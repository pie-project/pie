//! A buffer whose address never moves and whose memory comes and goes.
//!
//! The KV cache is the problem this exists for. Its size is not known when
//! the model loads -- it depends on how many sequences arrive and how long
//! they get -- and the obvious answers are both wrong. Allocating for the
//! worst case reserves tens of gigabytes that are usually idle, on a machine
//! where the GPU and the CPU are competing for the same DRAM. Reallocating
//! as it grows changes the buffer's GPU address, and the address is baked
//! into argument tables, into constant blocks, and into every kernel that
//! walks the cache by pointer.
//!
//! A placement-sparse buffer separates the two. The buffer is created once at
//! its full virtual size, so its `gpuAddress` is fixed for its lifetime and
//! nothing that recorded it ever has to be told. Physical memory is attached
//! and detached underneath in [`TILE`]-sized pieces, and only the attached
//! part costs anything.
//!
//! # Three sizes, and they are all different
//!
//! * [`TILE`] (16 KiB) is the sparse page: the granularity Metal will map at,
//!   and therefore what every offset and length here is rounded to.
//! * [`CHUNK`] (256 MiB) is the placement heap: the granularity physical
//!   memory is *acquired* at. Mapping a tile needs a heap to take it from,
//!   and a heap per tile would be tens of thousands of heaps.
//! * [`PAGE`] (2 MiB) is neither. It is the unit the budget is *reported* in,
//!   shared with the CUDA driver so the two agree on what a number means.
//!
//! Confusing the first two is the bug this layout is arranged to prevent: the
//! chunk is what gets allocated and freed, the tile is what gets mapped and
//! unmapped, and a chunk is released only once every tile in it is unmapped.
//!
//! # Growth is refusable, and the refusal is the point
//!
//! [`Arena`] holds a budget, and an [`Elastic`] that asks past it is told no
//! rather than being given memory the machine cannot spare. Under OS memory
//! pressure the budget shrinks -- but only for growth, never for what a model
//! already had to have. See [`Need`] for why that distinction is load-bearing
//! and what happened when it was not there.
//!
//! # Unmapping is a GPU operation, not a host one
//!
//! Tearing a tile out from under a running kernel is a fault. So a remap is a
//! point on the same timeline as a step -- it waits for the last committed
//! step and the next step waits for it -- which is why the operations that
//! change mappings take a [`Stepper`](super::Stepper) rather than doing it
//! behind its back. A heap is handed back only after the GPU has been
//! observed past the unmap that emptied it; until then it sits in
//! [`Arena::pending`].
//!
//! Destroying a buffer is the same rule wearing different clothes. A growth
//! is issued and not waited for, so an [`Elastic`] can reach its destructor
//! with an `updateBufferMappings` still queued against heaps it is about to
//! release. Dropping them there is a GPU page fault, not a leak -- so the
//! destructor waits for the mapping timeline first, and leaks rather than
//! frees if that wait runs out. See `Fence`.

use std::ptr::NonNull;
use std::sync::{Arc, Mutex, Weak};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4CommandQueue, MTLBuffer, MTLDevice, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLResidencySet, MTLResourceOptions, MTLSharedEvent, MTLSparsePageSize,
    MTLStorageMode,
};

use crate::gpu::device::context::Context;
use crate::error::{Error, Result};

/// The sparse page size: the granularity Metal maps at.
///
/// Every offset and length in this module is a multiple of it, and rounding
/// up rather than down is not a choice -- a request rounded down would leave
/// the last bytes of a caller's range unmapped, which faults on access
/// instead of failing at the ask.
pub const TILE: u64 = 16 * 1024;

/// The placement-heap size: the granularity physical memory is acquired at.
///
/// Large because a heap is an allocation with its own residency entry, and a
/// 40 GiB cache mapped a tile at a time would need two and a half million of
/// them. Large enough that the last chunk of a buffer is usually part-used,
/// which is why a chunk tracks how much of itself is mapped rather than being
/// all-or-nothing.
pub const CHUNK: u64 = 256 * 1024 * 1024;

/// The unit budgets are *reported* in.
///
/// Not a tile and not a chunk. It exists so that this driver and the CUDA one
/// quote the same number for the same thing; CUDA commits at this
/// granularity, this one does not, and a shared unit is what keeps a
/// cross-driver report comparable. Never use it to size a mapping.
pub const PAGE: u64 = 2 * 1024 * 1024;

/// How many [`PAGE`]s `bytes` occupies.
///
/// Zero bytes is zero pages -- not one -- because this feeds a report of how
/// much is in use, and an empty pool that reports a page in use is a pool
/// nobody can prove they released.
#[must_use]
pub const fn pages_for_bytes(bytes: u64) -> u64 {
    bytes.div_ceil(PAGE)
}

/// Round up to a whole number of [`TILE`]s, saturating.
const fn tiles_up(bytes: u64) -> u64 {
    match bytes.checked_next_multiple_of(TILE) {
        Some(rounded) => rounded,
        // Only reachable within one tile of `u64::MAX`, which is not an
        // allocation anyone will make; saturating beats wrapping to zero,
        // which would silently map nothing.
        None => u64::MAX,
    }
}

/// How far back from `end` a span can reach and stay in one chunk.
///
/// The last byte is at `end - 1`; whichever chunk that lands in, the span can
/// run back to that chunk's start. Callers only reach this with `end > 0`,
/// because a piece of no bytes is one the walk never asks for.
const fn tail_in_chunk(end: u64, chunk: u64) -> u64 {
    (end - 1) % chunk + 1
}

/// How many of the `want` bytes at `offset` lie before the next chunk seam.
const fn head_in_chunk(offset: u64, want: u64, chunk: u64) -> u64 {
    let to_seam = chunk - offset % chunk;
    if want < to_seam { want } else { to_seam }
}

/// Cut a move into pieces that each lie inside one chunk on both sides, and
/// hand them to `piece` in an order that does not smear an overlap.
///
/// # Why the order matters
///
/// A single `memmove` may overlap because it reads all of the source before
/// any of the destination exists to conflict with -- it decides internally
/// which direction to run. Cutting the move into pieces takes that decision
/// away from it: piece 1 has already landed by the time piece 2 reads, so a
/// forward walk with `dst > src` overwrites bytes that later pieces still
/// have to read, and copies the first piece over and over down the span. So
/// the walk runs front-to-back when the destination is below the source and
/// back-to-front when it is above -- the same rule, applied one level up.
///
/// Pieces are cut at whichever side's seam comes first: the two spans sit at
/// different offsets, so their chunk boundaries do not line up and either can
/// be the one that ends the piece.
///
/// # Errors
///
/// Whatever `piece` returns, at the first piece that returns one. Pieces
/// already handed over have already happened.
fn walk_move(
    dst: u64,
    src: u64,
    bytes: u64,
    chunk: u64,
    mut piece: impl FnMut(u64, u64, u64) -> Result<()>,
) -> Result<()> {
    if bytes == 0 || dst == src {
        return Ok(());
    }
    let mut done = 0;
    while done < bytes {
        let left = bytes - done;
        let (d, s, take) = if dst < src {
            let (d, s) = (dst + done, src + done);
            (
                d,
                s,
                head_in_chunk(d, left, chunk).min(head_in_chunk(s, left, chunk)),
            )
        } else {
            let (d_end, s_end) = (dst + left, src + left);
            let take = left
                .min(tail_in_chunk(d_end, chunk))
                .min(tail_in_chunk(s_end, chunk));
            (d_end - take, s_end - take, take)
        };
        piece(d, s, take)?;
        done += take;
    }
    Ok(())
}

/// A `u64` byte count as a `usize`, saturating.
///
/// Byte counts are `u64` because that is what Metal speaks; `ptr` operations
/// want `usize`. On a 64-bit host this is the identity, and every host that
/// runs this crate is one -- the saturation is so the cast is not silently
/// wrapping on a hypothetical 32-bit build.
const fn usize_of(v: u64) -> usize {
    if v > usize::MAX as u64 { usize::MAX } else { v as usize }
}

/// What the OS says about memory, as a level rather than an event.
///
/// The C++ subscribes a `dispatch_source` to memory-pressure notifications,
/// stores the level in a shared atomic, cancels the source in the destructor,
/// and exposes `set_memory_pressure_level_for_test` because none of that can
/// be driven from a test. Four pieces of machinery, one of which exists only
/// to work around the other three.
///
/// The level is also just readable, from the same sysctl the notification is
/// derived from, so here it is a value that is passed in. There is no
/// background thread to tear down, no shared atomic, and no test override:
/// a test that wants critical pressure passes [`Pressure::Critical`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Pressure {
    /// Nothing is tight. The full budget applies.
    #[default]
    Normal,
    /// The OS would like memory back. Growth is held to half the budget.
    Warn,
    /// The OS needs memory back. Growth is held to the floor.
    Critical,
}

impl Pressure {
    /// Read the current level from the kernel.
    ///
    /// `None` when the sysctl is missing or returns something unexpected,
    /// which is the honest answer and not [`Pressure::Normal`]: a probe that
    /// reports "fine" when it actually failed would turn an unreadable system
    /// into one that never clamps.
    #[must_use]
    pub fn probe() -> Option<Self> {
        let mut level: i32 = 0;
        let mut size = size_of::<i32>();
        let name = c"kern.memorystatus_vm_pressure_level";
        // SAFETY: `name` is a NUL-terminated C string, the output pointer is
        // to a live `i32` and `size` says so, and the new-value pointer is
        // null with a zero length, which is how sysctlbyname is told to read.
        let ok = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                std::ptr::from_mut(&mut level).cast(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if ok != 0 || size != size_of::<i32>() {
            return None;
        }
        // The kernel's constants are a bitfield: 1 normal, 2 warn, 4 critical.
        match level {
            1 => Some(Self::Normal),
            2 => Some(Self::Warn),
            4 => Some(Self::Critical),
            _ => None,
        }
    }
}

/// Whether an ask is something a step cannot run without.
///
/// This distinction is the whole reason the pressure clamp is usable, and it
/// was learned twice from the same model.
///
/// The first time, a floor of zero under critical pressure was not a floor
/// but an off switch: every ask failed, including the few megabytes per layer
/// that make a KV pool exist at all. And the load's own mapping is what
/// raised the pressure -- eighteen gigabytes of clean file cache takes free
/// memory to nothing on the way in -- so the model was refused by a state its
/// own admission created. Refusing there buys nothing either: the machine was
/// already checked against the whole budget before a byte was allocated, so
/// declining the mandatory part does not hand a page back, it turns an
/// admitted model into an unusable one.
///
/// The second time, the same argument turned out to reach one step further.
/// The initial commitment is what makes a buffer exist, not what makes the
/// first forward pass legal -- the step also needs scratch and KV rows sized
/// to the tokens actually in hand. Those are not growth either; they are the
/// same admitted requirement arriving a moment later. Only the initial
/// commitment had been recorded as mandatory, so the clamp declined the rest
/// and a model that had just loaded could not take a step.
///
/// The C++ carries this as a `serving_step_requirement` flag set on the
/// context for the span of one call and cleared by a destructor. Here the
/// caller says which kind of ask it is making, because it is the only one who
/// knows and a flag is a worse way to be told.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Need {
    /// Speculative: memory held against work that has not arrived. Clamped.
    #[default]
    Growth,
    /// A step cannot run without it. The full budget applies, because
    /// admission already checked the machine against the full budget.
    Step,
}

/// What the arena is allowed to hand out, and what it has.
///
/// Separate from the allocations so that the arithmetic can be tested without
/// a GPU -- and it is the arithmetic, not the Metal calls, that decides
/// whether a model runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Budget {
    /// The ceiling, as admission computed it.
    pub total: u64,
    /// The most pressure may clamp growth down to. Distinct from `mandatory`:
    /// this is a policy floor set once, that is a running total of what
    /// individual buffers declared they could not exist without.
    pub floor: u64,
    /// The sum of every live buffer's initial commitment.
    pub mandatory: u64,
    /// Bytes promised to buffers, whether or not the mapping has landed.
    /// Checked against rather than `committed`, so two asks in flight cannot
    /// both be told there is room for the same bytes.
    pub reserved: u64,
    /// Bytes actually mapped.
    pub committed: u64,
}

impl Budget {
    /// The ceiling that applies to one ask.
    ///
    /// Never below `mandatory`, and never above `total`. A [`Need::Step`] ask
    /// ignores pressure entirely -- see [`Need`] for the two occasions that
    /// taught this.
    #[must_use]
    pub fn effective(&self, pressure: Pressure, need: Need) -> u64 {
        if need == Need::Step || pressure == Pressure::Normal {
            return self.total;
        }
        let clamp = match pressure {
            Pressure::Critical => self.floor,
            // Halved rather than floored, because a warning is a request and
            // not a demand: something has to still be servable or a warning
            // becomes indistinguishable from a critical.
            _ => self.floor.max(self.total / 2),
        };
        self.total.min(self.mandatory.max(clamp))
    }

    /// What is still available under `effective`.
    #[must_use]
    pub fn headroom(&self, pressure: Pressure, need: Need) -> u64 {
        self.effective(pressure, need).saturating_sub(self.reserved)
    }
}

/// One placement heap and how much of it is in use.
struct Chunk {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    /// A buffer aliasing the whole heap.
    ///
    /// It exists first because a placement heap with no resource placed in it
    /// is a heap Metal may treat as unused, and because the C++ keeps one;
    /// dropping it is what releases the heap's memory alongside the heap
    /// itself. It is also the ONLY host address for these bytes -- the sparse
    /// buffer over them is private -- which is what [`Elastic::host_span`]
    /// hands out.
    alias: Retained<ProtocolObject<dyn MTLBuffer>>,
    bytes: u64,
    mapped: u64,
}

// SAFETY: the only thing in here that is not a plain integer is a `Retained`
// of a Metal heap and its alias buffer, and a `Retained` is refcounted
// thread-safely. Metal objects have no thread affinity; what `Send` grants is
// transfer, not sharing, and the `Mutex` is what provides sharing. Same
// argument as `Cached` in `pool`.
unsafe impl Send for Chunk {}

/// A heap that has been unmapped but not yet proven idle.
struct Pending {
    /// The timeline value at which the unmap will have happened on the GPU.
    through: u64,
    chunk: Chunk,
}

/// The last mapping operation issued over a buffer, and where it lands.
///
/// Recorded because a growth is deliberately not waited for -- see
/// [`Stepper::ensure`](super::Stepper::ensure) -- which leaves an
/// `updateBufferMappings` in flight that names heaps this buffer owns.
/// Nothing that only reads the buffer needs to care, and that is why the
/// growth may go unwaited. Destruction is not a read: releasing those heaps
/// while the operation that names them is still queued hands the GPU a
/// mapping into memory that has gone back to the system, which faults inside
/// the GPU driver rather than in this process.
struct Fence {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// The timeline value the last mapping over this buffer lands at.
    through: u64,
}

// SAFETY: a `Retained` of a shared event is refcounted thread-safely and
// Metal objects have no thread affinity; the same argument as `Chunk`.
unsafe impl Send for Fence {}

/// How long each probe of the teardown wait blocks for.
///
/// Probed rather than waited for in one call for the same reason
/// [`Stepper`](super::Stepper) probes: an unbounded wait in a destructor is a
/// hang with no message attached to it.
const TEARDOWN_PROBE_MS: u64 = 5_000;

/// How many probes before a destructor gives up and leaks instead.
const TEARDOWN_PROBES: u32 = 12;

/// Hand a Metal object's last reference to nobody, on purpose.
///
/// The one situation this is reached from is a teardown the GPU has not been
/// proven past -- see [`Elastic::drop`]. Releasing there is not the safe
/// choice and leaking is, so the leak is spelled out rather than left as an
/// omission a later reader would take for a bug.
fn leak<T: ?Sized + objc2::Message>(object: Retained<T>) {
    let _ = Retained::into_raw(object);
}

/// The arena's shared state: budget plus heaps waiting to be given back.
#[derive(Default)]
struct State {
    budget: Budget,
    pending: Vec<Pending>,
}

impl State {
    /// Release every heap whose unmap the GPU has passed.
    ///
    /// `signalled` is what the timeline has actually reached. Heaps at or
    /// below it are dropped; the rest stay, because a heap freed while a
    /// kernel still holds the mapping is a fault rather than a leak, and a
    /// leak is the safe half of a bad situation.
    fn collect(&mut self, signalled: u64, residency: &ProtocolObject<dyn MTLResidencySet>) {
        let mut freed = false;
        self.pending.retain(|entry| {
            if entry.through > signalled {
                return true;
            }
            residency.removeAllocation(ProtocolObject::from_ref(&*entry.chunk.heap));
            freed = true;
            false
        });
        if freed {
            residency.commit();
        }
    }
}

/// A budget, and the heaps it has handed out.
///
/// Cloneable, and a clone is the same arena: [`Elastic`] holds a weak
/// reference back so that dropping a buffer returns its bytes without the
/// arena having to be told.
#[derive(Clone)]
pub struct Arena {
    state: Arc<Mutex<State>>,
}

impl Arena {
    /// An arena with `total` bytes to give out and a pressure floor of
    /// `floor`.
    ///
    /// `floor` of zero is legal and means growth stops entirely under
    /// critical pressure. It does NOT mean existing buffers are taken back --
    /// see [`Need`].
    #[must_use]
    pub fn new(total: u64, floor: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                budget: Budget {
                    total,
                    floor,
                    ..Budget::default()
                },
                pending: Vec::new(),
            })),
        }
    }

    /// Whether two handles name the same arena.
    ///
    /// Cloning an `Arena` shares its budget, so equality here is identity of
    /// the accounting, not equality of the numbers. A batch growth prices
    /// itself against one budget and has to know that every buffer in the
    /// batch is drawing on that one.
    #[must_use]
    pub fn is(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    /// What the arena has promised and mapped, right now.
    #[must_use]
    pub fn budget(&self) -> Budget {
        self.lock().budget
    }

    /// How many heaps are unmapped but not yet given back.
    ///
    /// Non-zero means a trim has happened that the GPU has not been observed
    /// past. It falls to zero on its own as steps complete; a value that
    /// never falls means the timeline stopped moving.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.lock().pending.len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl std::fmt::Debug for Arena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.lock();
        f.debug_struct("Arena")
            .field("budget", &state.budget)
            .field("pending", &state.pending.len())
            .finish()
    }
}

/// A buffer with a fixed address and a variable amount of memory behind it.
///
/// Created at its full virtual size and mapped up to whatever it currently
/// needs. [`gpu_address`](Self::gpu_address) is stable for the whole life of
/// the value, which is the property the whole module exists to provide.
pub struct Elastic {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// The rounded-up virtual size: what can ever be mapped.
    virtual_bytes: u64,
    /// What the caller asked for, un-rounded. Bounds checks use this, so a
    /// caller cannot reach the rounding slack it never asked for.
    len: u64,
    committed: u64,
    /// What this buffer declared it cannot exist without, so that dropping it
    /// gives exactly that back.
    mandatory: u64,
    chunks: Vec<Chunk>,
    owner: Weak<Mutex<State>>,
    /// The residency set this buffer and its heaps were added to.
    ///
    /// Held as an owning handle rather than borrowed from the context so that
    /// the set cannot be released while this buffer is still named in it, and
    /// so that [`Drop`] has something to take the names back out of.
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    /// The last mapping issued over this buffer, if any. See [`Fence`].
    fence: Option<Fence>,
}

impl Elastic {
    /// The buffer's address, which does not change.
    #[must_use]
    pub fn gpu_address(&self) -> u64 {
        self.buffer.gpuAddress()
    }

    /// What the caller asked for. Not the rounded virtual size.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.len
    }

    /// Whether it was created with no bytes at all.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// How much memory is attached right now.
    ///
    /// A multiple of [`TILE`], and at least what was last successfully
    /// ensured -- rounding means it can be more.
    #[must_use]
    pub const fn committed(&self) -> u64 {
        self.committed
    }

    /// A host address for the `len` bytes at `offset`.
    ///
    /// The sparse buffer itself is private, because it is a page table and a
    /// page table has no contents. The memory is in the placement heaps, and
    /// `make_chunk` makes those `Shared` for exactly this reason -- "what
    /// lets the host stage into a KV page without a second copy". This is the
    /// accessor that makes that sentence true; without it the storage mode
    /// was a promise nothing kept, and the alias was documented as never
    /// read.
    ///
    /// The address is only valid while the span stays mapped. A `shrink`
    /// past it takes the memory back, so a caller holding one across a
    /// shrink is holding a dangling pointer -- ask again after.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfRange`] when the span runs past what is committed:
    /// address space with no memory attached has nothing to point at, and
    /// returning an address into it would fault on first touch rather than
    /// here. [`Error::Create`] for a zero-length span, and for one that
    /// crosses a chunk boundary -- two chunks are two heaps, and no single
    /// host pointer spans them.
    pub fn host_span(&self, offset: u64, len: u64) -> Result<NonNull<u8>> {
        if len == 0 {
            return Err(Error::Create {
                what: "elastic host span",
                message: "a span of no bytes has no address".to_string(),
            });
        }
        let end = offset.checked_add(len).ok_or(Error::OutOfRange {
            what: "elastic host span",
            offset,
            bytes: len,
            len: self.committed,
        })?;
        if end > self.committed {
            return Err(Error::OutOfRange {
                what: "elastic host span",
                offset,
                bytes: len,
                len: self.committed,
            });
        }
        // Chunk `i` covers `[i * CHUNK, i * CHUNK + size)` of the buffer's
        // address space, in order, which is what `grow` builds: it fills the
        // last chunk before pushing another.
        let index = usize::try_from(offset / CHUNK).map_err(|_| Error::Create {
            what: "elastic host span",
            message: format!("offset {offset} does not index a chunk on this host"),
        })?;
        let within = offset % CHUNK;
        let chunk = self.chunks.get(index).ok_or(Error::OutOfRange {
            what: "elastic host span",
            offset,
            bytes: len,
            len: self.committed,
        })?;
        if within + len > chunk.bytes {
            return Err(Error::Create {
                what: "elastic host span",
                message: format!(
                    "{len} bytes at {offset} cross the end of chunk {index}, which is \
                     a different heap; ask for the two sides separately"
                ),
            });
        }
        let base = chunk.alias.contents();
        // SAFETY: `within + len <= chunk.bytes`, and the alias is a buffer
        // over the whole heap, so the offset is inside the allocation.
        let at = unsafe { base.as_ptr().cast::<u8>().add(within as usize) };
        NonNull::new(at).ok_or(Error::Create {
            what: "elastic host span",
            message: "the heap alias has no host address".to_string(),
        })
    }

    /// Zero `bytes` at `offset`, across as many chunks as they span.
    ///
    /// [`host_span`](Self::host_span) refuses a span that crosses a chunk,
    /// because no single host pointer covers two heaps. A caller clearing a
    /// KV page does not care where the allocator's seams fell -- it wants the
    /// bytes cleared. This walks the chunks so the caller states the span it
    /// means rather than the tiling underneath it.
    ///
    /// # Errors
    ///
    /// As [`host_span`](Self::host_span), for the first span that is not
    /// mapped. What was already cleared stays cleared: a caller that needs
    /// all-or-nothing has to weigh the span against [`committed`] first.
    ///
    /// [`committed`]: Self::committed
    ///
    /// # Safety
    ///
    /// Nothing may be reading these bytes on the GPU. The pages are host
    /// addressable, not host owned -- between fires the host owns them
    /// outright, during one it does not.
    pub unsafe fn zero(&self, offset: u64, bytes: u64) -> Result<()> {
        let mut done = 0;
        while done < bytes {
            let at = offset + done;
            let take = head_in_chunk(at, bytes - done, CHUNK);
            let span = self.host_span(at, take)?;
            // SAFETY: `host_span` returned `take` writable bytes there.
            unsafe { std::ptr::write_bytes(span.as_ptr(), 0, usize_of(take)) };
            done += take;
        }
        Ok(())
    }

    /// Move `bytes` from `src` to `dst` within this buffer.
    ///
    /// A memmove, and the overlap is not hypothetical: a KV compaction slides
    /// live rows toward the front of the pool, so source and destination
    /// share bytes by construction. See `walk_move` for why that makes the
    /// order of the pieces load-bearing.
    ///
    /// # Errors
    ///
    /// As [`host_span`](Self::host_span), for the first piece on either side
    /// that is not mapped. Partial: what was moved before the refusal stays
    /// moved.
    ///
    /// # Safety
    ///
    /// As [`zero`](Self::zero).
    pub unsafe fn copy_within(&self, dst: u64, src: u64, bytes: u64) -> Result<()> {
        walk_move(dst, src, bytes, CHUNK, |d, s, take| {
            let to = self.host_span(d, take)?;
            let from = self.host_span(s, take)?;
            // SAFETY: both spans are `take` mapped bytes. They may point into
            // the same heap and may overlap, which `copy` (memmove) permits,
            // and `walk_move` orders the pieces so that one is never written
            // over bytes a later one still has to read.
            unsafe { std::ptr::copy(from.as_ptr(), to.as_ptr(), usize_of(take)) };
            Ok(())
        })
    }

    /// The underlying buffer, for binding.
    #[must_use]
    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// The arena this was created in, if it still exists.
    ///
    /// `None` after the arena has been dropped, which leaves the buffer
    /// usable at its current size but ungrowable -- there is nothing left to
    /// charge.
    #[must_use]
    pub fn arena(&self) -> Option<Arena> {
        self.owner.upgrade().map(|state| Arena { state })
    }

    /// Record that a mapping over this buffer lands at `through` on `event`.
    ///
    /// Called by [`Stepper`](super::Stepper) after every remap, because the
    /// stepper owns the timeline and this buffer owns the heaps the mapping
    /// names -- and the only moment both are known is here. Monotonic: the
    /// timeline only advances, and a lower value would let teardown stop
    /// waiting before the newest operation.
    pub(crate) fn fence_at(
        &mut self,
        event: &Retained<ProtocolObject<dyn MTLSharedEvent>>,
        through: u64,
    ) {
        match &mut self.fence {
            Some(fence) => fence.through = fence.through.max(through),
            None => {
                self.fence = Some(Fence {
                    event: event.clone(),
                    through,
                });
            }
        }
    }

    /// Block until every mapping issued over this buffer has landed.
    ///
    /// `true` when the GPU is known to be past all of them, which is the only
    /// condition under which this buffer's heaps may be released.
    fn drained(&self) -> bool {
        let Some(fence) = &self.fence else {
            return true;
        };
        if fence.event.signaledValue() >= fence.through {
            return true;
        }
        (0..TEARDOWN_PROBES).any(|_| {
            fence
                .event
                .waitUntilSignaledValue_timeoutMS(fence.through, TEARDOWN_PROBE_MS)
        })
    }
}

impl Drop for Elastic {
    fn drop(&mut self) {
        // The mappings are not torn down here: an unmap is a GPU operation
        // needing a timeline, and releasing the sparse buffer releases the
        // address space they lived in, so there is nothing left to unmap
        // from. What CANNOT be skipped is the wait. A growth is issued and
        // deliberately not waited for, so at this moment an
        // `updateBufferMappings` naming these heaps may still be queued; the
        // heaps go when their `Chunk`s do, a few lines below. Freeing them
        // under a live mapping operation is not a leak and not a wrong
        // answer -- it is a GPU page fault, raised inside the Metal driver,
        // which takes the machine down rather than this process.
        //
        // This was three kernel panics before it was a comment.
        if self.drained() {
            self.residency
                .removeAllocation(ProtocolObject::from_ref(&*self.buffer));
            for chunk in &self.chunks {
                self.residency
                    .removeAllocation(ProtocolObject::from_ref(&*chunk.heap));
            }
            self.residency.commit();
        } else {
            // The timeline stopped moving, so nothing will ever prove the
            // mapping landed. Leak the buffer and every heap rather than
            // free them: a leak costs this process memory it was already
            // holding, and the alternative costs the machine. Same trade as
            // `State::collect` makes, for the same reason.
            leak(self.buffer.clone());
            for chunk in self.chunks.drain(..) {
                let Chunk { heap, alias, .. } = chunk;
                leak(heap);
                leak(alias);
            }
        }

        // The accounting is given back either way, because an arena that
        // keeps counting a freed buffer refuses the next one.
        let Some(state) = self.owner.upgrade() else {
            return;
        };
        let mut state = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.budget.reserved = state.budget.reserved.saturating_sub(self.committed);
        state.budget.committed = state.budget.committed.saturating_sub(self.committed);
        state.budget.mandatory = state.budget.mandatory.saturating_sub(self.mandatory);
    }
}

impl std::fmt::Debug for Elastic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Elastic")
            .field("len", &self.len)
            .field("virtual_bytes", &self.virtual_bytes)
            .field("committed", &self.committed)
            .field("chunks", &self.chunks.len())
            .finish()
    }
}

/// Make a placement heap of `bytes` and a buffer aliasing all of it.
fn make_chunk(context: &Context, bytes: u64) -> Result<Chunk> {
    let descriptor = MTLHeapDescriptor::new();
    descriptor.setType(MTLHeapType::Placement);
    // Shared, not Private: on this hardware there is one pool of memory, and
    // a Shared placement heap is what lets the host stage into a KV page
    // without a second copy. The sparse buffer over it is still Private.
    descriptor.setStorageMode(MTLStorageMode::Shared);
    // Untracked because the timeline already orders every access to these
    // tiles. Leaving tracking on asks Metal to insert hazards for a
    // dependency the stepper has already expressed, per allocation, forever.
    descriptor.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
    descriptor.setSize(usize::try_from(bytes).unwrap_or(usize::MAX));
    descriptor.setMaxCompatiblePlacementSparsePageSize(MTLSparsePageSize::Size16);

    let heap = context
        .device()
        .newHeapWithDescriptor(&descriptor)
        .ok_or_else(|| Error::Create {
            what: "MTLHeap",
            message: format!("placement heap of {bytes} bytes refused"),
        })?;
    // SAFETY: the offset is zero and the length is the heap's own size, so
    // the placement is in bounds by construction; the descriptor above
    // declared the heap compatible with this storage mode.
    let alias = unsafe {
        heap.newBufferWithLength_options_offset(
            usize::try_from(bytes).unwrap_or(usize::MAX),
            MTLResourceOptions::StorageModeShared,
            0,
        )
    }
    .ok_or_else(|| Error::Create {
        what: "MTLBuffer",
        message: format!("alias over a {bytes}-byte placement heap refused"),
    })?;

    context
        .residency()
        .addAllocation(ProtocolObject::from_ref(&*heap));
    context.residency().commit();

    Ok(Chunk {
        heap,
        alias,
        bytes,
        mapped: 0,
    })
}

/// Create a sparse buffer of `len` bytes in `arena`.
///
/// The buffer starts with nothing mapped; [`Stepper::ensure`](super::Stepper)
/// is what attaches memory. Creating it costs address space and a residency
/// entry, not memory, which is why the size can be the worst case even when
/// the usage will not be.
///
/// # Errors
///
/// If Metal refuses the sparse buffer. A zero `len` is an error rather than
/// an empty buffer: a zero-length buffer has no address, and the address is
/// the only thing this type promises.
pub fn create(context: &Context, arena: &Arena, len: u64) -> Result<Elastic> {
    if len == 0 {
        return Err(Error::Create {
            what: "elastic buffer",
            message: "a zero-length sparse buffer has no address to promise".to_string(),
        });
    }
    let virtual_bytes = tiles_up(len);
    // SAFETY: the length is a whole number of tiles at the page size given,
    // which is what this call requires; nothing is mapped yet, so there is no
    // aliasing to violate.
    let buffer = unsafe {
        context
            .device()
            .newBufferWithLength_options_placementSparsePageSize(
                usize::try_from(virtual_bytes).unwrap_or(usize::MAX),
                MTLResourceOptions::StorageModePrivate,
                MTLSparsePageSize::Size16,
            )
    }
    .ok_or_else(|| Error::Create {
        what: "placement sparse buffer",
        message: format!("{virtual_bytes} bytes of sparse address space refused"),
    })?;

    context
        .residency()
        .addAllocation(ProtocolObject::from_ref(&*buffer));
    context.residency().commit();

    Ok(Elastic {
        buffer,
        virtual_bytes,
        len,
        committed: 0,
        mandatory: 0,
        chunks: Vec::new(),
        owner: Arc::downgrade(&arena.state),
        residency: context.residency_handle(),
        fence: None,
    })
}

/// The queue sparse remappings are issued on.
///
/// Separate from the step queue because a remap has to be ordered against
/// steps from *outside* them: it waits for the last committed step and the
/// next step waits for it. Issuing it on the step queue would put it inside
/// the ordering it is supposed to bracket.
pub(crate) struct Mappings {
    pub(crate) queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
}

impl Mappings {
    /// Build the mapping queue for `context`.
    pub(crate) fn new(context: &Context) -> Result<Self> {
        let queue = context
            .device()
            .newMTL4CommandQueue()
            .ok_or(Error::Create {
                what: "MTL4CommandQueue",
                message: String::new(),
            })?;
        Ok(Self { queue })
    }
}

/// How a mapping is put on the timeline: buffer, heap, first tile, tile
/// count, heap tile. Returns the timeline value it completes at.
///
/// A callback rather than a method because the timeline lives on
/// [`Stepper`](super::Stepper) and the heaps live here, and neither type
/// should have to know how the other works.
pub(crate) type Map<'a> = dyn FnMut(&ProtocolObject<dyn MTLBuffer>, &ProtocolObject<dyn MTLHeap>, u64, u64, u64) -> u64
    + 'a;

/// The unmap half: buffer, first tile, tile count. No heap -- Metal ignores
/// it for an unmap, and passing one would imply the tiles go back to that
/// particular heap, which is not what happens.
pub(crate) type Unmap<'a> = dyn FnMut(&ProtocolObject<dyn MTLBuffer>, u64, u64) -> u64 + 'a;

/// Attach memory to `buffer` until at least `bytes` of it is mapped.
///
/// Implemented here rather than on [`Elastic`] because it needs three things
/// no single value owns: the device (to make heaps), the arena (to check and
/// charge the budget) and the timeline (to order the remap against steps).
///
/// Returns the timeline value the mapping completes at, or `None` when
/// nothing had to change. Idempotent: an ask below what is already mapped
/// costs nothing, which is what lets a caller ask on every step.
pub(crate) fn grow(
    context: &Context,
    buffer: &mut Elastic,
    bytes: u64,
    pressure: Pressure,
    need: Need,
    schedule: &mut Map<'_>,
) -> Result<Option<u64>> {
    if bytes > buffer.len {
        return Err(Error::Create {
            what: "elastic growth",
            message: format!(
                "asked for {bytes} bytes of a buffer that is {} long",
                buffer.len
            ),
        });
    }
    let target = tiles_up(bytes).min(buffer.virtual_bytes);
    if target <= buffer.committed {
        return Ok(None);
    }
    let delta = target - buffer.committed;

    let Some(state) = buffer.owner.upgrade() else {
        return Err(Error::Create {
            what: "elastic growth",
            message: "the arena this buffer belongs to is gone".to_string(),
        });
    };
    {
        let mut state = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if delta > state.budget.headroom(pressure, need) {
            return Err(Error::Create {
                what: "elastic growth",
                message: format!(
                    "{delta} more bytes exceeds the {} available under {pressure:?} pressure \
                     for a {need:?} request",
                    state.budget.headroom(pressure, need)
                ),
            });
        }
        // Charged BEFORE the mapping, so that a second ask arriving while
        // this one is still attaching heaps cannot be told the same bytes are
        // free. Given back below if a heap is refused.
        state.budget.reserved += delta;
    }

    // A refcount bump so the sparse buffer can be handed to the scheduler
    // while `buffer.chunks` is borrowed mutably.
    let target_buffer = buffer.buffer.clone();
    let mut last = None;
    while buffer.committed < target {
        if buffer
            .chunks
            .last()
            .is_none_or(|chunk| chunk.mapped == chunk.bytes)
        {
            let offset = buffer.chunks.len() as u64 * CHUNK;
            let size = CHUNK.min(buffer.virtual_bytes - offset);
            match make_chunk(context, size) {
                Ok(chunk) => buffer.chunks.push(chunk),
                Err(error) => {
                    // Un-charge only what has not been mapped. The tiles that
                    // did land are real and still owed for.
                    let mut state = state
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    state.budget.reserved -= target - buffer.committed;
                    return Err(error);
                }
            }
        }

        let chunk = buffer.chunks.last_mut().expect("just pushed if empty");
        let grow = (target - buffer.committed).min(chunk.bytes - chunk.mapped);
        last = Some(schedule(
            &target_buffer,
            &chunk.heap,
            buffer.committed / TILE,
            grow / TILE,
            chunk.mapped / TILE,
        ));
        chunk.mapped += grow;
        buffer.committed += grow;

        let mut state = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.budget.committed += grow;
    }
    Ok(last)
}

/// Detach memory from `buffer` down to `bytes`, giving emptied heaps back.
///
/// The heaps are not dropped here. An unmap is a GPU operation, and a heap
/// freed before the GPU has passed the unmap is a fault rather than a leak;
/// they go to [`Arena::pending`] and are collected once the timeline is
/// observed past `through`.
///
/// Returns the timeline value the last unmap completes at, or `None` when
/// nothing had to change.
pub(crate) fn shrink(buffer: &mut Elastic, bytes: u64, schedule: &mut Unmap<'_>) -> Option<u64> {
    let target = tiles_up(bytes).min(buffer.virtual_bytes);
    if target >= buffer.committed {
        return None;
    }
    let state = buffer.owner.upgrade();
    let target_buffer = buffer.buffer.clone();
    let mut last = None;
    let mut released = 0u64;
    let mut emptied = Vec::new();

    while buffer.committed > target {
        let Some(chunk) = buffer.chunks.last_mut() else {
            break;
        };
        let shrink = (buffer.committed - target).min(chunk.mapped);
        last = Some(schedule(
            &target_buffer,
            (buffer.committed - shrink) / TILE,
            shrink / TILE,
        ));
        chunk.mapped -= shrink;
        buffer.committed -= shrink;
        released += shrink;
        if chunk.mapped == 0 {
            emptied.push(buffer.chunks.pop().expect("just inspected"));
        }
    }

    // A floor cannot name memory nobody holds.
    //
    // `declare_mandatory` only ever raises, because two callers declaring
    // different amounts for one buffer both have to be honoured. A trim is
    // not a second caller -- it is the same one saying it needs less, and it
    // has already given the bytes back by the time this runs. Left alone, the
    // arena would keep reserving against a pool that had shrunk, and the
    // pressure clamp's floor would stay where the pool used to be.
    let surrendered = buffer.mandatory - buffer.mandatory.min(buffer.committed);
    buffer.mandatory -= surrendered;

    if let (Some(state), Some(through)) = (state, last) {
        let mut state = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.budget.committed = state.budget.committed.saturating_sub(released);
        state.budget.reserved = state.budget.reserved.saturating_sub(released);
        state.budget.mandatory = state.budget.mandatory.saturating_sub(surrendered);
        state
            .pending
            .extend(emptied.into_iter().map(|chunk| Pending { through, chunk }));
    }
    last
}

/// Declare `bytes` of `buffer` to be what it cannot exist without.
///
/// Recorded on the arena so that the pressure clamp never reaches below the
/// sum of every buffer's commitment -- see [`Need`]. Called before the first
/// growth, because the growth is what consults the clamp.
pub(crate) fn declare_mandatory(buffer: &mut Elastic, bytes: u64) {
    let mandatory = tiles_up(bytes.min(buffer.virtual_bytes));
    let Some(state) = buffer.owner.upgrade() else {
        return;
    };
    let mut state = state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state.budget.mandatory += mandatory - buffer.mandatory.min(mandatory);
    buffer.mandatory = buffer.mandatory.max(mandatory);
}

impl Arena {
    /// Release every heap the GPU has been observed past.
    pub(crate) fn collect(&self, signalled: u64, residency: &ProtocolObject<dyn MTLResidencySet>) {
        self.lock().collect(signalled, residency);
    }

    /// What is available for one ask, without taking it.
    ///
    /// Public because the caller that has to decide whether to admit a
    /// sequence needs the number before it commits to asking for it.
    #[must_use]
    pub fn headroom(&self, pressure: Pressure, need: Need) -> u64 {
        self.lock().budget.headroom(pressure, need)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_length_is_rounded_up_to_a_whole_tile_because_down_would_fault() {
        assert_eq!(tiles_up(0), 0);
        assert_eq!(tiles_up(1), TILE);
        assert_eq!(tiles_up(TILE), TILE);
        assert_eq!(tiles_up(TILE + 1), 2 * TILE);
    }

    #[test]
    fn pages_are_the_reporting_unit_and_an_empty_pool_reports_none() {
        assert_eq!(pages_for_bytes(0), 0, "an empty pool must report nothing");
        assert_eq!(pages_for_bytes(1), 1);
        assert_eq!(pages_for_bytes(PAGE), 1);
        assert_eq!(pages_for_bytes(PAGE + 1), 2);
    }

    #[test]
    fn the_three_sizes_are_distinct_and_nest() {
        let (tile, page, chunk) = (TILE, PAGE, CHUNK);
        assert!(tile < page, "a tile is the smallest thing mapped");
        assert!(page < chunk, "a chunk holds many pages");
        assert!(
            chunk.is_multiple_of(tile),
            "a chunk must be a whole number of tiles, or the last tile of a \
             chunk would straddle two heaps"
        );
        assert!(
            page.is_multiple_of(tile),
            "a page must be a whole number of tiles"
        );
    }

    fn budget() -> Budget {
        Budget {
            total: 1000,
            floor: 100,
            mandatory: 200,
            reserved: 0,
            committed: 0,
        }
    }

    #[test]
    fn no_pressure_means_the_whole_budget() {
        assert_eq!(budget().effective(Pressure::Normal, Need::Growth), 1000);
    }

    #[test]
    fn a_step_requirement_ignores_pressure_entirely() {
        for pressure in [Pressure::Normal, Pressure::Warn, Pressure::Critical] {
            assert_eq!(
                budget().effective(pressure, Need::Step),
                1000,
                "a model that has been admitted must be able to take a step \
                 under {pressure:?}; refusing here does not hand a page back, \
                 it turns an admitted model into an unusable one"
            );
        }
    }

    #[test]
    fn a_warning_halves_growth_rather_than_stopping_it() {
        assert_eq!(
            budget().effective(Pressure::Warn, Need::Growth),
            500,
            "a warning that stopped growth would be indistinguishable from a \
             critical"
        );
    }

    #[test]
    fn critical_pressure_stops_growth_at_the_floor() {
        let mut b = budget();
        b.mandatory = 0;
        assert_eq!(b.effective(Pressure::Critical, Need::Growth), 100);
    }

    #[test]
    fn the_clamp_never_reaches_below_what_buffers_already_had_to_have() {
        // The floor is 100 and mandatory is 200. Clamping to 100 would mean
        // the arena believes it has handed out more than it is allowed to,
        // and every subsequent ask -- including a mandatory one -- fails.
        assert_eq!(
            budget().effective(Pressure::Critical, Need::Growth),
            200,
            "critical pressure clamped below the mandatory commitments that \
             are already mapped, which is the off-switch bug"
        );
    }

    #[test]
    fn a_zero_floor_stops_growth_but_does_not_reach_into_what_exists() {
        let b = Budget {
            total: 1000,
            floor: 0,
            mandatory: 300,
            reserved: 300,
            committed: 300,
        };
        assert_eq!(
            b.effective(Pressure::Critical, Need::Growth),
            300,
            "a floor of zero must stop growth, not unmap what a model needs \
             to exist"
        );
        assert_eq!(b.headroom(Pressure::Critical, Need::Growth), 0);
        assert_eq!(
            b.headroom(Pressure::Critical, Need::Step),
            700,
            "the step path still sees the budget admission checked against"
        );
    }

    #[test]
    fn headroom_counts_what_is_promised_and_not_only_what_is_mapped() {
        let b = Budget {
            total: 1000,
            floor: 0,
            mandatory: 0,
            // Promised but not yet mapped. Counting `committed` here would
            // let two asks in flight both be told there is room for the same
            // bytes.
            reserved: 800,
            committed: 100,
        };
        assert_eq!(b.headroom(Pressure::Normal, Need::Growth), 200);
    }

    #[test]
    fn headroom_saturates_rather_than_wrapping_when_over_committed() {
        let b = Budget {
            total: 100,
            floor: 0,
            mandatory: 0,
            reserved: 500,
            committed: 500,
        };
        assert_eq!(
            b.headroom(Pressure::Normal, Need::Growth),
            0,
            "an over-committed arena must report no room, not four exabytes"
        );
    }

    #[test]
    fn the_pressure_probe_reads_a_level_or_says_it_could_not() {
        // Whatever the machine is doing, the answer must be one of the three
        // levels and not a default. A probe that returned Normal on failure
        // would turn an unreadable system into one that never clamps.
        match Pressure::probe() {
            Some(level) => assert!(matches!(
                level,
                Pressure::Normal | Pressure::Warn | Pressure::Critical
            )),
            None => panic!("kern.memorystatus_vm_pressure_level is readable on macOS"),
        }
    }

    /// Run a move over a plain byte array the way [`Elastic::copy_within`]
    /// runs it over heaps: cut into per-chunk pieces, applied in the order
    /// [`walk_move`] hands them over.
    ///
    /// A tiny `chunk` is what makes this testable at all. The real one is 256
    /// MiB, so a move that crosses one is a quarter-gigabyte allocation --
    /// which is why the seam-crossing case had no test before the walk was
    /// separable from the heaps it walks.
    fn moved(bytes: &[u8], dst: u64, src: u64, len: u64, chunk: u64) -> Vec<u8> {
        let mut out = bytes.to_vec();
        walk_move(dst, src, len, chunk, |d, s, take| {
            let (d, s, take) = (usize_of(d), usize_of(s), usize_of(take));
            out.copy_within(s..s + take, d);
            Ok(())
        })
        .expect("the walk itself refuses nothing");
        out
    }

    #[test]
    fn a_move_inside_one_chunk_is_handed_over_whole() {
        let mut pieces = Vec::new();
        walk_move(0, 64, 32, 256, |d, s, take| {
            pieces.push((d, s, take));
            Ok(())
        })
        .expect("walk");
        assert_eq!(
            pieces,
            vec![(0, 64, 32)],
            "cutting a move that no seam crosses costs host_span calls and \
             buys nothing"
        );
    }

    #[test]
    fn a_move_is_cut_at_whichever_side_reaches_a_seam_first() {
        let mut pieces = Vec::new();
        // Destination seam at 256 (56 bytes in), source seam at 512 (12 in).
        walk_move(200, 500, 100, 256, |d, s, take| {
            pieces.push((d, s, take));
            Ok(())
        })
        .expect("walk");
        assert_eq!(
            pieces,
            vec![(200, 500, 12), (212, 512, 44), (256, 556, 44)],
            "the source's seam at 512 ends the first piece and the \
             destination's at 256 ends the second -- whichever side reaches \
             one first. Cutting on only one side would hand the other a span \
             that crosses two heaps, and host_span would refuse it"
        );
        assert_eq!(
            pieces.iter().map(|p| p.2).sum::<u64>(),
            100,
            "the pieces must add up to the move"
        );
    }

    #[test]
    fn an_overlapping_move_slides_rather_than_smearing_across_a_seam() {
        // The KV compaction's shape: live rows sliding toward the front of
        // the pool, far enough to cross a chunk seam.
        let source: Vec<u8> = (0..=255u8).collect();
        let chunk = 64;
        // Every case must genuinely overlap. A distance equal to the length
        // is a move that merely touches, and it lands correctly whichever
        // way the walk runs -- which is how the first draft of this test
        // passed against a deliberately forward-only walk.
        for (dst, src, len) in [(10u64, 100u64, 150u64), (100, 10, 150), (60, 62, 130)] {
            assert!(
                dst.abs_diff(src) < len,
                "a {len}-byte move between {src} and {dst} does not overlap,                  so it cannot tell the two walk directions apart"
            );
            let mut want = source.clone();
            want.copy_within(usize_of(src)..usize_of(src + len), usize_of(dst));
            assert_eq!(
                moved(&source, dst, src, len, chunk),
                want,
                "a {len}-byte move from {src} to {dst} across {chunk}-byte \
                 chunks must land where one memmove would; a walk that runs \
                 the wrong way copies its first piece down the whole span"
            );
        }
    }

    #[test]
    fn a_move_onto_itself_touches_nothing() {
        let mut pieces = 0;
        walk_move(48, 48, 16, 64, |_, _, _| {
            pieces += 1;
            Ok(())
        })
        .expect("walk");
        assert_eq!(pieces, 0, "a move to where the bytes already are is not a move");
    }
}
