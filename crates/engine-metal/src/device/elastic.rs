//! A buffer whose address never moves and whose memory comes and goes.
//!
//! The KV cache is the problem this exists for. Its size is not known when
//! the model loads -- it depends on how many sequences arrive and how long
//! they get -- and the obvious answers are both wrong. Allocating for the
//! worst case reserves tens of gigabytes that are usually idle, on a machine
//! where the GPU and the CPU are competing for the same DRAM. Reallocating
//! as it grows changes the buffer's GPU address, and the address is baked
//! into argument tables, into indirect command buffers, and into every
//! kernel that walks the cache by pointer.
//!
//! A placement-sparse buffer separates the two. The buffer is created once at
//! its full virtual size, so its `gpuAddress` is fixed for its lifetime and
//! nothing that recorded it ever has to be told. Physical memory is attached
//! and detached underneath in [`TILE`]-sized pieces, and only the attached
//! part costs anything.
//!
//! # Three sizes, and they are all different
//!
//! * [`TILE`] (256 KiB) is the sparse page: the granularity Metal will map at,
//!   and therefore what every offset and length here is rounded to.
//! * [`CHUNK`] (256 MiB) is the placement heap: the granularity physical
//!   memory is *acquired* at. Mapping a tile needs a heap to take it from,
//!   and a heap per tile would be tens of thousands of heaps.
//! * [`PAGE`] (2 MiB) is neither. It is the unit the budget is *reported* in,
//!   shared with the CUDA engine so the two agree on what a number means.
//!
//! Confusing the first two is the bug this layout is arranged to prevent: the
//! chunk is what gets allocated and freed, the tile is what gets mapped and
//! unmapped, and a chunk is released only once every tile in it is unmapped.
//!
//! # Growth is refusable, and the refusal is the point
//!
//! [`Arena`] holds a budget, and a batch of buffers that asks past it is
//! told no before anything is mapped, rather than being given memory the
//! machine cannot spare. The budget is what admission sized the pool at, so
//! a frame the runtime admitted always fits; the refusal is for the day the
//! two disagree.
//!
//! # Unmapping is a GPU operation, not a host one
//!
//! Tearing a tile out from under a running kernel is a fault. A remap goes
//! on the Metal 4 mapping queue ([`Sparse`]), and every batch of them is
//! waited for on the host before this module returns: a grown buffer's new
//! tiles are there before the frame that needs them is encoded, and a
//! shrunk buffer's heaps are handed back only after the unmap has landed.
//! What this module cannot know is whether a frame is still reading the
//! tiles a shrink takes away -- the shell's frame queue signals nothing --
//! so [`shrink_all`] is called only once the caller has drained its frames.
//!
//! Destroying a buffer is the same rule wearing different clothes. Freeing
//! a heap under a queued `updateBufferMappings` that names it is a GPU page
//! fault, not a leak -- so the destructor checks the mapping timeline is
//! past the last remap, and leaks rather than frees if that check fails.
//! See `Fence`.

use std::sync::{Arc, Mutex};

#[cfg(target_vendor = "apple")]
use std::ptr::NonNull;
#[cfg(target_vendor = "apple")]
use std::sync::Weak;

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor, MTLHeapType,
    MTLResidencySet, MTLResourceOptions, MTLSharedEvent, MTLSparsePageSize,
    MTLSparseTextureMappingMode, MTLStorageMode,
};

#[cfg(target_vendor = "apple")]
use super::sparse::Sparse;
#[cfg(target_vendor = "apple")]
use crate::error::Fault;
use crate::error::Result;

/// The sparse page size: the granularity Metal maps at.
///
/// Every offset and length in this module is a multiple of it, and rounding
/// up rather than down is not a choice -- a request rounded down would leave
/// the last bytes of a caller's range unmapped, which faults on access
/// instead of failing at the ask.
///
/// The largest page Metal offers rather than the smallest, because growth
/// is rounded to [`PAGE`] anyway, so a finer tile buys no granularity -- and
/// a kv plane read through a sparse buffer pays for its page table on every
/// access: sixteen times fewer tiles is sixteen times fewer translations.
pub const TILE: u64 = 256 * 1024;

/// The placement-heap size: the granularity physical memory is acquired at.
///
/// Large because a heap is an allocation with its own residency entry, and a
/// 40 GiB cache mapped a tile at a time would need two and a half million of
/// them. Large enough that the last chunk of a buffer is usually part-used,
/// which is why a chunk tracks how much of itself is mapped rather than being
/// all-or-nothing.
pub const CHUNK: u64 = 256 * 1024 * 1024;

/// The unit budgets are *reported* in, and the step growth is rounded to.
///
/// Not a tile and not a chunk. It exists so that this engine and the CUDA
/// one quote the same number for the same thing; CUDA commits at this
/// granularity, and rounding growth up to it here keeps a decode that adds
/// one kv page per layer from issuing one remap per layer per step.
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
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
const fn tiles_up(bytes: u64) -> u64 {
    match bytes.checked_next_multiple_of(TILE) {
        Some(rounded) => rounded,
        // Only reachable within one tile of `u64::MAX`, which is not an
        // allocation anyone will make; saturating beats wrapping to zero,
        // which would silently map nothing.
        None => u64::MAX,
    }
}

/// Round up to a whole number of [`PAGE`]s, saturating.
#[must_use]
pub const fn pages_up(bytes: u64) -> u64 {
    match bytes.checked_next_multiple_of(PAGE) {
        Some(rounded) => rounded,
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
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
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
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
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
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
const fn usize_of(v: u64) -> usize {
    if v > usize::MAX as u64 {
        usize::MAX
    } else {
        v as usize
    }
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
    /// Bytes promised to buffers, whether or not the mapping has landed.
    /// Checked against rather than `committed`, so two asks in flight cannot
    /// both be told there is room for the same bytes.
    pub reserved: u64,
    /// Bytes actually mapped.
    pub committed: u64,
    /// The most that has ever been mapped at once.
    pub high_water: u64,
}

impl Budget {
    /// What is still available: the ceiling less what is promised.
    #[must_use]
    pub fn headroom(&self) -> u64 {
        self.total.saturating_sub(self.reserved)
    }
}

/// One placement heap and how much of it is in use.
#[cfg(target_vendor = "apple")]
struct Chunk {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    /// A buffer aliasing the whole heap.
    ///
    /// It exists first because a placement heap with no resource placed in it
    /// is a heap Metal may treat as unused; dropping it is what releases the
    /// heap's memory alongside the heap itself. It is also the ONLY host
    /// address for these bytes -- the sparse buffer over them is private --
    /// which is what [`Elastic::host_span`] hands out.
    alias: Retained<ProtocolObject<dyn MTLBuffer>>,
    bytes: u64,
    mapped: u64,
}

// SAFETY: the only thing in here that is not a plain integer is a `Retained`
// of a Metal heap and its alias buffer, and a `Retained` is refcounted
// thread-safely. Metal objects have no thread affinity; what `Send` grants is
// transfer, not sharing, and the `Mutex` is what provides sharing.
#[cfg(target_vendor = "apple")]
unsafe impl Send for Chunk {}

/// A heap that has been unmapped but not yet proven idle.
#[cfg(target_vendor = "apple")]
struct Pending {
    /// The timeline value at which the unmap will have happened on the GPU.
    through: u64,
    chunk: Chunk,
}

/// The last mapping operation issued over a buffer, and where it lands.
///
/// Every batch of remaps is waited for before this module returns, so in
/// practice the fence is always satisfied by the time a buffer is dropped.
/// It is still recorded, because the one case where it is not -- a flush
/// whose wait ran out -- is the case where freeing the heaps is a GPU page
/// fault inside the Metal driver rather than a leak in this process.
#[cfg(target_vendor = "apple")]
struct Fence {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// The timeline value the last mapping over this buffer lands at.
    through: u64,
}

// SAFETY: a `Retained` of a shared event is refcounted thread-safely and
// Metal objects have no thread affinity; the same argument as `Chunk`.
#[cfg(target_vendor = "apple")]
unsafe impl Send for Fence {}

/// How long each probe of the teardown wait blocks for, and how many probes
/// before a destructor gives up and leaks instead.
#[cfg(target_vendor = "apple")]
const TEARDOWN_PROBE_MS: u64 = 5_000;
#[cfg(target_vendor = "apple")]
const TEARDOWN_PROBES: u32 = 12;

/// Hand a Metal object's last reference to nobody, on purpose.
///
/// The one situation this is reached from is a teardown the GPU has not been
/// proven past -- see [`Elastic::drop`]. Releasing there is not the safe
/// choice and leaking is, so the leak is spelled out rather than left as an
/// omission a later reader would take for a bug.
#[cfg(target_vendor = "apple")]
fn leak<T: ?Sized + objc2::Message>(object: Retained<T>) {
    let _ = Retained::into_raw(object);
}

/// The arena's shared state: budget plus heaps waiting to be given back.
#[derive(Default)]
struct State {
    budget: Budget,
    #[cfg(target_vendor = "apple")]
    pending: Vec<Pending>,
}

#[cfg(target_vendor = "apple")]
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
    /// An arena with `total` bytes to give out.
    #[must_use]
    pub fn new(total: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                budget: Budget {
                    total,
                    ..Budget::default()
                },
                #[cfg(target_vendor = "apple")]
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
    /// Non-zero means a shrink has happened that the GPU has not been
    /// observed past. It falls to zero as the timeline moves; a value that
    /// never falls means the timeline stopped moving.
    #[must_use]
    pub fn pending(&self) -> usize {
        #[cfg(target_vendor = "apple")]
        {
            self.lock().pending.len()
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            0
        }
    }

    /// What is available for one ask, without taking it.
    #[must_use]
    pub fn headroom(&self) -> u64 {
        self.lock().budget.headroom()
    }

    /// Release every heap the GPU has been observed past.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn collect(&self, signalled: u64, residency: &ProtocolObject<dyn MTLResidencySet>) {
        self.lock().collect(signalled, residency);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl std::fmt::Debug for Arena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.lock();
        f.debug_struct("Arena")
            .field("budget", &state.budget)
            .field("pending", &self.pending())
            .finish()
    }
}

/// A buffer with a fixed address and a variable amount of memory behind it.
///
/// Created at its full virtual size and mapped up to whatever it currently
/// needs. [`gpu_address`](Self::gpu_address) is stable for the whole life of
/// the value, which is the property the whole module exists to provide.
#[cfg(target_vendor = "apple")]
pub struct Elastic {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// The rounded-up virtual size: what can ever be mapped.
    virtual_bytes: u64,
    /// What the caller asked for, un-rounded. Bounds checks use this, so a
    /// caller cannot reach the rounding slack it never asked for.
    len: u64,
    committed: u64,
    chunks: Vec<Chunk>,
    owner: Weak<Mutex<State>>,
    /// The mapping side this buffer was created on: the device that makes
    /// its heaps, the queue that maps them, and the residency set they are
    /// named in.
    sparse: Arc<Sparse>,
    /// The last mapping issued over this buffer, if any. See [`Fence`].
    fence: Option<Fence>,
}

#[cfg(target_vendor = "apple")]
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
    /// grown to -- rounding means it can be more.
    #[must_use]
    pub const fn committed(&self) -> u64 {
        self.committed
    }

    /// A binding view over the sparse buffer, for encoders and blits.
    #[must_use]
    pub fn view(&self) -> super::Buffer {
        super::Buffer::sparse_view(self.buffer.clone(), self.len)
    }

    /// A host address for the `len` bytes at `offset`.
    ///
    /// The sparse buffer itself is private, because it is a page table and a
    /// page table has no contents. The memory is in the placement heaps, and
    /// `make_chunk` makes those `Shared` for exactly this reason -- what lets
    /// the host stage into a KV page without a second copy.
    ///
    /// The address is only valid while the span stays mapped. A shrink past
    /// it takes the memory back, so a caller holding one across a shrink is
    /// holding a dangling pointer -- ask again after.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] when the span runs past what is committed: address
    /// space with no memory attached has nothing to point at, and returning
    /// an address into it would fault on first touch rather than here.
    /// [`Fault::Device`] for a zero-length span, and for one that crosses a
    /// chunk boundary -- two chunks are two heaps, and no single host
    /// pointer spans them.
    pub fn host_span(&self, offset: u64, len: u64) -> Result<NonNull<u8>> {
        if len == 0 {
            return Err(Fault::Device {
                call: "elastic host span",
                why: "a span of no bytes has no address".to_string(),
            });
        }
        let end = offset.checked_add(len).ok_or(Fault::Ceiling {
            what: "bytes mapped under an elastic buffer",
            need: u64::MAX,
            have: self.committed,
        })?;
        if end > self.committed {
            return Err(Fault::Ceiling {
                what: "bytes mapped under an elastic buffer",
                need: end,
                have: self.committed,
            });
        }
        // Chunk `i` covers `[i * CHUNK, i * CHUNK + size)` of the buffer's
        // address space, in order, which is what `grow` builds: it fills the
        // last chunk before pushing another.
        let index = usize::try_from(offset / CHUNK).map_err(|_| Fault::Device {
            call: "elastic host span",
            why: format!("offset {offset} does not index a chunk on this host"),
        })?;
        let within = offset % CHUNK;
        let chunk = self.chunks.get(index).ok_or(Fault::Ceiling {
            what: "bytes mapped under an elastic buffer",
            need: end,
            have: self.committed,
        })?;
        if within + len > chunk.bytes {
            return Err(Fault::Device {
                call: "elastic host span",
                why: format!(
                    "{len} bytes at {offset} cross the end of chunk {index}, which is \
                     a different heap; ask for the two sides separately"
                ),
            });
        }
        let base = chunk.alias.contents();
        // SAFETY: `within + len <= chunk.bytes`, and the alias is a buffer
        // over the whole heap, so the offset is inside the allocation.
        let at = unsafe { base.as_ptr().cast::<u8>().add(usize_of(within)) };
        NonNull::new(at).ok_or(Fault::Device {
            call: "elastic host span",
            why: "the heap alias has no host address".to_string(),
        })
    }

    /// Walk `bytes` at `offset` one chunk at a time, handing `piece` the
    /// host address and length of each.
    fn walk_spans(
        &self,
        offset: u64,
        bytes: u64,
        mut piece: impl FnMut(NonNull<u8>, u64, u64) -> Result<()>,
    ) -> Result<()> {
        let mut done = 0;
        while done < bytes {
            let at = offset + done;
            let take = head_in_chunk(at, bytes - done, CHUNK);
            let span = self.host_span(at, take)?;
            piece(span, done, take)?;
            done += take;
        }
        Ok(())
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
    /// mapped. What was already cleared stays cleared.
    ///
    /// # Safety
    ///
    /// Nothing may be reading these bytes on the GPU. The pages are host
    /// addressable, not host owned -- between frames the host owns them
    /// outright, during one it does not.
    pub unsafe fn zero(&self, offset: u64, bytes: u64) -> Result<()> {
        self.walk_spans(offset, bytes, |span, _, take| {
            // SAFETY: `host_span` returned `take` writable bytes there.
            unsafe { std::ptr::write_bytes(span.as_ptr(), 0, usize_of(take)) };
            Ok(())
        })
    }

    /// Copy the mapped bytes at `offset` into `into`.
    ///
    /// # Errors
    ///
    /// As [`host_span`](Self::host_span), for the first span that is not
    /// mapped.
    pub fn read_into(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.walk_spans(offset, into.len() as u64, |span, done, take| {
            let into = &mut into[usize_of(done)..usize_of(done + take)];
            // SAFETY: `host_span` returned `take` readable bytes there, and
            // `into` is a distinct host allocation of that many bytes.
            unsafe { std::ptr::copy_nonoverlapping(span.as_ptr(), into.as_mut_ptr(), into.len()) };
            Ok(())
        })
    }

    /// Copy `from` into the mapped bytes at `offset`.
    ///
    /// # Errors
    ///
    /// As [`host_span`](Self::host_span), for the first span that is not
    /// mapped. Partial: what was written before the refusal stays written.
    ///
    /// # Safety
    ///
    /// As [`zero`](Self::zero).
    pub unsafe fn write_from(&self, offset: u64, from: &[u8]) -> Result<()> {
        self.walk_spans(offset, from.len() as u64, |span, done, take| {
            let from = &from[usize_of(done)..usize_of(done + take)];
            // SAFETY: `host_span` returned `take` writable bytes there, and
            // `from` is a distinct host allocation of that many bytes.
            unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), span.as_ptr(), from.len()) };
            Ok(())
        })
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

    /// Record that a mapping over this buffer lands at `through` on the
    /// mapping timeline. Monotonic: the timeline only advances, and a lower
    /// value would let teardown stop waiting before the newest operation.
    fn fence_at(&mut self, through: u64) {
        match &mut self.fence {
            Some(fence) => fence.through = fence.through.max(through),
            None => {
                self.fence = Some(Fence {
                    event: self.sparse.event().clone(),
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

#[cfg(target_vendor = "apple")]
impl Drop for Elastic {
    fn drop(&mut self) {
        // The mappings are not torn down here: an unmap is a GPU operation
        // needing a timeline, and releasing the sparse buffer releases the
        // address space they lived in, so there is nothing left to unmap
        // from. What CANNOT be skipped is the check that the last remap has
        // landed: the heaps go when their `Chunk`s do, a few lines below,
        // and freeing them under a live mapping operation is not a leak and
        // not a wrong answer -- it is a GPU page fault, raised inside the
        // Metal driver, which takes the machine down rather than this
        // process.
        if self.drained() {
            let residency = self.sparse.residency();
            residency.removeAllocation(ProtocolObject::from_ref(&*self.buffer));
            for chunk in &self.chunks {
                residency.removeAllocation(ProtocolObject::from_ref(&*chunk.heap));
            }
            residency.commit();
        } else {
            // The timeline stopped moving, so nothing will ever prove the
            // mapping landed. Leak the buffer and every heap rather than
            // free them: a leak costs this process memory it was already
            // holding, and the alternative costs the machine.
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
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.budget.reserved = state.budget.reserved.saturating_sub(self.committed);
        state.budget.committed = state.budget.committed.saturating_sub(self.committed);
    }
}

#[cfg(target_vendor = "apple")]
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
#[cfg(target_vendor = "apple")]
fn make_chunk(sparse: &Sparse, bytes: u64) -> Result<Chunk> {
    let descriptor = MTLHeapDescriptor::new();
    descriptor.setType(MTLHeapType::Placement);
    // Shared, not Private: on this hardware there is one pool of memory, and
    // a Shared placement heap is what lets the host stage into a KV page
    // without a second copy. The sparse buffer over it is still Private.
    descriptor.setStorageMode(MTLStorageMode::Shared);
    // Untracked: the sparse buffer over these tiles is the resource the
    // encoders bind and track; a second set of hazards on the heap would be
    // a dependency already expressed, per allocation, forever.
    descriptor.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
    descriptor.setSize(usize_of(bytes));
    descriptor.setMaxCompatiblePlacementSparsePageSize(MTLSparsePageSize::Size256);

    let heap = sparse
        .device()
        .newHeapWithDescriptor(&descriptor)
        .ok_or_else(|| Fault::Device {
            call: "newHeapWithDescriptor",
            why: format!("a placement heap of {bytes} bytes was refused"),
        })?;
    // SAFETY: the offset is zero and the length is the heap's own size, so
    // the placement is in bounds by construction; the descriptor above
    // declared the heap compatible with this storage mode.
    let alias = unsafe {
        heap.newBufferWithLength_options_offset(
            usize_of(bytes),
            MTLResourceOptions::StorageModeShared,
            0,
        )
    }
    .ok_or_else(|| Fault::Device {
        call: "newBufferWithLength:options:offset:",
        why: format!("an alias over a {bytes}-byte placement heap was refused"),
    })?;

    let residency = sparse.residency();
    residency.addAllocation(ProtocolObject::from_ref(&*heap));
    residency.commit();

    Ok(Chunk {
        heap,
        alias,
        bytes,
        mapped: 0,
    })
}

/// Create a sparse buffer of `len` bytes in `arena`.
///
/// The buffer starts with nothing mapped; [`grow_all`] is what attaches
/// memory. Creating it costs address space and a residency entry, not
/// memory, which is why the size can be the worst case even when the usage
/// will not be.
///
/// # Errors
///
/// If Metal refuses the sparse buffer. A zero `len` is an error rather than
/// an empty buffer: a zero-length buffer has no address, and the address is
/// the only thing this type promises.
#[cfg(target_vendor = "apple")]
pub fn create(sparse: &Arc<Sparse>, arena: &Arena, len: u64) -> Result<Elastic> {
    if len == 0 {
        return Err(Fault::Device {
            call: "elastic buffer",
            why: "a zero-length sparse buffer has no address to promise".to_string(),
        });
    }
    let virtual_bytes = tiles_up(len);
    // SAFETY: the length is a whole number of tiles at the page size given,
    // which is what this call requires; nothing is mapped yet, so there is no
    // aliasing to violate.
    let buffer = unsafe {
        sparse
            .device()
            .newBufferWithLength_options_placementSparsePageSize(
                usize_of(virtual_bytes),
                MTLResourceOptions::StorageModePrivate,
                MTLSparsePageSize::Size256,
            )
    }
    .ok_or_else(|| Fault::Device {
        call: "newBufferWithLength:options:placementSparsePageSize:",
        why: format!("{virtual_bytes} bytes of sparse address space were refused"),
    })?;

    let residency = sparse.residency();
    residency.addAllocation(ProtocolObject::from_ref(&*buffer));
    residency.commit();

    Ok(Elastic {
        buffer,
        virtual_bytes,
        len,
        committed: 0,
        chunks: Vec::new(),
        owner: Arc::downgrade(&arena.state),
        sparse: sparse.clone(),
        fence: None,
    })
}

/// Queue the remaps that attach memory to `buffer` until at least `bytes`
/// of it is mapped, charging the arena as it goes.
///
/// Returns the newly mapped range, or `None` when nothing had to change.
/// Idempotent: an ask below what is already mapped costs nothing, which is
/// what lets a caller ask on every step. The remaps have been issued, not
/// landed, when this returns -- [`grow_all`] is the caller that flushes.
#[cfg(target_vendor = "apple")]
fn grow(buffer: &mut Elastic, bytes: u64) -> Result<Option<(u64, u64)>> {
    if bytes > buffer.len {
        return Err(Fault::Ceiling {
            what: "bytes of an elastic buffer",
            need: bytes,
            have: buffer.len,
        });
    }
    let target = tiles_up(bytes).min(buffer.virtual_bytes);
    if target <= buffer.committed {
        return Ok(None);
    }
    let from = buffer.committed;
    let Some(state) = buffer.owner.upgrade() else {
        return Err(Fault::Device {
            call: "elastic growth",
            why: "the arena this buffer belongs to is gone".to_string(),
        });
    };
    let sparse = buffer.sparse.clone();
    // A refcount bump so the sparse buffer can be handed to the mapping
    // queue while `buffer.chunks` is borrowed mutably.
    let target_buffer = buffer.buffer.clone();
    while buffer.committed < target {
        if buffer
            .chunks
            .last()
            .is_none_or(|chunk| chunk.mapped == chunk.bytes)
        {
            let offset = buffer.chunks.len() as u64 * CHUNK;
            let size = CHUNK.min(buffer.virtual_bytes - offset);
            match make_chunk(&sparse, size) {
                Ok(chunk) => buffer.chunks.push(chunk),
                Err(error) => {
                    // Un-charge only what has not been mapped. The tiles that
                    // did land are real and still owed for.
                    let mut state = state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    state.budget.reserved = state
                        .budget
                        .reserved
                        .saturating_sub(target - buffer.committed);
                    return Err(error);
                }
            }
        }

        let chunk = buffer.chunks.last_mut().expect("just pushed if empty");
        let grow = (target - buffer.committed).min(chunk.bytes - chunk.mapped);
        sparse.issue(
            &target_buffer,
            Some(&chunk.heap),
            MTLSparseTextureMappingMode::Map,
            buffer.committed / TILE,
            grow / TILE,
            chunk.mapped / TILE,
        );
        chunk.mapped += grow;
        buffer.committed += grow;

        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.budget.committed += grow;
        state.budget.high_water = state.budget.high_water.max(state.budget.committed);
    }
    Ok(Some((from, buffer.committed)))
}

/// One buffer and how many bytes of it a batch growth wants mapped.
#[cfg(target_vendor = "apple")]
pub struct Target<'a> {
    pub buffer: &'a mut Elastic,
    pub bytes: u64,
}

/// Attach memory to every target until at least its `bytes` are mapped,
/// as one priced, waited-for batch.
///
/// The batch is priced once against one arena before anything is mapped: a
/// batch that does not fit is refused whole, with every buffer untouched,
/// rather than mapped part-way and refused at the buffer that broke the
/// budget. The remaps are then issued together and flushed once, so a
/// decode that grows every layer's plane by a page costs one signal and one
/// host wait rather than one per layer. Newly mapped bytes are zeroed
/// before this returns, because a placement heap's memory is whatever it
/// was last used for and a kv page that reads stale keys is a wrong answer
/// nothing downstream can detect.
///
/// Returns, per target, the range that was newly mapped.
///
/// # Errors
///
/// [`Fault::Ceiling`] when the batch exceeds the arena's headroom, or names
/// bytes past a buffer's length; [`Fault::Device`] when a heap or the
/// mapping queue refuses, or when the batch spans two arenas -- two budgets
/// cannot be priced as one.
#[cfg(target_vendor = "apple")]
pub fn grow_all(targets: &mut [Target<'_>]) -> Result<Vec<Option<(u64, u64)>>> {
    let mut arena: Option<Arena> = None;
    let mut delta = 0u64;
    for target in targets.iter() {
        if target.bytes > target.buffer.len {
            return Err(Fault::Ceiling {
                what: "bytes of an elastic buffer",
                need: target.bytes,
                have: target.buffer.len,
            });
        }
        let want = tiles_up(target.bytes).min(target.buffer.virtual_bytes);
        delta = delta.saturating_add(want.saturating_sub(target.buffer.committed));
        let Some(mine) = target.buffer.arena() else {
            return Err(Fault::Device {
                call: "elastic growth",
                why: "the arena this buffer belongs to is gone".to_string(),
            });
        };
        match &arena {
            Some(first) if !first.is(&mine) => {
                return Err(Fault::Device {
                    call: "elastic growth",
                    why: "a batch growth spans two arenas, which cannot be priced or \
                          rolled back as one"
                        .to_string(),
                });
            }
            Some(_) => {}
            None => arena = Some(mine),
        }
    }
    if delta == 0 {
        return Ok(targets.iter().map(|_| None).collect());
    }
    let arena = arena.expect("a non-zero delta came from some target");
    {
        let mut state = arena.lock();
        if delta > state.budget.headroom() {
            return Err(Fault::Ceiling {
                what: "bytes of elastic kv memory",
                need: state.budget.reserved.saturating_add(delta),
                have: state.budget.total,
            });
        }
        // Charged BEFORE the mapping, so that a second ask arriving while
        // this one is still attaching heaps cannot be told the same bytes
        // are free. `grow` gives back what a refused heap leaves unmapped.
        state.budget.reserved += delta;
    }

    let mut grown = Vec::with_capacity(targets.len());
    let mut first_fault = None;
    for (at, target) in targets.iter_mut().enumerate() {
        match grow(target.buffer, target.bytes) {
            Ok(range) => grown.push(range),
            Err(fault) => {
                // `grow` gave back what the failed buffer left unmapped;
                // the buffers after it were charged and never asked.
                let untouched: u64 = targets[at + 1..]
                    .iter()
                    .map(|target| {
                        tiles_up(target.bytes)
                            .min(target.buffer.virtual_bytes)
                            .saturating_sub(target.buffer.committed)
                    })
                    .sum();
                let mut state = arena.lock();
                state.budget.reserved = state.budget.reserved.saturating_sub(untouched);
                first_fault = Some(fault);
                break;
            }
        }
    }
    // Flushed even on a refusal: whatever was issued has to land before the
    // heaps it names can be trusted or released.
    let sparse = targets
        .first()
        .map(|target| target.buffer.sparse.clone())
        .expect("a non-zero delta came from some target");
    let through = sparse.flush()?;
    for target in targets.iter_mut() {
        target.buffer.fence_at(through);
    }
    if let Some(fault) = first_fault {
        return Err(fault);
    }
    for (target, range) in targets.iter().zip(&grown) {
        if let Some((from, to)) = *range {
            // SAFETY: the tiles were mapped by the flush just waited for,
            // and no frame has been encoded against them yet -- they did not
            // exist a moment ago.
            unsafe { target.buffer.zero(from, to - from)? };
        }
    }
    Ok(grown)
}

/// Queue the remaps that detach memory from `buffer` down to `bytes`, and
/// give back the accounting.
///
/// Returns whether anything was issued, and the heaps that emptied -- the
/// caller holds those until the unmap has landed.
#[cfg(target_vendor = "apple")]
fn shrink(buffer: &mut Elastic, bytes: u64) -> (bool, Vec<Chunk>) {
    let target = tiles_up(bytes).min(buffer.virtual_bytes);
    if target >= buffer.committed {
        return (false, Vec::new());
    }
    let target_buffer = buffer.buffer.clone();
    let mut released = 0u64;
    let mut emptied = Vec::new();

    while buffer.committed > target {
        let Some(chunk) = buffer.chunks.last_mut() else {
            break;
        };
        let shrink = (buffer.committed - target).min(chunk.mapped);
        buffer.sparse.issue(
            &target_buffer,
            None,
            MTLSparseTextureMappingMode::Unmap,
            (buffer.committed - shrink) / TILE,
            shrink / TILE,
            0,
        );
        chunk.mapped -= shrink;
        buffer.committed -= shrink;
        released += shrink;
        if chunk.mapped == 0 {
            emptied.push(buffer.chunks.pop().expect("just inspected"));
        }
    }

    if let Some(state) = buffer.owner.upgrade() {
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.budget.committed = state.budget.committed.saturating_sub(released);
        state.budget.reserved = state.budget.reserved.saturating_sub(released);
    }
    (true, emptied)
}

/// Detach memory from every target down to its `bytes`, as one waited-for
/// batch, and give the emptied heaps back.
///
/// # Safety
///
/// No frame may be reading the tiles this takes away: the caller has
/// drained every command buffer that could touch these buffers. The mapping
/// queue cannot check that, because the frame queue signals nothing.
///
/// # Errors
///
/// [`Fault::Device`] when the mapping queue does not reach the unmap; the
/// heaps then stay pending rather than being freed under it.
#[cfg(target_vendor = "apple")]
pub unsafe fn shrink_all(targets: &mut [Target<'_>]) -> Result<()> {
    let mut issued = false;
    let mut emptied: Vec<(Option<Arena>, Vec<Chunk>)> = Vec::new();
    for target in targets.iter_mut() {
        let (did, chunks) = shrink(target.buffer, target.bytes);
        issued |= did;
        if !chunks.is_empty() {
            emptied.push((target.buffer.arena(), chunks));
        }
    }
    if !issued {
        return Ok(());
    }
    let sparse = targets
        .first()
        .map(|target| target.buffer.sparse.clone())
        .expect("something was issued from some target");
    let through = sparse.flush()?;
    for target in targets.iter_mut() {
        target.buffer.fence_at(through);
    }
    for (arena, chunks) in emptied {
        let Some(arena) = arena else {
            // No arena to hold them pending; the flush above proved the
            // unmap landed, so they can go now.
            drop(chunks);
            continue;
        };
        arena
            .lock()
            .pending
            .extend(chunks.into_iter().map(|chunk| Pending { through, chunk }));
        arena.collect(sparse.signalled(), sparse.residency());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elastic_every_case() {
        a_length_is_rounded_up_to_a_whole_tile_because_down_would_fault();
        pages_are_the_reporting_unit_and_an_empty_pool_reports_none();
        the_three_sizes_are_distinct_and_nest();
        headroom_counts_what_is_promised_and_not_only_what_is_mapped();
        headroom_saturates_rather_than_wrapping_when_over_committed();
        a_move_inside_one_chunk_is_handed_over_whole();
        a_move_is_cut_at_whichever_side_reaches_a_seam_first();
        an_overlapping_move_slides_rather_than_smearing_across_a_seam();
        a_move_onto_itself_touches_nothing();
    }

    fn a_length_is_rounded_up_to_a_whole_tile_because_down_would_fault() {
        assert_eq!(tiles_up(0), 0);
        assert_eq!(tiles_up(1), TILE);
        assert_eq!(tiles_up(TILE), TILE);
        assert_eq!(tiles_up(TILE + 1), 2 * TILE);
        assert_eq!(pages_up(1), PAGE);
        assert_eq!(pages_up(PAGE + 1), 2 * PAGE);
    }

    fn pages_are_the_reporting_unit_and_an_empty_pool_reports_none() {
        assert_eq!(pages_for_bytes(0), 0, "an empty pool must report nothing");
        assert_eq!(pages_for_bytes(1), 1);
        assert_eq!(pages_for_bytes(PAGE), 1);
        assert_eq!(pages_for_bytes(PAGE + 1), 2);
    }

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

    fn headroom_counts_what_is_promised_and_not_only_what_is_mapped() {
        let b = Budget {
            total: 1000,
            // Promised but not yet mapped. Counting `committed` here would
            // let two asks in flight both be told there is room for the same
            // bytes.
            reserved: 800,
            committed: 100,
            high_water: 100,
        };
        assert_eq!(b.headroom(), 200);
    }

    fn headroom_saturates_rather_than_wrapping_when_over_committed() {
        let b = Budget {
            total: 100,
            reserved: 500,
            committed: 500,
            high_water: 500,
        };
        assert_eq!(
            b.headroom(),
            0,
            "an over-committed arena must report no room, not four exabytes"
        );
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
                "a {len}-byte move between {src} and {dst} does not overlap, \
                 so it cannot tell the two walk directions apart"
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

    fn a_move_onto_itself_touches_nothing() {
        let mut pieces = 0;
        walk_move(48, 48, 16, 64, |_, _, _| {
            pieces += 1;
            Ok(())
        })
        .expect("walk");
        assert_eq!(
            pieces, 0,
            "a move to where the bytes already are is not a move"
        );
    }
}
