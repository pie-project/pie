//! A pool of short-lived buffers, recycled instead of reallocated.
//!
//! # Why these do not come out of the heap
//!
//! [`Heap`](crate::gpu::device::heap::Heap) is a bump allocator: it hands out placements
//! in order and never takes one back. That is right for the weights and the
//! KV cache, which live as long as the model does, and wrong for the buffers
//! a step needs for the length of one forward pass -- a scratch reduction, a
//! staged set of arguments, a scatter index. Those recur every token, and a
//! bump allocator asked for them runs out.
//!
//! # Why they are not just allocated per step either
//!
//! `newBufferWithLength:` is a device allocation, and every one of them has
//! to be added to the residency set, which then has to be committed. On the
//! measured trace that pair cost 0.375 ms out of a 1.2 ms gap between two
//! forward passes -- for buffers a few hundred bytes wide. So they are kept.
//!
//! # Recycling is not optional here, it is automatic
//!
//! The C++ shell keys an `in_use` flag by raw pointer, because a `void*` can
//! be recycled twice, or never. Neither is expressible here: [`Transient`]
//! owns its buffer and returns it to the pool when it is dropped. Recycling
//! twice does not compile, and forgetting to recycle is not a leak.
//!
//! What that does NOT change is the hazard it shares with the C++: a buffer
//! is reusable the moment the host is done with it, and the host is done with
//! it before the GPU is. Dropping a [`Transient`] whose step has not signalled
//! hands the next step a buffer the previous one is still reading. The pool
//! cannot see that; the step boundary is what enforces it.
//!
//! # Size classes, and why the cache depth is not flat
//!
//! Powers of two from 256 bytes, so a pass that asks for 300 bytes and then
//! 400 gets the same buffer. The per-class cache depth is derived from the
//! class rather than fixed: one measured pass acquires about thirteen buffers
//! and most land in the smallest classes, so a flat depth of eight overflowed
//! on every pass -- and the overflow path is the expensive one, releasing a
//! buffer and re-allocating it next time. The byte budget is what actually
//! bounds the pool; the depth only stops one class hoarding it.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLAllocation, MTLBuffer, MTLDevice, MTLResidencySet, MTLResourceOptions};

use crate::gpu::device::context::Context;
use crate::error::{Error, Result};

/// The smallest size class, and the alignment every buffer therefore has.
pub const SMALLEST_CLASS: u64 = 256;

/// The default byte budget for cached and outstanding buffers together.
pub const DEFAULT_CAPACITY: u64 = 1 << 30;

/// How many buffers of one class are worth keeping.
///
/// A megabyte's worth, clamped: enough that a pass which leans on one class
/// does not overflow, few enough that a large class cannot fill the budget by
/// itself.
fn cache_depth(class: u64) -> usize {
    let by_bytes = (1u64 << 20) / class.max(1);
    usize::try_from(by_bytes.clamp(8, 64)).unwrap_or(8)
}

/// The smallest power-of-two class that holds `size` bytes.
fn size_class(size: u64) -> Option<u64> {
    if size == 0 {
        return None;
    }
    let mut class = SMALLEST_CLASS;
    while class < size {
        class = class.checked_mul(2)?;
    }
    Some(class)
}

/// A buffer on loan from a [`Pool`].
///
/// Returns itself when dropped. See the module docs for the one thing that
/// does not make safe: dropping it before the step that reads it has
/// signalled.
pub struct Transient {
    buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    residency: Registrar,
    contents: NonNull<c_void>,
    gpu_address: u64,
    /// What the caller asked for.
    size: u64,
    /// What was actually allocated, which is what the pool books.
    class: u64,
    owner: Weak<Mutex<State>>,
}

impl Transient {
    /// The Metal buffer, for binding.
    #[must_use]
    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        // The option is Some for the whole life of the value; `take` happens
        // only in `Drop`, which nothing can observe from.
        self.buffer
            .as_deref()
            .expect("a live transient has a buffer")
    }

    /// The GPU virtual address, for an argument table entry.
    #[must_use]
    pub const fn gpu_address(&self) -> u64 {
        self.gpu_address
    }

    /// The CPU-visible bytes.
    ///
    /// A pointer rather than a slice, for the same reason as
    /// [`Slot::contents`](crate::gpu::device::heap::Slot::contents): the GPU may be reading
    /// them.
    #[must_use]
    pub const fn contents(&self) -> NonNull<c_void> {
        self.contents
    }

    /// Length in bytes, as the caller asked for it.
    ///
    /// Not the size class. The rest of the class belongs to no one, and a
    /// previous borrower's bytes are still in it.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.size
    }

    /// Whether the loan is empty, which a real one never is.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// The size class this was drawn from.
    #[must_use]
    pub const fn class(&self) -> u64 {
        self.class
    }
}

impl Drop for Transient {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        let Some(owner) = self.owner.upgrade() else {
            // The pool is gone, so there is nothing to hand back to -- but the
            // residency set outlives both and would keep this buffer alive.
            self.residency.unregister_one(&buffer);
            return;
        };
        let mut state = owner.lock().unwrap_or_else(|e| e.into_inner());
        let residency = Registrar(self.residency.0.clone());
        state.give_back(Cached {
            buffer,
            residency,
            contents: self.contents,
            gpu_address: self.gpu_address,
            class: self.class,
        });
    }
}

impl std::fmt::Debug for Transient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Transient")
            .field("len", &self.size)
            .field("class", &self.class)
            .field("gpu_address", &format_args!("{:#x}", self.gpu_address))
            .finish_non_exhaustive()
    }
}

/// A buffer sitting in the pool rather than in a caller's hands.
struct Cached {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    contents: NonNull<c_void>,
    gpu_address: u64,
    class: u64,
    /// Carried by the buffer rather than held by the pool so that every way
    /// out -- eviction, drain, the pool being dropped, a loan outliving the
    /// pool -- has what it needs to unregister, instead of three of the four
    /// having it and one leaking.
    residency: Registrar,
}

// SAFETY: the pointer is a shared-storage buffer's contents, valid for as long
// as the buffer it came from, which is held beside it. `Retained` for a Metal
// object is itself sendable; the raw pointer is what stops the derive.
unsafe impl Send for Cached {}

/// The residency set a buffer was added to, carried so it can be taken out.
struct Registrar(Retained<ProtocolObject<dyn MTLResidencySet>>);

impl Registrar {
    /// Take `buffer` out of the set. The caller commits.
    ///
    /// Split from the commit because a commit is the expensive half and a
    /// batch of removals only needs one.
    fn unregister(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        let allocation: &ProtocolObject<dyn MTLAllocation> = ProtocolObject::from_ref(buffer);
        self.0.removeAllocation(allocation);
    }

    /// Take `buffer` out of the set and commit, for a lone buffer.
    fn unregister_one(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        self.unregister(buffer);
        self.0.commit();
    }
}

// SAFETY: a `Retained` Metal object is sendable -- the same claim `Cached`
// makes above for the buffers this set names. objc2 declines to derive it for
// a protocol object because one could stand for a main-thread-only class, and
// a residency set is not one.
unsafe impl Send for Registrar {}

/// What the pool has done, for a caller that wants to know.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Buffers created from the device.
    pub allocations: u64,
    /// Acquisitions served from the cache instead.
    pub reuse_hits: u64,
    /// Buffers handed back.
    pub recycles: u64,
    /// Buffers released because the budget or the depth said so.
    pub evictions: u64,
    /// Acquisitions refused.
    pub refusals: u64,
    /// Buffers the pool is responsible for, cached and outstanding.
    pub resident_buffers: usize,
    /// Their bytes, counted by size class.
    pub resident_bytes: u64,
    /// Buffers sitting in the free lists.
    pub cached_buffers: usize,
    /// Their bytes.
    pub cached_bytes: u64,
    /// The high-water mark of `resident_bytes`.
    pub peak_resident_bytes: u64,
    /// The budget `resident_bytes` is kept under.
    pub capacity_bytes: u64,
}

impl PoolStats {
    /// Buffers currently in a caller's hands.
    #[must_use]
    pub const fn outstanding_buffers(&self) -> usize {
        self.resident_buffers - self.cached_buffers
    }

    /// Their bytes, by size class.
    #[must_use]
    pub const fn outstanding_bytes(&self) -> u64 {
        self.resident_bytes - self.cached_bytes
    }
}

struct State {
    free: BTreeMap<u64, Vec<Cached>>,
    stats: PoolStats,
}

impl State {
    /// Take a buffer back, cache it if there is room, release it if not.
    fn give_back(&mut self, cached: Cached) {
        self.stats.recycles += 1;
        let class = cached.class;
        let depth = cache_depth(class);
        let bucket = self.free.entry(class).or_default();
        if bucket.len() < depth && self.stats.resident_bytes <= self.stats.capacity_bytes {
            bucket.push(cached);
            self.stats.cached_buffers += 1;
            self.stats.cached_bytes += class;
            return;
        }
        self.release(cached);
    }

    /// Drop a buffer the pool was counting, whether cached or outstanding.
    fn release(&mut self, cached: Cached) {
        self.release_all(vec![cached]);
    }

    /// Drop several buffers the pool was counting, for one residency commit.
    ///
    /// A buffer that is only dropped is not released: the residency set holds
    /// its own reference to every allocation it names, so the memory survives
    /// the `Retained` going away and the set grows without bound. It has to be
    /// taken out of the set first.
    ///
    /// The commit lands before the buffers drop, and that order is the point:
    /// the set must stop naming an allocation while the allocation is still
    /// there. Doing them together is also why this takes a batch -- eviction
    /// releases in a loop, and one commit for the loop beats one per buffer.
    fn release_all(&mut self, doomed: Vec<Cached>) {
        if doomed.is_empty() {
            return;
        }
        for cached in &doomed {
            self.stats.resident_buffers -= 1;
            self.stats.resident_bytes -= cached.class;
            self.stats.evictions += 1;
            cached.residency.unregister(&cached.buffer);
        }
        // One commit for the batch: every buffer here came from the same
        // context, so they are all edits to the same set.
        doomed[0].residency.0.commit();
        drop(doomed);
    }

    /// Release cached buffers, largest class first, until `wanted` fits.
    ///
    /// Largest first because that is the fewest objects released per byte
    /// recovered, and every release is a residency-set edit.
    fn make_room(&mut self, wanted: u64) {
        let mut doomed = Vec::new();
        // Counted here rather than taken off `resident_bytes` because
        // `release_all` is what takes it off, once, below.
        let mut freed = 0;
        while self.stats.resident_bytes - freed + wanted > self.stats.capacity_bytes {
            let Some(class) = self
                .free
                .iter()
                .rev()
                .find(|(_, bucket)| !bucket.is_empty())
                .map(|(class, _)| *class)
            else {
                break;
            };
            let Some(evicted) = self.free.get_mut(&class).and_then(Vec::pop) else {
                break;
            };
            self.stats.cached_buffers -= 1;
            self.stats.cached_bytes -= class;
            freed += class;
            doomed.push(evicted);
        }
        self.release_all(doomed);
    }
}

impl Drop for State {
    /// The residency set outlives the pool, so a pool that simply went away
    /// would leave every buffer it had cached named in the set forever.
    fn drop(&mut self) {
        let doomed: Vec<Cached> = std::mem::take(&mut self.free)
            .into_values()
            .flatten()
            .collect();
        self.release_all(doomed);
    }
}

/// Short-lived buffers, kept between steps.
///
/// Cheap to clone; a clone is the same pool.
#[derive(Clone)]
pub struct Pool {
    state: Arc<Mutex<State>>,
}

impl Default for Pool {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

impl Pool {
    /// A pool that will hold at most `capacity_bytes` of buffers.
    ///
    /// The budget covers cached AND outstanding buffers, because both are
    /// resident and it is residency the device is counting.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                free: BTreeMap::new(),
                stats: PoolStats {
                    capacity_bytes: capacity_bytes.max(SMALLEST_CLASS),
                    ..PoolStats::default()
                },
            })),
        }
    }

    /// Borrow a buffer of at least `size` bytes.
    ///
    /// Served from the cache when a buffer of the same class is free.
    pub fn acquire(&self, context: &Context, size: u64) -> Result<Transient> {
        let class = size_class(size).ok_or_else(|| Error::Create {
            what: "transient buffer",
            message: if size == 0 {
                "a buffer of no bytes".to_string()
            } else {
                format!("{size} bytes has no power-of-two size class")
            },
        })?;

        let mut state = self.locked();
        if class > state.stats.capacity_bytes {
            state.stats.refusals += 1;
            let capacity = state.stats.capacity_bytes;
            return Err(Error::Create {
                what: "transient buffer",
                message: format!(
                    "{size} bytes rounds to a {class}-byte class, past the pool's {capacity}"
                ),
            });
        }

        if let Some(cached) = state.free.get_mut(&class).and_then(Vec::pop) {
            state.stats.reuse_hits += 1;
            state.stats.cached_buffers -= 1;
            state.stats.cached_bytes -= class;
            return Ok(self.lend(cached, size));
        }

        state.make_room(class);
        if state.stats.resident_bytes + class > state.stats.capacity_bytes {
            state.stats.refusals += 1;
            let (used, capacity) = (state.stats.resident_bytes, state.stats.capacity_bytes);
            return Err(Error::Create {
                what: "transient buffer",
                message: format!(
                    "{class} bytes on top of {used} would pass the pool's {capacity}; \
                     the cache is empty, so the rest is in a caller's hands"
                ),
            });
        }

        // Released before touching the device: allocation is slow, and a
        // handler or another thread must not be blocked behind it.
        drop(state);
        let cached = Self::create(context, class)?;

        let mut state = self.locked();
        state.stats.allocations += 1;
        state.stats.resident_buffers += 1;
        state.stats.resident_bytes += class;
        state.stats.peak_resident_bytes = state
            .stats
            .peak_resident_bytes
            .max(state.stats.resident_bytes);
        Ok(self.lend(cached, size))
    }

    /// Create a shared-storage buffer and make it resident.
    ///
    /// The residency set grows across its lifetime -- adding and committing
    /// after the initial build is supported, which is what lets these exist at
    /// all alongside a heap that was made resident once.
    fn create(context: &Context, class: u64) -> Result<Cached> {
        context.check_working_set(class)?;
        let length = usize::try_from(class).map_err(|_| Error::Create {
            what: "transient buffer",
            message: format!("{class} bytes does not fit this host's usize"),
        })?;
        let buffer = context
            .device()
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| Error::Create {
                what: "transient buffer",
                message: format!("the device declined {class} bytes"),
            })?;
        let contents = buffer.contents();
        let gpu_address = buffer.gpuAddress();

        let allocation: &ProtocolObject<dyn MTLAllocation> = ProtocolObject::from_ref(&*buffer);
        context.residency().addAllocation(allocation);
        context.residency().commit();
        context.residency().requestResidency();

        Ok(Cached {
            buffer,
            contents,
            gpu_address,
            class,
            residency: Registrar(context.residency_handle()),
        })
    }

    fn lend(&self, cached: Cached, size: u64) -> Transient {
        Transient {
            buffer: Some(cached.buffer),
            residency: cached.residency,
            contents: cached.contents,
            gpu_address: cached.gpu_address,
            size,
            class: cached.class,
            owner: Arc::downgrade(&self.state),
        }
    }

    /// What the pool has done so far.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        self.locked().stats
    }

    /// Change the byte budget, releasing cached buffers to meet it.
    ///
    /// Only cached buffers can go. Outstanding ones are in a caller's hands
    /// and possibly in a running step, so lowering the budget below what is
    /// outstanding leaves it over budget until those come back.
    pub fn set_capacity(&self, bytes: u64) {
        let mut state = self.locked();
        state.stats.capacity_bytes = bytes.max(SMALLEST_CLASS);
        state.make_room(0);
    }

    /// Release every cached buffer, keeping outstanding ones.
    pub fn drain(&self) {
        let mut state = self.locked();
        let doomed: Vec<Cached> = std::mem::take(&mut state.free)
            .into_values()
            .flatten()
            .collect();
        state.stats.cached_buffers -= doomed.len();
        state.stats.cached_bytes -= doomed.iter().map(|cached| cached.class).sum::<u64>();
        state.release_all(doomed);
    }

    fn locked(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("stats", &self.stats())
            .finish()
    }
}

// SAFETY: `contents` is the buffer's shared-storage pointer, and the loan
// holds the buffer alive until it is dropped. `size` is what the borrower
// asked for, which is never more than the class that was allocated. Only one
// `Transient` can name a given buffer at a time -- that is what the pool's
// ownership is for.
unsafe impl crate::Region for Transient {
    fn contents(&self) -> NonNull<c_void> {
        self.contents
    }

    fn len(&self) -> u64 {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::{SMALLEST_CLASS, cache_depth, size_class};

    #[test]
    fn a_size_rounds_up_to_its_power_of_two() {
        assert_eq!(size_class(0), None);
        assert_eq!(size_class(1), Some(SMALLEST_CLASS));
        assert_eq!(size_class(SMALLEST_CLASS), Some(SMALLEST_CLASS));
        assert_eq!(size_class(SMALLEST_CLASS + 1), Some(512));
        assert_eq!(size_class(300), Some(512));
        assert_eq!(size_class(400), Some(512), "300 and 400 must share a class");
        assert_eq!(size_class(1 << 20), Some(1 << 20));
    }

    #[test]
    fn a_size_no_power_of_two_can_hold_is_refused_rather_than_wrapping() {
        assert_eq!(size_class(u64::MAX), None);
        assert_eq!(size_class((1 << 63) + 1), None);
        assert_eq!(size_class(1 << 63), Some(1 << 63));
    }

    #[test]
    fn the_cache_is_deeper_for_small_classes() {
        assert_eq!(cache_depth(SMALLEST_CLASS), 64, "clamped at the top");
        assert_eq!(cache_depth(1 << 20), 8, "clamped at the bottom");
        assert_eq!(cache_depth(1 << 16), 16);
        assert!(
            cache_depth(1 << 14) > cache_depth(1 << 18),
            "a smaller class must not be cached less deeply"
        );
        assert_eq!(cache_depth(0), 64, "a zero class must not divide by zero");
    }
}
