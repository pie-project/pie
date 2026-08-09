//! GPU timestamps, as a heap that owns itself.
//!
//! A [`Timestamps`] is a Metal 4 counter heap of `count` timestamp entries.
//! [`super::StepEncoder::mark_timestamp`] writes one of them at an index
//! during encode, and [`Timestamps::resolve`] reads the whole heap back once
//! the step that wrote it has completed. That pair is what per-dispatch and
//! per-phase attribution is: the GPU stamps its own clock at points the host
//! chose, so the numbers do not include submit latency, encode time, or the
//! host's own wait.
//!
//! # Why this is a value and not a handle
//!
//! The C++ shell spells the same thing as `void* create_timestamp_heap`,
//! `resolve_timestamps(void*, ...)` and `release_timestamp_heap(void*)`, plus
//! a context-wide array that retains every heap ever created. Both halves of
//! that follow from the `void*`: a raw pointer cannot own an Objective-C
//! object, so something else has to retain it, and once something else
//! retains it there has to be a call that un-retains it -- otherwise the
//! array grows for the life of the process. Forgetting that call is a leak
//! that nothing reports.
//!
//! Here the heap is a `Retained` field. Dropping the [`Timestamps`] releases
//! it, there is no release function to forget, and the context keeps no list.
//! The bound also travels with the heap, which is what lets
//! [`super::StepEncoder::mark_timestamp`] range-check an index that the C++
//! could only pass through to Metal.
//!
//! # Zero is refused
//!
//! `create_timestamp_heap(0)` returns `nullptr` in the C++, and a null heap
//! makes every later `mark_timestamp` a no-op and every `resolve_timestamps`
//! a no-op. A caller that asked for no timestamps and then marks them has a
//! bug, and the C++ answers that bug with silence and a buffer of zeroes.
//! [`Timestamps::new`] answers it with [`Error::Create`].

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4CounterHeap, MTL4CounterHeapDescriptor, MTL4CounterHeapType, MTL4TimestampGranularity,
    MTL4TimestampHeapEntry, MTLDevice,
};

use crate::device::context::{Context, describe};
use crate::error::{Error, Result};

/// How precisely a timestamp is asked to be taken.
///
/// A named pair rather than the C++'s `bool precise`, which at a call site
/// reads as `mark_timestamp(heap, 3, true)` and says nothing about which of
/// the two behaviours `true` selects -- or that there is a cost attached to
/// one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Granularity {
    /// Lowest overhead, at the cost of precision.
    ///
    /// The default, and the right one for per-step attribution: Metal is free
    /// to sample at command-encoder boundaries, which means it does not have
    /// to split the encoder to honour the request. Keeping the encoder whole
    /// is what preserves the single-command-buffer model the stepper is built
    /// around, so a relaxed mark costs approximately nothing.
    #[default]
    Relaxed,
    /// As precise as Metal can manage, which may split the encoder.
    ///
    /// Worth it only for boundary-accurate sampling -- attributing a specific
    /// dispatch rather than a phase. A split encoder is a real cost paid on
    /// every step that carries the mark, so this is not the default.
    Precise,
}

impl From<Granularity> for MTL4TimestampGranularity {
    fn from(g: Granularity) -> Self {
        match g {
            Granularity::Relaxed => Self::Relaxed,
            Granularity::Precise => Self::Precise,
        }
    }
}

/// A counter heap of GPU timestamp entries, released when dropped.
///
/// See the module docs for why this owns the heap rather than handing out a
/// pointer to one the context retains forever.
pub struct Timestamps {
    heap: Retained<ProtocolObject<dyn MTL4CounterHeap>>,
    count: u32,
}

impl Timestamps {
    /// Allocate a heap of `count` timestamp entries.
    ///
    /// # Errors
    ///
    /// [`Error::Create`] if `count` is zero -- see the module docs; a heap of
    /// no entries can only ever be marked out of range -- or if the device
    /// declines the heap, which on this stack means the OS predates Metal 4
    /// counter heaps rather than that the allocation was too large.
    pub fn new(context: &Context, count: u32) -> Result<Self> {
        if count == 0 {
            return Err(Error::Create {
                what: "MTL4CounterHeap",
                message: "a heap of zero timestamps has no index that can be marked; asking for \
                          one is a caller bug, not a heap"
                    .to_string(),
            });
        }
        let descriptor = MTL4CounterHeapDescriptor::new();
        descriptor.setType(MTL4CounterHeapType::Timestamp);
        // SAFETY: the setter is `unsafe` only because Metal does not bound
        // the entry count. `count` is a `u32` and non-zero, which is well
        // inside what the driver will allocate.
        unsafe { descriptor.setCount(count as usize) };

        let heap = context
            .device()
            .newCounterHeapWithDescriptor_error(&descriptor)
            .map_err(|e| Error::Create {
                what: "MTL4CounterHeap",
                message: format!("{count} timestamp entries: {}", describe(&e)),
            })?;

        Ok(Self { heap, count })
    }

    /// How many entries the heap holds.
    ///
    /// The bound [`super::StepEncoder::mark_timestamp`] checks an index
    /// against.
    #[must_use]
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// The heap, for the encoder's write.
    pub(super) fn heap(&self) -> &ProtocolObject<dyn MTL4CounterHeap> {
        &self.heap
    }

    /// Read every entry back, in GPU ticks.
    ///
    /// Only meaningful after the step that wrote the marks has completed, and
    /// it is the caller's job to have run one: entries that were never
    /// written resolve to zero rather than to an error. What makes the read
    /// safe once a step HAS run is the stepper -- [`super::Stepper::run`]
    /// does not return until it has waited the shared event the step signals,
    /// and a signalled event is exactly the synchronisation Metal names as
    /// sufficient for a CPU-timeline resolve. So there is nothing to wait for
    /// here, and nothing here to get wrong by not waiting.
    ///
    /// The result is truncated to the number of entries the returned data
    /// actually holds. That is not defensive rounding: `resolveCounterRange:`
    /// returns an `NSData` whose length the driver chooses, and the entry
    /// size is a driver-side decision that the device reports separately
    /// through `sizeOfCounterHeapEntry:`. Trusting the requested count
    /// instead would read past the buffer the driver returned.
    ///
    /// # Errors
    ///
    /// [`Error::Create`] if `resolveCounterRange:` returns nil. The C++
    /// prints that to stderr and returns, leaving the caller with a buffer of
    /// zeroes it cannot distinguish from a step that ran and measured
    /// nothing.
    pub fn resolve(&self) -> Result<Vec<u64>> {
        let entry = size_of::<MTL4TimestampHeapEntry>();
        let range = NSRange::new(0, self.count as usize);
        // SAFETY: the range is `unsafe` because Metal does not bounds-check
        // it. It starts at 0 and runs for `self.count`, which is the count the
        // heap was created with and has not changed since.
        let data = unsafe { self.heap.resolveCounterRange(range) }.ok_or(Error::Create {
            what: "timestamp resolve",
            message: format!("the driver returned no data for {} entries", self.count),
        })?;

        // SAFETY: `data` is a freshly resolved, immutable NSData that nothing
        // else holds, so no Objective-C code can mutate it while the slice
        // lives.
        let bytes = unsafe { data.as_bytes_unchecked() };

        Ok(bytes
            .chunks_exact(entry)
            .take(self.count as usize)
            .map(|chunk| {
                let mut ticks = [0u8; 8];
                ticks.copy_from_slice(&chunk[..8]);
                u64::from_ne_bytes(ticks)
            })
            .collect())
    }
}

impl std::fmt::Debug for Timestamps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Timestamps")
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relaxed_is_the_granularity_a_caller_gets_without_choosing() {
        assert_eq!(Granularity::default(), Granularity::Relaxed);
    }

    #[test]
    fn each_granularity_maps_to_the_metal_constant_of_the_same_name() {
        assert_eq!(
            MTL4TimestampGranularity::from(Granularity::Relaxed),
            MTL4TimestampGranularity::Relaxed
        );
        assert_eq!(
            MTL4TimestampGranularity::from(Granularity::Precise),
            MTL4TimestampGranularity::Precise
        );
    }

    /// The truncation in [`Timestamps::resolve`] is arithmetic on the data
    /// the driver returned, and that arithmetic is testable without a device.
    #[test]
    fn a_timestamp_entry_is_one_u64_so_resolved_data_is_tightly_packed_ticks() {
        assert_eq!(size_of::<MTL4TimestampHeapEntry>(), size_of::<u64>());
    }
}
