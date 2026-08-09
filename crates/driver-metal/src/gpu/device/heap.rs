//! The placement heap, and the slots placed in it.
//!
//! One heap, Shared storage, hazard tracking off, and every buffer the driver
//! holds for the life of a model placed inside it at an offset this module
//! chose. The three decisions are worth stating because none of them is a
//! default:
//!
//! * **Placement.** An automatic heap picks offsets itself and will not tell
//!   you which; a placement heap is address space plus a promise, and the
//!   offsets are ours. That is what makes a slot's location reproducible from
//!   one run to the next, which is what makes a captured command buffer
//!   replayable.
//! * **Shared.** On UMA there is no copy to avoid, so `contents()` is a real
//!   CPU pointer into the same bytes the GPU reads. Weight staging and the
//!   per-step scalar writes both go through it.
//! * **Untracked.** Metal's automatic hazard tracking would insert barriers
//!   per resource; this driver knows its own dependency graph and issues them
//!   per dispatch. Leaving tracking on would pay for both.
//!
//! # Residency
//!
//! The heap is added to the residency set ONCE, as a whole. A placement
//! sub-buffer created later is inside a range that is already resident, so it
//! needs no registration of its own -- which is the property that makes
//! allocating mid-run cheap rather than a residency rebuild.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLDevice, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLResidencySet, MTLResourceOptions, MTLStorageMode,
};

use crate::gpu::device::context::Context;
use crate::layout::bump::Bump;
use crate::error::{Error, Result};

/// How every buffer in this heap is created.
///
/// Shared storage and no hazard tracking, matching the heap itself. The
/// options have to agree with the heap's: `heapBufferSizeAndAlignWithLength:`
/// answers for THESE options, and a buffer created with different ones is
/// sized against a number that was computed for something else.
/// The alignment floor a constant placement asks for.
///
/// Metal's own buffer-offset requirement, and the same number the C++ uses as
/// `heap_alloc`'s default. The device may raise it; it may not lower it.
const CONSTANT_ALIGN: u64 = 256;

const BUFFER_OPTIONS: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

/// A buffer placed in the heap.
///
/// Borrowed: the heap owns every buffer it hands out, for its own lifetime.
/// The C++ shell says the same thing with a `void*` and a comment; here the
/// lifetime says it, which is the entire reason for the port.
#[derive(Debug)]
pub struct Slot<'heap> {
    buffer: &'heap ProtocolObject<dyn MTLBuffer>,
    contents: NonNull<c_void>,
    gpu_address: u64,
    offset: u64,
    size: u64,
}

impl<'heap> Slot<'heap> {
    /// The Metal buffer, for binding.
    #[must_use]
    pub fn buffer(&self) -> &'heap ProtocolObject<dyn MTLBuffer> {
        self.buffer
    }

    /// The GPU virtual address, for an argument table entry.
    #[must_use]
    pub const fn gpu_address(&self) -> u64 {
        self.gpu_address
    }

    /// Byte offset within the heap.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Length in bytes, as the caller asked for it.
    ///
    /// Not what the allocation consumed: the device rounds the request up and
    /// the padding belongs to no one. Writing past this is out of bounds even
    /// though the bytes exist.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.size
    }

    /// Whether the slot is empty, which a placed slot never is.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// The CPU-visible bytes.
    ///
    /// A real pointer rather than a slice: the GPU may be reading these bytes
    /// concurrently, and `&[u8]` would be claiming it is not. Shared storage
    /// on UMA makes the pointer valid for the heap's lifetime; what it does
    /// not make it is exclusive.
    #[must_use]
    pub const fn contents(&self) -> NonNull<c_void> {
        self.contents
    }
}

/// The single placement heap and the bump allocator that places into it.
pub struct Heap {
    heap: Retained<ProtocolObject<dyn MTLHeap>>,
    /// Every placed buffer, kept alive for the heap's lifetime.
    ///
    /// A placement sub-buffer does not keep the heap alive, and dropping one
    /// does not return its range -- the bump allocator has already moved past
    /// it. So the buffers are owned here and handed out by reference, which is
    /// the ownership the C++ shell expresses with an `NSMutableArray* retained`
    /// it never removes from.
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    bump: Bump,
    /// Placements memoised by the argument-table slot they are bound to.
    ///
    /// See [`Heap::constant`]. Small and looked up once per rebind, so a map
    /// rather than anything cleverer.
    constants: HashMap<u64, Constant>,
}

/// A placement the constant cache can hand out again.
///
/// Everything a [`Slot`] is except the buffer reference, which cannot be
/// stored beside the `Vec` that owns it and is looked up by index instead.
#[derive(Debug, Clone, Copy)]
struct Constant {
    buffer: usize,
    contents: NonNull<c_void>,
    gpu_address: u64,
    offset: u64,
    size: u64,
}

impl Heap {
    /// Create a heap of `capacity` bytes and make it resident.
    ///
    /// Refuses up front when the device will not hold the request resident,
    /// because Metal will not: `newHeapWithDescriptor:` succeeds well past the
    /// working set and the refusal arrives later, from inside a command
    /// buffer, without the numbers.
    pub fn new(context: &Context, capacity: u64) -> Result<Self> {
        context.check_working_set(capacity)?;

        let descriptor = MTLHeapDescriptor::new();
        descriptor.setType(MTLHeapType::Placement);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        descriptor.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
        descriptor.setSize(usize::try_from(capacity).map_err(|_| Error::HeapExhausted {
            requested: capacity,
            available: 0,
            capacity,
        })?);

        let heap = context
            .device()
            .newHeapWithDescriptor(&descriptor)
            .ok_or(Error::Create {
                what: "MTLHeap",
                message: format!("placement heap of {capacity} bytes"),
            })?;

        // Once, as a whole. Every placement sub-buffer allocated later lands
        // inside this range and inherits its residency.
        //
        // `MTLHeap` refines `MTLAllocation`, so this is a widening to the
        // supertrait's object rather than a conversion; `ProtocolObject`
        // models that with `from_ref` on the concrete protocol object.
        let allocation: &ProtocolObject<dyn MTLAllocation> = ProtocolObject::from_ref(&*heap);
        context.residency().addAllocation(allocation);
        context.residency().commit();
        context.residency().requestResidency();

        Ok(Self {
            heap,
            buffers: Vec::new(),
            bump: Bump::new(capacity),
            constants: HashMap::new(),
        })
    }

    /// Total bytes.
    #[must_use]
    pub const fn capacity(&self) -> u64 {
        self.bump.capacity()
    }

    /// Bytes handed out, including the padding alignment forced.
    #[must_use]
    pub const fn used(&self) -> u64 {
        self.bump.used()
    }

    /// How many slots have been placed.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.buffers.len()
    }

    /// A placement MEMOISED by the argument-table slot it will be bound to.
    ///
    /// The bump allocator never takes anything back, which is the right
    /// trade for weights and the wrong one for a constant that is rewritten
    /// every fire. A batch whose row count varies rebinds its constants each
    /// time, and a fresh [`alloc`](Self::alloc) per rebind walks the heap
    /// until there is nothing left -- at which point the model fails to set
    /// up its NEXT sequence, reporting a budget too small, some thousands of
    /// fires away from the allocation that actually spent it.
    ///
    /// The value at a given `(ordinal, index)` is the same constant every
    /// time, so it can be allocated once and rewritten. What makes rewriting
    /// safe is the step boundary: a rebind happens between steps and a step
    /// blocks on its completion fence, so nothing is reading the old bytes.
    ///
    /// A LARGER request at the same slot is a different constant and gets a
    /// fresh placement; the old one stays where it is, because the bump
    /// allocator has no way to take it back. That is a leak bounded by the
    /// number of distinct sizes a slot ever sees, which is one in every
    /// current caller.
    ///
    /// # Errors
    ///
    /// As [`alloc`](Self::alloc), and only when the placement is new.
    pub fn constant(
        &mut self,
        context: &Context,
        ordinal: u32,
        index: u8,
        bytes: u64,
    ) -> Result<Slot<'_>> {
        let key = (u64::from(ordinal) << 8) | u64::from(index);
        if let Some(hit) = self.constants.get(&key).copied()
            && hit.size >= bytes
        {
            return Ok(Slot {
                buffer: &self.buffers[hit.buffer],
                contents: hit.contents,
                gpu_address: hit.gpu_address,
                offset: hit.offset,
                // The request, not what was reserved. A slot that reported
                // the first, larger request would let a later smaller
                // constant write past itself, and the whole point of `len` is
                // that it is the bound.
                size: bytes,
            });
        }

        let fresh = {
            let slot = self.alloc(context, bytes, CONSTANT_ALIGN)?;
            Constant {
                buffer: 0,
                contents: slot.contents,
                gpu_address: slot.gpu_address,
                offset: slot.offset,
                size: slot.size,
            }
        };
        let record = Constant {
            buffer: self.buffers.len() - 1,
            ..fresh
        };
        self.constants.insert(key, record);
        Ok(Slot {
            buffer: &self.buffers[record.buffer],
            contents: record.contents,
            gpu_address: record.gpu_address,
            offset: record.offset,
            size: record.size,
        })
    }

    /// How many distinct constant slots have been placed.
    #[must_use]
    pub fn constant_count(&self) -> usize {
        self.constants.len()
    }

    /// Place `size` bytes, at least `align`-aligned.
    ///
    /// `align` is a floor, not the alignment: the device states its own
    /// requirement for these options and the larger of the two wins. A caller
    /// asking for less than the device needs gets the device's answer rather
    /// than a buffer the device did not agree to.
    ///
    /// A zero-length request is an error rather than an empty slot. Every
    /// caller of this is sizing a tensor, so zero is a plan that computed a
    /// dimension wrong, and handing back a slot no one can write to would
    /// move the failure to whichever kernel reads it first.
    pub fn alloc(&mut self, context: &Context, size: u64, align: u64) -> Result<Slot<'_>> {
        if size == 0 {
            return Err(Error::HeapExhausted {
                requested: 0,
                available: self.bump.available(),
                capacity: self.bump.capacity(),
            });
        }

        let length = usize::try_from(size).map_err(|_| Error::HeapExhausted {
            requested: size,
            available: self.bump.available(),
            capacity: self.bump.capacity(),
        })?;

        // What the DEVICE thinks this costs. The requested length is what the
        // caller may write; this is what the next allocation has to start
        // after, and the two are not the same number.
        let size_and_align = context
            .device()
            .heapBufferSizeAndAlignWithLength_options(length, BUFFER_OPTIONS);
        let device_size = size_and_align.size as u64;
        let device_align = size_and_align.align as u64;

        let placement = self
            .bump
            .alloc(device_size, align.max(device_align).max(1))
            .map_err(|e| Error::HeapExhausted {
                requested: e.requested,
                available: e.available,
                capacity: e.capacity,
            })?;

        let offset = usize::try_from(placement.offset).map_err(|_| Error::HeapExhausted {
            requested: device_size,
            available: self.bump.available(),
            capacity: self.bump.capacity(),
        })?;

        // SAFETY: `offset` is aligned to the device's own requirement for
        // `BUFFER_OPTIONS` and the range [offset, offset + device_size) is
        // inside the heap and disjoint from every range already handed out --
        // which is exactly what the bump allocator above guarantees, and why
        // it is a separate, tested type rather than three lines here.
        let buffer = unsafe {
            self.heap
                .newBufferWithLength_options_offset(length, BUFFER_OPTIONS, offset)
        }
        .ok_or(Error::Create {
            what: "placement MTLBuffer",
            message: format!("{size} bytes at offset {offset}"),
        })?;

        let contents = buffer.contents();
        let gpu_address = buffer.gpuAddress();
        self.buffers.push(buffer);

        Ok(Slot {
            buffer: self.buffers.last().expect("just pushed"),
            contents,
            gpu_address,
            offset: placement.offset,
            size,
        })
    }
}

impl std::fmt::Debug for Heap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Heap")
            .field("capacity", &self.capacity())
            .field("used", &self.used())
            .field("slots", &self.slot_count())
            .finish_non_exhaustive()
    }
}

// SAFETY: `contents` is the buffer's shared-storage pointer, valid for the
// heap's lifetime, and `size` is the length placement reserved for this slot
// alone. The bump allocator never overlaps two placements.
unsafe impl crate::Region for Slot<'_> {
    fn contents(&self) -> NonNull<c_void> {
        self.contents
    }

    fn len(&self) -> u64 {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16 MB: large enough that the device's alignment is not most of it,
    /// small enough to allocate on any machine that has a GPU at all.
    const TEST_HEAP: u64 = 16 << 20;

    fn fixture() -> Option<(Context, Heap)> {
        let context = match Context::new() {
            Ok(c) => c,
            Err(Error::NoDevice) => return None,
            Err(e) => panic!("context: {e}"),
        };
        let heap = Heap::new(&context, TEST_HEAP).expect("16 MB placement heap");
        Some((context, heap))
    }

    #[test]
    fn a_heap_builds_and_drops() {
        let Some((_context, heap)) = fixture() else {
            return;
        };
        assert_eq!(heap.capacity(), TEST_HEAP);
        assert_eq!(heap.used(), 0);
        drop(heap);
    }

    #[test]
    fn a_slot_is_writable_through_its_contents_pointer() {
        let Some((context, mut heap)) = fixture() else {
            return;
        };
        let slot = heap.alloc(&context, 4096, 1).expect("4 KB fits");
        assert_eq!(slot.len(), 4096);
        assert!(slot.gpu_address() != 0, "a placed buffer has an address");

        // The claim under test is that Shared storage on a placement heap
        // gives back a CPU pointer to the same bytes -- which is what every
        // weight upload and scalar write depends on, and which a
        // Private-storage heap would answer null for.
        let ptr = slot.contents().as_ptr().cast::<u8>();
        // SAFETY: `ptr` is the start of a Shared-storage buffer of at least
        // 4096 bytes, alive for as long as the heap, and nothing has been
        // encoded against it -- no GPU work exists that could be reading it.
        unsafe {
            std::ptr::write_bytes(ptr, 0xAB, 4096);
            assert_eq!(*ptr, 0xAB);
            assert_eq!(*ptr.add(4095), 0xAB);
        }
    }

    #[test]
    fn slots_do_not_overlap() {
        let Some((context, mut heap)) = fixture() else {
            return;
        };
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        for i in 1..=8u64 {
            let size = i * 1024;
            let slot = heap.alloc(&context, size, 1).expect("fits");
            ranges.push((slot.offset(), size));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            let (offset, size) = pair[0];
            let (next, _) = pair[1];
            assert!(
                offset + size <= next,
                "slot at {offset}+{size} overlaps the one at {next}"
            );
        }
        assert_eq!(heap.slot_count(), 8);
    }

    #[test]
    fn a_slot_larger_than_the_heap_is_refused_with_its_numbers() {
        let Some((context, mut heap)) = fixture() else {
            return;
        };
        let err = heap
            .alloc(&context, TEST_HEAP * 2, 1)
            .expect_err("twice the heap does not fit");
        match err {
            Error::HeapExhausted {
                requested,
                available,
                capacity,
            } => {
                assert!(requested >= TEST_HEAP * 2);
                assert_eq!(available, TEST_HEAP);
                assert_eq!(capacity, TEST_HEAP);
            }
            other => panic!("expected HeapExhausted, got {other}"),
        }
        // And the refusal cost nothing: the heap is still empty and usable.
        assert_eq!(heap.used(), 0);
        heap.alloc(&context, 1024, 1).expect("still allocatable");
    }

    #[test]
    fn a_zero_length_slot_is_refused() {
        let Some((context, mut heap)) = fixture() else {
            return;
        };
        heap.alloc(&context, 0, 1)
            .expect_err("zero is a bug upstream");
    }

    #[test]
    fn a_heap_past_the_working_set_is_refused_before_metal_sees_it() {
        let context = match Context::new() {
            Ok(c) => c,
            Err(Error::NoDevice) => return,
            Err(e) => panic!("context: {e}"),
        };
        let err = Heap::new(&context, context.working_set_bytes() + 1)
            .expect_err("one byte past what the device will hold");
        assert!(matches!(err, Error::WorkingSetExceeded { .. }), "{err}");
    }
}
