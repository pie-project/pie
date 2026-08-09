//! The launch path's buffer view, against real allocations.
//!
//! What is worth proving on a device is the arithmetic the C++ `subhandle`
//! never checked: that a slice's GPU address lands exactly where its host
//! pointer does, that the bounds refuse what the C++ silently minted, and
//! that a view keeps its allocation alive after every owner is gone.

#![allow(clippy::print_stdout)]

use driver_metal::{Error, Region};
use driver_metal::gpu::{Compiler, Context, Handle, Heap, Pool, Stepper, Tables};

const HEAP_BYTES: u64 = 1 << 20;
const SLOT_BYTES: u64 = 4096;

/// Writes `tag` into every element of the buffer at binding 0.
const FILL: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void fill(device uint* out [[buffer(0)]],
                 constant uint& tag [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
    out[gid] = tag;
}
";

struct Fixture {
    context: Context,
    heap: Heap,
}

fn fixture() -> Option<Fixture> {
    let context = match Context::new() {
        Ok(c) => c,
        Err(Error::NoDevice) => return None,
        Err(e) => panic!("context: {e}"),
    };
    let heap = Heap::new(&context, HEAP_BYTES).expect("heap");
    Some(Fixture { context, heap })
}

/// The whole parent span, read through its contents pointer.
fn bytes_of(handle: &Handle) -> Vec<u8> {
    // SAFETY: the handle promises `len` readable bytes and no step is running.
    unsafe {
        std::slice::from_raw_parts(
            handle.contents().as_ptr().cast::<u8>(),
            handle.len() as usize,
        )
    }
    .to_vec()
}

#[test]
fn a_slice_writes_the_bytes_its_offset_names_in_the_parent() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut heap = f.heap;
    let slot = heap.alloc(&f.context, SLOT_BYTES, 256).expect("slot");
    let parent = Handle::over(slot.buffer(), slot.len()).expect("view");
    // A heap placement is not zero-filled; establish a known ground first.
    unsafe { parent.zero(0, parent.len()) }.expect("zero");

    assert_eq!(
        parent.gpu_address(),
        slot.gpu_address(),
        "a whole-buffer view must start where the slot starts"
    );

    let slice = parent.slice(256, 64).expect("slice");
    assert_eq!(slice.gpu_address(), parent.gpu_address() + 256);
    assert_eq!(slice.len(), 64);

    unsafe { slice.write(0, b"pie!") }.expect("write");
    let all = bytes_of(&parent);
    assert_eq!(
        &all[256..260],
        b"pie!",
        "the slice did not land at its offset"
    );
    assert_eq!(all[255], 0, "the byte before the slice is not the slice's");
    assert_eq!(all[260], 0, "the byte after the write is not the write's");
}

#[test]
fn a_slices_gpu_address_lands_the_dispatch_inside_its_parent() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut heap = f.heap;
    let compiler = Compiler::new(&f.context).expect("compiler");
    let pipeline = compiler.compile(&f.context, FILL, "fill").expect("fill");

    const COUNT: usize = 64;
    const OFFSET: u64 = 512;

    let slot = heap.alloc(&f.context, SLOT_BYTES, 256).expect("slot");
    let parent = Handle::over(slot.buffer(), slot.len()).expect("view");
    unsafe { parent.zero(0, parent.len()) }.expect("zero");
    let slice = parent.slice(OFFSET, (COUNT * 4) as u64).expect("slice");

    let tag = heap.alloc(&f.context, 4, 256).expect("tag");
    // SAFETY: shared storage, one u32 wide, no step is running.
    unsafe { tag.contents().as_ptr().cast::<u32>().write(0xC3C3_C3C3) };

    let mut tables = Tables::new();
    tables
        .bind_address(&f.context, 0, 0, slice.gpu_address())
        .expect("bind out");
    tables
        .bind_address(&f.context, 0, 1, tag.gpu_address())
        .expect("bind tag");

    let mut stepper = Stepper::new(&f.context).expect("stepper");
    stepper
        .run(|step| {
            step.set_pipeline(&pipeline);
            step.set_argument_table_for(&tables, 0)?;
            step.dispatch([COUNT, 1, 1], [64, 1, 1])
        })
        .expect("the step ran");

    // SAFETY: the step signalled; the parent promises its whole span.
    let words = unsafe {
        std::slice::from_raw_parts(
            parent.contents().as_ptr().cast::<u32>(),
            (SLOT_BYTES / 4) as usize,
        )
    };
    let first = (OFFSET / 4) as usize;
    assert!(
        words[..first].iter().all(|&w| w == 0),
        "the dispatch wrote before the slice"
    );
    assert_eq!(
        &words[first..first + COUNT],
        &[0xC3C3_C3C3_u32; COUNT][..],
        "the GPU did not write through the slice's address"
    );
    assert!(
        words[first + COUNT..].iter().all(|&w| w == 0),
        "the dispatch wrote past the slice"
    );
}

#[test]
fn a_span_the_parent_never_had_is_refused_not_minted() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut heap = f.heap;
    let slot = heap.alloc(&f.context, SLOT_BYTES, 256).expect("slot");
    let parent = Handle::over(slot.buffer(), slot.len()).expect("view");

    assert!(
        parent.slice(SLOT_BYTES, 1).is_err(),
        "one byte past the end"
    );
    assert!(
        parent.slice(0, SLOT_BYTES + 1).is_err(),
        "one byte too long"
    );

    // The sum the obvious check gets wrong: offset + len wraps to something
    // small, and the C++ would have minted the handle.
    let err = parent.slice(u64::MAX, 2).expect_err("a wrapped span");
    assert!(matches!(err, Error::OutOfRange { .. }));

    // And a whole-buffer claim longer than the buffer itself.
    let err = Handle::over(slot.buffer(), u64::MAX).expect_err("a claim past the buffer");
    assert!(matches!(err, Error::OutOfRange { .. }));
}

#[test]
fn a_slice_of_a_slice_composes_like_one_slice_of_the_base() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut heap = f.heap;
    let slot = heap.alloc(&f.context, SLOT_BYTES, 256).expect("slot");
    let parent = Handle::over(slot.buffer(), slot.len()).expect("view");

    let outer = parent.slice(256, 512).expect("outer");
    let inner = outer.slice(128, 64).expect("inner");
    assert_eq!(inner.gpu_address(), parent.gpu_address() + 384);
    assert_eq!(inner.len(), 64);

    // The inner bound is the outer view's, not the base buffer's: the base
    // has plenty of room at offset 449, the outer view does not.
    assert!(outer.slice(448, 64).is_ok(), "flush against the outer end");
    assert!(
        outer.slice(449, 64).is_err(),
        "a slice reached back out of the span its parent was narrowed to"
    );
}

#[test]
fn an_empty_slice_at_the_boundary_is_a_view_of_nothing_not_an_error() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut heap = f.heap;
    let slot = heap.alloc(&f.context, SLOT_BYTES, 256).expect("slot");
    let parent = Handle::over(slot.buffer(), slot.len()).expect("view");

    let empty = parent.slice(SLOT_BYTES, 0).expect("empty at the end");
    assert!(empty.is_empty());
    assert!(
        parent.slice(SLOT_BYTES + 1, 0).is_err(),
        "empty past the end is still past the end"
    );
}

#[test]
fn a_view_keeps_its_allocation_alive_after_every_owner_is_gone() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let pool = Pool::new(1 << 20);
    let loan = pool.acquire(&f.context, 4096).expect("loan");
    let view = Handle::over(loan.buffer(), loan.len()).expect("view");

    // The C++ analogue of this sequence is a dangling `void*`: the loan goes
    // back to the pool, the pool is torn down, and the handle still points.
    drop(loan);
    drop(pool);

    unsafe { view.write(0, b"still mapped") }.expect("write");
    assert_eq!(&bytes_of(&view)[..12], b"still mapped");
}

#[test]
fn a_buffer_the_host_cannot_address_is_refused() {
    use objc2_metal::{MTLDevice, MTLResourceOptions};

    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let private = f
        .context
        .device()
        .newBufferWithLength_options(4096, MTLResourceOptions::StorageModePrivate)
        .expect("a private buffer");

    let message = Handle::over(&private, 4096)
        .expect_err("a private buffer has no host bytes to view")
        .to_string();
    assert!(
        message.contains("storage mode"),
        "the refusal does not say what is wrong: {message}"
    );
}
