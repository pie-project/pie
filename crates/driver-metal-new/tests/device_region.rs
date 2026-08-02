//! [`Region`] against real device memory.
//!
//! The arithmetic is proved in the unit tests, off-device, against a `Vec`.
//! What a `Vec` cannot prove is the premise the whole module rests on: that a
//! host `memset` into a shared-storage buffer is a write the GPU then reads,
//! with nothing encoded and no barrier issued. If that were false every test
//! in `src/region.rs` would still pass.
//!
//! The other claim here is the bound. A heap slot's neighbour is real memory
//! a few bytes away, and a slot that reported its rounded-up size instead of
//! its requested one would let a caller into it. Off-device that is a byte in
//! a `Vec`; here it is another allocation.

#![allow(clippy::print_stdout)]

use driver_metal_new::{Compiler, Context, Error, Heap, Pool, Region, Stepper, Tables};

/// Reads a buffer and reports what it found, so the GPU's view of the host's
/// writes is the thing under test rather than the host's own view of them.
const SUM: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void sum(device uint* out [[buffer(0)]],
                const device uint* in [[buffer(1)]],
                uint gid [[thread_position_in_grid]]) {
    uint total = 0;
    for (uint i = 0; i < 64; ++i) total += in[i];
    out[gid] = total;
}
";

fn context() -> Option<Context> {
    match Context::new() {
        Ok(c) => Some(c),
        Err(Error::NoDevice) => None,
        Err(e) => panic!("context: {e}"),
    }
}

fn read_u32(r: &impl Region) -> u32 {
    // SAFETY: shared storage, at least four bytes wide, and the step that
    // wrote it has signalled.
    unsafe { r.contents().as_ptr().cast::<u32>().read() }
}

#[test]
fn the_gpu_reads_what_the_host_wrote_and_the_zero_that_followed() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let pipeline = compiler.compile(&context, SUM, "sum").expect("sum");
    let pool = Pool::new(1 << 20);

    let out = pool.acquire(&context, 4).expect("out");
    let input = pool.acquire(&context, 256).expect("in");

    // Sixty-four ones, written host-side with no encoding of any kind.
    let ones: Vec<u8> = (0..64u32).flat_map(|_| 1u32.to_le_bytes()).collect();
    // SAFETY: nothing is in flight; the region is 256 bytes and so is `ones`.
    unsafe { input.write(0, &ones) }.expect("write");

    let mut tables = Tables::new();
    tables
        .bind_address(&context, 0, 0, out.gpu_address())
        .expect("0.0");
    tables
        .bind_address(&context, 0, 1, input.gpu_address())
        .expect("0.1");

    let mut stepper = Stepper::new(&context).expect("stepper");
    let fire = |stepper: &mut Stepper| {
        stepper
            .run(|step| {
                step.set_pipeline(&pipeline);
                step.set_argument_table_for(&tables, 0)?;
                step.dispatch([1, 1, 1], [1, 1, 1])
            })
            .expect("the step ran");
    };

    fire(&mut stepper);
    assert_eq!(
        read_u32(&out),
        64,
        "the GPU did not see a plain host write to shared storage"
    );

    // Zero the back half only. If `zero` wrote the whole buffer the sum would
    // be 0, and if it wrote nothing it would still be 64.
    // SAFETY: the previous step has signalled and no step is committed.
    unsafe { input.zero(128, 128) }.expect("zero");
    fire(&mut stepper);
    assert_eq!(read_u32(&out), 32, "the zeroed half did not reach the GPU");
}

#[test]
fn a_copy_between_a_slot_and_a_loan_reaches_the_gpu() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let pipeline = compiler.compile(&context, SUM, "sum").expect("sum");
    let mut heap = Heap::new(&context, 1 << 20).expect("heap");
    let pool = Pool::new(1 << 20);

    let staging = heap.alloc(&context, 256, 256).expect("staging");
    let threes: Vec<u8> = (0..64u32).flat_map(|_| 3u32.to_le_bytes()).collect();
    // SAFETY: nothing is in flight.
    unsafe { staging.write(0, &threes) }.expect("write");

    let input = pool.acquire(&context, 256).expect("in");
    let out = pool.acquire(&context, 4).expect("out");
    // SAFETY: as above. Two distinct allocations, so no overlap.
    unsafe { input.copy(0, &staging, 0, 256) }.expect("copy");

    let mut tables = Tables::new();
    tables
        .bind_address(&context, 0, 0, out.gpu_address())
        .expect("0.0");
    tables
        .bind_address(&context, 0, 1, input.gpu_address())
        .expect("0.1");

    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .run(|step| {
            step.set_pipeline(&pipeline);
            step.set_argument_table_for(&tables, 0)?;
            step.dispatch([1, 1, 1], [1, 1, 1])
        })
        .expect("the step ran");

    assert_eq!(
        read_u32(&out),
        192,
        "the copy did not land where the GPU was looking"
    );
}

/// A slot reports what was asked for, not what was allocated.
///
/// The device rounds a placement up and the bump allocator moves past the
/// rounded figure, so the bytes between one slot's request and the next
/// slot's start exist, are mapped, and are writable. Nothing faults if a
/// caller walks into them -- they simply belong to no one until the heap
/// hands out the next placement, and then they belong to someone else. The
/// only thing that ever says no is this bound, so it has to be reading the
/// request.
#[test]
fn a_slot_bounds_the_request_and_not_the_allocation() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut heap = Heap::new(&context, 1 << 20).expect("heap");

    let slot = heap.alloc(&context, 100, 256).expect("slot");
    assert_eq!(slot.len(), 100, "the slot reported its rounding-up");

    let err = unsafe { slot.zero(0, 101) }.unwrap_err();
    assert!(
        matches!(err, Error::OutOfRange { len: 100, .. }),
        "the bound is not the request: {err}"
    );
    let err = unsafe { slot.write(100, &[0]) }.unwrap_err();
    assert!(matches!(err, Error::OutOfRange { .. }), "{err}");

    // SAFETY: nothing is in flight, and the range is the whole request.
    unsafe { slot.write(0, &[0x5A; 100]) }.expect("the whole slot is fine");
    // SAFETY: shared storage, 100 bytes, nothing in flight.
    let seen = unsafe { std::slice::from_raw_parts(slot.contents().as_ptr().cast::<u8>(), 100) };
    assert_eq!(seen, [0x5A; 100]);
}
