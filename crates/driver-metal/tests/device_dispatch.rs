//! A dispatch that actually runs, proved by reading back what it wrote.
//!
//! Everything up to here can be green while the GPU does nothing. A pipeline
//! compiles, an argument table binds, a command buffer commits and an event
//! signals -- and if the dispatch was skipped, all of that still happens and
//! the step reports success. Metal skips a dispatch whose threadgroup exceeds
//! the pipeline's limit, silently, which is the failure the C++ shell found
//! only by noticing the model had begun answering nonsense.
//!
//! So the assertion is on the BYTES: a kernel writes a value the host did not
//! put there, and the host reads it back through the heap's shared pointer.
//!
//! Requires a Metal 4 GPU, and skips without one.


use driver_metal::Error;
use driver_metal::gpu::{ArgumentTable, Compiler, Context, Heap, Stepper, Visibility};

/// Writes `index + 1` to every element, so the check cannot pass on a buffer
/// that happens to be a constant -- and cannot pass on a dispatch that ran
/// with the wrong grid, because the tail would be missing.
const RAMP: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void ramp(device uint* out [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
    out[gid] = gid + 1u;
}
";

/// Doubles in place, to prove a barrier orders one dispatch after another
/// within a single encoder.
const DOUBLE: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void double_in_place(device uint* out [[buffer(0)]],
                            uint gid [[thread_position_in_grid]]) {
    out[gid] = out[gid] * 2u;
}
";

const COUNT: usize = 1024;
const HEAP_BYTES: u64 = 1 << 20;

struct Fixture {
    context: Context,
    compiler: Compiler,
    heap: Heap,
}

fn fixture() -> Option<Fixture> {
    let context = match Context::new() {
        Ok(c) => c,
        Err(Error::NoDevice) => return None,
        Err(e) => panic!("context: {e}"),
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let heap = Heap::new(&context, HEAP_BYTES).expect("heap");
    Some(Fixture {
        context,
        compiler,
        heap,
    })
}

/// Read `count` u32s out of a slot's shared storage.
///
/// # Safety
///
/// The caller must have waited for every dispatch that writes the slot, and
/// the slot must hold at least `count` u32s.
unsafe fn read_u32s(pointer: *const u32, count: usize) -> Vec<u32> {
    // SAFETY: forwarded to the caller's contract.
    unsafe { std::slice::from_raw_parts(pointer, count).to_vec() }
}

#[test]
fn a_dispatch_writes_bytes_the_host_reads_back() {
    let Some(Fixture {
        context,
        compiler,
        mut heap,
    }) = fixture()
    else {
        return;
    };

    let pipeline = compiler.compile(&context, RAMP, "ramp").expect("compiles");
    let bytes = (COUNT * size_of::<u32>()) as u64;
    let slot = heap.alloc(&context, bytes, 1).expect("slot");

    // Poisoned first, so a dispatch that does not run leaves a value the
    // assertion below can tell apart from the one it wants.
    let pointer = slot.contents().as_ptr().cast::<u32>();
    // SAFETY: `pointer` is the start of a Shared-storage slot of `bytes`, and
    // no GPU work has been encoded against it yet.
    unsafe { std::ptr::write_bytes(pointer.cast::<u8>(), 0xEE, bytes as usize) };

    let table = ArgumentTable::new(&context, 1).expect("table");
    table.bind(0, &slot).expect("bind");

    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .run(|step| {
            step.set_pipeline(&pipeline);
            step.set_argument_table(&table);
            step.dispatch([COUNT, 1, 1], [64, 1, 1])
        })
        .expect("the step ran");

    assert_eq!(stepper.steps(), 1);
    assert!(!stepper.is_wedged());

    // SAFETY: `stepper.run` returned, so the event this step signals has been
    // reached and no GPU work is still writing the slot.
    let out = unsafe { read_u32s(pointer, COUNT) };
    let want: Vec<u32> = (0..COUNT).map(|i| i as u32 + 1).collect();
    assert_eq!(out, want, "the dispatch did not write what the kernel says");
}

#[test]
fn a_barrier_orders_two_dispatches_in_one_encoder() {
    let Some(Fixture {
        context,
        compiler,
        mut heap,
    }) = fixture()
    else {
        return;
    };

    let ramp = compiler.compile(&context, RAMP, "ramp").expect("compiles");
    let double = compiler
        .compile(&context, DOUBLE, "double_in_place")
        .expect("compiles");

    let bytes = (COUNT * size_of::<u32>()) as u64;
    let slot = heap.alloc(&context, bytes, 1).expect("slot");
    let pointer = slot.contents().as_ptr().cast::<u32>();

    let table = ArgumentTable::new(&context, 1).expect("table");
    table.bind(0, &slot).expect("bind");

    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .run(|step| {
            step.set_argument_table(&table);
            step.set_pipeline(&ramp);
            step.dispatch([COUNT, 1, 1], [64, 1, 1])?;
            step.barrier(Visibility::Device);
            step.set_pipeline(&double);
            step.dispatch([COUNT, 1, 1], [64, 1, 1])
        })
        .expect("the step ran");

    // SAFETY: the step completed before `run` returned.
    let out = unsafe { read_u32s(pointer, COUNT) };
    let want: Vec<u32> = (0..COUNT).map(|i| (i as u32 + 1) * 2).collect();
    assert_eq!(
        out, want,
        "the second dispatch did not see the first one's writes"
    );
}

/// Two steps in a row, which is what exercises the allocator parity: the
/// second step resets the OTHER allocator, and a stepper that reset the one
/// still in use would fault here rather than in production.
#[test]
fn consecutive_steps_alternate_allocators() {
    let Some(Fixture {
        context,
        compiler,
        mut heap,
    }) = fixture()
    else {
        return;
    };

    let double = compiler
        .compile(&context, DOUBLE, "double_in_place")
        .expect("compiles");
    let bytes = (COUNT * size_of::<u32>()) as u64;
    let slot = heap.alloc(&context, bytes, 1).expect("slot");
    let pointer = slot.contents().as_ptr().cast::<u32>();
    // SAFETY: nothing has been encoded against the slot yet.
    unsafe {
        for i in 0..COUNT {
            pointer.add(i).write(1);
        }
    }

    let table = ArgumentTable::new(&context, 1).expect("table");
    table.bind(0, &slot).expect("bind");
    let mut stepper = Stepper::new(&context).expect("stepper");

    for _ in 0..4 {
        stepper
            .run(|step| {
                step.set_pipeline(&double);
                step.set_argument_table(&table);
                step.dispatch([COUNT, 1, 1], [64, 1, 1])
            })
            .expect("the step ran");
    }

    assert_eq!(stepper.steps(), 4);
    // SAFETY: every step completed before its `run` returned.
    let out = unsafe { read_u32s(pointer, COUNT) };
    assert!(
        out.iter().all(|&v| v == 16),
        "four doublings of 1 is 16, got {:?}",
        &out[..4]
    );
}

/// The refusal that exists because Metal does not refuse: a threadgroup past
/// the pipeline's limit makes the dispatch silently not happen.
#[test]
fn an_oversized_threadgroup_is_refused_rather_than_skipped() {
    let Some(Fixture {
        context,
        compiler,
        mut heap,
    }) = fixture()
    else {
        return;
    };

    let pipeline = compiler.compile(&context, RAMP, "ramp").expect("compiles");
    let bytes = (COUNT * size_of::<u32>()) as u64;
    let slot = heap.alloc(&context, bytes, 1).expect("slot");
    let table = ArgumentTable::new(&context, 1).expect("table");
    table.bind(0, &slot).expect("bind");

    let mut stepper = Stepper::new(&context).expect("stepper");
    let err = stepper
        .run(|step| {
            step.set_pipeline(&pipeline);
            step.set_argument_table(&table);
            // No pipeline on this device allows a 4096-thread threadgroup.
            step.dispatch([COUNT, 1, 1], [4096, 1, 1])
        })
        .expect_err("the dispatch is refused");
    assert!(matches!(err, Error::Create { .. }), "{err}");
    // The step was still closed and committed, so the stepper is usable.
    assert!(!stepper.is_wedged());
}

/// A dispatch with no pipeline set would run whatever the last one left, and
/// Metal reports nothing.
#[test]
fn a_dispatch_without_a_pipeline_is_refused() {
    let Some(Fixture { context, .. }) = fixture() else {
        return;
    };
    let mut stepper = Stepper::new(&context).expect("stepper");
    let err = stepper
        .run(|step| step.dispatch([64, 1, 1], [64, 1, 1]))
        .expect_err("no pipeline");
    assert!(matches!(err, Error::Create { .. }), "{err}");
}

/// Binding past the table's capacity is an error, not a no-op that surfaces
/// as a kernel reading address zero.
#[test]
fn a_binding_past_the_table_is_refused() {
    let Some(Fixture {
        context, mut heap, ..
    }) = fixture()
    else {
        return;
    };
    let slot = heap.alloc(&context, 4096, 1).expect("slot");
    let table = ArgumentTable::new(&context, 2).expect("table");
    table.bind(0, &slot).expect("in range");
    table.bind(1, &slot).expect("in range");
    table.bind(2, &slot).expect_err("past the bind count");
}
