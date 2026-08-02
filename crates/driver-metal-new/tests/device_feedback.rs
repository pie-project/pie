//! Commit feedback: the only place Metal 4 says a step actually worked.
//!
//! The load-bearing test is [`slow_and_fast_steps_are_told_apart`]: it makes
//! the GPU do two workloads that differ by two orders of magnitude and
//! asserts the reported times differ in the same direction. A stub returning
//! zeros, or a handler reading the wrong pair of timestamps, passes every
//! other check here and fails that one.

#![allow(clippy::print_stdout)]

use std::time::Duration;

use driver_metal_new::{Compiler, Context, Error, Heap, Stepper, Tables};

const HEAP_BYTES: u64 = 1 << 22;
const LANDING: Duration = Duration::from_secs(10);

/// Spins `rounds` times per thread so the step's GPU time is a knob.
const SPIN: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void spin(device uint* out [[buffer(0)]],
                 constant uint& rounds [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
    uint acc = gid;
    for (uint i = 0; i < rounds; ++i) {
        acc = acc * 1664525u + 1013904223u;
    }
    out[gid] = acc;
}
";

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

/// Run `rounds` of the spin kernel over `threads` threads and report the GPU
/// time the feedback handler received for it.
fn timed_step(rounds: u32, threads: usize) -> Option<Duration> {
    let f = fixture()?;
    let mut heap = f.heap;
    let pipeline = f.compiler.compile(&f.context, SPIN, "spin").expect("spin");

    let out = heap
        .alloc(&f.context, (threads * 4) as u64, 256)
        .expect("out");
    let out_addr = out.gpu_address();
    let knob = heap.alloc(&f.context, 4, 256).expect("knob");
    // SAFETY: shared storage, one u32 wide, nothing is running.
    unsafe { knob.contents().as_ptr().cast::<u32>().write(rounds) };

    let mut tables = Tables::new();
    tables
        .bind_address(&f.context, 0, 0, out_addr)
        .expect("0.0");
    tables
        .bind_address(&f.context, 0, 1, knob.gpu_address())
        .expect("0.1");

    let mut stepper = Stepper::new(&f.context).expect("stepper");
    let observer = stepper.feedback().clone();
    stepper
        .run(|step| {
            step.set_pipeline(&pipeline);
            step.set_argument_table_for(&tables, 0)?;
            step.dispatch([threads, 1, 1], [64, 1, 1])
        })
        .expect("the step ran");

    let got = observer
        .await_step(1, LANDING)
        .expect("no feedback landed within ten seconds");
    assert_eq!(got.step, 1);
    assert!(!got.failed(), "the GPU reported a fault: {:?}", got.error);
    Some(got.gpu_time())
}

#[test]
fn a_step_is_reported_on_with_a_time_that_is_not_zero() {
    let Some(gpu) = timed_step(4096, 1 << 16) else {
        println!("no Metal device; skipped");
        return;
    };
    assert!(
        gpu > Duration::ZERO,
        "the GPU reported zero time for a step that ran 64Ki threads"
    );
    assert!(
        gpu < Duration::from_secs(5),
        "a step that should take microseconds reported {gpu:?}; the timestamps are not a pair"
    );
    println!("gpu time: {gpu:?}");
}

#[test]
fn slow_and_fast_steps_are_told_apart() {
    let Some(fast) = timed_step(1, 1 << 14) else {
        println!("no Metal device; skipped");
        return;
    };
    let slow = timed_step(1 << 15, 1 << 16).expect("device");
    println!("fast: {fast:?}  slow: {slow:?}");
    assert!(
        slow > fast * 4,
        "a workload two orders of magnitude larger reported {slow:?} against {fast:?}; \
         the number does not measure the step"
    );
}

#[test]
fn feedback_has_not_landed_before_anything_has_run() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let stepper = Stepper::new(&f.context).expect("stepper");
    assert!(
        stepper.feedback().latest().is_none(),
        "feedback exists for a step that was never committed"
    );
    // Nothing will ever land, so the bounded wait must give up rather than
    // hang -- the same shape as the completion wait.
    assert!(
        stepper
            .feedback()
            .await_step(1, Duration::from_millis(50))
            .is_none()
    );
}

#[test]
fn later_steps_replace_the_report_of_earlier_ones() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut heap = f.heap;
    let pipeline = f.compiler.compile(&f.context, SPIN, "spin").expect("spin");
    let out = heap.alloc(&f.context, 4096, 256).expect("out");
    let out_addr = out.gpu_address();
    let knob = heap.alloc(&f.context, 4, 256).expect("knob");
    // SAFETY: shared storage, one u32 wide, nothing is running.
    unsafe { knob.contents().as_ptr().cast::<u32>().write(64) };

    let mut tables = Tables::new();
    tables
        .bind_address(&f.context, 0, 0, out_addr)
        .expect("0.0");
    tables
        .bind_address(&f.context, 0, 1, knob.gpu_address())
        .expect("0.1");

    let mut stepper = Stepper::new(&f.context).expect("stepper");
    let observer = stepper.feedback().clone();
    for _ in 0..4 {
        stepper
            .run(|step| {
                step.set_pipeline(&pipeline);
                step.set_argument_table_for(&tables, 0)?;
                step.dispatch([1024, 1, 1], [64, 1, 1])
            })
            .expect("the step ran");
    }

    let got = observer
        .await_step(4, LANDING)
        .expect("the fourth step was never reported on");
    assert_eq!(
        got.step, 4,
        "the newest report is for step {}, not the fourth",
        got.step
    );
    assert!(got.gpu_end >= got.gpu_start);
}
