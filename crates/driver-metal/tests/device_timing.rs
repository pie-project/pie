//! What a step cost, and the claim that the two halves are separately real.
//!
//! [`Timing`] splits a step into host encoding and GPU execution. A split is
//! only worth having if the two numbers actually move independently, so that
//! is what is checked here: a kernel whose iteration count is a constant is
//! compiled twice, once cheap and once expensive, and encoded identically
//! both times. If the split is real, `gpu_exec` moves by a large factor and
//! `encode` does not. If the implementation had read one clock and reported
//! it twice, or put the commit on the wrong side of the boundary, both would
//! move together and this fails.
//!
//! The other claim is that `gpu` -- the GPU's own report -- is smaller than
//! the host-observed `gpu_exec`, because it excludes queueing and the host's
//! wake-up. That is the entire reason a third number exists.

#![allow(clippy::print_stdout)]

use std::time::Duration;

use driver_metal::Error;
use driver_metal::device::{Context, Pool, Stepper, Tables};
use driver_metal::program::Compiler;

/// A loop the compiler cannot fold away: the result is stored, and the trip
/// count is a specialisation constant substituted into the source.
const SPIN: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void spin(device float* out [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
    float acc = float(gid);
    for (uint i = 0; i < ITERS; ++i) {
        acc = fma(acc, 1.0000001f, 1.0f);
    }
    out[gid] = acc;
}
";

const THREADS: usize = 4096;
const THREADGROUP: usize = 256;

fn source(iters: u32) -> String {
    format!("#define ITERS {iters}u\n{SPIN}")
}

fn context() -> Option<Context> {
    match Context::new() {
        Ok(c) => Some(c),
        Err(Error::NoDevice) => None,
        Err(e) => panic!("context: {e}"),
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

#[test]
fn encoding_and_execution_are_measured_separately() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let cheap = compiler
        .compile(&context, &source(1), "spin")
        .expect("cheap");
    let dear = compiler
        .compile(&context, &source(300_000), "spin")
        .expect("dear");

    let pool = Pool::new(1 << 20);
    let out = pool.acquire(&context, (THREADS * 4) as u64).expect("out");
    let mut tables = Tables::new();
    tables
        .bind_address(&context, 0, 0, out.gpu_address())
        .expect("bind");

    let mut stepper = Stepper::new(&context).expect("stepper");

    // Warm both paths first. The first submission of a process pays for
    // driver setup that belongs to neither half, and a first-run number
    // would make this test about start-up rather than about the split.
    for pipeline in [&cheap, &dear] {
        stepper
            .run(|step| {
                step.set_argument_table_for(&tables, 0)?;
                step.set_pipeline(pipeline);
                step.dispatch([THREADS, 1, 1], [THREADGROUP, 1, 1])
            })
            .expect("warm");
    }

    let mut run = |pipeline| {
        stepper
            .run(|step: &mut driver_metal::device::StepEncoder<'_>| {
                step.set_argument_table_for(&tables, 0)?;
                step.set_pipeline(pipeline);
                step.dispatch([THREADS, 1, 1], [THREADGROUP, 1, 1])
            })
            .expect("run")
    };

    let mut light = run(&cheap);
    let mut heavy = run(&dear);
    // `gpu_exec` is host-observed (commit to fence), so a busy machine — a
    // parallel test run is enough — can inflate the cheap step's number and
    // collapse the heavy/light ratio for one sample. The property under test
    // is about the boundary, not about one sample: re-measure a few times
    // and let ANY quiet window decide. If the commit really is on the wrong
    // side, no window will ever satisfy it and the assertions below still
    // fail on the last sample.
    for _ in 0..4 {
        if heavy.gpu_exec > light.gpu_exec * 4 && heavy.encode < heavy.gpu_exec {
            break;
        }
        light = run(&cheap);
        heavy = run(&dear);
    }

    println!(
        "cheap: encode {:.3} ms, gpu_exec {:.3} ms, gpu {:?}",
        ms(light.encode),
        ms(light.gpu_exec),
        light.gpu.map(ms)
    );
    println!(
        "dear:  encode {:.3} ms, gpu_exec {:.3} ms, gpu {:?}",
        ms(heavy.encode),
        ms(heavy.gpu_exec),
        heavy.gpu.map(ms)
    );

    assert!(
        light.encode > Duration::ZERO && light.gpu_exec > Duration::ZERO,
        "a real dispatch spent time in both halves, so neither may read zero: {light:?}"
    );

    // The load-bearing assertion. Both steps encode the SAME three calls;
    // only the kernel's trip count differs. So the GPU half must grow and
    // the host half must not.
    assert!(
        heavy.gpu_exec > light.gpu_exec * 4,
        "300000x the kernel work did not move gpu_exec ({:.3} ms vs {:.3} ms), \
         so the commit is on the wrong side of the boundary",
        ms(heavy.gpu_exec),
        ms(light.gpu_exec)
    );
    assert!(
        heavy.encode < heavy.gpu_exec,
        "encoding two identical dispatches cannot cost more than a kernel \
         that spins 300000 times: encode {:.3} ms, gpu_exec {:.3} ms",
        ms(heavy.encode),
        ms(heavy.gpu_exec)
    );

    assert_eq!(
        heavy.total(),
        heavy.encode + heavy.gpu_exec,
        "total is what the caller waited for"
    );
    assert!(
        heavy.step > light.step,
        "each step names its own timeline value"
    );

    // The previous step's feedback is sitting in the slot when this one is
    // assembled, and it is a real, plausible number about the wrong step.
    // A cheap step encoded right after an expensive one must not inherit it.
    let after = run(&cheap);
    println!("after: gpu {:?}", after.gpu.map(ms));
    if let Some(gpu) = after.gpu {
        assert!(
            gpu < heavy.gpu_exec / 4,
            "a 1-iteration kernel reported {:.3} ms, which is the previous \
             step's number: the feedback was not keyed to this submission",
            ms(gpu)
        );
    }
}

#[test]
fn the_gpus_own_report_excludes_the_host_wake_up() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let pipeline = compiler
        .compile(&context, &source(200_000), "spin")
        .expect("spin");
    let pool = Pool::new(1 << 20);
    let out = pool.acquire(&context, (THREADS * 4) as u64).expect("out");
    let mut tables = Tables::new();
    tables
        .bind_address(&context, 0, 0, out.gpu_address())
        .expect("bind");

    let mut stepper = Stepper::new(&context).expect("stepper");
    let timing = stepper
        .run(|step| {
            step.set_argument_table_for(&tables, 0)?;
            step.set_pipeline(&pipeline);
            step.dispatch([THREADS, 1, 1], [THREADGROUP, 1, 1])
        })
        .expect("run");

    // `timing.gpu` is allowed to be absent -- the feedback is asynchronous
    // and this is precisely why it is an Option rather than a zero. The
    // number is keyed by the step, so ask for it.
    let reported = timing.gpu.or_else(|| {
        stepper
            .feedback()
            .await_step(timing.step, Duration::from_secs(2))
            .map(|f| f.gpu_time())
    });
    let Some(gpu) = reported else {
        panic!("no commit feedback for step {} after 2 s", timing.step);
    };

    println!(
        "gpu_exec {:.3} ms, gpu {:.3} ms",
        ms(timing.gpu_exec),
        ms(gpu)
    );
    assert!(gpu > Duration::ZERO, "the GPU ran something");
    assert!(
        gpu <= timing.gpu_exec,
        "the GPU cannot have spent longer executing than the host spent \
         waiting for it: gpu {:.3} ms, gpu_exec {:.3} ms",
        ms(gpu),
        ms(timing.gpu_exec)
    );

    let with_report = driver_metal::device::Timing {
        gpu: Some(gpu),
        ..timing
    };
    assert_eq!(
        with_report.overhead(),
        Some(timing.gpu_exec - gpu),
        "overhead is the wait the GPU does not account for"
    );
}

#[test]
fn segments_accumulate_both_host_halves() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let pipeline = compiler
        .compile(&context, &source(100_000), "spin")
        .expect("spin");
    let pool = Pool::new(1 << 20);
    let out = pool.acquire(&context, (THREADS * 4) as u64).expect("out");
    let mut tables = Tables::new();
    tables
        .bind_address(&context, 0, 0, out.gpu_address())
        .expect("bind");

    let mut stepper = Stepper::new(&context).expect("stepper");
    let encode = |step: &mut driver_metal::device::StepEncoder<'_>| {
        step.set_argument_table_for(&tables, 0)?;
        step.set_pipeline(&pipeline);
        step.dispatch([THREADS, 1, 1], [THREADGROUP, 1, 1])
    };

    let one = stepper.run(|step| encode(step)).expect("warm");
    let first_value = one.step;

    const SEGMENTS: usize = 4;
    let mut between_calls = 0usize;
    let four = stepper
        .run_segments(
            SEGMENTS,
            |_, step| encode(step),
            |_| {
                between_calls += 1;
                // Host work between segments must NOT be charged to the step.
                std::thread::sleep(Duration::from_millis(20));
                Ok(())
            },
        )
        .expect("segments");

    assert_eq!(between_calls, SEGMENTS);
    println!(
        "one: encode {:.3} gpu_exec {:.3} | four: encode {:.3} gpu_exec {:.3}",
        ms(one.encode),
        ms(one.gpu_exec),
        ms(four.encode),
        ms(four.gpu_exec)
    );

    assert!(
        four.gpu_exec > one.gpu_exec * 2,
        "four segments of the same kernel must have accumulated, not replaced: \
         {:.3} ms vs {:.3} ms",
        ms(four.gpu_exec),
        ms(one.gpu_exec)
    );
    assert!(
        four.encode > one.encode,
        "encoding four segments costs more than encoding one"
    );
    // 4 x 20 ms of sleeping happened inside `between`. If that had been
    // folded into the step, the total would exceed it.
    assert!(
        four.total() < Duration::from_millis(80),
        "the host's own work between segments was charged to the GPU: {:.3} ms",
        ms(four.total())
    );
    assert_eq!(
        four.step,
        first_value + SEGMENTS as u64,
        "the accumulated timing names the LAST segment"
    );
}

#[test]
fn a_run_that_encodes_nothing_is_zero_rather_than_a_refusal() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut stepper = Stepper::new(&context).expect("stepper");

    let empty = stepper
        .run_parallel(0, |_, _| unreachable!("no buffers were asked for"))
        .expect("empty batch");
    assert_eq!(empty, driver_metal::device::Timing::default());
    assert_eq!(empty.total(), Duration::ZERO);

    let none = stepper
        .run_segments(0, |_, _| unreachable!(), |_| unreachable!())
        .expect("no segments");
    assert_eq!(none, driver_metal::device::Timing::default());
}
