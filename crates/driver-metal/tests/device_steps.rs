//! Submitting more than one command buffer per step.
//!
//! Three shapes and the difference between them is the whole test:
//!
//! * [`Stepper::run`] -- one buffer, one signal.
//! * [`Stepper::run_parallel`] -- N buffers, ONE `commit:count:options:` and
//!   ONE signal. The saving is `N - 1` submissions; the price is that the
//!   buffers race.
//! * [`Stepper::run_segments`] -- N buffers, each committed and waited for
//!   before the next is encoded, with the host running in between.
//!
//! The claim that separates the first two from the third is the timeline: a
//! parallel batch advances the event by one no matter how many buffers went
//! in, and segments advance it once per segment. That is checked directly,
//! because a `run_parallel` that quietly fell back to N submissions would
//! produce identical output.

#![allow(clippy::print_stdout)]

use driver_metal::device::{Context, Pool, Stepper, Tables};
use driver_metal::program::Compiler;
use driver_metal::{Error, Region};

/// Copies one value into a span, so what a buffer holds says which chunk
/// wrote it and when.
const SPREAD: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void spread(device uint* out [[buffer(0)]],
                   const device uint* src [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
    out[gid] = src[0];
}
";

const CHUNKS: usize = 4;
const PER_CHUNK: usize = 64;

fn context() -> Option<Context> {
    match Context::new() {
        Ok(c) => Some(c),
        Err(Error::NoDevice) => None,
        Err(e) => panic!("context: {e}"),
    }
}

fn read_u32s(r: &impl Region, count: usize) -> Vec<u32> {
    // SAFETY: shared storage, wide enough, and the step that wrote it has
    // signalled.
    unsafe { std::slice::from_raw_parts(r.contents().as_ptr().cast::<u32>(), count) }.to_vec()
}

#[test]
fn a_parallel_batch_is_one_submission_and_one_timeline_point() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let pipeline = compiler
        .compile(&context, SPREAD, "spread")
        .expect("spread");
    let pool = Pool::new(1 << 20);

    let out = pool
        .acquire(&context, (CHUNKS * PER_CHUNK * 4) as u64)
        .expect("out");
    let tags = pool.acquire(&context, (CHUNKS * 4) as u64).expect("tags");
    let want_tags: Vec<u32> = (0..CHUNKS as u32).map(|k| 100 + k).collect();
    let bytes: Vec<u8> = want_tags.iter().flat_map(|v| v.to_le_bytes()).collect();
    // SAFETY: nothing is in flight and the slice is exactly the region.
    unsafe { tags.write(0, &bytes) }.expect("tags");

    // One ordinal per chunk. Each names a disjoint slice of `out`, which is
    // what makes the chunks hazard-free -- the requirement `run_parallel`
    // states and cannot check.
    let mut tables = Tables::new();
    for k in 0..CHUNKS {
        let ordinal = k as u32;
        tables
            .bind_address(
                &context,
                ordinal,
                0,
                out.gpu_address() + (k * PER_CHUNK * 4) as u64,
            )
            .expect("out");
        tables
            .bind_address(&context, ordinal, 1, tags.gpu_address() + (k * 4) as u64)
            .expect("tag");
    }

    let mut stepper = Stepper::new(&context).expect("stepper");
    let before = stepper.steps();
    stepper
        .run_parallel(CHUNKS, |k, step| {
            step.set_pipeline(&pipeline);
            step.set_argument_table_for(&tables, k as u32)?;
            step.dispatch([PER_CHUNK, 1, 1], [64, 1, 1])
        })
        .expect("the batch ran");

    assert_eq!(
        stepper.steps() - before,
        1,
        "{CHUNKS} command buffers advanced the timeline more than once, so they \
         were not one commit"
    );

    let got = read_u32s(&out, CHUNKS * PER_CHUNK);
    let want: Vec<u32> = (0..CHUNKS)
        .flat_map(|k| std::iter::repeat_n(100 + k as u32, PER_CHUNK))
        .collect();
    assert_eq!(got, want, "a chunk did not write its own slice");
}

/// Segments are ordered, and the host between them can change what the next
/// one reads. Nothing else in the crate can express that.
#[test]
fn the_host_runs_between_segments_and_after_the_last() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let pipeline = compiler
        .compile(&context, SPREAD, "spread")
        .expect("spread");
    let pool = Pool::new(1 << 20);

    const SEGMENTS: usize = 3;
    let out = pool.acquire(&context, (SEGMENTS * 4) as u64).expect("out");
    let tag = pool.acquire(&context, 4).expect("tag");
    // SAFETY: nothing is in flight.
    unsafe { tag.write(0, &1u32.to_le_bytes()) }.expect("tag");

    let mut tables = Tables::new();
    for i in 0..SEGMENTS {
        let ordinal = i as u32;
        tables
            .bind_address(&context, ordinal, 0, out.gpu_address() + (i * 4) as u64)
            .expect("out");
        tables
            .bind_address(&context, ordinal, 1, tag.gpu_address())
            .expect("tag");
    }

    let mut stepper = Stepper::new(&context).expect("stepper");
    let before = stepper.steps();
    let mut visited = Vec::new();
    stepper
        .run_segments(
            SEGMENTS,
            |i, step| {
                step.set_pipeline(&pipeline);
                step.set_argument_table_for(&tables, i as u32)?;
                step.dispatch([1, 1, 1], [1, 1, 1])
            },
            |i| {
                visited.push(i);
                // Reads what the segment just computed and decides what the
                // next one will read. A GPU-side chain could not do this;
                // that is the point of the callback.
                let seen = read_u32s(&out, SEGMENTS)[i];
                // SAFETY: this segment has signalled and the next is not yet
                // encoded, which is exactly the boundary `write` requires.
                unsafe { tag.write(0, &(seen * 2 + 1).to_le_bytes()) }
            },
        )
        .expect("the segments ran");

    assert_eq!(visited, vec![0, 1, 2], "the last segment got no callback");
    assert_eq!(
        read_u32s(&out, SEGMENTS),
        vec![1, 3, 7],
        "a segment did not see what the host wrote after the one before it"
    );
    assert_eq!(
        read_u32s(&tag, 1),
        vec![15],
        "the last callback did not run"
    );
    assert_eq!(
        stepper.steps() - before,
        SEGMENTS as u64,
        "segments are one submission each; that is what they cost"
    );
}

#[test]
fn a_failed_encode_commits_nothing_and_a_failed_callback_stops_the_rest() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let pipeline = compiler
        .compile(&context, SPREAD, "spread")
        .expect("spread");
    let pool = Pool::new(1 << 20);
    let out = pool.acquire(&context, 16).expect("out");

    let mut tables = Tables::new();
    tables
        .bind_address(&context, 0, 0, out.gpu_address())
        .expect("out");
    tables
        .bind_address(&context, 0, 1, out.gpu_address())
        .expect("src");

    let mut stepper = Stepper::new(&context).expect("stepper");
    let before = stepper.steps();

    // Chunk 2 asks for an ordinal nobody bound. A partial batch is work the
    // caller did not ask for, so none of it goes.
    let mut encoded = 0;
    let err = stepper
        .run_parallel(CHUNKS, |k, step| {
            encoded += 1;
            step.set_pipeline(&pipeline);
            step.set_argument_table_for(&tables, if k == 2 { 99 } else { 0 })?;
            step.dispatch([4, 1, 1], [4, 1, 1])
        })
        .unwrap_err();
    println!("refused: {err}");
    assert_eq!(encoded, 3, "encoding continued past the failure");
    assert_eq!(stepper.steps(), before, "a partial batch was committed");

    // A callback that fails stops the remaining segments from being encoded.
    let mut seen = 0;
    let err = stepper
        .run_segments(
            CHUNKS,
            |_, step| {
                step.set_pipeline(&pipeline);
                step.set_argument_table_for(&tables, 0)?;
                step.dispatch([4, 1, 1], [4, 1, 1])
            },
            |i| {
                seen = i + 1;
                if i == 1 {
                    return Err(Error::NoDevice);
                }
                Ok(())
            },
        )
        .unwrap_err();
    assert!(matches!(err, Error::NoDevice), "{err}");
    assert_eq!(seen, 2, "segments ran on past a failed callback");
    assert_eq!(
        stepper.steps() - before,
        2,
        "the segments after the failure were still submitted"
    );
}

/// A batch of nothing is not an error and does not move the timeline.
#[test]
fn an_empty_batch_costs_nothing() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .run_parallel(0, |_, _| panic!("nothing to encode"))
        .expect("empty");
    stepper
        .run_segments(
            0,
            |_, _| panic!("nothing to encode"),
            |_| panic!("no segments"),
        )
        .expect("empty");
    assert_eq!(stepper.steps(), 0);
}

/// Two fires in flight: the second is QUEUED while the first still runs.
///
/// **This is the invariant pie is built on** (`.wiki/new-driver/next.md`,
/// priority 1), and the reason it needed a new verb. `Stepper::run` ends in
/// `await_value`, so the call that would queue step n+1 cannot return until
/// step n has finished — the engine's `frame_dispatch_depth` is a number the
/// engine honours and the driver then serialises.
///
/// What this measures is not "submit returned quickly", which a fast GPU
/// would satisfy by accident. It submits a step, and then — BEFORE waiting
/// for it — submits a second one and asks the timeline how far it has got.
/// Two distinct values outstanding from one thread is the property; a `run`
/// loop cannot produce it at all, because it never holds two.
#[test]
fn a_second_step_is_queued_while_the_first_is_still_outstanding() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let pipeline = compiler
        .compile(&context, SPREAD, "spread")
        .expect("spread");
    let pool = Pool::new(1 << 20);
    let out = pool.acquire(&context, (PER_CHUNK * 4) as u64).expect("out");
    let tags = pool.acquire(&context, 4).expect("tags");
    // SAFETY: nothing is in flight and the slice is exactly the region.
    unsafe { tags.write(0, &7u32.to_le_bytes()) }.expect("tags");

    let mut tables = Tables::new();
    tables
        .bind_address(&context, 0, 0, out.gpu_address())
        .expect("out");
    tables
        .bind_address(&context, 0, 1, tags.gpu_address())
        .expect("tag");

    let mut stepper = Stepper::new(&context).expect("stepper");
    let encode = |step: &mut driver_metal::device::StepEncoder<'_>| {
        step.set_pipeline(&pipeline);
        step.set_argument_table_for(&tables, 0)?;
        step.dispatch([PER_CHUNK, 1, 1], [64, 1, 1])
    };

    let first = stepper.submit(encode).expect("the first step submits");
    // The second is encoded and committed WITHOUT the first having been
    // waited for. `submit` only blocks on the step two back, and there is
    // none.
    let second = stepper.submit(encode).expect("the second step submits");

    assert_eq!(
        second,
        first + 1,
        "two submissions must be two timeline points, or they were one commit"
    );
    assert_eq!(
        stepper.steps(),
        second,
        "the stepper's committed count must be the value it last handed out"
    );

    // Both retire, and the query that says so does not block.
    stepper
        .wait_for(second)
        .expect("the GPU reaches the second");
    assert!(
        stepper.has_passed(first) && stepper.has_passed(second),
        "the event passed {second} but reports otherwise"
    );
    assert!(
        !stepper.has_passed(second + 1),
        "a value nothing signalled must not read as passed, or the completion \
         path retires fires that never ran"
    );

    let got = read_u32s(&out, PER_CHUNK);
    assert_eq!(got, vec![7u32; PER_CHUNK], "the queued steps did not run");
}
