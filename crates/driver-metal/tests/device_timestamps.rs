//! GPU timestamps that a real step actually wrote.
//!
//! The heap, the mark and the resolve can all succeed while measuring
//! nothing: a heap allocates, an out-of-range mark is undefined rather than
//! reported, and a heap that was never written resolves to a vector of
//! zeroes. So the load-bearing test here is not "resolve returned something"
//! -- it is that a step which dispatches a kernel between two marks produces
//! two NON-ZERO ticks in the order they were encoded.
//!
//! Requires a Metal 4 GPU, and skips without one.


use driver_metal::Error;
use driver_metal::device::{ArgumentTable, Context, Granularity, Heap, Stepper, Timestamps, Visibility};
use driver_metal::program::Compiler;

/// Enough arithmetic that the two marks cannot land in the same tick by the
/// GPU clock simply not having advanced.
const BUSY: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void busy(device uint* out [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
    uint acc = gid;
    for (uint i = 0; i < 4096u; ++i) { acc = acc * 1664525u + 1013904223u; }
    out[gid] = acc;
}
";

const COUNT: usize = 4096;
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

/// A heap of no entries is a heap every later mark misses. The C++ answers
/// this with `nullptr`, which turns every subsequent `mark_timestamp` into a
/// silent no-op.
#[test]
fn a_zero_count_heap_is_refused_rather_than_returned_as_nothing() {
    let Some(Fixture { context, .. }) = fixture() else {
        return;
    };
    let err = Timestamps::new(&context, 0).expect_err("zero entries is refused");
    assert!(matches!(err, Error::Create { .. }), "{err}");
}

#[test]
fn a_heap_reports_the_count_it_was_asked_for() {
    let Some(Fixture { context, .. }) = fixture() else {
        return;
    };
    let timestamps = Timestamps::new(&context, 8).expect("heap");
    assert_eq!(timestamps.count(), 8);
}

/// Metal's behaviour past the end of a counter heap is undefined and
/// unreported, so the refusal has to happen here -- and it has to name the
/// bound, because "out of range" without it does not say what the heap was
/// sized for.
#[test]
fn marking_an_out_of_range_index_is_refused_and_names_the_bound() {
    let Some(Fixture { context, .. }) = fixture() else {
        return;
    };
    let timestamps = Timestamps::new(&context, 2).expect("heap");
    let mut stepper = Stepper::new(&context).expect("stepper");

    let err = stepper
        .run(|step| step.mark_timestamp(&timestamps, 2, Granularity::Relaxed))
        .expect_err("index 2 is past a two-entry heap");

    match err {
        Error::OutOfRange {
            what, offset, len, ..
        } => {
            assert_eq!(what, "timestamp index");
            assert_eq!(offset, 2);
            assert_eq!(len, 2, "the message must carry the heap's count");
        }
        other => panic!("{other}"),
    }
    assert!(!stepper.is_wedged());
}

/// A heap no step ever wrote resolves to zeroes rather than failing, and
/// nothing about reading it is undefined -- the entries exist from creation.
#[test]
fn resolving_a_heap_that_was_never_written_yields_zeroes_without_panicking() {
    let Some(Fixture { context, .. }) = fixture() else {
        return;
    };
    let timestamps = Timestamps::new(&context, 4).expect("heap");
    let ticks = timestamps.resolve().expect("resolve");
    assert_eq!(ticks.len(), 4);
    assert!(
        ticks.iter().all(|&t| t == 0),
        "an unwritten heap should read as zeroes, got {ticks:?}"
    );
}

/// The one that fails if `mark_timestamp` does nothing.
///
/// Two marks around a dispatch that takes real time. Both must be non-zero --
/// zero is what an unwritten entry reads as -- and the second must not
/// precede the first, because they were encoded in that order into one
/// encoder with a barrier between them.
#[test]
fn a_step_writes_two_ticks_that_are_non_zero_and_in_encode_order() {
    let Some(Fixture {
        context,
        compiler,
        mut heap,
    }) = fixture()
    else {
        return;
    };

    let pipeline = compiler.compile(&context, BUSY, "busy").expect("compiles");
    let bytes = (COUNT * size_of::<u32>()) as u64;
    let slot = heap.alloc(&context, bytes, 1).expect("slot");
    let table = ArgumentTable::new(&context, 1).expect("table");
    table.bind(0, &slot).expect("bind");

    let timestamps = Timestamps::new(&context, 2).expect("heap");
    let mut stepper = Stepper::new(&context).expect("stepper");

    stepper
        .run(|step| {
            step.mark_timestamp(&timestamps, 0, Granularity::Precise)?;
            step.set_pipeline(&pipeline);
            step.set_argument_table(&table);
            step.dispatch([COUNT, 1, 1], [64, 1, 1])?;
            step.barrier(Visibility::Device);
            step.mark_timestamp(&timestamps, 1, Granularity::Precise)
        })
        .expect("the step ran");

    // `run` waited the shared event, which is exactly the synchronisation a
    // CPU-timeline resolve needs.
    let ticks = timestamps.resolve().expect("resolve");
    assert_eq!(ticks.len(), 2);
    assert!(
        ticks[0] != 0 && ticks[1] != 0,
        "an unmarked heap reads as zeroes, so {ticks:?} means nothing was written"
    );
    assert!(
        ticks[1] >= ticks[0],
        "the second mark was encoded after the first: {ticks:?}"
    );
}

/// A relaxed mark is the default and must still write, so that the cheap
/// granularity is not silently a no-op.
#[test]
fn a_relaxed_mark_writes_a_tick_too() {
    let Some(Fixture { context, .. }) = fixture() else {
        return;
    };
    let timestamps = Timestamps::new(&context, 1).expect("heap");
    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .run(|step| step.mark_timestamp(&timestamps, 0, Granularity::Relaxed))
        .expect("the step ran");

    let ticks = timestamps.resolve().expect("resolve");
    assert_eq!(ticks.len(), 1);
    assert_ne!(ticks[0], 0, "a relaxed mark wrote nothing");
}

/// Dropping the heap is the release. There is no `release_timestamp_heap` to
/// call, and no context-wide array that keeps the heap alive after this.
#[test]
fn dropping_a_heap_releases_it_without_a_release_call() {
    let Some(Fixture { context, .. }) = fixture() else {
        return;
    };
    for _ in 0..64 {
        let timestamps = Timestamps::new(&context, 16).expect("heap");
        assert_eq!(timestamps.count(), 16);
        drop(timestamps);
    }
}
