//! A run fences the staging slot it read on its own completion, not on an
//! output: a program that outputs nothing still holds its slot until it has
//! run, and the slot is written again only after. Skipped without a PJRT
//! plugin.

use engine_xla::inputs::ROW_WORDS;
use engine_xla::pjrt::Arg;
use engine_xla::store::stage::{Done, Ring};
use engine_xla::trace::{Signature, Source};

/// A program reading a staging slot of `rows` rows and returning nothing.
fn reads_and_returns_nothing(rows: u32) -> String {
    format!(
        "module @nothing {{\n\
         func.func public @main(%arg0: tensor<{rows}x{ROW_WORDS}xui32> \
         {{mhlo.memory_kind = \"pinned_host\"}}) -> () {{\n\
         \"func.return\"() : () -> ()\n\
         }}\n\
         }}\n"
    )
}

#[test]
fn an_output_less_run_fences_the_slot_it_read() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let device = engine_xla::device::Device::open(None, 0).expect("the device opens");
    let rows = 8;
    let mut ring = Ring::lay(&device, 1, rows).expect("a staging ring lays");
    let at = ring
        .write(&[7; ROW_WORDS])
        .expect("the slot takes a fire's inputs");
    let program = device
        .program(
            &reads_and_returns_nothing(rows),
            Signature {
                params: vec![Source::Stage],
                results: Vec::new(),
            },
        )
        .expect("an output-less program compiles");
    let (outs, done) = device
        .run_fenced(&program, vec![Arg::Keep(ring.buffer(at))], false)
        .expect("it runs");
    assert!(outs.is_empty(), "the program outputs nothing");
    let done = done.expect("a run not waited for answers the event it completes by");
    ring.read_by(at, Some(Done::of(done)));
    assert!(ring.fenced(at), "the slot waits on the run that read it");
    // Writing the slot again waits for that run, then frees the fence.
    ring.write(&[9; ROW_WORDS])
        .expect("the slot is written once the run has completed");
    assert!(
        !ring.fenced(at),
        "a written slot waits on nothing until a run reads it"
    );
}
