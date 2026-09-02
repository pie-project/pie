//! Pins the emitters over [`extended_stages`] — the traces that exist to reach
//! what `corpus_stages()` never does.
//!
//! `golden-{msl,cuda}/` are dumps of a C++ oracle that has since been deleted,
//! so their expected column can never be re-derived; adding cases there would
//! silently turn oracle-authored lines into self-authored ones. These cases are
//! therefore pinned separately, against this compiler's own output, and
//! `golden-extended/` says exactly that in its header.
//!
//! What that buys: not correctness — nothing here was ever checked against a
//! second implementation — but *change detection*. A rewrite that alters what
//! these 55 ops, 8 intrinsics, 4 schedules and 4 stages compile to has to say
//! so out loud instead of moving under a corpus that never looked.

#[path = "common/device_text.rs"]
mod device_text;
#[path = "common/msl_corpus.rs"]
mod msl_corpus;
#[path = "common/provenance.rs"]
mod provenance;

use std::fmt::Write as _;
use std::path::PathBuf;

use msl_corpus::{extended_stages, region_shape};

const HEADER: &str = "\
# A pin of the extended corpus (`msl_corpus::extended_traces`) -- the traces
# added to reach the ops, intrinsics, schedules and stages that
# `corpus_stages()` never compiles.
#
# Source of truth: THIS COMPILER. Unlike `golden-{msl,cuda}/`, no C++ oracle
# ever produced or reviewed these bytes -- they were never cross-checked against
# a second implementation. They pin *what the emitters do today*, so that a
# change to it shows up as a diff instead of passing unnoticed.
#
# A mismatch here is a question, not a verdict: decide whether the new output is
# right, then re-pin with `PTIR_REGEN=1 cargo test -p eta-compiler`.
#
# Where a kernel is digested, `bytes` and the hash cover what the *compiler*
# generated: the hand-written device text under `crates/eta-compiler/runtime/` is
# dropped first, so tuning a kernel does not re-pin every case. That the
# emitters still splice those blocks in whole is
# `every_live_device_file_is_spliced_into_a_kernel_verbatim`.
";

fn golden_extended_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden-extended")
}

fn compare(name: &str, body: &str) {
    let path = golden_extended_dir().join(format!("{name}.txt"));

    let how = provenance::Regenerate::Own { header: HEADER };
    if let Some(expected) = provenance::body_to_diff(&path, body, how) {
        provenance::assert_same_lines(body, &expected, name, "");
    }
}

/// The stage wire bytes, so a plan-encoding change is visible even where the
/// emitters happen to agree.
#[test]
fn extended_corpus_plans_are_pinned() {
    let mut body = String::new();
    for stage in &extended_stages() {
        let _ = writeln!(body, "=== {}", stage.id());
        let _ = writeln!(body, "stage_tag: {:#04x}", stage.stage_tag);
        let _ = writeln!(body, "signature: {:016x}", stage.plan.signature.hash);
        let _ = writeln!(
            body,
            "plan: lines={} fnv1a64=0x{:016x}",
            stage.debug.lines().count(),
            eta_ir::fnv1a64(stage.debug.as_bytes())
        );
        for (index, region) in stage.plan.fused.regions.iter().enumerate() {
            let _ = writeln!(body, "fused#{index}: {}", region_shape(region));
        }
    }
    compare("extended_plans", &body);
}

