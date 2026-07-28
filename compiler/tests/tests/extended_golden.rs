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
# right, then re-pin with `PTIR_REGEN=1 cargo test -p pie-compiler-tests`.
#
# Where a kernel is digested, `bytes` and the hash cover what the *compiler*
# generated: the hand-written device text under `compiler/codegen/runtime/` is
# dropped first, so tuning a kernel does not re-pin every case. That the
# emitters still splice those blocks in whole is
# `every_live_device_file_is_spliced_into_a_kernel_verbatim`.
";

fn golden_extended_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden-extended")
}

/// A kernel is pinned by length plus digest, not verbatim: the point is to
/// notice that it changed, and 200 KB of inlined runtime in the diff hides that
/// rather than showing it. For the same reason the hand-written device text is
/// dropped before hashing — `bytes` counts what the compiler generated, so a
/// kernel engineer tuning a `.cuh` does not re-pin all 55 ops.
fn digest(emitted: &Result<String, String>, device: &[String]) -> String {
    match emitted {
        Ok(source) => {
            let generated = device_text::elide_device_text(source, device);
            format!(
                "ok bytes={} fnv1a64=0x{:016x}",
                generated.len(),
                pie_ir::fnv1a64(generated.as_bytes())
            )
        }
        Err(error) => format!("err {error}"),
    }
}

/// The device text `digest` drops, read from the tree the emitters `include_str!`.
fn device_text() -> Vec<String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../codegen/runtime");
    let texts = device_text::live_device_text(&root);
    assert_eq!(
        texts.len(),
        8,
        "compiler/codegen/runtime/ holds eight hand-written device files; if that \
         changed, this pin is eliding a different set than it was written for"
    );
    texts
}

fn compare(name: &str, body: &str) {
    let path = golden_extended_dir().join(format!("{name}.txt"));

    let how = provenance::Regenerate::Own { header: HEADER };
    if let Some(expected) = provenance::body_to_diff(&path, body, how) {
        provenance::assert_same_lines(body, &expected, name, "");
    }
}

#[test]
fn extended_corpus_emitters_are_pinned() {
    use pie_codegen::cuda;
    use pie_codegen::metal;

    let stages = extended_stages();
    assert!(!stages.is_empty(), "extended corpus compiled to nothing");
    let device = device_text();

    let mut body = String::new();
    for stage in &stages {
        let signature = format!("{:016x}", stage.plan.signature.hash);
        for (partition_name, partition) in [
            ("singleton", &stage.plan.singleton),
            ("fused", &stage.plan.fused),
        ] {
            for (index, region) in partition.regions.iter().enumerate() {
                let entry = format!("ptir_fused_{signature}_r{index}");
                let _ = writeln!(body, "=== {} {partition_name}#{index}", stage.id());
                let _ = writeln!(body, "region: {}", region_shape(region));
                let _ = writeln!(
                    body,
                    "cuda_fused: {}",
                    digest(
                        &cuda::emit_fused_region(&entry, &stage.plan, region),
                        &device
                    )
                );
                let verdict = cuda::validate_generated_region(&stage.plan, region);
                let _ = writeln!(
                    body,
                    "cuda_validate: ok={} error={}",
                    verdict.is_ok(),
                    verdict.as_ref().err().map_or("", |error| error.as_str())
                );
                let _ = writeln!(
                    body,
                    "cuda_second_party: {}",
                    cuda::second_party_region_supported(&stage.plan, region)
                );
                let _ = writeln!(
                    body,
                    "metal_fused: {}",
                    digest(
                        &metal::emit_fused_region(&entry, &stage.plan, region),
                        &device
                    )
                );
                let _ = writeln!(
                    body,
                    "metal_grouped: {}",
                    digest(
                        &metal::emit_grouped_fused_region(&entry, &stage.plan, region),
                        &device
                    )
                );
            }
        }
    }
    compare("extended_emitters", &body);
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
            pie_ir::fnv1a64(stage.debug.as_bytes())
        );
        for (index, region) in stage.plan.fused.regions.iter().enumerate() {
            let _ = writeln!(body, "fused#{index}: {}", region_shape(region));
        }
    }
    compare("extended_plans", &body);
}
