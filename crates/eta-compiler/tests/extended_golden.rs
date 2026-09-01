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
use eta_compiler::codegen::error::EmitError;

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

/// A kernel is pinned by length plus digest, not verbatim: the point is to
/// notice that it changed, and 200 KB of inlined runtime in the diff hides that
/// rather than showing it. For the same reason the hand-written device text is
/// dropped before hashing — `bytes` counts what the compiler generated, so a
/// kernel engineer tuning a `.cuh` does not re-pin all 55 ops.
fn digest(emitted: &Result<String, EmitError>, device: &[String]) -> String {
    match emitted {
        Ok(source) => {
            let generated = device_text::elide_device_text(source, device);
            format!(
                "ok bytes={} fnv1a64=0x{:016x}",
                generated.len(),
                eta_ir::fnv1a64(generated.as_bytes())
            )
        }
        Err(error) => format!("err {error}"),
    }
}

/// The device text `digest` drops, read from the tree the emitters `include_str!`.
fn device_text() -> Vec<String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime");
    let texts = device_text::live_device_text(&root);
    assert_eq!(
        texts.len(),
        8,
        "crates/eta-compiler/runtime/ holds eight hand-written device files; if that \
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
    use eta_compiler::codegen::cuda;
    use eta_compiler::codegen::metal;

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
                    verdict
                        .as_ref()
                        .err()
                        .map_or(String::new(), |error| error.to_string())
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
            eta_ir::fnv1a64(stage.debug.as_bytes())
        );
        for (index, region) in stage.plan.fused.regions.iter().enumerate() {
            let _ = writeln!(body, "fused#{index}: {}", region_shape(region));
        }
    }
    compare("extended_plans", &body);
}

/// **THE SCORE RECTANGLE BINDS AT ITS OWN INDEX AND IS READ THROUGH ITS OWN
/// NAME** (`.wiki/alto/attn-score.md` §4, §6.2).
///
/// The golden above pins this region by length and digest, which says THAT
/// the emitted MSL moved and not what it says. What the observability door
/// actually needs is two sentences of that source, and both of them are the
/// mis-binding the whitelist exists to prevent read from the other side: the
/// kernel takes a parameter at the slot table's own index for
/// `IntrinsicId::AttnScore`, and the `INTRINSIC_VAL` op's `a0` is that
/// parameter rather than the trunk's `logits`. A regression in either would
/// leave a kernel that runs, faults nothing, and hands a guest the logits
/// under the score's name.
///
/// **THE GROUPED HALF IS THE SAME SENTENCE ABOUT A DIFFERENT KIND OF
/// BINDING**, and it used to be a refusal. That form takes no per-intrinsic
/// parameter at all — every rectangle it reads arrives as an address on the
/// lane record — so for as long as `lane.logits_base` was the only such
/// address the score slab was unreachable and the emitter said so by name.
/// `LaneRecord::attn_score_base` is the second address, and what is asserted
/// here is that the gather reads THAT one: a grouped kernel that reached for
/// `logits` instead would be the identical mis-binding, one indirection
/// further in and with no argument index to make it visible.
#[test]
fn the_score_rectangle_is_emitted_at_its_own_buffer_index() {
    use eta_compiler::codegen::metal;

    let at = metal::m2_intrinsic_buffer(eta_ir::op::intrinsic_tags::ATTN_SCORE)
        .expect("the score rectangle has a slot");
    let stage = extended_stages()
        .into_iter()
        .find(|stage| stage.id().starts_with("extended_attn_score"))
        .expect("the extended corpus carries a score-reading stage");
    let region = stage
        .plan
        .fused
        .regions
        .first()
        .expect("the score-reading stage fuses one region");

    let source = metal::emit_fused_region("ptir_fused_probe_r0", &stage.plan, region)
        .expect("the score rectangle is bindable on the single-lane path");
    let parameter = format!("const device uchar* intrinsic_7 [[buffer({at})]]");
    assert!(
        source.contains(&parameter),
        "the emitted kernel does not take `{parameter}`"
    );
    assert!(
        source.contains("ptir_m1_execute(160u, status, descriptors, params + 0, intrinsic_7,"),
        "the `intrinsic_val` op's a0 is not the score rectangle"
    );

    let grouped = metal::emit_grouped_fused_region("ptir_grouped_probe_r0", &stage.plan, region)
        .expect("the score rectangle has a lane-record address on the grouped path");
    assert!(
        grouped.contains(
            "const device float* score_planes = \
             reinterpret_cast<const device float*>(lane.attn_score_base);"
        ),
        "the grouped gather does not read the lane record's score base"
    );
    assert!(
        !grouped.contains("score_planes = reinterpret_cast<const device float*>(lane.logits_base)"),
        "the grouped gather reaches the score rectangle off the READOUT base, which is \
         the mis-binding this door exists to prevent"
    );
    // The F32 arm, stated where the M2 form states it as an element width: a
    // per-key mass is a probability a policy divides by, and reading the slab
    // as `bfloat` would halve every plane into the next one's keys.
    assert!(
        grouped.contains("device float* score_out = reinterpret_cast<device float*>("),
        "the grouped gather does not write the score rectangle as F32"
    );
    // A zero base is the absence of a captured block, and the device refuses
    // it rather than dereferencing null — the guest-plane half of
    // `program::session`'s third guard.
    assert!(
        grouped.contains("lane.attn_score_base == 0ul"),
        "the grouped gather does not refuse an unbound score base"
    );
}
