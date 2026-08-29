//! Where the emitter version constants meet the bytes they describe.
//!
//! Both engines key their compiled-kernel cache on the emitter version. A
//! version that stays put across a change to the emitted text is not a stale
//! comment — it is a cache that hands back a cubin or a `MTLLibrary` built from
//! the *old* source for a program the compiler now emits differently, with no
//! error anywhere along the way.
//!
//! Nothing enforced that. CUDA's only guard compared the constant to the number
//! recorded in a golden's provenance header, and regenerating a golden
//! deliberately preserves that header — so "changed body, unchanged constant"
//! passed. Metal had no guard at all.
//!
//! So each backend pins both numbers here: the version against the constant the
//! engines are compiled with, and a fingerprint against a hash of everything
//! `emit_program` produces for both corpora. Neither number can move without
//! failing this file — an emitter change moves the fingerprint, a constant
//! change moves the version — so both have to be restated here, in the commit
//! that changed them.
//!
//! What this does *not* do is make one imply the other. Every expected value
//! here is a literal in this file, so the cheapest way to go green after
//! changing an emitter is still to paste the new fingerprint and leave the
//! version alone. No test that lives beside the values it checks can close
//! that: the edit that repairs the check is free to move the number the check
//! reads. What it buys is that the change cannot pass unnoticed — someone has
//! to open this file and read a failure that names the constant to bump — and
//! that whichever version is written here is the one being shipped.
//!
//! Closing it for real needs a cache key the compiler cannot forget rather than
//! a stricter test: the engines are already handed the `source` they compile,
//! so a key derived from that would not need a maintained number at all.
//!
//! The hash is a fingerprint, not a golden: it says *that* the output changed,
//! and the golden dumps say what to.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;

use msl_corpus::{corpus_bound, corpus_stages, extended_stages};
use tensor_compiler::codegen::cuda::CUDA_GENERATED_EMITTER_VERSION;
use tensor_compiler::codegen::metal::METAL_M1_EMITTER_VERSION;
use tensor_compiler::codegen::program::{Backend, emit_program};

/// `(backend, version, fingerprint)`, both numbers written out as literals.
///
/// The version is a literal rather than the constant itself because a row that
/// reads the constant describes nothing: that half would be true whatever the
/// constant said, and the fingerprint beside it would stop recording which
/// compiler produced it.
///
/// A change that leaves the fingerprint alone did not change the emitted bytes
/// and must not move the version either — a gratuitous bump throws away every
/// engine's cache — so a constant that moves on its own fails
/// `the_pinned_versions_are_the_compiled_ones` until someone comes here and
/// says so in the same commit.
// Re-pinned WITHOUT a version bump when `lora_prologue` joined the corpus:
// the fingerprint is a hash over everything the corpus emits, so growing the
// corpus moves it even when no pre-existing case's bytes changed — and the
// oracle dumps (`golden-cuda/`, `golden-msl/`) show that growth was purely
// additive. Bumping the constants for that would discard every engine's
// compile cache over sources they would re-emit identically.
//
// Re-pinned again WITH both bumps already made, which is the case this file's
// preamble says it cannot force and here did not have to: cumsum/cumprod
// widened from F32-only to every numeric dtype, the m1 runtime's scan block
// now branches on the operand dtype, and that block is spliced verbatim into
// every emitted source on both backends. Whoever made it bumped `cuda` 19 -> 21
// and `metal` 35 -> 36 and re-pinned the oracle dumps — `golden-cuda/` and
// `golden-msl/` both carry that reason — and left only this table behind. So
// the versions below are restatements of constants that already describe their
// output, not a judgement about whether a cache should be discarded; the
// judgement was made and is recorded in the dumps.
// Re-pinned a third time, again without a bump, and again on the reasoning the
// paragraph above records rather than on convenience. `EmitError::
// GeneratedRegionHasBoundary` now carries the offending library op's own name,
// so the refusal reads `... (top_k)` instead of leaving the reader to add a
// print to find out which op the partitioner lifted. The fingerprint hashes
// `kernel.error` alongside `kernel.source`, deliberately, so it moved -- but
// not one byte of any emitted SOURCE did, and the regions whose text changed
// are exactly the ones that emit nothing at all. A cache keyed on version 22
// therefore cannot hand back a cubin built from anything: there is no cubin.
// Bumping would discard both engines' caches to re-emit identical sources.
const PINNED: &[(&str, u16, u64)] = &[
    ("cuda", 22, 0x5d98_7c38_15e8_2c41),
    ("metal", 36, 0x0f81_7250_caff_2a71),
];

/// Everything an engine receives for both corpora, hashed.
///
/// The extended corpus is included because it exists to reach what the base one
/// does not — further ops, further intrinsics, the hierarchical-row schedule —
/// and the emitters have per-op arms, so a change confined to those paths would
/// otherwise leave this hash untouched while changing what an engine runs.
///
/// Refusals are hashed alongside sources because a region that stops being
/// emittable changes what the engine runs just as much as one whose source
/// changes, and the version has to move for both.
fn fingerprint(backend: Backend) -> u64 {
    let stages: Vec<_> = corpus_stages()
        .into_iter()
        .chain(extended_stages())
        .map(|stage| stage.plan)
        .collect();
    let mut bytes = Vec::new();
    for kernel in emit_program(backend, &stages, &corpus_bound()) {
        bytes.extend_from_slice(&(kernel.kind as u32).to_le_bytes());
        bytes.extend_from_slice(&kernel.stage_index.to_le_bytes());
        bytes.extend_from_slice(&kernel.region_index.to_le_bytes());
        for text in [&kernel.entry_name, &kernel.source, &kernel.error] {
            bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
        }
    }
    tensor_ir::fnv1a64(&bytes)
}

fn backend_of(name: &str) -> Backend {
    match name {
        "cuda" => Backend::Cuda,
        "metal" => Backend::Metal,
        other => panic!("no backend named `{other}`"),
    }
}

/// The pinned version literals are the constants the engines are compiled
/// against, so `PINNED` describes this compiler rather than a past one.
#[test]
fn the_pinned_versions_are_the_compiled_ones() {
    for (name, version, _) in PINNED {
        let constant = match backend_of(name) {
            Backend::Cuda => CUDA_GENERATED_EMITTER_VERSION,
            Backend::Metal => METAL_M1_EMITTER_VERSION,
        };
        assert_eq!(
            *version, constant,
            "the {name} emitter version constant is {constant}, but PINNED still says {version}. \
             If the emitted bytes changed, update both; if they did not, the constant moved for \
             nothing and every engine's cache was discarded."
        );
    }
}

/// The emitted bytes are still the ones the pinned version was written for.
///
/// A failure means the engines would key a cache on a version that no longer
/// describes what they will be handed. The repair is to bump the backend's
/// constant and re-pin; this test sees only the re-pinning, so the bump is the
/// reader's to make.
#[test]
fn each_emitter_version_still_describes_its_output() {
    let mut moved = Vec::new();
    for (name, version, pinned) in PINNED {
        let actual = fingerprint(backend_of(name));
        if actual != *pinned {
            moved.push(format!(
                "{name}: emitter version {version} was pinned to {pinned:#018x} but now emits \
                 {actual:#018x}"
            ));
        }
    }
    assert!(
        moved.is_empty(),
        "the emitted bytes changed, so the engines' compile caches must be \
         invalidated:\n  {}\n\nBump the backend's emitter version constant and \
         put the new fingerprint in PINNED, in the same commit. If the change \
         was meant to be output-neutral, this is the bug report.",
        moved.join("\n  ")
    );
}

/// A fingerprint that does not depend on the emitted text would pass forever.
#[test]
fn the_fingerprint_reads_the_emitted_text() {
    assert_ne!(
        fingerprint(Backend::Cuda),
        fingerprint(Backend::Metal),
        "the two backends emit different source, so their fingerprints must differ"
    );
    assert_ne!(fingerprint(Backend::Cuda), 0);
}

/// Two runs of the same emitter agree, or the pin above could never hold.
#[test]
fn emission_is_deterministic() {
    for backend in [Backend::Cuda, Backend::Metal] {
        assert_eq!(fingerprint(backend), fingerprint(backend));
    }
}
