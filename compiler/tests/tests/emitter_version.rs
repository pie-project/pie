//! The emitter version constants change whenever the emitted bytes do.
//!
//! Both drivers key their compiled-kernel cache on the emitter version. A
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
//! So each backend pins its version against a hash of everything
//! `emit_program` produces for the corpus. Changing an emitter changes the
//! hash, and the only way to make this pass again is to write down a new
//! version next to a new hash, in one diff. The hash is a fingerprint, not a
//! golden: it says *that* the output changed, and the golden dumps say what to.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;

use msl_corpus::{corpus_bound, corpus_stages, extended_stages};
use pie_codegen::cuda::CUDA_GENERATED_EMITTER_VERSION;
use pie_codegen::metal::METAL_M1_EMITTER_VERSION;
use pie_codegen::program::{Backend, emit_program};

/// `(backend, version, fingerprint)`, both numbers written out as literals.
///
/// The version is a literal rather than the constant itself so that the two sit
/// on one line and neither can be updated without looking at the other.
/// Substituting the constant would make the version half true by construction,
/// and the minimal way to get a green test after changing an emitter would be
/// to paste a new fingerprint and leave the version alone — which is the exact
/// failure this file exists to prevent.
///
/// A change that leaves the fingerprint alone did not change the emitted bytes
/// and must not move the version either: a gratuitous bump throws away every
/// driver's cache.
const PINNED: &[(&str, u16, u64)] = &[
    ("cuda", 19, 0xe378_da9c_00f6_fda5),
    ("metal", 35, 0x16e9_a27d_8fc5_3b4b),
];

/// Everything a driver receives for both corpora, hashed.
///
/// The extended corpus is included because it exists to reach what the base one
/// does not — further ops, further intrinsics, the hierarchical-row schedule —
/// and the emitters have per-op arms, so a change confined to those paths would
/// otherwise leave this hash untouched while changing what a driver runs.
///
/// Refusals are hashed alongside sources because a region that stops being
/// emittable changes what the driver runs just as much as one whose source
/// changes, and the version has to move for both.
fn fingerprint(backend: Backend) -> u64 {
    let stages: Vec<_> = corpus_stages()
        .into_iter()
        .chain(extended_stages())
        .map(|stage| stage.plan)
        .collect();
    let mut bytes = Vec::new();
    for kernel in emit_program(backend, &stages, &corpus_bound()) {
        bytes.extend_from_slice(&kernel.kind.to_le_bytes());
        bytes.extend_from_slice(&kernel.stage_index.to_le_bytes());
        bytes.extend_from_slice(&kernel.region_index.to_le_bytes());
        for text in [&kernel.entry_name, &kernel.source, &kernel.error] {
            bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
        }
    }
    pie_ir::fnv1a64(&bytes)
}

fn backend_of(name: &str) -> Backend {
    match name {
        "cuda" => Backend::Cuda,
        "metal" => Backend::Metal,
        other => panic!("no backend named `{other}`"),
    }
}

/// The pinned version literals are the constants the drivers are compiled
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
             nothing and every driver's cache was discarded."
        );
    }
}

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
        "the emitted bytes changed, so the drivers' compile caches must be \
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
