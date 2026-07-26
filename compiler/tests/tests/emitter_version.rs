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

use msl_corpus::{corpus_bound, corpus_stages};
use pie_codegen::cuda::CUDA_GENERATED_EMITTER_VERSION;
use pie_codegen::metal::METAL_M1_EMITTER_VERSION;
use pie_codegen::program::{Backend, emit_program};

/// `(version, fingerprint)` for each backend.
///
/// Update both halves together. A change that leaves the fingerprint alone did
/// not change the emitted bytes and must not move the version either — a
/// gratuitous bump throws away every driver's cache.
const PINNED: &[(&str, u16, u64)] = &[
    (
        "cuda",
        CUDA_GENERATED_EMITTER_VERSION,
        0x0977_b2ba_53d8_6682,
    ),
    ("metal", METAL_M1_EMITTER_VERSION, 0xa20e_c47b_00d3_f732),
];

/// Everything a driver receives for the corpus, hashed.
///
/// Refusals are hashed alongside sources because a region that stops being
/// emittable changes what the driver runs just as much as one whose source
/// changes, and the version has to move for both.
fn fingerprint(backend: Backend) -> u64 {
    let stages: Vec<_> = corpus_stages()
        .into_iter()
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
