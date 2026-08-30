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
use eta_compiler::codegen::cuda::CUDA_GENERATED_EMITTER_VERSION;
use eta_compiler::codegen::metal::METAL_M1_EMITTER_VERSION;
use eta_compiler::codegen::program::{Backend, emit_program};

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
// Re-pinned a fourth and fifth time, WITH a cuda bump each, and both are
// changes in what an engine runs rather than in how its output is spelled.
// `emit_program` now writes source into slots that used to carry `generated
// region contains a non-generated boundary (...)` -- regions the CUDA shell
// had no way to run at all:
//
//   22 -> 23  `codegen::cuda::order` (then `topk`) took the single-node
//             `top_k` library region. `beam-search` routes every token
//             through one.
//   23 -> 24  the same emitter took `sort_desc`, which is `top_k` at `k = n`
//             and shares its kernel, and `codegen::cuda::scan` took
//             `cumsum`/`cumprod`. `locally-typical-sampling` and
//             `tail-free-sampling` cut their candidate set with `cumsum(p) -
//             p`; `mtp-speculative-decoding` builds its accept prefix with
//             `cumprod`.
//
// Nothing else moved in either: `emit_fused_region`'s bytes are unchanged
// (`golden-cuda/` needed no regeneration, only its version stamp), and metal is
// untouched, which is why only the cuda row moves.
//
// The bump is not bookkeeping here. `engine::cache_identity` folds the emitter
// version into the key `engine_cuda::program::Cache` files its NEGATIVE tier
// under, so the old version is the key a shell already remembers "this guest
// program does not compile here" for. Without the bump those remembered
// refusals outlive the fix for the life of the process.
// Re-pinned a sixth time, WITHOUT a bump, for the `tensor-*` -> `eta-*` crate
// rename. `include/ptir_rng.generated.metal` opens with a provenance line
// naming the Rust source it is generated from; that line moved with the
// directory (`crates/tensor-ir/src/rng.rs` -> `crates/eta-ir/src/rng.rs`) and
// `rng_contract` holds the file to its generator byte for byte. The preamble is
// spliced verbatim into every emitted Metal source, so one comment byte fewer
// reaches every kernel and this hash moves. Nothing an MSL compiler reads
// changed, and the reasoning is the third re-pin's: a cache keyed on version 36
// can only hand back a library built from a source that differs from today's in
// a comment. The bump is also not free here, which is why it was weighed rather
// than taken by reflex -- the version is spelled into generated function names
// (`ptir_m3_generic_ready_v36`), so moving it re-pins `golden-msl/`, and those
// dumps are the deleted C++ oracle's only surviving record. Discarding evidence
// to restamp a comment is the worse trade. CUDA does not splice this preamble,
// so its row does not move at all.
// Re-pinned a seventh time, WITHOUT a bump, for the first paragraph's reason:
// the corpus moved, not the emitters. `IntrinsicId::AttnScore` is Epilogue-only
// and rectangle-shaped now, so `extended_on_attn` reaches `Stage::OnAttn` on
// `Layer` alone and a new `extended_attn_score` reads the rectangle where it is
// legal. Both rows move because the entry name carries the plan signature and
// two signatures are new; every case whose trace did not change re-emitted byte
// for byte, which is why `golden-{cuda,msl}/` and the untouched half of
// `golden-extended/` needed no re-pin. A cache keyed on 24 or 36 can only hand
// back a build of the source those versions still emit.
// Re-pinned an eighth time, WITH a metal bump, and this one is a change in
// what an engine runs rather than in how it is spelled. `emit_fused_region`
// used to bind `logits [[buffer(6)]]` and make it the first argument of EVERY
// `INTRINSIC_VAL` op, so a region reading `mtp_logits` beside `logits` read one
// rectangle twice -- the draft column answered by the trunk's, with no fault
// anywhere. `codegen::metal::intrinsics` gives each intrinsic an argument index
// of its own (the trunk keeps 6; the rest come down from 30) and the emitter
// routes each op's `a0` to its own, so three cases in `golden-msl/` grew a
// buffer parameter and changed which pointer an op is handed. A library cached
// under 36 for one of those programs is built from the one-buffer source, which
// is exactly the hand-back this file's preamble names -- so:
//
//   36 -> 37  the M2 intrinsic slot table. `engine-metal` binds a rectangle per
//             intrinsic and points `IntrinsicId::MtpLogits` at the `mtp`
//             export, which is what let `has_mtp_logits` stop being a
//             statement about the ABI.
//
// CUDA is untouched -- its kernel has read a per-intrinsic slot table since it
// was written -- so its row does not move.
// Re-pinned a ninth time, WITH a metal bump, and it is the eighth's argument
// one intrinsic further on. `IntrinsicId::AttnScore` was the one id the M2 slot
// table left out, and the reason was never the table: a score plane is F32 and
// the `0xA0` handler read `bfloat`, so a slot for it would have been an index
// without a reader. `ptir_m1_runtime.metal`'s handler now branches on `p.intr`
// and gathers `float` for that id, `codegen::metal::intrinsics` gives it
// argument index 28, and `emit_fused_region` routes the op's `a0` there -- so a
// region reading `attn_score` emits MSL where it used to emit
// `UnbindableIntrinsic`, and the emitted source for it did not exist under 37
// at all:
//
//   37 -> 38  the attention-score door. `engine-metal` carves an observability
//             slab, the capture arm writes it, and an epilogue's
//             `IntrinsicId::AttnScore` is bound at the lane's block of it --
//             which is what let `has_attn_score` stop being a statement about
//             the element type.
//
// The runtime text changed, which is why `golden-msl/oracle-inputs/` exists
// now: those dumps are the deleted C++ oracle's only record and are compared
// against the text it was taken with (`device_text.rs`). What moved in them is
// exactly the version spelled into `ptir_m3_generic_{ready,commit}_v38`.
//
// CUDA is untouched -- its handler has read a per-binding storage mode since it
// was written -- so its row does not move.
// Re-pinned a tenth time, WITH a metal bump, and this one is a change in what
// an engine COMPUTES rather than in what it binds. `ptir_m1_runtime.metal`'s
// `0x58` (`PivotThreshold`) carried the two forms the CUDA runtime deleted:
// `RankLe` re-scanned the whole row per element and `CummassLe` re-scanned the
// already-picked list per candidate, which is O(len^2) and O(len^3) on ONE
// thread. This shell runs only `KernelKind::Fused`, so those loops ARE the
// sampler -- at qwen35-d0.8b's 248320-token vocabulary the second is >10^16
// steps, and every inferlet that samples hung the device rather than answering
// slowly. Both arms are now `ptir_m1_runtime_body.cuh`'s, verbatim: a 4-pass
// 8-bit MSB radix select on a new `m1_desc_key`, and the last pick's
// total-order key as the availability threshold:
//
//   38 -> 39  the pivot predicates stop being quadratic. The picks and the keep
//             bits are bit-identical -- `m1_sort_better` is a strict total
//             order and the radix key is monotone in it -- which is what
//             `engine-metal`'s `program_parity` holds them to.
//
// The runtime text is spliced into every fused source, so this moves the whole
// metal fingerprint; what moved in `golden-msl/` is again only the version
// spelled into `ptir_m3_generic_{ready,commit}_v39`. CUDA took this fix first
// and its row does not move.
const PINNED: &[(&str, u16, u64)] = &[
    ("cuda", 24, 0xc692_ce36_f07d_34df),
    ("metal", 39, 0x95ca_0859_8ea8_afeb),
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
    eta_ir::fnv1a64(&bytes)
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
