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
// Re-pinned an eleventh time, WITH a metal bump, and this one is a wrong
// answer rather than a slow one. `emit_grouped_fused_region`'s fused
// gather-and-argmax strode its columns by `constexpr uint am_w =
// METAL_M3_REGION_THREADS`, on the reading that a grouped launch is always
// that wide. It cannot be: a threadgroup's width is capped by the PIPELINE's
// `maxTotalThreadsPerThreadgroup`, which falls with register pressure, and
// `beam_epilogue`'s 67-op region measures 384 on an M1 Max. At any narrower
// launch the columns the missing threads owned were never visited and the
// argmax answered a confident wrong index. The stride is `m3_threads` now --
// the number of threads that actually exist -- so the column classes
// partition the row exactly at every width:
//
//   39 -> 40  the grouped argmax stops assuming its own launch width.
//             `engine-metal` binds `KernelKind::Grouped` for the first time in
//             this same wave (the twelve-channel ceiling and the single-thread
//             sampler are both the M2 form's), so this is the wave that could
//             first observe it -- and the bump is load-bearing rather than
//             hygienic, because a library cached under 39 for one of these
//             programs IS the 512-strided source.
//
// One golden moves, `emit_grouped_fused_region_msl.txt`, plus the version
// spelled into `ptir_m3_generic_{ready,commit}_v40`. `emit_fused_region`'s
// bytes are untouched -- the single-lane kernel has one thread and no stride
// to get wrong -- and CUDA does not share this emitter at all, so its row
// stands.
// Re-pinned a twelfth time, WITH a metal bump, and this one adds an argument
// rather than fixing one. `M3GroupLayout::vocab` is the grouped form's row
// pitch -- the kernel already multiplied a row index by it -- but the emitted
// gather spent that ONE number on two jobs: the pitch it strides the SOURCE by
// and the width it walks each row for. While the host wrote the reader's own
// declared width into it the two were the same number and the conflation was
// invisible; it is also exactly what made a narrow multi-row read
// inexpressible on this plane, because row `r` began `last` elements in rather
// than a whole rectangle row. `engine-metal`'s `program::launch` said so by
// name and refused the shape -- which is beam search's `[B, V]` logits read,
// and consensus decoding's.
//
// The two numbers are now two. The gather and the fused gather-and-argmax take
// their ROW WIDTH from `intrinsic_desc.last` / `am_in.last` -- the reader's own
// claim, which is where the single-lane `0xA0` handler reads it too -- and
// their PITCH from `layout->vocab`, which the host now fills with the
// RECTANGLE's width. The relation checked is the CUDA handler's and not a
// stricter one: `ptir_m1_runtime_body.cuh` faults on `stride < logical_width`
// and on nothing else about the two, so a declared row stays a CEILING.
//
//   40 -> 41  the grouped gather stops spending one word on the pitch and the
//             row width both. Where a reader is full-width the two are equal
//             and not a byte of behaviour moves; where it is narrow, this is
//             the whole of what `intrinsic_row_stride` buys the CUDA twin. The
//             bump is load-bearing rather than hygienic: a library cached
//             under 40 for one of these programs IS the source that strides
//             the source rows by the reader's width.
//
// One golden moves, `emit_grouped_fused_region_msl.txt`, plus the version
// spelled into `ptir_m3_generic_{ready,commit}_v41`. `emit_fused_region`'s
// bytes are untouched -- the single-lane form still has no stride to be told,
// and keeps its one-row refusal, now stated at encode where the form is known
// -- and CUDA does not share this emitter, so its row stands.
// Re-pinned a thirteenth time, WITH BOTH BUMPS, and it is the first entry in
// this ledger where the two rows move together — because what changed is the
// one struct they share. `LaneRecord` is declared once, in
// `codegen::layout`, and printed into the CUDA prologue, both MSL preambles
// and the `#[repr(C)]` the hosts pack. It grew two fields:
//
//   attn_score_base        the lane's block of the observability slab
//   attn_score_row_stride  that rectangle's plane pitch, in F32 elements
//
// **THE REASON IS THE GROUPED FORM'S ONE ABI RULE.** An M3 kernel binds no
// per-intrinsic buffer at all: every rectangle it reads arrives as a `ulong`
// on the lane record. `logits` and `mtp_logits` share one — the draft column
// is the same rectangle in a second row block, reached through
// `M3RowMeta::mtp_offset` — and `attn_score` was therefore the id
// `m3_intrinsic_bindable` refused by name, because the score slab is the
// SHELL's reservation (`engine_metal::scores`) and the readout is the arena's,
// so no displacement off `logits_base` was ever going to land in it. A grouped
// region reading it would have been emitted pointing at the trunk's rows,
// which is the silent mis-binding that whitelist exists to prevent. The
// rectangle has an address of its own now, `emit_grouped_fused_region` gathers
// its F32 planes from that base, and the M2 and M3 tables agree id for id:
//
//   41 -> 42  the score rectangle gets a lane-record address. What moved in
//             `golden-msl/` is exactly the +73 bytes of struct text the two
//             fields add to the M1 and M3 preambles, plus the version spelled
//             into `ptir_m3_generic_{ready,commit}_v42`; no corpus case reads
//             a score rectangle in a grouped region, so no kernel body grew an
//             arm. `golden-extended/`'s `extended_attn_score` is where the
//             behaviour shows: its `metal_grouped` column was
//             `UnbindableIntrinsic` and is emitted MSL.
//
//   24 -> 25  the same struct, on the plane that does not read the fields.
//             CUDA reaches the score rectangle through its five
//             per-(lane, intrinsic) side arrays and always has, so nothing
//             about its emitted logic moves — but the record is 96 -> 112
//             bytes on both planes and the lane table is an ARRAY of it. A
//             kernel cached under 24 strides sixteen bytes short per lane and
//             reads every lane after the first at the wrong offset, which is a
//             wrong answer rather than a refusal. That is precisely the
//             hand-back this file's preamble names, so the bump is
//             load-bearing on the row whose bytes did not otherwise change.
//
// The fields were APPENDED rather than folded in beside `logits_base`, so
// every offset a kernel already reads is where it was; only the stride between
// lanes moved. Both `golden-cuda/` bodies stand — `fused_block0.cuh` is
// kernel-side text with an oracle-era copy and `device_text.rs` puts it back —
// and only their version stamps move.
//
// **AND THE REASON THE RECORD GREW AT ALL IS A CEILING, NOT A CAPABILITY.**
// The M2 form can bind the score rectangle: it has had argument index 28 since
// 37 -> 38. What it cannot do is bind it beside many channels, because the
// intrinsic slots grow DOWN from 30 while the channels grow UP from 7 — so
// `fused_channel_ceiling` puts a score-reading region at ten channels instead
// of twelve. `trackb-h2o` wants ten channels and the scores, and was refused by
// an argument-space limit that has nothing to do with what it asked for. On the
// grouped form a channel is a row of the lane table and the score base is a
// word beside it, so neither crowds the other.
const PINNED: &[(&str, u16, u64)] = &[
    ("cuda", 25, 0xf481_2e2f_7393_1f3e),
    ("metal", 42, 0x166f_c41f_c8b1_ff87),
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
