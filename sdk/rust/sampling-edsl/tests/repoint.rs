//! #12/#15 option-(B) repoint validation gate (host-only, pre-GPU).
//!
//! `lower_sampler_standard` is the production lowering that replaces the sugar
//! `build_sampler` (which baked T/p as `constant_f32` immediates → param-variant,
//! un-recognizable bytecode). These tests are the **skew-proof**: the repointed
//! lowering emits the *canonical* `standard_program` — byte-identical to what the
//! driver baked, fixture-hash-matching — so `recognize()` hash-hits and
//! `extract_dedicated_params` reads the now-populated submit slots (T@key0,
//! p/min_p@key1). When this greens, the inferlet (once golf wires the helper in)
//! emits exactly what delta baked.

use pie_sampling_ir::program_hash;
use sampling_edsl::builder::Built;
use sampling_edsl::standard::{StandardSampler, standard_program};
use sampling_edsl::sugar::{SamplerSpec, lower_sampler_standard};
use sampling_edsl::CanonicalKind;

/// qwen3-0.6b — the HW-verify vocab; the byte-parity that matters on hardware.
const QWEN3_VOCAB: u32 = 151936;

fn hash_of(built: &Built) -> u64 {
    program_hash(&built.lower().bytecode)
}

fn f32le(x: f32) -> Vec<u8> {
    x.to_le_bytes().to_vec()
}

#[test]
fn repoint_emits_canonical_standard_program_with_fixture_hashes() {
    // The 3 phase-1 k-invariant RNG kinds the dedicated ladder extracts. Each
    // repointed lowering must be byte-identical to `standard_program(kind)` AND
    // hash to delta's exact baked hash (== foxtrot's committed fixture row).
    let cases = [
        (
            SamplerSpec::Multinomial { temperature: 0.8 },
            StandardSampler::Temperature,
            0x7d84_1977_776b_bb2d_u64,
        ),
        (
            SamplerSpec::MinP { temperature: 0.8, p: 0.05 },
            StandardSampler::MinP,
            0x9445_890b_0172_1734,
        ),
        (
            SamplerSpec::TopP { temperature: 0.8, p: 0.9 },
            StandardSampler::TopP,
            0xfdeb_b813_5fe2_48e7,
        ),
    ];
    for (spec, kind, fixture_hash) in cases {
        let (built, _submit) = lower_sampler_standard(spec, QWEN3_VOCAB).unwrap();
        let (std_bytes, _) = standard_program(kind, QWEN3_VOCAB).unwrap();
        assert_eq!(
            built.lower().bytecode,
            std_bytes,
            "repoint must emit the canonical standard_program for {kind:?}"
        );
        assert_eq!(
            hash_of(&built),
            fixture_hash,
            "{kind:?} program_hash must match delta's bake / the fixture"
        );
    }
}

#[test]
fn repoint_submit_values_are_keyed_t0_filter1() {
    // T@key0, p/min_p@key1 — exactly delta's `extract_dedicated_params` map.
    let (_b, submit) =
        lower_sampler_standard(SamplerSpec::TopP { temperature: 0.8, p: 0.9 }, QWEN3_VOCAB).unwrap();
    assert_eq!(submit, vec![(0u32, f32le(0.8)), (1u32, f32le(0.9))]);

    let (_b, submit) =
        lower_sampler_standard(SamplerSpec::MinP { temperature: 0.7, p: 0.05 }, QWEN3_VOCAB)
            .unwrap();
    assert_eq!(submit, vec![(0u32, f32le(0.7)), (1u32, f32le(0.05))]);

    let (_b, submit) =
        lower_sampler_standard(SamplerSpec::Multinomial { temperature: 0.9 }, QWEN3_VOCAB).unwrap();
    assert_eq!(submit, vec![(0u32, f32le(0.9))], "temperature submits T@key0 only");
}

#[test]
fn repoint_greedy_collapses_to_argmax_no_submit() {
    // T <= 0 → Argmax program (no params), regardless of the nominal sugar kind —
    // the `canonical_kind`/`infer_kind` greedy collapse, preserved by the repoint.
    let (argmax_bytes, _) = standard_program(StandardSampler::Argmax, QWEN3_VOCAB).unwrap();
    for spec in [
        SamplerSpec::TopP { temperature: 0.0, p: 0.9 },
        SamplerSpec::TopK { temperature: -1.0, k: 40 },
        SamplerSpec::Multinomial { temperature: 0.0 },
    ] {
        let (built, submit) = lower_sampler_standard(spec, QWEN3_VOCAB).unwrap();
        assert_eq!(built.canonical_kind, CanonicalKind::Argmax, "{spec:?} → Argmax");
        assert!(submit.is_empty(), "greedy submits no params for {spec:?}");
        assert_eq!(built.lower().bytecode, argmax_bytes);
    }
}

#[test]
fn repoint_k_bearing_bakes_k_and_submits_continuous_params() {
    // TopK: routes to build_standard(TopK{k}) (k baked into RankLe), submits T@key0.
    let (built, submit) =
        lower_sampler_standard(SamplerSpec::TopK { temperature: 0.8, k: 40 }, QWEN3_VOCAB).unwrap();
    assert_eq!(built.canonical_kind, CanonicalKind::TopK);
    let (std_bytes, _) = standard_program(StandardSampler::TopK { k: 40 }, QWEN3_VOCAB).unwrap();
    assert_eq!(built.lower().bytecode, std_bytes, "TopK repoint == build_standard(TopK{{40}})");
    assert_eq!(submit, vec![(0u32, f32le(0.8))]);

    // TopKTopP: k baked, submits T@key0 + p@key1.
    let (built, submit) = lower_sampler_standard(
        SamplerSpec::TopKTopP { temperature: 0.8, k: 40, p: 0.9 },
        QWEN3_VOCAB,
    )
    .unwrap();
    assert_eq!(built.canonical_kind, CanonicalKind::TopKTopP);
    let (std_bytes, _) =
        standard_program(StandardSampler::TopKTopP { k: 40 }, QWEN3_VOCAB).unwrap();
    assert_eq!(built.lower().bytecode, std_bytes);
    assert_eq!(submit, vec![(0u32, f32le(0.8)), (1u32, f32le(0.9))]);
}
