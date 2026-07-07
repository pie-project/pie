//! Structural + round-trip tests for the Phase-2 programs (mirostat, grammar,
//! spec-verify greedy/lossless) and the WS5 `Sampler` sugar, on PSIR v4.
//! Each program builds, round-trips through the canonical encoder/decoder, and
//! passes the canonical validator.
#![allow(deprecated)] // exercises the deprecated WS5 sugar (`lower_sampler`) until removal.

use sampling_edsl::ir::{self, Op};
use sampling_edsl::program::{
    grammar, grammar_sampled, mirostat, spec_verify_greedy, spec_verify_lossless,
};
use sampling_edsl::sugar::{SamplerSpec, build_sampler, lower_sampler};
use sampling_edsl::{Built, OutputKind};

const VOCAB: u32 = 32_000;

fn ops(b: &Built) -> &[Op] {
    &b.program.ops
}
fn has(b: &Built, p: impl Fn(&Op) -> bool) -> bool {
    ops(b).iter().any(p)
}
fn count(b: &Built, p: impl Fn(&Op) -> bool) -> usize {
    ops(b).iter().filter(|o| p(o)).count()
}
fn roundtrip(b: &Built) {
    let lowered = b.lower();
    assert_eq!(&lowered.bytecode[..4], &ir::MAGIC);
    let decoded = ir::decode(&lowered.bytecode).expect("decode");
    assert_eq!(decoded, b.program, "round-trip mismatch");
    decoded.validate().expect("validate");
}

// ── mirostat ────────────────────────────────────────────────────────────────
#[test]
fn mirostat_outputs_token_and_scalar() {
    let (b, keys) = mirostat(VOCAB).expect("builds");
    roundtrip(&b);
    assert_eq!(b.outputs, vec![OutputKind::Token, OutputKind::Scalar]);
    // μ is the only host input, submit-bound.
    assert_eq!(b.host_inputs.len(), 1);
    assert_eq!(b.host_inputs[0].key, keys.mu);
    assert_eq!(b.host_inputs[0].ready, ir::Readiness::Submit);
    // softmax (Exp), surprise (Log+Neg), keep (Ge), gumbel (stream:0), argmax, gather S.
    assert!(has(&b, |o| matches!(o, Op::Exp(_))));
    assert!(has(&b, |o| matches!(o, Op::Neg(_))));
    assert!(has(&b, |o| matches!(o, Op::Ge(_, _))));
    assert!(has(&b, |o| matches!(o, Op::Rng { kind: ir::RngKind::Gumbel, stream: 0, .. })));
    assert!(has(&b, |o| matches!(o, Op::ReduceArgmax(_))));
    assert!(has(&b, |o| matches!(o, Op::Gather { .. })));
}

// ── grammar ───────────────────────────────────────────────────────────────--
#[test]
fn grammar_greedy_mask_apply() {
    let (b, keys) = grammar(VOCAB).expect("builds");
    roundtrip(&b);
    assert_eq!(b.outputs, vec![OutputKind::Token]);
    assert_eq!(b.host_inputs.len(), 1);
    let m = b.host_inputs[0];
    assert_eq!(m.key, keys.mask);
    // Packed allowed-token bitmask: [ceil(vocab/32)] u32, late-bound.
    assert_eq!(m.dtype, ir::DType::U32);
    assert_eq!(m.shape, ir::Shape::vector(VOCAB.div_ceil(32)));
    assert_eq!(m.ready, ir::Readiness::Late);
    // argmax(mask_apply(logits, mask)); greedy => no RNG.
    assert!(has(&b, |o| matches!(o, Op::MaskApply { .. })));
    assert!(has(&b, |o| matches!(o, Op::ReduceArgmax(_))));
    assert!(!has(&b, |o| matches!(o, Op::Rng { .. })));
}

#[test]
fn grammar_sampled_adds_noise() {
    let (b, _keys) = grammar_sampled(VOCAB).expect("builds");
    roundtrip(&b);
    assert_eq!(b.outputs, vec![OutputKind::Token]);
    assert!(has(&b, |o| matches!(o, Op::Rng { kind: ir::RngKind::Gumbel, .. })));
}

// ── spec-verify greedy ────────────────────────────────────────────────────--
#[test]
fn spec_verify_greedy_structure() {
    let k = 5u32;
    let (b, keys) = spec_verify_greedy(VOCAB, k).expect("builds");
    roundtrip(&b);
    assert_eq!(b.outputs, vec![OutputKind::Token]);
    // intrinsic logits = [k, vocab]; draft = [k] i32 submit-bound. Binding-free
    // v4: find the Logits-bound slot and read its decl shape from program.inputs.
    let logits_slot = b
        .bindings
        .iter()
        .position(|x| matches!(x, ir::Binding::Logits))
        .expect("logits input");
    assert_eq!(b.program.inputs[logits_slot].shape, ir::Shape::matrix(k, VOCAB));
    let d = b.host_inputs.iter().find(|h| h.key == keys.draft).unwrap();
    assert_eq!(d.dtype, ir::DType::I32);
    assert_eq!(d.shape, ir::Shape::vector(k));
    // argmax -> eq -> cumprod -> select.
    assert!(has(&b, |o| matches!(o, Op::ReduceArgmax(_))));
    assert!(has(&b, |o| matches!(o, Op::Eq(_, _))));
    assert!(has(&b, |o| matches!(o, Op::CumProd(_))));
    assert!(has(&b, |o| matches!(o, Op::Select { .. })));
}

// ── spec-verify lossless ──────────────────────────────────────────────────--
#[test]
fn spec_verify_lossless_structure() {
    let k = 4u32;
    let (b, keys) = spec_verify_lossless(VOCAB, k).expect("builds");
    roundtrip(&b);
    assert_eq!(b.outputs, vec![OutputKind::Token]);
    // q [k,vocab] + draft [k] i32, both submit-bound.
    let q = b.host_inputs.iter().find(|h| h.key == keys.q).unwrap();
    assert_eq!(q.shape, ir::Shape::matrix(k, VOCAB));
    let d = b.host_inputs.iter().find(|h| h.key == keys.draft).unwrap();
    assert_eq!(d.dtype, ir::DType::I32);
    // GatherRow for p_at + q_at; Div+MinElem (ratio); Uniform+Gumbel RNG; sentinel CumSum.
    assert_eq!(count(&b, |o| matches!(o, Op::GatherRow { .. })), 2);
    assert!(has(&b, |o| matches!(o, Op::Div(_, _))));
    assert!(has(&b, |o| matches!(o, Op::MinElem(_, _))));
    assert!(has(&b, |o| matches!(o, Op::Rng { kind: ir::RngKind::Uniform, stream: 0, .. })));
    assert!(has(&b, |o| matches!(o, Op::Rng { kind: ir::RngKind::Gumbel, stream: 1, .. })));
    assert!(has(&b, |o| matches!(o, Op::CumSum(_))));
}

// ── WS5 sugar ─────────────────────────────────────────────────────────────--
#[test]
fn sugar_argmax_deterministic_no_rng() {
    let b = build_sampler(SamplerSpec::Argmax, VOCAB).expect("argmax");
    roundtrip(&b);
    assert_eq!(b.outputs, vec![OutputKind::Token]);
    assert_eq!(ops(&b).len(), 2); // Input(logits) + ReduceArgmax
    assert!(!has(&b, |o| matches!(o, Op::Rng { .. })));
}

#[test]
fn sugar_temperature_zero_collapses_to_argmax() {
    for spec in [
        SamplerSpec::TopP { temperature: 0.0, p: 0.9 },
        SamplerSpec::TopK { temperature: 0.0, k: 40 },
        SamplerSpec::MinP { temperature: 0.0, p: 0.05 },
        SamplerSpec::Multinomial { temperature: 0.0 },
    ] {
        let b = build_sampler(spec, VOCAB).expect("builds");
        roundtrip(&b);
        assert!(!has(&b, |o| matches!(o, Op::Rng { .. })), "{spec:?} at T=0 is greedy");
    }
}

#[test]
fn sugar_stochastic_uses_stream0_rng() {
    for spec in [
        SamplerSpec::TopP { temperature: 0.8, p: 0.9 },
        SamplerSpec::TopK { temperature: 1.0, k: 50 },
        SamplerSpec::MinP { temperature: 0.7, p: 0.05 },
        SamplerSpec::TopKTopP { temperature: 0.9, k: 40, p: 0.95 },
        SamplerSpec::Multinomial { temperature: 1.0 },
    ] {
        let b = build_sampler(spec, VOCAB).expect("builds");
        roundtrip(&b);
        assert!(has(&b, |o| matches!(o, Op::Rng { kind: ir::RngKind::Gumbel, stream: 0, .. })));
    }
}

#[test]
fn sugar_min_p_is_logit_space() {
    let b = build_sampler(SamplerSpec::MinP { temperature: 0.7, p: 0.05 }, VOCAB).unwrap();
    roundtrip(&b);
    assert!(has(&b, |o| matches!(o, Op::ReduceMax(_))));
    assert!(has(&b, |o| matches!(o, Op::Log(_))));
    assert!(has(&b, |o| matches!(o, Op::Ge(_, _))));
}

#[test]
fn sugar_top_k_top_p_has_both_predicates() {
    let b = build_sampler(SamplerSpec::TopKTopP { temperature: 0.9, k: 40, p: 0.95 }, VOCAB).unwrap();
    roundtrip(&b);
    assert!(has(&b, |o| matches!(o, Op::PivotThreshold { predicate: ir::Predicate::RankLe(_), .. })));
    assert!(has(&b, |o| matches!(o, Op::PivotThreshold { predicate: ir::Predicate::CummassLe(_), .. })));
}

#[test]
fn lower_sampler_emits_v4_bytecode() {
    let lowered = lower_sampler(SamplerSpec::Argmax, VOCAB).unwrap();
    assert_eq!(&lowered.bytecode[..4], &ir::MAGIC);
    let version = u16::from_le_bytes([lowered.bytecode[4], lowered.bytecode[5]]);
    assert_eq!(version, ir::VERSION);
    assert_eq!(lowered.outputs, vec![OutputKind::Token]);
}
