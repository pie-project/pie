//! Structural + numeric tests for the production standard-sampler programs
//! (`standard.rs`, Task #4 (B) batched de-hardwiring). These are the per-sequence
//! vector forms with params as scalar HostSubmit inputs + ambient batch-axis RNG
//! (`Op::Rng{stream:0}`). The batched `[N,V]` replication is the driver's
//! `batched=true` lowering (hotel's removal-gate covers that on-GPU); here we test
//! the single-sequence semantics through the CPU interpreter.

use pie_sampling_ir::eval::{InputBindings, Value as EvalValue, eval};
use pie_sampling_ir::{Binding, Op, SamplingProgram, decode};

use sampling_edsl::OutputKind;
use sampling_edsl::builder::Built;
use sampling_edsl::standard::{
    StandardSampler, StdParamKeys, argmax, build_standard, min_p, temperature, top_k, top_k_top_p,
    top_p,
};

fn prog(b: &Built) -> SamplingProgram {
    decode(&b.lower().bytecode).expect("decode")
}

/// Positional eval inputs (one `Value` per program slot, in slot order) — resolve
/// each slot from `Built::bindings`: `Logits` ← logits, `Tensor{key}` ← params.
fn bind(b: &Built, logits: &[f32], params: &[(u32, f32)]) -> Vec<EvalValue> {
    b.bindings
        .iter()
        .map(|binding| match binding {
            Binding::Logits => EvalValue::F32(logits.to_vec()),
            Binding::Tensor { key, .. } => {
                let v = params
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| *v)
                    .unwrap_or_else(|| panic!("no value for param key {key}"));
                EvalValue::F32(vec![v])
            }
        })
        .collect()
}

fn token_of(v: &EvalValue) -> i32 {
    match v {
        EvalValue::I32(x) => x[0],
        o => panic!("expected I32 token, got {o:?}"),
    }
}

fn argmax_idx(xs: &[f32]) -> i32 {
    let mut best = 0usize;
    for i in 1..xs.len() {
        if xs[i] > xs[best] {
            best = i;
        }
    }
    best as i32
}

fn logits8() -> Vec<f32> {
    vec![0.5, 3.0, -1.0, 2.0, 0.0, 4.0, 1.5, -2.0]
}

// ── structure ───────────────────────────────────────────────────────────────

#[test]
fn argmax_structure() {
    let b = argmax(32).unwrap();
    assert_eq!(b.outputs, vec![OutputKind::Token]);
    assert!(b.host_inputs.is_empty(), "argmax declares no host params");
    // Only the intrinsic logits input; the sole op is a per-row argmax.
    assert!(b.bindings.iter().all(|x| matches!(x, Binding::Logits)));
    let p = prog(&b);
    p.validate().unwrap();
    assert!(p.ops.iter().any(|o| matches!(o, Op::ReduceArgmax(_))));
}

#[test]
fn temperature_structure() {
    let (b, keys) = temperature(64).unwrap();
    assert_eq!(b.outputs, vec![OutputKind::Token]);
    // One scalar f32 submit param (T); selection RNG is ambient stream:0.
    assert_eq!(b.host_inputs.len(), 1);
    let t = keys.temperature.expect("temperature key");
    assert!(b.host_inputs.iter().any(|h| h.key == t && h.shape.is_scalar()));
    assert_eq!(keys.min_p, None);
    let p = prog(&b);
    p.validate().unwrap();
    assert!(p.ops.iter().any(|o| matches!(o, Op::Rng { stream: 0, .. })));
}

#[test]
fn min_p_structure() {
    let (b, keys) = min_p(64).unwrap();
    assert_eq!(b.outputs, vec![OutputKind::Token]);
    // Two scalar submit params: T + min_p.
    assert_eq!(b.host_inputs.len(), 2);
    assert!(keys.temperature.is_some() && keys.min_p.is_some());
    let p = prog(&b);
    p.validate().unwrap();
    assert!(p.ops.iter().any(|o| matches!(o, Op::Rng { stream: 0, .. })));
}

#[test]
fn build_standard_dispatch() {
    assert_eq!(
        build_standard(StandardSampler::Argmax, 16).unwrap().1,
        StdParamKeys::default()
    );
    assert!(build_standard(StandardSampler::Temperature, 16).unwrap().1.temperature.is_some());
    let mp = build_standard(StandardSampler::MinP, 16).unwrap().1;
    assert!(mp.temperature.is_some() && mp.min_p.is_some());
}

// ── eval: single-sequence semantics ──────────────────────────────────────────

#[test]
fn eval_argmax_exact() {
    let b = argmax(8).unwrap();
    let logits = logits8();
    let inputs = bind(&b, &logits, &[]);
    let out = eval(&prog(&b), &InputBindings::new(&inputs, 1)).unwrap();
    assert_eq!(token_of(&out[0]), argmax_idx(&logits)); // idx 5 (4.0)
}

#[test]
fn eval_temperature_low_t_is_greedy() {
    // As T→0 the scaled logits dominate the Gumbel noise → argmax.
    let (b, keys) = temperature(8).unwrap();
    let logits = logits8();
    let want = argmax_idx(&logits);
    for s in [1u32, 7, 42, 1000, 65535] {
        let inputs = bind(&b, &logits, &[(keys.temperature.unwrap(), 0.02)]);
        let out = eval(&prog(&b), &InputBindings::new(&inputs, s)).unwrap();
        assert_eq!(token_of(&out[0]), want, "seed {s}");
    }
}

#[test]
fn eval_min_p_token_in_kept_set() {
    // The chosen token must clear the logit-space min-p threshold
    // `max_logit + ln(p)`, for any seed.
    let (b, keys) = min_p(8).unwrap();
    let logits = logits8();
    let p = 0.3f32;
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let thr = max_logit + p.ln();
    for s in [1u32, 7, 42, 1000, 65535] {
        let inputs = bind(
            &b,
            &logits,
            &[(keys.temperature.unwrap(), 0.8), (keys.min_p.unwrap(), p)],
        );
        let out = eval(&prog(&b), &InputBindings::new(&inputs, s)).unwrap();
        let t = token_of(&out[0]) as usize;
        assert!(logits[t] >= thr - 1e-5, "seed {s}: token {t} below min-p threshold");
    }
}

// ── top-k / top-p / top-k-top-p ──────────────────────────────────────────────

fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|&x| x / s).collect()
}

fn top_k_set(logits: &[f32], k: usize) -> std::collections::HashSet<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &c| logits[c].partial_cmp(&logits[a]).unwrap());
    idx[..k].iter().cloned().collect()
}

/// The inclusive top-p nucleus on `softmax(logits)`: highest-prob tokens until
/// the cumulative mass first reaches `p`.
fn nucleus(logits: &[f32], p: f32) -> std::collections::HashSet<usize> {
    let probs = softmax(logits);
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &c| probs[c].partial_cmp(&probs[a]).unwrap());
    let mut cum = 0.0;
    let mut set = std::collections::HashSet::new();
    for &i in &idx {
        set.insert(i);
        cum += probs[i];
        if cum >= p {
            break;
        }
    }
    set
}

#[test]
fn top_k_structure() {
    let (b, keys) = top_k(64, 5).unwrap();
    assert_eq!(b.outputs, vec![OutputKind::Token]);
    assert_eq!(b.host_inputs.len(), 1); // temperature only; k is a baked immediate
    assert!(keys.temperature.is_some() && keys.top_p.is_none());
    let p = prog(&b);
    p.validate().unwrap();
    assert!(p.ops.iter().any(|o| matches!(
        o,
        Op::PivotThreshold { predicate: pie_sampling_ir::Predicate::RankLe(5), .. }
    )));
    assert!(p.ops.iter().any(|o| matches!(o, Op::Rng { stream: 0, .. })));
}

#[test]
fn top_p_structure() {
    let (b, keys) = top_p(64).unwrap();
    assert_eq!(b.host_inputs.len(), 2); // temperature + p
    assert!(keys.temperature.is_some() && keys.top_p.is_some());
    let p = prog(&b);
    p.validate().unwrap();
    assert!(p.ops.iter().any(|o| matches!(
        o,
        Op::PivotThreshold { predicate: pie_sampling_ir::Predicate::CummassLe(_), .. }
    )));
}

#[test]
fn top_k_top_p_structure() {
    let (b, keys) = top_k_top_p(64, 8).unwrap();
    assert_eq!(b.host_inputs.len(), 2); // temperature + p; k is a baked immediate
    assert!(keys.temperature.is_some() && keys.top_p.is_some());
    let p = prog(&b);
    p.validate().unwrap();
    assert!(p.ops.iter().any(|o| matches!(
        o,
        Op::PivotThreshold { predicate: pie_sampling_ir::Predicate::RankLe(8), .. }
    )));
    assert!(p.ops.iter().any(|o| matches!(
        o,
        Op::PivotThreshold { predicate: pie_sampling_ir::Predicate::CummassLe(_), .. }
    )));
}

#[test]
fn eval_top_k_token_in_top_k() {
    let k = 3u32;
    let (b, keys) = top_k(8, k).unwrap();
    let logits = logits8();
    let topk = top_k_set(&logits, k as usize);
    for s in [3u32, 11, 256, 4096, 50000] {
        let inputs = bind(&b, &logits, &[(keys.temperature.unwrap(), 1.0)]);
        let out = eval(&prog(&b), &InputBindings::new(&inputs, s)).unwrap();
        let t = token_of(&out[0]) as usize;
        assert!(topk.contains(&t), "seed {s}: token {t} not in top-{k}");
    }
}

#[test]
fn eval_top_p_token_in_nucleus() {
    let p = 0.9f32;
    let (b, keys) = top_p(8).unwrap();
    let logits = logits8();
    let nuc = nucleus(&logits, p); // temp=1.0 => softmax over raw logits
    for s in [1u32, 7, 42, 1000, 65535] {
        let inputs = bind(
            &b,
            &logits,
            &[(keys.temperature.unwrap(), 1.0), (keys.top_p.unwrap(), p)],
        );
        let out = eval(&prog(&b), &InputBindings::new(&inputs, s)).unwrap();
        let t = token_of(&out[0]) as usize;
        assert!(nuc.contains(&t), "seed {s}: token {t} not in top-p nucleus {nuc:?}");
    }
}

// ── canonical-kind recognizer (#8) ───────────────────────────────────────────

use sampling_edsl::{CanonicalKind, SamplerSpec, build_sampler, canonical_kind, infer_kind};

/// `canonical_kind` derives from the frozen ladder: for every sampler spec it
/// equals `infer_kind(its params)` — the SDK-sugar↔recognizer drift-guard.
#[test]
fn canonical_kind_matches_ladder() {
    let cases = [
        (SamplerSpec::Argmax, CanonicalKind::Argmax),
        (SamplerSpec::Multinomial { temperature: 0.8 }, CanonicalKind::Temperature),
        (SamplerSpec::TopP { temperature: 0.8, p: 0.9 }, CanonicalKind::TopP),
        (SamplerSpec::TopK { temperature: 0.8, k: 40 }, CanonicalKind::TopK),
        (SamplerSpec::MinP { temperature: 0.8, p: 0.05 }, CanonicalKind::MinP),
        (SamplerSpec::TopKTopP { temperature: 0.8, k: 40, p: 0.9 }, CanonicalKind::TopKTopP),
    ];
    for (spec, want) in cases {
        assert_eq!(canonical_kind(spec), want, "{spec:?}");
    }
}

/// `T<=0 -> Argmax` is unconditional (any k/p) — the greedy neutrality fix.
#[test]
fn greedy_collapses_to_argmax_any_filter() {
    for spec in [
        SamplerSpec::Multinomial { temperature: 0.0 },
        SamplerSpec::TopK { temperature: 0.0, k: 50 },
        SamplerSpec::TopP { temperature: -1.0, p: 0.9 },
        SamplerSpec::TopKTopP { temperature: 0.0, k: 50, p: 0.9 },
        SamplerSpec::MinP { temperature: 0.0, p: 0.1 },
    ] {
        assert_eq!(canonical_kind(spec), CanonicalKind::Argmax, "{spec:?}");
    }
}

/// The combined `top-k && top-p` arm precedes the standalone arms (no filter
/// dropped); degenerate filters (`p>=1`, `k=0`) fall through correctly.
#[test]
fn ladder_precedence_and_degenerate_filters() {
    assert_eq!(infer_kind(0.8, 40, 0.9, 0.0), CanonicalKind::TopKTopP); // both
    assert_eq!(infer_kind(0.8, 40, 1.0, 0.0), CanonicalKind::TopK); // p>=1 not a filter
    assert_eq!(infer_kind(0.8, 0, 0.9, 0.0), CanonicalKind::TopP);
    assert_eq!(infer_kind(0.8, 0, 1.0, 0.0), CanonicalKind::Temperature); // no filters
    assert_eq!(infer_kind(0.8, 0, 1.0, 0.2), CanonicalKind::MinP);
}

/// The stamp lands on `Built`/`LoweredProgram`: sugar → its kind; `build_standard`
/// → its kind; a `Graph`-authored program → `Custom`.
#[test]
fn built_carries_canonical_kind() {
    // sugar
    let b = build_sampler(SamplerSpec::TopK { temperature: 0.8, k: 40 }, 64).unwrap();
    assert_eq!(b.canonical_kind, CanonicalKind::TopK);
    assert_eq!(b.lower().canonical_kind, CanonicalKind::TopK);
    // greedy sugar collapses to Argmax (matches the program it actually builds)
    let g = build_sampler(SamplerSpec::TopP { temperature: 0.0, p: 0.9 }, 64).unwrap();
    assert_eq!(g.canonical_kind, CanonicalKind::Argmax);
    // standard
    let (s, _) = build_standard(StandardSampler::MinP, 64).unwrap();
    assert_eq!(s.canonical_kind, CanonicalKind::MinP);
    // custom (Graph-authored): grammar program is not a recognized standard kind
    let (custom, _) = sampling_edsl::program::grammar(64).unwrap();
    assert_eq!(custom.canonical_kind, CanonicalKind::Custom);
}

/// The `top_p<=0` edge (#7 reconciliation): a degenerate nucleus mass is NOT a
/// top-p filter — the recognizer's `0<top_p<1` correctly excludes it (the legacy
/// bare `top_p<1` is looser; #7 aligns the agreement check to this side).
#[test]
fn top_p_le_zero_is_not_a_filter() {
    assert_eq!(infer_kind(0.8, 0, 0.0, 0.0), CanonicalKind::Temperature); // T>0, no real filter
    assert_eq!(infer_kind(0.8, 0, -0.5, 0.0), CanonicalKind::Temperature);
    assert_eq!(infer_kind(0.8, 40, 0.0, 0.0), CanonicalKind::TopK); // top_p=0 => TopK, not TopKTopP
    assert_eq!(infer_kind(0.8, 0, 0.0, 0.2), CanonicalKind::MinP); // falls through to min_p
    // T<=0 still wins over everything (greedy), even with a degenerate top_p.
    assert_eq!(infer_kind(0.0, 40, 0.0, 0.0), CanonicalKind::Argmax);
}

// ── #12 reference set + entropy probe (#15 foundations) ──────────────────────

#[test]
fn standard_programs_reference_set() {
    use sampling_edsl::{standard_program, standard_programs};
    let progs = standard_programs(128).unwrap();
    // The 4 k-invariant kinds, in order, each mapped to its CanonicalKind.
    let kinds: Vec<CanonicalKind> = progs.iter().map(|(_, k)| *k).collect();
    assert_eq!(
        kinds,
        vec![
            CanonicalKind::Argmax,
            CanonicalKind::Temperature,
            CanonicalKind::MinP,
            CanonicalKind::TopP
        ]
    );
    // All-distinct bytecode => distinct hashes for alpha's {hash->kind} table.
    let set: std::collections::HashSet<&Vec<u8>> = progs.iter().map(|(b, _)| b).collect();
    assert_eq!(set.len(), 4, "standard program bytecode must be all-distinct");
    // k-bearing: classifies TopK; bytecode varies by k (the RankLe immediate).
    let (b40, k40) = standard_program(StandardSampler::TopK { k: 40 }, 128).unwrap();
    let (b50, k50) = standard_program(StandardSampler::TopK { k: 50 }, 128).unwrap();
    assert_eq!(k40, CanonicalKind::TopK);
    assert_eq!(k50, CanonicalKind::TopK);
    assert_ne!(b40, b50, "k-bearing bytecode varies by k");
}

#[test]
fn entropy_probe_structure_and_eval() {
    use sampling_edsl::program::entropy;
    let b = entropy(8).unwrap();
    // Scalar-output measurement, classified Custom (not a token-sampler kind).
    assert_eq!(b.outputs, vec![OutputKind::Scalar]);
    assert_eq!(b.canonical_kind, CanonicalKind::Custom);
    assert!(b.host_inputs.is_empty());

    let logits = logits8();
    let inputs = bind(&b, &logits, &[]);
    let out = eval(&prog(&b), &InputBindings::new(&inputs, 1)).unwrap();
    let h = match &out[0] {
        EvalValue::F32(v) => v[0],
        o => panic!("expected F32 scalar, got {o:?}"),
    };
    // Reference H = -Σ p·ln p.
    let probs = softmax(&logits);
    let expected: f32 = -probs.iter().map(|&p| p * p.ln()).sum::<f32>();
    assert!((h - expected).abs() < 1e-4, "entropy {h} != reference {expected}");
}
