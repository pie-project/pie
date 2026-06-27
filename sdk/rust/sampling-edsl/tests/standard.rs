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
