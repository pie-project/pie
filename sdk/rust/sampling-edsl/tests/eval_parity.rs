//! Numeric parity for the Phase-2 programs via the canonical `pie-sampling-ir`
//! CPU interpreter (`eval` feature, PSIR v4). RNG is Model B: the ambient seed
//! `S` is the 3rd arg to `eval`; multiple draws decorrelate via `stream` ids.

use pie_sampling_ir::eval::{InputBindings, Value as EvalValue, eval};
use pie_sampling_ir::{SamplingProgram, decode};

use sampling_edsl::builder::Built;
use sampling_edsl::program::{grammar, mirostat, spec_verify_lossless};
use sampling_edsl::sugar::{SamplerSpec, build_sampler};

fn prog(b: &Built) -> SamplingProgram {
    decode(&b.lower().bytecode).expect("decode")
}
fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|&x| x / s).collect()
}
fn argmax(xs: &[f32]) -> i32 {
    let mut best = 0usize;
    for i in 1..xs.len() {
        if xs[i] > xs[best] {
            best = i;
        }
    }
    best as i32
}
fn token_of(v: &EvalValue) -> i32 {
    match v {
        EvalValue::I32(x) => x[0],
        o => panic!("expected I32 token, got {o:?}"),
    }
}
fn scalar_of(v: &EvalValue) -> f32 {
    match v {
        EvalValue::F32(x) => x[0],
        o => panic!("expected F32 scalar, got {o:?}"),
    }
}
fn logits8() -> Vec<f32> {
    vec![0.5, 3.0, -1.0, 2.0, 0.0, 4.0, 1.5, -2.0]
}

/// Build the positional `eval` input list (one `Value` per program input slot,
/// in slot order) from the logits + the declared tensor keys. Binding-free v4
/// resolves each slot's binding from [`Built::bindings`].
fn bind(b: &Built, logits: &[f32], tensors: &[(u32, EvalValue)]) -> Vec<EvalValue> {
    b.bindings
        .iter()
        .map(|binding| match binding {
            pie_sampling_ir::Binding::Logits => EvalValue::F32(logits.to_vec()),
            pie_sampling_ir::Binding::Tensor { key, .. } => tensors
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| panic!("no binding supplied for tensor key {key}")),
        })
        .collect()
}

// ── sugar argmax: exact ───────────────────────────────────────────────────--
#[test]
fn eval_sugar_argmax_exact() {
    let b = build_sampler(SamplerSpec::Argmax, 8).unwrap();
    let logits = logits8();
    let out = eval(&prog(&b), &InputBindings::new(&bind(&b, &logits, &[]), 1)).unwrap();
    assert_eq!(token_of(&out[0]), argmax(&logits));
}

// ── greedy grammar: exact argmax(logits + mask) ──────────────────────────────
#[test]
fn eval_grammar_greedy_exact() {
    let (b, keys) = grammar(8).unwrap();
    let logits = logits8();
    let mut mask = vec![0.0f32; 8];
    mask[5] = f32::NEG_INFINITY; // ban the natural argmax (idx 5)
    mask[1] = f32::NEG_INFINITY; // and idx 1 -> best allowed = idx 3 (2.0)
    let tensors = [(keys.mask, EvalValue::F32(mask.clone()))];
    let out = eval(&prog(&b), &InputBindings::new(&bind(&b, &logits, &tensors), 7)).unwrap();
    let biased: Vec<f32> = logits.iter().zip(&mask).map(|(l, m)| l + m).collect();
    assert_eq!(token_of(&out[0]), argmax(&biased));
    assert_eq!(token_of(&out[0]), 3);
}

// ── mirostat: S == -ln p(token) for whatever token RNG picked ────────────────
#[test]
fn eval_mirostat_reports_surprise_of_chosen_token() {
    let (b, keys) = mirostat(8).unwrap();
    let logits = logits8();
    let probs = softmax(&logits);
    // μ large => all tokens kept; any draw admissible.
    let tensors = [(keys.mu, EvalValue::F32(vec![100.0]))];
    let out = eval(&prog(&b), &InputBindings::new(&bind(&b, &logits, &tensors), 12345)).unwrap();
    let token = token_of(&out[0]);
    assert!((0..8).contains(&token));
    let s = scalar_of(&out[1]);
    let expected = -(probs[token as usize].ln());
    assert!((s - expected).abs() < 1e-4, "S={s} != -ln p(token)={expected}");
}

// ── mirostat truncation: tiny μ forces the argmax token ─────────────────────-
#[test]
fn eval_mirostat_small_mu_forces_argmax() {
    let (b, keys) = mirostat(8).unwrap();
    let logits = logits8();
    let probs = softmax(&logits);
    let min_surprise = -(probs.iter().cloned().fold(0.0f32, f32::max).ln());
    let tensors = [(keys.mu, EvalValue::F32(vec![min_surprise + 1e-3]))];
    for s in [1u32, 7, 42, 1000] {
        let out = eval(&prog(&b), &InputBindings::new(&bind(&b, &logits, &tensors), s)).unwrap();
        assert_eq!(token_of(&out[0]), argmax(&logits), "seed {s}");
    }
}

// ── sugar min-p: chosen token clears the logit-space threshold ──────────────-
#[test]
fn eval_sugar_min_p_token_in_kept_set() {
    let p = 0.3f32;
    let b = build_sampler(SamplerSpec::MinP { temperature: 0.8, p }, 8).unwrap();
    let logits = logits8();
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let thr = max_logit + p.ln();
    for s in [1u32, 7, 42, 1000, 65535] {
        let out = eval(&prog(&b), &InputBindings::new(&bind(&b, &logits, &[]), s)).unwrap();
        let t = token_of(&out[0]) as usize;
        assert!(logits[t] >= thr - 1e-5, "seed {s}: token {t} below min-p threshold");
    }
}

// ── sugar top-k: chosen token is among the top-k by logit ───────────────────-
#[test]
fn eval_sugar_top_k_token_in_top_k() {
    let k = 3u32;
    let b = build_sampler(SamplerSpec::TopK { temperature: 1.0, k }, 8).unwrap();
    let logits = logits8();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &c| logits[c].partial_cmp(&logits[a]).unwrap());
    let topk: std::collections::HashSet<usize> = idx[..k as usize].iter().cloned().collect();
    for s in [3u32, 11, 256, 4096, 50000] {
        let out = eval(&prog(&b), &InputBindings::new(&bind(&b, &logits, &[]), s)).unwrap();
        let t = token_of(&out[0]) as usize;
        assert!(topk.contains(&t), "seed {s}: token {t} not in top-{k}");
    }
}

// ── lossless: forced-accept + reject->resample semantics ────────────────────-
#[test]
fn eval_lossless_accept_and_resample() {
    let (b, keys) = spec_verify_lossless(4, 2).unwrap();
    let program = prog(&b);
    // Row 0: target favors token 0, draft token 0 (p≫q) => always accepted.
    // Row 1: target favors token 2, draft token 3 (p(3)≈0) => reject -> residual
    //         support {2} => resample token 2.
    let logits = vec![
        10.0, 0.0, 0.0, 0.0, // row 0
        0.0, 0.0, 10.0, 0.0, // row 1
    ];
    let q = vec![
        0.5, 0.5, 0.0, 0.0, // row 0
        0.0, 0.0, 0.0, 1.0, // row 1
    ];
    let draft = vec![0i32, 3];
    for s in [1u32, 7, 99, 2024, 65000] {
        let tensors = [
            (keys.q, EvalValue::F32(q.clone())),
            (keys.draft, EvalValue::I32(draft.clone())),
        ];
        let out = eval(&program, &InputBindings::new(&bind(&b, &logits, &tensors), s)).unwrap();
        let toks = match &out[0] {
            EvalValue::I32(v) => v.clone(),
            o => panic!("expected I32 vector, got {o:?}"),
        };
        assert_eq!(toks[0], 0, "seed {s}: row 0 accepts draft 0");
        assert_eq!(toks[1], 2, "seed {s}: row 1 resamples token 2");
    }
}

// ── lossless: DISTRIBUTION preservation (the actual "lossless" guarantee) ────-
// P(out=t) = min(q,p) + max(0,p-q) = p(t).  Drive the program through eval over
// many independent (draft~q, ambient-seed S) trials; the output histogram must
// match the TARGET p (not the draft q). k=1 always emits a token.
#[test]
fn eval_lossless_preserves_target_distribution() {
    const V: usize = 5;
    let (b, keys) = spec_verify_lossless(V as u32, 1).unwrap();
    let program = prog(&b);

    let logits = vec![2.0f32, 0.5, -1.0, 1.0, 0.0];
    let p = softmax(&logits);
    let q = vec![0.10f32, 0.40, 0.20, 0.05, 0.25];

    let sample_q = |u: f32| -> i32 {
        let mut acc = 0.0;
        for (t, &qt) in q.iter().enumerate() {
            acc += qt;
            if u < acc {
                return t as i32;
            }
        }
        (V - 1) as i32
    };

    // Deterministic test-harness uniform stream (splitmix64).
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut next_u = || -> f32 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 + 0.5) * (1.0 / 16_777_216.0)
    };

    const N: usize = 40_000;
    let mut hist = [0u32; V];
    for _ in 0..N {
        let draft = sample_q(next_u()); // x ~ q (the lossless precondition)
        let s = (next_u() * 4.0e9) as u32; // ambient seed varies per trial
        let tensors = [
            (keys.q, EvalValue::F32(q.clone())),
            (keys.draft, EvalValue::I32(vec![draft])),
        ];
        let out = eval(&program, &InputBindings::new(&bind(&b, &logits, &tensors), s)).unwrap();
        let t = match &out[0] {
            EvalValue::I32(v) => v[0],
            o => panic!("expected I32, got {o:?}"),
        };
        assert!((0..V as i32).contains(&t));
        hist[t as usize] += 1;
    }

    for t in 0..V {
        let freq = hist[t] as f32 / N as f32;
        assert!(
            (freq - p[t]).abs() < 0.02,
            "token {t}: empirical {freq:.4} != target p {:.4} (q was {:.4})",
            p[t],
            q[t]
        );
    }
}
