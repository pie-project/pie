//! #19 mirostat re-tune — **host-eval** sweep (no GPU).
//!
//! hotel's GPU finding (`b8f197ae`/`dfa19f8b`): on the corrected 151936 vocab the
//! full-distribution **natural surprise ceiling is ≈ 2.2 nats**, so mirostat v2
//! (`μ ← μ − lr·(S − τ)`) can only converge for a target `τ` *below* that ceiling
//! (`τ=3.0` → μ runaway). But `τ=1.5` — though achievable on average — **collapses
//! ~70 % of fresh boots** into the repetition-attractor: μ drives the surprise gate
//! so low that the kept set shrinks to the argmax (greedy → repetition).
//!
//! This harness reproduces the *mechanism* on the CPU interpreter so we can bracket
//! a stable (τ, lr, μ0) without burning GPU. It drives the real `program::mirostat`
//! μ-loop through `eval` over a reproducible 151936-scale logit stream **calibrated
//! to the ~2.2-nat natural surprise** hotel measured, and reports, per config:
//!   - **convergence** — tail-mean S vs τ, and whether μ stayed bounded (no runaway);
//!   - **kept-set collapse** — the steady-state truncation-set size (the attractor
//!     proxy: size→1 ≡ greedy ≡ repetition-prone), and the fraction of trials that
//!     collapse;
//!   - **repetition lock** — under an optional autoregressive repeat-bias, the rate
//!     at which the token stream literally locks into a loop.
//!
//! It is a *model* (synthetic logits) — it brackets the params + proves the
//! mechanism; @delta's GPU convergence trace on real logits is ground truth.
//!
//! Run the full sweep:
//!   cargo test -p sampling-edsl --test mirostat_tune -- --ignored --nocapture

use pie_sampling_ir::eval::{InputBindings, Value as EvalValue, eval};
use pie_sampling_ir::{Binding, SamplingProgram, decode};
use sampling_edsl::builder::Built;
use sampling_edsl::program::{MirostatKeys, mirostat};

const VOCAB: u32 = 151936; // qwen3-0.6b output vocab (the corrected vocab)
const TARGET_NAT_SURPRISE: f32 = 2.2; // hotel's 4090-measured full-dist ceiling

// ── reproducible RNG (SplitMix64) ───────────────────────────────────────────
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn unit_f32(state: &mut u64) -> f32 {
    // (0,1) open interval, avoids ln(0) in Box-Muller.
    ((splitmix64(state) >> 40) as f32 + 0.5) / (1u64 << 24) as f32
}
fn gaussian(state: &mut u64) -> f32 {
    let u1 = unit_f32(state);
    let u2 = unit_f32(state);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

// ── synthetic LM logits: peaked Gaussian field, temperature-scaled ──────────
/// One step's logits: `sigma`-scaled standard-normal field (a smooth peaked
/// next-token distribution whose argmax/top-set shuffle every step). `boost`
/// optionally adds an autoregressive repeat-bias on `last_token`.
fn gen_logits(seed: u64, vocab: u32, sigma: f32, boost: Option<(u32, f32)>) -> Vec<f32> {
    let mut st = seed ^ 0xD1B5_4A32_D192_ED03;
    let mut v: Vec<f32> = (0..vocab).map(|_| sigma * gaussian(&mut st)).collect();
    if let Some((tok, b)) = boost {
        // Make the repeat token dominant by margin `b` over the current max — models
        // a real-LM that has become over-confident on the repeated token (p→1).
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        v[tok as usize] = m + b;
    }
    v
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = logits.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|&x| x / s).collect()
}
fn entropy_nats(probs: &[f32]) -> f32 {
    -probs.iter().filter(|&&p| p > 0.0).map(|&p| p * p.ln()).sum::<f32>()
}

/// Binary-search `sigma` so the mean full-distribution entropy ≈ `TARGET_NAT_SURPRISE`
/// (entropy decreases as sigma grows — more peaked). Averaged over a few draws.
fn calibrate_sigma(vocab: u32) -> f32 {
    let mean_entropy = |sigma: f32| -> f32 {
        let mut h = 0.0;
        let n = 8;
        for i in 0..n {
            let probs = softmax(&gen_logits(0xA11CE ^ i, vocab, sigma, None));
            h += entropy_nats(&probs);
        }
        h / n as f32
    };
    let (mut lo, mut hi) = (1.0f32, 60.0f32); // entropy(lo) high, entropy(hi) low
    for _ in 0..30 {
        let mid = 0.5 * (lo + hi);
        if mean_entropy(mid) > TARGET_NAT_SURPRISE {
            lo = mid; // too flat → raise sigma
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn prog(b: &Built) -> SamplingProgram {
    decode(&b.lower().bytecode).expect("decode")
}
fn bind(b: &Built, logits: &[f32], mu: f32, keys: &MirostatKeys) -> Vec<EvalValue> {
    b.bindings
        .iter()
        .map(|binding| match binding {
            Binding::Logits | Binding::MtpLogits => EvalValue::F32(logits.to_vec()),
            Binding::Tensor { key, .. } if *key == keys.mu => EvalValue::F32(vec![mu]),
            Binding::Tensor { key, .. } => panic!("unexpected tensor key {key}"),
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Config {
    tau: f32,
    lr: f32,
    mu0_factor: f32, // μ0 = factor·τ
}

struct TrialOut {
    tail_mean_s: f32,
    max_mu: f32,
    median_keep: f32,
    collapsed: bool,     // steady-state kept-set ≤ 1 (greedy collapse)
    repetition_lock: bool, // tail tokens locked to a single repeated id
}

/// Run one fresh-boot mirostat trial through the CPU interpreter.
#[allow(clippy::too_many_arguments)]
fn run_trial(
    p: &SamplingProgram,
    b: &Built,
    keys: &MirostatKeys,
    vocab: u32,
    sigma: f32,
    cfg: Config,
    n_steps: usize,
    trial_seed: u64,
    repeat_boost: Option<f32>,
) -> TrialOut {
    let mut mu = cfg.mu0_factor * cfg.tau;
    let mut last_token: Option<u32> = None;
    let mut surprises = Vec::with_capacity(n_steps);
    let mut keeps = Vec::with_capacity(n_steps);
    let mut mus = Vec::with_capacity(n_steps);
    let mut tokens = Vec::with_capacity(n_steps);

    for step in 0..n_steps {
        let boost = last_token.zip(repeat_boost);
        let logits = gen_logits(trial_seed.wrapping_add(step as u64 * 0x100), vocab, sigma, boost);
        let probs = softmax(&logits);
        let surpr: Vec<f32> = probs.iter().map(|&q| -q.ln()).collect();

        let rng_seed = (trial_seed ^ (step as u64).wrapping_mul(0x9E37_79B9)) as u32;
        let inputs = bind(b, &logits, mu, keys);
        let out = eval(p, &InputBindings::new(&inputs, rng_seed)).expect("eval");
        let token = match &out[0] {
            EvalValue::I32(x) => x[0] as u32,
            o => panic!("token: {o:?}"),
        };
        let s = match &out[1] {
            EvalValue::F32(x) => x[0],
            o => panic!("scalar: {o:?}"),
        };
        // kept-set size = tokens with surprise ≤ μ (what the gate admits).
        let keep = surpr.iter().filter(|&&x| x <= mu).count();

        surprises.push(s);
        keeps.push(keep as f32);
        tokens.push(token);
        // mirostat v2 update; μ stays non-negative (a negative gate is degenerate).
        mu = (mu - cfg.lr * (s - cfg.tau)).max(0.0);
        mus.push(mu);
        last_token = Some(token);
    }

    let tail = |v: &[f32]| -> Vec<f32> { v[v.len() / 2..].to_vec() };
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let median = |v: &[f32]| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    let tail_keep = tail(&keeps);
    let tail_tokens = &tokens[tokens.len() / 2..];
    let rep_lock = tail_tokens.windows(2).all(|w| w[0] == w[1]) && tail_tokens.len() >= 2;

    TrialOut {
        tail_mean_s: mean(&tail(&surprises)),
        max_mu: mus.iter().cloned().fold(0.0, f32::max),
        median_keep: median(&tail_keep),
        collapsed: median(&tail_keep) <= 1.0,
        repetition_lock: rep_lock,
    }
}

struct Agg {
    cfg: Config,
    mean_tail_s: f32,
    abs_err: f32,    // |mean_tail_s − τ|
    max_mu: f32,     // worst-trial μ (runaway detector)
    mean_keep: f32,  // mean steady-state kept-set size
    collapse_rate: f32,
    repeat_rate: f32,
}

#[allow(clippy::too_many_arguments)]
fn run_config(
    p: &SamplingProgram,
    b: &Built,
    keys: &MirostatKeys,
    vocab: u32,
    sigma: f32,
    cfg: Config,
    n_steps: usize,
    n_trials: usize,
    repeat_boost: Option<f32>,
) -> Agg {
    let mut s_sum = 0.0;
    let mut keep_sum = 0.0;
    let mut max_mu = 0.0f32;
    let mut collapses = 0;
    let mut reps = 0;
    for t in 0..n_trials {
        let seed = 0xBADC_0FFEE_u64.wrapping_mul(t as u64 + 1).wrapping_add(0x51ED);
        let o = run_trial(p, b, keys, vocab, sigma, cfg, n_steps, seed, repeat_boost);
        s_sum += o.tail_mean_s;
        keep_sum += o.median_keep;
        max_mu = max_mu.max(o.max_mu);
        if o.collapsed {
            collapses += 1;
        }
        if o.repetition_lock {
            reps += 1;
        }
    }
    let mean_tail_s = s_sum / n_trials as f32;
    Agg {
        cfg,
        mean_tail_s,
        abs_err: (mean_tail_s - cfg.tau).abs(),
        max_mu,
        mean_keep: keep_sum / n_trials as f32,
        collapse_rate: collapses as f32 / n_trials as f32,
        repeat_rate: reps as f32 / n_trials as f32,
    }
}

/// On-demand re-tune guard (run at the real 151936 vocab — the achievable surprise
/// ceiling is vocab-dependent, so this is too heavy for CI; `eval_parity.rs` guards
/// the mirostat program's correctness in CI). At the re-tuned config the host μ-loop
/// converges (μ bounded, tail-S near τ) and the kept set does NOT collapse to greedy.
#[test]
#[ignore = "heavy (151936-vocab μ-loop); run with --ignored --nocapture"]
fn mirostat_retuned_config_converges_without_collapse() {
    let vocab = VOCAB;
    let sigma = calibrate_sigma(vocab);
    let (b, keys) = mirostat(vocab).unwrap();
    let p = prog(&b);
    // Re-tune: τ=2.0 sits below the natural-surprise ceiling (converges) and gives
    // a kept-set with diversity headroom (far from the τ=1.5 greedy basin); the
    // mandatory μ0=2τ standard init avoids the low-init μ-underflow collapse.
    let cfg = Config { tau: 2.0, lr: 0.4, mu0_factor: 2.0 };
    let agg = run_config(&p, &b, &keys, vocab, sigma, cfg, 24, 8, None);
    eprintln!(
        "[guard] tau=2.0 lr=0.4 mu0=2.0τ → tail_S={:.3} |err|={:.3} max_mu={:.2} keep={:.1} collapse={:.0}%",
        agg.mean_tail_s, agg.abs_err, agg.max_mu, agg.mean_keep, agg.collapse_rate * 100.0
    );
    assert!(agg.max_mu < 8.0 * cfg.tau, "μ must stay bounded (no runaway)");
    assert!(agg.abs_err < 0.5, "tail-mean S must track τ");
    assert!(agg.mean_keep > 8.0, "kept set must hold diversity headroom (no collapse)");
    assert!(agg.collapse_rate < 0.1, "collapse rate must be low at the re-tuned config");
}

/// Full (τ, lr, μ0) sweep — the #19 diagnostic. Prints a table + the re-tune
/// recommendation, and contrasts the wrong-vocab τ=1.5 against it under an
/// autoregressive repeat-bias (the attractor).
#[test]
#[ignore = "diagnostic sweep; run with --ignored --nocapture"]
fn mirostat_retune_sweep_151936() {
    let sigma = calibrate_sigma(VOCAB);
    let probe_h = {
        let mut h = 0.0;
        for i in 0..8 {
            h += entropy_nats(&softmax(&gen_logits(7 ^ i, VOCAB, sigma, None)));
        }
        h / 8.0
    };
    let (b, keys) = mirostat(VOCAB).unwrap();
    let p = prog(&b);
    let (n_steps, n_trials) = (24, 16);

    eprintln!("\n=== #19 mirostat host-eval sweep (vocab={VOCAB}) ===");
    eprintln!("calibrated sigma={sigma:.3} → mean natural surprise ≈ {probe_h:.3} nats (target {TARGET_NAT_SURPRISE})");
    eprintln!("steps={n_steps} trials={n_trials}  [no repeat-bias: pure μ-dynamics]\n");
    eprintln!("  τ     lr   μ0       tail_S  |err|  max_μ   keep   collapse%");

    let taus = [1.5f32, 1.75, 2.0, 2.25];
    let lrs = [0.2f32, 0.4, 0.6];
    let mu0s = [1.0f32, 2.0];
    let mut best: Option<Agg> = None;
    for &tau in &taus {
        for &lr in &lrs {
            for &mu0_factor in &mu0s {
                let cfg = Config { tau, lr, mu0_factor };
                let a = run_config(&p, &b, &keys, VOCAB, sigma, cfg, n_steps, n_trials, None);
                eprintln!(
                    "  {:.2}  {:.1}  {:.1}τ    {:6.3}  {:5.3}  {:5.2}  {:5.1}   {:4.0}%",
                    tau, lr, mu0_factor, a.mean_tail_s, a.abs_err, a.max_mu, a.mean_keep,
                    a.collapse_rate * 100.0
                );
                // Re-tune target: converges (small err, μ bounded), collapse-free,
                // and a kept-set with diversity headroom (well clear of the greedy
                // basin) without degenerating into near-full sampling.
                let healthy = a.abs_err < 0.25
                    && a.max_mu < 6.0 * tau
                    && a.collapse_rate < 0.05
                    && (8.0..=50.0).contains(&a.mean_keep);
                // Prefer lowest |err|, tie-break toward more kept-set headroom.
                let score = |x: &Agg| x.abs_err - 0.002 * x.mean_keep;
                if healthy && best.as_ref().is_none_or(|bst| score(&a) < score(bst)) {
                    best = Some(a);
                }
            }
        }
    }

    eprintln!("\n[collapse modes] μ0-init sensitivity + a hard autoregressive repeat-lock");
    eprintln!("  (repeat-bias makes the last token dominant by margin Δ — an over-confident-LM repeat)");
    for (label, cfg, rb) in [
        ("τ=1.5 μ0=1.0τ low-init      ", Config { tau: 1.5, lr: 0.6, mu0_factor: 1.0 }, None),
        ("τ=1.5 μ0=2.0τ std-init      ", Config { tau: 1.5, lr: 0.6, mu0_factor: 2.0 }, None),
        ("τ=2.0 μ0=2.0τ re-tune       ", Config { tau: 2.0, lr: 0.4, mu0_factor: 2.0 }, None),
        ("τ=1.5 μ0=2.0τ + repeat Δ=4  ", Config { tau: 1.5, lr: 0.6, mu0_factor: 2.0 }, Some(4.0)),
        ("τ=2.0 μ0=2.0τ + repeat Δ=4  ", Config { tau: 2.0, lr: 0.4, mu0_factor: 2.0 }, Some(4.0)),
    ] {
        let a = run_config(&p, &b, &keys, VOCAB, sigma, cfg, n_steps, n_trials, rb);
        eprintln!(
            "  {label} → collapse={:3.0}%  repeat_lock={:3.0}%  keep={:6.1}  tail_S={:.3}  max_μ={:.2}",
            a.collapse_rate * 100.0, a.repeat_rate * 100.0, a.mean_keep, a.mean_tail_s, a.max_mu
        );
    }

    match best {
        Some(a) => eprintln!(
            "\n>>> RE-TUNE RECOMMENDATION (host-eval): τ={:.2}, lr={:.1}, μ0={:.1}τ  \
             (tail_S={:.3}, keep={:.1}, collapse={:.0}%)  → hand to @delta for the GPU trace.",
            a.cfg.tau, a.cfg.lr, a.cfg.mu0_factor, a.mean_tail_s, a.mean_keep,
            a.collapse_rate * 100.0
        ),
        None => eprintln!("\n>>> no config met the convergence+health bar — widen the grid."),
    }
}
