//! Entropy-based Dynamic Temperature (EDT) — Zhang et al.,
//! <https://arxiv.org/abs/2403.14541>, *EDT: Improving Large Language Models'
//! Generation by Entropy-based Dynamic Temperature Sampling*.
//!
//! The temperature is not a constant but a function of the step's own entropy:
//!
//! ```text
//! T = T0 · N^(θ / H)        with 0 < N < 1
//! ```
//!
//! Read the exponent, not the formula. `H` small (the model is confident) makes
//! `θ/H` large, and a base below one raised to a large power collapses toward
//! zero — so **a confident step gets a cold temperature and commits**. `H`
//! large (the model is confused) makes `θ/H` small, `N^small` approaches 1, and
//! the step is sampled at nearly the full `T0` — so **a confused step gets a
//! hot temperature and explores**. `T0` is therefore a genuine upper bound on
//! the temperature ever used, which is what makes the knob tunable in practice.
//!
//! The rationale from the paper: when the model cannot pick a good continuation
//! anyway, exploring costs little quality; when it can, sampling noise is pure
//! downside. Zhu et al., <https://arxiv.org/abs/2309.02772> (*Hot or Cold?
//! Adaptive Temperature Sampling for Code Generation*) reached the same
//! conclusion independently in the code-generation domain.
//!
//! ## Why this needs a device program
//!
//! The entropy has to be reduced over the full 151936-token logit row and then
//! fed *back into the same step's* temperature before sampling. On a black-box
//! server that is a host round-trip per token: read logits out, reduce, decide,
//! push back. Here it is four PTIR ops inside the forward pass, and the decode
//! loop keeps running ahead.
//!
//! Pie already had both halves of this. `entropycheck` measures entropy and
//! acts on nothing; `mirostat` drives a control word into the sampler but
//! derives it from surprise rather than entropy. This inferlet is the wire
//! between them.
//!
//! ## Numerical guards
//!
//! A one-hot distribution has `H = 0`, which would make `θ/H` infinite. Entropy
//! is floored at `1e-6` and the resulting temperature at `min_temperature`, so
//! the division into the logits stays finite. Both floors are inert for any
//! distribution a real model produces.
//!
//! ## Source
//!
//! Zhang et al., *EDT: Improving Large Language Models' Generation by
//! Entropy-based Dynamic Temperature Sampling* —
//! <https://arxiv.org/abs/2403.14541> (Eq. 7).
//!
//! Note that several secondary summaries state the rule as
//! `T = T₀ + θ·(H / log K)`, which *inverts* the behaviour. Equation 7 is
//! `T = T₀ · N^(θ/Entropy)` with `0 < N < 1`, which is what this implements.
//!
//! Faithfulness: **Exact**. See
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.

use inferlet::ptir::attention::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_t0")]
    t0: f32,
    #[serde(default = "default_theta")]
    theta: f32,
    #[serde(default = "default_n")]
    n: f32,
    #[serde(default = "default_min_temperature")]
    min_temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_seed")]
    seed: u32,
}

fn default_prompt() -> String {
    "Write a short paragraph about adaptive temperature.".into()
}

fn default_t0() -> f32 {
    1.0
}

fn default_theta() -> f32 {
    0.1
}

fn default_n() -> f32 {
    0.8
}

fn default_min_temperature() -> f32 {
    0.05
}

fn default_max_tokens() -> usize {
    32
}

fn default_seed() -> u32 {
    0x7ce1
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    t0: f32,
    theta: f32,
    n: f32,
    /// Mean entropy of the sampled distributions, in nats.
    mean_entropy: f32,
    /// Mean temperature actually used. Must sit at or below `t0`; equal to `t0`
    /// would mean the entropy signal is saturated and the knob is doing
    /// nothing.
    mean_temperature: f32,
    min_entropy: f32,
}

#[derive(Clone, Copy)]
struct Cfg {
    t0: f32,
    theta: f32,
    ln_n: f32,
    min_temperature: f32,
}

/// `T = T0 · N^(θ/H)`, evaluated as `T0 · exp((θ/H) · ln N)` because the op set
/// has `exp` but no general power.
///
/// Returns `(temperature, entropy)`, both scalars.
fn edt_temperature(logits: &Tensor, cfg: Cfg) -> (Tensor, Tensor) {
    let logprobs = log_softmax(logits);
    let probs = exp(&logprobs);
    let h = entropy_from_logprobs(&probs, &logprobs);

    // A one-hot row has H = 0; floor it so the reciprocal stays finite.
    let h_safe = max_elem(&h, 1e-6f32);
    let exponent = cfg.theta / &h_safe;
    let t = cfg.t0 * exp(&exponent * cfg.ln_n);
    (max_elem(&t, cfg.min_temperature), h)
}

/// One sampling step: derive the temperature from this row's entropy, then draw.
fn step(logits: Tensor, vocab: u32, cfg: Cfg, rng_state: &Tensor) -> (Tensor, Tensor, Tensor) {
    let (temperature, h) = edt_temperature(&logits, cfg);
    let scaled = &logits / broadcast(&temperature, [vocab]);
    (
        gumbel_max(scaled, rng_state),
        reshape(&h, [1]),
        reshape(&temperature, [1]),
    )
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.t0.is_finite() || input.t0 <= 0.0 {
        return Err("t0 must be finite and greater than 0".into());
    }
    if !input.theta.is_finite() || input.theta <= 0.0 {
        return Err("theta must be finite and greater than 0".into());
    }
    // n must sit strictly inside (0, 1): at n = 1 the temperature is constant,
    // and at n >= 1 the response inverts and confident steps would get hotter.
    if !input.n.is_finite() || input.n <= 0.0 || input.n >= 1.0 {
        return Err("n must be finite and in (0, 1)".into());
    }
    if !input.min_temperature.is_finite() || input.min_temperature <= 0.0 {
        return Err("min_temperature must be finite and greater than 0".into());
    }
    let max_tokens = input.max_tokens;
    let vocab = model::output_vocab_size();
    let cfg = Cfg {
        t0: input.t0,
        theta: input.theta,
        ln_n: input.n.ln(),
        min_temperature: input.min_temperature,
    };
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "entropy-adaptive-temperature",
            text: String::new(),
            count: 0,
            t0: cfg.t0,
            theta: cfg.theta,
            n: input.n,
            mean_entropy: 0.0,
            mean_temperature: 0.0,
            min_entropy: 0.0,
        });
    }

    let mut prompt = model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut s1: Vec<f32> = Vec::with_capacity(max_tokens);
    let mut s2: Vec<f32> = Vec::with_capacity(max_tokens);

    // ── PREFILL FIRE (N-wide): first sampled token comes off the prompt. ──
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from_iter(0..n).named("positions_p");
    let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
    let page_indptr_p = Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p = Channel::from_iter((0..n).map(|p| p / page_size)).named("w_slot_p");
    let w_off_p = Channel::from_iter((0..n).map(|p| p % page_size)).named("w_off_p");
    let kv_len_p = Channel::from([n]).named("kv_len_p");
    let rng_p = Channel::from([input.seed, 0]).named("rng_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
    let s1_out_p = Channel::new([1], dtype::f32).named("s1_out_p");
    let s2_out_p = Channel::new([1], dtype::f32).named("s2_out_p");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    fwd_p.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &kv_len_p,
            pages: &pages_p,
            page_indptr: &page_indptr_p,
            w_slot: &w_slot_p,
            w_off: &w_off_p,
            positions: &positions_p,
            mask: None,
        },
    )?;
    fwd_p.epilogue(move || {
        let r = rng_p.take();
        let logits = intrinsics::logits();
        let (token, a, b) = step(logits, vocab, cfg, &r);
        let r_next = &r + iota(2);
        tok_out_p.put(&token);
        s1_out_p.put(&a);
        s2_out_p.put(&b);
        rng_p.put(&r_next);
    });

    let pipe = Pipeline::new();
    fwd_p.submit(&pipe).context("prefill submit")?;

    let g0 = tok_out_p.take_host::<i32>().await?;
    let a0 = s1_out_p.take_host::<f32>().await?;
    let b0 = s2_out_p.take_host::<f32>().await?;
    generated.push(g0 as u32);
    s1.push(a0);
    s2.push(b0);

    // ── DECODE LOOP (1-wide, run-ahead). ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from([g0]).named("tok_in");
        let rng = Channel::from([input.seed ^ 0x5bd1, 0]).named("rng");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let s1_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("s1_out");
        let s2_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("s2_out");
        let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from([n]).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from([n / page_size]).named("w_slot");
        let w_off = Channel::from([n % page_size]).named("w_off");
        let kv_len = Channel::from([n + 1]).named("kv_len");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        fwd.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: (n / page_size)..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        )?;
        fwd.epilogue(move || {
            // Takes and compute first, puts last (value-id discipline).
            let length = kv_len.take();
            let r = rng.take();
            let logits = intrinsics::logits();
            let (token, a, b) = step(logits, vocab, cfg, &r);

            let r_next = &r + iota(2);
            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);

            tok_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(indptr(1, &page_count));
            tok_out.put(&token);
            s1_out.put(&a);
            s2_out.put(&b);
            rng.put(&r_next);
        });

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget as usize, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            let a = s1_out
                .take_host::<f32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            let b = s2_out
                .take_host::<f32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            generated.push(t as u32);
            s1.push(a);
            s2.push(b);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    let mean_s1 = s1.iter().sum::<f32>() / s1.len() as f32;
    let mean_s2 = s2.iter().sum::<f32>() / s2.len() as f32;
    let min_s1 = s1.iter().fold(f32::INFINITY, |a, &b| a.min(b)).max(0.0);
    if mean_s2 > cfg.t0 * 1.001 {
        return Err(format!(
            "mean temperature {mean_s2} exceeded the t0 bound {}",
            cfg.t0
        ));
    }

    Ok(Output {
        sampler: "entropy-adaptive-temperature",
        text: model::decode(&generated)?,
        count: generated.len(),
        t0: cfg.t0,
        theta: cfg.theta,
        n: input.n,
        mean_entropy: mean_s1,
        mean_temperature: mean_s2,
        min_entropy: min_s1,
    })
}
