//! Aaronson's Gumbel watermark — a *distortion-free* watermark, meaning the
//! watermarked text is drawn from exactly the model's own distribution.
//!
//! The greenlist scheme already in this directory
//! (`watermarked-sampling`, Kirchenbauer et al. 2301.10226) works by adding a
//! bias `delta` to half the vocabulary. That is detectable, but it provably
//! *changes what the model says*. The Gumbel scheme changes nothing:
//!
//! ```text
//! seed_t   = hash(secret, x_{t-h..t-1})       key from context, not a counter
//! r_{t,v}  ~ Uniform[0,1)                      keyed, reproducible by a detector
//! x_t      = argmax_v  r_{t,v}^(1 / p_{t,v})   Aaronson's rule
//! ```
//!
//! and that rule is the Gumbel-max trick in disguise. With `E_v = -log r_v`,
//! maximizing `r_v^(1/p_v)` means minimizing `E_v / p_v`, the exponential race
//! whose winner is distributed exactly as `p`. Equivalently `argmax_v (logit_v
//! + G_v)` for `G_v = -log(-log r_v)`. So for **any** fixed key the marginal
//! distribution of `x_t` is the model's own — the watermark is carried in the
//! *correlation* between the text and the key, not in a shifted distribution.
//!
//! ## Why Pie makes this nearly free
//!
//! Pie's sampler is already keyed Gumbel-max over a `[key, ctr]` rng state, so
//! the noise a detector must reproduce is already a pure function of that
//! state. Watermarking reduces to *choosing what goes in the state*: replace
//! the free-running counter with a hash of the preceding tokens.
//!
//! ```text
//! ctr_t = (((0 * 31 + x_{t-h}) * 31 + ...) * 31 + x_{t-1}) mod 2000003
//! state = [secret, ctr_t]
//! ```
//!
//! The multiplier and modulus are kept small deliberately: `h * 31 + token`
//! stays under 2^26 so the u32 arithmetic cannot overflow, and the driver's
//! counter-based RNG does the actual mixing. A weak input hash is fine when it
//! only has to be *distinct per context*.
//!
//! ## Detection
//!
//! A detector who knows the secret recomputes `r_{t,v}` and scores the observed
//! token with Aaronson's statistic:
//!
//! ```text
//! s_t = -log(1 - r_{t,x_t})
//! ```
//!
//! Under H0 — the text was not produced with this key — `r_{t,x_t}` is uniform
//! and `s_t ~ Exp(1)`, so `mean(s) = 1`. Under H1 the sampler *chose* `x_t`
//! partly because `r_{t,x_t}` was large, so `mean(s) > 1`.
//!
//! This inferlet scores both hypotheses in the same pass: `mean_score` uses the
//! real secret, `mean_null_score` re-scores the very same tokens under a
//! decoy secret. The gap between them is the watermark. Because the noise is
//! recovered with `gumbel(state, ..)` and `scalar_gather`, both statistics are
//! computed on the device, alongside the sampling they describe.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Keeps `h * 31 + token` far below `u32::MAX` so the device arithmetic is
/// exact, while still spreading contexts over two million counters.
const HASH_MODULUS: u32 = 2000003;
const HASH_MULTIPLIER: u32 = 31;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    /// Watermark secret. The detector needs this and nothing else.
    #[serde(default = "default_secret")]
    secret: u32,
    /// How many preceding tokens seed each step, `h` in the papers.
    #[serde(default = "default_context_width")]
    context_width: u32,
    /// When false the rng state free-runs, producing unwatermarked text that
    /// the same detector should score at the null mean.
    #[serde(default = "default_watermark")]
    watermark: bool,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_seed")]
    seed: u32,
}

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    secret: u32,
    context_width: u32,
    watermark: bool,
    /// Mean of `-log(1 - r)` under the real secret, over distinct contexts.
    /// `1.0` under H0, larger when the text really was sampled with this key.
    mean_score: f32,
    /// The same statistic under a decoy secret — the empirical null.
    mean_null_score: f32,
    /// `(mean_score - 1) * sqrt(n)`. `s_t ~ Exp(1)` has unit variance under
    /// H0, so this is a standard-normal z-score.
    z_score: f32,
    /// Steps that contributed to the statistic. A context-keyed watermark
    /// reuses its key whenever the context repeats, so repeats carry no new
    /// evidence and are dropped — the same de-duplication every published
    /// detector performs.
    unique_contexts: usize,
}

fn default_prompt() -> String {
    "List three interesting facts about the ocean.".into()
}

fn default_secret() -> u32 {
    0xA5A5_1234
}

fn default_context_width() -> u32 {
    4
}

fn default_watermark() -> bool {
    true
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> usize {
    32
}

fn default_seed() -> u32 {
    0x2f19
}

#[derive(Clone, Copy)]
struct Cfg {
    secret: u32,
    decoy: u32,
    context_width: u32,
    watermark: bool,
    temperature: f32,
    capacity: u32,
}

/// Rolling hash of the `context_width` tokens ending at `hlen - 1`.
///
/// Returns a `[1]` u32 counter. Positions before the start of the buffer are
/// clamped and contribute the sentinel, which is harmless: every step at the
/// same offset sees the same clamped prefix, so the counter stays a pure
/// function of the context.
fn context_counter(hist: &Tensor, hlen: &Tensor, cfg: Cfg) -> Tensor {
    let last = sub(cast(hlen, DType::I32), Tensor::constant(1i32));
    let mut h = broadcast(Tensor::constant(0u32), [1]);
    // Oldest token first, so the hash is the usual left-to-right polynomial.
    for d in (0..cfg.context_width).rev() {
        let idx = max_elem(
            sub(&last, Tensor::constant(d as i32)),
            Tensor::constant(0i32),
        );
        // `+1` lifts the `-1` padding sentinel to 0 and keeps every real token
        // distinguishable from it.
        let tok = cast(
            add(gather(hist, &idx), Tensor::constant(1i32)),
            DType::U32,
        );
        h = rem(
            add(mul(&h, Tensor::constant(HASH_MULTIPLIER)), tok),
            Tensor::constant(HASH_MODULUS),
        );
    }
    h
}

/// Assembles the `[2]` u32 `[key, ctr]` rng state from two `[1]` parts.
fn rng_state(key: u32, counter: &Tensor) -> Tensor {
    let is_counter = eq(iota(2), broadcast(Tensor::constant(1u32), [2]));
    select(
        &is_counter,
        broadcast(counter, [2]),
        broadcast(Tensor::constant(key), [2]),
    )
}

/// The host mirror of `context_counter`, used to find which steps saw a context
/// the detector has already scored.
///
/// It has to reproduce the device hash exactly, including the clamp at the
/// start of the buffer and the `+1` applied to every token.
fn host_counter(history: &[i32], hlen: usize, context_width: u32) -> u32 {
    let last = hlen as i64 - 1;
    let mut h: u32 = 0;
    for d in (0..context_width as i64).rev() {
        let idx = (last - d).max(0) as usize;
        let tok = (history[idx] + 1) as u32;
        h = (h * HASH_MULTIPLIER + tok) % HASH_MODULUS;
    }
    h
}

/// Aaronson's per-token score `-log(1 - r)` recovered from the Gumbel variate.
///
/// `G = -log(-log r)` inverts to `r = exp(-exp(-G))`. The clamp is what keeps
/// the statistic finite: a token the noise favoured strongly has `r` within a
/// float epsilon of 1, and `-log(1 - r)` would otherwise be infinite.
fn aaronson_score(noise: &Tensor, token: &Tensor) -> Tensor {
    let g = scalar_gather(noise, token);
    let r = exp(neg(exp(neg(&g))));
    let tail = max_elem(sub(Tensor::constant(1.0f32), &r), Tensor::constant(1e-7f32));
    reshape(neg(log(&tail)), [1])
}

/// One sampling step.
///
/// Returns `(token, hist_next, score, null_score)`.
fn step(
    logits: Tensor,
    vocab: u32,
    cfg: Cfg,
    hist: &Tensor,
    hlen: &Tensor,
    free_state: impl AsTensor + Copy,
) -> (Tensor, Tensor, Tensor, Tensor) {
    let scaled = if cfg.temperature == 1.0 {
        logits
    } else {
        div(&logits, cfg.temperature)
    };

    let counter = context_counter(hist, hlen, cfg);
    let keyed = rng_state(cfg.secret, &counter);
    // The decoy shares the context counter and differs only in the secret, so
    // the null is the *same* experiment with the wrong key — exactly the
    // question a detector asks.
    let decoy = rng_state(cfg.decoy, &counter);

    // Sampling is spelled out rather than delegated to `gumbel_max` because the
    // detector statistic needs the noise tensor itself, not just the argmax.
    let noise = gumbel(&keyed, [vocab]);
    let token = if cfg.watermark {
        reduce_argmax(add(&scaled, &noise))
    } else {
        gumbel_max(&scaled, free_state)
    };

    let score = aaronson_score(&noise, &token);
    let null_score = aaronson_score(&gumbel(&decoy, [vocab]), &token);
    let hist_next = scatter_set(hist, hlen, &token);
    (token, hist_next, score, null_score)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if input.context_width == 0 || input.context_width > 8 {
        return Err("context_width must satisfy 1 <= context_width <= 8".into());
    }
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }

    let max_tokens = input.max_tokens;
    let vocab = wit_model::output_vocab_size();
    let ws = WorkingSet::new();
    let page_size = ws.page_size();

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "gumbel-watermark",
            text: String::new(),
            count: 0,
            secret: input.secret,
            context_width: input.context_width,
            watermark: input.watermark,
            mean_score: 0.0,
            mean_null_score: 0.0,
            z_score: 0.0,
            unique_contexts: 0,
        });
    }

    let mut prompt = wit_model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages)
        .map_err(|e| format!("reserve KV: {e}"))?;

    let cfg = Cfg {
        secret: input.secret,
        // A decoy that shares no low bits with the secret, so the two keyed
        // noise fields are unrelated.
        decoy: input.secret.wrapping_add(0x9E37_79B9),
        context_width: input.context_width,
        watermark: input.watermark,
        temperature: input.temperature,
        capacity: n + max_tokens as u32,
    };

    let mut history: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    history.resize(cfg.capacity as usize, -1);

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut scores: Vec<f32> = Vec::with_capacity(max_tokens);
    let mut nulls: Vec<f32> = Vec::with_capacity(max_tokens);

    // ── PREFILL FIRE (N-wide): first sampled token comes off the prompt. ──
    let toks_p =
        Channel::from(history.iter().take(n as usize).copied().collect::<Vec<_>>()).named("toks_p");
    let embed_indptr_p = Channel::from(vec![0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from((0..n).collect::<Vec<_>>()).named("positions_p");
    let pages_p = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages_p");
    let page_indptr_p = Channel::from(vec![0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p =
        Channel::from((0..n).map(|p| p / page_size).collect::<Vec<_>>()).named("w_slot_p");
    let w_off_p = Channel::from((0..n).map(|p| p % page_size).collect::<Vec<_>>()).named("w_off_p");
    let kv_len_p = Channel::from(vec![n]).named("kv_len_p");
    let rng_p = Channel::from(vec![input.seed, 0]).named("rng_p");
    let hist_p = Channel::from(history.clone()).named("hist_p");
    let hlen_p = Channel::from(vec![n]).named("hlen_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
    let score_out_p = Channel::new([1], dtype::f32).named("score_out_p");
    let null_out_p = Channel::new([1], dtype::f32).named("null_out_p");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    fwd_p.attention(
        &ws,
        ..,
        ..,
        &kv_len_p,
        &pages_p,
        &page_indptr_p,
        &w_slot_p,
        &w_off_p,
        &positions_p,
        None,
    )?;
    fwd_p.epilogue(move || {
        let r = rng_p.take();
        let hist = hist_p.take().tensor();
        let hlen = hlen_p.take().tensor();
        let logits = intrinsics::logits();
        let (token, hist_next, score, null) = step(logits, vocab, cfg, &hist, &hlen, &r);
        let r_next = add(&r, iota(2));
        tok_out_p.put(&token);
        score_out_p.put(&score);
        null_out_p.put(&null);
        hist_p.put(&hist_next);
        hlen_p.put(add(&hlen, 1u32));
        rng_p.put(&r_next);
    });

    let pipe = Pipeline::new();
    fwd_p
        .submit(&pipe)
        .map_err(|e| format!("prefill submit: {e}"))?;

    let g0 = tok_out_p
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("g0 take: {e}"))?[0];
    let s0 = score_out_p
        .take()
        .get::<f32>()
        .await
        .map_err(|e| format!("score take: {e}"))?[0];
    let n0 = null_out_p
        .take()
        .get::<f32>()
        .await
        .map_err(|e| format!("null take: {e}"))?[0];
    generated.push(g0 as u32);
    scores.push(s0);
    nulls.push(n0);

    // ── DECODE LOOP (1-wide, run-ahead). ──
    if generated.len() < max_tokens {
        history[n as usize] = g0;

        let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
        let rng = Channel::from(vec![input.seed ^ 0x5bd1, 0]).named("rng");
        let hist_c = Channel::from(history.clone()).named("hist");
        let hlen_c = Channel::from(vec![n + 1]).named("hlen");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
            .named("tok_out");
        let score_out = Channel::new([1], dtype::f32)
            .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
            .named("score_out");
        let null_out = Channel::new([1], dtype::f32)
            .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
            .named("null_out");
        let lane1 = Channel::from(vec![0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from(vec![n]).named("positions");
        let pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages");
        let page_indptr =
            Channel::from(vec![0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from(vec![n / page_size]).named("w_slot");
        let w_off = Channel::from(vec![n % page_size]).named("w_off");
        let kv_len = Channel::from(vec![n + 1]).named("kv_len");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        fwd.attention(
            &ws,
            ..,
            (n / page_size)..,
            &kv_len,
            &pages,
            &page_indptr,
            &w_slot,
            &w_off,
            &positions,
            None,
        )?;
        fwd.epilogue(move || {
            // Takes and compute first, puts last (value-id discipline).
            let length = kv_len.take().tensor();
            let r = rng.take();
            let hist = hist_c.take().tensor();
            let hlen = hlen_c.take().tensor();
            let logits = intrinsics::logits();
            let (token, hist_next, score, null) = step(logits, vocab, cfg, &hist, &hlen, &r);

            let r_next = add(&r, iota(2));
            let next_length = add(&length, 1u32);
            let page_count = div(add(&next_length, page_size - 1), page_size);

            tok_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(div(&length, page_size));
            w_off.put(rem(&length, page_size));
            page_indptr.take();
            page_indptr.put(mul(iota(2), broadcast(&page_count, [2])));
            tok_out.put(&token);
            score_out.put(&score);
            null_out.put(&null);
            hist_c.put(&hist_next);
            hlen_c.put(add(&hlen, 1u32));
            rng.put(&r_next);
        });

        let budget = max_tokens - 1;
        let mut submitted = 0usize;
        let mut in_flight = 0usize;
        while in_flight < DEFAULT_RUNAHEAD_DEPTH && submitted < budget {
            fwd.submit(&pipe)
                .map_err(|e| format!("decode submit @{}: {e}", submitted + 1))?;
            submitted += 1;
            in_flight += 1;
        }
        while in_flight > 0 {
            let t = tok_out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("tok_out.take @{}: {e}", generated.len()))?[0];
            let s = score_out
                .take()
                .get::<f32>()
                .await
                .map_err(|e| format!("score_out.take @{}: {e}", generated.len()))?[0];
            let z = null_out
                .take()
                .get::<f32>()
                .await
                .map_err(|e| format!("null_out.take @{}: {e}", generated.len()))?[0];
            in_flight -= 1;
            generated.push(t as u32);
            scores.push(s);
            nulls.push(z);
            if submitted < budget {
                fwd.submit(&pipe)
                    .map_err(|e| format!("decode submit @{}: {e}", submitted + 1))?;
                submitted += 1;
                in_flight += 1;
            }
        }
    }
    pipe.close();

    // A context-keyed watermark draws the *same* noise whenever the context
    // repeats, so a repeated context re-reports evidence the detector has
    // already counted. Averaging over it would inflate or deflate the score
    // purely as a function of how much the model looped, so score each distinct
    // context once — what published detectors do by skipping repeated n-grams.
    let mut full = prompt.clone();
    full.extend_from_slice(&generated);
    let mut history_view: Vec<i32> = full.iter().map(|&t| t as i32).collect();
    history_view.resize(cfg.capacity.max(full.len() as u32) as usize, -1);

    let mut seen: HashSet<u32> = HashSet::new();
    let mut kept_score = 0.0f32;
    let mut kept_null = 0.0f32;
    let mut kept = 0usize;
    for (t, (&s, &z)) in scores.iter().zip(nulls.iter()).enumerate() {
        let counter = host_counter(&history_view, n as usize + t, cfg.context_width);
        if seen.insert(counter) {
            kept_score += s;
            kept_null += z;
            kept += 1;
        }
    }

    let denom = kept.max(1) as f32;
    let mean_score = kept_score / denom;
    let mean_null = kept_null / denom;
    // `s ~ Exp(1)` under H0, so the mean of n draws has standard error
    // `1 / sqrt(n)` and this is already a z-score.
    let z_score = (mean_score - 1.0) * denom.sqrt();

    if cfg.watermark && kept >= 16 && mean_score <= mean_null {
        return Err(format!(
            "watermark did not register: keyed score {mean_score} did not beat the decoy {mean_null} over {kept} distinct contexts — the context hash is not reaching the sampler"
        ));
    }

    Ok(Output {
        sampler: "gumbel-watermark",
        text: wit_model::decode(&generated)?,
        count: generated.len(),
        secret: cfg.secret,
        context_width: cfg.context_width,
        watermark: cfg.watermark,
        mean_score,
        mean_null_score: mean_null,
        z_score,
        unique_contexts: kept,
    })
}
