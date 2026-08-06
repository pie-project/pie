//! SynthID-Text tournament sampling (Dathathri et al., *Nature* 2024,
//! "Scalable watermarking for identifying large language model outputs").
//!
//! Tournament sampling draws `2^depth` candidate tokens i.i.d. from the model
//! distribution and runs `depth` knockout rounds; round ℓ scores each candidate
//! with a keyed Bernoulli(½) g-function `g_ℓ(context, token)` and the higher
//! score advances. The paper proves this is equivalent to a closed-form update
//! on the distribution itself, and that identity is what DeepMind's reference
//! implementation actually ships (`SynthIDTextWatermarkLogitsProcessor::update_scores`):
//!
//! ```text
//! p_0 = softmax(logits / T)
//! for ℓ in 0..depth:
//!     m_ℓ    = Σ_v p_ℓ(v) · g_ℓ(v)          expected g-mass under p_ℓ
//!     p_{ℓ+1}(v) = p_ℓ(v) · (1 + g_ℓ(v) − m_ℓ)
//! x_t ~ p_depth
//! ```
//!
//! Each factor lies in `[0, 2]` and `Σ_v p(1 + g − m) = 1 + m − m = 1`, so the
//! update is a normalised reweighting — no renormalisation step is needed, and
//! candidates the round's g-function favours gain mass in proportion to how
//! surprising that favour is. Running the `2^depth` tournament explicitly would
//! cost `2^depth` sampling ops; the closed form costs `depth` vector ops, so a
//! 9-layer watermark is 9 passes over `[vocab]` rather than 512 draws.
//!
//! ## Repeated contexts
//!
//! Both the reference generator and its detector special-case a context n-gram
//! that has been seen inside the last `history_size` steps: the generator
//! **skips watermarking** (`torch.where(is_repeated_context, scores, updated)`)
//! and the detector masks the step out. Reusing a key on a repeated context
//! would re-spend evidence the detector already counted and would visibly
//! distort looping text, so both sides are implemented here — the device keeps
//! a ring buffer of context hashes and the host mirrors it exactly.
//!
//! ## Detection
//!
//! The mean-score detector averages `g_ℓ(x_t)` over layers and unmasked steps.
//! Every `g` is a keyed fair coin, so the null mean is exactly ½ and
//! `z = 2·(mean − ½)·sqrt(depth · n)` is a standard normal. A decoy secret is
//! scored alongside to give the empirical null on the same text.
//!
//! ## Source
//!
//! Dathathri et al., *Scalable watermarking for identifying large language
//! model outputs*, *Nature* **634**, 818–823 (2024) —
//! <https://doi.org/10.1038/s41586-024-08025-4>.
//!
//! Faithfulness: **Exact (equivalent form)**. The paper specifies a knockout
//! tournament over `2^depth` sampled candidates; this implements the closed
//! form `P(w) = p(w)·(1 + g(w) − m)` with `m = E_p[g]`, which is that
//! tournament's exact per-round win probability and composes across layers
//! because round-ℓ winners are i.i.d. from `p_ℓ`. Derivation in
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.

use inferlet::chat;
use inferlet::ptir::attention::prelude::*;
use serde::{Deserialize, Serialize};

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
    /// `ngram_len - 1` in the reference config: how many preceding tokens seed
    /// each step's g-functions.
    #[serde(default = "default_context_width")]
    context_width: u32,
    /// Number of tournament layers. The reference config ships nine keys.
    #[serde(default = "default_depth")]
    depth: u32,
    /// How far back the repeated-context ring buffer remembers.
    #[serde(default = "default_history_size")]
    history_size: u32,
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
    depth: u32,
    watermark: bool,
    /// Mean g-value of the emitted tokens under the real secret, averaged over
    /// layers and unmasked steps. Exactly `0.5` under H0.
    mean_score: f32,
    /// The same statistic under a decoy secret — the empirical null.
    mean_null_score: f32,
    /// `2 * (mean_score - 0.5) * sqrt(depth * n)`. Each g is a fair coin under
    /// H0, so this is a standard-normal z-score.
    z_score: f32,
    /// Steps that contributed to the statistic; steps whose context n-gram
    /// repeated within the ring buffer are masked out on both sides.
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

fn default_depth() -> u32 {
    9
}

fn default_history_size() -> u32 {
    64
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
    depth: u32,
    history_size: u32,
    watermark: bool,
    temperature: f32,
    capacity: u32,
}

/// Salts that keep the three keyed streams — layer g-functions, the decoy's
/// g-functions, and the free-running sampler — from ever sharing an rng state.
const G_SALT: u32 = 0x0100_0000;
const DECOY_SALT: u32 = 0x0200_0000;

/// Layer ℓ's g-function: a keyed fair coin per vocabulary entry.
///
/// The reference implementation hashes `(context n-gram, candidate token,
/// layer key)` into a precomputed Bernoulli table. Here the driver's
/// counter-based RNG is the hash: the `[key, ctr]` state fixes the context and
/// the layer, and the element index within the `[vocab]` draw supplies the
/// candidate token, so `g` is a pure function of the same three inputs.
fn g_layer(secret: u32, counter: &Tensor, layer: u32, vocab: u32) -> Tensor {
    let state = rng_state(secret ^ (G_SALT.wrapping_mul(layer + 1)), counter);
    cast(ge(rng(&state, [vocab]), 0.5f32), dtype::f32)
}

/// Rolling hash of the `context_width` tokens ending at `hlen - 1`.
///
/// Returns a `[1]` u32 counter. Positions before the start of the buffer are
/// clamped and contribute the sentinel, which is harmless: every step at the
/// same offset sees the same clamped prefix, so the counter stays a pure
/// function of the context.
fn context_counter(hist: &Tensor, hlen: &Tensor, cfg: Cfg) -> Tensor {
    let last = cast(hlen, dtype::i32) - 1i32;
    let mut h = broadcast(0u32, [1]);
    // Oldest token first, so the hash is the usual left-to-right polynomial.
    for d in (0..cfg.context_width).rev() {
        let idx = max_elem(&last - d as i32, 0i32);
        // `+1` lifts the `-1` padding sentinel to 0 and keeps every real token
        // distinguishable from it.
        let tok = cast(gather(hist, &idx) + 1i32, dtype::u32);
        h = (&h * HASH_MULTIPLIER + tok) % HASH_MODULUS;
    }
    h
}

/// Assembles the `[2]` u32 `[key, ctr]` rng state from two `[1]` parts.
fn rng_state(key: u32, counter: &Tensor) -> Tensor {
    let is_counter = eq(iota(2), broadcast(1u32, [2]));
    select(&is_counter, broadcast(counter, [2]), broadcast(key, [2]))
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

/// The mean-score detector statistic: the emitted token's g-value averaged over
/// the `depth` layers. Exactly `0.5` in expectation under H0.
fn mean_g(gs: &[Tensor], token: &Tensor) -> Tensor {
    let mut total = scalar_gather(&gs[0], token);
    for g in &gs[1..] {
        total += scalar_gather(g, token);
    }
    reshape(total / gs.len() as f32, [1])
}

/// One sampling step.
///
/// Returns `(token, hist_next, chist_next, score, null_score)`.
fn step(
    logits: Tensor,
    vocab: u32,
    cfg: Cfg,
    hist: &Tensor,
    hlen: &Tensor,
    chist: &Tensor,
    free_state: &Tensor,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
    let scaled = if cfg.temperature == 1.0 {
        logits
    } else {
        &logits / cfg.temperature
    };

    let counter = context_counter(hist, hlen, cfg);
    // Has this exact context n-gram been seen inside the ring buffer? The
    // reference generator skips watermarking when it has.
    let repeated = gt(
        reduce_max(cast(
            eq(chist, broadcast(&counter, [cfg.history_size])),
            dtype::u32,
        )),
        0u32,
    );

    let base = softmax(&scaled);
    let gs: Vec<Tensor> = (0..cfg.depth)
        .map(|layer| g_layer(cfg.secret, &counter, layer, vocab))
        .collect();

    // The closed form of the 2^depth tournament: one multiplicative reweighting
    // per layer. Each factor is `1 + g - E_p[g]`, which is non-negative because
    // `g ∈ {0,1}` and `E_p[g] ∈ [0,1]`, and preserves normalisation exactly.
    let mut watermarked = base.clone();
    for g in &gs {
        let mass = reduce_sum(&watermarked * g);
        let offset = broadcast(1.0f32 - &mass, [vocab]);
        watermarked = &watermarked * (&offset + g);
    }

    let effective = select(broadcast(&repeated, [vocab]), &base, &watermarked);
    // `log` of a probability that underflowed would be -inf and would poison
    // the Gumbel-max comparison, so the floor is applied before the log.
    let wm_logits = log(max_elem(&effective, 1e-30f32));

    let token = if cfg.watermark {
        gumbel_max(&wm_logits, free_state)
    } else {
        gumbel_max(&scaled, free_state)
    };

    let score = mean_g(&gs, &token);
    // The decoy shares the context counter and differs only in the secret, so
    // the null is the *same* experiment with the wrong key.
    let decoy_gs: Vec<Tensor> = (0..cfg.depth)
        .map(|layer| g_layer(cfg.decoy ^ DECOY_SALT, &counter, layer, vocab))
        .collect();
    let null_score = mean_g(&decoy_gs, &token);

    let hist_next = scatter_set(hist, hlen, &token);
    let chist_next = scatter_set(chist, hlen % cfg.history_size, &counter);
    (token, hist_next, chist_next, score, null_score)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if input.context_width == 0 || input.context_width > 8 {
        return Err("context_width must satisfy 1 <= context_width <= 8".into());
    }
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }
    if input.depth == 0 || input.depth > 16 {
        return Err("depth must satisfy 1 <= depth <= 16".into());
    }
    if input.history_size == 0 || input.history_size > 1024 {
        return Err("history_size must satisfy 1 <= history_size <= 1024".into());
    }

    let max_tokens = input.max_tokens;
    let vocab = model::output_vocab_size();
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "synthid-tournament-sampling",
            text: String::new(),
            count: 0,
            secret: input.secret,
            context_width: input.context_width,
            depth: input.depth,
            watermark: input.watermark,
            mean_score: 0.0,
            mean_null_score: 0.0,
            z_score: 0.0,
            unique_contexts: 0,
        });
    }

    // The chat template matters here: watermark strength is proportional to the
    // entropy the sampler has to work with, and a raw completion prompt on an
    // instruction-tuned model degenerates into a low-entropy loop that no
    // reweighting scheme can mark.
    let mut prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;

    let cfg = Cfg {
        secret: input.secret,
        // A decoy that shares no low bits with the secret, so the two keyed
        // noise fields are unrelated.
        decoy: input.secret.wrapping_add(0x9E37_79B9),
        context_width: input.context_width,
        depth: input.depth,
        history_size: input.history_size,
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
    let toks_p = Channel::from_iter(history.iter().take(n as usize).copied()).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from_iter(0..n).named("positions_p");
    let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
    let page_indptr_p = Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p = Channel::from_iter((0..n).map(|p| p / page_size)).named("w_slot_p");
    let w_off_p = Channel::from_iter((0..n).map(|p| p % page_size)).named("w_off_p");
    let kv_len_p = Channel::from([n]).named("kv_len_p");
    let rng_p = Channel::from([input.seed, 0]).named("rng_p");
    let hist_p = Channel::from(history.clone()).named("hist_p");
    let hlen_p = Channel::from([n]).named("hlen_p");
    let chist_p =
        Channel::from(vec![u32::MAX; cfg.history_size as usize]).named("context_history_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
    let score_out_p = Channel::new([1], dtype::f32).named("score_out_p");
    let null_out_p = Channel::new([1], dtype::f32).named("null_out_p");

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
        let hist = hist_p.take();
        let hlen = hlen_p.take();
        let chist = chist_p.take();
        let logits = intrinsics::logits();
        let (token, hist_next, chist_next, score, null) =
            step(logits, vocab, cfg, &hist, &hlen, &chist, &r);
        let r_next = &r + iota(2);
        tok_out_p.put(&token);
        score_out_p.put(&score);
        null_out_p.put(&null);
        chist_p.put(&chist_next);
        hist_p.put(&hist_next);
        hlen_p.put(&hlen + 1u32);
        rng_p.put(&r_next);
    });

    let pipe = Pipeline::new();
    fwd_p.submit(&pipe).context("prefill submit")?;

    let g0 = tok_out_p.take_host::<i32>().await?;
    let s0 = score_out_p.take_host::<f32>().await?;
    let n0 = null_out_p.take_host::<f32>().await?;
    generated.push(g0 as u32);
    scores.push(s0);
    nulls.push(n0);

    // ── DECODE LOOP (1-wide, run-ahead). ──
    if generated.len() < max_tokens {
        history[n as usize] = g0;

        let tok_in = Channel::from([g0]).named("tok_in");
        let rng = Channel::from([input.seed ^ 0x5bd1, 0]).named("rng");
        let hist_c = Channel::from(history.clone()).named("hist");
        let hlen_c = Channel::from([n + 1]).named("hlen");
        // The decode fire gets its own channel, so the ring is seeded with the
        // entry the prefill fire wrote — otherwise the device and the host
        // mirror would disagree about the first context.
        let mut ring_seed = vec![u32::MAX; cfg.history_size as usize];
        ring_seed[(n % cfg.history_size) as usize] =
            host_counter(&history, n as usize, cfg.context_width);
        let chist_c = Channel::from(ring_seed).named("context_history");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let score_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("score_out");
        let null_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("null_out");
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
            let hist = hist_c.take();
            let hlen = hlen_c.take();
            let chist = chist_c.take();
            let logits = intrinsics::logits();
            let (token, hist_next, chist_next, score, null) =
                step(logits, vocab, cfg, &hist, &hlen, &chist, &r);

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
            score_out.put(&score);
            null_out.put(&null);
            chist_c.put(&chist_next);
            hist_c.put(&hist_next);
            hlen_c.put(&hlen + 1u32);
            rng.put(&r_next);
        });

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget as usize, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            let s = score_out
                .take_host::<f32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            let z = null_out
                .take_host::<f32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            generated.push(t as u32);
            scores.push(s);
            nulls.push(z);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    // Both sides of the reference implementation mask a step whose context
    // n-gram appeared inside the ring buffer: the generator skipped
    // watermarking there, so scoring it would only add noise. The ring is
    // replayed here exactly as the device maintained it.
    let mut full = prompt.clone();
    full.extend_from_slice(&generated);
    let mut history_view: Vec<i32> = full.iter().map(|&t| t as i32).collect();
    history_view.resize(cfg.capacity.max(full.len() as u32) as usize, -1);

    let mut ring = vec![u32::MAX; cfg.history_size as usize];
    let mut kept_score = 0.0f32;
    let mut kept_null = 0.0f32;
    let mut kept = 0usize;
    for (t, (&s, &z)) in scores.iter().zip(nulls.iter()).enumerate() {
        let position = n as usize + t;
        let counter = host_counter(&history_view, position, cfg.context_width);
        if !ring.contains(&counter) {
            kept_score += s;
            kept_null += z;
            kept += 1;
        }
        ring[position % cfg.history_size as usize] = counter;
    }

    let denom = kept.max(1) as f32;
    let mean_score = kept_score / denom;
    let mean_null = kept_null / denom;
    // Every g is an independent fair coin under H0, so the mean of
    // `depth * kept` of them has standard error `0.5 / sqrt(depth * kept)`.
    let z_score = 2.0 * (mean_score - 0.5) * (cfg.depth as f32 * denom).sqrt();

    if cfg.watermark && kept >= 16 && mean_score <= mean_null {
        return Err(format!(
            "watermark did not register: keyed mean g {mean_score} did not beat the decoy {mean_null} over {kept} unmasked contexts — the context hash is not reaching the tournament"
        ));
    }

    Ok(Output {
        sampler: "synthid-tournament-sampling",
        text: model::decode(&generated)?,
        count: generated.len(),
        secret: cfg.secret,
        context_width: cfg.context_width,
        depth: cfg.depth,
        watermark: cfg.watermark,
        mean_score,
        mean_null_score: mean_null,
        z_score,
        unique_contexts: kept,
    })
}
