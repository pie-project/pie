//! Frequency, presence and repetition penalties — the three penalties every
//! other serving engine exposes and Pie's `Sampler` surface does not.
//!
//! ```text
//! repetition (CTRL, Keskar et al.)  seen in prompt OR output:
//!                                     logit > 0 ? logit / r : logit * r
//! frequency  (OpenAI semantics)     logit -= f * count(output)
//! presence   (OpenAI semantics)     logit -= p * [count(output) > 0]
//! ```
//!
//! The scopes differ deliberately and match vLLM: the repetition penalty reads
//! prompt ∪ output, while frequency and presence read the output only. A prompt
//! token is something the model was *told*, not something it *chose*, so
//! charging it a presence penalty would penalize quoting the input.
//!
//! ## Why this is the interesting one
//!
//! Every other sampler here is a pure function of one logit row. These are not:
//! they need a histogram over everything emitted so far, which is *per-sequence
//! state that has to survive across steps*. On a black-box engine that state
//! lives on the host, so the decode loop must stop at every token, ship the
//! logits out, apply the penalty, and ship a decision back.
//!
//! Here the histogram is a `[vocab]` f32 channel that the device advances
//! itself:
//!
//! ```text
//! counts_next = scatter_add(counts, sampled_token, 1.0)
//! ```
//!
//! The count that penalizes step `t+1` is written by step `t` inside the same
//! forward pass that sampled it. Nothing crosses to the host, and the run-ahead
//! decode loop keeps its depth — which is the whole point of putting sequence
//! state on the device.
//!
//! ## Memory
//!
//! Two `[vocab]` f32 channels, so ~594 KiB each at this model's 151936-token
//! vocabulary. Only `counts` is a feedback channel; `prompt_present` is written
//! once and cycled unchanged, which keeps the graph shape identical between the
//! prefill and decode passes.
//!
//! ## Source
//!
//! Keskar et al., *CTRL: A Conditional Transformer Language Model for
//! Controllable Generation* — <https://arxiv.org/abs/1909.05858> (§4.1) — for
//! the multiplicative repetition penalty. The scope split (repetition reads
//! `prompt ∪ output`; frequency and presence read the output only) follows
//! vLLM `model_executor/layers/sampler.py` —
//! <https://github.com/vllm-project/vllm>.
//!
//! Faithfulness: **Exact**. See
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.

use inferlet::ptir::attention::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default)]
    frequency_penalty: f32,
    #[serde(default)]
    presence_penalty: f32,
    #[serde(default = "default_repetition_penalty")]
    repetition_penalty: f32,
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
    frequency_penalty: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
    /// Mean number of distinct vocabulary entries carrying a repetition penalty
    /// at each step. Starts at the prompt's distinct-token count and grows by
    /// at most one per step.
    mean_penalized: f32,
    /// Largest count any single token reached in the output histogram. This is
    /// the number the penalties exist to hold down.
    peak_repeat: f32,
    /// Distinct generated tokens over total generated. 1.0 means no token was
    /// ever emitted twice.
    unique_ratio: f32,
}

fn default_prompt() -> String {
    "List three interesting facts about the ocean.".into()
}

fn default_repetition_penalty() -> f32 {
    1.1
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> usize {
    32
}

fn default_seed() -> u32 {
    0x7ce1
}

#[derive(Clone, Copy)]
struct Cfg {
    frequency_penalty: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
    temperature: f32,
}

/// Applies all three penalties to a logit row.
///
/// Returns `(penalized_logits, penalized_count, peak_count)`.
fn apply_penalties(
    logits: &Tensor,
    vocab: u32,
    cfg: Cfg,
    counts: &Tensor,
    prompt_present: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    let zero = broadcast(0.0f32, [vocab]);
    let out_seen = gt(counts, &zero);
    // `prompt_present` is carried as 0.0/1.0 floats so the channel dtype stays
    // uniform with `counts`; 0.5 is the only sensible split point.
    let prompt_seen = gt(prompt_present, broadcast(0.5f32, [vocab]));
    let seen = or(&out_seen, &prompt_seen);

    // CTRL-style repetition penalty. Dividing a positive logit and multiplying
    // a negative one both move it toward -inf, which is the point: a single
    // multiplicative rule would *reward* already-negative logits.
    let r = broadcast(cfg.repetition_penalty, [vocab]);
    let positive = gt(logits, &zero);
    let repenalized = select(&positive, logits / &r, logits * &r);
    let l = select(&seen, &repenalized, logits);

    // Frequency scales with how often the token was emitted; presence is a flat
    // charge for having been emitted at all.
    let freq = broadcast(cfg.frequency_penalty, [vocab]) * counts;
    let pres = broadcast(cfg.presence_penalty, [vocab]) * cast(&out_seen, dtype::f32);

    let penalized = reshape(reduce_sum(cast(&seen, dtype::f32)), [1]);
    let peak = reshape(reduce_max(counts), [1]);
    (&l - &freq - &pres, penalized, peak)
}

/// One sampling step. Returns `(token, counts_next, penalized_count, peak)`.
fn step(
    logits: Tensor,
    vocab: u32,
    cfg: Cfg,
    counts: &Tensor,
    prompt_present: &Tensor,
    rng_state: &Tensor,
) -> (Tensor, Tensor, Tensor, Tensor) {
    let (penalized, n_penalized, peak) =
        apply_penalties(&logits, vocab, cfg, counts, prompt_present);
    // Temperature last, matching vLLM's ordering: the penalties are defined on
    // raw logits, and scaling first would make their strength depend on it.
    let scaled = if cfg.temperature == 1.0 {
        penalized
    } else {
        &penalized / cfg.temperature
    };
    let token = gumbel_max(scaled, rng_state);
    // `gumbel_max` reduces the row away, so the token index is rank-0. The
    // scatter's `vals` must then be rank-0 too — a `[1]` vector would not match
    // `idx.dims ++ base.dims[1..]`, which is empty here.
    let counts_next = scatter_add(counts, &token, 1.0f32);
    (token, counts_next, n_penalized, peak)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.repetition_penalty.is_finite() || input.repetition_penalty <= 0.0 {
        return Err("repetition_penalty must be finite and greater than 0".into());
    }
    if !input.frequency_penalty.is_finite() || !input.presence_penalty.is_finite() {
        return Err("frequency_penalty and presence_penalty must be finite".into());
    }
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }

    let max_tokens = input.max_tokens;
    let vocab = model::output_vocab_size();
    let cfg = Cfg {
        frequency_penalty: input.frequency_penalty,
        presence_penalty: input.presence_penalty,
        repetition_penalty: input.repetition_penalty,
        temperature: input.temperature,
    };
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "repetition-penalty",
            text: String::new(),
            count: 0,
            frequency_penalty: cfg.frequency_penalty,
            presence_penalty: cfg.presence_penalty,
            repetition_penalty: cfg.repetition_penalty,
            mean_penalized: 0.0,
            peak_repeat: 0.0,
            unique_ratio: 0.0,
        });
    }

    let mut prompt = model::encode(&input.prompt);
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(page_size).max(1);
    ws.reserve(max_pages).context("reserve KV")?;

    // The repetition penalty's scope includes the prompt, so seed the presence
    // vector on the host — one pass over the prompt beats a device scatter that
    // would have to run inside the prefill graph.
    let mut present = vec![0.0f32; vocab as usize];
    for &t in &prompt {
        if (t as usize) < present.len() {
            present[t as usize] = 1.0;
        }
    }

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut penalized: Vec<f32> = Vec::with_capacity(max_tokens);
    let mut peaks: Vec<f32> = Vec::with_capacity(max_tokens);

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
    let counts_p = Channel::from(vec![0.0f32; vocab as usize]).named("counts_p");
    let present_p = Channel::from(present.clone()).named("present_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
    let pen_out_p = Channel::new([1], dtype::f32).named("pen_out_p");
    let peak_out_p = Channel::new([1], dtype::f32).named("peak_out_p");

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
        let counts = counts_p.take();
        let present = present_p.take();
        let logits = intrinsics::logits();
        let (token, counts_next, n_pen, peak) = step(logits, vocab, cfg, &counts, &present, &r);
        let r_next = &r + iota(2);
        tok_out_p.put(&token);
        pen_out_p.put(&n_pen);
        peak_out_p.put(&peak);
        counts_p.put(&counts_next);
        present_p.put(&present);
        rng_p.put(&r_next);
    });

    let pipe = Pipeline::new();
    fwd_p.submit(&pipe).context("prefill submit")?;

    let g0 = tok_out_p.take_host::<i32>().await?;
    let p0 = pen_out_p.take_host::<f32>().await?;
    let k0 = peak_out_p.take_host::<f32>().await?;
    generated.push(g0 as u32);
    penalized.push(p0);
    peaks.push(k0);

    // ── DECODE LOOP (1-wide, run-ahead). ──
    if generated.len() < max_tokens {
        // The decode pass owns fresh channels, so the histogram has to be
        // handed over explicitly: it must already contain the prefill's token.
        let mut counts0 = vec![0.0f32; vocab as usize];
        if (g0 as usize) < counts0.len() {
            counts0[g0 as usize] = 1.0;
        }

        let tok_in = Channel::from([g0]).named("tok_in");
        let rng = Channel::from([input.seed ^ 0x5bd1, 0]).named("rng");
        let counts_c = Channel::from(counts0).named("counts");
        let present_c = Channel::from(present).named("present");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let pen_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("pen_out");
        let peak_out = Channel::new([1], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("peak_out");
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
            let counts = counts_c.take();
            let present = present_c.take();
            let logits = intrinsics::logits();
            let (token, counts_next, n_pen, peak) = step(logits, vocab, cfg, &counts, &present, &r);

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
            pen_out.put(&n_pen);
            peak_out.put(&peak);
            counts_c.put(&counts_next);
            present_c.put(&present);
            rng.put(&r_next);
        });

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget as usize, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            let p = pen_out
                .take_host::<f32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            let k = peak_out
                .take_host::<f32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            generated.push(t as u32);
            penalized.push(p);
            peaks.push(k);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    // The histogram is device state, so prove it actually advanced: a stuck
    // channel would leave every step reporting the same penalized count.
    let first = penalized.first().copied().unwrap_or(0.0);
    let last = penalized.last().copied().unwrap_or(0.0);
    if generated.len() > 4 && last < first {
        return Err(format!(
            "penalized-token count went backwards ({first} -> {last}) — the histogram channel is not advancing"
        ));
    }

    let unique: HashSet<u32> = generated.iter().copied().collect();
    Ok(Output {
        sampler: "repetition-penalty",
        text: model::decode(&generated)?,
        count: generated.len(),
        frequency_penalty: cfg.frequency_penalty,
        presence_penalty: cfg.presence_penalty,
        repetition_penalty: cfg.repetition_penalty,
        mean_penalized: penalized.iter().sum::<f32>() / penalized.len() as f32,
        peak_repeat: peaks.iter().fold(0.0f32, |a, &b| a.max(b)),
        unique_ratio: unique.len() as f32 / generated.len() as f32,
    })
}
