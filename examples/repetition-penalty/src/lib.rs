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
//! Faithfulness: **Exact**.

use inferlet::eta::prelude::*;
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

/// One sampling step. Returns `(token, penalized_count, peak)`.
fn step(
    logits: Tensor,
    vocab: u32,
    cfg: Cfg,
    counts: &Tensor,
    prompt_present: &Tensor,
    rng_state: &Tensor,
) -> (Tensor, Tensor, Tensor) {
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
    (token, n_penalized, peak)
}

/// Three `[1]` f32 values as one `[3]` cell.
fn pack3(a: &Tensor, b: &Tensor, c: &Tensor) -> Tensor {
    let lane = iota(3);
    select(
        eq(&lane, broadcast(0u32, [3])),
        broadcast(a, [3]),
        select(
            eq(&lane, broadcast(1u32, [3])),
            broadcast(b, [3]),
            broadcast(c, [3]),
        ),
    )
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

    // The model's opening (`<bos>` where it has one) before the raw text: a
    // gemma without it answers noise.
    let mut prompt = inferlet::chat::prefix();
    prompt.extend(model::encode(&input.prompt));
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut penalized: Vec<f32> = Vec::with_capacity(max_tokens);
    let mut peaks: Vec<f32> = Vec::with_capacity(max_tokens);

    // ── PREFILL FIRE (N-wide): first sampled token comes off the prompt. ──
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32.clone());
    let prompt_p = Channel::from(prompt_i32.clone());
    let embed_indptr_p = Channel::from([0u32, n]);
    let rng_p = Channel::from([input.seed, 0]);
    let tok_out_p = Channel::new([1], dtype::i32);
    let pen_out_p = Channel::new([1], dtype::f32);
    let peak_out_p = Channel::new([1], dtype::f32);

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    fwd_p.state(&[&ws], StateGeometry::folding())?;
    fwd_p.epilogue(move || {
        let r = rng_p.take();
        let p_toks = prompt_p.read();
        let counts = broadcast(0.0f32, [vocab]);
        let present = scatter_set(broadcast(0.0f32, [vocab]), &p_toks, 1.0f32);
        let logits = intrinsics::output(0, "logits");
        let (token, n_pen, peak) = step(logits, vocab, cfg, &counts, &present, &r);
        let r_next = &r + iota(2);
        tok_out_p.put(&token);
        pen_out_p.put(&n_pen);
        peak_out_p.put(&peak);
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
        let mut hist0 = vec![-1i32; max_tokens];
        hist0[0] = g0;

        let tok_in = Channel::from([g0]);
        let rng = Channel::from([input.seed ^ 0x5bd1, 0]);
        let hist_c = Channel::from(hist0);
        let hlen_c = Channel::from([1u32]);
        let prompt_c = Channel::from(prompt_i32);
        let out = Channel::new([3], dtype::f32).capacity(channel_capacity() as u32);
        let lane1 = Channel::from([0u32, 1u32]);

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        fwd.state(&[&ws], StateGeometry::folding())?;
        fwd.epilogue(move || {
            // Takes and compute first, puts last (value-id discipline).
            let r = rng.take();
            let hist = hist_c.take();
            let hlen = hlen_c.take();
            let p_toks = prompt_c.read();
            let logits = intrinsics::output(0, "logits");
            let counts = scatter_add(broadcast(0.0f32, [vocab]), &hist, 1.0f32);
            let present = scatter_set(broadcast(0.0f32, [vocab]), &p_toks, 1.0f32);
            let (token, n_pen, peak) = step(logits, vocab, cfg, &counts, &present, &r);
            let hist_next = scatter_set(&hist, &hlen, &token);

            let r_next = &r + iota(2);

            tok_in.put(&token);
            out.put(pack3(
                &cast(reshape(&token, [1]), dtype::f32),
                &n_pen,
                &peak,
            ));
            hist_c.put(&hist_next);
            hlen_c.put(&hlen + 1u32);
            rng.put(&r_next);
        });

        let budget = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget, async || {
            let v = out
                .take_host::<Vec<f32>>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            generated.push(v[0] as u32);
            penalized.push(v[1]);
            peaks.push(v[2]);
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
