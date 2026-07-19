//! Epsilon and eta truncation sampling (Hewitt et al.,
//! <https://arxiv.org/abs/2210.15191>, *Truncation Sampling as Language Model
//! Desmoothing*).
//!
//! Both modes are a probability floor rather than a rank or mass cut:
//!
//! * `epsilon` — keep every token with `p(x) >= epsilon`. A flat floor.
//! * `eta`     — keep every token with `p(x) >= min(epsilon, sqrt(epsilon) *
//!   exp(-H))`. The floor *relaxes* as the distribution's entropy rises, so a
//!   genuinely uncertain step is not truncated as aggressively as a confident
//!   one.
//!
//! Unlike `locally-typical-sampling`, this needs no ordering at all: it is a
//! single `pivot_threshold(probs, prob_ge(thr))`, which is `O(vocab)` rather
//! than `O(k * vocab)`. It is the cheapest possible shape for a custom sampler.
//!
//! A floor keeps the mask non-empty: the argmax token is OR-ed back in, so an
//! aggressive `epsilon` degenerates to greedy instead of to an empty set.
//!
//! ## Source
//!
//! Hewitt et al., *Truncation Sampling as Language Model Desmoothing* —
//! <https://arxiv.org/abs/2210.15191> (§3).
//!
//! Faithfulness: **Exact**. See
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default = "default_epsilon")]
    epsilon: f32,
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
    mode: String,
    text: String,
    count: usize,
    epsilon: f32,
    mean_kept: f32,
    min_kept: u32,
    mean_mass: f32,
}

fn default_prompt() -> String {
    "Write a short paragraph about truncation sampling.".into()
}

fn default_mode() -> String {
    "eta".into()
}

/// The eta-sampling paper's default order of magnitude for the floor.
fn default_epsilon() -> f32 {
    0.0003
}

fn default_temperature() -> f32 {
    1.0
}

fn default_max_tokens() -> usize {
    32
}

fn default_seed() -> u32 {
    0x3f17
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Epsilon,
    Eta,
}

/// The truncation keep-mask.
///
/// Returns `(keep_mask, kept_count, kept_mass)`. The argmax indicator is OR-ed
/// in so the mask is never empty regardless of `epsilon`.
fn truncation_keep(
    logits: &Tensor,
    vocab: u32,
    mode: Mode,
    epsilon: f32,
) -> (Tensor, Tensor, Tensor) {
    let logprobs = log_softmax(logits);
    let probs = exp(&logprobs);

    let threshold = match mode {
        Mode::Epsilon => Tensor::constant(epsilon),
        Mode::Eta => {
            // min(eps, sqrt(eps) * exp(-H)): the floor relaxes with entropy.
            let h = entropy_from_logprobs(&probs, &logprobs);
            let adaptive = mul(Tensor::constant(epsilon.sqrt()), exp(neg(&h)));
            min_elem(Tensor::constant(epsilon), adaptive)
        }
    };

    let base_mask = pivot_threshold(&probs, prob_ge(threshold));
    // Never-empty floor: the max-probability token's own indicator.
    let argmax_ind = ge(&probs, broadcast(reduce_max(&probs), [vocab]));
    let keep = or(base_mask, argmax_ind);

    let zeros = broadcast(Tensor::constant(0.0f32), [vocab]);
    let kept_mass = reshape(reduce_sum(select(&keep, &probs, &zeros)), [1]);
    let kept = reshape(reduce_sum(cast(&keep, DType::F32)), [1]);
    (keep, kept, kept_mass)
}

/// One sampling step: temperature, truncation mask, Gumbel-max draw.
fn truncation_step(
    logits: Tensor,
    vocab: u32,
    mode: Mode,
    temperature: f32,
    epsilon: f32,
    rng_state: impl AsTensor,
) -> (Tensor, Tensor, Tensor) {
    let scaled = if temperature == 1.0 {
        logits
    } else {
        div(&logits, temperature)
    };
    let (keep, kept, kept_mass) = truncation_keep(&scaled, vocab, mode, epsilon);
    let neg_inf = broadcast(Tensor::constant(f32::NEG_INFINITY), [vocab]);
    let masked = select(&keep, &scaled, &neg_inf);
    let token = gumbel_max(masked, rng_state);
    (token, kept, kept_mass)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.epsilon.is_finite() || input.epsilon <= 0.0 || input.epsilon >= 1.0 {
        return Err("epsilon must be finite and in (0, 1)".into());
    }
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }
    let mode = match input.mode.as_str() {
        "eta" => Mode::Eta,
        "epsilon" => Mode::Epsilon,
        other => return Err(format!("unknown mode: {other} (expected eta or epsilon)")),
    };
    let epsilon = input.epsilon;
    let temperature = input.temperature;
    let max_tokens = input.max_tokens;
    let vocab = wit_model::output_vocab_size();
    let ws = WorkingSet::new();
    let page_size = ws.page_size();

    if max_tokens == 0 {
        return Ok(Output {
            sampler: "truncation",
        mode: input.mode.clone(),
            text: String::new(),
            count: 0,
            epsilon,
            mean_kept: 0.0,
            min_kept: 0,
            mean_mass: 0.0,
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

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut kept_sizes: Vec<f32> = Vec::with_capacity(max_tokens);
    let mut kept_mass: Vec<f32> = Vec::with_capacity(max_tokens);

    // ── PREFILL FIRE (N-wide): first sampled token comes off the prompt. ──
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let embed_indptr_p = Channel::from(vec![0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from((0..n).collect::<Vec<_>>()).named("positions_p");
    let pages_p = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages_p");
    let page_indptr_p = Channel::from(vec![0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p =
        Channel::from((0..n).map(|p| p / page_size).collect::<Vec<_>>()).named("w_slot_p");
    let w_off_p = Channel::from((0..n).map(|p| p % page_size).collect::<Vec<_>>()).named("w_off_p");
    let kv_len_p = Channel::from(vec![n]).named("kv_len_p");
    let rng_p = Channel::from(vec![input.seed, 0]).named("rng_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
    let kept_out_p = Channel::new([1], dtype::f32).named("kept_out_p");
    let mass_out_p = Channel::new([1], dtype::f32).named("mass_out_p");

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
        let logits = intrinsics::logits();
        let (token, kept, kmass) = truncation_step(logits, vocab, mode, temperature, epsilon, &r);
        let r_next = add(&r, iota(2));
        tok_out_p.put(&token);
        kept_out_p.put(&kept);
        mass_out_p.put(&kmass);
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
    let k0 = kept_out_p
        .take()
        .get::<f32>()
        .await
        .map_err(|e| format!("k0 take: {e}"))?[0];
    let m0 = mass_out_p
        .take()
        .get::<f32>()
        .await
        .map_err(|e| format!("m0 take: {e}"))?[0];
    generated.push(g0 as u32);
    kept_sizes.push(k0);
    kept_mass.push(m0);

    // ── DECODE LOOP (1-wide, run-ahead). ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
        let rng = Channel::from(vec![input.seed ^ 0x5bd1, 0]).named("rng");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
            .named("tok_out");
        let kept_out = Channel::new([1], dtype::f32)
            .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
            .named("kept_out");
        let mass_out = Channel::new([1], dtype::f32)
            .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
            .named("mass_out");
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
            let logits = intrinsics::logits();
            let (token, kept, kmass) = truncation_step(logits, vocab, mode, temperature, epsilon, &r);

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
            kept_out.put(&kept);
            mass_out.put(&kmass);
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
            let k = kept_out
                .take()
                .get::<f32>()
                .await
                .map_err(|e| format!("kept_out.take @{}: {e}", generated.len()))?[0];
            let m = mass_out
                .take()
                .get::<f32>()
                .await
                .map_err(|e| format!("mass_out.take @{}: {e}", generated.len()))?[0];
            in_flight -= 1;
            generated.push(t as u32);
            kept_sizes.push(k);
            kept_mass.push(m);
            if submitted < budget {
                fwd.submit(&pipe)
                    .map_err(|e| format!("decode submit @{}: {e}", submitted + 1))?;
                submitted += 1;
                in_flight += 1;
            }
        }
    }
    pipe.close();

    let mean_kept = kept_sizes.iter().sum::<f32>() / kept_sizes.len() as f32;
    let min_kept = kept_sizes
        .iter()
        .fold(f32::INFINITY, |a, &b| a.min(b))
        .max(0.0) as u32;
    if min_kept == 0 {
        return Err("truncation keep-set was empty — the argmax floor failed".into());
    }

    Ok(Output {
        sampler: "truncation",
        mode: input.mode.clone(),
        text: wit_model::decode(&generated)?,
        count: generated.len(),
        epsilon,
        mean_kept,
        min_kept,
        mean_mass: kept_mass.iter().sum::<f32>() / kept_mass.len() as f32,
    })
}
