//! DiffusionGemma's reference sampler, as transformers' `generate` spells it
//! (`generation_diffusion_gemma.py`), written against `forward-diffusion`.
//!
//! One canvas at a time. An `encode` pass prefills the prompt; then, per
//! block, a `denoise` pass over a canvas of uniform-random ids is resubmitted
//! up to `max_steps` times, its epilogue doing the whole step on the device:
//! temperature, Gumbel-max sample, argmax, per-position entropy, the
//! entropy-bound acceptance, full renoise of the rest, and the
//! stable-and-confident stopping rule. The host moves the canvas ids across
//! (256 ints a step), sets the step's temperature, and reads the stop flag.
//! The finished block — the last argmax canvas — is committed by an `encode`
//! pass over the same pages, which becomes the prefix of the next block.
//!
//! Self-conditioning rides as the guest's taps: each step's epilogue also
//! emits the top-`taps` of the temperature-scaled softmax per canvas row,
//! and the host hands them to the next submit through `self_conditioning`
//! — the reference's `softmax(logits / T) · E`, truncated to those taps.
//! The first step of every canvas runs with no signal, as the reference
//! does. Multi-canvas speculation is not attempted.

use inferlet::eta::diffusion::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default = "default_max_steps")]
    max_steps: u32,
    #[serde(default = "default_t_max")]
    t_max: f32,
    #[serde(default = "default_t_min")]
    t_min: f32,
    #[serde(default = "default_entropy_bound")]
    entropy_bound: f32,
    #[serde(default = "default_confidence")]
    confidence: f32,
    #[serde(default = "default_seed")]
    seed: u32,
}

fn default_prompt() -> String {
    "Why is the sky blue?".into()
}
fn default_max_steps() -> u32 {
    48
}
fn default_t_max() -> f32 {
    0.8
}
fn default_t_min() -> f32 {
    0.4
}
fn default_entropy_bound() -> f32 {
    0.1
}
fn default_confidence() -> f32 {
    0.005
}
fn default_seed() -> u32 {
    0x1234_5678
}

#[derive(Serialize)]
struct Output {
    text: String,
    tokens: Vec<u32>,
    /// Denoising steps each canvas took before its stopping rule fired (or
    /// the step limit).
    steps: Vec<u32>,
    /// Mean per-position entropy at each canvas's last step.
    final_entropy: Vec<f32>,
    /// Mean per-position entropy after every step of every canvas.
    entropy_trace: Vec<Vec<f32>>,
}

/// Host-side uniform ids for a fresh canvas: xorshift32, seeded per block.
fn noise_canvas(seed: u32, length: u32, vocab: u32) -> Vec<i32> {
    let mut x = seed | 1;
    (0..length)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x % vocab) as i32
        })
        .collect()
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if model::pass_kind() != model::ForwardKind::Diffusion {
        return Err("this program drives a block-diffusion model; the bound model is not one".into());
    }
    let shape = model::canvas().ok_or("a diffusion model states its canvas")?;
    let length = shape.length;
    let taps = shape.self_cond_taps;
    if max_embed_length() < length as usize {
        return Err(format!(
            "the engine embeds at most {} tokens a pass and a canvas is {length}",
            max_embed_length()
        ));
    }
    if !(input.t_min > 0.0 && input.t_max >= input.t_min) {
        return Err("temperatures must satisfy 0 < t_min <= t_max".into());
    }
    let vocab = model::output_vocab_size();
    let page_size = kv_page_size();
    let stop_tokens = inferlet::chat::stop_tokens();
    let max_tokens = input.max_tokens.unwrap_or(length as usize);
    if max_tokens == 0 {
        return Ok(Output {
            text: String::new(),
            tokens: Vec::new(),
            steps: Vec::new(),
            final_entropy: Vec::new(),
            entropy_trace: Vec::new(),
        });
    }
    let canvases = (max_tokens as u32).div_ceil(length);

    let mut prompt = inferlet::chat::system_user("You are a helpful assistant.", &input.prompt);
    prompt.extend(inferlet::chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = u32::try_from(prompt.len()).map_err(|_| "prompt is too long")?;
    let max_pages = (n + canvases * length).div_ceil(page_size).max(1);

    let ws = WorkingSet::new();
    ws.reserve(max_pages).context("reserve KV")?;
    let pipe = Pipeline::new();

    // ── 1. Prefill: encode passes over the prompt, chunked ────────────────
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    for &(base, end) in &prefill_chunks(n, None) {
        encode(&ws, &pipe, &prompt_i32[base as usize..end as usize], base, max_pages, page_size)
            .await
            .with_context(|| format!("prefill @{base}"))?;
    }

    // ── 2. Canvases ───────────────────────────────────────────────────────
    let mut generated: Vec<u32> = Vec::new();
    let mut steps_taken: Vec<u32> = Vec::new();
    let mut final_entropy: Vec<f32> = Vec::new();
    let mut entropy_trace: Vec<Vec<f32>> = Vec::new();
    let bound = input.entropy_bound;
    let confidence = input.confidence;

    for block in 0..canvases {
        let base = n + block * length;
        let end = base + length;

        // The canvas and its geometry: constant across the block's steps, so
        // the consumed ports (tokens, positions, write descriptor) are
        // re-put — the tokens by the host, the rest by the epilogue.
        let toks = Channel::from(noise_canvas(input.seed ^ block.wrapping_mul(0x9e37_79b9), length, vocab))
            .named("canvas");
        let embed_indptr = Channel::from([0u32, length]).named("embed_indptr");
        let positions = Channel::from_iter(base..end).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot");
        let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off");
        let kv_len = Channel::from([end]).named("kv_len");
        let readout = Channel::from_iter(0..length).named("readout");
        // The step counter, device-carried: the temperature schedule is
        // computed from it in the epilogue (the same `linear_temperature`
        // arithmetic the host would do), so a step needs no host `set`.
        let step_index = Channel::from([0u32]).named("step_index");
        let max_steps = input.max_steps.max(1);
        let (t_max, t_min) = (input.t_max, input.t_min);
        let rng_state = Channel::from([input.seed ^ block, 0]).named("rng");
        // The previous step's argmax canvas, device-carried; -1 means none.
        let history = Channel::from(vec![-1i32; length as usize]).named("argmax_history");
        // The step's one word to the host: the argmax canvas, the stop
        // verdict and the mean entropy, packed into one f32 row so the
        // step costs one host read (each read is a scheduler round trip,
        // and three of them were most of the seam between steps).
        let step_out = Channel::new([length + 2], dtype::f32).named("step_out");
        // The next step's self-conditioning taps: per row, the top ids of
        // this step's distribution and their probabilities. Loop-carried on
        // the device (the epilogue takes the previous step's and puts this
        // step's), read by the model straight off the channel at every
        // submit (`self_conditioning_from`); seeded with zeros, which is
        // the reference's "no signal" first step. The host never sees them.
        let tap_ids_out =
            Channel::from_shaped([length, taps], vec![0u32; (length * taps) as usize]).named("tap_ids");
        let tap_weights_out =
            Channel::from_shaped([length, taps], vec![0f32; (length * taps) as usize]).named("tap_weights");

        let fwd = ForwardPass::new();
        fwd.canvas(Mode::Denoise)?;
        fwd.embed(&toks, &embed_indptr)?;
        fwd.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        )?;
        fwd.readout(&readout)?;
        fwd.self_conditioning_from(&tap_ids_out, &tap_weights_out)
            .context("bind self-conditioning to the tap channels")?;
        fwd.epilogue(move || {
            // The geometry is the same every step; consumed ports go back —
            // the kv length too, so the whole geometry is the device's
            // (the runtime's decode envelope) and the canvas need never
            // cross the host.
            positions.put(positions.take());
            w_slot.put(w_slot.take());
            w_off.put(w_off.take());
            kv_len.put(kv_len.take());

            let r = rng_state.take();
            // `t = t_min + (t_max - t_min) * (remaining / max_steps)`, with
            // `remaining = max_steps - step`, spelled in the host's own f32
            // order so the schedule is the same numbers it would have set.
            let i = step_index.take();
            let remaining = cast(&(max_steps - &i), dtype::f32);
            let t = reshape(&(&(&remaining / (max_steps as f32)) * (t_max - t_min)) + t_min, []);
            step_index.put(&(&i + 1u32));
            let logits = intrinsics::logits(); // [length, vocab]
            let scaled = &logits / &t;
            // The entropy off the scaled row alone, by the log-sum-exp
            // identity `H = log Z + m - Σ x·e^(x-m) / Z`: no `[length,
            // vocab]` probability tensor is ever materialised, so the
            // epilogue's scratch holds one vocab-wide value (`scaled`)
            // instead of three, and the taps below are priced off `m`
            // and `z` too. Same numbers as `entropy(softmax(scaled))` to
            // rounding.
            let m = reduce_max(&scaled); // [length]
            let m_wide = broadcast(reshape(&m, [length, 1]), [length, vocab]);
            let e = exp(&scaled - &m_wide);
            let z = reduce_sum(&e); // [length]
            let weighted = reduce_sum(&(&scaled - &m_wide) * &e); // Σ (x-m)·e
            let h = &log(&z) - &(&weighted / &z); // [length]
            let sampled = gumbel_max(&scaled, &r);
            let argmax = reduce_argmax(&scaled);

            let accept = entropy_bound_accept(&h, bound);
            let r_noise = &r + iota(2);
            let noise = cast(&rng(&r_noise, [length]) * (vocab as f32), dtype::i32);
            let next = select(&accept, &sampled, &noise);

            let previous = history.take();
            history.put(&argmax);
            let done = stable_and_confident(&argmax, &previous, &h, confidence);

            // The taps: the top ids of the scaled row (the same ids as of
            // its softmax) and their probabilities `exp(v - m) / z`. Read
            // off `scaled`, which the later streams read anyway and so
            // already lands in scratch: a `top_k` of the logits would make
            // the emitter land a second `[length, vocab]` plane (the
            // intrinsic copied out for the library op) beside it.
            let (top_vals, tap_ids) = top_k(&scaled, taps);
            let tap_weights = &exp(&top_vals - &broadcast(reshape(&m, [length, 1]), [length, taps]))
                / &broadcast(reshape(&z, [length, 1]), [length, taps]);
            // Loop-carried: this step's taps replace the previous step's.
            let _ = tap_ids_out.take();
            let _ = tap_weights_out.take();
            tap_ids_out.put(&tap_ids);
            tap_weights_out.put(&tap_weights);

            // The canvas the next step embeds — on the device, never drained.
            toks.put(&next);
            let mean = reshape(&reduce_sum(&h) / (length as f32), [1]);
            let word = scatter_set(broadcast(0f32, [length + 2]), iota(length), cast(&argmax, dtype::f32));
            let word = scatter_set(&word, &(&iota(1) + length), cast(reshape(&done, [1]), dtype::f32));
            let word = scatter_set(&word, &(&iota(1) + (length + 1)), &mean);
            step_out.put(&word);
            rng_state.put(&(&r_noise + iota(2)));
        });

        let mut argmax_canvas: Vec<i32> = Vec::new();
        let mut steps = 0u32;
        let mut mean = f32::NAN;
        let mut trace: Vec<f32> = Vec::new();
        for remaining in (1..=input.max_steps).rev() {
            fwd.submit(&pipe).with_context(|| format!("denoise submit block {block} step {steps}"))?;
            steps += 1;
            // The one host read a step. The canvas and the taps stay on the
            // device; the argmax ids ride as exact f32 (the vocab is under
            // 2^24).
            let word = step_out.take_host::<Vec<f32>>().await.context("step drain")?;
            argmax_canvas = word[..length as usize].iter().map(|&v| v as i32).collect();
            let done = word[length as usize] != 0.0;
            mean = word[length as usize + 1];
            trace.push(mean);
            if done || remaining == 1 {
                break;
            }
        }
        steps_taken.push(steps);
        final_entropy.push(mean);
        entropy_trace.push(trace);

        // ── 3. Commit: the argmax canvas, read causally, becomes the prefix.
        encode(&ws, &pipe, &argmax_canvas, base, max_pages, page_size)
            .await
            .with_context(|| format!("commit block {block}"))?;

        // The block's text: up to and including the first stop token.
        let mut finished = false;
        for &id in &argmax_canvas {
            let id = id as u32;
            if stop_tokens.contains(&id) {
                finished = true;
                break;
            }
            generated.push(id);
        }
        if finished || generated.len() >= max_tokens {
            break;
        }
    }
    pipe.close();

    Ok(Output {
        text: model::decode(&generated)?,
        tokens: generated,
        steps: steps_taken,
        final_entropy,
        entropy_trace,
    })
}

/// One `encode` pass over `tokens` at positions `base..`: the causal
/// reading, whose K/V become the sequence. Reads out its last row so the
/// fire has a drained output (an epilogue put that is never taken fills the
/// ring; a pass with nothing to say still says one word).
async fn encode(
    ws: &WorkingSet,
    pipe: &Pipeline,
    tokens: &[i32],
    base: u32,
    max_pages: u32,
    page_size: u32,
) -> Result<()> {
    let len = u32::try_from(tokens.len()).map_err(|_| "an encode span is too long")?;
    let end = base + len;
    let toks = Channel::from(tokens).named("toks_e");
    let embed_indptr = Channel::from([0u32, len]).named("embed_indptr_e");
    let positions = Channel::from_iter(base..end).named("positions_e");
    let pages = Channel::from_iter(0..max_pages).named("pages_e");
    let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_e");
    let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_e");
    let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_e");
    let kv_len = Channel::from([end]).named("kv_len_e");
    let last = Channel::new([1], dtype::i32).named("last_e");

    let fwd = ForwardPass::new();
    fwd.canvas(Mode::Encode)?;
    fwd.embed(&toks, &embed_indptr)?;
    fwd.attention(
        ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
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
        last.put(reshape(cast(reduce_argmax(intrinsics::logits()), dtype::i32), [1]));
    });
    fwd.submit(pipe)?;
    last.take_host::<i32>().await?;
    Ok(())
}
