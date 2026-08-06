//! Classifier-free guidance for language models (Sanchez et al., 2306.17806).
//!
//! Two forward passes run the bound model over two independent KV states: the
//! conditional stream sees the full prompt, the unconditional stream sees only
//! the negative prompt (empty by default). The paper's Eqn 4 is applied to
//! **log-probabilities**, not raw logits — the two streams have different
//! partition functions, so subtracting raw logits would inject a meaningless
//! per-stream constant:
//!
//! ```text
//! log P_cfg(w) ∝ log P(w | uncond) + γ · [log P(w | cond) − log P(w | uncond)]
//! ```
//!
//! and renormalised. γ = 1 collapses to plain conditional decoding, which the
//! reported `guidance_shift` statistic must confirm exactly.
//!
//! ## Source
//!
//! Sanchez et al., *Stay on topic with Classifier-Free Guidance* —
//! <https://arxiv.org/abs/2306.17806> (Eq. 7).
//!
//! Faithfulness: **Exact (equivalent form)** — the paper writes the rule over
//! log-probabilities and this works in logits, which differ by the per-stream
//! `logsumexp` constant. That constant is uniform over the vocabulary, so it
//! shifts every entry of the blended vector by the same amount and cancels in
//! the softmax. See
//! `inference-time-algorithms/10-implementation-faithfulness-audit.md`.

use inferlet::chat;
use inferlet::ptir::attention::prelude::*;
use serde::Deserialize;

const PAGE_T: u32 = 16;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default)]
    negative_prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_guidance")]
    guidance: f32,
}

fn default_prompt() -> String {
    "Explain why the sky appears blue.".into()
}

fn default_max_tokens() -> usize {
    32
}

fn default_guidance() -> f32 {
    1.5
}

/// Eqn 4 of 2306.17806 in the log-domain, plus the two diagnostics that prove
/// the γ = 1 identity: `shift` is 1 when guidance moved the argmax, and `kl` is
/// KL(P_cfg ‖ P_cond), which is exactly 0 at γ = 1.
fn guided_pick(uncond_logits: &Tensor, gamma: f32) -> (Tensor, Tensor, Tensor) {
    let cond = log_softmax(intrinsics::logits());
    let uncond = log_softmax(uncond_logits);
    let guided = log_softmax(&uncond + (&cond - &uncond) * gamma);

    let token = reduce_argmax(&guided);
    let cond_token = reduce_argmax(&cond);
    let shift = cast(ne(&token, &cond_token), dtype::i32);
    let kl = reduce_sum(exp(&guided) * (&guided - &cond));
    (
        reshape(cast(&token, dtype::i32), [1]),
        reshape(shift, [1]),
        reshape(kl, [1]),
    )
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if !input.guidance.is_finite() || input.guidance < 0.0 {
        return Err("guidance must be finite and non-negative".into());
    }
    if input.max_tokens == 0 {
        return Ok(String::new());
    }

    let max_tokens =
        u32::try_from(input.max_tokens).map_err(|_| "max_tokens exceeds the u32 range")?;
    let vocab = model::output_vocab_size();
    let gamma = input.guidance;
    let stop_tokens = chat::stop_tokens();

    let mut cond_prompt = chat::system_user("You are a helpful assistant.", &input.prompt);
    cond_prompt.extend(chat::cue());
    if cond_prompt.is_empty() {
        cond_prompt.push(0);
    }
    // The unconditional stream drops the conditioning text. With no negative
    // prompt it keeps only the chat scaffolding, which is the paper's
    // "unconditional" ∅ context for an instruction-tuned model.
    let mut uncond_prompt =
        chat::system_user("You are a helpful assistant.", &input.negative_prompt);
    uncond_prompt.extend(chat::cue());
    if uncond_prompt.is_empty() {
        uncond_prompt.push(0);
    }

    let nc = u32::try_from(cond_prompt.len()).map_err(|_| "prompt is too long")?;
    let nu = u32::try_from(uncond_prompt.len()).map_err(|_| "negative prompt is too long")?;
    let cond_pages = (nc + max_tokens + 1).div_ceil(PAGE_T);
    let uncond_pages = (nu + max_tokens + 1).div_ceil(PAGE_T);

    // ---- unconditional stream -------------------------------------------
    let uncond_ws = WorkingSet::new();
    uncond_ws
        .reserve(uncond_pages)
        .context("reserve unconditional KV")?;
    let uncond_prompt_i32 = uncond_prompt
        .iter()
        .map(|&token| token as i32)
        .collect::<Vec<_>>();

    let u_prompt_ch = Channel::from(uncond_prompt_i32).named("uncond_prompt");
    let u_pre_indptr = Channel::from([0u32, nu]).named("uncond_prefill_embed_indptr");
    let u_pre_pos = Channel::from_iter(0..nu).named("uncond_prefill_positions");
    let u_pre_pages = Channel::from_iter(0..uncond_pages);
    let u_pre_page_indptr = Channel::from([0u32, nu.div_ceil(PAGE_T)]);
    let u_pre_slot = Channel::from_iter((0..nu).map(|p| p / PAGE_T));
    let u_pre_off = Channel::from_iter((0..nu).map(|p| p % PAGE_T));
    let u_pre_klen = Channel::from([nu]);
    let u_pre_out = Channel::new([vocab], dtype::f32).named("uncond_prefill_logits");

    let uncond_prefill = ForwardPass::new();
    uncond_prefill.embed(&u_prompt_ch, &u_pre_indptr)?;
    uncond_prefill.attention(
        &uncond_ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &u_pre_klen,
            pages: &u_pre_pages,
            page_indptr: &u_pre_page_indptr,
            w_slot: &u_pre_slot,
            w_off: &u_pre_off,
            positions: &u_pre_pos,
            mask: None,
        },
    )?;
    uncond_prefill.epilogue(move || {
        u_pre_out.put(intrinsics::logits());
    });

    // One pipeline for the whole inferlet: a KV WorkingSet claims the first
    // pipeline it fires on and never migrates, so prefill and decode for both
    // streams must share a single scope.
    let pipeline = Pipeline::new();
    uncond_prefill.submit(&pipeline).context("uncond prefill")?;
    let first_uncond_logits = u_pre_out.take_host::<Vec<f32>>().await?;

    // ---- conditional stream ---------------------------------------------
    let cond_ws = WorkingSet::new();
    cond_ws
        .reserve(cond_pages)
        .context("reserve conditional KV")?;
    let cond_prompt_i32 = cond_prompt
        .iter()
        .map(|&token| token as i32)
        .collect::<Vec<_>>();

    let c_prompt_ch = Channel::from(cond_prompt_i32).named("cond_prompt");
    let c_pre_indptr = Channel::from([0u32, nc]).named("cond_prefill_embed_indptr");
    let c_pre_pos = Channel::from_iter(0..nc).named("cond_prefill_positions");
    let c_pre_pages = Channel::from_iter(0..cond_pages);
    let c_pre_page_indptr = Channel::from([0u32, nc.div_ceil(PAGE_T)]);
    let c_pre_slot = Channel::from_iter((0..nc).map(|p| p / PAGE_T));
    let c_pre_off = Channel::from_iter((0..nc).map(|p| p % PAGE_T));
    let c_pre_klen = Channel::from([nc]);
    let c_pre_uncond = Channel::new([vocab], dtype::f32).named("cond_prefill_uncond");
    let first_out = Channel::new([1], dtype::i32).named("first_token");
    let first_shift = Channel::new([1], dtype::i32).named("first_shift");
    let first_kl = Channel::new([1], dtype::f32).named("first_kl");

    let cond_prefill = ForwardPass::new();
    cond_prefill.embed(&c_prompt_ch, &c_pre_indptr)?;
    cond_prefill.attention(
        &cond_ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: ..,
            kv_len: &c_pre_klen,
            pages: &c_pre_pages,
            page_indptr: &c_pre_page_indptr,
            w_slot: &c_pre_slot,
            w_off: &c_pre_off,
            positions: &c_pre_pos,
            mask: None,
        },
    )?;
    cond_prefill.epilogue(move || {
        let (token, shift, kl) = guided_pick(&c_pre_uncond.take(), gamma);
        first_out.put(&token);
        first_shift.put(&shift);
        first_kl.put(&kl);
    });

    c_pre_uncond.put(first_uncond_logits);
    cond_prefill.submit(&pipeline).context("cond prefill")?;
    let first = first_out.take_host::<i32>().await? as u32;
    let mut shifts = first_shift.take_host::<i32>().await? as u64;
    let mut kl_total = first_kl.take_host::<f32>().await? as f64;
    let mut scored = 1u64;

    let mut generated = Vec::with_capacity(input.max_tokens);
    if !stop_tokens.contains(&first) {
        generated.push(first);
    }
    if generated.len() >= input.max_tokens || stop_tokens.contains(&first) {
        pipeline.close();
        return report(&generated, gamma, shifts, scored, kl_total);
    }

    // ---- decode loop: uncond runs one step ahead, cond consumes it -------
    let u_token = Channel::new([1], dtype::i32).named("uncond_token");
    let u_embed_indptr = Channel::from([0u32, 1]).named("uncond_embed_indptr");
    let u_pos = Channel::from([nu]).named("uncond_position");
    let u_klen = Channel::from([nu + 1]).named("uncond_kv_len");
    let u_pages = Channel::from_iter(0..uncond_pages).named("uncond_pages");
    let u_page_indptr =
        Channel::from([0u32, (nu + 1).div_ceil(PAGE_T)]).named("uncond_page_indptr");
    let u_slot = Channel::from([nu / PAGE_T]).named("uncond_write_slot");
    let u_off = Channel::from([nu % PAGE_T]).named("uncond_write_offset");
    let u_logits_out = Channel::new([vocab], dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("uncond_logits");

    let uncond_decode = ForwardPass::new();
    uncond_decode.embed(&u_token, &u_embed_indptr)?;
    uncond_decode.attention(
        &uncond_ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: (nu / kv_page_size())..,
            kv_len: &u_klen,
            pages: &u_pages,
            page_indptr: &u_page_indptr,
            w_slot: &u_slot,
            w_off: &u_off,
            positions: &u_pos,
            mask: None,
        },
    )?;
    uncond_decode.epilogue(move || {
        let length = u_klen.take();
        let next_length = &length + 1u32;
        let page_count = next_length.div_ceil(PAGE_T);

        u_logits_out.put(intrinsics::logits());
        u_klen.put(&next_length);
        u_pos.put(&length);
        u_slot.put(&length / PAGE_T);
        u_off.put(&length % PAGE_T);
        u_page_indptr.put(indptr(1, &page_count));
    });

    let c_token = Channel::from([first as i32]).named("cond_token");
    let c_embed_indptr = Channel::from([0u32, 1]).named("cond_embed_indptr");
    let c_pos = Channel::from([nc]).named("cond_position");
    let c_klen = Channel::from([nc + 1]).named("cond_kv_len");
    let c_pages = Channel::from_iter(0..cond_pages).named("cond_pages");
    let c_page_indptr = Channel::from([0u32, (nc + 1).div_ceil(PAGE_T)]).named("cond_page_indptr");
    let c_slot = Channel::from([nc / PAGE_T]).named("cond_write_slot");
    let c_off = Channel::from([nc % PAGE_T]).named("cond_write_offset");
    let c_uncond = Channel::writer([vocab], dtype::f32).named("cond_uncond_logits");
    let c_token_out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("cond_token_out");
    let c_shift_out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("cond_shift_out");
    let c_kl_out = Channel::new([1], dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("cond_kl_out");

    let cond_decode = ForwardPass::new();
    cond_decode.embed(&c_token, &c_embed_indptr)?;
    cond_decode.attention(
        &cond_ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: (nc / kv_page_size())..,
            kv_len: &c_klen,
            pages: &c_pages,
            page_indptr: &c_page_indptr,
            w_slot: &c_slot,
            w_off: &c_off,
            positions: &c_pos,
            mask: None,
        },
    )?;
    cond_decode.epilogue(move || {
        let length = c_klen.take();
        let (token, shift, kl) = guided_pick(&c_uncond.take(), gamma);
        let next_length = &length + 1u32;
        let page_count = next_length.div_ceil(PAGE_T);

        c_token.put(&token);
        c_klen.put(&next_length);
        c_pos.put(&length);
        c_slot.put(&length / PAGE_T);
        c_off.put(&length % PAGE_T);
        c_page_indptr.put(indptr(1, &page_count));
        c_token_out.put(&token);
        c_shift_out.put(&shift);
        c_kl_out.put(&kl);
    });

    let budget = input.max_tokens.saturating_sub(generated.len());
    // Strictly sequential: the unconditional stream must see the token that the
    // guided distribution actually emitted, so its input is only known after the
    // conditional fire lands. No run-ahead is possible without speculating.
    let mut previous = first;
    for _ in 0..budget {
        u_token.put(vec![previous as i32]);
        uncond_decode.submit(&pipeline).context("uncond decode")?;
        let uncond_logits = u_logits_out.take_host::<Vec<f32>>().await?;

        c_uncond.put(uncond_logits);
        cond_decode.submit(&pipeline).context("cond decode")?;
        let token = c_token_out.take_host::<i32>().await? as u32;
        shifts += c_shift_out.take_host::<i32>().await? as u64;
        kl_total += c_kl_out.take_host::<f32>().await? as f64;
        scored += 1;

        if stop_tokens.contains(&token) {
            break;
        }
        generated.push(token);
        previous = token;
    }
    pipeline.close();

    report(&generated, gamma, shifts, scored, kl_total)
}

fn report(
    generated: &[u32],
    gamma: f32,
    shifts: u64,
    scored: u64,
    kl_total: f64,
) -> Result<String> {
    let text = model::decode(generated)?;
    let mean_kl = if scored == 0 {
        0.0
    } else {
        kl_total / scored as f64
    };
    let identity = if (gamma - 1.0).abs() < 1e-6 && (shifts > 0 || mean_kl > 1e-3) {
        " IDENTITY-VIOLATION"
    } else {
        ""
    };
    Ok(format!(
        "{text}\n\n[cfg] guidance={gamma:.2} steps={scored} guidance_shift={:.3} mean_kl={mean_kl:.4}{identity}",
        shifts as f64 / scored.max(1) as f64
    ))
}
