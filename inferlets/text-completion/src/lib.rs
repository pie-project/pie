//! Simple text completion inferlet — **PTIR keep-core** rewrite.
//!
//! Migrated off the classic `forward-pass` (the sampling-gated
//! `carrier::submit_pass` machinery) onto the PTIR `Pipeline` surface
//! (`inferlet::ptir`). Chat generation = **prompt prefill in ONE N-wide fire**
//! (multi-query custom-mask pack + N-cell KV write — the driver capability
//! landed in `cuda_ptir_prefill_e2e`) followed by a **device-carried decode
//! loop** (each fire embeds the prior fire's sampled token via the loop-carried
//! `tok_in` channel; the epilogue evolves the geometry + mask in-graph — the
//! beam-designb / windowed-attention wire form). Sampling is a fully in-graph
//! **top-p + temperature** program (`softmax → pivot_threshold(cummass_le) →
//! gumbel-argmax`), device-codegen'd by the ptir dispatch.
//!
//! The decode LOOP + stop logic stay hand-written and visible here (In Gim's
//! SDK-minimize directive): the host reads each step's sampled token off the
//! `out` channel, feeds the chat detokenizer, and stops on a stop-token or the
//! `max_tokens` budget. Kept thin bindings: `chat` (templating + streaming
//! detok), `model` (tokenizer/vocab).
//!
//! NOTE (run-ahead): the classic path speculated the next fire before the
//! producer's token was known (depth-1 EOS rollback). This PTIR rewrite carries
//! the token DEVICE-side (the `tok_in` epilogue put — no host round-trip for the
//! token itself) but drives submission sequentially (read `out` each step to
//! apply the stop set). Host-side depth-1 speculation over the ptir FIFO is a
//! latency follow-up; correctness/coherence are unaffected.

use inferlet::ptir::prelude::*;
use inferlet::ptir::Taken;
use inferlet::{chat, model as wit_model, Result};
use serde::Deserialize;

const PAGE_T: u32 = 16; // tokens per pool page
const NUM_LAYERS: u32 = 28; // Qwen3-0.6B

#[derive(Deserialize)]
struct Input {
    /// The user prompt to complete.
    prompt: String,
    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// System message for the assistant.
    #[serde(default = "default_system")]
    system: String,
    /// Sampling temperature.
    #[serde(default = "default_temperature")]
    temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    #[serde(default = "default_top_p")]
    top_p: f32,
}

fn default_max_tokens() -> usize {
    256
}
fn default_system() -> String {
    "You are a helpful, respectful and honest assistant.".into()
}
fn default_temperature() -> f32 {
    0.6
}
fn default_top_p() -> f32 {
    0.95
}

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// In-graph top-p + temperature sampler over the read-out row logits `[1,vocab]`.
/// `r` is the taken `[2]` u32 rng state (`[key, ctr]`) driving the Gumbel noise.
/// Returns the sampled token `[1]` i32. Temperature is clamped to a small epsilon
/// so a `0` request degrades to near-greedy rather than dividing by zero.
fn sample_token(r: &Taken, temperature: f32, top_p: f32, vocab: u32) -> Tensor {
    let logits = intrinsics::logits(); // [1, vocab] f32 (read-out row)
    let scaled = div(&logits, temperature.max(1e-4));
    let probs = softmax(&scaled); // [1, vocab]
    let keep = pivot_threshold(&probs, cummass_le(top_p)); // nucleus keep-mask
    let masked = mask_apply(&scaled, &keep); // -inf outside the nucleus
    let g = gumbel(r, [vocab]); // Gumbel noise, same [vocab] shape as logits
    reduce_argmax(add(&masked, &g)) // [1] i32 = argmax over perturbed logits
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);

    let temperature = input.temperature;
    let top_p = input.top_p;
    let max_tokens = input.max_tokens;
    if max_tokens == 0 {
        return Ok(String::new());
    }

    // Chat-templated prompt (kept thin bindings). A deferred system folds into
    // the first user turn via `system_user` (mirrors the classic path).
    let mut prompt_tokens = chat::system_user(&input.system, &input.prompt);
    prompt_tokens.extend(chat::cue());
    if prompt_tokens.is_empty() {
        prompt_tokens.push(0);
    }
    let n = prompt_tokens.len() as u32;
    let stop = chat::stop_tokens();

    // Shared physical page pool: prompt + decode headroom, page-rounded.
    let pool_pages = ((n + max_tokens as u32 + 2) + PAGE_T - 1) / PAGE_T;
    let pool = pool_pages * PAGE_T;

    let ws: &'static WorkingSet = bx(WorkingSet::new());
    let slots = ws.reserve(pool_pages).map_err(|e| format!("ws.reserve: {e}"))?;
    let pool_ids: &'static Vec<u32> = bx(slots.ids().to_vec()); // physical page ids

    // ───────────────────────── 1. PREFILL FIRE (N-wide) ─────────────────────
    let prompt_i32: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
    let toks_p = bx(Channel::from(prompt_i32).named("toks_p")); // [N] i32 (seeded)
    let embed_indptr_p = Tensor::constant(vec![0u32, n]); // qo_indptr [0,N]

    // Explicit N-cell write descriptor: cell c → pool_ids[c/PAGE_T] @ c%PAGE_T.
    let w_slot_pv: Vec<u32> = (0..n).map(|c| pool_ids[(c / PAGE_T) as usize]).collect();
    let w_off_pv: Vec<u32> = (0..n).map(|c| c % PAGE_T).collect();
    let w_slot_p = bx(Channel::from(w_slot_pv).named("w_slot_p"));
    let w_off_p = bx(Channel::from(w_off_pv).named("w_off_p"));
    let klen_p = bx(Channel::from(vec![n; 1]).named("klen_p"));
    let pages_p = bx(Channel::from(pool_ids.clone()).named("pages_p"));
    let page_indptr_p = bx(Channel::from_shaped([2], vec![0u32, pool_pages]).named("pidx_p"));

    // Causal prefill mask [N, POOL]: query row i attends KV cols j <= i.
    let mask_pv: Vec<bool> = (0..n).flat_map(|i| (0..pool).map(move |j| j <= i)).collect();
    let mask_p = bx(Channel::from_shaped([n, pool], mask_pv).named("mask_p"));
    let rng_p = bx(Channel::from(vec![0x51ed_u32, 0]).named("rng_p"));
    let g0_ch = bx(Channel::new([1], dtype::i32).named("g0"));

    let fwd_p: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd_p.embed(toks_p, embed_indptr_p);
    fwd_p.attn_working_set(ws, klen_p);
    fwd_p.port_channel(Port::Pages, pages_p);
    fwd_p.port_channel(Port::PageIndptr, page_indptr_p);
    fwd_p.port_channel(Port::WSlot, w_slot_p);
    fwd_p.port_channel(Port::WOff, w_off_p);
    fwd_p.attn_mask(mask_p);
    fwd_p.epilogue(move || {
        let r = rng_p.take();
        let tok = sample_token(&r, temperature, top_p, vocab);
        let r_next = add(&r, iota(2)); // advance ctr: [key, ctr+1]
        g0_ch.put(&tok);
        rng_p.put(&r_next);
    });

    let prefill = Pipeline::new();
    fwd_p.submit(&prefill).map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = g0_ch.take().get::<i32>().map_err(|e| format!("g0 take: {e}"))?[0];
    prefill.close();

    let mut chat_dec = chat::Decoder::new();
    let mut text = String::new();
    let mut done = false;
    if !stop.contains(&(g0 as u32)) {
        if let chat::Event::Delta(s) | chat::Event::Done(s) = chat_dec.feed(&[g0 as u32])? {
            print!("{s}");
            text.push_str(&s);
        }
    } else {
        done = true;
    }

    // ───────────────────────── 2. DECODE LOOP (1-wide) ──────────────────────
    let phys_n = pool_ids[(n / PAGE_T) as usize];
    let tok_in = bx(Channel::from(vec![g0; 1]).named("tok_in"));
    let pos = bx(Channel::from(vec![n; 1]).named("pos"));
    let fill = bx(Channel::from(vec![n + 1; 1]).named("fill"));
    let klen = bx(Channel::from(vec![n + 1; 1]).named("klen"));
    let w_slot = bx(Channel::from(vec![phys_n; 1]).named("w_slot"));
    let w_off = bx(Channel::from(vec![n % PAGE_T; 1]).named("w_off"));
    let seed_mask: Vec<bool> = (0..pool).map(|j| j <= n).collect();
    let mask = bx(Channel::from_shaped([1, pool], seed_mask).named("mask"));
    let pages = bx(Channel::from(pool_ids.clone()).named("pages"));
    let page_indptr = bx(Channel::from_shaped([2], vec![0u32, pool_pages]).named("page_indptr"));
    let pool_ids_ch = bx(Channel::new([pool_pages], dtype::u32).named("pool_ids"));
    let out = bx(Channel::new([1], dtype::i32).named("out"));
    let rng = bx(Channel::from(vec![0x9e37_u32, 0]).named("rng"));
    let lane1 = Tensor::constant(vec![0u32, 1u32]);

    let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
    fwd.embed(tok_in, lane1);
    fwd.positions(pos);
    fwd.attn_working_set(ws, klen);
    fwd.port_channel(Port::Pages, pages);
    fwd.port_channel(Port::PageIndptr, page_indptr);
    fwd.port_channel(Port::WSlot, w_slot);
    fwd.port_channel(Port::WOff, w_off);
    fwd.attn_mask(mask);
    fwd.epilogue(move || {
        // TAKES + compute first, PUTS last (value-id discipline).
        let base = fill.take().tensor(); // [1] u32 — position this fire writes
        let pids = pool_ids_ch.take();
        let r = rng.take();

        let tok = sample_token(&r, temperature, top_p, vocab); // [1] i32
        let r_next = add(&r, iota(2));

        // Full causal mask for the query at `base`: attend all j <= base.
        let col = iota(pool);
        let base_b = broadcast(reshape(&base, [1]), [pool]);
        let new_mask = reshape(le(&col, &base_b), [1, pool]);

        let logical_slot = div(&base, PAGE_T);
        let w_slot_v = gather(&pids, &logical_slot);
        let w_off_v = rem(&base, PAGE_T);
        let klen_v = add(&base, 1u32);
        let next_free = add(&base, 1u32);
        let pages_v = reshape(&pids, [pool_pages]);
        let pidx_v = mul(&iota(2), pool_pages);

        tok_in.put(&tok);
        out.put(&tok);
        mask.put(&new_mask);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);
        klen.put(&klen_v);
        pos.put(&base);
        fill.put(&next_free);
        pages.put(&pages_v);
        page_indptr.put(&pidx_v);
        rng.put(&r_next);
    });

    let decode = Pipeline::new();
    let mut generated = 1usize; // g0 already emitted
    while !done && generated < max_tokens {
        pool_ids_ch.put(pool_ids.clone());
        fwd.submit(&decode).map_err(|e| format!("decode submit: {e}"))?;
        let t = out.take().get::<i32>().map_err(|e| format!("out.take: {e}"))?;
        let token = *t.first().unwrap_or(&0) as u32;
        generated += 1;
        if stop.contains(&token) {
            break;
        }
        match chat_dec.feed(&[token])? {
            chat::Event::Delta(s) => {
                print!("{s}");
                text.push_str(&s);
            }
            chat::Event::Done(s) => {
                text = s;
                break;
            }
            _ => {}
        }
    }
    decode.close();

    Ok(text)
}
