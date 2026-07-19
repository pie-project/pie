//! Chat completion with top-p and temperature sampling.
//!
//! The prompt is prefilled in one forward pass. A device-carried decode loop
//! then applies temperature scaling, nucleus filtering, and Gumbel-max sampling
//! in the PTIR epilogue.
//!
//! The host reads each sampled token for streaming detokenization and stop-token
//! handling; the sampled token itself remains device-carried into the next pass.

use inferlet::ptir::Taken;
use inferlet::ptir::prelude::*;
use inferlet::{Result, chat, model as wit_model};
use serde::Deserialize;

const PAGE_T: u32 = 16; // tokens per pool page
// Qwen3-0.6B

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

/// In-graph top-p + temperature sampler over the read-out row logits `[1,vocab]`.
/// `r` is the taken `[2]` u32 rng state (`[key, ctr]`) driving the Gumbel noise.
/// Returns the sampled token `[1]` i32. Zero temperature is exact greedy
/// decoding; positive temperatures use nucleus sampling.
fn sample_token(r: &Taken, temperature: f32, top_p: f32, vocab: u32) -> Tensor {
    let logits = intrinsics::logits(); // [1, vocab] f32 (read-out row)
    if temperature == 0.0 {
        return cast(&reduce_argmax(&logits), dtype::i32);
    }
    let scaled = div(&logits, temperature.max(1e-4));
    let _ = vocab;
    nucleus_sample(&scaled, top_p, r)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if !(0.0..=1.0).contains(&input.top_p) || input.top_p == 0.0 {
        return Err("top_p must be greater than 0 and at most 1".into());
    }
    if !input.temperature.is_finite() {
        return Err("temperature must be finite".into());
    }

    let vocab = wit_model::output_vocab_size();

    let temperature = input.temperature;
    let top_p = input.top_p;
    let max_tokens = input.max_tokens;
    if max_tokens == 0 {
        return Ok(String::new());
    }

    // Chat-template the system and user messages, then add the assistant cue.
    let mut prompt_tokens = chat::system_user(&input.system, &input.prompt);
    prompt_tokens.extend(chat::cue());
    if prompt_tokens.is_empty() {
        prompt_tokens.push(0);
    }
    let n = prompt_tokens.len() as u32;
    let stop = chat::stop_tokens();

    // Shared logical page pool: prompt + decode headroom, page-rounded.
    let pool_pages = (n + max_tokens as u32 + 2).div_ceil(PAGE_T);
    let pool = pool_pages * PAGE_T;

    let ws = WorkingSet::new();
    let slots = ws
        .reserve(pool_pages)
        .map_err(|e| format!("ws.reserve: {e}"))?;
    let pool_ids = slots.ids().to_vec();

    // ── ONE PIPELINE (R4-4): prefill and decode are one sequential stream.
    // The host g0 round-trip below stays — its take seeds the decode channels.
    let pipe = Pipeline::new();

    // ───────────────────────── 1. PREFILL FIRE (N-wide) ─────────────────────
    let g0 = {
        let prompt_i32: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
        let toks_p = Channel::from(prompt_i32).named("toks_p"); // [N] i32 (seeded)
        let embed_indptr_p = Channel::from(vec![0u32, n]).named("embed_indptr_p");
        let positions_p = Channel::from((0..n).collect::<Vec<_>>()).named("positions_p");

        // Explicit N-cell write descriptor: cell c → pool_ids[c/PAGE_T] @ c%PAGE_T.
        let w_slot_pv: Vec<u32> = (0..n).map(|c| pool_ids[(c / PAGE_T) as usize]).collect();
        let w_off_pv: Vec<u32> = (0..n).map(|c| c % PAGE_T).collect();
        let w_slot_p = Channel::from(w_slot_pv).named("w_slot_p");
        let w_off_p = Channel::from(w_off_pv).named("w_off_p");
        let klen_p = Channel::from(vec![n; 1]).named("klen_p");
        let pages_p = Channel::from(pool_ids.clone()).named("pages_p");
        let page_indptr_p = Channel::from_shaped([2], vec![0u32, pool_pages]).named("pidx_p");

        // Causal prefill mask [N, POOL]: query row i attends KV cols j <= i.
        let mask_pv: Vec<bool> = (0..n)
            .flat_map(|i| (0..pool).map(move |j| j <= i))
            .collect();
        let mask_p = Channel::from_shaped([n, pool], mask_pv).named("mask_p");
        let rng_p = Channel::from(vec![0x51ed_u32, 0]).named("rng_p");
        let g0_ch = Channel::new([1], dtype::i32).named("g0");

        let fwd_p = ForwardPass::new();
        fwd_p.embed(&toks_p, &embed_indptr_p)?;
        fwd_p.attention(
            &ws,
            ..,
            ..,
            &klen_p,
            &pages_p,
            &page_indptr_p,
            &w_slot_p,
            &w_off_p,
            &positions_p,
            Some(&mask_p),
        )?;
        fwd_p.epilogue(move || {
            let r = rng_p.take();
            let tok = sample_token(&r, temperature, top_p, vocab);
            let r_next = add(&r, iota(2)); // advance ctr: [key, ctr+1]
            g0_ch.put(&tok);
            rng_p.put(&r_next);
        });

        fwd_p
            .submit(&pipe)
            .map_err(|e| format!("prefill submit: {e}"))?;
        // max_tokens == 1: the prefill spends the whole budget, so it was
        // the stream's last submit — finish() right after it (F7).
        g0_ch
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("g0 take: {e}"))?[0]
    };

    let mut chat_dec = chat::Decoder::new();
    let mut text = String::new();
    let mut done = stop.contains(&(g0 as u32));
    if !stop.contains(&(g0 as u32)) {
        match chat_dec.feed(&[g0 as u32])? {
            chat::Event::Delta(s) => {
                print!("{s}");
                text.push_str(&s);
            }
            chat::Event::Done(s) => {
                text = s;
                done = true;
            }
            _ => {}
        }
    }

    // ───────────────────────── 2. DECODE LOOP (1-wide) ──────────────────────
    let slot_n = pool_ids[(n / PAGE_T) as usize];
    let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
    let pos = Channel::from(vec![n; 1]).named("pos");
    let fill = Channel::from(vec![n + 1; 1]).named("fill");
    let klen = Channel::from(vec![n + 1; 1]).named("klen");
    let w_slot = Channel::from(vec![slot_n; 1]).named("w_slot");
    let w_off = Channel::from(vec![n % PAGE_T; 1]).named("w_off");
    let seed_mask: Vec<bool> = (0..pool).map(|j| j <= n).collect();
    let mask = Channel::from_shaped([1, pool], seed_mask).named("mask");
    let pages = Channel::from(pool_ids.clone()).named("pages");
    let page_indptr = Channel::from_shaped([2], vec![0u32, pool_pages]).named("page_indptr");
    let pool_ids_ch = Channel::from(pool_ids.clone()).named("pool_ids");
    let out = Channel::new([1], dtype::i32)
        .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
        .named("out");
    let rng = Channel::from(vec![0x9e37_u32, 0]).named("rng");
    let lane1 = Channel::from(vec![0u32, 1u32]).named("embed_indptr");

    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &lane1)?;
    fwd.attention(
        &ws,
        ..,
        (n / ws.page_size())..,
        &klen,
        &pages,
        &page_indptr,
        &w_slot,
        &w_off,
        &pos,
        Some(&mask),
    )?;
    fwd.epilogue(move || {
        // TAKES + compute first, PUTS last (value-id discipline).
        let base = fill.take().tensor(); // [1] u32 — position this fire writes
        let pids = pool_ids_ch.take().tensor();
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
        let pidx_v = mul(iota(2), pool_pages);

        tok_in.put(&tok);
        out.put(&tok);
        mask.take();
        mask.put(&new_mask);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);
        klen.take();
        klen.put(&klen_v);
        pos.put(&base);
        fill.put(&next_free);
        pages.take();
        pages.put(&pages_v);
        page_indptr.take();
        page_indptr.put(&pidx_v);
        rng.put(&r_next);
        pool_ids_ch.put(&pids);
    });

    let budget = if done {
        0
    } else {
        max_tokens.saturating_sub(1) // g0 already emitted
    };
    let mut submitted = 0usize;
    let mut in_flight = 0usize;
    while in_flight < DEFAULT_RUNAHEAD_DEPTH && submitted < budget {
        fwd.submit(&pipe)
            .map_err(|e| format!("decode submit: {e}"))?;
        submitted += 1;
        in_flight += 1;
    }
    while !done && in_flight > 0 {
        let t = out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("out.take: {e}"))?;
        in_flight -= 1;
        let token = *t.first().unwrap_or(&0) as u32;
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
        if submitted < budget {
            fwd.submit(&pipe)
                .map_err(|e| format!("decode submit: {e}"))?;
            submitted += 1;
            in_flight += 1;
        }
    }
    while in_flight > 0 {
        out.take()
            .get::<i32>()
            .await
            .map_err(|e| format!("drain run-ahead output: {e}"))?;
        in_flight -= 1;
    }
    // Every submitted fire was drained above, so close only releases the
    // scheduler wait-set and rejects future submissions.
    pipe.close();

    Ok(text)
}
