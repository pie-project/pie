//! Chat completion with top-p and temperature sampling.
//!
//! The prompt is prefilled in one forward pass. A device-carried decode loop
//! then applies temperature scaling, nucleus filtering, and Gumbel-max sampling
//! in the PTIR epilogue.
//!
//! The host reads each sampled token for streaming detokenization and stop-token
//! handling; the sampled token itself remains device-carried into the next pass.

use inferlet::chat;
use inferlet::ptir::attention::prelude::*;
use serde::Deserialize;

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
fn sample_token(r: &Tensor, temperature: f32, top_p: f32, vocab: u32) -> Tensor {
    let logits = intrinsics::logits(); // [1, vocab] f32 (read-out row)
    if temperature == 0.0 {
        return reduce_argmax(&logits);
    }
    let scaled = &logits / temperature.max(1e-4);
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

    let vocab = model::output_vocab_size();

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

    // Tokens per pool page. Taken from the driver, never assumed: this pass
    // builds its own write descriptors (`cell c -> pool_ids[c/page_t] @
    // c%page_t`) and its own page CSR, and the driver reads both back at ITS
    // page size. A constant here that disagreed would put the tokens somewhere
    // other than where the CSR says they are -- and with no mask on the wire
    // any more, nothing downstream would notice.
    let page_t = kv_page_size();

    // Shared logical page pool: prompt + decode headroom, page-rounded.
    let pool_pages = (n + max_tokens as u32 + 2).div_ceil(page_t);

    let ws = WorkingSet::new();
    let slots = ws.reserve(pool_pages).context("ws.reserve")?;
    let pool_ids = slots.ids().to_vec();

    // ── ONE PIPELINE (R4-4): prefill and decode are one sequential stream.
    // The host g0 round-trip below stays — its take seeds the decode channels.
    let pipe = Pipeline::new();

    // ───────────────────────── 1. PREFILL FIRE (N-wide) ─────────────────────
    let g0 = {
        let prompt_i32: Vec<i32> = prompt_tokens.iter().map(|&t| t as i32).collect();
        let toks_p = Channel::from(prompt_i32).named("toks_p"); // [N] i32 (seeded)
        let embed_indptr_p = Channel::from([0u32, n]).named("embed_indptr_p");
        let positions_p = Channel::from_iter(0..n).named("positions_p");

        // Explicit N-cell write descriptor: cell c → pool_ids[c/page_t] @ c%page_t.
        let w_slot_pv: Vec<u32> = (0..n).map(|c| pool_ids[(c / page_t) as usize]).collect();
        let w_off_pv: Vec<u32> = (0..n).map(|c| c % page_t).collect();
        let w_slot_p = Channel::from(w_slot_pv).named("w_slot_p");
        let w_off_p = Channel::from(w_off_pv).named("w_off_p");
        let klen_p = Channel::from([n]).named("klen_p");
        let pages_p = Channel::from(pool_ids.clone()).named("pages_p");
        // The page CSR is the SOURCE OF TRUTH for kv_len on the wire: the driver
        // derives `kv_len = (page_count-1)*page_t + last_page_len`. Declaring the
        // whole pool here would claim a kv length the pass does not have and
        // silently corrupt attention — the count must track `kv_len` exactly.
        let page_indptr_p = Channel::from([0u32, n.div_ceil(page_t)]).named("pidx_p");

        // No AttnMask port. A prefill query attends every key up to its own
        // position, which is the bound the driver already derives from
        // `qo_indptr` and the page CSR — so a causal mask here only restates
        // the CSR two lines up, in `[N, POOL]` bools, over the wire, every
        // fire. The decode fire below has always said this; the prefill was
        // paying for the same statement twice.
        //
        // It was not free, either. A driver that cannot decode the wire form
        // has to refuse the launch or find the dense buffer some other way,
        // and the mask's length is a second claim about `kv_len` that can
        // disagree with the CSR's — which is a silent wrong answer when the
        // two are computed at different page sizes.
        let rng_p = Channel::from([0x51ed_u32, 0]).named("rng_p");
        let g0_ch = Channel::new([1], dtype::i32).named("g0");

        let fwd_p = ForwardPass::new();
        fwd_p.embed(&toks_p, &embed_indptr_p)?;
        fwd_p.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &klen_p,
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
            let tok = sample_token(&r, temperature, top_p, vocab);
            let r_next = &r + iota(2); // advance ctr: [key, ctr+1]
            g0_ch.put(&tok);
            rng_p.put(&r_next);
        });

        fwd_p.submit(&pipe).context("prefill submit")?;
        // max_tokens == 1: the prefill spends the whole budget, so it was
        // the stream's last submit — finish() right after it (F7).
        g0_ch.take_host::<i32>().await?
    };

    let chat_dec = chat::Decoder::new();
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
    let slot_n = pool_ids[(n / page_t) as usize];
    let tok_in = Channel::from([g0]).named("tok_in");
    let pos = Channel::from([n]).named("pos");
    let fill = Channel::from([n + 1]).named("fill");
    let klen = Channel::from([n + 1]).named("klen");
    let w_slot = Channel::from([slot_n]).named("w_slot");
    let w_off = Channel::from([n % page_t]).named("w_off");
    // No AttnMask port: a decode query attends every key up to `klen`, which
    // this pass already carries on device, so a causal mask would only restate
    // it. Binding one would also cost the whole run-ahead: the mask port is a
    // per-lane dense buffer, so a pass that binds it needs the driver to carry
    // per-lane mask state through batch composition, which the CUDA backend
    // does not advertise — the pass would fall out of the decode-envelope class
    // and lose the only descriptor resolver that works inside one frame.
    let pages = Channel::from(pool_ids.clone()).named("pages");
    let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_t)]).named("page_indptr");
    let pool_ids_ch = Channel::from(pool_ids.clone()).named("pool_ids");
    let out = Channel::new([1], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("out");
    let rng = Channel::from([0x9e37_u32, 0]).named("rng");
    let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");

    let fwd = ForwardPass::new();
    fwd.embed(&tok_in, &lane1)?;
    fwd.attention(
        &ws,
        KvGeometry {
            readable_pages: ..,
            writable_pages: (n / page_t)..,
            kv_len: &klen,
            pages: &pages,
            page_indptr: &page_indptr,
            w_slot: &w_slot,
            w_off: &w_off,
            positions: &pos,
            mask: None,
        },
    )?;
    fwd.epilogue(move || {
        // TAKES + compute first, PUTS last (value-id discipline).
        let base = fill.take(); // [1] u32 — position this fire writes
        let pids = pool_ids_ch.take();
        let r = rng.take();

        let tok = sample_token(&r, temperature, top_p, vocab); // [1] i32
        let r_next = &r + iota(2);

        let logical_slot = &base / page_t;
        let w_slot_v = gather(&pids, &logical_slot);
        let w_off_v = &base % page_t;
        let klen_v = &base + 1u32;
        let next_free = &base + 1u32;
        let pages_v = reshape(&pids, [pool_pages]);
        // Page count tracks the new kv length, never the pool size.
        let page_count = klen_v.div_ceil(page_t);
        let pidx_v = indptr(1, &page_count);

        // Device-resolved geometry is loop-carried: the host never drains
        // these rings, so every fire's values are re-put here.
        tok_in.put(&tok);
        out.put(&tok);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);
        klen.put(&klen_v);
        pos.put(&base);
        fill.put(&next_free);
        pages.put(&pages_v);
        page_indptr.put(&pidx_v);
        rng.put(&r_next);
        pool_ids_ch.put(&pids);
    });

    let budget = if done {
        0
    } else {
        max_tokens.saturating_sub(1) // g0 already emitted
    };
    run_ahead(&pipe, &fwd, budget, async || {
        let t = out.take_host::<Vec<i32>>().await?;
        let token = *t.first().unwrap_or(&0) as u32;
        if stop.contains(&token) {
            return Ok(ControlFlow::Break(()));
        }
        match chat_dec.feed(&[token])? {
            chat::Event::Delta(s) => {
                print!("{s}");
                text.push_str(&s);
            }
            chat::Event::Done(s) => {
                text = s;
                return Ok(ControlFlow::Break(()));
            }
            _ => {}
        }
        Ok(ControlFlow::Continue(()))
    })
    .await?;
    // Any fire still in flight after an early stop is left untaken; close
    // releases the scheduler wait-set, reclaims them, and rejects further
    // submissions.
    pipe.close();

    Ok(text)
}
