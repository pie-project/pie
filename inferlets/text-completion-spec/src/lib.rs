//! Text completion that exercised the speculative-decoding interface — **PTIR
//! keep-core** rewrite.
//!
//! NOTE: the classic `Generator::system_speculation()` was a **no-op** in the
//! shipped code — it only forced the sequential (non-pipelined) decode path; no
//! `output-speculative-tokens` was ever emitted, so every step yielded exactly
//! one token (`avg_tokens_per_step == 1.0`). The PTIR surface has no
//! host-visible speculative-token output on the plain LM head either (device
//! MTP drafts are `intrinsics::mtp_logits` — the separate `mtp-specdecode`
//! inferlet family), so this rewrite preserves the shipped behavior exactly:
//! prompt prefill in ONE N-wide fire (which also samples generation token 1 —
//! the classic combined prefill+first-step fire), then a **sequential**
//! 1-token-per-step decode loop on the ptir `Pipeline` (each step's `out` read
//! is awaited before the next submit — deliberately not run-ahead, mirroring
//! the classic forced-sequential path). Sampling is in-graph: pure argmax when
//! `temperature <= 0` (the classic `SamplerSpec::Argmax`), else top-p +
//! temperature (`softmax → pivot_threshold(cummass_le) → gumbel-argmax`).
//!
//! Emits one structured line on stdout when generation ends so the test
//! harness can compare runs:
//!
//!     SPEC_STATS prompt_tokens=N generated_tokens=M elapsed_ms=T tokens_per_sec=R steps=S avg_tokens_per_step=A

use inferlet::ptir::prelude::*;
use inferlet::ptir::Taken;
use inferlet::{chat, model as wit_model, session, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

const PAGE_T: u32 = 16; // tokens per pool page
const NUM_LAYERS: u32 = 28; // Qwen3-0.6B

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_system")]
    system: String,
    /// Default 0.0 (greedy).
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    decode_output: bool,
    #[serde(default)]
    start_signal: bool,
    #[serde(default)]
    emit_stats: bool,
    #[serde(default)]
    compact_output: bool,
}

fn default_max_tokens() -> usize {
    128
}
fn default_system() -> String {
    "You are a helpful, respectful and honest assistant.".into()
}
fn default_temperature() -> f32 {
    0.0
}
fn default_top_p() -> f32 {
    1.0
}

#[derive(Serialize)]
struct Output {
    text: String,
    generated_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    elapsed_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_tokens_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    steps: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    avg_tokens_per_step: Option<f64>,
}

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// In-graph top-p + temperature sampler over the read-out logits. `r` is the
/// taken `[2]` u32 rng state driving the Gumbel noise.
fn sample_topp(r: &Taken, temperature: f32, top_p: f32, vocab: u32) -> Tensor {
    let logits = intrinsics::logits();
    let scaled = div(&logits, temperature.max(1e-4));
    let probs = softmax(&scaled);
    let keep = pivot_threshold(&probs, cummass_le(top_p));
    let masked = mask_apply(&scaled, &keep);
    let g = gumbel(r, [vocab]);
    reduce_argmax(add(&masked, &g)) // [1] i32
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    let vocab = wit_model::output_vocab_size();
    model::configure(vocab, PAGE_T, NUM_LAYERS);

    // temperature <= 0 → pure argmax (the classic `SamplerSpec::Argmax`).
    let greedy = input.temperature <= 0.0;
    let temperature = input.temperature;
    let top_p = input.top_p;

    let mut prompt = chat::system_user(&input.system, &input.prompt);
    prompt.extend(chat::cue());
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;

    if input.start_signal {
        session::send("ready");
        let _ = session::receive().await;
    }

    let mut all_tokens: Vec<u32> = Vec::with_capacity(input.max_tokens);
    let mut steps: usize = 0;
    let mut prefill_elapsed = Duration::ZERO;
    let mut decode_elapsed = Duration::ZERO;

    if input.max_tokens > 0 {
        // Shared physical page pool: prompt + decode headroom, page-rounded.
        let pool_pages = (n + input.max_tokens as u32 + 2 + PAGE_T - 1) / PAGE_T;
        let pool = pool_pages * PAGE_T;
        let ws: &'static WorkingSet = bx(WorkingSet::new());
        let slots = ws.reserve(pool_pages).map_err(|e| format!("ws.reserve: {e}"))?;
        let pool_ids: &'static Vec<u32> = bx(slots.ids().to_vec());

        // ──────────── 1. PREFILL FIRE (N-wide; = classic step 1) ────────────
        // One fire prefills the whole prompt AND samples generation token 1 —
        // the classic first `fire()` combined prefill + first decode step.
        let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        let toks_p = bx(Channel::from(prompt_i32).named("toks_p"));
        let embed_indptr_p = Tensor::constant(vec![0u32, n]);

        let w_slot_pv: Vec<u32> = (0..n).map(|c| pool_ids[(c / PAGE_T) as usize]).collect();
        let w_off_pv: Vec<u32> = (0..n).map(|c| c % PAGE_T).collect();
        let w_slot_p = bx(Channel::from(w_slot_pv).named("w_slot_p"));
        let w_off_p = bx(Channel::from(w_off_pv).named("w_off_p"));
        let klen_p = bx(Channel::from(vec![n; 1]).named("klen_p"));
        let pages_p = bx(Channel::from(pool_ids.clone()).named("pages_p"));
        let page_indptr_p = bx(Channel::from_shaped([2], vec![0u32, pool_pages]).named("pidx_p"));
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
            if greedy {
                let tok = reduce_argmax(intrinsics::logits()); // [1] i32
                g0_ch.put(&tok);
            } else {
                let r = rng_p.take();
                let tok = sample_topp(&r, temperature, top_p, vocab);
                let r_next = add(&r, iota(2));
                g0_ch.put(&tok);
                rng_p.put(&r_next);
            }
        });

        let prefill_start = Instant::now();
        let prefill = Pipeline::new();
        fwd_p.submit(&prefill).map_err(|e| format!("prefill submit: {e}"))?;
        let g0 = g0_ch.take().get::<i32>().map_err(|e| format!("g0 take: {e}"))?[0];
        prefill.close();
        prefill_elapsed = prefill_start.elapsed();
        all_tokens.push(g0 as u32);
        steps += 1;

        // ─────────── 2. SEQUENTIAL DECODE LOOP (1 token per step) ───────────
        // The shipped no-op-spec behavior: one token per fire, each `out` read
        // awaited before the next submit. No stop-token check (the classic loop
        // ran to `max_tokens` unconditionally).
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

            let (tok, r_next) = if greedy {
                (reduce_argmax(intrinsics::logits()), None)
            } else {
                let r = rng.take();
                let tok = sample_topp(&r, temperature, top_p, vocab);
                let r_next = add(&r, iota(2));
                (tok, Some(r_next))
            };

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
            if let Some(r_next) = r_next {
                rng.put(&r_next);
            }
        });

        let decode_start = Instant::now();
        let decode = Pipeline::new();
        while all_tokens.len() < input.max_tokens {
            pool_ids_ch.put(pool_ids.clone());
            fwd.submit(&decode).map_err(|e| format!("decode submit: {e}"))?;
            let t = out.take().get::<i32>().map_err(|e| format!("out.take: {e}"))?;
            all_tokens.push(*t.first().unwrap_or(&0) as u32);
            steps += 1;
        }
        decode.close();
        decode_elapsed = decode_start.elapsed();
    }

    let elapsed = prefill_elapsed + decode_elapsed;

    let text = if input.decode_output {
        wit_model::decode(&all_tokens)?
    } else {
        String::new()
    };

    let elapsed_ms = elapsed.as_millis();
    let prefill_ms = prefill_elapsed.as_millis();
    let decode_ms = decode_elapsed.as_millis();
    let secs = elapsed.as_secs_f64();
    let decode_secs = decode_elapsed.as_secs_f64();
    let tps = if secs > 0.0 {
        all_tokens.len() as f64 / secs
    } else {
        0.0
    };
    let decode_tokens = all_tokens.len().saturating_sub(1);
    let decode_tps = if decode_secs > 0.0 && decode_tokens > 0 {
        decode_tokens as f64 / decode_secs
    } else {
        0.0
    };
    let avg_per_step = if steps > 0 {
        all_tokens.len() as f64 / steps as f64
    } else {
        0.0
    };

    if input.emit_stats {
        println!(
            "SPEC_STATS prompt_tokens={} generated_tokens={} elapsed_ms={} \
             prefill_ms={} decode_ms={} \
             tokens_per_sec={:.2} decode_tokens_per_sec={:.2} \
             steps={} avg_tokens_per_step={:.3}",
            input.prompt.split_whitespace().count(),
            all_tokens.len(),
            elapsed_ms,
            prefill_ms,
            decode_ms,
            tps,
            decode_tps,
            steps,
            avg_per_step,
        );
    }

    Ok(Output {
        text,
        generated_tokens: all_tokens.len(),
        elapsed_ms: (!input.compact_output).then_some(elapsed_ms),
        prefill_ms: (!input.compact_output).then_some(prefill_ms),
        decode_ms: (!input.compact_output).then_some(decode_ms),
        tokens_per_sec: (!input.compact_output).then_some(tps),
        decode_tokens_per_sec: (!input.compact_output).then_some(decode_tps),
        steps: (!input.compact_output).then_some(steps),
        avg_tokens_per_step: (!input.compact_output).then_some(avg_per_step),
    })
}
