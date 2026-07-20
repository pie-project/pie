//! Isolated top-p inferlet — a SINGLE TopP fire chain on a fresh context.
//! **`inferlet::ptir` bridge rewrite** (bravo).
//!
//! Phase-1 #12 token gate (un-confounded): the `multisamp` harness shares one
//! context across 4 kinds, so top-p fires after top-k's tokens. This inferlet
//! fires top-p ALONE on bare `"hello world"`, so its tokens depend only on the
//! prompt — a clean token-identity check: recognize TopP → extract(T=0.8,
//! p=0.9) → device Gumbel-max sample. The mask is built directly from the eDSL
//! primitives (`softmax` + `pivot_threshold(probs, cummass_le(p))`), the
//! author-facing equivalent of the deleted `sampler::sampler_program`
//! `SamplerSpec::TopP` shape.
//!
//! Run with `PIE_FIXED_SAMPLING_SEED=12345` for reproducibility (ambient
//! seed), and `PIE_SAMPLING_IR_TRACE=1` to confirm the FlashInfer dispatch.
//!
//! RNG state mirrors `text-completion`'s `sample_token`: `gumbel`'s `state`
//! operand is validated as an EXACT `[2]` u32 `[key, ctr]` pair (not a
//! scalar/`[1]` value) — a `[2]` channel is taken each fire and the ctr lane
//! advanced (`add(r, iota(2))`) and put back for the next fire.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const TEMPERATURE: f32 = 0.8;
const TOP_P: f32 = 0.9;
const MAX_TOKENS: usize = 4;

/// TopP mask + Gumbel-max sample over `logits` (already temperature-scaled):
/// keep = pivot_threshold(softmax(logits), cummass_le(p)); sample =
/// argmax(select(keep, logits, -inf) + gumbel(r)). `r` is the taken `[2]` u32
/// rng state (`[key, ctr]`).
fn topp_sample(logits: Tensor, vocab: u32, r: impl AsTensor) -> Tensor {
    let probs = softmax(&logits);
    let keep = pivot_threshold(probs, cummass_le(TOP_P));
    let neg_inf = broadcast(Tensor::constant(f32::NEG_INFINITY), [vocab]);
    let masked = select(&keep, &logits, &neg_inf);
    let g = gumbel(r, [vocab]);
    reduce_argmax(add(masked, g))
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    let ws = WorkingSet::new();
    model::configure(vocab, ws.page_size(), 1);

    let prompt = wit_model::encode("hello world");
    let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let n = prompt.len() as u32;
    let max_pages = (n + MAX_TOKENS as u32 + 1).div_ceil(ws.page_size());
    ws.reserve(max_pages)
        .map_err(|e| format!("ws.reserve: {e}"))?;

    // ───────────────────────── 1. PREFILL FIRE (N-wide) ─────────────────────
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let rng_p = Channel::from(vec![0x9e37_u32, 0]).named("rng_p");
    let g0_ch = Channel::new([1], dtype::i32).named("g0");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, Tensor::constant(vec![0u32, n]));
    let kv_len_p = Channel::from(vec![n]).named("kv_len_p");
    fwd_p.port_channel(Port::KvLen, &kv_len_p);
    fwd_p.attn_working_set(&ws, .., ..)?;
    fwd_p.derive_dense_geometry();
    fwd_p.epilogue(move || {
        let r = rng_p.take(); // [2] u32 rng state (key, ctr)
        let scaled = div(intrinsics::logits(), TEMPERATURE);
        let t = topp_sample(scaled, vocab, &r);
        let r_next = add(&r, iota(2));
        g0_ch.put(&t);
        rng_p.put(&r_next);
    });

    // ONE pipeline for the whole prefill→decode stream (R4-4): the decode
    // fires below are submitted on this same pipeline (MAX_TOKENS > 1, so
    // finish() (F7) lands after the last decode submit, not here).
    let pipe = Pipeline::new();
    fwd_p
        .submit(&pipe)
        .map_err(|e| format!("prefill submit: {e}"))?;
    let g0 = g0_ch
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("g0 take: {e}"))?[0];

    let mut got: Vec<u32> = Vec::with_capacity(MAX_TOKENS);
    got.push(g0 as u32);

    // ───────────────────────── 2. DECODE LOOP (1-wide) ──────────────────────
    if got.len() < MAX_TOKENS {
        let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
        let rng = Channel::from(vec![0x51ed_u32, 0]).named("rng");
        let out = Channel::new([1], dtype::i32).named("out");
        let lane1 = Tensor::constant(vec![0u32, 1u32]);

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, lane1);
        let kv_len = Channel::from(vec![n + 1]).named("kv_len");
        fwd.port_channel(Port::KvLen, &kv_len);
        fwd.attn_working_set(&ws, .., (n / ws.page_size())..)?;
        fwd.derive_dense_geometry();
        fwd.epilogue(move || {
            let length = kv_len.take().tensor();
            let r = rng.take(); // [2] u32 rng state
            let scaled = div(intrinsics::logits(), TEMPERATURE);
            let t = topp_sample(scaled, vocab, &r);

            let r_next = add(&r, iota(2));

            tok_in.put(&t);
            kv_len.put(add(&length, 1u32));
            out.put(&t);
            rng.put(&r_next);
        });

        for step in 1..MAX_TOKENS {
            // Fixed budget: the last submit is knowable at submit time →
            // finish() right after it (F7: end of stream; no close needed
            // after the drain).
            fwd.submit(&pipe)
                .map_err(|e| format!("decode submit @{step}: {e}"))?;
            if step + 1 == MAX_TOKENS {
                pipe.finish();
            }
            let t = out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("out.take @{step}: {e}"))?;
            let Some(&t0) = t.first() else {
                return Err(format!("out.take @{step}: empty tensor"));
            };
            got.push(t0 as u32);
        }
    }

    eprintln!("[ISOLATED_TOPP] tokens: {got:?}");
    Ok(format!("{{\"tokens\": {got:?}}}"))
}
