//! Mirostat v2 test inferlet (Phase 2, WS2) — **`inferlet::ptir` bridge
//! rewrite**. Demonstrates the **sequential late-bind** loop for a sampler
//! whose control value depends on the previous step's output. Each fire's
//! epilogue truncates to tokens with surprise `-log p ≤ μ` (a
//! `pivot_threshold(probs, prob_ge(exp(-μ)))` keep-mask), Gumbel-max samples
//! among the kept set, and publishes BOTH the sampled `token` and the observed
//! surprise `S = -log p(token)` (nats) — the eDSL-composed equivalent of the
//! deleted `sampling::program::mirostat*` shapes. The inferlet then runs the
//! mirostat v2 control update on the CPU — `μ ← μ − lr·(S − τ)` — and rebinds
//! the new μ as a **host-writer channel put** on the next fire (D2 instance
//! data, never a trace constant).
//!
//! Floor selection (keeps the keep-mask non-empty regardless of μ):
//!   - `"argmax"` (DEFAULT): OR the surprise mask with the max-prob token's own
//!     indicator (`ge(probs, broadcast(reduce_max(probs)))`) — proven-ops,
//!     never empty-keep.
//!   - `"rank"` with `k_min>0`: OR with a top-`k_min` rank mask
//!     (`pivot_threshold(logits, rank_le(k_min))`).
//!   - `k_min=0` (no floor override): the plain, degenerate control.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model, serde_json};

/// Default target surprise τ (nats); override via `_input` `"tau"`.
const TAU: f32 = 3.0;
/// Default control learning rate; override via `_input` `"lr"`.
const LR: f32 = 0.6;
/// Default number of tokens to generate; override via `_input` `"max_tokens"`.
const MAX_TOKENS: usize = 16;

#[derive(Clone, Copy)]
enum Floor {
    Argmax,
    Rank(u32),
    Plain,
}

fn json_f32(v: &serde_json::Value, key: &str, default: f32) -> f32 {
    v.get(key)
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or(default)
}
fn json_usize(v: &serde_json::Value, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(default)
}

/// The mirostat keep-mask + Gumbel-max sample + observed surprise, over the
/// FULL (untruncated) `probs`/`logits` distribution. `mu` is reshaped to a
/// scalar here — `pivot_threshold`'s predicate threshold operand is validated
/// as EXACTLY scalar (or a per-row `[rows]` vector for rank-2 input), not a
/// `[1]` vector, so a `[1]`-channel-sourced `mu` must be squeezed down. `r` is
/// the taken `[2]` u32 rng state (`[key, ctr]`) — `gumbel`'s `state` operand is
/// likewise validated as EXACTLY `[2]` u32. Returns `(token, surprise)`.
fn mirostat_step(
    floor: Floor,
    logits: Tensor,
    vocab: u32,
    mu: Tensor,
    r: impl AsTensor,
) -> (Tensor, Tensor) {
    let mu = reshape(mu, []); // [1] -> scalar
    let probs = softmax(&logits);
    let thr = exp(neg(&mu));
    let base_mask = pivot_threshold(&probs, prob_ge(thr));
    let mask = match floor {
        Floor::Argmax => {
            let argmax_ind = ge(&probs, broadcast(reduce_max(&probs), [vocab]));
            or(base_mask, argmax_ind)
        }
        Floor::Rank(k_min) if k_min > 0 => {
            let rank_mask = pivot_threshold(&logits, rank_le(k_min));
            or(base_mask, rank_mask)
        }
        _ => base_mask,
    };
    let neg_inf = broadcast(Tensor::constant(f32::NEG_INFINITY), [vocab]);
    let masked = select(&mask, &logits, &neg_inf);
    let g = gumbel(r, [vocab]);
    let token = reduce_argmax(add(masked, g)); // [1] i32
    let p_token = gather(&probs, cast(&token, DType::U32)); // [1] f32
    let surprise = neg(log(p_token)); // [1] f32
    (token, surprise)
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let params: serde_json::Value = serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let tau = json_f32(&params, "tau", TAU);
    let lr = json_f32(&params, "lr", LR);
    let max_tokens = json_usize(&params, "max_tokens", MAX_TOKENS);
    let k_min = json_usize(&params, "k_min", 8) as u32;
    let floor_name = params
        .get("floor")
        .and_then(|v| v.as_str())
        .unwrap_or("argmax");
    let floor = match floor_name {
        "argmax" => Floor::Argmax,
        _ if k_min > 0 => Floor::Rank(k_min),
        _ => Floor::Plain,
    };

    let vocab = wit_model::output_vocab_size();
    let ws = WorkingSet::new();
    model::configure(vocab, ws.page_size(), 1);

    let mu0_default = (vocab as f32).ln() + 1.0;
    let mut mu: f32 = json_f32(&params, "mu0", mu0_default);

    let mut prompt = wit_model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(ws.page_size());
    ws.reserve(max_pages)
        .map_err(|e| format!("ws.reserve: {e}"))?;

    let mut surprises: Vec<f32> = Vec::with_capacity(max_tokens);
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // ── PREFILL FIRE (N-wide) — first mirostat step over the prompt. ──
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let mu_p = Channel::new([1], dtype::f32).named("mu_p");
    let rng_p = Channel::from(vec![0x9e37_u32, 0]).named("rng_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");
    let s_out_p = Channel::new([1], dtype::f32).named("s_out_p");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, Tensor::constant(vec![0u32, n]));
    let kv_len_p = Channel::from(vec![n]).named("kv_len_p");
    fwd_p.port_channel(Port::KvLen, &kv_len_p);
    fwd_p.attn_working_set(&ws, .., ..)?;
    fwd_p.derive_dense_geometry();
    fwd_p.epilogue(move || {
        let mu_v = mu_p.take().tensor();
        let r = rng_p.take(); // [2] u32 rng state (key, ctr)
        let logits = intrinsics::logits(); // [vocab] f32 (single read-out row)
        let (token, surprise) = mirostat_step(floor, logits, vocab, mu_v, &r);
        let r_next = add(&r, iota(2));
        tok_out_p.put(&token);
        s_out_p.put(&surprise);
        rng_p.put(&r_next);
    });

    mu_p.put(vec![mu]);
    // ONE pipeline for the whole prefill→decode stream (R4-4): the decode
    // fires below are submitted on this same pipeline. The stream is finished
    // (F7) right after the prefill submit only in the degenerate case where
    // zero decode fires follow.
    let pipe = Pipeline::new();
    fwd_p
        .submit(&pipe)
        .map_err(|e| format!("prefill submit: {e}"))?;
    if max_tokens <= 1 {
        pipe.finish();
    }
    let g0 = tok_out_p
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("g0 take: {e}"))?[0];
    let s0 = s_out_p
        .take()
        .get::<f32>()
        .await
        .map_err(|e| format!("s0 take: {e}"))?[0];

    generated.push(g0 as u32);
    surprises.push(s0);
    mu -= lr * (s0 - tau);

    // ── DECODE LOOP (1-wide) ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
        let mu_ch = Channel::new([1], dtype::f32).named("mu_ch");
        let rng = Channel::from(vec![0x51ed_u32, 0]).named("rng");
        let tok_out = Channel::new([1], dtype::i32).named("tok_out");
        let s_out = Channel::new([1], dtype::f32).named("s_out");
        let lane1 = Tensor::constant(vec![0u32, 1u32]);

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, lane1);
        let kv_len = Channel::from(vec![n + 1]).named("kv_len");
        fwd.port_channel(Port::KvLen, &kv_len);
        fwd.attn_working_set(&ws, .., (n / ws.page_size())..)?;
        fwd.derive_dense_geometry();
        fwd.epilogue(move || {
            // Takes + compute first, PUTS last (value-id discipline).
            let length = kv_len.take().tensor();
            let mu_v = mu_ch.take().tensor(); // [1] f32, host-updated each step
            let r = rng.take(); // [2] u32 rng state
            let logits = intrinsics::logits(); // [vocab] f32
            let (token, surprise) = mirostat_step(floor, logits, vocab, mu_v, &r);

            let r_next = add(&r, iota(2));

            tok_in.put(&token);
            kv_len.put(add(&length, 1u32));
            tok_out.put(&token);
            s_out.put(&surprise);
            rng.put(&r_next);
        });

        for step in 1..max_tokens {
            mu_ch.put(vec![mu]);
            // Fixed budget (the host-updated mu changes the VALUES, not the
            // fire count): the last submit is knowable at submit time →
            // finish() right after it (F7: end of stream; no close needed
            // after the drain).
            fwd.submit(&pipe)
                .map_err(|e| format!("decode submit @{step}: {e}"))?;
            if step + 1 == max_tokens {
                pipe.finish();
            }
            let t = tok_out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("tok_out.take @{step}: {e}"))?[0];
            let s = s_out
                .take()
                .get::<f32>()
                .await
                .map_err(|e| format!("s_out.take @{step}: {e}"))?[0];
            generated.push(t as u32);
            surprises.push(s);
            mu -= lr * (s - tau);
        }
    }

    let mean_s = if surprises.is_empty() {
        f32::NAN
    } else {
        surprises.iter().sum::<f32>() / surprises.len() as f32
    };
    let tail_mean_s = if surprises.len() >= 2 {
        let tail = &surprises[surprises.len() / 2..];
        tail.iter().sum::<f32>() / tail.len() as f32
    } else {
        mean_s
    };
    let s_flowed = !surprises.is_empty();
    let tokens_json = serde_json::to_string(&generated).unwrap_or_else(|_| "[]".to_string());
    let result = format!(
        "{{\"sampler\":\"mirostat\",\"count\":{},\"tau\":{tau},\"final_mu\":{mu:.4},\
         \"mean_surprise\":{mean_s:.4},\"tail_mean_surprise\":{tail_mean_s:.4},\
         \"s_flowed\":{s_flowed},\"tokens\":{tokens_json}}}",
        generated.len(),
    );
    eprintln!("[MIROSTAT] {result}");
    Ok(result)
}
