//! **Pipelined ① copy of `isolatedtopp`** — the isolated single-TopP token
//! gate with SUBMIT-AHEAD depth, on the `inferlet::ptir` bridge. Where
//! `isolatedtopp`'s decode loop is synchronous (submit → take per fire), this
//! copy keeps `DEPTH` fires in flight: the loop-carried `tok_in`/`rng`
//! channels are device-side (each fire's epilogue puts the values the next
//! fire takes — no host round-trip), so the host submits fire `t+1` BEFORE
//! draining fire `t`'s token. The host-facing `out` ring is widened to `DEPTH`
//! cells to absorb the run-ahead (the `runahead` idiom).
//!
//! A COPY, not a rewrite-in-place: the original `isolatedtopp` is a
//! load-bearing token-identity baseline for the #12 gate — this validates the
//! SAME TopP mask + Gumbel-max epilogue runs PIPELINED without touching that
//! baseline.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const TEMPERATURE: f32 = 0.8;
const TOP_P: f32 = 0.9;
const MAX_TOKENS: usize = 4;
/// Submit-ahead window: fire `t+DEPTH-1` is submitted before fire `t`'s token
/// is drained. Also the host-read `out` ring capacity.
const DEPTH: usize = 2;

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

    // ──────────────── 1. PREFILL FIRE (N-wide, awaited) ─────────────────────
    // Identical to `isolatedtopp`: the prefill→decode handoff rides the host.
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

    // ONE pipeline for the whole prefill→decode stream (R4-4): the run-ahead
    // decode fires below are submitted on this same pipeline (MAX_TOKENS > 1,
    // so finish() (F7) lands after the last decode submit, not here).
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

    // ──────────────── 2. PIPELINED DECODE (1-wide, depth-DEPTH) ─────────────
    if got.len() < MAX_TOKENS {
        let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
        let rng = Channel::from(vec![0x51ed_u32, 0]).named("rng");
        // Host-read ring widened to the submit-ahead window.
        let out = Channel::new([1], dtype::i32)
            .capacity(DEPTH as u32)
            .named("out");
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

        // Prime + fill: launch up to DEPTH chain-linked fires upfront (none
        // awaited); finish() right after the last budget submit (F7 — the
        // budget is fixed, so the last submit is knowable at submit time);
        // then FIFO drain + refill one fire per drained token.
        let budget = MAX_TOKENS - 1;
        let mut submitted = 0usize;
        let mut inflight = 0usize;
        while inflight < DEPTH && submitted < budget {
            fwd.submit(&pipe)
                .map_err(|e| format!("decode submit @{submitted}: {e}"))?;
            submitted += 1;
            inflight += 1;
        }
        if submitted == budget {
            pipe.finish();
        }
        while inflight > 0 {
            let t = out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("out.take @{}: {e}", got.len()))?;
            inflight -= 1;
            let Some(&t0) = t.first() else {
                return Err(format!("out.take @{}: empty tensor", got.len()));
            };
            got.push(t0 as u32);
            if submitted < budget {
                fwd.submit(&pipe)
                    .map_err(|e| format!("decode submit @{submitted}: {e}"))?;
                submitted += 1;
                inflight += 1;
                if submitted == budget {
                    pipe.finish();
                }
            }
        }
    }

    eprintln!("[ISOLATED_TOPP_PIPELINED] tokens: {got:?}");
    Ok(format!("{{\"tokens\": {got:?}}}"))
}
