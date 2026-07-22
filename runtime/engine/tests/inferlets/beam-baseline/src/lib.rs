//! Hand-rolled beam baseline — the M3-G2 ±10% reference, on the
//! `inferlet::ptir` bridge.
//!
//! This is the **same beam math the fused `beam` inferlet does on-device, but
//! host-orchestrated per step**: every beam runs a plain raw-logits forward
//! (the epilogue publishes the FULL `[vocab]` logits row to a host-reader
//! channel), the host reads the logits back, computes `score + log_softmax`,
//! selects the per-beam top-B children, merges the `B*B` candidates and keeps
//! the global top-B, then RE-FORKS each survivor from its parent working set
//! (`WorkingSet::fork` — the CoW branching primitive) and re-feeds its token.
//! No on-device reorder / freeze / designated-child / compact — every
//! selection + KV reorder is a host round-trip.
//!
//! The M3 acceptance harness measures THIS inferlet's wall time into a
//! standalone comparison point: the fused `beam` inferlet's wall time must be
//! within 10% of it. Because the fused form folds the whole reorder/select
//! epilogue into ONE device pass (no per-step logits read-back, no host top-k,
//! no re-fork), it should be strictly faster — this baseline is the honest
//! hand-rolled upper bound the fused path must beat-or-match.
//!
//! FIDELITY: this is a token-domain beam over the model's real logits (real
//! forwards, real CoW KV fork), host-selected. It is the throughput/latency
//! reference for G2, not the on-device beam-geometry correctness capstone.
//!
//! It is also the workspace's only exercise of `WorkingSet::fork` on the
//! bridge surface (the fused beam reorders inside one working set instead).

use inferlet::ptir::prelude::*;
use inferlet::serde_json;
use inferlet::{Result, model as wit_model};

/// Beam width.
const BEAM_WIDTH: usize = 2;
/// Default decode steps; override via `{"max_tokens":N}`.
const MAX_TOKENS: usize = 12;

/// The top-`k` `(log_softmax, token)` children of one beam's raw logits.
/// `log_softmax(x)[v] = x[v] - (m + ln Σ exp(x-m))` (max-shift stable).
fn top_k_children(logits: &[f32], k: usize) -> Vec<(f32, u32)> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&x| (x - m).exp()).sum();
    let lse = m + sum.ln();
    // Partial top-k selection over the vocab (no full sort).
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for (v, &x) in logits.iter().enumerate() {
        let lsm = x - lse;
        if best.len() < k {
            best.push((lsm, v as u32));
            best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        } else if lsm > best[0].0 {
            best[0] = (lsm, v as u32);
            best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }
    }
    best
}

/// One live beam: its forked KV working set + sequence cursor, cumulative
/// log-prob, and token trail.
struct Beam {
    ws: WorkingSet,
    seq_len: u32,
    score: f32,
    tokens: Vec<u32>,
}

/// One raw-logits forward for `tokens` at `start` on `ws`: a fresh fire-once
/// pass (fresh channels — a channel attaches to only one pass for its
/// lifetime) whose epilogue publishes the read-out row's FULL `[vocab]` logits
/// to the host. Returns the host-read logits.
async fn forward_logits(
    on: &Pipeline,
    ws: &WorkingSet,
    start: u32,
    tokens: &[u32],
    vocab: u32,
    tag: &str,
) -> Result<Vec<f32>> {
    let n = tokens.len() as u32;
    let end = start + n;
    let page_size = ws.page_size();
    let pool_pages = ws.page_len();
    let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let toks = Channel::from(toks_i32).named("toks");
    let embed_indptr = Channel::from(vec![0u32, n]).named("embed_indptr");
    let positions = Channel::from((start..end).collect::<Vec<_>>()).named("positions");
    let pages = Channel::from((0..pool_pages).collect::<Vec<_>>()).named("pages");
    let page_indptr = Channel::from(vec![0u32, end.div_ceil(page_size)]).named("page_indptr");
    let w_slot = Channel::from(
        (start..end)
            .map(|position| position / page_size)
            .collect::<Vec<_>>(),
    )
    .named("w_slot");
    let w_off = Channel::from(
        (start..end)
            .map(|position| position % page_size)
            .collect::<Vec<_>>(),
    )
    .named("w_off");
    let logits_out = Channel::new([vocab], dtype::f32).named("logits_out");

    let fwd = ForwardPass::new();
    fwd.embed(&toks, &embed_indptr)?;
    let kv_len = Channel::from(vec![end]).named("kv_len");
    fwd.attention(
        ws,
        ..,
        (start / page_size)..,
        &kv_len,
        &pages,
        &page_indptr,
        &w_slot,
        &w_off,
        &positions,
        None,
    )?;
    fwd.epilogue(move || {
        // No on-device select: ship the raw logits row to the host.
        logits_out.put(&intrinsics::logits());
    });

    fwd.submit(on).map_err(|e| format!("submit {tag}: {e}"))?;
    logits_out
        .take()
        .get::<f32>()
        .await
        .map_err(|e| format!("logits take {tag}: {e}"))
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let params: serde_json::Value = serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let max_tokens = params
        .get("max_tokens")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(MAX_TOKENS);
    let beam_width = params
        .get("beam_width")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(BEAM_WIDTH);

    let vocab = wit_model::output_vocab_size();

    let mut prompt = wit_model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;

    // Root working set: reserve the whole logical envelope BEFORE any fork so
    // every CoW child shares one address space sized for prompt + decode.
    let root = WorkingSet::new();
    let max_pages = (n + max_tokens as u32 + 1).div_ceil(root.page_size());
    root.reserve(max_pages)
        .map_err(|e| format!("root reserve: {e}"))?;

    // One run-ahead ordering domain for the whole search: forks and fires
    // linearize on it, so a survivor's fork is ordered after its parent's
    // KV write and before the survivor's own re-feed forward.
    let pipe = Pipeline::new();

    // Seed `beam_width` beams from the prompt; each first forward feeds the
    // whole prompt, each subsequent forward feeds that beam's one new token.
    let mut beams: Vec<Beam> = Vec::with_capacity(beam_width);
    let mut pending: Vec<Vec<u32>> = Vec::with_capacity(beam_width);
    for _ in 0..beam_width {
        let ws = root.fork(&pipe).map_err(|e| format!("seed fork: {e}"))?;
        beams.push(Beam {
            ws,
            seq_len: 0,
            score: 0.0,
            tokens: Vec::with_capacity(max_tokens),
        });
        pending.push(prompt.clone());
    }

    for step in 0..max_tokens {
        // Expand: forward every beam, read logits back, host-select its top-B
        // children. On step 0 all beams are identical, so only expand the
        // first (avoids `beam_width` duplicate survivors from a shared prefix).
        let active = if step == 0 { 1 } else { beams.len() };
        let mut cand: Vec<(f32, usize, u32)> = Vec::with_capacity(active * beam_width);
        for b in 0..active {
            let start = beams[b].seq_len;
            let inp = pending[b].clone();
            let tag = format!("@{step}/{b}");
            let logits = forward_logits(&pipe, &beams[b].ws, start, &inp, vocab, &tag).await?;
            beams[b].seq_len += inp.len() as u32;

            let base_score = beams[b].score;
            for (lsm, tok) in top_k_children(&logits, beam_width) {
                cand.push((base_score + lsm, b, tok));
            }
        }

        // Global top-B over all candidates (host merge).
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        cand.truncate(beam_width);

        // Re-fork each survivor from its parent beam and re-feed its token.
        let mut next: Vec<Beam> = Vec::with_capacity(beam_width);
        let mut next_pending: Vec<Vec<u32>> = Vec::with_capacity(beam_width);
        for (score, parent, tok) in &cand {
            let ws = beams[*parent]
                .ws
                .fork(&pipe)
                .map_err(|e| format!("survivor fork: {e}"))?;
            let mut tokens = beams[*parent].tokens.clone();
            tokens.push(*tok);
            next.push(Beam {
                ws,
                seq_len: beams[*parent].seq_len,
                score: *score,
                tokens,
            });
            next_pending.push(vec![*tok]);
        }
        beams = next;
        pending = next_pending;
    }
    pipe.close();

    let best = beams
        .iter()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        .ok_or("no surviving beam")?;

    Ok(format!(
        "BEAM_BASELINE ok width={beam_width} steps={max_tokens} best_score={:.4} tokens={:?}",
        best.score, best.tokens
    ))
}
