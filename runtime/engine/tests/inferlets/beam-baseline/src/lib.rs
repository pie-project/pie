//! Hand-rolled beam baseline — the M3-G2 ±10% reference.
//!
//! This is the **same beam math the fused `beam` inferlet does on-device, but
//! host-orchestrated per step**: every beam runs a plain raw-logits forward,
//! the host reads the logits back, computes `score + log_softmax`, selects the
//! per-beam top-B children, merges the `B*B` candidates and keeps the global
//! top-B, then RE-FORKS each survivor from its parent working set and re-feeds
//! its token. No on-device reorder / freeze / designated-child / compact —
//! every selection + KV reorder is a host round-trip.
//!
//! The M3 acceptance harness measures THIS inferlet's wall time into
//! a standalone comparison point for the fused `beam` inferlet's wall
//! time is within 10% of it. Because the fused form folds the whole
//! reorder/select epilogue into ONE device pass (no per-step logits read-back,
//! no host top-k, no re-fork), it should be strictly faster — this baseline is
//! the honest hand-rolled upper bound the fused path must beat-or-match.
//!
//! FIDELITY: this is a token-domain beam over the model's real logits (real
//! forwards, real KV fork), host-selected. It is the throughput/latency
//! reference for G2, not the on-device beam-geometry correctness capstone
//! (that is charlie's tier-0 replay of `s6_2_beam_epilogue_binds`).
//!
//! Runs on the raw low-level WIT (keep-core): the `Context` facade's
//! fork/append/forward is a plain `KvWorkingSet` + `geometry::*` write +
//! raw `ForwardPass`, and the `edsl::logits` program's output tensor is read
//! back directly with `output().read()`.

use inferlet::geometry;
use inferlet::inference::ForwardPass;
use inferlet::mask::bf16_hi_to_f32;
use inferlet::program::resolve_bindings;
use inferlet::sampling::program as edsl;
use inferlet::serde_json;
use inferlet::working_set::KvWorkingSet;
use inferlet::{model, Result};

/// Beam width.
const BEAM_WIDTH: usize = 2;
/// Default decode steps; override via `{"max_tokens":N}`.
const MAX_TOKENS: usize = 12;

/// Read a raw-logits output tensor as `f32` (bf16-hi or f32 storage), exactly
/// as the driver materializes it — byte-identical to the on-device logits the
/// fused beam consumes.
fn logits_as_f32(bytes: &[u8], vocab: usize) -> Vec<f32> {
    if bytes.len() == vocab * 2 {
        bytes
            .chunks_exact(2)
            .map(|c| bf16_hi_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    } else {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

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
/// log-prob, and token trail. `fresh` fires the run-ahead generate-start signal
/// on the beam's first forward (keep-core replacement for the facade's
/// per-context `fresh_generate`).
struct Beam {
    kv: KvWorkingSet,
    seq_len: u32,
    fresh: bool,
    score: f32,
    tokens: Vec<u32>,
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

    let vocab = model::output_vocab_size();

    // Raw-logits program: forward -> [vocab] logits (no on-device select).
    let built = edsl::logits(vocab).map_err(|e| format!("logits build: {e:?}"))?;
    let program =
        inferlet::emit::emit_program(&built.program).map_err(|e| format!("logits emit: {e}"))?;

    let mut prompt = model::encode("hello world");
    if prompt.is_empty() {
        prompt.push(0);
    }

    // Seed `beam_width` beams from the prompt; each first forward feeds the
    // whole prompt, each subsequent forward feeds that beam's one new token.
    let root = KvWorkingSet::new();
    let page = root.page_size();
    let mut beams: Vec<Beam> = Vec::with_capacity(beam_width);
    let mut pending: Vec<Vec<u32>> = Vec::with_capacity(beam_width);
    for _ in 0..beam_width {
        let kv = root.fork().map_err(|e| format!("seed fork: {e}"))?;
        beams.push(Beam {
            kv,
            seq_len: 0,
            fresh: true,
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
            let n = inp.len() as u32;
            let decode_pos = start + n - 1;

            let pass = ForwardPass::new();
            if beams[b].fresh {
                pass.fresh_generate();
                beams[b].fresh = false;
            }
            let geom = geometry::ensure_pages(
                &beams[b].kv,
                geometry::kv_write_geometry(start, n, page),
            )
            .map_err(|e| format!("pages @{step}/{b}: {e}"))?;
            geometry::attach_kv_write(&pass, &beams[b].kv, &geom);
            let positions: Vec<u32> = (start..start + n).collect();
            pass.input_tokens(&inp, &positions);
            let bindings = resolve_bindings(&built.bindings, &built.host_inputs, &[decode_pos], &[])
                .map_err(|e| format!("bind @{step}/{b}: {e}"))?;
            pass.sampler(&program, bindings);
            pass.execute();
            beams[b].seq_len += n;

            let out = pass
                .output()
                .await
                .map_err(|e| format!("logits output @{step}/{b}: {e}"))?;
            let bytes = out
                .read()
                .map_err(|e| format!("read logits @{step}/{b}: {e:?}"))?;
            let logits = logits_as_f32(&bytes, vocab as usize);
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
            let kv = beams[*parent]
                .kv
                .fork()
                .map_err(|e| format!("survivor fork: {e}"))?;
            let mut tokens = beams[*parent].tokens.clone();
            tokens.push(*tok);
            next.push(Beam {
                kv,
                seq_len: beams[*parent].seq_len,
                fresh: true,
                score: *score,
                tokens,
            });
            next_pending.push(vec![*tok]);
        }
        beams = next;
        pending = next_pending;
    }

    let best = beams
        .iter()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        .ok_or("no surviving beam")?;

    Ok(format!(
        "BEAM_BASELINE ok width={beam_width} steps={max_tokens} best_score={:.4} tokens={:?}",
        best.score, best.tokens
    ))
}
