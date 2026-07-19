//! **Pipelined ① copy of `multisamp`** — every parametric truncated-sampling
//! kind (TopK / TopP / MinP / TopK+TopP joint) run with SUBMIT-AHEAD depth on
//! the `inferlet::ptir` bridge, proving each kind's device mask + Gumbel-max
//! epilogue survives run-ahead submission (not just the synchronous drain).
//!
//! A COPY, not a rewrite-in-place (the original `multisamp` is a load-bearing
//! #7 dispatch-parity baseline). It differs deliberately in ONE way, kept from
//! the original pipelined copy: each kind runs on its OWN fresh context
//! (independent pipelined prefill + decode) rather than the original's
//! shared-context continuation — the copy's purpose is run-ahead × kind
//! validation, not the dispatch-pollution probe.
//!
//! Run-ahead form (the `runahead` idiom): the loop-carried `tok_in`/`rng`
//! channels are device-side (each fire's epilogue puts the values the next
//! fire takes — no host round-trip), so the host submits fire `t+1` BEFORE
//! draining fire `t`'s token; the host-facing `out` ring is widened to `DEPTH`
//! cells to absorb the run-ahead.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const TEMPERATURE: f32 = 0.8;
const STEPS_PER_KIND: usize = 4;
/// Submit-ahead window: fire `t+DEPTH-1` is submitted before fire `t`'s token
/// is drained. Also the host-read `out` ring capacity.
const DEPTH: usize = 2;

#[derive(Clone, Copy)]
enum Kind {
    TopK { k: u32 },
    TopP { p: f32 },
    MinP { p: f32 },
    TopKTopP { k: u32, p: f32 },
}

/// This kind's keep-mask over `scaled` (temperature-scaled logits), `[vocab]`
/// bool — identical to `multisamp`'s author-facing mask shapes.
fn keep_mask(kind: Kind, scaled: &Tensor) -> Tensor {
    match kind {
        Kind::TopK { k } => pivot_threshold(scaled, rank_le(k)),
        Kind::TopP { p } => {
            let probs = softmax(scaled);
            pivot_threshold(probs, cummass_le(p))
        }
        Kind::MinP { p } => {
            let probs = softmax(scaled);
            let thr = mul(reduce_max(&probs), p); // scalar threshold (NOT broadcast to [vocab])
            pivot_threshold(probs, prob_ge(thr))
        }
        Kind::TopKTopP { k, p } => {
            let probs = softmax(scaled);
            let m1 = pivot_threshold(scaled, rank_le(k));
            let m2 = pivot_threshold(probs, cummass_le(p));
            and(m1, m2)
        }
    }
}

/// Masked Gumbel-max sample: `argmax(select(keep, scaled, -inf) + gumbel(r))`.
/// `r` is the taken `[2]` u32 rng state (`[key, ctr]`).
fn sample(kind: Kind, scaled: Tensor, vocab: u32, r: impl AsTensor) -> Tensor {
    let keep = keep_mask(kind, &scaled);
    let neg_inf = broadcast(Tensor::constant(f32::NEG_INFINITY), [vocab]);
    let masked = select(&keep, &scaled, &neg_inf);
    let g = gumbel(r, [vocab]);
    reduce_argmax(add(masked, g))
}

/// One kind's independent pipelined decode: fresh working set, N-wide prefill
/// (awaited — the handoff rides the host), then a depth-`DEPTH` submit-ahead
/// decode chain for the remaining `STEPS_PER_KIND - 1` tokens.
async fn run_kind(name: &str, idx: usize, kind: Kind, vocab: u32) -> Result<Vec<u32>> {
    let ws = WorkingSet::new();
    let page_size = ws.page_size();

    let prompt = wit_model::encode("hello world");
    let prompt: Vec<u32> = if prompt.is_empty() { vec![0] } else { prompt };
    let n = prompt.len() as u32;
    let max_pages = (n + STEPS_PER_KIND as u32 + 1).div_ceil(page_size);
    ws.reserve(max_pages)
        .map_err(|e| format!("{name} ws.reserve: {e}"))?;

    // ── PREFILL FIRE (N-wide, awaited) — this kind's first token. ──
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let toks_p = Channel::from(prompt_i32).named("toks_p");
    let embed_indptr_p = Channel::from(vec![0u32, n]).named("embed_indptr_p");
    let positions_p = Channel::from((0..n).collect::<Vec<_>>()).named("positions_p");
    let pages_p = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages_p");
    let page_indptr_p = Channel::from(vec![0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p =
        Channel::from((0..n).map(|p| p / page_size).collect::<Vec<_>>()).named("w_slot_p");
    let w_off_p = Channel::from((0..n).map(|p| p % page_size).collect::<Vec<_>>()).named("w_off_p");
    let rng_p = Channel::from(vec![0x9e37_u32 ^ (idx as u32), 0]).named("rng_p");
    let g0_ch = Channel::new([1], dtype::i32).named("g0");

    let fwd_p = ForwardPass::new();
    fwd_p.embed(&toks_p, &embed_indptr_p)?;
    let kv_len_p = Channel::from(vec![n]).named("kv_len_p");
    fwd_p.attention(
        &ws,
        ..,
        ..,
        &kv_len_p,
        &pages_p,
        &page_indptr_p,
        &w_slot_p,
        &w_off_p,
        &positions_p,
        None,
    )?;
    fwd_p.epilogue(move || {
        let r = rng_p.take(); // [2] u32 rng state (key, ctr)
        let scaled = div(intrinsics::logits(), TEMPERATURE);
        let t = sample(kind, scaled, vocab, &r);
        let r_next = add(&r, iota(2));
        g0_ch.put(&t);
        rng_p.put(&r_next);
    });
    // ONE pipeline for this kind-run (R4-4): the prefill and the pipelined
    // decode are one sequential stream, so both phases submit here. The
    // handoff still rides the host (awaited g0 take), unchanged.
    let pipe = Pipeline::new();
    fwd_p
        .submit(&pipe)
        .map_err(|e| format!("{name} prefill submit: {e}"))?;
    let g0 = g0_ch
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("{name} g0 take: {e}"))?[0];

    let mut got: Vec<u32> = Vec::with_capacity(STEPS_PER_KIND);
    got.push(g0 as u32);

    // ── PIPELINED DECODE (1-wide, depth-DEPTH submit-ahead) ──
    if got.len() < STEPS_PER_KIND {
        let tok_in = Channel::from(vec![g0; 1]).named("tok_in");
        let rng = Channel::from(vec![0x51ed_u32 ^ (idx as u32), 0]).named("rng");
        // Host-read ring widened to the submit-ahead window.
        let out = Channel::new([1], dtype::i32)
            .capacity(DEPTH as u32)
            .named("out");
        let lane1 = Channel::from(vec![0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from(vec![n]).named("positions");
        let pages = Channel::from((0..max_pages).collect::<Vec<_>>()).named("pages");
        let page_indptr =
            Channel::from(vec![0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from(vec![n / page_size]).named("w_slot");
        let w_off = Channel::from(vec![n % page_size]).named("w_off");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        let kv_len = Channel::from(vec![n + 1]).named("kv_len");
        fwd.attention(
            &ws,
            ..,
            (n / page_size)..,
            &kv_len,
            &pages,
            &page_indptr,
            &w_slot,
            &w_off,
            &positions,
            None,
        )?;
        fwd.epilogue(move || {
            let length = kv_len.take().tensor();
            let r = rng.take(); // [2] u32 rng state
            let scaled = div(intrinsics::logits(), TEMPERATURE);
            let t = sample(kind, scaled, vocab, &r);

            let r_next = add(&r, iota(2));
            let next_length = add(&length, 1u32);
            let page_count = div(add(&next_length, page_size - 1), page_size);

            tok_in.put(&t);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(div(&length, page_size));
            w_off.put(rem(&length, page_size));
            page_indptr.take();
            page_indptr.put(mul(iota(2), broadcast(&page_count, [2])));
            out.put(&t);
            rng.put(&r_next);
        });

        // Prime + fill: launch up to DEPTH chain-linked fires upfront (none
        // awaited); finish() right after the last budget submit (F7 — fixed
        // budget, so the last submit is knowable at submit time); then FIFO
        // drain + refill one fire per drained token.
        let budget = STEPS_PER_KIND - 1;
        let mut submitted = 0usize;
        let mut inflight = 0usize;
        while inflight < DEPTH && submitted < budget {
            fwd.submit(&pipe)
                .map_err(|e| format!("{name} decode submit @{submitted}: {e}"))?;
            submitted += 1;
            inflight += 1;
        }
        while inflight > 0 {
            let t = out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("{name} out.take @{}: {e}", got.len()))?;
            inflight -= 1;
            let Some(&t0) = t.first() else {
                return Err(format!("{name} out.take @{}: empty tensor", got.len()));
            };
            got.push(t0 as u32);
            if submitted < budget {
                fwd.submit(&pipe)
                    .map_err(|e| format!("{name} decode submit @{submitted}: {e}"))?;
                submitted += 1;
                inflight += 1;
            }
        }
    }
    pipe.close();

    eprintln!("[MULTISAMP_PIPELINED] {name} tokens: {got:?}");
    Ok(got)
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();

    let kinds: [(&str, Kind); 4] = [
        ("topk", Kind::TopK { k: 40 }),
        ("topp", Kind::TopP { p: 0.9 }),
        ("minp", Kind::MinP { p: 0.05 }),
        ("joint", Kind::TopKTopP { k: 40, p: 0.9 }),
    ];

    let mut all = Vec::new();
    for (idx, (name, kind)) in kinds.into_iter().enumerate() {
        // Independent fresh context per kind (the copy's run-ahead × kind check).
        let got = run_kind(name, idx, kind, vocab).await?;
        all.extend(got);
    }

    Ok(format!("{{\"tokens\": {all:?}}}"))
}
