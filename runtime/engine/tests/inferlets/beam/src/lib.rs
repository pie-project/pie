//! **Overview §6.2 beam search — the M3-G2 workload** (freeze / designated-child /
//! compact). SDK-authored via the `inferlet::ptir` bridge: the beam trace is a
//! faithful transcription of `sdk/rust/ptir-dsl/tests/p1_overview.rs::
//! s6_2_beam_epilogue_binds` (the validated reference), authored via the overview
//! surface (`ForwardPass`/`Channel`/`epilogue`), lowered to echo's canonical
//! container inside `forward-pass.new`, then run submit→take over a `Pipeline`.
//!
//! Reorder = index gathers; divergence = a *freeze* (NOT advancing the inherited
//! `lens` entry); the parent's designated child keeps filling its tail, siblings
//! open fresh slots — a surviving fork wastes nothing (overview §5.2). `klen`/`kvm`
//! are pure derivatives of `lens`, computed in the same epilogue; the tracer
//! auto-drains the put-without-take derivative channels (`lens` is take→put,
//! `klen`/`kvm` are put-only).
//!
//! A3: migrated off the deleted `register_program`/`ChannelSeed` WIT surface to
//! the unified `inferlet::ptir` bridge — one set of `Channel` objects owns both
//! the trace declaration and the host transport. Seeds ride each channel's own
//! pre-submit `put` (D2). The GPU e2e RUN is gated on charlie's [B,P]
//! fork/freeze/compact driver geometry (thrust-1+3).

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

/// Beam width (lanes) — small, matching the validated reference.
const B: u32 = 2;
/// Trace-known page-row capacity per lane (§6.1's P_MAX). SLACK absorbs
/// frozen-tail waste between compacts (D4).
const P: u32 = 4;
/// Tokens per KV page (model page size; mirrored at trace time).
const PAGE_T: u32 = 16;
/// BOS seed on every lane's input token.
const BOS: i32 = 1;
/// Decode steps for the run (the pre-stage keeps it short).
const MAX_STEPS: usize = 8;

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    let v = vocab;

    // 16 channels (overview §6.2 / echo's beam_trace). Per-instance geometry
    // (pages/lens/kvm) is `seeded` (D2); constant initials are `from`. Channel
    // tokens are Copy, so the epilogue closure and the host loop share them.
    let pages = Channel::seeded([B, P], dtype::u32).named("pages");
    let lens = Channel::seeded([B, P], dtype::u32).named("lens");
    let klen = Channel::from(vec![1u32; B as usize]).named("klen");
    let kvm = Channel::seeded([B, P * PAGE_T], dtype::bool).named("kvm");
    let pos = Channel::from(vec![0u32; B as usize]).named("pos");
    let np = Channel::from(vec![1u32; B as usize]).named("np");
    let tslot = Channel::from(vec![0u32; B as usize]).named("tslot");
    let tfill = Channel::from(vec![0u32; B as usize]).named("tfill");
    let w_slot = Channel::from(vec![0u32; B as usize]).named("w_slot");
    let w_off = Channel::from(vec![0u32; B as usize]).named("w_off");
    let toks = Channel::from(vec![BOS; B as usize]).named("toks");
    let scores = Channel::from(vec![0.0f32; B as usize]).named("scores");
    let fresh = Channel::new([B], dtype::u32).named("fresh");
    let out = Channel::new([B], dtype::i32).named("out");
    let out_par = Channel::new([B], dtype::u32).named("out_par");
    let out_scr = Channel::new([B], dtype::f32).named("out_scr");

    let ws = WorkingSet::new();

    let fwd = ForwardPass::new();
    let lanes_b = Channel::from((0u32..=B).collect::<Vec<_>>()).named("embed_indptr");
    let page_rows =
        Channel::from((0u32..=B).map(|i| i * P).collect::<Vec<_>>()).named("page_indptr");
    fwd.embed(&toks, &lanes_b)?;
    fwd.attention(
        &ws,
        ..,
        ..,
        &klen,
        &pages,
        &page_rows,
        &w_slot,
        &w_off,
        &pos,
        Some(&kvm),
    )?;
    fwd.epilogue(move || {
        // cand = running scores ⊕ log_softmax(logits); (s, i) = top_k over [B*V].
        let cand = add(
            broadcast(reshape(scores.take(), [B, 1]), [B, v]),
            log_softmax(intrinsics::logits()),
        );
        let (s, i) = top_k(reshape(cand, [B * v]), B);
        let parent = div(&i, v); // which lane each survivor came from
        // Reorder = row gathers by parent.
        let pg = gather(pages.take(), &parent);
        let pl = gather(lens.take(), &parent);
        let n = gather(np.take(), &parent);
        let tf = gather(tfill.take(), &parent);
        // Designated child: heir[p] = p's last child in lane order (last-wins scatter).
        let lanes = iota(B);
        let heir = scatter_set(&lanes, &parent, &lanes);
        let cont = and(eq(gather(heir, &parent), &lanes), lt(&tf, PAGE_T));
        // Continue the parent's tail iff designated child AND the tail has room;
        // else open a fresh slot at offset 0 (the freeze: siblings don't advance).
        let slot = select(&cont, gather(tslot.take(), &parent), fresh.take());
        let off = select(&cont, &tf, 0u32);
        let n2 = select(&cont, &n, add(&n, 1u32));
        let tcol = add(mul(&lanes, P), sub(&n2, 1u32)); // flat index of each lane's tail entry
        pages.put(reshape(
            scatter_set(reshape(pg, [B * P]), &tcol, &slot),
            [B, P],
        ));
        let off1 = add(&off, 1u32);
        let pl2 = reshape(scatter_set(reshape(pl, [B * P]), &tcol, &off1), [B, P]);
        lens.put(&pl2); // the source; klen/kvm are its two derivatives (put-only, tracer drains)
        klen.take();
        klen.put(add(mul(sub(&n2, 1u32), PAGE_T), &off1)); // physical span (frozen pages full)
        let io = reshape(iota(PAGE_T), [1, 1, PAGE_T]);
        let iob = broadcast(io, [B, P, PAGE_T]);
        let lb = broadcast(reshape(&pl2, [B, P, 1]), [B, P, PAGE_T]);
        kvm.take();
        kvm.put(reshape(lt(iob, lb), [B, P * PAGE_T])); // valid iff in-page offset < lens entry
        pos.put(add(pos.take(), 1u32)); // logical length (ping-pong)
        np.put(&n2);
        tslot.put(&slot);
        tfill.put(&off1);
        w_slot.put(&slot); // next step's write descriptor
        w_off.put(&off);
        let tok_i = cast(rem(&i, v), DType::I32);
        toks.put(&tok_i);
        scores.put(&s);
        out.put(&tok_i); // host-facing token (back-pressure)
        out_par.put(&parent); // reorder permutation, for host hypothesis backtracking
        out_scr.put(&s); // running scores (final ranking)
    });

    // Beam decode loop (overview §6.2 host): feed headroom slot ids, submit
    // run-ahead, harvest (token, parent, score) per step for hypothesis backtrack.
    // The first `fresh.put` (before the first submit's build) marks `fresh`
    // host-writer for the trace; one grant is staged per fire (capacity 1).
    let pipeline = Pipeline::new();
    let mut hyp_tokens: Vec<u32> = Vec::new();
    for step in 0..MAX_STEPS {
        // Fresh headroom for this fire: B grant ids from the working set (D2).
        let grant = ws
            .reserve(B)
            .map_err(|e| format!("ws.reserve @{step}: {e}"))?;
        fresh.put(grant);
        fwd.submit(&pipeline)
            .map_err(|e| format!("submit @{step}: {e}"))?;
        let picked = out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("out.take @{step}: {e}"))?;
        let _parents = out_par
            .take()
            .get::<u32>()
            .await
            .map_err(|e| format!("out_par.take @{step}: {e}"))?;
        let _scr = out_scr
            .take()
            .get::<f32>()
            .await
            .map_err(|e| format!("out_scr.take @{step}: {e}"))?;
        if let Some(&t0) = picked.first() {
            hyp_tokens.push(t0 as u32);
        }
    }
    pipeline.close();

    let result = format!(
        "BEAM B={B} steps={} tokens={hyp_tokens:?} (SDK-authored §6.2 beam, vocab={vocab})",
        hyp_tokens.len()
    );
    println!("BEAM_E2E {result}");
    Ok(result)
}
