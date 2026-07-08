//! **Overview §6.2 beam search — device-geometry form (B1).** The guest beam
//! epilogue traces the WIRE-FORM geometry in-graph: on top of the §6.2 [B,P]
//! slot geometry it computes the CSR `page_indptr = [0, CumSum(np)]` and the
//! densely-packed live pages, binding them to the `PageIndptr`/`Pages` ports —
//! so the driver reads `page_indptr[B]` live entries directly, with NO host
//! replay of the freeze / designated-child / page-turn arithmetic.
//!
//! This is the guest half of Track B's beam-deletion endgame: the program the
//! host `ptir_beam.rs` replay is being retired in favour of (B3/B5). Authored
//! on the `inferlet::ptir` bridge and submitted end-to-end through
//! `forward-pass.new` / `pipeline.submit`. The device e2e run is gated on
//! charlie's [B,P] fork/freeze driver geometry (thrust-1+3).
//!
//! The geometry contract is host-verified by
//! `sdk/rust/ptir-dsl/tests/beam_goldens.rs` (`wire_charlie_page_indptr_and_packed`,
//! `wire_continue_tail_packed`) on the CPU reference interpreter.

use inferlet::ptir::prelude::*;
use inferlet::{model as wit_model, Result};

const B: u32 = 2;
const P: u32 = 4; // page slots per beam (run's max length; D4 compaction out of scope)
const PAGE_T: u32 = 16;
const NUM_LAYERS: u32 = 28;
const BOS: i32 = 1;
const MAX_STEPS: usize = 8;

fn bx<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    let v = vocab;
    model::configure(vocab, PAGE_T, NUM_LAYERS);

    // §6.2 loop-carried slot geometry (seeded, D2).
    let pages = bx(Channel::seeded([B, P], dtype::u32).named("pages"));
    let lens = bx(Channel::seeded([B, P], dtype::u32).named("lens"));
    let klen = bx(Channel::from(vec![0u32; B as usize]).named("klen"));
    let kvm = bx(Channel::seeded([B, P * PAGE_T], dtype::bool).named("kvm"));
    let pos = bx(Channel::from(vec![0u32; B as usize]).named("pos"));
    let np = bx(Channel::from(vec![1u32; B as usize]).named("np"));
    let tslot = bx(Channel::from(vec![0u32; B as usize]).named("tslot"));
    let tfill = bx(Channel::from(vec![0u32; B as usize]).named("tfill"));
    let w_slot = bx(Channel::from(vec![0u32; B as usize]).named("w_slot"));
    let w_off = bx(Channel::from(vec![0u32; B as usize]).named("w_off"));
    let toks = bx(Channel::from(vec![BOS; B as usize]).named("toks"));
    let scores = bx(Channel::from(vec![0.0f32; B as usize]).named("scores"));
    let fresh = bx(Channel::new([B], dtype::u32).named("fresh"));
    let out = bx(Channel::new([B], dtype::i32).named("out"));
    let out_par = bx(Channel::new([B], dtype::u32).named("out_par"));
    let out_scr = bx(Channel::new([B], dtype::f32).named("out_scr"));
    // Wire-form derivative channels — device-produced each fire, read by the
    // descriptor (Pages/PageIndptr ports), so seeded for the fire-0 read.
    let page_indptr = bx(Channel::seeded([B + 1], dtype::u32).named("page_indptr"));
    let packed = bx(Channel::seeded([B * P], dtype::u32).named("packed"));
    // w_cont [B] bool: per-lane heir(true)/fork(false). A host-reader terminal
    // output the runtime reads at finalize to reclaim continuing heirs' unused
    // fresh page grants (PageLease.reclaim_after_fire).
    let w_cont_ch = bx(Channel::new([B], dtype::bool).named("w_cont"));

    let ws: &'static WorkingSet = bx(WorkingSet::new());

    let fwd: &'static ForwardPass<'static> = bx(ForwardPass::new());
    let lanes_b = Tensor::constant((0u32..=B).collect::<Vec<_>>());
    fwd.embed(toks, lanes_b);
    fwd.positions(pos);
    // Register the working set + bind KvLen via the sugar arity, then bind the
    // device-geometry ports directly: Pages ← densely-packed live pages,
    // PageIndptr ← CumSum(np) channel (the escape hatch the working-set sugar,
    // which takes a const page_indptr, can't express).
    fwd.attn_working_set(ws, klen);
    fwd.port_channel(Port::Pages, packed);
    fwd.port_channel(Port::PageIndptr, page_indptr);
    fwd.port_channel(Port::WSlot, w_slot);
    fwd.port_channel(Port::WOff, w_off);
    fwd.attn_mask(kvm);
    fwd.epilogue(move || {
        let cand = add(broadcast(reshape(scores.take(), [B, 1]), [B, v]), log_softmax(intrinsics::logits()));
        let (s, i) = top_k(reshape(cand, [B * v]), B);
        let parent = div(&i, v);
        let pg = gather(pages.take(), &parent);
        let pl = gather(lens.take(), &parent);
        let n = gather(np.take(), &parent);
        let tf = gather(tfill.take(), &parent);
        let lanes = iota(B);
        let heir = scatter_set(&lanes, &parent, &lanes);
        let cont = and(eq(gather(heir, &parent), &lanes), lt(&tf, PAGE_T));
        let slot = select(&cont, gather(tslot.take(), &parent), fresh.take());
        let off = select(&cont, &tf, 0u32);
        let n2 = select(&cont, &n, add(&n, 1u32));
        let tcol = add(mul(&lanes, P), sub(&n2, 1u32));
        let pg3 = reshape(scatter_set(reshape(pg, [B * P]), &tcol, &slot), [B, P]);
        pages.put(&pg3);
        let off1 = add(&off, 1u32);
        let pl2 = reshape(scatter_set(reshape(pl, [B * P]), &tcol, &off1), [B, P]);
        lens.put(&pl2);
        klen.put(add(mul(sub(&n2, 1u32), PAGE_T), &off1));
        let io = reshape(iota(PAGE_T), [1, 1, PAGE_T]);
        let iob = broadcast(io, [B, P, PAGE_T]);
        let lb = broadcast(reshape(&pl2, [B, P, 1]), [B, P, PAGE_T]);
        kvm.put(reshape(lt(iob, lb), [B, P * PAGE_T]));
        pos.put(add(pos.take(), 1u32));
        np.put(&n2);
        tslot.put(&slot);
        tfill.put(&off1);
        w_slot.put(&slot);
        w_off.put(&off);
        let tok_i = cast(rem(&i, v), DType::I32);
        toks.put(&tok_i);
        scores.put(&s);

        // ── wire-form geometry (emitted before the reader terminal puts) ──
        // page_indptr [B+1] = [0, CumSum(np)]; CumSum is F32-only so route np
        // through F32 (values are small, exact).
        let cps = cast(cumsum(cast(&n2, DType::F32)), DType::U32);
        let zeros_b1 = Tensor::constant(vec![0u32; (B + 1) as usize]);
        let pidx = scatter_set(zeros_b1, add(iota(B), 1u32), &cps);
        page_indptr.put(&pidx);
        // Densely pack the live [B,P] pages: entry (b,j) → packed[page_start[b]+j].
        let page_start = gather(&pidx, iota(B));
        let start_bp = broadcast(reshape(&page_start, [B, 1]), [B, P]);
        let col = broadcast(reshape(iota(P), [1, P]), [B, P]);
        let dest = reshape(add(start_bp, col), [B * P]);
        let pg3_flat = reshape(&pg3, [B * P]);
        packed.put(scatter_set(&pg3_flat, &dest, &pg3_flat));

        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
        w_cont_ch.put(&cont); // per-lane heir(true)/fork(false) for page reclaim
    });

    // Beam decode loop: feed headroom slot ids, submit run-ahead, harvest.
    let pipeline = Pipeline::new();
    let mut hyp_tokens: Vec<u32> = Vec::new();
    for step in 0..MAX_STEPS {
        let grant = ws.alloc(B).map_err(|e| format!("ws.alloc @{step}: {e}"))?;
        fresh.put(grant);
        pipeline.submit(fwd).map_err(|e| format!("submit @{step}: {e}"))?;
        let picked = out.take().get::<i32>().map_err(|e| format!("out.take @{step}: {e}"))?;
        let _parents = out_par.take().get::<u32>().map_err(|e| format!("out_par.take @{step}: {e}"))?;
        let _scr = out_scr.take().get::<f32>().map_err(|e| format!("out_scr.take @{step}: {e}"))?;
        if let Some(&t0) = picked.first() {
            hyp_tokens.push(t0 as u32);
        }
    }

    let result = format!(
        "BEAM_DEVGEO B={B} steps={} tokens={hyp_tokens:?} (device-geometry §6.2 beam, vocab={vocab})",
        hyp_tokens.len()
    );
    println!("BEAM_DEVGEO_E2E {result}");
    Ok(result)
}
