//! **Overview §6.2 beam search — the M3-G2 workload** (freeze / designated-child /
//! compact). SDK-authored: the beam trace is a faithful transcription of
//! `sdk/rust/ptir/tests/p1_overview.rs::s6_2_beam_epilogue_binds` (the validated
//! reference), authored via the `ptir` SDK (`ForwardPass`/`Channel`/`epilogue`),
//! `encode()`d to echo's canonical container, then run register→instantiate→
//! submit→take via the WIT (§6.4-style host-fed geometry).
//!
//! Reorder = index gathers; divergence = a *freeze* (NOT advancing the inherited
//! `lens` entry); the parent's designated child keeps filling its tail, siblings
//! open fresh slots — a surviving fork wastes nothing (overview §5.2). `klen`/`kvm`
//! are pure derivatives of `lens`, computed in the same epilogue; the tracer
//! auto-drains the put-without-take derivative channels (the drain-refill erratum —
//! carried faithfully here: `lens` is take→put, `klen`/`kvm` are put-only).
//!
//! HOST-VALIDATED NOW: `fwd.trace()` binds (echo's `bind`) — this inferlet builds
//! + traces + registers green to `wasm32-wasip2`. The GPU e2e RUN is gated on
//! charlie's [B,P] fork/freeze/compact driver geometry (thrust-1+3); when it lands
//! this is the M3-G2 workload (`PIE_M3_S62_INFERLET=beam`).

use inferlet::{Result, model as wit_model};
use ptir::prelude::*;
use ptir::DType;

/// Beam width (lanes) — small, matching the validated reference.
const B: u32 = 2;
/// Trace-known page-row capacity per lane (§6.1's P_MAX). SLACK absorbs
/// frozen-tail waste between compacts (D4).
const P: u32 = 4;
/// Tokens per KV page (model page size; mirrored at trace time).
const PAGE_T: u32 = 16;
const NUM_LAYERS: u32 = 28;
/// BOS seed on every lane's input token.
const BOS: i32 = 1;
/// Decode steps for the run (the pre-stage keeps it short).
const MAX_STEPS: usize = 8;

// Channel indices (declaration order = the container's dense channel ids).
const CH_FRESH: u32 = 12;
const CH_OUT: u32 = 13;
const CH_OUT_PAR: u32 = 14;
const CH_OUT_SCR: u32 = 15;

/// Build the §6.2 beam trace via the SDK and return the encoded container bytes.
/// Faithful transcription of `s6_2_beam_epilogue_binds` (V = the live model vocab).
fn beam_container(vocab: u32) -> core::result::Result<Vec<u8>, String> {
    ptir::model::configure(vocab, PAGE_T, NUM_LAYERS);
    let v = vocab;

    // 16 channels (overview §6.2 / echo's beam_trace). Per-instance geometry
    // (pages/lens/kvm) is `seeded` (D2); constant initials are `from`.
    let bx = |c: Channel| -> &'static Channel { Box::leak(Box::new(c)) };
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

    let ws: &'static WorkingSet = Box::leak(Box::new(WorkingSet::new()));
    fresh.put(ws.alloc(B)); // host-fed headroom, primed before submit

    let fwd = ForwardPass::new();
    let lanes_b = Tensor::constant((0u32..=B).collect::<Vec<_>>()); // indptr: one token per lane
    let page_rows = Tensor::constant((0u32..=B).map(|i| i * P).collect::<Vec<_>>()); // [0,P,2P]
    fwd.embed(toks, lanes_b);
    fwd.positions(pos);
    fwd.attn_working_set(ws, (pages, page_rows, klen, w_slot, w_off));
    fwd.attn_mask(kvm);
    fwd.epilogue(move || {
        // cand = running scores ⊕ log_softmax(logits); (s, i) = top_k over [B*V].
        let cand = add(broadcast(reshape(scores.take(), [B, 1]), [B, v]), log_softmax(intrinsics::logits()));
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
        pages.put(reshape(scatter_set(reshape(pg, [B * P]), &tcol, &slot), [B, P]));
        let off1 = add(&off, 1u32);
        let pl2 = reshape(scatter_set(reshape(pl, [B * P]), &tcol, &off1), [B, P]);
        lens.put(&pl2); // the source; klen/kvm are its two derivatives (put-only, tracer drains)
        klen.put(add(mul(sub(&n2, 1u32), PAGE_T), &off1)); // physical span (frozen pages full)
        let io = reshape(iota(PAGE_T), [1, 1, PAGE_T]);
        let iob = broadcast(io, [B, P, PAGE_T]);
        let lb = broadcast(reshape(&pl2, [B, P, 1]), [B, P, PAGE_T]);
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

    let traced = fwd.trace().map_err(|e| format!("beam trace: {e:?}"))?;
    Ok(traced.encode())
}

/// Encode a `[B]`-shaped u32 channel seed (little-endian) for `instantiate`.
fn seed_u32(n: u32, fill: u32) -> Vec<u8> {
    (0..n).flat_map(|_| fill.to_le_bytes()).collect()
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    use inferlet::pie::core::ptir::{ChannelSeed, Pipeline, register_program};

    let vocab = wit_model::output_vocab_size();
    let bytes = beam_container(vocab).map_err(|e| format!("author beam: {e}"))?;
    let program = register_program(&bytes).map_err(|e| format!("register-program: {e}"))?;

    // Instantiate: seed EVERY `seeded=true` channel (declaration order 0-11 —
    // `Channel::seeded` pages/lens/kvm + the `Channel::from` baked initials, which
    // the SDK currently surfaces as seeded=true since the container has no baked-
    // const value slot; flagged to echo). Channels 12-15 (fresh/out/out_par/out_scr)
    // are `Channel::new` → not seeded. The `from` initials: klen/pos/tslot/tfill/
    // w_slot/w_off/scores = 0, np = 1, toks = BOS(=1). bit patterns share u32 (i32
    // 1 == u32 1; f32 0.0 == u32 0).
    let seeds = vec![
        ChannelSeed { channel: 0, value: seed_u32(B * P, 0) },                     // pages [B,P]
        ChannelSeed { channel: 1, value: seed_u32(B * P, 0) },                     // lens  [B,P]
        ChannelSeed { channel: 2, value: seed_u32(B, 0) },                         // klen  [B] (from 0)
        ChannelSeed { channel: 3, value: vec![0u8; (B * P * PAGE_T) as usize] },   // kvm   [B,P*PAGE_T] bool
        ChannelSeed { channel: 4, value: seed_u32(B, 0) },                         // pos   [B] (from 0)
        ChannelSeed { channel: 5, value: seed_u32(B, 1) },                         // np    [B] (from 1)
        ChannelSeed { channel: 6, value: seed_u32(B, 0) },                         // tslot [B] (from 0)
        ChannelSeed { channel: 7, value: seed_u32(B, 0) },                         // tfill [B] (from 0)
        ChannelSeed { channel: 8, value: seed_u32(B, 0) },                         // w_slot[B] (from 0)
        ChannelSeed { channel: 9, value: seed_u32(B, 0) },                         // w_off [B] (from 0)
        ChannelSeed { channel: 10, value: seed_u32(B, BOS as u32) },               // toks  [B] i32 (from BOS)
        ChannelSeed { channel: 11, value: seed_u32(B, 0) },                        // scores[B] f32 (from 0.0)
    ];
    let pipeline =
        Pipeline::instantiate(program, &seeds).map_err(|e| format!("instantiate: {e}"))?;

    // Host endpoints on the host-facing channels.
    let fresh = pipeline.channel(CH_FRESH).map_err(|e| format!("channel(fresh): {e}"))?;
    let out = pipeline.channel(CH_OUT).map_err(|e| format!("channel(out): {e}"))?;
    let out_par = pipeline.channel(CH_OUT_PAR).map_err(|e| format!("channel(out_par): {e}"))?;
    let out_scr = pipeline.channel(CH_OUT_SCR).map_err(|e| format!("channel(out_scr): {e}"))?;

    // Beam decode loop (overview §6.2 host): feed headroom slot ids, submit
    // run-ahead, harvest (token, parent, score) per step for hypothesis backtrack.
    let mut hyp_tokens: Vec<u32> = Vec::new();
    let mut next_slot: u32 = B; // fresh slot-id cursor (host owns the working set)
    for step in 0..MAX_STEPS {
        let grant: Vec<u8> = (next_slot..next_slot + B).flat_map(|s| s.to_le_bytes()).collect();
        next_slot += B;
        fresh.put(&grant).map_err(|e| format!("fresh.put @{step}: {e}"))?;
        pipeline.submit().map_err(|e| format!("submit @{step}: {e}"))?;
        let picked = out.take().map_err(|e| format!("out.take @{step}: {e}"))?;
        let _parents = out_par.take().map_err(|e| format!("out_par.take @{step}: {e}"))?;
        let _scr = out_scr.take().map_err(|e| format!("out_scr.take @{step}: {e}"))?;
        // Each survivor token is a 4-byte i32; record lane 0's for the summary.
        if picked.len() >= 4 {
            hyp_tokens.push(u32::from_le_bytes([picked[0], picked[1], picked[2], picked[3]]));
        }
    }

    let result = format!(
        "BEAM B={B} steps={} tokens={hyp_tokens:?} (SDK-authored §6.2 beam, vocab={vocab})",
        hyp_tokens.len()
    );
    println!("BEAM_E2E {result}");
    Ok(result)
}
