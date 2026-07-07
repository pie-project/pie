//! **Overview §6.1 — native-MTP + grammar in ONE forward** (the M3-G1 real §6.1
//! pass). SDK-authored: spec-verify over the driver's NATIVE draft heads
//! (`intrinsics::mtp_logits()`) under a per-position grammar mask
//! (`mask_apply`), fused in one epilogue — the §6.1 discipline (golden
//! `rollout_trace` §2.2 is the reference), distinct from `mtpverify`'s host-fed
//! drafts.
//!
//! Per position the target's grammar-constrained greedy argmax is compared to the
//! MTP draft; the accept-prefix is the leading run of matches (`eq → cumprod →
//! select`, `-1` sentinel). Because the grammar mask applies BEFORE the argmax,
//! the verify accepts only grammar-legal tokens — the constraint composes with
//! speculation, losslessly (overview §6.1).
//!
//! HOST-VALIDATED NOW: `fwd.trace()` binds (echo's `bind`); builds green to
//! `wasm32-wasip2`. The GPU RUN is gated on charlie's MTP Stage-2 (the
//! `mtp_logits` driver head); the intrinsic API is stable, so this pre-stages the
//! §6.1 pass — `PIE_M3_S61_INFERLET=mtp-grammar` arms G1's accepted-tok/s when
//! Stage-2 lands.

use inferlet::{Result, model as wit_model};
use ptir::prelude::*;
use ptir::DType;

/// MTP draft width (K); the verify window is `[K+1, V]`.
const K: u32 = 3;
const PAGE_T: u32 = 16;
const NUM_LAYERS: u32 = 28;
const BOS: i32 = 1;

// Channel indices (declaration order).
const CH_GMASK: u32 = 0;
const CH_OUT: u32 = 2;

/// Build the §6.1 native-MTP + grammar spec-verify trace; return container bytes.
fn s61_container(vocab: u32) -> core::result::Result<Vec<u8>, String> {
    ptir::model::configure(vocab, PAGE_T, NUM_LAYERS);
    ptir::model::configure_gates(/* has_mtp_logits */ true, /* has_value_head */ false);
    let v = vocab;
    let kp1 = K + 1;

    let bx = |c: Channel| -> &'static Channel { Box::leak(Box::new(c)) };
    // gmask: host-fed per-position grammar mask [K+1, V] bool (host-writer).
    let gmask = bx(Channel::new([kp1, v], dtype::bool).named("gmask"));
    // toks: the K+1 verify-window input tokens (seeded per instance).
    let toks = bx(Channel::from(vec![BOS; kp1 as usize]).named("toks"));
    // out: the committed accept-prefix (host-reader).
    let out = bx(Channel::new([kp1], dtype::i32).named("out"));
    // gmask is host-fed each step (per-position grammar mask) — a host-side put
    // marks it host-writer + produces its value (mirrors the beam's `fresh.put`).
    gmask.put(vec![true; (kp1 * v) as usize]);

    let fwd = ForwardPass::new();
    let lanes = Tensor::constant((0u32..=kp1).collect::<Vec<_>>()); // one token per window row
    fwd.embed(toks, lanes);
    fwd.epilogue(move || {
        // Grammar mask FIRST, then the target argmax → grammar-legal picks.
        let masked = mask_apply(intrinsics::logits(), gmask.take()); // [K+1, V]
        let picked = reduce_argmax(&masked); // [K+1] i32 — the grammar-constrained target
        // NATIVE MTP: K distinct draft heads [K, V] (echo's §6.1 K-vs-K+1 contract;
        // charlie's Stage-2 resolves the MtpLogits rows from this decl).
        let mtp = intrinsics::mtp_logits(K); // [K, V]
        let draft = reduce_argmax(&mtp); // [K] i32
        // mtp_verify_tail: head = picked[0..K]; accept-prefix = leading run of matches.
        let head = gather(&picked, iota(K)); // [K]
        let hit = eq(&head, &draft); // [K] bool
        let ones = broadcast(Tensor::constant(1.0f32), [K]);
        let zeros = broadcast(Tensor::constant(0.0f32), [K]);
        let run = cumprod(select(&hit, &ones, &zeros)); // [K]
        let nacc = cast(reduce_sum(&run), DType::U32); // accepted-prefix length
        let keep = ge(broadcast(&nacc, [kp1]), iota(kp1)); // [K+1]
        let neg1 = broadcast(Tensor::constant(-1i32), [kp1]);
        let commit = select(&keep, &picked, &neg1); // accept-prefix + -1 sentinels
        out.put(&commit);
    });

    let traced = fwd.trace().map_err(|e| format!("§6.1 trace: {e:?}"))?;
    Ok(traced.encode())
}

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    use inferlet::pie::core::ptir::{ChannelSeed, Pipeline, register_program};

    let vocab = wit_model::output_vocab_size();
    let bytes = s61_container(vocab).map_err(|e| format!("author §6.1: {e}"))?;
    let program = register_program(&bytes).map_err(|e| format!("register-program: {e}"))?;

    // Seed the K+1 window tokens (per-instance); gmask is host-fed each step.
    let seeds: Vec<ChannelSeed> = Vec::new();
    let pipeline =
        Pipeline::instantiate(program, &seeds).map_err(|e| format!("instantiate: {e}"))?;

    let gmask = pipeline.channel(CH_GMASK).map_err(|e| format!("channel(gmask): {e}"))?;
    let out = pipeline.channel(CH_OUT).map_err(|e| format!("channel(out): {e}"))?;

    // One §6.1 verify step: feed an all-allow grammar mask, submit, harvest the
    // accept-prefix. (The multi-step accept-prefix decode loop + real grammar mask
    // land with charlie's MTP Stage-2 driver; the pre-stage proves the wire.)
    let kp1 = (K + 1) as usize;
    let mask_bytes = vec![0xFFu8; kp1 * ((vocab as usize).div_ceil(8))]; // all-allow (packed bool)
    gmask.put(&mask_bytes).map_err(|e| format!("gmask.put: {e}"))?;
    pipeline.submit().map_err(|e| format!("submit: {e}"))?;
    let committed = out.take().map_err(|e| format!("out.take: {e}"))?;

    let result = format!(
        "MTP_GRAMMAR K={K} committed_bytes={} (SDK-authored §6.1 native-MTP+grammar, vocab={vocab})",
        committed.len()
    );
    println!("MTP_GRAMMAR_E2E {result}");
    Ok(result)
}
