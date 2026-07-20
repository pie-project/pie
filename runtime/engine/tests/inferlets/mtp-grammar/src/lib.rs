//! **Overview §6.1 — native-MTP + grammar in ONE forward** (the M3-G1 real §6.1
//! pass). SDK-authored via the `inferlet::ptir` bridge: spec-verify over the
//! driver's NATIVE draft heads (`intrinsics::mtp_logits()`) under a per-position
//! grammar mask (`mask_apply`), fused in one epilogue — the §6.1 discipline
//! (golden `rollout_trace` §2.2 is the reference), distinct from `mtpverify`'s
//! host-fed drafts.
//!
//! Per position the target's grammar-constrained greedy argmax is compared to the
//! MTP draft; the accept-prefix is the leading run of matches (`eq → cumprod →
//! select`, `-1` sentinel). Because the grammar mask applies BEFORE the argmax,
//! the verify accepts only grammar-legal tokens — the constraint composes with
//! speculation, losslessly (overview §6.1).
//!
//! A3: migrated off the deleted `register_program`/`ChannelSeed` WIT surface to
//! the unified `inferlet::ptir` bridge. The GPU RUN is gated on charlie's MTP
//! Stage-2 (`mtp_logits` driver head); the intrinsic API is stable, so this
//! pre-stages the §6.1 pass.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

/// MTP draft width (K); the verify window is `[K+1, V]`.
const K: u32 = 3;
const PAGE_T: u32 = 16;
const NUM_LAYERS: u32 = 28;
const BOS: i32 = 1;

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    let v = vocab;
    let kp1 = K + 1;
    model::configure(vocab, PAGE_T, NUM_LAYERS);
    model::configure_gates(
        /* has_mtp_logits */ true, /* has_value_head */ false,
    );

    // gmask: host-fed per-position grammar mask [K+1, V] bool (host-writer).
    let gmask = Channel::new([kp1, v], dtype::bool).named("gmask");
    // toks: the K+1 verify-window input tokens (seeded per instance).
    let toks = Channel::from(vec![BOS; kp1 as usize]).named("toks");
    // out: the committed accept-prefix (host-reader).
    let out = Channel::new([kp1], dtype::i32).named("out");
    let ws = WorkingSet::new();
    ws.reserve(kp1.div_ceil(PAGE_T))
        .map_err(|e| format!("reserve KV: {e}"))?;

    let fwd = ForwardPass::new();
    let lanes = Tensor::constant((0u32..=kp1).collect::<Vec<_>>()); // one token per window row
    fwd.embed(&toks, lanes);
    let kv_len = Channel::from((1..=kp1).collect::<Vec<_>>()).named("kv_len");
    fwd.port_channel(Port::KvLen, &kv_len);
    fwd.attn_working_set(&ws, .., ..)?;
    fwd.derive_dense_geometry();
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

    // One §6.1 verify step: feed an all-allow grammar mask (the host-writer put
    // both marks `gmask` host-writer for the trace and supplies the fire-0
    // value), submit, harvest the accept-prefix. Bool payloads are dtype-native
    // (1 byte per bool; the wire packs to bits). The multi-step accept-prefix
    // decode loop + a real grammar mask land with charlie's MTP Stage-2 driver.
    let pipeline = Pipeline::new();
    gmask.put(vec![true; (kp1 * v) as usize]); // all-allow
    fwd.submit(&pipeline).map_err(|e| format!("submit: {e}"))?;
    let committed = out
        .take()
        .get::<i32>()
        .await
        .map_err(|e| format!("out.take: {e}"))?;

    let result = format!(
        "MTP_GRAMMAR K={K} committed={} (SDK-authored §6.1 native-MTP+grammar, vocab={vocab})",
        committed.len()
    );
    println!("MTP_GRAMMAR_E2E {result}");
    Ok(result)
}
