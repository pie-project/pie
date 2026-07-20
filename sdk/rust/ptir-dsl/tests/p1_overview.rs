//! P1 exit tests: the overview §3 greedy-decode pipeline lowers to a canonical
//! trace container and hashes stably (byte-identity locked to golden C3 hashes),
//! plus error-message snapshot tests for the lint set (double-endpoint,
//! readiness-direction conflict, sink misplacement).
//!
//! These drive the neutral [`Builder`] directly (the author-facing
//! `ForwardPass`/`WorkingSet` surface lives in `inferlet`). Idiom note: values
//! reused as op operands take `&`; a value used once is moved.

use ptir_dsl::builder::Builder;
use ptir_dsl::prelude::*;
use ptir_dsl::ptir::op::Op;
use ptir_dsl::{Channel, TraceError, Traced};

const VOCAB: u32 = 32_000;
const PAGE: u32 = 16;
const LAYERS: u32 = 32;

// Golden C3 identity hashes (FNV-1a over the canonical container bytes). These
// LOCK byte-identity of the lowering: a change here means the emitted container
// bytes moved. Captured at the A1 pre-refactor `ForwardPass` baseline.
const GOLDEN_S3: u64 = 13510327253018374923;
const GOLDEN_BEAM: u64 = 2712280431104955710;
const GOLDEN_MTP_GRAMMAR: u64 = 5700756782757024409;

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// Host stub: the root grammar mask (a `[vocab]` bool). Real matcher is host code.
fn initial_mask() -> Vec<bool> {
    vec![true; VOCAB as usize]
}

// ---------------------------------------------------------------------------
// overview §3 — greedy + grammar-masked decode, software-pipelined
// ---------------------------------------------------------------------------

/// Build the §3 forward pass verbatim (the trace-producing portion). Channels
/// live for `'static` via `Box::leak` (test-only; a real inferlet keeps them on
/// its stack).
fn build_s3() -> Traced {
    let ctr1: &'static Tensor = leak(Tensor::constant([0u32, 1]));
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32).named("tok"));
    let out: &'static Channel = leak(Channel::new([1], dtype::i32).named("out"));
    let mask: &'static Channel =
        leak(Channel::new([intrinsics::vocab()], dtype::bool).named("mask"));
    let len: &'static Channel = leak(Channel::from([1u32]).named("len"));
    let rng_ch: &'static Channel = leak(Channel::from([7u32, 0]).named("rng"));

    // seed token -> cell full
    tok.put([1i32]);

    let mut b = Builder::new();
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, Tensor::constant([0u32, 1]));
    b.bind_port(Port::KvLen, len);
    b.stage(Stage::Epilogue, move || {
        let logits = intrinsics::logits();
        let r = rng_ch.take();
        let g = gumbel(&r, [intrinsics::vocab()]);
        let t = reduce_argmax(add(mask_apply(logits, mask.take()), g));
        rng_ch.put(add(&r, ctr1));
        tok.put(&t);
        len.put(add(len.take(), 1u32));
        out.put(t);
    });

    // prime mask_0 before the first submit (§3 line 261) so `mask` has a producer.
    mask.put(initial_mask());
    b.build().expect("§3 must build to a validated container")
}

#[test]
fn s3_traces_and_validates() {
    ptir_dsl::model::configure(VOCAB, PAGE, LAYERS);
    let traced = build_s3();

    let c = traced.container();
    assert_eq!(c.stages.len(), 1, "one epilogue stage");
    assert_eq!(c.stages[0].stage, Stage::Epilogue);
    assert_eq!(c.channels.len(), 5, "tok/out/mask/len/rng");
    let puts = c.stages[0]
        .ops
        .iter()
        .filter(|op| matches!(op, Op::ChanPut { .. }))
        .count();
    assert_eq!(puts, 4, "rng, tok, len, out puts");
}

#[test]
fn s3_identity_hash_is_stable() {
    ptir_dsl::model::configure(VOCAB, PAGE, LAYERS);
    let a = build_s3().identity_hash();
    let b = build_s3().identity_hash();
    assert_eq!(a, b, "the same program hashes identically (C3)");
    assert_eq!(
        a, GOLDEN_S3,
        "byte-identical to the pre-A1 golden container"
    );
}

#[test]
fn different_program_hashes_differently() {
    ptir_dsl::model::configure(VOCAB, PAGE, LAYERS);
    let greedy = build_s3().identity_hash();

    let tok: &'static Channel = leak(Channel::new([1], dtype::i32));
    let rng_ch: &'static Channel = leak(Channel::from([7u32, 0]));
    tok.put([1i32]);
    let mut b = Builder::new();
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, Tensor::constant([0u32, 1]));
    b.stage(Stage::Epilogue, move || {
        let logits = intrinsics::logits();
        let r = rng_ch.take();
        let scaled = mul(logits, 2.0f32); // temperature != greedy
        let g = gumbel(&r, [intrinsics::vocab()]);
        let t = reduce_argmax(add(scaled, g));
        rng_ch.put(add(&r, Tensor::constant([0u32, 1])));
        tok.put(t);
    });
    let other = b.build().unwrap().identity_hash();
    assert_ne!(greedy, other, "different op graph => different identity");
}

// ---------------------------------------------------------------------------
// lint set — error-message snapshot tests (P1.3)
// ---------------------------------------------------------------------------

#[test]
fn lint_double_endpoint_host_both_ends() {
    ptir_dsl::model::configure(VOCAB, PAGE, LAYERS);
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32));
    // `dup` is claimed by the host as BOTH writer and reader (no pass endpoint
    // remains — SPSC violation). It is also consumed by the epilogue so it
    // enters the trace container.
    let dup: &'static Channel = leak(Channel::new([1], dtype::i32).named("dup"));
    tok.put([1i32]);
    dup.put([0i32]); // host writes
    let _ = dup.take(); // host also consumes

    let mut b = Builder::new();
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, Tensor::constant([0u32, 1]));
    b.stage(Stage::Epilogue, move || {
        let v = dup.take(); // pass consumes it too (so `dup` is interned)
        tok.put(add(&v, reduce_argmax(intrinsics::logits())));
    });

    let err = b.build().expect_err("host-both-endpoints must fail");
    let msg = err.to_string();
    assert!(
        err.0.iter().any(|e| matches!(
            e,
            TraceError::DoubleEndpoint { role: "host", channel, .. } if channel == "dup"
        )),
        "expected a host DoubleEndpoint on `dup`, got:\n{msg}"
    );
    assert!(msg.contains("two host endpoints"), "message:\n{msg}");
}

#[test]
fn lint_readiness_conflict_consumed_never_produced() {
    ptir_dsl::model::configure(VOCAB, PAGE, LAYERS);
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32));
    let orphan: &'static Channel = leak(Channel::new([1], dtype::i32).named("orphan"));
    tok.put([1i32]);

    let mut b = Builder::new();
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, Tensor::constant([0u32, 1]));
    b.stage(Stage::Epilogue, move || {
        let v = orphan.take();
        let _ = intrinsics::logits();
        tok.put(add(&v, 1u32));
    });

    let err = b
        .build()
        .expect_err("consuming an unproduced channel must fail");
    let msg = err.to_string();
    assert!(
        err.0.iter().any(|e| matches!(
            e,
            TraceError::ReadinessConflict { channel, .. } if channel == "orphan"
        )),
        "expected a ReadinessConflict on `orphan`, got:\n{msg}"
    );
    assert!(msg.contains("never produced"), "message:\n{msg}");
}

#[test]
fn lint_sink_misplacement_in_epilogue() {
    ptir_dsl::model::configure(VOCAB, PAGE, LAYERS);
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32));
    let budget: &'static Channel = leak(Channel::from([256u32]));
    tok.put([1i32]);

    let mut b = Builder::new();
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, Tensor::constant([0u32, 1]));
    b.stage(Stage::Epilogue, move || {
        let logits = intrinsics::logits();
        let mask = pivot_threshold(&logits, rank_le(budget.read()));
        intrinsics::kernel::attn_page_mask(mask);
        tok.put(reduce_argmax(&logits));
    });

    let err = b.build().expect_err("sink at epilogue must fail");
    let msg = err.to_string();
    assert!(
        err.0
            .iter()
            .any(|e| matches!(e, TraceError::SinkMisplacement { .. })),
        "expected a SinkMisplacement, got:\n{msg}"
    );
    assert!(msg.contains("attn_page_mask"), "message:\n{msg}");
}

// ---------------------------------------------------------------------------
// overview §6.2 — beam search (the second P1 exit gate): reorder = gathers,
// divergence = freeze. Exercises the full op set (top_k, log_softmax, gather,
// scatter_set, reshape, iota, broadcast, div/rem/mul/sub, lt/and/eq, cast).
// F8: no auto-drain synthesis — the loop-carried peek-port channels
// (klen/kvm) drain EXPLICITLY (`take()` directly before the re-put), which
// reproduces the previously-synthesized ops verbatim (same golden hash).
// ---------------------------------------------------------------------------

#[test]
fn s6_2_beam_epilogue_binds() {
    const B: u32 = 2;
    const V: u32 = 8;
    const P: u32 = 3;
    const PAGE_T: u32 = 4;
    ptir_dsl::model::configure(V, PAGE_T, 2);

    // channels 0..=15 as in overview §6.2 / echo's beam_trace.
    let pages: &'static Channel = leak(Channel::seeded([B, P], dtype::u32).named("pages"));
    let lens: &'static Channel = leak(Channel::seeded([B, P], dtype::u32).named("lens"));
    let klen: &'static Channel = leak(Channel::from(vec![0u32; B as usize]).named("klen"));
    let kvm: &'static Channel = leak(Channel::seeded([B, P * PAGE_T], dtype::bool).named("kvm"));
    let pos: &'static Channel = leak(Channel::from(vec![0u32; B as usize]).named("pos"));
    let np: &'static Channel = leak(Channel::from(vec![1u32; B as usize]).named("np"));
    let tslot: &'static Channel = leak(Channel::from(vec![0u32; B as usize]).named("tslot"));
    let tfill: &'static Channel = leak(Channel::from(vec![0u32; B as usize]).named("tfill"));
    let w_slot: &'static Channel = leak(Channel::from(vec![0u32; B as usize]).named("w_slot"));
    let w_off: &'static Channel = leak(Channel::from(vec![0u32; B as usize]).named("w_off"));
    let toks: &'static Channel = leak(Channel::from(vec![1i32; B as usize]).named("toks"));
    let scores: &'static Channel = leak(Channel::from(vec![0.0f32; B as usize]).named("scores"));
    let fresh: &'static Channel = leak(Channel::new([B], dtype::u32).named("fresh"));
    let out: &'static Channel = leak(Channel::new([B], dtype::i32).named("out"));
    let out_par: &'static Channel = leak(Channel::new([B], dtype::u32).named("out_par"));
    let out_scr: &'static Channel = leak(Channel::new([B], dtype::f32).named("out_scr"));

    // host-fed headroom (slot grants are per-instance data; overview §5.2's
    // `fresh.put(ws.alloc(B))`), primed before submit.
    fresh.put(vec![0u32; B as usize]);

    let lanes_b = Tensor::constant((0u32..=B).collect::<Vec<_>>()); // [0,1,2] indptr
    let page_rows = Tensor::constant((0u32..=B).map(|i| i * P).collect::<Vec<_>>()); // [0,P,2P]

    let mut b = Builder::new();
    b.bind_port(Port::EmbedTokens, toks);
    b.bind_port(Port::EmbedIndptr, lanes_b);
    b.bind_port(Port::Positions, pos);
    b.bind_port(Port::Pages, pages);
    b.bind_port(Port::PageIndptr, page_rows);
    b.bind_port(Port::KvLen, klen);
    b.bind_port(Port::WSlot, w_slot);
    b.bind_port(Port::WOff, w_off);
    b.bind_port(Port::AttnMask, kvm);
    b.stage(Stage::Epilogue, move || {
        let cand = add(
            broadcast(reshape(scores.take(), [B, 1]), [B, V]),
            log_softmax(intrinsics::logits()),
        );
        let (s, i) = top_k(reshape(cand, [B * V]), B);
        let parent = div(&i, V);
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
        pages.put(reshape(
            scatter_set(reshape(pg, [B * P]), &tcol, &slot),
            [B, P],
        ));
        let off1 = add(&off, 1u32);
        let pl2 = reshape(scatter_set(reshape(pl, [B * P]), &tcol, &off1), [B, P]);
        lens.put(&pl2);
        let klen_next = add(mul(sub(&n2, 1u32), PAGE_T), &off1);
        klen.take();
        klen.put(klen_next);
        let io = reshape(iota(PAGE_T), [1, 1, PAGE_T]);
        let iob = broadcast(io, [B, P, PAGE_T]);
        let lb = broadcast(reshape(&pl2, [B, P, 1]), [B, P, PAGE_T]);
        let kvm_next = reshape(lt(iob, lb), [B, P * PAGE_T]);
        kvm.take();
        kvm.put(kvm_next);
        pos.put(add(pos.take(), 1u32));
        np.put(&n2);
        tslot.put(&slot);
        tfill.put(&off1);
        w_slot.put(&slot);
        w_off.put(&off);
        let tok_u = rem(&i, V);
        let tok_i = cast(&tok_u, ptir_dsl::DType::I32);
        toks.put(&tok_i);
        scores.put(&s);
        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
    });

    let traced = b.build().expect("§6.2 beam epilogue must bind");
    let c = traced.container();
    assert_eq!(c.stages[0].stage, Stage::Epilogue);
    assert_eq!(c.channels.len(), 16, "16 beam channels");
    assert_eq!(
        traced.identity_hash(),
        GOLDEN_BEAM,
        "canonical bytes remain stable"
    );

    // Regression (G2 fire-0 seed round-trip): channel 0 (`pages`) is [B,P] (2D).
    // The [B,P] shape MUST survive encode→decode, else `validate_seeds` rejects
    // the [B,P] seed as a byte-length mismatch (numel collapse).
    assert_eq!(
        c.channels[0].shape.numel(),
        (B * P) as u64,
        "pages [B,P] numel in built container"
    );
    let decoded =
        ptir_dsl::ptir::container::decode(&traced.encode()).expect("decode beam container");
    assert_eq!(
        decoded.channels[0].shape.dims(),
        &[B, P],
        "pages 2D dims survive encode->decode"
    );
    assert_eq!(
        decoded.channels[0].shape.numel(),
        (B * P) as u64,
        "pages [B,P] numel after encode->decode"
    );

    // host_role (fix #3): out/out_par/out_scr are terminal program outputs (prog-put,
    // no program/descriptor consumer) → inferred host Reader so the guest's `take`
    // is accepted; fresh (host-put headroom) is a Writer.
    use ptir_dsl::ptir::container::HostRole;
    assert_eq!(
        decoded.channels[13].host_role,
        HostRole::Reader,
        "out (13) is host-Reader"
    );
    assert_eq!(
        decoded.channels[14].host_role,
        HostRole::Reader,
        "out_par (14) is host-Reader"
    );
    assert_eq!(
        decoded.channels[15].host_role,
        HostRole::Reader,
        "out_scr (15) is host-Reader"
    );
    assert_eq!(
        decoded.channels[12].host_role,
        HostRole::Writer,
        "fresh (12) is host-Writer"
    );
}

// overview §6.1 — native-MTP + grammar spec-verify binds (the M3-G1 §6.1 pass:
// `mtp-grammar` inferlet's trace). Grammar mask BEFORE the argmax → grammar-legal
// picks; accept-prefix = leading run of picked[0..K] == argmax(mtp_logits).
#[test]
fn s6_1_mtp_grammar_binds() {
    const V: u32 = 8;
    const K: u32 = 3;
    ptir_dsl::model::configure(V, 4, 2);
    ptir_dsl::model::configure_gates(true, false); // has_mtp_logits
    let kp1 = K + 1;
    let gmask: &'static Channel = leak(Channel::new([kp1, V], dtype::bool).named("gmask"));
    let toks: &'static Channel = leak(Channel::from(vec![1i32; kp1 as usize]).named("toks"));
    let out: &'static Channel = leak(Channel::new([kp1], dtype::i32).named("out"));
    // gmask is host-fed each step (per-position grammar mask) — a host-side put
    // marks it host-writer + produces its value (mirrors the beam's `fresh.put`).
    gmask.put(vec![true; (kp1 * V) as usize]);
    let lanes = Tensor::constant((0u32..=kp1).collect::<Vec<_>>());

    let mut b = Builder::new();
    b.bind_port(Port::EmbedTokens, toks);
    b.bind_port(Port::EmbedIndptr, lanes);
    b.stage(Stage::Epilogue, move || {
        let masked = mask_apply(intrinsics::logits(), gmask.take()); // [K+1, V]
        let picked = reduce_argmax(&masked); // [K+1] grammar-constrained target
        // NATIVE MTP: K distinct draft heads [K, V] (echo's §6.1 K-vs-K+1 contract).
        let mtp = intrinsics::mtp_logits(K); // [K, V]
        let draft = reduce_argmax(&mtp); // [K]
        // mtp_verify_tail: head = picked[0..K]; accept-prefix = leading run of matches.
        let head = gather(&picked, iota(K)); // [K]
        let hit = eq(&head, &draft); // [K] bool
        let ones = broadcast(Tensor::constant(1.0f32), [K]);
        let zeros = broadcast(Tensor::constant(0.0f32), [K]);
        let run = cumprod(select(&hit, &ones, &zeros)); // [K]
        let nacc = cast(reduce_sum(&run), ptir_dsl::DType::U32); // accepted-prefix length
        let keep = ge(broadcast(&nacc, [kp1]), iota(kp1)); // [K+1]
        let neg1 = broadcast(Tensor::constant(-1i32), [kp1]);
        let commit = select(&keep, &picked, &neg1); // accept-prefix + -1 sentinels
        out.put(&commit);
    });
    let traced = b.build().expect("§6.1 mtp-grammar epilogue must bind");
    assert_eq!(
        traced.identity_hash(),
        GOLDEN_MTP_GRAMMAR,
        "byte-identical to the pre-A1 golden"
    );
}
