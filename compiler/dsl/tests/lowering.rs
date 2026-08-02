//! Lowering a program to a canonical trace container.
//!
//! The greedy-decode pipeline, lowered and hashed for byte
//! identity against the golden FNV-1a hashes, plus error-message snapshots for
//! the lint set (double-endpoint, readiness-direction conflict, sink
//! misplacement).
//!
//! These drive the neutral [`Builder`] directly (the author-facing
//! `ForwardPass`/`WorkingSet` surface lives in `inferlet`). Idiom note: values
//! reused as op operands take `&`; a value used once is moved.

use pie_dsl::builder::Builder;
use pie_dsl::prelude::*;
use pie_dsl::ptir::op::Op;
use pie_dsl::{Channel, TraceError, Traced};

const VOCAB: u32 = 32_000;
const PAGE: u32 = 16;

// Golden C3 identity hashes (FNV-1a over the canonical container bytes). These
// LOCK byte-identity of the channel-only descriptor lowering: a change here
// means the emitted container bytes moved.
const GOLDEN_S3: u64 = 7015106236045798467;
const GOLDEN_BEAM: u64 = 16590305409635560083;
const GOLDEN_MTP_GRAMMAR: u64 = 12706234719755930498;

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// Host stub: the root grammar mask (a `[vocab]` bool). Real matcher is host code.
fn initial_mask() -> Vec<bool> {
    vec![true; VOCAB as usize]
}

// ---------------------------------------------------------------------------
// greedy + grammar-masked decode, software-pipelined
// ---------------------------------------------------------------------------

/// Build the greedy-decode forward pass (the trace-producing portion). Channels
/// live for `'static` via `Box::leak` (test-only; a real inferlet keeps them on
/// its stack).
fn build_s3() -> Traced {
    let ctr1: &'static Tensor = leak(Tensor::constant([0u32, 1]));
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32).named("tok"));
    let indptr: &'static Channel = leak(Channel::from([0u32, 1]).named("indptr"));
    let out: &'static Channel = leak(Channel::new([1], dtype::i32).named("out"));
    let mask: &'static Channel = leak(Channel::new([VOCAB], dtype::bool).named("mask"));
    let len: &'static Channel = leak(Channel::from([1u32]).named("len"));
    let rng_ch: &'static Channel = leak(Channel::from([7u32, 0]).named("rng"));

    // seed token -> cell full
    tok.put([1i32]);

    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr);
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

    // prime mask_0 before the first submit so `mask` has a producer.
    mask.put(initial_mask());
    b.build()
        .expect("greedy-decode must build to a validated container")
}

#[test]
fn s3_traces_and_validates() {
    let traced = build_s3();

    let c = traced.container();
    assert_eq!(c.stages.len(), 1, "one epilogue stage");
    assert_eq!(c.stages[0].stage, Stage::Epilogue);
    assert_eq!(c.channels.len(), 6, "tok/indptr/out/mask/len/rng");
    assert_eq!(
        c.channels[1].host_role,
        pie_dsl::ptir::container::HostRole::Writer,
        "a seeded descriptor-only channel supports device-visible host set"
    );
    let puts = c.stages[0]
        .ops
        .iter()
        .filter(|op| matches!(op, Op::ChanPut { .. }))
        .count();
    assert_eq!(puts, 4, "rng, tok, len, out puts");
}

#[test]
fn s3_identity_hash_is_stable() {
    let a = build_s3().identity_hash();
    let b = build_s3().identity_hash();
    assert_eq!(a, b, "the same program hashes identically");
    assert_eq!(
        a, GOLDEN_S3,
        "byte-identical to the channel-only descriptor golden"
    );
}

#[test]
fn different_program_hashes_differently() {
    let greedy = build_s3().identity_hash();

    let tok: &'static Channel = leak(Channel::new([1], dtype::i32));
    let indptr: &'static Channel = leak(Channel::from([0u32, 1]));
    let rng_ch: &'static Channel = leak(Channel::from([7u32, 0]));
    tok.put([1i32]);
    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr);
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
// lint set — error-message snapshot tests
// ---------------------------------------------------------------------------

#[test]
fn lint_double_endpoint_host_both_ends() {
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32));
    let indptr: &'static Channel = leak(Channel::from([0u32, 1]));
    // `dup` is claimed by the host as BOTH writer and reader (no pass endpoint
    // remains — SPSC violation). It is also consumed by the epilogue so it
    // enters the trace container.
    let dup: &'static Channel = leak(Channel::new([1], dtype::i32).named("dup"));
    tok.put([1i32]);
    dup.put([0i32]); // host writes
    dup.note_host_take(); // host also consumes

    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr);
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
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32));
    let indptr: &'static Channel = leak(Channel::from([0u32, 1]));
    let orphan: &'static Channel = leak(Channel::new([1], dtype::i32).named("orphan"));
    tok.put([1i32]);

    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr);
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
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32));
    let indptr: &'static Channel = leak(Channel::from([0u32, 1]));
    let budget: &'static Channel = leak(Channel::from([256u32]));
    tok.put([1i32]);

    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr);
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

/// The pass-wide `lora` sink is prologue-only (T11): an attn-proj placement is
/// legal for an attention-scoped sink but not for one the whole forward
/// consumes, so the pre-lint must refuse it before bind even sees it.
#[test]
fn lint_lora_sink_misplacement_outside_prologue() {
    let la: &'static Channel = leak(Channel::from_shaped([2u32, 2, 4], vec![0.0f32; 16]));
    let lb: &'static Channel = leak(Channel::from_shaped([2u32, 4, 2], vec![0.0f32; 16]));

    let mut b = Builder::new(VOCAB, PAGE);
    b.stage(Stage::OnAttnProj, move || {
        intrinsics::kernel::lora(la.read(), lb.read(), Tensor::constant([0u32, 1]));
    });

    let err = b
        .build()
        .expect_err("a pass-wide sink at attn-proj must fail");
    let msg = err.to_string();
    assert!(
        err.0
            .iter()
            .any(|e| matches!(e, TraceError::SinkMisplacement { .. })),
        "expected a SinkMisplacement, got:\n{msg}"
    );
    assert!(msg.contains("lora"), "message:\n{msg}");
}

// ---------------------------------------------------------------------------
// beam search (the second exit gate): reorder = gathers,
// divergence = freeze. Exercises the full op set (top_k, log_softmax, gather,
// scatter_set, reshape, iota, broadcast, div/rem/mul/sub, lt/and/eq, cast).
// No auto-drain synthesis — the loop-carried peek-port channels
// (klen/kvm) drain EXPLICITLY (`take()` directly before the re-put), which
// reproduces the same ops verbatim (same golden hash).
// ---------------------------------------------------------------------------

#[test]
fn s6_2_beam_epilogue_binds() {
    const B: u32 = 2;
    const V: u32 = 8;
    const P: u32 = 3;
    const PAGE_T: u32 = 4;

    // channels 0..=15 for the beam-search epilogue.
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

    // host-fed headroom (slot grants are per-instance data,
    // `fresh.put(ws.alloc(B))`), primed before submit.
    fresh.put(vec![0u32; B as usize]);

    let lanes_b = leak(Channel::from((0u32..=B).collect::<Vec<_>>())); // [0,1,2]
    let page_rows = leak(Channel::from((0u32..=B).map(|i| i * P).collect::<Vec<_>>())); // [0,P,2P]

    let mut b = Builder::new(V, PAGE_T);
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
        let tok_i = cast(&tok_u, pie_dsl::DType::I32);
        toks.put(&tok_i);
        scores.put(&s);
        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
    });

    let traced = b.build().expect("beam epilogue must bind");
    let c = traced.container();
    assert_eq!(c.stages[0].stage, Stage::Epilogue);
    assert_eq!(
        c.channels.len(),
        18,
        "16 state channels + 2 descriptor channels"
    );
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
        pie_dsl::ptir::container::decode(&traced.encode()).expect("decode beam container");
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
    use pie_dsl::ptir::container::HostRole;
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

// native-MTP + grammar spec-verify binds (the mtp-grammar inferlet's trace).
// Grammar mask BEFORE the argmax → grammar-legal
// picks; accept-prefix = leading run of picked[0..K] == argmax(mtp_logits).
#[test]
fn s6_1_mtp_grammar_binds() {
    const V: u32 = 8;
    const K: u32 = 3;
    let kp1 = K + 1;
    let gmask: &'static Channel = leak(Channel::new([kp1, V], dtype::bool).named("gmask"));
    let toks: &'static Channel = leak(Channel::from(vec![1i32; kp1 as usize]).named("toks"));
    let out: &'static Channel = leak(Channel::new([kp1], dtype::i32).named("out"));
    // gmask is host-fed each step (per-position grammar mask) — a host-side put
    // marks it host-writer + produces its value (mirrors the beam's `fresh.put`).
    gmask.put(vec![true; (kp1 * V) as usize]);
    let lanes = leak(Channel::from((0u32..=kp1).collect::<Vec<_>>()));

    let mut b = Builder::new(V, 4);
    b.bind_port(Port::EmbedTokens, toks);
    b.bind_port(Port::EmbedIndptr, lanes);
    b.stage(Stage::Epilogue, move || {
        let masked = mask_apply(intrinsics::logits(), gmask.take()); // [K+1, V]
        let picked = reduce_argmax(&masked); // [K+1] grammar-constrained target
        // NATIVE MTP: K distinct draft heads [K, V] (the K-vs-K+1 contract).
        let mtp = intrinsics::mtp_logits(K); // [K, V]
        let draft = reduce_argmax(&mtp); // [K]
        // mtp_verify_tail: head = picked[0..K]; accept-prefix = leading run of matches.
        let head = gather(&picked, iota(K)); // [K]
        let hit = eq(&head, &draft); // [K] bool
        let ones = broadcast(Tensor::constant(1.0f32), [K]);
        let zeros = broadcast(Tensor::constant(0.0f32), [K]);
        let run = cumprod(select(&hit, &ones, &zeros)); // [K]
        let nacc = cast(reduce_sum(&run), pie_dsl::DType::U32); // accepted-prefix length
        let keep = ge(broadcast(&nacc, [kp1]), iota(kp1)); // [K+1]
        let neg1 = broadcast(Tensor::constant(-1i32), [kp1]);
        let commit = select(&keep, &picked, &neg1); // accept-prefix + -1 sentinels
        out.put(&commit);
    });
    let traced = b.build().expect("mtp-grammar epilogue must bind");
    assert_eq!(
        traced.identity_hash(),
        GOLDEN_MTP_GRAMMAR,
        "byte-identical to the channel-only descriptor golden"
    );
}

// ---------------------------------------------------------------------------
// put auto-drains a peeked descriptor port
// ---------------------------------------------------------------------------

/// A channel bound to a port whose discipline is *read* keeps its cell full
/// across the descriptor phase, so a loop-carried re-put has to drain the stale
/// value or the ring grows by one every fire. `put` does that itself: the
/// author writes the same call whichever side of `Port::consumes` the port
/// falls on, and cannot leak occupancy by forgetting which.
#[test]
fn put_drains_a_peeked_port_and_leaves_a_consuming_one_alone() {
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32).named("tok"));
    let indptr: &'static Channel = leak(Channel::from([0u32, 1]).named("indptr"));
    // Peeked (`consumes()` false): the descriptor reads it, so `put` drains.
    let pidx: &'static Channel = leak(Channel::from([0u32, 1]).named("pidx"));
    // Consuming (`consumes()` true): the descriptor drains it, so `put` fills.
    let pos: &'static Channel = leak(Channel::from([0u32]).named("pos"));
    // Peeked, but drained EXPLICITLY — the author's take is honoured, not doubled.
    let len: &'static Channel = leak(Channel::from([1u32]).named("len"));
    tok.put([1i32]);

    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr);
    b.bind_port(Port::PageIndptr, pidx);
    b.bind_port(Port::Positions, pos);
    b.bind_port(Port::KvLen, len);
    b.stage(Stage::Epilogue, move || {
        let t = reduce_argmax(intrinsics::logits());
        tok.put(t);
        pidx.put(Tensor::constant([0u32, 1]));
        pos.put(Tensor::constant([0u32]));
        len.put(len.take() + 1u32);
    });

    let traced = b.build().expect("must build");
    let c = traced.container();
    let order = traced.channel_order();
    let dense = |ch: &Channel| {
        order
            .iter()
            .position(|gid| *gid == ch.gid())
            .expect("channel interned") as u32
    };
    let takes: Vec<u32> = c.stages[0]
        .ops
        .iter()
        .filter_map(|op| match op {
            Op::ChanTake(chan) => Some(*chan),
            _ => None,
        })
        .collect();

    assert_eq!(
        takes.iter().filter(|c| **c == dense(pidx)).count(),
        1,
        "a peeked port's re-put must drain exactly once: {takes:?}"
    );
    assert_eq!(
        takes.iter().filter(|c| **c == dense(pos)).count(),
        0,
        "a consuming port already drains; put must not add a take: {takes:?}"
    );
    assert_eq!(
        takes.iter().filter(|c| **c == dense(len)).count(),
        1,
        "an explicit take must not be doubled by put: {takes:?}"
    );

    // The drain precedes the put it makes room for.
    let pos_of =
        |pred: &dyn Fn(&Op) -> bool| c.stages[0].ops.iter().position(|op| pred(op)).expect("op");
    assert!(
        pos_of(&|op| matches!(op, Op::ChanTake(c) if *c == dense(pidx)))
            < pos_of(&|op| matches!(op, Op::ChanPut { chan, .. } if *chan == dense(pidx))),
        "the drain must precede its put"
    );
}

// ---------------------------------------------------------------------------
// div_ceil / indptr lower to exactly what they replaced
// ---------------------------------------------------------------------------

/// Both helpers are pure spelling: they must emit the SAME ops as the
/// arithmetic the guests wrote by hand, or migrating 81 call sites would move
/// the container bytes.
#[test]
fn div_ceil_and_indptr_match_the_arithmetic_they_replace() {
    const PAGE: u32 = 16;

    fn ops_of(body: impl Fn() + 'static) -> Vec<Op> {
        let tok: &'static Channel = leak(Channel::new([1], dtype::i32));
        let indptr_ch: &'static Channel = leak(Channel::from([0u32, 1]));
        tok.put([1i32]);
        let mut b = Builder::new(VOCAB, PAGE);
        b.bind_port(Port::EmbedTokens, tok);
        b.bind_port(Port::EmbedIndptr, indptr_ch);
        b.stage(Stage::Epilogue, move || {
            body();
            tok.put(reshape(reduce_argmax(intrinsics::logits()), [1]));
        });
        b.build().expect("must build").container().stages[0]
            .ops
            .clone()
    }

    let by_hand = ops_of(|| {
        let len = Tensor::constant([7u32]);
        let _ = (&len + (PAGE - 1)) / PAGE;
    });
    let by_helper = ops_of(|| {
        let len = Tensor::constant([7u32]);
        let _ = len.div_ceil(PAGE);
    });
    assert_eq!(by_hand, by_helper, "div_ceil must not change the lowering");

    let by_hand = ops_of(|| {
        let count = Tensor::constant([3u32]);
        let _ = iota(2) * broadcast(&count, [2]);
    });
    let by_helper = ops_of(|| {
        let count = Tensor::constant([3u32]);
        let _ = indptr(1, &count);
    });
    assert_eq!(by_hand, by_helper, "indptr must not change the lowering");
}
