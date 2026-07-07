//! P1 exit tests: the overview §3 greedy-decode pipeline compiles to a validated
//! trace and hashes stably, plus error-message snapshot tests for the lint set
//! (double-endpoint, readiness-direction conflict, sink misplacement).
//!
//! Idiom note: values reused as op operands take `&` (a taken value is used at
//! multiple sites); a value used once is moved. See the crate-doc deviations.

use ptir::prelude::*;
use ptir::ptir::op::Op;
use ptir::{Channel, Stage, TraceError};

const VOCAB: u32 = 32_000;
const PAGE: u32 = 16;
const LAYERS: u32 = 32;

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// Host stub: the root grammar mask (a `[vocab]` bool). Real matcher is host code.
fn initial_mask() -> Vec<bool> {
    vec![true; VOCAB as usize]
}

// ---------------------------------------------------------------------------
// overview §3 — greedy + grammar-masked decode, software-pipelined
// ---------------------------------------------------------------------------

/// Build the §3 forward pass verbatim (the trace-producing portion). We only
/// trace + validate here; the async host loop lands with the channel store (P3).
fn build_s3() -> ptir::ForwardPass<'static> {
    // Channels live for 'static via Box::leak so the returned pass owns nothing
    // borrowed (test-only; a real inferlet keeps them on its stack).
    let ctr1: &'static Tensor = Box::leak(Box::new(Tensor::constant([0u32, 1])));
    let tok: &'static Channel = Box::leak(Box::new(Channel::new([1], dtype::i32).named("tok")));
    let out: &'static Channel = Box::leak(Box::new(Channel::new([1], dtype::i32).named("out")));
    let mask: &'static Channel =
        Box::leak(Box::new(Channel::new([intrinsics::vocab()], dtype::bool).named("mask")));
    let len: &'static Channel = Box::leak(Box::new(Channel::from([1u32]).named("len")));
    let rng_ch: &'static Channel = Box::leak(Box::new(Channel::from([7u32, 0]).named("rng")));

    // seed token -> cell full
    let bos: i32 = 1;
    tok.put([bos]);

    let ws: &'static WorkingSet = Box::leak(Box::new(WorkingSet::new()));
    const MAX_TOKENS: u32 = 8;
    ws.alloc(div_ceil(1 + MAX_TOKENS, ws.page_size()));

    let fwd = ForwardPass::new();
    let lane_1 = Tensor::constant([0u32, 1]);
    fwd.embed(tok, lane_1);
    fwd.attn_working_set(ws, len);
    fwd.epilogue(move || {
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
    fwd
}

#[test]
fn s3_traces_and_validates() {
    ptir::model::configure(VOCAB, PAGE, LAYERS);
    let fwd = build_s3();
    let traced = fwd.trace().expect("§3 must trace to a validated container");

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
    ptir::model::configure(VOCAB, PAGE, LAYERS);
    let a = build_s3().trace().unwrap().identity_hash();
    let b = build_s3().trace().unwrap().identity_hash();
    assert_eq!(a, b, "the same program hashes identically (C3)");
    assert_ne!(a, 0);
}

#[test]
fn s3_trace_once_memoizes() {
    ptir::model::configure(VOCAB, PAGE, LAYERS);
    let fwd = build_s3();
    let t1 = fwd.trace().unwrap();
    let t2 = fwd.trace().unwrap();
    assert!(std::rc::Rc::ptr_eq(&t1, &t2), "closure traced once, then cached");
}

#[test]
fn different_program_hashes_differently() {
    ptir::model::configure(VOCAB, PAGE, LAYERS);
    let greedy = build_s3().trace().unwrap().identity_hash();

    let tok: &'static Channel = Box::leak(Box::new(Channel::new([1], dtype::i32)));
    let rng_ch: &'static Channel = Box::leak(Box::new(Channel::from([7u32, 0])));
    tok.put([1i32]);
    let fwd = ForwardPass::new();
    fwd.embed(tok, Tensor::constant([0u32, 1]));
    fwd.epilogue(move || {
        let logits = intrinsics::logits();
        let r = rng_ch.take();
        let scaled = mul(logits, 2.0f32); // temperature != greedy
        let g = gumbel(&r, [intrinsics::vocab()]);
        let t = reduce_argmax(add(scaled, g));
        rng_ch.put(add(&r, Tensor::constant([0u32, 1])));
        tok.put(t);
    });
    let other = fwd.trace().unwrap().identity_hash();
    assert_ne!(greedy, other, "different op graph => different identity");
}

// ---------------------------------------------------------------------------
// lint set — error-message snapshot tests (P1.3)
// ---------------------------------------------------------------------------

#[test]
fn lint_double_endpoint_host_both_ends() {
    ptir::model::configure(VOCAB, PAGE, LAYERS);
    let tok: &'static Channel = Box::leak(Box::new(Channel::new([1], dtype::i32)));
    // `dup` is claimed by the host as BOTH writer and reader (no pass endpoint
    // remains — SPSC violation). It is also consumed by the epilogue so it
    // enters the trace container.
    let dup: &'static Channel = Box::leak(Box::new(Channel::new([1], dtype::i32).named("dup")));
    tok.put([1i32]);
    dup.put([0i32]); // host writes
    let _ = dup.take(); // host also consumes

    let fwd = ForwardPass::new();
    fwd.embed(tok, Tensor::constant([0u32, 1]));
    fwd.epilogue(move || {
        let v = dup.take(); // pass consumes it too (so `dup` is interned)
        tok.put(add(&v, reduce_argmax(intrinsics::logits())));
    });

    let err = fwd.trace().expect_err("host-both-endpoints must fail");
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
    ptir::model::configure(VOCAB, PAGE, LAYERS);
    let tok: &'static Channel = Box::leak(Box::new(Channel::new([1], dtype::i32)));
    let orphan: &'static Channel =
        Box::leak(Box::new(Channel::new([1], dtype::i32).named("orphan")));
    tok.put([1i32]);

    let fwd = ForwardPass::new();
    fwd.embed(tok, Tensor::constant([0u32, 1]));
    fwd.epilogue(move || {
        let v = orphan.take();
        let _ = intrinsics::logits();
        tok.put(add(&v, 1u32));
    });

    let err = fwd.trace().expect_err("consuming an unproduced channel must fail");
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
    ptir::model::configure(VOCAB, PAGE, LAYERS);
    let tok: &'static Channel = Box::leak(Box::new(Channel::new([1], dtype::i32)));
    let budget: &'static Channel = Box::leak(Box::new(Channel::from([256u32])));
    tok.put([1i32]);

    let fwd = ForwardPass::new();
    fwd.embed(tok, Tensor::constant([0u32, 1]));
    fwd.epilogue(move || {
        let logits = intrinsics::logits();
        let mask = pivot_threshold(&logits, rank_le(budget.read()));
        intrinsics::kernel::attn_page_mask(mask);
        tok.put(reduce_argmax(&logits));
    });

    let err = fwd.trace().expect_err("sink at epilogue must fail");
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
// Verbatim: the tracer auto-drains pure-derivative device channels (klen/kvm)
// that are put-without-take (overview elides the drain; the tracer injects it).
// ---------------------------------------------------------------------------

#[test]
fn s6_2_beam_epilogue_binds() {
    const B: u32 = 2;
    const V: u32 = 8;
    const P: u32 = 3;
    const PAGE_T: u32 = 4;
    ptir::model::configure(V, PAGE_T, 2);

    // channels 0..=15 as in overview §6.2 / echo's beam_trace.
    let pages: &'static Channel = Box::leak(Box::new(Channel::seeded([B, P], dtype::u32).named("pages")));
    let lens: &'static Channel = Box::leak(Box::new(Channel::seeded([B, P], dtype::u32).named("lens")));
    let klen: &'static Channel = Box::leak(Box::new(Channel::from(vec![0u32; B as usize]).named("klen")));
    let kvm: &'static Channel = Box::leak(Box::new(Channel::seeded([B, P * PAGE_T], dtype::bool).named("kvm")));
    let pos: &'static Channel = Box::leak(Box::new(Channel::from(vec![0u32; B as usize]).named("pos")));
    let np: &'static Channel = Box::leak(Box::new(Channel::from(vec![1u32; B as usize]).named("np")));
    let tslot: &'static Channel = Box::leak(Box::new(Channel::from(vec![0u32; B as usize]).named("tslot")));
    let tfill: &'static Channel = Box::leak(Box::new(Channel::from(vec![0u32; B as usize]).named("tfill")));
    let w_slot: &'static Channel = Box::leak(Box::new(Channel::from(vec![0u32; B as usize]).named("w_slot")));
    let w_off: &'static Channel = Box::leak(Box::new(Channel::from(vec![0u32; B as usize]).named("w_off")));
    let toks: &'static Channel = Box::leak(Box::new(Channel::from(vec![1i32; B as usize]).named("toks")));
    let scores: &'static Channel = Box::leak(Box::new(Channel::from(vec![0.0f32; B as usize]).named("scores")));
    let fresh: &'static Channel = Box::leak(Box::new(Channel::new([B], dtype::u32).named("fresh")));
    let out: &'static Channel = Box::leak(Box::new(Channel::new([B], dtype::i32).named("out")));
    let out_par: &'static Channel = Box::leak(Box::new(Channel::new([B], dtype::u32).named("out_par")));
    let out_scr: &'static Channel = Box::leak(Box::new(Channel::new([B], dtype::f32).named("out_scr")));

    let ws: &'static WorkingSet = Box::leak(Box::new(WorkingSet::new()));

    // host-fed headroom + descriptor geometry, primed before submit.
    fresh.put(ws.alloc(B));

    let fwd = ForwardPass::new();
    let lanes_b = Tensor::constant((0u32..=B).collect::<Vec<_>>()); // [0,1,2] indptr
    let page_rows = Tensor::constant((0u32..=B).map(|i| i * P).collect::<Vec<_>>()); // [0,P,2P]
    fwd.embed(toks, lanes_b);
    fwd.positions(pos);
    fwd.attn_working_set(ws, (pages, page_rows, klen, w_slot, w_off));
    fwd.attn_mask(kvm);
    fwd.epilogue(move || {
        let cand = add(broadcast(reshape(scores.take(), [B, 1]), [B, V]), log_softmax(intrinsics::logits()));
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
        pages.put(reshape(scatter_set(reshape(pg, [B * P]), &tcol, &slot), [B, P]));
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
        let tok_u = rem(&i, V);
        let tok_i = cast(&tok_u, ptir::DType::I32);
        toks.put(&tok_i);
        scores.put(&s);
        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
    });

    let traced = fwd.trace().expect("§6.2 beam epilogue must bind");
    let c = traced.container();
    assert_eq!(c.stages[0].stage, Stage::Epilogue);
    assert_eq!(c.channels.len(), 16, "16 beam channels");
    assert_ne!(traced.identity_hash(), 0);

    // Regression (G2 fire-0 seed round-trip): channel 0 (`pages`) is [B,P] (2D).
    // The [B,P] shape MUST survive encode→decode, else `validate_seeds` rejects
    // the [B,P] seed as a byte-length mismatch (numel collapse).
    assert_eq!(c.channels[0].shape.numel(), (B * P) as u64, "pages [B,P] numel in built container");
    let decoded = pie_sampling_ir::ptir::container::decode(&traced.encode())
        .expect("decode beam container");
    assert_eq!(decoded.channels[0].shape.dims(), &[B, P], "pages 2D dims survive encode->decode");
    assert_eq!(
        decoded.channels[0].shape.numel(),
        (B * P) as u64,
        "pages [B,P] numel after encode->decode"
    );

    // host_role (fix #3): out/out_par/out_scr are terminal program outputs (prog-put,
    // no program/descriptor consumer) → inferred host Reader so the guest's `take`
    // is accepted; fresh (host-put headroom) is a Writer.
    use ptir::ptir::container::HostRole;
    assert_eq!(decoded.channels[13].host_role, HostRole::Reader, "out (13) is host-Reader");
    assert_eq!(decoded.channels[14].host_role, HostRole::Reader, "out_par (14) is host-Reader");
    assert_eq!(decoded.channels[15].host_role, HostRole::Reader, "out_scr (15) is host-Reader");
    assert_eq!(decoded.channels[12].host_role, HostRole::Writer, "fresh (12) is host-Writer");
}

// overview §6.1 — native-MTP + grammar spec-verify binds (the M3-G1 §6.1 pass:
// `mtp-grammar` inferlet's trace). Grammar mask BEFORE the argmax → grammar-legal
// picks; accept-prefix = leading run of picked[0..K] == argmax(mtp_logits).
#[test]
fn s6_1_mtp_grammar_binds() {
    const V: u32 = 8;
    const K: u32 = 3;
    ptir::model::configure(V, 4, 2);
    ptir::model::configure_gates(true, false); // has_mtp_logits
    let kp1 = K + 1;
    let gmask: &'static Channel = Box::leak(Box::new(Channel::new([kp1, V], dtype::bool).named("gmask")));
    let toks: &'static Channel = Box::leak(Box::new(Channel::from(vec![1i32; kp1 as usize]).named("toks")));
    let out: &'static Channel = Box::leak(Box::new(Channel::new([kp1], dtype::i32).named("out")));
    // gmask is host-fed each step (per-position grammar mask) — a host-side put
    // marks it host-writer + produces its value (mirrors the beam's `fresh.put`).
    gmask.put(vec![true; (kp1 * V) as usize]);
    let fwd = ForwardPass::new();
    let lanes = Tensor::constant((0u32..=kp1).collect::<Vec<_>>());
    fwd.embed(toks, lanes);
    fwd.epilogue(move || {
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
        let nacc = cast(reduce_sum(&run), ptir::DType::U32); // accepted-prefix length
        let keep = ge(broadcast(&nacc, [kp1]), iota(kp1)); // [K+1]
        let neg1 = broadcast(Tensor::constant(-1i32), [kp1]);
        let commit = select(&keep, &picked, &neg1); // accept-prefix + -1 sentinels
        out.put(&commit);
    });
    let traced = fwd.trace().expect("§6.1 mtp-grammar epilogue must bind");
    assert_ne!(traced.identity_hash(), 0);
}
