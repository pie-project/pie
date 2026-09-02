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

use eta_dsl::builder::Builder;
use eta_dsl::prelude::*;
use eta_dsl::{Channel, TraceError, Traced};

const VOCAB: u32 = 32_000;
const PAGE: u32 = 16;

// Golden C3 identity hashes (FNV-1a over the canonical container bytes). These
// LOCK byte-identity of the channel-only descriptor lowering: a change here
// means the emitted container bytes moved.
//
// Re-pinned once, for the `PTIR` -> `ETA` rename: `ETA_MAGIC` went from
// `*b"PTIR"` to `*b"ETA\0"`, still four bytes and still the container's first
// four, so every hash below moved and nothing else in the lowering did. The
// old values were 7015106236045798467 (s3), 16590305409635560083 (beam) and
// 12706234719755930498 (mtp-grammar); today's encoder cannot produce them
// again, which is why they are written down here rather than only in git.
const GOLDEN_S3: u64 = 4213522552817221928;

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
fn s3_identity_hash_is_stable() {
    let a = build_s3().identity_hash();
    let b = build_s3().identity_hash();
    assert_eq!(a, b, "the same program hashes identically");
    assert_eq!(
        a, GOLDEN_S3,
        "byte-identical to the channel-only descriptor golden"
    );
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

// ---------------------------------------------------------------------------
// beam search (the second exit gate): reorder = gathers,
// divergence = freeze. Exercises the full op set (top_k, log_softmax, gather,
// scatter_set, reshape, iota, broadcast, div/rem/mul/sub, lt/and/eq, cast).
// No auto-drain synthesis — the loop-carried peek-port channels
// (klen/kvm) drain EXPLICITLY (`take()` directly before the re-put), which
// reproduces the same ops verbatim (same golden hash).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// put auto-drains a peeked descriptor port
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// div_ceil / indptr lower to exactly what they replaced
// ---------------------------------------------------------------------------

