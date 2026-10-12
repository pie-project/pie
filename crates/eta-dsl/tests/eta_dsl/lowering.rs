use eta_dsl::builder::Builder;
use eta_dsl::prelude::*;
use eta_dsl::{Channel, TraceError, Traced};

const VOCAB: u32 = 32_000;
const PAGE: u32 = 16;

const GOLDEN_S3: u64 = 4213522552817221928;

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

fn initial_mask() -> Vec<bool> {
    vec![true; VOCAB as usize]
}

fn build_s3() -> Traced {
    let ctr1: &'static Tensor = leak(Tensor::constant([0u32, 1]));
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32).named("tok"));
    let indptr: &'static Channel = leak(Channel::from([0u32, 1]).named("indptr"));
    let out: &'static Channel = leak(Channel::new([1], dtype::i32).named("out"));
    let mask: &'static Channel = leak(Channel::new([VOCAB], dtype::bool).named("mask"));
    let len: &'static Channel = leak(Channel::from([1u32]).named("len"));
    let rng_ch: &'static Channel = leak(Channel::from([7u32, 0]).named("rng"));

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

    mask.put(initial_mask());
    b.build()
        .expect("greedy-decode must build to a validated container")
}

#[test]
fn lowering_every_case() {
    s3_identity_hash_is_stable();
    lint_double_endpoint_host_both_ends();
}

fn s3_identity_hash_is_stable() {
    let a = build_s3().identity_hash();
    let b = build_s3().identity_hash();
    assert_eq!(a, b, "the same program hashes identically");
    assert_eq!(
        a, GOLDEN_S3,
        "byte-identical to the channel-only descriptor golden"
    );
}

fn lint_double_endpoint_host_both_ends() {
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32));
    let indptr: &'static Channel = leak(Channel::from([0u32, 1]));
    let dup: &'static Channel = leak(Channel::new([1], dtype::i32).named("dup"));
    tok.put([1i32]);
    dup.put([0i32]);
    dup.note_host_take();

    let mut b = Builder::new(VOCAB, PAGE);
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr);
    b.stage(Stage::Epilogue, move || {
        let v = dup.take();
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
