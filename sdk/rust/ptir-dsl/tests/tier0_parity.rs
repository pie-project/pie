//! Tier-0 result-parity tests (the semantic correctness gate): the SDK-emitted
//! §3 / greedy container, bound and run on echo's reference interpreter
//! (`ptir::interp`, `eval` feature), yields the SAME TOKEN RESULTS as echo's
//! golden `greedy_argmax` / `section3_masked_gumbel` fixtures — and encode→decode
//! round-trips. This is the correctness contract (NOT hash-equality with echo's
//! hand-built containers; emission order may differ, results may not).
//!
//! The guest does not bind (D6): [`Builder::build`] lowers + lints only, and
//! these native parity tests bind explicitly against a test profile (the same
//! validator `forward-pass.new` runs host-side).

use pie_ptir::container;
use pie_ptir::interp::Value;
use pie_ptir::interp::{HostError, Instance, NoKernels, PassInputs};
use pie_ptir::validate::{bind, BoundTrace};

use ptir_dsl::builder::Builder;
use ptir_dsl::prelude::*;
use ptir_dsl::{Channel, Traced};

/// Bind a lowered [`Traced`] against the current model profile (native parity
/// only — host-side this is `forward-pass.new`'s job).
fn bound(traced: &Traced) -> BoundTrace {
    bind(traced.container().clone(), ptir_dsl::model::profile()).expect("container binds")
}

/// Dense channel index of a named channel.
fn idx(names: &[String], name: &str) -> u32 {
    names.iter().position(|n| n == name).unwrap_or_else(|| panic!("no channel `{name}`")) as u32
}

fn logits(v: Vec<f32>) -> PassInputs {
    PassInputs { logits: Some(Value::F32(v)), ..Default::default() }
}

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

// ---------------------------------------------------------------------------
// greedy_argmax (VOCAB=8): argmax(logits) -> token. Golden tokens: 2, then 0.
// ---------------------------------------------------------------------------

#[test]
fn greedy_argmax_tier0_matches_golden() {
    ptir_dsl::model::configure(8, 4, 2);
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32).named("tok"));
    let out: &'static Channel = leak(Channel::new([1], dtype::i32).named("out"));
    tok.put([1i32]); // seed BOS

    let mut b = Builder::new();
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, Tensor::constant([0u32, 1]));
    b.stage(Stage::Epilogue, move || {
        let t = reduce_argmax(intrinsics::logits());
        tok.put(&t);
        out.put(t);
    });
    let _ = out.take(); // host-reader signal (marks `out` HostRole::Reader)

    let traced = b.build().expect("greedy builds");
    let bound = bound(&traced);
    let names = traced.channel_names();
    let (tok_i, out_i) = (idx(names, "tok"), idx(names, "out"));

    // encode -> decode round-trips stably.
    let bytes = traced.encode();
    assert_eq!(container::decode(&bytes).unwrap(), *traced.container());

    let mut inst = Instance::new(&bound, &[(tok_i, Value::I32(vec![1]))]).unwrap();

    let r = inst.step(&bound, &logits(vec![0., 1., 9., 2., 0., 0., 0., 3.]), &mut NoKernels).unwrap();
    assert!(r.committed, "step 0 commits");
    assert_eq!(inst.host_take(&bound, out_i).unwrap(), Value::I32(vec![2]), "golden token 2");

    inst.step(&bound, &logits(vec![7., 1., 0., 2., 0., 0., 0., 3.]), &mut NoKernels).unwrap();
    assert_eq!(inst.host_take(&bound, out_i).unwrap(), Value::I32(vec![0]), "golden token 0");
}

// ---------------------------------------------------------------------------
// section3 (VOCAB=32): masked gumbel-greedy. Golden: token 7, late-mask miss
// (WouldBlock), recover to token 3.
// ---------------------------------------------------------------------------

const V32: u32 = 32;

fn build_section3() -> Traced {
    // Channels live for 'static (test-only) so the pass owns nothing borrowed.
    let ctr1: &'static Tensor = leak(Tensor::constant([0u32, 1]));
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32).named("tok"));
    let out: &'static Channel = leak(Channel::new([1], dtype::i32).named("out"));
    let mask: &'static Channel = leak(Channel::new([intrinsics::vocab()], dtype::bool).named("mask"));
    let len: &'static Channel = leak(Channel::from([1u32]).named("len"));
    let rng: &'static Channel = leak(Channel::from([1234u32, 0]).named("rng"));

    tok.put([1i32]); // seed BOS

    let mut b = Builder::new();
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, Tensor::constant([0u32, 1]));
    b.bind_port(Port::KvLen, len);
    b.stage(Stage::Epilogue, move || {
        let logits = intrinsics::logits();
        let r = rng.take();
        let g = gumbel(&r, [intrinsics::vocab()]);
        let t = reduce_argmax(add(mask_apply(logits, mask.take()), g));
        rng.put(add(&r, ctr1));
        tok.put(&t);
        len.put(add(len.take(), 1u32));
        out.put(t);
    });
    // Host-endpoint signals: mask is host-written each step, out is host-read
    // (the §3 host-loop `mask.put(..)` / `out.take()` — mark Writer/Reader).
    mask.put(vec![true; V32 as usize]);
    let _ = out.take();
    b.build().expect("section3 builds")
}

#[test]
fn section3_tier0_matches_golden() {
    ptir_dsl::model::configure(V32, 16, 32);
    let traced = build_section3();
    let bound = bound(&traced);
    let names = traced.channel_names();
    let (tok_i, out_i, mask_i, len_i, rng_i) = (
        idx(names, "tok"),
        idx(names, "out"),
        idx(names, "mask"),
        idx(names, "len"),
        idx(names, "rng"),
    );

    // encode -> decode round-trips stably.
    assert_eq!(container::decode(&traced.encode()).unwrap(), *traced.container());

    // Per-instance seeds (D2): tok=BOS, len=1, rng=[1234,0] (echo's golden seeds).
    let mut inst = Instance::new(
        &bound,
        &[
            (tok_i, Value::I32(vec![1])),
            (len_i, Value::U32(vec![1])),
            (rng_i, Value::U32(vec![1234, 0])),
        ],
    )
    .unwrap();

    // Logit 100 at index 7 (mask-/logit-dominated so gumbel doesn't flip it).
    let peaked = {
        let mut l = vec![0.0f32; V32 as usize];
        l[7] = 100.0;
        l
    };
    let allow_all = vec![true; V32 as usize];
    let allow_only_3 = {
        let mut m = vec![false; V32 as usize];
        m[3] = true;
        m
    };

    // step 0: mask=allow_all -> token 7.
    inst.host_put(&bound, mask_i, Value::Bool(allow_all)).unwrap();
    let r0 = inst.step(&bound, &logits(peaked.clone()), &mut NoKernels).unwrap();
    assert!(r0.committed, "step 0 commits");
    assert_eq!(inst.host_take(&bound, out_i).unwrap(), Value::I32(vec![7]), "golden token 7");

    // step 1: mask consumed + not re-fed -> readiness miss (dummy-run), out empty.
    let r1 = inst.step(&bound, &logits(peaked.clone()), &mut NoKernels).unwrap();
    assert!(!r1.committed, "step 1 is a late-mask miss");
    assert_eq!(inst.host_take(&bound, out_i), Err(HostError::WouldBlock), "no token on the miss");

    // step 2: mask=allow_only([3]) -> recover to token 3.
    inst.host_put(&bound, mask_i, Value::Bool(allow_only_3)).unwrap();
    let r2 = inst.step(&bound, &logits(peaked), &mut NoKernels).unwrap();
    assert!(r2.committed, "step 2 recovers");
    assert_eq!(inst.host_take(&bound, out_i).unwrap(), Value::I32(vec![3]), "golden token 3");
}
