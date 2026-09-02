//! Result parity with the IR's own fixtures — the semantic correctness gate.
//!
//! The SDK-emitted
//! greedy container, bound and run on the IR's reference interpreter
//! (`eta::interp`, `eval` feature), yields the SAME TOKEN RESULTS as the IR's
//! golden `greedy_argmax` / `section3_masked_gumbel` fixtures — and encode→decode
//! round-trips. This is the correctness contract (NOT hash-equality with the IR's
//! hand-built containers; emission order may differ, results may not).
//!
//! The guest does not bind: [`Builder::build`] lowers + lints only, and
//! these native parity tests bind explicitly against a test profile (the same
//! validator `forward-pass.new` runs host-side).

use eta_compiler::eval::interp::Value;
use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs};
use eta_ir::container;
use eta_ir::registry::ModelProfile;
use eta_ir::validate::{BoundTrace, bind};

use eta_dsl::builder::Builder;
use eta_dsl::prelude::*;
use eta_dsl::{Channel, Traced};

/// Bind a lowered [`Traced`] against the current model profile (native parity
/// only — host-side this is `forward-pass.program`'s job).
fn bound(traced: &Traced, vocab: u32, page_size: u32, num_layers: u32) -> BoundTrace {
    let profile = ModelProfile {
        vocab,
        page_size,
        num_layers,
        ..ModelProfile::dummy()
    };
    bind(traced.container().clone(), profile).expect("container binds")
}

/// Dense channel index of a named channel.
fn idx(names: &[String], name: &str) -> u32 {
    names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("no channel `{name}`")) as u32
}

fn logits(v: Vec<f32>) -> PassInputs {
    PassInputs {
        logits: Some(Value::F32(v)),
        ..Default::default()
    }
}

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

// ---------------------------------------------------------------------------
// greedy_argmax (VOCAB=8): argmax(logits) -> token. Golden tokens: 2, then 0.
// ---------------------------------------------------------------------------

#[test]
fn greedy_argmax_tier0_matches_golden() {
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32).named("tok"));
    let indptr: &'static Channel = leak(Channel::from([0u32, 1]).named("indptr"));
    let klen: &'static Channel = leak(Channel::from([1u32]).named("klen"));
    let positions: &'static Channel = leak(Channel::from([0u32]).named("positions"));
    let pages: &'static Channel = leak(Channel::from([0u32]).named("pages"));
    let page_indptr: &'static Channel = leak(Channel::from([0u32, 1]).named("page_indptr"));
    let w_slot: &'static Channel = leak(Channel::from([0u32]).named("w_slot"));
    let w_off: &'static Channel = leak(Channel::from([0u32]).named("w_off"));
    let out: &'static Channel = leak(Channel::new([1], dtype::i32).named("out"));
    tok.put([1i32]); // seed BOS

    let mut b = Builder::new(8, 4);
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr);
    b.bind_port(Port::KvLen, klen);
    b.bind_port(Port::Positions, positions);
    b.bind_port(Port::Pages, pages);
    b.bind_port(Port::PageIndptr, page_indptr);
    b.bind_port(Port::WSlot, w_slot);
    b.bind_port(Port::WOff, w_off);
    b.stage(Stage::Epilogue, move || {
        let t = reduce_argmax(intrinsics::logits());
        tok.put(&t);
        positions.put(Tensor::constant([0u32]));
        w_slot.put(Tensor::constant([0u32]));
        w_off.put(Tensor::constant([0u32]));
        out.put(t);
    });
    out.note_host_take(); // host-reader signal (marks `out` HostRole::Reader)

    let traced = b.build().expect("greedy builds");
    let bound = bound(&traced, 8, 4, 2);
    let names = traced.channel_names();
    let (tok_i, indptr_i, klen_i, positions_i, pages_i, page_indptr_i, w_slot_i, w_off_i, out_i) = (
        idx(names, "tok"),
        idx(names, "indptr"),
        idx(names, "klen"),
        idx(names, "positions"),
        idx(names, "pages"),
        idx(names, "page_indptr"),
        idx(names, "w_slot"),
        idx(names, "w_off"),
        idx(names, "out"),
    );

    // encode -> decode round-trips stably.
    let bytes = traced.encode();
    assert_eq!(container::decode(&bytes).unwrap(), *traced.container());

    let mut inst = Instance::new(
        &bound,
        &[
            (tok_i, Value::I32(vec![1])),
            (indptr_i, Value::U32(vec![0, 1])),
            (klen_i, Value::U32(vec![1])),
            (positions_i, Value::U32(vec![0])),
            (pages_i, Value::U32(vec![0])),
            (page_indptr_i, Value::U32(vec![0, 1])),
            (w_slot_i, Value::U32(vec![0])),
            (w_off_i, Value::U32(vec![0])),
        ],
    )
    .unwrap();

    let r = inst
        .step(
            &bound,
            &logits(vec![0., 1., 9., 2., 0., 0., 0., 3.]),
            &mut NoKernels,
        )
        .unwrap();
    assert!(r.committed, "step 0 commits");
    assert_eq!(
        inst.host_take(&bound, out_i).unwrap(),
        Value::I32(vec![2]),
        "golden token 2"
    );

    inst.step(
        &bound,
        &logits(vec![7., 1., 0., 2., 0., 0., 0., 3.]),
        &mut NoKernels,
    )
    .unwrap();
    assert_eq!(
        inst.host_take(&bound, out_i).unwrap(),
        Value::I32(vec![0]),
        "golden token 0"
    );
}

// ---------------------------------------------------------------------------
// section3 (VOCAB=32): masked gumbel-greedy. Golden: token 7, late-mask miss
// (WouldBlock), recover to token 3.
// ---------------------------------------------------------------------------

