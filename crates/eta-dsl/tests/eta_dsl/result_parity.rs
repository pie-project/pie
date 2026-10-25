use eta_compiler::eval::interp::Value;
use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs};
use eta_ir::container;
use eta_ir::registry::{EngineCaps, Graph, Readout, ReadoutKind};
use eta_ir::validate::BoundTrace;

use eta_dsl::builder::Builder;
use eta_dsl::prelude::*;
use eta_dsl::{Channel, Traced};

fn bound(traced: &Traced, vocab: u32, _page_size: u32, num_layers: u32) -> BoundTrace {
    served(traced.container().clone(), logits_lane(vocab), num_layers).expect("container binds")
}

fn served(
    container: eta_ir::container::TraceContainer,
    lanes: Vec<Option<Readout>>,
    num_layers: u32,
) -> Result<eta_ir::validate::BoundTrace, eta_ir::validate::ValidateError> {
    eta_ir::validate::bind(
        container,
        lanes,
        Graph {
            num_layers,
            ..Graph::default()
        },
        EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
}

fn logits_lane(vocab: u32) -> Vec<Option<Readout>> {
    vec![Some(Readout {
        kind: ReadoutKind::Logits,
        width: vocab,
    })]
}

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

#[test]
fn greedy_argmax_tier0_matches_golden() {
    let tok: &'static Channel = leak(Channel::new([1], dtype::i32).named("tok"));
    let indptr: &'static Channel = leak(Channel::from([0u32, 1]).named("indptr"));
    let klen: &'static Channel = leak(Channel::from([1u32]).named("klen"));
    let positions: &'static Channel = leak(Channel::from([0u32]).named("positions"));
    let out: &'static Channel = leak(Channel::new([1], dtype::i32).named("out"));
    tok.put([1i32]);

    let mut b = Builder::new(8, 4);
    b.bind_port(Port::EmbedTokens, tok);
    b.bind_port(Port::EmbedIndptr, indptr);
    b.bind_port(Port::KvLen, klen);
    b.bind_port(Port::Positions, positions);
    b.stage(Stage::Epilogue, move || {
        let t = reduce_argmax(intrinsics::logits());
        tok.put(&t);
        positions.put(Tensor::constant([0u32]));
        out.put(t);
    });
    out.note_host_take();

    let traced = b.build().expect("greedy builds");
    let bound = bound(&traced, 8, 4, 2);
    let names = traced.channel_names();
    let (tok_i, indptr_i, klen_i, positions_i, out_i) = (
        idx(names, "tok"),
        idx(names, "indptr"),
        idx(names, "klen"),
        idx(names, "positions"),
        idx(names, "out"),
    );

    let bytes = traced.encode();
    assert_eq!(container::decode(&bytes).unwrap(), *traced.container());

    let mut inst = Instance::new(
        &bound,
        &[
            (tok_i, Value::I32(vec![1])),
            (indptr_i, Value::U32(vec![0, 1])),
            (klen_i, Value::U32(vec![1])),
            (positions_i, Value::U32(vec![0])),
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
