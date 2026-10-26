//! An op's operand types are stated on its fields (`#[output(I32)]`), and
//! the validator names the field it refuses.

use poem::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Weight, ops, trace_hybrid};
use poem_ir::Turn;
use poem_ir::check::Port;
use poem_ir::{Fault, Operation, Sampling, Trace, Ty};

const HIDDEN: u64 = 64;

const VOCAB: u32 = 128;

struct Ranked;

impl ForwardHybrid for Ranked {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) {
        let table = Weight::sym("embed", [u64::from(VOCAB), HIDDEN], Dtype::Bf16);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let (values, _) = ops::sampling::top_k(&x, 4);
        poem::output("logits", &values, None);
    }
}

fn traced() -> Trace {
    trace_hybrid("ranked", &Ranked, Platform::Cuda)
}

#[test]
fn a_mistyped_port_is_refused_by_its_field_name() {
    let mut trace = traced();
    assert_eq!(poem_ir::check(&trace), Ok(()));

    let indices = trace
        .nodes
        .iter()
        .find_map(|node| match &node.op {
            Operation::Sampling(Sampling::TopK { indices, .. }) => Some(*indices),
            _ => None,
        })
        .expect("the trace holds its top-k");
    let Ty::Tensor { dtype, .. } = &mut trace.values[indices.0 as usize].ty else {
        panic!("top-k's indices are a tensor");
    };
    *dtype = Dtype::F32;

    let faults = poem_ir::check(&trace).expect_err("f32 indices are refused");
    assert!(
        faults.iter().any(|fault| matches!(
            fault,
            Fault::PortDtype {
                port: Port("indices"),
                want: Dtype::I32,
                ..
            }
        )),
        "{faults:?}"
    );
}

struct Turned;

impl ForwardHybrid for Turned {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) {
        let table = Weight::sym("embed", [u64::from(VOCAB), HIDDEN], Dtype::Bf16);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let logits = ops::elemwise::rope(
            &x,
            &inputs.positions(),
            32,
            1e4,
            Turn::Partial {
                rotary_dim: 16,
                span: 16 / 2,
            },
            None,
            None,
        )
        .0;
        poem::output("logits", &logits, None);
    }
}

/// The folded rotation names its positions port too, whichever turn it takes.
#[test]
fn a_rotation_at_f32_positions_is_refused_by_its_field_name() {
    let mut trace = trace_hybrid("turned", &Turned, Platform::Cuda);
    assert_eq!(poem_ir::check(&trace), Ok(()));
    let positions = trace
        .nodes
        .iter()
        .find_map(|node| match &node.op {
            Operation::Elemwise(poem_ir::Elemwise::Rope { positions, .. }) => Some(*positions),
            _ => None,
        })
        .expect("the trace holds its rotation");
    let Ty::Tensor { dtype, .. } = &mut trace.values[positions.0 as usize].ty else {
        panic!("positions are a tensor");
    };
    *dtype = Dtype::F32;
    let faults = poem_ir::check(&trace).expect_err("f32 positions are refused");
    assert!(
        faults.iter().any(|fault| matches!(
            fault,
            Fault::PortDtype {
                port: Port("positions"),
                want: Dtype::I32,
                ..
            }
        )),
        "{faults:?}"
    );
}

/// The folded lookup names its ids port, whichever form it takes.
#[test]
fn a_lookup_of_f32_ids_is_refused_by_its_field_name() {
    let mut trace = traced();
    let ids = trace
        .nodes
        .iter()
        .find_map(|node| match &node.op {
            Operation::Gather(poem_ir::Gather::Embed { ids, .. }) => Some(*ids),
            _ => None,
        })
        .expect("the trace holds its lookup");
    let Ty::Tensor { dtype, .. } = &mut trace.values[ids.0 as usize].ty else {
        panic!("ids are a tensor");
    };
    *dtype = Dtype::F32;
    let faults = poem_ir::check(&trace).expect_err("f32 ids are refused");
    assert!(
        faults.iter().any(|fault| matches!(
            fault,
            Fault::PortDtype {
                port: Port("ids"),
                want: Dtype::I32,
                ..
            }
        )),
        "{faults:?}"
    );
}
