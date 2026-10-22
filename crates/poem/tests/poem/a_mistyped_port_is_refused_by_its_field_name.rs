//! An op's operand types are stated on its fields (`#[output(I32)]`), and
//! the validator names the field it refuses.

use poem::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Value, Weight, ops, trace_hybrid};
use poem_ir::check::Port;
use poem_ir::{Fault, Operation, Sampling, Trace, Ty};

const HIDDEN: u64 = 64;

const VOCAB: u32 = 128;

struct Ranked;

impl ForwardHybrid for Ranked {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) -> Value {
        let table = Weight::sym("embed", [u64::from(VOCAB), HIDDEN], Dtype::Bf16);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let (values, _) = ops::sampling::top_k(&x, 4);
        values
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
                port: Port::Named("indices"),
                want: Dtype::I32,
                ..
            }
        )),
        "{faults:?}"
    );
}
