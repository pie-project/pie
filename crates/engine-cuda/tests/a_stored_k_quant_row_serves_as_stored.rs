use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, ops,
    trace_hybrid,
};
use model_ir::Trace;

const VOCAB: u32 = 1024;

const HIDDEN: u64 = 512;

const Q4_K: Dtype = Dtype::U4g32k;

const Q6_K: Dtype = Dtype::I6g16k;

struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Stored,
    Decoded,
}

struct Micro {
    embed: Weight,
    proj: Weight,
    head: Weight,
}

impl Micro {
    fn new(arm: Arm) -> Micro {
        let (proj, head) = match arm {
            Arm::Stored => (Q4_K, Q6_K),
            Arm::Decoded => (Dtype::Bf16, Dtype::Bf16),
        };
        Micro {
            embed: Weight::sym("embed", [u64::from(VOCAB), HIDDEN], Dtype::Bf16),
            proj: Weight::sym("proj", [HIDDEN, HIDDEN], proj),
            head: Weight::sym("lm_head", [u64::from(VOCAB), HIDDEN], head),
        }
    }
}

impl ForwardHybrid for Micro {
    type Facts = NoFacts;

    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let x = ops::layout::embed(&inputs.tokens(), &self.embed, VOCAB);
        let h = ops::linear::matmul(&x, &self.proj);
        ops::linear::lm_head(&h, &self.head)
    }
}

fn trace(arm: Arm) -> (Micro, Trace) {
    let m = Micro::new(arm);
    let trace = trace_hybrid("kquant-micro", &m, Platform::Cuda);
    (m, trace)
}

#[test]
fn a_stored_declaration_interns_one_byte_rectangle() {
    let (_, stored) = trace(Arm::Stored);
    let plane = |name: &str| {
        stored
            .params
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the trace interns `{name}`"))
    };
    assert_eq!(plane("proj").shape, vec![HIDDEN, 288]);
    assert_eq!(plane("proj").dtype, Q4_K);
    assert_eq!(plane("lm_head").shape, vec![u64::from(VOCAB), 420]);
    assert_eq!(plane("lm_head").dtype, Q6_K);
    assert_eq!(stored.params.len(), 3, "three planes for three weights");

    let (_, decoded) = trace(Arm::Decoded);
    assert_eq!(plane_of(&decoded, "proj").shape, vec![HIDDEN, HIDDEN]);
    assert_eq!(plane_of(&decoded, "proj").dtype, Dtype::Bf16);
}

fn plane_of<'a>(trace: &'a Trace, name: &str) -> &'a model_ir::Param {
    trace
        .params
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("the trace interns `{name}`"))
}
