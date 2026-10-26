//! A model stating q, k and v apart (each with its bias) and a gate and an
//! up apart is, once its products of one activation are joined, the model
//! stating them packed: one product by the joined weight, its bias added
//! once, split after; one packed gated unit.

use poem::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Weight, ops, trace_hybrid};
use poem_ir::canon;

const VOCAB: u32 = 64;
const HIDDEN: u64 = 128;
const Q: u64 = 128;
const KV: u64 = 32;
const INTER: u64 = 256;

fn weight(name: &str, rows: u64) -> Weight {
    Weight::sym(name, [rows, HIDDEN], Dtype::Bf16)
}

fn bias(name: &str, rows: u64, heads: u64) -> Weight {
    Weight::sym(name, [rows], Dtype::Bf16)
        .columns()
        .heads([heads])
}

struct Apart {
    embed: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    q_bias: Weight,
    k_bias: Weight,
    v_bias: Weight,
    gate: Weight,
    up: Weight,
    head: Weight,
}

struct Packed {
    embed: Weight,
    qkv: Weight,
    qkv_bias: Weight,
    gate_up: Weight,
    head: Weight,
}

fn embed() -> Weight {
    Weight::sym("embed", [u64::from(VOCAB), HIDDEN], Dtype::Bf16)
}

fn head() -> Weight {
    Weight::sym("head", [u64::from(VOCAB), HIDDEN], Dtype::Bf16)
}

impl ForwardHybrid for Apart {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) {
        let x = ops::gather::embed(&inputs.tokens(), &self.embed, VOCAB);
        let q = ops::elemwise::add_bias(&self.q_bias, &ops::linear::matmul(&x, &self.q));
        let k = ops::elemwise::add_bias(&self.k_bias, &ops::linear::matmul(&x, &self.k));
        let v = ops::elemwise::add_bias(&self.v_bias, &ops::linear::matmul(&x, &self.v));
        let h = ops::elemwise::residual_add(&ops::elemwise::residual_add(&q, &k), &v);
        let gate = ops::linear::matmul(&h, &self.gate);
        let up = ops::linear::matmul(&h, &self.up);
        let f = ops::linear::glu(&gate, poem_ir::GluAct::Swiglu, None, Some(&up), None, None);
        let logits = ops::linear::lm_head(&f, &self.head);
        poem::output("logits", &logits, None);
    }
}

impl ForwardHybrid for Packed {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) {
        let x = ops::gather::embed(&inputs.tokens(), &self.embed, VOCAB);
        let qkv = ops::elemwise::add_bias(&self.qkv_bias, &ops::linear::matmul(&x, &self.qkv));
        let (q, k, v) = ops::gather::split_qkv(&qkv, Q as u32, KV as u32);
        let h = ops::elemwise::residual_add(&ops::elemwise::residual_add(&q, &k), &v);
        let packed = ops::linear::matmul(&h, &self.gate_up);
        let f = ops::linear::glu(
            &packed,
            poem_ir::GluAct::Swiglu,
            Some(INTER as u32),
            None,
            None,
            None,
        );
        let logits = ops::linear::lm_head(&f, &self.head);
        poem::output("logits", &logits, None);
    }
}

#[test]
fn legs_join_into_what_a_package_once_packed() {
    let apart = Apart {
        embed: embed(),
        q: weight("q", Q).columns().heads([4]),
        k: weight("k", KV).columns().heads([1]),
        v: weight("v", KV).columns().heads([1]),
        q_bias: bias("q_bias", Q, 4),
        k_bias: bias("k_bias", KV, 1),
        v_bias: bias("v_bias", KV, 1),
        gate: weight("gate", INTER).columns(),
        up: weight("up", INTER).columns(),
        head: head(),
    };
    let packed = Packed {
        embed: embed(),
        qkv: weight("qkv", Q + 2 * KV)
            .packed([Q, KV, KV])
            .heads([4, 1, 1]),
        qkv_bias: Weight::sym("qkv_bias", [Q + 2 * KV], Dtype::Bf16)
            .packed([Q, KV, KV])
            .heads([4, 1, 1]),
        gate_up: weight("gate_up", 2 * INTER).packed([INTER, INTER]),
        head: head(),
    };
    let joined = poem_compiler::pack::pack(trace_hybrid("micro", &apart, Platform::Cuda));
    let want = trace_hybrid("micro", &packed, Platform::Cuda);
    assert_eq!(canon::difference(&want, &joined), None);
    let joins = poem_compiler::pack::joins(&trace_hybrid("micro", &apart, Platform::Cuda));
    let names: Vec<&str> = joins.iter().map(|j| j.name.as_str()).collect();
    assert_eq!(names, ["qkv", "qkv_bias", "gate_up"]);
}

const EXPERTS: u64 = 8;
const TOP_K: u32 = 2;

struct Banks {
    embed: Weight,
    router: Weight,
    gate: Option<Weight>,
    up: Option<Weight>,
    gate_up: Option<Weight>,
    down: Weight,
    head: Weight,
}

impl ForwardHybrid for Banks {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) {
        use ops::linear::{glu, matmul, moe_matmul_select, moe_route, moe_weighted_sum};
        let x = ops::gather::embed(&inputs.tokens(), &self.embed, VOCAB);
        let logits = matmul(&x, &self.router);
        let (routes, weights) = moe_route(
            &logits,
            EXPERTS as u32,
            TOP_K,
            poem_ir::RouteScore::Softmax,
            None,
            None,
            None,
            None,
            None,
        );
        let act = match (&self.gate, &self.up, &self.gate_up) {
            (Some(gate), Some(up), None) => {
                let gate = moe_matmul_select(&x, gate, &routes, TOP_K, None);
                let up = moe_matmul_select(&x, up, &routes, TOP_K, None);
                glu(&gate, poem_ir::GluAct::Swiglu, None, Some(&up), None, None)
            }
            (None, None, Some(gate_up)) => {
                let packed = moe_matmul_select(&x, gate_up, &routes, TOP_K, None);
                glu(
                    &packed,
                    poem_ir::GluAct::Swiglu,
                    Some(INTER as u32),
                    None,
                    None,
                    None,
                )
            }
            _ => unreachable!("a micro states its banks one way"),
        };
        let routed = moe_matmul_select(&act, &self.down, &routes, TOP_K, None);
        let y = moe_weighted_sum(&routed, &weights);
        let logits = ops::linear::lm_head(&y, &self.head);
        poem::output("logits", &logits, None);
    }
}

/// A gate's bank and an up's selected over one route, apart, join into the
/// packed bank a package once stated.
#[test]
fn routed_legs_join_into_one_bank() {
    let bank = |name: &str, rows: u64| Weight::sym(name, [EXPERTS, rows, HIDDEN], Dtype::Bf16);
    let banks = |gate: Option<Weight>, up: Option<Weight>, gate_up: Option<Weight>| Banks {
        embed: embed(),
        router: Weight::sym("router", [EXPERTS, HIDDEN], Dtype::Bf16),
        gate,
        up,
        gate_up,
        down: Weight::sym("down", [EXPERTS, HIDDEN, INTER], Dtype::Bf16).bank([HIDDEN]),
        head: head(),
    };
    let apart = banks(
        Some(bank("gate", INTER).bank([INTER])),
        Some(bank("up", INTER).bank([INTER])),
        None,
    );
    let packed = banks(
        None,
        None,
        Some(bank("gate_up", 2 * INTER).bank([INTER, INTER])),
    );
    let joined = poem_compiler::pack::pack(trace_hybrid("moe", &apart, Platform::Cuda));
    let want = trace_hybrid("moe", &packed, Platform::Cuda);
    assert_eq!(canon::difference(&want, &joined), None);
    let joins = poem_compiler::pack::joins(&trace_hybrid("moe", &apart, Platform::Cuda));
    assert_eq!(joins.len(), 1);
    assert_eq!((joins[0].name.as_str(), joins[0].axis), ("gate_up", 1));
}
