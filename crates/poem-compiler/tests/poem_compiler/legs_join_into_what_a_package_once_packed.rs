//! A model stating q, k and v apart (each with its bias) and a gate and an
//! up apart is, once its products of one activation are joined, the model
//! stating them packed: one product by the joined weight, its bias added
//! once, split after; one packed gated unit.

use poem::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Value, Weight, ops, trace_hybrid};
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

    fn forward(&self, inputs: Input) -> Value {
        let x = ops::gather::embed(&inputs.tokens(), &self.embed, VOCAB);
        let q = ops::elemwise::add_bias(&self.q_bias, &ops::linear::matmul(&x, &self.q));
        let k = ops::elemwise::add_bias(&self.k_bias, &ops::linear::matmul(&x, &self.k));
        let v = ops::elemwise::add_bias(&self.v_bias, &ops::linear::matmul(&x, &self.v));
        let h = ops::elemwise::residual_add(&ops::elemwise::residual_add(&q, &k), &v);
        let gate = ops::linear::matmul(&h, &self.gate);
        let up = ops::linear::matmul(&h, &self.up);
        let f = ops::linear::glu(&gate, poem_ir::GluAct::Swiglu, None, Some(&up), None, None);
        ops::linear::lm_head(&f, &self.head)
    }
}

impl ForwardHybrid for Packed {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) -> Value {
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
        ops::linear::lm_head(&f, &self.head)
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

    fn forward(&self, inputs: Input) -> Value {
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
        ops::linear::lm_head(&y, &self.head)
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

const QKV: u64 = 96;
const Z: u64 = 128;
const HEADS: u64 = 32;

/// A GDN mixer's projections, apart or packed: q, k and v and z, which a
/// seam marks the rows of, and beta and alpha, which its gates read side by
/// side.
struct Gdn {
    embed: Weight,
    legs: Option<[Weight; 4]>,
    packed: Option<[Weight; 2]>,
    head: Weight,
}

impl ForwardHybrid for Gdn {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) -> Value {
        use ops::linear::matmul;
        let x = ops::gather::embed(&inputs.tokens(), &self.embed, VOCAB);
        let (qkv, z, ba) = match (&self.legs, &self.packed) {
            (Some([qkv, z, b, a]), None) => {
                let (qkv, z) = (matmul(&x, qkv), matmul(&x, z));
                x.rec().seam("recurrent", &[&qkv, &z]);
                let ba = ops::gather::concat_cols(&[&matmul(&x, b), &matmul(&x, a)]);
                (qkv, z, ba)
            }
            (None, Some([qkvz, ba])) => {
                let qkvz = matmul(&x, qkvz);
                x.rec().seam("recurrent", &[&qkvz]);
                let (qkv, z) = ops::gather::split_rows(&qkvz, QKV as u32);
                (qkv, z, matmul(&x, ba))
            }
            _ => unreachable!("a micro states its projections one way"),
        };
        let h = ops::elemwise::residual_add(&ops::gather::concat_cols(&[&qkv, &ba]), &z);
        ops::linear::lm_head(&ops::linear::matmul(&h, &self.embed), &self.head)
    }
}

/// GDN's q, k and v and its z, a seam over both, join into the one product
/// split back by rows that a package once stated packed; its beta and alpha,
/// read side by side, into the one product their concat was.
#[test]
fn a_seam_and_a_concat_join_their_legs() {
    let w = |name: &str, rows: u64| weight(name, rows).columns();
    let apart = Gdn {
        embed: embed(),
        legs: Some([w("qkv", QKV), w("z", Z), w("b", HEADS / 2), w("a", HEADS / 2)]),
        packed: None,
        head: head(),
    };
    let packed = Gdn {
        embed: embed(),
        legs: None,
        packed: Some([
            weight("qkvz", QKV + Z).packed([QKV, Z]),
            weight("ba", HEADS).packed([HEADS / 2, HEADS / 2]),
        ]),
        head: head(),
    };
    let joined = poem_compiler::pack::pack(trace_hybrid("gdn", &apart, Platform::Cuda));
    let want = trace_hybrid("gdn", &packed, Platform::Cuda);
    assert_eq!(canon::difference(&want, &joined), None);
    let joins = poem_compiler::pack::joins(&trace_hybrid("gdn", &apart, Platform::Cuda));
    let names: Vec<&str> = joins.iter().map(|j| j.name.as_str()).collect();
    assert_eq!(names, ["qkvz", "ba"]);
}

/// q, k, v and an output gate of one activation: four legs join into one
/// product, the gate peeled off by rows and q, k and v split after.
struct Gated {
    embed: Weight,
    legs: Option<[Weight; 4]>,
    packed: Option<Weight>,
    head: Weight,
}

impl ForwardHybrid for Gated {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) -> Value {
        use ops::linear::matmul;
        let x = ops::gather::embed(&inputs.tokens(), &self.embed, VOCAB);
        let (q, k, v, g) = match (&self.legs, &self.packed) {
            (Some([q, k, v, g]), None) => (matmul(&x, q), matmul(&x, k), matmul(&x, v), matmul(&x, g)),
            (None, Some(qkvg)) => {
                let (qkv, g) = ops::gather::split_rows(&matmul(&x, qkvg), (Q + 2 * KV) as u32);
                let (q, k, v) = ops::gather::split_qkv(&qkv, Q as u32, KV as u32);
                (q, k, v, g)
            }
            _ => unreachable!("a micro states its projections one way"),
        };
        let kv = ops::gather::concat_cols(&[&k, &v, &k, &v]);
        let h = ops::elemwise::residual_add(&ops::elemwise::residual_add(&q, &g), &kv);
        ops::linear::lm_head(&h, &self.head)
    }
}

#[test]
fn four_legs_peel_their_last_and_split_the_rest() {
    let w = |name: &str, rows: u64, heads: u64| weight(name, rows).columns().heads([heads]);
    let apart = Gated {
        embed: embed(),
        legs: Some([w("q", Q, 4), w("k", KV, 1), w("v", KV, 1), w("g", Q, 4)]),
        packed: None,
        head: head(),
    };
    let packed = Gated {
        embed: embed(),
        legs: None,
        packed: Some(
            weight("qkvg", 2 * Q + 2 * KV)
                .packed([Q, KV, KV, Q])
                .heads([4, 1, 1, 4]),
        ),
        head: head(),
    };
    let joined = poem_compiler::pack::pack(trace_hybrid("gated", &apart, Platform::Cuda));
    let want = trace_hybrid("gated", &packed, Platform::Cuda);
    assert_eq!(canon::difference(&want, &joined), None);
}

/// A gate and an up a count of ranks holds whole, which one rank's dataflow
/// cuts by their columns, join whole as one rank's would join them cut.
#[test]
fn legs_held_whole_join_whole() {
    struct Mlp {
        embed: Weight,
        gate: Option<Weight>,
        up: Option<Weight>,
        gate_up: Option<Weight>,
        head: Weight,
    }
    impl ForwardHybrid for Mlp {
        fn caches(&self) -> HybridSpec {
            HybridSpec::new()
        }
        fn forward(&self, inputs: Input) -> Value {
            use ops::linear::{glu, matmul};
            let x = ops::gather::embed(&inputs.tokens(), &self.embed, VOCAB);
            let act = match (&self.gate, &self.up, &self.gate_up) {
                (Some(gate), Some(up), None) => glu(
                    &matmul(&x, gate),
                    poem_ir::GluAct::Swiglu,
                    None,
                    Some(&matmul(&x, up)),
                    None,
                    None,
                ),
                (None, None, Some(gate_up)) => glu(
                    &matmul(&x, gate_up),
                    poem_ir::GluAct::Swiglu,
                    Some(INTER as u32),
                    None,
                    None,
                    None,
                ),
                _ => unreachable!("a micro states its mlp one way"),
            };
            ops::linear::lm_head(&act, &self.head)
        }
    }
    let apart = Mlp {
        embed: embed(),
        gate: Some(weight("gate", INTER)),
        up: Some(weight("up", INTER)),
        gate_up: None,
        head: head(),
    };
    let traced = trace_hybrid("mlp", &apart, Platform::Cuda);
    // One rank's dataflow cuts both by their columns; the ranks hold them
    // whole.
    let shape: Vec<poem_ir::Param> = traced
        .params
        .iter()
        .map(|p| match p.name.as_str() {
            "gate" | "up" => poem_ir::Param {
                shard: poem_ir::Shard::Cut {
                    axis: 0,
                    segments: vec![INTER],
                    heads: Vec::new(),
                },
                ..p.clone()
            },
            _ => p.clone(),
        })
        .collect();
    let joined = poem_compiler::pack::pack_as(traced.clone(), &shape);
    let packed = Mlp {
        embed: embed(),
        gate: None,
        up: None,
        gate_up: Some(weight("gate_up", 2 * INTER)),
        head: head(),
    };
    let want = trace_hybrid("mlp", &packed, Platform::Cuda);
    assert_eq!(canon::difference(&want, &joined), None);
    // Stated whole and so cut by no dataflow, they stay apart.
    assert!(poem_compiler::pack::joins(&traced).is_empty());
}

/// A routed gate's and up's banks, each with a bias a row an expert, join
/// with their biases into the biased bank a package once stated.
#[test]
fn biased_routed_legs_join_with_their_biases() {
    struct Biased {
        embed: Weight,
        router: Weight,
        legs: Option<[Weight; 4]>,
        packed: Option<[Weight; 2]>,
        down: Weight,
        head: Weight,
    }
    impl ForwardHybrid for Biased {
        fn caches(&self) -> HybridSpec {
            HybridSpec::new()
        }
        fn forward(&self, inputs: Input) -> Value {
            use ops::linear::{glu, matmul, moe_matmul_select, moe_route, moe_weighted_sum};
            let x = ops::gather::embed(&inputs.tokens(), &self.embed, VOCAB);
            let (routes, weights) = moe_route(
                &matmul(&x, &self.router),
                EXPERTS as u32,
                TOP_K,
                poem_ir::RouteScore::Softmax,
                None,
                None,
                None,
                None,
                None,
            );
            let act = match (&self.legs, &self.packed) {
                (Some([gate, gate_bias, up, up_bias]), None) => glu(
                    &moe_matmul_select(&x, gate, &routes, TOP_K, Some(gate_bias)),
                    poem_ir::GluAct::Swiglu,
                    None,
                    Some(&moe_matmul_select(&x, up, &routes, TOP_K, Some(up_bias))),
                    None,
                    None,
                ),
                (None, Some([gate_up, bias])) => glu(
                    &moe_matmul_select(&x, gate_up, &routes, TOP_K, Some(bias)),
                    poem_ir::GluAct::Swiglu,
                    Some(INTER as u32),
                    None,
                    None,
                    None,
                ),
                _ => unreachable!("a micro states its banks one way"),
            };
            let routed = moe_matmul_select(&act, &self.down, &routes, TOP_K, None);
            ops::linear::lm_head(&moe_weighted_sum(&routed, &weights), &self.head)
        }
    }
    let bank = |name: &str, rows: u64| {
        Weight::sym(name, [EXPERTS, rows, HIDDEN], Dtype::Bf16).bank([rows])
    };
    let bias = |name: &str, rows: u64| Weight::sym(name, [EXPERTS, rows], Dtype::Bf16).bank([rows]);
    let micro = |legs, packed| Biased {
        embed: embed(),
        router: Weight::sym("router", [EXPERTS, HIDDEN], Dtype::Bf16),
        legs,
        packed,
        down: Weight::sym("down", [EXPERTS, HIDDEN, INTER], Dtype::Bf16).bank([HIDDEN]),
        head: head(),
    };
    let apart = micro(
        Some([bank("gate", INTER), bias("gate_bias", INTER), bank("up", INTER), bias("up_bias", INTER)]),
        None,
    );
    let packed = micro(
        None,
        Some([
            Weight::sym("gate_up", [EXPERTS, 2 * INTER, HIDDEN], Dtype::Bf16).bank([INTER, INTER]),
            Weight::sym("gate_up_bias", [EXPERTS, 2 * INTER], Dtype::Bf16).bank([INTER, INTER]),
        ]),
    );
    let joined = poem_compiler::pack::pack(trace_hybrid("biased", &apart, Platform::Cuda));
    let want = trace_hybrid("biased", &packed, Platform::Cuda);
    assert_eq!(canon::difference(&want, &joined), None);
    let joins = poem_compiler::pack::joins(&trace_hybrid("biased", &apart, Platform::Cuda));
    let names: Vec<(&str, u32)> = joins.iter().map(|j| (j.name.as_str(), j.axis)).collect();
    assert_eq!(names, [("gate_up", 1), ("gate_up_bias", 1)]);
}
