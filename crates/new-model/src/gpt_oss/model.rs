//! The gpt-oss declaration, de-genericized (design §5, decision #18): the old
//! `Model<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize>` phantom tree is
//! gone — `tp` is a runtime field, each weight carries its `Repr`, and the kv
//! dtype is nobody's parameter. The mxfp4 expert banks are ordinary `Weight`s
//! whose repr interns a packed codes plane and an `.scales` companion behind
//! one name (the declare surface's bank planes). Names and the per-layer
//! scheme are unchanged from the old crate: weights intern by name, so the
//! checkpoint mapping carries over untouched.

use new_model_dsl::{CacheRef, Norm, Weight, Yarn};
use new_model_ir::Repr;

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,
    pub embed: Weight,
    pub head: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Norm,
}

pub struct Layer {
    pub attn: Attn,
    pub attn_norm: Norm,
    pub mlp_norm: Norm,
    pub mlp: Moe,
}

pub struct Attn {
    pub window: Option<u32>,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub sm_scale: f32,
    pub rope: Yarn,
    pub q_proj: Weight,
    pub q_bias: Weight,
    pub k_proj: Weight,
    pub k_bias: Weight,
    pub v_proj: Weight,
    pub v_bias: Weight,
    pub o_proj: Weight,
    pub o_bias: Weight,
    pub sinks: Weight,
    pub kv: CacheRef,
}

pub struct Moe {
    pub experts: u32,
    pub top_k: u32,
    pub inter: u32,
    pub swiglu_limit: f32,
    pub swiglu_alpha: f32,
    pub router: Weight,
    pub router_bias: Weight,
    pub gate_up: Weight,
    pub gate_up_bias: Weight,
    pub down: Weight,
    pub down_bias: Weight,
}

struct Dims {
    hidden: u32,
    layers: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    theta: f32,
    yarn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_attention_factor: f32,
    yarn_original_max_position: u32,
    window: u32,
    experts: u32,
    top_k: u32,
    inter: u32,
    swiglu_limit: f32,
    swiglu_alpha: f32,
    vocab: u32,
    norm_eps: f32,
}

fn per_rank(d: Dims, tp: u32) -> Dims {
    let cut = |what, whole| new_model_dsl::per_rank(what, whole, tp as usize);
    Dims {
        q_heads: cut("q_heads", d.q_heads),
        kv_heads: cut("kv_heads", d.kv_heads),
        inter: cut("inter", d.inter),
        ..d
    }
}

impl Model {
    pub fn b20(w1: Repr, w2: Repr, tp: u32) -> Model {
        assemble(
            w1,
            w2,
            tp,
            Dims {
                hidden: 2880,
                layers: 24,
                q_heads: 64,
                kv_heads: 8,
                head_dim: 64,
                theta: 150_000.0,
                yarn_factor: 32.0,
                yarn_beta_fast: 32.0,
                yarn_beta_slow: 1.0,
                yarn_attention_factor: 1.346_573_6,
                yarn_original_max_position: 4096,
                window: 128,
                experts: 32,
                top_k: 4,
                inter: 2880,
                swiglu_limit: 7.0,
                swiglu_alpha: 1.702,
                vocab: 201_088,
                norm_eps: 1e-5,
            },
        )
    }

    pub fn b120(w1: Repr, w2: Repr, tp: u32) -> Model {
        assemble(
            w1,
            w2,
            tp,
            Dims {
                hidden: 2880,
                layers: 36,
                q_heads: 64,
                kv_heads: 8,
                head_dim: 64,
                theta: 150_000.0,
                yarn_factor: 32.0,
                yarn_beta_fast: 32.0,
                yarn_beta_slow: 1.0,
                yarn_attention_factor: 1.346_573_6,
                yarn_original_max_position: 4096,
                window: 128,
                experts: 128,
                top_k: 4,
                inter: 2880,
                swiglu_limit: 7.0,
                swiglu_alpha: 1.702,
                vocab: 201_088,
                norm_eps: 1e-5,
            },
        )
    }
}

fn assemble(w1: Repr, w2: Repr, tp: u32, d: Dims) -> Model {
    let d = per_rank(d, tp);
    let hidden = d.hidden as u64;
    let hd = d.head_dim as u64;
    let q_w = d.q_heads as u64 * hd;
    let kv_w = d.kv_heads as u64 * hd;
    let experts = d.experts as u64;
    let inter = d.inter as u64;
    let sliding_at = |l: u32| l.is_multiple_of(2);

    let layers = (0..d.layers)
        .map(|l| {
            let n = |s: &str| format!("layer.{l}.{s}");
            let norm = |s: &str, w: u64| Norm {
                weight: Weight::sym(n(s), [w], w1),
                eps: d.norm_eps,
            };
            Layer {
                attn: Attn {
                    window: sliding_at(l).then_some(d.window),
                    q_heads: d.q_heads,
                    kv_heads: d.kv_heads,
                    head_dim: d.head_dim,
                    sm_scale: (d.head_dim as f32).sqrt().recip(),
                    rope: Yarn {
                        theta: d.theta,
                        factor: d.yarn_factor,
                        beta_fast: d.yarn_beta_fast,
                        beta_slow: d.yarn_beta_slow,
                        attention_factor: d.yarn_attention_factor,
                        original_max_position: d.yarn_original_max_position,
                    },
                    q_proj: Weight::sym(n("q_proj"), [q_w, hidden], w1).columns(),
                    q_bias: Weight::sym(n("q_bias"), [q_w], w1).columns(),
                    k_proj: Weight::sym(n("k_proj"), [kv_w, hidden], w1).columns(),
                    k_bias: Weight::sym(n("k_bias"), [kv_w], w1).columns(),
                    v_proj: Weight::sym(n("v_proj"), [kv_w, hidden], w1).columns(),
                    v_bias: Weight::sym(n("v_bias"), [kv_w], w1).columns(),
                    o_proj: Weight::sym(n("o_proj"), [hidden, q_w], w1).rows(),
                    o_bias: Weight::sym(n("o_bias"), [hidden], w1),
                    sinks: Weight::sym(n("attn_sinks"), [d.q_heads as u64], w1).columns(),
                    kv: CacheRef::to(format!("kv.{l}")),
                },
                attn_norm: norm("attn_norm", hidden),
                mlp_norm: norm("mlp_norm", hidden),
                mlp: Moe {
                    experts: d.experts,
                    top_k: d.top_k,
                    inter: d.inter,
                    swiglu_limit: d.swiglu_limit,
                    swiglu_alpha: d.swiglu_alpha,
                    router: Weight::sym(n("router"), [experts, hidden], w1),
                    router_bias: Weight::sym(n("router_bias"), [experts], w1),
                    gate_up: Weight::sym(
                        n("expert_gate_up_bank"),
                        [experts, 2 * inter, hidden],
                        w2,
                    )
                    .bank([inter, inter]),
                    gate_up_bias: Weight::sym(n("expert_gate_up_bias"), [experts, 2 * inter], w1)
                        .bank([inter, inter]),
                    down: Weight::sym(n("expert_down_bank"), [experts, hidden, inter], w2).rows(),
                    down_bias: Weight::sym(n("expert_down_bias"), [experts, hidden], w1),
                },
            }
        })
        .collect();

    Model {
        hidden: d.hidden,
        vocab: d.vocab,
        tp,
        embed: Weight::sym("embed", [d.vocab as u64, hidden], w1),
        head: Weight::sym("lm_head", [d.vocab as u64, hidden], w1),
        layers,
        final_norm: Norm {
            weight: Weight::sym("final_norm", [hidden], w1),
            eps: d.norm_eps,
        },
    }
}
