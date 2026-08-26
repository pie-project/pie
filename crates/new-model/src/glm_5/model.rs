//! The GLM 5 declaration, de-genericized (design §5, decision #18): the old
//! `Model<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize>` phantom tree is
//! gone — `tp` is a runtime field, each weight carries its `Repr`, and the kv
//! dtype is nobody's parameter. Names and the per-layer scheme are unchanged
//! from the old crate: weights intern by name, so the checkpoint mapping
//! carries over untouched.

use new_model_dsl::{CacheRef, Norm, Weight};
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
    pub mlp: Mlp,
}

pub struct Attn {
    pub heads: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub q_a_proj: Weight,
    pub q_a_norm: Norm,
    pub q_b_proj: Weight,
    pub kv_a_proj: Weight,
    pub kv_a_norm: Norm,
    pub kv_b_proj: Weight,
    pub o_proj: Weight,
    pub indexer: Indexer,
    pub kv: CacheRef,
}

pub struct Indexer {
    pub heads: u32,
    pub head_dim: u32,
    pub top_k: u32,
    pub q_proj: Weight,
    pub k_proj: Weight,
    pub weights_proj: Weight,
    pub k_norm: Norm,
    pub k_norm_bias: Weight,
    pub keys: CacheRef,
}

#[allow(clippy::large_enum_variant)]
pub enum Mlp {
    Dense {
        gate_up: Weight,
        down: Weight,
        inter: u32,
    },
    Routed {
        router: Weight,
        gate_up: Weight,
        down: Weight,
        shared: Option<Shared>,
        experts: u32,
        top_k: u32,
        inter: u32,
        norm_weights: bool,
        scaling: f32,
    },
}

pub struct Shared {
    pub gate_up: Weight,
    pub down: Weight,
    pub inter: u32,
}

struct MoeDims {
    experts: u32,
    top_k: u32,
    inter: u32,
    shared_inter: u32,
    norm_weights: bool,
    scaling: f32,
}

struct Dims {
    hidden: u32,
    layers: u32,
    dense_layers: u32,
    heads: u32,
    q_lora_rank: u32,
    kv_lora_rank: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    v_head_dim: u32,
    theta: f32,
    index_heads: u32,
    index_head_dim: u32,
    index_top_k: u32,
    dense_inter: u32,
    moe: MoeDims,
    vocab: u32,
    norm_eps: f32,
}

fn per_rank(d: Dims, tp: u32) -> Dims {
    let cut = |what, whole| new_model_dsl::per_rank(what, whole, tp as usize);
    Dims {
        heads: cut("heads", d.heads),
        dense_inter: cut("dense inter", d.dense_inter),
        moe: MoeDims {
            inter: cut("moe inter", d.moe.inter),
            shared_inter: cut("shared inter", d.moe.shared_inter),
            ..d.moe
        },
        ..d
    }
}

impl Model {
    pub fn a12b(w1: Repr, w2: Repr, tp: u32) -> Model {
        assemble(
            w1,
            w2,
            tp,
            Dims {
                hidden: 4096,
                layers: 46,
                dense_layers: 3,
                heads: 96,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                theta: 10_000.0,
                index_heads: 64,
                index_head_dim: 128,
                index_top_k: 2048,
                dense_inter: 10_944,
                moe: MoeDims {
                    experts: 128,
                    top_k: 8,
                    inter: 1408,
                    shared_inter: 1408,
                    norm_weights: true,
                    scaling: 2.5,
                },
                vocab: 151_552,
                norm_eps: 1e-5,
            },
        )
    }
}

fn assemble(w1: Repr, w2: Repr, tp: u32, d: Dims) -> Model {
    let d = per_rank(d, tp);
    let hidden = d.hidden as u64;
    let q_lora = d.q_lora_rank as u64;
    let kv_lora = d.kv_lora_rank as u64;
    let qk_head_dim = (d.qk_nope_head_dim + d.qk_rope_head_dim) as u64;
    let q_b_width = d.heads as u64 * qk_head_dim;
    let kv_a_width = kv_lora + d.qk_rope_head_dim as u64;
    let kv_b_width = d.heads as u64 * (d.qk_nope_head_dim + d.v_head_dim) as u64;
    let v_width = d.heads as u64 * d.v_head_dim as u64;
    let index_width = d.index_heads as u64 * d.index_head_dim as u64;
    let dense_at = |l: u32| l < d.dense_layers;

    let layers = (0..d.layers)
        .map(|l| {
            let n = |s: &str| format!("layer.{l}.{s}");
            let norm = |s: &str, w: u64| Norm {
                weight: Weight::sym(n(s), [w], w1),
                eps: d.norm_eps,
            };
            let attn = Attn {
                heads: d.heads,
                kv_lora_rank: d.kv_lora_rank,
                qk_nope_head_dim: d.qk_nope_head_dim,
                qk_rope_head_dim: d.qk_rope_head_dim,
                v_head_dim: d.v_head_dim,
                theta: d.theta,
                sm_scale: (qk_head_dim as f32).sqrt().recip(),
                q_a_proj: Weight::sym(n("q_a_proj"), [q_lora, hidden], w1),
                q_a_norm: norm("q_a_norm", q_lora),
                q_b_proj: Weight::sym(n("q_b_proj"), [q_b_width, q_lora], w1).columns(),
                kv_a_proj: Weight::sym(n("kv_a_proj"), [kv_a_width, hidden], w1),
                kv_a_norm: norm("kv_a_norm", kv_lora),
                kv_b_proj: Weight::sym(n("kv_b_proj"), [kv_b_width, kv_lora], w1).columns(),
                o_proj: Weight::sym(n("o_proj"), [hidden, v_width], w1).rows(),
                indexer: Indexer {
                    heads: d.index_heads,
                    head_dim: d.index_head_dim,
                    top_k: d.index_top_k,
                    q_proj: Weight::sym(n("index_q_proj"), [index_width, q_lora], w1),
                    k_proj: Weight::sym(n("index_k_proj"), [d.index_head_dim as u64, hidden], w1),
                    weights_proj: Weight::sym(
                        n("index_weights"),
                        [d.index_heads as u64, q_lora],
                        w1,
                    ),
                    k_norm: norm("index_k_norm", d.index_head_dim as u64),
                    k_norm_bias: Weight::sym(n("index_k_norm_bias"), [d.index_head_dim as u64], w1),
                    keys: CacheRef::to(format!("index.{l}")),
                },
                kv: CacheRef::to(format!("kv.{l}")),
            };
            let mlp = if dense_at(l) {
                let iw = d.dense_inter as u64;
                Mlp::Dense {
                    gate_up: Weight::sym(n("gate_up"), [2 * iw, hidden], w1).packed([iw, iw]),
                    down: Weight::sym(n("down"), [hidden, iw], w1).rows(),
                    inter: d.dense_inter,
                }
            } else {
                let m = &d.moe;
                let iw = m.inter as u64;
                let sw = m.shared_inter as u64;
                Mlp::Routed {
                    router: Weight::sym(n("router"), [m.experts as u64, hidden], w1),
                    gate_up: Weight::sym(
                        n("experts_gate_up"),
                        [m.experts as u64, 2 * iw, hidden],
                        w2,
                    )
                    .bank([iw, iw]),
                    down: Weight::sym(n("experts_down"), [m.experts as u64, hidden, iw], w2).rows(),
                    shared: (m.shared_inter > 0).then(|| Shared {
                        gate_up: Weight::sym(n("shared_gate_up"), [2 * sw, hidden], w1)
                            .packed([sw, sw]),
                        down: Weight::sym(n("shared_down"), [hidden, sw], w1).rows(),
                        inter: m.shared_inter,
                    }),
                    experts: m.experts,
                    top_k: m.top_k,
                    inter: m.inter,
                    norm_weights: m.norm_weights,
                    scaling: m.scaling,
                }
            };
            Layer {
                attn,
                attn_norm: norm("attn_norm", hidden),
                mlp_norm: norm("mlp_norm", hidden),
                mlp,
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
