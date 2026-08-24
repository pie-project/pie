use std::marker::PhantomData;

use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{CacheRef, Norm, Tensor};

pub struct Model<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize = 1> {
    pub hidden: u32,
    pub vocab: u32,
    pub embed: Tensor<W1>,
    /// The lm head's own bank. This text does NOT tie: the tied reading was
    /// an enum arm no geometry constructed, and `tie_word_embeddings` is
    /// false in the config these dims are read from.
    pub head: Tensor<W1>,
    pub layers: Vec<Layer<W1, W2>>,
    pub final_norm: Norm<W1>,
    _kv: PhantomData<K>,
}

pub struct Layer<W1: Dtype, W2: Dtype> {
    pub attn: Attn<W1>,
    pub attn_norm: Norm<W1>,
    pub mlp_norm: Norm<W1>,
    pub mlp: Mlp<W1, W2>,
}

pub struct Attn<W1: Dtype> {
    pub heads: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub q_a_proj: Tensor<W1>,
    pub q_a_norm: Norm<W1>,
    pub q_b_proj: Tensor<W1>,
    pub kv_a_proj: Tensor<W1>,
    pub kv_a_norm: Norm<W1>,
    pub kv_b_proj: Tensor<W1>,
    pub o_proj: Tensor<W1>,
    pub indexer: Indexer<W1>,
    pub kv: CacheRef,
}

pub struct Indexer<W1: Dtype> {
    pub heads: u32,
    pub head_dim: u32,
    pub top_k: u32,
    pub q_proj: Tensor<W1>,
    pub k_proj: Tensor<W1>,
    pub weights_proj: Tensor<W1>,
    pub k_norm: Norm<W1>,
    pub k_norm_bias: Tensor<W1>,
    pub keys: CacheRef,
}

#[allow(clippy::large_enum_variant)] // a per-layer weight-bank record, built once at trace; boxing buys nothing and costs every reader a deref
pub enum Mlp<W1: Dtype, W2: Dtype> {
    Dense {
        gate_up: Tensor<W1>,
        down: Tensor<W1>,
        inter: u32,
    },
    Routed {
        router: Tensor<W1>,
        gate_up: Tensor<W2>,
        down: Tensor<W2>,
        shared: Option<Shared<W1>>,
        experts: u32,
        top_k: u32,
        inter: u32,
        norm_weights: bool,
        scaling: f32,
    },
}

pub struct Shared<W1: Dtype> {
    pub gate_up: Tensor<W1>,
    pub down: Tensor<W1>,
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

/// THE CUT, AND THE WHOLE OF IT: the dims a rank holds a share of at `TP`
/// ways, with `..d` saying that everything else is replicated.
///
/// The attention heads and the three intermediates. THE LATENT DOES NOT
/// DIVIDE, and that is what makes MLA cheap under a rank cut: `kv_a_proj`
/// writes the `[1, kv_lora_rank + qk_rope_head_dim]` row this text caches
/// and every rank writes and holds the whole of it, so `kv_lora_rank`,
/// `qk_rope_head_dim` and the `q_lora_rank` beside them stay put while
/// `q_b_proj` and `kv_b_proj` cut the head fan they expand it into.
///
/// The sparse indexer does not divide either: every one of its banks is
/// replicated, so each rank scores the same keys and reaches the same
/// `index.topk` selection — which is the only way a selection two ranks
/// attend through can be the same selection without a statement to make it
/// so.
fn per_rank<const TP: usize>(d: Dims) -> Dims {
    let cut = |what, whole| model_dsl::per_rank(what, whole, TP);
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

impl<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize> Model<W1, W2, K, TP> {
    /// UNVERIFIED against a checkpoint — no GLM-5 file is cached, so nothing
    /// has been held against these numbers.
    ///
    /// `serve::ROWS` advertises `glm5-a12b-bf16-bf16-kv-bf16` as arch
    /// `glm_moe_dsa`, and unlike deepseek-v4's and kimi's rows this text is
    /// a FULL-DEPTH reading: 46 layers with 3 dense, from the published
    /// A12B config. That makes it the more dangerous kind of unverified —
    /// it looks like a transcription and may be one, but the DSA index
    /// tower (`index_heads` / `index_head_dim` / `index_top_k`) and the
    /// MoE block are the parts a file would most likely disagree with, and
    /// `tie_word_embeddings` is read as false with nothing to check it
    /// against.
    pub fn a12b() -> Self {
        assemble(Dims {
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
        })
    }
}

fn assemble<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize>(d: Dims) -> Model<W1, W2, K, TP> {
    let d = per_rank::<TP>(d);
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
                weight: Tensor::sym(n(s), [w]),
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
                q_a_proj: Tensor::sym(n("q_a_proj"), [q_lora, hidden]),
                q_a_norm: norm("q_a_norm", q_lora),
                q_b_proj: Tensor::sym(n("q_b_proj"), [q_b_width, q_lora]).columns(),
                kv_a_proj: Tensor::sym(n("kv_a_proj"), [kv_a_width, hidden]),
                kv_a_norm: norm("kv_a_norm", kv_lora),
                kv_b_proj: Tensor::sym(n("kv_b_proj"), [kv_b_width, kv_lora]).columns(),
                o_proj: Tensor::sym(n("o_proj"), [hidden, v_width]).rows(),
                indexer: Indexer {
                    heads: d.index_heads,
                    head_dim: d.index_head_dim,
                    top_k: d.index_top_k,
                    q_proj: Tensor::sym(n("index_q_proj"), [index_width, q_lora]),
                    k_proj: Tensor::sym(n("index_k_proj"), [d.index_head_dim as u64, hidden]),
                    weights_proj: Tensor::sym(n("index_weights"), [d.index_heads as u64, q_lora]),
                    k_norm: norm("index_k_norm", d.index_head_dim as u64),
                    k_norm_bias: Tensor::sym(n("index_k_norm_bias"), [d.index_head_dim as u64]),
                    keys: CacheRef::to(format!("index.{l}")),
                },
                kv: CacheRef::to(format!("kv.{l}")),
            };
            let mlp = if dense_at(l) {
                let iw = d.dense_inter as u64;
                Mlp::Dense {
                    gate_up: Tensor::sym(n("gate_up"), [2 * iw, hidden]).packed([iw, iw]),
                    down: Tensor::sym(n("down"), [hidden, iw]).rows(),
                    inter: d.dense_inter,
                }
            } else {
                let m = &d.moe;
                let iw = m.inter as u64;
                let sw = m.shared_inter as u64;
                Mlp::Routed {
                    router: Tensor::sym(n("router"), [m.experts as u64, hidden]),
                    gate_up: Tensor::sym(n("experts_gate_up"), [m.experts as u64, 2 * iw, hidden])
                        .bank([iw, iw]),
                    down: Tensor::sym(n("experts_down"), [m.experts as u64, hidden, iw]).rows(),
                    shared: (m.shared_inter > 0).then(|| Shared {
                        gate_up: Tensor::sym(n("shared_gate_up"), [2 * sw, hidden])
                            .packed([sw, sw]),
                        down: Tensor::sym(n("shared_down"), [hidden, sw]).rows(),
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
        embed: Tensor::sym("embed", [d.vocab as u64, hidden]),
        head: Tensor::sym("lm_head", [d.vocab as u64, hidden]),
        layers,
        final_norm: Norm {
            weight: Tensor::sym("final_norm", [hidden]),
            eps: d.norm_eps,
        },
        _kv: PhantomData,
    }
}
