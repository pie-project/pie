//! The Kimi K3 declaration, de-genericized (design §5, decision #18): the old
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
    pub res_blend: Option<ResBlend>,
    pub mixer: Mixer,
    pub mixer_norm: Norm,
    pub mlp_norm: Norm,
    pub mlp: Mlp,
}

pub struct ResBlend {
    pub norm: Norm,
    pub proj: Weight,
}

pub enum Mixer {
    Mla(Mla),
    Kda(Kda),
}

pub struct Mla {
    pub heads: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub sm_scale: f32,
    pub q_a_proj: Weight,
    pub q_a_norm: Norm,
    pub q_b_proj: Weight,
    pub kv_a_proj: Weight,
    pub kv_a_norm: Norm,
    pub kv_b_proj: Weight,
    pub gate: Option<Weight>,
    pub o_proj: Weight,
    pub kv: CacheRef,
}

pub struct Kda {
    pub heads: u32,
    pub head_dim: u32,
    pub conv_kernel: u32,
    pub norm_eps: f32,
    pub qkv: Weight,
    pub conv: Weight,
    pub f_a: Weight,
    pub f_b: Weight,
    pub b: Weight,
    pub dt_bias: Weight,
    pub a_log: Weight,
    pub gate: Weight,
    pub o_norm: Norm,
    pub o_proj: Weight,
    pub conv_state: CacheRef,
    pub delta_state: CacheRef,
}

#[allow(clippy::large_enum_variant)]
pub enum Mlp {
    Dense {
        gate_up: Weight,
        down: Weight,
        inter: u32,
        beta: f32,
        up_cap: Option<f32>,
    },
    Routed {
        router: Weight,
        gate_up: Weight,
        down: Weight,
        shared: Option<Shared>,
        experts: u32,
        top_k: u32,
        routed_scaling: f32,
        inter: u32,
        beta: f32,
        up_cap: Option<f32>,
    },
}

pub struct Shared {
    pub gate_up: Weight,
    pub down: Weight,
    pub inter: u32,
}

struct MlaDims {
    heads: u32,
    q_lora_rank: u32,
    kv_lora_rank: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    v_head_dim: u32,
    output_gate: bool,
}

struct KdaDims {
    heads: u32,
    head_dim: u32,
    conv_kernel: u32,
    norm_eps: f32,
}

struct MoeDims {
    experts: u32,
    top_k: u32,
    routed_scaling: f32,
    inter: u32,
    shared_inter: u32,
}

struct Dims {
    hidden: u32,
    layers: u32,
    dense_layers: u32,
    full_attn_every: u32,
    res_block: u32,
    mla: MlaDims,
    kda: KdaDims,
    moe: MoeDims,
    dense_inter: u32,
    situ_beta: f32,
    situ_cap: Option<f32>,
    vocab: u32,
    norm_eps: f32,
}

fn per_rank(d: Dims, tp: u32) -> Dims {
    let cut = |what, whole| new_model_dsl::per_rank(what, whole, tp as usize);
    Dims {
        mla: MlaDims {
            heads: cut("mla heads", d.mla.heads),
            ..d.mla
        },
        kda: KdaDims {
            heads: cut("kda heads", d.kda.heads),
            ..d.kda
        },
        moe: MoeDims {
            inter: cut("moe inter", d.moe.inter),
            shared_inter: cut("shared inter", d.moe.shared_inter),
            ..d.moe
        },
        dense_inter: cut("dense inter", d.dense_inter),
        ..d
    }
}

impl Model {
    pub fn k3(w1: Repr, w2: Repr, tp: u32) -> Model {
        assemble(
            w1,
            w2,
            tp,
            Dims {
                hidden: 2048,
                layers: 8,
                dense_layers: 1,
                full_attn_every: 4,
                res_block: 4,
                mla: MlaDims {
                    heads: 16,
                    q_lora_rank: 768,
                    kv_lora_rank: 256,
                    qk_nope_head_dim: 128,
                    qk_rope_head_dim: 64,
                    v_head_dim: 128,
                    output_gate: true,
                },
                kda: KdaDims {
                    heads: 16,
                    head_dim: 128,
                    conv_kernel: 4,
                    norm_eps: 1e-5,
                },
                moe: MoeDims {
                    experts: 64,
                    top_k: 6,
                    routed_scaling: 2.0,
                    inter: 1024,
                    shared_inter: 1024,
                },
                dense_inter: 5632,
                situ_beta: 1.0,
                situ_cap: None,
                vocab: 163_840,
                norm_eps: 1e-5,
            },
        )
    }
}

fn assemble(w1: Repr, w2: Repr, tp: u32, d: Dims) -> Model {
    let d = per_rank(d, tp);
    let hidden = d.hidden as u64;
    let full_at = |l: u32| d.full_attn_every > 0 && (l + 1).is_multiple_of(d.full_attn_every);
    let moe_at = |l: u32| l >= d.dense_layers;
    let blend_at = |l: u32| d.res_block > 0 && l > 0 && l.is_multiple_of(d.res_block);

    let a = &d.mla;
    let k = &d.kda;
    let qk_head_dim = (a.qk_nope_head_dim + a.qk_rope_head_dim) as u64;
    let q_b_width = a.heads as u64 * qk_head_dim;
    let kv_a_width = (a.kv_lora_rank + a.qk_rope_head_dim) as u64;
    let kv_b_width = a.heads as u64 * (a.qk_nope_head_dim + a.v_head_dim) as u64;
    let v_width = a.heads as u64 * a.v_head_dim as u64;
    let kda_width = k.heads as u64 * k.head_dim as u64;

    let layers = (0..d.layers)
        .map(|l| {
            let n = |s: &str| format!("layer.{l}.{s}");
            let norm = |s: &str, w: u64| Norm {
                weight: Weight::sym(n(s), [w], w1),
                eps: d.norm_eps,
            };
            let mixer = if full_at(l) {
                Mixer::Mla(Mla {
                    heads: a.heads,
                    kv_lora_rank: a.kv_lora_rank,
                    qk_nope_head_dim: a.qk_nope_head_dim,
                    qk_rope_head_dim: a.qk_rope_head_dim,
                    v_head_dim: a.v_head_dim,
                    sm_scale: (qk_head_dim as f32).sqrt().recip(),
                    q_a_proj: Weight::sym(n("q_a_proj"), [a.q_lora_rank as u64, hidden], w1),
                    q_a_norm: norm("q_a_norm", a.q_lora_rank as u64),
                    q_b_proj: Weight::sym(n("q_b_proj"), [q_b_width, a.q_lora_rank as u64], w1)
                        .columns(),
                    kv_a_proj: Weight::sym(n("kv_a_proj"), [kv_a_width, hidden], w1),
                    kv_a_norm: norm("kv_a_norm", a.kv_lora_rank as u64),
                    kv_b_proj: Weight::sym(n("kv_b_proj"), [kv_b_width, a.kv_lora_rank as u64], w1)
                        .columns(),
                    gate: a
                        .output_gate
                        .then(|| Weight::sym(n("o_gate"), [v_width, hidden], w1).columns()),
                    o_proj: Weight::sym(n("o_proj"), [hidden, v_width], w1).rows(),
                    kv: CacheRef::to(format!("kv.{l}")),
                })
            } else {
                Mixer::Kda(Kda {
                    heads: k.heads,
                    head_dim: k.head_dim,
                    conv_kernel: k.conv_kernel,
                    norm_eps: k.norm_eps,
                    qkv: Weight::sym(n("kda_qkv"), [3 * kda_width, hidden], w1)
                        .packed([kda_width, kda_width, kda_width]),
                    conv: Weight::sym(n("kda_conv"), [3 * kda_width, k.conv_kernel as u64], w1)
                        .packed([kda_width, kda_width, kda_width]),
                    f_a: Weight::sym(n("kda_f_a"), [k.head_dim as u64, hidden], w1),
                    f_b: Weight::sym(n("kda_f_b"), [kda_width, k.head_dim as u64], w1).columns(),
                    b: Weight::sym(n("kda_b"), [k.heads as u64, hidden], w1).columns(),
                    dt_bias: Weight::sym(
                        n("kda_dt_bias"),
                        [k.heads as u64, k.head_dim as u64],
                        Repr::F32,
                    )
                    .columns(),
                    a_log: Weight::sym(n("kda_a_log"), [k.heads as u64], Repr::F32).columns(),
                    gate: Weight::sym(n("kda_gate"), [kda_width, hidden], w1).columns(),
                    o_norm: Norm {
                        weight: Weight::sym(n("kda_o_norm"), [k.head_dim as u64], w1),
                        eps: k.norm_eps,
                    },
                    o_proj: Weight::sym(n("kda_o_proj"), [hidden, kda_width], w1).rows(),
                    conv_state: CacheRef::to(format!("conv.{l}")),
                    delta_state: CacheRef::to(format!("delta.{l}")),
                })
            };
            let mlp = if moe_at(l) {
                let m = &d.moe;
                let inter = m.inter as u64;
                let shared_inter = m.shared_inter as u64;
                Mlp::Routed {
                    router: Weight::sym(n("router"), [m.experts as u64, hidden], w1),
                    gate_up: Weight::sym(
                        n("experts_gate_up"),
                        [m.experts as u64, 2 * inter, hidden],
                        w2,
                    )
                    .bank([inter, inter]),
                    down: Weight::sym(n("experts_down"), [m.experts as u64, hidden, inter], w2)
                        .rows(),
                    shared: (m.shared_inter > 0).then(|| Shared {
                        gate_up: Weight::sym(n("shared_gate_up"), [2 * shared_inter, hidden], w1)
                            .packed([shared_inter, shared_inter]),
                        down: Weight::sym(n("shared_down"), [hidden, shared_inter], w1).rows(),
                        inter: m.shared_inter,
                    }),
                    experts: m.experts,
                    top_k: m.top_k,
                    routed_scaling: m.routed_scaling,
                    inter: m.inter,
                    beta: d.situ_beta,
                    up_cap: d.situ_cap,
                }
            } else {
                let inter = d.dense_inter as u64;
                Mlp::Dense {
                    gate_up: Weight::sym(n("gate_up"), [2 * inter, hidden], w1)
                        .packed([inter, inter]),
                    down: Weight::sym(n("down"), [hidden, inter], w1).rows(),
                    inter: d.dense_inter,
                    beta: d.situ_beta,
                    up_cap: d.situ_cap,
                }
            };
            Layer {
                res_blend: blend_at(l).then(|| ResBlend {
                    norm: norm("res_norm", hidden),
                    proj: Weight::sym(n("res_proj"), [1, hidden], w1),
                }),
                mixer,
                mixer_norm: norm("mixer_norm", hidden),
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
