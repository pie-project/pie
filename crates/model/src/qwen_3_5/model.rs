//! The Qwen 3.5 declaration, de-genericized (design §5, decision #18): the
//! old `Model<W1: Dtype, K: KvDtype, const TP: usize>` phantom tree is gone —
//! `tp` is a runtime field, each weight carries its `Dtype`, and the SKU
//! constructors take every element choice as an argument: one dims table
//! serves every quantization of a SKU, and the catalog row that names the
//! shipped variant is where the weight, activation and kv-cache elements are
//! spelled. Names and the per-layer scheme are unchanged from the old crate:
//! weights intern by name, so the checkpoint mapping carries over untouched.

use model_dsl::{Dtype, Weight};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,
    /// Activation element — stated, not inherited silently.
    pub act: Dtype,
    /// Kv-cache element layout — drives the append kernel and row bytes.
    pub kv: Dtype,
    pub embed: Weight,
    pub head: Head,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,
}

pub enum Head {
    Tied,
    Bank(Weight),
}

pub struct Layer {
    pub mixer: Mixer,
    pub mixer_norm: Weight,
    pub mixer_norm_eps: f32,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: Mlp,
}

pub enum Mixer {
    Attn(Attn),
    Gdn(Gdn),
}

pub struct Attn {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub qg_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    pub o_proj: Weight,
    pub q_norm: Weight,
    pub q_norm_eps: f32,
    pub k_norm: Weight,
    pub k_norm_eps: f32,
    pub kv: String,
}

pub struct Gdn {
    pub k_heads: u32,
    pub v_heads: u32,
    pub k_dim: u32,
    pub v_dim: u32,
    pub conv_kernel: u32,
    pub in_qkvz: Weight,
    pub in_ba: Weight,
    pub conv: Weight,
    pub dt_bias: Weight,
    pub a_log: Weight,
    pub norm: Weight,
    pub norm_eps: f32,
    pub out_proj: Weight,
    pub conv_state: String,
    pub delta_state: String,
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
        shared_gate_up: Weight,
        shared_down: Weight,
        shared_gate: Weight,
        experts: u32,
        top_k: u32,
        inter: u32,
        shared_inter: u32,
    },
}

struct MoeDims {
    experts: u32,
    top_k: u32,
    inter: u32,
    shared_inter: u32,
}

enum MlpDims {
    Dense { inter: u32 },
    Routed(MoeDims),
}

struct Dims {
    hidden: u32,
    layers: u32,
    attn_every: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    conv_kernel: u32,
    mlp: MlpDims,
    vocab: u32,
    tied: bool,
    norm_eps: f32,
}

fn per_rank(d: Dims, tp: u32) -> Dims {
    let cut = |what, whole| model_dsl::per_rank(what, whole, tp as usize);
    Dims {
        q_heads: cut("q_heads", d.q_heads),
        kv_heads: cut("kv_heads", d.kv_heads),
        k_heads: cut("k_heads", d.k_heads),
        v_heads: cut("v_heads", d.v_heads),
        mlp: match d.mlp {
            MlpDims::Dense { inter } => MlpDims::Dense {
                inter: cut("inter", inter),
            },
            MlpDims::Routed(m) => MlpDims::Routed(MoeDims {
                inter: cut("moe inter", m.inter),
                shared_inter: cut("shared inter", m.shared_inter),
                ..m
            }),
        },
        ..d
    }
}

impl Model {
    pub fn a3b(w: Dtype, act: Dtype, kv: Dtype, tp: u32) -> Model {
        assemble(
            w,
            act,
            kv,
            tp,
            Dims {
                hidden: 2048,
                layers: 40,
                attn_every: 4,
                q_heads: 16,
                kv_heads: 2,
                head_dim: 256,
                rotary_dim: 64,
                theta: 10_000_000.0,
                k_heads: 16,
                v_heads: 32,
                k_dim: 128,
                v_dim: 128,
                conv_kernel: 4,
                mlp: MlpDims::Routed(MoeDims {
                    experts: 256,
                    top_k: 8,
                    inter: 512,
                    shared_inter: 512,
                }),
                vocab: 248_320,
                tied: false,
                norm_eps: 1e-6,
            },
        )
    }

    pub fn d0_8b(w: Dtype, act: Dtype, kv: Dtype, tp: u32) -> Model {
        assemble(
            w,
            act,
            kv,
            tp,
            Dims {
                hidden: 1024,
                layers: 24,
                attn_every: 4,
                q_heads: 8,
                kv_heads: 2,
                head_dim: 256,
                rotary_dim: 64,
                theta: 10_000_000.0,
                k_heads: 16,
                v_heads: 16,
                k_dim: 128,
                v_dim: 128,
                conv_kernel: 4,
                mlp: MlpDims::Dense { inter: 3584 },
                vocab: 248_320,
                tied: true,
                norm_eps: 1e-6,
            },
        )
    }

    pub fn d3b(w: Dtype, act: Dtype, kv: Dtype, tp: u32) -> Model {
        assemble(
            w,
            act,
            kv,
            tp,
            Dims {
                hidden: 2048,
                layers: 24,
                attn_every: 4,
                q_heads: 16,
                kv_heads: 2,
                head_dim: 256,
                rotary_dim: 64,
                theta: 10_000_000.0,
                k_heads: 16,
                v_heads: 32,
                k_dim: 128,
                v_dim: 128,
                conv_kernel: 4,
                mlp: MlpDims::Dense { inter: 8192 },
                vocab: 151_936,
                tied: true,
                norm_eps: 1e-6,
            },
        )
    }
}

fn assemble(w: Dtype, act: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
    let d = per_rank(d, tp);
    let hidden = d.hidden as u64;
    let attn_at = |l: u32| l % d.attn_every == d.attn_every - 1;

    let layers = (0..d.layers)
        .map(|l| {
            let n = |s: &str| format!("layer.{l}.{s}");
            let norm = |s: &str, dim: u64| Weight::sym(n(s), [dim], w);
            let mixer = if attn_at(l) {
                let hd = d.head_dim as u64;
                Mixer::Attn(Attn {
                    q_heads: d.q_heads,
                    kv_heads: d.kv_heads,
                    head_dim: d.head_dim,
                    rotary_dim: d.rotary_dim,
                    theta: d.theta,
                    sm_scale: (d.head_dim as f32).sqrt().recip(),
                    qg_proj: Weight::sym(n("qg_proj"), [2 * d.q_heads as u64 * hd, hidden], w)
                        .columns(),
                    k_proj: Weight::sym(n("k_proj"), [d.kv_heads as u64 * hd, hidden], w).columns(),
                    v_proj: Weight::sym(n("v_proj"), [d.kv_heads as u64 * hd, hidden], w).columns(),
                    o_proj: Weight::sym(n("o_proj"), [hidden, d.q_heads as u64 * hd], w).rows(),
                    q_norm: norm("q_norm", hd),
                    q_norm_eps: d.norm_eps,
                    k_norm: norm("k_norm", hd),
                    k_norm_eps: d.norm_eps,
                    kv: format!("kv.{l}"),
                })
            } else {
                let k_w = d.k_heads as u64 * d.k_dim as u64;
                let v_w = d.v_heads as u64 * d.v_dim as u64;
                let qkv = 2 * k_w + v_w;
                let qkvz = qkv + v_w;
                Mixer::Gdn(Gdn {
                    k_heads: d.k_heads,
                    v_heads: d.v_heads,
                    k_dim: d.k_dim,
                    v_dim: d.v_dim,
                    conv_kernel: d.conv_kernel,
                    in_qkvz: Weight::sym(n("in_qkvz"), [qkvz, hidden], w)
                        .packed([k_w, k_w, v_w, v_w]),
                    in_ba: Weight::sym(n("in_ba"), [2 * d.v_heads as u64, hidden], w)
                        .packed([d.v_heads as u64, d.v_heads as u64]),
                    conv: Weight::sym(n("conv"), [qkv, d.conv_kernel as u64], w)
                        .packed([k_w, k_w, v_w]),
                    dt_bias: Weight::sym(n("dt_bias"), [d.v_heads as u64], w).columns(),
                    a_log: Weight::sym(n("a_log"), [d.v_heads as u64], Dtype::F32).columns(),
                    norm: Weight::sym(n("gdn_norm"), [d.v_dim as u64], Dtype::F32),
                    norm_eps: d.norm_eps,
                    out_proj: Weight::sym(
                        n("out_proj"),
                        [hidden, d.v_heads as u64 * d.v_dim as u64],
                        w,
                    )
                    .rows(),
                    conv_state: format!("conv.{l}"),
                    delta_state: format!("delta.{l}"),
                })
            };
            let mlp = match &d.mlp {
                MlpDims::Dense { inter } => Mlp::Dense {
                    gate_up: Weight::sym(n("gate_up"), [2 * *inter as u64, hidden], w)
                        .packed([*inter as u64, *inter as u64]),
                    down: Weight::sym(n("down"), [hidden, *inter as u64], w).rows(),
                    inter: *inter,
                },
                MlpDims::Routed(m) => Mlp::Routed {
                    router: Weight::sym(n("router"), [m.experts as u64, hidden], w),

                    gate_up: Weight::sym(
                        n("experts_gate_up"),
                        [m.experts as u64, 2 * m.inter as u64, hidden],
                        w,
                    )
                    .bank([m.inter as u64, m.inter as u64]),
                    down: Weight::sym(
                        n("experts_down"),
                        [m.experts as u64, hidden, m.inter as u64],
                        w,
                    )
                    .rows(),
                    shared_gate_up: Weight::sym(
                        n("shared_gate_up"),
                        [2 * m.shared_inter as u64, hidden],
                        w,
                    )
                    .packed([m.shared_inter as u64, m.shared_inter as u64]),
                    shared_down: Weight::sym(n("shared_down"), [hidden, m.shared_inter as u64], w)
                        .rows(),
                    shared_gate: Weight::sym(n("shared_gate"), [1, hidden], w),
                    experts: m.experts,
                    top_k: m.top_k,
                    inter: m.inter,
                    shared_inter: m.shared_inter,
                },
            };
            Layer {
                mixer,
                mixer_norm: norm("mixer_norm", hidden),
                mixer_norm_eps: d.norm_eps,
                mlp_norm: norm("mlp_norm", hidden),
                mlp_norm_eps: d.norm_eps,
                mlp,
            }
        })
        .collect();

    Model {
        hidden: d.hidden,
        vocab: d.vocab,
        tp,
        act,
        kv,
        embed: Weight::sym("embed", [d.vocab as u64, hidden], w),
        head: if d.tied {
            Head::Tied
        } else {
            Head::Bank(Weight::sym("lm_head", [d.vocab as u64, hidden], w))
        },
        layers,
        final_norm: Weight::sym("final_norm", [hidden], w),
        final_norm_eps: d.norm_eps,
    }
}
