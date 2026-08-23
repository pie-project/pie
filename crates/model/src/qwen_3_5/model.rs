use std::marker::PhantomData;

use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{CacheRef, Norm, Tensor};

pub struct Model<W1: Dtype, K: KvDtype, const TP: usize = 1> {
    pub hidden: u32,
    pub embed: Tensor<W1>,
    pub head: Head<W1>,
    pub layers: Vec<Layer<W1>>,
    pub final_norm: Norm<W1>,
    _kv: PhantomData<K>,
}

pub enum Head<W1: Dtype> {
    Tied,
    Bank(Tensor<W1>),
}

pub struct Layer<W1: Dtype> {
    pub mixer: Mixer<W1>,
    pub mixer_norm: Norm<W1>,
    pub mlp_norm: Norm<W1>,
    pub mlp: Mlp<W1>,
}

pub enum Mixer<W1: Dtype> {
    Attn(Attn<W1>),
    Gdn(Gdn<W1>),
}

pub struct Attn<W1: Dtype> {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub qg_proj: Tensor<W1>,
    pub k_proj: Tensor<W1>,
    pub v_proj: Tensor<W1>,
    pub o_proj: Tensor<W1>,
    pub q_norm: Norm<W1>,
    pub k_norm: Norm<W1>,
    pub kv: CacheRef,
}

pub struct Gdn<W1: Dtype> {
    pub k_heads: u32,
    pub v_heads: u32,
    pub k_dim: u32,
    pub v_dim: u32,
    pub conv_kernel: u32,
    pub in_qkvz: Tensor<W1>,
    pub in_ba: Tensor<W1>,
    pub conv: Tensor<W1>,
    pub dt_bias: Tensor<W1>,
    pub a_log: Tensor<W1>,
    pub norm: Norm<W1>,
    pub out_proj: Tensor<W1>,
    pub conv_state: CacheRef,
    pub delta_state: CacheRef,
}

pub enum Mlp<W1: Dtype> {
    Dense {
        gate_up: Tensor<W1>,
        down: Tensor<W1>,
        inter: u32,
    },
    Routed {
        router: Tensor<W1>,
        gate_up: Tensor<W1>,
        down: Tensor<W1>,
        shared_gate_up: Tensor<W1>,
        shared_down: Tensor<W1>,
        shared_gate: Tensor<W1>,
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
    mlp_inter: u32,
    moe: Option<MoeDims>,
    vocab: u32,
    tied: bool,
    norm_eps: f32,
}

impl<W1: Dtype, K: KvDtype, const TP: usize> Model<W1, K, TP> {
    pub fn a3b() -> Self {
        assemble(Dims {
            hidden: 2048, layers: 48, attn_every: 4,
            q_heads: 16, kv_heads: 2, head_dim: 256, rotary_dim: 64, theta: 10_000_000.0,
            k_heads: 16, v_heads: 32, k_dim: 128, v_dim: 128, conv_kernel: 4,
            mlp_inter: 5120,
            moe: Some(MoeDims { experts: 512, top_k: 10, inter: 512, shared_inter: 512 }),
            vocab: 151_936, tied: false, norm_eps: 1e-6,
        })
    }

    pub fn d3b() -> Self {
        assemble(Dims {
            hidden: 2048, layers: 24, attn_every: 4,
            q_heads: 16, kv_heads: 2, head_dim: 256, rotary_dim: 64, theta: 10_000_000.0,
            k_heads: 16, v_heads: 32, k_dim: 128, v_dim: 128, conv_kernel: 4,
            mlp_inter: 8192,
            moe: None,
            vocab: 151_936, tied: true, norm_eps: 1e-6,
        })
    }
}

fn assemble<W1: Dtype, K: KvDtype, const TP: usize>(d: Dims) -> Model<W1, K, TP> {
    let hidden = d.hidden as u64;
    let attn_at = |l: u32| l % d.attn_every == d.attn_every - 1;

    let layers = (0..d.layers).map(|l| {
        let n = |s: &str| format!("layer.{l}.{s}");
        let norm = |s: &str, w: u64| Norm { weight: Tensor::sym(n(s), [w]), eps: d.norm_eps };
        let mixer = if attn_at(l) {
            let hd = d.head_dim as u64;
            Mixer::Attn(Attn {
                q_heads: d.q_heads,
                kv_heads: d.kv_heads,
                head_dim: d.head_dim,
                rotary_dim: d.rotary_dim,
                theta: d.theta,
                sm_scale: (d.head_dim as f32).sqrt().recip(),
                qg_proj: Tensor::sym(n("qg_proj"), [2 * d.q_heads as u64 * hd, hidden]).columns(),
                k_proj: Tensor::sym(n("k_proj"), [d.kv_heads as u64 * hd, hidden]).columns(),
                v_proj: Tensor::sym(n("v_proj"), [d.kv_heads as u64 * hd, hidden]).columns(),
                o_proj: Tensor::sym(n("o_proj"), [hidden, d.q_heads as u64 * hd]).rows(),
                q_norm: norm("q_norm", hd),
                k_norm: norm("k_norm", hd),
                kv: CacheRef::to(format!("kv.{l}")),
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
                in_qkvz: Tensor::sym(n("in_qkvz"), [qkvz, hidden]).packed([k_w, k_w, v_w, v_w]),
                in_ba: Tensor::sym(n("in_ba"), [2 * d.v_heads as u64, hidden]).packed([d.v_heads as u64, d.v_heads as u64]),
                conv: Tensor::sym(n("conv"), [qkv, d.conv_kernel as u64]).packed([k_w, k_w, v_w]),
                dt_bias: Tensor::sym(n("dt_bias"), [d.v_heads as u64]).columns(),
                a_log: Tensor::sym(n("a_log"), [d.v_heads as u64]).columns(),
                norm: norm("gdn_norm", d.v_dim as u64),
                out_proj: Tensor::sym(n("out_proj"), [hidden, d.v_heads as u64 * d.v_dim as u64]).rows(),
                conv_state: CacheRef::to(format!("conv.{l}")),
                delta_state: CacheRef::to(format!("delta.{l}")),
            })
        };
        let mlp = match &d.moe {
            None => Mlp::Dense {
                gate_up: Tensor::sym(n("gate_up"), [2 * d.mlp_inter as u64, hidden]).packed([d.mlp_inter as u64, d.mlp_inter as u64]),
                down: Tensor::sym(n("down"), [hidden, d.mlp_inter as u64]).rows(),
                inter: d.mlp_inter,
            },
            Some(m) => Mlp::Routed {
                router: Tensor::sym(n("router"), [m.experts as u64, hidden]),
                gate_up: Tensor::sym(n("experts_gate_up"), [m.experts as u64, 2 * m.inter as u64, hidden]).experts(),
                down: Tensor::sym(n("experts_down"), [m.experts as u64, hidden, m.inter as u64]).experts(),
                shared_gate_up: Tensor::sym(n("shared_gate_up"), [2 * m.shared_inter as u64, hidden]).packed([m.shared_inter as u64, m.shared_inter as u64]),
                shared_down: Tensor::sym(n("shared_down"), [hidden, m.shared_inter as u64]).rows(),
                shared_gate: Tensor::sym(n("shared_gate"), [1, hidden]),
                experts: m.experts,
                top_k: m.top_k,
                inter: m.inter,
                shared_inter: m.shared_inter,
            },
        };
        Layer {
            mixer,
            mixer_norm: norm("mixer_norm", hidden),
            mlp_norm: norm("mlp_norm", hidden),
            mlp,
        }
    }).collect();

    Model {
        hidden: d.hidden,
        embed: Tensor::sym("embed", [d.vocab as u64, hidden]),
        head: if d.tied {
            Head::Tied
        } else {
            Head::Bank(Tensor::sym("lm_head", [d.vocab as u64, hidden]))
        },
        layers,
        final_norm: Norm { weight: Tensor::sym("final_norm", [hidden]), eps: d.norm_eps },
        _kv: PhantomData,
    }
}
