use std::marker::PhantomData;

use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{CacheRef, Norm, Tensor};

pub struct Model<W1: Dtype, K: KvDtype, const TP: usize = 1> {
    pub hidden: u32,
    pub softcap: Option<f32>,
    pub embed: Tensor<W1>,
    pub head: Head<W1>,
    pub ple: Option<Ple<W1>>,
    pub layers: Vec<Layer<W1>>,
    pub final_norm: Norm<W1>,
    _kv: PhantomData<K>,
}

pub enum Head<W1: Dtype> {
    Tied,
    Bank(Tensor<W1>),
}

pub struct Ple<W1: Dtype> {
    pub dim: u32,
    pub table: Tensor<W1>,
    pub model_proj: Tensor<W1>,
    pub model_norm: Norm<W1>,
    pub per_layer: Vec<PleLayer<W1>>,
}

pub struct PleLayer<W1: Dtype> {
    pub gate: Tensor<W1>,
    pub proj: Tensor<W1>,
    pub norm: Norm<W1>,
    pub scalar: Tensor<W1>,
}

pub struct Layer<W1: Dtype> {
    pub attn: Attn<W1>,
    pub o_proj: Tensor<W1>,
    pub attn_norm: Norm<W1>,
    pub post_attn_norm: Norm<W1>,
    pub pre_ffw_norm: Norm<W1>,
    pub post_ffw_norm: Norm<W1>,
    pub gate_up: GateUp<W1>,
    pub down: Tensor<W1>,
}

pub enum GateUp<W1: Dtype> {
    Packed { bank: Tensor<W1>, inter: u32 },
    Split { gate: Tensor<W1>, up: Tensor<W1> },
}

pub struct Attn<W1: Dtype> {
    pub kind: AttnKind,
    pub q_heads: u32,
    pub sm_scale: f32,
    pub q_norm: Norm<W1>,
    pub kv: CacheRef,
    pub banks: AttnBanks<W1>,
}

pub enum AttnKind {
    Full { head_dim: u32, kv_heads: u32, rotary_dim: u32, theta: f32 },
    Sliding { head_dim: u32, kv_heads: u32, window: u32, theta: f32 },
}

impl AttnKind {
    pub fn head_dim(&self) -> u32 {
        match *self { AttnKind::Full { head_dim, .. } | AttnKind::Sliding { head_dim, .. } => head_dim }
    }
    pub fn kv_heads(&self) -> u32 {
        match *self { AttnKind::Full { kv_heads, .. } | AttnKind::Sliding { kv_heads, .. } => kv_heads }
    }
    pub fn theta(&self) -> f32 {
        match *self { AttnKind::Full { theta, .. } | AttnKind::Sliding { theta, .. } => theta }
    }
    pub fn window(&self) -> Option<u32> {
        match *self {
            AttnKind::Full { .. } => None,
            AttnKind::Sliding { window, .. } => Some(window),
        }
    }
    pub fn sliding(&self) -> bool {
        matches!(self, AttnKind::Sliding { .. })
    }
}

pub enum AttnBanks<W1: Dtype> {
    Owned { qkv: Qkv<W1>, k_norm: Norm<W1> },
    Shared { q_proj: Tensor<W1> },
}

pub enum Qkv<W1: Dtype> {
    Packed(Tensor<W1>),
    Split { q: Tensor<W1>, k: Tensor<W1>, v: Tensor<W1> },
}

struct Dims {
    hidden: u32,
    layers: u32,
    full_every: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    global_head_dim: u32,
    global_kv_heads: u32,
    global_rotary_dim: u32,
    theta_local: f32,
    theta_global: f32,
    sm_scale: f32,
    intermediate: u32,
    vocab: u32,
    shared_tail: u32,
    ple_dim: u32,
    double_wide_shared: bool,
    softcap: Option<f32>,
    window: u32,
    norm_eps: f32,
}

impl<W1: Dtype, K: KvDtype, const TP: usize> Model<W1, K, TP> {
    pub fn e4b() -> Self {
        assemble(Dims {
            hidden: 2560, layers: 42, full_every: 6,
            q_heads: 8, kv_heads: 2, head_dim: 256,
            global_head_dim: 512, global_kv_heads: 2, global_rotary_dim: 128,
            theta_local: 10_000.0, theta_global: 1_000_000.0, sm_scale: 1.0,
            intermediate: 10_240, vocab: 262_144,
            shared_tail: 18, ple_dim: 256, double_wide_shared: false,
            softcap: Some(30.0), window: 512, norm_eps: 1e-6,
        })
    }

    pub fn e2b() -> Self {
        assemble(Dims {
            hidden: 1536, layers: 35, full_every: 5,
            q_heads: 8, kv_heads: 1, head_dim: 256,
            global_head_dim: 512, global_kv_heads: 1, global_rotary_dim: 128,
            theta_local: 10_000.0, theta_global: 1_000_000.0, sm_scale: 1.0,
            intermediate: 6144, vocab: 262_144,
            shared_tail: 20, ple_dim: 256, double_wide_shared: true,
            softcap: Some(30.0), window: 512, norm_eps: 1e-6,
        })
    }

    pub fn b31() -> Self {
        assemble(Dims {
            hidden: 5376, layers: 60, full_every: 6,
            q_heads: 32, kv_heads: 16, head_dim: 256,
            global_head_dim: 512, global_kv_heads: 4, global_rotary_dim: 128,
            theta_local: 10_000.0, theta_global: 1_000_000.0, sm_scale: 1.0,
            intermediate: 21_504, vocab: 262_144,
            shared_tail: 0, ple_dim: 0, double_wide_shared: false,
            softcap: Some(30.0), window: 512, norm_eps: 1e-6,
        })
    }
}

fn assemble<W1: Dtype, K: KvDtype, const TP: usize>(d: Dims) -> Model<W1, K, TP> {
    let hidden = d.hidden as u64;
    let full_at = |l: u32| l % d.full_every == d.full_every - 1;
    let shared_at = |l: u32| l >= d.layers - d.shared_tail;
    let source = |l: u32| {
        (0..l).rev()
            .find(|&s| !shared_at(s) && full_at(s) == full_at(l))
            .unwrap_or(l)
    };
    let kind = |l: u32| {
        if full_at(l) {
            AttnKind::Full {
                head_dim: d.global_head_dim, kv_heads: d.global_kv_heads,
                rotary_dim: d.global_rotary_dim, theta: d.theta_global,
            }
        } else {
            AttnKind::Sliding {
                head_dim: d.head_dim, kv_heads: d.kv_heads,
                window: d.window, theta: d.theta_local,
            }
        }
    };
    let inter = |l: u32| {
        if d.double_wide_shared && shared_at(l) { 2 * d.intermediate } else { d.intermediate }
    };

    let layers = (0..d.layers).map(|l| {
        let n = |s: &str| format!("layer.{l}.{s}");
        let norm = |s: &str, w: u64| Norm { weight: Tensor::sym(n(s), [w]), eps: d.norm_eps };
        let ki = kind(l);
        let hd = ki.head_dim() as u64;
        let q_w = d.q_heads as u64 * hd;
        let kv_w = ki.kv_heads() as u64 * hd;
        let iw = inter(l) as u64;
        Layer {
            attn: Attn {
                q_heads: d.q_heads,
                sm_scale: d.sm_scale,
                q_norm: norm("q_norm", hd),
                kv: CacheRef::to(format!("kv.{}", if shared_at(l) { source(l) } else { l })),
                banks: if shared_at(l) {
                    AttnBanks::Shared {
                        q_proj: Tensor::sym(n("q_proj"), [q_w, hidden]).columns(),
                    }
                } else {
                    AttnBanks::Owned {
                        qkv: Qkv::Packed(Tensor::sym(n("qkv"), [q_w + 2 * kv_w, hidden]).packed([q_w, kv_w, kv_w])),
                        k_norm: norm("k_norm", hd),
                    }
                },
                kind: ki,
            },
            o_proj: Tensor::sym(n("o_proj"), [hidden, q_w]).rows(),
            attn_norm: norm("attn_norm", hidden),
            post_attn_norm: norm("post_attn_norm", hidden),
            pre_ffw_norm: norm("pre_ffw_norm", hidden),
            post_ffw_norm: norm("post_ffw_norm", hidden),
            gate_up: GateUp::Packed { bank: Tensor::sym(n("gate_up"), [2 * iw, hidden]).packed([iw, iw]), inter: inter(l) },
            down: Tensor::sym(n("down"), [hidden, iw]).rows(),
        }
    }).collect();

    Model {
        hidden: d.hidden,
        softcap: d.softcap,
        embed: Tensor::sym("embed", [d.vocab as u64, hidden]),
        head: Head::Tied,
        ple: (d.ple_dim > 0).then(|| {
            let ple = d.ple_dim as u64;
            Ple {
                dim: d.ple_dim,
                table: Tensor::sym("ple.table", [d.vocab as u64, d.layers as u64 * ple]),
                model_proj: Tensor::sym("ple.model_proj", [d.layers as u64 * ple, hidden]),
                model_norm: Norm { weight: Tensor::sym("ple.model_norm", [ple]), eps: d.norm_eps },
                per_layer: (0..d.layers).map(|l| PleLayer {
                    gate: Tensor::sym(format!("layer.{l}.ple_gate"), [ple, hidden]),
                    proj: Tensor::sym(format!("layer.{l}.ple_proj"), [hidden, ple]),
                    norm: Norm {
                        weight: Tensor::sym(format!("layer.{l}.ple_norm"), [hidden]),
                        eps: d.norm_eps,
                    },
                    scalar: Tensor::sym(format!("layer.{l}.ple_scalar"), [1]),
                }).collect(),
            }
        }),
        layers,
        final_norm: Norm { weight: Tensor::sym("final_norm", [hidden]), eps: d.norm_eps },
        _kv: PhantomData,
    }
}
