use std::marker::PhantomData;

use model_dsl::axes::{Dtype, F32, KvDtype};
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
    pub res_blend: Option<ResBlend<W1>>,
    pub mixer: Mixer<W1>,
    pub mixer_norm: Norm<W1>,
    pub mlp_norm: Norm<W1>,
    pub mlp: Mlp<W1, W2>,
}

pub struct ResBlend<W1: Dtype> {
    pub norm: Norm<W1>,
    pub proj: Tensor<W1>,
}

pub enum Mixer<W1: Dtype> {
    Mla(Mla<W1>),
    Kda(Kda<W1>),
}

pub struct Mla<W1: Dtype> {
    pub heads: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub sm_scale: f32,
    pub q_a_proj: Tensor<W1>,
    pub q_a_norm: Norm<W1>,
    pub q_b_proj: Tensor<W1>,
    pub kv_a_proj: Tensor<W1>,
    pub kv_a_norm: Norm<W1>,
    pub kv_b_proj: Tensor<W1>,
    pub gate: Option<Tensor<W1>>,
    pub o_proj: Tensor<W1>,
    pub kv: CacheRef,
}

pub struct Kda<W1: Dtype> {
    pub heads: u32,
    pub head_dim: u32,
    pub conv_kernel: u32,
    pub norm_eps: f32,
    pub qkv: Tensor<W1>,
    pub conv: Tensor<W1>,
    pub f_a: Tensor<W1>,
    pub f_b: Tensor<W1>,
    pub b: Tensor<W1>,
    /// THE DECAY PAIR IS F32, and the kernel is what says so:
    /// `ssm/kda.cuh`'s `kda_gate_beta` takes `const float* __restrict__
    /// A_log` beside `const float* __restrict__ dt_bias`, both `float`,
    /// and `ssm.kda_step` declares both slots `Const<Tensor<f32>>`. Qwen's
    /// gated-delta pair splits the other way — `A_log` f32, `dt_bias` at
    /// the model's element — because ITS kernel does.
    ///
    /// The shapes come from the same place. `dt_bias[h * D + d]` is read
    /// per CHANNEL (KDA's forget gate is channel-wise) and `A_log[h]` per
    /// head, so the two are `[heads, head_dim]` and `[heads]`.
    pub dt_bias: Tensor<F32>,
    pub a_log: Tensor<F32>,
    pub gate: Tensor<W1>,
    pub o_norm: Norm<W1>,
    pub o_proj: Tensor<W1>,
    pub conv_state: CacheRef,
    pub delta_state: CacheRef,
}

#[allow(clippy::large_enum_variant)] // a per-layer weight-bank record, built once at trace; boxing buys nothing and costs every reader a deref
pub enum Mlp<W1: Dtype, W2: Dtype> {
    Dense {
        gate_up: Tensor<W1>,
        down: Tensor<W1>,
        inter: u32,
        beta: f32,
        up_cap: Option<f32>,
    },
    Routed {
        router: Tensor<W1>,
        gate_up: Tensor<W2>,
        down: Tensor<W2>,
        shared: Option<Shared<W1>>,
        experts: u32,
        top_k: u32,
        routed_scaling: f32,
        inter: u32,
        beta: f32,
        up_cap: Option<f32>,
    },
}

pub struct Shared<W1: Dtype> {
    pub gate_up: Tensor<W1>,
    pub down: Tensor<W1>,
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

/// THE CUT, AND THE WHOLE OF IT: the dims a rank holds a share of at `TP`
/// ways, with `..d` saying that everything else is replicated.
///
/// Both mixers' head counts and all three intermediates. The MLA half cuts
/// only its head fan — `kv_lora_rank`, `q_lora_rank` and `qk_rope_head_dim`
/// name the latent row every rank writes and caches whole, which is the same
/// reading glm-5 states. The KDA half cuts `heads` and leaves `head_dim`,
/// so its packed `[q | k | v]` projection, its convolution, its per-head
/// `a_log`/`dt_bias` columns and both recurrent slabs come out narrower
/// together, and `kda_o_norm` stays a per-head norm over a width no cut
/// touches.
fn per_rank<const TP: usize>(d: Dims) -> Dims {
    let cut = |what, whole| model_dsl::per_rank(what, whole, TP);
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

impl<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize> Model<W1, W2, K, TP> {
    /// UNVERIFIED — an 8-LAYER STAND-IN, and no Kimi K3 checkpoint is cached
    /// to make it anything else.
    ///
    /// `serve::ROWS` advertises `kimik3-bf16-mxfp4-kv-bf16` as arch
    /// `kimi_k3`, which is the real architecture's name; `layers: 8` with
    /// `dense_layers: 1` is not the real architecture's depth. The depth
    /// that IS load-bearing here is the ratio: at `full_attn_every: 4` and
    /// `res_block: 4` an 8-layer tower traces 2 MLA layers, 6 KDA layers
    /// and 1 residual-ledger blend, which is what makes it exercise
    /// `norm.res_blend` (kimi's variadic point, whose only caller this is)
    /// and both KDA arms beside MLA. Every number is a plausible shape
    /// rather than a config key, and the join that would settle them has
    /// never run.
    pub fn k3() -> Self {
        assemble(Dims {
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
        })
    }
}

fn assemble<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize>(d: Dims) -> Model<W1, W2, K, TP> {
    let d = per_rank::<TP>(d);
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
                weight: Tensor::sym(n(s), [w]),
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
                    q_a_proj: Tensor::sym(n("q_a_proj"), [a.q_lora_rank as u64, hidden]),
                    q_a_norm: norm("q_a_norm", a.q_lora_rank as u64),
                    q_b_proj: Tensor::sym(n("q_b_proj"), [q_b_width, a.q_lora_rank as u64])
                        .columns(),
                    kv_a_proj: Tensor::sym(n("kv_a_proj"), [kv_a_width, hidden]),
                    kv_a_norm: norm("kv_a_norm", a.kv_lora_rank as u64),
                    kv_b_proj: Tensor::sym(n("kv_b_proj"), [kv_b_width, a.kv_lora_rank as u64])
                        .columns(),
                    gate: a
                        .output_gate
                        .then(|| Tensor::sym(n("o_gate"), [v_width, hidden]).columns()),
                    o_proj: Tensor::sym(n("o_proj"), [hidden, v_width]).rows(),
                    kv: CacheRef::to(format!("kv.{l}")),
                })
            } else {
                Mixer::Kda(Kda {
                    heads: k.heads,
                    head_dim: k.head_dim,
                    conv_kernel: k.conv_kernel,
                    norm_eps: k.norm_eps,
                    qkv: Tensor::sym(n("kda_qkv"), [3 * kda_width, hidden])
                        .packed([kda_width, kda_width, kda_width]),
                    conv: Tensor::sym(n("kda_conv"), [3 * kda_width, k.conv_kernel as u64])
                        .packed([kda_width, kda_width, kda_width]),
                    f_a: Tensor::sym(n("kda_f_a"), [k.head_dim as u64, hidden]),
                    f_b: Tensor::sym(n("kda_f_b"), [kda_width, k.head_dim as u64]).columns(),
                    b: Tensor::sym(n("kda_b"), [k.heads as u64, hidden]).columns(),
                    dt_bias: Tensor::<F32>::sym(
                        n("kda_dt_bias"),
                        [k.heads as u64, k.head_dim as u64],
                    )
                    .columns(),
                    a_log: Tensor::<F32>::sym(n("kda_a_log"), [k.heads as u64]).columns(),
                    gate: Tensor::sym(n("kda_gate"), [kda_width, hidden]).columns(),
                    o_norm: Norm {
                        weight: Tensor::sym(n("kda_o_norm"), [k.head_dim as u64]),
                        eps: k.norm_eps,
                    },
                    o_proj: Tensor::sym(n("kda_o_proj"), [hidden, kda_width]).rows(),
                    conv_state: CacheRef::to(format!("conv.{l}")),
                    delta_state: CacheRef::to(format!("delta.{l}")),
                })
            };
            let mlp = if moe_at(l) {
                let m = &d.moe;
                let inter = m.inter as u64;
                let shared_inter = m.shared_inter as u64;
                Mlp::Routed {
                    router: Tensor::sym(n("router"), [m.experts as u64, hidden]),
                    gate_up: Tensor::sym(
                        n("experts_gate_up"),
                        [m.experts as u64, 2 * inter, hidden],
                    )
                    .bank([inter, inter]),
                    down: Tensor::sym(n("experts_down"), [m.experts as u64, hidden, inter]).rows(),
                    shared: (m.shared_inter > 0).then(|| Shared {
                        gate_up: Tensor::sym(n("shared_gate_up"), [2 * shared_inter, hidden])
                            .packed([shared_inter, shared_inter]),
                        down: Tensor::sym(n("shared_down"), [hidden, shared_inter]).rows(),
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
                    gate_up: Tensor::sym(n("gate_up"), [2 * inter, hidden]).packed([inter, inter]),
                    down: Tensor::sym(n("down"), [hidden, inter]).rows(),
                    inter: d.dense_inter,
                    beta: d.situ_beta,
                    up_cap: d.situ_cap,
                }
            };
            Layer {
                res_blend: blend_at(l).then(|| ResBlend {
                    norm: norm("res_norm", hidden),
                    proj: Tensor::sym(n("res_proj"), [1, hidden]),
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
