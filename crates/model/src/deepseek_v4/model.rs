use std::marker::PhantomData;

use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{CacheRef, Norm, Tensor};

pub struct Model<W1: Dtype, K: KvDtype, const TP: usize = 1> {
    pub hidden: u32,
    pub vocab: u32,
    pub hyper: Hyper<W1>,
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

pub struct Hyper<W1: Dtype> {
    pub streams: u32,
    pub norm_eps: f32,
    pub gate_eps: f32,
    pub alpha: f32,
    pub sinkhorn: u32,
    pub head_scale: Tensor<W1>,
    pub head_base: Tensor<W1>,
}

pub struct Mix<W1: Dtype> {
    pub scale: Tensor<W1>,
    pub base: Tensor<W1>,
}

pub struct Layer<W1: Dtype> {
    pub attn_mix: Mix<W1>,
    pub attn: Attn<W1>,
    pub mlp_mix: Mix<W1>,
    pub mlp: Mlp<W1>,
}

pub struct Attn<W1: Dtype> {
    pub heads: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub window: u32,
    pub q_down: Tensor<W1>,
    pub q_norm: Norm<W1>,
    pub q_up: Tensor<W1>,
    pub kv_down: Tensor<W1>,
    pub kv_norm: Norm<W1>,
    pub o_down: Tensor<W1>,
    pub o_up: Tensor<W1>,
    pub sink: Tensor<W1>,
    pub kv: CacheRef,
    pub pool: Option<Pool>,
}

pub struct Pool {
    pub ratio: u32,
    pub entries: CacheRef,
}

pub enum Mlp<W1: Dtype> {
    Dense {
        gate_up: Tensor<W1>,
        down: Tensor<W1>,
        inter: u32,
        limit: f32,
    },
    Routed {
        router: Tensor<W1>,
        bias: Tensor<W1>,
        gate_up: Tensor<W1>,
        down: Tensor<W1>,
        experts: u32,
        top_k: u32,
        inter: u32,
        limit: f32,
        renorm: bool,
        scaling: f32,
    },
}

struct Dims {
    hidden: u32,
    layers: u32,
    dense_layers: u32,
    ratios: &'static [u32],
    heads: u32,
    head_dim: u32,
    q_lora: u32,
    o_lora: u32,
    rope_dim: u32,
    theta: f32,
    window: u32,
    streams: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
    dense_inter: u32,
    experts: u32,
    top_k: u32,
    moe_inter: u32,
    renorm: bool,
    scaling: f32,
    swiglu_limit: f32,
    vocab: u32,
    tied: bool,
    norm_eps: f32,
}

/// THE CUT, AND THE WHOLE OF IT: the dims a rank holds a share of at `TP`
/// ways, with `..d` saying that everything else is replicated.
///
/// The attention heads and both intermediates. `q_lora` and `o_lora` do not
/// divide: `q_down` and `o_up` sit on the replicated side of the two
/// projections that DO cut (`q_up` expands the lora into the head fan,
/// `o_down` contracts the head fan back into it), and `o_down`'s partial
/// rows are exactly what the `dist.all_reduce` immediately after it sums.
/// The hyper-connection stack (`streams`) is a factor of the residual width,
/// which the reduce closes over and no rank holds a piece of.
///
/// THE COMPRESSED PLANE CUTS WITH THE HEADS, and it is the reason `kv_down`
/// carries a mark at all. This family's cache row is `[1, heads * head_dim]`
/// — one plane per token serving as both k and v — so it is per-head, and a
/// rank attending its own `heads / world` slice of the query fan must hold
/// the matching slice of the plane. A plane replicated whole beside a cut
/// query fan would need a HEAD OFFSET at the fire, which no statement here
/// carries.
///
/// A SEAM THIS CUT DOES NOT CLOSE: `kv_norm` is a whole-row `norm.rmsnorm`
/// over that now-cut plane, and a root-mean-square over a row split across
/// ranks is a cross-rank sum that no point in this tree states. Every other
/// norm this text applies is either over a replicated row (`q_norm` on the
/// lora, the hyper norms on the stack) or per head
/// (`norm.rmsnorm_no_scale` at `head_dim`), so this is the one. The fix is a
/// model-truth question — DeepSeek normalizes the compressed latent, and
/// whether this plane's norm is per head is a checkpoint's answer, and no
/// dsv4 checkpoint is cached.
fn per_rank<const TP: usize>(d: Dims) -> Dims {
    let cut = |what, whole| model_dsl::per_rank(what, whole, TP);
    Dims {
        heads: cut("heads", d.heads),
        dense_inter: cut("dense inter", d.dense_inter),
        moe_inter: cut("moe inter", d.moe_inter),
        ..d
    }
}

impl<W1: Dtype, K: KvDtype, const TP: usize> Model<W1, K, TP> {
    pub fn base() -> Self {
        assemble(Dims {
            hidden: 2048, layers: 6, dense_layers: 1, ratios: &[1, 2, 4],
            heads: 16, head_dim: 128, q_lora: 768, o_lora: 512,
            rope_dim: 64, theta: 10_000.0, window: 2048,
            streams: 4, gate_eps: 1e-6, alpha: 2.0, sinkhorn: 20,
            dense_inter: 5632,
            experts: 64, top_k: 6, moe_inter: 1024,
            renorm: false, scaling: 2.5, swiglu_limit: 7.0,
            vocab: 129_280, tied: true, norm_eps: 1e-5,
        })
    }
}

fn assemble<W1: Dtype, K: KvDtype, const TP: usize>(d: Dims) -> Model<W1, K, TP> {
    let d = per_rank::<TP>(d);
    let hidden = d.hidden as u64;
    let mult = d.streams as u64;
    let q_w = d.heads as u64 * d.head_dim as u64;
    let q_lora = d.q_lora as u64;
    let o_lora = d.o_lora as u64;
    let dense_inter = d.dense_inter as u64;
    let moe_inter = d.moe_inter as u64;

    let layers = (0..d.layers).map(|l| {
        let n = |s: &str| format!("layer.{l}.{s}");
        let norm = |s: &str, w: u64| Norm { weight: Tensor::sym(n(s), [w]), eps: d.norm_eps };
        let mix = |s: &str| Mix {
            scale: Tensor::sym(n(&format!("{s}_scale")), [3]),
            base: Tensor::sym(n(&format!("{s}_base")), [2 * mult + mult * mult]),
        };
        Layer {
            attn_mix: mix("attn_mix"),
            attn: Attn {
                heads: d.heads,
                head_dim: d.head_dim,
                rope_dim: d.rope_dim,
                theta: d.theta,
                sm_scale: (d.head_dim as f32).sqrt().recip(),
                window: d.window,
                q_down: Tensor::sym(n("q_down"), [q_lora, hidden]),
                q_norm: norm("q_norm", q_lora),
                q_up: Tensor::sym(n("q_up"), [q_w, q_lora]).columns(),
                kv_down: Tensor::sym(n("kv_down"), [q_w, hidden]).columns(),
                kv_norm: Norm {
                    weight: Tensor::sym(n("kv_norm"), [q_w]).columns(),
                    eps: d.norm_eps,
                },
                o_down: Tensor::sym(n("o_down"), [o_lora, q_w]).rows(),
                o_up: Tensor::sym(n("o_up"), [hidden, o_lora]),
                sink: Tensor::sym(n("attn_sink"), [d.heads as u64]).columns(),
                kv: CacheRef::to(format!("kv.{l}")),
                pool: d.ratios.get(l as usize).copied().filter(|r| *r > 0).map(|ratio| Pool {
                    ratio,
                    entries: CacheRef::to(format!("pool.{l}")),
                }),
            },
            mlp_mix: mix("mlp_mix"),
            mlp: if l < d.dense_layers {
                Mlp::Dense {
                    gate_up: Tensor::sym(n("gate_up"), [2 * dense_inter, hidden]).packed([dense_inter, dense_inter]),
                    down: Tensor::sym(n("down"), [hidden, dense_inter]).rows(),
                    inter: d.dense_inter,
                    limit: d.swiglu_limit,
                }
            } else {
                Mlp::Routed {
                    router: Tensor::sym(n("router"), [d.experts as u64, hidden]),
                    bias: Tensor::sym(n("router_bias"), [d.experts as u64]),
                    gate_up: Tensor::sym(n("experts_gate_up"), [d.experts as u64, 2 * moe_inter, hidden]).bank([moe_inter, moe_inter]),
                    down: Tensor::sym(n("experts_down"), [d.experts as u64, hidden, moe_inter]).rows(),
                    experts: d.experts,
                    top_k: d.top_k,
                    inter: d.moe_inter,
                    limit: d.swiglu_limit,
                    renorm: d.renorm,
                    scaling: d.scaling,
                }
            },
        }
    }).collect();

    Model {
        hidden: d.hidden,
        vocab: d.vocab,
        hyper: Hyper {
            streams: d.streams,
            norm_eps: d.norm_eps,
            gate_eps: d.gate_eps,
            alpha: d.alpha,
            sinkhorn: d.sinkhorn,
            head_scale: Tensor::sym("hyper.head_scale", [1]),
            head_base: Tensor::sym("hyper.head_base", [mult]),
        },
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
