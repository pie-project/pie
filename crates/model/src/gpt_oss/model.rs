use std::marker::PhantomData;

use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{CacheRef, Norm, Tensor, Yarn};

pub struct Model<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize = 1> {
    pub hidden: u32,
    pub vocab: u32,
    pub embed: Tensor<W1>,
    pub head: Tensor<W1>,
    pub layers: Vec<Layer<W1, W2>>,
    pub final_norm: Norm<W1>,
    _kv: PhantomData<K>,
}

pub struct Layer<W1: Dtype, W2: Dtype> {
    pub attn: Attn<W1>,
    pub attn_norm: Norm<W1>,
    pub mlp_norm: Norm<W1>,
    pub mlp: Moe<W1, W2>,
}

pub struct Attn<W1: Dtype> {
    pub kind: AttnKind,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub sm_scale: f32,
    pub rope: Yarn,
    pub q_proj: Tensor<W1>,
    pub q_bias: Tensor<W1>,
    pub k_proj: Tensor<W1>,
    pub k_bias: Tensor<W1>,
    pub v_proj: Tensor<W1>,
    pub v_bias: Tensor<W1>,
    pub o_proj: Tensor<W1>,
    pub o_bias: Tensor<W1>,
    pub sinks: Tensor<W1>,
    pub kv: CacheRef,
}

pub enum AttnKind {
    Full,
    Sliding { window: u32 },
}

impl AttnKind {
    pub fn window(&self) -> Option<u32> {
        match *self {
            AttnKind::Full => None,
            AttnKind::Sliding { window } => Some(window),
        }
    }
}

pub struct Moe<W1: Dtype, W2: Dtype> {
    pub experts: u32,
    pub top_k: u32,
    pub inter: u32,
    pub swiglu_limit: f32,
    pub swiglu_alpha: f32,
    pub router: Tensor<W1>,
    pub router_bias: Tensor<W1>,
    pub gate_up: Tensor<W2>,
    pub gate_up_bias: Tensor<W1>,
    pub down: Tensor<W2>,
    pub down_bias: Tensor<W1>,
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

/// THE CUT, AND THE WHOLE OF IT: the dims a rank holds a share of at `TP`
/// ways, with `..d` saying that everything else is replicated.
///
/// The head counts and the expert intermediate. `head_dim` is one head's own
/// width and does not divide; `experts` is the fan the router scores across
/// and every rank scores the whole of it, because the routed leg is cut
/// INSIDE each expert (`expert_gate_up_bank` packed on its `2 * inter` axis,
/// `expert_down_bank` on its `inter` one) and not across the bank.
///
/// A SEAM THIS CUT DOES NOT CLOSE, and it is a numeric one rather than a
/// width: `o_bias` and `expert_down_bias` are added by the row-parallel
/// statement that precedes `dist.all_reduce`, so a `world`-way deployment
/// sums them `world` times. Every other family in this catalog lands its
/// attention and its feed-forward without a bias and is unaffected. Fixing
/// it is a statement change (a bias that lands after the reduce, or a rank
/// that owns it), which is the driver-side rank story and not this column.
fn per_rank<const TP: usize>(d: Dims) -> Dims {
    let cut = |what, whole| model_dsl::per_rank(what, whole, TP);
    Dims {
        q_heads: cut("q_heads", d.q_heads),
        kv_heads: cut("kv_heads", d.kv_heads),
        inter: cut("inter", d.inter),
        ..d
    }
}

impl<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize> Model<W1, W2, K, TP> {
    pub fn b20() -> Self {
        assemble(Dims {
            hidden: 2880, layers: 24,
            q_heads: 64, kv_heads: 8, head_dim: 64,
            theta: 150_000.0,
            yarn_factor: 32.0, yarn_beta_fast: 32.0, yarn_beta_slow: 1.0,
            yarn_attention_factor: 1.346_573_6, yarn_original_max_position: 4096,
            window: 128,
            experts: 32, top_k: 4, inter: 2880,
            swiglu_limit: 7.0, swiglu_alpha: 1.702,
            vocab: 201_088, norm_eps: 1e-5,
        })
    }

    pub fn b120() -> Self {
        assemble(Dims {
            hidden: 2880, layers: 36,
            q_heads: 64, kv_heads: 8, head_dim: 64,
            theta: 150_000.0,
            yarn_factor: 32.0, yarn_beta_fast: 32.0, yarn_beta_slow: 1.0,
            yarn_attention_factor: 1.346_573_6, yarn_original_max_position: 4096,
            window: 128,
            experts: 128, top_k: 4, inter: 2880,
            swiglu_limit: 7.0, swiglu_alpha: 1.702,
            vocab: 201_088, norm_eps: 1e-5,
        })
    }
}

fn assemble<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize>(d: Dims) -> Model<W1, W2, K, TP> {
    let d = per_rank::<TP>(d);
    let hidden = d.hidden as u64;
    let hd = d.head_dim as u64;
    let q_w = d.q_heads as u64 * hd;
    let kv_w = d.kv_heads as u64 * hd;
    let experts = d.experts as u64;
    let inter = d.inter as u64;
    let sliding_at = |l: u32| l.is_multiple_of(2);

    let layers = (0..d.layers).map(|l| {
        let n = |s: &str| format!("layer.{l}.{s}");
        let norm = |s: &str, w: u64| Norm { weight: Tensor::sym(n(s), [w]), eps: d.norm_eps };
        Layer {
            attn: Attn {
                kind: if sliding_at(l) {
                    AttnKind::Sliding { window: d.window }
                } else {
                    AttnKind::Full
                },
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
                q_proj: Tensor::sym(n("q_proj"), [q_w, hidden]).columns(),
                q_bias: Tensor::sym(n("q_bias"), [q_w]).columns(),
                k_proj: Tensor::sym(n("k_proj"), [kv_w, hidden]).columns(),
                k_bias: Tensor::sym(n("k_bias"), [kv_w]).columns(),
                v_proj: Tensor::sym(n("v_proj"), [kv_w, hidden]).columns(),
                v_bias: Tensor::sym(n("v_bias"), [kv_w]).columns(),
                o_proj: Tensor::sym(n("o_proj"), [hidden, q_w]).rows(),
                o_bias: Tensor::sym(n("o_bias"), [hidden]),
                sinks: Tensor::sym(n("attn_sinks"), [d.q_heads as u64]).columns(),
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
                router: Tensor::sym(n("router"), [experts, hidden]),
                router_bias: Tensor::sym(n("router_bias"), [experts]),
                gate_up: Tensor::sym(n("expert_gate_up_bank"), [experts, 2 * inter, hidden]).bank([inter, inter]),
                gate_up_bias: Tensor::sym(n("expert_gate_up_bias"), [experts, 2 * inter]).bank([inter, inter]),
                down: Tensor::sym(n("expert_down_bank"), [experts, hidden, inter]).rows(),
                down_bias: Tensor::sym(n("expert_down_bias"), [experts, hidden]),
            },
        }
    }).collect();

    Model {
        hidden: d.hidden,
        vocab: d.vocab,
        embed: Tensor::sym("embed", [d.vocab as u64, hidden]),
        head: Tensor::sym("lm_head", [d.vocab as u64, hidden]),
        layers,
        final_norm: Norm { weight: Tensor::sym("final_norm", [hidden]), eps: d.norm_eps },
        _kv: PhantomData,
    }
}
