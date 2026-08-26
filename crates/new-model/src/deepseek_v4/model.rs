//! The DeepSeek V4 declaration, de-genericized (design §5, decision #18): the
//! old `Model<W1: Dtype, K: KvDtype, const TP: usize>` phantom tree is gone —
//! `tp` is a runtime field, each weight carries its `Repr`, and the kv dtype
//! is nobody's parameter. Names and the per-layer scheme are unchanged from
//! the old crate: weights intern by name, so the checkpoint mapping carries
//! over untouched.

use new_model_dsl::{CacheRef, Norm, Weight};
use new_model_ir::Repr;

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,
    pub hyper: Hyper,

    pub embed: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Norm,
}

pub struct Hyper {
    pub streams: u32,
    pub norm_eps: f32,
    pub gate_eps: f32,
    pub alpha: f32,
    pub sinkhorn: u32,

    pub head_scale: Weight,
    pub head_base: Weight,
}

pub struct Mix {
    pub scale: Weight,
    pub base: Weight,
}

pub struct Layer {
    pub attn_mix: Mix,
    pub attn: Attn,
    pub mlp_mix: Mix,
    pub mlp: Mlp,
}

pub struct Attn {
    pub heads: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub window: u32,
    pub q_down: Weight,
    pub q_norm: Norm,
    pub q_up: Weight,
    pub kv_down: Weight,
    pub kv_norm: Norm,
    pub o_down: Weight,
    pub o_up: Weight,
    pub sink: Weight,
    pub kv: CacheRef,
    pub pool: Option<Pool>,
}

pub struct Pool {
    pub ratio: u32,
    pub entries: CacheRef,
}

pub enum Mlp {
    Dense {
        gate_up: Weight,
        down: Weight,
        inter: u32,
        limit: f32,
    },
    Routed {
        router: Weight,
        bias: Weight,
        gate_up: Weight,
        down: Weight,
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
    norm_eps: f32,
}

fn per_rank(d: Dims, tp: u32) -> Dims {
    let cut = |what, whole| new_model_dsl::per_rank(what, whole, tp as usize);
    Dims {
        heads: cut("heads", d.heads),
        dense_inter: cut("dense inter", d.dense_inter),
        moe_inter: cut("moe inter", d.moe_inter),
        ..d
    }
}

impl Model {
    pub fn base(w1: Repr, tp: u32) -> Model {
        assemble(
            w1,
            tp,
            Dims {
                hidden: 2048,
                layers: 6,
                dense_layers: 1,
                ratios: &[1, 2, 4],
                heads: 16,
                head_dim: 128,
                q_lora: 768,
                o_lora: 512,
                rope_dim: 64,
                theta: 10_000.0,
                window: 2048,
                streams: 4,
                gate_eps: 1e-6,
                alpha: 2.0,
                sinkhorn: 20,
                dense_inter: 5632,
                experts: 64,
                top_k: 6,
                moe_inter: 1024,
                renorm: false,
                scaling: 2.5,
                swiglu_limit: 7.0,
                vocab: 129_280,
                norm_eps: 1e-5,
            },
        )
    }
}

fn assemble(w1: Repr, tp: u32, d: Dims) -> Model {
    let d = per_rank(d, tp);
    let hidden = d.hidden as u64;
    let mult = d.streams as u64;
    let q_w = d.heads as u64 * d.head_dim as u64;
    let q_lora = d.q_lora as u64;
    let o_lora = d.o_lora as u64;
    let dense_inter = d.dense_inter as u64;
    let moe_inter = d.moe_inter as u64;

    let layers = (0..d.layers)
        .map(|l| {
            let n = |s: &str| format!("layer.{l}.{s}");
            let norm = |s: &str, w: u64| Norm {
                weight: Weight::sym(n(s), [w], w1),
                eps: d.norm_eps,
            };
            let mix = |s: &str| Mix {
                scale: Weight::sym(n(&format!("{s}_scale")), [3], Repr::F32),
                base: Weight::sym(n(&format!("{s}_base")), [2 * mult + mult * mult], Repr::F32),
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
                    q_down: Weight::sym(n("q_down"), [q_lora, hidden], w1),
                    q_norm: norm("q_norm", q_lora),
                    q_up: Weight::sym(n("q_up"), [q_w, q_lora], w1).columns(),
                    kv_down: Weight::sym(n("kv_down"), [q_w, hidden], w1).columns(),
                    kv_norm: Norm {
                        weight: Weight::sym(n("kv_norm"), [q_w], w1).columns(),
                        eps: d.norm_eps,
                    },
                    o_down: Weight::sym(n("o_down"), [o_lora, q_w], w1).rows(),
                    o_up: Weight::sym(n("o_up"), [hidden, o_lora], w1),
                    sink: Weight::sym(n("attn_sink"), [d.heads as u64], w1).columns(),
                    kv: CacheRef::to(format!("kv.{l}")),
                    pool: d
                        .ratios
                        .get(l as usize)
                        .copied()
                        .filter(|r| *r > 0)
                        .map(|ratio| Pool {
                            ratio,
                            entries: CacheRef::to(format!("pool.{l}")),
                        }),
                },
                mlp_mix: mix("mlp_mix"),
                mlp: if l < d.dense_layers {
                    Mlp::Dense {
                        gate_up: Weight::sym(n("gate_up"), [2 * dense_inter, hidden], w1)
                            .packed([dense_inter, dense_inter]),
                        down: Weight::sym(n("down"), [hidden, dense_inter], w1).rows(),
                        inter: d.dense_inter,
                        limit: d.swiglu_limit,
                    }
                } else {
                    Mlp::Routed {
                        router: Weight::sym(n("router"), [d.experts as u64, hidden], w1),
                        bias: Weight::sym(n("router_bias"), [d.experts as u64], w1),
                        gate_up: Weight::sym(
                            n("experts_gate_up"),
                            [d.experts as u64, 2 * moe_inter, hidden],
                            w1,
                        )
                        .bank([moe_inter, moe_inter]),
                        down: Weight::sym(
                            n("experts_down"),
                            [d.experts as u64, hidden, moe_inter],
                            w1,
                        )
                        .rows(),
                        experts: d.experts,
                        top_k: d.top_k,
                        inter: d.moe_inter,
                        limit: d.swiglu_limit,
                        renorm: d.renorm,
                        scaling: d.scaling,
                    }
                },
            }
        })
        .collect();

    Model {
        hidden: d.hidden,
        vocab: d.vocab,
        tp,
        hyper: Hyper {
            streams: d.streams,
            norm_eps: d.norm_eps,
            gate_eps: d.gate_eps,
            alpha: d.alpha,
            sinkhorn: d.sinkhorn,
            head_scale: Weight::sym("hyper.head_scale", [1], Repr::F32),
            head_base: Weight::sym("hyper.head_base", [mult], Repr::F32),
        },
        embed: Weight::sym("embed", [d.vocab as u64, hidden], w1),
        layers,
        final_norm: Norm {
            weight: Weight::sym("final_norm", [hidden], w1),
            eps: d.norm_eps,
        },
    }
}
