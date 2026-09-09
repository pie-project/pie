use model_dsl::{Dtype, Weight};

pub use crate::adapter::Adapters;

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub window: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub norm_eps: f32,

    pub adapters: Adapters,

    pub kv: Dtype,
    pub softcap: f32,
    pub output_multiplier: f32,
    pub embed: Weight,
    pub lm_head: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Reading {
    Sliding = 0,
    Full = 1,
}

pub struct Layer {
    pub reading: Reading,
    pub qkv: Weight,
    pub gate: Weight,
    pub o_proj: Weight,
    pub kv: String,

    pub attn_norm: Weight,
    pub attn_norm_eps: f32,
    pub post_attn_norm: Weight,
    pub post_attn_norm_eps: f32,
    pub pre_ffw_norm: Weight,
    pub pre_ffw_norm_eps: f32,
    pub post_ffw_norm: Weight,
    pub post_ffw_norm_eps: f32,

    pub gate_up: Weight,
    pub inter: u32,
    pub down: Weight,

    pub lora_a: Weight,
    pub lora_b: Weight,
}

struct Dims {
    hidden: u32,
    layers: u32,
    full_every: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    intermediate: u32,
    vocab: u32,
    window: u32,
    theta: f32,
    qk_scale: f32,
    softcap: f32,
    output_multiplier: f32,
    norm_eps: f32,
    post_norm_eps: f32,
}

impl Model {
    pub fn b30(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::b30_dims())
    }

    pub fn b30_mini(layers: u32, w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::b30_dims();
        d.layers = layers;
        Model::new(w, kv, tp, d)
    }

    fn b30_dims() -> Dims {
        Dims {
            hidden: 6656,
            layers: 52,
            full_every: 4,
            q_heads: 32,
            kv_heads: 2,
            head_dim: 128,
            intermediate: 19_968,
            vocab: 202_048,
            window: 2048,
            theta: 500_000.0,
            qk_scale: 3.87,
            softcap: 20.0,
            output_multiplier: 0.196_116_13,
            norm_eps: 1e-5,
            post_norm_eps: 1e-8,
        }
    }

    fn new(w: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2),
            "tp {tp} is not a world this catalog ships (two kv heads divide two ways)"
        );
        let dense = crate::dense(w);
        let proj = match w {
            Dtype::U4g64 => Dtype::U4g64tiled,
            other => other,
        };
        let q_heads = d.q_heads / tp;
        let kv_heads = d.kv_heads / tp;
        let intermediate = d.intermediate / tp;

        let hidden = u64::from(d.hidden);
        let hd = u64::from(d.head_dim);
        let q_w = u64::from(q_heads) * hd;
        let kv_w = u64::from(kv_heads) * hd;
        let iw = u64::from(intermediate);
        let full_at = |l: u32| l % d.full_every == d.full_every - 1;

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str| Weight::sym(n(s), [hidden], dense);
                let (lora_a, lora_b) =
                    crate::adapter::banks(&format!("layer.{l}"), ADAPTERS, hidden, dense);
                Layer {
                    reading: if full_at(l) {
                        Reading::Full
                    } else {
                        Reading::Sliding
                    },
                    qkv: Weight::sym(n("qkv"), [q_w + 2 * kv_w, hidden], proj)
                        .packed([q_w, kv_w, kv_w]),
                    gate: Weight::sym(n("gate"), [q_w, hidden], proj).columns(),
                    o_proj: Weight::sym(n("o_proj"), [hidden, q_w], proj).rows(),
                    kv: format!("kv.{l}"),
                    attn_norm: norm("attn_norm"),
                    attn_norm_eps: d.norm_eps,
                    post_attn_norm: norm("post_attn_norm"),
                    post_attn_norm_eps: d.post_norm_eps,
                    pre_ffw_norm: norm("pre_ffw_norm"),
                    pre_ffw_norm_eps: d.norm_eps,
                    post_ffw_norm: norm("post_ffw_norm"),
                    post_ffw_norm_eps: d.post_norm_eps,
                    gate_up: Weight::sym(n("gate_up"), [2 * iw, hidden], proj).packed([iw, iw]),
                    inter: intermediate,
                    down: Weight::sym(n("down"), [hidden, iw], proj).rows(),
                    lora_a,
                    lora_b,
                }
            })
            .collect();

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            q_heads,
            kv_heads,
            head_dim: d.head_dim,
            window: d.window,
            theta: d.theta,
            sm_scale: d.qk_scale / (d.head_dim as f32).sqrt(),
            norm_eps: d.norm_eps,
            adapters: ADAPTERS,
            kv,
            softcap: d.softcap,
            output_multiplier: d.output_multiplier,
            embed: Weight::sym("embed", [u64::from(d.vocab), hidden], w),
            lm_head: {
                let banded = tp > 1 && std::env::var_os("PIE_NO_VOCAB_SHARD").is_none();
                let rows = if banded {
                    u64::from(d.vocab / tp)
                } else {
                    u64::from(d.vocab)
                };
                let bank = Weight::sym("lm_head", [rows, hidden], proj);
                if banded { bank.packed([rows]) } else { bank }
            },
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            final_norm_eps: d.norm_eps,
        }
    }
}

const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };
