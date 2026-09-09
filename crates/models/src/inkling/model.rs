use model_dsl::{Dtype, Weight};

pub use crate::adapter::Adapters;

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub head_rows: u32,
    pub tp: u32,

    pub heads: u32,
    pub head_dim: u32,
    pub d_rel: u32,
    pub window: u32,
    pub conv_width: u32,
    pub sm_scale: f32,
    pub norm_eps: f32,
    pub head_scale: f32,
    pub log_scaling: (u32, f32),

    pub adapters: Adapters,

    pub kv: Dtype,
    pub embed: Weight,
    pub embed_norm: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub unembed: Weight,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Reading {
    Local = 0,
    Global = 1,
}

pub struct Layer {
    pub reading: Reading,
    pub kv_heads: u32,
    pub extent: u32,

    pub attn_norm: Weight,
    pub q_proj: Weight,
    pub k_proj: Weight,
    pub v_proj: Weight,
    pub r_proj: Weight,
    pub o_proj: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub rel_proj: Weight,
    pub k_conv: Weight,
    pub v_conv: Weight,
    pub attn_conv: Weight,
    pub mlp_conv: Weight,
    pub kv: String,
    pub k_state: String,
    pub v_state: String,
    pub attn_state: String,
    pub mlp_state: String,

    pub mlp_norm: Weight,
    pub mlp: Mlp,

    pub lora_a: Weight,
    pub lora_b: Weight,
}

#[allow(clippy::large_enum_variant)]
pub enum Mlp {
    Dense {
        gate_up: Weight,
        inter: u32,
        down: Weight,
        scale: Weight,
    },
    Routed {
        router: Weight,
        bias: Weight,
        scale: Weight,
        gate_up: Weight,
        down: Weight,
        experts: u32,
        top_k: u32,
        sink: u32,
        inter: u32,
        scaling: f32,
    },
}

struct Dims {
    hidden: u32,
    layers: u32,
    vocab: u32,
    head_rows: u32,
    heads: u32,
    head_dim: u32,
    local_kv_heads: u32,
    global_kv_heads: u32,
    d_rel: u32,
    window: u32,
    global_extent: u32,
    conv_width: u32,
    global_every: u32,
    dense_layers: u32,
    dense_inter: u32,
    experts: u32,
    top_k: u32,
    sink: u32,
    moe_inter: u32,
    route_scale: f32,
    mup: f32,
    norm_eps: f32,
    log_floor: u32,
    log_alpha: f32,
}

impl Model {
    pub fn full(w: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, kv, tp, Model::dims())
    }

    pub fn mini(layers: u32, experts: u32, w: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut d = Model::dims();
        d.layers = layers;
        d.experts = experts;
        d.top_k = d.top_k.min(experts);
        Model::new(w, kv, tp, d)
    }

    fn dims() -> Dims {
        Dims {
            hidden: 6144,
            layers: 66,
            vocab: 201_024,
            head_rows: 200_058,
            heads: 64,
            head_dim: 128,
            local_kv_heads: 16,
            global_kv_heads: 8,
            d_rel: 16,
            window: 512,
            global_extent: 1024,
            conv_width: 4,
            global_every: 6,
            dense_layers: 2,
            dense_inter: 24_576,
            experts: 256,
            top_k: 6,
            sink: 2,
            moe_inter: 3072,
            route_scale: 8.0,
            mup: 24.0,
            norm_eps: 1e-6,
            log_floor: 128_000,
            log_alpha: 0.1,
        }
    }

    fn new(w: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        let dense = crate::dense(w);
        let proj = match w {
            Dtype::U4g64 => Dtype::U4g64tiled,
            other => other,
        };
        let heads = d.heads / tp;
        let hidden = u64::from(d.hidden);
        let hd = u64::from(d.head_dim);
        let q_w = u64::from(heads) * hd;
        let r_w = u64::from(heads) * u64::from(d.d_rel);
        let kw = u64::from(d.conv_width);

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let vec = |s: &str, len: u64| Weight::sym(n(s), [len], dense);
                let (lora_a, lora_b) =
                    crate::adapter::banks(&format!("layer.{l}"), ADAPTERS, hidden, dense);
                let global = (l + 1) % d.global_every == 0;
                let (reading, kv_heads, extent) = if global {
                    (Reading::Global, d.global_kv_heads / tp, d.global_extent)
                } else {
                    (Reading::Local, d.local_kv_heads / tp, d.window)
                };
                let kv_w = u64::from(kv_heads) * hd;
                let conv =
                    |s: &str, channels: u64| Weight::sym(n(s), [channels, kw], dense).columns();
                let mlp = if l < d.dense_layers {
                    let iw = u64::from(d.dense_inter / tp);
                    Mlp::Dense {
                        gate_up: Weight::sym(n("gate_up"), [2 * iw, hidden], proj).packed([iw, iw]),
                        inter: d.dense_inter / tp,
                        down: Weight::sym(n("down"), [hidden, iw], proj).rows(),
                        scale: vec("mlp_scale", 1),
                    }
                } else {
                    let bank = u64::from(d.experts + d.sink);
                    let mi = u64::from(d.moe_inter / tp);
                    Mlp::Routed {
                        router: Weight::sym(n("router"), [bank, hidden], proj),
                        bias: Weight::sym(n("router_bias"), [u64::from(d.experts)], Dtype::F32),
                        scale: Weight::sym(n("router_scale"), [1], Dtype::F32),
                        gate_up: Weight::sym(n("experts_gate_up"), [bank, 2 * mi, hidden], w)
                            .bank([mi, mi]),
                        down: Weight::sym(n("experts_down"), [bank, hidden, mi], w).rows(),
                        experts: d.experts,
                        top_k: d.top_k,
                        sink: d.sink,
                        inter: d.moe_inter / tp,
                        scaling: d.route_scale,
                    }
                };
                Layer {
                    reading,
                    kv_heads,
                    extent,
                    attn_norm: vec("attn_norm", hidden),
                    q_proj: Weight::sym(n("q_proj"), [q_w, hidden], proj).columns(),
                    k_proj: Weight::sym(n("k_proj"), [kv_w, hidden], proj).columns(),
                    v_proj: Weight::sym(n("v_proj"), [kv_w, hidden], proj).columns(),
                    r_proj: Weight::sym(n("r_proj"), [r_w, hidden], proj).columns(),
                    o_proj: Weight::sym(n("o_proj"), [hidden, q_w], proj).rows(),
                    q_norm: vec("q_norm", hd),
                    k_norm: vec("k_norm", hd),
                    rel_proj: Weight::sym(
                        n("rel_proj"),
                        [u64::from(d.d_rel), u64::from(extent)],
                        dense,
                    ),
                    k_conv: conv("k_conv", kv_w),
                    v_conv: conv("v_conv", kv_w),
                    attn_conv: Weight::sym(n("attn_conv"), [hidden, kw], dense),
                    mlp_conv: Weight::sym(n("mlp_conv"), [hidden, kw], dense),
                    kv: format!("kv.{l}"),
                    k_state: format!("conv.{l}.k"),
                    v_state: format!("conv.{l}.v"),
                    attn_state: format!("conv.{l}.attn"),
                    mlp_state: format!("conv.{l}.mlp"),
                    mlp_norm: vec("mlp_norm", hidden),
                    mlp,
                    lora_a,
                    lora_b,
                }
            })
            .collect();

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            head_rows: d.head_rows,
            tp,
            heads,
            head_dim: d.head_dim,
            d_rel: d.d_rel,
            window: d.window,
            conv_width: d.conv_width,
            sm_scale: 1.0 / d.head_dim as f32,
            norm_eps: d.norm_eps,
            head_scale: 1.0 / d.mup,
            log_scaling: (d.log_floor, d.log_alpha),
            adapters: ADAPTERS,
            kv,
            embed: Weight::sym("embed", [u64::from(d.vocab), hidden], w),
            embed_norm: Weight::sym("embed_norm", [hidden], dense),
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            unembed: Weight::sym("unembed", [u64::from(d.head_rows), hidden], proj),
        }
    }
}

const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };
