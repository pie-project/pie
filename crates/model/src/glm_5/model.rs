use model_dsl::{Dtype, Weight};
use model_loader::contract::ModelContract;

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    /// The MLA reading, stated once for the whole trunk. Every layer scores
    /// the same latent plane the same way — `heads` queries against a
    /// `kv_lora_rank`-wide absorbed output — so the numbers the plan op
    /// states are facts about the model, not about a layer, and the one
    /// schedule built at the top of `forward` is carved for all of them.
    pub heads: u32,
    pub kv_lora_rank: u32,

    pub kv_dtype: Dtype,
    pub embed: Weight,
    pub head: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,
}

pub struct Layer {
    pub attn: Attn,
    pub attn_norm: Weight,
    pub attn_norm_eps: f32,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: Mlp,
}

pub struct Attn {
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub theta: f32,
    pub sm_scale: f32,
    pub q_a_proj: Weight,
    pub q_a_norm: Weight,
    pub q_a_norm_eps: f32,
    pub q_b_proj: Weight,
    pub kv_a_proj: Weight,
    pub kv_a_norm: Weight,
    pub kv_a_norm_eps: f32,
    pub kv_b_proj: Weight,
    pub o_proj: Weight,
    pub indexer: Indexer,
    pub kv: String,
}

pub struct Indexer {
    pub heads: u32,
    pub head_dim: u32,
    pub top_k: u32,
    pub rope_dim: u32,
    pub theta: f32,
    pub q_proj: Weight,
    pub k_proj: Weight,
    pub weights_proj: Weight,
    pub k_norm: Weight,
    pub k_norm_eps: f32,
    pub k_norm_bias: Weight,
    pub keys: String,
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
        shared: Option<Shared>,
        experts: u32,
        top_k: u32,
        inter: u32,
        norm_weights: bool,
        scaling: f32,
    },
}

pub struct Shared {
    pub gate_up: Weight,
    pub down: Weight,
    pub inter: u32,
}

struct MoeDims {
    experts: u32,
    top_k: u32,
    inter: u32,
    shared_inter: u32,
    norm_weights: bool,
    scaling: f32,
}

struct Dims {
    hidden: u32,
    layers: u32,
    dense_layers: u32,
    heads: u32,
    q_lora_rank: u32,
    kv_lora_rank: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    v_head_dim: u32,
    theta: f32,
    index_heads: u32,
    index_head_dim: u32,
    index_top_k: u32,
    dense_inter: u32,
    moe: MoeDims,
    vocab: u32,
    norm_eps: f32,
}

impl Model {
    pub fn a12b(w: Dtype, experts: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            experts,
            kv,
            tp,
            Dims {
                hidden: 4096,
                layers: 46,
                dense_layers: 3,
                heads: 96,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                theta: 10_000.0,
                index_heads: 64,
                index_head_dim: 128,
                index_top_k: 2048,
                dense_inter: 10_944,
                moe: MoeDims {
                    experts: 128,
                    top_k: 8,
                    inter: 1408,
                    shared_inter: 1408,
                    norm_weights: true,
                    scaling: 2.5,
                },
                vocab: 151_552,
                norm_eps: 1e-5,
            },
        )
    }

    fn new(w: Dtype, experts: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        let heads = d.heads / tp;
        let dense_inter = d.dense_inter / tp;
        let moe_inter = d.moe.inter / tp;
        let shared_inter = d.moe.shared_inter / tp;

        let hidden = d.hidden as u64;
        let q_lora = d.q_lora_rank as u64;
        let kv_lora = d.kv_lora_rank as u64;
        let qk_head_dim = (d.qk_nope_head_dim + d.qk_rope_head_dim) as u64;
        let q_b_width = heads as u64 * qk_head_dim;
        let kv_a_width = kv_lora + d.qk_rope_head_dim as u64;
        let kv_b_width = heads as u64 * (d.qk_nope_head_dim + d.v_head_dim) as u64;
        let v_width = heads as u64 * d.v_head_dim as u64;
        let index_width = d.index_heads as u64 * d.index_head_dim as u64;

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str, width: u64| Weight::sym(n(s), [width], w);
                let attn = Attn {
                    qk_nope_head_dim: d.qk_nope_head_dim,
                    qk_rope_head_dim: d.qk_rope_head_dim,
                    v_head_dim: d.v_head_dim,
                    theta: d.theta,
                    sm_scale: (qk_head_dim as f32).sqrt().recip(),
                    q_a_proj: Weight::sym(n("q_a_proj"), [q_lora, hidden], w),
                    q_a_norm: norm("q_a_norm", q_lora),
                    q_a_norm_eps: d.norm_eps,
                    q_b_proj: Weight::sym(n("q_b_proj"), [q_b_width, q_lora], w).columns(),
                    kv_a_proj: Weight::sym(n("kv_a_proj"), [kv_a_width, hidden], w),
                    kv_a_norm: norm("kv_a_norm", kv_lora),
                    kv_a_norm_eps: d.norm_eps,
                    kv_b_proj: Weight::sym(n("kv_b_proj"), [kv_b_width, kv_lora], w).columns(),
                    o_proj: Weight::sym(n("o_proj"), [hidden, v_width], w).rows(),
                    indexer: Indexer {
                        heads: d.index_heads,
                        head_dim: d.index_head_dim,
                        top_k: d.index_top_k,
                        rope_dim: d.qk_rope_head_dim,
                        theta: d.theta,
                        q_proj: Weight::sym(n("index_q_proj"), [index_width, q_lora], w),
                        k_proj: Weight::sym(
                            n("index_k_proj"),
                            [d.index_head_dim as u64, hidden],
                            w,
                        ),
                        weights_proj: Weight::sym(
                            n("index_weights"),
                            [d.index_heads as u64, q_lora],
                            w,
                        ),
                        k_norm: norm("index_k_norm", d.index_head_dim as u64),
                        k_norm_eps: d.norm_eps,
                        k_norm_bias: Weight::sym(
                            n("index_k_norm_bias"),
                            [d.index_head_dim as u64],
                            w,
                        ),
                        keys: format!("index.{l}"),
                    },
                    kv: format!("kv.{l}"),
                };
                let mlp = if l < d.dense_layers {
                    let iw = dense_inter as u64;
                    Mlp::Dense {
                        gate_up: Weight::sym(n("gate_up"), [2 * iw, hidden], w).packed([iw, iw]),
                        down: Weight::sym(n("down"), [hidden, iw], w).rows(),
                        inter: dense_inter,
                    }
                } else {
                    let m = &d.moe;
                    let iw = moe_inter as u64;
                    let sw = shared_inter as u64;
                    Mlp::Routed {
                        router: Weight::sym(n("router"), [m.experts as u64, hidden], w),
                        gate_up: Weight::sym(
                            n("experts_gate_up"),
                            [m.experts as u64, 2 * iw, hidden],
                            experts,
                        )
                        .bank([iw, iw]),
                        down: Weight::sym(
                            n("experts_down"),
                            [m.experts as u64, hidden, iw],
                            experts,
                        )
                        .rows(),
                        shared: (shared_inter > 0).then(|| Shared {
                            gate_up: Weight::sym(n("shared_gate_up"), [2 * sw, hidden], w)
                                .packed([sw, sw]),
                            down: Weight::sym(n("shared_down"), [hidden, sw], w).rows(),
                            inter: shared_inter,
                        }),
                        experts: m.experts,
                        top_k: m.top_k,
                        inter: moe_inter,
                        norm_weights: m.norm_weights,
                        scaling: m.scaling,
                    }
                };
                Layer {
                    attn,
                    attn_norm: norm("attn_norm", hidden),
                    attn_norm_eps: d.norm_eps,
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
            heads,
            kv_lora_rank: d.kv_lora_rank,
            kv_dtype: kv,
            embed: Weight::sym("embed", [d.vocab as u64, hidden], w),
            head: Weight::sym("lm_head", [d.vocab as u64, hidden], w),
            layers,
            final_norm: Weight::sym("final_norm", [hidden], w),
            final_norm_eps: d.norm_eps,
        }
    }
}

impl Model {
    pub fn load(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, crate::contract::ModelError> {
        let tp = self.tp;
        let claim = |w: &Weight| crate::contract::claim(w, tp);
        let mut claims = Vec::new();
        claims.push(claim(&self.embed));
        claims.push(claim(&self.final_norm));
        claims.push(claim(&self.head));
        for layer in &self.layers {
            let attn = &layer.attn;
            let index = &attn.indexer;
            claims.push(claim(&layer.attn_norm));
            claims.push(claim(&layer.mlp_norm));
            claims.push(claim(&attn.q_a_proj));
            claims.push(claim(&attn.q_a_norm));
            claims.push(claim(&attn.q_b_proj));
            claims.push(claim(&attn.kv_a_proj));
            claims.push(claim(&attn.kv_a_norm));
            claims.push(claim(&attn.kv_b_proj));
            claims.push(claim(&attn.o_proj));
            claims.push(claim(&index.q_proj));
            claims.push(claim(&index.k_proj));
            claims.push(claim(&index.weights_proj));
            claims.push(claim(&index.k_norm));
            claims.push(claim(&index.k_norm_bias));
            match &layer.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    claims.push(claim(gate_up));
                    claims.push(claim(down));
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared,
                    ..
                } => {
                    claims.push(claim(router));
                    claims.push(claim(gate_up));
                    claims.push(claim(down));
                    if let Some(shared) = shared {
                        claims.push(claim(&shared.gate_up));
                        claims.push(claim(&shared.down));
                    }
                }
            }
        }
        crate::contract::elaborate(src, claims)
    }
}
