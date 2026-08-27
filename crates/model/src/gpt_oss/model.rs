use model_dsl::{Dtype, Weight};
use model_loader::contract::ModelContract;

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    /// The reading every attention schedule is carved for. gpt-oss reads one
    /// page-id space two ways, windowed and full, and the two differ only in
    /// the window: the head counts and the head width are one model-wide
    /// fact, and `window` is the width the windowed reading states.
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub window: u32,

    pub kv: Dtype,
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
    pub mlp: Moe,
}

/// Which of the two readings a layer takes over the one page-id space. The
/// discriminant is the index into `forward`'s per-class pair of schedules,
/// whose two lines differ by exactly the window this names.
#[derive(Clone, Copy)]
pub enum Reading {
    Windowed = 0,
    Full = 1,
}

pub struct Attn {
    pub reading: Reading,
    pub sm_scale: f32,
    pub theta: f32,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub attention_factor: f32,
    pub original_max_position: u32,
    pub q_proj: Weight,
    pub q_bias: Weight,
    pub k_proj: Weight,
    pub k_bias: Weight,
    pub v_proj: Weight,
    pub v_bias: Weight,
    pub o_proj: Weight,
    pub o_bias: Weight,
    pub sinks: Weight,
    pub kv: String,
}

pub struct Moe {
    pub experts: u32,
    pub top_k: u32,
    pub inter: u32,
    pub swiglu_limit: f32,
    pub swiglu_alpha: f32,
    pub router: Weight,
    pub router_bias: Weight,
    pub gate_up: Weight,
    pub gate_up_bias: Weight,
    pub down: Weight,
    pub down_bias: Weight,
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

impl Model {
    pub fn b20(w: Dtype, experts: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            experts,
            kv,
            tp,
            Dims {
                hidden: 2880,
                layers: 24,
                q_heads: 64,
                kv_heads: 8,
                head_dim: 64,
                theta: 150_000.0,
                yarn_factor: 32.0,
                yarn_beta_fast: 32.0,
                yarn_beta_slow: 1.0,
                yarn_attention_factor: 1.346_573_6,
                yarn_original_max_position: 4096,
                window: 128,
                experts: 32,
                top_k: 4,
                inter: 2880,
                swiglu_limit: 7.0,
                swiglu_alpha: 1.702,
                vocab: 201_088,
                norm_eps: 1e-5,
            },
        )
    }

    pub fn b120(w: Dtype, experts: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            experts,
            kv,
            tp,
            Dims {
                hidden: 2880,
                layers: 36,
                q_heads: 64,
                kv_heads: 8,
                head_dim: 64,
                theta: 150_000.0,
                yarn_factor: 32.0,
                yarn_beta_fast: 32.0,
                yarn_beta_slow: 1.0,
                yarn_attention_factor: 1.346_573_6,
                yarn_original_max_position: 4096,
                window: 128,
                experts: 128,
                top_k: 4,
                inter: 2880,
                swiglu_limit: 7.0,
                swiglu_alpha: 1.702,
                vocab: 201_088,
                norm_eps: 1e-5,
            },
        )
    }

    fn new(weights: Dtype, experts: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        let q_heads = d.q_heads / tp;
        let kv_heads = d.kv_heads / tp;
        let inter = d.inter / tp;

        let hidden = d.hidden as u64;
        let hd = d.head_dim as u64;
        let q_w = q_heads as u64 * hd;
        let kv_w = kv_heads as u64 * hd;
        let n_experts = d.experts as u64;
        let inter_w = inter as u64;
        let sliding_window_on = |l: u32| l.is_multiple_of(2);

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str, cols: u64| Weight::sym(n(s), [cols], weights);
                Layer {
                    attn: Attn {
                        reading: if sliding_window_on(l) {
                            Reading::Windowed
                        } else {
                            Reading::Full
                        },
                        sm_scale: (d.head_dim as f32).sqrt().recip(),
                        theta: d.theta,
                        factor: d.yarn_factor,
                        beta_fast: d.yarn_beta_fast,
                        beta_slow: d.yarn_beta_slow,
                        attention_factor: d.yarn_attention_factor,
                        original_max_position: d.yarn_original_max_position,
                        q_proj: Weight::sym(n("q_proj"), [q_w, hidden], weights).columns(),
                        q_bias: Weight::sym(n("q_bias"), [q_w], weights).columns(),
                        k_proj: Weight::sym(n("k_proj"), [kv_w, hidden], weights).columns(),
                        k_bias: Weight::sym(n("k_bias"), [kv_w], weights).columns(),
                        v_proj: Weight::sym(n("v_proj"), [kv_w, hidden], weights).columns(),
                        v_bias: Weight::sym(n("v_bias"), [kv_w], weights).columns(),
                        o_proj: Weight::sym(n("o_proj"), [hidden, q_w], weights).rows(),
                        o_bias: Weight::sym(n("o_bias"), [hidden], weights),
                        sinks: Weight::sym(n("attn_sinks"), [q_heads as u64], weights).columns(),
                        kv: format!("kv.{l}"),
                    },
                    attn_norm: norm("attn_norm", hidden),
                    attn_norm_eps: d.norm_eps,
                    mlp_norm: norm("mlp_norm", hidden),
                    mlp_norm_eps: d.norm_eps,
                    mlp: Moe {
                        experts: d.experts,
                        top_k: d.top_k,
                        inter,
                        swiglu_limit: d.swiglu_limit,
                        swiglu_alpha: d.swiglu_alpha,
                        router: Weight::sym(n("router"), [n_experts, hidden], weights),
                        router_bias: Weight::sym(n("router_bias"), [n_experts], weights),
                        gate_up: Weight::sym(
                            n("expert_gate_up_bank"),
                            [n_experts, 2 * inter_w, hidden],
                            experts,
                        )
                        .bank([inter_w, inter_w]),
                        gate_up_bias: Weight::sym(
                            n("expert_gate_up_bias"),
                            [n_experts, 2 * inter_w],
                            weights,
                        )
                        .bank([inter_w, inter_w]),
                        down: Weight::sym(
                            n("expert_down_bank"),
                            [n_experts, hidden, inter_w],
                            experts,
                        )
                        .rows(),
                        down_bias: Weight::sym(n("expert_down_bias"), [n_experts, hidden], weights),
                    },
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
            kv,
            embed: Weight::sym("embed", [d.vocab as u64, hidden], weights),
            head: Weight::sym("lm_head", [d.vocab as u64, hidden], weights),
            layers,
            final_norm: Weight::sym("final_norm", [hidden], weights),
            final_norm_eps: d.norm_eps,
        }
    }
}

impl Model {
    pub fn load(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, crate::contract::ModelError> {
        let mut claims = vec![
            crate::contract::claim(&self.embed, self.tp),
            crate::contract::claim(&self.final_norm, self.tp),
            crate::contract::claim(&self.head, self.tp),
        ];

        for layer in &self.layers {
            let attn = &layer.attn;
            let mlp = &layer.mlp;

            claims.push(crate::contract::claim(&layer.attn_norm, self.tp));
            claims.push(crate::contract::claim(&layer.mlp_norm, self.tp));
            claims.push(crate::contract::claim(&attn.q_proj, self.tp));
            claims.push(crate::contract::claim(&attn.q_bias, self.tp));
            claims.push(crate::contract::claim(&attn.k_proj, self.tp));
            claims.push(crate::contract::claim(&attn.k_bias, self.tp));
            claims.push(crate::contract::claim(&attn.v_proj, self.tp));
            claims.push(crate::contract::claim(&attn.v_bias, self.tp));
            claims.push(crate::contract::claim(&attn.o_proj, self.tp));
            claims.push(crate::contract::claim(&attn.o_bias, self.tp));
            claims.push(crate::contract::claim(&attn.sinks, self.tp));
            claims.push(crate::contract::claim(&mlp.router, self.tp));
            claims.push(crate::contract::claim(&mlp.router_bias, self.tp));
            claims.push(crate::contract::claim(&mlp.gate_up, self.tp));
            claims.push(crate::contract::claim(&mlp.gate_up_bias, self.tp));
            claims.push(crate::contract::claim(&mlp.down, self.tp));
            claims.push(crate::contract::claim(&mlp.down_bias, self.tp));
        }

        crate::contract::elaborate(src, claims)
    }
}
