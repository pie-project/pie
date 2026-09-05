use model_dsl::{Dtype, Weight};

use crate::drafter::dflash::{self, DFlash};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,

    /// head counts and width are model-wide; `window` is the sliding-window width.
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub window: u32,

    /// Adapter bank shape (slots, rank); same at every layer.
    pub adapters: Adapters,

    pub kv: Dtype,
    pub embed: Weight,
    pub head: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,

    /// z-lab's block drafter (`gpt-oss-20b-DFlash`), when an overlay carries
    /// one — the same text every family carries it as (`crate::drafter::dflash`).
    pub dflash: Option<DFlash>,
}

pub use crate::adapter::Adapters;

pub struct Layer {
    pub attn: Attn,
    pub attn_norm: Weight,
    pub attn_norm_eps: f32,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: Moe,
    /// Adapter bank for the attention sublayer: `[slots, rank, hidden]` down,
    /// `[slots, hidden, rank]` up. Applied after `all_reduce` and after
    /// `o_bias`, on the replicated output — earlier would be summed `tp`
    /// times or correct only half the site.
    pub lora_a: Weight,
    pub lora_b: Weight,
}

/// Which attention reading a layer takes; indexes into `forward`'s per-class schedule pair.
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
    /// The 20B with z-lab's block drafter overlaid (`gpt-oss-20b-DFlash`).
    pub fn b20_dflash(w: Dtype, experts: Dtype, kv: Dtype, tp: u32) -> Model {
        let mut m = Model::b20(w, experts, kv, tp);
        let dense = crate::dense(w);
        m.dflash = Some(DFlash::declare(
            &dflash::GPTOSS_20B_DFLASH,
            "aux",
            &dflash::Trunk {
                hidden: u64::from(m.hidden),
                vocab: u64::from(m.vocab),
                norm_eps: m.final_norm_eps,
                weights: w,
                dense,
                tp,
            },
        ));
        m
    }

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

        // Dtype for norms, biases and sinks: never quantized like matmul banks.
        let dense = crate::dense(weights);
        // Router gate is quantized one width coarser than the rest of the stack
        // (U8g64 instead of U4g64); bf16 stacks keep the router bf16.
        let router = match weights {
            Dtype::U4g64 => Dtype::U8g64,
            other => other,
        };
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
                let norm = |s: &str, cols: u64| Weight::sym(n(s), [cols], dense);
                let (lora_a, lora_b) = crate::adapter::banks(
                    &format!("layer.{l}"),
                    ADAPTERS,
                    hidden,
                    dense,
                );
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
                        q_bias: Weight::sym(n("q_bias"), [q_w], dense).columns(),
                        k_proj: Weight::sym(n("k_proj"), [kv_w, hidden], weights).columns(),
                        k_bias: Weight::sym(n("k_bias"), [kv_w], dense).columns(),
                        v_proj: Weight::sym(n("v_proj"), [kv_w, hidden], weights).columns(),
                        v_bias: Weight::sym(n("v_bias"), [kv_w], dense).columns(),
                        o_proj: Weight::sym(n("o_proj"), [hidden, q_w], weights).rows(),
                        o_bias: Weight::sym(n("o_bias"), [hidden], dense),
                        sinks: Weight::sym(n("attn_sinks"), [q_heads as u64], dense).columns(),
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
                        router: Weight::sym(n("router"), [n_experts, hidden], router),
                        router_bias: Weight::sym(n("router_bias"), [n_experts], dense),
                        gate_up: Weight::sym(
                            n("expert_gate_up_bank"),
                            [n_experts, 2 * inter_w, hidden],
                            experts,
                        )
                        .bank([inter_w, inter_w]),
                        gate_up_bias: Weight::sym(
                            n("expert_gate_up_bias"),
                            [n_experts, 2 * inter_w],
                            dense,
                        )
                        .bank([inter_w, inter_w]),
                        down: Weight::sym(
                            n("expert_down_bank"),
                            [n_experts, hidden, inter_w],
                            experts,
                        )
                        .rows(),
                        down_bias: Weight::sym(n("expert_down_bias"), [n_experts, hidden], dense),
                    },
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
            adapters: ADAPTERS,
            head_dim: d.head_dim,
            window: d.window,
            kv,
            embed: Weight::sym("embed", [d.vocab as u64, hidden], weights),
            head: Weight::sym("lm_head", [d.vocab as u64, hidden], weights),
            layers,
            final_norm: Weight::sym("final_norm", [hidden], dense),
            final_norm_eps: d.norm_eps,
            dflash: None,
        }
    }
}

/// Adapter capacity for this family. A deployment choice, not a checkpoint
/// fact; changing it requires a re-trace.
const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

impl Model {
 }
