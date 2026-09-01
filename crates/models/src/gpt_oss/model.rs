use checkpoint::contract::ModelContract;
use model_dsl::{Dtype, Weight};

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

    /// The adapter banks this family seats (palo design §8). Per layer, and
    /// the same two numbers at every one of them: the correction is a
    /// per-lane axis, not a per-layer one.
    pub adapters: Adapters,

    pub kv: Dtype,
    pub embed: Weight,
    pub head: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,
}

pub use crate::adapter::Adapters;

pub struct Layer {
    pub attn: Attn,
    pub attn_norm: Weight,
    pub attn_norm_eps: f32,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: Moe,
    /// This layer's adapter bank, `[slots, rank, hidden]` and
    /// `[slots, hidden, rank]` — the down and up planes of one correction site
    /// (palo design §8, campaign A-6).
    ///
    /// **THE SITE IS THE ATTENTION SUBLAYER, AND IT IS THE SITE BECAUSE OF THE
    /// COLLECTIVE.** Both ends are REPLICATED values: the input is this
    /// layer's normed residual and the output is the attention's result AFTER
    /// `all_reduce` and after `o_bias`. A correction stated one statement
    /// earlier — on `o_proj`'s own output, which is what a checkpoint's
    /// `o_proj` LoRA names — reads a rows-cut partial product and lands before
    /// the reduce, so every rank would contribute the whole `ΔW·x` and the sum
    /// would carry it `tp` times. `MoeBiasSum` states the identical argument
    /// about the identical hazard one sublayer down, and takes the identical
    /// way out: say the additive term once, after the reduce, where it lands
    /// exactly once.
    ///
    /// AND AFTER THE BIAS, not between the reduce and it. `o_bias` is part of
    /// what the base projection answers, so a correction that landed first
    /// would be correcting half a site — the same reason the bias itself is
    /// added once and on the replicated value.
    pub lora_a: Weight,
    pub lora_b: Weight,
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

        // Everything this text declares that is NOT a matmul bank — the two
        // norms, every attention and expert bias, and the learned sinks. See
        // `crate::dense`: no checkpoint quantizes a layernorm, and MLX's own
        // rule is that a group of sixty-four codes needs sixty-four columns to
        // group, which a `[hidden]` norm or a `[experts]` router bias does not
        // have. This text had stamped `weights` on all of them, which was
        // right while `weights` was always bf16 and is a `32 % 64 != 0` panic
        // at `router_bias` the moment it is not.
        let dense = crate::dense(weights);
        // **THE ROUTER GATE IS QUANTIZED ONE WIDTH COARSER, BY THE FAMILY'S
        // OWN RULE.** `mlx_lm/models/gpt_oss.py` carries
        //
        // ```python
        // @property
        // def quant_predicate(self):
        //     def predicate(path, _):
        //         if path.endswith("router"):
        //             return {"group_size": 64, "bits": 8}
        //         return True
        // ```
        //
        // so every MLX conversion of this family publishes its twenty-four
        // router gates at eight bits whatever the rest of the stack is at, and
        // `mlx-community/gpt-oss-20b-MXFP4-Q4`'s `config.json` lists all
        // twenty-four under `bits: 8` beside ninety-eight affine-U4 entries.
        // The rule is the family's and not the SKU's — `qwen3_5.py` and
        // `gemma4_text.py` carry the same predicate for their own gates — so
        // it is stated here, where the family is, rather than as a fifth
        // parameter every caller would have to repeat.
        //
        // A bf16 stack's router stays bf16: the raise is from four bits to
        // eight, not to eight from anywhere.
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
        }
    }
}

/// What every SKU of this family seats.
///
/// Not a `Dims` field, because it is not a fact about the checkpoint the way
/// `hidden` and `layers` are — no pretrained artifact states it. It is the
/// DEPLOYMENT's ceiling written where a shape has to be written, and a
/// deployment that wants a different one changes this line and re-traces,
/// which is exactly the "load-time recompile, never a runtime extension"
/// design §9 asks for.
///
/// Eight slots of rank sixteen costs gpt-oss 1.41 MiB a layer — two planes of
/// `8 x 16 x 2880` in the compute element — so 33.8 MiB over b20's
/// twenty-four and 50.6 MiB over b120's thirty-six, against 12.8 GiB and
/// 60.8 GiB of banks.
const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

impl Model {
    pub fn load(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, checkpoint_dsl::Error> {
        let mut b = checkpoint_dsl::Builder::new(src, self.tp);
        b.read_own(&self.embed)?;
        b.read_own(&self.final_norm)?;
        b.read_own(&self.head)?;

        for layer in &self.layers {
            let attn = &layer.attn;
            let mlp = &layer.mlp;

            b.read_own(&layer.attn_norm)?;
            b.read_own(&layer.mlp_norm)?;
            b.read_own(&attn.q_proj)?;
            b.read_own(&attn.q_bias)?;
            b.read_own(&attn.k_proj)?;
            b.read_own(&attn.k_bias)?;
            b.read_own(&attn.v_proj)?;
            b.read_own(&attn.v_bias)?;
            b.read_own(&attn.o_proj)?;
            b.read_own(&attn.o_bias)?;
            b.read_own(&attn.sinks)?;
            b.read_own(&mlp.router)?;
            b.read_own(&mlp.router_bias)?;
            b.read_own(&mlp.gate_up)?;
            b.read_own(&mlp.gate_up_bias)?;
            b.read_own(&mlp.down)?;
            b.read_own(&mlp.down_bias)?;
        }

        Ok(b.build())
    }
}
