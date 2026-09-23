use model_dsl::{Dtype, Weight};

pub struct Model {
    pub hidden: u32,
    pub vocab: u32,
    pub tp: u32,
    /// `attn_res_block_size` under `AttnRes::Every`.
    pub res_block: u32,

    pub mla_heads: u32,
    pub kv_lora_rank: u32,

    pub adapters: Adapters,

    pub kv: Dtype,
    pub embed: Weight,
    pub head: Weight,
    pub layers: Vec<Layer>,
    pub final_norm: Weight,
    pub final_norm_eps: f32,
    /// Where the residual stream is blended with the closed blocks.
    pub attn_res: AttnRes,
    /// The released model blends the whole block stack once more before the
    /// final norm.
    pub output_res: Option<ResBlend>,
}

/// Kimi's AttnRes: the residual stream is carried as one prefix sum per block
/// plus the closed blocks' sums, and a sublayer reads a softmax mix of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnRes {
    /// The fixture's reading: blend once where a block opens, keep one
    /// running residual.
    AtBlockStart,
    /// The released model: every attention and every MLP input is a blend of
    /// the closed blocks and the current block's prefix sum, which resets
    /// where a block opens (`attn_res_block_size`).
    Every,
}

pub use crate::adapter::Adapters;

pub struct Layer {
    pub res_blend: Option<ResBlend>,
    /// `AttnRes::Every`: the blend before the MLP (`mlp_res_*`).
    pub mlp_res: Option<ResBlend>,
    pub mixer: Mixer,
    pub mixer_norm: Weight,
    pub mixer_norm_eps: f32,
    pub mlp_norm: Weight,
    pub mlp_norm_eps: f32,
    pub mlp: Mlp,
    pub lora_a: Weight,
    pub lora_b: Weight,
}

pub struct ResBlend {
    pub norm: Weight,
    pub norm_eps: f32,
    pub proj: Weight,
}

#[allow(clippy::large_enum_variant)]
pub enum Mixer {
    Mla(Mla),
    Kda(Kda),
}

pub struct Mla {
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub sm_scale: f32,
    pub q_a_proj: Weight,
    pub q_a_norm: Weight,
    pub q_a_norm_eps: f32,
    pub q_b_proj: Weight,
    pub kv_a_proj: Weight,
    pub kv_a_norm: Weight,
    pub kv_a_norm_eps: f32,
    pub kv_b_proj: Weight,
    pub gate: Option<Weight>,
    pub o_proj: Weight,
    pub kv: String,
}

pub struct Kda {
    pub heads: u32,
    pub head_dim: u32,
    pub conv_kernel: u32,
    pub norm_eps: f32,
    /// `gate_lower_bound`: the forget gate's floor (0 leaves it unbounded).
    pub gate_floor: f32,
    pub qkv: Weight,
    pub conv: Weight,
    pub f_a: Weight,
    pub f_b: Weight,
    pub b: Weight,
    pub dt_bias: Weight,
    pub a_log: Weight,
    pub gate: Weight,
    pub o_norm: Weight,
    pub o_norm_eps: f32,
    pub o_proj: Weight,
    pub conv_state: String,
    pub delta_state: String,
}

#[allow(clippy::large_enum_variant)]
pub enum Mlp {
    Dense {
        gate_up: Weight,
        down: Weight,
        inter: u32,
        beta: f32,
        up_cap: Option<f32>,
    },
    Routed {
        router: Weight,
        /// `e_score_correction_bias`: steers the selection, not the weights.
        bias: Option<Weight>,
        gate_up: Weight,
        down: Weight,
        shared: Option<Shared>,
        /// Kimi-K3's LatentMoE: the routed experts work in a narrower latent
        /// the token is projected into and back out of.
        latent: Option<Latent>,
        experts: u32,
        top_k: u32,
        renorm: bool,
        routed_scaling: f32,
        inter: u32,
        beta: f32,
        up_cap: Option<f32>,
    },
}

pub struct Shared {
    pub gate_up: Weight,
    pub down: Weight,
    pub inter: u32,
}

pub struct Latent {
    pub width: u32,
    pub down: Weight,
    pub norm: Option<Weight>,
    pub norm_eps: f32,
    pub up: Weight,
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
    f_rank: u32,
    conv_kernel: u32,
    norm_eps: f32,
}

struct MoeDims {
    experts: u32,
    top_k: u32,
    routed_scaling: f32,
    inter: u32,
    shared_inter: u32,
    /// The routed experts' input width when it is not the hidden width.
    latent: Option<u32>,
    latent_norm: bool,
    bias: bool,
    renorm: bool,
}

struct Dims {
    hidden: u32,
    layers: u32,
    dense_layers: u32,
    full_attn_every: u32,
    res_block: u32,
    attn_res: AttnRes,
    gate_floor: f32,
    mla: MlaDims,
    kda: KdaDims,
    moe: MoeDims,
    dense_inter: u32,
    situ_beta: f32,
    situ_cap: Option<f32>,
    vocab: u32,
    norm_eps: f32,
}

fn closes_a_block(l: u32, every: u32) -> bool {
    every > 0 && (l + 1).is_multiple_of(every)
}

impl Model {
    pub fn k3(w: Dtype, experts: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(
            w,
            experts,
            kv,
            tp,
            Dims {
                hidden: 2048,
                layers: 8,
                dense_layers: 1,
                full_attn_every: 4,
                res_block: 4,
                attn_res: AttnRes::AtBlockStart,
                gate_floor: 0.0,
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
                    f_rank: 128,
                    conv_kernel: 4,
                    norm_eps: 1e-5,
                },
                moe: MoeDims {
                    experts: 64,
                    top_k: 6,
                    routed_scaling: 2.0,
                    inter: 1024,
                    shared_inter: 1024,
                    latent: None,
                    latent_norm: false,
                    bias: false,
                    renorm: false,
                },
                dense_inter: 5632,
                situ_beta: 1.0,
                situ_cap: None,
                vocab: 163_840,
                norm_eps: 1e-5,
            },
        )
    }

    /// `moonshotai/Kimi-K3` as released: 93 layers (one dense, then KDA with
    /// an MLA layer every fourth), 896 latent-MoE experts of which 16 route,
    /// two shared experts, AttnRes blocks of 12.
    pub fn k3_released(w: Dtype, experts: Dtype, kv: Dtype, tp: u32) -> Model {
        Model::new(w, experts, kv, tp, Model::released_dims(93, 896, 12))
    }

    /// The first `layers` layers of Kimi-K3 at full width with the first
    /// `experts` routed experts (top-16 kept), AttnRes blocks of `res_block`:
    /// the miniature `scripts/bench/shrink_checkpoint.py` cuts with
    /// `--layers 0-7 --experts 32 --attn-res-block-size 4`.
    pub fn k3_mini(
        layers: u32,
        experts: u32,
        res_block: u32,
        w: Dtype,
        bank: Dtype,
        kv: Dtype,
        tp: u32,
    ) -> Model {
        Model::new(
            w,
            bank,
            kv,
            tp,
            Model::released_dims(layers, experts, res_block),
        )
    }

    fn released_dims(layers: u32, experts: u32, res_block: u32) -> Dims {
        Dims {
            hidden: 7168,
            layers,
            dense_layers: 1,
            full_attn_every: 4,
            res_block,
            attn_res: AttnRes::Every,
            gate_floor: -5.0,
            mla: MlaDims {
                heads: 96,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                output_gate: true,
            },
            kda: KdaDims {
                heads: 96,
                head_dim: 128,
                f_rank: 128,
                conv_kernel: 4,
                norm_eps: 1e-5,
            },
            moe: MoeDims {
                experts,
                top_k: 16.min(experts),
                routed_scaling: 1.0,
                inter: 3072,
                shared_inter: 2 * 3072,
                latent: Some(3584),
                latent_norm: true,
                bias: true,
                renorm: true,
            },
            dense_inter: 33_792,
            situ_beta: 4.0,
            situ_cap: Some(25.0),
            vocab: 163_840,
            norm_eps: 1e-5,
        }
    }

    fn new(weights: Dtype, experts: Dtype, kv: Dtype, tp: u32, d: Dims) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4 | 8),
            "tp {tp} is not a world this catalog ships"
        );
        let mla_heads = d.mla.heads / tp;
        let kda_heads = d.kda.heads / tp;
        let moe_inter = d.moe.inter / tp;
        let shared_inter = d.moe.shared_inter / tp;
        let dense_inter = d.dense_inter / tp;

        let hidden = d.hidden as u64;
        let moe_in = u64::from(d.moe.latent.unwrap_or(d.hidden));
        let full_at = |l: u32| closes_a_block(l, d.full_attn_every);
        let moe_at = |l: u32| l >= d.dense_layers;
        let blend_at = |l: u32| match d.attn_res {
            AttnRes::AtBlockStart => l > 0 && closes_a_block(l - 1, d.res_block),
            AttnRes::Every => true,
        };

        let a = &d.mla;
        let k = &d.kda;
        let qk_head_dim = (a.qk_nope_head_dim + a.qk_rope_head_dim) as u64;
        let q_b_width = mla_heads as u64 * qk_head_dim;
        let kv_a_width = (a.kv_lora_rank + a.qk_rope_head_dim) as u64;
        let kv_b_width = mla_heads as u64 * (a.qk_nope_head_dim + a.v_head_dim) as u64;
        let v_width = mla_heads as u64 * a.v_head_dim as u64;
        let kda_width = kda_heads as u64 * k.head_dim as u64;

        let layers = (0..d.layers)
            .map(|l| {
                let n = |s: &str| format!("layer.{l}.{s}");
                let norm = |s: &str, width: u64| Weight::sym(n(s), [width], weights);
                let mixer = if full_at(l) {
                    Mixer::Mla(Mla {
                        qk_nope_head_dim: a.qk_nope_head_dim,
                        qk_rope_head_dim: a.qk_rope_head_dim,
                        v_head_dim: a.v_head_dim,
                        sm_scale: (qk_head_dim as f32).sqrt().recip(),
                        q_a_proj: Weight::sym(
                            n("q_a_proj"),
                            [a.q_lora_rank as u64, hidden],
                            weights,
                        ),
                        q_a_norm: norm("q_a_norm", a.q_lora_rank as u64),
                        q_a_norm_eps: d.norm_eps,
                        q_b_proj: Weight::sym(
                            n("q_b_proj"),
                            [q_b_width, a.q_lora_rank as u64],
                            weights,
                        )
                        .columns(),
                        kv_a_proj: Weight::sym(n("kv_a_proj"), [kv_a_width, hidden], weights),
                        kv_a_norm: norm("kv_a_norm", a.kv_lora_rank as u64),
                        kv_a_norm_eps: d.norm_eps,
                        kv_b_proj: Weight::sym(
                            n("kv_b_proj"),
                            [kv_b_width, a.kv_lora_rank as u64],
                            weights,
                        )
                        .columns(),
                        gate: a.output_gate.then(|| {
                            Weight::sym(n("o_gate"), [v_width, hidden], weights).columns()
                        }),
                        o_proj: Weight::sym(n("o_proj"), [hidden, v_width], weights).rows(),
                        kv: format!("kv.{l}"),
                    })
                } else {
                    Mixer::Kda(Kda {
                        heads: kda_heads,
                        head_dim: k.head_dim,
                        conv_kernel: k.conv_kernel,
                        norm_eps: k.norm_eps,
                        gate_floor: d.gate_floor,
                        qkv: Weight::sym(n("kda_qkv"), [3 * kda_width, hidden], weights)
                            .packed([kda_width, kda_width, kda_width]),
                        conv: Weight::sym(
                            n("kda_conv"),
                            [3 * kda_width, k.conv_kernel as u64],
                            weights,
                        )
                        .packed([kda_width, kda_width, kda_width]),
                        f_a: Weight::sym(n("kda_f_a"), [k.f_rank as u64, hidden], weights),
                        f_b: Weight::sym(n("kda_f_b"), [kda_width, k.f_rank as u64], weights)
                            .columns(),
                        b: Weight::sym(n("kda_b"), [kda_heads as u64, hidden], weights).columns(),
                        dt_bias: Weight::sym(
                            n("kda_dt_bias"),
                            [kda_heads as u64, k.head_dim as u64],
                            Dtype::F32,
                        )
                        .columns(),
                        a_log: Weight::sym(n("kda_a_log"), [kda_heads as u64], Dtype::F32)
                            .columns(),
                        gate: Weight::sym(n("kda_gate"), [kda_width, hidden], weights).columns(),
                        o_norm: Weight::sym(n("kda_o_norm"), [k.head_dim as u64], Dtype::F32),
                        o_norm_eps: k.norm_eps,
                        o_proj: Weight::sym(n("kda_o_proj"), [hidden, kda_width], weights).rows(),
                        conv_state: format!("conv.{l}"),
                        delta_state: format!("delta.{l}"),
                    })
                };
                let mlp = if moe_at(l) {
                    let m = &d.moe;
                    let inter = moe_inter as u64;
                    let shared_width = shared_inter as u64;
                    Mlp::Routed {
                        router: Weight::sym(n("router"), [m.experts as u64, hidden], weights),
                        bias: m
                            .bias
                            .then(|| Weight::sym(n("router_bias"), [m.experts as u64], Dtype::F32)),
                        gate_up: Weight::sym(
                            n("experts_gate_up"),
                            [m.experts as u64, 2 * inter, moe_in],
                            experts,
                        )
                        .bank([inter, inter]),
                        down: Weight::sym(
                            n("experts_down"),
                            [m.experts as u64, moe_in, inter],
                            experts,
                        )
                        .rows(),
                        latent: m.latent.map(|width| Latent {
                            width,
                            down: Weight::sym(
                                n("latent_down"),
                                [u64::from(width), hidden],
                                weights,
                            ),
                            norm: m.latent_norm.then(|| {
                                Weight::sym(n("latent_norm"), [u64::from(width)], weights)
                            }),
                            norm_eps: d.norm_eps,
                            up: Weight::sym(n("latent_up"), [hidden, u64::from(width)], weights),
                        }),
                        shared: (shared_inter > 0).then(|| Shared {
                            gate_up: Weight::sym(
                                n("shared_gate_up"),
                                [2 * shared_width, hidden],
                                weights,
                            )
                            .packed([shared_width, shared_width]),
                            down: Weight::sym(n("shared_down"), [hidden, shared_width], weights)
                                .rows(),
                            inter: shared_inter,
                        }),
                        experts: m.experts,
                        top_k: m.top_k,
                        renorm: m.renorm,
                        routed_scaling: m.routed_scaling,
                        inter: moe_inter,
                        beta: d.situ_beta,
                        up_cap: d.situ_cap,
                    }
                } else {
                    let inter = dense_inter as u64;
                    Mlp::Dense {
                        gate_up: Weight::sym(n("gate_up"), [2 * inter, hidden], weights)
                            .packed([inter, inter]),
                        down: Weight::sym(n("down"), [hidden, inter], weights).rows(),
                        inter: dense_inter,
                        beta: d.situ_beta,
                        up_cap: d.situ_cap,
                    }
                };
                let (lora_a, lora_b) = crate::adapter::banks(
                    &format!("layer.{l}"),
                    ADAPTERS,
                    hidden,
                    crate::dense(weights),
                );
                Layer {
                    res_blend: blend_at(l).then(|| ResBlend {
                        norm: norm("res_norm", hidden),
                        norm_eps: d.norm_eps,
                        proj: Weight::sym(n("res_proj"), [1, hidden], weights),
                    }),
                    mlp_res: (d.attn_res == AttnRes::Every).then(|| ResBlend {
                        norm: norm("mlp_res_norm", hidden),
                        norm_eps: d.norm_eps,
                        proj: Weight::sym(n("mlp_res_proj"), [1, hidden], weights),
                    }),
                    mixer,
                    mixer_norm: norm("mixer_norm", hidden),
                    mixer_norm_eps: d.norm_eps,
                    mlp_norm: norm("mlp_norm", hidden),
                    mlp_norm_eps: d.norm_eps,
                    mlp,
                    lora_a,
                    lora_b,
                }
            })
            .collect();

        Model {
            hidden: d.hidden,
            vocab: d.vocab,
            tp,
            res_block: d.res_block,
            mla_heads,
            kv_lora_rank: a.kv_lora_rank,
            adapters: ADAPTERS,
            kv,
            embed: Weight::sym("embed", [d.vocab as u64, hidden], weights),
            head: {
                let banded = tp > 1 && std::env::var_os("PIE_NO_VOCAB_SHARD").is_none();
                let rows = if banded {
                    u64::from(d.vocab / tp)
                } else {
                    u64::from(d.vocab)
                };
                let bank = Weight::sym("lm_head", [rows, hidden], weights);
                if banded { bank.packed([rows]) } else { bank }
            },
            layers,
            final_norm: Weight::sym("final_norm", [hidden], weights),
            final_norm_eps: d.norm_eps,
            attn_res: d.attn_res,
            output_res: (d.attn_res == AttnRes::Every).then(|| ResBlend {
                norm: Weight::sym("output_res_norm", [hidden], weights),
                norm_eps: d.norm_eps,
                proj: Weight::sym("output_res_proj", [1, hidden], weights),
            }),
        }
    }

    /// Which layers open an AttnRes block (the prefix sum is pushed and reset).
    #[must_use]
    pub fn opens_block(&self, layer: u32) -> bool {
        self.res_block > 0 && layer.is_multiple_of(self.res_block)
    }
}

const ADAPTERS: Adapters = Adapters { slots: 8, rank: 16 };

impl Model {}
