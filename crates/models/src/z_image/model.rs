use poem::{Dtype, Weight};

pub const CHANNELS: u32 = 16;
pub const PATCH: u32 = 2;
pub const PATCH_FEATURES: u32 = CHANNELS * PATCH * PATCH;
pub const SPATIAL_COMPRESSION: u32 = 8;

pub const ADALN_DIM: u32 = 256;
pub const T_MID: u32 = 1024;
pub const T_FREQ_DIM: u32 = 256;
pub const T_MAX_PERIOD: f32 = 10_000.0;
pub const T_FLIP_SIN_COS: bool = true;
pub const T_SCALE: f32 = 1000.0;
pub const T_FLIP: f32 = 1000.0;
pub const TRAIN_STEPS: u32 = 1000;

pub const SEQ_MULTIPLE: u32 = 32;

pub const NORM_EPS: f32 = 1e-5;
pub const FINAL_LN_EPS: f32 = 1e-6;
pub const ROPE_AXES: u8 = 3;
pub const ROPE_THETA: f32 = 256.0;
pub const MOD_SLICES: u32 = 4;

pub const TE_HIDDEN: u32 = 2560;
pub const TE_VOCAB: u32 = 151_936;
pub const TE_Q_HEADS: u32 = 32;
pub const TE_KV_HEADS: u32 = 8;
pub const TE_HEAD_DIM: u32 = 128;
pub const TE_INTER: u32 = 9728;
pub const TE_THETA: f32 = 1_000_000.0;
pub const TE_EPS: f32 = 1e-6;
pub const TE_DEPTH: u32 = 36;
pub const TE_LAYERS: u32 = TE_DEPTH - 1;
pub const TE_MAX_TOKENS: u32 = 512;

pub mod port {
    pub const PAD_IMAGE: u8 = 0;
    pub const LATENTS: u8 = 1;
    pub const CONTEXT_REFINED: u8 = 2;
    pub const PAD_CAPTION: u8 = 0;
    pub const CAPTION: u8 = 0;
    pub const TIMESTEP: u8 = 0;
    pub const POSITIONS: u8 = 0;
    pub const VOXELS: u8 = 0;
    pub const PIXEL_VOXELS: u8 = 1;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub dim: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub inter: u32,
    pub joint_layers: u32,
    pub refiner_layers: u32,
    pub cap_width: u32,
    pub rope_dims: [u32; 4],
}

impl Dims {
    #[must_use]
    pub const fn turbo() -> Dims {
        Dims {
            dim: 3840,
            heads: 30,
            head_dim: 128,
            inter: 10_240,
            joint_layers: 30,
            refiner_layers: 2,
            cap_width: TE_HIDDEN,
            rope_dims: [32, 48, 48, 0],
        }
    }

    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            dim: 256,
            heads: 4,
            head_dim: 64,
            inter: 682,
            joint_layers: 2,
            refiner_layers: 2,
            cap_width: 64,
            rope_dims: [16, 24, 24, 0],
        }
    }

    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }
}

pub struct Linear {
    pub w: Weight,
    pub bias: Weight,
}

impl Linear {
    pub(super) fn at(name: &str, out: u32, in_: u32, banks: Dtype) -> Linear {
        Linear {
            w: Weight::sym(name, [u64::from(out), u64::from(in_)], banks),
            bias: Weight::sym(
                format!("{name}.bias"),
                [u64::from(out)],
                crate::dense(banks),
            ),
        }
    }
}

pub struct Attn {
    pub qkv: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub out: Weight,
}

pub struct Swiglu {
    pub gate_up: Weight,
    pub down: Weight,
}

pub struct Block {
    pub ada: Option<Linear>,
    pub attn_norm1: Weight,
    pub attn_norm2: Weight,
    pub ffn_norm1: Weight,
    pub ffn_norm2: Weight,
    pub attn: Attn,
    pub mlp: Swiglu,
}

impl Block {
    fn at(prefix: &str, d: &Dims, modulated: bool, banks: Dtype) -> Block {
        let dense = crate::dense(banks);
        let dim = u64::from(d.dim);
        let hd = u64::from(d.head_dim);
        let inter = u64::from(d.inter);
        let n = |s: &str| format!("{prefix}.{s}");
        let norm = |s: &str, width: u64| Weight::sym(n(s), [width], dense);
        Block {
            ada: modulated.then(|| Linear::at(&n("ada"), MOD_SLICES * d.dim, ADALN_DIM, banks)),
            attn_norm1: norm("attn_norm1", dim),
            attn_norm2: norm("attn_norm2", dim),
            ffn_norm1: norm("ffn_norm1", dim),
            ffn_norm2: norm("ffn_norm2", dim),
            attn: Attn {
                qkv: Weight::sym(n("qkv"), [3 * dim, dim], banks).packed([dim, dim, dim]),
                q_norm: norm("q_norm", hd),
                k_norm: norm("k_norm", hd),
                out: Weight::sym(n("out"), [dim, dim], banks),
            },
            mlp: Swiglu {
                gate_up: Weight::sym(n("gate_up"), [2 * inter, dim], banks).packed([inter, inter]),
                down: Weight::sym(n("down"), [dim, inter], banks),
            },
        }
    }
}

pub struct Dit {
    pub x_embed: Linear,
    pub x_pad_mod: Weight,
    pub cap_norm: Weight,
    pub cap_embed: Linear,
    pub cap_pad_mod: Weight,
    pub t_mlp0: Linear,
    pub t_mlp1: Linear,
    pub t_flip: Weight,
    pub noise_refiner: Vec<Block>,
    pub context_refiner: Vec<Block>,
    pub layers: Vec<Block>,
    pub final_ada: Linear,
    pub final_linear: Linear,
}

pub struct TeLayer {
    pub attn_norm: Weight,
    pub q: Weight,
    pub k: Weight,
    pub v: Weight,
    pub o: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub mlp_norm: Weight,
    pub gate_up: Weight,
    pub down: Weight,
    pub kv: String,
}

pub struct TextEncoder {
    pub hidden: u32,
    pub vocab: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub inter: u32,
    pub theta: f32,
    pub eps: f32,
    pub sm_scale: f32,
    pub embed: Weight,
    pub layers: Vec<TeLayer>,
}

impl TextEncoder {
    fn qwen3_4b(banks: Dtype) -> TextEncoder {
        let dense = crate::dense(banks);
        let hidden = u64::from(TE_HIDDEN);
        let hd = u64::from(TE_HEAD_DIM);
        let inter = u64::from(TE_INTER);
        let layers = (0..TE_LAYERS)
            .map(|l| {
                let n = |s: &str| format!("te.layer.{l}.{s}");
                TeLayer {
                    attn_norm: Weight::sym(n("attn_norm"), [hidden], dense),
                    q: Weight::sym(n("q"), [u64::from(TE_Q_HEADS) * hd, hidden], banks),
                    k: Weight::sym(n("k"), [u64::from(TE_KV_HEADS) * hd, hidden], banks),
                    v: Weight::sym(n("v"), [u64::from(TE_KV_HEADS) * hd, hidden], banks),
                    o: Weight::sym(n("o"), [hidden, u64::from(TE_Q_HEADS) * hd], banks),
                    q_norm: Weight::sym(n("q_norm"), [hd], dense),
                    k_norm: Weight::sym(n("k_norm"), [hd], dense),
                    mlp_norm: Weight::sym(n("mlp_norm"), [hidden], dense),
                    gate_up: Weight::sym(n("gate_up"), [2 * inter, hidden], banks)
                        .packed([inter, inter]),
                    down: Weight::sym(n("down"), [hidden, inter], banks),
                    kv: format!("te.kv.{l}"),
                }
            })
            .collect();
        TextEncoder {
            hidden: TE_HIDDEN,
            vocab: TE_VOCAB,
            q_heads: TE_Q_HEADS,
            kv_heads: TE_KV_HEADS,
            head_dim: TE_HEAD_DIM,
            inter: TE_INTER,
            theta: TE_THETA,
            eps: TE_EPS,
            sm_scale: (TE_HEAD_DIM as f32).sqrt().recip(),
            embed: Weight::sym("te.embed", [u64::from(TE_VOCAB), hidden], banks),
            layers,
        }
    }
}

pub struct Model {
    pub banks: Dtype,
    pub kv: Dtype,
    pub dims: Dims,
    pub dit: Dit,
    pub te: Option<TextEncoder>,
    pub vae: Option<super::vae::Vae>,
    pub shift: f32,
}

impl Model {
    #[must_use]
    pub fn turbo(banks: Dtype) -> Model {
        Model::new(
            banks,
            Dims::turbo(),
            Some(TextEncoder::qwen3_4b(banks)),
            Some(super::vae::Vae::flux(Dtype::Bf16)),
            3.0,
        )
    }

    #[must_use]
    pub fn mini(banks: Dtype) -> Model {
        Model::new(banks, Dims::mini(), None, None, 3.0)
    }

    fn new(
        banks: Dtype,
        d: Dims,
        te: Option<TextEncoder>,
        vae: Option<super::vae::Vae>,
        shift: f32,
    ) -> Model {
        assert_eq!(
            d.rope_dims.iter().sum::<u32>(),
            d.head_dim,
            "the rotary axes cover the whole head"
        );
        assert_eq!(
            d.heads * d.head_dim,
            d.dim,
            "plain MHA: heads × head_dim is the width"
        );
        let dense = crate::dense(banks);
        let dim = d.dim;
        let blocks = |stem: &str, count: u32, modulated: bool| -> Vec<Block> {
            (0..count)
                .map(|i| Block::at(&format!("dit.{stem}.{i}"), &d, modulated, banks))
                .collect()
        };
        let dit = Dit {
            x_embed: Linear::at("dit.x_embed", dim, PATCH_FEATURES, banks),
            x_pad_mod: Weight::sym("dit.x_pad_mod", [u64::from(2 * dim), 1], dense),
            cap_norm: Weight::sym("dit.cap_norm", [u64::from(d.cap_width)], dense),
            cap_embed: Linear::at("dit.cap_embed", dim, d.cap_width, banks),
            cap_pad_mod: Weight::sym("dit.cap_pad_mod", [u64::from(2 * dim), 1], dense),
            t_mlp0: Linear::at("dit.t_mlp0", T_MID, T_FREQ_DIM, banks),
            t_mlp1: Linear::at("dit.t_mlp1", ADALN_DIM, T_MID, banks),
            t_flip: Weight::sym("dit.t_flip", [1], Dtype::F32),
            noise_refiner: blocks("noise", d.refiner_layers, true),
            context_refiner: blocks("context", d.refiner_layers, false),
            layers: blocks("layer", d.joint_layers, true),
            final_ada: Linear::at("dit.final_ada", dim, ADALN_DIM, banks),
            final_linear: Linear::at("dit.final", PATCH_FEATURES, dim, banks),
        };
        Model {
            banks,
            kv: Dtype::Bf16,
            dims: d,
            dit,
            te,
            vae,
            shift,
        }
    }
}
