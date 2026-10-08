use poem::{Dtype, Weight};

pub const IN_CHANNELS: u32 = 128;
pub const VAE_CHANNELS: u32 = 32;
pub const PACK: u32 = 2;
pub const VAE_COMPRESSION: u32 = 8;
pub const TOKEN_COMPRESSION: u32 = VAE_COMPRESSION * PACK;

pub const HEAD_DIM: u32 = 128;
pub const ROPE_DIMS: [u32; 4] = [32, 32, 32, 32];
pub const ROPE_THETA: f32 = 2000.0;
pub const ROPE_AXES: u8 = 4;
pub const REFERENCE_TIME_STRIDE: u32 = 10;

pub const T_FREQ_DIM: u32 = 256;
pub const T_MAX_PERIOD: f32 = 10_000.0;
pub const T_FLIP_SIN_COS: bool = true;
pub const T_SCALE: f32 = 1.0;
pub const GUIDANCE_SCALE: f32 = 1000.0;

pub const NORM_EPS: f32 = 1e-6;
pub const SM_SCALE: f32 = 0.088_388_35;

pub const DOUBLE_MOD_SLICES: u32 = 6;
pub const SINGLE_MOD_SLICES: u32 = 3;

pub const MLP_RATIO: u32 = 3;

pub const TRAIN_STEPS: u32 = 1000;

pub const TE_HIDDEN: u32 = 2560;
pub const TE_VOCAB: u32 = 151_936;
pub const TE_Q_HEADS: u32 = 32;
pub const TE_KV_HEADS: u32 = 8;
pub const TE_HEAD_DIM: u32 = 128;
pub const TE_INTER: u32 = 9728;
pub const TE_THETA: f32 = 1_000_000.0;
pub const TE_EPS: f32 = 1e-6;
pub const TE_DEPTH: u32 = 36;
pub const TE_TAPS: [u32; 3] = [9, 18, 27];
pub const TE_LAYERS: u32 = 27;
pub const TE_MAX_TOKENS: u32 = 512;
pub const TE_CONTEXT_WIDTH: u32 = 3 * TE_HIDDEN;

pub mod port {
    pub const LATENTS: u8 = 0;
    pub const CONTEXT: u8 = 0;
    pub const TIMESTEP: u8 = 0;
    pub const GUIDANCE: u8 = 1;
    pub const POSITIONS: u8 = 0;
    pub const VOXELS: u8 = 0;
    pub const PIXEL_VOXELS: u8 = 1;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub dim: u32,
    pub heads: u32,
    pub inter: u32,
    pub context_in: u32,
    pub double_blocks: u32,
    pub single_blocks: u32,
    pub guidance_embeds: bool,
}

impl Dims {
    #[must_use]
    pub const fn klein_4b() -> Dims {
        Dims {
            dim: 3072,
            heads: 24,
            inter: 3072 * MLP_RATIO,
            context_in: TE_CONTEXT_WIDTH,
            double_blocks: 5,
            single_blocks: 20,
            guidance_embeds: false,
        }
    }

    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            dim: 256,
            heads: 2,
            inter: 256 * MLP_RATIO,
            context_in: 192,
            double_blocks: 2,
            single_blocks: 2,
            guidance_embeds: true,
        }
    }
}

pub type Linear = Weight;

pub struct Attn {
    pub qkv: Linear,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub out: Linear,
}

impl Attn {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Attn {
        let dense = crate::dense(banks);
        let dim = u64::from(d.dim);
        Attn {
            qkv: Weight::sym(format!("{prefix}.qkv"), [3 * dim, dim], banks)
                .packed([dim, dim, dim]),
            q_norm: Weight::sym(format!("{prefix}.q_norm"), [u64::from(HEAD_DIM)], dense),
            k_norm: Weight::sym(format!("{prefix}.k_norm"), [u64::from(HEAD_DIM)], dense),
            out: Weight::sym(format!("{prefix}.out"), [dim, dim], banks),
        }
    }
}

pub struct Swiglu {
    pub linear_in: Linear,
    pub linear_out: Linear,
}

impl Swiglu {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Swiglu {
        let (dim, inter) = (u64::from(d.dim), u64::from(d.inter));
        Swiglu {
            linear_in: Weight::sym(format!("{prefix}.in"), [2 * inter, dim], banks)
                .packed([inter, inter]),
            linear_out: Weight::sym(format!("{prefix}.out"), [dim, inter], banks),
        }
    }
}

pub struct Side {
    pub attn: Attn,
    pub ff: Swiglu,
}

impl Side {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Side {
        Side {
            attn: Attn::at(&format!("{prefix}.attn"), d, banks),
            ff: Swiglu::at(&format!("{prefix}.ff"), d, banks),
        }
    }
}

pub struct DoubleBlock {
    pub img: Side,
    pub txt: Side,
}

pub struct SingleBlock {
    pub in_proj: Linear,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub out_attn: Linear,
    pub out_mlp: Linear,
}

impl SingleBlock {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> SingleBlock {
        let dense = crate::dense(banks);
        let (dim, inter) = (u64::from(d.dim), u64::from(d.inter));
        SingleBlock {
            in_proj: Weight::sym(format!("{prefix}.in"), [3 * dim + 2 * inter, dim], banks)
                .packed([dim, dim, dim, inter, inter]),
            q_norm: Weight::sym(format!("{prefix}.q_norm"), [u64::from(HEAD_DIM)], dense),
            k_norm: Weight::sym(format!("{prefix}.k_norm"), [u64::from(HEAD_DIM)], dense),
            out_attn: Weight::sym(format!("{prefix}.out_attn"), [dim, dim], banks),
            out_mlp: Weight::sym(format!("{prefix}.out_mlp"), [dim, inter], banks),
        }
    }
}

pub struct Embedder {
    pub linear_1: Linear,
    pub linear_2: Linear,
}

impl Embedder {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Embedder {
        let dim = u64::from(d.dim);
        Embedder {
            linear_1: Weight::sym(format!("{prefix}.1"), [dim, u64::from(T_FREQ_DIM)], banks),
            linear_2: Weight::sym(format!("{prefix}.2"), [dim, dim], banks),
        }
    }
}

pub struct Dit {
    pub x_embed: Linear,
    pub context_embed: Option<Linear>,
    pub t_embed: Embedder,
    pub g_embed: Option<Embedder>,
    pub mod_img: Linear,
    pub mod_txt: Linear,
    pub mod_single: Linear,
    pub double: Vec<DoubleBlock>,
    pub single: Vec<SingleBlock>,
    pub norm_out: Linear,
    pub proj_out: Linear,
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
    pub context_embed: [Linear; 3],
}

impl TextEncoder {
    fn qwen3_4b(d: &Dims, banks: Dtype) -> TextEncoder {
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
        let dim = u64::from(d.dim);
        let tap = |i: usize| Weight::sym(format!("dit.context_embed.{i}"), [dim, hidden], banks);
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
            context_embed: [tap(0), tap(1), tap(2)],
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
}

impl Model {
    #[must_use]
    pub fn klein_4b(banks: Dtype) -> Model {
        let d = Dims::klein_4b();
        Model::new(
            banks,
            d,
            Some(TextEncoder::qwen3_4b(&d, banks)),
            Some(super::vae::Vae::flux2(Dtype::Bf16)),
        )
    }

    #[must_use]
    pub fn mini(banks: Dtype) -> Model {
        Model::new(banks, Dims::mini(), None, None)
    }

    fn new(banks: Dtype, d: Dims, te: Option<TextEncoder>, vae: Option<super::vae::Vae>) -> Model {
        assert_eq!(
            d.heads * HEAD_DIM,
            d.dim,
            "plain MHA over 128-wide heads: heads × 128 is the width"
        );
        assert_eq!(
            ROPE_DIMS.iter().sum::<u32>(),
            HEAD_DIM,
            "the four rotary axes cover the whole head"
        );
        let dim = u64::from(d.dim);
        let dit = Dit {
            x_embed: Weight::sym("dit.x_embed", [dim, u64::from(IN_CHANNELS)], banks),
            context_embed: te
                .is_none()
                .then(|| Weight::sym("dit.context_embed", [dim, u64::from(d.context_in)], banks)),
            t_embed: Embedder::at("dit.t_embed", &d, banks),
            g_embed: d
                .guidance_embeds
                .then(|| Embedder::at("dit.g_embed", &d, banks)),
            mod_img: Weight::sym(
                "dit.mod_img",
                [u64::from(DOUBLE_MOD_SLICES) * dim, dim],
                banks,
            ),
            mod_txt: Weight::sym(
                "dit.mod_txt",
                [u64::from(DOUBLE_MOD_SLICES) * dim, dim],
                banks,
            ),
            mod_single: Weight::sym(
                "dit.mod_single",
                [u64::from(SINGLE_MOD_SLICES) * dim, dim],
                banks,
            ),
            double: (0..d.double_blocks)
                .map(|i| DoubleBlock {
                    img: Side::at(&format!("dit.double.{i}.img"), &d, banks),
                    txt: Side::at(&format!("dit.double.{i}.txt"), &d, banks),
                })
                .collect(),
            single: (0..d.single_blocks)
                .map(|i| SingleBlock::at(&format!("dit.single.{i}"), &d, banks))
                .collect(),
            norm_out: Weight::sym("dit.norm_out", [2 * dim, dim], banks),
            proj_out: Weight::sym("dit.proj_out", [u64::from(IN_CHANNELS), dim], banks),
        };
        Model {
            banks,
            kv: Dtype::Bf16,
            dims: d,
            dit,
            te,
            vae,
        }
    }
}
