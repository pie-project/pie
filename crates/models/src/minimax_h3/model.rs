use poem_dsl::{Dtype, Stream, Weight};

pub const LATENT_CHANNELS: u32 = 24;
pub const AUDIO_CHANNELS: u32 = 32;
pub const PATCH_T: u32 = 1;
pub const PATCH_H: u32 = 2;
pub const PATCH_W: u32 = 2;
pub const VIDEO_FEATURES: u32 = LATENT_CHANNELS * PATCH_T * PATCH_H * PATCH_W;
pub const SPATIAL_COMPRESSION: u32 = 16;
pub const TEMPORAL_COMPRESSION: u32 = 4;

pub const HEAD_DIM: u32 = 128;

pub const NORM_EPS: f32 = 1e-5;

pub const T_MAX_PERIOD: f32 = 10_000.0;
pub const T_FLIP_SIN_COS: bool = true;
pub const T_SCALE: f32 = 1.0;

pub const ROPE_THETA: f32 = 10_000.0;
pub const ROPE_AXES: u8 = 3;

pub const ADALN_SLICES: u32 = 6;
pub const MODALITIES: u32 = 3;
pub const FINAL_SLICES: u32 = 2;

pub const TIMESTEP_SLOTS: u32 = 4;

pub const TRAIN_STEPS: u32 = 1;

pub const VIDEO_SHIFT: f32 = 12.0;
pub const AUDIO_SHIFT: f32 = 3.0;
pub const STEPS: u32 = 50;
pub const CONDITION_TIMESTEP: f32 = 0.999;

pub const TE_HIDDEN: u32 = 5120;
pub const TE_VOCAB: u32 = 151_936;
pub const TE_Q_HEADS: u32 = 64;
pub const TE_KV_HEADS: u32 = 8;
pub const TE_HEAD_DIM: u32 = 128;
pub const TE_INTER: u32 = 25_600;
pub const TE_THETA: f32 = 5_000_000.0;
pub const TE_EPS: f32 = 1e-6;
pub const TE_DEPTH: u32 = 64;
pub const TE_LAYERS: u32 = 50;
pub const TE_MAX_TOKENS: u32 = 262_144;

pub mod port {
    pub const LATENTS: u8 = 0;
    pub const REFERENCE: u8 = 1;
    pub const AUDIO: u8 = 2;
    pub const CONTEXT: u8 = 3;
    pub const CAPTION: u8 = 0;
    pub const TIMESTEP: u8 = 0;
    pub const POSITIONS: u8 = 0;
}

#[must_use]
pub const fn modality(stream: Stream) -> usize {
    match stream {
        Stream::Video | Stream::Reference | Stream::Image => 0,
        Stream::Audio => 2,
        Stream::Text | Stream::Context => 1,
    }
}

#[must_use]
pub const fn timestep_slot(stream: Stream) -> u32 {
    match stream {
        Stream::Video | Stream::Text | Stream::Context => 0,
        Stream::Reference | Stream::Image => 1,
        Stream::Audio => 2,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub dim: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub inter: u32,
    pub blocks: u32,
    pub refiners: u32,
    pub text_dim: u32,
    pub t_freq: u32,
    pub t_hidden: u32,
    pub t_dim: u32,
    pub rope_freqs: u32,
}

impl Dims {
    #[must_use]
    pub const fn h3(tp: u32) -> Dims {
        Dims {
            dim: 5376,
            heads: 56 / tp,
            head_dim: HEAD_DIM,
            inter: 14336 / tp,
            blocks: 50,
            refiners: 2,
            text_dim: TE_HIDDEN,
            t_freq: 256,
            t_hidden: 5376,
            t_dim: 2688,
            rope_freqs: 16,
        }
    }

    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            dim: 128,
            heads: 2,
            head_dim: 64,
            inter: 256,
            blocks: 2,
            refiners: 1,
            text_dim: 64,
            t_freq: 32,
            t_hidden: 128,
            t_dim: 64,
            rope_freqs: 8,
        }
    }

    #[must_use]
    pub const fn inner(&self) -> u32 {
        self.heads * self.head_dim
    }

    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        let per = 2 * self.rope_freqs;
        [per, per, per, 0]
    }

    #[must_use]
    pub const fn rotary_dim(&self) -> u32 {
        6 * self.rope_freqs
    }

    #[must_use]
    pub const fn adaln_width(&self) -> u32 {
        ADALN_SLICES * self.dim
    }
}

pub struct Linear {
    pub w: Weight,
    pub bias: Weight,
}

impl Linear {
    fn at(name: &str, out: u64, inp: u64, banks: Dtype) -> Linear {
        Linear {
            w: Weight::sym(name, [out, inp], banks),
            bias: Weight::sym(format!("{name}.bias"), [out], crate::dense(banks)),
        }
    }
}

pub struct Attn {
    pub qkv: Weight,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub out: Weight,
}

impl Attn {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Attn {
        let dense = crate::dense(banks);
        let (dim, inner) = (u64::from(d.dim), u64::from(d.inner()));
        Attn {
            qkv: Weight::sym(format!("{prefix}.qkv"), [3 * inner, dim], banks)
                .packed([inner, inner, inner]),
            q_norm: Weight::sym(format!("{prefix}.q_norm"), [u64::from(d.head_dim)], dense),
            k_norm: Weight::sym(format!("{prefix}.k_norm"), [u64::from(d.head_dim)], dense),
            out: Weight::sym(format!("{prefix}.out"), [dim, inner], banks),
        }
    }
}

pub struct Mlp {
    pub fc1: Weight,
    pub fc2: Weight,
}

impl Mlp {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Mlp {
        let (dim, inter) = (u64::from(d.dim), u64::from(d.inter));
        Mlp {
            fc1: Weight::sym(format!("{prefix}.fc1"), [2 * inter, dim], banks)
                .packed([inter, inter]),
            fc2: Weight::sym(format!("{prefix}.fc2"), [dim, inter], banks),
        }
    }
}

pub struct Block {
    pub norm1: Weight,
    pub norm2: Weight,
    pub attn: Attn,
    pub mlp: Mlp,
    pub adaln: [Linear; MODALITIES as usize],
}

impl Block {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Block {
        let width = u64::from(d.adaln_width());
        let t_dim = u64::from(d.t_dim);
        Block {
            norm1: Weight::sym(
                format!("{prefix}.norm1"),
                [u64::from(d.dim)],
                crate::dense(banks),
            ),
            norm2: Weight::sym(
                format!("{prefix}.norm2"),
                [u64::from(d.dim)],
                crate::dense(banks),
            ),
            attn: Attn::at(&format!("{prefix}.attn"), d, banks),
            mlp: Mlp::at(&format!("{prefix}.mlp"), d, banks),
            adaln: std::array::from_fn(|m| {
                Linear::at(&format!("{prefix}.adaln.{m}"), width, t_dim, banks)
            }),
        }
    }
}

pub struct Refiner {
    pub norm1: Weight,
    pub norm2: Weight,
    pub attn: Attn,
    pub mlp: Mlp,
}

impl Refiner {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Refiner {
        Refiner {
            norm1: Weight::sym(
                format!("{prefix}.norm1"),
                [u64::from(d.dim)],
                crate::dense(banks),
            ),
            norm2: Weight::sym(
                format!("{prefix}.norm2"),
                [u64::from(d.dim)],
                crate::dense(banks),
            ),
            attn: Attn::at(&format!("{prefix}.attn"), d, banks),
            mlp: Mlp::at(&format!("{prefix}.mlp"), d, banks),
        }
    }
}

pub struct Dit {
    pub video_patch: Linear,
    pub audio_patch: Linear,
    pub condition: Linear,
    pub t_in: Linear,
    pub t_out: Linear,
    pub refine: Vec<Refiner>,
    pub refine_norm: Weight,
    pub blocks: Vec<Block>,
    pub final_norm: Weight,
    pub final_adaln: Linear,
    pub video_out: Linear,
    pub audio_out: Linear,
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
    fn qwen3_vl_32b(banks: Dtype, tp: u32) -> TextEncoder {
        let dense = crate::dense(banks);
        let hidden = u64::from(TE_HIDDEN);
        let hd = u64::from(TE_HEAD_DIM);
        let inter = u64::from(TE_INTER / tp);
        let q_heads = TE_Q_HEADS / tp;
        let kv_heads = TE_KV_HEADS / tp;
        let layers = (0..TE_LAYERS)
            .map(|l| {
                let n = |s: &str| format!("te.layer.{l}.{s}");
                TeLayer {
                    attn_norm: Weight::sym(n("attn_norm"), [hidden], dense),
                    q: Weight::sym(n("q"), [u64::from(q_heads) * hd, hidden], banks).columns(),
                    k: Weight::sym(n("k"), [u64::from(kv_heads) * hd, hidden], banks).columns(),
                    v: Weight::sym(n("v"), [u64::from(kv_heads) * hd, hidden], banks).columns(),
                    o: Weight::sym(n("o"), [hidden, u64::from(q_heads) * hd], banks).rows(),
                    q_norm: Weight::sym(n("q_norm"), [hd], dense),
                    k_norm: Weight::sym(n("k_norm"), [hd], dense),
                    mlp_norm: Weight::sym(n("mlp_norm"), [hidden], dense),
                    gate_up: Weight::sym(n("gate_up"), [2 * inter, hidden], banks)
                        .packed([inter, inter]),
                    down: Weight::sym(n("down"), [hidden, inter], banks).rows(),
                    kv: format!("te.kv.{l}"),
                }
            })
            .collect();
        TextEncoder {
            hidden: TE_HIDDEN,
            vocab: TE_VOCAB,
            q_heads,
            kv_heads,
            head_dim: TE_HEAD_DIM,
            inter: TE_INTER / tp,
            theta: TE_THETA,
            eps: TE_EPS,
            sm_scale: (TE_HEAD_DIM as f32).sqrt().recip(),
            embed: Weight::sym("te.embed", [u64::from(TE_VOCAB), hidden], banks),
            layers,
        }
    }
}

pub struct Model {
    pub tp: u32,
    pub banks: Dtype,
    pub kv: Dtype,
    pub dims: Dims,
    pub dit: Dit,
    pub te: Option<TextEncoder>,
}

impl Model {
    #[must_use]
    pub fn fl2va(banks: Dtype, tp: u32) -> Model {
        Model::new(
            banks,
            tp,
            Dims::h3(tp),
            Some(TextEncoder::qwen3_vl_32b(banks, tp)),
        )
    }

    #[must_use]
    pub fn mini(banks: Dtype, tp: u32) -> Model {
        Model::new(banks, tp, Dims::mini(), None)
    }

    fn new(banks: Dtype, tp: u32, d: Dims, te: Option<TextEncoder>) -> Model {
        assert!(
            matches!(tp, 1 | 2 | 4),
            "tp {tp} is not a world this text ships"
        );
        assert_eq!(
            d.rotary_dim() % 2,
            0,
            "each rotary axis owns an even channel count"
        );
        assert!(
            d.rotary_dim() <= d.head_dim,
            "the three rotary axes cover {} of a {}-wide head",
            d.rotary_dim(),
            d.head_dim
        );
        let dense = crate::dense(banks);
        let dim = u64::from(d.dim);
        let dit = Dit {
            video_patch: Linear::at("dit.video_patch", dim, u64::from(VIDEO_FEATURES), banks),
            audio_patch: Linear::at("dit.audio_patch", dim, u64::from(AUDIO_CHANNELS), banks),
            condition: Linear::at("dit.condition", dim, u64::from(d.text_dim), banks),
            t_in: Linear::at(
                "dit.t_in",
                u64::from(d.t_hidden),
                u64::from(d.t_freq),
                banks,
            ),
            t_out: Linear::at(
                "dit.t_out",
                u64::from(d.t_dim),
                u64::from(d.t_hidden),
                banks,
            ),
            refine: (0..d.refiners)
                .map(|i| Refiner::at(&format!("dit.refine.{i}"), &d, banks))
                .collect(),
            refine_norm: Weight::sym("dit.refine_norm", [dim], dense),
            blocks: (0..d.blocks)
                .map(|i| Block::at(&format!("dit.block.{i}"), &d, banks))
                .collect(),
            final_norm: Weight::sym("dit.final_norm", [dim], dense),
            final_adaln: Linear::at(
                "dit.final_adaln",
                u64::from(FINAL_SLICES) * dim,
                u64::from(d.t_dim),
                banks,
            ),
            video_out: Linear::at("dit.video_out", u64::from(VIDEO_FEATURES), dim, banks),
            audio_out: Linear::at("dit.audio_out", u64::from(AUDIO_CHANNELS), dim, banks),
        };
        Model {
            tp,
            banks,
            kv: Dtype::Bf16,
            dims: d,
            dit,
            te,
        }
    }
}
