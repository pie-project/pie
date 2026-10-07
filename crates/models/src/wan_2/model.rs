use poem_dsl::{Dtype, Weight};

pub const PATCH_T: u32 = 1;
pub const PATCH_H: u32 = 2;
pub const PATCH_W: u32 = 2;
pub const PATCH_VOL: u32 = PATCH_T * PATCH_H * PATCH_W;

pub const VAE_SPATIAL_COMPRESSION: u32 = 16;
pub const VAE_TEMPORAL_COMPRESSION: u32 = 4;

pub const T_MAX_PERIOD: f32 = 10_000.0;
pub const T_FLIP_SIN_COS: bool = true;
pub const T_SCALE: f32 = 1.0;

pub const NORM_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 10_000.0;
pub const ROPE_AXES: u8 = 3;

pub const MOD_SLICES: u32 = 6;
pub const HEAD_SLICES: u32 = 2;

pub const TRAIN_STEPS: u32 = 1000;
pub const SHIFT_TI2V: f32 = 5.0;

pub const CONTEXT_LEN: u32 = 512;

pub const TE_HIDDEN: u32 = 4096;
pub const TE_VOCAB: u32 = 256_384;
pub const TE_HEADS: u32 = 64;
pub const TE_HEAD_DIM: u32 = 64;
pub const TE_INTER: u32 = 10_240;
pub const TE_LAYERS: u32 = 24;
pub const TE_EPS: f32 = 1e-6;
pub const TE_BUCKETS: u32 = 32;
pub const TE_MAX_DISTANCE: f32 = 128.0;
pub const TE_MAX_TOKENS: u32 = 512;

pub const VAE_Z: u32 = 48;
pub const VAE_PIX_CHANNELS: u32 = 12;
pub const VAE_RGB: u32 = 3;
pub const VAE_PATCH: u32 = 2;
pub const VAE_DECODER_DIMS: [u32; 5] = [1024, 1024, 1024, 512, 256];
pub const VAE_RESNETS: u32 = 3;
pub const VAE_TEMPORAL_UP: [bool; 4] = [true, true, false, false];
pub const VAE_ENCODER_DIMS: [u32; 5] = [160, 160, 320, 640, 640];
pub const VAE_ENC_RESNETS: u32 = 2;
pub const VAE_TEMPORAL_DOWN: [bool; 4] = [false, true, true, false];
pub const VAE_EPS: f32 = 1e-12;
pub const VAE_LATENTS_MEAN: [f32; VAE_Z as usize] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813,
    0.0891, 0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108,
    -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313,
    -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944,
    0.3573, -0.0548, -0.1681, -0.0667,
];
pub const VAE_LATENTS_STD: [f32; VAE_Z as usize] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

pub const VAE_MAX_LATENT_PLANE: u64 = 44 * 80;

pub mod port {
    pub const LATENTS: u8 = 0;
    pub const CONTEXT: u8 = 0;
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
    pub ffn: u32,
    pub layers: u32,
    pub in_channels: u32,
    pub out_channels: u32,
    pub text_dim: u32,
    pub freq_dim: u32,
}

impl Dims {
    #[must_use]
    pub const fn ti2v_5b() -> Dims {
        Dims {
            dim: 3072,
            heads: 24,
            head_dim: 128,
            ffn: 14_336,
            layers: 30,
            in_channels: 48,
            out_channels: 48,
            text_dim: 4096,
            freq_dim: 256,
        }
    }

    #[must_use]
    pub const fn mini_d128() -> Dims {
        Dims {
            dim: 256,
            heads: 2,
            head_dim: 128,
            ffn: 512,
            layers: 2,
            in_channels: 16,
            out_channels: 16,
            text_dim: 64,
            freq_dim: 256,
        }
    }

    #[must_use]
    pub const fn mini_nano() -> Dims {
        Dims {
            dim: 48,
            heads: 2,
            head_dim: 24,
            ffn: 128,
            layers: 2,
            in_channels: 16,
            out_channels: 16,
            text_dim: 64,
            freq_dim: 32,
        }
    }

    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        let hw = 2 * (self.head_dim / 6);
        [self.head_dim - 2 * hw, hw, hw, 0]
    }

    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    #[must_use]
    pub const fn patch_in(&self) -> u32 {
        self.in_channels * PATCH_VOL
    }

    #[must_use]
    pub const fn patch_out(&self) -> u32 {
        self.out_channels * PATCH_VOL
    }
}

pub struct Linear {
    pub w: Weight,
    pub bias: Weight,
}

impl Linear {
    fn at(name: &str, out: u32, in_: u32, banks: Dtype) -> Linear {
        Linear {
            w: Weight::sym(name, [u64::from(out), u64::from(in_)], banks),
            bias: Weight::sym(
                format!("{name}.bias"),
                [u64::from(out)],
                crate::dense(banks),
            ),
        }
    }

    fn packed(name: &str, seams: &[u32], in_: u32, banks: Dtype) -> Linear {
        let out: u64 = seams.iter().map(|&s| u64::from(s)).sum();
        let seams: Vec<u64> = seams.iter().map(|&s| u64::from(s)).collect();
        Linear {
            w: Weight::sym(name, [out, u64::from(in_)], banks).packed(seams.clone()),
            bias: Weight::sym(format!("{name}.bias"), [out], crate::dense(banks)).packed(seams),
        }
    }
}

pub struct SelfAttn {
    pub qkv: Linear,
    pub norm_q: Weight,
    pub norm_k: Weight,
    pub out: Linear,
}

pub struct CrossAttn {
    pub q: Linear,
    pub kv: Linear,
    pub norm_q: Weight,
    pub norm_k: Weight,
    pub out: Linear,
}

pub struct Ffn {
    pub up: Linear,
    pub down: Linear,
}

pub struct Block {
    pub table: Weight,
    pub self_attn: SelfAttn,
    pub norm2: Weight,
    pub norm2_bias: Weight,
    pub cross: CrossAttn,
    pub ffn: Ffn,
}

impl Block {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Block {
        let dense = crate::dense(banks);
        let dim = d.dim;
        let n = |s: &str| format!("{prefix}.{s}");
        let gain = |s: &str| Weight::sym(n(s), [u64::from(dim)], dense);
        Block {
            table: Weight::sym(n("table"), [u64::from(MOD_SLICES * dim)], Dtype::F32),
            self_attn: SelfAttn {
                qkv: Linear::packed(&n("self.qkv"), &[dim, dim, dim], dim, banks),
                norm_q: gain("self.norm_q"),
                norm_k: gain("self.norm_k"),
                out: Linear::at(&n("self.out"), dim, dim, banks),
            },
            norm2: gain("norm2"),
            norm2_bias: gain("norm2.bias"),
            cross: CrossAttn {
                q: Linear::at(&n("cross.q"), dim, dim, banks),
                kv: Linear::packed(&n("cross.kv"), &[dim, dim], dim, banks),
                norm_q: gain("cross.norm_q"),
                norm_k: gain("cross.norm_k"),
                out: Linear::at(&n("cross.out"), dim, dim, banks),
            },
            ffn: Ffn {
                up: Linear::at(&n("ffn.up"), d.ffn, dim, banks),
                down: Linear::at(&n("ffn.down"), dim, d.ffn, banks),
            },
        }
    }
}

pub struct Embedder {
    pub linear_1: Linear,
    pub linear_2: Linear,
}

impl Embedder {
    fn at(prefix: &str, in_: u32, dim: u32, banks: Dtype) -> Embedder {
        Embedder {
            linear_1: Linear::at(&format!("{prefix}.1"), dim, in_, banks),
            linear_2: Linear::at(&format!("{prefix}.2"), dim, dim, banks),
        }
    }
}

pub struct Dit {
    pub patch_embed: Linear,
    pub text_embed: Embedder,
    pub time_embed: Embedder,
    pub time_proj: Linear,
    pub head_proj: Linear,
    pub blocks: Vec<Block>,
    pub head_table: Weight,
    pub proj_out: Linear,
}

pub struct TeLayer {
    pub attn_norm: Weight,
    pub q: Weight,
    pub k: Weight,
    pub v: Weight,
    pub o: Weight,
    pub rel_bias: Weight,
    pub ffn_norm: Weight,
    pub wi_0: Weight,
    pub wi_1: Weight,
    pub wo: Weight,
}

pub struct TextEncoder {
    pub embed: Weight,
    pub layers: Vec<TeLayer>,
    pub final_norm: Weight,
}

impl TextEncoder {
    fn umt5_xxl(banks: Dtype) -> TextEncoder {
        let dense = crate::dense(banks);
        let hidden = u64::from(TE_HIDDEN);
        let inner = u64::from(TE_HEADS * TE_HEAD_DIM);
        let inter = u64::from(TE_INTER);
        let layers = (0..TE_LAYERS)
            .map(|l| {
                let n = |s: &str| format!("te.layer.{l}.{s}");
                TeLayer {
                    attn_norm: Weight::sym(n("attn_norm"), [hidden], dense),
                    q: Weight::sym(n("q"), [inner, hidden], banks),
                    k: Weight::sym(n("k"), [inner, hidden], banks),
                    v: Weight::sym(n("v"), [inner, hidden], banks),
                    o: Weight::sym(n("o"), [hidden, inner], banks),
                    rel_bias: Weight::sym(
                        n("rel_bias"),
                        [u64::from(TE_BUCKETS), u64::from(TE_HEADS)],
                        dense,
                    ),
                    ffn_norm: Weight::sym(n("ffn_norm"), [hidden], dense),
                    wi_0: Weight::sym(n("wi_0"), [inter, hidden], banks),
                    wi_1: Weight::sym(n("wi_1"), [inter, hidden], banks),
                    wo: Weight::sym(n("wo"), [hidden, inter], banks),
                }
            })
            .collect();
        TextEncoder {
            embed: Weight::sym("te.embed", [u64::from(TE_VOCAB), hidden], banks),
            layers,
            final_norm: Weight::sym("te.final_norm", [hidden], dense),
        }
    }
}

pub struct Conv {
    pub w: Weight,
    pub bias: Weight,
    pub c_in: u32,
    pub c_out: u32,
    pub k: [u32; 3],
    pub front: u32,
    pub cache: Option<String>,
    pub plane: u64,
}

impl Conv {
    fn at(name: &str, c_out: u32, c_in: u32, k: [u32; 3], plane: u64, banks: Dtype) -> Conv {
        let taps = k[0] * k[1] * k[2];
        Conv {
            w: Weight::sym(
                name,
                [u64::from(c_out), u64::from(c_in) * u64::from(taps)],
                banks,
            )
            .conv_taps_major(c_in, taps),
            bias: Weight::sym(format!("{name}.bias"), [u64::from(c_out)], Dtype::F32),
            c_in,
            c_out,
            k,
            front: k[0].saturating_sub(1),
            cache: (k[0] > 1).then(|| format!("{name}.frames")),
            plane,
        }
    }

    #[must_use]
    fn fronting(mut self, front: u32) -> Conv {
        self.front = front;
        self
    }

    #[must_use]
    pub fn slab(&self) -> [u64; 2] {
        [u64::from(self.front) * self.plane, u64::from(self.c_in)]
    }
}

pub struct Resnet {
    pub norm1: Weight,
    pub conv1: Conv,
    pub norm2: Weight,
    pub conv2: Conv,
    pub shortcut: Option<Conv>,
}

impl Resnet {
    fn at(prefix: &str, c_in: u32, c_out: u32, plane: u64, banks: Dtype) -> Resnet {
        let dense = crate::dense(banks);
        Resnet {
            norm1: Weight::sym(format!("{prefix}.norm1"), [u64::from(c_in)], dense),
            conv1: Conv::at(
                &format!("{prefix}.conv1"),
                c_out,
                c_in,
                [3, 3, 3],
                plane,
                banks,
            ),
            norm2: Weight::sym(format!("{prefix}.norm2"), [u64::from(c_out)], dense),
            conv2: Conv::at(
                &format!("{prefix}.conv2"),
                c_out,
                c_out,
                [3, 3, 3],
                plane,
                banks,
            ),
            shortcut: (c_in != c_out).then(|| {
                Conv::at(
                    &format!("{prefix}.shortcut"),
                    c_out,
                    c_in,
                    [1, 1, 1],
                    plane,
                    banks,
                )
            }),
        }
    }
}

pub struct MidAttention {
    pub norm: Weight,
    pub qkv: Linear,
    pub proj: Linear,
}

pub struct Upsampler {
    pub time_conv: Option<Conv>,
    pub resample: Conv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shortcut {
    Nearest222,
    ShuffleH,
}

pub struct UpBlock {
    pub resnets: Vec<Resnet>,
    pub upsampler: Option<Upsampler>,
    pub shortcut: Option<Shortcut>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AvgDown {
    pub factor: [u32; 3],
    pub group: u32,
}

impl AvgDown {
    #[must_use]
    pub fn at(c_in: u32, c_out: u32, factor_t: u32, factor_s: u32) -> AvgDown {
        let factor = [factor_t, factor_s, factor_s];
        let volume = factor_t * factor_s * factor_s;
        assert_eq!(
            c_in * volume % c_out,
            0,
            "an AvgDown3D's widened channels must fold into whole groups"
        );
        AvgDown {
            factor,
            group: c_in * volume / c_out,
        }
    }
}

pub struct Downsampler {
    pub resample: Conv,
    pub time_conv: Option<Conv>,
}

pub struct DownBlock {
    pub resnets: Vec<Resnet>,
    pub downsampler: Option<Downsampler>,
    pub shortcut: AvgDown,
}

pub struct VaeEncoder {
    pub conv_in: Conv,
    pub down: Vec<DownBlock>,
    pub mid_res0: Resnet,
    pub mid_attn: MidAttention,
    pub mid_res1: Resnet,
    pub norm_out: Weight,
    pub conv_out: Conv,
    pub quant: Conv,
    pub norm_bias: Weight,
    pub norm_scale: Weight,
}

impl VaeEncoder {
    fn wan22(banks: Dtype) -> VaeEncoder {
        let dense = crate::dense(banks);
        let dims = VAE_ENCODER_DIMS;
        let p0 = VAE_MAX_LATENT_PLANE;
        let mut plane = 64 * p0;
        let mut down = Vec::new();
        for i in 0..4 {
            let (c_in, c_out) = (dims[i], dims[i + 1]);
            let prefix = format!("vae.enc.down.{i}");
            let resnets = (0..VAE_ENC_RESNETS)
                .map(|r| {
                    Resnet::at(
                        &format!("{prefix}.res.{r}"),
                        if r == 0 { c_in } else { c_out },
                        c_out,
                        plane,
                        banks,
                    )
                })
                .collect();
            let down_flag = i != 3;
            let temporal = VAE_TEMPORAL_DOWN[i];
            let downsampler = down_flag.then(|| Downsampler {
                resample: Conv::at(
                    &format!("{prefix}.resample"),
                    c_out,
                    c_out,
                    [1, 3, 3],
                    plane,
                    banks,
                ),
                time_conv: temporal.then(|| {
                    Conv::at(
                        &format!("{prefix}.time_conv"),
                        c_out,
                        c_out,
                        [3, 1, 1],
                        plane / 4,
                        banks,
                    )
                    .fronting(1)
                }),
            });
            down.push(DownBlock {
                resnets,
                downsampler,
                shortcut: AvgDown::at(
                    c_in,
                    c_out,
                    if temporal { 2 } else { 1 },
                    if down_flag { 2 } else { 1 },
                ),
            });
            if down_flag {
                plane /= 4;
            }
        }
        let top = dims[4];
        VaeEncoder {
            conv_in: Conv::at(
                "vae.enc.conv_in",
                dims[0],
                VAE_PIX_CHANNELS,
                [3, 3, 3],
                64 * p0,
                banks,
            ),
            down,
            mid_res0: Resnet::at("vae.enc.mid.res.0", top, top, plane, banks),
            mid_attn: MidAttention {
                norm: Weight::sym("vae.enc.mid.attn.norm", [u64::from(top)], dense),
                qkv: Linear::at("vae.enc.mid.attn.qkv", 3 * top, top, banks),
                proj: Linear::at("vae.enc.mid.attn.proj", top, top, banks),
            },
            mid_res1: Resnet::at("vae.enc.mid.res.1", top, top, plane, banks),
            norm_out: Weight::sym("vae.enc.norm_out", [u64::from(top)], dense),
            conv_out: Conv::at("vae.enc.conv_out", 2 * VAE_Z, top, [3, 3, 3], plane, banks),
            quant: Conv::at("vae.enc.quant", VAE_Z, 2 * VAE_Z, [1, 1, 1], plane, banks),
            norm_bias: Weight::sym("vae.enc.norm_bias", [u64::from(VAE_Z)], dense),
            norm_scale: Weight::sym("vae.enc.norm_scale", [u64::from(VAE_Z)], dense),
        }
    }

    pub fn cached_convs(&self) -> impl Iterator<Item = &Conv> + '_ {
        let mut out: Vec<&Conv> = vec![&self.conv_in];
        for block in &self.down {
            for r in &block.resnets {
                out.push(&r.conv1);
                out.push(&r.conv2);
            }
            if let Some(Downsampler {
                time_conv: Some(tc),
                ..
            }) = &block.downsampler
            {
                out.push(tc);
            }
        }
        for r in [&self.mid_res0, &self.mid_res1] {
            out.push(&r.conv1);
            out.push(&r.conv2);
        }
        out.push(&self.conv_out);
        out.into_iter().filter(|c| c.cache.is_some())
    }
}

pub struct Vae {
    pub denorm_bias: Weight,
    pub denorm_scale: Weight,
    pub post_quant: Conv,
    pub conv_in: Conv,
    pub mid_res0: Resnet,
    pub mid_attn: MidAttention,
    pub mid_res1: Resnet,
    pub up: Vec<UpBlock>,
    pub norm_out: Weight,
    pub conv_out: Conv,
    pub enc: VaeEncoder,
}

impl Vae {
    fn wan22(banks: Dtype) -> Vae {
        let dense = crate::dense(banks);
        let dims = VAE_DECODER_DIMS;
        let top = dims[0];
        let p0 = VAE_MAX_LATENT_PLANE;
        let mut plane = p0;
        let mut up = Vec::new();
        for i in 0..4 {
            let (c_in, c_out) = (dims[i], dims[i + 1]);
            let prefix = format!("vae.up.{i}");
            let resnets = (0..VAE_RESNETS)
                .map(|r| {
                    Resnet::at(
                        &format!("{prefix}.res.{r}"),
                        if r == 0 { c_in } else { c_out },
                        c_out,
                        plane,
                        banks,
                    )
                })
                .collect();
            let up_flag = i != 3;
            let upsampler = up_flag.then(|| Upsampler {
                time_conv: VAE_TEMPORAL_UP[i].then(|| {
                    Conv::at(
                        &format!("{prefix}.time_conv"),
                        2 * c_out,
                        c_out,
                        [3, 1, 1],
                        plane,
                        banks,
                    )
                }),
                resample: Conv::at(
                    &format!("{prefix}.resample"),
                    c_out,
                    c_out,
                    [1, 3, 3],
                    4 * plane,
                    banks,
                ),
            });
            let shortcut = up_flag.then(|| {
                if VAE_TEMPORAL_UP[i] {
                    assert_eq!(c_in, c_out, "a (2, 2, 2) DupUp3D keeps the width");
                    Shortcut::Nearest222
                } else {
                    assert_eq!(c_in, 2 * c_out, "a (1, 2, 2) DupUp3D halves the width");
                    Shortcut::ShuffleH
                }
            });
            up.push(UpBlock {
                resnets,
                upsampler,
                shortcut,
            });
            if up_flag {
                plane *= 4;
            }
        }
        let last = dims[4];
        Vae {
            denorm_bias: Weight::sym("vae.denorm_bias", [u64::from(VAE_Z)], dense),
            denorm_scale: Weight::sym("vae.denorm_scale", [u64::from(VAE_Z)], dense),
            post_quant: Conv::at("vae.post_quant", VAE_Z, VAE_Z, [1, 1, 1], p0, banks),
            conv_in: Conv::at("vae.conv_in", top, VAE_Z, [3, 3, 3], p0, banks),
            mid_res0: Resnet::at("vae.mid.res.0", top, top, p0, banks),
            mid_attn: MidAttention {
                norm: Weight::sym("vae.mid.attn.norm", [u64::from(top)], dense),
                qkv: Linear::at("vae.mid.attn.qkv", 3 * top, top, banks),
                proj: Linear::at("vae.mid.attn.proj", top, top, banks),
            },
            mid_res1: Resnet::at("vae.mid.res.1", top, top, p0, banks),
            up,
            norm_out: Weight::sym("vae.norm_out", [u64::from(last)], dense),
            conv_out: Conv::at(
                "vae.conv_out",
                VAE_PIX_CHANNELS,
                last,
                [3, 3, 3],
                plane,
                banks,
            ),
            enc: VaeEncoder::wan22(banks),
        }
    }

    pub fn cached_convs(&self) -> impl Iterator<Item = &Conv> + '_ {
        let mut out: Vec<&Conv> = vec![&self.conv_in];
        for r in [&self.mid_res0, &self.mid_res1] {
            out.push(&r.conv1);
            out.push(&r.conv2);
        }
        for block in &self.up {
            for r in &block.resnets {
                out.push(&r.conv1);
                out.push(&r.conv2);
            }
            if let Some(Upsampler {
                time_conv: Some(tc),
                ..
            }) = &block.upsampler
            {
                out.push(tc);
            }
        }
        out.push(&self.conv_out);
        out.into_iter().filter(|c| c.cache.is_some())
    }
}

pub struct Model {
    pub tp: u32,
    pub banks: Dtype,
    pub dims: Dims,
    pub dit: Dit,
    pub te: Option<TextEncoder>,
    pub vae: Option<Vae>,
    pub shift: f32,
}

impl Model {
    #[must_use]
    pub fn ti2v_5b(banks: Dtype, tp: u32) -> Model {
        Model::new(
            banks,
            tp,
            Dims::ti2v_5b(),
            Some(TextEncoder::umt5_xxl(banks)),
            Some(Vae::wan22(Dtype::Bf16)),
            SHIFT_TI2V,
        )
    }

    #[must_use]
    pub fn mini_d128(banks: Dtype, tp: u32) -> Model {
        Model::new(banks, tp, Dims::mini_d128(), None, None, SHIFT_TI2V)
    }

    #[must_use]
    pub fn mini_nano(banks: Dtype, tp: u32) -> Model {
        Model::new(banks, tp, Dims::mini_nano(), None, None, SHIFT_TI2V)
    }

    fn new(
        banks: Dtype,
        tp: u32,
        d: Dims,
        te: Option<TextEncoder>,
        vae: Option<Vae>,
        shift: f32,
    ) -> Model {
        assert_eq!(
            tp, 1,
            "this text ships one-rank rows; tp {tp} is not a world it states"
        );
        assert_eq!(
            d.heads * d.head_dim,
            d.dim,
            "plain MHA: heads × head_dim is the width"
        );
        assert_eq!(
            d.rope_dims().iter().sum::<u32>(),
            d.head_dim,
            "the three rotary axes cover the whole head"
        );
        if te.is_some() {
            assert_eq!(d.text_dim, TE_HIDDEN, "the context rows are the encoder's");
        }
        if vae.is_some() {
            assert_eq!(d.in_channels, VAE_Z, "the latent is the VAE's");
        }
        let dim = d.dim;
        let dit = Dit {
            patch_embed: Linear::at("dit.patch_embed", dim, d.patch_in(), banks),
            text_embed: Embedder::at("dit.text_embed", d.text_dim, dim, banks),
            time_embed: Embedder::at("dit.time_embed", d.freq_dim, dim, banks),
            time_proj: Linear::at("dit.time_proj", MOD_SLICES * dim, dim, banks),
            head_proj: Linear::at("dit.head_proj", HEAD_SLICES * dim, dim, banks),
            blocks: (0..d.layers)
                .map(|i| Block::at(&format!("dit.block.{i}"), &d, banks))
                .collect(),
            head_table: Weight::sym("dit.head_table", [u64::from(HEAD_SLICES * dim)], Dtype::F32),
            proj_out: Linear::at("dit.proj_out", d.patch_out(), dim, banks),
        };
        Model {
            tp,
            banks,
            dims: d,
            dit,
            te,
            vae,
            shift,
        }
    }
}
