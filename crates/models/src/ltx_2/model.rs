use poem_dsl::{Dtype, Weight};

pub const PATCH_T: u32 = 1;
pub const PATCH_H: u32 = 1;
pub const PATCH_W: u32 = 1;

pub const VAE_SPATIAL_COMPRESSION: u32 = 32;
pub const VAE_TEMPORAL_COMPRESSION: u32 = 8;
pub const VAE_Z: u32 = 128;

pub const VAE_RGB: u32 = 3;
pub const VAE_PATCH: u32 = 4;
pub const VAE_EPS: f32 = 1e-8;
pub const VAE_DECODER_DIMS: [u32; 5] = [1024, 512, 512, 256, 128];
pub const VAE_MID_RESNETS: u32 = 2;
pub const VAE_UP_RESNETS: [u32; 4] = [2, 4, 6, 4];
pub const VAE_UP_STRIDES: [[u32; 3]; 4] = [[2, 2, 2], [2, 2, 2], [2, 1, 1], [1, 2, 2]];

pub const T_FREQ_DIM: u32 = 256;
pub const T_MAX_PERIOD: f32 = 10_000.0;
pub const T_FLIP_SIN_COS: bool = true;
pub const T_SCALE: f32 = 1.0;

pub const NORM_EPS: f32 = 1e-6;

pub const ROPE_THETA: f32 = 10_000.0;
pub const ROPE_MAX_POS: [f32; 3] = [20.0, 2048.0, 2048.0];
pub const AUDIO_ROPE_MAX_POS: f32 = 20.0;
pub const CROSS_ROPE_MAX_POS: f32 = 20.0;
pub const ROPE_AXES: u8 = 3;
pub const AUDIO_ROPE_AXES: u8 = 1;

pub const VIDEO_SCALE: [f32; 3] = [8.0, 32.0, 32.0];
pub const AUDIO_SCALE: f32 = 4.0;
pub const CAUSAL_OFFSET: f32 = 1.0;
pub const AUDIO_SAMPLING_RATE: f32 = 16_000.0;
pub const AUDIO_HOP: f32 = 160.0;

pub const GATE_SCALE: f32 = 2.0;

pub const MOD_SLICES: u32 = 9;
pub const AV_SS_SLICES: u32 = 4;
pub const AV_GATE_SLICES: u32 = 1;
pub const PROMPT_SLICES: u32 = 2;
pub const HEAD_SLICES: u32 = 2;

pub const AV_GATE_TIMESTEP_SCALE: f32 = 1.0;

pub const TRAIN_STEPS: u32 = 1000;
pub const DISTILLED_SIGMAS: [f32; 8] = [
    1.0, 0.993_75, 0.987_5, 0.981_25, 0.975, 0.909_375, 0.725, 0.421_875,
];
pub const STAGE2_SIGMAS: [f32; 3] = [0.909_375, 0.725, 0.421_875];

pub const TEXT_LAYERS: u32 = 49;
pub const TEXT_LEN: u32 = 1024;
pub const CONN_REGISTERS: u32 = 128;
pub const CONN_ROPE_BASE: f32 = 4096.0;
pub const CONN_FF_MULT: u32 = 4;

pub mod port {
    pub const LATENTS: u8 = 0;
    pub const CONTEXT: u8 = 0;
    pub const AUDIO_CONTEXT: u8 = 1;
    pub const TEXT: u8 = 1;
    pub const TIMESTEP: u8 = 0;
    pub const POSITIONS: u8 = 0;
    pub const TIME_POSITIONS: u8 = 1;
    pub const VOXELS: u8 = 0;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub layers: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub audio_heads: u32,
    pub audio_head_dim: u32,
    pub channels: u32,
    pub cross_dim: u32,
    pub audio_cross_dim: u32,
    pub ff_mult: u32,
    pub caption: u32,
    pub conn_layers: u32,
}

impl Dims {
    #[must_use]
    pub const fn ltx_2_5() -> Dims {
        Dims {
            layers: 48,
            heads: 32,
            head_dim: 128,
            audio_heads: 32,
            audio_head_dim: 64,
            channels: 128,
            cross_dim: 4096,
            audio_cross_dim: 2048,
            ff_mult: 4,
            caption: 3840,
            conn_layers: 8,
        }
    }

    #[must_use]
    pub const fn mini() -> Dims {
        Dims {
            layers: 2,
            heads: 2,
            head_dim: 128,
            audio_heads: 2,
            audio_head_dim: 64,
            channels: 128,
            cross_dim: 256,
            audio_cross_dim: 128,
            ff_mult: 4,
            caption: 16,
            conn_layers: 1,
        }
    }

    #[must_use]
    pub const fn dim(&self) -> u32 {
        self.heads * self.head_dim
    }

    #[must_use]
    pub const fn audio_dim(&self) -> u32 {
        self.audio_heads * self.audio_head_dim
    }

    #[must_use]
    pub const fn av_inner(&self) -> u32 {
        self.audio_heads * self.audio_head_dim
    }

    #[must_use]
    pub const fn text_in(&self) -> u32 {
        self.caption * TEXT_LAYERS
    }

    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    #[must_use]
    pub fn audio_sm_scale(&self) -> f32 {
        (self.audio_head_dim as f32).sqrt().recip()
    }

    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        let f = self.dim() / (2 * ROPE_AXES as u32);
        [2 * f, 2 * f, 2 * f, 0]
    }

    #[must_use]
    pub const fn audio_rope_dims(&self) -> [u32; 4] {
        [self.audio_dim(), 0, 0, 0]
    }

    #[must_use]
    pub const fn av_rope_dims(&self) -> [u32; 4] {
        [self.av_inner(), 0, 0, 0]
    }
}

pub struct Linear {
    pub w: Weight,
    pub bias: Option<Weight>,
}

impl Linear {
    fn at(name: &str, out: u32, in_: u32, banks: Dtype) -> Linear {
        Linear {
            w: Weight::sym(name, [u64::from(out), u64::from(in_)], banks),
            bias: Some(Weight::sym(
                format!("{name}.bias"),
                [u64::from(out)],
                crate::dense(banks),
            )),
        }
    }

    fn plain(name: &str, out: u32, in_: u32, banks: Dtype) -> Linear {
        Linear {
            w: Weight::sym(name, [u64::from(out), u64::from(in_)], banks),
            bias: None,
        }
    }

    fn packed(name: &str, seams: &[u32], in_: u32, banks: Dtype) -> Linear {
        let out: u64 = seams.iter().map(|&s| u64::from(s)).sum();
        let seams: Vec<u64> = seams.iter().map(|&s| u64::from(s)).collect();
        Linear {
            w: Weight::sym(name, [out, u64::from(in_)], banks).packed(seams.clone()),
            bias: Some(
                Weight::sym(format!("{name}.bias"), [out], crate::dense(banks)).packed(seams),
            ),
        }
    }
}

pub struct Attn {
    pub qkv: Linear,
    pub kv: Option<Linear>,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub gate: Linear,
    pub out: Linear,
    pub heads: u32,
    pub head_dim: u32,
}

impl Attn {
    fn own(prefix: &str, dim: u32, heads: u32, head_dim: u32, banks: Dtype) -> Attn {
        let inner = heads * head_dim;
        Attn {
            qkv: Linear::packed(&format!("{prefix}.qkv"), &[inner, inner, inner], dim, banks),
            kv: None,
            q_norm: gain(&format!("{prefix}.q_norm"), inner, banks),
            k_norm: gain(&format!("{prefix}.k_norm"), inner, banks),
            gate: Linear::at(&format!("{prefix}.gate"), heads, dim, banks),
            out: Linear::at(&format!("{prefix}.out"), dim, inner, banks),
            heads,
            head_dim,
        }
    }

    fn cross(prefix: &str, dim: u32, ctx: u32, heads: u32, head_dim: u32, banks: Dtype) -> Attn {
        let inner = heads * head_dim;
        Attn {
            qkv: Linear::at(&format!("{prefix}.q"), inner, dim, banks),
            kv: Some(Linear::packed(
                &format!("{prefix}.kv"),
                &[inner, inner],
                ctx,
                banks,
            )),
            q_norm: gain(&format!("{prefix}.q_norm"), inner, banks),
            k_norm: gain(&format!("{prefix}.k_norm"), inner, banks),
            gate: Linear::at(&format!("{prefix}.gate"), heads, dim, banks),
            out: Linear::at(&format!("{prefix}.out"), dim, inner, banks),
            heads,
            head_dim,
        }
    }

    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    #[must_use]
    pub const fn inner(&self) -> u32 {
        self.heads * self.head_dim
    }
}

fn gain(name: &str, width: u32, banks: Dtype) -> Weight {
    Weight::sym(name, [u64::from(width)], crate::dense(banks))
}

pub struct Ffn {
    pub up: Linear,
    pub down: Linear,
}

impl Ffn {
    fn at(prefix: &str, dim: u32, mult: u32, bias: bool, banks: Dtype) -> Ffn {
        let inner = dim * mult;
        let make = |name: String, out, in_| {
            if bias {
                Linear::at(&name, out, in_, banks)
            } else {
                Linear::plain(&name, out, in_, banks)
            }
        };
        Ffn {
            up: make(format!("{prefix}.up"), inner, dim),
            down: make(format!("{prefix}.down"), dim, inner),
        }
    }
}

pub struct Side {
    pub table: Weight,
    pub av_ss_table: Weight,
    pub av_gate_table: Weight,
    pub prompt_table: Weight,
    pub self_attn: Attn,
    pub cross: Attn,
    pub ffn: Ffn,
}

pub struct Block {
    pub video: Side,
    pub audio: Side,
    pub a2v: Attn,
    pub v2a: Attn,
}

impl Block {
    fn at(prefix: &str, d: &Dims, banks: Dtype) -> Block {
        let (dim, adim) = (d.dim(), d.audio_dim());
        let table = |name: String, slices: u32, width: u32| {
            Weight::sym(name, [u64::from(slices * width)], Dtype::F32)
        };
        let side =
            |stem: String, width: u32, heads: u32, head_dim: u32, ctx: u32, bias: bool| Side {
                table: table(format!("{stem}.table"), MOD_SLICES, width),
                av_ss_table: table(format!("{stem}.av_ss_table"), AV_SS_SLICES, width),
                av_gate_table: table(format!("{stem}.av_gate_table"), AV_GATE_SLICES, width),
                prompt_table: table(format!("{stem}.prompt_table"), PROMPT_SLICES, width),
                self_attn: Attn::own(&format!("{stem}.self"), width, heads, head_dim, banks),
                cross: Attn::cross(&format!("{stem}.cross"), width, ctx, heads, head_dim, banks),
                ffn: Ffn::at(&format!("{stem}.ffn"), width, d.ff_mult, bias, banks),
            };
        Block {
            video: side(
                format!("{prefix}.video"),
                dim,
                d.heads,
                d.head_dim,
                d.cross_dim,
                false,
            ),
            audio: side(
                format!("{prefix}.audio"),
                adim,
                d.audio_heads,
                d.audio_head_dim,
                d.audio_cross_dim,
                true,
            ),
            a2v: Attn::cross(
                &format!("{prefix}.a2v"),
                dim,
                adim,
                d.audio_heads,
                d.audio_head_dim,
                banks,
            ),
            v2a: Attn::cross(
                &format!("{prefix}.v2a"),
                adim,
                dim,
                d.audio_heads,
                d.audio_head_dim,
                banks,
            ),
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

pub struct AdaLn {
    pub embed: Embedder,
    pub proj: Linear,
    pub slices: u32,
}

impl AdaLn {
    fn at(prefix: &str, dim: u32, slices: u32, banks: Dtype) -> AdaLn {
        AdaLn {
            embed: Embedder::at(&format!("{prefix}.emb"), T_FREQ_DIM, dim, banks),
            proj: Linear::at(&format!("{prefix}.proj"), slices * dim, dim, banks),
            slices,
        }
    }
}

pub struct Stream {
    pub patchify: Linear,
    pub adaln: AdaLn,
    pub av_ss: AdaLn,
    pub av_gate: AdaLn,
    pub head_proj: Linear,
    pub head_table: Weight,
    pub proj_out: Linear,
}

impl Stream {
    fn at(prefix: &str, channels: u32, dim: u32, banks: Dtype) -> Stream {
        Stream {
            patchify: Linear::at(&format!("{prefix}.patchify"), dim, channels, banks),
            adaln: AdaLn::at(&format!("{prefix}.adaln"), dim, MOD_SLICES, banks),
            av_ss: AdaLn::at(&format!("{prefix}.av_ss"), dim, AV_SS_SLICES, banks),
            av_gate: AdaLn::at(&format!("{prefix}.av_gate"), dim, AV_GATE_SLICES, banks),
            head_proj: Linear::at(
                &format!("{prefix}.head_proj"),
                HEAD_SLICES * dim,
                dim,
                banks,
            ),
            head_table: Weight::sym(
                format!("{prefix}.head_table"),
                [u64::from(HEAD_SLICES * dim)],
                Dtype::F32,
            ),
            proj_out: Linear::at(&format!("{prefix}.proj_out"), channels, dim, banks),
        }
    }
}

pub struct Dit {
    pub video: Stream,
    pub audio: Stream,
    pub prompt: AdaLn,
    pub audio_prompt: AdaLn,
    pub blocks: Vec<Block>,
}

pub struct ConnBlock {
    pub attn: Attn,
    pub ffn: Ffn,
}

pub struct Connector {
    pub aggregate: Linear,
    pub blocks: Vec<ConnBlock>,
    pub dim: u32,
    pub heads: u32,
    pub head_dim: u32,
}

impl Connector {
    fn at(
        prefix: &str,
        text_in: u32,
        dim: u32,
        heads: u32,
        layers: u32,
        banks: Dtype,
    ) -> Connector {
        let head_dim = dim / heads;
        Connector {
            aggregate: Linear::at(&format!("{prefix}.aggregate"), dim, text_in, banks),
            blocks: (0..layers)
                .map(|l| ConnBlock {
                    attn: Attn::own(
                        &format!("{prefix}.block.{l}.attn"),
                        dim,
                        heads,
                        head_dim,
                        banks,
                    ),
                    ffn: Ffn::at(
                        &format!("{prefix}.block.{l}.ffn"),
                        dim,
                        CONN_FF_MULT,
                        true,
                        banks,
                    ),
                })
                .collect(),
            dim,
            heads,
            head_dim,
        }
    }

    #[must_use]
    pub fn sm_scale(&self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    #[must_use]
    pub const fn rope_dims(&self) -> [u32; 4] {
        [self.dim, 0, 0, 0]
    }

    #[must_use]
    pub fn rescale(&self, caption: u32) -> f32 {
        (f64::from(self.dim) / f64::from(caption)).sqrt() as f32
    }
}

pub struct VaeConv {
    pub w: Weight,
    pub bias: Weight,
    pub c_in: u32,
    pub c_out: u32,
}

impl VaeConv {
    fn at(name: &str, c_out: u32, c_in: u32, banks: Dtype) -> VaeConv {
        let taps = 27;
        VaeConv {
            w: Weight::sym(
                name,
                [u64::from(c_out), u64::from(c_in) * u64::from(taps)],
                banks,
            )
            .conv_taps_major(c_in, taps),
            bias: Weight::sym(format!("{name}.bias"), [u64::from(c_out)], Dtype::F32),
            c_in,
            c_out,
        }
    }
}

pub struct VaeResnet {
    pub conv1: VaeConv,
    pub conv2: VaeConv,
}

impl VaeResnet {
    fn at(prefix: &str, c: u32, banks: Dtype) -> VaeResnet {
        VaeResnet {
            conv1: VaeConv::at(&format!("{prefix}.conv1"), c, c, banks),
            conv2: VaeConv::at(&format!("{prefix}.conv2"), c, c, banks),
        }
    }
}

pub struct VaeUpBlock {
    pub upsampler: VaeConv,
    pub stride: [u32; 3],
    pub resnets: Vec<VaeResnet>,
}

pub struct Vae {
    pub latents_mean: Weight,
    pub latents_std: Weight,
    pub zero: Weight,
    pub conv_in: VaeConv,
    pub mid: Vec<VaeResnet>,
    pub up: Vec<VaeUpBlock>,
    pub conv_out: VaeConv,
}

impl Vae {
    fn ltx_2_5(banks: Dtype) -> Vae {
        let dims = VAE_DECODER_DIMS;
        let top = dims[0];
        let up = (0..4)
            .map(|i| {
                let (c_in, c_out) = (dims[i], dims[i + 1]);
                let stride = VAE_UP_STRIDES[i];
                let prefix = format!("vae.up.{i}");
                VaeUpBlock {
                    upsampler: VaeConv::at(
                        &format!("{prefix}.upsampler"),
                        c_out * stride[0] * stride[1] * stride[2],
                        c_in,
                        banks,
                    ),
                    stride,
                    resnets: (0..VAE_UP_RESNETS[i])
                        .map(|r| VaeResnet::at(&format!("{prefix}.res.{r}"), c_out, banks))
                        .collect(),
                }
            })
            .collect();
        let last = dims[4];
        let dense = crate::dense(banks);
        let row = |name: &str| Weight::sym(name, [u64::from(VAE_Z)], dense);
        Vae {
            latents_mean: row("vae.latents_mean"),
            latents_std: row("vae.latents_std"),
            zero: row("vae.zero"),
            conv_in: VaeConv::at("vae.conv_in", top, VAE_Z, banks),
            mid: (0..VAE_MID_RESNETS)
                .map(|r| VaeResnet::at(&format!("vae.mid.res.{r}"), top, banks))
                .collect(),
            up,
            conv_out: VaeConv::at("vae.conv_out", VAE_RGB * VAE_PATCH * VAE_PATCH, last, banks),
        }
    }

    pub fn convs(&self) -> impl Iterator<Item = &VaeConv> + '_ {
        let mut out: Vec<&VaeConv> = vec![&self.conv_in];
        for r in &self.mid {
            out.push(&r.conv1);
            out.push(&r.conv2);
        }
        for block in &self.up {
            out.push(&block.upsampler);
            for r in &block.resnets {
                out.push(&r.conv1);
                out.push(&r.conv2);
            }
        }
        out.push(&self.conv_out);
        out.into_iter()
    }
}

pub struct Model {
    pub banks: Dtype,
    pub dims: Dims,
    pub dit: Dit,
    pub connectors: (Connector, Connector),
    pub vae: Option<Vae>,
}

impl Model {
    #[must_use]
    pub fn ltx_2_5(banks: Dtype) -> Model {
        Model::new(banks, Dims::ltx_2_5(), Some(Vae::ltx_2_5(Dtype::Bf16)))
    }

    #[must_use]
    pub fn mini(banks: Dtype) -> Model {
        Model::new(banks, Dims::mini(), None)
    }

    fn new(banks: Dtype, d: Dims, vae: Option<Vae>) -> Model {
        assert_eq!(
            d.rope_dims().iter().sum::<u32>() + 2 * rope_pad(d.dim(), ROPE_AXES),
            d.dim(),
            "the video ladder and its identity pad cover the row"
        );
        assert_eq!(
            rope_pad(d.audio_dim(), AUDIO_ROPE_AXES),
            0,
            "a one-axis ladder pads nothing"
        );
        assert_eq!(
            d.av_inner(),
            d.audio_cross_dim,
            "the cross-modal rope is built at `audio_cross_attention_dim`, which is the \
             pair's inner width"
        );
        let dit = Dit {
            video: Stream::at("dit.video", d.channels, d.dim(), banks),
            audio: Stream::at("dit.audio", d.channels, d.audio_dim(), banks),
            prompt: AdaLn::at("dit.prompt", d.dim(), PROMPT_SLICES, banks),
            audio_prompt: AdaLn::at("dit.audio_prompt", d.audio_dim(), PROMPT_SLICES, banks),
            blocks: (0..d.layers)
                .map(|i| Block::at(&format!("dit.block.{i}"), &d, banks))
                .collect(),
        };
        if vae.is_some() {
            assert_eq!(d.channels, VAE_Z, "the latent is the VAE's");
        }
        Model {
            banks,
            dims: d,
            dit,
            vae,
            connectors: (
                Connector::at(
                    "connectors.video",
                    d.text_in(),
                    d.cross_dim,
                    d.heads,
                    d.conn_layers,
                    banks,
                ),
                Connector::at(
                    "connectors.audio",
                    d.text_in(),
                    d.audio_cross_dim,
                    d.audio_heads,
                    d.conn_layers,
                    banks,
                ),
            ),
        }
    }
}

#[must_use]
pub const fn rope_pad(dim: u32, axes: u8) -> u32 {
    let axes = axes as u32;
    dim / 2 - axes * (dim / (2 * axes))
}
