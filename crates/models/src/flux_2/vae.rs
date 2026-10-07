use poem_dsl::ops::spatial::{self, Conv};
use poem_dsl::{Dtype, Input, Value, Weight, ops, seam};

use super::forward::Facts;
use super::model::{IN_CHANNELS, Linear, PACK, VAE_CHANNELS, port};

pub const BLOCK_CHANNELS: [u32; 4] = [128, 256, 512, 512];
pub const GN_GROUPS: u32 = 32;
pub const GN_EPS: f32 = 1e-6;
pub const BN_EPS: f32 = 1e-4;
pub const LAYERS_PER_BLOCK: u32 = 2;
pub const DECODER_RESNETS: u32 = LAYERS_PER_BLOCK + 1;
pub const RGB: u32 = 3;
const TAPS3: u32 = 9;
pub const POSTERIOR_STORED: u32 = 2 * VAE_CHANNELS;

pub struct ConvW {
    pub w: Weight,
    pub bias: Weight,
    pub c_in: u32,
    pub c_out: u32,
    pub taps: u32,
}

impl ConvW {
    fn at(name: &str, c_out: u32, c_in: u32, taps: u32, banks: Dtype) -> ConvW {
        ConvW {
            w: Weight::sym(
                name,
                [u64::from(c_out), u64::from(c_in) * u64::from(taps)],
                banks,
            )
            .conv_taps_major(c_in, taps),
            bias: Weight::sym(format!("{name}.bias"), [u64::from(c_out)], Dtype::F32),
            c_in,
            c_out,
            taps,
        }
    }
}

pub struct Norm {
    pub weight: Weight,
    pub bias: Weight,
}

impl Norm {
    fn at(name: &str, c: u32) -> Norm {
        Norm {
            weight: Weight::sym(format!("{name}.weight"), [u64::from(c)], Dtype::F32),
            bias: Weight::sym(format!("{name}.bias"), [u64::from(c)], Dtype::F32),
        }
    }
}

pub struct Proj {
    pub w: Linear,
    pub bias: Weight,
}

impl Proj {
    fn at(name: &str, c: u32, banks: Dtype, dense: Dtype) -> Proj {
        let c = u64::from(c);
        Proj {
            w: Weight::sym(name.to_string(), [c, c], banks),
            bias: Weight::sym(format!("{name}.bias"), [c], dense),
        }
    }
}

pub struct ResBlock {
    pub norm1: Norm,
    pub conv1: ConvW,
    pub norm2: Norm,
    pub conv2: ConvW,
    pub shortcut: Option<ConvW>,
}

impl ResBlock {
    fn at(name: &str, c_in: u32, c_out: u32, banks: Dtype) -> ResBlock {
        ResBlock {
            norm1: Norm::at(&format!("{name}.norm1"), c_in),
            conv1: ConvW::at(&format!("{name}.conv1"), c_out, c_in, TAPS3, banks),
            norm2: Norm::at(&format!("{name}.norm2"), c_out),
            conv2: ConvW::at(&format!("{name}.conv2"), c_out, c_out, TAPS3, banks),
            shortcut: (c_in != c_out)
                .then(|| ConvW::at(&format!("{name}.shortcut"), c_out, c_in, 1, banks)),
        }
    }
}

pub struct AttnBlock {
    pub norm: Norm,
    pub q: Proj,
    pub k: Proj,
    pub v: Proj,
    pub out: Proj,
    pub width: u32,
}

impl AttnBlock {
    fn at(name: &str, c: u32, banks: Dtype, dense: Dtype) -> AttnBlock {
        AttnBlock {
            norm: Norm::at(&format!("{name}.norm"), c),
            q: Proj::at(&format!("{name}.q"), c, banks, dense),
            k: Proj::at(&format!("{name}.k"), c, banks, dense),
            v: Proj::at(&format!("{name}.v"), c, banks, dense),
            out: Proj::at(&format!("{name}.out"), c, banks, dense),
            width: c,
        }
    }
}

pub struct Mid {
    pub res0: ResBlock,
    pub attn: AttnBlock,
    pub res1: ResBlock,
}

impl Mid {
    fn at(name: &str, c: u32, banks: Dtype, dense: Dtype) -> Mid {
        Mid {
            res0: ResBlock::at(&format!("{name}.res0"), c, c, banks),
            attn: AttnBlock::at(&format!("{name}.attn"), c, banks, dense),
            res1: ResBlock::at(&format!("{name}.res1"), c, c, banks),
        }
    }
}

pub struct UpBlock {
    pub resnets: Vec<ResBlock>,
    pub upsample: Option<ConvW>,
}

pub struct DownBlock {
    pub resnets: Vec<ResBlock>,
    pub downsample: Option<ConvW>,
}

pub struct Decoder {
    pub conv_in: ConvW,
    pub mid: Mid,
    pub up: Vec<UpBlock>,
    pub norm_out: Norm,
    pub conv_out: ConvW,
}

pub struct Encoder {
    pub conv_in: ConvW,
    pub down: Vec<DownBlock>,
    pub mid: Mid,
    pub norm_out: Norm,
    pub conv_out: ConvW,
}

pub struct Vae {
    pub bn_zero: Weight,
    pub bn_scale: Weight,
    pub bn_rscale: Weight,
    pub bn_mean: Weight,
    pub quant_conv: ConvW,
    pub post_quant_conv: ConvW,
    pub decoder: Decoder,
    pub encoder: Encoder,
}

impl Vae {
    #[must_use]
    pub fn flux2(banks: Dtype) -> Vae {
        let dense = crate::dense(banks);
        let top = BLOCK_CHANNELS[BLOCK_CHANNELS.len() - 1];
        let mut up = Vec::new();
        let mut c_prev = top;
        for (i, &c) in BLOCK_CHANNELS.iter().rev().enumerate() {
            let name = format!("vae.dec.up{i}");
            let resnets = (0..DECODER_RESNETS)
                .map(|r| {
                    let block = ResBlock::at(&format!("{name}.res{r}"), c_prev, c, banks);
                    c_prev = c;
                    block
                })
                .collect();
            let last = i + 1 == BLOCK_CHANNELS.len();
            up.push(UpBlock {
                resnets,
                upsample: (!last)
                    .then(|| ConvW::at(&format!("{name}.upsample"), c, c, TAPS3, banks)),
            });
        }
        let mut down = Vec::new();
        let mut c_prev = BLOCK_CHANNELS[0];
        for (i, &c) in BLOCK_CHANNELS.iter().enumerate() {
            let name = format!("vae.enc.down{i}");
            let resnets = (0..LAYERS_PER_BLOCK)
                .map(|r| {
                    let block = ResBlock::at(&format!("{name}.res{r}"), c_prev, c, banks);
                    c_prev = c;
                    block
                })
                .collect();
            let last = i + 1 == BLOCK_CHANNELS.len();
            down.push(DownBlock {
                resnets,
                downsample: (!last)
                    .then(|| ConvW::at(&format!("{name}.downsample"), c, c, TAPS3, banks)),
            });
        }
        let bn =
            |tail: &str| Weight::sym(format!("vae.bn.{tail}"), [u64::from(IN_CHANNELS)], dense);
        Vae {
            bn_zero: bn("zero"),
            bn_scale: bn("scale"),
            bn_rscale: bn("rscale"),
            bn_mean: bn("mean"),
            quant_conv: ConvW::at("vae.quant", VAE_CHANNELS, POSTERIOR_STORED, 1, banks),
            post_quant_conv: ConvW::at("vae.post_quant", VAE_CHANNELS, VAE_CHANNELS, 1, banks),
            decoder: Decoder {
                conv_in: ConvW::at("vae.dec.conv_in", top, VAE_CHANNELS, TAPS3, banks),
                mid: Mid::at("vae.dec.mid", top, banks, dense),
                up,
                norm_out: Norm::at("vae.dec.norm_out", BLOCK_CHANNELS[0]),
                conv_out: ConvW::at("vae.dec.conv_out", RGB, BLOCK_CHANNELS[0], TAPS3, banks),
            },
            encoder: Encoder {
                conv_in: ConvW::at("vae.enc.conv_in", BLOCK_CHANNELS[0], RGB, TAPS3, banks),
                down,
                mid: Mid::at("vae.enc.mid", top, banks, dense),
                norm_out: Norm::at("vae.enc.norm_out", top),
                conv_out: ConvW::at("vae.enc.conv_out", POSTERIOR_STORED, top, TAPS3, banks),
            },
        }
    }
}

pub fn decode(arm: &Input<Facts>, vae: &Vae) -> Value {
    let d = &vae.decoder;
    let g0 = arm.grid();
    let z = arm.voxels(port::VOXELS, IN_CHANNELS, Dtype::Bf16);
    let (z, g0) = spatial::upsample_nearest(&z, &g0, [1, 1, 1], false);
    let z = ops::elemwise::standardize(&z, &vae.bn_zero, &vae.bn_scale);
    let z = ops::elemwise::add_bias(&vae.bn_mean, &z);
    let (z, mut grid) = spatial::pixel_shuffle(&z, &g0, [1, PACK, PACK]);
    let z = conv(&z, &grid, &vae.post_quant_conv);
    let mut h = conv(&z, &grid, &d.conv_in);
    h = mid(&h, &grid, &d.mid);
    for block in &d.up {
        for res in &block.resnets {
            h = resnet(&h, &grid, res);
        }
        if let Some(c) = &block.upsample {
            let (up, up_grid) = spatial::upsample_nearest(&h, &grid, [1, 2, 2], false);
            grid = up_grid;
            h = conv(&up, &grid, c);
        }
    }
    let h = group_norm(&h, &grid, &d.norm_out, true);
    let y = conv(&h, &grid, &d.conv_out);
    seam::at(seam::PIXELS, &[&y, &grid]);
    y
}

pub fn encode(arm: &Input<Facts>, vae: &Vae) -> Value {
    let e = &vae.encoder;
    let mut grid = arm.grid();
    let x = arm.voxels(port::PIXEL_VOXELS, RGB, Dtype::Bf16);
    let mut h = conv(&x, &grid, &e.conv_in);
    for block in &e.down {
        for res in &block.resnets {
            h = resnet(&h, &grid, res);
        }
        if let Some(c) = &block.downsample {
            let (y, y_grid) = spatial::conv3d(
                &h,
                &grid,
                &c.w,
                Some(&c.bias),
                Conv::conv2d([3, 3], [2, 2], [0, 0]).pad_back([0, 1, 1]),
                None,
            );
            h = y;
            grid = y_grid;
        }
    }
    h = mid(&h, &grid, &e.mid);
    let h = group_norm(&h, &grid, &e.norm_out, true);
    let h = conv(&h, &grid, &e.conv_out);
    let mean = conv(&h, &grid, &vae.quant_conv);
    let (packed, grid) = spatial::pixel_unshuffle(&mean, &grid, [1, PACK, PACK]);
    let z = ops::elemwise::standardize(&packed, &vae.bn_mean, &vae.bn_rscale);
    seam::at(seam::PIXELS, &[&z, &grid]);
    z
}

fn conv(x: &Value, grid: &Value, c: &ConvW) -> Value {
    let shape = match c.taps {
        1 => Conv::conv2d([1, 1], [1, 1], [0, 0]),
        _ => Conv::conv2d([3, 3], [1, 1], [1, 1]),
    };
    spatial::conv3d(x, grid, &c.w, Some(&c.bias), shape, None).0
}

fn group_norm(x: &Value, grid: &Value, norm: &Norm, silu: bool) -> Value {
    spatial::group_norm(x, grid, GN_GROUPS, &norm.weight, &norm.bias, GN_EPS, silu)
}

fn resnet(x: &Value, grid: &Value, r: &ResBlock) -> Value {
    let h = group_norm(x, grid, &r.norm1, true);
    let h = conv(&h, grid, &r.conv1);
    let h = group_norm(&h, grid, &r.norm2, true);
    let h = conv(&h, grid, &r.conv2);
    let skip = match &r.shortcut {
        Some(c) => conv(x, grid, c),
        None => x.clone(),
    };
    ops::elemwise::add(&skip, &h)
}

fn mid(x: &Value, grid: &Value, m: &Mid) -> Value {
    let h = resnet(x, grid, &m.res0);
    let h = attention(&h, grid, &m.attn);
    resnet(&h, grid, &m.res1)
}

fn attention(x: &Value, grid: &Value, a: &AttnBlock) -> Value {
    let h = group_norm(x, grid, &a.norm, false);
    let linear =
        |p: &Proj, x: &Value| ops::elemwise::add_bias(&p.bias, &ops::linear::matmul(x, &p.w));
    let q = linear(&a.q, &h);
    let k = linear(&a.k, &h);
    let v = linear(&a.v, &h);
    let o = spatial::attention(&q, &k, &v, grid, (a.width as f32).sqrt().recip());
    let o = linear(&a.out, &o);
    ops::elemwise::add(x, &o)
}
