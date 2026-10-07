use poem_dsl::ops::spatial::{self, Conv};
use poem_dsl::{Dtype, Input, Value, Weight, ops, seam};

use super::model::{CHANNELS, Linear, port};

pub const SCALING_FACTOR: f32 = 0.3611;
pub const SHIFT_FACTOR: f32 = 0.1159;
pub const GN_GROUPS: u32 = 32;
pub const GN_EPS: f32 = 1e-6;
pub const BLOCK_CHANNELS: [u32; 4] = [128, 256, 512, 512];
pub const LAYERS_PER_BLOCK: u32 = 2;
pub const RGB: u32 = 3;
const TAPS3: u32 = 9;

pub struct ConvW {
    pub w: Weight,
    pub bias: Weight,
    pub c_in: u32,
    pub c_out: u32,
    pub taps: u32,
}

impl ConvW {
    fn at(name: &str, c_out: u32, c_in: u32, taps: u32) -> ConvW {
        ConvW {
            w: Weight::sym(
                name,
                [u64::from(c_out), u64::from(c_in) * u64::from(taps)],
                Dtype::Bf16,
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

pub struct ResBlock {
    pub norm1: Norm,
    pub conv1: ConvW,
    pub norm2: Norm,
    pub conv2: ConvW,
    pub shortcut: Option<ConvW>,
}

impl ResBlock {
    fn at(name: &str, c_in: u32, c_out: u32) -> ResBlock {
        ResBlock {
            norm1: Norm::at(&format!("{name}.norm1"), c_in),
            conv1: ConvW::at(&format!("{name}.conv1"), c_out, c_in, TAPS3),
            norm2: Norm::at(&format!("{name}.norm2"), c_out),
            conv2: ConvW::at(&format!("{name}.conv2"), c_out, c_out, TAPS3),
            shortcut: (c_in != c_out)
                .then(|| ConvW::at(&format!("{name}.shortcut"), c_out, c_in, 1)),
        }
    }
}

pub struct AttnBlock {
    pub norm: Norm,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    pub width: u32,
}

impl AttnBlock {
    fn at(name: &str, c: u32, banks: Dtype) -> AttnBlock {
        AttnBlock {
            norm: Norm::at(&format!("{name}.norm"), c),
            q: Linear::at(&format!("{name}.q"), c, c, banks),
            k: Linear::at(&format!("{name}.k"), c, c, banks),
            v: Linear::at(&format!("{name}.v"), c, c, banks),
            out: Linear::at(&format!("{name}.out"), c, c, banks),
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
    fn at(name: &str, c: u32, banks: Dtype) -> Mid {
        Mid {
            res0: ResBlock::at(&format!("{name}.res0"), c, c),
            attn: AttnBlock::at(&format!("{name}.attn"), c, banks),
            res1: ResBlock::at(&format!("{name}.res1"), c, c),
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
    pub shift: Weight,
    pub encoder_out_stored: u32,
    pub decoder: Decoder,
    pub encoder: Encoder,
}

impl Vae {
    #[must_use]
    pub fn flux(banks: Dtype) -> Vae {
        let top = BLOCK_CHANNELS[BLOCK_CHANNELS.len() - 1];
        let mut up = Vec::new();
        let mut c_prev = top;
        for (i, &c) in BLOCK_CHANNELS.iter().rev().enumerate() {
            let name = format!("vae.dec.up{i}");
            let resnets = (0..=LAYERS_PER_BLOCK)
                .map(|r| {
                    let block = ResBlock::at(&format!("{name}.res{r}"), c_prev, c);
                    c_prev = c;
                    block
                })
                .collect();
            let last = i + 1 == BLOCK_CHANNELS.len();
            up.push(UpBlock {
                resnets,
                upsample: (!last).then(|| ConvW::at(&format!("{name}.upsample"), c, c, TAPS3)),
            });
        }
        let mut down = Vec::new();
        let mut c_prev = BLOCK_CHANNELS[0];
        for (i, &c) in BLOCK_CHANNELS.iter().enumerate() {
            let name = format!("vae.enc.down{i}");
            let resnets = (0..LAYERS_PER_BLOCK)
                .map(|r| {
                    let block = ResBlock::at(&format!("{name}.res{r}"), c_prev, c);
                    c_prev = c;
                    block
                })
                .collect();
            let last = i + 1 == BLOCK_CHANNELS.len();
            down.push(DownBlock {
                resnets,
                downsample: (!last).then(|| ConvW::at(&format!("{name}.downsample"), c, c, TAPS3)),
            });
        }
        Vae {
            shift: Weight::sym("vae.shift", [u64::from(CHANNELS)], crate::dense(banks)),
            encoder_out_stored: 2 * CHANNELS,
            decoder: Decoder {
                conv_in: ConvW::at("vae.dec.conv_in", top, CHANNELS, TAPS3),
                mid: Mid::at("vae.dec.mid", top, banks),
                up,
                norm_out: Norm::at("vae.dec.norm_out", BLOCK_CHANNELS[0]),
                conv_out: ConvW::at("vae.dec.conv_out", RGB, BLOCK_CHANNELS[0], TAPS3),
            },
            encoder: Encoder {
                conv_in: ConvW::at("vae.enc.conv_in", BLOCK_CHANNELS[0], RGB, TAPS3),
                down,
                mid: Mid::at("vae.enc.mid", top, banks),
                norm_out: Norm::at("vae.enc.norm_out", top),
                conv_out: ConvW::at("vae.enc.conv_out", CHANNELS, top, TAPS3),
            },
        }
    }
}

pub fn decode(arm: &Input, vae: &Vae) -> Value {
    let d = &vae.decoder;
    let mut grid = arm.grid();
    let z = arm.voxels(port::VOXELS, CHANNELS, Dtype::Bf16);
    let z = ops::elemwise::add(&z, &z);
    let z = ops::elemwise::add_bias(
        &vae.shift,
        &ops::elemwise::mul_scalar(0.5 * SCALING_FACTOR.recip(), &z),
    );
    let mut h = conv3(&z, &grid, &d.conv_in);
    h = mid(&h, &grid, &d.mid);
    for block in &d.up {
        for res in &block.resnets {
            h = resnet(&h, &grid, res);
        }
        if let Some(conv) = &block.upsample {
            let (up, up_grid) = spatial::upsample_nearest(&h, &grid, [1, 2, 2], false);
            grid = up_grid;
            h = conv3(&up, &grid, conv);
        }
    }
    let h = group_norm(&h, &grid, &d.norm_out, true);
    let y = conv3(&h, &grid, &d.conv_out);
    seam::at(seam::PIXELS, &[&y, &grid]);
    y
}

pub fn encode(arm: &Input, vae: &Vae) -> Value {
    let e = &vae.encoder;
    let mut grid = arm.grid();
    let x = arm.voxels(port::PIXEL_VOXELS, RGB, Dtype::Bf16);
    let mut h = conv3(&x, &grid, &e.conv_in);
    for block in &e.down {
        for res in &block.resnets {
            h = resnet(&h, &grid, res);
        }
        if let Some(conv) = &block.downsample {
            let (y, y_grid) = spatial::conv3d(
                &h,
                &grid,
                &conv.w,
                Some(&conv.bias),
                Conv::conv2d([3, 3], [2, 2], [0, 0]).pad_back([0, 1, 1]),
                None,
            );
            h = y;
            grid = y_grid;
        }
    }
    h = mid(&h, &grid, &e.mid);
    let h = group_norm(&h, &grid, &e.norm_out, true);
    let mean = conv3(&h, &grid, &e.conv_out);
    seam::at(seam::PIXELS, &[&mean, &grid]);
    mean
}

fn conv3(x: &Value, grid: &Value, conv: &ConvW) -> Value {
    let shape = match conv.taps {
        1 => Conv::conv2d([1, 1], [1, 1], [0, 0]),
        _ => Conv::conv2d([3, 3], [1, 1], [1, 1]),
    };
    spatial::conv3d(x, grid, &conv.w, Some(&conv.bias), shape, None).0
}

fn group_norm(x: &Value, grid: &Value, norm: &Norm, silu: bool) -> Value {
    spatial::group_norm(x, grid, GN_GROUPS, &norm.weight, &norm.bias, GN_EPS, silu)
}

fn resnet(x: &Value, grid: &Value, r: &ResBlock) -> Value {
    let h = group_norm(x, grid, &r.norm1, true);
    let h = conv3(&h, grid, &r.conv1);
    let h = group_norm(&h, grid, &r.norm2, true);
    let h = conv3(&h, grid, &r.conv2);
    let skip = match &r.shortcut {
        Some(conv) => conv3(x, grid, conv),
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
        |w: &Linear, x: &Value| ops::elemwise::add_bias(&w.bias, &ops::linear::matmul(x, &w.w));
    let q = linear(&a.q, &h);
    let k = linear(&a.k, &h);
    let v = linear(&a.v, &h);
    let o = spatial::attention(&q, &k, &v, grid, (a.width as f32).sqrt().recip());
    let o = linear(&a.out, &o);
    ops::elemwise::add(x, &o)
}
