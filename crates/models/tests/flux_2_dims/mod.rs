//! The widths, constants and schedule FLUX.2's package declares, which the
//! tests of its rows, its import and its VAE hold the package to.
#![allow(dead_code)]

pub const IN_CHANNELS: u32 = 128;
pub const VAE_CHANNELS: u32 = 32;
pub const PACK: u32 = 2;
pub const VAE_COMPRESSION: u32 = 8;
pub const TOKEN_COMPRESSION: u32 = VAE_COMPRESSION * PACK;

pub const HEAD_DIM: u32 = 128;
pub const ROPE_DIMS: [u32; 4] = [32, 32, 32, 32];
pub const ROPE_THETA: f32 = 2000.0;
pub const ROPE_AXES: u8 = 4;

pub const T_FREQ_DIM: u32 = 256;
pub const T_MAX_PERIOD: f32 = 10_000.0;
pub const T_FLIP_SIN_COS: bool = true;
pub const T_SCALE: f32 = 1.0;
pub const GUIDANCE_SCALE: f32 = 1000.0;

pub const MLP_RATIO: u32 = 3;
pub const TRAIN_STEPS: u32 = 1000;

pub const TE_HIDDEN: u32 = 2560;
pub const TE_HEAD_DIM: u32 = 128;
pub const TE_LAYERS: u32 = 27;
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

pub mod vae {
    pub const GN_GROUPS: u32 = 32;
    pub const GN_EPS: f32 = 1e-6;
    pub const RGB: u32 = 3;
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

/// The reading each of a row's passes runs, by name, as its generative facts
/// number them: (text, denoise, vae.decode, vae.encode).
#[must_use]
pub fn readings(row: &str) -> (Option<u8>, u8, Option<u8>, Option<u8>) {
    let facts = models::deployment(row)
        .and_then(|d| d.generative.clone())
        .unwrap_or_else(|| panic!("`{row}` is a generative row"));
    let index = |name: &str| {
        facts
            .readings
            .iter()
            .find(|r| r.name == name)
            .map(|r| r.index)
    };
    (
        index("text"),
        index("denoise").expect("a denoise reading"),
        index("vae.decode"),
        index("vae.encode"),
    )
}

/// The flow schedule's shift for `image_rows` and `steps`, as the reference
/// pipeline computes it.
#[must_use]
pub fn empirical_mu(image_rows: u32, steps: u32) -> f32 {
    const A1: f64 = 8.738_095_24e-5;
    const B1: f64 = 1.898_333_33;
    const A2: f64 = 0.000_169_27;
    const B2: f64 = 0.456_666_66;
    let rows = f64::from(image_rows);
    if image_rows > 4300 {
        return (A2 * rows + B2) as f32;
    }
    let m_200 = A2 * rows + B2;
    let m_10 = A1 * rows + B1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    (a * f64::from(steps) + b) as f32
}

/// The reference pipeline's sigmas for `image_rows` and `steps`.
#[must_use]
pub fn sigmas(image_rows: u32, steps: u32) -> Vec<f32> {
    let steps = steps.max(1);
    let shift = f64::from(empirical_mu(image_rows, steps)).exp();
    (0..steps)
        .map(|i| {
            let n = f64::from(steps);
            let sigma = 1.0 - f64::from(i) * (1.0 - 1.0 / n) / (n - 1.0).max(1.0);
            (shift / (shift + 1.0 / sigma - 1.0)) as f32
        })
        .collect()
}
