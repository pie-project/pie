//! The partition: the plan a shape makes under a preset's keys (a pinned plan's or the search's:
//! [`super::pins`], [`super::search`]), by the layout its layers take ([`Layout`]). Each layout
//! reads its own keys (the snake's unit slots, the stacked columns' arrangement, the row runs' dims
//! and head groups, the split's attention / MoE columns) and refuses the shapes its cut and
//! kernels cannot take by name. Nothing physical is decided here.

use super::kernels::KernelSet;
use super::plan::{LayerKind, Plan, Preset, Shape};

pub mod runs;
pub mod snake;
pub mod split;
pub mod stacked;

type R<T> = Result<T, String>;

/// The plan of `shape` under `preset`: the partition and contract the preset's keys make for the
/// model, by its layout.
pub fn plan(shape: &Shape, preset: &Preset) -> R<Plan> {
    match KernelSet::of_shape(shape) {
        KernelSet::Own => snake::plan(shape, preset),
        KernelSet::Shared => stacked::plan(shape, preset),
        KernelSet::Windows => runs::plan(shape, preset),
        KernelSet::Experts => split::plan(shape, preset),
    }
}

/// Layers `layers` in units of `k`, every unit the same kinds; the unit count.
pub fn units_of(layers: &[LayerKind], k: u32, layout: &str) -> R<u32> {
    let n = layers.len() as u32;
    if k == 0 || !n.is_multiple_of(k) {
        return Err(format!("{layout}: {n} layers in units of {k}"));
    }
    Ok(n / k)
}

/// `n / d` rounded up to a multiple of `m`.
pub fn ceil_to(n: u32, d: u32, m: u32) -> u32 {
    n.div_ceil(d).div_ceil(m) * m
}
