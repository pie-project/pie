//! Program families: the kernels a model's program runs. A family turns the model's shape and a
//! preset (a pinned plan's keys, or the search's: [`super::pins`]) into the plan, states what its kernels read off each block the composer
//! places ([`super::compose`]), and builds each kind's kernel from the plan. Everything else (the floor, the wiring, the fabric, the emitters, the
//! placer, the host) is generic over the blocks. Which family serves a model follows from what
//! its shape needs ([`serving`]), never from its name.

use super::abi::Kernels;
use super::compose::Layout;
use super::plan::{FamilyId, LayerKind, Plan, Preset, Shape};
use super::program::Program;
use super::tiles::spec::Spec;

pub mod dense;
pub mod gemma;
pub mod gemma12;
pub mod gptoss;

type R<T> = Result<T, String>;

/// A program family. Every method is pure: same inputs, same outputs.
pub trait Family: Send + Sync {
    /// The plan of `shape` under `preset`: the partition and contract the preset's choices make
    /// for the model. Nothing physical is decided here.
    fn plan(&self, shape: &Shape, preset: &Preset) -> R<Plan>;

    /// Each kind's kernel and its params (levers applied by [`super::emit::params::kernels`]).
    fn kernels(&self, prog: &Program) -> R<Kernels>;

    /// Every name the kernels export, with its element type, in export order: the ABI between
    /// the kernels and the host's memcpy.
    fn symbols(&self) -> &'static [(&'static str, &'static str)];

    /// The cut of the model's params into the tiles the plan's bindings place, as data
    /// ([`super::tiles::spec`]); `shapes` gives a param's shape where the trace holds it.
    fn tiles(&self, plan: &Plan, shapes: &dyn Fn(&str) -> Option<Vec<u64>>) -> R<Spec>;
}

/// A family by its id.
pub fn of(id: FamilyId) -> Box<dyn Family> {
    match id {
        FamilyId::Dense => Box::new(dense::Dense),
        FamilyId::GptOss => Box::new(gptoss::GptOss),
        FamilyId::Gemma => Box::new(gemma::Gemma),
        FamilyId::Gemma12 => Box::new(gemma12::Gemma12),
    }
}

/// The family of a layout's kernels.
pub fn of_layout(layout: Layout) -> FamilyId {
    match layout {
        Layout::Snake => FamilyId::Dense,
        Layout::Stacked => FamilyId::Gemma,
        Layout::Runs => FamilyId::Gemma12,
        Layout::Split => FamilyId::GptOss,
    }
}

/// The family whose blocks `shape` needs: routed experts take the MoE columns and the band head
/// (gpt-oss); per-layer inputs or layers over another's KV take the PLE storage and the shared-KV
/// line (Gemma 4 E2B / E4B); sliding layers otherwise take the attention / plain row runs and the
/// grouped head (Gemma 4 12B); the rest (GDN, gated or plain attention) the snake of units
/// (Qwen).
pub fn serving(shape: &Shape) -> FamilyId {
    let any = |f: fn(&LayerKind) -> bool| shape.layers.iter().any(f);
    if shape.moe.is_some() {
        FamilyId::GptOss
    } else if shape.ple.is_some() || any(|k| matches!(k, LayerKind::SharedKv { .. })) {
        FamilyId::Gemma
    } else if any(|k| matches!(k, LayerKind::Sliding { .. })) {
        FamilyId::Gemma12
    } else {
        FamilyId::Dense
    }
}

/// Layers `layers` in units of `k`, every unit the same kinds; the unit count.
pub fn units_of(layers: &[LayerKind], k: u32, family: &str) -> R<u32> {
    let n = layers.len() as u32;
    if k == 0 || !n.is_multiple_of(k) {
        return Err(format!("{family}: {n} layers in units of {k}"));
    }
    Ok(n / k)
}

/// `n / d` rounded up to a multiple of `m`.
pub fn ceil_to(n: u32, d: u32, m: u32) -> u32 {
    n.div_ceil(d).div_ceil(m) * m
}
