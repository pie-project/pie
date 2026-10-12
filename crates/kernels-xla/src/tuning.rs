//! Every hardware-dependent decision the XLA kernels read, declared for the
//! XLA profile registry, and [`Tuning`]: the values one load runs under.
//!
//! The defaults are the numbers measured on a TPU v6e, the one generation
//! tuned so far: XLA compiles for any, and a guess would serve no better.

use profile::{Anywhere, Decision, Profile, Registry};

/// How a routed (mixture-of-experts) matmul is formulated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.path", default = "auto")]
pub enum MoePath {
    /// Chosen per fire from the bank's bytes, the pairs and the costs
    /// measured on v6e.
    Auto,
    /// Each pair's expert sliced out of the bank and multiplied by its row.
    Slice,
    /// The loop over the experts the fire routes to.
    Loop,
    /// The loop over active experts, or the whole bank when more are active
    /// than the loop is worth.
    Active,
    /// Every row against every expert.
    All,
    /// The ragged dot.
    Ragged,
    /// The grouped-matmul kernel, where the bank is one it reads; else
    /// every expert.
    Gmm,
}

/// Whether the automatic formulation may take the grouped-matmul kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.gmm", flag, default = true)]
pub struct MoeGmm;

/// Tiles the loop over active experts multiplies per step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.loop_unroll", constant(range(1..=8)), default = 2)]
pub struct MoeLoopUnroll(pub u32);

/// The most rows one tile of the loop over active experts holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.loop_tile_max", constant(range(8..=128)), default = 128)]
pub struct MoeLoopTileMax(pub u32);

/// Register every decision this crate reads.
pub fn register(registry: &mut Registry) {
    registry
        .add::<MoePath>()
        .add::<MoeGmm>()
        .add::<MoeLoopUnroll>()
        .add::<MoeLoopTileMax>();
}

/// The kernel decisions one load runs under, read once from its profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    /// A formulation the profile forces, or `Auto`.
    pub moe_path: MoePath,
    pub moe_gmm: bool,
    pub moe_loop_unroll: i64,
    pub moe_loop_tile_max: i64,
}

impl Tuning {
    /// The registry defaults, for an emitter no load has tuned.
    pub const DEFAULT: Tuning = Tuning {
        moe_path: MoePath::Auto,
        moe_gmm: true,
        moe_loop_unroll: 2,
        moe_loop_tile_max: 128,
    };

    #[must_use]
    pub fn of(profile: &Profile) -> Tuning {
        Tuning {
            moe_path: *profile.get::<MoePath>(),
            moe_gmm: profile.on::<MoeGmm>(&Anywhere),
            moe_loop_unroll: i64::from(profile.get::<MoeLoopUnroll>().0),
            moe_loop_tile_max: i64::from(profile.get::<MoeLoopTileMax>().0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_tuning_is_the_registry_defaults() {
        let mut r = Registry::new("xla", &[], &["rows"]);
        register(&mut r);
        assert_eq!(Tuning::of(&r.defaults()), Tuning::DEFAULT);
    }
}
