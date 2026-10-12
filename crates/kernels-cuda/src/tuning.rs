//! Every hardware-dependent decision the CUDA kernels read, declared for the
//! CUDA profile registry, and [`Tuning`]: the values one load runs under.
//!
//! Which GPU runs which value lives in `crates/engine-cuda/profiles/`.

use profile::{Anywhere, Decision, Profile, Registry};

/// Whether routed matmuls over bf16 or f16 banks take the grouped GEMM once
/// an expert's routes pay for it; off, every route reads its expert alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.grouped", flag, default = true, tune = search)]
pub struct MoeGrouped;

/// The same for mxfp4 banks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_grouped", flag, default = true, tune = search)]
pub struct MoeMxfp4Grouped;

/// Whether the grouped mxfp4 kernel multiplies bf16 rows with tensor cores
/// (WMMA) rather than the scalar grouped kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.mxfp4_wmma", flag, default = true, tune = search)]
pub struct MoeMxfp4Wmma;

/// Whether a small decode batch still splits its keys across blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.split_small", flag, default = false, tune = search)]
pub struct DecodeSplitSmall;

/// Whether a sliding-window decode splits its keys across blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.split_window", flag, default = false, tune = search)]
pub struct DecodeSplitWindow;

/// Whether a split decode chunk holds at least 128 keys, as the scheduler
/// once floored it, rather than one page.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.chunk_floor_128", flag, default = false, tune = search)]
pub struct DecodeChunkFloor128;

/// Register every decision this crate reads.
pub fn register(registry: &mut Registry) {
    registry
        .add::<MoeGrouped>()
        .add::<MoeMxfp4Grouped>()
        .add::<MoeMxfp4Wmma>()
        .add::<DecodeSplitSmall>()
        .add::<DecodeSplitWindow>()
        .add::<DecodeChunkFloor128>();
}

/// The kernel decisions one load runs under, read once from its profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    pub moe_grouped: bool,
    pub mxfp4_grouped: bool,
    pub mxfp4_wmma: bool,
    pub decode_split_small: bool,
    pub decode_split_window: bool,
    pub decode_chunk_floor_128: bool,
}

impl Tuning {
    /// The registry defaults, for a context no load has tuned (kernel
    /// tests); a test holds the two equal.
    pub const DEFAULT: Tuning = Tuning {
        moe_grouped: true,
        mxfp4_grouped: true,
        mxfp4_wmma: true,
        decode_split_small: false,
        decode_split_window: false,
        decode_chunk_floor_128: false,
    };

    #[must_use]
    pub fn of(profile: &Profile) -> Tuning {
        Tuning {
            moe_grouped: profile.on::<MoeGrouped>(&Anywhere),
            mxfp4_grouped: profile.on::<MoeMxfp4Grouped>(&Anywhere),
            mxfp4_wmma: profile.on::<MoeMxfp4Wmma>(&Anywhere),
            decode_split_small: profile.on::<DecodeSplitSmall>(&Anywhere),
            decode_split_window: profile.on::<DecodeSplitWindow>(&Anywhere),
            decode_chunk_floor_128: profile.on::<DecodeChunkFloor128>(&Anywhere),
        }
    }
}

impl Default for Tuning {
    fn default() -> Tuning {
        Tuning::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_tuning_is_the_registry_defaults() {
        let mut r = Registry::new("cuda", &[], &["rows"]);
        register(&mut r);
        assert_eq!(Tuning::of(&r.defaults()), Tuning::default());
    }
}
