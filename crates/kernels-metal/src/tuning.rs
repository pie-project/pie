//! Every hardware-dependent decision the Metal kernels read, declared for the
//! Metal profile registry, and [`Tuning`]: the values one load runs under.
//!
//! A decision here is data a chip's profile sets; the formulas that use it
//! stay in the kernels. Which chip runs which value lives in
//! `crates/engine-metal/profiles/`, never in this crate.

use profile::{Decision, Profile, Registry};

/// How dense affine-quantised matmuls read prompt rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.path", default = "tiled")]
pub enum QmmPath {
    /// The tiled and split-k kernels, and the row-folded vector kernel below
    /// them.
    Tiled,
    /// The Metal 4 tensor-op (MPP) kernel first, for the shapes it serves;
    /// the tiled path for the rest.
    Mpp,
}

/// The fewest rows a dense matmul over fp16-format weights takes the tiled
/// kernel for; fewer take the vector kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.min_rows", constant(range(1..=4096)), default = 5)]
pub struct QmmMinRows(pub u32);

/// The same threshold for weights the tiled kernel dequantises in the loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.min_rows_emulated", constant(range(1..=4096)), default = 12)]
pub struct QmmMinRowsEmulated(pub u32);

/// Threadgroups a dense matmul must fill before a wider column tile or a
/// coarser row fold pays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.crossover_tg", constant(range(1..=1_000_000)), default = 160)]
pub struct QmmCrossoverTg(pub u32);

/// Whether 4-bit, 64-group weights take the fp16 GEMM format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.fp16", flag, default = true)]
pub struct QmmFp16;

/// The most rows the vector kernel folds into one threadgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmv.rows_max", constant(range(1..=8)), default = 8)]
pub struct QmvRowsMax(pub u32);

/// Codes the row-folded vector kernel reads per load.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmv.packs", constant(set(1, 2)), default = 1)]
pub struct QmvPacks(pub u32);

/// How routed (mixture-of-experts) matmuls read their sorted rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.path", default = "tiled")]
pub enum MoePath {
    Tiled,
    /// The MPP kernel for the 4-bit, 64-group banks it serves.
    Mpp,
}

/// The row tile a routed matmul sorts its pairs into, by pairs per expert;
/// 1 leaves the pairs unsorted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.moe.tile_rows",
    shape(per_expert: derived),
    constant(set(1, 8, 16, 32, 64)),
    default = r#"[{ when = "per_expert < 2", value = 1 }, { when = "per_expert >= 32", value = 32 }, { value = 16 }]"#
)]
pub struct MoeTileRows(pub u32);

/// How prompt rows attend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.prefill", default = "tiled")]
pub enum SdpaPrefill {
    Tiled,
    /// The MPP matmul for the shapes it serves, and the split paged kernel.
    Mpp,
}

/// The fewest rows per request that take the tiled attention kernel; fewer
/// take the vector kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.tile_rows_per_request", constant(range(1..=4096)), default = 32)]
pub struct SdpaTileRowsPerRequest(pub u32);

/// Whether 64-wide heads attend through the simdgroup-matrix kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.mma", flag, default = true)]
pub struct SdpaMma;

/// Whether attention reads keys and values through the 8-bit kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.kv_q8", flag, default = false)]
pub struct SdpaKvQ8;

/// The gated-delta recurrent scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gdn.scan", default = r#"{ use = "simd", rows = 4 }"#)]
pub enum GdnScan {
    /// 32 lanes a head, `rows` value rows a simdgroup.
    Simd {
        #[constant(set(1, 2, 4, 8), default = 4)]
        rows: u32,
    },
    /// 8 lanes a head and 2 value rows, with the staged prompt kernel where
    /// a head is 128 wide.
    Staged,
}

/// Register every decision this crate reads.
pub fn register(registry: &mut Registry) {
    registry
        .add::<QmmPath>()
        .add::<QmmMinRows>()
        .add::<QmmMinRowsEmulated>()
        .add::<QmmCrossoverTg>()
        .add::<QmmFp16>()
        .add::<QmvRowsMax>()
        .add::<QmvPacks>()
        .add::<MoePath>()
        .add::<MoeTileRows>()
        .add::<SdpaPrefill>()
        .add::<SdpaTileRowsPerRequest>()
        .add::<SdpaMma>()
        .add::<SdpaKvQ8>()
        .add::<GdnScan>();
}

/// What one load's kernels run under: the device's resolved profile, and
/// what the model asks of every load on any chip.
#[derive(Clone, Debug)]
pub struct Tuning {
    profile: Profile,
    wide_range: bool,
}

impl Tuning {
    /// `wide_range` is the model's: a deployment whose activations leave the
    /// fp16 range (diffusion latents) takes the wide tiled kernel and no fp16
    /// GEMM format.
    #[must_use]
    pub fn new(profile: Profile, wide_range: bool) -> Tuning {
        Tuning {
            profile,
            wide_range,
        }
    }

    #[must_use]
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    #[must_use]
    pub fn wide_range(&self) -> bool {
        self.wide_range
    }

    #[must_use]
    pub fn qmm_mpp(&self) -> bool {
        *self.profile.get::<QmmPath>() == QmmPath::Mpp
    }

    /// The fewest rows that take the tiled kernel, for weights in the fp16
    /// GEMM format or not.
    #[must_use]
    pub fn qmm_min_rows(&self, fp16: bool) -> u32 {
        if fp16 {
            self.profile.get::<QmmMinRows>().0
        } else {
            self.profile.get::<QmmMinRowsEmulated>().0
        }
    }

    #[must_use]
    pub fn qmm_crossover_tg(&self) -> u32 {
        self.profile.get::<QmmCrossoverTg>().0
    }

    /// Whether a bank of `bits`-bit codes in `group`-wide groups takes the
    /// fp16 GEMM format.
    #[must_use]
    pub fn fp16_gemm_format(&self, bits: u32, group: u32) -> bool {
        !self.wide_range
            && self.profile.on::<QmmFp16>(&profile::Anywhere)
            && bits == 4
            && group == 64
    }

    #[must_use]
    pub fn qmv_rows_max(&self) -> u32 {
        self.profile.get::<QmvRowsMax>().0
    }

    #[must_use]
    pub fn qmv_packs(&self) -> u32 {
        self.profile.get::<QmvPacks>().0
    }

    #[must_use]
    pub fn moe_mpp(&self) -> bool {
        *self.profile.get::<MoePath>() == MoePath::Mpp
    }

    /// The row tile `pairs` routed pairs over `experts` experts sort into;
    /// 1 leaves them unsorted.
    #[must_use]
    pub fn moe_tile_rows(&self, pairs: u32, experts: u32) -> u32 {
        if experts == 0 {
            return 1;
        }
        let at = MoeTileRowsAt {
            per_expert: i64::from(pairs / experts),
        };
        self.profile.pick::<MoeTileRows>(&at).0
    }

    /// Every row tile this profile can sort into, for sizing the stack.
    pub fn moe_tiles(&self) -> impl Iterator<Item = u32> + '_ {
        self.profile.ladder::<MoeTileRows>().values().map(|t| t.0)
    }

    #[must_use]
    pub fn sdpa_mpp(&self) -> bool {
        *self.profile.get::<SdpaPrefill>() == SdpaPrefill::Mpp
    }

    #[must_use]
    pub fn sdpa_tile_rows_per_request(&self) -> u32 {
        self.profile.get::<SdpaTileRowsPerRequest>().0
    }

    #[must_use]
    pub fn sdpa_mma(&self) -> bool {
        self.profile.on::<SdpaMma>(&profile::Anywhere)
    }

    #[must_use]
    pub fn kv_q8(&self) -> bool {
        self.profile.on::<SdpaKvQ8>(&profile::Anywhere)
    }

    /// The gated-delta scan as `(lanes a head, value rows a simdgroup)`.
    #[must_use]
    pub fn gdn_scan(&self) -> (u32, u32) {
        match *self.profile.get::<GdnScan>() {
            GdnScan::Simd { rows } => (32, rows),
            GdnScan::Staged => (8, 2),
        }
    }
}
