//! Every hardware-dependent decision the Metal kernels read, declared for the
//! Metal profile registry, and [`Tuning`]: the values one load runs under.
//!
//! A decision here is data a chip's profile sets; the formulas that use it
//! stay in the kernels. Which chip runs which value lives in
//! `crates/engine-metal/profiles/`, never in this crate.

use profile::{Decision, Profile, Registry};

/// The kernel a dense affine-quantised matmul runs: the first rung whose
/// shape it fits and whose scratch it finds. `format` is the bank's: `fp16`
/// where its 4-bit, 64-group codes take the fp16 GEMM format, else
/// `emulated`. `fill` is how full the device is: eighths of a wave of
/// 32x64 output tiles over its cores, so a rung can say "fills the device"
/// in a way that holds on chips of any size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.qmm",
    shape(rows, fill: derived; n, k, bits, group, format: enum(fp16, emulated)),
    tune = search,
    default = r#"[
        { when = "rows >= 5 and (bits == 2 or format == fp16)", use = "tiled", crossover_tg = 160 },
        { when = "rows >= 12", use = "tiled", crossover_tg = 160 },
        { use = "qmv", rows_max = 8, packs = 1, crossover_tg = 160 },
    ]"#,
    fits = Qmm::fits_shape
)]
pub enum Qmm {
    /// The Metal 4 tensor-op (MPP) kernel: 4-bit, 64-group banks, at least
    /// 8 rows, a whole number of its column tiles.
    #[partial]
    Mpp,
    /// The tiled kernels (fp16 precast, split-k, plain), where the
    /// contraction is whole 32-code tiles and a tile fills the device.
    #[partial]
    Tiled {
        /// Threadgroups a tiling must fill before a wider column tile pays.
        #[constant(range(1..=1_000_000), default = 160)]
        crossover_tg: u32,
    },
    /// The vector kernels: rows folded into a threadgroup where that fills
    /// the device, else one row at a time.
    Qmv {
        /// The most rows folded into one threadgroup.
        #[constant(range(1..=8), default = 8)]
        rows_max: u32,
        /// Codes the row-folded kernel reads per load.
        #[constant(set(1, 2), default = 1)]
        packs: u32,
        /// Threadgroups a fold must fill before a coarser one pays.
        #[constant(range(1..=1_000_000), default = 160)]
        crossover_tg: u32,
    },
}

impl Qmm {
    fn fits_shape(&self, at: &QmmAt) -> bool {
        match self {
            Qmm::Mpp => {
                let bn = if at.rows <= 8 { 64 } else { 128 };
                at.bits == 4 && at.group == 64 && at.rows >= 8 && at.n % bn == 0
            }
            Qmm::Tiled { .. } => at.k % i64::from(crate::linear::quant::QMM_BK) == 0,
            Qmm::Qmv { .. } => true,
        }
    }
}

/// Whether 4-bit, 64-group weights take the fp16 GEMM format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm_fp16", flag, default = true, tune = search)]
pub struct QmmFp16;

/// How routed (mixture-of-experts) matmuls read their sorted rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.path", default = "tiled", tune = search)]
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
    tune = search,
    constant(set(1, 8, 16, 32, 64)),
    default = r#"[{ when = "per_expert < 2", value = 1 }, { when = "per_expert >= 32", value = 32 }, { value = 16 }]"#
)]
pub struct MoeTileRows(pub u32);

/// How prompt rows attend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.prefill", default = "tiled", tune = search)]
pub enum SdpaPrefill {
    Tiled,
    /// The MPP matmul for the shapes it serves, and the split paged kernel.
    Mpp,
}

/// The fewest rows per request that take the tiled attention kernel; fewer
/// take the vector kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.sdpa.tile_rows_per_request",
    constant(range(1..=4096)),
    default = 32,
    tune = search
)]
pub struct SdpaTileRowsPerRequest(pub u32);

/// Whether 64-wide heads attend through the simdgroup-matrix kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.mma", flag, default = true, tune = search)]
pub struct SdpaMma;

/// Whether attention reads keys and values through the 8-bit kernel. It
/// halves the pool and changes what attention reads: the operator's call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.kv_q8", flag, default = false, tune = never)]
pub struct SdpaKvQ8;

/// The gated-delta recurrent scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gdn.scan", default = r#"{ use = "simd", rows = 4 }"#, tune = search)]
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
        .add::<Qmm>()
        .add::<QmmFp16>()
        .add::<MoePath>()
        .add::<MoeTileRows>()
        .add::<SdpaPrefill>()
        .add::<SdpaTileRowsPerRequest>()
        .add::<SdpaMma>()
        .add::<SdpaKvQ8>()
        .add::<GdnScan>();
}

/// What one load's kernels run under: the device's resolved profile, its
/// core count, and what the model asks of every load on any chip.
#[derive(Clone, Debug)]
pub struct Tuning {
    profile: Profile,
    wide_range: bool,
    cores: u32,
}

impl Tuning {
    /// `wide_range` is the model's: a deployment whose activations leave the
    /// fp16 range (diffusion latents) takes the wide tiled kernel and no fp16
    /// GEMM format.
    /// `cores` is the device's GPU core count, which `fill` divides by.
    #[must_use]
    pub fn new(profile: Profile, wide_range: bool, cores: u32) -> Tuning {
        Tuning {
            profile,
            wide_range,
            cores: cores.max(1),
        }
    }

    /// The point a dense matmul of `rows` rows into `n` columns over a
    /// `k`-long contraction of a `bits`-bit, `group`-wide bank is chosen at.
    #[must_use]
    pub fn qmm_at(&self, rows: u32, n: u32, k: u32, bits: u32, group: u32) -> QmmAt {
        let tiles = u64::from(rows.div_ceil(32)) * u64::from(n.div_ceil(64));
        QmmAt {
            rows: i64::from(rows),
            fill: i64::try_from(tiles * 8 / u64::from(self.cores)).unwrap_or(i64::MAX),
            n: i64::from(n),
            k: i64::from(k),
            bits: i64::from(bits),
            group: i64::from(group),
            format: if self.fp16_gemm_format(bits, group) {
                "fp16"
            } else {
                "emulated"
            },
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

    /// The kernels a dense matmul at `at` may run, most preferred first.
    #[must_use]
    pub fn qmm(&self, at: &QmmAt) -> Vec<Qmm> {
        self.profile
            .ladder::<Qmm>()
            .candidates(at)
            .copied()
            .collect()
    }

    /// Whether any dense matmul may take the MPP kernel: its scratch is
    /// reserved at load.
    #[must_use]
    pub fn qmm_mpp(&self) -> bool {
        self.profile
            .ladder::<Qmm>()
            .values()
            .any(|q| *q == Qmm::Mpp)
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
