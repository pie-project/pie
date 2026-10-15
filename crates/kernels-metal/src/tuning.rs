//! The hardware-dependent decisions the Metal kernels read, and [`Tuning`]:
//! the values one load runs under.

use profile::{Decision, Profile, Registry};

/// The kernel a dense affine-quantised matmul runs: the first rung whose
/// shape it fits and whose scratch it finds. `format` is the bank's: `fp16`
/// where its 4-bit, 64-group codes take the fp16 GEMM format, else
/// `emulated`. `fill` is how full the device is: eighths of a wave of
/// 32x64 output tiles over its cores, so a rung can say "fills the device"
/// in a way that holds on chips of any size; unknown on a device that does
/// not say its cores, where a rung that reads it does not hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.qmm",
    shape(rows, fill: derived optional; n, k, bits, group, format: enum(fp16, emulated)),
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
    #[partial]
    Staged,
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

/// The MPP matmul's `(row tile, column tile)` for `rows` rows: what the
/// dense and the routed kernels both stamp.
#[must_use]
pub fn mpp_tile(rows: u32) -> (u32, u32) {
    let bm = match rows {
        0..=8 => 8,
        9..=16 => 16,
        17..=24 => 24,
        _ => 32,
    };
    (bm, if bm == 8 { 64 } else { 128 })
}

/// Eighths of a wave of 32x64 output tiles over `cores` GPU cores: how full
/// `rows` rows into `n` columns keep the device.
#[must_use]
pub fn fill(rows: u32, n: u32, cores: u32) -> i64 {
    let tiles = u64::from(rows.div_ceil(32)) * u64::from(n.div_ceil(64));
    i64::try_from(tiles * 8 / u64::from(cores.max(1))).unwrap_or(i64::MAX)
}

impl Qmm {
    fn fits_shape(&self, at: &QmmAt) -> bool {
        match self {
            Qmm::Mpp => {
                let (_, bn) = mpp_tile(u32::try_from(at.rows).unwrap_or(u32::MAX));
                at.bits == 4 && at.group == 64 && at.rows >= 8 && at.n % i64::from(bn) == 0
            }
            Qmm::Staged => at.bits == 4 && at.group == 64 && at.rows <= 8 && at.n % 64 == 0,
            Qmm::Tiled { .. } => at.k % i64::from(crate::linear::quant::QMM_BK) == 0,
            Qmm::Qmv { .. } => true,
        }
    }
}

/// Whether 4-bit, 64-group weights take the fp16 GEMM format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm_fp16", flag, default = true, tune = search)]
pub struct QmmFp16;

/// Split-K for skinny dense matmuls: a decode-sized tiled matmul whose
/// tiles fill fewer than `fill_tg` threadgroups splits its contraction into
/// up to `max` slices and reduces the partials.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.qmm_splitk",
    default = r#"{ use = "fill", fill_tg = 512, max = 8 }"#,
    tune = search
)]
pub enum QmmSplitK {
    Off,
    Fill {
        #[constant(range(16..=8192), default = 512)]
        fill_tg: u32,
        #[constant(set(2, 4, 8), default = 8)]
        max: u32,
    },
}

/// The fewest rows a tile must hold before activations are cast to fp16 once
/// and the fp16 GEMM reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm_precast", constant(set(16, 32, 64)), default = 16, tune = search)]
pub struct QmmPrecastMinBm(pub u32);

/// The widest row tile a padded tiled matmul widens to when a wider tile
/// still fills the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm_widen", constant(set(16, 32, 64)), default = 32, tune = search)]
pub struct QmmWiden(pub u32);

macro_rules! counts {
    ($($ty:ident = $id:literal, $lo:literal ..= $hi:literal, $default:literal, $doc:literal;)*) => {
        $(
            #[doc = $doc]
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
            #[decision(id = $id, constant(range($lo..=$hi)), default = $default, tune = search)]
            pub struct $ty(pub u32);
        )*
    };
}

counts! {
    MppSplitRows = "kernel.qmm_mpp.split_rows", 8..=256, 32,
        "Up to this many rows (or twice it over few columns) the MPP matmul splits its contraction across partitions.";
    MppSplitCols = "kernel.qmm_mpp.split_cols", 1024..=65536, 8192,
        "The most columns the MPP matmul splits at twice `split_rows` rows.";
    MppPipelineCols = "kernel.qmm_mpp.pipeline_cols", 1024..=65536, 8192,
        "The most columns the MPP matmul pipelines its loads over.";
    MppLinearFrom = "kernel.qmm_mpp.linear_from", 256..=8192, 1024,
        "From this many rows a packed bank's MPP matmul walks its tiles linearly.";
    SdpaMppFrom = "kernel.sdpa.mpp_rows.from", 8..=1024, 32,
        "The fewest prompt rows that attend through the MPP kernels.";
    SdpaMppDirectFrom = "kernel.sdpa.mpp_rows.direct_from", 64..=8192, 512,
        "From this many rows 256-wide heads take the direct-visit MPP kernel.";
    SdpaPairedBelow = "kernel.sdpa.mpp_rows.paired_below", 1..=256, 32,
        "Below this many rows 256-wide grouped-query heads decode in pairs.";
}

/// The most rows the MPP matmul pipelines its loads over.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm_mpp.pipeline_rows", constant(set(8, 16, 24, 32)), default = 16, tune = search)]
pub struct MppPipelineRows(pub u32);

/// The `kernel.qmm_mpp.*` decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QmmMpp {
    pub split_rows: u32,
    pub split_cols: u32,
    pub pipeline_rows: u32,
    pub pipeline_cols: u32,
    pub linear_from: u32,
}

/// The `kernel.sdpa.mpp_rows.*` decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SdpaMppRows {
    pub from: u32,
    pub direct_from: u32,
    pub paired_below: u32,
}

/// How many slices the split paged-attention kernel aims for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.split", constant(set(8, 16, 32)), default = 32, tune = search)]
pub struct SdpaSplit(pub u32);

/// How routed (mixture-of-experts) matmuls read their sorted rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.path", default = "tiled", tune = search)]
pub enum MoePath {
    Tiled,
    /// The MPP kernel for the 4-bit, 64-group banks it serves.
    Mpp,
}

/// The row tile a routed matmul sorts its pairs into, by pairs per expert;
/// 1 leaves the pairs unsorted. An 8-row tile is stamped only for fp16 4-bit
/// 64-group banks; any other bank runs it at 16.
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
    /// a head is 128 wide and a request brings at least `min_rows` rows.
    Staged {
        #[constant(range(1..=4096), default = 32)]
        min_rows: u32,
    },
}

/// Register every decision this crate reads.
pub fn register(registry: &mut Registry) {
    registry
        .add::<Qmm>()
        .add::<QmmFp16>()
        .add::<QmmSplitK>()
        .add::<QmmPrecastMinBm>()
        .add::<QmmWiden>()
        .add::<MppSplitRows>()
        .add::<MppSplitCols>()
        .add::<MppPipelineRows>()
        .add::<MppPipelineCols>()
        .add::<MppLinearFrom>()
        .add::<SdpaMppFrom>()
        .add::<SdpaMppDirectFrom>()
        .add::<SdpaPairedBelow>()
        .add::<SdpaSplit>()
        .add::<MoePath>()
        .add::<MoeTileRows>()
        .add::<SdpaPrefill>()
        .add::<SdpaTileRowsPerRequest>()
        .add::<SdpaMma>()
        .add::<SdpaKvQ8>()
        .add::<GdnScan>();
}

#[derive(Clone, Debug)]
pub struct Tuning {
    profile: Profile,
    wide_range: bool,
    cores: Option<u32>,
}

impl Tuning {
    /// `wide_range`: the model's activations leave the fp16 range (diffusion
    /// latents), so it takes the wide tiled kernel and no fp16 GEMM format.
    /// `cores`: the GPU's, where the device says.
    #[must_use]
    pub fn new(profile: Profile, wide_range: bool, cores: Option<u32>) -> Tuning {
        Tuning {
            profile,
            wide_range,
            cores,
        }
    }

    #[must_use]
    pub fn qmm_at(&self, rows: u32, n: u32, k: u32, bits: u32, group: u32) -> QmmAt {
        QmmAt {
            rows: i64::from(rows),
            fill: self.cores.map(|cores| fill(rows, n, cores)),
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

    /// `(fill threadgroups, most slices)` of split-K, or `None` when off.
    #[must_use]
    pub fn qmm_splitk(&self) -> Option<(u32, u32)> {
        match *self.profile.get::<QmmSplitK>() {
            QmmSplitK::Off => None,
            QmmSplitK::Fill { fill_tg, max } => Some((fill_tg, max)),
        }
    }

    /// The most slices split-K may cut, whatever a load decides: what its
    /// scratch is sized for.
    #[must_use]
    pub fn qmm_splitk_max(&self) -> u32 {
        self.profile
            .ladder::<QmmSplitK>()
            .values()
            .map(|v| match v {
                QmmSplitK::Off => 1,
                QmmSplitK::Fill { max, .. } => *max,
            })
            .max()
            .unwrap_or(1)
    }

    #[must_use]
    pub fn qmm_precast_min_bm(&self) -> u32 {
        self.profile.get::<QmmPrecastMinBm>().0
    }

    #[must_use]
    pub fn qmm_widen(&self) -> u32 {
        self.profile.get::<QmmWiden>().0
    }

    #[must_use]
    pub fn qmm_mpp_tiles(&self) -> QmmMpp {
        QmmMpp {
            split_rows: self.profile.get::<MppSplitRows>().0,
            split_cols: self.profile.get::<MppSplitCols>().0,
            pipeline_rows: self.profile.get::<MppPipelineRows>().0,
            pipeline_cols: self.profile.get::<MppPipelineCols>().0,
            linear_from: self.profile.get::<MppLinearFrom>().0,
        }
    }

    #[must_use]
    pub fn sdpa_mpp_rows(&self) -> SdpaMppRows {
        SdpaMppRows {
            from: self.profile.get::<SdpaMppFrom>().0,
            direct_from: self.profile.get::<SdpaMppDirectFrom>().0,
            paired_below: self.profile.get::<SdpaPairedBelow>().0,
        }
    }

    #[must_use]
    pub fn sdpa_split(&self) -> u32 {
        self.profile.get::<SdpaSplit>().0
    }

    #[must_use]
    pub fn gdn_scan(&self) -> GdnScan {
        *self.profile.get::<GdnScan>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rung_that_reads_fill_does_not_hold_without_the_cores() {
        let mut registry = Registry::new("metal", &[], &["rows"]);
        register(&mut registry);
        let profile = registry.with_overrides(
            "[kernel]\nqmm = [{ when = \"fill >= 8\", use = \"tiled\", crossover_tg = 160 }, \"qmv\"]\n",
        );
        let first = |cores| {
            let t = Tuning::new(profile.clone(), false, cores);
            t.qmm(&t.qmm_at(64, 4096, 4096, 4, 64))[0]
        };
        assert_eq!(first(Some(10)), Qmm::Tiled { crossover_tg: 160 });
        assert_eq!(
            first(None),
            Qmm::Qmv {
                rows_max: 8,
                packs: 1,
                crossover_tg: 160
            },
            "the next rung serves"
        );
    }
}
