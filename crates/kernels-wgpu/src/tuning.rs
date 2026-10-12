//! The hardware-dependent decisions the WebGPU kernels read, and [`Tuning`],
//! the values one load runs under.

use profile::{Decision, Domain, Profile, Registry, Var};

pub const FACTS: &[Var] = &[
    Var::fixed("vendor", Domain::Enum(profile::vendor::NAMES)),
    // Absent without the subgroup feature.
    Var::fixed("subgroup_min", Domain::COUNT),
    Var::fixed("cores", Domain::COUNT),
    Var::fixed("f16", Domain::Bool),
];

pub const AXES: &[&str] = &["rows"];

/// The kernel a dense affine-quantised matmul runs: the first rung whose
/// shape it fits that can run. `fill` is how full the device is: eighths of
/// a wave of 32x64 output tiles over its cores, as on Vulkan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.qmm",
    shape(rows, fill: derived; n, k, bits, group),
    default = r#"[
        { when = "rows >= 12", use = "tiled", crossover_tg = 160 },
        "qmv",
    ]"#,
    fits = Qmm::fits_shape,
    tune = search
)]
pub enum Qmm {
    /// The tiled kernel, through workgroup memory.
    #[partial]
    Tiled {
        /// Workgroups a tiling must fill before a 32-wide column tile pays.
        #[constant(range(1..=1_000_000), default = 160)]
        crossover_tg: u32,
    },
    /// The vector kernel.
    Qmv,
}

impl Qmm {
    fn fits_shape(&self, at: &QmmAt) -> bool {
        match self {
            Qmm::Tiled { .. } => at.k % i64::from(crate::linear::quant::QMM_BK) == 0,
            Qmm::Qmv => true,
        }
    }
}

/// The row tile a routed matmul sorts its pairs into, by pairs per expert;
/// 1 leaves the pairs unsorted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.moe.tile_rows",
    shape(per_expert: derived),
    constant(set(1, 16, 32, 64)),
    default = r#"[{ when = "per_expert < 2", value = 1 }, { when = "per_expert >= 32", value = 32 }, { value = 16 }]"#,
    tune = search
)]
pub struct MoeTileRows(pub u32);

/// The fewest rows per request that take the tiled attention kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.tile_rows_per_request", constant(range(1..=4096)), default = 32, tune = search)]
pub struct SdpaTileRowsPerRequest(pub u32);

/// How many workgroups a decode row's keys may split across, as on Vulkan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.sdpa.decode.split",
    default = r#"{ use = "per_core", per = 1 }"#,
    tune = search
)]
pub enum DecodeSplit {
    /// A fixed workgroup count.
    Fixed {
        #[constant(range(1..=65_536), default = 128)]
        groups: u32,
    },
    /// `per` workgroups for each core the engine sizes work by (the count
    /// the native driver reports, else a fallback).
    PerCore {
        #[constant(range(1..=64), default = 1)]
        per: u32,
    },
}

/// The fewest keys a split slice holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.split_min_keys", constant(range(1..=65_536)), default = 256, tune = search)]
pub struct DecodeSplitMinKeys(pub u32);

pub fn register(registry: &mut Registry) {
    registry
        .add::<Qmm>()
        .add::<MoeTileRows>()
        .add::<SdpaTileRowsPerRequest>()
        .add::<DecodeSplit>()
        .add::<DecodeSplitMinKeys>();
}

#[derive(Clone, Debug)]
pub struct Tuning {
    profile: Profile,
    cores: u32,
    /// Storage buffers a shader stage may bind; 0 where unknown.
    storage_buffers: u32,
}

impl Tuning {
    #[must_use]
    pub fn new(profile: Profile, cores: u32, storage_buffers: u32) -> Tuning {
        Tuning {
            profile,
            cores,
            storage_buffers,
        }
    }

    #[must_use]
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    #[must_use]
    pub fn storage_buffers(&self) -> u32 {
        self.storage_buffers
    }

    #[must_use]
    pub fn qmm_at(&self, rows: u32, n: u32, k: u32, bits: u32, group: u32) -> QmmAt {
        let tiles = u64::from(rows.div_ceil(32)) * u64::from(n.div_ceil(64));
        QmmAt {
            rows: i64::from(rows),
            fill: i64::try_from(tiles * 8 / u64::from(self.cores.max(1))).unwrap_or(i64::MAX),
            n: i64::from(n),
            k: i64::from(k),
            bits: i64::from(bits),
            group: i64::from(group),
        }
    }

    #[must_use]
    pub fn qmm(&self, at: &QmmAt) -> Vec<Qmm> {
        self.profile
            .ladder::<Qmm>()
            .candidates(at)
            .copied()
            .collect()
    }

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
    pub fn sdpa_tile_rows_per_request(&self) -> u32 {
        self.profile.get::<SdpaTileRowsPerRequest>().0
    }

    #[must_use]
    pub fn sdpa_split_groups(&self) -> u32 {
        match *self.profile.get::<DecodeSplit>() {
            DecodeSplit::Fixed { groups } => groups,
            DecodeSplit::PerCore { per } => self.cores.saturating_mul(per).max(1),
        }
    }

    #[must_use]
    pub fn sdpa_split_min_keys(&self) -> u32 {
        self.profile.get::<DecodeSplitMinKeys>().0
    }
}
