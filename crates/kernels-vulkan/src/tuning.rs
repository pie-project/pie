//! Every hardware-dependent decision the Vulkan kernels read, declared for
//! the Vulkan profile registry, and [`Tuning`]: the values one load runs
//! under.
//!
//! An alternative that needs the device to have something says so in
//! `requires`: the cooperative-matrix kernels need `coopmat`, and the
//! one-row vector kernel folds with subgroup shuffles that assume at least
//! 32 lanes. A profile that picks one on a device without it is refused
//! whole, so no file can run a kernel the device answers wrongly.

use profile::{Anywhere, Decision, Domain, Profile, Registry, Var};

/// What the Vulkan engine measures about its device, for `target.assumes`,
/// `target.match` and what an alternative requires.
pub const FACTS: &[Var] = &[
    Var::fixed("vendor", Domain::Enum(profile::vendor::NAMES)),
    // The narrowest subgroup a pipeline may be compiled at.
    Var::fixed("subgroup", Domain::COUNT),
    // Multiprocessors or compute units; absent where no extension says.
    Var::fixed("cores", Domain::COUNT),
    Var::fixed("coopmat", Domain::Bool),
];

/// The recording axes a dynamic variable may name.
pub const AXES: &[&str] = &["rows"];

/// How dense affine-quantised matmuls read prompt rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.path", default = "tiled", tune = search)]
pub enum QmmPath {
    /// The tiled kernels, and the vector kernels below them.
    Tiled,
    /// The cooperative-matrix kernel first, where its tile divides the
    /// contraction.
    #[requires("coopmat")]
    Coopmat,
}

/// The fewest rows a dense matmul takes a tiled or cooperative-matrix
/// kernel for; fewer take the vector kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.min_rows", constant(range(1..=4096)), default = 12, tune = search)]
pub struct QmmMinRows(pub u32);

/// Workgroups a dense matmul must fill before a wider column tile pays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.crossover_tg", constant(range(1..=1_000_000)), default = 160, tune = search)]
pub struct QmmCrossoverTg(pub u32);

/// Whether batched projections take the register-tiled point (a 4x4 block
/// per invocation) rather than the plain tiled one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.register_tiles", flag, default = false, tune = search)]
pub struct QmmRegisterTiles;

/// How a one-row quantised projection (decode) reads its bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmv.path", default = "plain", tune = search)]
pub enum QmvPath {
    Plain,
    /// 128-bit loads folded with subgroup shuffles.
    #[requires("subgroup >= 32")]
    OneRow,
}

/// How unquantised matmuls read prompt rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.path", default = "tiled", tune = search)]
pub enum GemmPath {
    Tiled,
    #[requires("coopmat")]
    Coopmat,
}

/// How routed (mixture-of-experts) matmuls read their sorted rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.path", default = "tiled", tune = search)]
pub enum MoePath {
    Tiled,
    #[requires("coopmat")]
    Coopmat,
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

/// Whether 64-wide heads attend through the matrix kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.mma", flag, default = false, tune = search)]
pub struct SdpaMma;

/// How many workgroups a decode row's keys may split across.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.sdpa.decode.split",
    default = r#"{ use = "fixed", groups = 128 }"#,
    tune = search
)]
pub enum DecodeSplit {
    /// A fixed workgroup count.
    Fixed {
        #[constant(range(1..=65_536), default = 128)]
        groups: u32,
    },
    /// `per` workgroups for each core the device reports.
    #[requires("cores > 0")]
    PerCore {
        #[constant(range(1..=64), default = 1)]
        per: u32,
    },
}

/// The fewest keys a split slice holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.split_min_keys", constant(range(1..=65_536)), default = 256, tune = search)]
pub struct DecodeSplitMinKeys(pub u32);

/// Register every decision this crate reads.
pub fn register(registry: &mut Registry) {
    registry
        .add::<QmmPath>()
        .add::<QmmMinRows>()
        .add::<QmmCrossoverTg>()
        .add::<QmmRegisterTiles>()
        .add::<QmvPath>()
        .add::<GemmPath>()
        .add::<MoePath>()
        .add::<MoeTileRows>()
        .add::<SdpaTileRowsPerRequest>()
        .add::<SdpaMma>()
        .add::<DecodeSplit>()
        .add::<DecodeSplitMinKeys>();
}

/// What one load's kernels run under: the device's resolved profile, and
/// the core count a per-core decision multiplies.
#[derive(Clone, Debug)]
pub struct Tuning {
    profile: Profile,
    cores: u32,
}

impl Tuning {
    #[must_use]
    pub fn new(profile: Profile, cores: u32) -> Tuning {
        Tuning { profile, cores }
    }

    #[must_use]
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    #[must_use]
    pub fn qmm_coopmat(&self) -> bool {
        *self.profile.get::<QmmPath>() == QmmPath::Coopmat
    }

    #[must_use]
    pub fn qmm_min_rows(&self) -> u32 {
        self.profile.get::<QmmMinRows>().0
    }

    #[must_use]
    pub fn qmm_crossover_tg(&self) -> u32 {
        self.profile.get::<QmmCrossoverTg>().0
    }

    #[must_use]
    pub fn qmm_register_tiles(&self) -> bool {
        self.profile.on::<QmmRegisterTiles>(&Anywhere)
    }

    #[must_use]
    pub fn qmv_one_row(&self) -> bool {
        *self.profile.get::<QmvPath>() == QmvPath::OneRow
    }

    #[must_use]
    pub fn gemm_coopmat(&self) -> bool {
        *self.profile.get::<GemmPath>() == GemmPath::Coopmat
    }

    #[must_use]
    pub fn moe_coopmat(&self) -> bool {
        *self.profile.get::<MoePath>() == MoePath::Coopmat
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
    pub fn sdpa_tile_rows_per_request(&self) -> u32 {
        self.profile.get::<SdpaTileRowsPerRequest>().0
    }

    #[must_use]
    pub fn sdpa_mma(&self) -> bool {
        self.profile.on::<SdpaMma>(&Anywhere)
    }

    /// The workgroups a decode row's keys may split across.
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
