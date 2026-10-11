//! Every hardware-dependent decision the WebGPU kernels read, declared for
//! the wgpu profile registry, and [`Tuning`]: the values one load runs under.
//!
//! wgpu is the browser's engine. A browser masks the adapter, so there are
//! no named profiles: a load runs the registry defaults with the facts the
//! adapter does report, or a user's file where pie runs natively.

use profile::{Anywhere, Decision, Domain, Profile, Registry, Var};

/// What the wgpu engine reads off its adapter.
pub const FACTS: &[Var] = &[
    Var::fixed(
        "vendor",
        Domain::Enum(&["nvidia", "amd", "intel", "apple", "qualcomm", "other"]),
    ),
    // The narrowest subgroup the adapter may run a dispatch at; absent
    // without the subgroup feature.
    Var::fixed("subgroup_min", Domain::COUNT),
    // Multiprocessors or compute units, where the native driver says.
    Var::fixed("cores", Domain::COUNT),
    Var::fixed("f16", Domain::Bool),
];

/// The recording axes a dynamic variable may name.
pub const AXES: &[&str] = &["rows"];

/// The fewest rows a dense matmul takes the tiled kernel for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.min_rows", constant(range(1..=4096)), default = 12)]
pub struct QmmMinRows(pub u32);

/// Workgroups a dense matmul must fill before a wider column tile pays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.qmm.crossover_tg", constant(range(1..=1_000_000)), default = 160)]
pub struct QmmCrossoverTg(pub u32);

/// The row tile a routed matmul sorts its pairs into, by pairs per expert;
/// 1 leaves the pairs unsorted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.moe.tile_rows",
    shape(per_expert: derived),
    constant(set(1, 16, 32, 64)),
    default = r#"[{ when = "per_expert < 2", value = 1 }, { when = "per_expert >= 32", value = 32 }, { value = 16 }]"#
)]
pub struct MoeTileRows(pub u32);

/// The fewest rows per request that take the tiled attention kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.tile_rows_per_request", constant(range(1..=4096)), default = 32)]
pub struct SdpaTileRowsPerRequest(pub u32);

/// Whether 64-wide heads attend through the matrix kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.mma", flag, default = false)]
pub struct SdpaMma;

/// The most slices a decode row's keys split into; 1 never splits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.split_max", constant(range(1..=1024)), default = 32)]
pub struct DecodeSplitMax(pub u32);

/// The fewest keys a split slice holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.split_min_keys", constant(range(1..=65_536)), default = 256)]
pub struct DecodeSplitMinKeys(pub u32);

/// Register every decision this crate reads.
pub fn register(registry: &mut Registry) {
    registry
        .add::<QmmMinRows>()
        .add::<QmmCrossoverTg>()
        .add::<MoeTileRows>()
        .add::<SdpaTileRowsPerRequest>()
        .add::<SdpaMma>()
        .add::<DecodeSplitMax>()
        .add::<DecodeSplitMinKeys>();
}

/// What one load's kernels run under: the resolved profile, and the device
/// limits a decision is weighed against.
#[derive(Clone, Debug)]
pub struct Tuning {
    profile: Profile,
    /// The cores a decode split fills.
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
    pub fn cores(&self) -> u32 {
        self.cores
    }

    #[must_use]
    pub fn storage_buffers(&self) -> u32 {
        self.storage_buffers
    }

    #[must_use]
    pub fn qmm_min_rows(&self) -> u32 {
        self.profile.get::<QmmMinRows>().0
    }

    #[must_use]
    pub fn qmm_crossover_tg(&self) -> u32 {
        self.profile.get::<QmmCrossoverTg>().0
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

    #[must_use]
    pub fn sdpa_split_max(&self) -> u32 {
        self.profile.get::<DecodeSplitMax>().0
    }

    #[must_use]
    pub fn sdpa_split_min_keys(&self) -> u32 {
        self.profile.get::<DecodeSplitMinKeys>().0
    }
}
