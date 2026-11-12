//! The hardware-dependent decisions engine-wgpu reads when it fires
//! kernels-slang's entries, and [`Tuning`], the values one load runs under.

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
            Qmm::Tiled { .. } => at.k % i64::from(kernels_slang::linear::qmm::TILE_K) == 0,
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

    /// The rows a batched route sorts into: every touched expert's run
    /// padded to whole `kernel.moe.tile_rows` tiles; the pairs themselves
    /// where the route stays unsorted.
    #[must_use]
    pub fn moe_sorted_rows(&self, pairs: u32, experts: u32) -> u32 {
        let tile = self.moe_tile_rows(pairs, experts);
        if tile <= 1 {
            return pairs;
        }
        let touched = pairs.min(experts);
        let bound = pairs.saturating_add(touched.saturating_mul(tile - 1));
        bound.div_ceil(tile) * tile
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

/// WGSL has no subgroup tier; four strips a workgroup (kernels-slang.md §15).
const KQUANT: kernels_slang::linear::kquant::Tactic = kernels_slang::linear::kquant::Tactic {
    tier: kernels_slang::Tier::Baseline,
    slices: 4,
};

/// Storage buffers a split decode binds (its LSE form one more); a device
/// offering fewer (SwiftShader: 10) walks each row whole.
const SPLIT_STORAGE_BUFFERS: u32 = 12;

/// A tab's prefill dispatches, in rows times keys. A GPU cannot leave a
/// dispatch to draw the screen, and macOS ends a command buffer once that has
/// gone on for about 20 seconds (`Impacting Interactivity`), which costs the
/// tab its device. Smaller command buffers do not help: on an M2 Max,
/// dispatches of 95 ms are ended and these, of 48 ms, are not. A browser hides
/// its adapter from wgpu, so every tab pays for them, about 3% of a prefill.
const TAB_PREFILL_WORK: u64 = 1 << 19;

impl engine_gpu::backend::Tactics for Tuning {
    fn profile(&self) -> &profile::Profile {
        Tuning::profile(self)
    }

    /// The device's `kernel.qmm` ladder at this shape, as kernels-slang's
    /// rungs: the vector rung reads two activation rows a workgroup, which
    /// on the 4090 beat the old hand-written WGSL one-row kernel at every row
    /// count from 1 to 11 where four rows lost at two (kernels-slang.md §17).
    fn qmm_ladder(
        &self,
        rows: u32,
        columns: u32,
        contraction: u32,
        bank: kernels_slang::Bank,
    ) -> Vec<kernels_slang::linear::qmm::Qmm> {
        use kernels_slang::linear::qmm::Qmm as Rung;
        let at = self.qmm_at(rows, columns, contraction, bank.bits, bank.group);
        self.qmm(&at)
            .iter()
            .map(|q| match *q {
                Qmm::Tiled { crossover_tg } => Rung::Tiled { crossover_tg },
                Qmm::Qmv => Rung::Qmv { vecs: 2 },
            })
            .collect()
    }

    fn kquant(&self) -> kernels_slang::linear::kquant::Tactic {
        KQUANT
    }

    /// The vector kernel folds by shuffles where the adapter's subgroups
    /// hold a 32-lane strip.
    fn gemm(&self, tier: kernels_slang::Tier) -> kernels_slang::linear::gemm::Tactic {
        kernels_slang::linear::gemm::Tactic { tier }
    }

    fn batch(&self, pairs: u32, experts: u32) -> kernels_slang::linear::moe::select::Batch {
        kernels_slang::linear::moe::select::Batch {
            tile: self.moe_tile_rows(pairs, experts),
            padded: self.moe_sorted_rows(pairs, experts),
            tier: kernels_slang::Tier::Baseline,
            sort_lanes: 256,
        }
    }

    fn routed_tier(&self, tier: kernels_slang::Tier) -> kernels_slang::Tier {
        tier
    }

    fn lora_tier(&self) -> kernels_slang::Tier {
        kernels_slang::Tier::Baseline
    }

    /// The adapter's tier (its subgroup prefill tile where every subgroup
    /// is 32 lanes), a decode workgroup of 256 invocations (128 measured
    /// 26-61% slower through wgpu on an RTX 4090) and prefill tiles of 16
    /// rows of 16 lanes, the 256 invocations WebGPU's default workgroup
    /// holds.
    fn sdpa(&self, tier: kernels_slang::Tier, _head_dim: u32) -> kernels_slang::attn::sdpa::Tactic {
        kernels_slang::attn::sdpa::Tactic {
            tier,
            narrow: false,
            decode_threads: 256,
            tile_rows: 16,
            tile_lanes: 16,
            split_groups: self.sdpa_split_groups(),
            split_min_keys: self.sdpa_split_min_keys(),
            prefill_work: if cfg!(target_arch = "wasm32") {
                TAB_PREFILL_WORK
            } else {
                0
            },
        }
    }

    fn binds_split(&self, lse: bool) -> bool {
        let buffers = self.storage_buffers();
        buffers == 0 || buffers >= SPLIT_STORAGE_BUFFERS + u32::from(lse)
    }

    /// The recurrences fold their norms through workgroup memory, as WGSL
    /// does everywhere; a chunked or committed recurrence holds its state in
    /// registers from 64 rows, the threshold this engine has always run at.
    fn recurrent(&self) -> kernels_slang::ssm::gdn_recurrent::Tactic {
        kernels_slang::ssm::gdn_recurrent::Tactic {
            tier: kernels_slang::Tier::Baseline,
            register_min_rows: 64,
        }
    }

    fn moe_tile_rows(&self, pairs: u32, experts: u32) -> u32 {
        Tuning::moe_tile_rows(self, pairs, experts)
    }

    fn moe_sorted_rows(&self, pairs: u32, experts: u32) -> u32 {
        Tuning::moe_sorted_rows(self, pairs, experts)
    }

    fn sdpa_tile_rows_per_request(&self) -> u32 {
        Tuning::sdpa_tile_rows_per_request(self)
    }

    fn sdpa_split_groups(&self) -> u32 {
        Tuning::sdpa_split_groups(self)
    }

    fn sdpa_split_min_keys(&self) -> u32 {
        Tuning::sdpa_split_min_keys(self)
    }
}
