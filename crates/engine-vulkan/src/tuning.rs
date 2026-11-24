//! The hardware-dependent decisions engine-vulkan reads when it fires
//! kernels-slang's entries, and [`Tuning`], the values one load runs under.

use engine_gpu::tuning::{DecodeSplitMinKeys, MoeTileRows, SdpaTileRowsPerRequest};
use profile::{Decision, Domain, Profile, Registry, Var};

pub const FACTS: &[Var] = &[
    Var::fixed("vendor", Domain::Enum(profile::vendor::NAMES)),
    // The narrowest subgroup a pipeline runs at.
    Var::fixed("subgroup", Domain::COUNT),
    // Multiprocessors or compute units; absent where no extension says.
    Var::fixed("cores", Domain::COUNT),
    Var::fixed("coopmat", Domain::Bool),
    // A discrete GPU (`VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`).
    Var::fixed("discrete", Domain::Bool),
];

pub const AXES: &[&str] = &["rows"];

/// The most command buffers the device's driver holds live, a fact of the
/// device stated in its committed profile where measured (past it a driver
/// may hand back a dead buffer, with no error, and fault on its first use):
/// the arm walk stops below it, the rest stated eager. 0: none stated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.max_live_command_buffers", constant(range(0..=1_000_000)), default = 0, tune = never)]
pub struct MaxLiveCommandBuffers(pub u32);

/// The most memory maps the device's host lets a process hold
/// (vm.max_map_count), stated in its profile for a host that does not let
/// the process read it (Android): the arm walk keeps its recordings' maps
/// under it. 0: none stated (read from the host where it can be).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.max_map_count", constant(range(0..=100_000_000)), default = 0, tune = never)]
pub struct MaxMapCount(pub u32);

/// The kernel a dense affine-quantised matmul runs: the first rung whose
/// shape it fits that can run. `fill` is how full the device is: eighths of
/// a wave of 32x64 output tiles over its cores, so a rung can say "fills the
/// device" in a way that holds on GPUs of any size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.qmm",
    shape(rows, fill: derived; n, k, bits, group),
    default = r#"[
        { when = "rows >= 12", use = "tiled", crossover_tg = 160 },
        { use = "qmv", vecs = 4 },
    ]"#,
    fits = Qmm::fits_shape,
    tune = search
)]
pub enum Qmm {
    /// The cooperative-matrix kernel: 64x64 output tiles on tensor cores.
    #[requires("coopmat")]
    #[partial]
    Coopmat,
    /// The register-tiled kernel: an 8x4 output block an invocation, which
    /// suits GPUs whose workgroup memory is slow next to their ALUs
    /// (Adreno, PowerVR).
    #[partial]
    RegisterTiled {
        /// Rows a workgroup covers.
        #[constant(set(32, 64), default = 32)]
        bm: u32,
        /// A 16-wide contraction block: 2.5x faster on PowerVR, faster on Adreno.
        #[constant(default = true)]
        k16: bool,
    },
    /// The tiled kernel, through workgroup memory.
    #[partial]
    Tiled {
        /// Workgroups a tiling must fill before a 32-wide column tile pays.
        #[constant(range(1..=1_000_000), default = 160)]
        crossover_tg: u32,
    },
    /// One row (decode), 128-bit loads folded with subgroup shuffles.
    #[requires("subgroup >= 32")]
    #[partial]
    OneRow {
        /// Lanes that walk one output row.
        #[constant(set(8, 16, 32), default = 32)]
        lanes: u32,
        /// Output rows one lane carries, sharing the activations it loads.
        #[constant(set(1, 2), default = 1)]
        rows: u32,
    },
    /// The vector kernel, `vecs` activation rows a workgroup.
    Qmv {
        #[constant(set(1, 2, 4, 8), default = 4)]
        vecs: u32,
    },
}

impl Qmm {
    fn fits_shape(&self, at: &QmmAt) -> bool {
        use kernels_slang::linear::{qmm, qmv};
        let (group, bits) = (
            u32::try_from(at.group).unwrap_or(0),
            u32::try_from(at.bits).unwrap_or(0),
        );
        let stamped = qmm::stamped(group, bits);
        let whole = at.k % i64::from(qmm::TILE_K) == 0;
        match self {
            Qmm::Coopmat | Qmm::RegisterTiled { .. } => stamped && whole,
            Qmm::Tiled { .. } => stamped && whole && at.n % 16 == 0,
            Qmm::OneRow { .. } => {
                at.rows == 1
                    && stamped
                    && qmv::row_fits(group, bits)
                    && (at.k * at.bits) % 128 == 0
                    && at.k % 8 == 0
            }
            Qmm::Qmv { .. } => true,
        }
    }
}

/// How unquantised matmuls read prompt rows, by the rows a fire lands: the
/// cooperative-matrix tile is 64 rows tall, so below its crossover it
/// launches a row of workgroups that each walk the whole contraction, and
/// the tiled or register-tiled kernel lands the same rows sooner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.gemm.path", shape(rows), default = "tiled", tune = search)]
pub enum GemmPath {
    Tiled,
    #[requires("coopmat")]
    Coopmat,
}

/// Cooperative-matrix tiles where the path takes them.
fn gemm_tactic(coopmat: bool) -> kernels_slang::linear::gemm::Tactic {
    kernels_slang::linear::gemm::Tactic {
        tier: if coopmat {
            kernels_slang::Tier::Coopmat
        } else {
            kernels_slang::Tier::Baseline
        },
        vector: 0,
        // The wide tile where no cooperative matrix serves: the matmul curve
        // is timed on it; a load weighs both tiles a rung.
        wide: true,
    }
}

/// How routed (mixture-of-experts) matmuls read their sorted rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.moe.path", default = "tiled", tune = search)]
pub enum MoePath {
    Tiled,
    #[requires("coopmat")]
    Coopmat,
}

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

/// Invocations a decode attention workgroup walks a row's keys with: more
/// walk more keys at once and merge more partial sums. The device's
/// workgroup memory caps it whatever the profile says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.threads", constant(set(128, 256)), default = 256, tune = search)]
pub struct DecodeThreads(pub u32);

/// Query rows a tiled attention workgroup holds, 32 lanes each: fewer rows
/// mean more workgroups and less reuse of each key. The device's invocation
/// limit caps it whatever the profile says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.tile_rows", constant(set(8, 16, 32)), default = 32, tune = search)]
pub struct SdpaTileRows(pub u32);

pub fn register(registry: &mut Registry) {
    registry
        .add::<Qmm>()
        .add::<GemmPath>()
        .add::<MoePath>()
        .add::<MoeTileRows>()
        .add::<SdpaTileRowsPerRequest>()
        .add::<DecodeSplit>()
        .add::<DecodeSplitMinKeys>()
        .add::<DecodeThreads>()
        .add::<SdpaTileRows>()
        .add::<MaxLiveCommandBuffers>()
        .add::<MaxMapCount>();
}

/// The point a dense matmul is chosen at on a device with `cores` cores.
#[must_use]
pub fn qmm_at(rows: u32, n: u32, k: u32, bits: u32, group: u32, cores: u32) -> QmmAt {
    let tiles = u64::from(rows.div_ceil(32)) * u64::from(n.div_ceil(64));
    QmmAt {
        rows: i64::from(rows),
        fill: i64::try_from(tiles * 8 / u64::from(cores.max(1))).unwrap_or(i64::MAX),
        n: i64::from(n),
        k: i64::from(k),
        bits: i64::from(bits),
        group: i64::from(group),
    }
}

/// What a device allows a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub cores: u32,
    /// The narrowest subgroup a pipeline runs at.
    pub lanes: u32,
    /// `maxComputeSharedMemorySize`.
    pub shared_bytes: u32,
    /// `maxComputeWorkGroupInvocations`.
    pub invocations: u32,
}

impl Limits {
    /// What a test assumes of a device it does not open.
    pub const ASSUMED: Limits = Limits {
        cores: 16,
        lanes: 32,
        shared_bytes: 32 << 10,
        invocations: 1024,
    };
}

#[derive(Clone, Debug)]
pub struct Tuning {
    profile: Profile,
    limits: Limits,
}

impl Tuning {
    #[must_use]
    pub fn new(profile: Profile, limits: Limits) -> Tuning {
        Tuning { profile, limits }
    }

    #[must_use]
    pub fn limits(&self) -> Limits {
        self.limits
    }

    #[must_use]
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    #[must_use]
    pub fn qmm_at(&self, rows: u32, n: u32, k: u32, bits: u32, group: u32) -> QmmAt {
        qmm_at(rows, n, k, bits, group, self.limits.cores)
    }

    #[must_use]
    pub fn qmm(&self, at: &QmmAt) -> Vec<Qmm> {
        self.profile
            .ladder::<Qmm>()
            .candidates(at)
            .copied()
            .collect()
    }

    /// Every rung a batch between the shapes `first` and `last` (one row
    /// and the most a fire holds, the other widths the same) reaches: its
    /// region meets the shapes between them and it fits one of the two,
    /// since a rung tells rows apart only at one row ([`Qmm::fits_shape`]).
    #[must_use]
    pub fn qmm_between(&self, first: &QmmAt, last: &QmmAt) -> Vec<Qmm> {
        use profile::Point as _;
        let cell: Vec<profile::Set> = (0..Qmm::spec().vars.len())
            .map(|var| match (first.coord(var), last.coord(var)) {
                (Some(lo), Some(hi)) => profile::Set::span(lo, hi),
                _ => unreachable!("a `kernel.qmm` point states every variable"),
            })
            .collect();
        self.profile
            .ladder::<Qmm>()
            .meeting(&cell)
            .filter(|rung| rung.fits(first) || rung.fits(last))
            .copied()
            .collect()
    }

    /// Whether a dense matmul over `rows` rows takes cooperative matrices.
    #[must_use]
    pub fn gemm_coopmat(&self, rows: u32) -> bool {
        let at = GemmPathAt {
            rows: i64::from(rows),
        };
        *self.profile.pick::<GemmPath>(&at) == GemmPath::Coopmat
    }

    /// Every path a dense matmul of 1 to `capacity` rows takes, in the
    /// profile's order.
    #[must_use]
    pub fn gemm_paths(&self, capacity: u32) -> Vec<GemmPath> {
        let rows = [profile::Set::span(1, i64::from(capacity.max(1)))];
        self.profile
            .ladder::<GemmPath>()
            .reached(&rows)
            .into_iter()
            .copied()
            .collect()
    }

    #[must_use]
    pub fn moe_coopmat(&self) -> bool {
        *self.profile.get::<MoePath>() == MoePath::Coopmat
    }

    /// The profile's, halved until 32 lanes a row fit the invocation limit.
    #[must_use]
    pub fn sdpa_tile_rows(&self) -> u32 {
        let fits = (self.limits.invocations / 32).max(8);
        let mut rows = self.profile.get::<SdpaTileRows>().0;
        while rows > fits && rows > 8 {
            rows /= 2;
        }
        rows
    }

    /// The profile's, or 128 where 256 would pass the workgroup memory (one
    /// partial row a 32-lane subgroup) or the invocation limit.
    #[must_use]
    pub fn sdpa_decode_threads(&self, head_dim: u32) -> u32 {
        let want = self.profile.get::<DecodeThreads>().0;
        let need = |threads: u32| (threads / 32) * (head_dim * 4 + 8);
        if want > 128 && (need(want) > self.limits.shared_bytes || want > self.limits.invocations) {
            128
        } else {
            want
        }
    }

    #[must_use]
    pub fn sdpa_split_groups(&self) -> u32 {
        match *self.profile.get::<DecodeSplit>() {
            DecodeSplit::Fixed { groups } => groups,
            DecodeSplit::PerCore { per } => self.limits.cores.saturating_mul(per).max(1),
        }
    }

    #[must_use]
    pub fn sort_lanes(&self) -> u32 {
        if self.limits.invocations >= 1024 {
            1024
        } else {
            256
        }
    }
}

impl engine_gpu::backend::Tactics for Tuning {
    fn profile(&self) -> &profile::Profile {
        Tuning::profile(self)
    }

    fn qmm_ladder(
        &self,
        rows: u32,
        columns: u32,
        contraction: u32,
        bank: kernels_slang::Bank,
    ) -> Vec<kernels_slang::linear::qmm::Qmm> {
        crate::encode::qmm_ladder(self, rows, columns, contraction, bank)
    }

    fn qmm_rungs(
        &self,
        capacity: u32,
        columns: u32,
        contraction: u32,
        bank: kernels_slang::Bank,
    ) -> Vec<kernels_slang::linear::qmm::Qmm> {
        crate::encode::qmm_rungs(self, capacity, columns, contraction, bank)
    }

    /// The subgroup tier where every subgroup holds a 32-lane strip; four
    /// strips a workgroup, the 4090's best at every scheme and row count
    /// measured (kernels-slang.md §15).
    fn kquant(&self) -> kernels_slang::linear::kquant::Tactic {
        kernels_slang::linear::kquant::Tactic {
            tier: if self.limits.lanes >= 32 {
                kernels_slang::Tier::Subgroup
            } else {
                kernels_slang::Tier::Baseline
            },
            slices: 4,
        }
    }

    /// Cooperative-matrix tiles where the profile takes them at `rows`.
    fn gemm(&self, _tier: kernels_slang::Tier, rows: u32) -> kernels_slang::linear::gemm::Tactic {
        gemm_tactic(self.gemm_coopmat(rows))
    }

    /// The tactic of each path the profile takes over those rows.
    fn gemm_tactics(
        &self,
        _tier: kernels_slang::Tier,
        capacity: u32,
    ) -> Vec<kernels_slang::linear::gemm::Tactic> {
        let mut tactics = Vec::new();
        for path in self.gemm_paths(capacity) {
            let tactic = gemm_tactic(path == GemmPath::Coopmat);
            if !tactics.contains(&tactic) {
                tactics.push(tactic);
            }
        }
        tactics
    }

    fn batch(&self, pairs: u32, experts: u32) -> kernels_slang::linear::moe::select::Batch {
        kernels_slang::linear::moe::select::Batch {
            tile: self.moe_tile_rows(pairs, experts),
            padded: self.moe_sorted_rows(pairs, experts),
            tier: if self.moe_coopmat() {
                kernels_slang::Tier::Coopmat
            } else {
                kernels_slang::Tier::Baseline
            },
            sort_lanes: self.sort_lanes(),
        }
    }

    fn routed_tier(&self, _tier: kernels_slang::Tier) -> kernels_slang::Tier {
        kernels_slang::Tier::Baseline
    }

    fn lora_tier(&self) -> kernels_slang::Tier {
        kernels_slang::Tier::Subgroup
    }

    /// Prefill tiles 16 rows of 16 lanes folding by shuffles where every
    /// subgroup holds a row and the workgroup its 256 invocations (3.5x the
    /// 32-lane tile on an RTX 4090), else 32 lanes a row through workgroup
    /// memory.
    fn sdpa(&self, _tier: kernels_slang::Tier, head_dim: u32) -> kernels_slang::attn::sdpa::Tactic {
        let limits = self.limits;
        let shuffles = limits.lanes >= 16 && limits.invocations >= 256;
        kernels_slang::attn::sdpa::Tactic {
            tier: kernels_slang::Tier::Subgroup,
            narrow: limits.lanes < 32,
            decode_threads: self.sdpa_decode_threads(head_dim),
            tile_rows: if shuffles { 16 } else { self.sdpa_tile_rows() },
            tile_lanes: if shuffles { 16 } else { 32 },
            split_groups: self.sdpa_split_groups(),
            split_min_keys: self.sdpa_split_min_keys(),
            prefill_work: 0,
        }
    }

    /// The recurrences fold their norms through subgroups, which every
    /// Vulkan device this engine runs on has, any width; a register-state
    /// recurrence runs at every row count.
    fn recurrent(&self) -> kernels_slang::ssm::gdn_recurrent::Tactic {
        kernels_slang::ssm::gdn_recurrent::Tactic {
            tier: kernels_slang::Tier::Subgroup,
            register_min_rows: 0,
        }
    }

    fn sdpa_split_groups(&self) -> u32 {
        Tuning::sdpa_split_groups(self)
    }

    fn workgroup_bytes(&self) -> u32 {
        self.limits.shared_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_ladder_tiles_from_twelve_rows_and_folds_below() {
        let mut r = Registry::new("vulkan-test", FACTS, AXES);
        register(&mut r);
        let t = Tuning::new(r.defaults(), Limits::ASSUMED);
        let at = |rows| t.qmm_at(rows, 4096, 4096, 4, 64);
        assert_eq!(t.qmm(&at(12))[0], Qmm::Tiled { crossover_tg: 160 });
        assert_eq!(t.qmm(&at(11))[0], Qmm::Qmv { vecs: 4 });
        // An odd contraction leaves the tiled rung for the vector kernel.
        assert_eq!(
            t.qmm(&t.qmm_at(64, 4096, 4100, 4, 64))[0],
            Qmm::Qmv { vecs: 4 }
        );
    }

    /// The rungs between one row and a capacity hold the rungs of every
    /// row count up to it.
    #[test]
    fn the_rungs_between_one_row_and_a_capacity_hold_every_row_counts_rungs() {
        let mut r = Registry::new("vulkan-test", FACTS, AXES);
        register(&mut r);
        let t = Tuning::new(r.defaults(), Limits::ASSUMED);
        for (n, k) in [(4096, 4096), (1024, 4100), (64, 64)] {
            let at = |rows| t.qmm_at(rows, n, k, 4, 64);
            for capacity in [1, 11, 12, 64, 300] {
                let between = t.qmm_between(&at(1), &at(capacity));
                for rows in 1..=capacity {
                    for rung in t.qmm(&at(rows)) {
                        assert!(
                            between.contains(&rung),
                            "{rows} rows of {n} columns over {k} reach {rung:?}, which the \
                             rungs up to {capacity} rows leave out"
                        );
                    }
                }
            }
        }
    }

    /// A load whose rows cross a measured `kernel.gemm.path` crossover
    /// names the pipelines of both paths: every kernel a fire of any row
    /// count up to its capacity takes is one its walk named. A load whose
    /// rows stay below the crossover names no cooperative-matrix pipeline.
    #[test]
    fn a_row_crossover_inside_the_load_names_both_paths_pipelines() {
        use engine_gpu::backend::Tactics as _;
        use kernels_slang::linear::gemm;

        let mut r = Registry::new("vulkan-test", FACTS, AXES);
        register(&mut r);
        let (profile, report) = r.read(&[profile::Candidate {
            source: profile::Source::User("crossover".into()),
            text: "api = 1\n[target]\nassumes = { coopmat = true }\n[kernel]\n\
                   gemm.path = [{ when = \"rows < 96\", use = \"tiled\" }, \"coopmat\"]\n"
                .into(),
        }]);
        assert!(report.rejected.is_empty(), "{:?}", report.rejected);
        let t = Tuning::new(profile, Limits::ASSUMED);
        let tier = kernels_slang::Tier::Coopmat;
        let kernels = |tactic| {
            gemm::kernels(
                kernels_slang::Target::Spirv,
                tactic,
                "linear.matmul",
                1024,
                1024,
            )
            .unwrap()
            .into_iter()
            .map(|fire| (fire.entrypoint, fire.tier))
            .collect::<Vec<_>>()
        };
        let coopmat = ("linear_gemm_tiled_bm_64_bf16", kernels_slang::Tier::Coopmat);
        for capacity in [1, 64, 95, 96, 128, 256] {
            let named: Vec<_> = t
                .gemm_tactics(tier, capacity)
                .into_iter()
                .flat_map(kernels)
                .collect();
            for rows in 1..=capacity {
                for kernel in kernels(t.gemm(tier, rows)) {
                    assert!(
                        named.contains(&kernel),
                        "a fire of {rows} rows takes {kernel:?}, which a load of up to \
                         {capacity} rows did not name"
                    );
                }
            }
            assert_eq!(named.contains(&coopmat), capacity >= 96, "{capacity} rows");
        }
    }
}
