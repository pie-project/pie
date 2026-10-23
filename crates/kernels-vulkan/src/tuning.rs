//! The hardware-dependent decisions the Vulkan kernels read, and [`Tuning`],
//! the values one load runs under.

use profile::{Decision, Domain, Profile, Registry, Var};

pub const FACTS: &[Var] = &[
    Var::fixed("vendor", Domain::Enum(profile::vendor::NAMES)),
    // The narrowest subgroup a pipeline runs at.
    Var::fixed("subgroup", Domain::COUNT),
    // Multiprocessors or compute units; absent where no extension says.
    Var::fixed("cores", Domain::COUNT),
    Var::fixed("coopmat", Domain::Bool),
];

pub const AXES: &[&str] = &["rows"];

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
    /// The register-tiled kernel: a 4x4 output block an invocation, which
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
        use crate::linear::quant::{QMM_BK, qmm_stamped, qmv1_fits};
        let stamped = qmm_stamped(at.group, at.bits);
        let whole = at.k % i64::from(QMM_BK) == 0;
        match self {
            Qmm::Coopmat | Qmm::RegisterTiled { .. } => stamped && whole,
            Qmm::Tiled { .. } => stamped && whole && at.n % 16 == 0,
            Qmm::OneRow { .. } => {
                at.rows == 1
                    && stamped
                    && qmv1_fits(at.group, at.bits)
                    && (at.k * at.bits) % 128 == 0
                    && at.k % 8 == 0
            }
            Qmm::Qmv { .. } => true,
        }
    }
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
        .add::<SdpaTileRows>();
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

    #[must_use]
    pub fn gemm_coopmat(&self) -> bool {
        *self.profile.get::<GemmPath>() == GemmPath::Coopmat
    }

    #[must_use]
    pub fn moe_coopmat(&self) -> bool {
        *self.profile.get::<MoePath>() == MoePath::Coopmat
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
    pub fn sdpa_split_min_keys(&self) -> u32 {
        self.profile.get::<DecodeSplitMinKeys>().0
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
}
