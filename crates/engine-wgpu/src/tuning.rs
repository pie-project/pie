//! The hardware-dependent decisions engine-wgpu reads when it fires
//! kernels-slang's entries, and [`Tuning`], the values one load runs under.

use engine_gpu::tuning::{DecodeSplitMinKeys, MoeTileRows, SdpaTileRowsPerRequest};
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
    /// `per` times the workgroups the device holds at once, as its rates
    /// step measured them (WebGPU reports no count of an adapter's compute
    /// units: `profile::fire::RESIDENT_WORKGROUPS`).
    PerCore {
        #[constant(range(1..=64), default = 1)]
        per: u32,
    },
}

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
    /// The workgroups the device holds at once, as its profile states
    /// them (`[rates] resident_workgroups`, measured at the first load).
    resident: Option<u32>,
    /// The decodes (head width, LSE) whose split walk the adapter binds,
    /// as the load held them ([`Tuning::hold_decodes`]).
    splits: Vec<(u32, bool)>,
    /// The workgroup memory the adapter states a kernel may take.
    workgroup_bytes: u32,
}

/// The workgroup memory every WebGPU adapter allows
/// (`maxComputeWorkgroupStorageSize`'s floor).
pub const WORKGROUP_BYTES_FLOOR: u32 = 16 << 10;

impl Tuning {
    #[must_use]
    pub fn new(profile: Profile) -> Tuning {
        let resident = profile
            .rates()
            .resident_workgroups
            .map(|workgroups| workgroups.round().clamp(1.0, f64::from(u32::MAX)) as u32);
        Tuning {
            profile,
            resident,
            splits: Vec::new(),
            workgroup_bytes: WORKGROUP_BYTES_FLOOR,
        }
    }

    /// The workgroups the device holds at once, as its profile states them;
    /// `None` before the load's rates step has measured them.
    #[must_use]
    pub fn resident(&self) -> Option<u32> {
        self.resident
    }

    /// What sizes work to the device: the workgroups it holds at once; one
    /// before the rates step has measured them, in that step's own tuning,
    /// whose timed kernels read neither the qmm's fill nor a decode's split
    /// (a load's tuning states them, [`crate::rates::at_load`]).
    fn sized(&self) -> u32 {
        self.resident.unwrap_or(1)
    }

    /// With the workgroup memory the adapter states.
    #[must_use]
    pub fn with_workgroup_bytes(mut self, bytes: u32) -> Tuning {
        self.workgroup_bytes = bytes;
        self
    }

    /// Holds the decodes a load walks (`decodes`: head width, LSE) to
    /// `limit`, the storage buffers the adapter binds a shader stage, each
    /// kernel's count read from the module it is compiled from. A decode
    /// whose walk of a row's keys whole does not bind refuses the load by
    /// that kernel, its buffers and the limit; its split walk is taken
    /// where that binds too, which the load states.
    #[cfg(feature = "wgpu")]
    pub fn hold_decodes(
        &mut self,
        tier: kernels_slang::Tier,
        decodes: &[(u32, bool)],
        windowed: &[(u32, bool)],
        limit: u32,
    ) -> engine_gpu::Result<()> {
        use engine_gpu::backend::Tactics as _;
        self.splits.clear();
        // A windowed decode's ring form binds one table more: held as the
        // plain form is, its split taken only where both forms' bind.
        let forms = decodes
            .iter()
            .map(|&decode| (decode, false))
            .chain(windowed.iter().map(|&decode| (decode, true)));
        let mut unsplit: Vec<(u32, bool)> = Vec::new();
        for ((head_dim, lse), ring) in forms {
            let [whole, split] = kernels_slang::attn::sdpa::decode_kernels(
                kernels_slang::Target::Wgsl,
                self.sdpa(tier, head_dim),
                head_dim,
                lse,
                ring,
            )?
            .map(|fire| {
                let key = kernels_slang::key(fire.entrypoint, fire.tier).ok_or(
                    engine_gpu::Fault::Shader {
                        file: fire.file,
                        entrypoint: fire.entrypoint,
                        why: format!("not instantiated at the {} tier", fire.tier.name()),
                    },
                )?;
                let (own, pool) = crate::device::pipelines::storage(fire.file, key)?;
                Ok::<_, engine_gpu::Fault>((key, own, pool))
            });
            let (key, own, pool) = whole?;
            engine_gpu::storage_buffers(key, own, pool, limit)?;
            let (key, own, pool) = split?;
            let binds = engine_gpu::storage_buffers(key, own, pool, limit).is_ok();
            tracing::info!(
                kernel = key,
                own,
                pool,
                limit,
                split = binds,
                "a decode splits a row's keys where the adapter binds the split walk's storage \
                 buffers"
            );
            if !binds {
                unsplit.push((head_dim, lse));
            } else if !ring && !self.splits.contains(&(head_dim, lse)) {
                self.splits.push((head_dim, lse));
            }
        }
        self.splits.retain(|decode| !unsplit.contains(decode));
        Ok(())
    }

    #[must_use]
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    #[must_use]
    pub fn qmm_at(&self, rows: u32, n: u32, k: u32, bits: u32, group: u32) -> QmmAt {
        let tiles = u64::from(rows.div_ceil(32)) * u64::from(n.div_ceil(64));
        QmmAt {
            rows: i64::from(rows),
            fill: i64::try_from(tiles * 8 / u64::from(self.sized())).unwrap_or(i64::MAX),
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

    /// Every rung a batch between the shapes `first` and `last` (one row
    /// and the most a fire holds, the other widths the same) reaches: its
    /// region meets the shapes between them and it fits one of the two,
    /// since no rung tells rows apart ([`Qmm::fits_shape`]).
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

    #[must_use]
    pub fn sdpa_split_groups(&self) -> u32 {
        match *self.profile.get::<DecodeSplit>() {
            DecodeSplit::Fixed { groups } => groups,
            DecodeSplit::PerCore { per } => self.sized().saturating_mul(per).max(1),
        }
    }
}

/// WGSL has no subgroup tier; four strips a workgroup (kernels-slang.md §15).
const KQUANT: kernels_slang::linear::kquant::Tactic = kernels_slang::linear::kquant::Tactic {
    tier: kernels_slang::Tier::Baseline,
    slices: 4,
};

/// A tab's prefill dispatches, in rows times keys. A GPU cannot leave a
/// dispatch to draw the screen, and macOS ends a command buffer once that has
/// gone on for about 20 seconds (`Impacting Interactivity`), which costs the
/// tab its device. Smaller command buffers do not help: on an M2 Max,
/// dispatches of 95 ms are ended and these, of 48 ms, are not. A browser hides
/// its adapter from wgpu, so every tab pays for them, about 3% of a prefill.
const TAB_PREFILL_WORK: u64 = 1 << 19;

fn qmm_rung(rung: Qmm) -> kernels_slang::linear::qmm::Qmm {
    use kernels_slang::linear::qmm::Qmm as Rung;
    match rung {
        Qmm::Tiled { crossover_tg } => Rung::Tiled { crossover_tg },
        Qmm::Qmv => Rung::Qmv { vecs: 2 },
    }
}

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
        let at = self.qmm_at(rows, columns, contraction, bank.bits, bank.group);
        self.qmm(&at).into_iter().map(qmm_rung).collect()
    }

    fn qmm_rungs(
        &self,
        capacity: u32,
        columns: u32,
        contraction: u32,
        bank: kernels_slang::Bank,
    ) -> Vec<kernels_slang::linear::qmm::Qmm> {
        let at = |rows| self.qmm_at(rows, columns, contraction, bank.bits, bank.group);
        self.qmm_between(&at(1), &at(capacity.max(1)))
            .into_iter()
            .map(qmm_rung)
            .collect()
    }

    fn kquant(&self) -> kernels_slang::linear::kquant::Tactic {
        KQUANT
    }

    /// The vector kernel folds by shuffles where the adapter's subgroups
    /// hold a 32-lane strip.
    fn gemm(&self, tier: kernels_slang::Tier, _rows: u32) -> kernels_slang::linear::gemm::Tactic {
        kernels_slang::linear::gemm::Tactic {
            tier,
            vector: 0,
            wide: true,
        }
    }

    /// One tactic at every row count.
    fn gemm_tactics(
        &self,
        tier: kernels_slang::Tier,
        _capacity: u32,
    ) -> Vec<kernels_slang::linear::gemm::Tactic> {
        vec![kernels_slang::linear::gemm::Tactic {
            tier,
            vector: 0,
            wide: true,
        }]
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

    /// As the load held it: the split walk's storage buffers are within
    /// what the adapter binds a stage.
    fn binds_split(&self, head_dim: u32, lse: bool) -> bool {
        self.splits.contains(&(head_dim, lse))
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

    fn sdpa_split_groups(&self) -> u32 {
        Tuning::sdpa_split_groups(self)
    }

    fn workgroup_bytes(&self) -> u32 {
        self.workgroup_bytes
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use engine_gpu::backend::Tactics as _;

    use super::*;

    fn tuning() -> Tuning {
        let adapter = crate::chip::Device {
            name: "an adapter".to_string(),
            vendor: "intel",
            subgroup_min: None,
            cores: None,
            f16: true,
        };
        let (profile, _) = crate::chip::resolve(&adapter, None);
        Tuning::new(profile)
    }

    /// The rungs between one row and a capacity hold the rungs of every
    /// row count up to it.
    #[test]
    fn the_rungs_between_one_row_and_a_capacity_hold_every_row_counts_rungs() {
        let t = tuning();
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

    /// An adapter binding a stage 10 storage buffers (Chrome on macOS,
    /// SwiftShader) cannot bind the plain decode, 8 buffers of its own and
    /// the pool's 4 segments as its module declares them: the load is
    /// refused by that kernel, both counts and the limit.
    #[test]
    fn an_adapter_of_10_storage_buffers_refuses_the_plain_decode_by_name() {
        let mut tuning = tuning();
        let refused = tuning
            .hold_decodes(kernels_slang::Tier::Baseline, &[(64, false)], &[], 10)
            .unwrap_err();
        assert_eq!(
            refused.to_string(),
            "`attn_sdpa_decode_bf16_d_64` binds 12 storage buffers in its shader stage (8 of \
             its own and 4 pool segments) and this adapter binds a stage 10 \
             (`maxStorageBuffersPerShaderStage`)"
        );
        assert!(!tuning.binds_split(64, false));
    }

    /// At 16 the load is taken and every decode splits; between the whole
    /// walk's count and the split's the load is taken with the rows walked
    /// whole, each form by its own count.
    #[test]
    fn an_adapter_of_16_storage_buffers_loads_and_splits_by_the_same_count() {
        let decodes = [(64, false), (64, true)];
        let mut tuning = tuning();
        tuning
            .hold_decodes(kernels_slang::Tier::Baseline, &decodes, &[], 16)
            .unwrap();
        assert!(tuning.binds_split(64, false) && tuning.binds_split(64, true));
        // A decode the load did not hold is not split.
        assert!(!tuning.binds_split(128, false));

        // The split walk binds 14, its LSE form 15; the whole LSE walk 13.
        tuning
            .hold_decodes(kernels_slang::Tier::Baseline, &decodes, &[], 14)
            .unwrap();
        assert!(tuning.binds_split(64, false) && !tuning.binds_split(64, true));
        tuning
            .hold_decodes(kernels_slang::Tier::Baseline, &decodes, &[], 13)
            .unwrap();
        assert!(!tuning.binds_split(64, false) && !tuning.binds_split(64, true));
        let refused = tuning
            .hold_decodes(kernels_slang::Tier::Baseline, &decodes, &[], 12)
            .unwrap_err();
        assert!(
            matches!(
                &refused,
                engine_gpu::Fault::StorageBuffers { kernel, own: 9, pool: 4, limit: 12 }
                    if kernel == "attn_sdpa_decode_lse_bf16_d_64"
            ),
            "{refused:?}"
        );
    }

    /// A windowed decode's ring form binds one table more than the plain
    /// form: at 14 both whole walks and the plain split bind and the ring
    /// split does not, so the decode is walked whole; at 15 it splits; at
    /// 12 the ring walk is refused by name.
    #[test]
    fn a_windowed_decode_splits_where_its_ring_form_binds() {
        let decodes = [(64, false)];
        let mut tuning = tuning();
        tuning
            .hold_decodes(kernels_slang::Tier::Baseline, &decodes, &decodes, 14)
            .unwrap();
        assert!(!tuning.binds_split(64, false));
        tuning
            .hold_decodes(kernels_slang::Tier::Baseline, &decodes, &decodes, 15)
            .unwrap();
        assert!(tuning.binds_split(64, false));
        let refused = tuning
            .hold_decodes(kernels_slang::Tier::Baseline, &decodes, &decodes, 12)
            .unwrap_err();
        assert!(
            matches!(
                &refused,
                engine_gpu::Fault::StorageBuffers { kernel, own: 9, pool: 4, limit: 12 }
                    if kernel == "attn_sdpa_decode_ring_bf16_d_64"
            ),
            "{refused:?}"
        );
    }
}
