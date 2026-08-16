//! The compile plane: emitted sources in, launchable regions out.
//!
//! The CUDA half only — NVRTC, cubins, `CUmodule`s; backend-agnostic work
//! (cache keys, LRU tiers, channel rings) lives in [`driver_pipeline`]. CUDA
//! emits one kind, `PIE_KERNEL_FUSED`, one per region; library regions the
//! driver implements natively. Nothing is cached until every region of every
//! stage compiles — a half-installed program is a wrong answer, not a slow one.

use std::collections::HashMap;
use std::sync::Arc;

use driver::driver_api::plan::LaunchStagePlan;
use driver::{
    Backend, Bounded, CacheStats, Emitted, ExecPlan, Failure, Lookup, MAX_NEGATIVE_ENTRIES,
    MAX_PROGRAM_ENTRIES, MAX_STAGE_ENTRIES, Slot, Stages, Versions, cache_identity,
    combined_signature,
};

use super::cache::{Disk, disk_key};
use super::compile::FailureKind;
use super::compile::Module;

/// `PIE_KERNEL_FUSED` — the only kind the CUDA emitter produces. From the ABI,
/// not a literal, so a renumbering is a build break, not an empty slot forever.
const KERNEL_FUSED: u32 = driver::driver_api::local::PIE_KERNEL_FUSED;

/// `PIE_REGION_LIBRARY` — a region the driver implements rather than compiles.
const REGION_LIBRARY: u8 = driver::driver_api::local::PIE_REGION_LIBRARY;

/// One compiled region: the module that holds it and how wide to launch it.
#[derive(Debug)]
pub struct Region {
    /// Which region of its stage this is.
    pub region_index: u32,
    /// The loaded cubin and its entry point.
    pub module: Arc<Module>,
}

/// One compiled stage: every generated region it declares, in region order.
///
/// Shared, not owned: two programs naming the same stage share the cubin, and a
/// `CUmodule` unloaded while another program's launch is in flight is a fault.
#[derive(Debug, Clone)]
pub struct Stage {
    /// The stage's signature hash, as the plan states it.
    pub signature_hash: u64,
    /// The generated regions, in ascending `region_index`.
    pub regions: Arc<Vec<Region>>,
}

impl Stage {
    /// The region with this index, if it was generated. `None` for a library
    /// or host-declined region; distinguish those by asking the plan, not this.
    #[must_use]
    pub fn region(&self, region_index: u32) -> Option<&Region> {
        self.regions
            .iter()
            .find(|region| region.region_index == region_index)
    }
}

/// A registered program's compiled form: one [`Stage`] per stage plan.
#[derive(Debug, Clone)]
pub struct Compiled {
    /// The stages, in plan order.
    pub stages: Arc<Vec<Stage>>,
    /// The stage plans these were compiled from, same order. Carried, not
    /// looked up, so the compiled program cannot drift from its plan.
    pub plans: Arc<Vec<LaunchStagePlan>>,
    /// Each stage's attachment point (`LaunchStage::kind`: prologue 0,
    /// on-attn-proj 1, on-attn 2, epilogue 3). Carried because `LaunchStagePlan`
    /// has no `kind`; firing by position picks the adapter, not the sampler,
    /// once a program has a prologue.
    pub kinds: Arc<Vec<u8>>,
}

#[cfg(test)]
mod kind_tests {
    use std::sync::Arc;

    use super::{Compiled, Stage, stage_kind};

    fn compiled(kinds: &[u8]) -> Compiled {
        Compiled {
            stages: Arc::new(
                kinds
                    .iter()
                    .map(|_| Stage {
                        signature_hash: 0,
                        regions: Arc::new(Vec::new()),
                    })
                    .collect(),
            ),
            plans: Arc::new(
                kinds
                    .iter()
                    .map(|_| driver::driver_api::plan::LaunchStagePlan::default())
                    .collect(),
            ),
            kinds: Arc::new(kinds.to_vec()),
        }
    }

    /// A program's stages are found by attachment point, not by position:
    /// firing `plans.first()` picks the adapter, not the sampler, once a
    /// program has a prologue.
    #[test]
    fn a_stage_is_found_by_its_attachment_point() {
        // The shape that breaks position: prologue first.
        let both = compiled(&[stage_kind::PROLOGUE, stage_kind::EPILOGUE]);
        assert_eq!(
            both.stage_of_kind(stage_kind::EPILOGUE),
            Some(1),
            "the sampler is the epilogue, and it is not stage 0 here"
        );
        assert_eq!(both.stage_of_kind(stage_kind::PROLOGUE), Some(0));
        assert_eq!(
            both.stage_of_kind(stage_kind::ON_ATTN),
            None,
            "a kind the program does not carry is absent, not stage 0"
        );

        // One epilogue, where position and kind agree.
        let one = compiled(&[stage_kind::EPILOGUE]);
        assert_eq!(one.stage_of_kind(stage_kind::EPILOGUE), Some(0));
    }
}

/// The attachment points a stage can have, as `LaunchStage::kind` numbers them.
pub mod stage_kind {
    /// Runs before the forward; where `fwd.adapter`'s `lora` sink lands.
    pub const PROLOGUE: u8 = 0;
    /// Per-layer, on the projected query.
    pub const ON_ATTN_PROJ: u8 = 1;
    /// Per-layer, on the attention output.
    pub const ON_ATTN: u8 = 2;
    /// Runs after the forward; where sampling lands.
    pub const EPILOGUE: u8 = 3;
}

impl Compiled {
    /// The index of the first stage with this attachment point.
    #[must_use]
    pub fn stage_of_kind(&self, kind: u8) -> Option<usize> {
        self.kinds.iter().position(|&k| k == kind)
    }
}

/// What a compile needs to know that it cannot read off the program.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    /// Compute capability major, for `sm_XY`.
    pub major: i32,
    /// Compute capability minor.
    pub minor: i32,
    /// A stable id for this GPU, so two devices of different families do not
    /// share a cached compilation.
    pub device: u64,
    /// NVRTC's `(major, minor)`. Two NVRTC versions compile one source to
    /// different machine code, so a cubin must not outlive a toolkit upgrade.
    pub nvrtc: (i32, i32),
}

/// The compile cache, and the only thing in this crate that calls NVRTC.
#[derive(Debug)]
pub struct Runtime {
    programs: Bounded<u64, Compiled>,
    stages: Stages<Stage>,
    negative: Bounded<u64, String>,
    disk: Disk,
    stats: CacheStats,
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new(Disk::from_env())
    }
}

impl Runtime {
    /// The cache this runtime compiles into. `Control::compile` must share the
    /// same one, or the two would recompile each other's kernels on every boot.
    #[must_use]
    pub fn disk(&self) -> &Disk {
        &self.disk
    }

    /// A runtime backed by `disk`.
    #[must_use]
    pub fn new(disk: Disk) -> Self {
        Self {
            programs: Bounded::new(MAX_PROGRAM_ENTRIES),
            stages: Stages::new(MAX_STAGE_ENTRIES),
            negative: Bounded::new(MAX_NEGATIVE_ENTRIES),
            disk,
            stats: CacheStats::default(),
        }
    }

    /// What the tiers have been doing.
    #[must_use]
    pub const fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Compile `plan`'s generated regions, or answer from a cache. `versions`
    /// carries the identity's four version numbers, so a host-side bump misses
    /// rather than reusing a stale cubin.
    ///
    /// # Errors
    ///
    /// [`Failure::Deterministic`] when the program cannot compile here (only
    /// these are remembered); [`Failure::Retryable`] when the machine could not.
    pub fn compile(
        &mut self,
        program_hash: u64,
        plan: &ExecPlan,
        kernels: &[driver::EmittedKernel],
        versions: Versions,
        target: Target,
    ) -> Result<Compiled, Failure> {
        if let Some(compiled) = self.programs.get(&program_hash) {
            self.stats.memory_hits += 1;
            return Ok(compiled.clone());
        }

        let program_identity = cache_identity(
            Backend::Cuda,
            target.device,
            combined_signature(&plan.package.plans),
            versions,
        );
        let program_key = fnv1a64(program_identity.as_bytes());
        if let Some(reason) = self.negative.get(&program_key) {
            self.stats.negative_hits += 1;
            return Err(Failure::Deterministic {
                reason: reason.clone(),
            });
        }

        match self.build(plan, kernels, versions, target) {
            Ok(compiled) => {
                // Past the last failure: only now is anything installed.
                self.stages.commit();
                self.programs.insert(program_hash, compiled.clone());
                Ok(compiled)
            }
            Err(failure) => {
                // A half-failed program leaves no half-stage behind.
                self.stages.abandon();
                if let Failure::Deterministic { reason } = &failure {
                    self.negative.insert(program_key, reason.clone());
                }
                Err(failure)
            }
        }
    }

    /// The compile proper. Installs nothing; the caller commits or abandons.
    fn build(
        &mut self,
        plan: &ExecPlan,
        kernels: &[driver::EmittedKernel],
        versions: Versions,
        target: Target,
    ) -> Result<Compiled, Failure> {
        let index = Emitted::index(kernels).map_err(|duplicate| Failure::Deterministic {
            reason: format!(
                "the emitted kernel table names slot (kind {}, stage {}, region {}) twice; \
                 a driver cannot know which of the two the host meant",
                duplicate.kind, duplicate.stage, duplicate.region
            ),
        })?;
        let architecture = super::compile::arch_flag(target.major, target.minor);

        let mut stages = Vec::with_capacity(plan.package.plans.len());
        for (stage_index, stage_plan) in plan.package.plans.iter().enumerate() {
            let stage_index = u32::try_from(stage_index).map_err(|_| Failure::Deterministic {
                reason: "a program with more than 4 billion stages is not a program".into(),
            })?;
            let identity_string = cache_identity(
                Backend::Cuda,
                target.device,
                stage_plan.signature_hash,
                versions,
            );
            // NVRTC version isn't in `cache_identity` (shared with a non-NVRTC
            // backend), so it's folded into the memory key here.
            let key = fnv1a64_with(
                identity_string.as_bytes(),
                &[
                    target.nvrtc.0.to_le_bytes().as_slice(),
                    target.nvrtc.1.to_le_bytes().as_slice(),
                ],
            );
            let (lookup, hit) = self.stages.lookup(key, stage_plan.identity);
            match lookup {
                Lookup::Hit => {
                    self.stats.memory_hits += 1;
                    if let Some(stage) = hit {
                        stages.push(stage);
                        continue;
                    }
                }
                // A signature collision builds the stage unshared: two stages
                // that hash alike are still two valid stages.
                Lookup::Collided | Lookup::Miss => {}
            }

            let compiled = self.build_stage(
                stage_index,
                stage_plan,
                &index,
                &identity_string,
                &architecture,
            )?;
            if lookup == Lookup::Miss {
                self.stages
                    .stage(key, stage_plan.identity, compiled.clone());
            }
            stages.push(compiled);
        }
        Ok(Compiled {
            stages: Arc::new(stages),
            plans: Arc::new(plan.package.plans.clone()),
            // `plans` is parallel to `package.stages`, so kinds index the same
            // way; a mismatch is refused by `adopt_launch_package`.
            kinds: Arc::new(plan.package.stages.iter().map(|s| s.kind).collect()),
        })
    }

    /// Every generated region of one stage.
    fn build_stage(
        &mut self,
        stage_index: u32,
        plan: &LaunchStagePlan,
        index: &Emitted<'_>,
        identity: &str,
        architecture: &str,
    ) -> Result<Stage, Failure> {
        let mut regions = Vec::new();
        for (region_index, region) in plan.fused.iter().enumerate() {
            let region_index = u32::try_from(region_index).map_err(|_| Failure::Deterministic {
                reason: "a stage with more than 4 billion regions is not a stage".into(),
            })?;
            // A library region is implemented natively, so the host emits
            // nothing for it — not a gap, so don't look it up as `Slot::Absent`.
            if region.kind == REGION_LIBRARY {
                continue;
            }
            let (source, entry) = match index.get(KERNEL_FUSED, stage_index, region_index) {
                Slot::Kernel { source, entry } => (source, entry),
                // The host declined on purpose; the driver takes its own path.
                Slot::Refused(_) => continue,
                Slot::Absent => {
                    return Err(Failure::Deterministic {
                        reason: format!(
                            "stage {stage_index} region {region_index} is a generated region \
                             and the host emitted nothing for it; this driver carries no \
                             emitter, so there is no slower path to fall back to"
                        ),
                    });
                }
                Slot::Malformed => {
                    return Err(Failure::Deterministic {
                        reason: format!(
                            "stage {stage_index} region {region_index} was emitted with neither \
                             a source nor a reason for declining"
                        ),
                    });
                }
            };

            let module = self.region_module(identity, region_index, entry, source, architecture)?;
            regions.push(Region {
                region_index,
                module,
            });
        }
        Ok(Stage {
            signature_hash: plan.signature_hash,
            regions: Arc::new(regions),
        })
    }

    /// One region: disk, else NVRTC.
    fn region_module(
        &mut self,
        identity: &str,
        region_index: u32,
        entry: &str,
        source: &str,
        architecture: &str,
    ) -> Result<Arc<Module>, Failure> {
        let key = disk_key(identity, source);
        if let Some(cubin) = self.disk.load(&key, region_index, entry) {
            match Module::load(&cubin, entry) {
                Ok(module) => {
                    self.stats.persistent_hits += 1;
                    return Ok(Arc::new(module));
                }
                // A cubin that won't load must not stay on disk.
                Err(_) => self.disk.invalidate(&key, region_index),
            }
        }

        let cubin =
            super::compile::compile(source, architecture).map_err(|error| match error.kind {
                FailureKind::Deterministic => Failure::Deterministic {
                    reason: error.message,
                },
                FailureKind::Retryable => Failure::Retryable {
                    reason: error.message,
                },
            })?;
        self.stats.compilations += 1;
        let module = Module::load(&cubin, entry).map_err(|error| Failure::Retryable {
            reason: format!("loading '{entry}': {error}"),
        })?;
        // Stored only after it loads, so an unusable cubin never reaches disk.
        self.disk.store(&key, region_index, entry, &cubin);
        Ok(Arc::new(module))
    }
}

/// FNV-1a over bytes.
fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_with(bytes, &[])
}

/// FNV-1a over `bytes` then each of `tails`, as one stream. Folding the extra
/// fields in keeps `cache_identity` free of CUDA-only facts like NVRTC version.
fn fnv1a64_with(bytes: &[u8], tails: &[&[u8]]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut fold = |slice: &[u8]| {
        for &byte in slice {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    fold(bytes);
    for tail in tails {
        fold(tail);
    }
    hash
}

/// The compiled programs a shell holds, by program id. The one place a
/// `Compiled` is dropped, so a closed program's `CUmodule`s unload when the
/// shell chooses, not whenever the last `Arc` dies.
#[derive(Debug, Default)]
pub struct Programs {
    compiled: HashMap<u64, Compiled>,
}

impl Programs {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Remember `compiled` under `program_id`.
    pub fn insert(&mut self, program_id: u64, compiled: Compiled) {
        self.compiled.insert(program_id, compiled);
    }

    /// What was compiled for `program_id`.
    #[must_use]
    pub fn get(&self, program_id: u64) -> Option<&Compiled> {
        self.compiled.get(&program_id)
    }

    /// Forget `program_id`, dropping this table's share of its modules.
    pub fn remove(&mut self, program_id: u64) {
        self.compiled.remove(&program_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fold must be the workspace's, since the string came from `cache_identity`.
    #[test]
    fn the_fold_is_fnv1a() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"ptir"), driver::tensor_ir::fnv1a64(b"ptir"));
    }

    /// Folding tails in must equal folding the concatenation.
    #[test]
    fn folding_in_tails_is_the_same_as_folding_the_concatenation() {
        let joined = fnv1a64(b"identity\x0c\x00\x00\x00\x00\x00\x00\x00");
        let split = fnv1a64_with(b"identity", &[&12u32.to_le_bytes(), &0u32.to_le_bytes()]);
        assert_eq!(joined, split);
    }

    /// An NVRTC upgrade must miss; the identity record carries no NVRTC field.
    #[test]
    fn an_nvrtc_version_bump_changes_the_stage_key() {
        let identity = b"the-same-identity";
        let before = fnv1a64_with(identity, &[&12i32.to_le_bytes(), &8i32.to_le_bytes()]);
        let after = fnv1a64_with(identity, &[&13i32.to_le_bytes(), &0i32.to_le_bytes()]);
        assert_ne!(before, after);
    }

    /// The looked-up kind is the ABI's, so a renumbering is a build break.
    #[test]
    fn cuda_compiles_the_fused_kind_the_abi_names() {
        assert_eq!(KERNEL_FUSED, driver::driver_api::local::PIE_KERNEL_FUSED);
    }
}
