//! The compile plane: emitted sources in, launchable regions out, cached at
//! three tiers.
//!
//! # What this owns and what it deliberately does not
//!
//! Everything about a PTIR program that is *not* CUDA lives in
//! [`driver_pipeline`]: adopting the launch package, indexing the emitted
//! table, the cache keys, the LRU tiers, the channel rings, the reference
//! pass. This file is the CUDA half and only the CUDA half — NVRTC, cubins,
//! `CUmodule`s — and the fact that it is this short is the claim the shared
//! crate was extracted to make.
//!
//! # Why CUDA only ever compiles fused regions
//!
//! The host emitter's CUDA arm emits exactly one kind of kernel:
//! `PIE_KERNEL_FUSED`, one per generated region, named
//! `ptir_fused_{signature:016x}_r{region}`. It emits no readiness kernel, no
//! commit kernel and no grouped kernel, because on CUDA those are prebuilt —
//! and the fused kernel is already lane-parallel (`dispatch_lane = blockIdx.x`,
//! one CTA per lane), which is what Metal needs a separate grouped emission
//! for. So [`Runtime::compile`] walks one kind, and a region that is a
//! *library* region (nucleus, top-k, sort, scan, matmul) is not compiled here
//! at all: the driver implements those natively and the host emits nothing for
//! them.
//!
//! # The three tiers, and why a miss at each one is different
//!
//! 1. **Program cache**, keyed on `program_hash`. A program is registered once
//!    and bound many times; this is what makes the second bind free.
//! 2. **Stage cache**, keyed on the identity hash with a second identity
//!    compared on hit. Two programs that share a stage share its cubin.
//! 3. **Disk cache**, keyed on the identity *plus the source fingerprint*. See
//!    [`disk`](super::disk) for why that last part is not optional.
//!
//! and a fourth that is not a tier but an answer: the **negative cache**, for
//! compiles that failed deterministically. Recompiling a program NVRTC will
//! reject again, once per fire, is the difference between slow and unusable.
//!
//! # Assembly happens past the last failure
//!
//! Nothing is installed into any cache until every region of every stage has
//! compiled. A program that fails halfway leaves the caches exactly as it
//! found them, because the alternative — a half-installed program that a later
//! bind finds and believes — is a wrong answer rather than a slow one.

use std::collections::HashMap;
use std::sync::Arc;

use driver::driver_api::plan::LaunchStagePlan;
use driver::{
    Backend, Bounded, CacheStats, Emitted, ExecPlan, Failure, Lookup, MAX_NEGATIVE_ENTRIES,
    MAX_PROGRAM_ENTRIES, MAX_STAGE_ENTRIES, Slot, Stages, Versions, cache_identity,
    combined_signature,
};

use super::disk::{Disk, disk_key};
use super::module::Module;
use super::nvrtc::{self, FailureKind};

/// `PIE_KERNEL_FUSED` — the only kind the CUDA emitter produces.
///
/// Taken from the ABI rather than written as a literal: the numbering is the
/// host's, and a driver that hardcodes it looks up an empty slot forever after
/// a renumbering instead of failing at the seam.
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
/// Shared rather than owned because two programs that name the same stage
/// share the cubin — that is the whole point of the stage tier — and a
/// `CUmodule` unloaded while another program's launch is in flight is a fault
/// inside the driver rather than an error anybody can report.
#[derive(Debug, Clone)]
pub struct Stage {
    /// The stage's signature hash, as the plan states it.
    pub signature_hash: u64,
    /// The generated regions, in ascending `region_index`.
    pub regions: Arc<Vec<Region>>,
}

impl Stage {
    /// The region with this index, if it was generated.
    ///
    /// `None` for a library region — those are not compiled here — and for a
    /// region the host declined. A caller must distinguish those two by asking
    /// the plan, not by asking this.
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
    /// The stage plans these were compiled FROM, in the same order.
    ///
    /// Carried rather than looked up, and that is the point: firing needs
    /// the plan — its ops say which channels the stage reads and puts, its
    /// value types size the scratch — and a driver that kept the two in
    /// separate tables keyed by the same id would have two things to keep
    /// in step. A compiled program that could disagree with the plan it
    /// came from is exactly the drift `Compiled` exists to make
    /// impossible.
    pub plans: Arc<Vec<LaunchStagePlan>>,
    /// Each stage's ATTACHMENT POINT — `PieLaunchStage::kind`: Prologue
    /// 0, OnAttnProj 1, OnAttn 2, Epilogue 3.
    ///
    /// Carried because `LaunchStagePlan` has no `kind` field and the
    /// package's `stages` table is not kept: the driver could see how
    /// many stages a program had and not what any of them WAS. It fired
    /// `plans.first()`, which is the epilogue only by the accident that
    /// no program in the tree has a prologue — the moment one does
    /// (`fwd.adapter` puts its `lora` sink there), `first()` is the
    /// adapter and the sampler never runs.
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
                    .map(|_| Stage { signature_hash: 0, regions: Arc::new(Vec::new()) })
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

    /// A program's stages are found by what they ARE, not by position.
    ///
    /// `run_program` fired `plans.first()` because `Compiled` had no
    /// kinds to ask. That is the epilogue only while no program has a
    /// prologue — and `fwd.adapter` puts its `lora` sink in one, so the
    /// first adapter-carrying program would have fired its ADAPTER as
    /// the sampler and never run the sampler at all.
    #[test]
    fn a_stage_is_found_by_its_attachment_point() {
        // The shape that breaks position: prologue FIRST.
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

        // Today's shape: one epilogue, where position and kind agree —
        // which is exactly why the bug was invisible.
        let one = compiled(&[stage_kind::EPILOGUE]);
        assert_eq!(one.stage_of_kind(stage_kind::EPILOGUE), Some(0));
    }
}

/// The attachment points a stage can have, as `PieLaunchStage::kind`
/// numbers them.
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
    /// The cache this runtime compiles into.
    ///
    /// `Control::compile` wants one too, and it must be the SAME one: the
    /// control kernels and the program kernels are cached by the same
    /// key scheme, so a second directory would recompile both on every
    /// boot and neither would ever hit.
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

    /// Compile `plan`'s generated regions, or answer from a cache.
    ///
    /// `kernels` is the host's emitted table, already indexed. `versions`
    /// carries the four numbers the identity is keyed on — the emitter version
    /// among them, taken from the registration rather than hardcoded, so a
    /// host-side bump misses instead of reusing a stale cubin.
    ///
    /// # Errors
    ///
    /// [`Failure::Deterministic`] when the program cannot compile on this
    /// driver — a malformed emitted table, a source NVRTC rejects — and
    /// [`Failure::Retryable`] when the machine could not, this time. Only the
    /// first is remembered.
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
                // A program that failed halfway must leave no half-stage
                // behind for the next program to find and believe.
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
        let architecture = nvrtc::arch_flag(target.major, target.minor);

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
            // The NVRTC version is not in `cache_identity` -- that record is
            // shared with a backend that has no NVRTC -- so it is folded into
            // the memory key here, where it is a CUDA fact.
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
                // A signature collision builds the stage UNSHARED rather than
                // rejecting the program: two stages that hash alike are still
                // two valid stages, and the C++ refused the second one.
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
            // `plans` is parallel to `package.stages` — the plans are the
            // stages' own, in the same order — so the kinds index the
            // same way. A missing entry would be a package whose two
            // tables disagree, which `adopt_launch_package` refuses.
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
            // A library region -- nucleus, top-k, sort, scan, matmul -- is
            // implemented by the driver natively, so the host emits nothing
            // for it. That is not a gap in the table and must not be looked up
            // as one: `Slot::Absent` here would refuse a program that is fine.
            if region.kind == REGION_LIBRARY {
                continue;
            }
            let (source, entry) = match index.get(KERNEL_FUSED, stage_index, region_index) {
                Slot::Kernel { source, entry } => (source, entry),
                // The host declined ON PURPOSE and said why. Not a failure:
                // the driver takes its own path for this region. The C++
                // collapsed this into "no kernel" and could not tell the two
                // apart.
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
                // A cubin that will not load is a cubin that must not stay on
                // disk: the alternative is paying this read, and this failure,
                // on every launch forever.
                Err(_) => self.disk.invalidate(&key, region_index),
            }
        }

        let cubin = nvrtc::compile(source, architecture).map_err(|error| match error.kind {
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
        // Stored only after it has been proven loadable, so a cubin that
        // cannot be used never reaches the disk in the first place.
        self.disk.store(&key, region_index, entry, &cubin);
        Ok(Arc::new(module))
    }
}

/// FNV-1a over bytes.
fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_with(bytes, &[])
}

/// FNV-1a over `bytes` followed by each of `tails`, as one stream.
///
/// Folding the extra fields in rather than formatting them into the string
/// keeps `cache_identity`'s record — which is shared with a backend that has
/// no NVRTC — free of CUDA facts.
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

/// The compiled programs a shell holds, by program id.
///
/// A thin map rather than a type: what it exists for is to be the one place a
/// `Compiled` is dropped, so the `CUmodule`s a closed program owns are unloaded
/// at a point the shell chose rather than whenever the last `Arc` happens to
/// die.
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

    /// The fold must be the one the rest of the workspace uses, since the
    /// string being folded came from `cache_identity`.
    #[test]
    fn the_fold_is_fnv1a() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(
            fnv1a64(b"ptir"),
            driver::tensor_ir::fnv1a64(b"ptir")
        );
    }

    /// Folding the tails in must be identical to folding the concatenation,
    /// or the key depends on how it was assembled rather than on what is in it.
    #[test]
    fn folding_in_tails_is_the_same_as_folding_the_concatenation() {
        let joined = fnv1a64(b"identity\x0c\x00\x00\x00\x00\x00\x00\x00");
        let split = fnv1a64_with(b"identity", &[&12u32.to_le_bytes(), &0u32.to_le_bytes()]);
        assert_eq!(joined, split);
    }

    /// An NVRTC upgrade must miss. The identity record has no NVRTC field --
    /// it is shared with a backend that has none -- so this is the only thing
    /// standing between a toolkit bump and a cubin compiled by the old one.
    #[test]
    fn an_nvrtc_version_bump_changes_the_stage_key() {
        let identity = b"the-same-identity";
        let before = fnv1a64_with(identity, &[&12i32.to_le_bytes(), &8i32.to_le_bytes()]);
        let after = fnv1a64_with(identity, &[&13i32.to_le_bytes(), &0i32.to_le_bytes()]);
        assert_ne!(before, after);
    }

    /// The kind this driver looks up is the ABI's, not a literal, so a
    /// renumbering is a build break rather than an empty table at run time.
    #[test]
    fn cuda_compiles_the_fused_kind_the_abi_names() {
        assert_eq!(
            KERNEL_FUSED,
            driver::driver_api::local::PIE_KERNEL_FUSED
        );
    }
}
