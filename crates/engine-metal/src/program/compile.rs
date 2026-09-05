//! Compile cache for guest MSL: two tiers (memory, negative), keyed after
//! splicing in the RNG include. Fast math is disabled for determinism.

use std::sync::Arc;

use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_compiler::codegen::program::KernelKind;
use eta_compiler::plan::{LibraryOp, RegionKind};
use eta_exec::{
    Backend, Bounded, CacheStats, Emitted, EmittedKernel, ExecPlan, Failure, Lookup,
    MAX_NEGATIVE_ENTRIES, MAX_PROGRAM_ENTRIES, MAX_STAGE_ENTRIES, Slot, Stages, Versions,
    cache_identity, combined_signature,
};
use eta_ir::registry::Stage as Attach;

use crate::device::Context;
use crate::error::Result;

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

/// The single-lane kernel kind, whose channels are argument slots.
const KERNEL_FUSED: KernelKind = KernelKind::Fused;

/// The grouped kernel kind, over a lane table. Handles what the single-lane
/// form cannot: more than twelve channels (Metal's last argument index
/// is 30), and vocabulary-width gathers split across a threadgroup.
const KERNEL_GROUPED: KernelKind = KernelKind::Grouped;

/// Which emitted form a compiled region is; fixed at compile time, since the
/// pipeline was built for one of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// Single-lane: status at buffer 0, each channel's cells at `7 + 2k` /
    /// `8 + 2k`, one thread.
    Fused,
    /// Grouped kernel over the lane table, a threadgroup per lane.
    Grouped,
    /// Grouped library sampler (nucleus/top-k): same eleven bindings, one
    /// threadgroup per (lane, row), requires exactly 256 threads.
    GroupedLibrary,
}

/// The include line every emitted kernel carries.
const RNG_INCLUDE: &str = "#include \"ptir_rng.generated.metal\"";

/// Device identity for the cache key: MSL compiles per-device, so no
/// separate architecture/toolkit version is needed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    /// The device's registry id, as `cache_identity` takes it.
    pub device: u64,
}

impl Target {
    /// Read the bound device's identity.
    ///
    /// # Errors
    ///
    /// [`Fault::Deviceless`](crate::Fault::Deviceless) off Apple.
    pub fn of(context: &Context) -> Result<Target> {
        #[cfg(target_vendor = "apple")]
        {
            Ok(Target {
                device: context.device().registryID(),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = context;
            Err(crate::error::Fault::Deviceless)
        }
    }
}

/// One compiled entrypoint: the library and the pipeline state a dispatch
/// binds. No `Drop`; ARC releases both when the last `Arc<Module>` goes.
pub struct Module {
    /// Kept alive so a symbolicated GPU error names the function, not an
    /// address.
    #[cfg(target_vendor = "apple")]
    #[allow(dead_code)]
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    #[cfg(target_vendor = "apple")]
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    entry: String,
}

// SAFETY: `MTLLibrary` and `MTLComputePipelineState` are documented thread-safe.
unsafe impl Send for Module {}
// SAFETY: both objects are immutable once built.
unsafe impl Sync for Module {}

impl std::fmt::Debug for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Module").field("entry", &self.entry).finish()
    }
}

impl Module {
    /// The entrypoint this module was built for.
    #[must_use]
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// The pipeline a dispatch binds.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn pipeline(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.pipeline
    }

    /// The widest threadgroup this pipeline will accept; a property of the
    /// compiled kernel (register pressure), not of the device.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn max_threads(&self) -> usize {
        self.pipeline.maxTotalThreadsPerThreadgroup()
    }

    /// Compile one owned MSL source and build the pipeline for `entry`. A
    /// rejected source is remembered in the negative tier; any other failure
    /// is not.
    #[cfg(target_vendor = "apple")]
    fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
        entry: &str,
    ) -> std::result::Result<Module, Failure> {
        use objc2_metal::MTLCompileOptions;

        let options = MTLCompileOptions::new();
        // Metal defaults fast math on; turn it off for determinism.
        set_safe_math(&options);
        let text = crate::device::ctx::nsstring(source);
        let library = device
            .newLibraryWithSource_options_error(&text, Some(&options))
            .map_err(|error| classify(entry, &error))?;
        let name = crate::device::ctx::nsstring(entry);
        let function = library
            .newFunctionWithName(&name)
            .ok_or_else(|| Failure::Deterministic {
                reason: format!(
                    "the library compiled and holds no `{entry}`; the emitter and the \
                     engine disagree about this region's entry name"
                ),
            })?;
        // The source already compiled, so a pipeline build failure is not a
        // permanent rejection.
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| classify(entry, &error))?;
        Ok(Module {
            library,
            pipeline,
            entry: entry.to_string(),
        })
    }
}

/// Turn fast math off. `setMathMode:` only exists from macOS 15 / Metal 3.2;
/// an unrecognised selector aborts the process, so `respondsToSelector:` is
/// checked first and the deprecated `setFastMathEnabled:` is the fallback.
#[cfg(target_vendor = "apple")]
fn set_safe_math(options: &objc2_metal::MTLCompileOptions) {
    use objc2::runtime::NSObjectProtocol as _;

    if options.respondsToSelector(objc2::sel!(setMathMode:)) {
        options.setMathMode(objc2_metal::MTLMathMode::Safe);
    } else {
        // Older fallback selector; turns off both halves.
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
    }
    if options.respondsToSelector(objc2::sel!(setMathFloatingPointFunctions:)) {
        options
            .setMathFloatingPointFunctions(objc2_metal::MTLMathFloatingPointFunctions::Precise);
    }
}

/// An `NSError` from the shader compiler, split into the shared vocabulary.
/// Only `Internal` is `Retryable`; everything else, including a syntax
/// error's `MTLLibraryError::Unsupported`, is treated as `Deterministic`.
#[cfg(target_vendor = "apple")]
fn classify(entry: &str, error: &objc2_foundation::NSError) -> Failure {
    use objc2_metal::MTLLibraryError;

    let reason = format!(
        "`{entry}`: {} (domain {}, code {})",
        error.localizedDescription(),
        error.domain(),
        error.code()
    );
    if error.code() == MTLLibraryError::Internal.0 as isize {
        Failure::Retryable { reason }
    } else {
        Failure::Deterministic { reason }
    }
}

/// The emitted source with its one unresolved include spliced in:
/// `newLibraryWithSource:` has no header search path, so the include cannot be
/// left for the compiler.
fn expand(source: &str) -> String {
    if !source.contains(RNG_INCLUDE) {
        return source.to_string();
    }
    source.replace(RNG_INCLUDE, &eta_compiler::codegen::rng::generate_msl_preamble())
}

/// One compiled region: the module that holds it, and which region it is.
#[derive(Debug)]
pub struct Region {
    /// Index into the plan's fused partition. The grouped emitter names its
    /// fused-region kernels at `singleton.len() + region_index`.
    pub region_index: u32,
    /// Which emitted form this region's pipeline was built from.
    pub form: Form,
    /// The compiled library and its pipeline.
    pub module: Arc<Module>,
}

impl Region {
    /// The pipeline a dispatch binds.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn pipeline(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        self.module.pipeline()
    }
}

/// What the grouped form answered for one region.
enum GroupedAnswer {
    /// The grouped kernel compiled; this is it.
    Served(Region),
    /// It did not; the clause the caller's refusal message appends.
    Declined(String),
}

/// One compiled stage: every generated region it declares, in region order.
/// Shared: two programs naming the same stage share one compiled library;
/// ARC keeps the pipeline alive for as long as any holder does.
#[derive(Debug, Clone)]
pub struct Stage {
    /// The stage's signature hash, as its plan states it.
    pub signature_hash: u64,
    /// The generated regions, in ascending `region_index`.
    pub regions: Arc<Vec<Region>>,
}

impl Stage {
    /// The region with this index, if it was compiled.
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
    /// The stage plans these were compiled from, in the same order, so a
    /// compiled program cannot drift from its plan.
    pub plans: Arc<Vec<LaunchStagePlan>>,
    /// Each stage's attachment point (`LaunchStage::kind`); `LaunchStagePlan`
    /// itself carries no `kind`.
    pub kinds: Arc<Vec<Attach>>,
}

impl Compiled {
    /// The index of the first stage with this attachment point.
    #[must_use]
    pub fn stage_of_kind(&self, kind: Attach) -> Option<usize> {
        self.kinds.iter().position(|&k| k == kind)
    }
}

/// The compile cache: the only thing in this crate that compiles guest MSL.
#[derive(Debug)]
pub struct Cache {
    programs: Bounded<u64, Compiled>,
    stages: Stages<Stage>,
    negative: Bounded<u64, String>,
    stats: CacheStats,
}

impl Default for Cache {
    fn default() -> Cache {
        Cache::new()
    }
}

impl Cache {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Cache {
        Cache {
            programs: Bounded::new(MAX_PROGRAM_ENTRIES),
            stages: Stages::new(MAX_STAGE_ENTRIES),
            negative: Bounded::new(MAX_NEGATIVE_ENTRIES),
            stats: CacheStats::default(),
        }
    }

    /// What the tiers have been doing.
    #[must_use]
    pub const fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Compile `plan`'s generated regions, or answer from a tier.
    ///
    /// `versions` carries the identity's version numbers, so a host-side
    /// bump misses rather than reusing a stale pipeline.
    ///
    /// # Errors
    ///
    /// [`Failure::Deterministic`] when the program cannot compile here —
    /// only these are remembered — and [`Failure::Retryable`] when the
    /// machine could not.
    pub fn compile(
        &mut self,
        context: &Context,
        program_hash: u64,
        plan: &ExecPlan,
        kernels: &[EmittedKernel],
        versions: Versions,
        target: Target,
    ) -> std::result::Result<Compiled, Failure> {
        if let Some(compiled) = self.programs.get(&program_hash) {
            self.stats.memory_hits += 1;
            return Ok(compiled.clone());
        }

        let program_identity = cache_identity(
            Backend::Metal,
            target.device,
            combined_signature(&plan.package.plans),
            versions,
        );
        let program_key = eta_ir::fnv1a64(program_identity.as_bytes());
        if let Some(reason) = self.negative.get(&program_key) {
            self.stats.negative_hits += 1;
            return Err(Failure::Deterministic {
                reason: reason.clone(),
            });
        }

        match self.build(context, plan, kernels, versions, target) {
            Ok(compiled) => {
                self.stages.commit();
                self.programs.insert(program_hash, compiled.clone());
                Ok(compiled)
            }
            Err(failure) => {
                // No half-stage is left behind on failure.
                self.stages.abandon();
                if let Failure::Deterministic { reason } = &failure {
                    self.negative.insert(program_key, reason.clone());
                }
                Err(failure)
            }
        }
    }

    /// Forget `program_hash`, dropping this cache's share of its modules.
    pub fn forget(&mut self, program_hash: u64) {
        self.programs.remove(&program_hash);
    }

    /// The compile proper. Installs nothing; the caller commits or abandons.
    fn build(
        &mut self,
        context: &Context,
        plan: &ExecPlan,
        kernels: &[EmittedKernel],
        versions: Versions,
        target: Target,
    ) -> std::result::Result<Compiled, Failure> {
        let index = Emitted::index(kernels).map_err(|duplicate| Failure::Deterministic {
            reason: format!(
                "the emitted kernel table names slot (kind {}, stage {}, region {}) twice; \
                 an engine cannot know which of the two the host meant",
                duplicate.kind as u32, duplicate.stage, duplicate.region
            ),
        })?;

        let mut stages = Vec::with_capacity(plan.package.plans.len());
        for (stage_index, stage_plan) in plan.package.plans.iter().enumerate() {
            let stage_index = u32::try_from(stage_index).map_err(|_| Failure::Deterministic {
                reason: "a program with more than four billion stages is not a program".into(),
            })?;
            let identity = cache_identity(
                Backend::Metal,
                target.device,
                stage_plan.signature_hash,
                versions,
            );
            // Nothing folded in beside the identity: on this plane the
            // device is the toolchain.
            let key = eta_ir::fnv1a64(identity.as_bytes());
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

            let compiled = self.build_stage(context, stage_index, stage_plan, &index)?;
            if lookup == Lookup::Miss {
                self.stages
                    .stage(key, stage_plan.identity, compiled.clone());
            }
            stages.push(compiled);
        }
        Ok(Compiled {
            stages: Arc::new(stages),
            plans: Arc::new(plan.package.plans.clone()),
            // `plans` is parallel to `package.stages`, so kinds index the same way.
            kinds: Arc::new(plan.package.stages.iter().map(|s| s.stage).collect()),
        })
    }

    /// Every generated region of one stage.
    fn build_stage(
        &mut self,
        context: &Context,
        stage_index: u32,
        plan: &LaunchStagePlan,
        index: &Emitted<'_>,
    ) -> std::result::Result<Stage, Failure> {
        let mut regions = Vec::new();
        for region_index in 0..plan.fused.len() {
            let region_index = u32::try_from(region_index).map_err(|_| Failure::Deterministic {
                reason: "a stage with more than four billion regions is not a stage".into(),
            })?;
            // A second-party region has no generated kernel; that is not a
            // compile failure.
            if plan.fused.get(region_index as usize).is_some_and(|region| {
                region.kind == RegionKind::Library(LibraryOp::SecondParty)
            }) {
                continue;
            }
            let grouped_declined =
                match self.grouped_region(context, stage_index, region_index, plan, index)? {
                    GroupedAnswer::Served(region) => {
                        regions.push(region);
                        continue;
                    }
                    GroupedAnswer::Declined(why) => why,
                };
            let (source, entry) = match index.get(KERNEL_FUSED, stage_index, region_index) {
                Slot::Kernel { source, entry } => (source, entry),
                // A declined region has no fallback path; skipping it would
                // silently run with the fire's memset zeros.
                Slot::Refused(why) => {
                    return Err(Failure::Deterministic {
                        reason: format!(
                            "stage {stage_index} region {region_index} was declined by the \
                             emitter ({why}), and the grouped form could not serve it \
                             either ({grouped_declined}); this shell runs only compiled \
                             regions, so a declined one would silently not run at all"
                        ),
                    });
                }
                Slot::Absent => {
                    return Err(Failure::Deterministic {
                        reason: format!(
                            "stage {stage_index} region {region_index} is a generated region \
                             and the host emitted nothing for it; this shell carries no \
                             emitter, so there is no slower path to fall back to"
                        ),
                    });
                }
                Slot::Malformed => {
                    return Err(Failure::Deterministic {
                        reason: format!(
                            "stage {stage_index} region {region_index} was emitted with \
                             neither a source nor a reason for declining"
                        ),
                    });
                }
            };

            let module = self.region_module(context, entry, source)?;
            regions.push(Region {
                region_index,
                form: Form::Fused,
                module,
            });
        }
        Ok(Stage {
            signature_hash: plan.signature_hash,
            regions: Arc::new(regions),
        })
    }

    /// The grouped kernel for one fused region, when it is used: the
    /// single-lane emitter refused it, it is a library sampler (nucleus/top-k,
    /// needing 256 threads), or it walks the vocabulary through an intrinsic
    /// gather. Otherwise the region stays on the single-lane kernel.
    ///
    /// # Errors
    ///
    /// A decline is not an error: it answers [`GroupedAnswer::Declined`] and
    /// the caller falls back to the single-lane path.
    fn grouped_region(
        &mut self,
        context: &Context,
        stage_index: u32,
        region_index: u32,
        plan: &LaunchStagePlan,
        index: &Emitted<'_>,
    ) -> std::result::Result<GroupedAnswer, Failure> {
        if !plan.needs.grouped_valid {
            return Ok(GroupedAnswer::Declined(if plan.error.is_empty() {
                "the plan says the grouped path cannot cover this stage, and states \
                 no reason"
                    .to_string()
            } else {
                format!(
                    "the plan says the grouped path cannot cover this stage: {}",
                    plan.error
                )
            }));
        }
        let region = plan.fused.get(region_index as usize);
        let library = matches!(
            region.map(|region| region.kind),
            Some(RegionKind::Library(LibraryOp::NucleusSample | LibraryOp::TopK))
        );
        let refused = matches!(
            index.get(KERNEL_FUSED, stage_index, region_index),
            Slot::Refused(_)
        );
        let gathers = region.is_some_and(|region| {
            region.nodes.iter().any(|&node| {
                plan.ops
                    .get(node as usize)
                    .is_some_and(|op| op.intrinsic.is_some())
            })
        });
        if !(library || refused || gathers) {
            return Ok(GroupedAnswer::Declined(
                "this shell keeps a region on the single-lane form unless the grouped \
                 one buys something: a library sampler, a refusal to route around, or \
                 an intrinsic gather. This region is none of the three"
                    .to_string(),
            ));
        }
        // The grouped table names a fused region at `singleton.len() + i`.
        let slot = match u32::try_from(plan.singleton.len())
            .ok()
            .and_then(|offset| offset.checked_add(region_index))
        {
            Some(slot) => slot,
            None => {
                return Ok(GroupedAnswer::Declined(format!(
                    "the grouped slot index overflows: {} singleton regions plus \
                     region {region_index}",
                    plan.singleton.len()
                )));
            }
        };
        let (source, entry) = match index.get(KERNEL_GROUPED, stage_index, slot) {
            Slot::Kernel { source, entry } => (source, entry),
            Slot::Refused(why) => {
                return Ok(GroupedAnswer::Declined(format!(
                    "the grouped emitter declined it too ({why})"
                )));
            }
            Slot::Absent => {
                return Ok(GroupedAnswer::Declined(format!(
                    "the emitted table has no grouped kernel at (stage {stage_index}, \
                     region {slot}), which is where this shell reads a fused region's \
                     grouped form — {} singleton regions plus region {region_index}",
                    plan.singleton.len()
                )));
            }
            Slot::Malformed => {
                return Ok(GroupedAnswer::Declined(format!(
                    "the grouped kernel at (stage {stage_index}, region {slot}) was \
                     emitted with neither a source nor a reason for declining"
                )));
            }
        };
        let module = self.region_module(context, entry, source)?;
        #[cfg(target_vendor = "apple")]
        if library && module.max_threads() < super::launch::LIBRARY_SAMPLER_THREADS {
            // Compiled and dropped; the caller falls back to the single-lane form.
            return Ok(GroupedAnswer::Declined(format!(
                "the grouped library sampler opens by declining any width but \
                 {}, and this pipeline's own limit is {}",
                super::launch::LIBRARY_SAMPLER_THREADS,
                module.max_threads()
            )));
        }
        Ok(GroupedAnswer::Served(Region {
            region_index,
            form: if library {
                Form::GroupedLibrary
            } else {
                Form::Grouped
            },
            module,
        }))
    }

    /// One region: expand its include, compile it, build its pipeline.
    fn region_module(
        &mut self,
        context: &Context,
        entry: &str,
        source: &str,
    ) -> std::result::Result<Arc<Module>, Failure> {
        let expanded = expand(source);
        #[cfg(target_vendor = "apple")]
        {
            let module = Module::build(context.device(), &expanded, entry)?;
            self.stats.compilations += 1;
            Ok(Arc::new(module))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (context, entry, expanded);
            Err(Failure::Retryable {
                reason: "this build has no Metal in it: the target is not an Apple one".into(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_replaces_the_include_and_leaves_everything_else() {
        let source = format!("// head\n{RNG_INCLUDE}\n// tail\n");
        let expanded = expand(&source);
        assert!(
            !expanded.contains(RNG_INCLUDE),
            "the include survived the expansion"
        );
        assert!(expanded.contains("// head") && expanded.contains("// tail"));
        assert!(
            expanded.len() > source.len(),
            "the rng source was spliced in"
        );
    }

    #[test]
    fn a_source_with_no_include_is_handed_over_unchanged() {
        let source = "kernel void nothing() {}\n";
        assert_eq!(expand(source), source);
    }

    /// The identity separates the two backends and the two emitters, which
    /// version independently.
    #[test]
    fn the_two_emitters_never_share_a_cache_identity() {
        use eta_compiler::codegen::program::Backend as Emitter;

        assert_ne!(
            Emitter::Metal.emitter_version(),
            Emitter::Cuda.emitter_version(),
            "the identity folds the emitter version in, so the two must differ"
        );
        assert_ne!(
            cache_identity(
                Backend::Metal,
                7,
                99,
                Versions::from_compiler(Emitter::Metal.emitter_version()),
            ),
            cache_identity(
                Backend::Cuda,
                7,
                99,
                Versions::from_compiler(Emitter::Cuda.emitter_version()),
            ),
        );
    }

}
