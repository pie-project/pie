//! Compiling a guest program for this device: emitted MSL in, a pipeline per
//! generated region out, held in two tiers.
//!
//! **THIS IS A SECOND `newLibraryWithSource:` CALLER AND THAT IS
//! DELIBERATE.** [`Pipelines`](crate::device::Pipelines) compiles the
//! SHIPPED shaders and is keyed by `(&'static str file, &'static str
//! entrypoint)` off `kernels_metal::SOURCES`; a guest program's MSL is an
//! owned `String` produced this second by `tensor-compiler`, with no file
//! and no static name. Sharing that cache would mean either leaking every
//! guest entry name to `'static` or keying it by something it is not. So
//! this module compiles its own, and the two share only the framework — the
//! same argument the CUDA sibling makes about not reaching into
//! `kernels_cuda::jit`.
//!
//! # The emitted source is not complete until this module completes it
//!
//! Every M2 kernel `tensor_compiler::codegen::metal` emits begins with
//! `RUNTIME_TEMPLATE`, and the template's second line is
//! `#include "ptir_rng.generated.metal"`. `newLibraryWithSource:` has no
//! header search path, so the compile fails with `fatal error:
//! 'ptir_rng.generated.metal' file not found` — measured on an M1 Max
//! against all five parity subjects. The emitter leaves the include
//! deliberately (its own doc says the expansion is the driver's), and the
//! file it names is one `kernels-metal` already ships, so the splice is a
//! lookup in `kernels_metal::source` rather than a second copy of a
//! generated table. **The expansion happens BEFORE the source reaches the
//! cache key**: two guests whose only difference was the rng text would
//! otherwise share one key.
//!
//! # There is no persistent tier, and the reason is not laziness
//!
//! The CUDA sibling has three: memory, disk, negative. Its disk tier stores
//! a CUBIN — a device image `cuModuleLoadData` reloads — under a key folded
//! from the identity and the emitted source. Metal has no counterpart:
//! `newLibraryWithSource:` yields a live `MTLLibrary` with no serializable
//! image, and the nearest thing, `MTLBinaryArchive`, archives compiled
//! PIPELINES keyed to the device AND the driver build, is written through
//! `serializeToURL:` rather than read back as bytes, and does not fit
//! `Disk::{load, store, invalidate}`'s `Option<Vec<u8>>` shape at all. So
//! this cache is memory and negative, and a fresh process compiles. What it
//! costs is measured rather than guessed: the five parity subjects are
//! 42–47 KB of MSL each and compile in well under a second apiece on an M1
//! Max, against a CUDA disk tier that exists because NVRTC is slow.
//!
//! # Fast math is off, and that is a determinism clause
//!
//! The CUDA half passes `--fmad=false --prec-div=true --prec-sqrt=true` and
//! its own doc calls that a determinism clause rather than a tuning flag:
//! the channel plane promises bit-for-bit agreement with a host interpreter
//! that has no FMA, and `program_parity` is the test that would fail. Metal's
//! DEFAULT compile options enable fast math, so a metal shell that passed
//! `None` — as [`crate::device::library`] does for the shipped shaders,
//! which are diffed against no interpreter — would silently lose that
//! promise on the first float-touching guest. The options object is
//! constructed here for that reason and nothing else.

use std::sync::Arc;

use driver::driver_api::program::{KernelKind, LaunchStagePlan, LibraryOp, RegionKind};
use driver::tensor_ir::registry::Stage as Attach;
use driver::{
    Backend, Bounded, CacheStats, Emitted, EmittedKernel, ExecPlan, Failure, Lookup,
    MAX_NEGATIVE_ENTRIES, MAX_PROGRAM_ENTRIES, MAX_STAGE_ENTRIES, Slot, Stages, Versions,
    cache_identity, combined_signature,
};

use crate::device::Context;
use crate::error::Result;

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

/// The only kernel kind this shell runs. The Metal emitter also produces
/// singleton, grouped, readiness and commit kernels into the same table;
/// this plane binds none of them (readiness and commit are host-side by
/// ruling — build log 15 and 18 — and the grouped forms need device
/// addresses this shell does not hand out), so the lookup names one kind and
/// the extras are ignored rather than half-bound.
const KERNEL_FUSED: KernelKind = KernelKind::Fused;

/// The include line every emitted kernel carries, and the source that
/// answers it.
const RNG_INCLUDE: &str = "#include \"ptir_rng.generated.metal\"";

/// Where the rng source lives in `kernels-metal`'s shipped table.
const RNG_SOURCE: &str = "ptir/ptir_rng.generated.metal";

/// What this device is, for the cache identity.
///
/// **A TARGET IS A DEVICE HERE, NOT AN ARCHITECTURE AND A TOOLKIT.** The
/// CUDA twin carries a compute capability (which selects `-arch=sm_XX`) and
/// NVRTC's own version (which changes the cubin for identical source). MSL
/// is compiled BY the device it will run on, through the framework that
/// shipped with the OS, so there is no architecture flag to choose and no
/// separate toolkit version to fold in — the device's registry id is the
/// whole of it, and a machine with two GPUs keys them apart.
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

/// One compiled entrypoint: the library that holds it and the pipeline state
/// a dispatch binds.
///
/// **NO `Drop`, AND THAT IS THE PLATFORM RATHER THAN AN OVERSIGHT.** The
/// CUDA twin ends in `cuModuleUnload`, and its plane's `close_program`
/// refuses to unload a module under a live instance because that is a launch
/// into freed machine code. ARC releases both objects here when the last
/// `Arc<Module>` goes, and a `Session` holding a `Region` holds the
/// pipeline through it — so the ordering is enforced by the type rather than
/// by the refusal. The refusal stays anyway: closing a program with live
/// instances is still a caller's bug.
pub struct Module {
    /// Held for its LIFETIME and nothing else: a `MTLComputePipelineState`
    /// is independent of the library that produced it, but keeping the
    /// library alive beside it is what makes a symbolicated GPU error name
    /// the function rather than an address.
    #[cfg(target_vendor = "apple")]
    #[allow(dead_code)]
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    #[cfg(target_vendor = "apple")]
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    entry: String,
}

// SAFETY: `MTLLibrary` and `MTLComputePipelineState` are documented
// thread-safe; the same argument `device::Buffer`'s own `Send` makes.
unsafe impl Send for Module {}
// SAFETY: as above — the two objects are immutable once built.
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

    /// Compile one owned MSL source and build the pipeline for `entry`.
    ///
    /// The two failure kinds are the shared vocabulary's, and the split is
    /// what the negative tier is for: a source the compiler REJECTED will be
    /// rejected forever on this device, so it is remembered; anything else
    /// was the moment (an out-of-memory, a device that went away) and must
    /// not be.
    #[cfg(target_vendor = "apple")]
    fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
        entry: &str,
    ) -> std::result::Result<Module, Failure> {
        use objc2_metal::MTLCompileOptions;

        let options = MTLCompileOptions::new();
        // **THE DETERMINISM CLAUSE.** See the module doc: Metal's default is
        // fast math ON, and the channel plane is diffed against a host
        // interpreter with no fused multiply-add.
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
                     driver disagree about this region's entry name"
                ),
            })?;
        // Through `classify` rather than straight to `Deterministic`: the
        // source already compiled, so a pipeline that will not build is the
        // device declining to specialize it right now — an out-of-memory, a
        // device that went away — and remembering that forever would retire a
        // valid program over one bad moment.
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

/// Turn fast math off on a compile options object, in both of its halves.
///
/// **THE SELECTOR IS ASKED FOR RATHER THAN ASSUMED, AND THAT IS NOT
/// DEFENSIVENESS.** `objc2-metal` 0.3 publishes BOTH spellings of this
/// switch — `setFastMathEnabled:`, which its headers deprecate in favour of
/// `setMathMode:`, and `setMathMode:` itself. The crate does no availability
/// checking, and `setMathMode:` is only on `MTLCompileOptions` from macOS 15
/// / Metal 3.2 onwards: sending it to an older framework is an unrecognised
/// selector, which raises an Objective-C exception and aborts the process
/// rather than returning an error this shell could report. So the new
/// spelling is preferred where the class answers to it, and the deprecated
/// one is the fallback.
///
/// Two settings, because the CUDA clause this mirrors is two flags.
/// [`MTLMathMode::Safe`](objc2_metal::MTLMathMode::Safe) is `--fmad=false`:
/// no unsafe floating-point rewrites, so no contracted multiply-add.
/// [`MTLMathFloatingPointFunctions::Precise`](objc2_metal::MTLMathFloatingPointFunctions::Precise)
/// is `--prec-div=true --prec-sqrt=true`: the `metal::precise` variants
/// rather than the `metal::fast` ones the language standard defaults to.
/// The deprecated setter covers both at once, which is why its own header
/// comment mentions the high-precision math variants.
#[cfg(target_vendor = "apple")]
fn set_safe_math(options: &objc2_metal::MTLCompileOptions) {
    use objc2::runtime::NSObjectProtocol as _;

    if options.respondsToSelector(objc2::sel!(setMathMode:)) {
        options.setMathMode(objc2_metal::MTLMathMode::Safe);
    } else {
        // The one call the older framework has, and it turns off both halves.
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
    }
    if options.respondsToSelector(objc2::sel!(setMathFloatingPointFunctions:)) {
        options
            .setMathFloatingPointFunctions(objc2_metal::MTLMathFloatingPointFunctions::Precise);
    }
}

/// An `NSError` from the shader compiler, split into the shared vocabulary.
///
/// **DETERMINISTIC IS THE DEFAULT HERE, AND THAT IS A MEASUREMENT RATHER
/// THAN A PREFERENCE.** The obvious rule — `Deterministic` exactly when the
/// code is `MTLLibraryError::CompileFailure` — was written first and is
/// wrong: on macOS 27 a source with a syntax error in it comes back at code
/// **1** (`MTLLibraryError::Unsupported`), not 3, so that rule classified
/// every real rejection as `Retryable` and left the negative tier dead. It
/// is also the wrong way round on the merits: `newLibraryWithSource:` is
/// handed TEXT, and a compiler that will not accept a piece of text will not
/// accept it on the next call either. So the split is inverted — everything
/// is the source's fault except `Internal`, which is the one code that names
/// the framework rather than the input.
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

/// The emitted source with its one unresolved include spliced in.
///
/// # Errors
///
/// [`Failure::Deterministic`] when `kernels-metal` does not ship the rng
/// source the template names — which would be this tree disagreeing with
/// itself, and is deterministic in the exact sense the negative tier means.
fn expand(source: &str) -> std::result::Result<String, Failure> {
    if !source.contains(RNG_INCLUDE) {
        return Ok(source.to_string());
    }
    let rng = kernels_metal::source(RNG_SOURCE).ok_or_else(|| Failure::Deterministic {
        reason: format!(
            "the emitted runtime includes `{RNG_SOURCE}` and `kernels-metal` ships no \
             such source; `newLibraryWithSource:` has no header search path, so the \
             include cannot be left for the compiler"
        ),
    })?;
    Ok(source.replace(RNG_INCLUDE, rng))
}

// ─────────────────────────────────────────────────────────────────────────────
// The compiled program
// ─────────────────────────────────────────────────────────────────────────────

/// One compiled region: the module that holds it, and which region it is.
#[derive(Debug)]
pub struct Region {
    /// Which region of its stage this is.
    pub region_index: u32,
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

/// One compiled stage: every generated region it declares, in region order.
///
/// Shared rather than owned: two programs naming the same stage share one
/// compiled library, and — unlike the CUDA twin, where sharing a `CUmodule`
/// is what makes the unload order dangerous — sharing here is free, because
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
    /// The stage plans these were compiled from, in the same order. Carried
    /// rather than looked up, so a compiled program cannot drift from its
    /// plan.
    pub plans: Arc<Vec<LaunchStagePlan>>,
    /// Each stage's attachment point (`LaunchStage::kind`). Carried because
    /// `LaunchStagePlan` has no `kind`, and firing by position picks the
    /// adapter rather than the sampler once a program has a prologue.
    pub kinds: Arc<Vec<Attach>>,
}

impl Compiled {
    /// The index of the first stage with this attachment point.
    #[must_use]
    pub fn stage_of_kind(&self, kind: Attach) -> Option<usize> {
        self.kinds.iter().position(|&k| k == kind)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The cache
// ─────────────────────────────────────────────────────────────────────────────

/// The compile cache: the only thing in this crate that compiles guest MSL.
///
/// Two tiers, where the CUDA twin has three — see the module doc for why the
/// persistent one is absent rather than unwritten.
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
    ///
    /// **AN ABSENCE HAS NO OUTPUT**, so the one claim worth asserting about
    /// a cache — that the second bind of a program compiles nothing — is
    /// only reachable through [`CacheStats::compilations`].
    #[must_use]
    pub const fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Compile `plan`'s generated regions, or answer from a tier.
    ///
    /// `versions` carries the identity's four version numbers, so a
    /// host-side bump misses rather than reusing a stale pipeline.
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
        let program_key = driver::tensor_ir::fnv1a64(program_identity.as_bytes());
        if let Some(reason) = self.negative.get(&program_key) {
            self.stats.negative_hits += 1;
            return Err(Failure::Deterministic {
                reason: reason.clone(),
            });
        }

        match self.build(context, plan, kernels, versions, target) {
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
                 a driver cannot know which of the two the host meant",
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
            // **NOTHING IS FOLDED IN BESIDE THE IDENTITY**, where the CUDA
            // twin folds NVRTC's version: `cache_identity` already carries
            // the device, and on this plane the device IS the toolchain.
            let key = driver::tensor_ir::fnv1a64(identity.as_bytes());
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
            // `plans` is parallel to `package.stages` — `adopt_launch_package`
            // refuses a package where it is not — so kinds index the same way.
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
            // A SECOND-PARTY region has no generated kernel and never will:
            // it is a `kernel_call` or a `sink_call`, which is a NAME the
            // shell launches itself rather than a body the emitter could
            // write. The emitter declines it correctly, and reading that
            // decline as a compile failure would refuse every adapter
            // program this shell can actually run. It is the LIBRARY tag
            // that says so — an emitter that declined a genuinely generated
            // region still has to be a failure, which is what the arms below
            // are for.
            if plan.fused.get(region_index as usize).is_some_and(|region| {
                region.kind == RegionKind::Library(LibraryOp::SecondParty)
            }) {
                continue;
            }
            let (source, entry) = match index.get(KERNEL_FUSED, stage_index, region_index) {
                Slot::Kernel { source, entry } => (source, entry),
                // NOT a `continue`. "The host declined on purpose" presumes a
                // shell with its own path for the region, and this one has
                // none — every region it runs is a compiled
                // `KernelKind::Fused`. Skipping a refusal drops the region's
                // ops from the fire while the plan still budgets their
                // scratch, so they read back as the zeros the fire memset and
                // publish a confident wrong answer. A reason nobody can act
                // on still beats an answer nobody can distinguish.
                Slot::Refused(why) => {
                    return Err(Failure::Deterministic {
                        reason: format!(
                            "stage {stage_index} region {region_index} was declined by the \
                             emitter ({why}); this shell runs only compiled regions, so a \
                             declined one would silently not run at all"
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
                module,
            });
        }
        Ok(Stage {
            signature_hash: plan.signature_hash,
            regions: Arc::new(regions),
        })
    }

    /// One region: expand its include, compile it, build its pipeline.
    fn region_module(
        &mut self,
        context: &Context,
        entry: &str,
        source: &str,
    ) -> std::result::Result<Arc<Module>, Failure> {
        let expanded = expand(source)?;
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
    fn the_one_kernel_kind_this_shell_runs_is_the_fused_one() {
        assert_eq!(KERNEL_FUSED, KernelKind::Fused);
    }

    #[test]
    fn the_rng_include_the_emitter_leaves_is_one_this_tree_ships() {
        assert!(
            kernels_metal::source(RNG_SOURCE).is_some(),
            "`{RNG_SOURCE}` is what every emitted runtime includes, and \
             `newLibraryWithSource:` has no search path to find it with"
        );
    }

    #[test]
    fn expansion_replaces_the_include_and_leaves_everything_else() {
        let source = format!("// head\n{RNG_INCLUDE}\n// tail\n");
        let expanded = expand(&source).expect("the rng source ships");
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
        assert_eq!(expand(source).expect("no include"), source);
    }

    /// The identity separates the two BACKENDS and the two EMITTERS.
    /// `Versions::from_compiler` is handed `registration.emitter_version`,
    /// which a host computes as `Backend::Metal.emitter_version()`, and the
    /// two emitters version independently — one plan emitted twice is two
    /// different translation units and must never share a key.
    #[test]
    fn the_two_emitters_never_share_a_cache_identity() {
        use tensor_compiler::codegen::program::Backend as Emitter;

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

    /// Fast math off is the determinism clause, so it is asserted rather than
    /// reviewed. Needs a Metal framework but no DEVICE — `MTLCompileOptions`
    /// is a plain object — so it runs anywhere the crate is built for Apple.
    #[cfg(target_vendor = "apple")]
    #[test]
    fn the_compile_options_turn_fast_math_off() {
        use objc2::runtime::NSObjectProtocol as _;

        let options = objc2_metal::MTLCompileOptions::new();
        set_safe_math(&options);
        if options.respondsToSelector(objc2::sel!(mathMode)) {
            assert_eq!(
                options.mathMode(),
                objc2_metal::MTLMathMode::Safe,
                "a contracted multiply-add moves a lane off the interpreter's answer"
            );
        }
        #[allow(deprecated)]
        {
            assert!(
                !options.fastMathEnabled(),
                "the deprecated reading of the same switch has to agree"
            );
        }
    }
}
