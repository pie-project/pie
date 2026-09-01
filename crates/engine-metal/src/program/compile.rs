//! Compiling a guest program for this device: emitted MSL in, a pipeline per
//! generated region out, held in two tiers.
//!
//! **THIS IS A SECOND `newLibraryWithSource:` CALLER AND THAT IS
//! DELIBERATE.** [`Pipelines`](crate::device::Pipelines) compiles the
//! SHIPPED shaders and is keyed by `(&'static str file, &'static str
//! entrypoint)` off `kernels_metal::SOURCES`; a guest program's MSL is an
//! owned `String` produced this second by `eta-compiler`, with no file
//! and no static name. Sharing that cache would mean either leaking every
//! guest entry name to `'static` or keying it by something it is not. So
//! this module compiles its own, and the two share only the framework — the
//! same argument the CUDA sibling makes about not reaching into
//! `kernels_cuda::jit`.
//!
//! # The emitted source is not complete until this module completes it
//!
//! Every M2 kernel `eta_compiler::codegen::metal` emits begins with
//! `RUNTIME_TEMPLATE`, and the template's second line is
//! `#include "ptir_rng.generated.metal"`. `newLibraryWithSource:` has no
//! header search path, so the compile fails with `fatal error:
//! 'ptir_rng.generated.metal' file not found` — measured on an M1 Max
//! against all five parity subjects. The emitter leaves the include
//! deliberately (its own doc says the expansion is the engine's), and the
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

/// The grouped kernel kind, whose channels are rows of a lane table.
///
/// **THIS SHELL RUNS TWO KINDS NOW, AND THE SECOND ONE IS NOT AN
/// OPTIMISATION.** The ruling this constant replaces said the plane binds
/// only [`KERNEL_FUSED`], because "the grouped forms need device addresses
/// this shell does not hand out". It hands them out now
/// (`device::alloc::Buffer::address_at`), and what that buys is two things
/// the single-lane form cannot do at all rather than two it does slowly:
///
/// * a region with more than twelve channels. The M2 kernel binds each
///   channel's committed and pending cells at `7 + 2k` and `8 + 2k`, and
///   Metal's last argument index is 30 — so the emitter REFUSES a wider
///   region, and this shell used to turn that refusal into a compile
///   failure for the whole program. `beam_epilogue` (sixteen channels) and
///   `pentathlon_iter` (thirteen) are the corpus's own examples.
/// * a region that walks the vocabulary. The M2 kernel is one thread
///   (`if (gid != 0) return;`), so a 248k-wide gather is 248k serial
///   iterations; the grouped kernel gets a threadgroup, splits the gather
///   across it, and elides it entirely where its only consumer is an
///   argmax.
///
/// Readiness and commit stay host-side, unchanged: that ruling was about
/// where a ring is gated, not about which kernel computes.
const KERNEL_GROUPED: KernelKind = KernelKind::Grouped;

/// Which emitted form a compiled region is, and therefore what
/// [`Prepared::encode_into`](super::launch::Prepared::encode_into) binds and
/// how wide it dispatches.
///
/// The three are three kernel SIGNATURES, not three speeds. A region's form
/// is decided once at compile — where the emitter's refusals can be read —
/// and never re-decided at fire time, because the pipeline was built for one
/// of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// The M2 single-lane kernel: status at buffer 0, each channel's two
    /// cells bound at `7 + 2k` / `8 + 2k`, one thread.
    Fused,
    /// The M3 grouped kernel over the lane table, a threadgroup per lane at
    /// `METAL_M3_REGION_THREADS`.
    Grouped,
    /// The M3 grouped LIBRARY sampler — nucleus or top-k — which takes the
    /// same eleven bindings but decomposes its grid one threadgroup per
    /// (lane, row) and demands exactly 256 threads.
    GroupedLibrary,
}

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

    /// The widest threadgroup this pipeline will accept.
    ///
    /// Read once, at compile: the grouped forms need an EXACT width and this
    /// is what says whether the device can give it. A pipeline's limit falls
    /// with its register pressure, so it is a property of the compiled
    /// kernel rather than of the device.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn max_threads(&self) -> usize {
        self.pipeline.maxTotalThreadsPerThreadgroup()
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
                     engine disagree about this region's entry name"
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
    /// Which region of its stage this is — an index into the plan's FUSED
    /// partition, whichever form the kernel took. The grouped emitter names
    /// its fused-region kernels at `singleton.len() + region_index`, and
    /// that offset is a fact about the emitted TABLE rather than about the
    /// region, so it lives at the lookup and nowhere else.
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
///
/// **A DECLINE CARRIES ITS REASON, AND IT DID NOT USED TO.** The single-lane
/// refusal this feeds ends `…and the grouped form could not serve it either`,
/// which was a CONSTANT: the sentence was printed whether the planner had
/// refused the stage, the emitter had refused the region, the table had no
/// kernel at the slot this shell reads, or the pipeline was too narrow for a
/// library sampler. Four different doors behind one sentence, and the metal
/// verify queue spent two sessions attributing `trackb-h2o`'s decline to the
/// wrong one. The reason is a `String` rather than an enum because its only
/// consumer is that message — nothing branches on which door it was.
enum GroupedAnswer {
    /// The grouped kernel compiled; this is it.
    Served(Region),
    /// It did not, and this is the clause the caller's refusal appends.
    Declined(String),
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
        let program_key = eta_ir::fnv1a64(program_identity.as_bytes());
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
            // **NOTHING IS FOLDED IN BESIDE THE IDENTITY**, where the CUDA
            // twin folds NVRTC's version: `cache_identity` already carries
            // the device, and on this plane the device IS the toolchain.
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
            // ── **THE GROUPED FORM IS TRIED FIRST WHERE IT BUYS SOMETHING**,
            //    and "something" is one of two things the single-lane form
            //    cannot do rather than a speed it does badly. See
            //    [`KERNEL_GROUPED`]. Everything else stays on the M2 kernel
            //    byte for byte: an existing green region is not moved onto a
            //    second ABI to buy nothing.
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
                // NOT a `continue`. "The host declined on purpose" presumes a
                // shell with its own path for the region, and this one has
                // none — every region it runs is a compiled kernel, in one of
                // the two forms above. Skipping a refusal drops the region's
                // ops from the fire while the plan still budgets their
                // scratch, so they read back as the zeros the fire memset and
                // publish a confident wrong answer. A reason nobody can act
                // on still beats an answer nobody can distinguish.
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

    /// The grouped kernel for one fused region, when that is the form to run
    /// it in; `None` to fall through to the single-lane one.
    ///
    /// **THE CHOICE IS MADE ON WHAT THE OTHER FORM CANNOT DO, NOT ON WHICH
    /// IS FASTER.** Three things move a region here and nothing else does:
    ///
    /// * the M2 emitter REFUSED it. Above twelve channels there is no
    ///   single-lane kernel at all, and until the grouped form was bound that
    ///   refusal failed the whole program's compile.
    /// * it is a library sampler. The nucleus and top-k library ops have a
    ///   grouped kernel of their own — a radix ordering across 256 threads —
    ///   and the single-lane path lowers the same region to the generic op
    ///   switch on one thread.
    /// * it walks the vocabulary through a `logits` gather. That gather is
    ///   the entire cost of a decode step's guest half (the emitter measured
    ///   85ms of an 89ms step), it splits across the threadgroup, and where
    ///   its only consumer is an argmax the grouped emitter removes it
    ///   outright.
    ///
    /// A region that is none of the three keeps the M2 kernel it has always
    /// had. That is deliberate: the two forms agree byte for byte — the
    /// grouped runtime partitions only argmax and copies, everything else
    /// runs on thread 0 through the same `ptir_m1_execute` — so moving a
    /// green region would be a second ABI bought for nothing.
    ///
    /// **A LIBRARY SAMPLER NEEDS AN EXACT WIDTH AND THIS IS WHERE THE DEVICE
    /// IS ASKED FOR IT.** The nucleus and top-k kernels open with
    /// `if (threads != 256u …) return;` — they size threadgroup arrays as
    /// `256 * 16` and decline any other width rather than adapting, so a
    /// pipeline whose register pressure caps it lower has no way to run one
    /// and the region goes back to the M2 form. A grouped FUSED region has no
    /// such requirement: it strides by `m3_threads` and is correct at any
    /// width up to the reduction buffer's, which is why this check is the
    /// library one alone. It used to be both, and that cost `beam_epilogue`
    /// its widest region — a 67-op kernel measures 384 on an M1 Max.
    ///
    /// **AND THE SCORE RECTANGLE IS ON THIS ROAD NOW, NOT OFF IT.** This
    /// paragraph used to end by naming a region reading the F32 score plane
    /// as the example of a decline: `m3_intrinsic_bindable` refused
    /// `AttnScore`, because a grouped kernel reads every rectangle as an
    /// ADDRESS off the lane record and the record carried only
    /// `logits_base`. It carries `attn_score_base` now (emitter 42), the two
    /// intrinsic tables agree id for id, and that matters here rather than
    /// only in the compiler: a score-reading region also has the LOWEST M2
    /// ceiling there is — `fused_channel_ceiling` puts it at ten channels
    /// instead of twelve — so the single-lane form is exactly what cannot
    /// serve it, and this is the only path that can. `trackb-h2o` is the
    /// shape that proves it; `eta-compiler`'s
    /// `the_score_rectangle_beside_many_channels` pins the compiler's half.
    ///
    /// # Errors
    ///
    /// Whatever the compile said. A REFUSAL is not an error: it answers
    /// [`GroupedAnswer::Declined`] with the reason, and the caller takes the
    /// single-lane path — appending that reason if the single-lane form has
    /// no kernel either.
    fn grouped_region(
        &mut self,
        context: &Context,
        stage_index: u32,
        region_index: u32,
        plan: &LaunchStagePlan,
        index: &Emitted<'_>,
    ) -> std::result::Result<GroupedAnswer, Failure> {
        if !plan.needs.grouped_valid {
            // The planner's own words when it has them: `LaunchStagePlan`
            // sets `error` and clears the bit together, and a plan that
            // cleared the bit with nothing to say is itself worth seeing.
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
        // The grouped table names a fused region at `singleton.len() + i`,
        // because the singleton partition's regions take the low indices of
        // the same `KernelKind::Grouped` slot space. Read off
        // `eta_compiler::codegen::program::emit_metal_stage` rather than
        // guessed; `program_parity`'s
        // `the_grouped_table_names_a_fused_region_where_this_shell_looks`
        // holds the two against each other over the whole corpus.
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
            // **NAMED, BECAUSE THE THREE ARE DIFFERENT BUGS.** A refusal is
            // the emitter's own sentence and belongs to the compiler; an
            // absence at this slot with kernels present at others is the
            // `singleton.len()` offset disagreeing between the two sides
            // (`program_parity`'s `the_grouped_table_names_a_fused_region_
            // where_this_shell_looks` is the standing check); a malformed one
            // is a table built with neither half.
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
            // Compiled and dropped. A rare enough answer that carrying a
            // second cache tier for it would be a table nobody reads, and the
            // caller's fallback compiles the M2 kernel this pipeline can run.
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
