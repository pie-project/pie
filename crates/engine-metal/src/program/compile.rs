#![allow(unsafe_code)]
use std::sync::Arc;

use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_compiler::codegen::metal::StepNodes;
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

const KERNEL_FUSED: KernelKind = KernelKind::Fused;

const KERNEL_GROUPED: KernelKind = KernelKind::Grouped;

const WIDE_REGION_ELEMENTS: u32 = 256;

fn op_width(op: &eta_compiler::codegen::launch::LaunchOp) -> u32 {
    op.shape
        .iter()
        .try_fold(1u32, |acc, &dim| acc.checked_mul(dim))
        .unwrap_or(u32::MAX)
}

fn region_ops(plan: &LaunchStagePlan, region_index: u32) -> usize {
    plan.fused
        .get(region_index as usize)
        .map_or(0, |region| region.nodes.len())
}

fn region_tags(plan: &LaunchStagePlan, region_index: u32) -> String {
    plan.fused
        .get(region_index as usize)
        .map(|region| {
            region
                .nodes
                .iter()
                .filter_map(|&node| plan.ops.get(node as usize))
                .map(|op| match plan.names.get(op.name_index as usize) {
                    Some(name) if op.tag == eta_ir::op::tags::OUTPUT => {
                        format!("{:#04x}[{name}]", op.tag)
                    }
                    _ => format!("{:#04x}", op.tag),
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default()
}

fn region_widest(plan: &LaunchStagePlan, region_index: u32) -> u32 {
    plan.fused.get(region_index as usize).map_or(0, |region| {
        region
            .nodes
            .iter()
            .filter_map(|&node| plan.ops.get(node as usize))
            .map(op_width)
            .max()
            .unwrap_or(0)
    })
}

fn region_trace() -> bool {
    crate::diag::on().region_trace
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    Fused,
    Grouped,
    GroupedLibrary,
    Streamed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamedStep {
    pub node: u32,
    pub kind: eta_compiler::codegen::metal::StepKind,
    pub result: u32,
    pub input: u32,
}

const RNG_INCLUDE: &str = "#include \"ptir_rng.generated.metal\"";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub device: u64,
}

impl Target {
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

/// The step pipelines every region module has built ([`Module::build`]).
static STEP_PIPELINES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The step pipelines region modules have built so far, process-wide.
#[must_use]
pub fn step_pipelines_built() -> u64 {
    STEP_PIPELINES.load(std::sync::atomic::Ordering::Relaxed)
}

pub struct Module {
    /// Held for the life of the pipelines made from it, never read.
    #[cfg(target_vendor = "apple")]
    #[allow(dead_code)]
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    #[cfg(target_vendor = "apple")]
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// A streamed region's pipeline a step (`m4_fixed_step`): each keeps
    /// its step's case alone, and with it only that case's registers and
    /// threadgroup memory. Empty where the region is one dispatch.
    #[cfg(target_vendor = "apple")]
    steps: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    entry: String,
}

// SAFETY: `MTLLibrary` and `MTLComputePipelineState` are documented thread-safe.
unsafe impl Send for Module {}
// SAFETY: both objects are immutable once built.
unsafe impl Sync for Module {}

impl std::fmt::Debug for Module {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Module")
            .field("entry", &self.entry)
            .finish()
    }
}

impl Module {
    #[must_use]
    pub fn entry(&self) -> &str {
        &self.entry
    }

    #[cfg(target_vendor = "apple")]
    pub(crate) fn pipeline(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.pipeline
    }

    /// Step `step`'s own pipeline, where it was built one.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn step_pipelines(&self) -> &[Retained<ProtocolObject<dyn MTLComputePipelineState>>] {
        &self.steps
    }

    #[cfg(target_vendor = "apple")]
    pub(crate) fn max_threads(&self) -> usize {
        self.pipeline.maxTotalThreadsPerThreadgroup()
    }

    #[cfg(target_vendor = "apple")]
    pub(crate) fn execution_width(&self) -> usize {
        self.pipeline.threadExecutionWidth()
    }

    #[cfg(target_vendor = "apple")]
    fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
        entry: &str,
        steps: usize,
        threads: usize,
    ) -> std::result::Result<Module, Failure> {
        let options = crate::device::ctx::compile_options();
        set_safe_math(&options);
        if let Some(dir) = crate::diag::on().kernel_dump.as_deref() {
            let path = dir.join(format!("{entry}.metal"));
            let _ = std::fs::write(path, source);
        }
        crate::diag::made(&format!("guest library+pipeline {entry}"), 0);
        let text = crate::device::ctx::nsstring(source);
        let library = device
            .newLibraryWithSource_options_error(&text, Some(&options))
            .map_err(|error| classify(entry, &error))?;
        // A streamed kernel declares its step as a function constant: the
        // pipeline for every step leaves it undefined, a step's states it.
        let pipeline_for = |library: &ProtocolObject<dyn MTLLibrary>, step: Option<u32>| {
            let name = crate::device::ctx::nsstring(entry);
            let constants = objc2_metal::MTLFunctionConstantValues::new();
            if let Some(step) = step {
                // SAFETY: a uint constant at this index, read from a live u32.
                unsafe {
                    constants.setConstantValue_type_atIndex(
                        std::ptr::NonNull::from(&step).cast(),
                        objc2_metal::MTLDataType::UInt,
                        eta_compiler::codegen::metal::M4_STEP_CONSTANT,
                    );
                }
            }
            let function = library
                .newFunctionWithName_constantValues_error(&name, &constants)
                .map_err(|_| Failure::Deterministic {
                    reason: format!(
                        "the library compiled and holds no `{entry}`; the emitter and the \
                         engine disagree about this region's entry name"
                    ),
                })?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| classify(entry, &error))
        };
        let made = pipeline_for(&library, None).map(Made)?;
        // Every step's pipeline is built here, with the region's library, as
        // the registration compiles: none is built where a step fires.
        let shared = Made(&*library);
        let steps = in_parallel(threads, steps, |step| {
            let shared = &shared;
            pipeline_for(shared.0, Some(u32::try_from(step).unwrap_or(u32::MAX))).map(Made)
        })
        .into_iter()
        .map(|made| made.map(|Made(pipeline)| pipeline))
        .collect::<std::result::Result<Vec<_>, _>>()?;
        STEP_PIPELINES.fetch_add(steps.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(Module {
            pipeline: made.0,
            library,
            steps,
            entry: entry.to_string(),
        })
    }
}

/// A library or a pipeline state, made on one thread and taken on another:
/// `MTLLibrary` and `MTLComputePipelineState` are documented thread-safe.
#[cfg(target_vendor = "apple")]
struct Made<T>(T);
// SAFETY: as stated above; `Made` holds nothing else.
#[cfg(target_vendor = "apple")]
unsafe impl<T> Send for Made<T> {}
// SAFETY: as above.
#[cfg(target_vendor = "apple")]
unsafe impl<T> Sync for Made<T> {}

/// `make` of every index below `count`, on up to `threads` threads, answered
/// in index order whatever order they ran in.
pub(crate) fn in_parallel<T: Send>(
    threads: usize,
    count: usize,
    make: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let threads = threads.min(count);
    if threads < 2 {
        return (0..count).map(make).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<T>>> = (0..count).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let at = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(slot) = slots.get(at) else { break };
                let made = make(at);
                *slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(made);
            });
        }
    });
    slots
        .into_iter()
        .filter_map(|slot| slot.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner))
        .collect()
}

/// The modules jobs run at once share, by text: each text is built by the
/// first job to ask, and the others wait for it.
#[derive(Default)]
#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
pub struct Sharing {
    modules: std::sync::Mutex<std::collections::HashMap<u64, SharedModule>>,
    threads: usize,
}

type SharedModule = Arc<std::sync::OnceLock<std::result::Result<Arc<Module>, Failure>>>;

#[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
impl Sharing {
    /// `threads`: the compiles the device runs at once.
    #[must_use]
    pub fn over(threads: usize) -> Sharing {
        Sharing {
            threads,
            ..Sharing::default()
        }
    }

    /// The module of `key`, and whether this call built it.
    fn module(
        &self,
        key: u64,
        build: impl FnOnce() -> std::result::Result<Arc<Module>, Failure>,
    ) -> (std::result::Result<Arc<Module>, Failure>, bool) {
        let slot = self
            .modules
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(key)
            .or_default()
            .clone();
        let mut built = false;
        let module = slot
            .get_or_init(|| {
                built = true;
                build()
            })
            .clone();
        (module, built)
    }
}

/// Safe math and precise floating-point functions. Both options are macOS
/// 15's; the engine runs on macOS 26 or newer only (Metal 4, which
/// `Context::bind` refuses a device without), so no older form is set.
#[cfg(target_vendor = "apple")]
fn set_safe_math(options: &objc2_metal::MTLCompileOptions) {
    options.setMathMode(objc2_metal::MTLMathMode::Safe);
    options.setMathFloatingPointFunctions(objc2_metal::MTLMathFloatingPointFunctions::Precise);
}

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

fn expand(source: &str) -> String {
    if !source.contains(RNG_INCLUDE) {
        return source.to_string();
    }
    source.replace(
        RNG_INCLUDE,
        &eta_compiler::codegen::rng::generate_msl_preamble(),
    )
}

#[derive(Debug)]
pub struct Region {
    pub region_index: u32,
    pub steps: Arc<Vec<StreamedStep>>,
    pub form: Form,
    pub module: Arc<Module>,
    /// A streamed region's nodes as its dispatches compute them, a list a
    /// step, and the views no step computes (`launch::launches`); empty
    /// where the emitter published no node lists or left a node out of
    /// them, and the region is one launch.
    pub launches: Arc<StepNodes>,
}

impl Region {
    #[cfg(target_vendor = "apple")]
    pub(crate) fn pipeline(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        self.module.pipeline()
    }
}

/// The device a registration's compile builds its libraries and pipelines
/// on, owned so the compile can run on any thread
/// (`engine::Engine::prepare_program`): Metal states `MTLDevice` thread-safe
/// for making libraries and pipeline states.
pub struct CompileDevice {
    #[cfg(target_vendor = "apple")]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

// SAFETY: the device is only asked to make libraries and pipeline states,
// which Metal states safe from any thread.
#[allow(unsafe_code)]
unsafe impl Send for CompileDevice {}

impl CompileDevice {
    /// Has the device compile on as many threads as the OS lets it
    /// (`on`), or as it does by default, and answers how many that is.
    #[must_use]
    pub fn compile_wide(&self, on: bool) -> usize {
        #[cfg(target_vendor = "apple")]
        {
            self.device.setShouldMaximizeConcurrentCompilation(on);
            self.device.maximumConcurrentCompilationTaskCount().max(1)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = on;
            1
        }
    }

    #[must_use]
    pub fn of(context: &Context) -> CompileDevice {
        #[cfg(target_vendor = "apple")]
        {
            use objc2::Message as _;
            CompileDevice {
                device: context.device().retain(),
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = context;
            CompileDevice {}
        }
    }
}

/// What a registration still has to compile, planned on the engine's
/// thread ([`Cache::plan`]): the stages the cache did not hold, built on
/// any thread ([`Job::run`]), then committed on the engine's thread
/// ([`Cache::finish`]).
pub struct Job {
    device: CompileDevice,
    program_hash: u64,
    program_key: u64,
    plans: Vec<LaunchStagePlan>,
    kinds: Vec<Attach>,
    kernels: Vec<EmittedKernel>,
    /// Each stage: the cache's (a hit) or `None` (to build), with its key,
    /// its identity and whether the cache takes it once built.
    slots: Vec<(Option<Stage>, u64, u64, bool)>,
    /// The region modules the cache held when the job was planned, by text
    /// ([`Cache::modules`]).
    modules: std::collections::HashMap<u64, Arc<Module>>,
}

// SAFETY: the stages a job holds and builds are immutable libraries and
// pipeline states (and their plans), which Metal states safe to use from
// any thread; nothing in a job is touched by two threads at once.
#[allow(unsafe_code)]
unsafe impl Send for Job {}

/// A job run ([`Job::run`]): its stages, or why they did not build.
pub struct Built {
    job: Job,
    stages: std::result::Result<Vec<Stage>, Failure>,
    compilations: u64,
    memory_hits: u64,
    /// The wall time its region compiles took, in nanoseconds.
    compile_nanos: u64,
    /// The region modules the job built, by text, for the cache.
    made: std::collections::HashMap<u64, Arc<Module>>,
}

// SAFETY: as `Job`.
#[allow(unsafe_code)]
unsafe impl Send for Built {}

/// A registration's compile, planned ([`Cache::plan`]): held already, or a
/// job to run.
pub enum Planned {
    Held(Compiled),
    Run(Job),
}

impl Job {
    /// Builds every stage the cache did not hold: the libraries and
    /// pipelines, on whatever thread runs it.
    #[must_use]
    pub fn run(self) -> Built {
        self.run_sharing(&Sharing::over(1))
    }

    /// [`Job::run`], its modules shared with the jobs run beside it.
    #[must_use]
    pub fn run_sharing(self, sharing: &Sharing) -> Built {
        let mut builder = Builder {
            device: &self.device,
            sharing,
            modules: &self.modules,
            made: std::collections::HashMap::new(),
            compilations: 0,
            memory_hits: 0,
            compile_nanos: 0,
        };
        let stages = (|| {
            let index = Emitted::index(&self.kernels).map_err(|duplicate| Failure::Deterministic {
                reason: format!(
                    "the emitted kernel table names slot (kind {}, stage {}, region {}) twice; \
                     an engine cannot know which of the two the host meant",
                    duplicate.kind as u32, duplicate.stage, duplicate.region
                ),
            })?;
            let mut stages = Vec::with_capacity(self.slots.len());
            for (at, (held, ..)) in self.slots.iter().enumerate() {
                if let Some(stage) = held {
                    stages.push(stage.clone());
                    continue;
                }
                let stage_index = u32::try_from(at).map_err(|_| Failure::Deterministic {
                    reason: "a program with more than four billion stages is not a program".into(),
                })?;
                stages.push(builder.build_stage(stage_index, &self.plans[at], &index)?);
            }
            Ok(stages)
        })();
        let Builder {
            compilations,
            memory_hits,
            compile_nanos,
            made,
            ..
        } = builder;
        Built {
            job: self,
            stages,
            compilations,
            memory_hits,
            compile_nanos,
            made,
        }
    }
}

/// The region compiles a job runs, on the device it owns.
struct Builder<'a> {
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    device: &'a CompileDevice,
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    sharing: &'a Sharing,
    modules: &'a std::collections::HashMap<u64, Arc<Module>>,
    made: std::collections::HashMap<u64, Arc<Module>>,
    compilations: u64,
    memory_hits: u64,
    compile_nanos: u64,
}

enum GroupedAnswer {
    Served(Region),
    Declined(String),
}

#[derive(Debug, Clone)]
pub struct Stage {
    pub signature_hash: u64,
    pub regions: Arc<Vec<Region>>,
}

impl Stage {
    #[must_use]
    pub fn region(&self, region_index: u32) -> Option<&Region> {
        self.regions
            .iter()
            .find(|region| region.region_index == region_index)
    }
}

#[derive(Debug, Clone)]
pub struct Compiled {
    pub stages: Arc<Vec<Stage>>,
    pub plans: Arc<Vec<LaunchStagePlan>>,
    pub kinds: Arc<Vec<Attach>>,
}

#[derive(Debug)]
pub struct Cache {
    programs: Bounded<u64, Compiled>,
    stages: Stages<Stage>,
    negative: Bounded<u64, String>,
    /// Each region's module by its source text: a program whose
    /// stages differ from a built one's (another binding, another request
    /// shape) and whose regions read the same reuses them, as the runtime's
    /// prewarm of its standard samplers counts on (regions keyed by text).
    modules: std::collections::HashMap<u64, Arc<Module>>,
    stats: CacheStats,
    /// The wall time every region compile since load took, in nanoseconds:
    /// a compile the OS's shader cache serves reads apart from a cold one.
    compile_nanos: u64,
}

impl Default for Cache {
    fn default() -> Cache {
        Cache::new()
    }
}

impl Cache {
    #[must_use]
    pub fn new() -> Cache {
        Cache {
            programs: Bounded::new(MAX_PROGRAM_ENTRIES),
            stages: Stages::new(MAX_STAGE_ENTRIES),
            negative: Bounded::new(MAX_NEGATIVE_ENTRIES),
            modules: std::collections::HashMap::new(),
            stats: CacheStats::default(),
            compile_nanos: 0,
        }
    }

    #[must_use]
    pub const fn stats(&self) -> CacheStats {
        self.stats
    }

    /// The wall time every region compile since load took, in nanoseconds.
    #[must_use]
    pub const fn compile_nanos(&self) -> u64 {
        self.compile_nanos
    }

    pub fn compile(
        &mut self,
        context: &Context,
        program_hash: u64,
        plan: &ExecPlan,
        kernels: &[EmittedKernel],
        versions: Versions,
        target: Target,
    ) -> std::result::Result<Compiled, Failure> {
        match self.plan(context, program_hash, plan, kernels, versions, target)? {
            Planned::Held(compiled) => Ok(compiled),
            Planned::Run(job) => self.finish(job.run()),
        }
    }

    /// The first half of [`Cache::compile`], on the engine's thread: the
    /// program the cache holds already, a deterministic refusal it
    /// remembers, or a job of the stages it does not hold.
    pub fn plan(
        &mut self,
        context: &Context,
        program_hash: u64,
        plan: &ExecPlan,
        kernels: &[EmittedKernel],
        versions: Versions,
        target: Target,
    ) -> std::result::Result<Planned, Failure> {
        if let Some(compiled) = self.programs.get(&program_hash) {
            self.stats.memory_hits += 1;
            return Ok(Planned::Held(compiled.clone()));
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

        let mut slots = Vec::with_capacity(plan.package.plans.len());
        for stage_plan in &plan.package.plans {
            let identity = cache_identity(
                Backend::Metal,
                target.device,
                stage_plan.signature_hash,
                versions,
            );
            let key = eta_ir::fnv1a64(identity.as_bytes());
            let (lookup, hit) = self.stages.lookup(key, stage_plan.identity);
            match (lookup, hit) {
                (Lookup::Hit, Some(stage)) => {
                    self.stats.memory_hits += 1;
                    slots.push((Some(stage), key, stage_plan.identity, false));
                }
                (lookup, _) => {
                    slots.push((None, key, stage_plan.identity, lookup == Lookup::Miss));
                }
            }
        }
        Ok(Planned::Run(Job {
            device: CompileDevice::of(context),
            program_hash,
            program_key,
            plans: plan.package.plans.clone(),
            kinds: plan.package.stages.iter().map(|s| s.stage).collect(),
            kernels: kernels.to_vec(),
            slots,
            modules: self.modules.clone(),
        }))
    }

    /// The second half of [`Cache::compile`], on the engine's thread: a
    /// built job's stages taken into the cache and its program held, or its
    /// refusal remembered where it is deterministic.
    pub fn finish(&mut self, built: Built) -> std::result::Result<Compiled, Failure> {
        self.stats.compilations += built.compilations;
        self.stats.memory_hits += built.memory_hits;
        self.compile_nanos = self.compile_nanos.saturating_add(built.compile_nanos);
        self.modules.extend(built.made);
        let job = built.job;
        match built.stages {
            Ok(stages) => {
                for ((held, key, identity, keep), stage) in job.slots.iter().zip(&stages) {
                    if held.is_none() && *keep {
                        self.stages.stage(*key, *identity, stage.clone());
                    }
                }
                self.stages.commit();
                let compiled = Compiled {
                    stages: Arc::new(stages),
                    plans: Arc::new(job.plans),
                    kinds: Arc::new(job.kinds),
                };
                self.programs.insert(job.program_hash, compiled.clone());
                Ok(compiled)
            }
            Err(failure) => {
                self.stages.abandon();
                if let Failure::Deterministic { reason } = &failure {
                    self.negative.insert(job.program_key, reason.clone());
                }
                Err(failure)
            }
        }
    }

    pub fn forget(&mut self, program_hash: u64) {
        self.programs.remove(&program_hash);
    }

}

impl Builder<'_> {
    fn build_stage(
        &mut self,
        stage_index: u32,
        plan: &LaunchStagePlan,
        index: &Emitted<'_>,
    ) -> std::result::Result<Stage, Failure> {
        let mut regions = Vec::new();
        for region_index in 0..plan.fused.len() {
            let region_index = u32::try_from(region_index).map_err(|_| Failure::Deterministic {
                reason: "a stage with more than four billion regions is not a stage".into(),
            })?;
            if plan
                .fused
                .get(region_index as usize)
                .is_some_and(|region| region.kind == RegionKind::Library(LibraryOp::SecondParty))
            {
                continue;
            }
            if let Some(region) =
                self.streamed_region(stage_index, region_index, plan, index)?
            {
                if region_trace() {
                    eprintln!(
                        "region: stage {stage_index} region {region_index} takes the Streamed \
                         form ({} step(s), {} op(s), widest value {} element(s)): {}",
                        region.steps.len(),
                        region_ops(plan, region_index),
                        region_widest(plan, region_index),
                        region_tags(plan, region_index),
                    );
                }
                regions.push(region);
                continue;
            }
            let grouped_declined =
                match self.grouped_region(stage_index, region_index, plan, index)? {
                    GroupedAnswer::Served(region) => {
                        if region_trace() {
                            eprintln!(
                                "region: stage {stage_index} region {region_index} takes the \
                                 {:?} form ({} op(s), widest value {} element(s)): {}",
                                region.form,
                                region_ops(plan, region_index),
                                region_widest(plan, region_index),
                                region_tags(plan, region_index),
                            );
                        }
                        regions.push(region);
                        continue;
                    }
                    GroupedAnswer::Declined(why) => why,
                };
            if region_trace() {
                eprintln!(
                    "region: stage {stage_index} region {region_index} falls back to the \
                     single-lane form ({} op(s), widest value {} element(s)): {}: {grouped_declined}",
                    region_ops(plan, region_index),
                    region_widest(plan, region_index),
                    region_tags(plan, region_index),
                );
            }
            let (source, entry) = match index.get(KERNEL_FUSED, stage_index, region_index) {
                Slot::Kernel { source, entry, .. } => (source, entry),
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

            let module = self.region_module(entry, source, 0)?;
            regions.push(Region {
                region_index,
                steps: Arc::new(Vec::new()),
                form: Form::Fused,
                module,
                launches: Arc::default(),
            });
        }
        Ok(Stage {
            signature_hash: plan.signature_hash,
            regions: Arc::new(regions),
        })
    }

    fn grouped_region(
        &mut self,
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
            Some(RegionKind::Library(
                LibraryOp::NucleusSample | LibraryOp::TopK
            ))
        );
        let refused = matches!(
            index.get(KERNEL_FUSED, stage_index, region_index),
            Slot::Refused(_)
        );
        let gathers = region.is_some_and(|region| {
            region.nodes.iter().any(|&node| {
                plan.ops
                    .get(node as usize)
                    .is_some_and(|op| op.tag == eta_ir::op::tags::OUTPUT)
            })
        });
        let wide = region.is_some_and(|region| {
            region.nodes.iter().any(|&node| {
                plan.ops
                    .get(node as usize)
                    .is_some_and(|op| op_width(op) >= WIDE_REGION_ELEMENTS)
            })
        });
        if !(library || refused || gathers || wide) {
            return Ok(GroupedAnswer::Declined(format!(
                "this shell keeps a region on the single-lane form unless the grouped \
                 one buys something: a library sampler, a refusal to route around, an \
                 intrinsic gather, or a value of at least {WIDE_REGION_ELEMENTS} elements \
                 to split across the threadgroup. This region is none of the four"
            )));
        }
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
            Slot::Kernel { source, entry, .. } => (source, entry),
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
        let module = self.region_module(entry, source, 0)?;
        #[cfg(target_vendor = "apple")]
        if library && module.max_threads() < super::launch::LIBRARY_SAMPLER_THREADS {
            return Ok(GroupedAnswer::Declined(format!(
                "the grouped library sampler opens by declining any width but \
                 {}, and this pipeline's own limit is {}",
                super::launch::LIBRARY_SAMPLER_THREADS,
                module.max_threads()
            )));
        }
        Ok(GroupedAnswer::Served(Region {
            region_index,
            steps: Arc::new(Vec::new()),
            form: if library {
                Form::GroupedLibrary
            } else {
                Form::Grouped
            },
            module,
            launches: Arc::default(),
        }))
    }

    fn streamed_region(
        &mut self,
        stage_index: u32,
        region_index: u32,
        plan: &LaunchStagePlan,
        index: &Emitted<'_>,
    ) -> std::result::Result<Option<Region>, Failure> {
        if !plan.needs.grouped_valid {
            return Ok(None);
        }
        if plan.fused.get(region_index as usize).is_none() {
            return Ok(None);
        }

        let (source, entry, table) =
            match index.get(KernelKind::Streamed, stage_index, region_index) {
                Slot::Kernel {
                    source,
                    entry,
                    steps,
                } => (source, entry, steps),
                _ => return Ok(None),
            };
        let (table, published) = eta_compiler::codegen::metal::step_node_table(table);
        let launches = published
            .map(|published| step_launches(published, &plan.fused[region_index as usize].nodes))
            .unwrap_or_default();
        let mut steps = Vec::with_capacity(table.len());
        for &word in table {
            let value = eta_compiler::codegen::metal::step_value(word);
            let Some(kind) = eta_compiler::codegen::metal::step_kind(word) else {
                return Ok(None);
            };
            steps.push(StreamedStep {
                node: value,
                kind,
                result: value,
                input: value,
            });
        }
        let module = self.region_module(entry, source, steps.len())?;
        #[cfg(target_vendor = "apple")]
        if module.execution_width() != 32 {
            if region_trace() {
                eprintln!(
                    "region: stage {stage_index} region {region_index} declines the Streamed \
                     form: the pipeline's execution width is {}, not 32",
                    module.execution_width()
                );
            }
            return Ok(None);
        }
        Ok(Some(Region {
            region_index,
            steps: Arc::new(steps),
            form: Form::Streamed,
            module,
            launches: Arc::new(launches),
        }))
    }

    fn region_module(
        &mut self,
        entry: &str,
        source: &str,
        steps: usize,
    ) -> std::result::Result<Arc<Module>, Failure> {
        let expanded = expand(source);
        // Keyed by its text under one symbol (each program names its
        // regions' entries after itself), as the runtime's prewarm test
        // compares them; a module keeps its own pipeline, whatever it is
        // named.
        let key = eta_ir::fnv1a64(expanded.replace(entry, "ptir_region").as_bytes());
        if let Some(module) = self.modules.get(&key).or_else(|| self.made.get(&key)) {
            self.memory_hits += 1;
            return Ok(module.clone());
        }
        #[cfg(target_vendor = "apple")]
        {
            let began = std::time::Instant::now();
            let (module, built) = self.sharing.module(key, || {
                Module::build(&self.device.device, &expanded, entry, steps, self.sharing.threads)
                    .map(Arc::new)
            });
            let module = module?;
            if built {
                self.compile_nanos += u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX);
                self.compilations += 1;
            } else {
                self.memory_hits += 1;
            }
            self.made.insert(key, module.clone());
            Ok(module)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (entry, expanded, steps);
            Err(Failure::Retryable {
                reason: "this build has no Metal in it: the target is not an Apple one".into(),
            })
        }
    }
}

/// A streamed region's launches: each step a dispatch of its own, with the
/// nodes it computes (a node a reduction dispatches in a partial step and
/// finishes in a later one lies in both, its bytes live from the first to
/// the last), and the views no step computes. Empty, and the region one
/// launch, where a node of `region` is neither in a step's list nor a
/// view: what the emitter did not account for gets no reuse.
fn step_launches(published: StepNodes, region: &[u32]) -> StepNodes {
    let accounted = |node: &u32| {
        published.views.contains(node)
            || published.folded.iter().any(|&(_, folded)| folded == *node)
            || published.steps.iter().any(|nodes| nodes.contains(node))
    };
    if region.iter().all(accounted) {
        published
    } else {
        StepNodes::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod sampler_scratch {

    }

    #[test]
    fn compile_every_case() {
        expansion_replaces_the_include_and_leaves_everything_else();
        a_source_with_no_include_is_handed_over_unchanged();
        the_two_emitters_never_share_a_cache_identity();
    }

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

    fn a_source_with_no_include_is_handed_over_unchanged() {
        let source = "kernel void nothing() {}\n";
        assert_eq!(expand(source), source);
    }

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

#[cfg(test)]
mod parallel {
    use super::*;

    /// Answers come in index order, each index made once, at any thread
    /// count; a text's failure is every asker's and is built once.
    #[test]
    fn parallel_builds_answer_in_order_and_build_a_text_once() {
        for threads in [1, 2, 18] {
            let made = std::sync::atomic::AtomicUsize::new(0);
            let out = in_parallel(threads, 40, |at| {
                made.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                at
            });
            assert_eq!(out, (0..40).collect::<Vec<_>>());
            assert_eq!(made.into_inner(), 40);
            let sharing = Sharing::over(threads);
            let asked = in_parallel(threads, 8, |_| {
                sharing.module(7, || {
                    Err(Failure::Deterministic { reason: "`e`: refused".into() })
                })
            });
            assert_eq!(asked.iter().filter(|(_, built)| *built).count(), 1);
            for (module, _) in asked {
                let refused = Failure::Deterministic { reason: "`e`: refused".into() };
                assert_eq!(module.map(|_| ()), Err(refused));
            }
        }
    }
}
