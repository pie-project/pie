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
        let name = crate::device::ctx::nsstring(entry);
        // A streamed kernel declares its step as a function constant: the
        // pipeline for every step leaves it undefined, a step's states it.
        let pipeline_for = |step: Option<u32>| {
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
        let pipeline = pipeline_for(None)?;
        // Every step's pipeline is built here, with the region's library, as
        // the registration compiles: none is built where a step fires.
        let steps = (0..steps)
            .map(|step| pipeline_for(Some(u32::try_from(step).unwrap_or(u32::MAX))))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        STEP_PIPELINES.fetch_add(steps.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(Module {
            library,
            pipeline,
            steps,
            entry: entry.to_string(),
        })
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
        let mut builder = Builder {
            device: &self.device,
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
            let module = Arc::new(Module::build(&self.device.device, &expanded, entry, steps)?);
            self.compile_nanos += u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX);
            self.compilations += 1;
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
        use super::super::super::launch;
        use super::super::{KernelKind, StepNodes, step_launches};
        use eta_compiler::codegen::launch::LaunchPackage;
        use eta_compiler::codegen::program::{Backend as Emitter, emit_program};
        use eta_compiler::plan::compile_bound;
        use eta_dsl::Traced;
        use eta_dsl::prelude::*;
        use eta_exec::{Emitted, Slot};
        use eta_exec::{Extents, describe};
        use eta_ir::registry::{Axis, EngineCaps, OutputDecl, Port};

        const QWEN_VOCAB: u32 = 151_936;

        fn leak(channel: Channel) -> &'static Channel {
            Box::leak(Box::new(channel))
        }

        fn logits(vocab: u32) -> Vec<Vec<OutputDecl>> {
            vec![vec![OutputDecl {
                name: "logits".into(),
                axis: Axis::Row,
                width: vocab,
                dtype: eta_ir::Dtype::F32,
                layers: 0,
                readout: true,
            }]]
        }

        fn builder(vocab: u32) -> (Builder<'static>, &'static Channel) {
            let tok_in = leak(Channel::from([42i32]));
            let mut b = Builder::new(vocab, 16);
            b.outputs(logits(vocab));
            b.bind_port(Port::EmbedTokens, tok_in);
            b.bind_port(Port::EmbedIndptr, leak(Channel::from([0u32, 1])));
            b.bind_port(Port::Fold, leak(Channel::from([u32::MAX])));
            (b, tok_in)
        }

        fn top_p(vocab: u32) -> Traced {
            let (temperature, top_p) = (0.6f32, 0.95f32);
            let rng = leak(Channel::from([0x7ce1u32, 0]));
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let logits = intrinsics::output(0, "logits");
                let scaled = logits / temperature;
                let probs = softmax(&scaled);
                let keep = pivot_threshold(probs, cummass_le(top_p));
                let neg_inf = broadcast(f32::NEG_INFINITY, [vocab]);
                let masked = select(&keep, &scaled, &neg_inf);
                let r = rng.take();
                let g = gumbel(&r, [vocab]);
                let r_next = &r + iota(2);
                rng.put(&r_next);
                let t = reshape(reduce_argmax(masked + g), [1]);
                tok_in.put(&t);
                out.put(&t);
            });
            out.note_host_take();
            b.build().expect("the top-p epilogue traces")
        }

        fn top_k_top_p(vocab: u32) -> Traced {
            let temperature = 0.8f32;
            let rng = leak(Channel::from([0x7ce1u32, 0]));
            let out = leak(Channel::new([1], dtype::i32).capacity(8));
            let (mut b, tok_in) = builder(vocab);
            b.stage(Stage::Epilogue, move || {
                let r = rng.take();
                let logits = reshape(intrinsics::output(0, "logits"), [vocab]);
                let scaled = &logits / temperature;
                let probs = softmax(&scaled);
                let ranked = pivot_threshold(&probs, rank_le(40u32));
                let nucleus = pivot_threshold(&probs, cummass_le(0.9f32));
                let keep = and(&ranked, &nucleus);
                let token = reshape(gumbel_max(mask_apply(&scaled, keep), &r), [1]);
                rng.put(&r + iota(2));
                tok_in.put(&token);
                out.put(&token);
            });
            out.note_host_take();
            b.build().expect("the top-k top-p epilogue traces")
        }

        /// A lane's scratch for the samplers inferlets ship, at Qwen3's
        /// vocabulary: `text-completion-bench`'s top-p draw and `compat`'s
        /// top-k + top-p. Each streamed step is a launch of its own and a
        /// view takes its source's bytes, so the values lay in what they
        /// take alive at once ([`eta_exec::most_alive`]), beside the
        /// temporary area; the stride is far below each value its own
        /// bytes, and below what runs of steps sharing a node laid.
        #[test]
        fn a_samplers_values_lay_in_what_is_alive_at_once() {
            // (name, each value its own, runs of steps sharing a node, now,
            // (the values' bytes laid, what they take alive at once))
            let want = [
                (
                    "top-p",
                    9_312_512u64,
                    6_930_688u64,
                    5_107_456u64,
                    (2_583_552u64, 2_583_552u64),
                ),
                (
                    "top-k + top-p",
                    9_616_896,
                    7_691_264,
                    5_411_584,
                    (2_887_680, 2_887_680),
                ),
            ];
            let programs = [top_p(QWEN_VOCAB), top_k_top_p(QWEN_VOCAB)];
            for ((name, own_was, runs_was, now, laid_alive), traced) in
                want.into_iter().zip(programs)
            {
                let caps = EngineCaps {
                    lora: true,
                    attn_page_mask: true,
                };
                let bound =
                    eta_ir::validate::bind(traced.container().clone(), logits(QWEN_VOCAB), caps)
                        .expect("the program binds");
                let stages = compile_bound(&bound);
                let package: LaunchPackage = eta_compiler::codegen::launch::build(&bound, &stages);
                let kernels = emit_program(Emitter::Metal, &stages, &bound);
                let index = Emitted::index(&kernels).expect("one kernel a slot");
                let [plan] = &package.plans[..] else {
                    panic!("{name}: one stage");
                };
                let published: Vec<StepNodes> = (0..plan.fused.len() as u32)
                    .map(|region| {
                        let Slot::Kernel { steps, .. } = index.get(KernelKind::Streamed, 0, region)
                        else {
                            panic!("{name}: region {region} has a streamed kernel");
                        };
                        let (_, published) = eta_compiler::codegen::metal::step_node_table(steps);
                        let launches = step_launches(
                            published.expect("node lists"),
                            &plan.fused[region as usize].nodes,
                        );
                        assert!(
                            !launches.steps.is_empty(),
                            "{name}: region {region} accounts for every node"
                        );
                        launches
                    })
                    .collect();
                let published: Vec<Option<&StepNodes>> = published.iter().map(Some).collect();
                let descriptors: Vec<_> = plan
                    .value_types
                    .iter()
                    .map(|value| describe(value, &Extents::default()).expect("static"))
                    .collect();
                let run = launch::published_launches(plan, &published, &descriptors)
                    .expect("the temporary area is stated");
                let lifetimes =
                    eta_exec::value_lifetimes(plan, &run).expect("every node launched or idle");
                let laid = launch::lay_out_launches(plan, &run, &descriptors).expect("fits");
                let values_end = descriptors
                    .iter()
                    .zip(&lifetimes)
                    .zip(&laid.values)
                    .filter(|((_, life), _)| !life.dead)
                    .map(|((descriptor, _), &at)| {
                        at + descriptor
                            .device_bytes()
                            .next_multiple_of(eta_exec::SCRATCH_ALIGN)
                    })
                    .max()
                    .unwrap_or(0);
                let alive = eta_exec::most_alive(&descriptors, &lifetimes);
                eprintln!(
                    "{name}: {} bytes a lane ({values_end} of values, {alive} alive at once; \
                     {own_was} each value its own, {runs_was} by runs of steps)",
                    laid.total
                );
                assert_eq!(
                    eta_exec::layout(&descriptors).expect("fits").total,
                    own_was,
                    "{name}"
                );
                assert_eq!(
                    (values_end - eta_exec::SCRATCH_DUMMY_BYTES, alive),
                    laid_alive,
                    "{name}: the values lay in what is alive at once, but for what a \
                     slot's alignment leaves"
                );
                assert_eq!(laid.total, now, "{name}");
                assert!(now < runs_was, "{name}");
            }
        }
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

