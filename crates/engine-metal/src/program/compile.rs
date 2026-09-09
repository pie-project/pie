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
                .map(|op| match op.intrinsic {
                    Some(id) => format!("{:#04x}[{id:?}]", op.tag),
                    None => format!("{:#04x}", op.tag),
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

pub struct Module {
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
    ) -> std::result::Result<Module, Failure> {
        use objc2_metal::MTLCompileOptions;

        let options = MTLCompileOptions::new();
        set_safe_math(&options);
        if let Some(dir) = crate::diag::on().kernel_dump.as_deref() {
            let path = dir.join(format!("{entry}.metal"));
            let _ = std::fs::write(path, source);
        }
        let text = crate::device::ctx::nsstring(source);
        let library = device
            .newLibraryWithSource_options_error(&text, Some(&options))
            .map_err(|error| classify(entry, &error))?;
        let name = crate::device::ctx::nsstring(entry);
        let function =
            library
                .newFunctionWithName(&name)
                .ok_or_else(|| Failure::Deterministic {
                    reason: format!(
                        "the library compiled and holds no `{entry}`; the emitter and the \
                     engine disagree about this region's entry name"
                    ),
                })?;
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

#[cfg(target_vendor = "apple")]
fn set_safe_math(options: &objc2_metal::MTLCompileOptions) {
    use objc2::runtime::NSObjectProtocol as _;

    if options.respondsToSelector(objc2::sel!(setMathMode:)) {
        options.setMathMode(objc2_metal::MTLMathMode::Safe);
    } else {
        #[allow(deprecated)]
        options.setFastMathEnabled(false);
    }
    if options.respondsToSelector(objc2::sel!(setMathFloatingPointFunctions:)) {
        options.setMathFloatingPointFunctions(objc2_metal::MTLMathFloatingPointFunctions::Precise);
    }
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
}

impl Region {
    #[cfg(target_vendor = "apple")]
    pub(crate) fn pipeline(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        self.module.pipeline()
    }
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

impl Compiled {
    #[must_use]
    pub fn stage_of_kind(&self, kind: Attach) -> Option<usize> {
        self.kinds.iter().position(|&k| k == kind)
    }
}

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
    #[must_use]
    pub fn new() -> Cache {
        Cache {
            programs: Bounded::new(MAX_PROGRAM_ENTRIES),
            stages: Stages::new(MAX_STAGE_ENTRIES),
            negative: Bounded::new(MAX_NEGATIVE_ENTRIES),
            stats: CacheStats::default(),
        }
    }

    #[must_use]
    pub const fn stats(&self) -> CacheStats {
        self.stats
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
                self.stages.abandon();
                if let Failure::Deterministic { reason } = &failure {
                    self.negative.insert(program_key, reason.clone());
                }
                Err(failure)
            }
        }
    }

    pub fn forget(&mut self, program_hash: u64) {
        self.programs.remove(&program_hash);
    }

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
            kinds: Arc::new(plan.package.stages.iter().map(|s| s.stage).collect()),
        })
    }

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
            if plan
                .fused
                .get(region_index as usize)
                .is_some_and(|region| region.kind == RegionKind::Library(LibraryOp::SecondParty))
            {
                continue;
            }
            if let Some(region) =
                self.streamed_region(context, stage_index, region_index, plan, index)?
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
                match self.grouped_region(context, stage_index, region_index, plan, index)? {
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

            let module = self.region_module(context, entry, source)?;
            regions.push(Region {
                region_index,
                steps: Arc::new(Vec::new()),
                form: Form::Fused,
                module,
            });
        }
        Ok(Stage {
            signature_hash: plan.signature_hash,
            regions: Arc::new(regions),
        })
    }

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
                    .is_some_and(|op| op.intrinsic.is_some())
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
        let module = self.region_module(context, entry, source)?;
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
        }))
    }

    fn streamed_region(
        &mut self,
        context: &Context,
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
        let module = self.region_module(context, entry, source)?;
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
        }))
    }

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
