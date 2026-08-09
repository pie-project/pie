//! Whole-program emission: a bound trace in, one kernel table out.
//!
//! The per-region emitters in [`crate::codegen::cuda`] and [`crate::codegen::metal`] handle one
//! region at a time. This module owns the walk above them: which emitter each
//! region goes through, and what its entry point is called. That decision
//! belongs here rather than in a driver, so a driver receives a table and
//! compiles it instead of re-deriving from the plan what the host already
//! worked out.
//!
//! Entry names and the emitter-selection rules are shared with
//! `crates/driver-metal/csrc/src/pipeline/m1_runtime.cpp` and
//! `crates/driver-cuda/csrc/src/pipeline/generated/module_cache.hpp` — a driver reading
//! this table has to find exactly the names it looks up, so the naming scheme
//! is an ABI and not a formatting choice.

use crate::codegen::error::{EmitError, EmitterKind};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::plan::{CompiledStage, LibraryOp, Region, RegionKind};
use tensor_ir::op::tags;
use tensor_ir::validate::BoundTrace;

/// Kind discriminants. Re-exported from the driver ABI rather than restated:
/// [`EmittedKernel::kind`] is handed straight to the driver, so a second
/// spelling of these numbers here would be a second thing to keep right.
pub use driver_api::local::{
    PIE_KERNEL_COMMIT, PIE_KERNEL_FUSED, PIE_KERNEL_GROUPED, PIE_KERNEL_READINESS,
    PIE_KERNEL_SINGLETON,
};

/// One emitted kernel, or the reason it could not be emitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmittedKernel {
    /// The `PIE_KERNEL_*` discriminant naming which kernel family this is.
    pub kind: u32,
    /// The stage this kernel was emitted for.
    pub stage_index: u32,
    /// The region within that stage this kernel was emitted for.
    pub region_index: u32,
    /// The entry-point symbol a driver looks the kernel up by; empty when
    /// emission failed.
    pub entry_name: String,
    /// The generated kernel source, or empty when emission failed.
    pub source: String,
    /// The refusal text when emission failed, empty on success; copied across
    /// the C boundary for a human to read.
    pub error: String,
}

impl EmittedKernel {
    fn new(
        kind: u32,
        stage_index: usize,
        region_index: usize,
        entry_name: String,
        emitted: Result<String, EmitError>,
    ) -> Self {
        // The typed refusal becomes text here and only here: `error` is ABI,
        // copied across the C boundary for a human to read.
        let (source, error) = match emitted {
            Ok(source) => (source, String::new()),
            Err(error) => (String::new(), error.to_string()),
        };
        Self {
            kind,
            stage_index: stage_index as u32,
            region_index: region_index as u32,
            entry_name: if source.is_empty() {
                String::new()
            } else {
                entry_name
            },
            source,
            error,
        }
    }
}

/// The backends the host can generate for. The string form is what a driver
/// advertises in `DriverCapabilities::codegen_backend`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    /// CUDA, compiled by NVRTC; advertised as `"cuda"`.
    Cuda,
    /// Metal, compiled from MSL; advertised as `"metal"`.
    Metal,
}

impl Backend {
    /// Every backend, for callers that mean "all of them".
    ///
    /// A test that wants both backends would otherwise write the pair out,
    /// and a third backend would leave that test quietly covering two of
    /// three. `every_backend_is_in_all` keeps this honest against [`parse`].
    ///
    /// [`parse`]: Backend::parse
    pub const ALL: &'static [Backend] = &[Backend::Cuda, Backend::Metal];

    /// Parse a driver's advertised backend. Unknown names mean "no host code
    /// generation", never a guess.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "cuda" => Some(Backend::Cuda),
            "metal" => Some(Backend::Metal),
            _ => None,
        }
    }

    /// The name a driver advertises. Inverse of [`Backend::parse`].
    pub fn name(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
        }
    }

    /// The emitter version a driver's compile cache must key on.
    pub fn emitter_version(self) -> u32 {
        match self {
            Backend::Cuda => crate::codegen::cuda::CUDA_GENERATED_EMITTER_VERSION as u32,
            Backend::Metal => crate::codegen::metal::METAL_M1_EMITTER_VERSION as u32,
        }
    }
}

/// Emit every kernel a driver needs for `stages`, in stage then region order.
///
/// One `match` owns the whole backend decision, and everything a backend does
/// lives inside its arm.
///
/// Program-level work placed outside the match — an `if backend ==
/// Backend::Metal` after the loop, say — is invisible to exhaustiveness
/// checking, so a third backend would compile, run, and quietly emit no
/// readiness or commit kernels at all. Inside the match, adding a backend is a
/// compile error here rather than a missing kernel in a driver.
pub fn emit_program(
    backend: Backend,
    stages: &[CompiledStage],
    bound: &BoundTrace,
) -> Vec<EmittedKernel> {
    let mut kernels = Vec::new();
    match backend {
        Backend::Cuda => {
            for (stage_index, stage) in stages.iter().enumerate() {
                emit_cuda_stage(stage, stage_index, &mut kernels);
            }
            // No program-level effect kernels: the CUDA driver's readiness and
            // commit are prebuilt tier-0 kernels, not generated ones.
        }
        Backend::Metal => {
            for (stage_index, stage) in stages.iter().enumerate() {
                emit_metal_stage(stage, stage_index, &mut kernels);
            }
            emit_metal_program_effects(bound, &mut kernels);
        }
    }
    kernels
}

fn signature(stage: &CompiledStage) -> String {
    format!("{:016x}", stage.signature.hash)
}

fn emit_cuda_stage(stage: &CompiledStage, stage_index: usize, out: &mut Vec<EmittedKernel>) {
    let signature = signature(stage);
    // The CUDA driver compiles one fused kernel per generated region and falls
    // back to the prebuilt tier-0 kernels elsewhere, so singleton regions need
    // no emission — `module_cache.hpp` only ever calls `emit_fused_region_cuda`.
    for (region_index, region) in stage.fused.regions.iter().enumerate() {
        let entry = format!("ptir_fused_{signature}_r{region_index}");
        let emitted = crate::codegen::cuda::emit_fused_region(&entry, stage, region);
        out.push(EmittedKernel::new(
            PIE_KERNEL_FUSED,
            stage_index,
            region_index,
            entry,
            emitted,
        ));
    }
}

fn emit_metal_stage(stage: &CompiledStage, stage_index: usize, out: &mut Vec<EmittedKernel>) {
    let signature = signature(stage);

    // M1: one dispatch per op. The kernel is a function of the op tag alone.
    match crate::codegen::metal::validate_singleton_plan(stage) {
        Ok(operations) => {
            for (region_index, meta) in operations.iter().enumerate() {
                let entry = format!("ptir_m1_{signature}_r{region_index}");
                let source = crate::codegen::metal::emit_singleton_region(&entry, meta.op.tag);
                out.push(EmittedKernel::new(
                    PIE_KERNEL_SINGLETON,
                    stage_index,
                    region_index,
                    entry,
                    Ok(source),
                ));
            }
        }
        Err(error) => {
            // The whole stage is unrepresentable on the singleton path; say so
            // once rather than per region.
            out.push(EmittedKernel::new(
                PIE_KERNEL_SINGLETON,
                stage_index,
                0,
                String::new(),
                Err(error),
            ));
        }
    }

    // M2: one kernel per fused region, bound directly to channel cells. The
    // driver refuses this form above `kMetalM2MaxFusedChannels`, and so do we —
    // emitting it anyway would produce a kernel that cannot be bound.
    let fused_supported = stage.normalized.channel_bindings.len()
        <= crate::codegen::metal::METAL_M2_MAX_FUSED_CHANNELS;
    for (region_index, region) in stage.fused.regions.iter().enumerate() {
        let entry = format!("ptir_m2_{signature}_r{region_index}");
        let emitted = if fused_supported {
            crate::codegen::metal::emit_fused_region(&entry, stage, region)
        } else {
            Err(EmitError::ChannelLimitExceeded {
                emitter: EmitterKind::MetalFused,
                limit: crate::codegen::metal::METAL_M2_MAX_FUSED_CHANNELS,
            })
        };
        out.push(EmittedKernel::new(
            PIE_KERNEL_FUSED,
            stage_index,
            region_index,
            entry,
            emitted,
        ));
    }

    // M3: the grouped forms, which serve every lane in a group from one launch.
    // Singleton regions get the grouped-fused treatment; fused regions pick the
    // library kernel their `library_op` names.
    for (region_index, region) in stage.singleton.regions.iter().enumerate() {
        let entry = format!("ptir_m3s_{signature}_r{region_index}");
        let emitted = crate::codegen::metal::emit_grouped_fused_region(&entry, stage, region);
        out.push(EmittedKernel::new(
            PIE_KERNEL_GROUPED,
            stage_index,
            region_index,
            entry,
            emitted,
        ));
    }
    for (region_index, region) in stage.fused.regions.iter().enumerate() {
        let entry = format!("ptir_m3_{signature}_r{region_index}");
        let emitted = match grouped_library(stage, region) {
            Some(LibraryOp::NucleusSample) => {
                crate::codegen::metal::emit_grouped_nucleus(&entry, stage, region)
            }
            Some(LibraryOp::TopK) => {
                crate::codegen::metal::emit_grouped_topk(&entry, stage, region)
            }
            _ => crate::codegen::metal::emit_grouped_fused_region(&entry, stage, region),
        };
        out.push(EmittedKernel::new(
            PIE_KERNEL_GROUPED,
            stage_index,
            stage.singleton.regions.len() + region_index,
            entry,
            emitted,
        ));
    }

    // The grouped readiness and commit kernels are shared across a group, so
    // they are named by emitter version rather than by program. They sit at
    // region 0; the per-program single-lane forms the M1 and M2 launch paths
    // bind are emitted once per program at region 1 by `emit_program`, because
    // their channel effects are program-wide and a stage cannot see them.
    let version = crate::codegen::metal::METAL_M1_EMITTER_VERSION;
    let ready = format!("ptir_m3_generic_ready_v{version}");
    let source = crate::codegen::metal::emit_grouped_readiness(&ready);
    out.push(EmittedKernel::new(
        PIE_KERNEL_READINESS,
        stage_index,
        0,
        ready,
        Ok(source),
    ));
    let commit = format!("ptir_m3_generic_commit_v{version}");
    let source = crate::codegen::metal::emit_grouped_commit(&commit);
    out.push(EmittedKernel::new(
        PIE_KERNEL_COMMIT,
        stage_index,
        0,
        commit,
        Ok(source),
    ));
}

/// The single-lane readiness and commit kernels, specialised to this program's
/// channel effects.
///
/// These are what `m1_runtime.cpp` binds on the M1 and M2 paths — a different
/// buffer shape from the grouped M3 forms above, so they cannot share a slot.
/// They are program-wide rather than per-stage, hence `(stage 0, region 1)`.
fn emit_metal_program_effects(bound: &BoundTrace, out: &mut Vec<EmittedKernel>) {
    let effects = crate::codegen::metal::channel_effects(bound);
    let signature = format!("{:016x}", bound.hash);
    let ready = format!("ptir_m1_{signature}_ready");
    let source = crate::codegen::metal::emit_readiness(&ready, &effects);
    out.push(EmittedKernel::new(
        PIE_KERNEL_READINESS,
        0,
        1,
        ready,
        source,
    ));
    let commit = format!("ptir_m1_{signature}_commit");
    let source = crate::codegen::metal::emit_commit(&commit, &effects);
    out.push(EmittedKernel::new(PIE_KERNEL_COMMIT, 0, 1, commit, source));
}

/// Which library kernel a grouped region should use, reproducing the driver's
/// `parallel_nucleus` / `parallel_topk` tests.
fn grouped_library(stage: &CompiledStage, region: &Region) -> Option<LibraryOp> {
    let RegionKind::Library(op) = region.kind else {
        return None;
    };
    match op {
        LibraryOp::NucleusSample => Some(LibraryOp::NucleusSample),
        LibraryOp::TopK => {
            // The driver additionally checks the node really is a `top_k`, so a
            // mislabelled region falls to the generic emitter instead of a
            // kernel that would read the wrong operands.
            let node = region.nodes.first()?.index();
            let op = stage.normalized.ops.get(node)?;
            (crate::codegen::op_view::OpView::of(op).tag == tags::TOP_K).then_some(LibraryOp::TopK)
        }
        // Listed rather than caught by `_`: a new library op has no grouped
        // kernel until someone writes one, and it should be this match that
        // says so.
        LibraryOp::Sort | LibraryOp::Scan | LibraryOp::MatMul | LibraryOp::SecondParty => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Walk the enum by successor. Odd on purpose: the `match` is exhaustive,
    /// so a new variant does not compile until it is given a place in the
    /// order, and the walk that place produces is what [`Backend::ALL`] is
    /// then required to equal. A plain `for backend in Backend::ALL` cannot do
    /// that — it only ever visits what the list already contains, which is the
    /// thing in question.
    fn walk() -> Vec<Backend> {
        let mut out = Vec::new();
        let mut next = Some(Backend::Cuda);
        while let Some(backend) = next {
            out.push(backend);
            next = match backend {
                Backend::Cuda => Some(Backend::Metal),
                Backend::Metal => None,
            };
        }
        out
    }

    /// [`Backend::ALL`] is the whole enum, and every entry round-trips through
    /// the two string conversions beside it. Without this, a third backend
    /// could reach `name`, `emitter_version` and `emit_program` — all three of
    /// which the compiler does insist on — and still quietly halve every "for
    /// both backends" test that iterates `ALL`.
    #[test]
    fn all_is_the_whole_enum_and_round_trips() {
        assert_eq!(Backend::ALL, walk().as_slice());
        for backend in Backend::ALL {
            assert_eq!(Backend::parse(backend.name()), Some(*backend));
        }
        assert_eq!(Backend::parse("vulkan"), None);
    }
}
