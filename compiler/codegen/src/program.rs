//! Whole-program emission: a bound trace in, one kernel table out.
//!
//! The per-region emitters in [`crate::cuda`] and [`crate::metal`] are faithful
//! ports of what the drivers used to call, one region at a time. This module is
//! the part that used to live in the drivers' *runtime* files — the walk that
//! decides which emitter each region goes through, and what the entry point is
//! called. Keeping that decision here is the point of the whole move: the
//! driver receives a table and compiles it, instead of re-deriving from the
//! plan what the host already worked out.
//!
//! Naming and the emitter-selection rules are reproduced from
//! `driver/metal/src/pipeline/m1_runtime.cpp` and
//! `driver/cuda/src/pipeline/generated/module_cache.hpp`; a driver that reads
//! this table must find the same entry names it used to generate.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use pie_plan::{CompiledStage, LibraryOp, Region, RegionKind};

/// Kind discriminants, matching `PIE_KERNEL_*` in the driver ABI.
pub const KERNEL_SINGLETON: u32 = 0;
pub const KERNEL_FUSED: u32 = 1;
pub const KERNEL_GROUPED: u32 = 2;
pub const KERNEL_READINESS: u32 = 3;
pub const KERNEL_COMMIT: u32 = 4;

/// One emitted kernel, or the reason it could not be emitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmittedKernel {
    pub kind: u32,
    pub stage_index: u32,
    pub region_index: u32,
    pub entry_name: String,
    pub source: String,
    pub error: String,
}

impl EmittedKernel {
    fn new(
        kind: u32,
        stage_index: usize,
        region_index: usize,
        entry_name: String,
        emitted: Result<String, String>,
    ) -> Self {
        let (source, error) = match emitted {
            Ok(source) => (source, String::new()),
            Err(error) => (String::new(), error),
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
    Cuda,
    Metal,
}

impl Backend {
    /// Parse a driver's advertised backend. Unknown names mean "no host code
    /// generation", never a guess.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "cuda" => Some(Backend::Cuda),
            "metal" => Some(Backend::Metal),
            _ => None,
        }
    }

    /// The emitter version a driver's compile cache must key on.
    pub fn emitter_version(self) -> u32 {
        match self {
            Backend::Cuda => crate::cuda::CUDA_GENERATED_EMITTER_VERSION as u32,
            Backend::Metal => crate::metal::METAL_M1_EMITTER_VERSION as u32,
        }
    }
}

/// Emit every kernel a driver needs for `stages`, in stage then region order.
pub fn emit_program(backend: Backend, stages: &[CompiledStage]) -> Vec<EmittedKernel> {
    let mut kernels = Vec::new();
    for (stage_index, stage) in stages.iter().enumerate() {
        match backend {
            Backend::Cuda => emit_cuda_stage(stage, stage_index, &mut kernels),
            Backend::Metal => emit_metal_stage(stage, stage_index, &mut kernels),
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
        let emitted = crate::cuda::emit_fused_region(&entry, stage, region);
        out.push(EmittedKernel::new(
            KERNEL_FUSED,
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
    match crate::metal::validate_singleton_plan(stage) {
        Ok(operations) => {
            for (region_index, meta) in operations.iter().enumerate() {
                let entry = format!("ptir_m1_{signature}_r{region_index}");
                let source = crate::metal::emit_singleton_region(&entry, meta.op.tag);
                out.push(EmittedKernel::new(
                    KERNEL_SINGLETON,
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
                KERNEL_SINGLETON,
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
    let fused_supported =
        stage.normalized.channel_bindings.len() <= crate::metal::METAL_M2_MAX_FUSED_CHANNELS;
    for (region_index, region) in stage.fused.regions.iter().enumerate() {
        let entry = format!("ptir_m2_{signature}_r{region_index}");
        let emitted = if fused_supported {
            crate::metal::emit_fused_region(&entry, stage, region)
        } else {
            Err("fused region exceeds the 12-channel direct-binding limit".to_string())
        };
        out.push(EmittedKernel::new(
            KERNEL_FUSED,
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
        let emitted = crate::metal::emit_grouped_fused_region(&entry, stage, region);
        out.push(EmittedKernel::new(
            KERNEL_GROUPED,
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
                crate::metal::emit_grouped_nucleus(&entry, stage, region)
            }
            Some(LibraryOp::TopK) => crate::metal::emit_grouped_topk(&entry, stage, region),
            _ => crate::metal::emit_grouped_fused_region(&entry, stage, region),
        };
        out.push(EmittedKernel::new(
            KERNEL_GROUPED,
            stage_index,
            stage.singleton.regions.len() + region_index,
            entry,
            emitted,
        ));
    }

    // The readiness and commit kernels are shared across a group, so they are
    // named by emitter version rather than by program.
    let version = crate::metal::METAL_M1_EMITTER_VERSION;
    let ready = format!("ptir_m3_generic_ready_v{version}");
    let source = crate::metal::emit_grouped_readiness(&ready);
    out.push(EmittedKernel::new(
        KERNEL_READINESS,
        stage_index,
        0,
        ready,
        Ok(source),
    ));
    let commit = format!("ptir_m3_generic_commit_v{version}");
    let source = crate::metal::emit_grouped_commit(&commit);
    out.push(EmittedKernel::new(
        KERNEL_COMMIT,
        stage_index,
        0,
        commit,
        Ok(source),
    ));
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
            let node = *region.nodes.first()? as usize;
            let op = stage.normalized.ops.get(node)?;
            (crate::op_view::OpView::of(op).tag == crate::metal::validate::OP_TOP_K)
                .then_some(LibraryOp::TopK)
        }
        _ => None,
    }
}
