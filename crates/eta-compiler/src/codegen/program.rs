//! Whole-program emission: a bound trace in, one kernel table out. Owns the
//! walk over the per-region emitters in [`crate::codegen::cuda`] and
//! [`crate::codegen::metal`]: which emitter each region goes through, and
//! what its entry point is called. Entry names and emitter-selection rules
//! are an ABI shared with the engines' runtime C++.

use serde::{Deserialize, Serialize};

use crate::codegen::error::{EmitError, EmitterKind};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::plan::{CompiledStage, LibraryOp, Region, RegionKind};
use eta_ir::op::tags;
use eta_ir::validate::BoundTrace;

/// What an emitted kernel is for. Discriminants are a wire numbering an
/// engine reads, written out explicitly rather than left implicit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[derive(Serialize, Deserialize)]
#[repr(u32)]
pub enum KernelKind {
    /// One region, launched alone.
    #[default]
    Singleton = 0,
    /// A fused run of regions.
    Fused = 1,
    /// The grouped launch covering a whole stage.
    Grouped = 2,
    /// The readiness control kernel.
    Readiness = 3,
    /// The commit control kernel.
    Commit = 4,
}

/// One emitted kernel, or the reason it could not be emitted — the record
/// an engine receives, not a private staging form of one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[derive(Serialize, Deserialize)]
pub struct EmittedKernel {
    /// Which kernel family this is.
    pub kind: KernelKind,
    /// The stage this kernel was emitted for.
    pub stage_index: u32,
    /// The region within that stage this kernel was emitted for.
    pub region_index: u32,
    /// The entry-point symbol an engine looks the kernel up by; empty when
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
        kind: KernelKind,
        stage_index: usize,
        region_index: usize,
        entry_name: String,
        emitted: Result<String, EmitError>,
    ) -> Self {
        // The typed refusal becomes text here and only here.
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

/// The backends the host can generate for. The string form is what an engine
/// advertises in `EngineCapabilities::codegen_backend`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    /// CUDA, compiled by NVRTC; advertised as `"cuda"`.
    Cuda,
    /// Metal, compiled from MSL; advertised as `"metal"`.
    Metal,
}

impl Backend {
    /// Every backend, for callers that mean "all of them".
    pub const ALL: &'static [Backend] = &[Backend::Cuda, Backend::Metal];

    /// Parse an engine's advertised backend. Unknown names mean "no host code
    /// generation", never a guess.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "cuda" => Some(Backend::Cuda),
            "metal" => Some(Backend::Metal),
            _ => None,
        }
    }

    /// The name an engine advertises. Inverse of [`Backend::parse`].
    pub fn name(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
        }
    }

    /// The emitter version an engine's compile cache must key on.
    pub fn emitter_version(self) -> u32 {
        match self {
            Backend::Cuda => crate::codegen::cuda::CUDA_GENERATED_EMITTER_VERSION as u32,
            Backend::Metal => crate::codegen::metal::METAL_M1_EMITTER_VERSION as u32,
        }
    }
}

/// Emit every kernel an engine needs for `stages`, in stage then region
/// order. One `match` owns the whole backend decision so a new backend is a
/// compile error here rather than a missing kernel in an engine.
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
            // No program-level effect kernels: the CUDA engine's readiness and
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
    // Singleton regions need no emission: the shell only ever reads the
    // `KernelKind::Fused` slot; CUDA falls back to prebuilt tier-0 kernels.
    for (region_index, region) in stage.fused.regions.iter().enumerate() {
        let entry = format!("ptir_fused_{signature}_r{region_index}");
        let emitted = crate::codegen::cuda::emit_region(&entry, stage, region);
        out.push(EmittedKernel::new(
            KernelKind::Fused,
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
                    KernelKind::Singleton,
                    stage_index,
                    region_index,
                    entry,
                    Ok(source),
                ));
            }
        }
        Err(error) => {
            // Unrepresentable on the singleton path; say so once, not per region.
            out.push(EmittedKernel::new(
                KernelKind::Singleton,
                stage_index,
                0,
                String::new(),
                Err(error),
            ));
        }
    }

    // M2: one kernel per fused region, bound directly to channel cells.
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
            KernelKind::Fused,
            stage_index,
            region_index,
            entry,
            emitted,
        ));
    }

    // M3: grouped forms, serving every lane in a group from one launch.
    for (region_index, region) in stage.singleton.regions.iter().enumerate() {
        let entry = format!("ptir_m3s_{signature}_r{region_index}");
        let emitted = crate::codegen::metal::emit_grouped_fused_region(&entry, stage, region);
        out.push(EmittedKernel::new(
            KernelKind::Grouped,
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
            KernelKind::Grouped,
            stage_index,
            stage.singleton.regions.len() + region_index,
            entry,
            emitted,
        ));
    }

    // Shared across a group, so named by emitter version, not program; sit
    // at region 0 (the per-program single-lane forms are region 1).
    let version = crate::codegen::metal::METAL_M1_EMITTER_VERSION;
    let ready = format!("ptir_m3_generic_ready_v{version}");
    let source = crate::codegen::metal::emit_grouped_readiness(&ready);
    out.push(EmittedKernel::new(
        KernelKind::Readiness,
        stage_index,
        0,
        ready,
        Ok(source),
    ));
    let commit = format!("ptir_m3_generic_commit_v{version}");
    let source = crate::codegen::metal::emit_grouped_commit(&commit);
    out.push(EmittedKernel::new(
        KernelKind::Commit,
        stage_index,
        0,
        commit,
        Ok(source),
    ));
}

/// The single-lane readiness and commit kernels, specialised to this
/// program's channel effects — a different buffer shape from the grouped M3
/// forms, so they cannot share a slot. Program-wide, hence `(stage 0, region 1)`.
fn emit_metal_program_effects(bound: &BoundTrace, out: &mut Vec<EmittedKernel>) {
    let effects = crate::codegen::metal::channel_effects(bound);
    let signature = format!("{:016x}", bound.hash);
    let ready = format!("ptir_m1_{signature}_ready");
    let source = crate::codegen::metal::emit_readiness(&ready, &effects);
    out.push(EmittedKernel::new(
        KernelKind::Readiness,
        0,
        1,
        ready,
        source,
    ));
    let commit = format!("ptir_m1_{signature}_commit");
    let source = crate::codegen::metal::emit_commit(&commit, &effects);
    out.push(EmittedKernel::new(KernelKind::Commit, 0, 1, commit, source));
}

/// Which library kernel a grouped region should use, reproducing the engine's
/// `parallel_nucleus` / `parallel_topk` tests.
fn grouped_library(stage: &CompiledStage, region: &Region) -> Option<LibraryOp> {
    let RegionKind::Library(op) = region.kind else {
        return None;
    };
    match op {
        LibraryOp::NucleusSample => Some(LibraryOp::NucleusSample),
        LibraryOp::TopK => {
            // A mislabelled region falls to the generic emitter instead of a
            // kernel that would read the wrong operands.
            let node = region.nodes.first()?.index();
            let op = stage.normalized.ops.get(node)?;
            (crate::codegen::op_view::OpView::of(op).tag == tags::TOP_K).then_some(LibraryOp::TopK)
        }
        // Listed rather than caught by `_` so a new library op is a compile error.
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
