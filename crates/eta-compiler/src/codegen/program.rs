use serde::{Deserialize, Serialize};

use crate::codegen::error::{EmitError, EmitterKind};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::plan::{CompiledStage, LibraryOp, Region, RegionKind};
use eta_ir::op::tags;
use eta_ir::validate::BoundTrace;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum KernelKind {
    #[default]
    Singleton = 0,
    Fused = 1,
    Grouped = 2,
    Readiness = 3,
    Commit = 4,
    Streamed = 5,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmittedKernel {
    pub kind: KernelKind,
    pub stage_index: u32,
    pub region_index: u32,
    pub entry_name: String,
    pub source: String,
    pub error: String,
    #[serde(default)]
    pub steps: Vec<u32>,
}

impl EmittedKernel {
    fn new(
        kind: KernelKind,
        stage_index: usize,
        region_index: usize,
        entry_name: String,
        emitted: Result<String, EmitError>,
    ) -> Self {
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
            steps: Vec::new(),
            source,
            error,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    Cuda,
    Metal,
}

impl Backend {
    pub const ALL: &'static [Backend] = &[Backend::Cuda, Backend::Metal];

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "cuda" => Some(Backend::Cuda),
            "metal" => Some(Backend::Metal),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
        }
    }

    pub fn emitter_version(self) -> u32 {
        match self {
            Backend::Cuda => crate::codegen::cuda::CUDA_GENERATED_EMITTER_VERSION as u32,
            Backend::Metal => crate::codegen::metal::METAL_M1_EMITTER_VERSION as u32,
        }
    }
}

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
            out.push(EmittedKernel::new(
                KernelKind::Singleton,
                stage_index,
                0,
                String::new(),
                Err(error),
            ));
        }
    }

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

    for (region_index, region) in stage.fused.regions.iter().enumerate() {
        let entry = format!("ptir_m4_{signature}_r{region_index}");
        let answer = match grouped_library(stage, region) {
            Some(LibraryOp::TopK) => {
                crate::codegen::metal::emit_streamed_topk(&entry, stage, region)
            }
            Some(LibraryOp::NucleusSample) | None => {
                crate::codegen::metal::emit_streamed_region(&entry, stage, region)
            }
            Some(_) => Err(crate::codegen::error::EmitError::LibraryRegionAbiInvalid(
                crate::codegen::error::RegionForm::GroupedFused,
            )),
        };
        let (emitted, steps) = match answer {
            Ok((source, steps)) => (Ok(source), steps),
            Err(error) => (Err(error), Vec::new()),
        };
        let mut kernel = EmittedKernel::new(
            KernelKind::Streamed,
            stage_index,
            region_index,
            entry,
            emitted,
        );
        kernel.steps = steps;
        out.push(kernel);
    }

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

fn grouped_library(stage: &CompiledStage, region: &Region) -> Option<LibraryOp> {
    let RegionKind::Library(op) = region.kind else {
        return None;
    };
    match op {
        LibraryOp::NucleusSample => Some(LibraryOp::NucleusSample),
        LibraryOp::TopK => {
            let node = region.nodes.first()?.index();
            let op = stage.normalized.ops.get(node)?;
            (crate::codegen::op_view::OpView::of(op).tag == tags::TOP_K).then_some(LibraryOp::TopK)
        }
        LibraryOp::Sort | LibraryOp::Scan | LibraryOp::MatMul | LibraryOp::SecondParty => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

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

    #[test]
    fn all_is_the_whole_enum_and_round_trips() {
        assert_eq!(Backend::ALL, walk().as_slice());
        for backend in Backend::ALL {
            assert_eq!(Backend::parse(backend.name()), Some(*backend));
        }
        assert_eq!(Backend::parse("vulkan"), None);
    }
}
