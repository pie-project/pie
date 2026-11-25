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
            // Held for the program's life: the emitter's growth is not.
            Ok(mut source) => {
                source.shrink_to_fit();
                (source, String::new())
            }
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
    Hip,
    Metal,
}

impl Backend {
    pub const ALL: &'static [Backend] = &[Backend::Cuda, Backend::Hip, Backend::Metal];

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "cuda" => Some(Backend::Cuda),
            "hip" => Some(Backend::Hip),
            "metal" => Some(Backend::Metal),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Hip => "hip",
            Backend::Metal => "metal",
        }
    }

    pub fn emitter_version(self) -> u32 {
        match self {
            Backend::Cuda => crate::codegen::cuda::CUDA_GENERATED_EMITTER_VERSION as u32,
            Backend::Hip => crate::codegen::hip::HIP_GENERATED_EMITTER_VERSION as u32,
            Backend::Metal => crate::codegen::metal::METAL_M1_EMITTER_VERSION as u32,
        }
    }

    /// The library forms this backend's emitters serve
    /// ([`crate::plan::LibraryForms`]): what the planner may form for it.
    #[must_use]
    pub const fn library_forms(self) -> crate::plan::LibraryForms {
        match self {
            Backend::Metal => crate::codegen::metal::LIBRARY_FORMS,
            Backend::Cuda | Backend::Hip => crate::plan::LibraryForms::ALL,
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
                emit_simt_stage(
                    crate::codegen::cuda::DIALECT,
                    stage,
                    stage_index,
                    &mut kernels,
                );
            }
        }
        Backend::Hip => {
            for (stage_index, stage) in stages.iter().enumerate() {
                emit_simt_stage(
                    crate::codegen::hip::DIALECT,
                    stage,
                    stage_index,
                    &mut kernels,
                );
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

fn emit_simt_stage(
    dialect: crate::codegen::simt::Dialect,
    stage: &CompiledStage,
    stage_index: usize,
    out: &mut Vec<EmittedKernel>,
) {
    let signature = signature(stage);
    for (region_index, region) in stage.fused.regions.iter().enumerate() {
        let entry = format!("ptir_fused_{signature}_r{region_index}");
        let emitted = crate::codegen::simt::emit_region(dialect, &entry, stage, region);
        out.push(EmittedKernel::new(
            KernelKind::Fused,
            stage_index,
            region_index,
            entry,
            emitted,
        ));
    }
}

/// The entry a Metal kernel is emitted under, until its text names it
/// ([`push_metal`]).
const METAL_UNNAMED: &str = "ptir_metal_unnamed_entry";

/// Takes a Metal kernel emitted under [`METAL_UNNAMED`], its entry named by
/// a hash of that text: macOS keeps a compiled library by its source, so
/// regions of one text are one source whatever program, stage or region
/// they are of.
fn push_metal(
    out: &mut Vec<EmittedKernel>,
    kind: KernelKind,
    stage_index: usize,
    region_index: usize,
    prefix: &str,
    emitted: Result<String, EmitError>,
) {
    let (entry, emitted) = match emitted {
        Ok(source) => {
            let hash = eta_ir::fnv1a64(source.as_bytes());
            let entry = format!("ptir_{prefix}_{hash:016x}");
            let source = source.replace(METAL_UNNAMED, &entry);
            (entry, Ok(source))
        }
        Err(error) => (String::new(), Err(error)),
    };
    out.push(EmittedKernel::new(kind, stage_index, region_index, entry, emitted));
}

fn emit_metal_stage(stage: &CompiledStage, stage_index: usize, out: &mut Vec<EmittedKernel>) {
    let entry = METAL_UNNAMED;

    let fused_supported = stage.normalized.channel_bindings.len()
        <= crate::codegen::metal::METAL_M2_MAX_FUSED_CHANNELS;
    for (region_index, region) in stage.fused.regions.iter().enumerate() {
        let emitted = if fused_supported {
            crate::codegen::metal::emit_fused_region(entry, stage, region)
        } else {
            Err(EmitError::ChannelLimitExceeded {
                emitter: EmitterKind::MetalFused,
                limit: crate::codegen::metal::METAL_M2_MAX_FUSED_CHANNELS,
            })
        };
        push_metal(
            out,
            KernelKind::Fused,
            stage_index,
            region_index,
            "m2",
            emitted,
        );
    }

    // Past the singleton partition's slots, which hold no kernel: the
    // engine asks a region's grouped form at that offset.
    for (region_index, region) in stage.fused.regions.iter().enumerate() {
        let emitted = match grouped_library(stage, region) {
            Some(LibraryOp::NucleusSample) => {
                crate::codegen::metal::emit_grouped_nucleus(entry, stage, region)
            }
            Some(LibraryOp::TopK) => {
                crate::codegen::metal::emit_grouped_topk(entry, stage, region)
            }
            _ => crate::codegen::metal::emit_grouped_fused_region(entry, stage, region),
        };
        push_metal(
            out,
            KernelKind::Grouped,
            stage_index,
            stage.singleton.regions.len() + region_index,
            "m3",
            emitted,
        );
    }

    for (region_index, region) in stage.fused.regions.iter().enumerate() {
        let answer = match grouped_library(stage, region) {
            Some(LibraryOp::TopK) => {
                crate::codegen::metal::emit_streamed_topk(entry, stage, region)
            }
            Some(LibraryOp::NucleusSample) | None => {
                crate::codegen::metal::emit_streamed_region(entry, stage, region)
            }
            Some(_) => Err(crate::codegen::error::EmitError::LibraryRegionAbiInvalid(
                crate::codegen::error::RegionForm::GroupedFused,
            )),
        };
        let (emitted, steps) = match answer {
            Ok((source, steps)) => (Ok(source), steps),
            Err(error) => (Err(error), Vec::new()),
        };
        push_metal(
            out,
            KernelKind::Streamed,
            stage_index,
            region_index,
            "m4",
            emitted,
        );
        if let Some(kernel) = out.last_mut() {
            kernel.steps = steps;
        }
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
    for (kind, prefix, source) in [
        (
            KernelKind::Readiness,
            "m1_ready",
            crate::codegen::metal::emit_readiness(METAL_UNNAMED, &effects),
        ),
        (
            KernelKind::Commit,
            "m1_commit",
            crate::codegen::metal::emit_commit(METAL_UNNAMED, &effects),
        ),
    ] {
        push_metal(out, kind, 0, 1, prefix, source);
    }
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
        LibraryOp::Sort
        | LibraryOp::Scan
        | LibraryOp::MatMul
        | LibraryOp::SecondParty
        | LibraryOp::RankSample
        | LibraryOp::LaneCut => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::{String, ToString};
    use alloc::vec;
    use alloc::vec::Vec;

    fn walk() -> Vec<Backend> {
        let mut out = Vec::new();
        let mut next = Some(Backend::Cuda);
        while let Some(backend) = next {
            out.push(backend);
            next = match backend {
                Backend::Cuda => Some(Backend::Hip),
                Backend::Hip => Some(Backend::Metal),
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

    /// Programs whose regions are one text emit one source, whatever else
    /// their stages sign: a channel's capacity, a literal (an op param). A
    /// different op is a different text, named apart.
    #[test]
    fn a_metal_region_is_named_by_its_text() {
        use crate::plan::compile_bound;
        use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
        use eta_ir::op::Op;
        use eta_ir::registry::{EngineCaps, Stage};
        use eta_ir::types::{Dtype, Literal, Shape};
        let emit = |op: Op, scale: f32, capacity: u32| {
            let decl = |host_role, capacity| ChannelDecl {
                shape: Shape::vector(64),
                dtype: ChanDType::Concrete(Dtype::F32),
                capacity,
                host_role,
                seeded: false,
            };
            let container = TraceContainer {
                channels: vec![decl(HostRole::Writer, 1), decl(HostRole::Reader, capacity)],
                stages: vec![StageProgram {
                    stage: Stage::Epilogue,
                    ops: vec![
                        Op::ChanTake(0),
                        Op::Const(Literal::F32(scale)),
                        op,
                        Op::ChanPut { chan: 1, value: 2 },
                    ],
                }],
                ..TraceContainer::default()
            };
            let bound = eta_ir::validate::bind(container, vec![Vec::new()], EngineCaps::default())
                .expect("binds");
            let stages = compile_bound(&bound);
            let regions: Vec<(String, String)> = emit_program(Backend::Metal, &stages, &bound)
                .into_iter()
                .filter(|kernel| !matches!(kernel.kind, KernelKind::Readiness | KernelKind::Commit))
                .map(|kernel| (kernel.entry_name.to_string(), kernel.source.to_string()))
                .collect();
            (stages[0].signature.hash, regions)
        };
        let (signed, regions) = emit(Op::Mul(0, 1), 0.5, 1);
        assert!(!regions.is_empty());
        for (other, scale, capacity) in [("capacity", 0.5, 4), ("literal", 0.25, 1)] {
            let (apart, same) = emit(Op::Mul(0, 1), scale, capacity);
            assert_ne!(signed, apart, "{other}");
            assert_eq!(regions, same, "{other}");
        }
        let (_, added) = emit(Op::Add(0, 1), 0.5, 1);
        for (at, (entry, source)) in added.iter().enumerate() {
            let (was, text) = &regions[at];
            assert_eq!(entry == was, source == text, "region {at}");
        }
        assert_ne!(regions, added);
    }

    /// Metal emits what its engine asks: no singleton kernel, and a grouped
    /// one only at a fused region's slot, past the singleton partition's.
    #[test]
    fn metal_emits_no_singleton_kernel() {
        use crate::plan::compile_bound;
        use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
        use eta_ir::op::Op;
        use eta_ir::registry::{EngineCaps, Stage};
        use eta_ir::types::{Dtype, Shape};
        let decl = |host_role| ChannelDecl {
            shape: Shape::vector(64),
            dtype: ChanDType::Concrete(Dtype::F32),
            capacity: 1,
            host_role,
            seeded: false,
        };
        let container = TraceContainer {
            channels: vec![decl(HostRole::Writer), decl(HostRole::Reader)],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0),
                    Op::Exp(0),
                    Op::Neg(1),
                    Op::ChanPut { chan: 1, value: 2 },
                ],
            }],
            ..TraceContainer::default()
        };
        let bound = eta_ir::validate::bind(container, vec![Vec::new()], EngineCaps::default())
            .expect("binds");
        let stages = compile_bound(&bound);
        let singles = stages[0].singleton.regions.len() as u32;
        let kernels = emit_program(Backend::Metal, &stages, &bound);
        assert!(kernels.iter().all(|kernel| kernel.kind != KernelKind::Singleton));
        let grouped: Vec<u32> = kernels
            .iter()
            .filter(|kernel| kernel.kind == KernelKind::Grouped)
            .map(|kernel| kernel.region_index)
            .collect();
        let regions = stages[0].fused.regions.len() as u32;
        assert_eq!(grouped, (singles..singles + regions).collect::<Vec<_>>());
    }
}
