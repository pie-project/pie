use crate::plan::index::PlanIndex;
use crate::plan::{
    LoadPlan, SourceExtent, StorageInstr, StorageTarget, TILE_MAP_BIAS, TILE_MAP_CAST,
    TILE_MAP_DECODE, TILE_MAP_ENCODE, TILE_MAP_REBLOCK, TILE_MAP_REPACK, TILE_MAP_SCALE,
    TILE_MAP_UNARY, TileMapKind,
};
use crate::types::{BackendKind, BufferId, DType, Encoding, QuantScheme};

pub const CUDA_CAST_FP32_TO_BF16: &str = "quant::cast_fp32_to";
pub const CUDA_SCALE_ROWS_BF16: &str = "quant::scale_rows";

pub const CUDA_TILE_MAP_MASK: u32 =
    TILE_MAP_CAST | TILE_MAP_SCALE | TILE_MAP_DECODE | TILE_MAP_BIAS;

pub const METAL_TILE_MAP_MASK: u32 =
    TILE_MAP_CAST | TILE_MAP_SCALE | TILE_MAP_DECODE | TILE_MAP_BIAS;

pub const VULKAN_TILE_MAP_MASK: u32 = TILE_MAP_CAST | TILE_MAP_SCALE | TILE_MAP_DECODE;

pub const WGPU_TILE_MAP_MASK: u32 = TILE_MAP_CAST | TILE_MAP_SCALE | TILE_MAP_DECODE;

pub const HOST_TILE_MAP_MASK: u32 =
    TILE_MAP_CAST | TILE_MAP_REBLOCK | TILE_MAP_SCALE | TILE_MAP_BIAS | TILE_MAP_UNARY;

pub const CONVERT_TILE_MAP_MASK: u32 =
    HOST_TILE_MAP_MASK | TILE_MAP_ENCODE | TILE_MAP_DECODE | TILE_MAP_REPACK;

pub fn compilable_tile_maps(backend: BackendKind) -> u32 {
    match backend {
        BackendKind::Cuda => CUDA_TILE_MAP_MASK,
        BackendKind::Metal => METAL_TILE_MAP_MASK,
        BackendKind::Vulkan => VULKAN_TILE_MAP_MASK,
        BackendKind::Wgpu => WGPU_TILE_MAP_MASK,
        BackendKind::Unknown => HOST_TILE_MAP_MASK,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileMapFacts {
    pub kind: TileMapKind,
    pub transform_from: Option<QuantScheme>,
    pub transform_to: Option<QuantScheme>,
    pub source_dtype: Option<DType>,
    pub has_source: bool,
    pub compact_source: bool,
    pub shape: Option<(u64, u64)>,
    pub max_tile_bytes: u64,
    pub dest_dtype: Option<DType>,
    pub in_place: bool,
    pub blocked_scale: bool,
    pub operands_in_arena: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileLowering {
    pub kernel: Option<&'static str>,
}

pub fn lower(plan: &mut LoadPlan) -> usize {
    let target = plan.target.clone();
    let index = PlanIndex::new(plan);

    let facts: Vec<Option<TileMapFacts>> = plan
        .instrs
        .iter()
        .map(|instr| tile_map_facts(plan, &index, instr))
        .collect();

    let mut named = 0;
    for (instr, facts) in plan.instrs.iter_mut().zip(facts) {
        let (Some(facts), StorageInstr::TileMap { transform, .. }) = (facts, instr) else {
            continue;
        };
        let lowering = lower_tile_map(&facts, &target);
        transform.kernel = lowering.kernel.map(str::to_string);
        named += usize::from(lowering.kernel.is_some());
    }
    named
}

pub(super) fn lower_backend_tiling(plan: &mut LoadPlan) -> crate::error::Result<usize> {
    Ok(lower(plan))
}

fn lower_tile_map(facts: &TileMapFacts, target: &StorageTarget) -> TileLowering {
    let kernel = |chosen| facts.operands_in_arena.then_some(chosen).flatten();
    match target.backend {
        BackendKind::Cuda => TileLowering {
            kernel: kernel(cuda_kernel(facts)),
        },
        BackendKind::Metal | BackendKind::Vulkan | BackendKind::Wgpu | BackendKind::Unknown => {
            TileLowering::default()
        }
    }
}

pub(crate) fn kernel_for(facts: &TileMapFacts, target: &StorageTarget) -> Option<&'static str> {
    lower_tile_map(facts, target).kernel
}

pub(crate) fn facts_of(
    plan: &LoadPlan,
    index: &PlanIndex,
    instr: &StorageInstr,
) -> Option<TileMapFacts> {
    tile_map_facts(plan, index, instr)
}

fn cuda_kernel(facts: &TileMapFacts) -> Option<&'static str> {
    match facts.kind {
        TileMapKind::Cast => (facts.source_dtype == Some(DType::F32)
            && facts.dest_dtype == Some(DType::Bf16))
        .then_some(CUDA_CAST_FP32_TO_BF16),
        TileMapKind::Scale => (facts.blocked_scale
            && facts.in_place
            && facts.source_dtype == Some(DType::Bf16)
            && facts.dest_dtype == Some(DType::Bf16)
            && facts.shape.is_some())
        .then_some(CUDA_SCALE_ROWS_BF16),
        TileMapKind::Bias => None,
        TileMapKind::Encode
        | TileMapKind::Decode
        | TileMapKind::Transcode
        | TileMapKind::Reblock
        | TileMapKind::Unary
        | TileMapKind::Repack => None,
    }
}

fn extent_is_compact(extent: &crate::extent::Extent) -> bool {
    let mut stride = i64::from(extent.element_bytes);
    for dim in extent.dims.iter().rev() {
        if dim.src_stride != stride || dim.dst_stride != stride {
            return false;
        }
        match stride.checked_mul(dim.count) {
            Some(next) => stride = next,
            None => return false,
        }
    }
    true
}

fn encoding_dtype(encoding: &Encoding) -> DType {
    match encoding {
        Encoding::Raw(dtype) => *dtype,
        Encoding::Quant(spec) => spec.logical_dtype,
    }
}

fn tile_map_facts(
    plan: &LoadPlan,
    index: &PlanIndex,
    instr: &StorageInstr,
) -> Option<TileMapFacts> {
    let StorageInstr::TileMap {
        kind,
        source,
        dest,
        inputs,
        outputs,
        tile,
        transform,
        ..
    } = instr
    else {
        return None;
    };
    Some(TileMapFacts {
        kind: *kind,
        transform_from: transform.from,
        transform_to: transform.to,
        source_dtype: source_dtype(plan, index, source.as_ref(), inputs),
        has_source: source.is_some(),
        compact_source: source
            .as_ref()
            .is_none_or(|source| extent_is_compact(&source.stride)),
        shape: outputs
            .first()
            .and_then(|buffer| logical_shape(plan, *buffer)),
        max_tile_bytes: tile.max_tile_bytes,
        dest_dtype: outputs.first().and_then(|buffer| raw_dtype(plan, *buffer)),
        in_place: rewrites_in_place(plan, source.as_ref(), inputs, outputs, dest.as_ref()),
        blocked_scale: !transform.scale_blocks.is_empty(),
        operands_in_arena: inputs
            .iter()
            .chain(outputs)
            .chain(dest.as_ref().map(|dest| &dest.buffer))
            .all(|buffer| in_arena(plan, *buffer)),
    })
}

fn in_arena(plan: &LoadPlan, id: BufferId) -> bool {
    let mut id = id;
    for _ in 0..MAX_VIEW_HOPS {
        let Ok(decl) = plan.buffer(id) else {
            return false;
        };
        if decl.arena_offset().is_some() {
            return true;
        }
        let base = plan.instrs.iter().find_map(|instr| match instr {
            StorageInstr::CreateView { input, output, .. } if *output == id => Some(*input),
            _ => None,
        });
        match base {
            Some(base) => id = base,
            None => return false,
        }
    }
    false
}

const MAX_VIEW_HOPS: usize = 16;

fn raw_dtype(plan: &LoadPlan, buffer: BufferId) -> Option<DType> {
    match plan.buffer(buffer).ok()?.ty.encoding {
        Encoding::Raw(dtype) => Some(dtype),
        Encoding::Quant(_) => None,
    }
}

fn rewrites_in_place(
    plan: &LoadPlan,
    source: Option<&SourceExtent>,
    inputs: &[BufferId],
    outputs: &[BufferId],
    dest: Option<&crate::plan::DestExtent>,
) -> bool {
    if source.is_some() {
        return false;
    }
    let Some(&src) = inputs.first() else {
        return false;
    };
    match dest {
        Some(dest) => {
            let Ok(decl) = plan.buffer(dest.buffer) else {
                return false;
            };
            dest.buffer == src
                && dest.offset + dest.stride.base_offset == 0
                && extent_bytes(&dest.stride) == decl.bytes
        }
        None => outputs.first() == Some(&src),
    }
}

fn extent_bytes(extent: &crate::extent::Extent) -> u64 {
    extent
        .dims
        .iter()
        .try_fold(1u64, |n, d| u64::try_from(d.count).ok().map(|c| n * c))
        .unwrap_or(0)
        * u64::from(extent.element_bytes)
}

fn source_dtype(
    plan: &LoadPlan,
    index: &PlanIndex,
    source: Option<&SourceExtent>,
    inputs: &[BufferId],
) -> Option<DType> {
    if let Some(source) = source {
        return index
            .source(plan, source.tensor_id)
            .map(|decl| encoding_dtype(&decl.encoding));
    }
    plan.buffer(*inputs.first()?)
        .ok()
        .map(|decl| encoding_dtype(&decl.ty.encoding))
}

fn logical_shape(plan: &LoadPlan, buffer: BufferId) -> Option<(u64, u64)> {
    let (rows, cols) = crate::types::rectangle(&plan.buffer(buffer).ok()?.ty.shape)?;
    Some((u64::try_from(rows).ok()?, u64::try_from(cols).ok()?))
}

#[cfg(test)]
mod tests;
