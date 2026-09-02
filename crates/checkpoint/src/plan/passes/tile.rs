//! Backend tile-map lowering.
//!
//! A fact the engine can *measure* is data and travels in [`StorageTarget`];
//! a rule that differs per backend and cannot be parameterized is code and
//! lives here. Deciding the kernel/tile choice here (not at execution time)
//! puts the answer in the plan and thus in the plan hash.

use crate::plan::index::PlanIndex;
use crate::plan::{
    LoadPlan, SourceExtent, StorageInstr, StorageTarget, TILE_MAP_BIAS, TILE_MAP_CAST,
    TILE_MAP_DECODE, TILE_MAP_ENCODE, TILE_MAP_REBLOCK, TILE_MAP_REPACK, TILE_MAP_SCALE,
    TileMapKind,
};
use crate::types::{BackendKind, BufferId, DType, Encoding, QuantScheme};

/// The kernel rows a load may run on the device, by table symbol. Named
/// here rather than reached for out of `kernels-cuda`, so a plan for a CUDA
/// target compiles on machines with no CUDA toolchain. These strings are
/// written into a checked-in golden plan, so they don't track the Rust
/// namespace.
pub const CUDA_CAST_FP32_TO_BF16: &str = "quant::cast_fp32_to";
pub const CUDA_SCALE_ROWS_BF16: &str = "quant::scale_rows";

/// The transforms a plan for a CUDA target may carry. `Repack` and
/// `Reblock` have no host implementation, so a plan naming either is
/// refused at compile time rather than failing at execution. No serving
/// plan carries `Encode`: quantization runs once on the host at `pie model
/// import` (against [`CONVERT_TILE_MAP_MASK`]), not per boot.
pub const CUDA_TILE_MAP_MASK: u32 =
    TILE_MAP_CAST | TILE_MAP_SCALE | TILE_MAP_DECODE | TILE_MAP_BIAS;

/// The transforms a plan for a Metal target may carry. `engine-metal`'s
/// backing implements neither [`ArenaBacking::runs_named_kernels`] nor
/// `run_tile_map`, so every transform in a Metal plan runs on the host. No
/// serving plan carries `Encode`; see [`CUDA_TILE_MAP_MASK`].
///
/// [`ArenaBacking::runs_named_kernels`]: crate::executor::arena::ArenaBacking::runs_named_kernels
pub const METAL_TILE_MAP_MASK: u32 =
    TILE_MAP_CAST | TILE_MAP_SCALE | TILE_MAP_DECODE | TILE_MAP_BIAS;

/// The transforms a Vulkan plan may carry. Equal to Metal's, written out
/// rather than aliased so the two can diverge later without a silent
/// change.
///
/// [`ArenaBacking::runs_named_kernels`]: crate::executor::arena::ArenaBacking::runs_named_kernels
pub const VULKAN_TILE_MAP_MASK: u32 = TILE_MAP_CAST | TILE_MAP_SCALE | TILE_MAP_DECODE;

/// The transforms `host_executor` implements. Not a device capability: it is
/// what a plan compiled for no device may carry, which is the reference the
/// device answers are checked against.
pub const HOST_TILE_MAP_MASK: u32 =
    TILE_MAP_CAST | TILE_MAP_REBLOCK | TILE_MAP_SCALE | TILE_MAP_BIAS;

/// The mask offline conversion compiles against: everything `replay` runs
/// plus `Encode`, `Decode` and `Repack`. Separate from
/// [`HOST_TILE_MAP_MASK`], which is the verification surface for replaying
/// a *device* plan, while this mask's plan writes output back to a
/// checkpoint. `Encode`/`Repack` are import-time transforms; no device mask
/// carries either bit.
pub const CONVERT_TILE_MAP_MASK: u32 =
    HOST_TILE_MAP_MASK | TILE_MAP_ENCODE | TILE_MAP_DECODE | TILE_MAP_REPACK;

/// Which transforms a plan compiled for `backend` may carry. A compile-time
/// property, deliberately not "what the device runs" — that is a property
/// of the [`ArenaBacking`](crate::executor::arena::ArenaBacking) the caller
/// handed over and can be narrower on any given load; every transform this
/// admits also has a host implementation, so the narrower case is a slower
/// load rather than a failed one.
pub fn compilable_tile_maps(backend: BackendKind) -> u32 {
    match backend {
        BackendKind::Cuda => CUDA_TILE_MAP_MASK,
        BackendKind::Metal => METAL_TILE_MAP_MASK,
        BackendKind::Vulkan => VULKAN_TILE_MAP_MASK,
        BackendKind::Unknown => HOST_TILE_MAP_MASK,
    }
}

/// Everything the lowering rule is allowed to see about one `TileMap`.
/// Deliberately not the instruction itself: the plan walking happens once,
/// below, so the rule contains only the rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileMapFacts {
    pub kind: TileMapKind,
    pub transform_from: Option<QuantScheme>,
    pub transform_to: Option<QuantScheme>,
    /// The dtype the transform reads: the source tensor's for a checkpoint
    /// source, the input buffer's tensor dtype otherwise. `None` when neither
    /// resolves, which forces the conservative answer everywhere below.
    pub source_dtype: Option<DType>,
    pub has_source: bool,
    /// Whether the source extent is one contiguous run. A strided source
    /// can't be sliced by rows without re-deriving the stride per tile.
    pub compact_source: bool,
    /// Declared 2-D shape of the primary output, or `None` if it is not 2-D.
    /// Read from the instruction, not the target, since the build may narrow
    /// the budget for one instruction.
    pub shape: Option<(u64, u64)>,
    pub max_tile_bytes: u64,
    /// The dtype the transform WRITES, when the primary output is
    /// unquantized. `None` for a quantized destination, whose scheme is
    /// [`transform_to`](Self::transform_to) instead.
    pub dest_dtype: Option<DType>,
    /// Whether the transform rewrites its input where it lies. The per-row
    /// scale kernel multiplies in place, so a different destination buffer
    /// is not one it can run.
    pub in_place: bool,
    /// Whether a [`TileMapKind::Scale`] reads per-group factors from an
    /// operand, rather than multiplying by the uniform constant in
    /// `scale_factor_bits`.
    pub blocked_scale: bool,
    /// Whether every operand resolves to a span of the arena — a
    /// precondition of naming a kernel row: an operand the arena doesn't
    /// hold means `kernel = None`, so the plan visibly says the host runs
    /// this one instead of the executor discovering it at launch.
    pub operands_in_arena: bool,
}

/// What the lowering decided. Written into the instruction verbatim.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileLowering {
    /// The backend entry point these operands run as, or `None` for the host.
    /// See [`TransformSpec::kernel`](crate::plan::TransformSpec::kernel).
    pub kernel: Option<&'static str>,
}

/// Fill in every backend decision the plan carries. Only ever writes
/// decision fields — never adds, removes, or reorders instructions — so a
/// backend that declines to decide leaves the plan bit-identical.
pub fn lower(plan: &mut LoadPlan) -> usize {
    let target = plan.target.clone();
    let index = PlanIndex::new(plan);

    // Resolved up front: the write loop holds `&mut` on an instruction while
    // these lookups still need to read declarations.
    let facts: Vec<Option<TileMapFacts>> = plan
        .instrs
        .iter()
        .map(|instr| tile_map_facts(plan, &index, instr))
        .collect();

    let mut named = 0;
    for (instr, facts) in plan.instrs.iter_mut().zip(facts) {
        let (
            Some(facts),
            StorageInstr::TileMap {
                transform, ..
            },
        ) = (facts, instr)
        else {
            continue;
        };
        let lowering = lower_tile_map(&facts, &target);
        transform.kernel = lowering.kernel.map(str::to_string);
        named += usize::from(lowering.kernel.is_some());
    }
    named
}

/// [`lower`], as the pipeline runs it. The count is the kernels it named.
pub(super) fn lower_backend_tiling(plan: &mut LoadPlan) -> crate::error::Result<usize> {
    Ok(lower(plan))
}

fn lower_tile_map(facts: &TileMapFacts, target: &StorageTarget) -> TileLowering {
    // Nothing to name a row for: the operands are not where a kernel could
    // read them.
    let kernel = |chosen| facts.operands_in_arena.then_some(chosen).flatten();
    match target.backend {
        BackendKind::Cuda => TileLowering {
            kernel: kernel(cuda_kernel(facts)),
        },
        // Neither Metal nor Vulkan runs a transform: the host executor
        // derives its own tiling from `max_tile_bytes` at run time.
        BackendKind::Metal | BackendKind::Vulkan | BackendKind::Unknown => TileLowering::default(),
    }
}

/// Which row this target would run these operands as, or `None` for the
/// host. [`stage_device_transforms`] asks this about operands that don't
/// exist yet, to know before rewriting whether it buys anything.
///
/// [`stage_device_transforms`]: super::stage::stage_device_transforms
pub(crate) fn kernel_for(facts: &TileMapFacts, target: &StorageTarget) -> Option<&'static str> {
    lower_tile_map(facts, target).kernel
}

/// Everything the lowering rule reads about one instruction, extracted from
/// the plan.
///
/// `pub(crate)` for the staging pass, which asks [`kernel_for`] about a
/// modified copy of what this returns.
pub(crate) fn facts_of(
    plan: &LoadPlan,
    index: &PlanIndex,
    instr: &StorageInstr,
) -> Option<TileMapFacts> {
    tile_map_facts(plan, index, instr)
}

/// Which kernel row runs these operands, or `None` for the host. Deciding
/// here (rather than at launch, as the engine used to) means a refusal can
/// name the tensor rather than just "no kernel for these bytes".
fn cuda_kernel(facts: &TileMapFacts) -> Option<&'static str> {
    match facts.kind {
        // The one cast the table implements. Any other pair is refused, not
        // approximated: a cast with no kernel must never become a copy.
        TileMapKind::Cast => (facts.source_dtype == Some(DType::F32)
            && facts.dest_dtype == Some(DType::Bf16))
        .then_some(CUDA_CAST_FP32_TO_BF16),
        // `scale_rows` multiplies in place and reads its factors from an
        // operand; a uniform factor has no row.
        TileMapKind::Scale => (facts.blocked_scale
            && facts.in_place
            && facts.source_dtype == Some(DType::Bf16)
            && facts.dest_dtype == Some(DType::Bf16)
            && facts.shape.is_some())
        .then_some(CUDA_SCALE_ROWS_BF16),
        // A bias reconciles a checkpoint format's constant at import time,
        // which runs on the host; a device plan asking for one is refused.
        TileMapKind::Bias => None,
        // No device mask admits `Encode`; it runs on the host at
        // `pie model import`.
        TileMapKind::Encode
        | TileMapKind::Decode
        | TileMapKind::Transcode
        | TileMapKind::Reblock
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

/// The dtype a transform sees, which for a quantized encoding is the logical
/// one.
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

/// Whether a buffer resolves to a span of the arena, through views. The same
/// walk the executor's `resolve` does — a window on a resident buffer is in
/// the arena.
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

/// How deep a chain of views may go before the walk gives up; the same guard
/// `passes::arena` and `passes::validate` use, for the same reason.
const MAX_VIEW_HOPS: usize = 16;

/// The dtype behind a buffer, when it is unquantized. `None` for a quantized
/// destination rather than its logical dtype, so a rule can't read "bf16"
/// off an MXFP4 output.
fn raw_dtype(plan: &LoadPlan, buffer: BufferId) -> Option<DType> {
    match plan.buffer(buffer).ok()?.ty.encoding {
        Encoding::Raw(dtype) => Some(dtype),
        Encoding::Quant(_) => None,
    }
}

/// Whether the transform's destination is the same bytes as its input.
/// Answered by buffer identity (the same buffer, covered whole) rather than
/// arena-span comparison, so it can be asked before placement exists.
///
/// A checkpoint source is never in place: its bytes are on disk.
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

/// The bytes one extent covers: the product of its counts, times the width of
/// the contiguous inner block.
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
    // The buffer's own type, not its tensor's, so an intermediate a transform
    // chain produces (not a bound tensor) still types.
    plan.buffer(*inputs.first()?)
        .ok()
        .map(|decl| encoding_dtype(&decl.ty.encoding))
}

/// The declared rectangle behind a buffer. MXFP4 outputs are allocated flat
/// (`u8[bytes]`), so the buffer's own size says nothing about rows and
/// columns; the logical shape lives on the buffer's declared type. Read
/// through [`crate::types::rectangle`], which folds a rank-3 bank's leading
/// axes into the row count the way the kernels index it.
fn logical_shape(plan: &LoadPlan, buffer: BufferId) -> Option<(u64, u64)> {
    let (rows, cols) = crate::types::rectangle(&plan.buffer(buffer).ok()?.ty.shape)?;
    Some((u64::try_from(rows).ok()?, u64::try_from(cols).ok()?))
}

#[cfg(test)]
mod tests;
