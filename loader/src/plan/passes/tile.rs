//! Backend tile-map lowering.
//!
//! `architecture.md` §9 draws the line this file sits on. A fact the driver can
//! *measure* is data and travels in [`StorageTarget`]; a rule that differs per
//! backend and cannot be parameterized is code and lives here. It is the same
//! split LLVM draws between target features and `TargetLowering`.
//!
//! Everything here answers a question the C++ driver used to answer at run
//! time. That mattered less for being wrong than for being *unrecorded*:
//! `driver/cuda/src/loader/transcode_engine.hpp` chose a tile size and a fusion
//! strategy while executing, so one plan could run two different kernel
//! sequences and nothing in the plan said which (`architecture.md` §8.1).
//! Deciding here puts the answer in the plan, which puts it in the plan hash,
//! which makes "the plan determines execution" true rather than aspirational.
//!
//! It used to be a `trait Backend` with three implementations, of which two
//! returned a constant and the third was one function. A trait is the right
//! shape for a decision with several *independent* implementations; this is one
//! decision with one implementation and two abstentions, so it is a `match`.

use crate::plan::index::PlanIndex;
use crate::plan::{
    FUSION_FP8_TO_MXFP4, LoadPlan, SourceExtent, StorageInstr, StorageTarget, TILE_MAP_CAST,
    TILE_MAP_DECODE, TILE_MAP_ENCODE, TILE_MAP_REBLOCK, TILE_MAP_REPACK, TILE_MAP_SCALE,
    TileMapKind, TransformFusion,
};
use crate::types::{BackendKind, BufferId, DType, Encoding, QuantScheme, TensorDecl};

/// The transforms `driver/cuda`'s kernels implement. Mirrored in C++ as
/// `kCudaTileMapMask`, which is defined in terms of the generated bits rather
/// than restated, so the two cannot drift.
pub const CUDA_TILE_MAP_MASK: u32 =
    TILE_MAP_CAST | TILE_MAP_ENCODE | TILE_MAP_REBLOCK | TILE_MAP_REPACK | TILE_MAP_SCALE;

/// The transforms `driver/metal`'s load-time kernels implement. Mirrored in C++
/// as `kMetalTileMapMask`, which is defined in terms of the generated bits
/// rather than restated, so the two cannot drift.
///
/// `SCALE` decodes a block-scaled scheme to values and `CAST` re-encodes them as
/// the affine-U4 that driver's matvecs read, which is what lets it load the
/// published MXFP4 gpt-oss checkpoint directly. It has no repacking or
/// reblocking kernels, so those bits stay clear.
pub const METAL_TILE_MAP_MASK: u32 = TILE_MAP_CAST | TILE_MAP_ENCODE | TILE_MAP_SCALE;

/// The transforms `host_executor` implements. Not a device capability, so it is
/// not part of the C surface.
pub const HOST_TILE_MAP_MASK: u32 = TILE_MAP_CAST | TILE_MAP_REBLOCK | TILE_MAP_SCALE;

/// The mask offline conversion compiles against: everything `replay` runs plus
/// `Encode` and `Decode`, which the host executor implements for exactly that
/// command.
///
/// A separate constant rather than a wider [`HOST_TILE_MAP_MASK`] because the
/// two answer different questions. HOST is the verification surface — what a
/// `replay` of a *device* plan may be asked to reproduce — and widening it
/// would quietly change which plans `replay` accepts. Conversion is its own
/// target: a plan compiled against this mask is meant to be executed on the
/// host and its output written back to a checkpoint, not compared against a
/// device.
///
/// `Decode` is here and in no device mask: the schemes it covers carry their
/// scales inside the payload (GGUF blocks), which no device kernel reads —
/// decoding them offline into a checkpoint the device *can* read is the whole
/// point of conversion.
pub const CONVERT_TILE_MAP_MASK: u32 = HOST_TILE_MAP_MASK | TILE_MAP_ENCODE | TILE_MAP_DECODE;

/// The loader's own model of what a backend's kernels do.
///
/// The driver states the same thing independently in
/// `PieLoaderTargetSpec::tile_map_mask`, and `storage_target` cross-checks the
/// two, which makes this load-bearing rather than descriptive: a driver
/// claiming a transform the loader has no model of is rejected at compile time
/// instead of surfacing as a failed kernel dispatch.
///
/// The comparison is deliberately one-sided. `architecture.md` §9 makes the
/// driver the authority on what its kernels do, so a *narrower* driver mask is
/// fine — it implements fewer transforms than the loader can lower. A *wider*
/// one is a claim about kernels the loader cannot reason about, and is refused.
pub fn tile_map_mask(backend: BackendKind) -> u32 {
    match backend {
        BackendKind::Cuda => CUDA_TILE_MAP_MASK,
        BackendKind::Metal => METAL_TILE_MAP_MASK,
        BackendKind::Unknown => HOST_TILE_MAP_MASK,
    }
}

/// Everything the lowering rule is allowed to see about one `TileMap`.
///
/// Deliberately not the instruction itself. The plan walking — resolving a
/// source tensor's dtype, recovering a flat buffer's logical shape — happens
/// once, below, so the rule contains only the rule. A rule that needed a fact
/// not listed here is a signal that the fact belongs in [`StorageTarget`], not
/// that this struct should grow a plan reference.
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
    /// Whether the source extent is one contiguous run. A strided source cannot
    /// be sliced by rows without re-deriving the stride per tile, which is why
    /// the driver refused to tile those.
    pub compact_source: bool,
    /// Declared 2-D shape of the primary output, or `None` if it is not 2-D.
    ///
    /// Read from the *instruction*, not the target: the build may narrow the
    /// budget for one instruction, and a rule that consulted the target would
    /// silently ignore that.
    pub shape: Option<(u64, u64)>,
    pub max_tile_bytes: u64,
}

/// What the lowering decided. Written into the instruction verbatim.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileLowering {
    /// Rows of the output the driver transforms per launch. `0` means "no
    /// tiling" — do the whole tensor in one pass — which is both the answer for
    /// instructions that cannot be tiled and the answer when one tile covers
    /// everything.
    pub rows_per_tile: u32,
    pub fusion: TransformFusion,
}

/// Fill in every backend decision the plan carries.
///
/// Runs over a finished plan, so a decision may depend on anything the plan
/// says. It only ever *writes* decision fields — never adds, removes, or
/// reorders instructions — which is what lets a backend that declines to decide
/// leave the plan bit-identical. That is also why it is not a [`Pass`]: it
/// rewrites nothing there would be anything to count.
///
/// [`Pass`]: crate::plan::pass::Pass
pub fn lower(plan: &mut LoadPlan) {
    let target = plan.target.clone();
    let index = PlanIndex::new(plan);

    // Resolved up front: the write loop holds `&mut` on an instruction while
    // these lookups still need to read buffer and tensor declarations.
    let facts: Vec<Option<TileMapFacts>> = plan
        .instrs
        .iter()
        .map(|instr| tile_map_facts(plan, &index, instr))
        .collect();

    for (instr, facts) in plan.instrs.iter_mut().zip(facts) {
        let (
            Some(facts),
            StorageInstr::TileMap {
                tile, transform, ..
            },
        ) = (facts, instr)
        else {
            continue;
        };
        let lowering = lower_tile_map(&facts, &target);
        tile.rows_per_tile = lowering.rows_per_tile;
        transform.fusion = lowering.fusion;
    }
}

fn lower_tile_map(facts: &TileMapFacts, target: &StorageTarget) -> TileLowering {
    match target.backend {
        BackendKind::Cuda => cuda_encode(facts, target),
        // Metal runs no transforms, and the host executor derives its own
        // tiling from `max_tile_bytes` at run time — which it is allowed to do
        // precisely because it is not the thing whose execution the plan is
        // supposed to determine (`architecture.md` §10.3).
        BackendKind::Metal | BackendKind::Unknown => TileLowering::default(),
    }
}

/// CUDA's one rule, ported from `transcode_engine.hpp`.
///
/// The behaviour is deliberately unchanged from the C++ it replaces — this
/// moves *where* the decision happens, not *what* it decides, so the existing
/// kernel parity tests stay valid as the safety net (`architecture.md` §8.1).
fn cuda_encode(facts: &TileMapFacts, target: &StorageTarget) -> TileLowering {
    if facts.kind != TileMapKind::Encode {
        return TileLowering::default();
    }
    TileLowering {
        rows_per_tile: encode_rows_per_tile(facts, target),
        fusion: encode_fusion(facts, target),
    }
}

/// Rows of the output the driver transforms per launch, or `0` for "all at
/// once".
///
/// Ported including its use of the *logical* dtype width for a quantized
/// source. That is not the true on-disk row size, but reproducing the
/// arithmetic exactly is what keeps this a pure relocation; changing the budget
/// is a separate question.
///
/// Every number this reads is now stated by the target. The rule that is left —
/// scratch is the source plus the dequant buffer, unless they are the same
/// dtype — is the part that cannot be a field, because it is *how the kernel is
/// written*, not what the device is.
fn encode_rows_per_tile(facts: &TileMapFacts, target: &StorageTarget) -> u32 {
    let Some((rows, cols)) = facts.shape else {
        return 0;
    };
    // A strided source would need its stride re-derived per tile, so only a
    // contiguous run qualifies. An Encode with no source reads a device buffer,
    // which is contiguous by construction.
    if facts.has_source && !facts.compact_source {
        return 0;
    }
    let Some(source_dtype) = facts.source_dtype else {
        return 0;
    };
    // A block-scaled source carries one scale per `[block_scale_rows, N]` tile,
    // so slicing the dequant by an arbitrary row count would cut a scale block
    // in half. GLM-5.1's expert weights at [2048, 6144] fit in ~50 MB of
    // scratch, so refusing to tile them costs nothing.
    if target.block_scale_rows != 0 && source_dtype.is_block_scaled() {
        return 0;
    }
    let scratch_dtype = target.encode_scratch_dtype;
    let source_row_bytes = cols.saturating_mul(source_dtype.bytes());
    let scratch_row_bytes = cols.saturating_mul(scratch_dtype.bytes());
    let scratch_per_row = if source_dtype == scratch_dtype {
        scratch_row_bytes
    } else {
        source_row_bytes.saturating_add(scratch_row_bytes)
    };
    let rows_per_tile = rows_under_budget(rows, scratch_per_row, facts.max_tile_bytes);
    // One tile covering everything is the untiled case; say so, rather than
    // making the driver compare a row count against the shape to find out.
    if u64::from(rows_per_tile) >= rows {
        0
    } else {
        rows_per_tile
    }
}

/// Whether to transcode FP8 straight to MXFP4, skipping the BF16 HBM
/// round-trip.
///
/// Bit-identical to the two-step path and kernel parity-tested. What changed is
/// that the opt-out is now a bit in `StorageTarget::fusion_mask` — a compile
/// input — instead of `PIE_CUDA_DISABLE_FUSED_TRANSCODE` read inside the
/// executor. An environment variable that silently selects different kernels
/// for the same plan is exactly the thing `architecture.md` §8.1 objects to; as
/// a target field it produces a *different plan*, which is the honest
/// representation of different execution.
///
/// The *chain* stays here rather than becoming data, and that is the
/// `architecture.md` §9 line: which two steps `Fp8ToMxfp4` collapses, and the
/// proof that collapsing them is bit-identical, is the loader's model of the
/// transform. Whether the kernel exists is the driver's, and that is the bit.
fn encode_fusion(facts: &TileMapFacts, target: &StorageTarget) -> TransformFusion {
    let fusable = target.fusion_mask & FUSION_FP8_TO_MXFP4 != 0
        && facts.transform_to == Some(QuantScheme::Mxfp4E2M1E8M0)
        && facts.has_source
        && facts.source_dtype == Some(DType::F8E4M3);
    if fusable {
        TransformFusion::Fp8ToMxfp4
    } else {
        TransformFusion::None
    }
}

/// Tile a transform by rows under a byte budget.
///
/// Separate because it is arithmetic rather than policy: given the cost of one
/// row and the budget for one tile, this is the row count. What differs per
/// backend is `scratch_per_row` and whether tiling is legal at all.
fn rows_under_budget(rows: u64, scratch_per_row: u64, max_tile_bytes: u64) -> u32 {
    if max_tile_bytes == 0 || scratch_per_row == 0 {
        return clamp_rows(rows);
    }
    let per_tile = (max_tile_bytes / scratch_per_row).max(1);
    clamp_rows(rows.min(per_tile))
}

fn clamp_rows(rows: u64) -> u32 {
    u32::try_from(rows).unwrap_or(u32::MAX)
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
/// one. This is the value the driver reads off `PieLoaderSourceTensorView`, so
/// the two must agree.
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
            .and_then(|buffer| logical_shape(plan, index, *buffer)),
        max_tile_bytes: tile.max_tile_bytes,
    })
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
    index
        .buffer_tensor(plan, *inputs.first()?)
        .map(TensorDecl::dtype)
}

/// The declared 2-D shape behind a buffer.
///
/// MXFP4 outputs are allocated flat (`u8[bytes]`), so the buffer's own size
/// says nothing about rows and columns; the logical shape lives on the tensor
/// declaration. Same recovery `encode_tile_map` did in C++.
fn logical_shape(plan: &LoadPlan, index: &PlanIndex, buffer: BufferId) -> Option<(u64, u64)> {
    match index.buffer_tensor(plan, buffer)?.shape.as_slice() {
        [rows, cols] => Some((u64::try_from(*rows).ok()?, u64::try_from(*cols).ok()?)),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
