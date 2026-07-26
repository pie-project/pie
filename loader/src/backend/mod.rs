//! Backend-specific compilation decisions.
//!
//! §9 draws the line these modules sit on. A fact the driver can *measure* is
//! data and travels in [`StorageTarget`]; a rule that differs per backend and
//! cannot be parameterized is code and lives here. It is the same split LLVM
//! draws between target features and `TargetLowering`.
//!
//! Everything here answers a question the C++ driver used to answer at run time.
//! That mattered less for being wrong than for being *unrecorded*:
//! `driver/cuda/src/loader/transcode_engine.hpp` chose a tile size and a fusion
//! strategy while executing, so one plan could run two different kernel
//! sequences and nothing in the plan said which (§8.1). Deciding here puts the
//! answer in the plan, which puts it in the plan hash, which makes "the plan
//! determines execution" true rather than aspirational.

pub mod cuda;
pub mod host;
pub mod metal;

use crate::load_plan::{
    LoadPlan, SourceExtent, StorageInstr, StorageTarget, StridedExtent, TileMapKind,
    TransformFusion,
};
use crate::types::{BackendKind, BufferId, DType, Encoding, QuantScheme, TensorDecl};

/// Everything a backend is allowed to see about one `TileMap` instruction.
///
/// Deliberately not the instruction itself. The lowering pass does the plan
/// walking — resolving a source tensor's dtype, recovering a flat buffer's
/// logical shape — so a backend module contains only the rule. A backend that
/// needs a fact not listed here is a signal that the fact belongs in
/// [`StorageTarget`], not that this struct should grow a plan reference.
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
    pub shape: Option<(u64, u64)>,
    pub max_tile_bytes: u64,
}

/// What the backend decided. Written into the instruction verbatim.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileLowering {
    /// Rows of the output the driver transforms per launch. `0` means "no
    /// tiling" — do the whole tensor in one pass — which is both the answer for
    /// instructions that cannot be tiled and the answer when one tile covers
    /// everything.
    pub rows_per_tile: u32,
    pub fusion: TransformFusion,
}

/// The rules that cannot be expressed as [`StorageTarget`] data.
pub trait Backend {
    fn name(&self) -> &'static str;

    /// The tile-map transforms this backend's kernels implement.
    ///
    /// The loader's own model of the backend, which the driver states
    /// independently in `PieLoaderTargetSpec::tile_map_mask` (C++
    /// `kCudaTileMapMask` and friends). `storage_target` cross-checks the two
    /// (step 8), which makes this method load-bearing rather than descriptive:
    /// a driver claiming a transform that is not in this mask is rejected at
    /// compile time instead of surfacing as a failed kernel dispatch.
    ///
    /// The comparison is deliberately one-sided. §9 makes the driver the
    /// authority on what its kernels do, so a *narrower* driver mask is fine —
    /// it implements fewer transforms than the loader can lower. A *wider* one
    /// is a claim about kernels the loader has no model of, and is refused.
    fn tile_map_mask(&self) -> u32;

    fn lower_tile_map(&self, facts: &TileMapFacts, target: &StorageTarget) -> TileLowering;
}

pub fn for_backend(kind: BackendKind) -> &'static dyn Backend {
    match kind {
        BackendKind::Cuda => &cuda::Cuda,
        BackendKind::Metal => &metal::Metal,
        BackendKind::Unknown => &host::Host,
    }
}

/// Fill in every backend decision the plan carries.
///
/// Runs over a finished plan, so a decision may depend on anything the plan
/// says. It only ever *writes* decision fields — never adds, removes, or
/// reorders instructions — which is what lets a backend that declines to decide
/// leave the plan bit-identical.
pub fn lower(plan: &mut LoadPlan) {
    let backend = for_backend(plan.target.backend);
    let target = plan.target.clone();

    // Resolved up front: the write loop holds `&mut` on an instruction while
    // these lookups still need to read buffer and tensor declarations.
    let facts: Vec<Option<TileMapFacts>> = plan
        .instrs
        .iter()
        .map(|instr| tile_map_facts(plan, instr))
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
        let lowering = backend.lower_tile_map(&facts, &target);
        tile.rows_per_tile = lowering.rows_per_tile;
        transform.fusion = lowering.fusion;
    }
}

/// Tile a transform by rows under a byte budget.
///
/// Shared because it is arithmetic rather than policy: given the cost of one row
/// and the budget for one tile, this is the row count. What differs per backend
/// is `scratch_per_row` and whether tiling is legal at all.
pub(crate) fn rows_under_budget(rows: u64, scratch_per_row: u64, max_tile_bytes: u64) -> u32 {
    if max_tile_bytes == 0 || scratch_per_row == 0 {
        return clamp_rows(rows);
    }
    let per_tile = (max_tile_bytes / scratch_per_row).max(1);
    clamp_rows(rows.min(per_tile))
}

fn clamp_rows(rows: u64) -> u32 {
    u32::try_from(rows).unwrap_or(u32::MAX)
}

fn extent_is_compact(extent: &StridedExtent) -> bool {
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

fn tile_map_facts(plan: &LoadPlan, instr: &StorageInstr) -> Option<TileMapFacts> {
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
        source_dtype: source_dtype(plan, source.as_ref(), inputs),
        has_source: source.is_some(),
        compact_source: source
            .as_ref()
            .is_none_or(|source| extent_is_compact(&source.stride)),
        shape: outputs
            .first()
            .and_then(|buffer| logical_shape(plan, *buffer)),
        max_tile_bytes: tile.max_tile_bytes,
    })
}

fn source_dtype(
    plan: &LoadPlan,
    source: Option<&SourceExtent>,
    inputs: &[BufferId],
) -> Option<DType> {
    if let Some(source) = source {
        return plan
            .sources
            .iter()
            .find(|decl| decl.id == source.tensor_id)
            .map(|decl| encoding_dtype(&decl.encoding));
    }
    buffer_tensor(plan, *inputs.first()?).map(TensorDecl::dtype)
}

/// The declared 2-D shape behind a buffer.
///
/// MXFP4 outputs are allocated flat (`u8[bytes]`), so the buffer's own size says
/// nothing about rows and columns; the logical shape lives on the tensor
/// declaration. Same recovery `encode_tile_map` did in C++.
fn logical_shape(plan: &LoadPlan, buffer: BufferId) -> Option<(u64, u64)> {
    match buffer_tensor(plan, buffer)?.shape.as_slice() {
        [rows, cols] => Some((u64::try_from(*rows).ok()?, u64::try_from(*cols).ok()?)),
        _ => None,
    }
}

fn buffer_tensor(plan: &LoadPlan, buffer: BufferId) -> Option<&TensorDecl> {
    let tensor = plan
        .buffers
        .iter()
        .find(|decl| decl.id == buffer)
        .and_then(|decl| decl.tensor)?;
    plan.tensors.iter().find(|decl| decl.id == tensor)
}

#[cfg(test)]
mod tests;
