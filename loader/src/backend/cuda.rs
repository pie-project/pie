//! CUDA lowering rules.
//!
//! Each function here replaces a decision `driver/cuda/src/loader/
//! transcode_engine.hpp` used to make while executing. The behaviour is
//! deliberately unchanged — this step moves *where* the decision happens, not
//! *what* it decides, so the existing kernel parity tests stay valid as the
//! safety net (§8.1).

use super::{Backend, TileLowering, TileMapFacts, rows_under_budget};
use crate::load_plan::{
    StorageTarget, TILE_MAP_CAST, TILE_MAP_ENCODE, TILE_MAP_REBLOCK, TILE_MAP_REORDER,
    TILE_MAP_REPACK, TileMapKind, TransformFusion,
};
use crate::types::{DType, QuantScheme};

/// The transforms `driver/cuda`'s kernels implement. Mirrored in C++ as
/// `kCudaTileMapMask`, which is defined in terms of the generated bits rather
/// than restated, so the two cannot drift.
pub const TILE_MAP_MASK: u32 =
    TILE_MAP_CAST | TILE_MAP_ENCODE | TILE_MAP_REBLOCK | TILE_MAP_REORDER | TILE_MAP_REPACK;

/// Tile budget used when the target declares none.
///
/// A target that reports `max_tile_bytes == 0` is not saying "no limit"; it is
/// saying it did not measure one. The value matches what the driver would have
/// sent (`kCudaMaxTileBytes`), so a request that omits the field compiles to the
/// same plan as one that states it.
pub const FALLBACK_TILE_BYTES: u64 = 64 * 1024 * 1024;

pub struct Cuda;

impl Backend for Cuda {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn tile_map_mask(&self) -> u32 {
        TILE_MAP_MASK
    }

    fn lower_tile_map(&self, facts: &TileMapFacts, target: &StorageTarget) -> TileLowering {
        if facts.kind != TileMapKind::Encode {
            return TileLowering::default();
        }
        TileLowering {
            rows_per_tile: encode_rows_per_tile(facts),
            fusion: encode_fusion(facts, target),
        }
    }
}

/// Whether an Encode may be split into row tiles.
///
/// A strided source would need its stride re-derived per tile, so only a
/// contiguous run qualifies. An Encode with no source reads a device buffer,
/// which is contiguous by construction.
fn can_tile_encode(facts: &TileMapFacts) -> bool {
    !facts.has_source || facts.compact_source
}

/// Rows of the output the driver transforms per launch, or `0` for "all at
/// once".
///
/// Ported from `transcode_engine.hpp::encode_rows_per_tile`, including its use
/// of the *logical* dtype width for a quantized source. That is not the true
/// on-disk row size, but reproducing the arithmetic exactly is what keeps this
/// step a pure relocation; changing the budget is a separate question.
fn encode_rows_per_tile(facts: &TileMapFacts) -> u32 {
    let Some((rows, cols)) = facts.shape else {
        return 0;
    };
    if !can_tile_encode(facts) {
        return 0;
    }
    let Some(source_dtype) = facts.source_dtype else {
        return 0;
    };
    // An FP8 Encode source carries a [rows/128, cols/128] block scale, so
    // slicing the dequant by an arbitrary row count would cut through a 128-row
    // block boundary. GLM-5.1's expert weights at [2048, 6144] fit in ~50 MB of
    // BF16 scratch, so refusing to tile them costs nothing.
    if matches!(source_dtype, DType::F8E4M3 | DType::F8E5M2) {
        return 0;
    }
    let max_tile_bytes = if facts.max_tile_bytes == 0 {
        FALLBACK_TILE_BYTES
    } else {
        facts.max_tile_bytes
    };
    let source_row_bytes = cols.saturating_mul(source_dtype.bytes());
    let bf16_row_bytes = cols.saturating_mul(DType::BF16.bytes());
    let scratch_per_row = if source_dtype == DType::BF16 {
        bf16_row_bytes
    } else {
        source_row_bytes.saturating_add(bf16_row_bytes)
    };
    let rows_per_tile = rows_under_budget(rows, scratch_per_row, max_tile_bytes);
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
/// that the opt-out is now `StorageTarget::fused_transcode` — a compile input —
/// instead of `PIE_CUDA_DISABLE_FUSED_TRANSCODE` read inside the executor. An
/// environment variable that silently selects different kernels for the same
/// plan is exactly the thing §8.1 objects to; as a target field it produces a
/// *different plan*, which is the honest representation of different execution.
fn encode_fusion(facts: &TileMapFacts, target: &StorageTarget) -> TransformFusion {
    let fusable = target.fused_transcode
        && facts.transform_to == Some(QuantScheme::Mxfp4E2M1E8M0)
        && facts.has_source
        && facts.source_dtype == Some(DType::F8E4M3);
    if fusable {
        TransformFusion::Fp8ToMxfp4
    } else {
        TransformFusion::None
    }
}
