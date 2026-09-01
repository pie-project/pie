//! Backend lowering rules.
//!
//! These test the *decisions*, not the plumbing: each case pins one branch of
//! the CUDA table so a later change to it has to be deliberate. The end-to-end
//! wiring (that `lower` reaches every `TileMap` and writes the field back) is
//! covered by the plan-level tests in `tests/storage_compiler.rs`.
//!
//! **THIS FILE WAS THE ENCODE LOWERING'S SUITE** and §M-3 shut that door.
//! Fifteen cases here budgeted an encode's scratch rows, fused an FP8 source
//! into MXFP4, or read a fact only `encode_rows_per_tile` consumed; every one
//! of them tested a decision no plan can now reach, because no device mask
//! carries an encode and the transform runs on the host at `pie model
//! import`. What is left is the table that survives — one cast, one row
//! scale — and the statement that each backend resolves to its own mask.

use super::*;
use crate::plan::StorageTarget;

const MIB: u64 = 1024 * 1024;

/// One instruction's facts, with the two shape-and-place answers a device row
/// needs already true. Each test below sets only the fields its own row reads.
fn facts(kind: TileMapKind) -> TileMapFacts {
    TileMapFacts {
        kind,
        transform_from: None,
        transform_to: None,
        source_dtype: None,
        has_source: true,
        compact_source: true,
        shape: Some((4096, 4096)),
        max_tile_bytes: 4 * MIB,
        dest_dtype: None,
        in_place: false,
        blocked_scale: false,
        // These operands are on the device; every case here is about which
        // row a target picks, not about where the bytes are.
        operands_in_arena: true,
    }
}

fn cuda(facts: &TileMapFacts) -> Option<&'static str> {
    lower_tile_map(
        facts,
        &StorageTarget {
            backend: BackendKind::Cuda,
            ..StorageTarget::default()
        },
    )
    .kernel
}

#[test]
fn the_cuda_table_is_one_cast_and_one_row_scale() {
    let mut cast = facts(TileMapKind::Cast);
    cast.source_dtype = Some(DType::F32);
    cast.dest_dtype = Some(DType::Bf16);
    assert_eq!(cuda(&cast), Some(CUDA_CAST_FP32_TO_BF16));

    let mut scale = facts(TileMapKind::Scale);
    scale.source_dtype = Some(DType::Bf16);
    scale.dest_dtype = Some(DType::Bf16);
    scale.in_place = true;
    scale.blocked_scale = true;
    assert_eq!(cuda(&scale), Some(CUDA_SCALE_ROWS_BF16));
}

#[test]
fn a_cast_with_no_row_is_refused_and_never_approximated() {
    // A cast with no kernel must not become a copy: the bytes would be the
    // source's representation under the destination's name, and no later
    // stage can detect that.
    let mut f16 = facts(TileMapKind::Cast);
    f16.source_dtype = Some(DType::F16);
    f16.dest_dtype = Some(DType::Bf16);
    assert_eq!(cuda(&f16), None);
}

#[test]
fn no_device_row_quantizes() {
    // The two encode rows this table held are gone with the door. Asked for
    // one now, CUDA answers the host — which is the only place an encode
    // runs, and only under `CONVERT_TILE_MAP_MASK`.
    let mut encode = facts(TileMapKind::Encode);
    encode.source_dtype = Some(DType::Bf16);
    encode.transform_to = Some(QuantScheme::Mxfp4E2M1E8M0);
    assert_eq!(cuda(&encode), None);

    let mut fp8 = facts(TileMapKind::Encode);
    fp8.source_dtype = Some(DType::Bf16);
    fp8.transform_to = Some(QuantScheme::Fp8E4M3);
    assert_eq!(cuda(&fp8), None);
}

#[test]
fn operands_the_device_cannot_reach_name_no_row() {
    // Stated once in `lower_tile_map` rather than in each backend's table,
    // because it is true of every device and of no host.
    let mut cast = facts(TileMapKind::Cast);
    cast.source_dtype = Some(DType::F32);
    cast.dest_dtype = Some(DType::Bf16);
    cast.operands_in_arena = false;
    assert_eq!(cuda(&cast), None);
}

#[test]
fn backends_without_transform_kernels_decide_nothing() {
    // Metal runs its transforms on the host, over the shared heap it is about
    // to bind, so it advertises the kinds it executes and asks for none of the
    // lowering the table can offer: there is no kernel to name.
    //
    // `TILE_MAP_DECODE` is in the mask and makes no difference to that, which
    // is the point being restated: a device mask says which transforms a plan
    // may CARRY. Metal runs every one of them on the host either way.
    //
    // `TILE_MAP_BIAS` joined for exactly that reason and no other. It is the
    // same host fold `TILE_MAP_SCALE` beside it already is — one constant
    // per element instead of one factor — and leaving it out did not stop a
    // kernel Metal lacks, because Metal lacks all of them. What it stopped
    // was `Expr::Bias` reaching a Metal plan at all, which is what a
    // `mlx_lm` checkpoint of qwen3.5 needs to undo the `+1` that converter
    // folds into its norm weights.
    //
    // `TILE_MAP_ENCODE` was in this list and is gone with the door (§M-3):
    // no device mask carries an encode any more, and the property is pinned
    // for every backend at once in `tile::mask_tests`.
    assert_eq!(
        METAL_TILE_MAP_MASK,
        TILE_MAP_CAST | TILE_MAP_SCALE | TILE_MAP_DECODE | TILE_MAP_BIAS
    );
    let target = StorageTarget {
        backend: BackendKind::Metal,
        ..StorageTarget::default()
    };
    let mut cast = facts(TileMapKind::Cast);
    cast.source_dtype = Some(DType::F32);
    cast.dest_dtype = Some(DType::Bf16);
    assert_eq!(lower_tile_map(&cast, &target), TileLowering::default());
}

#[test]
fn the_reference_backend_declines_every_optimization() {
    // `host_executor` is a correctness oracle: it derives its own tiling from
    // the budget at run time and names no kernel, so a difference between
    // `cuda` and `host` output is always a backend rule and never an accident.
    let mut cast = facts(TileMapKind::Cast);
    cast.source_dtype = Some(DType::F32);
    cast.dest_dtype = Some(DType::Bf16);
    assert_eq!(
        lower_tile_map(&cast, &StorageTarget::default()),
        TileLowering::default()
    );
}

#[test]
fn each_backend_resolves_to_its_own_rules() {
    assert_eq!(compilable_tile_maps(BackendKind::Cuda), CUDA_TILE_MAP_MASK);
    assert_eq!(
        compilable_tile_maps(BackendKind::Metal),
        METAL_TILE_MAP_MASK
    );
    assert_eq!(
        compilable_tile_maps(BackendKind::Unknown),
        HOST_TILE_MAP_MASK
    );
}
