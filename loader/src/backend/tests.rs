//! Backend lowering rules.
//!
//! These test the *decisions*, not the plumbing: each case pins one branch of
//! the CUDA rules ported from `transcode_engine.hpp` so a later change to the
//! arithmetic has to be deliberate. The end-to-end wiring (that `lower` reaches
//! every `TileMap` and writes the fields back) is covered separately by the
//! plan-level tests in `tests/storage_compiler.rs`.

use super::*;
use crate::load_plan::{
    BufferDecl, DimSpec, LoadPlan, SourceTensorDecl, StorageInstr, TileSpec, TransformSpec,
};
use crate::types::{FileId, InstrId, QuantSpec, TensorId};

const MIB: u64 = 1024 * 1024;

fn facts(source_dtype: DType, rows: u64, cols: u64, max_tile_bytes: u64) -> TileMapFacts {
    TileMapFacts {
        kind: TileMapKind::Encode,
        transform_from: None,
        transform_to: Some(QuantScheme::Mxfp4E2M1E8M0),
        source_dtype: Some(source_dtype),
        has_source: true,
        compact_source: true,
        shape: Some((rows, cols)),
        max_tile_bytes,
    }
}

fn cuda_lower(facts: &TileMapFacts, fused_transcode: bool) -> TileLowering {
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        fused_transcode,
        ..StorageTarget::default()
    };
    cuda::Cuda.lower_tile_map(facts, &target)
}

fn rows_per_tile(facts: &TileMapFacts) -> u32 {
    cuda_lower(facts, false).rows_per_tile
}

#[test]
fn fp8_sources_are_never_tiled() {
    // An FP8 source carries a [rows/128, cols/128] block scale, so an arbitrary
    // row split would cut a scale block in half. Budget is irrelevant: even a
    // 1 KiB budget must not tile.
    for dtype in [DType::F8E4M3, DType::F8E5M2] {
        assert_eq!(
            rows_per_tile(&facts(dtype, 4096, 4096, 1024)),
            0,
            "{dtype:?}"
        );
    }
}

#[test]
fn bf16_source_budgets_only_the_output_scratch() {
    // BF16 needs no dequant buffer, so one row costs `cols * 2`. 4 MiB / 8 KiB
    // is 512 rows, which is under the 4096-row tensor and so a real tile.
    let facts = facts(DType::BF16, 4096, 4096, 4 * MIB);
    assert_eq!(rows_per_tile(&facts), 512);
}

#[test]
fn non_bf16_source_budgets_both_sides_of_the_dequant() {
    // FP16 is read and then written as BF16, so a row costs `cols * 2` twice.
    // Same budget as the BF16 case, half the rows.
    let facts = facts(DType::F16, 4096, 4096, 4 * MIB);
    assert_eq!(rows_per_tile(&facts), 256);
}

#[test]
fn a_budget_that_covers_the_tensor_reports_untiled() {
    // 4096 rows * 8 KiB is 32 MiB, so a 64 MiB budget fits the whole tensor.
    // The plan says `0` rather than `4096` because the driver should not have to
    // compare a row count against the shape to learn there is one tile.
    let facts = facts(DType::BF16, 4096, 4096, 64 * MIB);
    assert_eq!(rows_per_tile(&facts), 0);
}

#[test]
fn an_unmeasured_budget_falls_back_rather_than_meaning_unlimited() {
    // `max_tile_bytes == 0` is "the driver did not measure one", so the 64 MiB
    // fallback applies. At 64 KiB per row that is 1024 rows, not all 65536.
    let facts = facts(DType::BF16, 65536, 32768, 0);
    assert_eq!(rows_per_tile(&facts), 1024);
    assert_eq!(cuda::FALLBACK_TILE_BYTES, 64 * MIB);
}

#[test]
fn a_budget_too_small_for_one_row_still_yields_one_row() {
    // Tiling below a row is not expressible, so the floor is 1 rather than 0 —
    // which would mean "untiled" and silently ask for the opposite.
    let facts = facts(DType::BF16, 4096, 4096, 16);
    assert_eq!(rows_per_tile(&facts), 1);
}

#[test]
fn a_strided_source_is_not_tiled() {
    // Slicing by rows would need the stride re-derived per tile.
    let mut facts = facts(DType::BF16, 4096, 4096, 4 * MIB);
    facts.compact_source = false;
    assert_eq!(rows_per_tile(&facts), 0);
}

#[test]
fn a_buffer_input_is_contiguous_and_so_still_tiles() {
    // No source means the input is a device buffer, which is contiguous by
    // construction; `compact_source` says nothing in that case.
    let mut facts = facts(DType::BF16, 4096, 4096, 4 * MIB);
    facts.has_source = false;
    facts.compact_source = false;
    assert_eq!(rows_per_tile(&facts), 512);
}

#[test]
fn an_unresolvable_shape_or_dtype_declines_to_tile() {
    let mut no_shape = facts(DType::BF16, 4096, 4096, 4 * MIB);
    no_shape.shape = None;
    assert_eq!(rows_per_tile(&no_shape), 0);

    let mut no_dtype = facts(DType::BF16, 4096, 4096, 4 * MIB);
    no_dtype.source_dtype = None;
    assert_eq!(rows_per_tile(&no_dtype), 0);
}

#[test]
fn only_encode_is_lowered() {
    // Cast and Reblock have no scratch to budget; the executor streams them.
    for kind in [
        TileMapKind::Cast,
        TileMapKind::Reblock,
        TileMapKind::Reorder,
    ] {
        let mut f = facts(DType::F8E4M3, 4096, 4096, 4 * MIB);
        f.kind = kind;
        assert_eq!(cuda_lower(&f, true), TileLowering::default(), "{kind:?}");
    }
}

#[test]
fn fusion_needs_every_conjunct() {
    let base = facts(DType::F8E4M3, 4096, 4096, 4 * MIB);
    assert_eq!(cuda_lower(&base, true).fusion, TransformFusion::Fp8ToMxfp4);

    // The driver's opt-out, formerly PIE_CUDA_DISABLE_FUSED_TRANSCODE.
    assert_eq!(cuda_lower(&base, false).fusion, TransformFusion::None);

    // Only MXFP4 has a fused kernel.
    let mut to_fp8 = base;
    to_fp8.transform_to = Some(QuantScheme::Fp8E4M3);
    assert_eq!(cuda_lower(&to_fp8, true).fusion, TransformFusion::None);

    // The fused kernel reads the checkpoint directly; a device buffer input has
    // already paid for the round-trip this fusion exists to skip.
    let mut buffered = base;
    buffered.has_source = false;
    assert_eq!(cuda_lower(&buffered, true).fusion, TransformFusion::None);

    // E5M2 is not the format the fused kernel decodes.
    let mut e5m2 = base;
    e5m2.source_dtype = Some(DType::F8E5M2);
    assert_eq!(cuda_lower(&e5m2, true).fusion, TransformFusion::None);
}

#[test]
fn backends_without_transform_kernels_decide_nothing() {
    // Metal binds tensors into a heap and runs no transforms, so its mask is
    // empty and every lowering is the default.
    assert_eq!(metal::TILE_MAP_MASK, 0);
    let target = StorageTarget {
        backend: BackendKind::Metal,
        fused_transcode: true,
        ..StorageTarget::default()
    };
    let facts = facts(DType::F8E4M3, 4096, 4096, 4 * MIB);
    assert_eq!(
        metal::Metal.lower_tile_map(&facts, &target),
        TileLowering::default()
    );
}

#[test]
fn the_reference_backend_declines_every_optimization() {
    // `host_executor` is a correctness oracle: it derives its own tiling from
    // the budget at run time and has no fused kernels, so a difference between
    // `cuda` and `host` output is always a backend rule and never an accident.
    let target = StorageTarget {
        fused_transcode: true,
        ..StorageTarget::default()
    };
    let facts = facts(DType::F8E4M3, 4096, 4096, 4 * MIB);
    assert_eq!(
        host::Host.lower_tile_map(&facts, &target),
        TileLowering::default()
    );
}

#[test]
fn each_backend_resolves_to_its_own_rules() {
    assert_eq!(for_backend(BackendKind::Cuda).name(), "cuda");
    assert_eq!(for_backend(BackendKind::Metal).name(), "metal");
    assert_eq!(for_backend(BackendKind::Unknown).name(), "host");
    assert_eq!(
        for_backend(BackendKind::Cuda).tile_map_mask(),
        cuda::TILE_MAP_MASK
    );
}

// ---------------------------------------------------------------------------
// Fact extraction: what the pass reads out of the plan before a backend sees it.
// ---------------------------------------------------------------------------

fn compact_extent(rows: i64, cols: i64, element_bytes: u32) -> StridedExtent {
    let eb = i64::from(element_bytes);
    StridedExtent {
        base_offset: 0,
        element_bytes,
        dims: vec![
            DimSpec {
                count: rows,
                src_stride: cols * eb,
                dst_stride: cols * eb,
            },
            DimSpec {
                count: cols,
                src_stride: eb,
                dst_stride: eb,
            },
        ],
    }
}

/// A plan with one Encode reading tensor 0 into buffer 0, declared `[rows, cols]`
/// but allocated flat as MXFP4 output is.
fn encode_plan(encoding: Encoding, rows: i64, cols: i64) -> LoadPlan {
    let mut plan = LoadPlan::empty(StorageTarget {
        backend: BackendKind::Cuda,
        max_tile_bytes: 4 * MIB,
        fused_transcode: true,
        ..StorageTarget::default()
    });
    plan.sources.push(SourceTensorDecl {
        id: TensorId(0),
        name: "w".to_string(),
        file_id: FileId(0),
        file_offset: 0,
        span_bytes: 0,
        shape: vec![rows, cols],
        encoding,
    });
    plan.tensors.push(TensorDecl {
        id: TensorId(0),
        name: "w".to_string(),
        shape: vec![rows, cols],
        encoding: Encoding::Raw(DType::BF16),
        alignment: 1,
    });
    plan.buffers.push(BufferDecl {
        id: BufferId(0),
        tensor: Some(TensorId(0)),
        bytes: 0,
        alignment: 1,
        temporary: false,
        persistent_offset: None,
    });
    plan.instrs.push(StorageInstr::TileMap {
        id: InstrId(0),
        kind: TileMapKind::Encode,
        source: Some(SourceExtent {
            file_id: FileId(0),
            tensor_id: TensorId(0),
            file_offset: 0,
            span_bytes: 0,
            stride: compact_extent(rows, cols, 2),
        }),
        dest: None,
        inputs: Vec::new(),
        outputs: vec![BufferId(0)],
        tile: TileSpec {
            max_tile_bytes: 4 * MIB,
            rows_per_tile: 0,
        },
        transform: TransformSpec {
            to: Some(QuantScheme::Mxfp4E2M1E8M0),
            ..TransformSpec::default()
        },
    });
    plan
}

fn only_tile_map(plan: &LoadPlan) -> (&TileSpec, &TransformSpec) {
    match &plan.instrs[0] {
        StorageInstr::TileMap {
            tile, transform, ..
        } => (tile, transform),
        other => panic!("expected TileMap, got {other:?}"),
    }
}

#[test]
fn lowering_writes_decisions_back_into_the_instruction() {
    let mut plan = encode_plan(Encoding::Raw(DType::F8E4M3), 4096, 4096);
    lower(&mut plan);
    let (tile, transform) = only_tile_map(&plan);
    assert_eq!(tile.rows_per_tile, 0);
    assert_eq!(transform.fusion, TransformFusion::Fp8ToMxfp4);
    // The budget is an input and must survive the pass unchanged.
    assert_eq!(tile.max_tile_bytes, 4 * MIB);
}

#[test]
fn a_quantized_source_resolves_to_its_logical_dtype() {
    // The driver reads `PieLoaderSourceTensorView::dtype`, which is the logical
    // dtype for a quantized encoding. If this pass used the storage dtype
    // instead, the two would disagree about what the transform reads.
    let mut plan = encode_plan(
        Encoding::Quant(QuantSpec {
            scheme: QuantScheme::Fp8E4M3,
            logical_dtype: DType::F8E4M3,
            bits_per_element: 8,
            group_size: 0,
            channel_axis: None,
            scale_dtype: None,
            zero_point_dtype: None,
            block_shape: Vec::new(),
        }),
        4096,
        4096,
    );
    lower(&mut plan);
    let (_, transform) = only_tile_map(&plan);
    assert_eq!(transform.fusion, TransformFusion::Fp8ToMxfp4);
}

#[test]
fn a_non_2d_output_is_left_untiled() {
    let mut plan = encode_plan(Encoding::Raw(DType::BF16), 4096, 4096);
    plan.tensors[0].shape = vec![4096];
    lower(&mut plan);
    assert_eq!(only_tile_map(&plan).0.rows_per_tile, 0);
}

#[test]
fn lowering_is_idempotent() {
    // The pass only writes decision fields, so running it twice must not drift.
    let mut once = encode_plan(Encoding::Raw(DType::BF16), 4096, 4096);
    lower(&mut once);
    let mut twice = once.clone();
    lower(&mut twice);
    assert_eq!(once, twice);
    assert_eq!(only_tile_map(&once).0.rows_per_tile, 512);
}

#[test]
fn a_strided_plan_source_declines_to_tile() {
    let mut plan = encode_plan(Encoding::Raw(DType::BF16), 4096, 4096);
    if let StorageInstr::TileMap { source, .. } = &mut plan.instrs[0] {
        source.as_mut().unwrap().stride.dims[1].src_stride = 4;
    }
    lower(&mut plan);
    assert_eq!(only_tile_map(&plan).0.rows_per_tile, 0);
}
