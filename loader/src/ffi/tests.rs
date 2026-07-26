//! Round-trip tests for the POD boundary.
//!
//! Each test builds a `LoadPlan` by hand, converts it, and reads the result back
//! through raw pointers. The assertions target the *synthesized* fields — the
//! ones the C++ parser produced that have no direct counterpart in the Rust IR —
//! because those are the only places the conversion can silently disagree with
//! the executor that reads it.

use super::arena::{self, view};
use super::entry::{
    PieLoaderComponent, PieLoaderDiagnostics, PieLoaderModelSpec, PieLoaderMxfp4MoeRequest,
    PieLoaderRequest, PieLoaderStatus, PieLoaderTargetSpec,
};
use super::types::*;
use crate::load_plan::*;
use crate::optimizer::{OptimizerPassStats, OptimizerReport};
use crate::types::*;

fn target() -> StorageTarget {
    StorageTarget {
        backend: BackendKind::Cuda,
        tp_rank: 1,
        tp_size: 4,
        max_tile_bytes: 1 << 20,
        preferred_alignment: 256,
        tile_map_mask: CUDA_TILE_MAP_MASK,
        mxfp4_moe: Mxfp4MoePolicy::NativeGemm,
        native_mxfp4_moe: true,
        fused_transcode: true,
    }
}

fn stride(element_bytes: u32, dims: &[(i64, i64, i64)]) -> StridedExtent {
    StridedExtent {
        base_offset: 0,
        element_bytes,
        dims: dims
            .iter()
            .map(|(count, src_stride, dst_stride)| DimSpec {
                count: *count,
                src_stride: *src_stride,
                dst_stride: *dst_stride,
            })
            .collect(),
    }
}

fn source_extent(span_bytes: u64) -> SourceExtent {
    SourceExtent {
        file_id: FileId(0),
        tensor_id: TensorId(7),
        file_offset: 512,
        span_bytes,
        stride: stride(2, &[(4, 8, 16)]),
    }
}

fn dest_extent(buffer: u32) -> DestExtent {
    DestExtent {
        buffer: BufferId(buffer),
        offset: 64,
        stride: stride(2, &[(4, 16, 8)]),
    }
}

/// A real file on disk for the fixture's file table to point at.
///
/// `verify` stats every declared file, so a fixture that named a path which does
/// not exist would report a staleness error on every test that verifies. Backing
/// the table with a real file keeps that check live rather than stubbed out.
fn fixture_file() -> &'static (String, u64) {
    static FILE: std::sync::OnceLock<(String, u64)> = std::sync::OnceLock::new();
    FILE.get_or_init(|| {
        let path = std::env::temp_dir().join(format!(
            "pie_loader_ffi_fixture_{}.safetensors",
            std::process::id()
        ));
        // Large enough to contain every source span the fixture plan declares:
        // verification checks that each read lands inside the file it names, so
        // a fixture smaller than its own plan is a fixture bug, not a finding.
        let bytes = vec![0u8; 64 << 10];
        std::fs::write(&path, &bytes).expect("write fixture checkpoint");
        (path.to_string_lossy().into_owned(), bytes.len() as u64)
    })
}

/// Build a plan carrying every instruction variant exactly once.
fn plan_with_every_instr() -> LoadPlan {
    let mut plan = LoadPlan::empty(target());
    let (path, size_bytes) = fixture_file();
    plan.files.push(CheckpointFileDecl {
        id: FileId(0),
        path: path.clone(),
        size_bytes: *size_bytes,
        format: CheckpointFormat::Safetensors,
    });
    plan.tensors.push(TensorDecl {
        id: TensorId(7),
        name: "model.layers.0.mlp.gate_proj.weight".to_string(),
        shape: vec![4096, 11008],
        encoding: Encoding::Quant(QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(0)),
            scale_dtype: Some(DType::U8),
            zero_point_dtype: None,
            block_shape: vec![32],
        }),
        layout: Layout::dense(256),
        alignment: 256,
    });
    plan.tensors.push(TensorDecl {
        id: TensorId(8),
        name: "model.norm.weight".to_string(),
        shape: vec![4096],
        encoding: Encoding::Raw(DType::BF16),
        layout: Layout::dense(256),
        alignment: 256,
    });
    plan.sources.push(SourceTensorDecl {
        id: TensorId(7),
        name: "raw.gate_proj".to_string(),
        file_id: FileId(0),
        file_offset: 512,
        span_bytes: 4096,
        shape: vec![4096, 11008],
        encoding: Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::F16,
            bits_per_element: 4,
            group_size: 64,
            channel_axis: None,
            scale_dtype: None,
            zero_point_dtype: None,
            block_shape: vec![],
        }),
    });
    plan.sources.push(SourceTensorDecl {
        id: TensorId(8),
        name: "raw.norm".to_string(),
        file_id: FileId(0),
        file_offset: 8192,
        span_bytes: 8192,
        shape: vec![4096],
        encoding: Encoding::Raw(DType::F32),
    });
    plan.buffers.push(BufferDecl {
        id: BufferId(0),
        tensor: Some(TensorId(7)),
        bytes: 4096,
        alignment: 256,
        temporary: false,
        persistent_offset: Some(0),
    });
    plan.buffers.push(BufferDecl {
        id: BufferId(1),
        tensor: None,
        bytes: 1024,
        alignment: 16,
        temporary: true,
        persistent_offset: None,
    });

    plan.instrs = vec![
        StorageInstr::Allocate {
            id: InstrId(0),
            buffer: BufferId(0),
        },
        StorageInstr::ExtentWrite {
            id: InstrId(1),
            source: source_extent(4096),
            dest: dest_extent(9),
        },
        StorageInstr::BulkExtentWrite {
            id: InstrId(2),
            source: source_extent(65536),
            dest_offset: 4096,
        },
        StorageInstr::SlabScatter {
            id: InstrId(3),
            file_id: FileId(0),
            file_offset: 128,
            span_bytes: 2048,
            placements: vec![
                SlabPlacement {
                    src_offset: 0,
                    dest_offset: 16,
                    bytes: 1024,
                },
                SlabPlacement {
                    src_offset: 1024,
                    dest_offset: 2048,
                    bytes: 1024,
                },
            ],
        },
        StorageInstr::TileMap {
            id: InstrId(4),
            kind: TileMapKind::Repack,
            source: Some(source_extent(256)),
            dest: Some(dest_extent(11)),
            inputs: vec![BufferId(1), BufferId(2)],
            outputs: vec![BufferId(11), BufferId(12)],
            tile: TileSpec {
                max_tile_bytes: 4096,
                rows_per_tile: 64,
            },
            transform: TransformSpec {
                from: Some(QuantScheme::MlxAffineU4),
                to: Some(QuantScheme::Mxfp4E2M1E8M0),
                repack: RepackSpec {
                    layout: RepackLayout::MarlinMxfp4Weight,
                    row_map: RowMap::Odd,
                    batch: 2,
                    source_rows: 32,
                    source_row_offset: 4,
                    target_rows: 64,
                    valid_rows: 30,
                    source_stride_cols: 128,
                    source_col_offset: 8,
                    source_cols: 96,
                    target_cols: 112,
                },
                scratch_bytes: 8192,
                fusion: TransformFusion::Fp8ToMxfp4,
            },
        },
        StorageInstr::CreateView {
            id: InstrId(5),
            input: BufferId(0),
            output: BufferId(13),
            view: dest_extent(13),
            layout: Layout::dense(256),
        },
        StorageInstr::Release {
            id: InstrId(7),
            buffer: BufferId(1),
        },
        StorageInstr::Finalize {
            id: InstrId(8),
            tensor: BufferId(0),
            name: "model.layers.0.mlp.gate_proj.weight".to_string(),
        },
        // The second tensor decl needs a Finalize of its own: `verify` treats a
        // declared-but-never-finalized tensor as a weight the load would leave
        // absent, which is the coverage check §8.2 asks for.
        StorageInstr::Finalize {
            id: InstrId(9),
            tensor: BufferId(1),
            name: "model.norm.weight".to_string(),
        },
    ];
    plan.schedule = (0..plan.instrs.len() as u32).map(InstrId).collect();
    plan.memory = MemoryPlan {
        persistent_bytes: 1 << 30,
        temporary_peak_bytes: 1 << 20,
        transform_scratch_peak_bytes: 8192,
        checkpoint_read_bytes: 1 << 31,
        device_write_bytes: 1 << 30,
    };
    plan.optimizer = OptimizerReport {
        passes: vec![OptimizerPassStats {
            name: "coalesce".to_string(),
            exprs_before: 900,
            exprs_after: 120,
            rewrites: 41,
        }],
    };
    plan
}

/// Runs `body` against a built plan and releases it afterwards.
fn with_plan(plan: &LoadPlan, body: impl FnOnce(*mut PieLoaderPlan)) {
    let pod = arena::build(plan, &arena::PlanExtras::default());
    assert!(!pod.is_null());
    body(pod);
    unsafe { arena::release(pod) };
}

#[test]
fn header_carries_target_and_versions() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let header = unsafe { &*pod };
        assert_eq!(header.version, LOAD_PLAN_VERSION);
        assert_eq!(header.compiler_version, compiler_version());
        assert_eq!(header.target.backend, PieLoaderBackendKind::Cuda);
        assert_eq!(header.target.tp_rank, 1);
        assert_eq!(header.target.tp_size, 4);
        assert_eq!(header.target.max_tile_bytes, 1 << 20);
        assert_eq!(header.target.preferred_alignment, 256);
        assert_eq!(header.target.tile_map_mask, CUDA_TILE_MAP_MASK);
        assert_eq!(header.target.mxfp4_moe, PieLoaderMxfp4MoePolicy::NativeGemm);
        assert!(header.target.native_mxfp4_moe);
        assert!(header.target.fused_transcode);
        assert_eq!(header.memory.persistent_bytes, 1 << 30);
        assert_eq!(header.memory.checkpoint_read_bytes, 1 << 31);
    });
}

#[test]
fn tensors_split_encoding_into_flat_fields() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let tensors = unsafe { view::tensors(pod) };
        assert_eq!(tensors.len(), 2);

        let quant = &tensors[0];
        assert_eq!(quant.id, 7);
        assert_eq!(
            unsafe { view::bytes(&quant.name) },
            "model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(quant.encoding_kind, PieLoaderEncodingKind::Quant);
        // The logical dtype, not the storage dtype: a reader asking "what does
        // this decode to" must not have to know the scheme.
        assert_eq!(quant.dtype, PieLoaderDType::BF16);
        assert_eq!(quant.quant_scheme, PieLoaderQuantScheme::Mxfp4E2M1E8M0);
        assert_eq!(quant.quant_bits_per_element, 4);
        assert_eq!(quant.quant_group_size, 32);
        assert_eq!(unsafe { view::i64s(&quant.shape) }, &[4096, 11008]);
        assert_eq!(quant.alignment, 256);

        // Raw leaves the quant fields at rest rather than filling in defaults,
        // so "unquantized" is distinguishable from "8-bit".
        let raw = &tensors[1];
        assert_eq!(raw.encoding_kind, PieLoaderEncodingKind::Raw);
        assert_eq!(raw.dtype, PieLoaderDType::BF16);
        assert_eq!(raw.quant_scheme, PieLoaderQuantScheme::None);
        assert_eq!(raw.quant_bits_per_element, 0);
        assert_eq!(raw.quant_group_size, 0);
    });
}

#[test]
fn the_plan_declares_the_files_its_offsets_are_relative_to() {
    let plan = plan_with_every_instr();
    let (path, size_bytes) = fixture_file();
    with_plan(&plan, |pod| {
        let files = unsafe { view::files(pod) };
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].id, 0);
        assert_eq!(unsafe { view::bytes(&files[0].path) }, path.as_str());
        assert_eq!(files[0].size_bytes, *size_bytes);
        assert_eq!(files[0].format, PieLoaderCheckpointFormat::Safetensors);
    });
}

/// Verify `plan` against a request that agrees with it in every other respect,
/// so any diagnostic returned is attributable to the mutation under test.
fn verify_diagnostics(plan: &LoadPlan) -> String {
    let pod = arena::build(plan, &arena::PlanExtras::default());
    let mut req = request("/nonexistent");
    req.target.tp_rank = 1;
    req.target.tp_size = 4;
    req.target.native_mxfp4_moe = true;
    req.mxfp4_moe = PieLoaderMxfp4MoeRequest::NativeGemm as u32;
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let _ = unsafe { super::pie_loader_verify(pod, &req, &mut diags) };
    let text = first_message(diags);
    unsafe { super::pie_loader_release_diagnostics(diags) };
    unsafe { super::pie_loader_release(pod) };
    text
}

#[test]
fn verify_rejects_a_plan_whose_checkpoint_changed_size() {
    // The offsets in a plan are byte positions inside these exact files. If the
    // file was replaced between compile and load, the plan would still "work" —
    // it would just read the wrong bytes.
    let mut plan = plan_with_every_instr();
    plan.files[0].size_bytes += 1;
    let text = verify_diagnostics(&plan);
    assert!(
        text.contains("the plan was compiled against"),
        "expected a size-mismatch diagnostic, got {text:?}"
    );
}

#[test]
fn verify_rejects_a_source_that_reads_from_an_undeclared_file() {
    let mut plan = plan_with_every_instr();
    plan.sources[0].file_id = FileId(9);
    let text = verify_diagnostics(&plan);
    assert!(
        text.contains("but the plan declares 1 files"),
        "expected a dangling-file diagnostic, got {text:?}"
    );
}

#[test]
fn verify_rejects_a_tensor_the_schedule_never_finalizes() {
    // §8.2's coverage question, in the half the loader can answer without an
    // externally-supplied contract: a declared tensor that nothing produces is a
    // weight the driver will look up and not find.
    let mut plan = plan_with_every_instr();
    plan.instrs.retain(|instr| {
        !matches!(instr, StorageInstr::Finalize { name, .. } if name == "model.norm.weight")
    });
    plan.schedule = (0..plan.instrs.len() as u32).map(InstrId).collect();
    let text = verify_diagnostics(&plan);
    assert!(
        text.contains("model.norm.weight") && text.contains("never finalized"),
        "expected a coverage diagnostic, got {text:?}"
    );
}

#[test]
fn verify_rejects_a_file_table_whose_ids_are_not_its_indices() {
    let mut plan = plan_with_every_instr();
    plan.files[0].id = FileId(1);
    let text = verify_diagnostics(&plan);
    assert!(
        text.contains("ids must equal their index"),
        "expected an id/index diagnostic, got {text:?}"
    );
}

#[test]
fn sources_carry_file_identity_and_encoding() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let sources = unsafe { view::sources(pod) };
        assert_eq!(sources.len(), 2);
        let quant = &sources[0];
        assert_eq!(unsafe { view::bytes(&quant.name) }, "raw.gate_proj");
        assert_eq!(quant.file_id, 0);
        assert_eq!(quant.file_offset, 512);
        assert_eq!(quant.span_bytes, 4096);
        assert_eq!(quant.quant_scheme, PieLoaderQuantScheme::MlxAffineU4);
        assert_eq!(quant.dtype, PieLoaderDType::F16);
        assert_eq!(quant.quant_bits_per_element, 4);
        assert_eq!(quant.quant_group_size, 64);

        let raw = &sources[1];
        assert_eq!(raw.encoding_kind, PieLoaderEncodingKind::Raw);
        assert_eq!(raw.dtype, PieLoaderDType::F32);
        assert_eq!(unsafe { view::i64s(&raw.shape) }, &[4096]);
    });
}

#[test]
fn optional_buffer_fields_become_explicit_flags() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let buffers = unsafe { view::buffers(pod) };
        assert_eq!(buffers.len(), 2);

        assert!(buffers[0].has_tensor);
        assert_eq!(buffers[0].tensor_id, 7);
        assert!(buffers[0].has_persistent_offset);
        assert_eq!(buffers[0].persistent_offset, 0);
        assert!(!buffers[0].temporary);

        // `None` must reach C as the sentinel *and* a false flag: a reader that
        // checks only the id would otherwise read u32::MAX as a tensor.
        assert!(!buffers[1].has_tensor);
        assert_eq!(buffers[1].tensor_id, PIE_LOADER_NO_BUFFER);
        assert!(!buffers[1].has_persistent_offset);
        assert_eq!(buffers[1].persistent_offset, 0);
        assert!(buffers[1].temporary);
    });
}

#[test]
fn allocate_and_release_carry_only_a_buffer() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let instrs = unsafe { view::instrs(pod) };
        let alloc = &instrs[0];
        assert_eq!(alloc.kind, PieLoaderStorageInstrKind::Allocate);
        assert_eq!(alloc.buffer_id, 0);
        assert!(!alloc.has_source);
        assert!(!alloc.has_dest);
        assert_eq!(alloc.tile_kind, PieLoaderTileMapKind::None);

        let release = &instrs[6];
        assert_eq!(release.kind, PieLoaderStorageInstrKind::Release);
        assert_eq!(release.buffer_id, 1);
        assert!(!release.has_source);
        assert!(!release.has_dest);
    });
}

#[test]
fn extent_write_republishes_dest_buffer_as_instruction_buffer() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let write = &unsafe { view::instrs(pod) }[1];
        assert_eq!(write.kind, PieLoaderStorageInstrKind::ExtentWrite);
        assert!(write.has_source);
        assert!(write.has_dest);
        assert_eq!(write.dest.buffer_id, 9);
        assert_eq!(write.buffer_id, 9);
        assert_eq!(write.source.file_id, 0);
        assert_eq!(write.source.tensor_id, 7);
        assert_eq!(write.source.span_bytes, 4096);
        assert_eq!(write.source.stride.element_bytes, 2);
        assert_eq!(
            unsafe { view::dims(&write.source.stride.dims) },
            &[PieLoaderDimSpecView {
                count: 4,
                src_stride: 8,
                dst_stride: 16,
            }]
        );
        assert_eq!(
            unsafe { view::dims(&write.dest.stride.dims) },
            &[PieLoaderDimSpecView {
                count: 4,
                src_stride: 16,
                dst_stride: 8,
            }]
        );
    });
}

/// The one conversion with no counterpart in the Rust IR: `BulkExtentWrite`
/// carries a bare `dest_offset`, and the executor reads a `DestExtent`. The
/// synthesized extent must be a flat byte run against the sentinel buffer, or an
/// arena-relative write is silently reinterpreted as buffer-relative.
#[test]
fn bulk_extent_write_synthesizes_a_flat_arena_relative_dest() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let bulk = &unsafe { view::instrs(pod) }[2];
        assert_eq!(bulk.kind, PieLoaderStorageInstrKind::BulkExtentWrite);
        assert!(bulk.has_source);
        assert!(bulk.has_dest);
        assert_eq!(bulk.dest.buffer_id, PIE_LOADER_NO_BUFFER);
        assert_eq!(bulk.buffer_id, PIE_LOADER_NO_BUFFER);
        assert_eq!(bulk.dest.offset, 4096);
        assert_eq!(bulk.dest.stride.base_offset, 0);
        assert_eq!(bulk.dest.stride.element_bytes, 1);
        assert_eq!(
            unsafe { view::dims(&bulk.dest.stride.dims) },
            &[PieLoaderDimSpecView {
                count: 65536,
                src_stride: 1,
                dst_stride: 1,
            }]
        );
    });
}

#[test]
fn slab_scatter_carries_placements_and_no_buffer() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let slab = &unsafe { view::instrs(pod) }[3];
        assert_eq!(slab.kind, PieLoaderStorageInstrKind::SlabScatter);
        assert_eq!(slab.slab_file_id, 0);
        assert_eq!(slab.slab_file_offset, 128);
        assert_eq!(slab.slab_span_bytes, 2048);
        assert_eq!(slab.buffer_id, PIE_LOADER_NO_BUFFER);
        assert_eq!(
            unsafe { view::slabs(&slab.slab_placements) },
            &[
                PieLoaderSlabPlacementView {
                    src_offset: 0,
                    dest_offset: 16,
                    bytes: 1024,
                },
                PieLoaderSlabPlacementView {
                    src_offset: 1024,
                    dest_offset: 2048,
                    bytes: 1024,
                },
            ]
        );
    });
}

#[test]
fn tile_map_flattens_transform_and_takes_first_output_as_buffer() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let tile = &unsafe { view::instrs(pod) }[4];
        assert_eq!(tile.kind, PieLoaderStorageInstrKind::TileMap);
        assert_eq!(tile.tile_kind, PieLoaderTileMapKind::Repack);
        assert_eq!(unsafe { view::u32s(&tile.input_buffers) }, &[1, 2]);
        assert_eq!(unsafe { view::u32s(&tile.output_buffers) }, &[11, 12]);
        assert_eq!(tile.buffer_id, 11);
        assert_eq!(tile.rows_per_tile, 64);
        assert_eq!(tile.transform_fusion, PieLoaderTransformFusion::Fp8ToMxfp4);
        assert_eq!(tile.transform_from, PieLoaderQuantScheme::MlxAffineU4);
        assert_eq!(tile.transform_to, PieLoaderQuantScheme::Mxfp4E2M1E8M0);
        assert_eq!(tile.repack_layout, PieLoaderRepackLayout::MarlinMxfp4Weight);
        assert_eq!(tile.row_map, PieLoaderRowMap::Odd);
        assert_eq!(tile.transform_batch, 2);
        assert_eq!(tile.transform_source_rows, 32);
        assert_eq!(tile.transform_source_row_offset, 4);
        assert_eq!(tile.transform_target_rows, 64);
        assert_eq!(tile.transform_valid_rows, 30);
        assert_eq!(tile.transform_source_stride_cols, 128);
        assert_eq!(tile.transform_source_col_offset, 8);
        assert_eq!(tile.transform_source_cols, 96);
        assert_eq!(tile.transform_target_cols, 112);
        assert_eq!(tile.transform_scratch_bytes, 8192);
    });
}

#[test]
fn tile_map_without_outputs_keeps_the_sentinel_buffer() {
    let mut plan = LoadPlan::empty(target());
    plan.instrs.push(StorageInstr::TileMap {
        id: InstrId(0),
        kind: TileMapKind::Cast,
        source: None,
        dest: None,
        inputs: vec![],
        outputs: vec![],
        tile: TileSpec {
            max_tile_bytes: 0,
            rows_per_tile: 0,
        },
        transform: TransformSpec::default(),
    });
    plan.schedule = vec![InstrId(0)];
    with_plan(&plan, |pod| {
        let tile = &unsafe { view::instrs(pod) }[0];
        assert_eq!(tile.buffer_id, PIE_LOADER_NO_BUFFER);
        assert!(!tile.has_source);
        assert!(!tile.has_dest);
        assert_eq!(unsafe { view::u32s(&tile.output_buffers) }, &[] as &[u32]);
        assert_eq!(tile.transform_from, PieLoaderQuantScheme::None);
        assert_eq!(tile.transform_to, PieLoaderQuantScheme::None);
    });
}

#[test]
fn scalar_operands_are_published_as_one_element_runs() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let instrs = unsafe { view::instrs(pod) };

        let view_instr = &instrs[5];
        assert_eq!(view_instr.kind, PieLoaderStorageInstrKind::CreateView);
        assert_eq!(unsafe { view::u32s(&view_instr.input_buffers) }, &[0]);
        assert_eq!(unsafe { view::u32s(&view_instr.output_buffers) }, &[13]);
        assert_eq!(view_instr.buffer_id, 13);
        assert!(view_instr.has_dest);
        assert_eq!(view_instr.dest.buffer_id, 13);

        let finalize = &instrs[7];
        assert_eq!(finalize.kind, PieLoaderStorageInstrKind::Finalize);
        assert_eq!(finalize.buffer_id, 0);
        assert_eq!(unsafe { view::u32s(&finalize.output_buffers) }, &[0]);
        assert_eq!(
            unsafe { view::bytes(&finalize.name) },
            "model.layers.0.mlp.gate_proj.weight"
        );
    });
}

#[test]
fn schedule_and_optimizer_report_survive() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        assert_eq!(
            unsafe { view::schedule(pod) },
            &[0, 1, 2, 3, 4, 5, 6, 7, 8][..]
        );
        let passes = unsafe { view::passes(pod) };
        assert_eq!(passes.len(), 1);
        assert_eq!(unsafe { view::bytes(&passes[0].name) }, "coalesce");
        assert_eq!(passes[0].exprs_before, 900);
        assert_eq!(passes[0].exprs_after, 120);
        assert_eq!(passes[0].rewrites, 41);
    });
}

#[test]
fn empty_plan_publishes_empty_slices_not_dangling_ones() {
    let plan = LoadPlan::empty(target());
    with_plan(&plan, |pod| {
        assert_eq!(unsafe { view::instrs(pod) }.len(), 0);
        assert_eq!(unsafe { view::tensors(pod) }.len(), 0);
        assert_eq!(unsafe { view::sources(pod) }.len(), 0);
        assert_eq!(unsafe { view::buffers(pod) }.len(), 0);
        assert_eq!(unsafe { view::schedule(pod) }.len(), 0);
        assert_eq!(unsafe { view::passes(pod) }.len(), 0);
    });
}

/// Growing the instruction vector must not move the strings and runs already
/// handed out. This is why each run is its own boxed slice rather than a shared
/// growable buffer, and why the C++ it replaces used `std::deque`.
#[test]
fn nested_slices_survive_arena_growth() {
    let mut plan = LoadPlan::empty(target());
    for i in 0..256u32 {
        plan.instrs.push(StorageInstr::Finalize {
            id: InstrId(i),
            tensor: BufferId(i),
            name: format!("tensor.{i}"),
        });
    }
    plan.schedule = (0..256).map(InstrId).collect();
    with_plan(&plan, |pod| {
        let instrs = unsafe { view::instrs(pod) };
        assert_eq!(instrs.len(), 256);
        for (i, instr) in instrs.iter().enumerate() {
            let i = i as u32;
            assert_eq!(unsafe { view::bytes(&instr.name) }, format!("tensor.{i}"));
            assert_eq!(unsafe { view::u32s(&instr.output_buffers) }, &[i]);
        }
    });
}

#[test]
fn quant_scheme_discriminants_are_stable() {
    // The generated header is the only definition of these values, so a
    // reordering of `crate::types::QuantScheme` must not silently renumber the
    // wire format. Pinning the two that differ from the enum this replaces is
    // enough to catch a reorder.
    assert_eq!(
        PieLoaderQuantScheme::from(QuantScheme::MlxAffineU4) as u32,
        8
    );
    assert_eq!(PieLoaderQuantScheme::from(QuantScheme::GgufQ8_0) as u32, 13);
    assert_eq!(PieLoaderQuantScheme::from(QuantScheme::None) as u32, 0);
    assert_eq!(PieLoaderTileMapKind::None as u32, 7);
    assert_eq!(PieLoaderBackendKind::Unknown as u32, 255);
}

#[test]
fn plans_can_be_built_and_released_from_other_threads() {
    // Ranks compile in parallel, so the boundary must not depend on the plan
    // being built and freed on one thread.
    let plan = plan_with_every_instr();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let plan = plan.clone();
            std::thread::spawn(move || {
                let pod = arena::build(&plan, &arena::PlanExtras::default());
                let count = unsafe { view::instrs(pod) }.len();
                (pod as usize, count)
            })
        })
        .collect();
    for handle in handles {
        let (pod, count) = handle.join().unwrap();
        assert_eq!(count, 9);
        unsafe { arena::release(pod as *mut PieLoaderPlan) };
    }
}

#[test]
fn releasing_null_is_a_no_op() {
    unsafe { arena::release(std::ptr::null_mut()) };
    unsafe { super::pie_loader_release(std::ptr::null_mut()) };
    unsafe { super::pie_loader_release_diagnostics(std::ptr::null_mut()) };
}

// ---------------------------------------------------------------------------
// Request validation.
//
// Every enum-valued field of a request arrives as a `uint32_t` precisely so
// these cases can be *rejected* rather than being undefined behaviour, so the
// rejection is the thing worth testing.
// ---------------------------------------------------------------------------

fn bytes(s: &str) -> PieLoaderBytes {
    PieLoaderBytes {
        ptr: s.as_ptr(),
        len: s.len(),
    }
}

fn request(dir: &str) -> PieLoaderRequest {
    PieLoaderRequest {
        snapshot_dir: bytes(dir),
        target: PieLoaderTargetSpec {
            backend: PieLoaderBackendKind::Cuda as u32,
            tp_rank: 0,
            tp_size: 1,
            max_tile_bytes: 1 << 20,
            preferred_alignment: 256,
            tile_map_mask: CUDA_TILE_MAP_MASK,
            fp8_native: false,
            native_mxfp4_moe: false,
            fused_transcode: true,
        },
        model: PieLoaderModelSpec {
            model_type: bytes("llama"),
            quant_method: bytes(""),
            num_hidden_layers: 1,
            num_experts: 0,
            num_experts_per_tok: 0,
        },
        runtime_quant: bytes(""),
        mxfp4_moe: PieLoaderMxfp4MoeRequest::Auto as u32,
        component: PieLoaderComponent::Full as u32,
        demands: PieLoaderTensorDemandSlice::default(),
    }
}

/// Drive `pie_loader_compile` and return the status plus the first diagnostic.
fn compile(req: &PieLoaderRequest) -> (PieLoaderStatus, String) {
    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::pie_loader_compile(req, &mut plan, &mut diags) };
    let message = if diags.is_null() {
        String::new()
    } else {
        let items = unsafe { std::slice::from_raw_parts((*diags).items, (*diags).len) };
        let first = items
            .first()
            .map(|d| {
                let raw = unsafe { std::slice::from_raw_parts(d.message.ptr, d.message.len) };
                String::from_utf8_lossy(raw).into_owned()
            })
            .unwrap_or_default();
        unsafe { super::pie_loader_release_diagnostics(diags) };
        first
    };
    if !plan.is_null() {
        unsafe { super::pie_loader_release(plan) };
    }
    (status, message)
}

#[test]
fn out_of_range_backend_is_rejected_not_transmuted() {
    let mut req = request("/nonexistent");
    req.target.backend = 7;
    let (status, message) = compile(&req);
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(
        message.contains("backend") && message.contains('7'),
        "unexpected diagnostic: {message}"
    );
}

#[test]
fn a_target_claiming_an_unknown_tile_map_transform_is_rejected() {
    // The driver is the authority on what its kernels implement (§9), so it may
    // claim fewer transforms than the loader knows how to lower. It may not
    // claim one the loader has never heard of: nothing downstream would notice
    // until a kernel dispatch failed at load time.
    let mut req = request("/nonexistent");
    req.target.tile_map_mask = CUDA_TILE_MAP_MASK | (1 << 30);
    let (status, message) = compile(&req);
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(
        message.contains("does not define") && message.contains("cuda"),
        "got: {message}"
    );
}

#[test]
fn a_target_claiming_fewer_transforms_than_the_loader_knows_is_accepted() {
    // The converse: a driver that implements a subset is well-formed. Only the
    // plan's *use* of a transform is checked against the claim, and that check
    // already lives in `validate_target_support`.
    let mut req = request("/nonexistent");
    req.target.tile_map_mask = CUDA_TILE_MAP_MASK & !TILE_MAP_REPACK;
    let (status, message) = compile(&req);
    assert_ne!(
        status,
        PieLoaderStatus::InvalidRequest,
        "a narrower mask is not a malformed request: {message}"
    );
}

#[test]
fn out_of_range_mxfp4_moe_is_rejected() {
    let mut req = request("/nonexistent");
    req.mxfp4_moe = 99;
    let (status, message) = compile(&req);
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(message.contains("mxfp4_moe"), "got: {message}");
}

#[test]
fn out_of_range_component_is_rejected() {
    let mut req = request("/nonexistent");
    req.component = u32::MAX;
    let (status, message) = compile(&req);
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(message.contains("component"), "got: {message}");
}

#[test]
fn every_declared_enum_value_is_accepted_by_its_checked_conversion() {
    for (v, want) in [
        (0, PieLoaderBackendKind::Cuda),
        (1, PieLoaderBackendKind::Metal),
        (255, PieLoaderBackendKind::Unknown),
    ] {
        assert_eq!(PieLoaderBackendKind::try_from(v), Ok(want));
    }
    for v in 0..=3u32 {
        assert!(PieLoaderMxfp4MoeRequest::try_from(v).is_ok());
    }
    assert!(PieLoaderMxfp4MoeRequest::try_from(4).is_err());
    for v in 0..=2u32 {
        assert!(PieLoaderComponent::try_from(v).is_ok());
    }
    assert!(PieLoaderComponent::try_from(3).is_err());
}

#[test]
fn native_gemm_without_device_support_is_rejected() {
    let mut req = request("/nonexistent");
    req.mxfp4_moe = PieLoaderMxfp4MoeRequest::NativeGemm as u32;
    req.target.native_mxfp4_moe = false;
    let (status, message) = compile(&req);
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(message.contains("native_mxfp4_moe=false"), "got: {message}");
}

#[test]
fn zero_alignment_and_bad_tp_shape_are_rejected() {
    let mut req = request("/nonexistent");
    req.target.preferred_alignment = 0;
    assert_eq!(compile(&req).0, PieLoaderStatus::InvalidRequest);

    let mut req = request("/nonexistent");
    req.target.tp_size = 0;
    assert_eq!(compile(&req).0, PieLoaderStatus::InvalidRequest);

    let mut req = request("/nonexistent");
    req.target.tp_rank = 4;
    req.target.tp_size = 4;
    assert_eq!(compile(&req).0, PieLoaderStatus::InvalidRequest);
}

#[test]
fn null_arguments_never_dereference() {
    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();

    let status = unsafe { super::pie_loader_compile(std::ptr::null(), &mut plan, &mut diags) };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(plan.is_null());
    assert!(!diags.is_null(), "a null request should still be explained");
    unsafe { super::pie_loader_release_diagnostics(diags) };

    let req = request("/nonexistent");
    let status =
        unsafe { super::pie_loader_compile(&req, std::ptr::null_mut(), std::ptr::null_mut()) };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);

    let status = unsafe { super::pie_loader_verify(std::ptr::null(), &req, std::ptr::null_mut()) };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
}

#[test]
fn verify_rejects_a_plan_compiled_under_a_different_moe_policy() {
    // The plan below resolved to NativeGemm. A request that resolves to
    // RoutedDecode must not verify against it, even though every other field
    // agrees — this is exactly the rank-divergence case of §6.2.
    let pod = arena::build(&plan_with_every_instr(), &arena::PlanExtras::default());

    let mut req = request("/nonexistent");
    req.target.tp_rank = 1;
    req.target.tp_size = 4;
    req.target.native_mxfp4_moe = true;
    req.mxfp4_moe = PieLoaderMxfp4MoeRequest::NativeGemm as u32;
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::pie_loader_verify(pod, &req, &mut diags) };
    assert_eq!(
        status,
        PieLoaderStatus::Ok,
        "matching policy should verify; diagnostics: {}",
        first_message(diags)
    );
    unsafe { super::pie_loader_release_diagnostics(diags) };

    req.mxfp4_moe = PieLoaderMxfp4MoeRequest::EagerBf16 as u32;
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::pie_loader_verify(pod, &req, &mut diags) };
    assert_eq!(status, PieLoaderStatus::ContractViolation);
    assert!(first_message(diags).contains("mxfp4_moe"));
    unsafe { super::pie_loader_release_diagnostics(diags) };

    unsafe { super::pie_loader_release(pod) };
}

#[test]
fn verify_rejects_a_plan_compiled_with_a_different_fusion_setting() {
    // The two settings name different kernel sequences for the same weights.
    // Accepting a fused plan on a driver that has fusion disabled would run
    // something the plan does not describe — and, because the driver caches the
    // materialized artifact, would do so from a cache entry the other setting
    // wrote (§8.1).
    let pod = arena::build(&plan_with_every_instr(), &arena::PlanExtras::default());

    let mut req = request("/nonexistent");
    req.target.tp_rank = 1;
    req.target.tp_size = 4;
    req.target.native_mxfp4_moe = true;
    req.mxfp4_moe = PieLoaderMxfp4MoeRequest::NativeGemm as u32;
    req.target.fused_transcode = false;

    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::pie_loader_verify(pod, &req, &mut diags) };
    assert_eq!(status, PieLoaderStatus::ContractViolation);
    assert!(first_message(diags).contains("fused_transcode"));
    unsafe { super::pie_loader_release_diagnostics(diags) };

    unsafe { super::pie_loader_release(pod) };
}

fn first_message(diags: *mut PieLoaderDiagnostics) -> String {
    if diags.is_null() {
        return String::new();
    }
    let items = unsafe { std::slice::from_raw_parts((*diags).items, (*diags).len) };
    items
        .iter()
        .map(|d| {
            let raw = unsafe { std::slice::from_raw_parts(d.message.ptr, d.message.len) };
            String::from_utf8_lossy(raw).into_owned()
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// A plan compiled from a real checkpoint must survive its own `verify`, at
/// every tensor-parallel width, without the instruction count exploding.
///
/// Both drivers call `verify` immediately after `compile` (§8), so a check that
/// is too strict does not fail a test — it fails every model load. The unit
/// fixtures cannot catch that, because they are hand-built to satisfy the
/// checks; only a real checkpoint exercises the compiler's actual output.
///
/// The instruction budget is the guard on the lowering's cost model. A sharded
/// tensor is thousands of contiguous runs and must still be *one* strided copy;
/// if that folding ever stops working the plan grows by three orders of
/// magnitude, which nothing else here would notice until a model took minutes
/// to load. Sharding is also the only thing `tp_size` changes, so sweeping it
/// is what exercises the strided path at all.
/// Build a borrowed demand list. The backing storage must outlive the call, so
/// callers keep the returned vectors alive.
fn demands(
    entries: &[(&'static str, Vec<i64>, bool)],
) -> (Vec<PieLoaderTensorDemand>, Vec<Vec<i64>>) {
    let shapes: Vec<Vec<i64>> = entries.iter().map(|(_, shape, _)| shape.clone()).collect();
    let demands = entries
        .iter()
        .zip(&shapes)
        .map(|((name, _, optional), shape)| PieLoaderTensorDemand {
            name: bytes(name),
            shape: PieLoaderI64Slice {
                ptr: if shape.is_empty() {
                    std::ptr::null()
                } else {
                    shape.as_ptr()
                },
                len: shape.len(),
            },
            optional: *optional,
        })
        .collect();
    (demands, shapes)
}

fn verify_with(
    pod: *const PieLoaderPlan,
    entries: &[(&'static str, Vec<i64>, bool)],
) -> (PieLoaderStatus, String) {
    let (demands, _shapes) = demands(entries);
    // Match the target `plan_with_every_instr` was built for, so the only thing
    // under test is the contract.
    let mut req = request("/nonexistent");
    req.target.tp_rank = 1;
    req.target.tp_size = 4;
    req.target.native_mxfp4_moe = true;
    req.mxfp4_moe = PieLoaderMxfp4MoeRequest::NativeGemm as u32;
    req.demands = PieLoaderTensorDemandSlice {
        ptr: demands.as_ptr(),
        len: demands.len(),
    };
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::pie_loader_verify(pod, &req, &mut diags) };
    let message = first_message(diags);
    unsafe { super::pie_loader_release_diagnostics(diags) };
    (status, message)
}

/// The point of the whole exercise: a contract the loader did not author,
/// checked against the plan it did.
#[test]
fn a_driver_declared_contract_is_checked_against_the_plan() {
    let pod = arena::build(&plan_with_every_instr(), &arena::PlanExtras::default());

    let (status, message) = verify_with(
        pod,
        &[
            (
                "model.layers.0.mlp.gate_proj.weight",
                vec![4096, 11008],
                false,
            ),
            ("model.norm.weight", vec![4096], false),
        ],
    );
    assert_eq!(status, PieLoaderStatus::Ok, "diagnostics: {message}");

    unsafe { super::pie_loader_release(pod) };
}

#[test]
fn a_declared_demand_the_plan_does_not_deliver_is_a_violation() {
    let pod = arena::build(&plan_with_every_instr(), &arena::PlanExtras::default());

    let (status, message) = verify_with(
        pod,
        &[("model.embed_tokens.weight", vec![32000, 4096], false)],
    );
    assert_eq!(status, PieLoaderStatus::ContractViolation);
    assert!(
        message.contains("model.embed_tokens.weight") && message.contains("does not declare it"),
        "unexpected diagnostic: {message}"
    );

    // ...unless the driver said the weight is conditional. `tie_word_embeddings`
    // is the real case: the binder falls back to the embedding table.
    let (status, message) = verify_with(
        pod,
        &[("model.embed_tokens.weight", vec![32000, 4096], true)],
    );
    assert_eq!(status, PieLoaderStatus::Ok, "diagnostics: {message}");

    unsafe { super::pie_loader_release(pod) };
}

/// The check that actually earns its keep. A pass that computes a shape the
/// driver disagrees with used to bind silently and produce garbage.
#[test]
fn a_declared_shape_that_disagrees_with_the_plan_is_a_violation() {
    let pod = arena::build(&plan_with_every_instr(), &arena::PlanExtras::default());

    let (status, message) = verify_with(pod, &[("model.norm.weight", vec![8192], false)]);
    assert_eq!(status, PieLoaderStatus::ContractViolation);
    assert!(
        message.contains("model.norm.weight") && message.contains("8192"),
        "unexpected diagnostic: {message}"
    );

    // An empty shape is "unstated", not "scalar": presence is still demanded,
    // the shape simply is not compared.
    let (status, message) = verify_with(pod, &[("model.norm.weight", vec![], false)]);
    assert_eq!(status, PieLoaderStatus::Ok, "diagnostics: {message}");

    unsafe { super::pie_loader_release(pod) };
}

/// A driver that declares nothing must keep verifying exactly as before —
/// otherwise adopting the declaration could not be incremental.
#[test]
fn an_undeclared_contract_still_verifies_the_plan_itself() {
    let pod = arena::build(&plan_with_every_instr(), &arena::PlanExtras::default());
    let (status, message) = verify_with(pod, &[]);
    assert_eq!(status, PieLoaderStatus::Ok, "diagnostics: {message}");
    unsafe { super::pie_loader_release(pod) };
}

#[test]
fn a_malformed_demand_is_rejected_rather_than_dereferenced() {
    let pod = arena::build(&plan_with_every_instr(), &arena::PlanExtras::default());

    let (status, message) = verify_with(pod, &[("", vec![4096], false)]);
    assert_eq!(status, PieLoaderStatus::ContractViolation);
    assert!(message.contains("empty name"), "unexpected: {message}");

    let (status, message) = verify_with(pod, &[("model.norm.weight", vec![0], false)]);
    assert_eq!(status, PieLoaderStatus::ContractViolation);
    assert!(message.contains("non-positive"), "unexpected: {message}");

    unsafe { super::pie_loader_release(pod) };
}

/// The payoff of §12 row 7b-4b: the contract the *driver* declares, checked
/// against a plan compiled for a real checkpoint.
///
/// The declaration is not recomputed here. It is read from a JSON file written
/// by `driver/cuda/tests/model_contract_test.cpp`, which calls the real
/// `declare_model_contract`. Transcribing the formulas into Rust instead would
/// make this test agree with a copy rather than with the driver — and the copy
/// had already drifted (`infer_qk_norm` in `model/config.cpp` treats `qwen3_5`
/// as qk-normed too) before the transcription was deleted.
///
/// Nothing here knows about any family, so declaring a new one is a change to
/// `model_contracts.hpp` alone:
///
/// ```text
/// BD=$(ls -dt target/debug/build/pie-worker-*/out/cuda/build | head -1)
/// PIE_TEST_SNAPSHOT=$d PIE_TEST_CONTRACT_DUMP=/tmp/c.json $BD/bin/test_model_contract
/// PIE_TEST_SNAPSHOT=$d PIE_TEST_CONTRACT_DUMP=/tmp/c.json \
///   cargo test -p pie-loader --lib -- --ignored --exact \
///   ffi::tests::a_declared_contract_matches_a_real_plan
/// ```
#[test]
#[ignore = "requires PIE_TEST_SNAPSHOT and PIE_TEST_CONTRACT_DUMP"]
fn a_declared_contract_matches_a_real_plan() {
    let snapshot = std::env::var("PIE_TEST_SNAPSHOT").expect("set PIE_TEST_SNAPSHOT");
    let dump_path = std::env::var("PIE_TEST_CONTRACT_DUMP").expect("set PIE_TEST_CONTRACT_DUMP");
    let dump: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&dump_path).expect("contract dump exists"))
            .expect("contract dump parses");

    // A family with no declaration yet dumps an empty array. That is the
    // designed no-op, not a pass to be counted, so say so and stop.
    let by_tp = &dump["by_tp_size"];
    if by_tp["1"].as_array().is_none_or(Vec::is_empty) {
        eprintln!(
            "{} declares no contract yet; nothing to check",
            dump["model_type"]
        );
        return;
    }

    let read = |tp: u32| -> Vec<(String, Vec<i64>, bool)> {
        by_tp[tp.to_string()]
            .as_array()
            .expect("a declaration per tp size")
            .iter()
            .map(|d| {
                (
                    d["name"].as_str().expect("name").to_string(),
                    d["shape"]
                        .as_array()
                        .expect("shape")
                        .iter()
                        .map(|v| v.as_i64().expect("dim"))
                        .collect(),
                    d["optional"].as_bool().expect("optional"),
                )
            })
            .collect()
    };

    // Guards against the whole test going vacuous. Every declared shape is
    // checked, so if sharding were silently dropped from the declaration the
    // unsharded shapes would still have to match a sharded plan — which they
    // must not. Assert the declaration actually changes with `tp_size`.
    let unsharded = read(1);
    let sharded = read(4);
    assert!(
        unsharded
            .iter()
            .zip(&sharded)
            .any(|((_, a, _), (_, b, _))| a != b),
        "the declared contract must depend on tp_size"
    );

    let cfg_model = crate::ffi::inproc::parse_model_config(std::path::Path::new(&snapshot), "")
        .expect("snapshot has a readable config.json");

    for (tp_size, tp_rank) in [(1, 0), (2, 1), (4, 3)] {
        let declared = read(tp_size);
        let entries: Vec<PieLoaderTensorDemand> = declared
            .iter()
            .map(|(name, shape, optional)| PieLoaderTensorDemand {
                name: PieLoaderBytes {
                    ptr: name.as_ptr(),
                    len: name.len(),
                },
                shape: PieLoaderI64Slice {
                    ptr: shape.as_ptr(),
                    len: shape.len(),
                },
                optional: *optional,
            })
            .collect();

        let mut req = request(&snapshot);
        req.model = PieLoaderModelSpec {
            model_type: bytes(&cfg_model.model_type),
            quant_method: bytes(&cfg_model.quant_method),
            num_hidden_layers: cfg_model.num_hidden_layers,
            num_experts: cfg_model.num_experts,
            num_experts_per_tok: cfg_model.num_experts_per_tok,
        };
        req.target.fp8_native = true;
        req.target.tp_size = tp_size;
        req.target.tp_rank = tp_rank;
        req.demands = PieLoaderTensorDemandSlice {
            ptr: entries.as_ptr(),
            len: entries.len(),
        };

        let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
        let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
        let status = unsafe { super::pie_loader_compile(&req, &mut plan, &mut diags) };
        assert_eq!(status, PieLoaderStatus::Ok, "{}", first_message(diags));
        unsafe { super::pie_loader_release_diagnostics(diags) };

        let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
        let status = unsafe { super::pie_loader_verify(plan, &req, &mut diags) };
        assert_eq!(
            status,
            PieLoaderStatus::Ok,
            "the driver's declared contract disagrees with the plan at tp {tp_rank}/{tp_size}: {}",
            first_message(diags)
        );
        eprintln!(
            "tp {tp_rank}/{tp_size}: {} declared tensors matched",
            entries.len()
        );
        unsafe { super::pie_loader_release_diagnostics(diags) };
        unsafe { super::pie_loader_release(plan) };
    }
}

#[test]
#[ignore = "requires PIE_TEST_SNAPSHOT"]
fn a_real_checkpoint_compiles_to_a_plan_that_verifies() {
    let snapshot = std::env::var("PIE_TEST_SNAPSHOT").expect("set PIE_TEST_SNAPSHOT");
    let runtime_quant = std::env::var("PIE_TEST_RUNTIME_QUANT").unwrap_or_default();
    // A driver would state these; a test holding only a snapshot directory
    // reads them the way the CLI does.
    let cfg = crate::ffi::inproc::parse_model_config(std::path::Path::new(&snapshot), "")
        .expect("snapshot has a readable config.json");
    for (tp_size, tp_rank) in [(1, 0), (2, 0), (2, 1), (4, 3)] {
        let mut req = request(&snapshot);
        req.model = PieLoaderModelSpec {
            model_type: bytes(&cfg.model_type),
            quant_method: bytes(&cfg.quant_method),
            num_hidden_layers: cfg.num_hidden_layers,
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
        };
        req.target.fp8_native = true;
        req.target.tp_size = tp_size;
        req.target.tp_rank = tp_rank;
        req.runtime_quant = bytes(&runtime_quant);

        let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
        let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
        let status = unsafe { super::pie_loader_compile(&req, &mut plan, &mut diags) };
        assert_eq!(
            status,
            PieLoaderStatus::Ok,
            "compile failed at tp {tp_rank}/{tp_size}: {}",
            first_message(diags)
        );
        unsafe { super::pie_loader_release_diagnostics(diags) };

        let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
        let status = unsafe { super::pie_loader_verify(plan, &req, &mut diags) };
        assert_eq!(
            status,
            PieLoaderStatus::Ok,
            "verify rejected a freshly compiled plan at tp {tp_rank}/{tp_size}: {}",
            first_message(diags)
        );
        unsafe { super::pie_loader_release_diagnostics(diags) };

        let files = unsafe { std::slice::from_raw_parts((*plan).files.ptr, (*plan).files.len) };
        assert!(
            !files.is_empty(),
            "the plan must declare the files it reads"
        );

        let instrs = unsafe { std::slice::from_raw_parts((*plan).instrs.ptr, (*plan).instrs.len) };
        let finalized = instrs
            .iter()
            .filter(|i| i.kind == PieLoaderStorageInstrKind::Finalize)
            .count();
        assert!(finalized > 0, "the plan finalizes nothing");
        assert!(
            instrs.len() <= finalized * 8,
            "tp {tp_rank}/{tp_size}: {} instructions for {finalized} tensors — \
             the lowering has stopped folding strided copies",
            instrs.len()
        );
        unsafe { super::pie_loader_release(plan) };
    }
}

/// The encode component must yield a plan that finalizes the encoder towers and
/// nothing else. Migrated here from the worker when plan compilation moved
/// behind the FFI: the scoping is the loader's rule, so the test that pins it
/// belongs to the loader.
#[test]
#[ignore = "requires PIE_TEST_GEMMA4_SNAPSHOT"]
fn gemma4_encode_component_finalizes_only_encoder_tensors() {
    let snapshot = std::env::var("PIE_TEST_GEMMA4_SNAPSHOT").expect("set PIE_TEST_GEMMA4_SNAPSHOT");
    let mut req = request(&snapshot);
    req.target.fp8_native = true;
    req.component = PieLoaderComponent::Encode as u32;

    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::pie_loader_compile(&req, &mut plan, &mut diags) };
    assert_eq!(
        status,
        PieLoaderStatus::Ok,
        "compile failed: {}",
        first_message(diags)
    );
    unsafe { super::pie_loader_release_diagnostics(diags) };

    let instrs = unsafe { std::slice::from_raw_parts((*plan).instrs.ptr, (*plan).instrs.len) };
    let finalized = instrs
        .iter()
        .filter(|i| i.kind == PieLoaderStorageInstrKind::Finalize)
        .map(|i| {
            let raw = unsafe { std::slice::from_raw_parts(i.name.ptr, i.name.len) };
            String::from_utf8_lossy(raw).into_owned()
        })
        .collect::<Vec<_>>();

    assert!(finalized.len() > 10);
    assert!(
        finalized
            .iter()
            .any(|name| name.starts_with("model.vision_tower."))
    );
    assert!(
        finalized
            .iter()
            .any(|name| name.starts_with("model.audio_tower."))
    );
    assert!(finalized.iter().all(|name| {
        name.starts_with("model.vision_tower.")
            || name.starts_with("model.embed_vision.")
            || name.starts_with("model.audio_tower.")
            || name.starts_with("model.embed_audio.")
    }));

    unsafe { super::pie_loader_release(plan) };
}

/// The POD views carry `dtype` and `quant_scheme` as flat enums, and
/// verification rebuilds an [`Encoding`] from them to compare a marshalled
/// tensor against a typed one. That only works if the two conversion directions
/// agree, and nothing else forces them to.
#[test]
fn dtype_survives_the_c_boundary() {
    use crate::types::DType;
    for dtype in [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F8E4M3,
        DType::F8E5M2,
        DType::I32,
        DType::I16,
        DType::I8,
        DType::U32,
        DType::U16,
        DType::U8,
        DType::Bool,
    ] {
        let round_tripped: DType = PieLoaderDType::from(dtype).into();
        assert_eq!(round_tripped, dtype);
    }
}

#[test]
fn quant_scheme_survives_the_c_boundary() {
    use crate::types::QuantScheme;
    for scheme in [
        QuantScheme::None,
        QuantScheme::Fp8E4M3,
        QuantScheme::Fp8E5M2,
        QuantScheme::Int8Symmetric,
        QuantScheme::Int8Asymmetric,
        QuantScheme::AwqInt4,
        QuantScheme::GptqInt4,
        QuantScheme::Mxfp4E2M1E8M0,
        QuantScheme::MlxAffineU4,
        QuantScheme::GgufQ4_0,
        QuantScheme::GgufQ4K,
        QuantScheme::GgufQ5_0,
        QuantScheme::GgufQ5K,
        QuantScheme::GgufQ8_0,
    ] {
        let round_tripped: QuantScheme = PieLoaderQuantScheme::from(scheme).into();
        assert_eq!(round_tripped, scheme);
    }
}
