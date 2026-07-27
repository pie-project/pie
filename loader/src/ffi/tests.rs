//! Round-trip tests for the POD boundary.
//!
//! Each test builds a `LoadPlan` by hand, converts it, and reads the result back
//! through raw pointers. The assertions target the *synthesized* fields — the
//! ones the C++ parser produced that have no direct counterpart in the Rust IR —
//! because those are the only places the conversion can silently disagree with
//! the executor that reads it.

use super::arena::{self, view};
use super::entry::{PieLoaderDiagnostics, PieLoaderStatus, PieLoaderTargetSpec};
use super::types::*;
use crate::load_plan::*;
use crate::types::*;

fn target() -> StorageTarget {
    StorageTarget {
        backend: BackendKind::Cuda,
        tp_rank: 1,
        tp_size: 4,
        max_tile_bytes: 1 << 20,
        preferred_alignment: 256,
        tile_map_mask: CUDA_TILE_MAP_MASK,
        native_mxfp4_moe: true,
        fusion_mask: FUSION_FP8_TO_MXFP4,
        encode_scratch_dtype: DType::BF16,
        block_scale_rows: 128,
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
        alignment: 256,
    });
    plan.tensors.push(TensorDecl {
        id: TensorId(8),
        name: "model.norm.weight".to_string(),
        shape: vec![4096],
        encoding: Encoding::Raw(DType::BF16),
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
                metadata_source: Some(TensorId(2)),
            },
        },
        StorageInstr::CreateView {
            id: InstrId(5),
            input: BufferId(0),
            output: BufferId(13),
            view: dest_extent(13),
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
fn header_carries_target_and_compiler_version() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let header = unsafe { &*pod };
        assert_eq!(header.compiler_version, compiler_version());
        assert_eq!(header.target.backend, PieLoaderBackendKind::Cuda);
        assert_eq!(header.target.tp_rank, 1);
        assert_eq!(header.target.tp_size, 4);
        assert_eq!(header.target.max_tile_bytes, 1 << 20);
        assert_eq!(header.target.preferred_alignment, 256);
        assert_eq!(header.target.tile_map_mask, CUDA_TILE_MAP_MASK);
        assert!(header.target.native_mxfp4_moe);
        assert_eq!(header.target.fusion_mask, FUSION_FP8_TO_MXFP4);
        assert_eq!(header.target.encode_scratch_dtype, PieLoaderDType::BF16);
        assert_eq!(header.target.block_scale_rows, 128);
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
///
/// The contract names one tensor the plan does deliver: a plan may deliver more
/// than a contract names, so the narrowest legal contract leaves only the plan's
/// *internal* consistency under test — which is what every caller of this helper
/// is mutating.
fn verify_diagnostics(plan: &LoadPlan) -> String {
    let pod = arena::build(plan, &arena::PlanExtras::default());
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let owned = crate::contract_writer::write_contract(&minimal_contract());
    let mut req = contract_request(handle, owned.view());
    req.target.tp_rank = 1;
    req.target.tp_size = 4;
    req.target.native_mxfp4_moe = true;
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let _ = unsafe { super::entry::pie_loader_verify_contract(pod, &req, &mut diags) };
    let text = all_messages(diags);
    unsafe { super::entry::pie_loader_release_diagnostics(diags) };
    unsafe { super::entry::pie_loader_release(pod) };
    unsafe { super::entry::pie_loader_close_checkpoint(handle) };
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
        assert_eq!(tile.transform_metadata_source, 2);
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
fn the_schedule_survives() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        assert_eq!(
            unsafe { view::schedule(pod) },
            &[0, 1, 2, 3, 4, 5, 6, 7, 8][..]
        );
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
// Target validation.
//
// Every enum-valued field of a request arrives as a `uint32_t` precisely so
// these cases can be *rejected* rather than being undefined behaviour, so the
// rejection is the thing worth testing. Since §12 row 12 the request carries no
// model and no policy, so the target is all there is left to get wrong — and it
// is the half the loader cannot re-derive, which makes refusing a malformed one
// the only defence.
// ---------------------------------------------------------------------------

fn bytes(s: &str) -> PieLoaderBytes {
    PieLoaderBytes {
        ptr: s.as_ptr(),
        len: s.len(),
    }
}

fn target_spec() -> PieLoaderTargetSpec {
    PieLoaderTargetSpec {
        backend: PieLoaderBackendKind::Cuda as u32,
        tp_rank: 0,
        tp_size: 1,
        max_tile_bytes: 1 << 20,
        preferred_alignment: 256,
        tile_map_mask: CUDA_TILE_MAP_MASK,
        native_mxfp4_moe: false,
        fusion_mask: FUSION_FP8_TO_MXFP4,
        encode_scratch_dtype: PieLoaderDType::BF16 as u32,
        block_scale_rows: 128,
    }
}

/// Compile the fixture contract against a target the caller has broken, and
/// report what came back.
///
/// A real checkpoint is opened even though every case here is expected to be
/// refused before the checkpoint is read: a request that carried a null handle
/// would be rejected for *that* reason instead, and the test would pass without
/// exercising the field it names.
fn compile_with_target(mutate: impl FnOnce(&mut PieLoaderTargetSpec)) -> (PieLoaderStatus, String) {
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let owned = crate::contract_writer::write_contract(&fused_contract());
    let mut req = contract_request(handle, owned.view());
    mutate(&mut req.target);

    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::entry::pie_loader_compile_contract(&req, &mut plan, &mut diags) };
    let message = drain(diags);
    if !plan.is_null() {
        unsafe { super::entry::pie_loader_release(plan) };
    }
    unsafe { super::entry::pie_loader_close_checkpoint(handle) };
    (status, message)
}

#[test]
fn out_of_range_backend_is_rejected_not_transmuted() {
    let (status, message) = compile_with_target(|t| t.backend = 7);
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
    let (status, message) =
        compile_with_target(|t| t.tile_map_mask = CUDA_TILE_MAP_MASK | (1 << 30));
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
    let (status, message) =
        compile_with_target(|t| t.tile_map_mask = CUDA_TILE_MAP_MASK & !TILE_MAP_REPACK);
    assert_eq!(
        status,
        PieLoaderStatus::Ok,
        "a narrower mask is not a malformed request: {message}"
    );
}

#[test]
fn a_target_that_states_no_tile_budget_is_rejected_rather_than_guessed_for() {
    // §9: a device constant the loader cannot measure has no safe default. The
    // number decides how much scratch every Encode allocates, so guessing it
    // would be a performance contract nobody signed.
    let (status, message) = compile_with_target(|t| t.max_tile_bytes = 0);
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(message.contains("max_tile_bytes"), "got: {message}");
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
    assert!(PieLoaderBackendKind::try_from(2).is_err());
}

#[test]
fn a_bad_tp_shape_is_rejected() {
    assert_eq!(
        compile_with_target(|t| t.tp_size = 0).0,
        PieLoaderStatus::InvalidRequest
    );
    assert_eq!(
        compile_with_target(|t| {
            t.tp_rank = 4;
            t.tp_size = 4;
        })
        .0,
        PieLoaderStatus::InvalidRequest
    );
}

#[test]
fn null_arguments_never_dereference() {
    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();

    let status = unsafe {
        super::entry::pie_loader_compile_contract(std::ptr::null(), &mut plan, &mut diags)
    };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(plan.is_null());
    assert!(!diags.is_null(), "a null request should still be explained");
    unsafe { super::entry::pie_loader_release_diagnostics(diags) };

    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let owned = crate::contract_writer::write_contract(&fused_contract());
    let req = contract_request(handle, owned.view());

    let status = unsafe {
        super::entry::pie_loader_compile_contract(&req, std::ptr::null_mut(), std::ptr::null_mut())
    };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);

    let status = unsafe {
        super::entry::pie_loader_verify_contract(std::ptr::null(), &req, std::ptr::null_mut())
    };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);

    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe {
        super::entry::pie_loader_open_checkpoint(
            bytes("/nonexistent"),
            std::ptr::null_mut(),
            &mut diags,
        )
    };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    unsafe { super::entry::pie_loader_release_diagnostics(diags) };

    unsafe { super::entry::pie_loader_close_checkpoint(handle) };
}

#[test]
fn verify_rejects_a_plan_compiled_with_a_different_fusion_setting() {
    // The two settings name different kernel sequences for the same weights.
    // Accepting a fused plan on a driver that has fusion disabled would run
    // something the plan does not describe — and, because the driver caches the
    // materialized artifact, would do so from a cache entry the other setting
    // wrote (§8.1).
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let owned = crate::contract_writer::write_contract(&fused_contract());
    let req = contract_request(handle, owned.view());

    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::entry::pie_loader_compile_contract(&req, &mut plan, &mut diags) };
    assert_eq!(
        status,
        PieLoaderStatus::Ok,
        "compile failed: {}",
        drain(diags)
    );

    let mut diverged = req;
    diverged.target.fusion_mask = 0;
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::entry::pie_loader_verify_contract(plan, &diverged, &mut diags) };
    assert_eq!(status, PieLoaderStatus::ContractViolation);
    assert!(drain(diags).contains("fusion_mask"));

    unsafe { super::entry::pie_loader_release(plan) };
    unsafe { super::entry::pie_loader_close_checkpoint(handle) };
}

/// Join every diagnostic in the array. Verification reports *all* violations,
/// not the first, so a test that read only one could pass on the wrong reason.
fn all_messages(diags: *mut PieLoaderDiagnostics) -> String {
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

/// Verify a plan against a contract *other* than the one it was compiled from.
///
/// This is the only way a contract and a plan can disagree now: compiling from
/// a contract makes them agree by construction, so the failure mode left is the
/// §6.2 one — two ranks, or a cache hit and a fresh compile, holding different
/// programs and each believing the other's plan.
fn verify_against(
    plan: *const PieLoaderPlan,
    contract: &crate::contract::ModelContract,
) -> (PieLoaderStatus, String) {
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let owned = crate::contract_writer::write_contract(contract);
    let req = contract_request(handle, owned.view());
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::entry::pie_loader_verify_contract(plan, &req, &mut diags) };
    let message = all_messages(diags);
    unsafe { super::entry::pie_loader_release_diagnostics(diags) };
    unsafe { super::entry::pie_loader_close_checkpoint(handle) };
    (status, message)
}

/// Compile the fixture contract and hand the plan to `body`.
fn with_fixture_plan(body: impl FnOnce(*mut PieLoaderPlan)) {
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let owned = crate::contract_writer::write_contract(&fused_contract());
    let req = contract_request(handle, owned.view());
    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::entry::pie_loader_compile_contract(&req, &mut plan, &mut diags) };
    assert_eq!(
        status,
        PieLoaderStatus::Ok,
        "compile failed: {}",
        drain(diags)
    );
    body(plan);
    unsafe { super::entry::pie_loader_release(plan) };
    unsafe { super::entry::pie_loader_close_checkpoint(handle) };
}

/// The point of the whole exercise: a contract the loader did not author,
/// checked against the plan it did.
#[test]
fn the_contract_a_plan_was_compiled_from_verifies_against_it() {
    with_fixture_plan(|plan| {
        let (status, message) = verify_against(plan, &fused_contract());
        assert_eq!(status, PieLoaderStatus::Ok, "diagnostics: {message}");
    });
}

#[test]
fn a_contract_naming_a_tensor_the_plan_does_not_deliver_is_a_violation() {
    use crate::contract::{Expr, ModelContract, TensorContract};
    with_fixture_plan(|plan| {
        let mut other = fused_contract();
        other.tensors.push(TensorContract::new(
            "c.weight",
            Expr::Src("a.weight".to_string()),
            vec![2, 4],
            Encoding::Raw(DType::BF16),
        ));
        let (status, message) = verify_against(plan, &other);
        assert_eq!(status, PieLoaderStatus::ContractViolation);
        assert!(
            message.contains("c.weight") && message.contains("does not declare it"),
            "unexpected diagnostic: {message}"
        );

        // A contract may not be empty: one that declared nothing would compile
        // to a plan that loads nothing, which is never what a caller meant.
        let (status, message) = verify_against(
            plan,
            &ModelContract {
                abi_version: 1,
                alignment: 256,
                tensors: Vec::new(),
            },
        );
        assert_eq!(status, PieLoaderStatus::ContractViolation);
        assert!(message.contains("is empty"), "unexpected: {message}");
    });
}

/// The check that actually earns its keep. A pass that computes a shape the
/// driver disagrees with used to bind silently and produce garbage.
#[test]
fn a_declared_shape_that_disagrees_with_the_plan_is_a_violation() {
    with_fixture_plan(|plan| {
        let mut other = fused_contract();
        other.tensors[0].shape = Some(vec![8, 4]);
        let (status, message) = verify_against(plan, &other);
        assert_eq!(status, PieLoaderStatus::ContractViolation);
        assert!(
            message.contains("ab.weight") && message.contains('8'),
            "unexpected diagnostic: {message}"
        );

        // An unstated shape is "unstated", not "scalar": presence is still
        // demanded, the shape simply is not compared.
        let mut unstated = fused_contract();
        unstated.tensors[0].shape = None;
        let (status, message) = verify_against(plan, &unstated);
        assert_eq!(status, PieLoaderStatus::Ok, "diagnostics: {message}");
    });
}

#[test]
fn a_declared_encoding_that_disagrees_with_the_plan_is_a_violation() {
    with_fixture_plan(|plan| {
        let mut other = fused_contract();
        other.tensors[0].encoding = Encoding::Raw(DType::F32);
        let (status, message) = verify_against(plan, &other);
        assert_eq!(status, PieLoaderStatus::ContractViolation);
        assert!(message.contains("ab.weight"), "unexpected: {message}");
    });
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

// ---------------------------------------------------------------------------
// The contract entry point, end to end.
//
// `pie_loader_compile` is checked above against a request that names a model
// and lets the loader guess; these drive the path that has no model in it at
// all. What is being tested is the whole chain — open a real checkpoint, state
// a program over the tensors it reports, get a plan — because each half is
// already covered on its own and the failure mode that is left is the two
// halves disagreeing about what a name refers to.
// ---------------------------------------------------------------------------

/// Write a two-tensor safetensors file and return its directory.
///
/// A real file, not a sized placeholder: `pie_loader_open_checkpoint` parses
/// the header, so a fixture whose header did not exist would test nothing.
fn contract_fixture() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("pie_loader_contract_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create fixture directory");
    let path = dir.join("model.safetensors");
    // Two BF16 [2, 4] tensors, 16 bytes each, laid out back to back.
    let header = r#"{"a.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]},"b.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[16,32]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0u8; 32]);
    // Published by rename so a parallel reader never sees a partial header.
    let staging = dir.join(format!("model.{:?}.partial", std::thread::current().id()));
    std::fs::write(&staging, &bytes).expect("write fixture checkpoint");
    std::fs::rename(&staging, &path).expect("publish fixture checkpoint");
    dir
}

fn open_checkpoint(dir: &std::path::Path) -> *mut super::checkpoint::PieLoaderCheckpoint {
    let text = dir.to_string_lossy().into_owned();
    let mut handle = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status =
        unsafe { super::entry::pie_loader_open_checkpoint(bytes(&text), &mut handle, &mut diags) };
    let message = drain(diags);
    assert_eq!(status, PieLoaderStatus::Ok, "open failed: {message}");
    assert!(!handle.is_null());
    handle
}

/// Take the first diagnostic and release the array.
fn drain(diags: *mut PieLoaderDiagnostics) -> String {
    if diags.is_null() {
        return String::new();
    }
    let items = unsafe { std::slice::from_raw_parts((*diags).items, (*diags).len) };
    let first = items
        .first()
        .map(|d| {
            let raw = unsafe { std::slice::from_raw_parts(d.message.ptr, d.message.len) };
            String::from_utf8_lossy(raw).into_owned()
        })
        .unwrap_or_default();
    unsafe { super::entry::pie_loader_release_diagnostics(diags) };
    first
}

fn contract_request(
    checkpoint: *const super::checkpoint::PieLoaderCheckpoint,
    contract: super::contract::PieLoaderModelContractView,
) -> super::entry::PieLoaderContractRequest {
    super::entry::PieLoaderContractRequest {
        checkpoint,
        target: target_spec(),
        contract,
    }
}

/// The narrowest contract `plan_with_every_instr` satisfies.
///
/// A contract may not be empty — one that declared nothing would compile to a
/// plan that loads nothing — so "state as little as possible" is one tensor
/// with its shape left unstated.
fn minimal_contract() -> crate::contract::ModelContract {
    use crate::contract::{Expr, ModelContract, TensorContract};
    ModelContract {
        abi_version: 1,
        alignment: 256,
        tensors: vec![TensorContract::inferred(
            "model.norm.weight",
            Expr::Src("a.weight".to_string()),
            Encoding::Raw(DType::BF16),
        )],
    }
}

/// A contract that fuses the fixture's two tensors along axis 0.
fn fused_contract() -> crate::contract::ModelContract {
    use crate::contract::{Expr, ModelContract, TensorContract};
    ModelContract {
        abi_version: 1,
        alignment: 256,
        tensors: vec![TensorContract::new(
            "ab.weight",
            Expr::Cat {
                axis: Axis(0),
                parts: vec![
                    Expr::Src("a.weight".to_string()),
                    Expr::Src("b.weight".to_string()),
                ],
            },
            vec![4, 4],
            Encoding::Raw(DType::BF16),
        )],
    }
}

#[test]
fn an_opened_checkpoint_reports_the_tensors_a_contract_can_name() {
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let view = unsafe { &*handle };
    assert_eq!(view.files.len, 1);
    let tensors = unsafe { std::slice::from_raw_parts(view.tensors.ptr, view.tensors.len) };
    let names: Vec<String> = tensors
        .iter()
        .map(|t| {
            let raw = unsafe { std::slice::from_raw_parts(t.name.ptr, t.name.len) };
            String::from_utf8_lossy(raw).into_owned()
        })
        .collect();
    assert_eq!(names, vec!["a.weight".to_string(), "b.weight".to_string()]);
    let shape = unsafe { std::slice::from_raw_parts(tensors[0].shape.ptr, tensors[0].shape.len) };
    assert_eq!(shape, &[2, 4]);
    assert_eq!(tensors[0].span_bytes, 16);
    assert_eq!(tensors[1].file_offset - tensors[0].file_offset, 16);
    unsafe { super::entry::pie_loader_close_checkpoint(handle) };
}

#[test]
fn a_contract_compiles_and_verifies_without_naming_a_model() {
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let owned = crate::contract_writer::write_contract(&fused_contract());
    let req = contract_request(handle, owned.view());

    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::entry::pie_loader_compile_contract(&req, &mut plan, &mut diags) };
    assert_eq!(
        status,
        PieLoaderStatus::Ok,
        "compile failed: {}",
        drain(diags)
    );
    assert!(!plan.is_null());

    let declared = unsafe { std::slice::from_raw_parts((*plan).tensors.ptr, (*plan).tensors.len) };
    assert_eq!(declared.len(), 1);
    let name = unsafe { std::slice::from_raw_parts(declared[0].name.ptr, declared[0].name.len) };
    assert_eq!(std::str::from_utf8(name).unwrap(), "ab.weight");

    let mut vdiags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let verified = unsafe { super::entry::pie_loader_verify_contract(plan, &req, &mut vdiags) };
    assert_eq!(
        verified,
        PieLoaderStatus::Ok,
        "verify failed: {}",
        drain(vdiags)
    );

    unsafe { super::pie_loader_release(plan) };
    unsafe { super::entry::pie_loader_close_checkpoint(handle) };
}

#[test]
fn a_contract_naming_a_tensor_the_checkpoint_lacks_is_a_message_not_a_crash() {
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let mut model = fused_contract();
    model.tensors[0].expr = crate::contract::Expr::Src("missing.weight".to_string());
    let owned = crate::contract_writer::write_contract(&model);
    let req = contract_request(handle, owned.view());

    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::entry::pie_loader_compile_contract(&req, &mut plan, &mut diags) };
    assert_ne!(status, PieLoaderStatus::Ok);
    assert!(plan.is_null());
    assert!(
        drain(diags).contains("missing.weight"),
        "the message should name the tensor that is not there"
    );
    unsafe { super::entry::pie_loader_close_checkpoint(handle) };
}

#[test]
fn a_contract_whose_declared_shape_is_wrong_fails_to_compile() {
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let mut model = fused_contract();
    model.tensors[0].shape = Some(vec![8, 4]);
    let owned = crate::contract_writer::write_contract(&model);
    let req = contract_request(handle, owned.view());

    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::entry::pie_loader_compile_contract(&req, &mut plan, &mut diags) };
    assert_ne!(status, PieLoaderStatus::Ok);
    let message = drain(diags);
    assert!(
        message.contains("[8, 4]") && message.contains("[4, 4]"),
        "the message should show both the claim and the truth: {message}"
    );
    unsafe { super::entry::pie_loader_close_checkpoint(handle) };
}

#[test]
fn a_contract_request_with_no_checkpoint_is_rejected() {
    let owned = crate::contract_writer::write_contract(&fused_contract());
    let req = contract_request(std::ptr::null(), owned.view());
    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::entry::pie_loader_compile_contract(&req, &mut plan, &mut diags) };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(drain(diags).contains("checkpoint is null"));
}

#[test]
fn closing_a_null_checkpoint_is_a_no_op() {
    unsafe { super::entry::pie_loader_close_checkpoint(std::ptr::null_mut()) };
}
