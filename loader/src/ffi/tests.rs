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
use crate::plan::*;
use crate::types::*;

/// Bind an instruction's operands, asserting the operation it is.
///
/// The flat form let a test read `instr.transform_batch` on an `Allocate` and
/// see 0; here that does not compile, so a test that names the wrong operation
/// says so instead of quietly passing.
macro_rules! operands {
    ($instr:expr, $variant:ident { $($field:ident),* $(,)? }) => {
        match &$instr.op {
            PieLoaderStorageOp::$variant { $($field,)* .. } => ($($field,)*),
            other => panic!(
                "expected {}, found {other:?}",
                stringify!($variant)
            ),
        }
    };
}

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

fn stride(element_bytes: u32, dims: &[(i64, i64, i64)]) -> Extent {
    Extent {
        base_offset: 0,
        element_bytes,
        dims: dims
            .iter()
            .map(|(count, src_stride, dst_stride)| Dim {
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
        dtype: DType::BF16,
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
        }),
        alignment: 256,
        visibility: Default::default(),
    });
    plan.tensors.push(TensorDecl {
        id: TensorId(8),
        name: "model.norm.weight".to_string(),
        shape: vec![4096],
        encoding: Encoding::Raw(DType::BF16),
        alignment: 256,
        visibility: Default::default(),
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
        StorageInstr::TileMap {
            id: InstrId(3),
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
                repack: Some(RepackSpec {
                    layout: RepackLayout::MarlinMxfp4Weight,
                    batch: 2,
                    source_rows: 32,
                    target_rows: 64,
                    source_cols: 96,
                    target_cols: 112,
                }),
                scratch_bytes: 8192,
                fusion: TransformFusion::Fp8ToMxfp4,
                metadata_source: Some(TensorId(2)),
                scale_factor_bits: 0.5f32.to_bits(),
                scale_blocks: vec![1, 32],
            },
        },
        StorageInstr::CreateView {
            id: InstrId(5),
            input: BufferId(0),
            output: BufferId(13),
            view: dest_extent(13),
        },
        StorageInstr::Finalize {
            id: InstrId(8),
            tensor: BufferId(0),
            name: "model.layers.0.mlp.gate_proj.weight".to_string(),
        },
        // The second tensor decl needs a Finalize of its own: `verify` treats a
        // declared-but-never-finalized tensor as a weight the load would leave
        // absent, which is the coverage check `architecture.md` §8.2 asks for.
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
    let pod = arena::build(plan, arena::UNKEYED);
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
    let pod = arena::build(plan, arena::UNKEYED);
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let owned = crate::testkit::contract_writer::write_contract(&minimal_contract());
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
    // `architecture.md` §8.2's coverage question, in the half the loader can
    // answer without an externally-supplied contract: a declared tensor that
    // nothing produces is a weight the driver will look up and not find.
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
fn allocate_carries_only_a_buffer() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let instrs = unsafe { view::instrs(pod) };
        let (buffer_id,) = operands!(&instrs[0], Allocate { buffer_id });
        assert_eq!(*buffer_id, 0);
    });
}

#[test]
fn extent_write_carries_both_sides_unconditionally() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let write = &unsafe { view::instrs(pod) }[1];
        let (source, dest) = operands!(write, ExtentWrite { source, dest });
        assert_eq!(dest.buffer_id, 9);
        assert_eq!(source.file_id, 0);
        assert_eq!(source.tensor_id, 7);
        assert_eq!(source.span_bytes, 4096);
        assert_eq!(source.stride.element_bytes, 2);
        assert_eq!(
            unsafe { view::dims(&source.stride.dims) },
            &[PieLoaderDimSpecView {
                count: 4,
                src_stride: 8,
                dst_stride: 16,
            }]
        );
        assert_eq!(
            unsafe { view::dims(&dest.stride.dims) },
            &[PieLoaderDimSpecView {
                count: 4,
                src_stride: 16,
                dst_stride: 8,
            }]
        );
    });
}

/// A bulk write's destination is an arena offset, and now says so.
///
/// The flat form had no way to express that: every instruction carried a
/// `DestExtent`, so this one fabricated a rank-1 byte run against the sentinel
/// buffer — an arena allocation per instruction whose only field the executor
/// read was `offset`. Getting the sentinel wrong would have reinterpreted an
/// arena-relative write as buffer-relative, which is why that fabrication used
/// to need a test of its own.
#[test]
fn bulk_extent_write_carries_a_bare_arena_offset() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let bulk = &unsafe { view::instrs(pod) }[2];
        let (source, dest_offset) = operands!(
            bulk,
            BulkExtentWrite {
                source,
                dest_offset
            }
        );
        assert_eq!(*dest_offset, 4096);
        assert_eq!(source.span_bytes, 65536);
    });
}

#[test]
fn tile_map_carries_the_whole_transform() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let tile = &unsafe { view::instrs(pod) }[3];
        let (
            tile_kind,
            input_buffers,
            output_buffers,
            rows_per_tile,
            transform_fusion,
            transform_from,
            transform_to,
            repack_layout,
        ) = operands!(
            tile,
            TileMap {
                tile_kind,
                input_buffers,
                output_buffers,
                rows_per_tile,
                transform_fusion,
                transform_from,
                transform_to,
                repack_layout,
            }
        );
        assert_eq!(*tile_kind, PieLoaderTileMapKind::Repack);
        assert_eq!(unsafe { view::u32s(input_buffers) }, &[1, 2]);
        assert_eq!(unsafe { view::u32s(output_buffers) }, &[11, 12]);
        assert_eq!(*rows_per_tile, 64);
        assert_eq!(*transform_fusion, PieLoaderTransformFusion::Fp8ToMxfp4);
        assert_eq!(*transform_from, PieLoaderQuantScheme::MlxAffineU4);
        assert_eq!(*transform_to, PieLoaderQuantScheme::Mxfp4E2M1E8M0);
        assert_eq!(*repack_layout, PieLoaderRepackLayout::MarlinMxfp4Weight);

        let (batch, source_rows, target_rows) = operands!(
            tile,
            TileMap {
                transform_batch,
                transform_source_rows,
                transform_target_rows,
            }
        );
        assert_eq!((*batch, *source_rows, *target_rows), (2, 32, 64));

        let (cols, target_cols, scratch, metadata, factor) = operands!(
            tile,
            TileMap {
                transform_source_cols,
                transform_target_cols,
                transform_scratch_bytes,
                transform_metadata_source,
                transform_scale_factor_bits,
            }
        );
        assert_eq!((*cols, *target_cols), (96, 112));
        assert_eq!((*scratch, *metadata), (8192, 2));
        assert_eq!(f32::from_bits(*factor), 0.5);
    });
}

/// A tile map that transforms a buffer already on the device has no source, and
/// the flag is the only thing that says so.
///
/// This is the one optional operand the union keeps, because it is genuinely
/// optional for this operation and for no other. It used to be optional for all
/// eight, which is how `ffi::view` came to record a checkpoint read of file 0
/// for every device-side transform: the resting `SourceExtentView` has
/// `file_id == 0`, and nothing distinguished that from a real read of the first
/// file.
#[test]
fn a_device_side_tile_map_has_no_source() {
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
        let (has_source, has_dest, output_buffers, transform_from, transform_to) = operands!(
            tile,
            TileMap {
                has_source,
                has_dest,
                output_buffers,
                transform_from,
                transform_to,
            }
        );
        assert!(!has_source);
        assert!(!has_dest);
        assert_eq!(unsafe { view::u32s(output_buffers) }, &[] as &[u32]);
        assert_eq!(*transform_from, PieLoaderQuantScheme::None);
        assert_eq!(*transform_to, PieLoaderQuantScheme::None);
    });
}

#[test]
fn create_view_and_finalize_name_their_operands() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        let instrs = unsafe { view::instrs(pod) };

        let (input_buffer, output_buffer, view_extent) = operands!(
            &instrs[4],
            CreateView {
                input_buffer,
                output_buffer,
                view,
            }
        );
        assert_eq!(*input_buffer, 0);
        assert_eq!(*output_buffer, 13);
        assert_eq!(view_extent.buffer_id, 13);

        let (buffer_id, name) = operands!(&instrs[5], Finalize { buffer_id, name });
        assert_eq!(*buffer_id, 0);
        assert_eq!(
            unsafe { view::bytes(name) },
            "model.layers.0.mlp.gate_proj.weight"
        );
    });
}

#[test]
fn the_schedule_survives() {
    let plan = plan_with_every_instr();
    with_plan(&plan, |pod| {
        assert_eq!(unsafe { view::schedule(pod) }, &[0, 1, 2, 3, 4, 5, 6][..]);
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
            let (buffer_id, name) = operands!(instr, Finalize { buffer_id, name });
            assert_eq!(unsafe { view::bytes(name) }, format!("tensor.{i}"));
            assert_eq!(*buffer_id, i);
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
                let pod = arena::build(&plan, arena::UNKEYED);
                let count = unsafe { view::instrs(pod) }.len();
                (pod as usize, count)
            })
        })
        .collect();
    for handle in handles {
        let (pod, count) = handle.join().unwrap();
        assert_eq!(count, 7);
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
// rejection is the thing worth testing. Since `architecture.md` §12 row 12 the request carries no
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
    let owned = crate::testkit::contract_writer::write_contract(&fused_contract());
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
    // The driver is the authority on what its kernels implement
    // (`architecture.md` §9), so it may claim fewer transforms than the loader
    // knows how to lower. It may not claim one the loader has never heard of:
    // nothing downstream would notice until a kernel dispatch failed at load
    // time.
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
    // `architecture.md` §9: a device constant the loader cannot measure has no
    // safe default. The number decides how much scratch every Encode allocates,
    // so guessing it would be a performance contract nobody signed.
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
    let owned = crate::testkit::contract_writer::write_contract(&fused_contract());
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
    // wrote (`architecture.md` §8.1).
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let owned = crate::testkit::contract_writer::write_contract(&fused_contract());
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
/// `architecture.md` §6.2 one — two ranks, or a cache hit and a fresh compile,
/// holding different programs and each believing the other's plan.
fn verify_against(
    plan: *const PieLoaderPlan,
    contract: &crate::contract::ModelContract,
) -> (PieLoaderStatus, String) {
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let owned = crate::testkit::contract_writer::write_contract(contract);
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
    let owned = crate::testkit::contract_writer::write_contract(&fused_contract());
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
                alignment: 256,
                tensors: Vec::new(),
                groups: Vec::new(),
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
        DType::I64,
        DType::U64,
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
        alignment: 256,
        tensors: vec![TensorContract::inferred(
            "model.norm.weight",
            Expr::Src("a.weight".to_string()),
            Encoding::Raw(DType::BF16),
        )],
        groups: Vec::new(),
    }
}

/// A contract that fuses the fixture's two tensors along axis 0.
fn fused_contract() -> crate::contract::ModelContract {
    use crate::contract::{Expr, ModelContract, TensorContract};
    ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "ab.weight",
            Expr::Concat {
                axis: Axis(0),
                parts: vec![
                    Expr::Src("a.weight".to_string()),
                    Expr::Src("b.weight".to_string()),
                ],
            },
            vec![4, 4],
            Encoding::Raw(DType::BF16),
        )],
        groups: Vec::new(),
    }
}

/// A contract that publishes a weight and declares a second tensor as its
/// scales, so the `scales` half of the contract ABI has something to carry.
fn scaled_contract() -> crate::contract::ModelContract {
    use crate::contract::{Expr, ModelContract, Scales, TensorContract};
    use crate::types::{QuantGranularity, ScaleForm};
    ModelContract {
        alignment: 256,
        tensors: vec![
            TensorContract::new(
                "a.weight",
                Expr::Src("a.weight".to_string()),
                vec![2, 4],
                Encoding::Raw(DType::BF16),
            ),
            TensorContract::new(
                "a.scale",
                Expr::Src("b.weight".to_string()),
                vec![2, 4],
                Encoding::Raw(DType::BF16),
            )
            .scaling(Scales {
                of: "a.weight".to_string(),
                granularity: QuantGranularity::PerChannel,
                group_size: 0,
                channel_axis: 0,
                form: ScaleForm::F32Factors,
            }),
        ],
        groups: Vec::new(),
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
    let owned = crate::testkit::contract_writer::write_contract(&fused_contract());
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
    let owned = crate::testkit::contract_writer::write_contract(&model);
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
    let owned = crate::testkit::contract_writer::write_contract(&model);
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

/// A driver-written enum field is read as an integer and validated, never
/// transmuted.
///
/// `PieLoaderScalesView` sits inside `PieLoaderModelContractView`, which the
/// *driver* fills in. A Rust enum holding a value outside its variants is
/// undefined behaviour the moment `read_contract` borrows the tensor slice —
/// before any `match` could reject it — which is why every input-side enum on
/// this boundary is spelled `u32`. This is the check that it still is.
#[test]
fn an_out_of_range_scale_granularity_is_rejected_not_transmuted() {
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let mut owned = crate::testkit::contract_writer::write_contract(&scaled_contract());
    let scaled = owned.first_scaled().expect("the fixture declares scales");
    owned.set_raw_scale_codes(scaled, 9, 1);
    let req = contract_request(handle, owned.view());

    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::entry::pie_loader_compile_contract(&req, &mut plan, &mut diags) };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(plan.is_null());
    let message = drain(diags);
    assert!(
        message.contains("granularity") && message.contains('9'),
        "the message should name the field and the value: {message}"
    );
    unsafe { super::entry::pie_loader_close_checkpoint(handle) };
}

#[test]
fn an_out_of_range_scale_form_is_rejected_not_transmuted() {
    let dir = contract_fixture();
    let handle = open_checkpoint(&dir);
    let mut owned = crate::testkit::contract_writer::write_contract(&scaled_contract());
    let scaled = owned.first_scaled().expect("the fixture declares scales");
    owned.set_raw_scale_codes(scaled, 0, 4);
    let req = contract_request(handle, owned.view());

    let mut plan: *mut PieLoaderPlan = std::ptr::null_mut();
    let mut diags: *mut PieLoaderDiagnostics = std::ptr::null_mut();
    let status = unsafe { super::entry::pie_loader_compile_contract(&req, &mut plan, &mut diags) };
    assert_eq!(status, PieLoaderStatus::InvalidRequest);
    assert!(plan.is_null());
    let message = drain(diags);
    assert!(
        message.contains("form") && message.contains('4'),
        "the message should name the field and the value: {message}"
    );
    unsafe { super::entry::pie_loader_close_checkpoint(handle) };
}

#[test]
fn a_contract_request_with_no_checkpoint_is_rejected() {
    let owned = crate::testkit::contract_writer::write_contract(&fused_contract());
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

/// The mirror enums exist to pin the C discriminants, and nothing checked that.
///
/// `PieLoaderDType` is not `DType` written twice. It is `DType`'s *wire
/// numbering*, held still while the Rust enum is free to be reordered for
/// reading. `766e9b029` is the proof: it inserted `E8M0` at position 5 of
/// `DType`, next to the other float formats where it belongs, and appended it
/// at 12 here, because 5 through 11 were already spoken for by drivers. The
/// `From` impls map by *name*, so that divergence is not a bug; it is the
/// mechanism.
///
/// The compiler already refuses a variant added to the Rust enum and not to the
/// mirror — the `From` match stops being exhaustive. What it cannot see is
/// someone tidying the mirror to match the original, which renumbers seven
/// dtypes under every driver built against the old header and compiles
/// perfectly. These assertions are the numbering itself, written down. A
/// variant appended with the next free number needs a line added here; an
/// existing line that has to *change* means the ABI broke.
#[test]
fn mirror_enum_discriminants_are_pinned() {
    // PieLoaderBackendKind
    assert_eq!(PieLoaderBackendKind::Cuda as u32, 0);
    assert_eq!(PieLoaderBackendKind::Metal as u32, 1);
    assert_eq!(PieLoaderBackendKind::Unknown as u32, 255);

    // PieLoaderDType
    assert_eq!(PieLoaderDType::F32 as u32, 0);
    assert_eq!(PieLoaderDType::F16 as u32, 1);
    assert_eq!(PieLoaderDType::BF16 as u32, 2);
    assert_eq!(PieLoaderDType::F8E4M3 as u32, 3);
    assert_eq!(PieLoaderDType::F8E5M2 as u32, 4);
    assert_eq!(PieLoaderDType::I32 as u32, 5);
    assert_eq!(PieLoaderDType::I16 as u32, 6);
    assert_eq!(PieLoaderDType::I8 as u32, 7);
    assert_eq!(PieLoaderDType::U32 as u32, 8);
    assert_eq!(PieLoaderDType::U16 as u32, 9);
    assert_eq!(PieLoaderDType::U8 as u32, 10);
    assert_eq!(PieLoaderDType::Bool as u32, 11);
    assert_eq!(PieLoaderDType::E8M0 as u32, 12);
    assert_eq!(PieLoaderDType::I64 as u32, 13);
    assert_eq!(PieLoaderDType::U64 as u32, 14);

    // PieLoaderEncodingKind
    assert_eq!(PieLoaderEncodingKind::Raw as u32, 0);
    assert_eq!(PieLoaderEncodingKind::Quant as u32, 1);

    // PieLoaderQuantScheme
    assert_eq!(PieLoaderQuantScheme::None as u32, 0);
    assert_eq!(PieLoaderQuantScheme::Fp8E4M3 as u32, 1);
    assert_eq!(PieLoaderQuantScheme::Fp8E5M2 as u32, 2);
    assert_eq!(PieLoaderQuantScheme::Int8Symmetric as u32, 3);
    assert_eq!(PieLoaderQuantScheme::Int8Asymmetric as u32, 4);
    assert_eq!(PieLoaderQuantScheme::AwqInt4 as u32, 5);
    assert_eq!(PieLoaderQuantScheme::GptqInt4 as u32, 6);
    assert_eq!(PieLoaderQuantScheme::Mxfp4E2M1E8M0 as u32, 7);
    assert_eq!(PieLoaderQuantScheme::MlxAffineU4 as u32, 8);
    assert_eq!(PieLoaderQuantScheme::GgufQ4_0 as u32, 9);
    assert_eq!(PieLoaderQuantScheme::GgufQ4K as u32, 10);
    assert_eq!(PieLoaderQuantScheme::GgufQ5_0 as u32, 11);
    assert_eq!(PieLoaderQuantScheme::GgufQ5K as u32, 12);
    assert_eq!(PieLoaderQuantScheme::GgufQ8_0 as u32, 13);

    // PieLoaderRepackLayout
    assert_eq!(PieLoaderRepackLayout::None as u32, 0);
    assert_eq!(PieLoaderRepackLayout::MarlinMxfp4Weight as u32, 1);
    assert_eq!(PieLoaderRepackLayout::MarlinMxfp4Scale as u32, 2);

    // PieLoaderTileMapKind
    assert_eq!(PieLoaderTileMapKind::Cast as u32, 0);
    assert_eq!(PieLoaderTileMapKind::Decode as u32, 1);
    assert_eq!(PieLoaderTileMapKind::Encode as u32, 2);
    assert_eq!(PieLoaderTileMapKind::Transcode as u32, 3);
    assert_eq!(PieLoaderTileMapKind::Reblock as u32, 4);
    assert_eq!(PieLoaderTileMapKind::Repack as u32, 6);
    assert_eq!(PieLoaderTileMapKind::None as u32, 7);

    // PieLoaderTransformFusion
    assert_eq!(PieLoaderTransformFusion::None as u32, 0);
    assert_eq!(PieLoaderTransformFusion::Fp8ToMxfp4 as u32, 1);

    // PieLoaderQuantGranularity
    assert_eq!(PieLoaderQuantGranularity::PerChannel as u32, 0);
    assert_eq!(PieLoaderQuantGranularity::PerGroup as u32, 1);

    // PieLoaderScaleForm
    assert_eq!(PieLoaderScaleForm::RawE8M0 as u32, 0);
    assert_eq!(PieLoaderScaleForm::F32Factors as u32, 1);

    // PieLoaderCheckpointFormat
    assert_eq!(PieLoaderCheckpointFormat::Safetensors as u32, 0);
    assert_eq!(PieLoaderCheckpointFormat::Gguf as u32, 1);
    assert_eq!(PieLoaderCheckpointFormat::Unknown as u32, 2);
    assert_eq!(PieLoaderCheckpointFormat::Zt as u32, 3);
    assert_eq!(PieLoaderCheckpointFormat::Npz as u32, 4);
    assert_eq!(PieLoaderCheckpointFormat::Pt as u32, 5);
    assert_eq!(PieLoaderCheckpointFormat::Hdf5 as u32, 6);
    assert_eq!(PieLoaderCheckpointFormat::Onnx as u32, 7);
}

/// The operation tag is the first four bytes of `PieLoaderStorageOp`, and this
/// reads them rather than a second enum that claims to agree.
///
/// The old flat form had a separate `PieLoaderStorageInstrKind`, so a test could
/// only assert that *it* was numbered as intended and trust the marshaller to
/// write it. Here the numbering and the wire value are the same thing, so the
/// assertion is against the bytes a driver will actually switch on.
///
/// 4 is missing on purpose. A retired instruction had it, and drivers built
/// against a header that said `Finalize = 5` are the reason it stays missing.
#[test]
fn storage_op_tags_are_the_wire_values() {
    fn tag(op: PieLoaderStorageOp) -> u32 {
        // `#[repr(C, u32)]` puts the discriminant first, which is the whole
        // reason the driver may switch on it.
        unsafe { *(&raw const op).cast::<u32>() }
    }
    use PieLoaderStorageOp as Op;

    let source = PieLoaderSourceExtentView::default();
    let dest = PieLoaderDestExtentView::default();
    assert_eq!(tag(Op::Allocate { buffer_id: 0 }), 0);
    assert_eq!(tag(Op::ExtentWrite { source, dest }), 1);
    assert_eq!(
        tag(Op::TileMap {
            tile_kind: PieLoaderTileMapKind::None,
            source,
            has_source: false,
            dest,
            has_dest: false,
            input_buffers: PieLoaderU32Slice::default(),
            output_buffers: PieLoaderU32Slice::default(),
            rows_per_tile: 0,
            transform_fusion: PieLoaderTransformFusion::None,
            transform_from: PieLoaderQuantScheme::None,
            transform_to: PieLoaderQuantScheme::None,
            repack_layout: PieLoaderRepackLayout::None,
            transform_batch: 0,
            transform_source_rows: 0,
            transform_target_rows: 0,
            transform_source_cols: 0,
            transform_target_cols: 0,
            transform_scratch_bytes: 0,
            transform_metadata_source: PIE_LOADER_NO_TENSOR,
            transform_scale_factor_bits: 0,
            transform_scale_blocks: PieLoaderI64Slice::default(),
        }),
        2
    );
    assert_eq!(
        tag(Op::CreateView {
            input_buffer: 0,
            output_buffer: 0,
            view: dest,
        }),
        3
    );
    assert_eq!(
        tag(Op::Finalize {
            buffer_id: 0,
            name: PieLoaderBytes::default(),
        }),
        5
    );
    assert_eq!(
        tag(Op::BulkExtentWrite {
            source,
            dest_offset: 0,
        }),
        6
    );
    assert_eq!(tag(Op::Fill { buffer_id: 0 }), 8);
}

/// A `Gather`'s indices are the first variable-length payload on an expression
/// node that is neither a shape nor a child list, so they are the one field
/// whose round trip nothing else would exercise.
#[test]
fn a_gather_carries_its_indices_across_the_ffi() {
    use crate::contract::{Expr, ModelContract, TensorContract};
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::inferred(
            "permuted",
            Expr::src("a.weight").gather(0, vec![3, 0, 7, 0]),
            Encoding::Raw(DType::BF16),
        )],
        groups: Vec::new(),
    };
    let owned = crate::testkit::contract_writer::write_contract(&contract);
    let read = unsafe { super::contract::read_contract(&owned.view()) }.expect("reads back");
    assert_eq!(read, contract);
}

/// A group is the one thing on the contract ABI that is not a flat list: the
/// C++ side keeps its declarations in storage of its own, and the flattened
/// form has to say which of the shared tensor entries belong to which group.
/// Both index nodes ride along, `Select`'s stride in the `start` field, so a
/// whole-contract equality is the check that pins that encoding down.
#[test]
fn a_group_survives_the_contract_ffi() {
    use crate::contract::{Expr, GroupContract, ModelContract, TensorContract};
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::inferred(
            "norm",
            Expr::src("model.norm.weight"),
            Encoding::Raw(DType::BF16),
        )],
        groups: vec![
            GroupContract {
                name: "experts".to_string(),
                arity: 128,
                tensors: vec![
                    TensorContract::inferred(
                        "gate_up",
                        Expr::select(Expr::src("experts.gate_up_blocks"), 0, 1, 1),
                        Encoding::Raw(DType::BF16),
                    ),
                    TensorContract::inferred(
                        "down",
                        Expr::src_indexed("model.layers.0.experts.{}.down.weight"),
                        Encoding::Raw(DType::BF16),
                    ),
                ],
            },
            GroupContract {
                name: "layers".to_string(),
                arity: 4,
                tensors: vec![TensorContract::inferred(
                    "attn",
                    Expr::src_indexed("model.layers.{}.attn.weight"),
                    Encoding::Raw(DType::BF16),
                )],
            },
        ],
    };
    let owned = crate::testkit::contract_writer::write_contract(&contract);
    let read = unsafe { super::contract::read_contract(&owned.view()) }.expect("reads back");
    assert_eq!(read, contract);
}

/// The rule `RepackLayout` used to state as a variant, now stated where it can
/// still be violated.
///
/// Zero is what an all-zero node carries, and a `Repack` that names no kernel
/// would otherwise reach the device as a transform that does nothing to bytes
/// a GEMM is about to read as swizzled.
#[test]
fn a_repack_that_names_no_kernel_is_refused_at_the_boundary() {
    use crate::contract::{Expr, ModelContract, TensorContract, TensorType};
    use crate::types::RepackLayout;
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::inferred(
            "packed",
            Expr::src("blocks").repack(
                RepackLayout::MarlinMxfp4Weight,
                TensorType::raw(vec![2, 32, 64], DType::BF16),
            ),
            Encoding::Raw(DType::BF16),
        )],
        groups: Vec::new(),
    };
    let mut owned = crate::testkit::contract_writer::write_contract(&contract);
    let node = owned.first_repack().expect("the contract has a Repack");
    owned.set_raw_repack_layout(node, PieLoaderRepackLayout::None as u32);
    let err = unsafe { super::contract::read_contract(&owned.view()) }.unwrap_err();
    assert!(err.contains("names a kernel"), "{err}");
}
