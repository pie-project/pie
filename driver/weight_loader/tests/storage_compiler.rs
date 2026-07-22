use pie_weight_loader::ir::{LayoutExpr, LayoutPlan};
use pie_weight_loader::source::{CheckpointFile, CheckpointMetadata, RawTensor};
use pie_weight_loader::storage::{StorageInstr, StorageTarget, TileMapKind};
use pie_weight_loader::storage_compiler::{compile_storage_program, lower_layout_plan};
use pie_weight_loader::types::{
    Axis, BackendKind, CheckpointFormat, DType, Encoding, FileId, Layout, Mxfp4MoePolicy,
    QuantScheme, QuantSpec, RepackLayout, RowMap, Sharding, TensorDecl, TensorId,
};

#[test]
fn buffer_join_tile_maps_carry_destination_offsets() {
    let mut plan = LayoutPlan::new();
    let a_decl = decl(0, "a", &[2], Encoding::Raw(DType::F32));
    let b_decl = decl(1, "b", &[2], Encoding::Raw(DType::F32));
    let a = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: a_decl.clone(),
    });
    let b = plan.push(LayoutExpr::Source {
        tensor: TensorId(1),
        decl: b_decl.clone(),
    });
    let a_cast_decl = decl(2, "a.cast", &[2], Encoding::Raw(DType::BF16));
    let b_cast_decl = decl(3, "b.cast", &[2], Encoding::Raw(DType::BF16));
    let a_cast = plan.push(LayoutExpr::Cast {
        input: a,
        dtype: DType::BF16,
        decl: a_cast_decl,
    });
    let b_cast = plan.push(LayoutExpr::Cast {
        input: b,
        dtype: DType::BF16,
        decl: b_cast_decl,
    });
    let joined_decl = decl(4, "joined", &[4], Encoding::Raw(DType::BF16));
    let joined = plan.push(LayoutExpr::Join {
        inputs: vec![a_cast, b_cast],
        axis: pie_weight_loader::types::Axis(0),
        decl: joined_decl.clone(),
    });
    let realized = plan.push(LayoutExpr::Realize {
        input: joined,
        runtime_name: "joined".to_string(),
        decl: joined_decl,
    });
    plan.outputs.push(realized);

    let program = lower_layout_plan(&metadata(), &plan, StorageTarget::default()).unwrap();
    let reblocks: Vec<_> = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap {
                kind: TileMapKind::Reblock,
                dest,
                ..
            } => dest.as_ref(),
            _ => None,
        })
        .collect();

    assert_eq!(reblocks.len(), 2);
    assert_eq!(reblocks[0].offset, 0);
    assert_eq!(reblocks[1].offset, 4);
    assert_eq!(reblocks[0].stride.element_bytes, 2);
    assert_eq!(reblocks[1].stride.element_bytes, 2);
    assert_eq!(program.memory.device_write_bytes, 16);
    assert_eq!(program.memory.persistent_bytes, 8);
    assert_eq!(program.memory.temporary_peak_bytes, 8);
}

#[test]
fn direct_copy_lowers_to_identity_extent_write() {
    let mut plan = LayoutPlan::new();
    let runtime_decl = decl(7, "runtime.weight", &[2, 2], Encoding::Raw(DType::BF16));
    let source_decl = decl(0, "checkpoint.weight", &[2, 2], Encoding::Raw(DType::BF16));
    let source = plan.push(LayoutExpr::Source {
        tensor: TensorId(6),
        decl: source_decl,
    });
    let realized = plan.push(LayoutExpr::Realize {
        input: source,
        runtime_name: "runtime.weight".to_string(),
        decl: runtime_decl,
    });
    plan.outputs.push(realized);

    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1024,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![RawTensor {
            id: TensorId(6),
            name: "checkpoint.weight".to_string(),
            file_id: FileId(0),
            file_offset: 512,
            span_bytes: 8,
            shape: vec![2, 2],
            encoding: Encoding::Raw(DType::BF16),
            layout: Layout::dense(1),
        }],
    };

    let program = lower_layout_plan(&metadata, &plan, StorageTarget::default()).unwrap();
    let writes: Vec<_> = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::ExtentWrite { id, source, dest } => {
                Some((id, source, dest.offset))
            }
            StorageInstr::BulkExtentWrite {
                id,
                source,
                dest_offset,
            } => Some((id, source, *dest_offset)),
            _ => None,
        })
        .collect();
    assert_eq!(writes.len(), 1);
    let (write_id, source, dest) = writes[0];
    assert_eq!(source.tensor_id, TensorId(6));
    assert_eq!(source.file_offset, 512);
    assert_eq!(source.span_bytes, 8);
    assert_eq!(dest, 0);
    assert_eq!(
        program.schedule,
        program.instrs.iter().map(instr_id).collect::<Vec<_>>()
    );
    assert!(program.schedule.contains(write_id));
    assert_eq!(program.memory.checkpoint_read_bytes, 8);
    assert_eq!(program.memory.device_write_bytes, 8);
    assert_eq!(program.memory.persistent_bytes, 8);
}

#[test]
fn packed_quant_row_select_uses_byte_exact_offsets() {
    let mut plan = LayoutPlan::new();
    let q_decl = decl(
        0,
        "q",
        &[4, 8],
        Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
    );
    let q = plan.push(LayoutExpr::Source {
        tensor: TensorId(2),
        decl: q_decl,
    });
    let selected_decl = decl(
        1,
        "q.row",
        &[1, 8],
        Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
    );
    let selected = plan.push(LayoutExpr::Select {
        input: q,
        axis: Axis(0),
        start: 2,
        length: 1,
        decl: selected_decl.clone(),
    });
    let realized = plan.push(LayoutExpr::Realize {
        input: selected,
        runtime_name: "q.row".to_string(),
        decl: selected_decl,
    });
    plan.outputs.push(realized);

    let program = lower_layout_plan(&quant_metadata(), &plan, StorageTarget::default()).unwrap();
    let write = program
        .instrs
        .iter()
        .find_map(|instr| match instr {
            StorageInstr::ExtentWrite { source, .. } => Some(source),
            StorageInstr::BulkExtentWrite { source, .. } => Some(source),
            _ => None,
        })
        .unwrap();
    assert_eq!(write.file_offset, 200 + 8);
    assert_eq!(write.span_bytes, 4);
    assert_eq!(program.memory.persistent_bytes, 4);
    assert_eq!(program.memory.device_write_bytes, 4);
}

#[test]
fn packed_quant_inner_select_must_be_byte_aligned() {
    let mut plan = LayoutPlan::new();
    let q_decl = decl(
        0,
        "q",
        &[4, 7],
        Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
    );
    let q = plan.push(LayoutExpr::Source {
        tensor: TensorId(4),
        decl: q_decl,
    });
    let selected_decl = decl(
        1,
        "q.bad",
        &[4, 1],
        Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
    );
    let selected = plan.push(LayoutExpr::Select {
        input: q,
        axis: Axis(1),
        start: 1,
        length: 1,
        decl: selected_decl.clone(),
    });
    let realized = plan.push(LayoutExpr::Realize {
        input: selected,
        runtime_name: "q.bad".to_string(),
        decl: selected_decl,
    });
    plan.outputs.push(realized);

    let err = lower_layout_plan(&quant_metadata(), &plan, StorageTarget::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("byte-aligned"));
}

#[test]
fn target_support_rejects_cuda_decode_at_compile_time() {
    let mut plan = LayoutPlan::new();
    let q_decl = decl(
        0,
        "q",
        &[4],
        Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::BF16)),
    );
    let q = plan.push(LayoutExpr::Source {
        tensor: TensorId(3),
        decl: q_decl,
    });
    let decoded_decl = decl(1, "decoded", &[4], Encoding::Raw(DType::BF16));
    let decoded = plan.push(LayoutExpr::Decode {
        scheme: QuantScheme::Fp8E4M3,
        data: q,
        metadata: Vec::new(),
        decl: decoded_decl.clone(),
    });
    let realized = plan.push(LayoutExpr::Realize {
        input: decoded,
        runtime_name: "decoded".to_string(),
        decl: decoded_decl,
    });
    plan.outputs.push(realized);

    let err = lower_layout_plan(
        &quant_metadata(),
        &plan,
        StorageTarget {
            backend: BackendKind::Cuda,
            ..StorageTarget::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("does not support Decode"));
}

#[test]
fn packed_quant_source_requires_exact_affine_size() {
    let mut plan = LayoutPlan::new();
    let q_decl = decl(
        0,
        "blocked",
        &[4, 8],
        Encoding::Quant(quant(QuantScheme::GgufQ4_0, DType::BF16)),
    );
    let q = plan.push(LayoutExpr::Source {
        tensor: TensorId(5),
        decl: q_decl.clone(),
    });
    let realized = plan.push(LayoutExpr::Realize {
        input: q,
        runtime_name: "blocked".to_string(),
        decl: q_decl,
    });
    plan.outputs.push(realized);

    let mut metadata = quant_metadata();
    metadata.tensors.push(RawTensor {
        id: TensorId(5),
        name: "blocked".to_string(),
        file_id: FileId(0),
        file_offset: 240,
        span_bytes: 32,
        shape: vec![4, 8],
        encoding: Encoding::Quant(quant(QuantScheme::GgufQ4_0, DType::BF16)),
        layout: Layout::dense(1),
    });

    let err = lower_layout_plan(&metadata, &plan, StorageTarget::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("non-affine physical size"));
}

#[test]
fn gpt_oss_native_mxfp4_default_abi_lowers_to_repack_tile_maps() {
    let cfg = pie_weight_loader::config::ModelConfig {
        model_type: "gpt_oss".to_string(),
        quant_method: String::new(),
        runtime_quant: String::new(),
        num_hidden_layers: 1,
        num_experts: 2,
        num_experts_per_tok: 2,
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        mxfp4_moe: Mxfp4MoePolicy::NativeGemm,
        native_mxfp4_moe: true,
        ..StorageTarget::default()
    };
    let metadata = gpt_oss_mxfp4_metadata();
    let abi =
        pie_weight_loader::abi::RuntimeAbi::default_for_target(&metadata, &cfg, &target).unwrap();
    let program = compile_storage_program(&metadata, &cfg, &abi, target).unwrap();

    let repacks: Vec<_> = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap {
                kind: TileMapKind::Repack,
                transform,
                ..
            } => Some(transform),
            _ => None,
        })
        .collect();
    assert_eq!(repacks.len(), 8);
    assert!(
        repacks
            .iter()
            .any(|spec| spec.repack.layout == RepackLayout::MarlinMxfp4Weight)
    );
    assert!(
        repacks
            .iter()
            .any(|spec| spec.repack.layout == RepackLayout::MarlinMxfp4Scale)
    );
    assert!(
        repacks
            .iter()
            .any(|spec| spec.repack.layout == RepackLayout::DenseRowGather)
    );
    let names = program
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<Vec<_>>();
    assert!(names.contains(&"model.layers.0.mlp.experts.gate_proj.weight"));
    assert!(names.contains(&"model.layers.0.mlp.experts.up_proj.weight"));
    assert!(names.contains(&"model.layers.0.mlp.experts.down_proj.weight"));
    assert!(!names.contains(&"model.layers.0.mlp.experts.gate_up_proj.weight"));
    assert!(program.memory.transform_scratch_peak_bytes > 0);
}

#[test]
fn gpt_oss_native_mxfp4_tp_uses_row_and_column_offset_repack_contracts() {
    let cfg = pie_weight_loader::config::ModelConfig {
        model_type: "gpt_oss".to_string(),
        quant_method: String::new(),
        runtime_quant: String::new(),
        num_hidden_layers: 1,
        num_experts: 2,
        num_experts_per_tok: 2,
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tp_rank: 1,
        tp_size: 2,
        mxfp4_moe: Mxfp4MoePolicy::NativeGemm,
        native_mxfp4_moe: true,
        ..StorageTarget::default()
    };
    let metadata = gpt_oss_mxfp4_metadata_with_intermediate(128);
    let abi =
        pie_weight_loader::abi::RuntimeAbi::default_for_target(&metadata, &cfg, &target).unwrap();
    let abi_repacks = abi
        .tensors
        .iter()
        .filter_map(|contract| match contract.source {
            pie_weight_loader::abi::RuntimeTensorSource::Repack { spec, .. } => Some(spec),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        abi_repacks
            .iter()
            .any(|spec| spec.layout == RepackLayout::MarlinMxfp4Weight
                && spec.row_map == RowMap::Even
                && spec.source_row_offset == 64
                && spec.valid_rows == 64
                && spec.source_stride_cols == 64
                && spec.source_col_offset == 0)
    );
    assert!(
        abi_repacks
            .iter()
            .any(|spec| spec.layout == RepackLayout::MarlinMxfp4Weight
                && spec.row_map == RowMap::Identity
                && spec.source_col_offset == 64
                && spec.source_stride_cols == 128
                && spec.source_cols == 64)
    );

    let program = compile_storage_program(&metadata, &cfg, &abi, target).unwrap();

    let repacks = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap {
                kind: TileMapKind::Repack,
                transform,
                ..
            } => Some(transform.repack),
            _ => None,
        })
        .collect::<Vec<_>>();
    let repack_sources = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap {
                kind: TileMapKind::Repack,
                source,
                transform,
                ..
            } => source.as_ref().map(|source| (transform.repack, source)),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(repacks.len(), 8);
    assert!(
        repacks
            .iter()
            .any(|spec| spec.layout == RepackLayout::MarlinMxfp4Weight
                && spec.row_map == RowMap::Even
                && spec.source_rows == 128
                && spec.source_row_offset == 0
                && spec.valid_rows == 64
                && spec.target_rows == 128
                && spec.source_stride_cols == 64
                && spec.source_col_offset == 0
                && spec.source_cols == 64
                && spec.target_cols == 64)
    );
    assert!(
        repacks
            .iter()
            .any(|spec| spec.layout == RepackLayout::DenseRowGather
                && spec.row_map == RowMap::Odd
                && spec.source_rows == 128
                && spec.source_row_offset == 0
                && spec.valid_rows == 64
                && spec.target_rows == 64)
    );
    assert!(
        repacks
            .iter()
            .any(|spec| spec.layout == RepackLayout::MarlinMxfp4Weight
                && spec.row_map == RowMap::Identity
                && spec.source_col_offset == 0
                && spec.source_stride_cols == 64
                && spec.source_cols == 64
                && spec.target_cols == 128)
    );
    assert!(
        repacks
            .iter()
            .any(|spec| spec.layout == RepackLayout::MarlinMxfp4Scale
                && spec.row_map == RowMap::Identity
                && spec.source_col_offset == 0
                && spec.source_stride_cols == 2
                && spec.source_cols == 2
                && spec.target_cols == 4)
    );
    assert!(repack_sources.iter().any(|(spec, source)| spec.layout
        == RepackLayout::MarlinMxfp4Weight
        && spec.row_map == RowMap::Even
        && source.span_bytes == 8192));
    assert!(repack_sources.iter().any(|(spec, source)| spec.layout
        == RepackLayout::MarlinMxfp4Weight
        && spec.row_map == RowMap::Identity
        && source.span_bytes == 4096));
    assert_eq!(program.memory.checkpoint_read_bytes, 23_040);
}

#[test]
fn nemotron_h_default_abi_packs_experts_and_exposes_views() {
    let cfg = pie_weight_loader::config::ModelConfig {
        model_type: "nemotron_h".to_string(),
        quant_method: String::new(),
        runtime_quant: String::new(),
        num_hidden_layers: 1,
        num_experts: 2,
        num_experts_per_tok: 2,
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tp_rank: 1,
        tp_size: 2,
        ..StorageTarget::default()
    };
    let metadata = nemotron_h_expert_metadata();
    let abi =
        pie_weight_loader::abi::RuntimeAbi::default_for_target(&metadata, &cfg, &target).unwrap();

    assert!(abi.tensors.iter().any(|contract| {
        contract.output_name
            == "language_model.backbone.layers.0.mixer.experts.up_proj.packed.weight"
            && contract.shape == vec![4, 3]
    }));
    assert!(abi.tensors.iter().any(|contract| {
        contract.output_name
            == "language_model.backbone.layers.0.mixer.experts.down_proj.packed.weight"
            && contract.shape == vec![6, 4]
            && contract.shard_axis == Some(Axis(1))
    }));
    assert!(abi.tensors.iter().any(|contract| {
        contract.output_name == "language_model.backbone.layers.0.mixer.experts.0.up_proj.weight"
            && contract.shape == vec![2, 3]
    }));
    assert!(abi.tensors.iter().any(|contract| {
        contract.output_name == "language_model.backbone.layers.0.mixer.experts.1.down_proj.weight"
            && contract.shape == vec![3, 2]
    }));

    let program = compile_storage_program(&metadata, &cfg, &abi, target).unwrap();
    let names = program
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<Vec<_>>();
    assert!(
        names.contains(&"language_model.backbone.layers.0.mixer.experts.up_proj.packed.weight")
    );
    assert!(
        names.contains(&"language_model.backbone.layers.0.mixer.experts.down_proj.packed.weight")
    );
    assert!(names.contains(&"language_model.backbone.layers.0.mixer.experts.0.up_proj.weight"));
    assert!(names.contains(&"language_model.backbone.layers.0.mixer.experts.1.down_proj.weight"));

    let writes = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::ExtentWrite { source, dest, .. } => {
                Some((source.tensor_id, source.span_bytes, dest.offset))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    // Experts are packed contiguously *within* their backing buffer (tight
    // 0/12 offsets), so the exposed `*.packed.weight` view is contiguous.
    assert!(writes.iter().any(|(_, bytes, off)| *bytes == 12 && *off == 0));
    assert!(writes.iter().any(|(_, bytes, off)| *bytes == 12 && *off == 12));

    // Each expert pack is one persistent backing buffer (2 experts × 12 B =
    // 24 B). Backing bases are aligned to PERSISTENT_OPERAND_ALIGNMENT (256)
    // so cuBLAS(Lt) selects its fast `align8` tensor kernels rather than the
    // slow `align1` fallback. Packing tightness is *internal* to each
    // backing and unaffected by the base alignment.
    let backings = program
        .buffers
        .iter()
        .filter_map(|b| b.persistent_offset.map(|o| (b.bytes, o)))
        .collect::<Vec<_>>();
    assert_eq!(backings.len(), 2);
    for (bytes, offset) in &backings {
        assert_eq!(*bytes, 24, "each backing packs 2 experts × 12 B");
        assert_eq!(*offset % 256, 0, "operand base must be 256-aligned");
    }

    // Raw data moved is unchanged (4 experts × 12 B); persistent arena grows
    // only by the per-backing alignment padding (2nd backing at offset 256).
    assert_eq!(program.memory.checkpoint_read_bytes, 48);
    assert_eq!(program.memory.device_write_bytes, 48);
    assert_eq!(program.memory.persistent_bytes, 280);
}

fn metadata() -> CheckpointMetadata {
    CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 16,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            raw(0, "a", 0, &[2], DType::F32),
            raw(1, "b", 8, &[2], DType::F32),
        ],
    }
}

fn nemotron_h_expert_metadata() -> CheckpointMetadata {
    let mut offset = 0u64;
    let mut tensors = Vec::new();
    let specs = [
        (
            0,
            "language_model.backbone.layers.0.mixer.experts.0.up_proj.weight",
            vec![4, 3],
        ),
        (
            1,
            "language_model.backbone.layers.0.mixer.experts.1.up_proj.weight",
            vec![4, 3],
        ),
        (
            2,
            "language_model.backbone.layers.0.mixer.experts.0.down_proj.weight",
            vec![3, 4],
        ),
        (
            3,
            "language_model.backbone.layers.0.mixer.experts.1.down_proj.weight",
            vec![3, 4],
        ),
    ];
    for (id, name, shape) in specs {
        let bytes = tensor_bytes(&shape, DType::BF16);
        tensors.push(RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: offset,
            span_bytes: bytes,
            shape,
            encoding: Encoding::Raw(DType::BF16),
            layout: Layout::dense(1),
        });
        offset += bytes;
    }
    CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "nemotron.safetensors".to_string(),
            size_bytes: offset,
            format: CheckpointFormat::Safetensors,
        }],
        tensors,
    }
}

fn quant_metadata() -> CheckpointMetadata {
    CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 256,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            RawTensor {
                id: TensorId(2),
                name: "q".to_string(),
                file_id: FileId(0),
                file_offset: 200,
                span_bytes: 16,
                shape: vec![4, 8],
                encoding: Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
                layout: Layout::dense(1),
            },
            RawTensor {
                id: TensorId(3),
                name: "fp8".to_string(),
                file_id: FileId(0),
                file_offset: 216,
                span_bytes: 4,
                shape: vec![4],
                encoding: Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::BF16)),
                layout: Layout::dense(1),
            },
            RawTensor {
                id: TensorId(4),
                name: "q_odd".to_string(),
                file_id: FileId(0),
                file_offset: 220,
                span_bytes: 14,
                shape: vec![4, 7],
                encoding: Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
                layout: Layout::dense(1),
            },
        ],
    }
}

fn gpt_oss_mxfp4_metadata() -> CheckpointMetadata {
    gpt_oss_mxfp4_metadata_with_intermediate(64)
}

fn gpt_oss_mxfp4_metadata_with_intermediate(intermediate: i64) -> CheckpointMetadata {
    assert!(intermediate % 32 == 0);
    let mut offset = 0u64;
    let mut tensors = Vec::new();
    let hidden = 64;
    let hidden_groups = hidden / 32;
    let intermediate_groups = intermediate / 32;
    let specs = [
        (
            10,
            "model.layers.0.mlp.experts.gate_up_proj_blocks",
            vec![2, 2 * intermediate, hidden_groups, 16],
            DType::U8,
        ),
        (
            11,
            "model.layers.0.mlp.experts.gate_up_proj_scales",
            vec![2, 2 * intermediate, hidden_groups],
            DType::U8,
        ),
        (
            12,
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            vec![2, 2 * intermediate],
            DType::BF16,
        ),
        (
            13,
            "model.layers.0.mlp.experts.down_proj_blocks",
            vec![2, hidden, intermediate_groups, 16],
            DType::U8,
        ),
        (
            14,
            "model.layers.0.mlp.experts.down_proj_scales",
            vec![2, hidden, intermediate_groups],
            DType::U8,
        ),
        (
            15,
            "model.layers.0.mlp.experts.down_proj_bias",
            vec![2, hidden],
            DType::BF16,
        ),
    ];
    for (id, name, shape, dtype) in specs {
        let bytes = tensor_bytes(&shape, dtype);
        tensors.push(RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: offset,
            span_bytes: bytes,
            shape,
            encoding: Encoding::Raw(dtype),
            layout: Layout::dense(1),
        });
        offset += bytes;
    }
    CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "gpt_oss.safetensors".to_string(),
            size_bytes: offset,
            format: CheckpointFormat::Safetensors,
        }],
        tensors,
    }
}

fn raw(id: u32, name: &str, offset: u64, shape: &[i64], dtype: DType) -> RawTensor {
    RawTensor {
        id: TensorId(id),
        name: name.to_string(),
        file_id: FileId(0),
        file_offset: offset,
        span_bytes: 8,
        shape: shape.to_vec(),
        encoding: Encoding::Raw(dtype),
        layout: Layout::dense(1),
    }
}

fn decl(id: u32, name: &str, shape: &[i64], encoding: Encoding) -> TensorDecl {
    TensorDecl {
        id: TensorId(id),
        name: name.to_string(),
        shape: shape.to_vec(),
        encoding,
        layout: Layout::dense(1),
        sharding: Sharding::replicated(),
        alignment: 1,
    }
}

fn quant(scheme: QuantScheme, dtype: DType) -> QuantSpec {
    QuantSpec {
        scheme,
        logical_dtype: dtype,
        bits_per_element: scheme.default_bits(),
        group_size: scheme.default_group_size(),
        channel_axis: None,
        scale_dtype: Some(DType::F32),
        zero_point_dtype: None,
        block_shape: Vec::new(),
    }
}

fn tensor_bytes(shape: &[i64], dtype: DType) -> u64 {
    shape
        .iter()
        .fold(dtype.bytes(), |acc, dim| acc * u64::try_from(*dim).unwrap())
}

// ── SlabScatter lowering tests ──────────────────────────────────────

#[test]
fn slab_scatter_merges_nearby_bulk_extent_writes() {
    // Three tensors with small gaps in the file. The gaps prevent
    // coalesce_persistent_arena_writes from merging them into a single
    // BulkExtentWrite, but they're close enough for slab_scatter to
    // group them (gap < 64 MiB, overread < 5/4).
    let chunk = 2 * 1024 * 1024; // 2 MiB per tensor
    let gap = 1024; // 1 KiB gap between tensors in the file
    let file_size = chunk * 3 + gap * 2 + 4096;
    let meta = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "big.safetensors".to_string(),
            size_bytes: file_size,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            raw_big(0, "t0", 0, chunk, DType::BF16),
            raw_big(1, "t1", chunk + gap, chunk, DType::BF16),
            raw_big(2, "t2", 2 * (chunk + gap), chunk, DType::BF16),
        ],
    };
    let mut plan = LayoutPlan::new();
    let mut ids = Vec::new();
    for i in 0..3u32 {
        let cols = (chunk / 2) as i64; // BF16 = 2 bytes
        let src = plan.push(LayoutExpr::Source {
            tensor: TensorId(i),
            decl: decl(i, &format!("t{i}"), &[cols], Encoding::Raw(DType::BF16)),
        });
        let r = plan.push(LayoutExpr::Realize {
            input: src,
            runtime_name: format!("t{i}"),
            decl: decl(i, &format!("t{i}"), &[cols], Encoding::Raw(DType::BF16)),
        });
        ids.push(r);
    }
    plan.outputs = ids;

    let target = StorageTarget {
        backend: BackendKind::Cuda,
        ..StorageTarget::default()
    };
    let program = lower_layout_plan(&meta, &plan, target).unwrap();

    let slabs: Vec<_> = program
        .instrs
        .iter()
        .filter(|i| matches!(i, StorageInstr::SlabScatter { .. }))
        .collect();
    assert!(
        !slabs.is_empty(),
        "expected at least one SlabScatter, got none; instrs: {:#?}",
        program.instrs,
    );
    if let StorageInstr::SlabScatter { placements, span_bytes, .. } = slabs[0] {
        assert!(
            placements.len() >= 2,
            "slab must have at least 2 placements, got {}",
            placements.len()
        );
        assert!(
            *span_bytes >= chunk * 2,
            "slab span_bytes {} should cover at least two chunks",
            span_bytes,
        );
        for p in placements {
            assert!(
                p.bytes > 0,
                "placement bytes must be non-zero"
            );
        }
    }
}

#[test]
fn slab_scatter_rejects_excessive_overread() {
    // Two small tensors with a huge gap — overread exceeds 5/4 threshold.
    let small = 1024 * 1024; // 1 MiB each
    let gap = 256 * 1024 * 1024; // 256 MiB gap
    let file_size = small + gap + small;
    let meta = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "sparse.safetensors".to_string(),
            size_bytes: file_size,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            raw_big(0, "near", 0, small, DType::BF16),
            raw_big(1, "far", small + gap, small, DType::BF16),
        ],
    };
    let mut plan = LayoutPlan::new();
    for i in 0..2u32 {
        let cols = (small / 2) as i64;
        let src = plan.push(LayoutExpr::Source {
            tensor: TensorId(i),
            decl: decl(i, &format!("t{i}"), &[cols], Encoding::Raw(DType::BF16)),
        });
        let r = plan.push(LayoutExpr::Realize {
            input: src,
            runtime_name: format!("t{i}"),
            decl: decl(i, &format!("t{i}"), &[cols], Encoding::Raw(DType::BF16)),
        });
        plan.outputs.push(r);
    }
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        ..StorageTarget::default()
    };
    let program = lower_layout_plan(&meta, &plan, target).unwrap();
    let slabs: Vec<_> = program
        .instrs
        .iter()
        .filter(|i| matches!(i, StorageInstr::SlabScatter { .. }))
        .collect();
    assert!(
        slabs.is_empty(),
        "should NOT merge into SlabScatter when overread is excessive; got {:#?}",
        slabs,
    );
}

#[test]
fn slab_scatter_placement_offsets_are_within_span() {
    let chunk = 4 * 1024 * 1024;
    let gap = 512 * 1024; // small gap
    let file_size = chunk * 3 + gap * 2;
    let meta = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "layout.safetensors".to_string(),
            size_bytes: file_size,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            raw_big(0, "a", 0, chunk, DType::BF16),
            raw_big(1, "b", chunk + gap, chunk, DType::BF16),
            raw_big(2, "c", 2 * (chunk + gap), chunk, DType::BF16),
        ],
    };
    let mut plan = LayoutPlan::new();
    for i in 0..3u32 {
        let cols = (chunk / 2) as i64;
        let src = plan.push(LayoutExpr::Source {
            tensor: TensorId(i),
            decl: decl(i, &format!("p{i}"), &[cols], Encoding::Raw(DType::BF16)),
        });
        let r = plan.push(LayoutExpr::Realize {
            input: src,
            runtime_name: format!("p{i}"),
            decl: decl(i, &format!("p{i}"), &[cols], Encoding::Raw(DType::BF16)),
        });
        plan.outputs.push(r);
    }
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        ..StorageTarget::default()
    };
    let program = lower_layout_plan(&meta, &plan, target).unwrap();
    for instr in &program.instrs {
        if let StorageInstr::SlabScatter {
            span_bytes,
            placements,
            ..
        } = instr
        {
            for (idx, p) in placements.iter().enumerate() {
                assert!(
                    p.src_offset + p.bytes <= *span_bytes,
                    "placement {idx}: src_offset {} + bytes {} exceeds span_bytes {}",
                    p.src_offset,
                    p.bytes,
                    span_bytes,
                );
            }
        }
    }
}

fn raw_big(id: u32, name: &str, offset: u64, span_bytes: u64, dtype: DType) -> RawTensor {
    let elem = dtype.bytes();
    let count = (span_bytes / elem) as i64;
    RawTensor {
        id: TensorId(id),
        name: name.to_string(),
        file_id: FileId(0),
        file_offset: offset,
        span_bytes,
        shape: vec![count],
        encoding: Encoding::Raw(dtype),
        layout: Layout::dense(1),
    }
}

// ── MLA weight fusion tests ─────────────────────────────────────────

#[test]
fn mla_q_kv_a_fusion_produces_joined_tensor() {
    use pie_weight_loader::config::ModelConfig;
    use pie_weight_loader::storage_compiler::compile_storage_program;

    let h = 128i64;
    let q_lora = 32i64;
    let kv_lora_rope = 16i64;
    let mut offset = 0u64;
    let mut tensors = Vec::new();
    let specs: Vec<(u32, &str, Vec<i64>)> = vec![
        (0, "model.layers.0.self_attn.q_a_proj.weight", vec![q_lora, h]),
        (1, "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", vec![kv_lora_rope, h]),
        (2, "model.layers.0.self_attn.q_a_layernorm.weight", vec![q_lora]),
        (3, "model.layers.0.self_attn.q_b_proj.weight", vec![64, q_lora]),
        (4, "model.layers.0.self_attn.kv_a_layernorm.weight", vec![12]),
        (5, "model.layers.0.self_attn.kv_b_proj.weight", vec![64, 12]),
        (6, "model.layers.0.self_attn.o_proj.weight", vec![h, 32]),
        (7, "model.layers.0.input_layernorm.weight", vec![h]),
        (8, "model.layers.0.post_attention_layernorm.weight", vec![h]),
        (9, "model.layers.0.mlp.gate_proj.weight", vec![h, h]),
        (10, "model.layers.0.mlp.up_proj.weight", vec![h, h]),
        (11, "model.layers.0.mlp.down_proj.weight", vec![h, h]),
    ];
    for (id, name, shape) in &specs {
        let bytes = shape.iter().fold(2u64, |acc, d| acc * *d as u64);
        tensors.push(RawTensor {
            id: TensorId(*id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: offset,
            span_bytes: bytes,
            shape: shape.clone(),
            encoding: Encoding::Raw(DType::BF16),
            layout: Layout::dense(1),
        });
        offset += bytes;
    }
    let meta = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: offset,
            format: CheckpointFormat::Safetensors,
        }],
        tensors,
    };
    let cfg = ModelConfig {
        model_type: "kimi_k2".to_string(),
        num_hidden_layers: 1,
        ..ModelConfig::default()
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        ..StorageTarget::default()
    };
    let abi = pie_weight_loader::abi::RuntimeAbi::default_for_target(&meta, &cfg, &target).unwrap();
    let program = compile_storage_program(&meta, &cfg, &abi, target).unwrap();
    let summary = program.summary();

    // The fusion should have joined q_a_proj + kv_a_proj into one tensor
    let has_fused = program.tensors.iter().any(|t| {
        t.name.contains("q_kv_a_proj.fused")
    });
    assert!(
        has_fused,
        "Expected fused q_kv_a_proj tensor; summary: {summary}\ntensors: {:?}",
        program.tensors.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    // The fused tensor should have rows = q_lora + kv_lora_rope
    let fused = program.tensors.iter().find(|t| t.name.contains("q_kv_a_proj.fused")).unwrap();
    assert_eq!(fused.shape[0], q_lora + kv_lora_rope);
    assert_eq!(fused.shape[1], h);

    // Summary should show the program compiled successfully
    assert!(summary.tensor_count > 0);
    assert!(summary.finalize_count > 0);
}

// ── SSD expert streaming (DeepSeek-V4) ──────────────────────────────

#[test]
fn dsv4_stream_routed_experts_excludes_expert_tensors_from_program() {
    // Minimal DSv4-shaped checkpoint: one layer with a norm, a router, a
    // shared expert, and two routed experts (w1/w2/w3 weight+scale each).
    let mut offset = 0u64;
    let mut tensors = Vec::new();
    let mut push = |id: u32, name: &str, shape: Vec<i64>, dtype: DType| {
        let bytes = shape.iter().fold(dtype.bytes(), |acc, d| acc * *d as u64);
        tensors.push(RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: offset,
            span_bytes: bytes,
            shape,
            encoding: Encoding::Raw(dtype),
            layout: Layout::dense(1),
        });
        offset += bytes;
    };
    let mut id = 0u32;
    let mut next = || {
        let v = id;
        id += 1;
        v
    };
    push(next(), "layers.0.ffn_norm.weight", vec![64], DType::BF16);
    push(next(), "layers.0.ffn.gate.weight", vec![2, 64], DType::BF16);
    push(next(), "layers.0.ffn.shared_experts.w1.weight", vec![32, 64], DType::BF16);
    push(next(), "layers.0.ffn.shared_experts.w2.weight", vec![64, 32], DType::BF16);
    push(next(), "layers.0.ffn.shared_experts.w3.weight", vec![32, 64], DType::BF16);
    for e in 0..2 {
        for w in ["w1", "w2", "w3"] {
            push(
                next(),
                &format!("layers.0.ffn.experts.{e}.{w}.weight"),
                vec![32, 32],
                DType::U8,
            );
            push(
                next(),
                &format!("layers.0.ffn.experts.{e}.{w}.scale"),
                vec![32, 1],
                DType::U8,
            );
        }
    }
    // MTP routed experts share the `.ffn.experts.` naming but must stay
    // resident — the stream cache only pages the main `layers.*` bank.
    for w in ["w1", "w2", "w3"] {
        push(
            next(),
            &format!("mtp.0.ffn.experts.0.{w}.weight"),
            vec![32, 32],
            DType::U8,
        );
        push(
            next(),
            &format!("mtp.0.ffn.experts.0.{w}.scale"),
            vec![32, 1],
            DType::U8,
        );
    }
    let meta = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: offset,
            format: CheckpointFormat::Safetensors,
        }],
        tensors,
    };
    let cfg = pie_weight_loader::config::ModelConfig {
        model_type: "deepseek_v4".to_string(),
        num_hidden_layers: 1,
        num_experts: 2,
        num_experts_per_tok: 2,
        ..pie_weight_loader::config::ModelConfig::default()
    };

    let compile = |stream: bool| {
        let target = StorageTarget {
            backend: BackendKind::Cuda,
            stream_routed_experts: stream,
            ..StorageTarget::default()
        };
        let abi =
            pie_weight_loader::abi::RuntimeAbi::default_for_target(&meta, &cfg, &target).unwrap();
        compile_storage_program(&meta, &cfg, &abi, target).unwrap()
    };

    let resident = compile(false);
    let streamed = compile(true);

    assert!(
        resident.tensors.iter().any(|t| t.name.starts_with("layers.0.ffn.experts.")),
        "baseline program must materialize main-stack routed experts"
    );
    assert!(
        !streamed
            .tensors
            .iter()
            .any(|t| t.name.starts_with("layers.") && t.name.contains(".ffn.experts.")),
        "streamed program must not declare main-stack routed-expert tensors; got: {:?}",
        streamed.tensors.iter().map(|t| &t.name).collect::<Vec<_>>()
    );
    // Non-expert tensors (norm, router, shared experts) stay resident.
    assert!(streamed.tensors.iter().any(|t| t.name == "layers.0.ffn.gate.weight"));
    assert!(
        streamed
            .tensors
            .iter()
            .any(|t| t.name == "layers.0.ffn.shared_experts.w1.weight")
    );
    assert!(
        streamed
            .tensors
            .iter()
            .any(|t| t.name.starts_with("mtp.0.ffn.experts.")),
        "MTP experts must remain resident; got: {:?}",
        streamed.tensors.iter().map(|t| &t.name).collect::<Vec<_>>()
    );
    assert!(
        streamed.memory.persistent_bytes < resident.memory.persistent_bytes,
        "streaming must shrink the persistent arena ({} vs {})",
        streamed.memory.persistent_bytes,
        resident.memory.persistent_bytes
    );

    // Deferred stream plan: template + bindings, not on the resident schedule.
    assert!(!streamed.stream.is_empty(), "stream plan must be present");
    assert_eq!(streamed.stream.num_layers, 1);
    assert_eq!(streamed.stream.num_experts, 2);
    assert_eq!(streamed.stream.sections_per_expert, 6);
    assert_eq!(streamed.stream.template.len(), 6);
    assert_eq!(streamed.stream.bindings.len(), 1 * 2 * 6);
    assert!(streamed.stream.slot_bytes > 0);
    for id in &streamed.stream.template {
        assert!(
            !streamed.schedule.contains(id),
            "stream template instr {id:?} must not be on the resident schedule"
        );
        assert!(
            streamed.instrs.iter().any(|instr| instr_id(instr) == *id),
            "stream template instr {id:?} must exist in instrs"
        );
    }
    // Template instrs are ExtentWrites with slot-relative dests.
    for id in &streamed.stream.template {
        let instr = streamed
            .instrs
            .iter()
            .find(|instr| instr_id(instr) == *id)
            .unwrap();
        match instr {
            StorageInstr::ExtentWrite { dest, .. } => {
                assert_eq!(dest.buffer.0, u32::MAX, "slot-base sentinel");
            }
            other => panic!("expected ExtentWrite in stream template, got {other:?}"),
        }
    }
}

#[test]
fn dsv4_stream_routed_experts_rejects_tensor_parallel() {
    // Enough of a streamed tensor for skip_streamed to fire before the TP check.
    let meta = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".into(),
            size_bytes: 32,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "layers.0.ffn.experts.0.w1.weight".into(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 32,
            shape: vec![32],
            encoding: Encoding::Raw(DType::U8),
            layout: Layout::dense(1),
        }],
    };
    let cfg = pie_weight_loader::config::ModelConfig {
        model_type: "deepseek_v4".to_string(),
        num_hidden_layers: 1,
        num_experts: 2,
        num_experts_per_tok: 2,
        ..pie_weight_loader::config::ModelConfig::default()
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tp_rank: 0,
        tp_size: 2,
        stream_routed_experts: true,
        ..StorageTarget::default()
    };
    let err = pie_weight_loader::abi::RuntimeAbi::default_for_target(&meta, &cfg, &target)
        .unwrap_err()
        .to_string();
    assert!(err.contains("tp_size=1"), "unexpected error: {err}");
}

#[test]
fn stream_routed_experts_rejects_unsupported_arch() {
    let meta = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".into(),
            size_bytes: 8,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "layers.0.ffn.experts.0.w1.weight".into(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 8,
            shape: vec![8],
            encoding: Encoding::Raw(DType::U8),
            layout: Layout::dense(1),
        }],
    };
    let cfg = pie_weight_loader::config::ModelConfig {
        model_type: "llama".to_string(),
        num_hidden_layers: 1,
        ..pie_weight_loader::config::ModelConfig::default()
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        stream_routed_experts: true,
        ..StorageTarget::default()
    };
    let err = pie_weight_loader::abi::RuntimeAbi::default_for_target(&meta, &cfg, &target)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not supported"), "unexpected error: {err}");
}

fn instr_id(instr: &StorageInstr) -> pie_weight_loader::types::InstrId {
    match instr {
        StorageInstr::Allocate { id, .. }
        | StorageInstr::ExtentWrite { id, .. }
        | StorageInstr::BulkExtentWrite { id, .. }
        | StorageInstr::SlabScatter { id, .. }
        | StorageInstr::TileMap { id, .. }
        | StorageInstr::CreateView { id, .. }
        | StorageInstr::Attach { id, .. }
        | StorageInstr::Release { id, .. }
        | StorageInstr::Finalize { id, .. } => *id,
    }
}

#[test]
fn mixtral_stream_routed_experts_excludes_expert_tensors_from_program() {
    // Minimal Mixtral-shaped checkpoint: 1 layer × 2 experts × w1/w2/w3 BF16.
    let mut offset = 0u64;
    let mut tensors = Vec::new();
    let mut push = |id: u32, name: &str, shape: Vec<i64>, dtype: DType| {
        let bytes = shape.iter().fold(dtype.bytes(), |acc, d| acc * *d as u64);
        tensors.push(RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: offset,
            span_bytes: bytes,
            shape,
            encoding: Encoding::Raw(dtype),
            layout: Layout::dense(1),
        });
        offset += bytes;
    };
    let mut id = 0u32;
    let mut next = || {
        let v = id;
        id += 1;
        v
    };
    push(next(), "model.embed_tokens.weight", vec![64], DType::BF16);
    push(next(), "model.layers.0.input_layernorm.weight", vec![64], DType::BF16);
    push(
        next(),
        "model.layers.0.block_sparse_moe.gate.weight",
        vec![2, 64],
        DType::BF16,
    );
    for e in 0..2 {
        for w in ["w1", "w2", "w3"] {
            push(
                next(),
                &format!("model.layers.0.block_sparse_moe.experts.{e}.{w}.weight"),
                vec![32, 32],
                DType::BF16,
            );
        }
    }
    let meta = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "mixtral.safetensors".into(),
            size_bytes: offset,
            format: CheckpointFormat::Safetensors,
        }],
        tensors,
    };
    let cfg = pie_weight_loader::config::ModelConfig {
        model_type: "mixtral".to_string(),
        num_hidden_layers: 1,
        num_experts: 2,
        num_experts_per_tok: 2,
        ..pie_weight_loader::config::ModelConfig::default()
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        stream_routed_experts: true,
        ..StorageTarget::default()
    };
    let abi =
        pie_weight_loader::abi::RuntimeAbi::default_for_target(&meta, &cfg, &target).unwrap();
    let streamed = compile_storage_program(&meta, &cfg, &abi, target).unwrap();

    assert!(
        !streamed.tensors.iter().any(|t| t.name.contains("block_sparse_moe.experts.")),
        "streamed program must not declare Mixtral expert tensors; got: {:?}",
        streamed.tensors.iter().map(|t| &t.name).collect::<Vec<_>>()
    );
    assert!(
        streamed
            .tensors
            .iter()
            .any(|t| t.name.contains("block_sparse_moe.gate.weight")),
        "router must remain resident; got: {:?}",
        streamed.tensors.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    assert!(!streamed.stream.is_empty());
    assert_eq!(streamed.stream.num_layers, 1);
    assert_eq!(streamed.stream.num_experts, 2);
    assert_eq!(streamed.stream.sections_per_expert, 3);
    assert_eq!(streamed.stream.template.len(), 3);
    assert_eq!(streamed.stream.bindings.len(), 1 * 2 * 3);
    for id in &streamed.stream.template {
        assert!(!streamed.schedule.contains(id));
        assert!(streamed.instrs.iter().any(|instr| instr_id(instr) == *id));
    }
}

#[test]
fn mixtral_stream_routed_experts_rejects_tensor_parallel() {
    let meta = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "mixtral.safetensors".into(),
            size_bytes: 64,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "model.layers.0.block_sparse_moe.experts.0.w1.weight".into(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 64,
            shape: vec![32, 1],
            encoding: Encoding::Raw(DType::BF16),
            layout: Layout::dense(1),
        }],
    };
    let cfg = pie_weight_loader::config::ModelConfig {
        model_type: "mixtral".to_string(),
        num_hidden_layers: 1,
        num_experts: 2,
        num_experts_per_tok: 2,
        ..pie_weight_loader::config::ModelConfig::default()
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tp_rank: 0,
        tp_size: 2,
        stream_routed_experts: true,
        ..StorageTarget::default()
    };
    let err = pie_weight_loader::abi::RuntimeAbi::default_for_target(&meta, &cfg, &target)
        .unwrap_err()
        .to_string();
    assert!(err.contains("tp_size=1"), "unexpected error: {err}");
}

#[test]
fn gpt_oss_stream_routed_experts_fused_plan() {
    // Minimal fused-bank fixture: 1 layer × 2 experts (not the real 32).
    let mut offset = 0u64;
    let mut tensors = Vec::new();
    let mut push = |id: u32, name: &str, span: u64| {
        tensors.push(RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: offset,
            span_bytes: span,
            shape: vec![span as i64],
            encoding: Encoding::Raw(DType::U8),
            layout: Layout::dense(1),
        });
        offset += span;
    };
    let mut id = 0u32;
    let mut next = || {
        let v = id;
        id += 1;
        v
    };
    push(next(), "model.embed_tokens.weight", 64);
    push(next(), "model.layers.0.input_layernorm.weight", 32);
    push(next(), "model.layers.0.mlp.router.weight", 64);
    push(next(), "model.layers.0.mlp.router.bias", 4);
    push(next(), "model.layers.0.mlp.experts.gate_up_proj_blocks", 200);
    push(next(), "model.layers.0.mlp.experts.gate_up_proj_scales", 40);
    push(next(), "model.layers.0.mlp.experts.gate_up_proj_bias", 16);
    push(next(), "model.layers.0.mlp.experts.down_proj_blocks", 100);
    push(next(), "model.layers.0.mlp.experts.down_proj_scales", 20);
    push(next(), "model.layers.0.mlp.experts.down_proj_bias", 8);
    let meta = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "gpt-oss.safetensors".into(),
            size_bytes: offset,
            format: CheckpointFormat::Safetensors,
        }],
        tensors,
    };
    let cfg = pie_weight_loader::config::ModelConfig {
        model_type: "gpt_oss".to_string(),
        num_hidden_layers: 1,
        num_experts: 2,
        num_experts_per_tok: 2,
        ..pie_weight_loader::config::ModelConfig::default()
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        stream_routed_experts: true,
        mxfp4_moe: pie_weight_loader::types::Mxfp4MoePolicy::RoutedDecode,
        ..StorageTarget::default()
    };
    let abi =
        pie_weight_loader::abi::RuntimeAbi::default_for_target(&meta, &cfg, &target).unwrap();
    let streamed = compile_storage_program(&meta, &cfg, &abi, target).unwrap();

    // Biases stay resident; packs/scales do not.
    assert!(
        streamed
            .tensors
            .iter()
            .any(|t| t.name.contains("gate_up_proj.bias")),
        "biases must remain resident; got: {:?}",
        streamed.tensors.iter().map(|t| &t.name).collect::<Vec<_>>()
    );
    assert!(
        !streamed
            .tensors
            .iter()
            .any(|t| t.name.contains("gate_up_proj.weight")
                || t.name.contains("down_proj.weight")
                || t.name.contains("weight_scale")),
        "streamed packs/scales must not be resident; got: {:?}",
        streamed.tensors.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    assert!(!streamed.stream.is_empty());
    assert_eq!(streamed.stream.num_layers, 1);
    assert_eq!(streamed.stream.num_experts, 2);
    assert_eq!(streamed.stream.sections_per_expert, 4);
    assert_eq!(streamed.stream.template.len(), 4);
    assert_eq!(streamed.stream.bindings.len(), 1 * 2 * 4);
    // Per-expert slices: gate_up weight 100, scale 20, down weight 50, scale 10.
    assert_eq!(streamed.stream.section_bytes, vec![100, 20, 50, 10]);
    assert_eq!(streamed.stream.bindings[0].span_bytes, 100);
    assert_eq!(
        streamed.stream.bindings[4].file_offset,
        streamed.stream.bindings[0].file_offset + 100
    );
}

#[test]
fn gpt_oss_stream_rejects_native_mxfp4() {
    // Enough of a streamed tensor for skip_streamed to fire before the policy check.
    let meta = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "gpt-oss.safetensors".into(),
            size_bytes: 8,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "model.layers.0.mlp.experts.gate_up_proj_blocks".into(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 8,
            shape: vec![8],
            encoding: Encoding::Raw(DType::U8),
            layout: Layout::dense(1),
        }],
    };
    let cfg = pie_weight_loader::config::ModelConfig {
        model_type: "gpt_oss".to_string(),
        num_hidden_layers: 1,
        num_experts: 2,
        num_experts_per_tok: 2,
        ..pie_weight_loader::config::ModelConfig::default()
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        stream_routed_experts: true,
        mxfp4_moe: pie_weight_loader::types::Mxfp4MoePolicy::NativeGemm,
        native_mxfp4_moe: true,
        ..StorageTarget::default()
    };
    let err = pie_weight_loader::abi::RuntimeAbi::default_for_target(&meta, &cfg, &target)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("routed_dequant") || err.contains("ExtentWrite"),
        "unexpected error: {err}"
    );
}

