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
            StorageInstr::ExtentWrite { id, source, dest } => Some((id, source, dest)),
            _ => None,
        })
        .collect();
    assert_eq!(writes.len(), 1);
    let (write_id, source, dest) = writes[0];
    assert_eq!(source.tensor_id, TensorId(6));
    assert_eq!(source.file_offset, 512);
    assert_eq!(source.span_bytes, 8);
    assert_eq!(dest.offset, 0);
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

fn instr_id(instr: &StorageInstr) -> pie_weight_loader::types::InstrId {
    match instr {
        StorageInstr::Allocate { id, .. }
        | StorageInstr::ExtentWrite { id, .. }
        | StorageInstr::TileMap { id, .. }
        | StorageInstr::CreateView { id, .. }
        | StorageInstr::Attach { id, .. }
        | StorageInstr::Release { id, .. }
        | StorageInstr::Finalize { id, .. } => *id,
    }
}
