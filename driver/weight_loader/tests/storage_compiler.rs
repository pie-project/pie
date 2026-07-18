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
fn gptq_int4_default_abi_lowers_checkpoint_triplet_to_marlin() {
    let metadata = offline_int4_metadata(QuantScheme::GptqInt4);
    let cfg = pie_weight_loader::config::ModelConfig {
        model_type: "qwen2".to_string(),
        quant_method: "gptq".to_string(),
        quant_bits: 4,
        quant_group_size: 16,
        quant_symmetric: true,
        num_hidden_layers: 1,
        ..Default::default()
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        preferred_alignment: 256,
        ..StorageTarget::default()
    };
    let abi =
        pie_weight_loader::abi::RuntimeAbi::default_for_target(&metadata, &cfg, &target).unwrap();
    let weight = abi
        .tensors
        .iter()
        .find(|tensor| tensor.output_name == "model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(weight.shape, vec![64, 32]);
    assert!(matches!(
        &weight.encoding,
        Encoding::Quant(spec)
            if spec.scheme == QuantScheme::GptqInt4 && spec.group_size == 16
    ));
    assert!(abi.tensors.iter().any(|tensor| {
        tensor.output_name == "model.layers.0.self_attn.q_proj.weight_scale_inv"
            && tensor.encoding == Encoding::Raw(DType::BF16)
    }));
    assert!(!abi.tensors.iter().any(|tensor| {
        tensor.output_name.ends_with(".qweight")
            || tensor.output_name.ends_with(".qzeros")
            || tensor.output_name.ends_with(".g_idx")
    }));

    let program = compile_storage_program(&metadata, &cfg, &abi, target).unwrap();
    let layouts = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap {
                kind: TileMapKind::Repack,
                transform,
                ..
            } => Some(transform.repack.layout),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(layouts.contains(&RepackLayout::MarlinGptqWeight));
    assert!(layouts.contains(&RepackLayout::MarlinInt4Scale));
    assert_eq!(program.memory.persistent_bytes, 1280);
}

#[test]
fn awq_int4_default_abi_lowers_checkpoint_triplet_to_marlin() {
    let metadata = offline_int4_metadata(QuantScheme::AwqInt4);
    let cfg = pie_weight_loader::config::ModelConfig {
        model_type: "qwen2".to_string(),
        quant_method: "awq".to_string(),
        quant_bits: 4,
        quant_group_size: 16,
        quant_symmetric: false,
        quant_zero_point: true,
        num_hidden_layers: 1,
        ..Default::default()
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        preferred_alignment: 256,
        ..StorageTarget::default()
    };
    let abi =
        pie_weight_loader::abi::RuntimeAbi::default_for_target(&metadata, &cfg, &target).unwrap();
    assert!(abi.tensors.iter().any(|tensor| {
        tensor.output_name == "model.layers.0.self_attn.q_proj.weight_zero_point"
            && tensor.encoding == Encoding::Raw(DType::I32)
    }));
    let program = compile_storage_program(&metadata, &cfg, &abi, target).unwrap();
    let layouts = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap {
                kind: TileMapKind::Repack,
                transform,
                ..
            } => Some(transform.repack.layout),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(layouts.contains(&RepackLayout::MarlinAwqWeight));
    assert!(layouts.contains(&RepackLayout::MarlinInt4Scale));
    assert!(layouts.contains(&RepackLayout::MarlinAwqZeroPoint));
    assert!(program.memory.transform_scratch_peak_bytes > 0);
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
        ..Default::default()
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
        ..Default::default()
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
        ..Default::default()
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

fn offline_int4_metadata(scheme: QuantScheme) -> CheckpointMetadata {
    let prefix = "model.layers.0.self_attn.q_proj";
    let (qweight_shape, qweight_bytes) = match scheme {
        QuantScheme::GptqInt4 => (vec![4, 64], 4 * 64 * 4),
        QuantScheme::AwqInt4 => (vec![32, 8], 32 * 8 * 4),
        _ => panic!("offline_int4_metadata requires GPTQ or AWQ"),
    };
    let mut offset = 0_u64;
    let mut tensor = |id: u32,
                      suffix: &str,
                      shape: Vec<i64>,
                      dtype: DType,
                      bytes: u64| {
        let raw = RawTensor {
            id: TensorId(id),
            name: format!("{prefix}.{suffix}"),
            file_id: FileId(0),
            file_offset: offset,
            span_bytes: bytes,
            shape,
            encoding: Encoding::Raw(dtype),
            layout: Layout::dense(1),
        };
        offset += bytes;
        raw
    };
    let qweight = tensor(0, "qweight", qweight_shape, DType::I32, qweight_bytes);
    let qzeros = tensor(1, "qzeros", vec![2, 8], DType::I32, 2 * 8 * 4);
    let scales = tensor(2, "scales", vec![2, 64], DType::F16, 2 * 64 * 2);
    let g_idx = tensor(3, "g_idx", vec![32], DType::I32, 32 * 4);
    CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: offset,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![qweight, qzeros, scales, g_idx],
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
