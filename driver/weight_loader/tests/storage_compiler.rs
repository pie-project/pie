use pie_weight_loader::ir::{LayoutExpr, LayoutPlan};
use pie_weight_loader::source::{CheckpointFile, CheckpointMetadata, RawTensor};
use pie_weight_loader::storage::{StorageInstr, StorageTarget, TileMapKind};
use pie_weight_loader::storage_compiler::lower_layout_plan;
use pie_weight_loader::types::{
    Axis, BackendKind, CheckpointFormat, DType, Encoding, FileId, Layout, QuantScheme, QuantSpec,
    Sharding, TensorDecl, TensorId,
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
