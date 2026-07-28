/// A contract stored next to the test that compiles it.
///
/// The families these exercise — GPT-OSS's native MXFP4 expert groups,
/// Nemotron-H's packed experts, Kimi's MLA joins — are authored by the driver
/// now, so there is no function in this crate that could rebuild one. That is
/// the point: what these tests are about is the *plan* the compiler produces
/// for a contract shaped like that, and the contract is an input like any
/// other.
fn stored_contract(name: &str) -> ModelContract {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/contracts")
        .join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("{name}: cannot read {}: {err}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("{name}: parsing: {err}"))
}
use pie_loader::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use pie_loader::contract::{Expr, ModelContract, Scales, TensorContract, TensorType};
use pie_loader::plan::compile as compile_load_plan;
use pie_loader::plan::{StorageInstr, StorageTarget, TileMapKind};
use pie_loader::types::{
    Axis, BackendKind, CheckpointFormat, DType, Encoding, FileId, QuantGranularity, QuantScheme,
    QuantSpec, RepackLayout, RowMap, ScaleForm, TensorId,
};

#[test]
fn metal_qwen35_schema_emits_canonical_affine_u4_arena() {
    let specs = [
        ("lm_head.weight", vec![2, 8], DType::U32),
        ("lm_head.scales", vec![2, 1], DType::BF16),
        ("lm_head.biases", vec![2, 1], DType::BF16),
        ("model.language_model.norm.weight", vec![64], DType::BF16),
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            vec![64, 8],
            DType::U32,
        ),
        (
            "model.language_model.layers.0.self_attn.q_proj.scales",
            vec![64, 1],
            DType::BF16,
        ),
        (
            "model.language_model.layers.0.self_attn.q_proj.biases",
            vec![64, 1],
            DType::BF16,
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            vec![16, 64],
            DType::BF16,
        ),
        ("model.visual.patch.weight", vec![1], DType::BF16),
        ("mtp.fc.weight", vec![1], DType::BF16),
    ];
    let mut offset = 0u64;
    let tensors = specs
        .into_iter()
        .enumerate()
        .map(|(index, (name, shape, dtype))| {
            let span_bytes = shape.iter().product::<i64>() as u64 * dtype.bytes();
            let tensor = RawTensor {
                id: TensorId(index as u32),
                name: name.to_string(),
                file_id: FileId(0),
                file_offset: offset,
                span_bytes,
                shape,
                encoding: Encoding::Raw(dtype),
            };
            offset += span_bytes;
            tensor
        })
        .collect();
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: offset,
            format: CheckpointFormat::Safetensors,
        }],
        tensors,
    };
    let target = StorageTarget {
        backend: BackendKind::Metal,
        tile_map_mask: pie_loader::plan::METAL_TILE_MAP_MASK,
        max_tile_bytes: 64 << 20,
        preferred_alignment: 256,
        ..StorageTarget::default()
    };
    // The MLX schema states 4-bit weights that the checkpoint packs eight to a
    // u32 word: a bitcast to the logical shape and an affine-U4 encoding, with
    // the scales and biases named as the tensors they are. The driver authors
    // this; the test states it directly, because what is under test here is the
    // arena the compiler builds from it, not who wrote it down.
    let affine_u4 = |group_size: u32| {
        Encoding::Quant(
            QuantSpec {
                scheme: QuantScheme::MlxAffineU4,
                logical_dtype: DType::BF16,
                bits_per_element: 4,
                group_size,
                channel_axis: Some(Axis(1)),
            }
            .normalized(),
        )
    };
    let packed = |source: &str, output: &str, rows: i64, cols: i64| {
        let ty = pie_loader::contract::TensorType::new(vec![rows, cols], affine_u4(64));
        TensorContract::new(
            output.to_string(),
            Expr::src(source.to_string()).bitcast(ty),
            vec![rows, cols],
            affine_u4(64),
        )
    };
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![
            packed("lm_head.weight", "lm_head.weight", 2, 64),
            TensorContract::new(
                "final_norm.weight",
                Expr::src("model.language_model.norm.weight"),
                vec![64],
                Encoding::Raw(DType::BF16),
            ),
            packed(
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.q_proj.weight",
                64,
                64,
            ),
            TensorContract::new(
                "layers.0.linear_attn.in_proj_a.weight",
                Expr::src("model.language_model.layers.0.linear_attn.in_proj_a.weight"),
                vec![16, 64],
                Encoding::Raw(DType::BF16),
            ),
        ],
    };

    let program = compile_load_plan(&metadata, &contract, target).unwrap();
    assert!(
        !program
            .instrs
            .iter()
            .any(|instr| matches!(instr, StorageInstr::TileMap { .. }))
    );
    assert!(
        program
            .buffers
            .iter()
            .filter_map(|buffer| buffer.persistent_offset)
            .all(|offset| offset % 256 == 0)
    );
    assert_eq!(program.sources.len(), metadata.tensors.len());
}

#[test]
fn buffer_join_tile_maps_carry_destination_offsets() {
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![
            TensorContract::new(
                "a.cast",
                Expr::src("a"),
                vec![2],
                Encoding::Raw(DType::BF16),
            ),
            TensorContract::new(
                "b.cast",
                Expr::src("b"),
                vec![2],
                Encoding::Raw(DType::BF16),
            ),
            TensorContract::new(
                "joined",
                Expr::cat(0, vec![Expr::out("a.cast"), Expr::out("b.cast")]),
                vec![4],
                Encoding::Raw(DType::BF16),
            ),
        ],
    };

    let program = compile_load_plan(&metadata(), &contract, StorageTarget::default()).unwrap();
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
    // The public contract has no ephemeral declarations, so the two cast inputs
    // to the join are now named tensors and count toward persistent memory
    // instead of temporary peak.
    assert_eq!(program.memory.persistent_bytes, 16);
    assert_eq!(program.memory.temporary_peak_bytes, 0);
}

#[test]
fn direct_copy_lowers_to_identity_extent_write() {
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
        }],
    };

    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "runtime.weight",
            Expr::src("checkpoint.weight"),
            vec![2, 2],
            Encoding::Raw(DType::BF16),
        )],
    };

    let program = compile_load_plan(&metadata, &contract, StorageTarget::default()).unwrap();
    let writes: Vec<_> = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::ExtentWrite { id, source, dest } => Some((id, source, dest.offset)),
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
    // Row 2 of a [4, 8] int4 tensor: 8 elements at 4 bits is 4 bytes a row.
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "q.row",
            Expr::src("q").slice(0, 2, 1),
            vec![1, 8],
            Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
        )],
    };

    let program =
        compile_load_plan(&quant_metadata(), &contract, StorageTarget::default()).unwrap();
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

/// An expression bigger than the tensor it is declared for is refused.
///
/// This used to inject a gather whose byte offsets ran past its output, which a
/// contract cannot author: every destination offset the builder emits is
/// derived from a declared shape. The byte-level property did not go away, it
/// moved down to where the offsets are actually produced —
/// `reference::replay` refuses a lowering that writes outside its output, and
/// `tests/algebra.rs` pins that. What is checkable *here* is the declaration
/// that would have led to one.
#[test]
fn an_expression_may_not_outgrow_the_tensor_it_is_declared_for() {
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "q.bad",
            Expr::src("q").slice(0, 0, 1),
            vec![1, 4],
            Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
        )],
    };

    let err = compile_load_plan(&quant_metadata(), &contract, StorageTarget::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("declares shape [1, 4]"), "{err}");
    assert!(err.contains("yields [1, 8]"), "{err}");
}

#[test]
fn target_support_rejects_cuda_decode_at_compile_time() {
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "decoded",
            Expr::src("fp8"),
            vec![4],
            Encoding::Raw(DType::BF16),
        )],
    };

    let err = compile_load_plan(
        &quant_metadata(),
        &contract,
        StorageTarget {
            backend: BackendKind::Cuda,
            tile_map_mask: pie_loader::plan::CUDA_TILE_MAP_MASK,
            ..StorageTarget::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("does not support Decode"), "{err}");
}

#[test]
fn packed_quant_source_requires_exact_affine_size() {
    let mut metadata = quant_metadata();
    metadata.tensors.push(RawTensor {
        id: TensorId(5),
        name: "blocked".to_string(),
        file_id: FileId(0),
        file_offset: 240,
        span_bytes: 32,
        shape: vec![4, 8],
        encoding: Encoding::Quant(quant(QuantScheme::GgufQ4_0, DType::BF16)),
    });

    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "blocked",
            Expr::src("blocked"),
            vec![4, 8],
            Encoding::Quant(quant(QuantScheme::GgufQ4_0, DType::BF16)),
        )],
    };

    let err = compile_load_plan(&metadata, &contract, StorageTarget::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("non-affine physical size"));
}

#[test]
fn gpt_oss_native_mxfp4_default_abi_lowers_to_repack_tile_maps() {
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: pie_loader::plan::CUDA_TILE_MAP_MASK,
        native_mxfp4_moe: true,
        ..StorageTarget::default()
    };
    let metadata = gpt_oss_mxfp4_metadata();
    let contract = stored_contract("gpt_oss_native_mxfp4");
    let program = compile_load_plan(&metadata, &contract, target).unwrap();

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

/// GPT-OSS reads one `gate_up_proj` block twice, once for the even rows and
/// once for the odd ones. The compiler cannot fold the two reads into one, but
/// it can schedule them back to back so an executor that keeps the block it just
/// staged serves the second one without going back to the checkpoint.
#[test]
fn gpt_oss_native_mxfp4_schedules_shared_source_reads_together() {
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: pie_loader::plan::CUDA_TILE_MAP_MASK,
        native_mxfp4_moe: true,
        ..StorageTarget::default()
    };
    let program = compile_load_plan(
        &gpt_oss_mxfp4_metadata(),
        &stored_contract("gpt_oss_native_mxfp4"),
        target,
    )
    .unwrap();

    let reads: Vec<(FileId, u64, u64)> = program
        .schedule
        .iter()
        .filter_map(|id| program.instrs.iter().find(|instr| instr_id(instr) == *id))
        .filter_map(|instr| match instr {
            StorageInstr::TileMap {
                source: Some(source),
                inputs,
                ..
            } if inputs.is_empty() => Some((source.file_id, source.file_offset, source.span_bytes)),
            _ => None,
        })
        .collect();

    let mut shared = 0;
    for (position, read) in reads.iter().enumerate() {
        let readers = reads.iter().filter(|other| *other == read).count();
        if readers == 1 {
            continue;
        }
        shared += 1;
        let adjacent =
            (position > 0 && reads[position - 1] == *read) || reads.get(position + 1) == Some(read);
        assert!(
            adjacent,
            "read {read:?} at {position} is separated from the tile map that shares it: {reads:?}"
        );
    }
    assert_eq!(
        shared, 6,
        "expected the three gate/up pairs to share a source"
    );
}

#[test]
fn gpt_oss_native_mxfp4_tp_uses_row_and_column_offset_repack_contracts() {
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: pie_loader::plan::CUDA_TILE_MAP_MASK,
        tp_rank: 1,
        tp_size: 2,
        native_mxfp4_moe: true,
        ..StorageTarget::default()
    };
    let metadata = gpt_oss_mxfp4_metadata_with_intermediate(128);
    let contract = stored_contract("gpt_oss_native_mxfp4_tp1_of_2");
    let abi_repacks = contract
        .tensors
        .iter()
        .filter_map(|contract| match &contract.expr {
            pie_loader::contract::Expr::Repack { spec, .. } => Some(*spec),
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

    let program = compile_load_plan(&metadata, &contract, target).unwrap();

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
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: pie_loader::plan::CUDA_TILE_MAP_MASK,
        tp_rank: 1,
        tp_size: 2,
        preferred_alignment: 256,
        ..StorageTarget::default()
    };
    let metadata = nemotron_h_expert_metadata();
    let contract = stored_contract("nemotron_h_packed_experts_tp1_of_2");

    assert!(contract.tensors.iter().any(|contract| {
        contract.name == "language_model.backbone.layers.0.mixer.experts.up_proj.packed.weight"
            && contract.shape.as_deref() == Some(&[4, 3][..])
    }));
    assert!(contract.tensors.iter().any(|contract| {
        contract.name
            == "language_model.backbone.layers.0.mixer.experts.down_proj.packed.weight"
            && contract.shape.as_deref() == Some(&[6, 2][..])
            && matches!(&contract.expr, pie_loader::contract::Expr::Shard { axis, .. } if *axis == Axis(1))
    }));
    assert!(contract.tensors.iter().any(|contract| {
        contract.name == "language_model.backbone.layers.0.mixer.experts.0.up_proj.weight"
            && contract.shape.as_deref() == Some(&[2, 3][..])
    }));
    assert!(contract.tensors.iter().any(|contract| {
        contract.name == "language_model.backbone.layers.0.mixer.experts.1.down_proj.weight"
            && contract.shape.as_deref() == Some(&[3, 2][..])
    }));

    let program = compile_load_plan(&metadata, &contract, target).unwrap();
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
    assert!(
        writes
            .iter()
            .any(|(_, bytes, off)| *bytes == 12 && *off == 0)
    );
    assert!(
        writes
            .iter()
            .any(|(_, bytes, off)| *bytes == 12 && *off == 12)
    );

    // Each expert pack is one persistent backing buffer (2 experts × 12 B =
    // 24 B). The CUDA target reports a 256-byte operand alignment, so
    // cuBLAS(Lt) can select its fast `align8` kernels. Packing tightness is
    // internal to each backing and unaffected by the base alignment.
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

#[test]
fn a_contract_that_declares_a_name_twice_is_rejected() {
    let one = |name: &str| {
        pie_loader::contract::TensorContract::new(
            name,
            pie_loader::contract::Expr::src("a"),
            vec![2],
            Encoding::Raw(DType::F32),
        )
    };
    let contract = pie_loader::contract::ModelContract {
        alignment: 256,
        tensors: vec![one("dup"), one("dup")],
    };
    let error = compile_load_plan(&metadata(), &contract, StorageTarget::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("declares 'dup' twice"), "{error}");
}

#[test]
fn a_contract_whose_declared_shape_is_wrong_is_rejected() {
    let contract = pie_loader::contract::ModelContract {
        alignment: 256,
        tensors: vec![pie_loader::contract::TensorContract::new(
            "a",
            pie_loader::contract::Expr::src("a"),
            vec![4],
            Encoding::Raw(DType::F32),
        )],
    };
    let error = compile_load_plan(&metadata(), &contract, StorageTarget::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("declares shape [4]"), "{error}");
    assert!(error.contains("yields [2]"), "{error}");
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
            },
            RawTensor {
                id: TensorId(3),
                name: "fp8".to_string(),
                file_id: FileId(0),
                file_offset: 216,
                span_bytes: 4,
                shape: vec![4],
                encoding: Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::BF16)),
            },
            RawTensor {
                id: TensorId(4),
                name: "q_odd".to_string(),
                file_id: FileId(0),
                file_offset: 220,
                span_bytes: 14,
                shape: vec![4, 7],
                encoding: Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
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
    }
}

fn sized_raw(
    id: u32,
    name: &str,
    offset: u64,
    span_bytes: u64,
    shape: &[i64],
    dtype: DType,
) -> RawTensor {
    RawTensor {
        span_bytes,
        ..raw(id, name, offset, shape, dtype)
    }
}

fn quant(scheme: QuantScheme, dtype: DType) -> QuantSpec {
    QuantSpec {
        scheme,
        logical_dtype: dtype,
        bits_per_element: scheme.default_bits(),
        group_size: scheme.default_group_size(),
        channel_axis: None,
    }
}

fn tensor_bytes(shape: &[i64], dtype: DType) -> u64 {
    shape
        .iter()
        .fold(dtype.bytes(), |acc, dim| acc * u64::try_from(*dim).unwrap())
}

// ── MLA weight fusion tests ─────────────────────────────────────────

#[test]
fn mla_q_kv_a_fusion_produces_joined_tensor() {
    let h = 128i64;
    let q_lora = 32i64;
    let kv_lora_rope = 16i64;
    let mut offset = 0u64;
    let mut tensors = Vec::new();
    let specs: Vec<(u32, &str, Vec<i64>)> = vec![
        (
            0,
            "model.layers.0.self_attn.q_a_proj.weight",
            vec![q_lora, h],
        ),
        (
            1,
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            vec![kv_lora_rope, h],
        ),
        (
            2,
            "model.layers.0.self_attn.q_a_layernorm.weight",
            vec![q_lora],
        ),
        (
            3,
            "model.layers.0.self_attn.q_b_proj.weight",
            vec![64, q_lora],
        ),
        (
            4,
            "model.layers.0.self_attn.kv_a_layernorm.weight",
            vec![12],
        ),
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
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: pie_loader::plan::CUDA_TILE_MAP_MASK,
        ..StorageTarget::default()
    };
    let contract = stored_contract("kimi_k2_mla_fusion");
    let program = compile_load_plan(&meta, &contract, target).unwrap();
    let summary = pie_loader::dump::describe(&program);

    // The fusion should have joined q_a_proj + kv_a_proj into one tensor
    let has_fused = program
        .tensors
        .iter()
        .any(|t| t.name.contains("q_kv_a_proj.fused"));
    assert!(
        has_fused,
        "Expected fused q_kv_a_proj tensor; summary: {summary}\ntensors: {:?}",
        program.tensors.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    // The fused tensor should have rows = q_lora + kv_lora_rope
    let fused = program
        .tensors
        .iter()
        .find(|t| t.name.contains("q_kv_a_proj.fused"))
        .unwrap();
    assert_eq!(fused.shape[0], q_lora + kv_lora_rope);
    assert_eq!(fused.shape[1], h);

    // Summary should show the plan compiled successfully
    assert!(!program.tensors.is_empty());
    assert!(
        program
            .instrs
            .iter()
            .any(|instr| matches!(instr, StorageInstr::Finalize { .. })),
        "{summary}"
    );
}

fn instr_id(instr: &StorageInstr) -> pie_loader::types::InstrId {
    match instr {
        StorageInstr::Allocate { id, .. }
        | StorageInstr::Fill { id, .. }
        | StorageInstr::ExtentWrite { id, .. }
        | StorageInstr::BulkExtentWrite { id, .. }
        | StorageInstr::TileMap { id, .. }
        | StorageInstr::CreateView { id, .. }
        | StorageInstr::Finalize { id, .. } => *id,
    }
}

/// A block-scaled FP8 source names its scale tensor on the instruction.
///
/// The executor used to rebuild the name by appending `_scale_inv` and looking
/// it up (`driver/cuda/src/loader/transcode_engine.hpp`), which is the same
/// guess-what-the-loader-decided pattern `attachments` removed from the output
/// side. The loader has the tensor table, so it answers; this pins that the
/// answer travels.
#[test]
fn a_block_scaled_fp8_source_carries_its_scale_tensor() {
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            sized_raw(0, "w.weight", 0, 4096, &[64, 64], DType::F8E4M3),
            sized_raw(1, "w.weight_scale_inv", 4096, 4, &[1, 1], DType::F32),
        ],
    };

    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: u32::MAX,
        ..StorageTarget::default()
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "runtime.w",
            Expr::src("w.weight"),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Mxfp4E2M1E8M0, DType::BF16)),
        )],
    };
    let program = compile_load_plan(&metadata, &contract, target).unwrap();
    let encodes: Vec<_> = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap {
                kind: TileMapKind::Encode,
                transform,
                ..
            } => Some(transform.metadata_source),
            _ => None,
        })
        .collect();
    assert_eq!(encodes, vec![Some(TensorId(1))]);
}

/// ...and a source with no such sibling says so, rather than naming tensor 0.
#[test]
fn a_source_without_a_scale_sibling_names_none() {
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![sized_raw(0, "w.weight", 0, 8192, &[64, 64], DType::BF16)],
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: u32::MAX,
        ..StorageTarget::default()
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "runtime.w",
            Expr::src("w.weight"),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Mxfp4E2M1E8M0, DType::BF16)),
        )],
    };
    let program = compile_load_plan(&metadata, &contract, target).unwrap();
    for instr in &program.instrs {
        if let StorageInstr::TileMap { transform, .. } = instr {
            assert_eq!(transform.metadata_source, None);
        }
    }
}

#[test]
fn a_padded_head_dim_zeroes_the_buffer_before_it_writes_the_rows() {
    // `driver/cuda/src/model/llama_like/llama_like.cpp:640` wants this: pad Q/K/V
    // up to a head_dim the attention kernel takes. Until `Fill` existed the
    // compiler priced the pad (spec.md §3.3) and then refused to build it.
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1024,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "q_proj.weight".to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 32,
            shape: vec![4, 4],
            encoding: Encoding::Raw(DType::BF16),
        }],
    };

    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "q_proj.weight",
            Expr::src("q_proj.weight").pad(1, 0, 1),
            vec![4, 5],
            Encoding::Raw(DType::BF16),
        )],
    };

    let program = compile_load_plan(&metadata, &contract, StorageTarget::default()).unwrap();

    let fills: Vec<_> = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::Fill { id, buffer } => Some((*id, *buffer)),
            _ => None,
        })
        .collect();
    assert_eq!(fills.len(), 1, "one fill, not one per band");
    let (fill_id, filled) = fills[0];

    // Four rows, because the destination stride is wider than the row and
    // `fold` will not skip in the destination. spec.md §3.3 prices this at 5.
    let writes: Vec<_> = program
        .instrs
        .iter()
        .filter(|instr| {
            matches!(
                instr,
                StorageInstr::ExtentWrite { .. } | StorageInstr::BulkExtentWrite { .. }
            )
        })
        .collect();
    assert_eq!(writes.len(), 4);

    // The fill has to run before every one of them, or it erases what they wrote.
    let at = |want| program.schedule.iter().position(|id| *id == want).unwrap();
    let fill_at = at(fill_id);
    for write in &writes {
        assert!(fill_at < at(instr_id(write)), "the fill must come first");
    }

    // The padded column is real memory that no source covers.
    assert_eq!(program.memory.checkpoint_read_bytes, 32);
    assert_eq!(program.memory.device_write_bytes, 32);
    assert_eq!(program.memory.persistent_bytes, 40);
    assert_eq!(program.buffer(filled).unwrap().bytes, 40);
}

/// A block scale is bytes in the file and an exponent to the GEMM.
///
/// DeepSeek-V4 pairs FP8-E4M3 weights with OCP Microscaling E8M0 scales --
/// a combination `QuantScheme::Mxfp4E2M1E8M0` cannot name, because that symbol
/// bundles the element format together with the scale format. The driver used
/// to bridge the gap by copying the scales to the host and running `ldexpf`
/// over them at bind time.
///
/// No new expression is needed for this. `Bitcast` names the reading of the
/// bytes and the declaration names the type wanted, so the existing
/// dtype-mismatch rule inserts the cast -- which is the whole argument for
/// `E8M0` being a dtype rather than an arithmetic escape hatch.
#[test]
fn an_e8m0_block_scale_read_as_fp32_lowers_to_a_cast() {
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![sized_raw(0, "w.scale", 0, 64, &[8, 8], DType::U8)],
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: u32::MAX,
        ..StorageTarget::default()
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "runtime.w.scale",
            Expr::Bitcast {
                src: Box::new(Expr::src("w.scale")),
                out: TensorType {
                    shape: vec![8, 8],
                    encoding: Encoding::Raw(DType::E8M0),
                },
            },
            vec![8, 8],
            Encoding::Raw(DType::F32),
        )],
    };
    let program = compile_load_plan(&metadata, &contract, target).unwrap();
    let casts = program
        .instrs
        .iter()
        .filter(|instr| {
            matches!(
                instr,
                StorageInstr::TileMap {
                    kind: TileMapKind::Cast,
                    ..
                }
            )
        })
        .count();
    assert_eq!(casts, 1, "expected exactly one Cast, got plan {program:#?}");
}

// ── quant attachments ──────────────────────────────────
//
// A quantized weight and its scales are two runtime tensors, and the driver has
// to know they belong together. The loader used to work that out after the fact,
// by matching `_scale_inv` / `.scale` suffixes over the finished tensor table
// with the group size hardcoded to 128 beside them, in
// `plan::derive_quant_attachments`. Every entry is now recorded by whoever
// declared the scale tensor, at the point of declaring it — which is why these
// exercise `compile` rather than a name-matching function: there is no longer
// anything to call in between.

fn scale_target() -> StorageTarget {
    StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: u32::MAX,
        ..StorageTarget::default()
    }
}

/// The MXFP4 GEMM asserts its scale operand is U8, so a plan that asks the
/// driver to expand these to F32 makes the kernel reject the load.
#[test]
fn scales_the_loader_writes_while_encoding_mxfp4_stay_raw_e8m0() {
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![sized_raw(0, "w.weight", 0, 8192, &[64, 64], DType::BF16)],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "runtime.w",
            Expr::src("w.weight"),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Mxfp4E2M1E8M0, DType::BF16)),
        )],
    };
    let program = compile_load_plan(&metadata, &contract, scale_target()).unwrap();
    assert_eq!(program.attachments.len(), 1, "{:#?}", program.attachments);
    let attach = program.attachments[0];
    assert_eq!(attach.scale_form, ScaleForm::RawE8M0);
    assert_eq!(attach.granularity, QuantGranularity::PerGroup);
    assert_eq!(attach.group_size, 32);
    // Both halves name real entries, which is the part the name matching could
    // get wrong without anything noticing.
    assert_eq!(program.tensors[attach.tensor.0 as usize].name, "runtime.w");
    assert_eq!(
        program.tensors[attach.scale_tensor.0 as usize].name,
        "runtime.w_scale"
    );
}

/// The same, for the per-channel schemes: one F32 factor per output row.
#[test]
fn scales_the_loader_writes_while_encoding_fp8_are_f32_factors() {
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![sized_raw(0, "w.weight", 0, 8192, &[64, 64], DType::BF16)],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "runtime.w",
            Expr::src("w.weight"),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::BF16)),
        )],
    };
    let program = compile_load_plan(&metadata, &contract, scale_target()).unwrap();
    assert_eq!(program.attachments.len(), 1, "{:#?}", program.attachments);
    assert_eq!(program.attachments[0].scale_form, ScaleForm::F32Factors);
    assert_eq!(
        program.attachments[0].granularity,
        QuantGranularity::PerChannel
    );
}

/// Block-scaled FP8 (DeepSeek-V3 and its descendants) arrives already
/// quantized: the loader writes no scales, so the contract states the pairing.
///
/// This is the case the suffix matching existed for, and the case it was worst
/// at. It tried `{name}_scale_inv` and then `{base}.scale`, and hardcoded
/// `group_size: 128` — while the only authoring site in the tree,
/// `dsv4_block_scales_to_fp32`, publishes `<base>.scale` and knows the real
/// block size. Note the 64 here: a checkpoint that blocks by anything other
/// than 128 used to be silently mislabelled.
#[test]
fn scales_the_checkpoint_shipped_are_paired_by_the_contract() {
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            sized_raw(0, "w.weight", 0, 4096, &[64, 64], DType::F8E4M3),
            sized_raw(1, "w.scale", 4096, 4, &[1, 1], DType::F32),
        ],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![
            TensorContract::new(
                "runtime.w",
                Expr::src("w.weight"),
                vec![64, 64],
                Encoding::Raw(DType::F8E4M3),
            ),
            TensorContract::new(
                "runtime.w_scale",
                Expr::src("w.scale"),
                vec![1, 1],
                Encoding::Raw(DType::F32),
            )
            .scaling(Scales {
                of: "runtime.w".to_string(),
                granularity: QuantGranularity::PerGroup,
                group_size: 64,
                channel_axis: 0,
                form: ScaleForm::F32Factors,
            }),
        ],
    };
    let program = compile_load_plan(&metadata, &contract, scale_target()).unwrap();
    assert_eq!(program.attachments.len(), 1, "{:#?}", program.attachments);
    let attach = program.attachments[0];
    assert_eq!(attach.group_size, 64);
    assert_eq!(attach.scale_form, ScaleForm::F32Factors);
    assert_eq!(program.tensors[attach.tensor.0 as usize].name, "runtime.w");
    assert_eq!(
        program.tensors[attach.scale_tensor.0 as usize].name,
        "runtime.w_scale"
    );
}

/// A contract that names a tensor no earlier entry declares is rejected.
///
/// The suffix matching could not fail: a name that resolved to nothing produced
/// no attachment, and the plan came out of the compiler silently missing the
/// metadata its kernels would go looking for at bind time.
#[test]
fn scales_named_for_a_weight_the_loader_quantizes_are_a_contract_error() {
    // The loader writes this weight's scales itself while encoding it, and
    // states that pairing. A contract that names a second set would attach
    // quant metadata to one weight twice, which the driver discovers only at
    // load time.
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            sized_raw(0, "w.weight", 0, 8192, &[64, 64], DType::BF16),
            sized_raw(1, "w.weight_scale_inv", 8192, 4, &[1, 1], DType::F32),
        ],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![
            TensorContract::new(
                "runtime.w",
                Expr::src("w.weight"),
                vec![64, 64],
                Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::BF16)),
            ),
            TensorContract::new(
                "runtime.w_shipped_scales",
                Expr::src("w.weight_scale_inv"),
                vec![1, 1],
                Encoding::Raw(DType::F32),
            )
            .scaling(Scales {
                of: "runtime.w".to_string(),
                granularity: QuantGranularity::PerGroup,
                group_size: 128,
                channel_axis: 0,
                form: ScaleForm::F32Factors,
            }),
        ],
    };
    let err = compile_load_plan(&metadata, &contract, scale_target())
        .unwrap_err()
        .to_string();
    assert!(err.contains("already has scales"), "{err}");
}

#[test]
fn scales_naming_an_undeclared_tensor_are_a_contract_error() {
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![sized_raw(0, "w.scale", 0, 4, &[1, 1], DType::F32)],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![
            TensorContract::new(
                "runtime.w_scale",
                Expr::src("w.scale"),
                vec![1, 1],
                Encoding::Raw(DType::F32),
            )
            .scaling(Scales {
                of: "runtime.w".to_string(),
                granularity: QuantGranularity::PerGroup,
                group_size: 128,
                channel_axis: 0,
                form: ScaleForm::F32Factors,
            }),
        ],
    };
    let error = compile_load_plan(&metadata, &contract, scale_target())
        .unwrap_err()
        .to_string();
    assert!(error.contains("runtime.w"), "{error}");
    assert!(error.contains("the contract does not declare"), "{error}");
}

/// Scales may be declared before the tensor they scale.
///
/// `dsv4_block_scales_to_fp32` runs before `author_dense_contract`, so the real
/// contract declares every block scale ahead of its weight. Resolving `of`
/// against earlier entries only — the rule `Expr::Out` follows — would reject
/// every DeepSeek-V4 contract in the tree.
#[test]
fn scales_may_name_a_tensor_declared_after_them() {
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            sized_raw(0, "w.scale", 0, 4, &[1, 1], DType::F32),
            sized_raw(1, "w.weight", 4096, 4096, &[64, 64], DType::F8E4M3),
        ],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![
            TensorContract::new(
                "runtime.w_scale",
                Expr::src("w.scale"),
                vec![1, 1],
                Encoding::Raw(DType::F32),
            )
            .scaling(Scales {
                of: "runtime.w".to_string(),
                granularity: QuantGranularity::PerGroup,
                group_size: 64,
                channel_axis: 0,
                form: ScaleForm::F32Factors,
            }),
            TensorContract::new(
                "runtime.w",
                Expr::src("w.weight"),
                vec![64, 64],
                Encoding::Raw(DType::F8E4M3),
            ),
        ],
    };
    let program = compile_load_plan(&metadata, &contract, scale_target()).unwrap();
    assert_eq!(program.attachments.len(), 1, "{:#?}", program.attachments);
    let attach = program.attachments[0];
    assert_eq!(program.tensors[attach.tensor.0 as usize].name, "runtime.w");
    assert_eq!(
        program.tensors[attach.scale_tensor.0 as usize].name,
        "runtime.w_scale"
    );
}
