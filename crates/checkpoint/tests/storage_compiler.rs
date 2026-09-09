fn stored_contract(name: &str) -> ModelContract {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/contracts")
        .join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("{name}: cannot read {}: {err}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("{name}: parsing: {err}"))
}
use checkpoint::contract::{Expr, ModelContract, Scales, TensorContract, TensorType, UnaryOp};
use checkpoint::file::{File, Metadata, RawTensor};
use checkpoint::plan::compile as compile_load_plan;
use checkpoint::plan::{LoadPlan, StorageInstr, StorageTarget, TileMapKind};
use checkpoint::types::{
    Axis, BackendKind, CheckpointFormat, DType, Encoding, FileId, QuantGranularity, QuantScheme,
    QuantSpec, RepackLayout, ScaleForm, TensorId,
};

#[test]
fn storage_compiler_every_case() {
    metal_qwen35_schema_emits_canonical_affine_u4_arena();
    buffer_join_tile_maps_carry_destination_offsets();
    direct_copy_lowers_to_identity_extent_write();
    packed_quant_row_select_uses_byte_exact_offsets();
    an_expression_may_not_outgrow_the_tensor_it_is_declared_for();
    target_support_rejects_cuda_decode_at_compile_time();
    a_quantized_tensor_is_re_encoded_through_a_decoded_intermediate();
    a_serving_target_refuses_the_encode_a_conversion_target_runs();
    a_quantized_tensor_may_not_be_cast_straight_to_another_scheme();
    packed_quant_source_requires_exact_affine_size();
    gpt_oss_native_mxfp4_default_abi_lowers_to_repack_tile_maps();
    a_repack_declaration_is_checked_against_its_transform();
    gpt_oss_native_mxfp4_reads_each_interleaved_half_once();
    gpt_oss_native_mxfp4_tp_resolves_the_rank_from_the_target();
    nemotron_h_default_abi_packs_experts_and_exposes_views();
    a_contract_that_declares_a_name_twice_is_rejected();
    a_contract_whose_declared_shape_is_wrong_is_rejected();
    a_head_boundary_shard_is_one_contiguous_run();
    a_head_boundary_shard_rejects_an_indivisible_world();
    a_scale_by_zero_is_rejected_at_compile_time();
    a_scale_by_a_non_finite_factor_is_rejected_at_compile_time();
    a_scale_over_integer_elements_is_rejected_at_compile_time();
    a_scale_over_quantized_elements_is_rejected_at_compile_time();
    a_scale_whose_declared_shape_is_wrong_is_rejected();
    every_path_names_the_contract_its_error_came_from();
    a_block_scaled_dequant_is_one_scale_with_its_factors_as_an_operand();
    a_sharded_block_scaled_dequant_scales_only_its_own_rank();
    a_scale_by_a_tensor_no_contract_declares_is_rejected();
    a_scale_blocks_every_axis_the_factors_divide();
    a_scale_by_factors_of_a_different_rank_is_rejected();
    a_scale_by_factors_that_do_not_divide_the_payload_is_rejected();
    a_scale_by_one_factor_per_element_is_rejected();
    a_scale_by_an_undeclared_expression_is_rejected();
    mla_q_kv_a_fusion_produces_joined_tensor();
    a_block_scaled_fp8_source_carries_its_scale_tensor();
    a_source_without_a_scale_sibling_names_none();
    a_padded_head_dim_zeroes_the_buffer_before_it_writes_the_rows();
    a_padded_head_dim_materializes_zeros_where_no_source_covers();
    an_e8m0_block_scale_read_as_fp32_lowers_to_a_cast();
    scales_the_loader_writes_while_encoding_mxfp4_stay_raw_e8m0();
    scales_the_loader_writes_while_encoding_fp8_are_f32_factors();
    scales_the_checkpoint_shipped_are_paired_by_the_contract();
    scales_named_for_a_weight_the_loader_quantizes_are_a_contract_error();
    scales_naming_an_undeclared_tensor_are_a_contract_error();
    scales_may_name_a_tensor_declared_after_them();
    a_rank_3_bank_encodes_and_its_scales_keep_the_expert_axis();
    an_encode_that_cannot_place_its_scales_is_refused();
    an_encode_whose_columns_do_not_fill_a_block_is_refused();
    an_encode_into_a_scheme_no_kernel_writes_is_refused();
    re_encoding_one_quantized_scheme_as_another_is_refused();
    a_declaration_that_disagrees_with_its_expression_is_a_mistake_not_a_kernel();
    the_same_pair_with_the_cast_written_down_encodes();
    a_unary_takes_the_logarithm_of_a_negated_plane();
    a_unary_refuses_an_element_outside_its_domain();
    a_serving_target_refuses_the_unary_a_conversion_target_runs();
    the_tiled_repack_is_the_documented_permutation_at_every_shipped_shape();
}

fn metal_qwen35_schema_emits_canonical_affine_u4_arena() {
    let specs = [
        ("lm_head.weight", vec![2, 8], DType::U32),
        ("lm_head.scales", vec![2, 1], DType::Bf16),
        ("lm_head.biases", vec![2, 1], DType::Bf16),
        ("model.language_model.norm.weight", vec![64], DType::Bf16),
        (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            vec![64, 8],
            DType::U32,
        ),
        (
            "model.language_model.layers.0.self_attn.q_proj.scales",
            vec![64, 1],
            DType::Bf16,
        ),
        (
            "model.language_model.layers.0.self_attn.q_proj.biases",
            vec![64, 1],
            DType::Bf16,
        ),
        (
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            vec![16, 64],
            DType::Bf16,
        ),
        ("model.visual.patch.weight", vec![1], DType::Bf16),
        ("mtp.fc.weight", vec![1], DType::Bf16),
    ];
    let mut offset = 0u64;
    let tensors = specs
        .into_iter()
        .enumerate()
        .map(|(index, (name, shape, dtype))| {
            let span_bytes = shape.iter().product::<i64>() as u64 * dtype.bytes_ceil();
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
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: offset,
            format: CheckpointFormat::Safetensors,
        }],
        tensors,
    };
    let target = StorageTarget {
        backend: BackendKind::Metal,
        tile_map_mask: checkpoint::plan::METAL_TILE_MAP_MASK,
        max_tile_bytes: 64 << 20,
        preferred_alignment: 256,
        ..StorageTarget::default()
    };
    let affine_u4 = |group_size: u32| {
        Encoding::Quant(
            QuantSpec {
                scheme: QuantScheme::MlxAffineU4,
                logical_dtype: DType::Bf16,
                bits_per_element: 4,
                group_size,
                channel_axis: Some(Axis(1)),
            }
            .normalized(),
        )
    };
    let packed = |source: &str, output: &str, rows: i64, cols: i64| {
        let ty = checkpoint::contract::TensorType::new(vec![rows, cols], affine_u4(64));
        TensorContract::new(
            output.to_string(),
            Expr::src(source.to_string()).transmute(ty),
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
                Encoding::Raw(DType::Bf16),
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
                Encoding::Raw(DType::Bf16),
            ),
        ],
        groups: Vec::new(),
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

fn buffer_join_tile_maps_carry_destination_offsets() {
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![
            TensorContract::new(
                "a.cast",
                Expr::src("a").cast(Encoding::Raw(DType::Bf16)),
                vec![2],
                Encoding::Raw(DType::Bf16),
            ),
            TensorContract::new(
                "b.cast",
                Expr::src("b").cast(Encoding::Raw(DType::Bf16)),
                vec![2],
                Encoding::Raw(DType::Bf16),
            ),
            TensorContract::new(
                "joined",
                Expr::concat(0, vec![Expr::out("a.cast"), Expr::out("b.cast")]),
                vec![4],
                Encoding::Raw(DType::Bf16),
            ),
        ],
        groups: Vec::new(),
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
    assert_eq!(program.memory.persistent_bytes, 16);
    assert_eq!(program.memory.temporary_peak_bytes, 0);
}

fn direct_copy_lowers_to_identity_extent_write() {
    let metadata = Metadata {
        files: vec![File {
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
            encoding: Encoding::Raw(DType::Bf16),
        }],
    };

    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "runtime.weight",
            Expr::src("checkpoint.weight"),
            vec![2, 2],
            Encoding::Raw(DType::Bf16),
        )],
        groups: Vec::new(),
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

fn packed_quant_row_select_uses_byte_exact_offsets() {
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "q.row",
            Expr::src("q").slice(0, 2, 1),
            vec![1, 8],
            Encoding::Quant(quant(QuantScheme::AwqInt4, DType::Bf16)),
        )],
        groups: Vec::new(),
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

fn an_expression_may_not_outgrow_the_tensor_it_is_declared_for() {
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "q.bad",
            Expr::src("q").slice(0, 0, 1),
            vec![1, 4],
            Encoding::Quant(quant(QuantScheme::AwqInt4, DType::Bf16)),
        )],
        groups: Vec::new(),
    };

    let err = compile_load_plan(&quant_metadata(), &contract, StorageTarget::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("declares shape [1, 4]"), "{err}");
    assert!(err.contains("yields [1, 8]"), "{err}");
}

fn target_support_rejects_cuda_decode_at_compile_time() {
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "decoded",
            Expr::src("fp8").cast(Encoding::Raw(DType::Bf16)),
            vec![4],
            Encoding::Raw(DType::Bf16),
        )],
        groups: Vec::new(),
    };

    let err = compile_load_plan(
        &quant_metadata(),
        &contract,
        StorageTarget {
            backend: BackendKind::Cuda,
            tile_map_mask: checkpoint::plan::CUDA_TILE_MAP_MASK,
            ..StorageTarget::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("does not support Decode"), "{err}");
}

fn a_quantized_tensor_is_re_encoded_through_a_decoded_intermediate() {
    let int8 = Encoding::Quant(QuantSpec {
        scheme: QuantScheme::Int8Symmetric,
        logical_dtype: DType::Bf16,
        bits_per_element: 8,
        group_size: 32,
        channel_axis: Some(Axis(1)),
    });
    let mut contract = block_scaled_contract("scales", "s", vec![4, 1]);
    contract.tensors[1] = contract.tensors[1].clone().internal();
    contract.tensors.push(TensorContract::new(
        "w_int8",
        Expr::out("w").cast(int8.clone()),
        vec![4, 32],
        int8,
    ));

    let target = StorageTarget {
        tile_map_mask: checkpoint::plan::CONVERT_TILE_MAP_MASK,
        ..StorageTarget::default()
    };
    let plan = compile_load_plan(&block_scaled_metadata(), &contract, target)
        .expect("the documented two-step must compile");

    let kinds: Vec<TileMapKind> = plan
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap { kind, .. } => Some(*kind),
            _ => None,
        })
        .collect();
    assert_eq!(
        kinds,
        vec![TileMapKind::Scale, TileMapKind::Encode],
        "decode then encode, each its own kernel"
    );

    assert_eq!(plan.memory.checkpoint_read_bytes, 68);
    let bound: Vec<&str> = plan
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::Finalize { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        bound,
        vec!["scales", "w_int8_scale_inv", "w_int8"],
        "the intermediate is not bound"
    );
}

fn a_serving_target_refuses_the_encode_a_conversion_target_runs() {
    let int8 = Encoding::Quant(QuantSpec {
        scheme: QuantScheme::Int8Symmetric,
        logical_dtype: DType::Bf16,
        bits_per_element: 8,
        group_size: 32,
        channel_axis: Some(Axis(1)),
    });
    let mut contract = block_scaled_contract("scales", "s", vec![4, 1]);
    contract.tensors[1] = contract.tensors[1].clone().internal();
    contract.tensors.push(TensorContract::new(
        "w_int8",
        Expr::out("w").cast(int8.clone()),
        vec![4, 32],
        int8,
    ));

    for backend in [
        BackendKind::Cuda,
        BackendKind::Metal,
        BackendKind::Vulkan,
        BackendKind::Wgpu,
        BackendKind::Unknown,
    ] {
        let refused = compile_load_plan(
            &block_scaled_metadata(),
            &contract,
            StorageTarget::for_backend(backend, 0, 1),
        )
        .expect_err("a serving plan does not convert");
        let said = refused.to_string();
        assert!(
            said.contains("pie model import"),
            "{backend:?} refuses the encode without naming the command that \
             runs it: {said}"
        );
    }

    let converting = compile_load_plan(
        &block_scaled_metadata(),
        &contract,
        StorageTarget {
            tile_map_mask: checkpoint::plan::CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        },
    )
    .expect("conversion still runs it");
    let encodes = converting
        .instrs
        .iter()
        .filter(|instr| {
            matches!(
                instr,
                StorageInstr::TileMap {
                    kind: TileMapKind::Encode,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        encodes, 1,
        "the conversion plan carries the encode, so `pie model import` writes \
         the codes the load now insists on"
    );
}

fn a_quantized_tensor_may_not_be_cast_straight_to_another_scheme() {
    let err = compile_load_plan(
        &quant_metadata(),
        &ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src("q").cast(Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::Bf16))),
                vec![4, 8],
                Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::Bf16)),
            )],
            groups: Vec::new(),
        },
        StorageTarget::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("no kernel does that in one step"), "{err}");
}

fn packed_quant_source_requires_exact_affine_size() {
    let mut metadata = quant_metadata();
    metadata.tensors.push(RawTensor {
        id: TensorId(5),
        name: "blocked".to_string(),
        file_id: FileId(0),
        file_offset: 240,
        span_bytes: 32,
        shape: vec![4, 8],
        encoding: Encoding::Quant(quant(QuantScheme::GgufQ4_0, DType::Bf16)),
    });

    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "blocked",
            Expr::src("blocked"),
            vec![4, 8],
            Encoding::Quant(quant(QuantScheme::GgufQ4_0, DType::Bf16)),
        )],
        groups: Vec::new(),
    };

    let err = compile_load_plan(&metadata, &contract, StorageTarget::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("non-affine physical size"));
}

fn gpt_oss_native_mxfp4_default_abi_lowers_to_repack_tile_maps() {
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: checkpoint::plan::CUDA_TILE_MAP_MASK | checkpoint::plan::TILE_MAP_REPACK,
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
    assert_eq!(repacks.len(), 6);
    assert!(repacks.iter().any(|spec| {
        spec.repack
            .is_some_and(|r| r.layout == RepackLayout::MarlinMxfp4Weight)
    }));
    assert!(repacks.iter().any(|spec| {
        spec.repack
            .is_some_and(|r| r.layout == RepackLayout::MarlinMxfp4Scale)
    }));
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

fn a_repack_declaration_is_checked_against_its_transform() {
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: checkpoint::plan::CUDA_TILE_MAP_MASK | checkpoint::plan::TILE_MAP_REPACK,
        native_mxfp4_moe: true,
        ..StorageTarget::default()
    };
    let mut contract = stored_contract("gpt_oss_native_mxfp4");
    let repacked = contract
        .tensors
        .iter_mut()
        .find(|tensor| matches!(tensor.expr, Expr::Repack { .. }))
        .expect("the gpt-oss contract repacks");
    let name = repacked.name.clone();
    repacked
        .shape
        .as_mut()
        .expect("a repack declaration states its shape")[1] += 7;

    let error = compile_load_plan(&gpt_oss_mxfp4_metadata(), &contract, target)
        .unwrap_err()
        .to_string();
    assert!(error.contains(&name), "{error}");
    assert!(error.contains("declares shape"), "{error}");
}

fn gpt_oss_native_mxfp4_reads_each_interleaved_half_once() {
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: checkpoint::plan::CUDA_TILE_MAP_MASK | checkpoint::plan::TILE_MAP_REPACK,
        native_mxfp4_moe: true,
        ..StorageTarget::default()
    };
    let program = compile_load_plan(
        &gpt_oss_mxfp4_metadata(),
        &stored_contract("gpt_oss_native_mxfp4"),
        target,
    )
    .unwrap();

    let blocks = TensorId(10);
    let mut halves: Vec<(u64, u64)> = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap {
                source: Some(source),
                ..
            } if source.tensor_id == blocks => Some((source.file_offset, source.span_bytes)),
            _ => None,
        })
        .collect();
    halves.sort_unstable();
    assert_eq!(halves.len(), 2, "{halves:?}");
    assert_eq!(halves[0].1, halves[1].1, "the halves are the same size");
    assert_eq!(halves[1].0 - halves[0].0, 32, "one row apart");
    assert_eq!(halves[0].1 + halves[1].1, 8192, "the block, once");
}

fn gpt_oss_native_mxfp4_tp_resolves_the_rank_from_the_target() {
    let metadata = gpt_oss_mxfp4_metadata_with_intermediate(128);
    let contract = stored_contract("gpt_oss_native_mxfp4_tp1_of_2");

    let repacks = contract
        .tensors
        .iter()
        .filter(|tensor| matches!(tensor.expr, checkpoint::contract::Expr::Repack { .. }))
        .count();
    assert_eq!(
        repacks, 6,
        "a weight and a scale for each of gate, up and down -- the biases are affine"
    );
    assert!(
        contract.tensors.iter().all(|tensor| !matches!(
            &tensor.expr,
            checkpoint::contract::Expr::Repack { src, .. } if matches!(**src, checkpoint::contract::Expr::Src(_))
        )),
        "a repack whose operand is a bare source has nowhere to have put the shard"
    );

    let plan_at = |rank: u32| {
        let target = StorageTarget {
            backend: BackendKind::Cuda,
            tile_map_mask: checkpoint::plan::CUDA_TILE_MAP_MASK | checkpoint::plan::TILE_MAP_REPACK,
            tp_rank: rank,
            tp_size: 2,
            native_mxfp4_moe: true,
            ..StorageTarget::default()
        };
        let program = compile_load_plan(&metadata, &contract, target).unwrap();
        let mut reads: Vec<(u32, u64, u64)> = program
            .instrs
            .iter()
            .filter_map(|instr| match instr {
                StorageInstr::TileMap {
                    source: Some(source),
                    ..
                } => Some((source.tensor_id.0, source.file_offset, source.span_bytes)),
                _ => None,
            })
            .collect();
        reads.sort_unstable();
        (reads, program.memory.checkpoint_read_bytes)
    };

    let (rank0, bytes0) = plan_at(0);
    let (rank1, bytes1) = plan_at(1);
    assert_ne!(
        rank0, rank1,
        "the two ranks must not read the same bytes: {rank0:?}"
    );
    assert_eq!(bytes0, bytes1, "and each rank must read the same volume");

    let band_start = |reads: &[(u32, u64, u64)]| {
        reads
            .iter()
            .filter(|(id, ..)| *id == 10)
            .map(|(_, offset, _)| *offset)
            .min()
            .unwrap()
    };
    assert_eq!(band_start(&rank0), 0);
    assert_eq!(band_start(&rank1), 128 * 32);
}

fn nemotron_h_default_abi_packs_experts_and_exposes_views() {
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: checkpoint::plan::CUDA_TILE_MAP_MASK,
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
            && contract.shape.as_deref() == Some(&[6, 4][..])
            && matches!(&contract.expr, checkpoint::contract::Expr::Shard { axis, .. } if *axis == Axis(1))
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

    assert_eq!(program.memory.checkpoint_read_bytes, 48);
    assert_eq!(program.memory.device_write_bytes, 48);
    assert_eq!(program.memory.persistent_bytes, 280);
}

fn a_contract_that_declares_a_name_twice_is_rejected() {
    let one = |name: &str| {
        checkpoint::contract::TensorContract::new(
            name,
            checkpoint::contract::Expr::src("a"),
            vec![2],
            Encoding::Raw(DType::F32),
        )
    };
    let contract = checkpoint::contract::ModelContract {
        alignment: 256,
        tensors: vec![one("dup"), one("dup")],
        groups: Vec::new(),
    };
    let error = compile_load_plan(&metadata(), &contract, StorageTarget::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("declares 'dup' twice"), "{error}");
}

fn a_contract_whose_declared_shape_is_wrong_is_rejected() {
    let contract = checkpoint::contract::ModelContract {
        alignment: 256,
        tensors: vec![checkpoint::contract::TensorContract::new(
            "a",
            checkpoint::contract::Expr::src("a"),
            vec![4],
            Encoding::Raw(DType::F32),
        )],
        groups: Vec::new(),
    };
    let error = compile_load_plan(&metadata(), &contract, StorageTarget::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("declares shape [4]"), "{error}");
    assert!(error.contains("yields [2]"), "{error}");
}

fn a_head_boundary_shard_is_one_contiguous_run() {
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 64,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![sized_raw(0, "w", 0, 64, &[8, 2], DType::F32)],
    };
    let expr = Expr::src("w")
        .transmute(TensorType::raw(vec![4, 4], DType::F32))
        .shard(0)
        .transmute(TensorType::raw(vec![-1, 2], DType::F32));
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "local",
            expr,
            vec![8, 2],
            Encoding::Raw(DType::F32),
        )],
        groups: Vec::new(),
    };
    let target = StorageTarget {
        tp_rank: 1,
        tp_size: 2,
        ..StorageTarget::default()
    };
    let program = compile_load_plan(&metadata, &contract, target).unwrap();
    let reads: Vec<&checkpoint::plan::SourceExtent> = program
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::BulkExtentWrite { source, .. } => Some(source),
            StorageInstr::ExtentWrite { source, .. } => Some(source),
            _ => None,
        })
        .collect();
    assert_eq!(reads.len(), 1, "{reads:#?}");
    assert_eq!(reads[0].file_offset, 32);
    assert_eq!(reads[0].span_bytes, 32);
}

fn a_head_boundary_shard_rejects_an_indivisible_world() {
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 48,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![sized_raw(0, "w", 0, 48, &[6, 2], DType::F32)],
    };
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "local",
            Expr::src("w")
                .transmute(TensorType::raw(vec![3, 4], DType::F32))
                .shard(0)
                .transmute(TensorType::raw(vec![-1, 2], DType::F32)),
            vec![6, 2],
            Encoding::Raw(DType::F32),
        )],
        groups: Vec::new(),
    };
    let target = StorageTarget {
        tp_rank: 1,
        tp_size: 2,
        ..StorageTarget::default()
    };
    let error = compile_load_plan(&metadata, &contract, target)
        .unwrap_err()
        .to_string();
    assert!(error.contains('3') && error.contains('2'), "{error}");
}

fn metadata() -> Metadata {
    Metadata {
        files: vec![File {
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

fn scale_contract(factor: f32, source: &str, dtype: DType) -> ModelContract {
    ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "out",
            Expr::src(source).scale(factor),
            vec![2],
            Encoding::Raw(dtype),
        )],
        groups: Vec::new(),
    }
}

fn a_scale_by_zero_is_rejected_at_compile_time() {
    let error = compile_load_plan(
        &metadata(),
        &scale_contract(0.0, "a", DType::F32),
        StorageTarget::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("zero"), "{error}");
}

fn a_scale_by_a_non_finite_factor_is_rejected_at_compile_time() {
    let error = compile_load_plan(
        &metadata(),
        &scale_contract(f32::NAN, "a", DType::F32),
        StorageTarget::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("finite"), "{error}");
}

fn a_scale_over_integer_elements_is_rejected_at_compile_time() {
    let metadata = Metadata {
        tensors: vec![raw(0, "ids", 0, &[2], DType::I32)],
        ..metadata()
    };
    let error = compile_load_plan(
        &metadata,
        &scale_contract(0.5, "ids", DType::I32),
        StorageTarget::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("I32"), "{error}");
}

fn a_scale_over_quantized_elements_is_rejected_at_compile_time() {
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "out",
            Expr::src("q").scale(0.5),
            vec![4, 8],
            Encoding::Quant(quant(QuantScheme::AwqInt4, DType::Bf16)),
        )],
        groups: Vec::new(),
    };
    let error = compile_load_plan(&quant_metadata(), &contract, StorageTarget::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("quantized"), "{error}");
}

fn a_scale_whose_declared_shape_is_wrong_is_rejected() {
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "out",
            Expr::src("a").scale(0.5),
            vec![4],
            Encoding::Raw(DType::F32),
        )],
        groups: Vec::new(),
    };
    let error = compile_load_plan(&metadata(), &contract, StorageTarget::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("declares shape [4]"), "{error}");
}

fn every_path_names_the_contract_its_error_came_from() {
    for expr in [
        Expr::src("a"),
        Expr::src("a").scale(0.5),
        Expr::src("a").cast(Encoding::Raw(DType::F16)),
    ] {
        let node = expr.node_name();
        let contract = ModelContract {
            alignment: 256,
            tensors: vec![TensorContract::new(
                "out",
                expr,
                vec![99],
                Encoding::Raw(DType::F32),
            )],
            groups: Vec::new(),
        };
        let error = compile_load_plan(&metadata(), &contract, StorageTarget::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("'out'"), "{node}: {error}");
    }
}

fn block_scaled_metadata() -> Metadata {
    Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 256,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            sized_raw(0, "w", 0, 64, &[4, 16], DType::U8),
            sized_raw(1, "s", 64, 4, &[4, 1], DType::U8),
            sized_raw(2, "s3", 68, 3, &[3, 1], DType::U8),
            sized_raw(3, "s128", 71, 128, &[128], DType::U8),
        ],
    }
}

fn mxfp4(channel_axis: u8) -> QuantSpec {
    QuantSpec {
        channel_axis: Some(Axis(channel_axis)),
        ..quant(QuantScheme::Mxfp4E2M1E8M0, DType::Bf16)
    }
}

fn block_scaled_contract(factors: &str, from: &str, shape: Vec<i64>) -> ModelContract {
    ModelContract {
        alignment: 256,
        tensors: vec![
            TensorContract::new(
                "scales",
                Expr::src(from).transmute(TensorType {
                    shape: shape.clone(),
                    encoding: Encoding::Raw(DType::E8m0),
                }),
                shape,
                Encoding::Raw(DType::E8m0),
            ),
            TensorContract::new(
                "w",
                Expr::src("w")
                    .transmute(TensorType {
                        shape: vec![4, 32],
                        encoding: Encoding::Quant(mxfp4(1)),
                    })
                    .scale_per_block(Expr::out(factors)),
                vec![4, 32],
                Encoding::Raw(DType::Bf16),
            ),
        ],
        groups: Vec::new(),
    }
}

fn a_block_scaled_dequant_is_one_scale_with_its_factors_as_an_operand() {
    let plan = compile_load_plan(
        &block_scaled_metadata(),
        &block_scaled_contract("scales", "s", vec![4, 1]),
        StorageTarget::default(),
    )
    .expect("block-scaled dequant should compile");

    let scales: Vec<_> = plan
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap {
                kind: TileMapKind::Scale,
                inputs,
                transform,
                ..
            } => Some((inputs.clone(), transform.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(scales.len(), 1, "{:#?}", plan.instrs);
    let (inputs, transform) = &scales[0];
    assert_eq!(transform.scale_blocks, vec![1, 32]);
    assert_eq!(transform.from, Some(QuantScheme::Mxfp4E2M1E8M0));
    assert_eq!(
        transform.scale_factor_bits, 0,
        "the uniform factor must stay unset so the two forms cannot be confused"
    );
    assert_eq!(
        inputs.len(),
        1,
        "the payload is the source extent, so the one operand is the factors"
    );
}

fn a_sharded_block_scaled_dequant_scales_only_its_own_rank() {
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![
            TensorContract::new(
                "scales",
                Expr::src("s")
                    .transmute(TensorType {
                        shape: vec![4, 1],
                        encoding: Encoding::Raw(DType::E8m0),
                    })
                    .shard(0),
                vec![4, 1],
                Encoding::Raw(DType::E8m0),
            ),
            TensorContract::new(
                "w",
                Expr::src("w")
                    .transmute(TensorType {
                        shape: vec![4, 32],
                        encoding: Encoding::Quant(mxfp4(1)),
                    })
                    .shard(0)
                    .scale_per_block(Expr::out("scales")),
                vec![4, 32],
                Encoding::Raw(DType::Bf16),
            ),
        ],
        groups: Vec::new(),
    };
    let target = StorageTarget {
        tp_size: 2,
        tp_rank: 1,
        ..StorageTarget::default()
    };
    let plan = compile_load_plan(&block_scaled_metadata(), &contract, target)
        .expect("a sharded block-scaled dequant should compile");

    let scale = plan
        .instrs
        .iter()
        .find_map(|instr| match instr {
            StorageInstr::TileMap {
                kind: TileMapKind::Scale,
                source,
                inputs,
                transform,
                ..
            } => Some((source.clone(), inputs.clone(), transform.clone())),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no Scale instruction: {:#?}", plan.instrs));
    let (source, inputs, transform) = scale;
    assert_eq!(transform.scale_blocks, vec![1, 32]);
    assert_eq!(inputs.len(), 1, "the one operand is the factors");
    let source = source.expect("rank 1's rows are contiguous, so they stay a source read");
    assert_eq!(
        source.span_bytes, 32,
        "only rank 1's half of the packed bytes is read"
    );
    let out = plan
        .tensors
        .iter()
        .find(|tensor| tensor.name == "w")
        .expect("the contract publishes 'w'");
    assert_eq!(
        out.shape,
        vec![2, 32],
        "the output is this rank's rows, dequantized"
    );
}

fn a_scale_by_a_tensor_no_contract_declares_is_rejected() {
    let error = compile_load_plan(
        &block_scaled_metadata(),
        &block_scaled_contract("absent", "s", vec![4, 1]),
        StorageTarget::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("is declared before this one"), "{error}");
}

fn a_scale_blocks_every_axis_the_factors_divide() {
    let plan = compile_load_plan(
        &block_scaled_metadata(),
        &block_scaled_contract("scales", "s", vec![2, 2]),
        StorageTarget::default(),
    )
    .expect("a two-dimensional blocking should compile");

    let blocks: Vec<_> = plan
        .instrs
        .iter()
        .filter_map(|instr| match instr {
            StorageInstr::TileMap {
                kind: TileMapKind::Scale,
                transform,
                ..
            } => Some(transform.scale_blocks.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(blocks, vec![vec![2, 16]]);
}

fn a_scale_by_factors_of_a_different_rank_is_rejected() {
    let error = compile_load_plan(
        &block_scaled_metadata(),
        &block_scaled_contract("scales", "s", vec![4]),
        StorageTarget::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("same rank and dividing each axis"),
        "{error}"
    );
}

fn a_scale_by_factors_that_do_not_divide_the_payload_is_rejected() {
    let error = compile_load_plan(
        &block_scaled_metadata(),
        &block_scaled_contract("scales", "s3", vec![3, 1]),
        StorageTarget::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("axis 0 of [4, 32] is not a whole number of blocks"),
        "{error}"
    );
}

fn a_scale_by_one_factor_per_element_is_rejected() {
    let error = compile_load_plan(
        &block_scaled_metadata(),
        &block_scaled_contract("scales", "s128", vec![4, 32]),
        StorageTarget::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("they group nothing"), "{error}");
}

fn a_scale_by_an_undeclared_expression_is_rejected() {
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "w",
            Expr::src("w")
                .transmute(TensorType {
                    shape: vec![4, 32],
                    encoding: Encoding::Quant(mxfp4(1)),
                })
                .scale_per_block(Expr::src("s").transmute(TensorType {
                    shape: vec![4, 1],
                    encoding: Encoding::Raw(DType::E8m0),
                })),
            vec![4, 32],
            Encoding::Raw(DType::Bf16),
        )],
        groups: Vec::new(),
    };
    let error = compile_load_plan(
        &block_scaled_metadata(),
        &contract,
        StorageTarget::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("declare them first"), "{error}");
}

fn nemotron_h_expert_metadata() -> Metadata {
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
        let bytes = tensor_bytes(&shape, DType::Bf16);
        tensors.push(RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: offset,
            span_bytes: bytes,
            shape,
            encoding: Encoding::Raw(DType::Bf16),
        });
        offset += bytes;
    }
    Metadata {
        files: vec![File {
            id: FileId(0),
            path: "nemotron.safetensors".to_string(),
            size_bytes: offset,
            format: CheckpointFormat::Safetensors,
        }],
        tensors,
    }
}

fn quant_metadata() -> Metadata {
    Metadata {
        files: vec![File {
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
                encoding: Encoding::Quant(quant(QuantScheme::AwqInt4, DType::Bf16)),
            },
            RawTensor {
                id: TensorId(3),
                name: "fp8".to_string(),
                file_id: FileId(0),
                file_offset: 216,
                span_bytes: 4,
                shape: vec![4],
                encoding: Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::Bf16)),
            },
            RawTensor {
                id: TensorId(4),
                name: "q_odd".to_string(),
                file_id: FileId(0),
                file_offset: 220,
                span_bytes: 14,
                shape: vec![4, 7],
                encoding: Encoding::Quant(quant(QuantScheme::AwqInt4, DType::Bf16)),
            },
        ],
    }
}

fn gpt_oss_mxfp4_metadata() -> Metadata {
    gpt_oss_mxfp4_metadata_with_intermediate(64)
}

fn gpt_oss_mxfp4_metadata_with_intermediate(intermediate: i64) -> Metadata {
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
            DType::Bf16,
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
            DType::Bf16,
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
    Metadata {
        files: vec![File {
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
    shape.iter().fold(dtype.bytes_ceil(), |acc, dim| {
        acc * u64::try_from(*dim).unwrap()
    })
}

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
            encoding: Encoding::Raw(DType::Bf16),
        });
        offset += bytes;
    }
    let meta = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: offset,
            format: CheckpointFormat::Safetensors,
        }],
        tensors,
    };
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: checkpoint::plan::CUDA_TILE_MAP_MASK,
        ..StorageTarget::default()
    };
    let contract = stored_contract("kimi_k2_mla_fusion");
    let program = compile_load_plan(&meta, &contract, target).unwrap();
    let summary = checkpoint::dump::describe(&program);

    let has_fused = program
        .tensors
        .iter()
        .any(|t| t.name.contains("q_kv_a_proj.fused"));
    assert!(
        has_fused,
        "Expected fused q_kv_a_proj tensor; summary: {summary}\ntensors: {:?}",
        program.tensors.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    let fused = program
        .tensors
        .iter()
        .find(|t| t.name.contains("q_kv_a_proj.fused"))
        .unwrap();
    assert_eq!(fused.shape[0], q_lora + kv_lora_rope);
    assert_eq!(fused.shape[1], h);

    assert!(!program.tensors.is_empty());
    assert!(
        program
            .instrs
            .iter()
            .any(|instr| matches!(instr, StorageInstr::Finalize { .. })),
        "{summary}"
    );
}

fn instr_id(instr: &StorageInstr) -> checkpoint::types::InstrId {
    match instr {
        StorageInstr::Allocate { id, .. }
        | StorageInstr::Fill { id, .. }
        | StorageInstr::ExtentWrite { id, .. }
        | StorageInstr::BulkExtentWrite { id, .. }
        | StorageInstr::GatherWrite { id, .. }
        | StorageInstr::TileMap { id, .. }
        | StorageInstr::CreateView { id, .. }
        | StorageInstr::Finalize { id, .. } => *id,
    }
}

fn a_block_scaled_fp8_source_carries_its_scale_tensor() {
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            sized_raw(0, "w.weight", 0, 4096, &[64, 64], DType::E4m3),
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
            Expr::src("w.weight").cast(Encoding::Quant(quant(
                QuantScheme::Mxfp4E2M1E8M0,
                DType::Bf16,
            ))),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Mxfp4E2M1E8M0, DType::Bf16)),
        )],
        groups: Vec::new(),
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

fn a_source_without_a_scale_sibling_names_none() {
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![sized_raw(0, "w.weight", 0, 8192, &[64, 64], DType::Bf16)],
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
            Expr::src("w.weight").cast(Encoding::Quant(quant(
                QuantScheme::Mxfp4E2M1E8M0,
                DType::Bf16,
            ))),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Mxfp4E2M1E8M0, DType::Bf16)),
        )],
        groups: Vec::new(),
    };
    let program = compile_load_plan(&metadata, &contract, target).unwrap();
    for instr in &program.instrs {
        if let StorageInstr::TileMap { transform, .. } = instr {
            assert_eq!(transform.metadata_source, None);
        }
    }
}

fn a_padded_head_dim_zeroes_the_buffer_before_it_writes_the_rows() {
    let metadata = Metadata {
        files: vec![File {
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
            encoding: Encoding::Raw(DType::Bf16),
        }],
    };

    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "q_proj.weight",
            Expr::concat(
                1,
                vec![
                    Expr::src("q_proj.weight"),
                    Expr::fill(0.0, TensorType::raw(vec![4, 1], DType::Bf16)),
                ],
            ),
            vec![4, 5],
            Encoding::Raw(DType::Bf16),
        )],
        groups: Vec::new(),
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

    let at = |want| program.schedule.iter().position(|id| *id == want).unwrap();
    let fill_at = at(fill_id);
    for write in &writes {
        assert!(fill_at < at(instr_id(write)), "the fill must come first");
    }

    assert_eq!(program.memory.checkpoint_read_bytes, 32);
    assert_eq!(program.memory.device_write_bytes, 32);
    assert_eq!(program.memory.persistent_bytes, 40);
    assert_eq!(program.buffer(filled).unwrap().bytes, 40);
}

fn a_padded_head_dim_materializes_zeros_where_no_source_covers() {
    let dir = std::env::temp_dir().join(format!("pie_fill_replay_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let snapshot = dir.join("model.safetensors");

    let source: Vec<u8> = (0..32).map(|i| (i as u8) | 0x80).collect();
    std::fs::write(&snapshot, &source).unwrap();

    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: source.len() as u64,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "q_proj.weight".to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 32,
            shape: vec![4, 4],
            encoding: Encoding::Raw(DType::Bf16),
        }],
    };
    let contract = ModelContract {
        groups: Vec::new(),
        alignment: 1,
        tensors: vec![TensorContract::new(
            "q_proj.weight",
            Expr::concat(
                1,
                vec![
                    Expr::src("q_proj.weight"),
                    Expr::fill(0.0, TensorType::raw(vec![4, 1], DType::Bf16)),
                ],
            ),
            vec![4, 5],
            Encoding::Raw(DType::Bf16),
        )],
    };

    let plan = compile_load_plan(&metadata, &contract, StorageTarget::default()).unwrap();
    let storage = checkpoint::executor::Execution::new(&plan, &dir)
        .run()
        .expect("the padded plan does not execute");
    let got = storage.tensors.get("q_proj.weight").expect("materialized");

    assert_eq!(got.len(), 40, "four rows of five bf16 elements");
    for row in 0..4 {
        let at = row * 10;
        assert_eq!(
            &got[at..at + 8],
            &source[row * 8..row * 8 + 8],
            "row {row} did not get its source bytes"
        );
        assert_eq!(
            &got[at + 8..at + 10],
            &[0, 0],
            "row {row}'s padded column is not zero"
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

fn an_e8m0_block_scale_read_as_fp32_lowers_to_a_cast() {
    let metadata = Metadata {
        files: vec![File {
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
            Expr::src("w.scale")
                .transmute(TensorType::raw(vec![8, 8], DType::E8m0))
                .cast(Encoding::Raw(DType::F32)),
            vec![8, 8],
            Encoding::Raw(DType::F32),
        )],
        groups: Vec::new(),
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

fn scale_target() -> StorageTarget {
    StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: u32::MAX,
        ..StorageTarget::default()
    }
}

fn scales_the_loader_writes_while_encoding_mxfp4_stay_raw_e8m0() {
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![sized_raw(0, "w.weight", 0, 8192, &[64, 64], DType::Bf16)],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "runtime.w",
            Expr::src("w.weight").cast(Encoding::Quant(quant(
                QuantScheme::Mxfp4E2M1E8M0,
                DType::Bf16,
            ))),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Mxfp4E2M1E8M0, DType::Bf16)),
        )],
        groups: Vec::new(),
    };
    let program = compile_load_plan(&metadata, &contract, scale_target()).unwrap();
    assert_eq!(program.attachments.len(), 1, "{:#?}", program.attachments);
    let attach = program.attachments[0];
    assert_eq!(attach.scale_form, ScaleForm::RawE8M0);
    assert_eq!(attach.granularity, QuantGranularity::PerGroup);
    assert_eq!(attach.group_size, 32);
    assert_eq!(program.tensors[attach.tensor.0 as usize].name, "runtime.w");
    assert_eq!(
        program.tensors[attach.scale_tensor.0 as usize].name,
        "runtime.w.scales"
    );
}

fn scales_the_loader_writes_while_encoding_fp8_are_f32_factors() {
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![sized_raw(0, "w.weight", 0, 8192, &[64, 64], DType::Bf16)],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "runtime.w",
            Expr::src("w.weight").cast(Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::Bf16))),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::Bf16)),
        )],
        groups: Vec::new(),
    };
    let program = compile_load_plan(&metadata, &contract, scale_target()).unwrap();
    assert_eq!(program.attachments.len(), 1, "{:#?}", program.attachments);
    assert_eq!(program.attachments[0].scale_form, ScaleForm::F32Factors);
    assert_eq!(
        program.attachments[0].granularity,
        QuantGranularity::PerChannel
    );
}

fn scales_the_checkpoint_shipped_are_paired_by_the_contract() {
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            sized_raw(0, "w.weight", 0, 4096, &[64, 64], DType::E4m3),
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
                Encoding::Raw(DType::E4m3),
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
        groups: Vec::new(),
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

fn scales_named_for_a_weight_the_loader_quantizes_are_a_contract_error() {
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            sized_raw(0, "w.weight", 0, 8192, &[64, 64], DType::Bf16),
            sized_raw(1, "w.weight_scale_inv", 8192, 4, &[1, 1], DType::F32),
        ],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![
            TensorContract::new(
                "runtime.w",
                Expr::src("w.weight")
                    .cast(Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::Bf16))),
                vec![64, 64],
                Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::Bf16)),
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
        groups: Vec::new(),
    };
    let err = compile_load_plan(&metadata, &contract, scale_target())
        .unwrap_err()
        .to_string();
    assert!(err.contains("already has scales"), "{err}");
}

fn scales_naming_an_undeclared_tensor_are_a_contract_error() {
    let metadata = Metadata {
        files: vec![File {
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
        groups: Vec::new(),
    };
    let error = compile_load_plan(&metadata, &contract, scale_target())
        .unwrap_err()
        .to_string();
    assert!(error.contains("runtime.w"), "{error}");
    assert!(error.contains("the contract does not declare"), "{error}");
}

fn scales_may_name_a_tensor_declared_after_them() {
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            sized_raw(0, "w.scale", 0, 4, &[1, 1], DType::F32),
            sized_raw(1, "w.weight", 4096, 4096, &[64, 64], DType::E4m3),
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
                Encoding::Raw(DType::E4m3),
            ),
        ],
        groups: Vec::new(),
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

fn encode_to(
    name: &str,
    shape: &[i64],
    scheme: QuantScheme,
) -> Result<LoadPlan, checkpoint::error::Error> {
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![sized_raw(
            0,
            "w.weight",
            0,
            (shape.iter().product::<i64>() * 2) as u64,
            shape,
            DType::Bf16,
        )],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            name,
            Expr::src("w.weight").cast(Encoding::Quant(quant(scheme, DType::Bf16))),
            shape.to_vec(),
            Encoding::Quant(quant(scheme, DType::Bf16)),
        )],
        groups: Vec::new(),
    };
    compile_load_plan(&metadata, &contract, scale_target())
}

fn a_rank_3_bank_encodes_and_its_scales_keep_the_expert_axis() {
    let plan = encode_to("runtime.w", &[2, 64, 64], QuantScheme::Mxfp4E2M1E8M0).unwrap();
    assert_eq!(plan.attachments.len(), 1, "{:#?}", plan.attachments);
    let attach = plan.attachments[0];
    let scales = &plan.tensors[attach.scale_tensor.0 as usize];
    assert_eq!(scales.name, "runtime.w.scales");
    assert_eq!(scales.shape, vec![2, 64, 2]);
    assert_eq!(attach.channel_axis, 2);
    assert_eq!(attach.group_size, 32);
}

fn an_encode_that_cannot_place_its_scales_is_refused() {
    let err = encode_to("runtime.w", &[64], QuantScheme::Mxfp4E2M1E8M0).unwrap_err();
    assert!(err.to_string().contains("rank-1"), "{err}");
    assert!(err.to_string().contains("runtime.w"), "{err}");
}

fn an_encode_whose_columns_do_not_fill_a_block_is_refused() {
    let err = encode_to("runtime.w", &[64, 48], QuantScheme::Mxfp4E2M1E8M0).unwrap_err();
    assert!(err.to_string().contains("blocks 32 columns"), "{err}");
}

fn an_encode_into_a_scheme_no_kernel_writes_is_refused() {
    let err = encode_to("runtime.w", &[64, 64], QuantScheme::AwqInt4).unwrap_err();
    assert!(err.to_string().contains("no encode kernel writes"), "{err}");
}

fn re_encoding_one_quantized_scheme_as_another_is_refused() {
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "w.weight".to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 4096,
            shape: vec![64, 64],
            encoding: Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::Bf16)),
        }],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "runtime.w",
            Expr::src("w.weight").cast(Encoding::Quant(quant(
                QuantScheme::Mxfp4E2M1E8M0,
                DType::Bf16,
            ))),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Mxfp4E2M1E8M0, DType::Bf16)),
        )],
        groups: Vec::new(),
    };
    let err = compile_load_plan(&metadata, &contract, scale_target()).unwrap_err();
    assert!(err.to_string().contains("re-encodes Fp8E4M3"), "{err}");
    assert!(
        err.to_string().contains("cast to a raw type first"),
        "{err}"
    );
}

fn a_declaration_that_disagrees_with_its_expression_is_a_mistake_not_a_kernel() {
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 1 << 20,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![sized_raw(0, "w.weight", 0, 8192, &[64, 64], DType::Bf16)],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "runtime.w",
            Expr::src("w.weight"),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::Bf16)),
        )],
        groups: Vec::new(),
    };
    let err = compile_load_plan(&metadata, &contract, scale_target())
        .unwrap_err()
        .to_string();
    assert!(err.contains("but its expression yields Raw(Bf16)"), "{err}");
    assert!(err.contains("explicit cast"), "{err}");
}

fn the_same_pair_with_the_cast_written_down_encodes() {
    let plan = encode_to("runtime.w", &[64, 64], QuantScheme::Fp8E4M3).unwrap();
    let names: Vec<&str> = plan.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"runtime.w"), "{names:?}");
    assert!(names.contains(&"runtime.w_scale_inv"), "{names:?}");
}

fn negative_plane(dir: &std::path::Path, values: &[f32]) -> Metadata {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("model.safetensors"), &bytes).unwrap();
    Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: bytes.len() as u64,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "ssm_a".to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: bytes.len() as u64,
            shape: vec![values.len() as i64],
            encoding: Encoding::Raw(DType::F32),
        }],
    }
}

fn neg_ln_contract() -> ModelContract {
    ModelContract {
        groups: Vec::new(),
        alignment: 1,
        tensors: vec![TensorContract::new(
            "a_log",
            Expr::src("ssm_a").unary(UnaryOp::NegLn),
            vec![4],
            Encoding::Raw(DType::F32),
        )],
    }
}

fn a_unary_takes_the_logarithm_of_a_negated_plane() {
    let dir = std::env::temp_dir().join(format!("pie_unary_{}", std::process::id()));
    let stored = [-1.294_096_f32, -0.131_171, -0.119_433, -0.567_561];
    let metadata = negative_plane(&dir, &stored);

    let plan = compile_load_plan(&metadata, &neg_ln_contract(), StorageTarget::default()).unwrap();
    let storage = checkpoint::executor::Execution::new(&plan, &dir)
        .run()
        .expect("the unary plan executes");
    let got = storage.tensors.get("a_log").expect("materialized");

    let read: Vec<f32> = got
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    for (at, (&raw, &want)) in stored
        .iter()
        .zip([0.257_812_f32, -2.031_253, -2.125, -0.566_407].iter())
        .enumerate()
    {
        assert!(
            (read[at] - want).abs() < 1e-4,
            "element {at}: ln(-{raw}) read as {} not {want}",
            read[at]
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

fn a_unary_refuses_an_element_outside_its_domain() {
    let dir = std::env::temp_dir().join(format!("pie_unary_bad_{}", std::process::id()));
    let metadata = negative_plane(&dir, &[-1.0, -2.0, 0.5, -4.0]);

    let plan = compile_load_plan(&metadata, &neg_ln_contract(), StorageTarget::default()).unwrap();
    let err = checkpoint::executor::Execution::new(&plan, &dir)
        .run()
        .expect_err("a positive element is refused")
        .to_string();
    assert!(err.contains("strictly negative"), "{err}");
    assert!(err.contains("element 2"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}

fn a_serving_target_refuses_the_unary_a_conversion_target_runs() {
    let dir = std::env::temp_dir().join(format!("pie_unary_mask_{}", std::process::id()));
    let metadata = negative_plane(&dir, &[-1.0, -2.0, -3.0, -4.0]);
    let contract = neg_ln_contract();

    for (backend, mask) in [
        (BackendKind::Cuda, checkpoint::plan::CUDA_TILE_MAP_MASK),
        (BackendKind::Metal, checkpoint::plan::METAL_TILE_MAP_MASK),
        (BackendKind::Vulkan, checkpoint::plan::VULKAN_TILE_MAP_MASK),
        (BackendKind::Wgpu, checkpoint::plan::WGPU_TILE_MAP_MASK),
    ] {
        let err = compile_load_plan(
            &metadata,
            &contract,
            StorageTarget {
                backend,
                tile_map_mask: mask,
                ..StorageTarget::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("would apply Some(NegLn) on the way in"),
            "{backend:?}: {err}"
        );
    }

    compile_load_plan(
        &metadata,
        &contract,
        StorageTarget {
            tile_map_mask: checkpoint::plan::CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        },
    )
    .expect("a conversion target compiles the unary");
    std::fs::remove_dir_all(&dir).ok();
}

const TILED_ROWS: [(usize, usize); 4] = [(32, 1024), (1024, 2048), (7168, 1024), (1024, 3584)];

fn tiled_codes(dir: &std::path::Path, rows: usize, k: usize) -> Metadata {
    let mut bytes = vec![0u8; rows * k / 2];
    let mut state = 0x2545_f491_4f6c_dd1d_u64 ^ (rows as u64) << 32 ^ k as u64;
    for byte in &mut bytes {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = (state >> 24) as u8;
    }
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("model.safetensors"), &bytes).unwrap();
    Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: bytes.len() as u64,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![RawTensor {
            id: TensorId(0),
            name: "proj".to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: bytes.len() as u64,
            shape: vec![rows as i64, k as i64],
            encoding: tiled_u4(),
        }],
    }
}

fn tiled_u4() -> Encoding {
    Encoding::Quant(
        QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 64,
            channel_axis: Some(Axis(1)),
        }
        .normalized(),
    )
}

fn the_tiled_repack_is_the_documented_permutation_at_every_shipped_shape() {
    const BAND: usize = 16;
    const STEP: usize = 64;
    for (rows, k) in TILED_ROWS {
        let dir = std::env::temp_dir().join(format!("pie_tiled_{}_{rows}x{k}", std::process::id()));
        let metadata = tiled_codes(&dir, rows, k);
        let banded = rows.div_ceil(BAND) * BAND;
        let contract = ModelContract {
            groups: Vec::new(),
            alignment: 1,
            tensors: vec![TensorContract::new(
                "proj.tiled",
                Expr::src("proj").repack(
                    RepackLayout::TiledAffineU4Weight,
                    TensorType::new(vec![banded as i64, k as i64], tiled_u4()),
                ),
                vec![banded as i64, k as i64],
                tiled_u4(),
            )],
        };

        let plan = compile_load_plan(
            &metadata,
            &contract,
            StorageTarget {
                tile_map_mask: checkpoint::plan::CONVERT_TILE_MAP_MASK,
                ..StorageTarget::default()
            },
        )
        .unwrap_or_else(|err| panic!("{rows}x{k}: the repack plan compiles: {err}"));
        let storage = checkpoint::executor::Execution::new(&plan, &dir)
            .run()
            .unwrap_or_else(|err| panic!("{rows}x{k}: the repack executes: {err}"));
        let got = storage
            .tensors
            .get("proj.tiled")
            .unwrap_or_else(|| panic!("{rows}x{k}: materialized"));

        let source = std::fs::read(dir.join("model.safetensors")).unwrap();
        let row_bytes = k / 2;
        assert_eq!(
            got.len(),
            banded * row_bytes,
            "{rows}x{k}: the placed plane is the banded rectangle"
        );
        assert_ne!(
            &got[..source.len()],
            &source[..],
            "{rows}x{k}: the placed plane is the source verbatim -- no repack ran"
        );

        let mut seen = vec![0u8; rows * k];
        let mut back = vec![0u8; rows * k];
        let quad = STEP / BAND;
        let quads = (k / BAND) / quad;
        let mut at = 0usize;
        for b in 0..banded / BAND {
            for kq in 0..quads {
                for lane in 0..32usize {
                    for word in 0..quad {
                        let kt = kq * quad + word;
                        let k_base = kt * BAND + 2 * (lane % 4);
                        let res = u32::from_le_bytes(
                            got[at * 4..at * 4 + 4].try_into().expect("four bytes"),
                        );
                        at += 1;
                        for s in 0..4usize {
                            let col = b * BAND + lane / 4 + usize::from(s >= 2) * 8;
                            for h in 0..2usize {
                                if col >= rows {
                                    continue;
                                }
                                let kk = k_base + usize::from(s % 2 == 1) * 8 + h;
                                let flat = col * k + kk;
                                back[flat] = ((res >> (4 * (s + 4 * h))) & 0xF) as u8;
                                seen[flat] += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            at * 4,
            got.len(),
            "{rows}x{k}: the word walk covers the placed plane exactly"
        );
        if let Some(flat) = seen.iter().position(|&n| n != 1) {
            panic!(
                "{rows}x{k}: code at (n {}, k {}) is written {} times, not once",
                flat / k,
                flat % k,
                seen[flat]
            );
        }
        for flat in 0..rows * k {
            let byte = source[flat / 2];
            let want = if flat % 2 == 0 { byte & 0xF } else { byte >> 4 };
            assert_eq!(
                back[flat],
                want,
                "{rows}x{k}: code at (n {}, k {}) came back as {} not {want}",
                flat / k,
                flat % k,
                back[flat]
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
