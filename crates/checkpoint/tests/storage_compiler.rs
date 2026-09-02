/// Loads a contract fixture stored next to the test that compiles it.
fn stored_contract(name: &str) -> ModelContract {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/contracts")
        .join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("{name}: cannot read {}: {err}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("{name}: parsing: {err}"))
}
use checkpoint::file::{File, Metadata, RawTensor};
use checkpoint::contract::{Expr, ModelContract, Scales, TensorContract, TensorType};
use checkpoint::plan::compile as compile_load_plan;
use checkpoint::plan::{LoadPlan, StorageInstr, StorageTarget, TileMapKind};
use checkpoint::types::{
    Axis, BackendKind, CheckpointFormat, DType, Encoding, FileId, QuantGranularity, QuantScheme,
    QuantSpec, RepackLayout, ScaleForm, TensorId,
};

#[test]
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
    // 4-bit weights packed eight to a u32 word: bitcast to logical shape,
    // affine-U4 encoding, scales/biases as named tensors.
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

#[test]
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
    // No ephemeral declarations, so the two cast inputs count toward
    // persistent memory instead of temporary peak.
    assert_eq!(program.memory.persistent_bytes, 16);
    assert_eq!(program.memory.temporary_peak_bytes, 0);
}

#[test]
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

#[test]
fn packed_quant_row_select_uses_byte_exact_offsets() {
    // Row 2 of a [4, 8] int4 tensor: 8 elements at 4 bits is 4 bytes a row.
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

/// An expression bigger than the tensor it is declared for is refused at
/// the declaration, before any offset is produced.
#[test]
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

#[test]
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

/// A quantized tensor can't cast straight to another scheme, so the route
/// is decode (a per-group `Scale`) then `Cast` over the intermediate.
#[test]
fn a_quantized_tensor_is_re_encoded_through_a_decoded_intermediate() {
    let int8 = Encoding::Quant(QuantSpec {
        scheme: QuantScheme::Int8Symmetric,
        logical_dtype: DType::Bf16,
        bits_per_element: 8,
        group_size: 32,
        channel_axis: Some(Axis(1)),
    });
    let mut contract = block_scaled_contract("scales", "s", vec![4, 1]);
    // `w` is the decoded BF16 tensor the fixture publishes.
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

    // 64 payload + 4 exponents, read once.
    assert_eq!(plan.memory.checkpoint_read_bytes, 68);
    // `Finalize` puts a name in the engine's bind table; internal
    // declarations get none.
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
        // `w`, the decoded intermediate, is the one name missing.
        vec!["scales", "w_int8_scale_inv", "w_int8"],
        "the intermediate is not bound"
    );
}

/// A serving target refuses the encode a conversion target runs: the stored
/// form is the served form, so quantizing on the way in is refused, naming
/// `pie model import` as the fix.
#[test]
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

/// Casting one quantization scheme straight to another stays refused — the
/// two-step exists because the one-step does not.
#[test]
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

#[test]
fn gpt_oss_native_mxfp4_default_abi_lowers_to_repack_tile_maps() {
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        // A target claiming the native MXFP4 GEMM must also claim the Marlin
        // repack that builds its operand.
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
    // Six, not eight: the two biases are a row selection, affine, never
    // reaching a kernel.
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

/// A repack declaration that disagrees with its transform must be refused,
/// not silently replaced by the transform's own shape.
#[test]
fn a_repack_declaration_is_checked_against_its_transform() {
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        // A target claiming the native MXFP4 GEMM must also claim the Marlin
        // repack that builds its operand.
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

/// GPT-OSS's gate and up halves are the even and odd rows of one block:
/// each repack reads only the rows it wants, covering the block once.
#[test]
fn gpt_oss_native_mxfp4_reads_each_interleaved_half_once() {
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        // A target claiming the native MXFP4 GEMM must also claim the Marlin
        // repack that builds its operand.
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

    // `gate_up_proj_blocks` is [2, 128, 2, 16] u8: a row is 32 bytes, the
    // whole tensor is 8192.
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

/// The repack's source selection is an `Expr::Shard`, so one contract serves
/// every rank: compiling it at two ranks reads two different bands.
#[test]
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
            tile_map_mask: checkpoint::plan::CUDA_TILE_MAP_MASK
                | checkpoint::plan::TILE_MAP_REPACK,
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

    // The gate/up block is [2, 256, 2, 16]: a row is 32 bytes, an expert is
    // 8192. Rank one's band starts halfway into each expert.
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

#[test]
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
    // The packed bank declares the whole `[6, 4]`; the `Shard` on axis 1
    // says this rank binds `[6, 2]` of it.
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
    // Experts are packed contiguously within their backing buffer (tight
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

    // Each expert pack is one persistent backing buffer (2 experts x 12 B =
    // 24 B), 256-byte aligned for cuBLAS(Lt)'s fast `align8` kernels.
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

    // Raw data moved is unchanged (4 experts x 12 B); persistent arena grows
    // only by the per-backing alignment padding (2nd backing at offset 256).
    assert_eq!(program.memory.checkpoint_read_bytes, 48);
    assert_eq!(program.memory.device_write_bytes, 48);
    assert_eq!(program.memory.persistent_bytes, 280);
}

#[test]
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

#[test]
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

/// A shard that must land on a unit boundary is said by reshaping the axis
/// first, sharding, then reshaping back — a byte identity that still
/// compiles to one contiguous run.
#[test]
fn a_head_boundary_shard_is_one_contiguous_run() {
    // 4 heads of 2 rows, 2 columns; rank 1 of 2 takes heads 2 and 3, which is
    // rows 4..8, which is bytes 32..64 of an F32 tensor.
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
            // The whole tensor: this rank binds `[4, 2]`, and the `Shard`
            // is what says so.
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

/// A `tp_size` that does not divide the unit count is rejected, not rounded.
#[test]
fn a_head_boundary_shard_rejects_an_indivisible_world() {
    let metadata = Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 48,
            format: CheckpointFormat::Safetensors,
        }],
        // 3 heads of 2 rows: 6 rows divides by 2, 3 heads does not.
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
            // The whole tensor's shape, so what this pins is the shard being
            // refused and not a declaration mismatch.
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

/// A contract that scales `source` by `factor`, for pinning `compile`
/// against `infer_scale`'s rules.
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

/// Every path through the compiler names the contract its error came from —
/// annotated once, at the boundary, for every path (affine and kernel alike).
#[test]
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

/// A checkpoint holding one block-scaled MXFP4 tensor (4 rows of 32
/// elements) and its factors (one E8M0 exponent per row), both raw `U8`.
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
            // Rejection-test factors: 3 exponents don't divide 4 rows; 128
            // give one per element.
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

/// A block-scaled dequant: the factors' shape is the whole statement of how
/// the payload is blocked — `[4, 1]` over `[4, 32]` is one factor per row,
/// `[2, 2]` is 2x16 tiles.
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

#[test]
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

/// Each rank dequantizes only the shard it will compute with: `infer`
/// compares the sharded factors and sharded payload after both are
/// specialized for the rank.
#[test]
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
                // Both declarations are the whole tensor's; rank 1 binds
                // half of each, per the two `Shard`s.
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

#[test]
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

/// Blocks on two axes at once: `[2, 2]` factors over a `[4, 32]` payload is
/// a 2x16 tile, neither number named in the contract.
#[test]
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

#[test]
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

#[test]
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

/// One factor per element groups nothing, so it is an elementwise product and
/// not a block scale. Symmetry rule A: a node may not denote its operand.
#[test]
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

/// Scaling by an expression, rather than a published tensor, is refused —
/// the engine reads factors by name.
#[test]
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
    shape
        .iter()
        .fold(dtype.bytes_ceil(), |acc, dim| acc * u64::try_from(*dim).unwrap())
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

/// A block-scaled FP8 source names its scale tensor on the instruction,
/// rather than the executor reconstructing it by appending `_scale_inv`.
#[test]
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

/// ...and a source with no such sibling says so, rather than naming tensor 0.
#[test]
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

#[test]
fn a_padded_head_dim_zeroes_the_buffer_before_it_writes_the_rows() {
    // Pads Q/K/V up to a head_dim the attention kernel takes.
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

    // Four rows: the destination stride is wider than the row and `fold`
    // will not skip in the destination.
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

/// The padding is actually zero when the plan runs — the test above only
/// proves the plan says `Fill`, not that executing it produces zeros.
#[test]
fn a_padded_head_dim_materializes_zeros_where_no_source_covers() {
    let dir = std::env::temp_dir().join(format!("pie_fill_replay_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let snapshot = dir.join("model.safetensors");

    // Every byte non-zero, so an uninitialised pad cannot pass by
    // coincidence and a pad written from the wrong offset shows up as
    // source bytes rather than zeros.
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

/// Reading raw bytes as `E8m0` and declaring `F32` hits the ordinary
/// dtype-mismatch rule, which inserts the cast.
#[test]
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

// ── quant attachments ──────────────────────────────────
// A quantized weight and its scales are two runtime tensors, paired by
// whoever declares the scale tensor.

fn scale_target() -> StorageTarget {
    StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: u32::MAX,
        ..StorageTarget::default()
    }
}

/// The MXFP4 GEMM asserts its scale operand is U8, so a plan that asks the
/// engine to expand these to F32 makes the kernel reject the load.
#[test]
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
    // Both halves name real entries.
    assert_eq!(program.tensors[attach.tensor.0 as usize].name, "runtime.w");
    // `.scales` is the one spelling: an encoded plane binds under the same
    // name as a shipped one, `<w>.scales`. See `ScaleLayout::for_encode`.
    assert_eq!(
        program.tensors[attach.scale_tensor.0 as usize].name,
        "runtime.w.scales"
    );
}

/// The same, for the per-channel schemes: one F32 factor per output row.
#[test]
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

/// Block-scaled FP8 arrives already quantized: the loader writes no
/// scales, so the contract states the pairing.
#[test]
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

/// A contract that names a tensor no earlier entry declares is rejected,
/// rather than silently producing a plan with no attachment.
#[test]
fn scales_named_for_a_weight_the_loader_quantizes_are_a_contract_error() {
    // The loader writes this weight's scales itself while encoding it. A
    // contract naming a second set would attach quant metadata twice.
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

#[test]
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

/// Scales may be declared before the tensor they scale.
#[test]
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

/// A quantized weight without its scales is not a smaller weight, it is an
/// unreadable one.
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

/// A rank-3 expert bank encodes, and its scales keep the leading (expert)
/// axis, since the engine binds the plane at its declared rank.
#[test]
fn a_rank_3_bank_encodes_and_its_scales_keep_the_expert_axis() {
    let plan = encode_to("runtime.w", &[2, 64, 64], QuantScheme::Mxfp4E2M1E8M0).unwrap();
    assert_eq!(plan.attachments.len(), 1, "{:#?}", plan.attachments);
    let attach = plan.attachments[0];
    let scales = &plan.tensors[attach.scale_tensor.0 as usize];
    assert_eq!(scales.name, "runtime.w.scales");
    assert_eq!(scales.shape, vec![2, 64, 2]);
    // The blocked axis is the last one, whatever rank the bank has.
    assert_eq!(attach.channel_axis, 2);
    assert_eq!(attach.group_size, 32);
}

/// A rank-1 declaration has no axis left to hold one scale per row, so it
/// is refused rather than folded into a plausible one-row weight.
#[test]
fn an_encode_that_cannot_place_its_scales_is_refused() {
    let err = encode_to("runtime.w", &[64], QuantScheme::Mxfp4E2M1E8M0).unwrap_err();
    assert!(err.to_string().contains("rank-1"), "{err}");
    assert!(err.to_string().contains("runtime.w"), "{err}");
}

/// The column count is part of the scale layout, not just of the payload.
#[test]
fn an_encode_whose_columns_do_not_fill_a_block_is_refused() {
    let err = encode_to("runtime.w", &[64, 48], QuantScheme::Mxfp4E2M1E8M0).unwrap_err();
    assert!(err.to_string().contains("blocks 32 columns"), "{err}");
}

/// `QuantScheme` names every format the loader can *read*; only three have
/// an encoder.
#[test]
fn an_encode_into_a_scheme_no_kernel_writes_is_refused() {
    let err = encode_to("runtime.w", &[64, 64], QuantScheme::AwqInt4).unwrap_err();
    assert!(err.to_string().contains("no encode kernel writes"), "{err}");
}

/// Re-encoding one quantized scheme as another is refused — no backend
/// implements a `Transcode`.
#[test]
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

/// A declaration that disagrees with its expression is a compile error, not
/// something that silently triggers a conversion kernel.
#[test]
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

/// With the cast written down, the same pair compiles and publishes the
/// scales the encoder needs.
#[test]
fn the_same_pair_with_the_cast_written_down_encodes() {
    let plan = encode_to("runtime.w", &[64, 64], QuantScheme::Fp8E4M3).unwrap();
    let names: Vec<&str> = plan.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"runtime.w"), "{names:?}");
    assert!(names.contains(&"runtime.w_scale_inv"), "{names:?}");
}
