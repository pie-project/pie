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
use pie_loader::plan::{LoadPlan, StorageInstr, StorageTarget, TileMapKind};
use pie_loader::types::{
    Axis, BackendKind, CheckpointFormat, DType, Encoding, FileId, QuantGranularity, QuantScheme,
    QuantSpec, RepackLayout, ScaleForm, TensorId,
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
                Expr::src("a").cast(Encoding::Raw(DType::BF16)),
                vec![2],
                Encoding::Raw(DType::BF16),
            ),
            TensorContract::new(
                "b.cast",
                Expr::src("b").cast(Encoding::Raw(DType::BF16)),
                vec![2],
                Encoding::Raw(DType::BF16),
            ),
            TensorContract::new(
                "joined",
                Expr::concat(0, vec![Expr::out("a.cast"), Expr::out("b.cast")]),
                vec![4],
                Encoding::Raw(DType::BF16),
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
            Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
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
            Expr::src("fp8").cast(Encoding::Raw(DType::BF16)),
            vec![4],
            Encoding::Raw(DType::BF16),
        )],
        groups: Vec::new(),
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

/// The two-step re-encode [`Expr::Cast`]'s doc promises, actually spelled.
///
/// A quantized tensor cannot be cast straight to another scheme -- no kernel
/// does it, and the destination's scales are not a function of the source's --
/// so the doc offers the route through a decoded intermediate. That route was
/// unspellable until the kernel-operand rule stopped refusing an `Expr::Out`:
/// step two is a `Cast` over the intermediate, and `plan::build` rejected any
/// kernel operand whose lowering read another contract. The refusal was an
/// artifact of the aliasing path claiming the declaration's own tensor id, not
/// anything about the algebra, so the escape route the doc names is the test.
///
/// The decode here is a per-group `Scale`, which is what decodes a block-scaled
/// scheme; `TileMapKind::Decode` is in the plan vocabulary but no backend
/// implements it yet.
#[test]
fn a_quantized_tensor_is_re_encoded_through_a_decoded_intermediate() {
    let int8 = Encoding::Quant(QuantSpec {
        scheme: QuantScheme::Int8Symmetric,
        logical_dtype: DType::BF16,
        bits_per_element: 8,
        group_size: 32,
        channel_axis: Some(Axis(1)),
    });
    let mut contract = block_scaled_contract("scales", "s", vec![4, 1]);
    // `w` is the decoded BF16 tensor the fixture publishes. Make it the
    // intermediate and re-encode it.
    contract.tensors[1] = contract.tensors[1].clone().internal();
    contract.tensors.push(TensorContract::new(
        "w_int8",
        Expr::out("w").cast(int8.clone()),
        vec![4, 32],
        int8,
    ));

    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: pie_loader::plan::CUDA_TILE_MAP_MASK,
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

    // 64 payload + 4 exponents, read once. If the encode had gone back to the
    // checkpoint rather than reading what the decode wrote, this would be more.
    assert_eq!(plan.memory.checkpoint_read_bytes, 68);
    // `Finalize` is what puts a name in the driver's bind table, and an
    // internal declaration gets none. Asserted on the instruction rather than
    // on `plan.tensors`, which lists internal declarations too so a kernel can
    // know their type.
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
        // `w_int8_scale_inv` is the encoder's own factors -- the second half of
        // what a raw-to-quantized cast publishes. `w`, the decoded
        // intermediate, is the one name missing, which is the point.
        vec!["scales", "w_int8_scale_inv", "w_int8"],
        "the intermediate is not bound"
    );
}

/// Casting one quantization scheme straight to another stays refused.
///
/// The companion to the test above: the two-step exists because the one-step
/// does not, so if this ever starts compiling the intermediate is dead weight.
#[test]
fn a_quantized_tensor_may_not_be_cast_straight_to_another_scheme() {
    let err = compile_load_plan(
        &quant_metadata(),
        &ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src("q").cast(Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::BF16))),
                vec![4, 8],
                Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::BF16)),
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
    // Six, not eight: the two biases are a row selection and nothing else, so
    // they are affine and never reach a kernel.
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

/// A repack declaration is checked against its transform like every other node.
///
/// The `Repack` arm used to be the one path through `Builder::tensor` that
/// inferred a type and then discarded it, taking `to.shape` as the answer
/// instead of comparing the two. A declaration that disagreed with the
/// transform compiled silently, and the plan carried the transform's shape
/// under the declaration's name. Found by adding 7 to each of gpt-oss's six
/// repack declarations and watching a clean plan come out; this is that
/// experiment kept.
#[test]
fn a_repack_declaration_is_checked_against_its_transform() {
    let target = StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: pie_loader::plan::CUDA_TILE_MAP_MASK,
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

/// GPT-OSS's gate and up halves are the even and odd rows of one block.
///
/// Each is now an `Expr::Stride`, so each repack reads only the rows it wants:
/// two interleaved gathers of half the block instead of two full reads of all
/// of it that an executor had to cache to make cheap. The two spans are equal,
/// they start one row apart, and together they are the block exactly once.
#[test]
fn gpt_oss_native_mxfp4_reads_each_interleaved_half_once() {
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

    // `gate_up_proj_blocks` is [2, 128, 2, 16] u8, so a row is 32 bytes and the
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

/// The acceptance test for moving the repack's source selection into the
/// algebra.
///
/// The old contract carried `source_row_offset: 64` -- `rank * local` for rank
/// one -- as an integer inside the spec, so the *contract* was valid for
/// exactly one rank and the driver had to re-author it per rank. Flipping the
/// target's rank then changed exactly one line of the plan: the recorded
/// `tp_rank`. Every read offset was identical.
///
/// Now the selection is an `Expr::Shard`, so one contract serves every rank and
/// the rank reaches the plan only through the target. Both halves of that are
/// asserted here: the contract holds no rank-derived integer, and compiling it
/// at two ranks reads two different bands.
#[test]
fn gpt_oss_native_mxfp4_tp_resolves_the_rank_from_the_target() {
    let metadata = gpt_oss_mxfp4_metadata_with_intermediate(128);
    let contract = stored_contract("gpt_oss_native_mxfp4_tp1_of_2");

    // Every repack now takes a composed operand, which is the shape of the
    // claim: the selection is stated in the algebra, not in the spec.
    let repacks = contract
        .tensors
        .iter()
        .filter(|tensor| matches!(tensor.expr, pie_loader::contract::Expr::Repack { .. }))
        .count();
    assert_eq!(
        repacks, 6,
        "a weight and a scale for each of gate, up and down -- the biases are affine"
    );
    assert!(
        contract.tensors.iter().all(|tensor| !matches!(
            &tensor.expr,
            pie_loader::contract::Expr::Repack { src, .. } if matches!(**src, pie_loader::contract::Expr::Src(_))
        )),
        "a repack whose operand is a bare source has nowhere to have put the shard"
    );

    let plan_at = |rank: u32| {
        let target = StorageTarget {
            backend: BackendKind::Cuda,
            tile_map_mask: pie_loader::plan::CUDA_TILE_MAP_MASK,
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

    // The gate/up block is [2, 256, 2, 16], so a row is 32 bytes and an expert
    // is 8192. Rank one's band starts halfway into each expert.
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
        groups: Vec::new(),
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
        groups: Vec::new(),
    };
    let error = compile_load_plan(&metadata(), &contract, StorageTarget::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("declares shape [4]"), "{error}");
    assert!(error.contains("yields [2]"), "{error}");
}

/// A shard that must land on a unit boundary is said by reshaping the axis.
///
/// `Shard` splits an extent, and an extent cannot say what it is a
/// concatenation *of*: `[heads * head_dim, cols]` divides by a `tp_size` that
/// does not divide `heads`, and the split then cuts a head in half. Reshaping
/// to `[heads, head_dim * cols]` first moves the divisibility question onto the
/// thing that has to answer it, and reshaping back afterwards costs nothing --
/// both are byte identities, so the whole composition still compiles to one
/// contiguous run.
///
/// Nemotron-H's Mamba mixer depends on this for every band of its fused
/// `in_proj`. The driver used to shard those on the host after the load, which
/// is how the rule ended up restated as a hand-written divisibility check.
#[test]
fn a_head_boundary_shard_is_one_contiguous_run() {
    // 4 heads of 2 rows, 2 columns; rank 1 of 2 takes heads 2 and 3, which is
    // rows 4..8, which is bytes 32..64 of an F32 tensor.
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
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
            vec![4, 2],
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
    let reads: Vec<&pie_loader::plan::SourceExtent> = program
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
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
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
            vec![4, 2],
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

/// The escape hatches lower through their own arm of `Builder::tensor` instead
/// of through `affine`, so each one has to reach the type checker on its own.
///
/// `Scale` shipped without doing so, and the unit tests over `infer_scale` all
/// passed: they call `infer_type` directly, which the arm that handles the only
/// supported form of the node never did. A zero factor — the value an FFI node
/// carries when the author forgets to set it — compiled to a plan that loaded a
/// tensor of zeros, cached it, and ran. Everything asserted below is a rule
/// `infer_scale` already stated; what is under test is that `compile` asks.
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
    let metadata = CheckpointMetadata {
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
            Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
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

/// Every path through the compiler names the contract its error came from.
///
/// `Builder::tensor` used to annotate the affine path at the call site and the
/// kernel paths call by call, so the same mistake read `'out': declares shape
/// [4] ...` through one and `declares shape [4] ...` through the other. In a
/// contract with hundreds of tensors the second message names nothing. The
/// annotation now happens once, at the boundary, for every path.
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

/// A checkpoint holding one block-scaled MXFP4 tensor and its factors.
///
/// Both arrive as the bytes a safetensors file actually stores — `U8` payloads
/// with no encoding of their own — because that is what a checkpoint records.
/// The contract is what says one of them is four rows of thirty-two MXFP4
/// elements and the other is one E8M0 exponent per row.
fn block_scaled_metadata() -> CheckpointMetadata {
    CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: 256,
            format: CheckpointFormat::Safetensors,
        }],
        tensors: vec![
            sized_raw(0, "w", 0, 64, &[4, 16], DType::U8),
            sized_raw(1, "s", 64, 4, &[4, 1], DType::U8),
            // Two more factor tensors, sized so that the shapes the rejection
            // tests need are legal renames of *something*: three exponents do
            // not divide four rows, and 128 give one per element.
            sized_raw(2, "s3", 68, 3, &[3, 1], DType::U8),
            sized_raw(3, "s128", 71, 128, &[128], DType::U8),
        ],
    }
}

fn mxfp4(channel_axis: u8) -> QuantSpec {
    QuantSpec {
        channel_axis: Some(Axis(channel_axis)),
        ..quant(QuantScheme::Mxfp4E2M1E8M0, DType::BF16)
    }
}

/// The dequantization a driver used to do by hand, said as one declaration.
///
/// `factors` names the tensor holding the exponents, and is the whole point:
/// the payload and its factors are paired by a name the contract already
/// checks, not by a suffix the executor appends to a name and hopes for.
///
/// `from` and `shape` are what the factors are, because **the factors' shape
/// is the whole statement of how the payload is blocked**. There is no group
/// size and no axis to pass: `[4, 1]` over `[4, 32]` says one factor per row,
/// `[2, 2]` says 2x16 tiles. Every rejection below is therefore a shape.
fn block_scaled_contract(factors: &str, from: &str, shape: Vec<i64>) -> ModelContract {
    ModelContract {
        alignment: 256,
        tensors: vec![
            TensorContract::new(
                "scales",
                Expr::src(from).transmute(TensorType {
                    shape: shape.clone(),
                    encoding: Encoding::Raw(DType::E8M0),
                }),
                shape,
                Encoding::Raw(DType::E8M0),
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
                Encoding::Raw(DType::BF16),
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

/// The capability the routed-expert families need: each rank dequantizes the
/// shard it will compute with, and never sees the rest.
///
/// The shard is byte spans and the multiply is a kernel, so the two compose
/// without either one learning about the other. What makes this worth pinning
/// is the alignment it depends on: the factors were sharded by the contract
/// that published them, and the payload is sharded here, and `infer` compares
/// the two *after* both have been specialized for the rank.
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
                        encoding: Encoding::Raw(DType::E8M0),
                    })
                    .shard(0),
                vec![2, 1],
                Encoding::Raw(DType::E8M0),
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
                vec![2, 32],
                Encoding::Raw(DType::BF16),
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

/// Blocks on two axes at once, which one group size could not have said.
///
/// `[2, 2]` factors over a `[4, 32]` payload is a 2x16 tile. Nothing in the
/// contract names either number: both fall out of the ratio, and the plan is
/// where they first appear as numbers -- which is what makes them checkable
/// against the two shapes rather than trusted.
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

/// Scaling by an expression, rather than by a tensor some contract published,
/// is refused — and refused with the fix in the message.
///
/// The restriction is what keeps one copy of the factors: a driver reads them
/// anyway, so the contract that publishes them is the one that should say what
/// they are.
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
                    encoding: Encoding::Raw(DType::E8M0),
                })),
            vec![4, 32],
            Encoding::Raw(DType::BF16),
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
            Expr::src("w.weight").cast(Encoding::Quant(quant(
                QuantScheme::Mxfp4E2M1E8M0,
                DType::BF16,
            ))),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Mxfp4E2M1E8M0, DType::BF16)),
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
            Expr::src("w.weight").cast(Encoding::Quant(quant(
                QuantScheme::Mxfp4E2M1E8M0,
                DType::BF16,
            ))),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Mxfp4E2M1E8M0, DType::BF16)),
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
            Expr::concat(
                1,
                vec![
                    Expr::src("q_proj.weight"),
                    Expr::fill(0.0, TensorType::raw(vec![4, 1], DType::BF16)),
                ],
            ),
            vec![4, 5],
            Encoding::Raw(DType::BF16),
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

/// And the padding is actually zero when the plan runs.
///
/// The test above proves the *plan* says `Fill`; nothing proved that executing
/// it produces zeros, because no golden carries a `Fill` and the goldens are
/// what drive the host executor. So `HostExecutor::fill` was reachable code
/// that no test had ever run. The pad is the one region of a materialized
/// tensor whose bytes come from no source, which makes it exactly the region a
/// missing zeroing would leave holding whatever the allocator last had.
#[test]
fn a_padded_head_dim_materializes_zeros_where_no_source_covers() {
    let dir = std::env::temp_dir().join(format!("pie_fill_replay_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let snapshot = dir.join("model.safetensors");

    // Four rows of four bf16 elements, every byte non-zero, so a pad that was
    // left uninitialised cannot pass by coincidence and a pad written from the
    // wrong offset shows up as source bytes rather than zeros.
    let source: Vec<u8> = (0..32).map(|i| (i as u8) | 0x80).collect();
    std::fs::write(&snapshot, &source).unwrap();

    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
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
            encoding: Encoding::Raw(DType::BF16),
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
                    Expr::fill(0.0, TensorType::raw(vec![4, 1], DType::BF16)),
                ],
            ),
            vec![4, 5],
            Encoding::Raw(DType::BF16),
        )],
    };

    let plan = compile_load_plan(&metadata, &contract, StorageTarget::default()).unwrap();
    let storage = pie_loader::executor::host::execute_plan(&plan, &dir)
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
            Expr::src("w.scale")
                .transmute(TensorType::raw(vec![8, 8], DType::E8M0))
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
            Expr::src("w.weight").cast(Encoding::Quant(quant(
                QuantScheme::Mxfp4E2M1E8M0,
                DType::BF16,
            ))),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Mxfp4E2M1E8M0, DType::BF16)),
        )],
        groups: Vec::new(),
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
            Expr::src("w.weight").cast(Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::BF16))),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::BF16)),
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
                Expr::src("w.weight")
                    .cast(Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::BF16))),
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
        groups: Vec::new(),
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
        groups: Vec::new(),
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
/// unreadable one. Each of these used to compile, publish exactly one tensor,
/// and hand the driver an encoding whose scales did not exist.
fn encode_to(
    name: &str,
    shape: &[i64],
    scheme: QuantScheme,
) -> Result<LoadPlan, pie_loader::error::Error> {
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
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
            DType::BF16,
        )],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            name,
            Expr::src("w.weight").cast(Encoding::Quant(quant(scheme, DType::BF16))),
            shape.to_vec(),
            Encoding::Quant(quant(scheme, DType::BF16)),
        )],
        groups: Vec::new(),
    };
    compile_load_plan(&metadata, &contract, scale_target())
}

/// The encode kernels walk `[rows, cols]` tiles, so a rank-3 output has no
/// scale layout at all -- not a different one.
#[test]
fn an_encode_that_cannot_place_its_scales_is_refused() {
    let err = encode_to("runtime.w", &[2, 64, 64], QuantScheme::Mxfp4E2M1E8M0).unwrap_err();
    assert!(err.to_string().contains("rank-3"), "{err}");
    assert!(err.to_string().contains("runtime.w"), "{err}");
}

/// The column count is part of the scale layout, not just of the payload.
#[test]
fn an_encode_whose_columns_do_not_fill_a_block_is_refused() {
    let err = encode_to("runtime.w", &[64, 48], QuantScheme::Mxfp4E2M1E8M0).unwrap_err();
    assert!(err.to_string().contains("blocks 32 columns"), "{err}");
}

/// `QuantScheme` names every format the loader can *read*. Only three have an
/// encoder, and asking for one of the others used to be accepted in silence.
#[test]
fn an_encode_into_a_scheme_no_kernel_writes_is_refused() {
    let err = encode_to("runtime.w", &[64, 64], QuantScheme::AwqInt4).unwrap_err();
    assert!(err.to_string().contains("no encode kernel writes"), "{err}");
}

/// Re-encoding one quantized scheme as another produced a `Transcode` neither
/// backend implements, and no scale tensor for the new scheme's blocks. It is
/// now refused where it is written.
#[test]
fn re_encoding_one_quantized_scheme_as_another_is_refused() {
    let metadata = CheckpointMetadata {
        files: vec![CheckpointFile {
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
            encoding: Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::BF16)),
        }],
    };
    let contract = ModelContract {
        alignment: 1,
        tensors: vec![TensorContract::new(
            "runtime.w",
            Expr::src("w.weight").cast(Encoding::Quant(quant(
                QuantScheme::Mxfp4E2M1E8M0,
                DType::BF16,
            ))),
            vec![64, 64],
            Encoding::Quant(quant(QuantScheme::Mxfp4E2M1E8M0, DType::BF16)),
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

/// The defect `Expr::Cast` closes: a declaration that disagreed with its
/// expression used to *cause* a conversion, so two lines that were merely
/// inconsistent silently ran a kernel and — encoding — invented a scale tensor
/// nothing in the type layer had checked.
#[test]
fn a_declaration_that_disagrees_with_its_expression_is_a_mistake_not_a_kernel() {
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
        groups: Vec::new(),
    };
    let err = compile_load_plan(&metadata, &contract, scale_target())
        .unwrap_err()
        .to_string();
    assert!(err.contains("but its expression yields Raw(BF16)"), "{err}");
    assert!(err.contains("explicit cast"), "{err}");
}

/// And with the cast written down, the same pair compiles and publishes the
/// scales the encoder needs — so the assertion is not merely refusing work.
#[test]
fn the_same_pair_with_the_cast_written_down_encodes() {
    let plan = encode_to("runtime.w", &[64, 64], QuantScheme::Fp8E4M3).unwrap();
    let names: Vec<&str> = plan.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"runtime.w"), "{names:?}");
    assert!(names.contains(&"runtime.w_scale_inv"), "{names:?}");
}
