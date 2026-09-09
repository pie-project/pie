use checkpoint::contract::{Expr, GroupContract, ModelContract, TensorContract};
use checkpoint::file::{File, Metadata, RawTensor};
use checkpoint::plan::{StorageInstr, StorageTarget, compile};
use checkpoint::types::{BackendKind, CheckpointFormat, DType, Encoding, FileId, TensorId};

const EXPERTS: u32 = 4;
const ROWS: i64 = 8;
const COLS: i64 = 16;

fn target() -> StorageTarget {
    StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: checkpoint::plan::CUDA_TILE_MAP_MASK,
        ..StorageTarget::default()
    }
}

fn tensor(id: u32, name: &str, offset: u64, shape: &[i64]) -> RawTensor {
    let span = shape.iter().product::<i64>() as u64 * DType::Bf16.bytes_ceil();
    RawTensor {
        id: TensorId(id),
        name: name.to_string(),
        file_id: FileId(0),
        file_offset: offset,
        span_bytes: span,
        shape: shape.to_vec(),
        encoding: Encoding::Raw(DType::Bf16),
    }
}

fn checkpoint(tensors: Vec<RawTensor>) -> Metadata {
    let size = tensors
        .iter()
        .map(|t| t.file_offset + t.span_bytes)
        .max()
        .unwrap_or(0);
    Metadata {
        files: vec![File {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: size,
            format: CheckpointFormat::Safetensors,
        }],
        tensors,
    }
}

fn fused_checkpoint() -> Metadata {
    checkpoint(vec![tensor(
        0,
        "experts.bank",
        0,
        &[EXPERTS as i64, ROWS, COLS],
    )])
}

fn named_checkpoint() -> Metadata {
    let span = (ROWS * COLS) as u64 * DType::Bf16.bytes_ceil();
    checkpoint(
        (0..EXPERTS)
            .map(|e| {
                tensor(
                    e,
                    &format!("experts.{e}.w"),
                    u64::from(e) * span,
                    &[ROWS, COLS],
                )
            })
            .collect(),
    )
}

fn group(expr: Expr) -> GroupContract {
    GroupContract {
        name: "experts".to_string(),
        arity: EXPERTS,
        tensors: vec![TensorContract::new(
            "w",
            expr,
            vec![ROWS, COLS],
            Encoding::Raw(DType::Bf16),
        )],
    }
}

fn banded(expr: Expr) -> GroupContract {
    let mut g = group(expr);
    g.tensors[0].shape = Some(vec![1, ROWS, COLS]);
    g
}

fn contract(group: GroupContract) -> ModelContract {
    ModelContract {
        alignment: 256,
        tensors: Vec::new(),
        groups: vec![group],
    }
}

#[test]
fn groups_every_case() {
    a_selected_band_of_a_fused_bank_is_one_plan_and_a_table_of_offsets();
    an_indexed_source_name_resolves_once_per_instance();
    both_spellings_of_a_group_compile_to_the_same_program();
    an_instance_of_a_different_shape_is_rejected();
    a_missing_instance_names_the_index_and_the_resolved_name();
    an_arity_wider_than_its_bank_is_rejected();
    a_template_without_a_placeholder_is_rejected();
    a_template_with_two_placeholders_is_rejected();
    an_index_node_outside_a_group_is_rejected();
    a_group_of_arity_zero_is_rejected();
    two_groups_of_one_name_are_rejected();
    every_instance_of_a_group_is_checked();
    an_instance_that_reads_past_its_file_is_rejected();
    a_group_composes_with_a_shard();
}

fn a_selected_band_of_a_fused_bank_is_one_plan_and_a_table_of_offsets() {
    let expr = Expr::src("experts.bank").select(0, 1, 1);
    let plan = compile(&fused_checkpoint(), &contract(banded(expr)), target()).unwrap();

    assert_eq!(plan.tensors.len(), 0, "a group publishes nothing resident");
    assert_eq!(plan.groups.len(), 1);
    let group = &plan.groups[0];
    assert_eq!(group.name, "experts");
    assert_eq!(group.arity, EXPERTS);
    assert_eq!(group.bindings.len(), EXPERTS as usize);

    let band = (ROWS * COLS) as u64 * DType::Bf16.bytes_ceil();
    for (index, binding) in group.bindings.iter().enumerate() {
        assert_eq!(binding.len(), 1, "index {index} reads once");
        assert_eq!(binding[0].tensor_id, TensorId(0));
        assert_eq!(binding[0].file_offset, index as u64 * band);
    }

    assert!(
        group
            .plan
            .instrs
            .iter()
            .any(|i| matches!(i, StorageInstr::Finalize { name, .. } if name == "w")),
        "{:#?}",
        group.plan.instrs
    );
}

fn an_indexed_source_name_resolves_once_per_instance() {
    let plan = compile(
        &named_checkpoint(),
        &contract(group(Expr::src_indexed("experts.{}.w"))),
        target(),
    )
    .unwrap();

    let group = &plan.groups[0];
    for (index, binding) in group.bindings.iter().enumerate() {
        assert_eq!(binding.len(), 1);
        assert_eq!(binding[0].tensor_id, TensorId(index as u32));
    }
}

fn both_spellings_of_a_group_compile_to_the_same_program() {
    let fused = compile(
        &fused_checkpoint(),
        &contract(banded(Expr::src("experts.bank").select(0, 1, 1))),
        target(),
    )
    .unwrap();
    let named = compile(
        &named_checkpoint(),
        &contract(group(Expr::src_indexed("experts.{}.w"))),
        target(),
    )
    .unwrap();

    let fused_buffers = &fused.groups[0].plan.buffers;
    let named_buffers = &named.groups[0].plan.buffers;
    assert_eq!(fused_buffers.len(), named_buffers.len());
    for (fused, named) in fused_buffers.iter().zip(named_buffers) {
        assert_eq!(
            (
                fused.id,
                fused.tensor,
                fused.bytes,
                fused.alignment,
                fused.temporary,
                fused.persistent_offset,
                fused.scratch_offset,
                &fused.ty.encoding,
            ),
            (
                named.id,
                named.tensor,
                named.bytes,
                named.alignment,
                named.temporary,
                named.persistent_offset,
                named.scratch_offset,
                &named.ty.encoding,
            ),
            "the same tensor needs the same buffer either way"
        );
        assert_eq!(
            fused.ty.element_count().unwrap(),
            named.ty.element_count().unwrap(),
            "and the two ranks describe the same elements"
        );
    }
    assert_eq!(fused.groups[0].plan.memory, named.groups[0].plan.memory);
}

fn an_instance_of_a_different_shape_is_rejected() {
    let span = (ROWS * COLS) as u64 * DType::Bf16.bytes_ceil();
    let mut tensors: Vec<RawTensor> = (0..EXPERTS)
        .map(|e| {
            tensor(
                e,
                &format!("experts.{e}.w"),
                u64::from(e) * span,
                &[ROWS, COLS],
            )
        })
        .collect();
    tensors[3] = tensor(3, "experts.3.w", 3 * span, &[ROWS / 2, COLS]);

    let err = compile(
        &checkpoint(tensors),
        &contract(group(Expr::src_indexed("experts.{}.w"))),
        target(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("index 3"), "{err}");
    assert!(err.contains("experts"), "{err}");
}

fn a_missing_instance_names_the_index_and_the_resolved_name() {
    let mut tensors = named_checkpoint().tensors;
    tensors.retain(|t| t.name != "experts.2.w");

    let err = compile(
        &checkpoint(tensors),
        &contract(group(Expr::src_indexed("experts.{}.w"))),
        target(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("index 2"), "{err}");
    assert!(err.contains("experts.2.w"), "{err}");
}

fn an_arity_wider_than_its_bank_is_rejected() {
    let mut group = banded(Expr::src("experts.bank").select(0, 1, 1));
    group.arity = EXPERTS + 1;

    let err = compile(&fused_checkpoint(), &contract(group), target())
        .unwrap_err()
        .to_string();
    assert!(err.contains(&format!("index {EXPERTS}")), "{err}");
}

fn a_template_without_a_placeholder_is_rejected() {
    let err = compile(
        &named_checkpoint(),
        &contract(group(Expr::src_indexed("experts.0.w"))),
        target(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("{}"), "{err}");
}

fn a_template_with_two_placeholders_is_rejected() {
    let err = compile(
        &named_checkpoint(),
        &contract(group(Expr::src_indexed("experts.{}.{}.w"))),
        target(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("{}"), "{err}");
}

fn an_index_node_outside_a_group_is_rejected() {
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "w",
            Expr::src_indexed("experts.{}.w"),
            vec![ROWS, COLS],
            Encoding::Raw(DType::Bf16),
        )],
        groups: Vec::new(),
    };
    let err = compile(&named_checkpoint(), &contract, target())
        .unwrap_err()
        .to_string();
    assert!(err.contains("group"), "{err}");
}

fn a_group_of_arity_zero_is_rejected() {
    let mut group = group(Expr::src_indexed("experts.{}.w"));
    group.arity = 0;
    let err = compile(&named_checkpoint(), &contract(group), target())
        .unwrap_err()
        .to_string();
    assert!(err.contains("arity 0"), "{err}");
}

fn two_groups_of_one_name_are_rejected() {
    let mut c = contract(group(Expr::src_indexed("experts.{}.w")));
    c.groups.push(c.groups[0].clone());
    let err = compile(&named_checkpoint(), &c, target())
        .unwrap_err()
        .to_string();
    assert!(err.contains("twice"), "{err}");
}

fn on_disk(mut meta: Metadata, tag: &str) -> (Metadata, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("pie-loader-groups-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    std::fs::write(&path, vec![0u8; meta.files[0].size_bytes as usize]).unwrap();
    meta.files[0].path = path.to_string_lossy().into_owned();
    (meta, dir)
}

fn every_instance_of_a_group_is_checked() {
    use checkpoint::verify::verify_plan;

    let (meta, dir) = on_disk(named_checkpoint(), "abi");
    let plan = compile(
        &meta,
        &contract(group(Expr::src_indexed("experts.{}.w"))),
        target(),
    )
    .unwrap();
    verify_plan(&plan, None).expect("a well-formed group verifies");
    assert_eq!(plan.groups.len(), 1);
    let g = &plan.groups[0];
    assert_eq!(g.arity, EXPERTS);
    assert_eq!(g.bindings.len(), EXPERTS as usize);
    assert!(!g.plan.instrs.is_empty(), "the child plan is a real plan");
    for (index, bindings) in g.bindings.iter().enumerate() {
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].tensor_id.0, index as u32);
    }
    let _ = std::fs::remove_dir_all(dir);
}

fn an_instance_that_reads_past_its_file_is_rejected() {
    use checkpoint::verify::verify_plan;

    let (meta, dir) = on_disk(named_checkpoint(), "past-end");
    let mut plan = compile(
        &meta,
        &contract(group(Expr::src_indexed("experts.{}.w"))),
        target(),
    )
    .unwrap();
    let last = plan.groups[0].bindings.last_mut().unwrap();
    last[0].file_offset = 1 << 40;

    let violations = verify_plan(&plan, None).unwrap_err();
    let text = violations
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ");
    assert!(text.contains("index 3"), "{text}");
    let _ = std::fs::remove_dir_all(dir);
}

fn a_group_composes_with_a_shard() {
    let expr = Expr::src_indexed("experts.{}.w").shard(0);
    let contract = ModelContract {
        alignment: 256,
        tensors: Vec::new(),
        groups: vec![GroupContract {
            name: "experts".to_string(),
            arity: EXPERTS,
            tensors: vec![TensorContract::new(
                "w",
                expr,
                vec![ROWS, COLS],
                Encoding::Raw(DType::Bf16),
            )],
        }],
    };
    let rank1 = StorageTarget {
        tp_rank: 1,
        tp_size: 2,
        ..target()
    };
    let plan = compile(&named_checkpoint(), &contract, rank1).unwrap();

    let half = (ROWS / 2 * COLS) as u64 * DType::Bf16.bytes_ceil();
    let span = (ROWS * COLS) as u64 * DType::Bf16.bytes_ceil();
    for (index, binding) in plan.groups[0].bindings.iter().enumerate() {
        assert_eq!(binding[0].tensor_id, TensorId(index as u32));
        assert_eq!(binding[0].file_offset, index as u64 * span + half);
    }
}

mod streamability {
    use super::*;
    use std::collections::{BTreeSet, HashMap};

    #[derive(Default)]
    struct Streamable {
        whole: BTreeSet<String>,
        banded: BTreeSet<String>,
    }

    fn streamable_tensors(
        plan: &checkpoint::plan::LoadPlan,
        metadata: &checkpoint::file::Metadata,
    ) -> Streamable {
        let by_id: HashMap<_, _> = metadata.tensors.iter().map(|t| (t.id, t)).collect();
        let mut out = Streamable::default();
        for group in &plan.groups {
            for instance in &group.bindings {
                for binding in instance {
                    let Some(tensor) = by_id.get(&binding.tensor_id) else {
                        continue;
                    };
                    if binding.file_offset == tensor.file_offset {
                        out.whole.insert(tensor.name.clone());
                    } else {
                        out.banded.insert(tensor.name.clone());
                    }
                }
            }
        }
        for name in &out.banded {
            out.whole.remove(name);
        }
        out
    }

    #[test]
    fn groups_1_every_case() {
        a_fused_bank_cannot_be_paged_by_instance();
        separately_named_instances_are_each_pageable();
        a_contract_without_groups_offers_nothing_to_stream();
    }

    fn a_fused_bank_cannot_be_paged_by_instance() {
        let expr = Expr::src("experts.bank").select(0, 1, 1);
        let metadata = fused_checkpoint();
        let plan = compile(&metadata, &contract(banded(expr)), target()).unwrap();

        let streamable = streamable_tensors(&plan, &metadata);
        assert_eq!(
            streamable.banded.iter().cloned().collect::<Vec<_>>(),
            vec!["experts.bank".to_string()]
        );
        assert!(
            streamable.whole.is_empty(),
            "instance 0 binds the bank's own offset and must not be mistaken for a \
             whole tensor on that evidence: {:?}",
            streamable.whole
        );
    }

    fn separately_named_instances_are_each_pageable() {
        let expr = Expr::src_indexed("experts.{}.w");
        let metadata = named_checkpoint();
        let plan = compile(&metadata, &contract(group(expr)), target()).unwrap();

        let streamable = streamable_tensors(&plan, &metadata);
        assert!(streamable.banded.is_empty(), "{:?}", streamable.banded);
        assert_eq!(
            streamable.whole.iter().cloned().collect::<Vec<_>>(),
            (0..EXPERTS)
                .map(|e| format!("experts.{e}.w"))
                .collect::<Vec<_>>()
        );
    }

    fn a_contract_without_groups_offers_nothing_to_stream() {
        let metadata = named_checkpoint();
        let plan = compile(
            &metadata,
            &ModelContract {
                alignment: 256,
                tensors: Vec::new(),
                groups: Vec::new(),
            },
            target(),
        )
        .unwrap();

        let streamable = streamable_tensors(&plan, &metadata);
        assert!(streamable.whole.is_empty() && streamable.banded.is_empty());
    }
}
