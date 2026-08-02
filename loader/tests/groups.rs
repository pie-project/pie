//! Groups: the loader's claim that a set of tensors is interchangeable.
//!
//! Two ways to write a group, and they are the two ways a checkpoint stores a
//! repeated structure: as one fused bank indexed along an axis
//! ([`Expr::select`]), or as separately named tensors distinguished by a number
//! in the name ([`Expr::src_indexed`]). Everything else here is about the one
//! guarantee the plan makes for both -- that instance `i` differs from instance
//! 0 only in which bytes it reads.

use pie_loader::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
use pie_loader::contract::{Expr, GroupContract, ModelContract, TensorContract};
use pie_loader::plan::{StorageInstr, StorageTarget, compile};
use pie_loader::types::{BackendKind, CheckpointFormat, DType, Encoding, FileId, TensorId};

const EXPERTS: u32 = 4;
const ROWS: i64 = 8;
const COLS: i64 = 16;

fn target() -> StorageTarget {
    StorageTarget {
        backend: BackendKind::Cuda,
        tile_map_mask: pie_loader::plan::CUDA_TILE_MAP_MASK,
        ..StorageTarget::default()
    }
}

fn tensor(id: u32, name: &str, offset: u64, shape: &[i64]) -> RawTensor {
    let span = shape.iter().product::<i64>() as u64 * DType::BF16.bytes();
    RawTensor {
        id: TensorId(id),
        name: name.to_string(),
        file_id: FileId(0),
        file_offset: offset,
        span_bytes: span,
        shape: shape.to_vec(),
        encoding: Encoding::Raw(DType::BF16),
    }
}

fn checkpoint(tensors: Vec<RawTensor>) -> CheckpointMetadata {
    let size = tensors
        .iter()
        .map(|t| t.file_offset + t.span_bytes)
        .max()
        .unwrap_or(0);
    CheckpointMetadata {
        files: vec![CheckpointFile {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes: size,
            format: CheckpointFormat::Safetensors,
        }],
        tensors,
    }
}

/// One `[E, ROWS, COLS]` bank, the way GPT-OSS and Qwen store their experts.
fn fused_checkpoint() -> CheckpointMetadata {
    checkpoint(vec![tensor(
        0,
        "experts.bank",
        0,
        &[EXPERTS as i64, ROWS, COLS],
    )])
}

/// `EXPERTS` separately named tensors, the way DeepSeek and Mixtral store them.
fn named_checkpoint() -> CheckpointMetadata {
    let span = (ROWS * COLS) as u64 * DType::BF16.bytes();
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
            Encoding::Raw(DType::BF16),
        )],
    }
}

/// The same group, declared at the fused bank's rank: a selected band keeps
/// the axis it was selected along, with an extent of one.
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

/// Selecting index `i` out of a fused bank: the group's expression drops the
/// bank's leading axis, and the plan says where each instance's slab begins.
#[test]
fn a_selected_band_of_a_fused_bank_is_one_plan_and_a_table_of_offsets() {
    let expr = Expr::src("experts.bank").select(0, 1, 1);
    let plan = compile(&fused_checkpoint(), &contract(banded(expr)), target()).unwrap();

    assert_eq!(plan.tensors.len(), 0, "a group publishes nothing resident");
    assert_eq!(plan.groups.len(), 1);
    let group = &plan.groups[0];
    assert_eq!(group.name, "experts");
    assert_eq!(group.arity, EXPERTS);
    assert_eq!(group.bindings.len(), EXPERTS as usize);

    // One read per instance, and the offsets march by exactly one band.
    let band = (ROWS * COLS) as u64 * DType::BF16.bytes();
    for (index, binding) in group.bindings.iter().enumerate() {
        assert_eq!(binding.len(), 1, "index {index} reads once");
        assert_eq!(binding[0].tensor_id, TensorId(0));
        assert_eq!(binding[0].file_offset, index as u64 * band);
    }

    // The template is a real plan, and it publishes the group's tensor.
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

/// Naming index `i`: the same group, written against a checkpoint that stores
/// each instance as its own tensor.
#[test]
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
        // Each instance reads a *different tensor*, which is the thing a
        // selected band cannot express and this node exists for.
        assert_eq!(binding[0].tensor_id, TensorId(index as u32));
    }
}

/// The two spellings of the same group must produce the same program, because
/// they describe the same load. Only the bindings differ.
#[test]
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

    assert_eq!(
        fused.groups[0].plan.buffers, named.groups[0].plan.buffers,
        "the same tensor needs the same buffer either way"
    );
    assert_eq!(fused.groups[0].plan.memory, named.groups[0].plan.memory);
}

/// A group whose instances are not the same shape is not a group. Caught at
/// compile time, with the index and the name it resolved to.
#[test]
fn an_instance_of_a_different_shape_is_rejected() {
    let span = (ROWS * COLS) as u64 * DType::BF16.bytes();
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
    // The last one is half as tall.
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

/// A template that names a tensor the checkpoint does not have reports both
/// the index and the name it produced -- the name is not in any source file,
/// so without the index there is nothing to search for.
#[test]
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

/// An `arity` wider than the bank runs off the end, and says so rather than
/// producing a slot of whatever follows in the file.
#[test]
fn an_arity_wider_than_its_bank_is_rejected() {
    let mut group = banded(Expr::src("experts.bank").select(0, 1, 1));
    group.arity = EXPERTS + 1;

    let err = compile(&fused_checkpoint(), &contract(group), target())
        .unwrap_err()
        .to_string();
    assert!(err.contains(&format!("index {EXPERTS}")), "{err}");
}

/// The template language is one `{}` and nothing else. A template with none is
/// a constant, and a constant instance is not an instance.
#[test]
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

#[test]
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

/// An index node outside a group has nothing to stand for, and the message
/// says that rather than defaulting to zero.
#[test]
fn an_index_node_outside_a_group_is_rejected() {
    let contract = ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "w",
            Expr::src_indexed("experts.{}.w"),
            vec![ROWS, COLS],
            Encoding::Raw(DType::BF16),
        )],
        groups: Vec::new(),
    };
    let err = compile(&named_checkpoint(), &contract, target())
        .unwrap_err()
        .to_string();
    assert!(err.contains("group"), "{err}");
}

#[test]
fn a_group_of_arity_zero_is_rejected() {
    let mut group = group(Expr::src_indexed("experts.{}.w"));
    group.arity = 0;
    let err = compile(&named_checkpoint(), &contract(group), target())
        .unwrap_err()
        .to_string();
    assert!(err.contains("arity 0"), "{err}");
}

#[test]
fn two_groups_of_one_name_are_rejected() {
    let mut c = contract(group(Expr::src_indexed("experts.{}.w")));
    c.groups.push(c.groups[0].clone());
    let err = compile(&named_checkpoint(), &c, target())
        .unwrap_err()
        .to_string();
    assert!(err.contains("twice"), "{err}");
}

/// Put the checkpoint on disk, because verification stats the file.
///
/// `verify` reads file sizes from the filesystem rather than believing the
/// plan, which is the whole value of the check. So a test that verifies has to
/// have a file.
fn on_disk(mut meta: CheckpointMetadata, tag: &str) -> (CheckpointMetadata, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("pie-loader-groups-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.safetensors");
    std::fs::write(&path, vec![0u8; meta.files[0].size_bytes as usize]).unwrap();
    meta.files[0].path = path.to_string_lossy().into_owned();
    (meta, dir)
}

/// The plan the driver sees carries its groups, and every instance of every
/// group is verified -- not just the index the template was compiled at.
#[test]
fn a_marshalled_group_survives_the_abi_and_every_instance_is_checked() {
    use pie_loader_capi::arena;
    use pie_loader_capi::view::verify_marshalled;

    let (meta, dir) = on_disk(named_checkpoint(), "abi");
    let plan = compile(
        &meta,
        &contract(group(Expr::src_indexed("experts.{}.w"))),
        target(),
    )
    .unwrap();
    verify_marshalled(&plan, None).expect("a well-formed group verifies");

    // Round-trips as POD, with the child plan reachable and the bindings laid
    // out instance-major.
    let pod = arena::build(&plan, "0000000000000000");
    assert!(!pod.is_null());
    unsafe {
        let groups = std::slice::from_raw_parts((*pod).groups.ptr, (*pod).groups.len);
        assert_eq!(groups.len(), 1);
        let view = &groups[0];
        assert_eq!(view.arity, EXPERTS);
        assert_eq!(view.bindings_per_instance, 1);
        assert_eq!(view.bindings.len, EXPERTS as usize);
        assert!(!view.plan.is_null());
        assert!((*view.plan).instrs.len > 0, "the child plan is a real plan");
        let bindings = std::slice::from_raw_parts(view.bindings.ptr, view.bindings.len);
        for (index, binding) in bindings.iter().enumerate() {
            assert_eq!(binding.tensor_id, index as u32);
        }
        arena::release(pod);
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// A binding that reads off the end of its file is caught even though the
/// template -- index 0 -- is perfectly in bounds.
#[test]
fn an_instance_that_reads_past_its_file_is_rejected() {
    use pie_loader_capi::view::verify_marshalled;

    let (meta, dir) = on_disk(named_checkpoint(), "past-end");
    let mut plan = compile(
        &meta,
        &contract(group(Expr::src_indexed("experts.{}.w"))),
        target(),
    )
    .unwrap();
    // Only the last instance is moved, and only past the end. Index 0 still
    // verifies, which is exactly the case a template-only check would miss.
    let last = plan.groups[0].bindings.last_mut().unwrap();
    last[0].file_offset = 1 << 40;

    let violations = verify_marshalled(&plan, None).unwrap_err();
    let text = violations
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ");
    assert!(text.contains("index 3"), "{text}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
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
                vec![ROWS / 2, COLS],
                Encoding::Raw(DType::BF16),
            )],
        }],
    };
    let rank1 = StorageTarget {
        tp_rank: 1,
        tp_size: 2,
        ..target()
    };
    let plan = compile(&named_checkpoint(), &contract, rank1).unwrap();

    // Rank 1 reads the second half of each instance's rows: the instance
    // chooses the tensor, the rank chooses the offset within it.
    let half = (ROWS / 2 * COLS) as u64 * DType::BF16.bytes();
    let span = (ROWS * COLS) as u64 * DType::BF16.bytes();
    for (index, binding) in plan.groups[0].bindings.iter().enumerate() {
        assert_eq!(binding[0].tensor_id, TensorId(index as u32));
        assert_eq!(binding[0].file_offset, index as u64 * span + half);
    }
}

/// Which storage shape a group uses decides whether it can be streamed at all,
/// and the plan is where that shows: the loader states it in the bindings
/// rather than leaving a reader to infer it from names.
///
/// Streaming reads a weight where it lies, through a mapping, so an instance
/// has to begin on a page of its own. When each instance is its own tensor,
/// placing that tensor on a page places the instance -- the instance *is* the
/// tensor. When every instance is a band of one fused bank, placing the bank
/// places instance 0 and no other: the rest begin at `base + i * stride`, and
/// the stride is whatever the shape makes it. No choice of where to put the
/// bank changes that, which is why the two shapes are not interchangeable.
mod streamability {
    use super::*;
    use std::collections::{BTreeSet, HashMap};

    #[derive(Default)]
    struct Streamable {
        /// Tensors a group instance reads in full: the instance *is* the
        /// tensor, so it can begin on a page of its own.
        whole: BTreeSet<String>,
        /// Fused banks. Every instance reads a slice of one tensor, so
        /// instance 0 can begin on a page and no other: the rest start at
        /// `base + i * stride`, and the stride is whatever the shape makes it
        /// -- in GPT-OSS-20B, 4147200 bytes, which no page size divides.
        banded: BTreeSet<String>,
    }

    /// Reads the distinction off the plan. A binding whose file offset is its
    /// tensor's own offset reads that tensor whole; one that is displaced from
    /// it reads a band.
    fn streamable_tensors(
        plan: &pie_loader::plan::LoadPlan,
        metadata: &pie_loader::checkpoint::CheckpointMetadata,
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
        // A bank whose instance 0 binds its base looks whole from that one
        // binding. What settles it is any sibling binding that does not.
        for name in &out.banded {
            out.whole.remove(name);
        }
        out
    }

    /// A bank is reported as a bank. Streaming one would fault in all four
    /// experts to read one -- correct, slower than not streaming, and with
    /// nothing on disk to say so.
    #[test]
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

    /// Separately named instances are each pageable, and every one of them is
    /// reported -- a reader told about only some would leave the rest streaming
    /// off pages they share.
    #[test]
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

    /// A contract with no groups declares nothing interchangeable, so there is
    /// nothing to stream.
    #[test]
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
