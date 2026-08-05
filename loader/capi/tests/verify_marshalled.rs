//! Does a marshalled plan honour its contract?
//!
//! The verifier's unit tests, relocated from `loader/src/verify.rs` when the
//! ABI split out: every one of them checks the verdict *through the
//! marshalled view*, and in-crate tests cannot do that any more — the
//! dev-only cycle means the capi this test links was built against the
//! external `pie_loader`, whose types are not the test build's `crate::`.
//! As an integration test both crates are external and the types unify.

use pie_loader::plan::*;
use pie_loader::types::*;
use pie_loader::verify::*;
use pie_loader_capi::view::verify_marshalled;

fn decl(name: &str) -> TensorDecl {
    TensorDecl {
        id: TensorId(0),
        name: name.to_string(),
        shape: vec![1],
        encoding: Encoding::Raw(pie_loader::types::DType::U8),
        alignment: 1,
        visibility: Visibility::Public,
    }
}

fn internal(decl: TensorDecl) -> TensorDecl {
    TensorDecl {
        visibility: Visibility::Internal,
        ..decl
    }
}

fn plan_with(tensors: Vec<TensorDecl>, instrs: Vec<StorageInstr>) -> LoadPlan {
    let schedule = (0..instrs.len() as u32).map(InstrId).collect();
    LoadPlan {
        compiler_version: compiler_version(),
        tensors,
        instrs,
        schedule,
        ..LoadPlan::empty(StorageTarget::default())
    }
}

fn finalize(index: u32, name: &str) -> StorageInstr {
    StorageInstr::Finalize {
        id: InstrId(index),
        tensor: BufferId(index),
        name: name.to_string(),
    }
}

#[test]
fn a_read_that_runs_past_the_end_of_its_file_is_a_violation() {
    // The plan's *sources* and its *reads* are different claims, and only
    // the second is what the executor is handed. A source declaring 8 legal
    // bytes says nothing about an instruction that reads 4096 from the same
    // file, and until `ReadView` carried the range nothing could.
    use pie_loader::plan::{CheckpointFileDecl, DestExtent, SourceExtent};
    use pie_loader::types::{CheckpointFormat, FileId};

    let mut plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
    plan.files.push(CheckpointFileDecl {
        id: FileId(0),
        path: "model.safetensors".to_string(),
        size_bytes: 8,
        format: CheckpointFormat::Safetensors,
    });
    plan.instrs.push(StorageInstr::ExtentWrite {
        id: InstrId(1),
        source: SourceExtent {
            file_id: FileId(0),
            tensor_id: TensorId(0),
            file_offset: 0,
            span_bytes: 4096,
            stride: pie_loader::extent::Extent::byte_run(4096),
            dtype: pie_loader::types::DType::U8,
        },
        dest: DestExtent {
            buffer: BufferId(0),
            offset: 0,
            stride: pie_loader::extent::Extent::byte_run(4096),
        },
    });
    plan.schedule.push(InstrId(1));

    let violations = verify_marshalled(&plan, None).unwrap_err();
    assert!(
        violations
            .iter()
            .any(|v| v.message.contains("reads bytes [0, 4096)")),
        "got: {violations:?}"
    );
}

#[test]
fn a_plan_that_finalizes_what_it_declares_verifies() {
    let plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
    let certificate = verify_marshalled(&plan, None).expect("should verify");
    assert_eq!(certificate.tensors, 1);
    assert_eq!(certificate.instructions, 1);
}

#[test]
fn a_declared_tensor_that_is_never_finalized_is_a_violation() {
    let plan = plan_with(vec![decl("w")], Vec::new());
    let violations = verify_marshalled(&plan, None).unwrap_err();
    assert_eq!(violations[0].tensor.as_deref(), Some("w"));
    assert!(violations[0].message.contains("never finalized"));
}

#[test]
fn an_internal_tensor_that_is_never_finalized_verifies() {
    let plan = plan_with(vec![internal(decl("factors"))], Vec::new());
    verify_marshalled(&plan, None).expect("an internal name is not a missing weight");
}

#[test]
fn finalizing_an_internal_tensor_is_a_violation() {
    let plan = plan_with(
        vec![internal(decl("factors"))],
        vec![finalize(0, "factors")],
    );
    let violations = verify_marshalled(&plan, None).unwrap_err();
    assert_eq!(violations[0].tensor.as_deref(), Some("factors"));
    assert!(violations[0].message.contains("is internal but finalized"));
}

#[test]
fn finalizing_a_name_nobody_declared_is_a_violation() {
    let plan = plan_with(Vec::new(), vec![finalize(0, "ghost")]);
    let violations = verify_marshalled(&plan, None).unwrap_err();
    assert!(
        violations
            .iter()
            .any(|v| v.message.contains("never declared"))
    );
}

#[test]
fn finalizing_the_same_tensor_twice_is_a_violation() {
    let plan = plan_with(vec![decl("w")], vec![finalize(0, "w"), finalize(1, "w")]);
    let violations = verify_marshalled(&plan, None).unwrap_err();
    assert!(violations.iter().any(|v| v.message.contains("2 times")));
}

#[test]
fn an_instruction_left_out_of_the_schedule_is_a_violation() {
    let mut plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
    plan.schedule.clear();
    let violations = verify_marshalled(&plan, None).unwrap_err();
    assert!(
        violations
            .iter()
            .any(|v| v.message.contains("never scheduled"))
    );
}

#[test]
fn scheduling_an_instruction_twice_is_a_violation() {
    let mut plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
    plan.schedule.push(InstrId(0));
    let violations = verify_marshalled(&plan, None).unwrap_err();
    assert!(
        violations
            .iter()
            .any(|v| v.message.contains("more than once"))
    );
}

fn contract(tensors: &[(&'static str, Vec<i64>, Encoding)]) -> ContractView<'static> {
    // Leaked so the view's borrows outlive the call; a test process is
    // exactly where that is the cheap answer.
    ContractView {
        tensors: tensors
            .iter()
            .map(|(name, shape, encoding)| {
                TensorDemand::authored(
                    name,
                    Some(Box::leak(shape.clone().into_boxed_slice())),
                    encoding,
                )
            })
            .collect(),
    }
}

#[test]
fn a_plan_that_delivers_the_contract_verifies() {
    let plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
    let demanded = contract(&[("w", vec![1], Encoding::Raw(pie_loader::types::DType::U8))]);
    verify_marshalled(&plan, Some(&demanded)).expect("should verify");
}

#[test]
fn a_contract_tensor_the_plan_omits_is_a_violation() {
    let plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
    let demanded = contract(&[
        ("w", vec![1], Encoding::Raw(pie_loader::types::DType::U8)),
        ("bias", vec![1], Encoding::Raw(pie_loader::types::DType::U8)),
    ]);
    let violations = verify_marshalled(&plan, Some(&demanded)).unwrap_err();
    assert_eq!(violations[0].tensor.as_deref(), Some("bias"));
    assert!(violations[0].message.contains("does not declare it"));
}

#[test]
fn a_plan_delivering_the_wrong_shape_is_a_violation() {
    let plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
    let demanded = contract(&[("w", vec![2, 3], Encoding::Raw(pie_loader::types::DType::U8))]);
    let violations = verify_marshalled(&plan, Some(&demanded)).unwrap_err();
    assert!(violations[0].message.contains("contract demands [2, 3]"));
}

#[test]
fn a_plan_delivering_the_wrong_encoding_is_a_violation() {
    let plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
    let demanded = contract(&[("w", vec![1], Encoding::Raw(pie_loader::types::DType::BF16))]);
    let violations = verify_marshalled(&plan, Some(&demanded)).unwrap_err();
    assert!(violations[0].message.contains("BF16"));
}

/// Publishing more than was demanded is not a failure: a contract may
/// republish a borrowed view — packed MoE experts under their original
/// names — that a given driver never binds.
#[test]
fn a_plan_delivering_more_than_the_contract_demands_verifies() {
    let plan = plan_with(
        vec![decl("w"), decl("view")],
        vec![finalize(0, "w"), finalize(1, "view")],
    );
    let demanded = contract(&[("w", vec![1], Encoding::Raw(pie_loader::types::DType::U8))]);
    verify_marshalled(&plan, Some(&demanded)).expect("should verify");
}

#[test]
fn a_stale_compiler_version_is_a_violation() {
    let mut plan = plan_with(vec![decl("w")], vec![finalize(0, "w")]);
    plan.compiler_version ^= 1;
    let violations = verify_marshalled(&plan, None).unwrap_err();
    assert!(violations[0].message.contains("compiler version"));
}
