//! The DSL's op constructors record what they claim to record.
//!
//! Each test traces a program, then checks both halves of the claim: the
//! emitted op tags, and what the reference interpreter computes from them.
//! Checking only the tags would pass for a program that assembles the right
//! ops in the wrong order, and checking only the result would pass for one
//! that reaches the answer through a library op the backends do not have.

use pie_dsl::builder::Builder;
use pie_dsl::prelude::*;
use pie_dsl::ptir::op::Op;
use pie_dsl::ptir::registry::ModelProfile;
use pie_dsl::ptir::types::{Literal, Predicate, RngKind};
use pie_dsl::ptir::validate::{BoundTrace, bind};
use pie_dsl::{Channel, Traced};
use pie_eval::interp::{Instance, NoKernels, PassInputs, Value};
use pie_plan::{LibraryOp, NodeIndex, RegionKind, compile_stage, debug_stage_plan};

#[test]
fn row_membership_is_general_ssa_and_evaluates_per_row() {
    let rows = Channel::seeded([2, 3], dtype::u32);
    let keys = Channel::seeded([4], dtype::u32);
    let output = Channel::new([2, 4], dtype::bool);
    let mut builder = Builder::new(32_000, 16);
    builder.stage(Stage::Epilogue, || {
        output.put(row_membership(rows.take(), keys.take()));
    });
    let traced = builder.build().unwrap();
    let ops = &traced.container().stages[0].ops;
    let tags: Vec<u8> = ops.iter().map(Op::tag).collect();
    for tag in [0x39, 0x64, 0x13, 0x1f, 0x12, 0x10, 0x60, 0x18, 0x07, 0x31] {
        assert!(tags.contains(&tag), "row_membership missing tag {tag:#x}");
    }
    let bound = bind(traced.container().clone(), ModelProfile::dummy()).unwrap();
    let mut instance = Instance::new(
        &bound,
        &[
            (0, Value::U32(vec![0, 1, 2, 1, 2, 3])),
            (1, Value::U32(vec![0, 1, 2, 3])),
        ],
    )
    .unwrap();
    assert!(
        instance
            .step(&bound, &PassInputs::default(), &mut NoKernels)
            .unwrap()
            .committed
    );
    assert_eq!(
        instance.host_take(&bound, 2).unwrap(),
        Value::Bool(vec![true, true, true, false, false, true, true, true])
    );
}

fn nucleus_trace(explicit: bool, escape_intermediate: bool) -> Traced {
    let logits = Channel::seeded([2, 4], dtype::f32);
    let top_p = Channel::seeded([2], dtype::f32);
    let rng = Channel::seeded([2], dtype::u32);
    let output = Channel::new([2], dtype::i32);
    let escaped = escape_intermediate.then(|| Channel::new([2, 4], dtype::f32));
    let mut builder = Builder::new(32_000, 16);
    builder.stage(Stage::Epilogue, || {
        let logits = logits.take();
        let top_p = top_p.take();
        let rng = rng.take();
        let token = if explicit {
            let probabilities = softmax(&logits);
            if let Some(escaped) = &escaped {
                escaped.put(&probabilities);
            }
            let keep = pivot_threshold(&probabilities, cummass_le(&top_p));
            let masked = mask_apply(&logits, keep);
            gumbel_max(masked, rng)
        } else {
            nucleus_sample(logits, top_p, rng)
        };
        output.put(token);
    });
    builder.build().unwrap()
}

#[test]
fn nucleus_helper_is_byte_identical_to_general_ssa_and_compiles_as_library() {
    let helper = nucleus_trace(false, false);
    let explicit = nucleus_trace(true, false);
    assert_eq!(helper.encode(), explicit.encode());
    assert_eq!(helper.identity_hash(), explicit.identity_hash());
    let ops = &helper.container().stages[0].ops;
    assert!(ops.iter().any(|op| matches!(
        op,
        Op::PivotThreshold {
            predicate: Predicate::CummassLe(_),
            ..
        }
    )));
    assert!(ops.iter().any(|op| matches!(
        op,
        Op::Const(Literal::F32(value))
            if value.to_bits() == f32::NEG_INFINITY.to_bits()
    )));
    assert!(ops.iter().any(|op| matches!(
        op,
        Op::RngKeyed {
            kind: RngKind::Gumbel,
            ..
        }
    )));
    assert!(matches!(ops.get(15), Some(Op::ReduceArgmax(_))));

    let helper_bound = bind(helper.container().clone(), ModelProfile::dummy()).unwrap();
    let explicit_bound = bind(explicit.container().clone(), ModelProfile::dummy()).unwrap();
    let helper_compiled = compile_stage(&helper_bound, Stage::Epilogue).unwrap();
    let explicit_compiled = compile_stage(&explicit_bound, Stage::Epilogue).unwrap();
    assert_eq!(helper_compiled.signature, explicit_compiled.signature);
    assert_eq!(helper_compiled.fused, explicit_compiled.fused);
    assert_eq!(
        debug_stage_plan(&helper_compiled),
        debug_stage_plan(&explicit_compiled),
        "the helper and the explicit spelling must compile to the same plan"
    );
    let region = helper_compiled
        .fused
        .regions
        .iter()
        .find(|region| region.kind == RegionKind::Library(LibraryOp::NucleusSample))
        .expect("nucleus library region");
    assert_eq!(region.nodes, (3..=15).map(NodeIndex).collect::<Vec<_>>());
    assert_eq!(region.inputs, vec![0, 1, 2]);
    assert_eq!(region.outputs, vec![15]);
}

#[test]
fn recognized_nucleus_reference_executes_generic_ops_on_adversarial_values() {
    let helper = nucleus_trace(false, false);
    let explicit = nucleus_trace(true, true);
    let helper = bind(helper.container().clone(), ModelProfile::dummy()).unwrap();
    let explicit = bind(explicit.container().clone(), ModelProfile::dummy()).unwrap();
    assert!(
        compile_stage(&helper, Stage::Epilogue)
            .unwrap()
            .fused
            .regions
            .iter()
            .any(|region| region.kind == RegionKind::Library(LibraryOp::NucleusSample))
    );
    assert!(
        !compile_stage(&explicit, Stage::Epilogue)
            .unwrap()
            .fused
            .regions
            .iter()
            .any(|region| region.kind == RegionKind::Library(LibraryOp::NucleusSample))
    );

    let logits = Value::F32(vec![
        4.0,
        4.0,
        3.0,
        f32::NEG_INFINITY,
        f32::NAN,
        1.0,
        1.0,
        f32::NEG_INFINITY,
    ]);
    for (case, top_p) in [
        vec![0.5, 1.0],
        vec![0.0, f32::NAN],
        vec![f32::INFINITY, -1.0],
    ]
    .into_iter()
    .enumerate()
    {
        let seeds = [
            (0, logits.clone()),
            (1, Value::F32(top_p)),
            (2, Value::U32(vec![17, case as u32])),
        ];
        let execute = |bound: &BoundTrace| {
            let mut instance = Instance::new(bound, &seeds).unwrap();
            assert!(
                instance
                    .step(bound, &PassInputs::default(), &mut NoKernels)
                    .unwrap()
                    .committed
            );
            instance.host_take(bound, 3).unwrap()
        };
        let recognized_reference = execute(&helper);
        let generic_expansion = execute(&explicit);
        assert_eq!(recognized_reference, generic_expansion, "case {case}");
        if case == 1 {
            assert_eq!(recognized_reference, Value::I32(vec![0, 0]));
        }
    }
}

fn top_k_trace(beam_named: bool) -> Traced {
    let input = Channel::seeded([2, 8], dtype::f32).named(if beam_named {
        "beam candidates"
    } else {
        "generic input"
    });
    let values = Channel::new([2, 2], dtype::f32).named(if beam_named {
        "beam scores"
    } else {
        "top values"
    });
    let indices = Channel::new([2, 2], dtype::u32).named(if beam_named {
        "beam indices"
    } else {
        "top indices"
    });
    let mut builder = Builder::new(32_000, 16);
    builder.stage(Stage::Epilogue, || {
        let (top_values, top_indices) = top_k(input.take(), 2);
        values.put(top_values);
        indices.put(top_indices);
    });
    builder.build().unwrap()
}

#[test]
fn top_k_bytes_and_signature_have_no_beam_distinction() {
    let generic = top_k_trace(false);
    let beam_named = top_k_trace(true);
    assert_eq!(generic.encode(), beam_named.encode());
    assert_eq!(generic.identity_hash(), beam_named.identity_hash());

    let generic = bind(generic.container().clone(), ModelProfile::dummy()).unwrap();
    let beam_named = bind(beam_named.container().clone(), ModelProfile::dummy()).unwrap();
    assert_eq!(
        compile_stage(&generic, Stage::Epilogue).unwrap().signature,
        compile_stage(&beam_named, Stage::Epilogue)
            .unwrap()
            .signature
    );
}

/// `Abs`/`Sign`/`Recip`/`SortDesc` were reachable in the IR, the validator, the
/// interpreter and the encoder, but no DSL function emitted them, so no guest
/// could ask for them. Pin the behaviour now that they are exposed.
#[test]
fn newly_exposed_unary_ops_evaluate() {
    let input = Channel::seeded([4], dtype::f32);
    let a = Channel::new([4], dtype::f32);
    let s = Channel::new([4], dtype::f32);
    let r = Channel::new([4], dtype::f32);
    let mut builder = Builder::new(32_000, 16);
    builder.stage(Stage::Epilogue, || {
        let x = input.take();
        a.put(abs(&x));
        s.put(sign(&x));
        r.put(recip(&x));
    });
    let traced = builder.build().unwrap();
    let bound = bind(traced.container().clone(), ModelProfile::dummy()).unwrap();
    let mut instance =
        Instance::new(&bound, &[(0, Value::F32(vec![-2.0, -0.5, 0.5, 4.0]))]).unwrap();
    assert!(
        instance
            .step(&bound, &PassInputs::default(), &mut NoKernels)
            .unwrap()
            .committed
    );
    assert_eq!(
        instance.host_take(&bound, 1).unwrap(),
        Value::F32(vec![2.0, 0.5, 0.5, 4.0])
    );
    assert_eq!(
        instance.host_take(&bound, 2).unwrap(),
        Value::F32(vec![-1.0, -1.0, 1.0, 1.0])
    );
    assert_eq!(
        instance.host_take(&bound, 3).unwrap(),
        Value::F32(vec![-0.5, -2.0, 2.0, 0.25])
    );
}

#[test]
fn sort_desc_returns_values_and_original_indices() {
    let input = Channel::seeded([4], dtype::f32);
    let values = Channel::new([4], dtype::f32);
    let indices = Channel::new([4], dtype::u32);
    let mut builder = Builder::new(32_000, 16);
    builder.stage(Stage::Epilogue, || {
        let (v, i) = sort_desc(input.take());
        values.put(v);
        indices.put(i);
    });
    let traced = builder.build().unwrap();
    let bound = bind(traced.container().clone(), ModelProfile::dummy()).unwrap();
    let mut instance =
        Instance::new(&bound, &[(0, Value::F32(vec![1.0, 4.0, -2.0, 3.0]))]).unwrap();
    assert!(
        instance
            .step(&bound, &PassInputs::default(), &mut NoKernels)
            .unwrap()
            .committed
    );
    assert_eq!(
        instance.host_take(&bound, 1).unwrap(),
        Value::F32(vec![4.0, 3.0, 1.0, -2.0])
    );
    assert_eq!(
        instance.host_take(&bound, 2).unwrap(),
        Value::U32(vec![1, 3, 0, 2])
    );

    // SortDesc is a hard fusion barrier: it always lands in its own region.
    let plan = compile_stage(&bound, Stage::Epilogue).unwrap();
    assert!(
        plan.fused
            .regions
            .iter()
            .any(|r| matches!(r.kind, RegionKind::Library(LibraryOp::Sort))),
        "SortDesc should compile to its own library region"
    );
}
