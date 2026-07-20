use ptir_dsl::builder::Builder;
use ptir_dsl::prelude::*;
use ptir_dsl::ptir::compiler::{LibraryOp, RegionKind, compile_stage, encode_stage_plan};
use ptir_dsl::ptir::interp::{Instance, NoKernels, PassInputs, Value};
use ptir_dsl::ptir::op::Op;
use ptir_dsl::ptir::registry::ModelProfile;
use ptir_dsl::ptir::types::{Literal, Predicate, RngKind};
use ptir_dsl::ptir::validate::{BoundTrace, bind};
use ptir_dsl::{Channel, Traced};

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
        let logits = logits.take().tensor();
        let top_p = top_p.take().tensor();
        let rng = rng.take().tensor();
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
        encode_stage_plan(&helper_compiled),
        encode_stage_plan(&explicit_compiled)
    );
    let region = helper_compiled
        .fused
        .regions
        .iter()
        .find(|region| region.kind == RegionKind::Library(LibraryOp::NucleusSample))
        .expect("nucleus library region");
    assert_eq!(region.nodes, (3..=15).collect::<Vec<_>>());
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
