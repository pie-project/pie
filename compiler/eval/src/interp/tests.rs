//! Tier-0 interpreter tests.

use super::numeric::*;
use super::*;
use pie_ir::container::{ChanDType, ChannelDecl, StageProgram, TraceContainer};
use pie_ir::op::Op;
use pie_ir::registry::ModelProfile;
use pie_ir::types::{Literal, Predicate, RngKind};
use pie_ir::validate::bind;

fn chan(shape: Shape, dtype: DType, host_role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded,
    }
}

/// Minimal ping-pong: counter channel c, out channel o.
/// epilogue: x = c.take(); y = x + 1; c.put(y); o.put(y)
fn counter_trace() -> TraceContainer {
    TraceContainer {
        names: vec![],
        channels: vec![
            chan(Shape::vector(1), DType::U32, HostRole::None, true), // 0 c
            chan(Shape::vector(1), DType::U32, HostRole::Reader, false), // 1 o
        ],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),            // 0
                Op::Const(Literal::U32(1)), // 1
                Op::Add(0, 1),              // 2
                Op::ChanPut { chan: 0, value: 2 },
                Op::ChanPut { chan: 1, value: 2 },
            ],
        }],
        externs: Vec::new(),
    }
}

#[test]
fn ping_pong_commits_and_back_pressures() {
    let b = bind(counter_trace(), ModelProfile::dummy()).unwrap();
    let mut inst = Instance::new(&b, &[(0, Value::U32(vec![10]))]).unwrap();
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(r.committed);
    assert_eq!(inst.host_take(&b, 1).unwrap(), Value::U32(vec![11]));
    // Second step commits (out drained).
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(r.committed);
    // Third step: out (cap 1) still full ⇒ leading-put NeedsEmpty fails ⇒
    // dummy-run, no commit, counter unchanged (back-pressure).
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(!r.committed);
    assert_eq!(r.missed.unwrap().0, 1);
    assert_eq!(inst.host_take(&b, 1).unwrap(), Value::U32(vec![12]));
    // Resubmission after the harvest commits and continues exactly.
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(r.committed);
    assert_eq!(inst.host_take(&b, 1).unwrap(), Value::U32(vec![13]));
}

#[test]
fn poison_makes_host_ops_error() {
    let b = bind(counter_trace(), ModelProfile::dummy()).unwrap();
    let mut inst = Instance::new(&b, &[(0, Value::U32(vec![0]))]).unwrap();
    inst.poison();
    assert_eq!(inst.host_take(&b, 1), Err(HostError::Poisoned));
    assert!(matches!(
        inst.step(&b, &PassInputs::default(), &mut NoKernels),
        Err(StepError::Poisoned)
    ));
}

#[test]
fn register_rule_put_then_take_reads_pending() {
    // epilogue: c.put(5); x = c.take(); o.put(x)  — x must be 5 (pending),
    // and the net effect on c is one put (queue: seed consumed? c had no
    // take of committed → committed cell remains, put lands: capacity 1
    // seeded ⇒ overflow fault. Use unseeded c.)
    let c = TraceContainer {
        names: vec![],
        channels: vec![
            chan(Shape::SCALAR, DType::U32, HostRole::None, false), // 0 c (empty)
            chan(Shape::SCALAR, DType::U32, HostRole::Reader, false), // 1 o
        ],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Const(Literal::U32(5)),        // 0
                Op::ChanPut { chan: 0, value: 0 }, // pending c = 5
                Op::ChanTake(0),                   // 1 = 5 (register rule)
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: alloc::vec::Vec::new(),
    };
    let b = bind(c, ModelProfile::dummy()).unwrap();
    let mut inst = Instance::new(&b, &[]).unwrap();
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(r.committed, "missed: {:?}", r.missed);
    assert_eq!(inst.host_take(&b, 1).unwrap(), Value::U32(vec![5]));
    // c: take popped nothing (was empty), put landed → now full with 5.
    assert_eq!(inst.len(0), 1);
}

#[test]
fn dummy_run_on_late_host_edge_then_recover() {
    // mask-style: host-fed m; epilogue takes m, adds to counter.
    let c = TraceContainer {
        names: vec![],
        channels: vec![
            chan(Shape::vector(1), DType::U32, HostRole::Writer, false), // 0 m
            chan(Shape::vector(1), DType::U32, HostRole::None, true),    // 1 acc
            chan(Shape::vector(1), DType::U32, HostRole::Reader, false), // 2 out
        ],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0), // 0 m
                Op::ChanTake(1), // 1 acc
                Op::Add(0, 1),   // 2
                Op::ChanPut { chan: 1, value: 2 },
                Op::ChanPut { chan: 2, value: 2 },
            ],
        }],
        externs: alloc::vec::Vec::new(),
    };
    let b = bind(c, ModelProfile::dummy()).unwrap();
    let mut inst = Instance::new(&b, &[(1, Value::U32(vec![100]))]).unwrap();
    // No mask yet: dummy-run (m's dummy = zeros), nothing commits.
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(!r.committed);
    assert_eq!(r.missed.unwrap().0, 0);
    assert_eq!(inst.host_take(&b, 2), Err(HostError::WouldBlock));
    assert_eq!(inst.len(1), 1, "acc untouched");
    // Host feeds m ⇒ resubmission commits with the real value.
    inst.host_put(&b, 0, Value::U32(vec![7])).unwrap();
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(r.committed);
    assert_eq!(inst.host_take(&b, 2).unwrap(), Value::U32(vec![107]));
}

#[test]
fn rng_keyed_is_pure_function_of_state() {
    let mk = || TraceContainer {
        names: vec![],
        channels: vec![
            chan(Shape::vector(2), DType::U32, HostRole::None, true), // rng
            chan(Shape::vector(4), DType::F32, HostRole::Reader, false), // out
        ],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::RngKeyed {
                    state: 0,
                    shape: Shape::vector(4),
                    kind: RngKind::Gumbel,
                },
                Op::ChanPut { chan: 0, value: 0 }, // ping-pong same state (replay!)
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: alloc::vec::Vec::new(),
    };
    let b = bind(mk(), ModelProfile::dummy()).unwrap();
    let seeds = [(0u32, Value::U32(vec![42, 7]))];
    let mut a = Instance::new(&b, &seeds).unwrap();
    let mut c = Instance::new(&b, &seeds).unwrap();
    a.step(&b, &PassInputs::default(), &mut NoKernels).unwrap();
    c.step(&b, &PassInputs::default(), &mut NoKernels).unwrap();
    assert_eq!(a.host_take(&b, 1).unwrap(), c.host_take(&b, 1).unwrap());
}

fn nucleus_expansion(escape_intermediate: bool) -> TraceContainer {
    let shape = Shape::matrix(2, 4);
    let mut channels = vec![
        chan(Shape::vector(2), DType::U32, HostRole::None, true),
        chan(Shape::vector(2), DType::F32, HostRole::None, true),
        chan(Shape::vector(2), DType::I32, HostRole::Reader, false),
    ];
    let mut ops = vec![
        Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape,
            dtype: DType::F32,
        },
        Op::ChanRead(0),
        Op::ChanRead(1),
        Op::ReduceMax(0),
        Op::Broadcast { value: 3, shape },
        Op::Sub(0, 4),
        Op::Exp(5),
        Op::ReduceSum(6),
        Op::Broadcast { value: 7, shape },
        Op::Div(6, 8),
        Op::PivotThreshold {
            input: 9,
            predicate: Predicate::CummassLe(2),
        },
        Op::Const(Literal::F32(f32::NEG_INFINITY)),
        Op::Select {
            cond: 10,
            a: 0,
            b: 11,
        },
        Op::RngKeyed {
            state: 1,
            shape,
            kind: RngKind::Gumbel,
        },
        Op::Add(12, 13),
        Op::ReduceArgmax(14),
        Op::ChanPut { chan: 2, value: 15 },
    ];
    if escape_intermediate {
        channels.push(chan(shape, DType::F32, HostRole::Reader, false));
        ops.push(Op::ChanPut { chan: 3, value: 6 });
    }
    TraceContainer {
        channels,
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        ..TraceContainer::default()
    }
}

#[test]
fn recognized_nucleus_reference_matches_generic_ssa_fallback() {
    use pie_plan::{LibraryOp, RegionKind, compile_stage};

    let mut profile = ModelProfile::dummy();
    profile.vocab = 4;
    let recognized = bind(nucleus_expansion(false), profile.clone()).unwrap();
    let generic = bind(nucleus_expansion(true), profile).unwrap();
    assert!(
        compile_stage(&recognized, Stage::Epilogue)
            .unwrap()
            .fused
            .regions
            .iter()
            .any(|region| region.kind == RegionKind::Library(LibraryOp::NucleusSample))
    );
    assert!(
        !compile_stage(&generic, Stage::Epilogue)
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
        let execute = |bound: &BoundTrace| {
            let mut instance = Instance::new(
                bound,
                &[
                    (0, Value::U32(vec![17, case as u32])),
                    (1, Value::F32(top_p.clone())),
                ],
            )
            .unwrap();
            assert!(
                instance
                    .step(
                        bound,
                        &PassInputs {
                            logits: Some(logits.clone()),
                            ..PassInputs::default()
                        },
                        &mut NoKernels,
                    )
                    .unwrap()
                    .committed
            );
            instance.host_take(bound, 2).unwrap()
        };
        assert_eq!(execute(&recognized), execute(&generic), "case {case}");
    }
}

#[test]
fn per_layer_tap_accumulates_via_register_rule() {
    // on_attn: stats.put(scatter_set(stats.take(), [layer], imp)) with
    // imp = layer as f32 vector — after one pass over 2 layers, stats =
    // [0., 1.] (each invocation writes its row; register semantics chain
    // the pending value between invocations).
    let c = TraceContainer {
        names: vec![],
        channels: vec![chan(Shape::vector(2), DType::F32, HostRole::None, true)], // stats
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::OnAttn,
            ops: vec![
                Op::ChanTake(0), // 0 stats
                Op::IntrinsicVal {
                    intr: IntrinsicId::Layer,
                    shape: Shape::SCALAR,
                    dtype: DType::U32,
                }, // 1
                Op::Cast {
                    value: 1,
                    dtype: DType::F32,
                }, // 2 imp (scalar)
                Op::ScatterSet {
                    base: 0,
                    idx: 1,
                    vals: 2,
                }, // 3
                Op::ChanPut { chan: 0, value: 3 },
            ],
        }],
        externs: alloc::vec::Vec::new(),
    };
    let b = bind(c, ModelProfile::dummy()).unwrap(); // num_layers = 2
    let mut inst = Instance::new(&b, &[(0, Value::F32(vec![-1.0, -1.0]))]).unwrap();
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(r.committed);
    // host can't read a device-private channel; inspect via a second
    // step's take: instead check internal state through len + dummy: use
    // the Overlay path — simplest: poison-free peek through queue.
    assert_eq!(inst.peek_front(0).unwrap(), Value::F32(vec![0.0, 1.0]));
}

#[test]
fn kernel_fault_poisons() {
    let c = TraceContainer {
        names: vec!["boom".into()],
        channels: vec![chan(Shape::vector(1), DType::F32, HostRole::None, true)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::KernelCall {
                    name: 0,
                    args: vec![],
                    shape: Shape::vector(1),
                    dtype: DType::F32,
                },
                Op::ChanTake(0),
                Op::Add(0, 1),
                Op::ChanPut { chan: 0, value: 2 },
            ],
        }],
        externs: alloc::vec::Vec::new(),
    };
    let mut profile = ModelProfile::dummy();
    profile.kernels.push(pie_ir::registry::KernelInfo {
        name: "boom".into(),
        sink_scope: None,
        replayable: true,
    });
    let b = bind(c, profile).unwrap();
    let mut inst = Instance::new(&b, &[(0, Value::F32(vec![0.0]))]).unwrap();
    let e = inst.step(&b, &PassInputs::default(), &mut NoKernels);
    assert!(matches!(e, Err(StepError::KernelFault { .. })));
    assert!(inst.is_poisoned());
}

#[test]
fn numeric_contract_argmax_and_topk() {
    // NaN never selected; ties → lower index.
    assert_eq!(argmax_row(&[f32::NAN, 1.0, 1.0]), 1);
    assert_eq!(argmax_row(&[f32::NAN, f32::NAN]), 0);
    assert_eq!(
        sort_desc_order(&[1.0, f32::NAN, 2.0, 1.0]),
        vec![2, 0, 3, 1]
    );
    let cancellation: [f32; 4] = [1.0e20, 1.0, -1.0e20, 1.0];
    assert_eq!(
        canonical_reduce(&cancellation, 0.0f32, |a, b| a + b).to_bits(),
        2.0f32.to_bits(),
        "width-32 tree order is part of the numeric contract"
    );
    assert_eq!(
        canonical_reduce(
            &[f32::NAN, -3.0, f32::NAN],
            f32::NEG_INFINITY,
            canonical_max
        ),
        -3.0
    );
    assert_eq!(
        canonical_reduce(&[-0.0, 0.0], f32::NEG_INFINITY, canonical_max).to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(
        canonical_reduce(&[-0.0, 0.0], f32::INFINITY, canonical_min).to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(argmax_ordered(&[16_777_216u32, 16_777_217]), 1);
    assert_eq!(argmax_ordered(&[-2i32, -1]), 1);
}

/// The two axes of [`extremum`], every combination, on the inputs IEEE
/// leaves to the caller. A single table makes mis-copying an `&&` or
/// identity visible in the test output instead of hiding it behind a
/// comparison that happened to agree for a different reason.
#[test]
fn the_extremum_rule_pins_nan_and_signed_zero() {
    const NAN: f32 = f32::NAN;
    // (left, right, canonical_max, canonical_min, element_max, element_min)
    let table: &[(f32, f32, f32, f32, f32, f32)] = &[
        // A NaN pair: reductions fold to their identity, elementwise keeps left.
        (NAN, NAN, f32::NEG_INFINITY, f32::INFINITY, NAN, NAN),
        // One NaN: dropped by both forms, either side.
        (NAN, -3.0, -3.0, -3.0, -3.0, -3.0),
        (-3.0, NAN, -3.0, -3.0, -3.0, -3.0),
        // Signed zeros: max is negative only when both are, min when either is.
        (-0.0, 0.0, 0.0, -0.0, 0.0, -0.0),
        (0.0, -0.0, 0.0, -0.0, 0.0, -0.0),
        (-0.0, -0.0, -0.0, -0.0, -0.0, -0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        // Ordinary values: no policy, just the order.
        (1.0, 2.0, 2.0, 1.0, 2.0, 1.0),
    ];
    for &(left, right, c_max, c_min, e_max, e_min) in table {
        for (name, got, want) in [
            ("canonical_max", canonical_max(left, right), c_max),
            ("canonical_min", canonical_min(left, right), c_min),
            ("element_max", element_max(left, right), e_max),
            ("element_min", element_min(left, right), e_min),
        ] {
            if want.is_nan() {
                assert!(got.is_nan(), "{name}({left}, {right}) = {got}, want NaN");
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{name}({left}, {right}) = {got} ({:#010x}), want {want} ({:#010x})",
                    got.to_bits(),
                    want.to_bits()
                );
            }
        }
    }
}

/// A `u32` prefix scan is exact where an `f32` one is not.
///
/// This is the reason `cumsum` stopped being F32-only. Ragged row offsets are
/// built by scanning per-row lengths, the offsets are `u32`, and the only way
/// to scan them under the old rule was `u32 -> f32 -> u32`. That round trip is
/// exact below 2^24 and rounds above it, so a long enough context produced
/// offsets that were wrong by one element and nothing anywhere said so.
///
/// The `f32` column is computed here rather than asserted from memory: the
/// claim is not "16777217 is unrepresentable", it is "the old lowering and the
/// new one disagree, and the new one is right".
#[test]
fn an_integer_scan_stays_exact_past_the_float_mantissa() {
    let lengths: Vec<u32> = vec![16_777_216, 1, 1, 1];

    let exact = scan_rows(&lengths, 1, 0u32, u32::wrapping_add);
    assert_eq!(exact, vec![16_777_216, 16_777_217, 16_777_218, 16_777_219]);

    let through_f32: Vec<u32> = scan_rows(
        &lengths.iter().map(|&n| n as f32).collect::<Vec<f32>>(),
        1,
        0.0f32,
        |a, b| a + b,
    )
    .into_iter()
    .map(|x| x as u32)
    .collect();
    assert_eq!(
        through_f32,
        vec![16_777_216; 4],
        "the f32 round trip this op was widened to remove: every +1 past the \
         mantissa rounds back to even, so three more tokens move no offset"
    );

    // Rows scan independently, and a product wraps rather than panicking.
    assert_eq!(
        scan_rows(&[1u32, 2, 3, 4, 5, 6], 2, 0u32, u32::wrapping_add),
        vec![1, 3, 6, 4, 9, 15]
    );
    assert_eq!(
        scan_rows(&[1u32 << 31, 2, 3], 1, 1u32, u32::wrapping_mul),
        vec![1 << 31, 0, 0]
    );
}
