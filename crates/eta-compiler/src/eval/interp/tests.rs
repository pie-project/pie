use super::numeric::*;
use super::*;
use eta_ir::container::{ChanDType, ChannelDecl, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, OutputDecl};
use eta_ir::types::{Literal, RngKind};
use eta_ir::validate::bind;

fn served(c: TraceContainer) -> BoundTrace {
    bind(
        c,
        vec![vec![OutputDecl {
            name: "logits".into(),
            axis: Axis::Row,
            width: 32,
            dtype: Dtype::F32,
            layers: 0,
            readout: true,
        }]],
        EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
    .unwrap()
}

fn chan(shape: Shape, dtype: Dtype, host_role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded,
    }
}

fn counter_trace() -> TraceContainer {
    TraceContainer {
        names: vec![],
        channels: vec![
            chan(Shape::vector(1), Dtype::U32, HostRole::None, true),
            chan(Shape::vector(1), Dtype::U32, HostRole::Reader, false),
        ],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::Const(Literal::U32(1)),
                Op::Add(0, 1),
                Op::ChanPut { chan: 0, value: 2 },
                Op::ChanPut { chan: 1, value: 2 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

#[test]
fn tests_every_case() {
    ping_pong_commits_and_back_pressures();
    poison_makes_host_ops_error();
    register_rule_put_then_take_reads_pending();
    dummy_run_on_late_host_edge_then_recover();
    rng_keyed_is_pure_function_of_state();
    numeric_contract_argmax_and_topk();
}

fn ping_pong_commits_and_back_pressures() {
    let b = served(counter_trace());
    let mut inst = Instance::new(&b, &[(0, Value::U32(vec![10]))]).unwrap();
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(r.committed);
    assert_eq!(inst.host_take(&b, 1).unwrap(), Value::U32(vec![11]));
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(r.committed);
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(!r.committed);
    assert_eq!(r.missed.unwrap().0, 1);
    assert_eq!(inst.host_take(&b, 1).unwrap(), Value::U32(vec![12]));
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(r.committed);
    assert_eq!(inst.host_take(&b, 1).unwrap(), Value::U32(vec![13]));
}

fn poison_makes_host_ops_error() {
    let b = served(counter_trace());
    let mut inst = Instance::new(&b, &[(0, Value::U32(vec![0]))]).unwrap();
    inst.poison();
    assert_eq!(inst.host_take(&b, 1), Err(HostError::Poisoned));
    assert!(matches!(
        inst.step(&b, &PassInputs::default(), &mut NoKernels),
        Err(StepError::Poisoned)
    ));
}

fn register_rule_put_then_take_reads_pending() {
    let c = TraceContainer {
        names: vec![],
        channels: vec![
            chan(Shape::SCALAR, Dtype::U32, HostRole::None, false),
            chan(Shape::SCALAR, Dtype::U32, HostRole::Reader, false),
        ],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Const(Literal::U32(5)),
                Op::ChanPut { chan: 0, value: 0 },
                Op::ChanTake(0),
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: alloc::vec::Vec::new(),
        inputs: Vec::new(),
    };
    let b = served(c);
    let mut inst = Instance::new(&b, &[]).unwrap();
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(r.committed, "missed: {:?}", r.missed);
    assert_eq!(inst.host_take(&b, 1).unwrap(), Value::U32(vec![5]));
    assert_eq!(inst.len(0), 1);
}

fn dummy_run_on_late_host_edge_then_recover() {
    let c = TraceContainer {
        names: vec![],
        channels: vec![
            chan(Shape::vector(1), Dtype::U32, HostRole::Writer, false),
            chan(Shape::vector(1), Dtype::U32, HostRole::None, true),
            chan(Shape::vector(1), Dtype::U32, HostRole::Reader, false),
        ],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::ChanTake(1),
                Op::Add(0, 1),
                Op::ChanPut { chan: 1, value: 2 },
                Op::ChanPut { chan: 2, value: 2 },
            ],
        }],
        externs: alloc::vec::Vec::new(),
        inputs: Vec::new(),
    };
    let b = served(c);
    let mut inst = Instance::new(&b, &[(1, Value::U32(vec![100]))]).unwrap();
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(!r.committed);
    assert_eq!(r.missed.unwrap().0, 0);
    assert_eq!(inst.host_take(&b, 2), Err(HostError::WouldBlock));
    assert_eq!(inst.len(1), 1, "acc untouched");
    inst.host_put(&b, 0, Value::U32(vec![7])).unwrap();
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(r.committed);
    assert_eq!(inst.host_take(&b, 2).unwrap(), Value::U32(vec![107]));
}

fn rng_keyed_is_pure_function_of_state() {
    let mk = || TraceContainer {
        names: vec![],
        channels: vec![
            chan(Shape::vector(2), Dtype::U32, HostRole::None, true),
            chan(Shape::vector(4), Dtype::F32, HostRole::Reader, false),
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
                Op::ChanPut { chan: 0, value: 0 },
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: alloc::vec::Vec::new(),
        inputs: Vec::new(),
    };
    let b = served(mk());
    let seeds = [(0u32, Value::U32(vec![42, 7]))];
    let mut a = Instance::new(&b, &seeds).unwrap();
    let mut c = Instance::new(&b, &seeds).unwrap();
    a.step(&b, &PassInputs::default(), &mut NoKernels).unwrap();
    c.step(&b, &PassInputs::default(), &mut NoKernels).unwrap();
    assert_eq!(a.host_take(&b, 1).unwrap(), c.host_take(&b, 1).unwrap());
}

fn numeric_contract_argmax_and_topk() {
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

/// The interpreter's rounding rule ([`eval_op::narrows`]) names `bf16`
/// exactly where eta-ir's types do, for every op over a bf16 value 0, an
/// f32 value 1, a u32 value 2, a bool value 3 (each `[4]`) and an f32
/// scalar 4.
#[test]
fn the_interpreter_narrows_where_the_ir_types_bf16() {
    use eta_ir::infer::{BodyCtx, body_types};
    use eta_ir::types::Predicate;
    let chans = [
        ValueType::vector(4, Dtype::Bf16),
        ValueType::vector(4, Dtype::F32),
        ValueType::vector(4, Dtype::U32),
        ValueType::vector(4, Dtype::Bool),
        ValueType::scalar(Dtype::F32),
    ];
    let ctx = BodyCtx {
        channel_types: &chans,
        n_names: 1,
    };
    let cases = [
        Op::Exp(0),
        Op::Neg(0),
        Op::Cast { value: 1, dtype: Dtype::Bf16 },
        Op::Cast { value: 0, dtype: Dtype::I32 },
        Op::Add(0, 1),
        Op::Add(1, 1),
        Op::Rem(1, 0),
        Op::Gt(0, 1),
        Op::Select { cond: 3, a: 1, b: 0 },
        Op::Select { cond: 3, a: 1, b: 1 },
        Op::ReduceSum(0),
        Op::ReduceMax(0),
        Op::ReduceArgmax(0),
        Op::CumSum(0),
        Op::CumProd(0),
        Op::SortDesc(0),
        Op::TopK { input: 0, k: 2 },
        Op::PivotThreshold { input: 0, predicate: Predicate::CummassLe(4) },
        Op::Gather { src: 0, idx: 2 },
        Op::ScatterAdd { base: 0, idx: 2, vals: 1 },
        Op::ScatterSet { base: 1, idx: 2, vals: 0 },
        Op::Broadcast { value: 0, shape: eta_ir::types::Shape::vector(4) },
        Op::Reshape { value: 0, shape: eta_ir::types::Shape::matrix(2, 2) },
    ];
    for op in cases {
        let mut ops: Vec<Op> = (0..5).map(Op::ChanTake).collect();
        ops.push(op.clone());
        let types = body_types(&ops, &ctx).unwrap_or_else(|e| panic!("{op:?}: {e}"));
        let typed = types[5].dtype == Dtype::Bf16;
        let bf16 = |id: u32| chans[id as usize].dtype == Dtype::Bf16;
        assert_eq!(eval_op::narrows(&op, bf16), typed, "{op:?}");
    }
}
