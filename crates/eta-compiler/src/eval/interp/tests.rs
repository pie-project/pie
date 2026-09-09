use super::numeric::*;
use super::*;
use eta_ir::container::{ChanDType, ChannelDecl, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::ModelProfile;
use eta_ir::types::{Literal, RngKind};
use eta_ir::validate::bind;

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
    }
}

#[test]
fn tests_every_case() {
    ping_pong_commits_and_back_pressures();
    poison_makes_host_ops_error();
    register_rule_put_then_take_reads_pending();
    dummy_run_on_late_host_edge_then_recover();
    rng_keyed_is_pure_function_of_state();
    per_layer_tap_accumulates_via_register_rule();
    kernel_fault_poisons();
    numeric_contract_argmax_and_topk();
}

fn ping_pong_commits_and_back_pressures() {
    let b = bind(counter_trace(), ModelProfile::dummy()).unwrap();
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
    let b = bind(counter_trace(), ModelProfile::dummy()).unwrap();
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
    };
    let b = bind(c, ModelProfile::dummy()).unwrap();
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
    };
    let b = bind(c, ModelProfile::dummy()).unwrap();
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
    };
    let b = bind(mk(), ModelProfile::dummy()).unwrap();
    let seeds = [(0u32, Value::U32(vec![42, 7]))];
    let mut a = Instance::new(&b, &seeds).unwrap();
    let mut c = Instance::new(&b, &seeds).unwrap();
    a.step(&b, &PassInputs::default(), &mut NoKernels).unwrap();
    c.step(&b, &PassInputs::default(), &mut NoKernels).unwrap();
    assert_eq!(a.host_take(&b, 1).unwrap(), c.host_take(&b, 1).unwrap());
}

fn per_layer_tap_accumulates_via_register_rule() {
    let c = TraceContainer {
        names: vec![],
        channels: vec![chan(Shape::vector(2), Dtype::F32, HostRole::None, true)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::OnAttn,
            ops: vec![
                Op::ChanTake(0),
                Op::IntrinsicVal {
                    intr: IntrinsicId::Layer,
                    shape: Shape::SCALAR,
                    dtype: Dtype::U32,
                },
                Op::Cast {
                    value: 1,
                    dtype: Dtype::F32,
                },
                Op::ScatterSet {
                    base: 0,
                    idx: 1,
                    vals: 2,
                },
                Op::ChanPut { chan: 0, value: 3 },
            ],
        }],
        externs: alloc::vec::Vec::new(),
    };
    let b = bind(c, ModelProfile::dummy()).unwrap();
    let mut inst = Instance::new(&b, &[(0, Value::F32(vec![-1.0, -1.0]))]).unwrap();
    let r = inst
        .step(&b, &PassInputs::default(), &mut NoKernels)
        .unwrap();
    assert!(r.committed);
    assert_eq!(inst.peek_front(0).unwrap(), Value::F32(vec![0.0, 1.0]));
}

fn kernel_fault_poisons() {
    let c = TraceContainer {
        names: vec!["boom".into()],
        channels: vec![chan(Shape::vector(1), Dtype::F32, HostRole::None, true)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::KernelCall {
                    name: 0,
                    args: vec![],
                    shape: Shape::vector(1),
                    dtype: Dtype::F32,
                },
                Op::ChanTake(0),
                Op::Add(0, 1),
                Op::ChanPut { chan: 0, value: 2 },
            ],
        }],
        externs: alloc::vec::Vec::new(),
    };
    let mut profile = ModelProfile::dummy();
    profile.kernels.push(eta_ir::registry::KernelInfo {
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
