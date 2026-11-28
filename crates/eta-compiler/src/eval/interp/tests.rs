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

/// `gumbel_argmax` over f32 logits is the draw it fuses, bit for bit:
/// `reduce_argmax(select(keep, logits * scale, -inf) + rng_keyed(state))`.
#[test]
fn gumbel_argmax_is_the_draw_it_fuses() {
    use eta_ir::types::Shape;
    let n = 997usize;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 7919) % 1000) as f32 / 37.0 - 9.0)
        .collect();
    let keep: Vec<bool> = (0..n).map(|i| i % 5 != 3).collect();
    for seed in 0..40u32 {
        let vals = vec![
            Value::F32(x.clone()),
            Value::Bool(keep.clone()),
            Value::F32(vec![1.0 / 0.7]),
            Value::U32(vec![seed, seed * 3 + 1]),
        ];
        let ty = |id: u32| match id {
            0 => ValueType::vector(n as u32, Dtype::F32),
            1 => ValueType::vector(n as u32, Dtype::Bool),
            2 => ValueType::scalar(Dtype::F32),
            _ => ValueType::vector(2, Dtype::U32),
        };
        let fused = eval_op(
            &Op::GumbelArgmax {
                logits: 0,
                keep: 1,
                scale: 2,
                state: 3,
            },
            &vals,
            &ty,
            &PassInputs::default(),
            &[],
        )
        .unwrap();
        let Evaled::One(Value::I32(fused)) = fused else {
            panic!("an index")
        };
        let noise = eval_op(
            &Op::RngKeyed {
                state: 3,
                shape: Shape::vector(n as u32),
                kind: RngKind::Gumbel,
            },
            &vals,
            &ty,
            &PassInputs::default(),
            &[],
        )
        .unwrap();
        let Evaled::One(Value::F32(noise)) = noise else {
            panic!("noise")
        };
        let perturbed: Vec<f32> = (0..n)
            .map(|i| {
                let scaled = if keep[i] {
                    x[i] * (1.0 / 0.7)
                } else {
                    f32::NEG_INFINITY
                };
                scaled + noise[i]
            })
            .collect();
        assert_eq!(fused, vec![argmax_row(&perturbed)], "seed {seed}");
    }
}

/// Over a bf16 row the draw is unbiased between two tokens whose logits
/// round to one bf16 value: each perturbed lane is f32, so neither the
/// rounding nor the lowest index decides it.
#[test]
fn a_bf16_draw_splits_a_tie_evenly() {
    let row = [24.0f32, 24.0];
    let bits: Vec<u16> = row.iter().map(|&a| eta_ir::types::bf16_bits(a)).collect();
    let (mut second, trials) = (0u32, 4000u32);
    for seed in 0..trials {
        let vals = vec![
            Value::Bf16(bits.clone()),
            Value::Bool(vec![true]),
            Value::F32(vec![1.0]),
            Value::U32(vec![seed, 7]),
        ];
        let ty = |id: u32| match id {
            0 => ValueType::vector(2, Dtype::Bf16),
            1 => ValueType::scalar(Dtype::Bool),
            2 => ValueType::scalar(Dtype::F32),
            _ => ValueType::vector(2, Dtype::U32),
        };
        let Evaled::One(Value::I32(i)) = eval_op(
            &Op::GumbelArgmax {
                logits: 0,
                keep: 1,
                scale: 2,
                state: 3,
            },
            &vals,
            &ty,
            &PassInputs::default(),
            &[],
        )
        .unwrap() else {
            panic!("an index")
        };
        second += i[0] as u32;
    }
    let share = f64::from(second) / f64::from(trials);
    assert!(
        (share - 0.5).abs() < 0.03,
        "token 1 drawn {share:.3} of the time"
    );
}
