use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{EngineCaps, Graph, Port, Readout, ReadoutKind, Stage};
use eta_ir::types::{Dtype, Literal, Shape};
use eta_ir::validate::{BoundTrace, ValidateError, bind};

fn served(c: TraceContainer, caps: EngineCaps) -> Result<BoundTrace, ValidateError> {
    let logits = Readout {
        kind: ReadoutKind::Logits,
        width: 32,
    };
    let graph = Graph {
        num_layers: 2,
        mtp_logits: true,
        mtp_depth: 1,
        attn_score: true,
    };
    bind(c, vec![Some(logits)], graph, caps)
}

const EVERY: EngineCaps = EngineCaps {
    lora: true,
    attn_page_mask: true,
};

fn chan(shape: Shape, dtype: Dtype, host_role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded,
    }
}

fn u32_port(port: Port, shape: Shape, values: &[u32]) -> PortBinding {
    PortBinding {
        port,
        source: PortSource::Const {
            dtype: Dtype::U32,
            shape,
            data: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        },
    }
}

fn section3() -> TraceContainer {
    let vocab = 32u32;
    let channels = vec![
        chan(Shape::vector(1), Dtype::I32, HostRole::None, true),
        chan(Shape::vector(1), Dtype::I32, HostRole::Reader, false),
        chan(Shape::vector(vocab), Dtype::Bool, HostRole::Writer, false),
        chan(Shape::vector(1), Dtype::U32, HostRole::None, true),
        chan(Shape::vector(2), Dtype::U32, HostRole::None, true),
    ];
    let mut ops: Vec<Op> = vec![
        Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            lane: 0,
            shape: Shape::matrix(1, vocab),
            dtype: Dtype::F32,
        },
        Op::Reshape {
            value: 0,
            shape: Shape::vector(vocab),
        },
        Op::ChanTake(4),
        Op::ChanTake(2),
    ];
    let g = eta_ir::expand::gumbel(&mut ops, 2, Shape::vector(vocab));
    let masked = eta_ir::expand::mask_apply(&mut ops, 1, 3);
    let sum = eta_ir::expand::next_id(&ops);
    ops.push(Op::Add(masked, g));
    ops.push(Op::ReduceArgmax(sum));
    let t = sum + 1;
    ops.push(Op::Iota { len: 2 });
    ops.push(Op::Cast {
        value: t + 1,
        dtype: Dtype::U32,
    });
    ops.push(Op::Add(2, t + 2));
    ops.push(Op::ChanPut {
        chan: 4,
        value: t + 3,
    });
    ops.push(Op::Reshape {
        value: t,
        shape: Shape::vector(1),
    });
    ops.push(Op::ChanPut {
        chan: 0,
        value: t + 4,
    });
    ops.push(Op::ChanTake(3));
    ops.push(Op::Const(Literal::U32(1)));
    ops.push(Op::Add(t + 5, t + 6));
    ops.push(Op::ChanPut {
        chan: 3,
        value: t + 7,
    });
    ops.push(Op::ChanPut {
        chan: 1,
        value: t + 4,
    });

    TraceContainer {
        names: vec![],
        channels,
        ports: vec![
            PortBinding {
                port: Port::EmbedTokens,
                source: PortSource::Channel(0),
            },
            PortBinding {
                port: Port::EmbedIndptr,
                source: PortSource::Const {
                    dtype: Dtype::U32,
                    shape: Shape::vector(2),
                    data: [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect(),
                },
            },
            u32_port(Port::Positions, Shape::vector(1), &[0]),
            PortBinding {
                port: Port::KvLen,
                source: PortSource::Channel(3),
            },
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

#[test]
fn validate_every_case() {
    geometry_ports_are_optional();
    spsc_second_producer_rejected();
    sink_precedence_t11();
    lora_honour_gate_and_placement();
    name_table_must_be_strictly_sorted_and_unique();
}

fn geometry_ports_are_optional() {
    let mut c = section3();
    c.ports
        .retain(|binding| !matches!(binding.port, Port::KvLen | Port::Positions));
    assert!(served(c, EVERY).is_ok());
}

fn spsc_second_producer_rejected() {
    let mut c = section3();
    c.stages[0].ops.push(Op::Const(Literal::Bool(true)));
    let id = eta_ir::expand::next_id(&c.stages[0].ops) - 1;
    c.stages[0].ops.push(Op::Broadcast {
        value: id,
        shape: Shape::vector(32),
    });
    c.stages[0].ops.push(Op::ChanPut {
        chan: 2,
        value: id + 1,
    });
    assert!(matches!(
        served(c, EVERY),
        Err(ValidateError::SecondProducer { chan: 2, .. })
    ));
}

fn sink_precedence_t11() {
    let mk = |stage: Stage| TraceContainer {
        names: vec!["lora".to_string()],
        channels: vec![chan(Shape::vector(4), Dtype::F32, HostRole::None, true)],
        ports: vec![],
        stages: vec![StageProgram {
            stage,
            ops: vec![
                Op::ChanRead(0),
                Op::SinkCall {
                    name: 0,
                    args: vec![0],
                },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    assert!(served(mk(Stage::Prologue), EVERY).is_ok());
    assert!(matches!(
        served(mk(Stage::Epilogue), EVERY),
        Err(ValidateError::SinkMisplaced { .. })
    ));
    let apm = TraceContainer {
        names: vec!["attn_page_mask".to_string()],
        channels: vec![chan(Shape::vector(4), Dtype::F32, HostRole::None, true)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::OnAttnProj,
            ops: vec![
                Op::ChanRead(0),
                Op::PivotThreshold {
                    input: 0,
                    predicate: eta_ir::types::Predicate::ProbGe(1),
                },
                Op::SinkCall {
                    name: 0,
                    args: vec![2],
                },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    let apm = {
        let mut c = apm;
        c.stages[0].ops = vec![
            Op::ChanRead(0),
            Op::Const(Literal::F32(0.5)),
            Op::PivotThreshold {
                input: 0,
                predicate: eta_ir::types::Predicate::ProbGe(1),
            },
            Op::SinkCall {
                name: 0,
                args: vec![2],
            },
        ];
        c
    };
    assert!(served(apm.clone(), EVERY).is_ok());
    let mut late = apm;
    late.stages[0].stage = Stage::OnAttn;
    assert!(matches!(
        served(late, EVERY),
        Err(ValidateError::SinkMisplaced { .. })
    ));
}

fn lora_honour_gate_and_placement() {
    let mk = |stage: Stage| TraceContainer {
        names: vec!["lora".to_string()],
        channels: vec![
            chan(
                Shape::new(&[2, 2, 4]).unwrap(),
                Dtype::F32,
                HostRole::None,
                true,
            ),
            chan(
                Shape::new(&[2, 4, 2]).unwrap(),
                Dtype::F32,
                HostRole::None,
                true,
            ),
            chan(Shape::vector(4), Dtype::U32, HostRole::None, true),
        ],
        ports: vec![],
        stages: vec![StageProgram {
            stage,
            ops: vec![
                Op::ChanRead(0),
                Op::ChanRead(1),
                Op::ChanRead(2),
                Op::SinkCall {
                    name: 0,
                    args: vec![0, 1, 2],
                },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    assert!(served(mk(Stage::Prologue), EVERY).is_ok());
    let no_lora = EngineCaps {
        lora: false,
        ..EVERY
    };
    assert!(matches!(
        served(mk(Stage::Prologue), no_lora),
        Err(ValidateError::KernelUnavailable { name, .. }) if name == "lora"
    ));
    assert!(matches!(
        served(mk(Stage::OnAttnProj), EVERY),
        Err(ValidateError::SinkMisplaced { .. })
    ));
    assert!(matches!(
        served(mk(Stage::Epilogue), EVERY),
        Err(ValidateError::SinkMisplaced { .. })
    ));
}

fn name_table_must_be_strictly_sorted_and_unique() {
    let unsorted = TraceContainer {
        names: vec!["z".into(), "a".into()],
        ..TraceContainer::default()
    };
    assert!(matches!(
        served(unsorted, EVERY),
        Err(ValidateError::NamesUnsortedOrDuplicate)
    ));

    let duplicate = TraceContainer {
        names: vec!["a".into(), "a".into()],
        ..TraceContainer::default()
    };
    assert!(matches!(
        served(duplicate, EVERY),
        Err(ValidateError::NamesUnsortedOrDuplicate)
    ));
}
