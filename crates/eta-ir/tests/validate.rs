//! `bind` as its callers see it: an integration test through the one public
//! entry point only, so a pass reshuffle can't quietly break the contract.

use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{ModelProfile, Port, SinkScope, Stage};
use eta_ir::types::{Dtype, Literal, Shape};
use eta_ir::validate::{ValidateError, bind};

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

/// The canonical decode-loop shape: tok (loop), out (host-read), mask (host-fed,
/// bool), len (counter), rng (state) + greedy-gumbel epilogue.
fn section3() -> TraceContainer {
    let vocab = 32u32;
    let channels = vec![
        chan(Shape::vector(1), Dtype::I32, HostRole::None, true), // 0 tok
        chan(Shape::vector(1), Dtype::I32, HostRole::Reader, false), // 1 out
        chan(Shape::vector(vocab), Dtype::Bool, HostRole::Writer, false), // 2 mask
        chan(Shape::vector(1), Dtype::U32, HostRole::None, true), // 3 len
        chan(Shape::vector(2), Dtype::U32, HostRole::None, true), // 4 rng
    ];
    let mut ops: Vec<Op> = vec![
        Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(1, vocab),
            dtype: Dtype::F32,
        }, // 0
        Op::Reshape {
            value: 0,
            shape: Shape::vector(vocab),
        }, // 1
        Op::ChanTake(4), // 2 r = rng.take()
        Op::ChanTake(2), // 3 m = mask.take()
    ];
    let g = eta_ir::expand::gumbel(&mut ops, 2, Shape::vector(vocab)); // 4
    let masked = eta_ir::expand::mask_apply(&mut ops, 1, 3); // 5,6
    let sum = eta_ir::expand::next_id(&ops);
    ops.push(Op::Add(masked, g)); // sum
    ops.push(Op::ReduceArgmax(sum)); // t = sum+1
    let t = sum + 1;
    // rng.put(add(r, CTR1)): CTR1=[0,1] via iota(2), not a scalar const.
    ops.push(Op::Iota { len: 2 }); // t+1
    ops.push(Op::Cast {
        value: t + 1,
        dtype: Dtype::U32,
    }); // t+2 (identity; keeps ids readable)
    ops.push(Op::Add(2, t + 2)); // t+3
    ops.push(Op::ChanPut {
        chan: 4,
        value: t + 3,
    });
    // tok.put(t) — argmax over [vocab] gives scalar; reshape to [1].
    ops.push(Op::Reshape {
        value: t,
        shape: Shape::vector(1),
    }); // t+4
    ops.push(Op::ChanPut {
        chan: 0,
        value: t + 4,
    });
    // len.put(len.take() + 1)
    ops.push(Op::ChanTake(3)); // t+5
    ops.push(Op::Const(Literal::U32(1))); // t+6
    ops.push(Op::Add(t + 5, t + 6)); // t+7
    ops.push(Op::ChanPut {
        chan: 3,
        value: t + 7,
    });
    // out.put(t)
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
            u32_port(Port::Pages, Shape::vector(1), &[0]),
            u32_port(Port::PageIndptr, Shape::vector(2), &[0, 1]),
            PortBinding {
                port: Port::KvLen,
                source: PortSource::Channel(3),
            },
            u32_port(Port::WSlot, Shape::vector(1), &[0]),
            u32_port(Port::WOff, Shape::vector(1), &[0]),
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

#[test]
fn embed_tokens_requires_kv_len() {
    let mut c = section3();
    c.ports.retain(|binding| binding.port != Port::KvLen);
    assert!(matches!(
        bind(c, ModelProfile::dummy()),
        Err(ValidateError::EmbedTokensWithoutKvLen)
    ));
}

#[test]
fn spsc_second_producer_rejected() {
    let mut c = section3();
    // Host writes `mask` (chan 2); a stage put to it is a bind error.
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
        bind(c, ModelProfile::dummy()),
        Err(ValidateError::SecondProducer { chan: 2, .. })
    ));
}

#[test]
fn sink_precedence_t11() {
    let mut profile = ModelProfile::dummy();
    profile.kernels.push(eta_ir::registry::KernelInfo {
        name: "lora".to_string(),
        sink_scope: Some(SinkScope::PassWide),
        replayable: true,
    });
    // lora (pass-wide) in the prologue: OK.
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
    };
    assert!(bind(mk(Stage::Prologue), profile.clone()).is_ok());
    // lora at the epilogue: nothing after it consumes → T11 error.
    assert!(matches!(
        bind(mk(Stage::Epilogue), profile),
        Err(ValidateError::SinkMisplaced { .. })
    ));
    // attn_page_mask allowed at attn-proj (that layer)…
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
    };
    // needs a threshold operand: insert const before pivot — rebuild:
    let apm = {
        let mut c = apm;
        c.stages[0].ops = vec![
            Op::ChanRead(0),              // 0
            Op::Const(Literal::F32(0.5)), // 1
            Op::PivotThreshold {
                input: 0,
                predicate: eta_ir::types::Predicate::ProbGe(1),
            }, // 2
            Op::SinkCall {
                name: 0,
                args: vec![2],
            },
        ];
        c
    };
    assert!(bind(apm.clone(), ModelProfile::dummy()).is_ok());
    // …but not at on_attn (post-attention).
    let mut late = apm;
    late.stages[0].stage = Stage::OnAttn;
    assert!(matches!(
        bind(late, ModelProfile::dummy()),
        Err(ValidateError::SinkMisplaced { .. })
    ));
}

/// `lora` is a pass-wide sink whose name always type-checks; the backend's
/// ability to honour it is checked at bind: refused without `has_lora`, and
/// refused at any stage but the prologue.
#[test]
fn lora_honour_gate_and_placement() {
    let mk = |stage: Stage| TraceContainer {
        names: vec!["lora".to_string()],
        channels: vec![
            chan(
                Shape::new(&[2, 2, 4]).unwrap(),
                Dtype::F32,
                HostRole::None,
                true,
            ), // A
            chan(
                Shape::new(&[2, 4, 2]).unwrap(),
                Dtype::F32,
                HostRole::None,
                true,
            ), // B
            chan(Shape::vector(4), Dtype::U32, HostRole::None, true), // SITES
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
    };
    assert!(bind(mk(Stage::Prologue), ModelProfile::dummy()).is_ok());
    // Without the capability, refused at bind rather than a silent no-op adapter.
    let mut no_lora = ModelProfile::dummy();
    no_lora.has_lora = false;
    assert!(matches!(
        bind(mk(Stage::Prologue), no_lora),
        Err(ValidateError::KernelUnavailable { name, .. }) if name == "lora"
    ));
    // pass-wide sink => prologue only.
    assert!(matches!(
        bind(mk(Stage::OnAttnProj), ModelProfile::dummy()),
        Err(ValidateError::SinkMisplaced { .. })
    ));
    assert!(matches!(
        bind(mk(Stage::Epilogue), ModelProfile::dummy()),
        Err(ValidateError::SinkMisplaced { .. })
    ));
}

#[test]
fn name_table_must_be_strictly_sorted_and_unique() {
    let unsorted = TraceContainer {
        names: vec!["z".into(), "a".into()],
        ..TraceContainer::default()
    };
    assert!(matches!(
        bind(unsorted, ModelProfile::dummy()),
        Err(ValidateError::NamesUnsortedOrDuplicate)
    ));

    let duplicate = TraceContainer {
        names: vec!["a".into(), "a".into()],
        ..TraceContainer::default()
    };
    assert!(matches!(
        bind(duplicate, ModelProfile::dummy()),
        Err(ValidateError::NamesUnsortedOrDuplicate)
    ));
}
