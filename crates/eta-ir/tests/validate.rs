//! `bind` as its callers see it.
//!
//! An integration test on purpose: everything here goes in through the one
//! public entry point and reads the [`BoundTrace`] that comes out, which is
//! exactly what `eta-compiler` does. Nothing in this file can reach a pass
//! directly, so a refactor that reshuffles the passes cannot quietly break
//! the contract without a test noticing.
//!
//! The one check that *does* call a pass directly stays inline in
//! `src/validate.rs`, because it is asserting an internal property: that a
//! pass which normally runs second still answers correctly when run first.

use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{ModelProfile, Phase, Port, SinkScope, Stage};
use eta_ir::types::{Dtype, Literal, Shape};
use eta_ir::validate::{ChannelClass, Direction, ValidateError, bind};

/// The published score-row width both the DSL and the backend read.
const KVMAX: u32 = eta_ir::registry::ATTN_SCORE_KV_MAX;

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
    // rng.put(add(r, CTR1)) — CTR1 = [0,1] not expressible as a scalar
    // const; use iota(2) (=[0,1]) as the counter increment.
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
fn section3_binds_and_hashes_stably() {
    let c = section3();
    let h1 = c.hash();
    let bound = bind(c.clone(), ModelProfile::dummy()).expect("bind");
    assert_eq!(bound.hash, h1);
    assert_eq!(bind(c, ModelProfile::dummy()).unwrap().hash, h1);
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
fn embed_tokens_requires_complete_geometry() {
    let mut c = section3();
    c.ports.retain(|binding| binding.port != Port::Pages);
    assert!(matches!(
        bind(c, ModelProfile::dummy()),
        Err(ValidateError::EmbedTokensWithoutGeometry { port: Port::Pages })
    ));
}

#[test]
fn section3_readiness_table() {
    let c = section3();
    let b = bind(c, ModelProfile::dummy()).unwrap();
    // tok: first op = descriptor take (embed) → NeedsFull @ Descriptor.
    // out: first op = epilogue put → NeedsEmpty (back-pressure).
    // mask: epilogue take → NeedsFull. len: descriptor peek → NeedsFull.
    // rng: epilogue take → NeedsFull.
    let get = |ch: u32| b.readiness.iter().find(|e| e.chan == ch).copied().unwrap();
    assert_eq!(get(0).phase, Phase::Descriptor);
    assert_eq!(get(0).dir, Direction::NeedsFull);
    assert_eq!(get(1).dir, Direction::NeedsEmpty);
    assert_eq!(get(1).phase, Phase::Epilogue);
    assert_eq!(get(2).dir, Direction::NeedsFull);
    assert_eq!(get(3).phase, Phase::Descriptor);
    assert_eq!(get(4).dir, Direction::NeedsFull);
}

#[test]
fn section3_channel_classes() {
    let c = section3();
    let b = bind(c, ModelProfile::dummy()).unwrap();
    // tok: taken by descriptor (embed) + put by epilogue → not linear in
    // one stage → FullRing. out/mask host-visible → FullRing.
    // len: descriptor peek + epilogue take→put → extra consumer → FullRing.
    // rng: pure epilogue take→put ping-pong, epilogue is last stage and
    // the epilogue itself is the only fallible stage (mask) → InPlace.
    assert_eq!(b.classes[0], ChannelClass::FullRing);
    assert_eq!(b.classes[1], ChannelClass::FullRing);
    assert_eq!(b.classes[2], ChannelClass::FullRing);
    assert_eq!(b.classes[3], ChannelClass::FullRing);
    assert_eq!(b.classes[4], ChannelClass::InPlace);
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
fn spsc_second_consumer_rejected() {
    let mut c = section3();
    // Host reads `out` (chan 1); a stage read of it is a bind error.
    c.stages[0].ops.push(Op::ChanRead(1));
    assert!(matches!(
        bind(c, ModelProfile::dummy()),
        Err(ValidateError::SecondConsumer { chan: 1, .. })
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

/// §6.5: `lora` is a first-party pass-wide sink. The name always type-checks,
/// so the backend's ability to HONOUR it has to be checked at bind (mirroring
/// `attn_page_mask`'s gate): refused without `has_lora`, and — T11 — refused
/// at any stage but the prologue.
#[test]
fn lora_honour_gate_and_placement() {
    // A `[num_layers, R, d]`, B `[num_layers, d_out, R]`, SITES a small
    // trace-known site vector; all three peeked, per §6.5.
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
    // The dummy profile advertises the sink: a prologue call binds.
    assert!(bind(mk(Stage::Prologue), ModelProfile::dummy()).is_ok());
    // Without the capability the same program must be refused at bind — the
    // fire-time alternative is an adapter that silently never applies.
    let mut no_lora = ModelProfile::dummy();
    no_lora.has_lora = false;
    assert!(matches!(
        bind(mk(Stage::Prologue), no_lora),
        Err(ValidateError::KernelUnavailable { name, .. }) if name == "lora"
    ));
    // T11: pass-wide ⇒ prologue only. An attention-stage call comes after
    // part of the forward it configures, and an epilogue call after all of it.
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
fn t10_non_replayable_kernel_rejected() {
    let mut profile = ModelProfile::dummy();
    profile.kernels.push(eta_ir::registry::KernelInfo {
        name: "gpu_load".to_string(),
        sink_scope: None,
        replayable: false,
    });
    let c = TraceContainer {
        names: vec!["gpu_load".to_string()],
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
        externs: Vec::new(),
    };
    // The resolved name, not just the index, is what makes this diagnosable
    // by an operator who has a container but not its name table.
    assert!(matches!(
        bind(c, profile),
        Err(ValidateError::NotReplayable { name_index: 0, ref name })
            if name == "gpu_load"
    ));
}

#[test]
fn model_gated_intrinsic_rejected_when_absent() {
    let mut profile = ModelProfile::dummy();
    profile.has_mtp_logits = false;
    let c = TraceContainer {
        names: vec![],
        channels: vec![chan(Shape::vector(4), Dtype::I32, HostRole::Reader, false)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::MtpLogits,
                    shape: Shape::matrix(4, 32),
                    dtype: Dtype::F32,
                },
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
        externs: Vec::new(),
    };
    assert!(matches!(
        bind(c, profile),
        Err(ValidateError::IntrinsicUnavailable {
            intr: IntrinsicId::MtpLogits
        })
    ));
}

#[test]
fn intrinsic_stage_scope_enforced() {
    // logits at the prologue is out of scope.
    let c = TraceContainer {
        names: vec![],
        channels: vec![chan(Shape::vector(1), Dtype::I32, HostRole::Reader, false)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Prologue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::Logits,
                    shape: Shape::matrix(1, 32),
                    dtype: Dtype::F32,
                },
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
        externs: Vec::new(),
    };
    assert!(matches!(
        bind(c, ModelProfile::dummy()),
        Err(ValidateError::IntrinsicWrongStage {
            intr: IntrinsicId::Logits,
            stage: Stage::Prologue
        })
    ));
}

/// **THE MID-GRAPH TAP IS GONE, AND ITS STAGES ARE REFUSED BY NAME**
/// (`.wiki/alto/attn-score.md` §4). `attn_score` used to be `OnAttn`-only —
/// a per-layer tap that tore the fire's one captured graph open once per
/// layer, which is the third boundary palo §9 abolished. It is epilogue-only
/// now, and BOTH anatomical taps must refuse it: a program written against
/// the old registration has to fail at bind rather than silently read a
/// rectangle at a stage where the capture arm has not finished writing it.
#[test]
fn attn_score_rejected_at_the_per_layer_taps() {
    for stage in [Stage::OnAttnProj, Stage::OnAttn] {
        let c = TraceContainer {
            names: vec![],
            channels: vec![chan(Shape::vector(4), Dtype::I32, HostRole::Reader, false)],
            ports: vec![],
            stages: vec![StageProgram {
                stage,
                ops: vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::AttnScore,
                        shape: Shape::matrix(4, KVMAX),
                        dtype: Dtype::F32,
                    },
                    Op::ReduceArgmax(0),
                    Op::ChanPut { chan: 0, value: 1 },
                ],
            }],
            externs: Vec::new(),
        };
        let said = bind(c, ModelProfile::dummy());
        assert!(
            matches!(
                said,
                Err(ValidateError::IntrinsicWrongStage {
                    intr: IntrinsicId::AttnScore,
                    stage: at,
                }) if at == stage
            ),
            "{stage:?} admitted the score rectangle: {said:?}"
        );
    }
}

/// The epilogue is where the whole rectangle is legible: the capture arm has
/// written every exported (layer, head) plane by then, and the boundary that
/// reads them already exists.
#[test]
fn attn_score_accepted_at_the_epilogue() {
    let c = TraceContainer {
        names: vec![],
        channels: vec![chan(Shape::vector(4), Dtype::I32, HostRole::Reader, false)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::AttnScore,
                    shape: Shape::matrix(4, KVMAX),
                    dtype: Dtype::F32,
                },
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
        externs: Vec::new(),
    };
    assert!(bind(c, ModelProfile::dummy()).is_ok());
}

/// An engine without the capture path must make the program fail at BIND.
/// The buffer is otherwise simply never written, and a policy reading it
/// would silently rank on garbage.
#[test]
fn attn_score_rejected_when_engine_lacks_capture() {
    let mut profile = ModelProfile::dummy();
    profile.has_attn_score = false;
    let c = TraceContainer {
        names: vec![],
        channels: vec![chan(Shape::vector(4), Dtype::I32, HostRole::Reader, false)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::AttnScore,
                    shape: Shape::matrix(4, KVMAX),
                    dtype: Dtype::F32,
                },
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
        externs: Vec::new(),
    };
    assert!(matches!(
        bind(c, profile),
        Err(ValidateError::IntrinsicUnavailable {
            intr: IntrinsicId::AttnScore
        })
    ));
}

/// **THE WIDTH IS THE SLAB'S PITCH AND THE PROGRAM MAY NOT RESTATE IT.**
///
/// The rectangle is `[planes, ATTN_SCORE_KV_MAX]` F32. A rank-1 read is the
/// old head-folded row and a different tensor; a non-F32 read is a different
/// element; and a row a hair narrower or wider than the published ceiling is
/// the failure this intrinsic exists to make impossible — not a padded row
/// but a row read at the wrong stride, striping one plane's mass across the
/// next one's. All four must fail at bind.
#[test]
fn attn_score_type_rule_pins_the_rectangle() {
    for (shape, dtype, chan_shape) in [
        (Shape::vector(KVMAX), Dtype::F32, Shape::SCALAR),
        (Shape::matrix(4, KVMAX), Dtype::I32, Shape::vector(4)),
        (Shape::matrix(4, KVMAX - 1), Dtype::F32, Shape::vector(4)),
        (Shape::matrix(4, KVMAX + 1), Dtype::F32, Shape::vector(4)),
    ] {
        let c = TraceContainer {
            names: vec![],
            channels: vec![chan(chan_shape, Dtype::I32, HostRole::Reader, false)],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::AttnScore,
                        shape,
                        dtype,
                    },
                    Op::ReduceArgmax(0),
                    Op::ChanPut { chan: 0, value: 1 },
                ],
            }],
            externs: Vec::new(),
        };
        let said = bind(c, ModelProfile::dummy());
        assert!(
            matches!(
                said,
                Err(ValidateError::IntrinsicTypeRule {
                    intr: IntrinsicId::AttnScore,
                    ..
                })
            ),
            "{shape:?} {dtype:?} bound: {said:?}"
        );
    }
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
/// Two rules broken at once. Which complaint you hear is decided by the
/// call order in [`bind`] and by nothing else, so it is pinned here: a
/// reorder is a test failure with a diff, not a silent change of contract.
#[test]
fn the_pass_order_decides_which_complaint_wins() {
    let container = TraceContainer {
        names: vec!["b".to_string(), "a".to_string()],
        channels: vec![chan(Shape::vector(1), Dtype::I32, HostRole::None, true)],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Prologue,
            ops: vec![Op::ChanTake(9)],
        }],
        externs: vec![],
    };
    // Unsorted names (check_structure) and a bad channel index
    // (check_bodies). Structure runs first.
    assert!(matches!(
        bind(container, ModelProfile::dummy()).unwrap_err(),
        ValidateError::NamesUnsortedOrDuplicate
    ));
}
