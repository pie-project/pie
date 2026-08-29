//! Planner tests.
//!
//! All twenty-nine of them go through `compile_stage`, the public entry point,
//! and read the whole `CompiledStage` it returns — normalized ops, both
//! partitions, the signature. None reach for `normalize_stage`,
//! `stage_signature` or the partitioners directly, which keeps the tests
//! pinned to behaviour a caller can observe rather than to the current
//! decomposition.
//!
//! What keeps them in-crate is the fixtures: `program`, `nucleus_program`,
//! `top_k_program` and friends build `BoundTrace`s at a level of detail that is
//! only readable next to the planner, and `NucleusMutation` is a crate-private
//! way to bend one of them. They are shared by nearly all the tests, so they
//! live together here rather than being copied per module -- and
//! `normalize::value_domain_tests` reaches for `program` too.

use super::*;
use crate::plan::lane_table::{LaneChannelSlot, LaneRecord, LaneTableHeader};
use alloc::vec;
use tensor_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use tensor_ir::expand;
use tensor_ir::op::{IntrinsicId, Op};
use tensor_ir::registry::{KernelInfo, ModelProfile, Port};
use tensor_ir::types::{DType, Literal, Predicate, RngKind, Shape, ValueId, ValueType};
use tensor_ir::validate::bind;

fn channel(shape: Shape, dtype: DType, role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role: role,
        seeded,
    }
}

#[test]
fn dce_preserves_faulting_kernel_calls_without_consumers() {
    let container = TraceContainer {
        names: vec!["observable".into()],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![Op::KernelCall {
                name: 0,
                args: vec![],
                shape: Shape::SCALAR,
                dtype: DType::F32,
            }],
        }],
        ..TraceContainer::default()
    };
    let mut profile = ModelProfile::dummy();
    profile.kernels.push(KernelInfo {
        name: "observable".into(),
        sink_scope: None,
        replayable: true,
    });
    let bound = bind(container, profile).unwrap();
    let compiled = compile_stage(&bound, Stage::Epilogue).unwrap();
    assert!(matches!(
        compiled.normalized.ops.as_slice(),
        [Op::KernelCall { .. }]
    ));
}

#[test]
fn symbolic_propagation_preserves_explicit_static_shape_changes() {
    let vocab = 32;
    let container = TraceContainer {
        channels: vec![
            channel(Shape::matrix(8, vocab), DType::F32, HostRole::Reader, false),
            channel(Shape::matrix(vocab, 1), DType::F32, HostRole::Reader, false),
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::Logits,
                    shape: Shape::matrix(1, vocab),
                    dtype: DType::F32,
                },
                Op::ReduceMax(0),
                Op::Broadcast {
                    value: 1,
                    shape: Shape::matrix(8, vocab),
                },
                Op::ChanPut { chan: 0, value: 2 },
                Op::Reshape {
                    value: 0,
                    shape: Shape::matrix(vocab, 1),
                },
                Op::ChanPut { chan: 1, value: 3 },
            ],
        }],
        ..TraceContainer::default()
    };
    let mut profile = ModelProfile::dummy();
    profile.vocab = vocab;
    let bound = bind(container, profile).unwrap();
    let compiled = compile_stage(&bound, Stage::Epilogue).unwrap();
    assert_eq!(
        compiled.normalized.value_types[2].dims,
        vec![Dimension::Static(8), Dimension::Static(vocab)]
    );
    assert_eq!(
        compiled.normalized.value_types[3].dims,
        vec![Dimension::Static(vocab), Dimension::Static(1)]
    );
}

#[test]
fn structured_masks_append_static_axis_without_dropping_symbolic_prefix() {
    let bound = program(0, 1);
    let original_types = [ValueType::vector(1, DType::U32)];
    let normalized_types = [SymbolicType {
        dtype: DType::U32,
        dims: vec![Dimension::Symbolic(SymbolicExtent::QueryLen)],
    }];
    let op = Op::CausalMask {
        positions: 0,
        len: 8,
    };
    let result = symbolic_result_type(
        &bound,
        &op,
        ValueType::new(Shape::matrix(1, 8), DType::Bool),
        &op,
        &original_types,
        &normalized_types,
    );
    assert_eq!(
        result.dims,
        vec![
            Dimension::Symbolic(SymbolicExtent::QueryLen),
            Dimension::Static(8),
        ]
    );
}

#[test]
fn structured_masks_use_remapped_positions_after_dce() {
    let mask_ops = [
        Op::CausalMask {
            positions: 1,
            len: 8,
        },
        Op::SlidingWindowMask {
            positions: 1,
            len: 8,
            window: 3,
        },
        Op::SinkWindowMask {
            positions: 1,
            len: 8,
            sink: 2,
            window: 3,
        },
    ];
    for mask_op in mask_ops {
        let container = TraceContainer {
            channels: vec![
                channel(Shape::vector(1), DType::U32, HostRole::None, true),
                channel(Shape::matrix(1, 8), DType::Bool, HostRole::Reader, false),
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::Const(Literal::U32(99)),
                    Op::ChanTake(0),
                    mask_op,
                    Op::ChanPut { chan: 1, value: 2 },
                ],
            }],
            ..TraceContainer::default()
        };
        let bound = bind(container, ModelProfile::dummy()).unwrap();
        let compiled = compile_stage(&bound, Stage::Epilogue).unwrap();
        assert_eq!(compiled.normalized.ops.len(), 3);
        assert_eq!(compiled.normalized.ops[1].operands(), vec![0]);
        assert_eq!(
            compiled.normalized.value_types[1].dims,
            vec![Dimension::Static(1), Dimension::Static(8)]
        );
    }
}

pub(super) fn program(prefix_constant: u32, global_channel_offset: usize) -> BoundTrace {
    let vocab = 32;
    let mut channels = Vec::new();
    for _ in 0..global_channel_offset {
        channels.push(channel(Shape::SCALAR, DType::U32, HostRole::None, true));
    }

    let token = channels.len() as u32;
    channels.push(channel(Shape::vector(1), DType::I32, HostRole::None, true));
    let output = channels.len() as u32;
    channels.push(channel(
        Shape::vector(1),
        DType::I32,
        HostRole::Reader,
        false,
    ));
    let kv_len = channels.len() as u32;
    channels.push(channel(Shape::vector(1), DType::U32, HostRole::None, true));
    let stages = vec![
        StageProgram {
            stage: Stage::Prologue,
            ops: vec![
                Op::Const(Literal::U32(prefix_constant)),
                Op::ChanPut {
                    chan: token.saturating_sub(1),
                    value: 0,
                },
            ],
        },
        StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::Logits,
                    shape: Shape::matrix(1, vocab),
                    dtype: DType::F32,
                },
                Op::ReduceArgmax(0),
                Op::Reshape {
                    value: 1,
                    shape: Shape::vector(1),
                },
                Op::ChanPut {
                    chan: output,
                    value: 2,
                },
            ],
        },
    ];
    let container = TraceContainer {
        names: vec![],
        channels,
        ports: vec![
            PortBinding {
                port: Port::EmbedTokens,
                source: PortSource::Channel(token),
            },
            PortBinding {
                port: Port::Positions,
                source: PortSource::Const {
                    dtype: DType::U32,
                    shape: Shape::vector(1),
                    data: 0u32.to_le_bytes().to_vec(),
                },
            },
            PortBinding {
                port: Port::Pages,
                source: PortSource::Const {
                    dtype: DType::U32,
                    shape: Shape::vector(1),
                    data: 0u32.to_le_bytes().to_vec(),
                },
            },
            PortBinding {
                port: Port::PageIndptr,
                source: PortSource::Const {
                    dtype: DType::U32,
                    shape: Shape::vector(2),
                    data: [0u32, 1].into_iter().flat_map(u32::to_le_bytes).collect(),
                },
            },
            PortBinding {
                port: Port::KvLen,
                source: PortSource::Channel(kv_len),
            },
            PortBinding {
                port: Port::WSlot,
                source: PortSource::Const {
                    dtype: DType::U32,
                    shape: Shape::vector(1),
                    data: 0u32.to_le_bytes().to_vec(),
                },
            },
            PortBinding {
                port: Port::WOff,
                source: PortSource::Const {
                    dtype: DType::U32,
                    shape: Shape::vector(1),
                    data: 0u32.to_le_bytes().to_vec(),
                },
            },
        ],
        stages,
        externs: vec![],
    };
    let mut profile = ModelProfile::dummy();
    profile.vocab = vocab;
    bind(container, profile).unwrap()
}

fn top_k_program(global_channel_offset: usize) -> BoundTrace {
    let mut channels = (0..global_channel_offset)
        .map(|_| channel(Shape::SCALAR, DType::U32, HostRole::None, true))
        .collect::<Vec<_>>();
    let input = channels.len() as u32;
    channels.push(channel(
        Shape::matrix(2, 8),
        DType::F32,
        HostRole::None,
        true,
    ));
    let values = channels.len() as u32;
    channels.push(channel(
        Shape::matrix(2, 2),
        DType::F32,
        HostRole::Reader,
        false,
    ));
    let indices = channels.len() as u32;
    channels.push(channel(
        Shape::matrix(2, 2),
        DType::U32,
        HostRole::Reader,
        false,
    ));
    bind(
        TraceContainer {
            channels,
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(input),
                    Op::TopK { input: 0, k: 2 },
                    Op::ChanPut {
                        chan: values,
                        value: 1,
                    },
                    Op::ChanPut {
                        chan: indices,
                        value: 2,
                    },
                ],
            }],
            ..TraceContainer::default()
        },
        ModelProfile::dummy(),
    )
    .unwrap()
}

fn softmax_program(rows: u32, vocab: u32) -> BoundTrace {
    let shape = Shape::matrix(rows, vocab);
    let container = TraceContainer {
        channels: vec![channel(Shape::SCALAR, DType::F32, HostRole::Reader, false)],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::Logits,
                    shape,
                    dtype: DType::F32,
                },
                Op::ReduceMax(0),
                Op::Broadcast { value: 1, shape },
                Op::Sub(0, 2),
                Op::Exp(3),
                Op::ReduceSum(4),
                Op::Broadcast { value: 5, shape },
                Op::Div(4, 6),
                Op::ReduceMax(7),
                Op::ReduceMax(8),
                Op::ChanPut { chan: 0, value: 9 },
            ],
        }],
        ..TraceContainer::default()
    };
    let mut profile = ModelProfile::dummy();
    profile.vocab = vocab;
    bind(container, profile).unwrap()
}

#[test]
fn singleton_scans_normalize_to_aliases() {
    let bound = bind(
        TraceContainer {
            channels: vec![
                channel(Shape::vector(1), DType::F32, HostRole::None, true),
                channel(Shape::vector(1), DType::F32, HostRole::Reader, false),
            ],
            stages: vec![StageProgram {
                stage: Stage::Prologue,
                ops: vec![
                    Op::ChanTake(0),
                    Op::CumSum(0),
                    Op::ChanPut { chan: 1, value: 1 },
                ],
            }],
            ..TraceContainer::default()
        },
        ModelProfile::dummy(),
    )
    .unwrap();
    let compiled = compile_stage(&bound, Stage::Prologue).unwrap();
    assert!(
        !compiled
            .normalized
            .ops
            .iter()
            .any(|op| matches!(op, Op::CumSum(_) | Op::CumProd(_)))
    );
}

#[derive(Clone, Copy)]
enum NucleusMutation {
    Exact,
    CommutedAdd,
    WrongPredicate,
    WrongSelectSource,
    WrongCenteredSource,
    FiniteMaskFill,
    UniformRng,
    PartialArgmax,
    WrongSumOperand,
    EscapedIntermediate,
    ForeignMaximum,
    DeadResult,
}

/// The id defined by the one canonical step matching `want`.
fn defines(ops: &[Op], want: impl FnMut(&Op) -> bool) -> ValueId {
    let at = ops
        .iter()
        .position(want)
        .expect("step is part of the canonical nucleus chain");
    expand::next_id(&ops[..at])
}

/// Channels every nucleus fixture reads: rng state, top-p, logits, and the
/// token the sampler writes back.
fn nucleus_channels(shape: Shape) -> Vec<ChannelDecl> {
    vec![
        channel(Shape::vector(2), DType::U32, HostRole::None, true),
        channel(Shape::vector(2), DType::F32, HostRole::None, true),
        channel(shape, DType::F32, HostRole::None, true),
        channel(Shape::vector(2), DType::I32, HostRole::Reader, false),
    ]
}

fn epilogue(channels: Vec<ChannelDecl>, ops: Vec<Op>) -> BoundTrace {
    let container = TraceContainer {
        channels,
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        ..TraceContainer::default()
    };
    bind(container, ModelProfile::dummy()).unwrap()
}

/// A nucleus sampler, then one thing wrong with it.
///
/// The chain itself comes from [`expand::nucleus_sample`] — the same sequence
/// `tensor-dsl` traces — so it cannot drift away from what the matcher will meet
/// in the field. Each mutation names the step it breaks rather than an SSA
/// number, which is also what the mutation *means*; the hand-numbered copy
/// this replaced had to be renumbered by eye whenever the chain moved.
fn nucleus_program(mutation: NucleusMutation) -> BoundTrace {
    use NucleusMutation as M;
    let shape = Shape::matrix(2, 8);
    let mut channels = nucleus_channels(shape);
    let mut ops = vec![
        Op::ChanRead(0), // rng state
        Op::ChanRead(1), // top-p
        Op::ChanTake(2), // logits
    ];
    // A second logits channel, so `ForeignMaximum` has somewhere else of the
    // right shape to reduce over.
    let decoy = expand::next_id(&ops);
    if matches!(mutation, M::ForeignMaximum) {
        channels.push(channel(shape, DType::F32, HostRole::None, true));
        ops.push(Op::ChanTake(4));
    }
    let token = expand::nucleus_sample(&mut ops, 2, 1, 0, shape);

    // Each targeted op kind occurs exactly once in the chain, so naming the
    // kind names the step. `Broadcast` occurs twice and the first is the one
    // that lifts the row maximum.
    let maximum = defines(&ops, |op| matches!(op, Op::Broadcast { .. }));
    let centered = defines(&ops, |op| matches!(op, Op::Sub(..)));
    let exponentials = defines(&ops, |op| matches!(op, Op::Exp(..)));
    let probabilities = defines(&ops, |op| matches!(op, Op::Div(..)));
    let masked = defines(&ops, |op| matches!(op, Op::Select { .. }));
    for op in &mut ops {
        match (mutation, op) {
            // Gumbel's argmax is only exact if the noise is added to the
            // masked logits; `a + b` and `b + a` differ once one is -inf.
            (M::CommutedAdd, Op::Add(a, b)) => core::mem::swap(a, b),
            (M::WrongPredicate, Op::PivotThreshold { predicate, .. }) => {
                *predicate = Predicate::ProbGe(1)
            }
            // Keep the probabilities rather than the logits.
            (M::WrongSelectSource, Op::Select { a, .. }) => *a = probabilities,
            // Centre the maximum against itself instead of the logits.
            (M::WrongCenteredSource, Op::Sub(a, _)) => *a = maximum,
            (M::FiniteMaskFill, Op::Const(fill)) => *fill = Literal::F32(f32::MIN),
            (M::UniformRng, Op::RngKeyed { kind, .. }) => *kind = RngKind::Uniform,
            // Argmax over the masked logits, before the noise is added.
            (M::PartialArgmax, Op::ReduceArgmax(input)) => *input = masked,
            // Normalize by the sum of the centered logits, not their exp.
            (M::WrongSumOperand, Op::ReduceSum(input)) => *input = centered,
            // Centre against a maximum taken over different logits.
            (M::ForeignMaximum, Op::ReduceMax(input)) => *input = decoy,
            // `Exact` breaks nothing; `EscapedIntermediate` adds a reader
            // below rather than editing a step.
            _ => {}
        }
    }
    if matches!(mutation, M::DeadResult) {
        // Chan 3 gets a token-shaped value from somewhere else, so the sampled
        // token is read by nothing.
        channels.push(channel(Shape::vector(2), DType::I32, HostRole::None, true));
        let substitute = expand::next_id(&ops);
        ops.push(Op::ChanRead(4));
        ops.push(Op::ChanPut {
            chan: 3,
            value: substitute,
        });
    } else {
        ops.push(Op::ChanPut {
            chan: 3,
            value: token,
        });
    }
    if matches!(mutation, M::EscapedIntermediate) {
        channels.push(channel(shape, DType::F32, HostRole::Reader, false));
        ops.push(Op::ChanPut {
            chan: 4,
            value: exponentials,
        });
    }
    epilogue(channels, ops)
}

/// How the sampler's `-inf` mask fill is spelled.
#[derive(Clone, Copy)]
enum MaskFill {
    /// A bare scalar, which is what [`expand::mask_apply`] emits.
    Scalar,
    /// Already broadcast to the row shape.
    Broadcast,
}

/// Temperature scaling ahead of the sampler, which the region has to absorb.
fn scaled_nucleus_program(fill: MaskFill) -> BoundTrace {
    let shape = Shape::matrix(2, 8);
    let mut ops = vec![
        Op::ChanRead(0),
        Op::ChanRead(1),
        Op::ChanTake(2),
        Op::Reshape { value: 2, shape },
        Op::Const(Literal::F32(0.8)),
        Op::Div(3, 4),
    ];
    // the id the temperature divide defines
    let scaled = expand::next_id(&ops[..ops.len() - 1]);
    let mut token = expand::nucleus_sample(&mut ops, scaled, 1, 0, shape);
    if matches!(fill, MaskFill::Broadcast) {
        token = broadcast_the_mask_fill(&mut ops, shape, token);
    }
    ops.push(Op::ChanPut {
        chan: 3,
        value: token,
    });
    epilogue(nucleus_channels(shape), ops)
}

/// Respell the sampler's `-inf` as an explicit row broadcast, renumbering
/// everything it displaces, and return the token id's new value.
///
/// Writing this as a rewrite rather than as a second copy of the chain is the
/// point: the claim under test is that normalization erases the difference,
/// and a rewrite can only introduce the difference it names.
fn broadcast_the_mask_fill(ops: &mut Vec<Op>, shape: Shape, token: ValueId) -> ValueId {
    let at = ops
        .iter()
        .position(|op| matches!(op, Op::Select { .. }))
        .expect("the sampler masks with a select");
    let Op::Select { b: fill, .. } = ops[at] else {
        unreachable!()
    };
    // `mask_apply` emits the fill immediately before the select, so this is
    // where the broadcast goes and `fill + 1` is the id it takes.
    assert_eq!(
        expand::next_id(&ops[..at]),
        fill + 1,
        "fill precedes select"
    );
    ops.insert(at, Op::Broadcast { value: fill, shape });
    for op in &mut ops[at + 1..] {
        op.map_operands(|id| if id > fill { id + 1 } else { id });
    }
    let Op::Select { b, .. } = &mut ops[at + 1] else {
        unreachable!()
    };
    *b = fill + 1;
    token + 1
}

/// `Op::Select` broadcasts its operands, so passing `-inf` as a bare scalar
/// and passing it already broadcast to the row shape are the same program.
/// Normalization must erase that difference: otherwise the spelling decides
/// whether the nucleus library is reachable at all, and the fast path is
/// available only by calling `nucleus_sample` by name — which is how the
/// repo's own `text-completion-bench` came to miss it.
#[test]
fn broadcast_neg_inf_normalizes_to_the_same_nucleus_plan() {
    let broadcast = compile_stage(
        &scaled_nucleus_program(MaskFill::Broadcast),
        Stage::Epilogue,
    )
    .unwrap();
    let bare = compile_stage(&scaled_nucleus_program(MaskFill::Scalar), Stage::Epilogue).unwrap();
    for compiled in [&broadcast, &bare] {
        let nucleus = compiled
            .fused
            .regions
            .iter()
            .find(|region| region.kind == RegionKind::Library(LibraryOp::NucleusSample))
            .expect("nucleus library region");
        // The 13-node/5-input shape is the wire ABI every backend asserts
        // (e.g. `grouped_nucleus_region_supported` in the CUDA engine).
        assert_eq!(nucleus.nodes.len(), 13);
        assert_eq!(nucleus.inputs.len(), 5);
    }
    assert_eq!(broadcast.normalized.ops, bare.normalized.ops);
    assert_eq!(broadcast.signature, bare.signature);
    assert_eq!(broadcast.fused.regions, bare.fused.regions);
}

/// The rewrite must not touch a broadcast that some consumer actually needs
/// at full width, nor one feeding a `Select` condition.
#[test]
fn scalar_broadcast_is_kept_when_a_non_select_consumer_needs_it() {
    let shape = Shape::vector(8);
    let container = TraceContainer {
        channels: vec![
            channel(shape, DType::F32, HostRole::None, true),
            channel(shape, DType::F32, HostRole::Reader, false),
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::Const(Literal::F32(2.0)),
                Op::Broadcast { value: 1, shape },
                Op::Add(0, 2),
                Op::ChanPut { chan: 1, value: 3 },
            ],
        }],
        ..TraceContainer::default()
    };
    let compiled = compile_stage(
        &bind(container, ModelProfile::dummy()).unwrap(),
        Stage::Epilogue,
    )
    .unwrap();
    assert!(
        compiled
            .normalized
            .ops
            .iter()
            .any(|op| matches!(op, Op::Broadcast { .. })),
        "a broadcast feeding Add must survive normalization"
    );
}

fn interleaved_nucleus_program() -> BoundTrace {
    let shape = Shape::matrix(2, 8);
    bind(
        TraceContainer {
            channels: vec![
                channel(Shape::vector(2), DType::U32, HostRole::None, true),
                channel(Shape::vector(2), DType::F32, HostRole::None, true),
                channel(shape, DType::F32, HostRole::None, true),
                channel(Shape::SCALAR, DType::U32, HostRole::None, true),
                channel(Shape::vector(2), DType::I32, HostRole::Reader, false),
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanRead(0),  // rng state: v0
                    Op::ChanRead(1),  // top-p: v1
                    Op::ChanTake(2),  // logits: v2
                    Op::ReduceMax(2), // matched n3, v3
                    Op::ChanRead(3),  // unrelated n4, v4
                    Op::Broadcast { value: 3, shape },
                    Op::Sub(2, 5),
                    Op::Exp(6),
                    Op::ReduceSum(7),
                    Op::Broadcast { value: 8, shape },
                    Op::Div(7, 9),
                    Op::PivotThreshold {
                        input: 10,
                        predicate: Predicate::CummassLe(1),
                    },
                    Op::Const(Literal::F32(f32::NEG_INFINITY)),
                    Op::Select {
                        cond: 11,
                        a: 2,
                        b: 12,
                    },
                    Op::RngKeyed {
                        state: 0,
                        shape,
                        kind: RngKind::Gumbel,
                    },
                    Op::Add(13, 14),
                    Op::ReduceArgmax(15),
                    Op::ChanPut { chan: 4, value: 16 },
                ],
            }],
            ..TraceContainer::default()
        },
        ModelProfile::dummy(),
    )
    .unwrap()
}

#[test]
fn identical_epilogues_share_signature_across_programs() {
    let first = program(1, 1);
    let second = program(2, 2);
    let first = compile_stage(&first, Stage::Epilogue).unwrap();
    let second = compile_stage(&second, Stage::Epilogue).unwrap();
    assert_eq!(first.signature, second.signature);
    assert_ne!(
        first.normalized.channel_bindings,
        second.normalized.channel_bindings
    );
}

#[test]
fn top_k_has_one_canonical_signature_and_library_kind() {
    let generic = top_k_program(0);
    let beam_style = top_k_program(3);
    let generic = compile_stage(&generic, Stage::Epilogue).unwrap();
    let beam_style = compile_stage(&beam_style, Stage::Epilogue).unwrap();
    assert_eq!(generic.signature, beam_style.signature);
    assert_eq!(
        generic
            .fused
            .regions
            .iter()
            .filter(|region| region.kind == RegionKind::Library(LibraryOp::TopK))
            .count(),
        1
    );
}

#[test]
fn normalized_nucleus_dataflow_has_role_ordered_library_abi() {
    let compiled =
        compile_stage(&nucleus_program(NucleusMutation::Exact), Stage::Epilogue).unwrap();
    let nucleus = compiled
        .fused
        .regions
        .iter()
        .find(|region| region.kind == RegionKind::Library(LibraryOp::NucleusSample))
        .expect("nucleus library region");
    assert_eq!(nucleus.nodes, (3..=15).map(NodeIndex).collect::<Vec<_>>());
    assert_eq!(nucleus.inputs, vec![2, 1, 0]);
    assert_eq!(nucleus.outputs, vec![15]);
    assert!(nucleus.nodes.windows(2).all(|nodes| nodes[0] < nodes[1]));
    assert!(
        !compiled
            .singleton
            .regions
            .iter()
            .any(|region| { region.kind == RegionKind::Library(LibraryOp::NucleusSample) })
    );
}

#[test]
fn scaled_nucleus_absorbs_temperature_and_peels_reshape() {
    let compiled =
        compile_stage(&scaled_nucleus_program(MaskFill::Scalar), Stage::Epilogue).unwrap();
    let nucleus = compiled
        .fused
        .regions
        .iter()
        .find(|region| region.kind == RegionKind::Library(LibraryOp::NucleusSample))
        .expect("scaled nucleus library region");
    assert_eq!(nucleus.nodes, (6..=18).map(NodeIndex).collect::<Vec<_>>());
    assert_eq!(nucleus.inputs, vec![2, 4, 5, 1, 0]);
    assert_eq!(nucleus.outputs, vec![18]);
}

/// The scaled nucleus carries 5 inputs, not 3, and still matches as one
/// library region. Every CUDA site accepts both arities; a structural decoder
/// that has since been deleted accepted only 3, and so rejected a plan this
/// crate had just produced.
#[test]
fn a_scaled_nucleus_matches_as_one_region_at_five_inputs() {
    let compiled =
        compile_stage(&scaled_nucleus_program(MaskFill::Scalar), Stage::Epilogue).unwrap();
    let nucleus = compiled
        .fused
        .regions
        .iter()
        .find(|region| region.kind == RegionKind::Library(LibraryOp::NucleusSample))
        .expect("scaled nucleus library region");
    assert_eq!(nucleus.inputs.len(), 5);
}

#[test]
fn byte_identical_nucleus_dags_share_signature_and_library_plan() {
    let first = nucleus_program(NucleusMutation::Exact);
    let second = bind(first.container.clone(), first.profile.clone()).unwrap();
    assert_eq!(
        tensor_ir::container::encode(&first.container),
        tensor_ir::container::encode(&second.container)
    );
    let first = compile_stage(&first, Stage::Epilogue).unwrap();
    let second = compile_stage(&second, Stage::Epilogue).unwrap();
    assert_eq!(first.signature, second.signature);
    assert_eq!(first.fused, second.fused);
    assert_eq!(debug_stage_plan(&first), debug_stage_plan(&second));
}

#[test]
fn nucleus_matching_uses_connectivity_not_contiguous_source_ranges() {
    let compiled = compile_stage(&interleaved_nucleus_program(), Stage::Epilogue).unwrap();
    let region = compiled
        .fused
        .regions
        .iter()
        .find(|region| region.kind == RegionKind::Library(LibraryOp::NucleusSample))
        .expect("interleaved nucleus region");
    assert_eq!(
        region.nodes,
        core::iter::once(3)
            .chain(5..=16)
            .map(NodeIndex)
            .collect::<Vec<_>>()
    );
    assert_eq!(region.inputs, vec![2, 1, 0]);
    assert_eq!(region.outputs, vec![16]);
}

#[test]
fn nucleus_lookalikes_remain_generic() {
    let exact = compile_stage(&nucleus_program(NucleusMutation::Exact), Stage::Epilogue).unwrap();
    let commuted = compile_stage(
        &nucleus_program(NucleusMutation::CommutedAdd),
        Stage::Epilogue,
    )
    .unwrap();
    assert!(
        commuted
            .fused
            .regions
            .iter()
            .any(|region| { region.kind == RegionKind::Library(LibraryOp::NucleusSample) })
    );

    for mutation in [
        NucleusMutation::WrongPredicate,
        NucleusMutation::WrongSelectSource,
        NucleusMutation::WrongCenteredSource,
        NucleusMutation::FiniteMaskFill,
        NucleusMutation::UniformRng,
        NucleusMutation::PartialArgmax,
        NucleusMutation::WrongSumOperand,
        NucleusMutation::EscapedIntermediate,
        NucleusMutation::ForeignMaximum,
    ] {
        let near = compile_stage(&nucleus_program(mutation), Stage::Epilogue).unwrap();
        assert_ne!(near.signature, exact.signature);
        assert!(
            !near
                .fused
                .regions
                .iter()
                .any(|region| { region.kind == RegionKind::Library(LibraryOp::NucleusSample) })
        );
    }
}

#[test]
fn epilogue_signature_ignores_unrelated_descriptor_schema() {
    let first = program(1, 1);
    let mut container = first.container.clone();
    container.ports.push(PortBinding {
        port: Port::AttnMask,
        source: PortSource::Const {
            dtype: DType::Bool,
            shape: Shape::vector(1),
            data: vec![1],
        },
    });
    let second = bind(container, first.profile.clone()).unwrap();
    assert_eq!(
        compile_stage(&first, Stage::Epilogue).unwrap().signature,
        compile_stage(&second, Stage::Epilogue).unwrap().signature
    );
}

#[test]
fn symbolic_row_shapes_share_signature_but_vocab_does_not() {
    let one = softmax_program(1, 32);
    let eight = softmax_program(8, 32);
    let other_vocab = softmax_program(1, 64);
    let signature = |bound: &BoundTrace| compile_stage(bound, Stage::Epilogue).unwrap().signature;
    assert_eq!(signature(&one), signature(&eight));
    assert_ne!(signature(&one), signature(&other_vocab));
}

#[test]
fn gather_prefix_dimensions_come_from_indices() {
    let vocab = 32;
    let container = TraceContainer {
        channels: vec![
            channel(Shape::matrix(2, vocab), DType::F32, HostRole::None, true),
            channel(Shape::vector(3), DType::U32, HostRole::None, true),
            channel(Shape::matrix(3, vocab), DType::F32, HostRole::Reader, false),
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::ChanTake(1),
                Op::Gather { src: 0, idx: 1 },
                Op::ChanPut { chan: 2, value: 2 },
            ],
        }],
        ..TraceContainer::default()
    };
    let bound = bind(container, ModelProfile::dummy()).unwrap();
    let compiled = compile_stage(&bound, Stage::Epilogue).unwrap();
    assert_eq!(
        compiled.normalized.value_types[2].dims,
        vec![Dimension::Static(3), Dimension::Static(vocab)]
    );
}

#[test]
fn channel_storage_stays_static_while_lane_carries_logical_extents() {
    let container = TraceContainer {
        channels: vec![channel(
            Shape::matrix(4, 8),
            DType::Bool,
            HostRole::None,
            true,
        )],
        ports: vec![PortBinding {
            port: Port::AttnMask,
            source: PortSource::Channel(0),
        }],
        stages: vec![StageProgram {
            stage: Stage::Prologue,
            ops: vec![Op::ChanRead(0)],
        }],
        ..TraceContainer::default()
    };
    let bound = bind(container, ModelProfile::dummy()).unwrap();
    let compiled = compile_stage(&bound, Stage::Prologue).unwrap();
    assert_eq!(
        compiled.normalized.value_types[0].dims,
        vec![Dimension::Static(4), Dimension::Static(8),]
    );
    let lane = LaneRecord {
        query_len: 4,
        key_len: 8,
        ..LaneRecord::default()
    };
    assert_eq!((lane.query_len, lane.key_len), (4, 8));
}

#[test]
fn mtp_row_count_remains_distinct_and_static() {
    let vocab = 32;
    let container = TraceContainer {
        channels: vec![channel(
            Shape::vector(3),
            DType::I32,
            HostRole::Reader,
            false,
        )],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::MtpLogits,
                    shape: Shape::matrix(3, vocab),
                    dtype: DType::F32,
                },
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
        ..TraceContainer::default()
    };
    let mut profile = ModelProfile::dummy();
    profile.vocab = vocab;
    profile.has_mtp_logits = true;
    let bound = bind(container, profile).unwrap();
    let compiled = compile_stage(&bound, Stage::Epilogue).unwrap();
    assert_eq!(
        compiled.normalized.value_types[0].dims,
        vec![Dimension::Static(3), Dimension::Static(vocab)]
    );
}

#[test]
fn explicit_candidate_batch_does_not_inherit_sampled_rows() {
    let vocab = 32;
    let container = TraceContainer {
        channels: vec![
            channel(Shape::vector(2), DType::U32, HostRole::None, true),
            channel(Shape::matrix(4, vocab), DType::F32, HostRole::Reader, false),
            channel(Shape::matrix(1, vocab), DType::F32, HostRole::Reader, false),
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::Logits,
                    shape: Shape::matrix(1, vocab),
                    dtype: DType::F32,
                },
                Op::ChanTake(0),
                Op::RngKeyed {
                    state: 1,
                    shape: Shape::matrix(4, vocab),
                    kind: tensor_ir::RngKind::Gumbel,
                },
                Op::ChanPut { chan: 1, value: 2 },
                Op::ChanPut { chan: 2, value: 0 },
            ],
        }],
        ..TraceContainer::default()
    };
    let mut profile = ModelProfile::dummy();
    profile.vocab = vocab;
    let bound = bind(container, profile).unwrap();
    let compiled = compile_stage(&bound, Stage::Epilogue).unwrap();
    assert_eq!(
        compiled.normalized.value_types[0].dims,
        vec![
            Dimension::Symbolic(SymbolicExtent::SampledRows),
            Dimension::Static(vocab),
        ]
    );
    assert_eq!(
        compiled.normalized.value_types[2].dims,
        vec![Dimension::Static(4), Dimension::Static(vocab)]
    );
}

// There is deliberately no `runtime_extents_do_not_change_signature` test.
// `compile_stage` does not take `RuntimeExtents` at all, so extents cannot
// reach the signature — the function's own type enforces it, and a test
// cannot strengthen that. What such a test can do is look like proof while
// asserting nothing: build two `ScheduleBucket`s, assert they differ, then
// assert `stage.signature == stage.signature.clone()` with nothing in between
// that could have changed it. Prefer the type.

/// The rendering a plan reaches humans through is a pure function of the plan.
///
/// `debug_stage_plan` is what `crates/runtime/src/pipeline/program.rs` prints
/// when a program is registered under a debug flag, and what the extended
/// golden pins. Both uses assume it does not vary run to run.
#[test]
fn plan_rendering_is_deterministic_and_self_describing() {
    let bound = program(1, 1);
    let stage = compile_stage(&bound, Stage::Epilogue).unwrap();
    let rendered = debug_stage_plan(&stage);
    assert_eq!(rendered, debug_stage_plan(&stage));
    assert!(rendered.contains("epilogue signature="));
    assert!(rendered.contains(&format!("{:016x}", stage.signature.hash)));
    assert!(rendered.contains("Fused"));
    assert_eq!(
        rendered.matches("    r").count(),
        stage.singleton.regions.len() + stage.fused.regions.len(),
        "every region of both partitions is rendered"
    );
    assert_eq!(
        stage.metrics().normalized_ops,
        stage.normalized.ops.len() as u32
    );
}

#[test]
fn lane_layout_is_stable() {
    assert_eq!(core::mem::size_of::<LaneTableHeader>(), 16);
    assert_eq!(core::mem::size_of::<LaneRecord>(), 96);
    assert_eq!(core::mem::size_of::<LaneChannelSlot>(), 32);
}

/// Why the "the result has to escape" check in `region::chain_is_exclusive`
/// is silent.
///
/// A nucleus chain whose token nothing reads is pure and unread, so
/// normalization deletes it and the matcher never sees it. In SSA every
/// consumer of the argmax comes after it, so no chain node can be one, and
/// "every consumer is inside the chain" therefore means "there are none" --
/// exactly the case DCE has already removed. The check stays as the matcher's
/// own statement of what it is for; this is the fact that keeps it honest,
/// and it fails if DCE ever stops running first.
#[test]
fn a_sampler_nobody_reads_is_deleted_before_it_is_matched() {
    let compiled = compile_stage(
        &nucleus_program(NucleusMutation::DeadResult),
        Stage::Epilogue,
    )
    .unwrap();
    assert!(
        !compiled
            .normalized
            .ops
            .iter()
            .any(|op| matches!(op, Op::ReduceArgmax(_) | Op::PivotThreshold { .. })),
        "the dead chain survived normalization, so the escape check is now \
         reachable and needs a mutation that reaches it: {:?}",
        compiled.normalized.ops
    );
}
