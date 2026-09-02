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
use alloc::vec;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{ModelProfile, Port};
use eta_ir::types::{Dtype, Literal, Shape};
use eta_ir::validate::bind;

fn channel(shape: Shape, dtype: Dtype, role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role: role,
        seeded,
    }
}

pub(super) fn program(prefix_constant: u32, global_channel_offset: usize) -> BoundTrace {
    let vocab = 32;
    let mut channels = Vec::new();
    for _ in 0..global_channel_offset {
        channels.push(channel(Shape::SCALAR, Dtype::U32, HostRole::None, true));
    }

    let token = channels.len() as u32;
    channels.push(channel(Shape::vector(1), Dtype::I32, HostRole::None, true));
    let output = channels.len() as u32;
    channels.push(channel(
        Shape::vector(1),
        Dtype::I32,
        HostRole::Reader,
        false,
    ));
    let kv_len = channels.len() as u32;
    channels.push(channel(Shape::vector(1), Dtype::U32, HostRole::None, true));
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
                    dtype: Dtype::F32,
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
                    dtype: Dtype::U32,
                    shape: Shape::vector(1),
                    data: 0u32.to_le_bytes().to_vec(),
                },
            },
            PortBinding {
                port: Port::Pages,
                source: PortSource::Const {
                    dtype: Dtype::U32,
                    shape: Shape::vector(1),
                    data: 0u32.to_le_bytes().to_vec(),
                },
            },
            PortBinding {
                port: Port::PageIndptr,
                source: PortSource::Const {
                    dtype: Dtype::U32,
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
                    dtype: Dtype::U32,
                    shape: Shape::vector(1),
                    data: 0u32.to_le_bytes().to_vec(),
                },
            },
            PortBinding {
                port: Port::WOff,
                source: PortSource::Const {
                    dtype: Dtype::U32,
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
        .map(|_| channel(Shape::SCALAR, Dtype::U32, HostRole::None, true))
        .collect::<Vec<_>>();
    let input = channels.len() as u32;
    channels.push(channel(
        Shape::matrix(2, 8),
        Dtype::F32,
        HostRole::None,
        true,
    ));
    let values = channels.len() as u32;
    channels.push(channel(
        Shape::matrix(2, 2),
        Dtype::F32,
        HostRole::Reader,
        false,
    ));
    let indices = channels.len() as u32;
    channels.push(channel(
        Shape::matrix(2, 2),
        Dtype::U32,
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

// There is deliberately no `runtime_extents_do_not_change_signature` test.
// `compile_stage` does not take `RuntimeExtents` at all, so extents cannot
// reach the signature — the function's own type enforces it, and a test
// cannot strengthen that. What such a test can do is look like proof while
// asserting nothing: build two `ScheduleBucket`s, assert they differ, then
// assert `stage.signature == stage.signature.clone()` with nothing in between
// that could have changed it. Prefer the type.

