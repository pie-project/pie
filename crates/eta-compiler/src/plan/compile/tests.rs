use super::*;
use alloc::vec;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, OutputDecl, Port};
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

fn served(c: TraceContainer, vocab: u32) -> BoundTrace {
    bind(
        c,
        vec![vec![OutputDecl {
            name: "logits".into(),
            axis: Axis::Row,
            width: vocab,
            dtype: Dtype::F32,
            layers: 0,
            readout: true,
        }]],
        EngineCaps::default(),
    )
    .unwrap()
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
                Op::Output {
                    index: 0,
                    lane: 0,
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
        names: vec!["logits".into()],
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
                port: Port::KvLen,
                source: PortSource::Channel(kv_len),
            },
        ],
        stages,
        externs: vec![],
        inputs: Vec::new(),
    };
    served(container, vocab)
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
    served(
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
        32,
    )
}

#[test]
fn tests_every_case() {
    identical_epilogues_share_signature_across_programs();
    top_k_has_one_canonical_signature_and_library_kind();
}

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

/// `0 + -0` is `+0` in every float width: a one-element cumsum or an
/// `x + 0` of a half-width float is not its element, as in f32; in integer
/// arithmetic it is.
#[test]
fn no_float_width_folds_an_add_of_zero() {
    use super::fold::simplify_alias;
    use super::symbolic::{Dimension, SymbolicType};
    let one = |dtype| SymbolicType {
        dtype,
        dims: vec![Dimension::Static(1)],
    };
    let literals = [None, Some(Literal::I32(0))];
    for dtype in [Dtype::F32, Dtype::F16, Dtype::Bf16] {
        assert_eq!(
            simplify_alias(&Op::CumSum(0), &one(dtype), &literals),
            None,
            "{dtype:?}"
        );
        assert_eq!(
            simplify_alias(&Op::Add(0, 1), &one(dtype), &literals),
            None,
            "{dtype:?}"
        );
    }
    for dtype in [Dtype::I32, Dtype::U32] {
        assert_eq!(
            simplify_alias(&Op::CumSum(0), &one(dtype), &literals),
            Some(0),
            "{dtype:?}"
        );
    }
    assert_eq!(
        simplify_alias(&Op::Add(0, 1), &one(Dtype::I32), &literals),
        Some(0)
    );
}
