use crate::msl_corpus::Served;
use eta_compiler::codegen::error::{EmitError, EmitterKind};
use eta_compiler::codegen::metal::{METAL_M2_MAX_FUSED_CHANNELS, emit_fused_region};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, Shape};

const VOCAB: u32 = 8;

const CHANNELS: usize = 11;

const SCORE_CEILING: usize = 10;

const _: () = assert!(CHANNELS <= METAL_M2_MAX_FUSED_CHANNELS);

fn writer() -> ChannelDecl {
    ChannelDecl {
        shape: Shape::vector(1),
        dtype: ChanDType::Concrete(Dtype::F32),
        capacity: 1,
        host_role: HostRole::Reader,
        seeded: false,
    }
}

fn token_out() -> ChannelDecl {
    ChannelDecl {
        shape: Shape::new(&[]).expect("a scalar shape"),
        dtype: ChanDType::Concrete(Dtype::I32),
        capacity: 1,
        host_role: HostRole::Reader,
        seeded: false,
    }
}

fn subject() -> TraceContainer {
    let mut channels = vec![token_out()];
    channels.extend(std::iter::repeat_with(writer).take(CHANNELS - 1));

    let mut ops = vec![
        Op::Output {
            index: 1,
            lane: 0,
            shape: Shape::matrix(1, VOCAB),
            dtype: Dtype::F32,
        },
        Op::Reshape {
            value: 0,
            shape: Shape::vector(VOCAB),
        },
        Op::ReduceArgmax(1),
        Op::ChanPut { chan: 0, value: 2 },
        Op::Output {
            index: 0,
            lane: 0,
            shape: Shape::matrix(1, 2048),
            dtype: Dtype::F32,
        },
        Op::ReduceSum(3),
    ];
    for chan in 1..CHANNELS {
        ops.push(Op::ChanPut {
            chan: chan as u32,
            value: 4,
        });
    }

    TraceContainer {
        names: vec!["attn.scores".into(), "logits".into()],
        channels,
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn served() -> Served {
    Served::of(VOCAB)
}

#[test]
fn the_single_lane_form_declines_this_shape_by_the_score_ceiling() {
    let bound = served().bind(subject()).expect("the subject binds");
    let stages = compile_bound(&bound);
    let stage = stages.first().expect("one stage");
    assert_eq!(
        stage.normalized.channel_bindings.len(),
        CHANNELS,
        "the subject stopped binding the channel count that makes it the subject"
    );
    let region = stage.fused.regions.first().expect("one fused region");
    let declined = emit_fused_region("m2", stage, region).expect_err("M2 must decline");
    assert_eq!(
        declined,
        EmitError::ChannelLimitExceeded {
            emitter: EmitterKind::MetalFused,
            limit: SCORE_CEILING,
        }
    );
    assert_eq!(
        declined.to_string(),
        format!("fused region exceeds the {SCORE_CEILING}-channel direct-binding limit")
    );
}
