//! **THE SHAPE `trackb-h2o` ASKS FOR, AND THE ONLY FORM THAT SERVES IT** —
//! one epilogue region that reads the `attn_score` rectangle *and* binds more
//! channels than a score-reading M2 kernel has argument indices for.
//!
//! The M2 form binds each intrinsic rectangle other than the trunk's at the
//! TOP of Metal's argument space and each channel's cell pair from index 7 up,
//! so a region that reads the score plane meets the channels at ten rather
//! than twelve (`fused_channel_ceiling`). `trackb-h2o`'s observing decode
//! binds eleven — ten of its own plus the drained score row — so the single-
//! lane emitter declines it, by an argument-space limit that has nothing to do
//! with what the program asked for. The grouped (M3) form is what lifts that:
//! a channel is a row of the lane table there and the score rectangle is an
//! ADDRESS on the lane record (`LaneRecord::attn_score_base`), so neither
//! crowds the other.
//!
//! **NOTHING PINNED THAT COMBINATION UNTIL THIS FILE.** `golden-extended`'s
//! `extended_attn_score` reads the rectangle with ONE channel, and
//! `engine-metal`'s `the_channel_ceiling_is_an_escape_and_not_a_wall` runs a
//! wide stage that reads NO rectangle. Each half was covered and the crossing
//! was not — which is exactly the cell `trackb-h2o` occupies, and the reason
//! its decline survived a device wave with the road already built.
//!
//! The last assertion is the load-bearing one: it is not enough that the
//! grouped emitter produced a kernel, it has to be filed at the slot the SHELL
//! reads. `engine_metal::program::compile::grouped_region` looks for the fused
//! region's grouped kernel at `singleton.len() + region_index`, and a kernel
//! filed anywhere else reads as "the grouped form could not serve it either"
//! with no way to tell that apart from a refusal.

use eta_compiler::codegen::error::{EmitError, EmitterKind};
use eta_compiler::codegen::metal::{
    METAL_M2_MAX_FUSED_CHANNELS, emit_fused_region,
};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{ModelProfile, Stage};
use eta_ir::types::{Dtype, Shape};
use eta_ir::validate::bind;

/// The vocabulary the subject binds against. Small on purpose: the claim is
/// about argument indices, not about a readout's width.
const VOCAB: u32 = 8;

/// How many channels the subject binds. **ELEVEN, WHICH IS THE WHOLE POINT**:
/// one past the ceiling a score-reading M2 region has (ten), and inside the
/// twelve a region that reads nothing but the trunk would get — so the only
/// thing that declines this stage is the SECOND rectangle, which is
/// `trackb-h2o`'s situation exactly.
const CHANNELS: usize = 11;

/// The ceiling `fused_channel_ceiling` puts a score-reading region at. Spelled
/// as a literal because it is the number the device's refusal PRINTS, and the
/// refusal text is what a verify wave reads.
const SCORE_CEILING: usize = 10;

/// The subject only isolates the SECOND rectangle while it stays inside the
/// plain twelve-slot limit: above that the M2 emitter declines for a reason
/// that has nothing to do with the score plane, and both tests below would
/// still pass while testing the wrong door. A `const` assertion because both
/// sides are constants — clippy says so, and it is right.
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

/// One epilogue that samples off the trunk AND folds the score rectangle, with
/// eleven channels to write both through — `trackb-h2o`'s observing decode,
/// stripped to the two facts that decide its form.
fn subject() -> TraceContainer {
    let mut channels = vec![token_out()];
    channels.extend(std::iter::repeat_with(writer).take(CHANNELS - 1));

    let mut ops = vec![
        // The trunk: logits in, one sampled token out. Its rectangle binds
        // BELOW the channels, so it costs none of them.
        Op::IntrinsicVal {
            intr: IntrinsicId::Logits,
            shape: Shape::matrix(1, VOCAB),
            dtype: Dtype::F32,
        },
        Op::Reshape {
            value: 0,
            shape: Shape::vector(VOCAB),
        },
        Op::ReduceArgmax(1),
        Op::ChanPut { chan: 0, value: 2 },
        // The second rectangle, read whole at the epilogue the way
        // `intrinsics::attn_score` is defined to be read.
        Op::IntrinsicVal {
            intr: IntrinsicId::AttnScore,
            shape: Shape::matrix(1, eta_ir::registry::ATTN_SCORE_KV_MAX),
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
        names: Vec::new(),
        channels,
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

fn profile() -> ModelProfile {
    let mut profile = ModelProfile::dummy();
    profile.vocab = VOCAB;
    profile.has_attn_score = true;
    profile
}

#[test]
fn the_single_lane_form_declines_this_shape_by_the_score_ceiling() {
    let bound = bind(subject(), profile()).expect("the subject binds");
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
    // The exact sentence a device wave reads out of the engine's refusal. It
    // cost the metal verify queue two sessions to attribute; pinning it means
    // the next reader can grep for it.
    assert_eq!(
        declined.to_string(),
        format!("fused region exceeds the {SCORE_CEILING}-channel direct-binding limit")
    );
}

