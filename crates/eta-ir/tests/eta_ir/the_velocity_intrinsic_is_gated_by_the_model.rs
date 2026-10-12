use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{ModelProfile, Stage, intrinsic_available, intrinsic_stages};
use eta_ir::types::{Dtype, Shape};
use eta_ir::validate::{ValidateError, bind};

const ROWS: u32 = 4;
const CHANNELS: u32 = 16;

fn profile() -> ModelProfile {
    ModelProfile {
        has_velocity: true,
        velocity_width: CHANNELS,
        ..ModelProfile::dummy()
    }
}

fn epilogue_reading(shape: Shape, dtype: Dtype) -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        channels: vec![ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::Velocity,
                    shape,
                    dtype,
                },
                Op::ChanPut { chan: 0, value: 0 },
            ],
        }],
        externs: Vec::new(),
    }
}

#[test]
fn the_velocity_intrinsic_is_gated_by_the_model_every_case() {
    a_denoising_model_serves_the_velocity_and_a_text_model_refuses_it();
    the_declared_width_must_be_the_models_own();
    the_velocity_is_an_epilogue_value_only();
}

fn a_denoising_model_serves_the_velocity_and_a_text_model_refuses_it() {
    let plane = Shape::matrix(ROWS, CHANNELS);
    bind(epilogue_reading(plane, Dtype::F32), profile())
        .expect("a model that predicts a velocity binds the reading");

    let text_only = ModelProfile {
        has_velocity: false,
        ..profile()
    };
    let refusal = bind(epilogue_reading(plane, Dtype::F32), text_only)
        .expect_err("a model with no velocity seam must refuse the program");
    assert!(
        matches!(
            refusal,
            ValidateError::IntrinsicUnavailable {
                intr: IntrinsicId::Velocity
            }
        ),
        "refused for the wrong reason: {refusal:?}"
    );
}

fn the_declared_width_must_be_the_models_own() {
    let wrong = Shape::matrix(ROWS, CHANNELS + 1);
    let refusal = bind(epilogue_reading(wrong, Dtype::F32), profile())
        .expect_err("a width that is not the model's must be refused");
    assert!(
        matches!(
            refusal,
            ValidateError::IntrinsicTypeRule {
                intr: IntrinsicId::Velocity,
                ..
            }
        ),
        "refused for the wrong reason: {refusal:?}"
    );

    for (shape, dtype) in [
        (Shape::vector(CHANNELS), Dtype::F32),
        (Shape::matrix(ROWS, CHANNELS), Dtype::I32),
    ] {
        assert!(
            bind(epilogue_reading(shape, dtype), profile()).is_err(),
            "bound a velocity of shape {shape:?} dtype {dtype:?}"
        );
    }
}

fn the_velocity_is_an_epilogue_value_only() {
    assert_eq!(
        intrinsic_stages(IntrinsicId::Velocity),
        &[Stage::Epilogue],
        "the velocity is the forward's answer, so no per-layer tap can read it"
    );
    assert!(intrinsic_available(IntrinsicId::Velocity, &profile()));
    assert!(!intrinsic_available(
        IntrinsicId::Velocity,
        &ModelProfile {
            has_velocity: false,
            ..profile()
        }
    ));
    assert!(intrinsic_available(
        IntrinsicId::Hidden,
        &ModelProfile {
            has_velocity: false,
            ..profile()
        }
    ));
}
