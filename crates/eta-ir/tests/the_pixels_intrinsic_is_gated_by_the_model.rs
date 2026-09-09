use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{ModelProfile, Stage, intrinsic_available, intrinsic_stages};
use eta_ir::types::{Dtype, Shape};
use eta_ir::validate::{ValidateError, bind};

const ROWS: u32 = 64;
const RGB: u32 = 3;

fn profile(pixels_width: u32) -> ModelProfile {
    ModelProfile {
        has_pixels: true,
        pixels_width,
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
                    intr: IntrinsicId::Pixels,
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
fn the_pixels_intrinsic_is_gated_by_the_model_every_case() {
    a_model_with_a_vae_serves_the_pixels_and_one_without_refuses_them();
    the_declared_width_is_the_models_when_it_states_one();
    the_pixels_are_an_epilogue_value_only();
}

fn a_model_with_a_vae_serves_the_pixels_and_one_without_refuses_them() {
    let plane = Shape::matrix(ROWS, RGB);
    bind(epilogue_reading(plane, Dtype::F32), profile(RGB))
        .expect("a model that lands pixels binds the reading");

    let vaeless = ModelProfile {
        has_pixels: false,
        ..profile(RGB)
    };
    let refusal = bind(epilogue_reading(plane, Dtype::F32), vaeless)
        .expect_err("a model with no pixels seam must refuse the program");
    assert!(
        matches!(
            refusal,
            ValidateError::IntrinsicUnavailable {
                intr: IntrinsicId::Pixels
            }
        ),
        "refused for the wrong reason: {refusal:?}"
    );
}

fn the_declared_width_is_the_models_when_it_states_one() {
    let wrong = Shape::matrix(ROWS, RGB + 1);
    let refusal = bind(epilogue_reading(wrong, Dtype::F32), profile(RGB))
        .expect_err("a width that is not the model's must be refused");
    assert!(
        matches!(
            refusal,
            ValidateError::IntrinsicTypeRule {
                intr: IntrinsicId::Pixels,
                ..
            }
        ),
        "refused for the wrong reason: {refusal:?}"
    );
    for width in [RGB, 16] {
        bind(
            epilogue_reading(Shape::matrix(ROWS, width), Dtype::F32),
            profile(0),
        )
        .unwrap_or_else(|why| panic!("a {width}-wide plane against an unstated width: {why:?}"));
    }
    for (shape, dtype) in [
        (Shape::vector(RGB), Dtype::F32),
        (Shape::matrix(ROWS, RGB), Dtype::I32),
    ] {
        assert!(
            bind(epilogue_reading(shape, dtype), profile(0)).is_err(),
            "bound pixels of shape {shape:?} dtype {dtype:?}"
        );
    }
}

fn the_pixels_are_an_epilogue_value_only() {
    assert_eq!(
        intrinsic_stages(IntrinsicId::Pixels),
        &[Stage::Epilogue],
        "the pixels are the fire's answer, so no per-layer tap can read them"
    );
    assert!(intrinsic_available(IntrinsicId::Pixels, &profile(RGB)));
    assert!(!intrinsic_available(
        IntrinsicId::Pixels,
        &ModelProfile {
            has_pixels: false,
            ..profile(RGB)
        }
    ));
}
