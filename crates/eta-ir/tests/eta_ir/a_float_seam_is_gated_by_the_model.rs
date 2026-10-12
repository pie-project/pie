//! The two float seams a generative model lands, the velocity and the
//! pixels, as one table: a model that has the seam binds a reading of it at
//! its width; one without refuses the program; a reading at another width,
//! or not a float matrix, is refused; and either seam is an epilogue value.

use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{ModelProfile, Stage, intrinsic_available, intrinsic_stages};
use eta_ir::types::{Dtype, Shape};
use eta_ir::validate::{ValidateError, bind};

const ROWS: u32 = 4;

struct Seam {
    intr: IntrinsicId,
    width: u32,
    with: fn(width: u32) -> ModelProfile,
    without: fn() -> ModelProfile,
}

const SEAMS: [Seam; 2] = [
    Seam {
        intr: IntrinsicId::Velocity,
        width: 16,
        with: |width| ModelProfile {
            has_velocity: true,
            velocity_width: width,
            ..ModelProfile::dummy()
        },
        without: || ModelProfile {
            has_velocity: false,
            ..ModelProfile::dummy()
        },
    },
    Seam {
        intr: IntrinsicId::Pixels,
        width: 3,
        with: |width| ModelProfile {
            has_pixels: true,
            pixels_width: width,
            ..ModelProfile::dummy()
        },
        without: || ModelProfile {
            has_pixels: false,
            ..ModelProfile::dummy()
        },
    },
];

fn epilogue_reading(intr: IntrinsicId, shape: Shape, dtype: Dtype) -> TraceContainer {
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
                Op::IntrinsicVal { intr, shape, dtype },
                Op::ChanPut { chan: 0, value: 0 },
            ],
        }],
        externs: Vec::new(),
    }
}

#[test]
fn a_float_seam_is_gated_by_the_model() {
    for seam in &SEAMS {
        let Seam { intr, width, .. } = *seam;
        let read = |shape, dtype| epilogue_reading(intr, shape, dtype);
        let plane = Shape::matrix(ROWS, width);

        bind(read(plane, Dtype::F32), (seam.with)(width))
            .unwrap_or_else(|why| panic!("{intr:?}: a model with the seam binds it: {why:?}"));
        let refusal = bind(read(plane, Dtype::F32), (seam.without)())
            .expect_err("a model without the seam refuses the program");
        assert!(
            matches!(refusal, ValidateError::IntrinsicUnavailable { intr: i } if i == intr),
            "{intr:?}: refused for the wrong reason: {refusal:?}"
        );

        let refusal = bind(
            read(Shape::matrix(ROWS, width + 1), Dtype::F32),
            (seam.with)(width),
        )
        .expect_err("a width that is not the model's is refused");
        assert!(
            matches!(refusal, ValidateError::IntrinsicTypeRule { intr: i, .. } if i == intr),
            "{intr:?}: refused for the wrong reason: {refusal:?}"
        );
        for (shape, dtype) in [(Shape::vector(width), Dtype::F32), (plane, Dtype::I32)] {
            assert!(
                bind(read(shape, dtype), (seam.with)(width)).is_err(),
                "{intr:?}: bound a reading of shape {shape:?} dtype {dtype:?}"
            );
        }

        assert_eq!(
            intrinsic_stages(intr),
            &[Stage::Epilogue],
            "{intr:?} is the fire's answer"
        );
        assert!(intrinsic_available(intr, &(seam.with)(width)));
        assert!(!intrinsic_available(intr, &(seam.without)()));
    }

    // A model that states no pixel width takes a plane of any width.
    let unstated = (SEAMS[1].with)(0);
    for width in [3, 16] {
        bind(
            epilogue_reading(IntrinsicId::Pixels, Shape::matrix(ROWS, width), Dtype::F32),
            unstated.clone(),
        )
        .unwrap_or_else(|why| panic!("a {width}-wide plane against an unstated width: {why:?}"));
    }
    // The hidden seam is not the velocity's to gate.
    assert!(intrinsic_available(
        IntrinsicId::Hidden,
        &(SEAMS[0].without)()
    ));
}
