//! A float seam binds only at the width of a lane of the pass that reads it out.

use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op, intrinsic_pitch};
use eta_ir::registry::{EngineCaps, Graph, Readout, ReadoutKind, Stage, intrinsic_stages};
use eta_ir::types::{Dtype, Shape};
use eta_ir::validate::{BoundTrace, ValidateError, bind};

const ROWS: u32 = 4;

struct Seam {
    intr: IntrinsicId,
    kind: ReadoutKind,
    width: u32,
}

const SEAMS: [Seam; 2] = [
    Seam {
        intr: IntrinsicId::Velocity,
        kind: ReadoutKind::Velocity,
        width: 16,
    },
    Seam {
        intr: IntrinsicId::Pixels,
        kind: ReadoutKind::Pixels,
        width: 3,
    },
];

fn bound(
    container: TraceContainer,
    lanes: Vec<Option<Readout>>,
) -> Result<BoundTrace, ValidateError> {
    bind(container, lanes, Graph::default(), EngineCaps::default())
}

fn epilogue_read(intr: IntrinsicId, lane: u32, shape: Shape, dtype: Dtype) -> TraceContainer {
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
                    intr,
                    lane,
                    shape,
                    dtype,
                },
                Op::ChanPut { chan: 0, value: 0 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

#[test]
fn a_float_seam_is_gated_by_the_model() {
    for seam in &SEAMS {
        let Seam { intr, kind, width } = *seam;
        let read = |shape, dtype| epilogue_read(intr, 0, shape, dtype);
        let plane = Shape::matrix(ROWS, width);
        let reads = Some(Readout { kind, width });
        let other = Some(Readout {
            kind: ReadoutKind::Logits,
            width: 32,
        });

        bound(read(plane, Dtype::F32), vec![reads])
            .unwrap_or_else(|why| panic!("{intr:?}: a lane reading the seam binds it: {why:?}"));
        let refusal = bound(read(plane, Dtype::F32), vec![other])
            .expect_err("a lane reading something else refuses the program");
        assert!(
            matches!(refusal, ValidateError::IntrinsicUnavailable { intr: i } if i == intr),
            "{intr:?}: refused for the wrong reason: {refusal:?}"
        );

        let refusal = bound(
            read(Shape::matrix(ROWS, width + 1), Dtype::F32),
            vec![reads],
        )
        .expect_err("a width that is not the lane's is refused");
        assert!(
            matches!(refusal, ValidateError::IntrinsicTypeRule { intr: i, .. } if i == intr),
            "{intr:?}: refused for the wrong reason: {refusal:?}"
        );
        for (shape, dtype) in [(Shape::vector(width), Dtype::F32), (plane, Dtype::I32)] {
            assert!(
                bound(read(shape, dtype), vec![reads]).is_err(),
                "{intr:?}: bound a read of shape {shape:?} dtype {dtype:?}"
            );
        }

        bound(
            epilogue_read(intr, 1, plane, Dtype::F32),
            vec![other, reads],
        )
        .unwrap_or_else(|why| panic!("{intr:?}: lane 1 reads the seam: {why:?}"));
        let refusal = bound(
            epilogue_read(intr, 2, plane, Dtype::F32),
            vec![other, reads],
        )
        .expect_err("a lane past the pass's is refused");
        assert!(
            matches!(refusal, ValidateError::IntrinsicLane { intr: i, lane: 2, lanes: 2 } if i == intr),
            "{intr:?}: refused for the wrong reason: {refusal:?}"
        );

        let mut lanes = vec![other; 21];
        lanes[20] = reads;
        let wide = bound(epilogue_read(intr, 20, plane, Dtype::F32), lanes)
            .unwrap_or_else(|why| panic!("{intr:?}: lane 20 reads the seam: {why:?}"));
        assert_eq!(
            intrinsic_pitch(&wide.container.stages[0].ops),
            IntrinsicId::SLOTS * 21
        );

        assert_eq!(
            intrinsic_stages(intr),
            &[Stage::Epilogue],
            "{intr:?} is the fire's answer"
        );
    }

    let refusal = bound(
        epilogue_read(IntrinsicId::Logits, 1, Shape::matrix(1, 32), Dtype::F32),
        vec![
            Some(Readout {
                kind: ReadoutKind::Logits,
                width: 32
            });
            2
        ],
    )
    .expect_err("logits name no lane");
    assert!(matches!(refusal, ValidateError::IntrinsicLane { .. }));
}
