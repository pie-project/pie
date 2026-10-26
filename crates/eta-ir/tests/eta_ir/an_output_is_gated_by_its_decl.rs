//! An output binds only as the lane reading it declares it.

use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{Op, output_pitch};
use eta_ir::registry::{Axis, EngineCaps, OutputDecl, Stage};
use eta_ir::types::{Dtype, Shape};
use eta_ir::validate::{BoundTrace, ValidateError, bind};

const ROWS: u32 = 4;

fn decl(name: &str, axis: Axis, width: u32, layers: u32) -> OutputDecl {
    OutputDecl {
        name: name.into(),
        axis,
        width,
        dtype: Dtype::F32,
        layers,
        readout: false,
    }
}

fn bound(
    container: TraceContainer,
    lanes: Vec<Vec<OutputDecl>>,
) -> Result<BoundTrace, ValidateError> {
    bind(container, lanes, EngineCaps::default())
}

fn read(name: &str, stage: Stage, lane: u32, shape: Shape, dtype: Dtype) -> TraceContainer {
    TraceContainer {
        names: vec![name.into()],
        channels: vec![ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage,
            ops: vec![
                Op::Output {
                    index: 0,
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
fn an_output_is_gated_by_its_decl() {
    let velocity = decl("velocity", Axis::Row, 16, 0);
    let other = decl("logits", Axis::Row, 32, 0);
    let plane = Shape::matrix(ROWS, 16);
    let at = |lane, shape, dtype| read("velocity", Stage::Epilogue, lane, shape, dtype);

    bound(at(0, plane, Dtype::F32), vec![vec![velocity.clone()]])
        .unwrap_or_else(|why| panic!("a lane reading it binds it: {why:?}"));
    let refusal = bound(at(0, plane, Dtype::F32), vec![vec![other.clone()]])
        .expect_err("a lane reading something else refuses the program");
    assert!(matches!(
        refusal,
        ValidateError::OutputUnavailable { lane: 0, .. }
    ));
    for (shape, dtype) in [
        (Shape::matrix(ROWS, 17), Dtype::F32),
        (Shape::vector(16), Dtype::F32),
        (plane, Dtype::I32),
    ] {
        let refusal = bound(at(0, shape, dtype), vec![vec![velocity.clone()]])
            .expect_err("a read not the decl's is refused");
        assert!(matches!(refusal, ValidateError::OutputTypeRule { .. }));
    }
    let refusal = bound(
        read("velocity", Stage::Prologue, 0, plane, Dtype::F32),
        vec![vec![velocity.clone()]],
    )
    .expect_err("an output is read at the epilogue");
    assert!(matches!(refusal, ValidateError::OutputWrongStage { .. }));

    let lanes = vec![vec![other.clone()], vec![velocity.clone()]];
    bound(at(1, plane, Dtype::F32), lanes.clone())
        .unwrap_or_else(|why| panic!("lane 1 reads it: {why:?}"));
    let refusal = bound(at(2, plane, Dtype::F32), lanes).expect_err("a lane past the pass's");
    assert!(matches!(
        refusal,
        ValidateError::OutputLane {
            lane: 2,
            lanes: 2,
            ..
        }
    ));

    let mut lanes = vec![vec![other]; 21];
    lanes[20] = vec![velocity];
    let wide = bound(at(20, plane, Dtype::F32), lanes)
        .unwrap_or_else(|why| panic!("lane 20 reads it: {why:?}"));
    assert_eq!(output_pitch(&wide.container.stages[0].ops), 21);

    // A lane output reads a row, or its layers stacked.
    let scores = |layers| decl("attn.scores", Axis::Lane, 8, layers);
    let lane_read = |rows| {
        read(
            "attn.scores",
            Stage::Epilogue,
            0,
            Shape::matrix(rows, 8),
            Dtype::F32,
        )
    };
    bound(lane_read(1), vec![vec![scores(0)]]).expect("one row a lane");
    bound(lane_read(3), vec![vec![scores(3)]]).expect("a row a layer");
    assert!(bound(lane_read(2), vec![vec![scores(3)]]).is_err());
}
