#![cfg(feature = "device")]

use crate::common_two_axis::{Rig, WIDTH, Weights, assert_close, frame, trace, velocity_epilogue};
use engine::Engine;
use engine::fire::{Attachment, ReadoutSeam};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, Literal, Shape};

const ROWS: u32 = 8;

fn latent() -> Vec<f32> {
    (0..ROWS * WIDTH)
        .map(|i| crate::common_two_axis::bf(0.01 * i as f32 - 1.0))
        .collect()
}

/// A program whose prologue puts the latent into a device-only input the
/// host never writes, and whose epilogue takes it back and lands the
/// velocity the forward read over it; the timestep and positions are host
/// inputs it never touches.
fn noising_prologue() -> TraceContainer {
    let decl = |shape: Shape, host_role: HostRole| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(Dtype::F32),
        capacity: 2,
        host_role,
        seeded: false,
    };
    let n = ROWS * WIDTH;
    TraceContainer {
        channels: vec![
            decl(Shape::matrix(ROWS, WIDTH), HostRole::None),
            decl(Shape::matrix(1, 1), HostRole::Writer),
            decl(Shape::matrix(ROWS, 2), HostRole::Writer),
            decl(Shape::matrix(ROWS, WIDTH), HostRole::Reader),
        ],
        stages: vec![
            StageProgram {
                stage: Stage::Prologue,
                ops: vec![
                    Op::Iota { len: n },
                    Op::Cast {
                        value: 0,
                        dtype: Dtype::F32,
                    },
                    Op::Const(Literal::F32(0.01)),
                    Op::Broadcast {
                        value: 2,
                        shape: Shape::vector(n),
                    },
                    Op::Mul(1, 3),
                    Op::Const(Literal::F32(-1.0)),
                    Op::Broadcast {
                        value: 5,
                        shape: Shape::vector(n),
                    },
                    Op::Add(4, 6),
                    Op::Reshape {
                        value: 7,
                        shape: Shape::matrix(ROWS, WIDTH),
                    },
                    Op::ChanPut { chan: 0, value: 8 },
                ],
            },
            StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0),
                    Op::IntrinsicVal {
                        intr: IntrinsicId::Velocity,
                        lane: 0,
                        shape: Shape::matrix(ROWS, WIDTH),
                        dtype: Dtype::F32,
                    },
                    Op::ChanPut { chan: 3, value: 1 },
                ],
            },
        ],
        inputs: vec![0, 1, 2],
        ..TraceContainer::default()
    }
}

#[test]
fn the_forward_reads_the_latent_its_prologue_put() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let weights = Weights::random(&trace(), 0x3c);
    let mut rig = Rig::load(&weights, 32, vec![16, 32], 8);
    let positions: Vec<f32> = (0..ROWS).flat_map(|r| [r as f32, 0.5 * r as f32]).collect();

    let noising = rig.register(noising_prologue(), 1);
    let latent_cell = rig.channel(vec![ROWS, WIDTH], HostRole::None);
    let timestep = rig.channel(vec![1, 1], HostRole::Writer);
    let position_cell = rig.channel(vec![ROWS, 2], HostRole::Writer);
    let landed = rig.channel(vec![ROWS, WIDTH], HostRole::Reader);
    let noised = rig.bind(
        noising,
        vec![latent_cell, timestep, position_cell, landed],
        ROWS,
    );
    // Three fires on a ring of two: each prologue finds the latent its
    // fire's epilogue took, so none blocks; each epilogue reads the
    // velocity of the forward that read its own prologue's put.
    // The timestep and positions are read and never taken: one cell each
    // serves every fire.
    rig.publish(noised, 1, &[0.7]);
    rig.publish(noised, 2, &positions);
    let mut fires = Vec::new();
    for fire in 0..3 {
        let mut ticket = rig
            .engine
            .submit(&frame(
                vec![rig.dit_lane(0, ROWS, latent_cell, timestep, position_cell)],
                vec![Attachment {
                    lane: 0,
                    lanes: 1,
                    instance: noised,
                    adapter_only: false,
                }],
                Vec::new(),
            ))
            .unwrap_or_else(|why| {
                panic!("fire {fire}: a prologue that puts its input fires: {why}")
            });
        rig.engine
            .settle_frame(&mut ticket)
            .unwrap_or_else(|why| panic!("fire {fire}: the noising fire settles: {why}"));
        let readout = ticket.steps[0].readouts[0].clone();
        assert_eq!(readout.seam, ReadoutSeam::Velocity);
        let epilogue = rig.take(noised, 3);
        assert_close(
            &epilogue,
            &readout.values,
            &format!("fire {fire}: the velocity the epilogue read after the forward"),
        );
        fires.push(readout);
    }

    let fed = rig.register(velocity_epilogue(ROWS), 2);
    let host_latent = rig.channel(vec![ROWS, WIDTH], HostRole::Writer);
    let host_timestep = rig.channel(vec![1, 1], HostRole::Writer);
    let host_positions = rig.channel(vec![ROWS, 2], HostRole::Writer);
    let out = rig.channel(vec![ROWS, WIDTH], HostRole::Reader);
    let host = rig.bind(
        fed,
        vec![host_latent, host_timestep, host_positions, out],
        ROWS,
    );
    rig.publish(host, 0, &latent());
    rig.publish(host, 1, &[0.7]);
    rig.publish(host, 2, &positions);
    let mut ticket = rig
        .engine
        .submit(&frame(
            vec![rig.dit_lane(0, ROWS, host_latent, host_timestep, host_positions)],
            vec![crate::common_two_axis::attach(0, host)],
            Vec::new(),
        ))
        .expect("the host-fed fire fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the host-fed fire settles");
    let want = rig.take(host, 3);
    assert!(
        want.iter().any(|v| v.abs() > 1e-6),
        "the host-fed forward answered nothing"
    );
    for (fire, readout) in fires.iter().enumerate() {
        assert_close(
            &readout.values,
            &want,
            &format!("fire {fire}: the velocity over the latent the prologue put"),
        );
    }
}
