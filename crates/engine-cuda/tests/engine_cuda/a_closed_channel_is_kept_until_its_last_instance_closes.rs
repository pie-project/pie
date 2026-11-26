#![cfg(feature = "device")]

//! A channel closed while an instance is still bound on it stays
//! registered until that instance closes (`Channels::close`): a
//! process closes a channel while it is parked, with its instance still
//! live, and the restore's rebind of the channel's ring must still be
//! served, else the instance would read its ring where it no longer lies.

use crate::common_two_axis::{Rig, WIDTH, Weights, assert_close, frame, trace};
use engine::chunk::SlotRole;
use engine::fire::Attachment;
use engine::program::Rebind;
use engine::{Channels, Engine, Programs};
use engine_cuda::program::Endpoint;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype, Literal, Shape};

const ROWS: u32 = 8;
/// A byte no cell of the ring holds: what the old slots are filled with
/// once the ring has moved.
const MARK: u8 = 0xa5;

/// A prologue putting a latent into a device-only ring (channel 0) that
/// the forward reads and the epilogue takes back out to the host
/// (channel 3): every fire reads and writes the ring where it lies.
fn prologue_ring() -> TraceContainer {
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
                    Op::Output {
                        index: 0,
                        lane: 0,
                        shape: Shape::matrix(ROWS, WIDTH),
                        dtype: Dtype::F32,
                    },
                    Op::ChanPut { chan: 3, value: 1 },
                ],
            },
        ],
        inputs: vec![0, 1, 2],
        names: vec!["velocity".to_string()],
        ..TraceContainer::default()
    }
}

/// Where the ring's cells lie, and what its old slots hold there.
fn ring_bytes(rig: &Rig, slots: &[u32], ring: &Endpoint) -> (u64, Vec<u8>) {
    let shell = rig.engine.shell().expect("the engine is loaded");
    let span = Endpoint::cell_span(ring.wire_bytes(), ring.cap1() - 1);
    let at = shell
        .guest_span(SlotRole::Channel, slots, span)
        .expect("the ring's slots are a channel item");
    let mut held = vec![0u8; usize::try_from(span).expect("a ring's span")];
    engine_cuda::device::alloc::copy_d2h_on(shell.stream(), at, &mut held)
        .expect("the ring's slots read");
    (at, held)
}

#[test]
fn a_closed_channel_is_kept_until_its_last_instance_closes() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let weights = Weights::random(&trace(), 0x71);
    let mut rig = Rig::load(&weights, 32, vec![16, 32], 8);
    let positions: Vec<f32> = (0..ROWS).flat_map(|r| [r as f32, 0.5 * r as f32]).collect();

    // A channel no instance is bound on is released by its close.
    let alone = rig.channel(vec![ROWS, WIDTH], HostRole::None);
    assert!(rig.engine.has_channel(alone));
    Channels::close(&mut rig.engine, alone).expect("the channel closes");
    assert!(
        !rig.engine.has_channel(alone),
        "a channel no instance is bound on is released at its close"
    );

    let program = rig.register(prologue_ring(), 0x71);
    let ring = rig.channel(vec![ROWS, WIDTH], HostRole::None);
    let timestep = rig.channel(vec![1, 1], HostRole::Writer);
    let position = rig.channel(vec![ROWS, 2], HostRole::Writer);
    let landed = rig.channel(vec![ROWS, WIDTH], HostRole::Reader);
    let instance = rig.bind(program, vec![ring, timestep, position, landed], ROWS);
    rig.publish(instance, 1, &[0.7]);
    rig.publish(instance, 2, &positions);

    let fire = |rig: &mut Rig, what: &str| -> Vec<f32> {
        let mut ticket = rig
            .engine
            .submit(&frame(
                vec![rig.dit_lane(0, ROWS, ring, timestep, position)],
                vec![Attachment {
                    lane: 0,
                    lanes: 1,
                    instance,
                    adapter_only: false,
                    scratch: ::engine::ScratchPlace::None,
                }],
                Vec::new(),
            ))
            .unwrap_or_else(|why| panic!("{what}: the fire submits: {why}"));
        rig.engine
            .settle_frame(&mut ticket)
            .unwrap_or_else(|why| panic!("{what}: the fire settles: {why}"));
        let readout = ticket.steps[0].readouts[0].clone();
        let taken = rig.take(instance, 3);
        assert_close(
            &taken,
            &readout.values,
            &format!("{what}: what the epilogue took from the ring, against the forward's read"),
        );
        readout.values
    };
    let before = fire(&mut rig, "before the park");

    // The process parks: the ring's bytes go to the host and its device
    // slots to someone else. Closed while parked, with its instance live.
    let endpoint = rig.engine.endpoint(ring).expect("the ring is registered");
    let old = endpoint.slots();
    assert!(!old.is_empty(), "a device-only ring lies in Channel slots");
    let (old_at, parked) = ring_bytes(&rig, &old, &endpoint);
    Channels::close(&mut rig.engine, ring).expect("the channel closes");
    assert!(
        rig.engine.has_channel(ring),
        "a channel closed with its instance still bound stays registered"
    );

    // The restore: the ring's bytes come back into fresh slots, the old
    // ones are another item's now (marked), and the rebind is served.
    let fresh = rig.fresh_ring(vec![ROWS, WIDTH], HostRole::None);
    assert_ne!(fresh, old);
    {
        let shell = rig.engine.shell().expect("the engine is loaded");
        let span = Endpoint::cell_span(endpoint.wire_bytes(), endpoint.cap1() - 1);
        let fresh_at = shell
            .guest_span(SlotRole::Channel, &fresh, span)
            .expect("the fresh slots are a channel item");
        engine_cuda::device::alloc::stage_raw(shell.stream(), fresh_at, &parked)
            .expect("the parked ring copies back");
        engine_cuda::device::alloc::stage_raw(shell.stream(), old_at, &vec![MARK; parked.len()])
            .expect("the old slots are overwritten");
    }
    rig.engine
        .rebind(Rebind::Channel {
            id: ring,
            slots: fresh.clone(),
        })
        .expect("a closed channel's ring rebinds while its instance is bound");
    assert_eq!(endpoint.slots(), fresh, "the ring lies in the fresh slots");
    let (fresh_at, _) = ring_bytes(&rig, &fresh, &endpoint);
    assert_eq!(endpoint.device_cells(), Some(fresh_at));

    // No new instance binds on a closed channel.
    let refused = rig
        .engine
        .bind(&engine::program::InstanceBinding {
            program,
            channels: vec![ring, timestep, position, landed],
            ..engine::program::InstanceBinding::default()
        })
        .expect_err("a bind on a closed channel is refused");
    assert!(refused.to_string().contains("is closed"), "{refused}");

    // The instance fires on, reading and writing its ring where it lies now.
    let after = fire(&mut rig, "after the restore");
    assert_close(&after, &before, "the restored instance's fire");
    let (_, old_held) = ring_bytes(&rig, &old, &endpoint);
    assert!(
        old_held.iter().all(|&byte| byte == MARK),
        "the fire read and wrote the ring in its fresh slots, not the old ones"
    );

    // Its last instance's close releases the channel.
    Programs::close(&mut rig.engine, instance).expect("the instance closes");
    assert!(
        !rig.engine.has_channel(ring),
        "the last instance's close releases the closed channel"
    );
    assert!(rig.engine.endpoint(ring).is_none());
}
