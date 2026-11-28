//! A registered channel that instances of two programs carry on the device
//! (as an inferlet's sampler state between its prefill and decode programs)
//! keeps one cell in the pool's `Channel` slots: a pass of one instance sees
//! what the last pass of any other left, after the one that last held the
//! cell closed too. Each pass adds one to the cell's second word and puts it
//! on a reader ring. Skipped without a PJRT plugin.

use engine::channel::{ChannelId, ChannelRegistration, ChannelSeed};
use engine::program::{BindExtents, InstanceBinding, InstanceId, ProgramId};
use engine_xla::store::layout::CHANNEL;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, Stage};
use eta_ir::types::{Dtype, Shape};

use crate::a_device_geometry_beam_answers_the_host_path::{host_lane, plan};
use crate::common_dit::{self, Rig, Weights, attach};

const COUNTER: ChannelId = ChannelId(900);
const SEED: [u32; 2] = [7, 100];

/// Takes the counter, adds `[0, 1]`, and puts the sum back and on the
/// reader ring.
fn counter() -> TraceContainer {
    TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![
            ChannelDecl {
                shape: Shape::vector(2),
                dtype: ChanDType::Concrete(Dtype::U32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: true,
            },
            ChannelDecl {
                shape: Shape::vector(2),
                dtype: ChanDType::Concrete(Dtype::U32),
                capacity: 2,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::Iota { len: 2 },
                Op::Add(0, 1),
                Op::ChanPut { chan: 0, value: 2 },
                Op::ChanPut { chan: 1, value: 2 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bind(rig: &mut Rig, program: ProgramId, seeded: bool) -> InstanceId {
    let out = rig.channel_of(vec![2], Dtype::U32, HostRole::Reader);
    engine::Programs::bind(
        &mut rig.engine,
        &InstanceBinding {
            program,
            channels: vec![COUNTER, out],
            seeds: if seeded {
                vec![ChannelSeed {
                    channel: 0,
                    bytes: words(&SEED),
                }]
            } else {
                Vec::new()
            },
            geometry: GeometryClass::Host,
            device_resolved: Default::default(),
            instance_slots: Vec::new(),
            // The counter's ring is the registered channel's: the runtime
            // takes none for it here.
            ring_slots: Vec::new(),
            adapter_slots: Vec::new(),
            extents: BindExtents::default(),
            scratch: engine::ScratchPlace::None,
            pair: engine::ScratchPlace::None,
            pair_generation: 0,
        },
    )
    .expect("an instance bound to the registered counter binds")
    .id
}

/// One pass of `instance` on a lane of one token: the count it put.
fn pass(rig: &mut Rig, instance: InstanceId, page: u32) -> u32 {
    let kv = crate::common_llm::Refs::of_facts(rig.loaded.caps.chunks.clone())
        .shapes(|s| s.role == engine::chunk::SlotRole::State)[0]
        .id;
    rig.fire(
        vec![host_lane(kv, 1, vec![3], vec![page], 0)],
        vec![attach(0, instance)],
        Vec::new(),
    );
    let bytes = engine::Channels::take(&mut rig.engine, instance, 1)
        .expect("the reader ring reads")
        .expect("the pass put a cell");
    let cell: Vec<u32> = (bytes.as_chunks::<4>().0.iter())
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    assert_eq!(cell[0], SEED[0], "the first word rides unchanged");
    cell[1]
}

fn contiguous(refs: &mut crate::common_llm::Refs, n: u32) -> Vec<u32> {
    let mut got: Vec<u32> = (0..n).map(|_| refs.take(CHANNEL)).collect();
    got.sort_unstable();
    assert!(
        got.windows(2).all(|w| w[1] == w[0] + 1),
        "one contiguous run: {got:?}"
    );
    got
}

#[test]
fn a_registered_channel_carries_one_cell_across_its_instances() {
    let Some(_device) = common_dit::device() else {
        return;
    };
    let plan = plan();
    let weights = Weights::random(&plan, 17);
    let mut rig = Rig::load(plan, &weights, 64, vec![1, 2, 16, 32, 64], None);
    let mut refs = crate::common_llm::Refs::of_facts(rig.loaded.caps.chunks.clone());
    let kv = refs.shapes(|s| s.role == engine::chunk::SlotRole::State)[0].id;
    let page = refs.take(kv);

    let mut registration = ChannelRegistration {
        id: COUNTER,
        shape: vec![2],
        dtype: ChanDType::Concrete(Dtype::U32),
        host_role: HostRole::None,
        seeded: true,
        capacity: 1,
        ..ChannelRegistration::default()
    };
    let ring = engine::Channels::ring_facts(&rig.engine, &registration)
        .expect("the engine answers")
        .expect("a channel an instance may carry keeps its ring in Channel slots");
    assert_eq!(ring.footprint.slots, 1, "one 1 KiB slot holds the cell");
    registration.slots = contiguous(&mut refs, ring.footprint.slots);
    engine::Channels::open(&mut rig.engine, &registration)
        .expect("the counter registers over its slots");

    // Two programs of one content, as an inferlet's prefill and decode
    // programs share their sampler's channel.
    let first = rig.register(counter());
    let second = rig.register(counter());
    let facts = engine::Programs::facts(&rig.engine, first).expect("the program's facts");
    assert!(
        facts.rings.iter().any(|r| r.channel == 0),
        "the counter is carried on the device: {facts:?}"
    );

    let x = bind(&mut rig, first, true);
    let y = bind(&mut rig, second, false);
    let device0 = rig.engine.guest_passes();
    let mut counts = vec![pass(&mut rig, x, page), pass(&mut rig, x, page)];
    counts.push(pass(&mut rig, y, page));

    // `y` closes holding the cell; a fresh instance carries on from it.
    engine::Programs::close(&mut rig.engine, y).expect("y closes");
    let z = bind(&mut rig, first, true);
    counts.push(pass(&mut rig, z, page));
    counts.push(pass(&mut rig, x, page));
    let device1 = rig.engine.guest_passes();
    let want: Vec<u32> = (1..=counts.len() as u32).map(|k| SEED[1] + k).collect();
    assert_eq!(
        counts, want,
        "every pass sees the last pass of any instance bound to the counter"
    );
    assert_eq!(
        device1 - device0,
        counts.len() as u64,
        "every pass ran on the device"
    );
}

/// Takes a cell the host wrote, adds `[0, 1]`, and puts the sum on the
/// reader ring: an instance the host gave no cell blocks.
fn incrementer() -> TraceContainer {
    let mut program = counter();
    program.channels[0].host_role = HostRole::Writer;
    program.stages[0].ops = vec![
        Op::ChanTake(0),
        Op::Iota { len: 2 },
        Op::Add(0, 1),
        Op::ChanPut { chan: 1, value: 2 },
    ];
    program
}

/// One instance's pass faulting in a step faults its lanes alone: the
/// step's outcome names that instance's lane, and the other instance's
/// pass in the same step commits its output.
#[test]
fn a_guest_fault_fails_its_own_lanes_alone() {
    let Some(_device) = common_dit::device() else {
        return;
    };
    let plan = plan();
    let weights = Weights::random(&plan, 17);
    let mut rig = Rig::load(plan, &weights, 64, vec![1, 2, 16, 32, 64], None);
    let mut refs = crate::common_llm::Refs::of_facts(rig.loaded.caps.chunks.clone());
    let kv = refs.shapes(|s| s.role == engine::chunk::SlotRole::State)[0].id;
    let (page_a, page_b) = (refs.take(kv), refs.take(kv));
    let program = rig.register(incrementer());
    let bound = |rig: &mut Rig, seeded: bool| -> InstanceId {
        let input = rig.channel_of(vec![2], Dtype::U32, HostRole::Writer);
        let out = rig.channel_of(vec![2], Dtype::U32, HostRole::Reader);
        engine::Programs::bind(
            &mut rig.engine,
            &InstanceBinding {
                program,
                channels: vec![input, out],
                seeds: if seeded {
                    vec![ChannelSeed {
                        channel: 0,
                        bytes: words(&SEED),
                    }]
                } else {
                    Vec::new()
                },
                geometry: GeometryClass::Host,
                device_resolved: Default::default(),
                instance_slots: Vec::new(),
                ring_slots: Vec::new(),
                adapter_slots: Vec::new(),
                extents: BindExtents::default(),
                scratch: engine::ScratchPlace::None,
                pair: engine::ScratchPlace::None,
                pair_generation: 0,
            },
        )
        .expect("the instance binds")
        .id
    };
    let fed = bound(&mut rig, true);
    let starved = bound(&mut rig, false);
    let outcomes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let heard = std::sync::Arc::clone(&outcomes);
    engine::Completion::on_complete(
        &mut rig.engine,
        std::sync::Arc::new(move |_, outcome| {
            heard.lock().expect("outcomes lock").push(outcome);
        }),
    );
    rig.fire(
        vec![
            host_lane(kv, 1, vec![3], vec![page_a], 0),
            host_lane(kv, 2, vec![3], vec![page_b], 0),
        ],
        vec![attach(0, fed), attach(1, starved)],
        Vec::new(),
    );
    let outcomes = outcomes.lock().expect("outcomes lock").clone();
    assert_eq!(outcomes.len(), 1, "the step lands once: {outcomes:?}");
    let faulted = &outcomes[0].faulted;
    assert_eq!(faulted.len(), 1, "one lane faults: {faulted:?}");
    assert_eq!(faulted[0].lane, 1, "the starved instance's lane");
    assert_eq!(faulted[0].instance, Some(starved));
    let bytes = engine::Channels::take(&mut rig.engine, fed, 1)
        .expect("the reader ring reads")
        .expect("the fed instance's pass committed its cell");
    let cell: Vec<u32> = (bytes.as_chunks::<4>().0.iter())
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    assert_eq!(cell, vec![SEED[0], SEED[1] + 1], "the fed instance's sum");
}
