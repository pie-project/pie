//! A guest instance whose launch tables lie in `Instance` pool slots and
//! whose kept cell lies in `Channel` slots (REDESIGN step 4) reads the same
//! outputs after an evacuation moved their chunks and after a park through
//! the host tier moved them to other refs.

use engine::ChannelId;
use engine::Engine;
use engine::channel::ChannelSeed;
use engine::chunk::{Evacuation, ShapeId, SlotCopy};
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step};
use engine::program::{InstanceBinding, InstanceId, Rebind, RingSlots};
use engine::transfer::MemoryDomain;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::Request;

use super::outputs::{LAYERS, Loaded, Outputs, VOCAB, WIDTH, channel, loaded_with, output_decls, registration_of};
use super::slots::Slots;
use crate::backend;

/// Channels: `ids` (doubled) and `stack` the host reads, `t` it writes,
/// `last` the instance's own (seeded; `WIDE` copies of the pass's id, which
/// the device keeps between passes where it keeps cells: one of capacity 1
/// and 64 KiB, so a run of `Channel` slots), `seen` the host reads: twice
/// what `last` held when the pass began.
fn program() -> TraceContainer {
    let decl = |shape: Shape, dtype: EtaDtype, host_role: HostRole, capacity: u32| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity,
        host_role,
        seeded: false,
    };
    let ids = || Op::Output { index: 0, lane: 0, shape: Shape::matrix(1, 1), dtype: EtaDtype::I32 };
    TraceContainer {
        names: vec!["ids".to_string(), "stack".to_string()],
        channels: vec![
            decl(Shape::matrix(1, 1), EtaDtype::I32, HostRole::Reader, 2),
            decl(Shape::matrix(LAYERS, WIDTH), EtaDtype::F32, HostRole::Reader, 2),
            decl(Shape::matrix(1, 1), EtaDtype::F32, HostRole::Writer, 2),
            ChannelDecl {
                seeded: true,
                ..decl(Shape::matrix(1, WIDE), EtaDtype::I32, HostRole::None, 1)
            },
            decl(Shape::matrix(1, WIDE), EtaDtype::I32, HostRole::Reader, 2),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                // Doubled on the device, so a pass whose tables are gone
                // shows: its puts would be the heap's stale words.
                ids(),
                Op::Add(0, 0),
                Op::ChanPut { chan: 0, value: 1 },
                Op::Output { index: 1, lane: 0, shape: Shape::matrix(LAYERS, WIDTH), dtype: EtaDtype::F32 },
                Op::ChanPut { chan: 1, value: 2 },
                Op::ChanTake(3),
                Op::Add(3, 3),
                Op::ChanPut { chan: 4, value: 4 },
                ids(),
                Op::Broadcast { value: 5, shape: Shape::matrix(1, WIDE) },
                Op::ChanPut { chan: 3, value: 6 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// What the seeded `last` holds before the first pass.
const SEED: i32 = -5;

/// The ids `last` carries: 64 KiB of them, what the device keeps between
/// passes without a deferral.
const WIDE: u32 = 16 << 10;

struct Bound {
    engine: backend::Engine,
    instance: InstanceId,
    channels: Vec<ChannelId>,
    t_input: u32,
    word: u64,
    /// The argmax of each embedding row.
    argmax: Vec<i32>,
}

impl Bound {
    /// One pass reading token `token` last: (`ids`, `seen`).
    fn pass(&mut self, token: u32) -> Result<(i32, i32), String> {
        let lane = Lane {
            slot: 0,
            word: self.word,
            tokens: vec![3, 11, token],
            readout: Readout::Last,
            kv_less: true,
            inputs: vec![InputFeed { id: self.t_input, channel: self.channels[2] }],
            ..Lane::default()
        };
        let attachment =
            Attachment { lane: 0, lanes: 1, instance: self.instance, adapter_only: false, scratch: ::engine::ScratchPlace::None };
        let step = Step { lanes: vec![lane], attachments: vec![attachment], media: Vec::new(), voxels: Vec::new(), shape: None };
        let mut ticket = self.engine.submit(&FrameSubmission::of(step)).map_err(|e| e.to_string())?;
        self.engine.settle_frame(&mut ticket).map_err(|e| e.to_string())?;
        let mut take = |dense: u32| -> Result<i32, String> {
            let cell = engine::Channels::take(&mut self.engine, self.instance, dense)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("channel {dense} holds no cell"))?;
            let first = i32::from_le_bytes(cell[..4].try_into().expect("an i32"));
            if !cell.as_chunks::<4>().0.iter().all(|word| i32::from_le_bytes(*word) == first) {
                return Err(format!("channel {dense}'s cell holds unlike ids"));
            }
            Ok(first)
        };
        let got = (take(0)?, take(4)?);
        // The stacked rows are another test's: drained so the ring has room.
        engine::Channels::take(&mut self.engine, self.instance, 1).map_err(|e| e.to_string())?;
        Ok(got)
    }
}

#[test]
fn an_instance_in_pool_slots_fires_through_moves() {
    let Some(Loaded { mut engine, caps, trace, planes, dir }) = loaded_with("roles", &Outputs, Some(64 << 20), crate::gpu::device_boot()) else {
        return;
    };
    let argmax: Vec<i32> = (0..VOCAB as usize)
        .map(|token| {
            let row = &planes["embed"].1[token * WIDTH as usize..][..WIDTH as usize];
            (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b])).expect("a row") as i32
        })
        .collect();
    let t_input = trace.inputs.iter().position(|input| input.name == "t").expect("the plan reads `t`") as u32;
    let chunks = caps.caps.chunks.clone();
    let top = chunks.segment_chunks;
    assert!(top >= 10 && chunks.host_chunks >= 2, "segment 0 and the host tier hold the test's chunks");
    // The guests' chunks, near the top of the committed segment: tables,
    // cells, two blank ones, two the evacuation moves onto, and the two an
    // unpark lands in.
    let [tables, cells, blank, blank_cells, onto, onto_cells, fresh_tables, fresh_cells] =
        [1, 2, 3, 4, 5, 6, 7, 8].map(|k| top - k);
    let mut slots = Slots::of(&chunks, tables, cells);
    let instance_shape = slots.instance.expect("an Instance shape");
    let channel_shape = slots
        .channel
        .expect("every backend keeps a channel's cells on the device (`engine_gpu::guest::defer`)");
    let program = engine::Programs::register(&mut engine, &registration_of(&output_decls(&caps), program(), 0x5107))
        .expect("the program registers");
    // Channel 3 (`last`) is the instance's own; the rest are registered.
    let channels: Vec<ChannelId> = (1..=5).map(ChannelId).collect();
    let registered: Vec<ChannelId> = [0usize, 1, 2, 4].map(|k| channels[k]).to_vec();
    for (k, (shape, dtype, host_role)) in [
        (vec![1, 1], EtaDtype::I32, HostRole::Reader),
        (vec![LAYERS, WIDTH], EtaDtype::F32, HostRole::Reader),
        (vec![1, 1], EtaDtype::F32, HostRole::Writer),
        (vec![1, WIDE], EtaDtype::I32, HostRole::Reader),
    ]
    .into_iter()
    .enumerate()
    {
        slots.register(&mut engine, channel(registered[k], shape, dtype, host_role, 2));
    }
    let binding = slots.binding(
        &engine,
        InstanceBinding {
            program,
            channels: channels.clone(),
            seeds: vec![ChannelSeed { channel: 3, bytes: SEED.to_le_bytes().repeat(WIDE as usize) }],
            geometry: GeometryClass::Host,
            extents: super::outputs::EXTENTS,
            ..InstanceBinding::default()
        },
        &registered,
    );
    assert!(!binding.instance_slots.is_empty() && !binding.ring_slots.is_empty(), "the tables and the kept cell take slots");
    let instance = engine::Programs::bind(&mut engine, &binding).expect("the instance binds").id;
    assert!(
        engine::Channels::publish(&mut engine, instance, 2, &super::outputs::T.to_le_bytes()).expect("the timestep publishes"),
        "the ring had room"
    );
    let mut bound = Bound {
        engine,
        instance,
        channels,
        t_input,
        word: trace.facts.word(&Request::new(super::outputs::ROWS, false)),
        argmax,
    };
    // Rounds read distinct tokens whose argmax differ from each other and
    // from zero, so a stale or blank read shows.
    let mut tokens: Vec<u32> = Vec::new();
    for token in 0..VOCAB {
        let id = bound.argmax[token as usize];
        if id != 0 && tokens.iter().all(|&p| bound.argmax[p as usize] != id) {
            tokens.push(token);
        }
    }
    assert!(tokens.len() >= 4, "enough distinct rows: {tokens:?}");
    // `last` carries the previous pass's ids, on the device.
    let mut last = SEED;
    let mut round = |bound: &mut Bound, token: u32, what: &str| {
        let got = bound.pass(token).unwrap_or_else(|why| panic!("{what}: the pass fires: {why}"));
        let want = bound.argmax[token as usize];
        assert_eq!(got, (2 * want, 2 * last), "{what}: (ids, seen), each doubled");
        last = want;
    };
    round(&mut bound, tokens[0], "first pass");
    round(&mut bound, tokens[1], "second pass");

    // Each guest chunk onto another, and a blank chunk onto the place each
    // left: what the guest bound there is gone. The engine rebinds as it
    // evacuates.
    engine::Pool::evacuate(&mut bound.engine, &Evacuation {
        moves: vec![(tables, onto), (cells, onto_cells), (blank, tables), (blank_cells, cells)],
    })
    .expect("the evacuation");
    round(&mut bound, tokens[2], "after an evacuation");

    // Park and unpark: each slot copied to the host tier, its device place
    // blanked, copied back into fresh refs, and rebound there.
    let device = <backend::Api as engine_gpu::api::Host>::domain(bound.engine.boot());
    let held: [(ShapeId, u32, &Vec<u32>, u32, u32); 2] = [
        (instance_shape.id, instance_shape.per_chunk, &binding.instance_slots, blank, fresh_tables),
        (channel_shape.id, channel_shape.per_chunk, &binding.ring_slots[0].slots, blank_cells, fresh_cells),
    ];
    let mut fresh: Vec<Vec<u32>> = Vec::new();
    for (k, &(shape, per, refs, blank, chunk)) in held.iter().enumerate() {
        // Host chunk 0 holds the tables, 1 the cells.
        let host: Vec<u32> = (0..refs.len() as u32).map(|i| k as u32 * per + i).collect();
        let back: Vec<u32> = (0..refs.len() as u32).map(|i| chunk * per + i).collect();
        let blanks = vec![blank * per; refs.len()];
        for (src, dst, from, to) in [
            (device, MemoryDomain::HostPinned, refs.as_slice(), &host),
            (device, device, blanks.as_slice(), refs),
            (MemoryDomain::HostPinned, device, host.as_slice(), &back),
        ] {
            engine::Tiers::copy(&mut bound.engine, &SlotCopy {
                shape,
                src,
                dst,
                slots: from.iter().copied().zip(to.iter().copied()).collect(),
                runs: Vec::new(),
            })
            .expect("the slots copy");
        }
        fresh.push(back);
    }
    engine::Programs::rebind(&mut bound.engine, Rebind::Instance {
        id: bound.instance,
        instance: fresh[0].clone(),
        rings: vec![RingSlots { channel: binding.ring_slots[0].channel, slots: fresh[1].clone() }],
        adapter: Vec::new(),
        // The guest's state comes back where it lay (`guest_state` moves it
        // to fresh room).
        scratch: binding.scratch.clone(),
        pair: binding.pair.clone(),
        pair_generation: binding.pair_generation,
    })
    .expect("the unparked slots rebind");
    round(&mut bound, tokens[3], "after a park and unpark");
    round(&mut bound, tokens[0], "a pass later");
    assert_eq!(
        bound.engine.live().tallied(engine::live::Tally::BuiltInFire),
        0,
        "a fire through pool slots and their moves built a pipeline the load did not"
    );
    drop(bound);
    let _ = std::fs::remove_dir_all(&dir);
}
