//! A guest program's state lies in its instance's and its pair's places in
//! the pool (`Engine::guest_state`, `engine_gpu::guest::state`), never in a
//! fire's span: the engine states no fire term for it, so 64 bound lanes
//! fire in the rung's span as it stood before any bound; and a park and an
//! unpark move the state to fresh room, nothing copied, the passes after
//! reading what they read before.

use engine::ChannelId;
use engine::Engine;
use engine::channel::ChannelRegistration;
use engine::chunk::Evacuation;
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step};
use engine::program::{BindExtents, InstanceBinding, InstanceId, Rebind, ScratchPlace};
use engine_gpu::backend::Buffer as _;
use eta_ir::container::{ChanDType, HostRole};
use eta_ir::registry::GeometryClass;
use eta_ir::types::Dtype as EtaDtype;
use poem::Request;

use super::outputs::{
    LAYERS, Loaded, Outputs, ROWS, T, WIDTH, epilogue, loaded_bound, output_decls,
    registration_of,
};
use super::slots::Slots;
use crate::backend;

const EXTENTS: BindExtents = BindExtents {
    row_count: 1,
    token_count: ROWS,
    sampled_rows: 1,
    query_len: ROWS,
    key_len: 1,
};

/// Binds an instance of `program` over three channels of its own (`ids`
/// and `stack` the host reads, `t` it writes) from id `first`, and
/// publishes `t`.
fn bind(
    engine: &mut backend::Engine,
    slots: &mut Slots,
    program: engine::program::ProgramId,
    first: u64,
) -> (InstanceId, Vec<ChannelId>, InstanceBinding) {
    bind_in(engine, slots, program, first, None)
}

/// [`bind`], its tables and state in `room`'s slots where given: the slots
/// another instance held, handed on as the runtime hands freed slots on.
fn bind_in(
    engine: &mut backend::Engine,
    slots: &mut Slots,
    program: engine::program::ProgramId,
    first: u64,
    room: Option<&InstanceBinding>,
) -> (InstanceId, Vec<ChannelId>, InstanceBinding) {
    try_bind_in(engine, slots, program, first, room).expect("the instance binds")
}

/// [`bind_in`], its refusal returned.
fn try_bind_in(
    engine: &mut backend::Engine,
    slots: &mut Slots,
    program: engine::program::ProgramId,
    first: u64,
    room: Option<&InstanceBinding>,
) -> engine::Result<(InstanceId, Vec<ChannelId>, InstanceBinding)> {
    let channels: Vec<ChannelId> = [
        (vec![1, 1], EtaDtype::I32, HostRole::Reader),
        (vec![LAYERS, WIDTH], EtaDtype::F32, HostRole::Reader),
        (vec![1, 1], EtaDtype::F32, HostRole::Writer),
    ]
    .into_iter()
    .enumerate()
    .map(|(at, (shape, dtype, host_role))| {
        let id = ChannelId(first + at as u64);
        slots.register(
            engine,
            ChannelRegistration {
                id,
                shape,
                dtype: ChanDType::Concrete(dtype),
                host_role,
                seeded: false,
                extern_dir: None,
                capacity: 2,
                extern_name: Vec::new(),
                slots: Vec::new(),
            },
        );
        id
    })
    .collect();
    let binding = slots.binding(
        engine,
        InstanceBinding {
            program,
            channels: channels.clone(),
            seeds: Vec::new(),
            geometry: GeometryClass::Host,
            extents: EXTENTS,
            ..InstanceBinding::default()
        },
        &channels,
    );
    let binding = match room {
        Some(room) => InstanceBinding {
            instance_slots: room.instance_slots.clone(),
            scratch: room.scratch.clone(),
            pair: room.pair.clone(),
            pair_generation: room.pair_generation,
            ..binding
        },
        None => binding,
    };
    let instance = engine.bind_instance(&binding)?.id;
    assert!(
        engine
            .publish_channel(instance, 2, &T.to_le_bytes())
            .expect("the timestep publishes"),
        "the ring had room"
    );
    Ok((instance, channels, binding))
}

/// One step: lane `k` reads token `tokens[k]` last and feeds instance
/// `bound[k]`. Each instance's `ids`, and `t` published again for the next.
fn fire(
    engine: &mut backend::Engine,
    word: u64,
    t_input: u32,
    bound: &[(InstanceId, Vec<ChannelId>)],
    tokens: &[u32],
) -> Vec<i32> {
    let lanes = bound
        .iter()
        .zip(tokens)
        .enumerate()
        .map(|(k, ((_, channels), &token))| Lane {
            slot: k as u32,
            word,
            tokens: vec![3, 11, token],
            readout: Readout::Last,
            kv_less: true,
            inputs: vec![InputFeed {
                id: t_input,
                channel: channels[2],
            }],
            ..Lane::default()
        })
        .collect();
    let attachments = bound
        .iter()
        .enumerate()
        .map(|(k, &(instance, _))| Attachment {
            lane: k as u32,
            lanes: 1,
            instance,
            adapter_only: false,
            scratch: ::engine::ScratchPlace::None,
        })
        .collect();
    let mut ticket = engine
        .submit(&FrameSubmission::of(Step {
            lanes,
            attachments,
            media: Vec::new(),
            voxels: Vec::new(),
        }))
        .expect("the frame fires");
    engine.settle_frame(&mut ticket).expect("the frame settles");
    bound
        .iter()
        .map(|&(instance, _)| {
            let ids = engine
                .take_channel(instance, 0)
                .expect("the channel reads")
                .expect("the channel holds a cell");
            engine
                .take_channel(instance, 1)
                .expect("the channel reads")
                .expect("the channel holds a cell");
            engine
                .publish_channel(instance, 2, &T.to_le_bytes())
                .expect("the timestep publishes");
            i32::from_le_bytes(ids.try_into().expect("one i32"))
        })
        .collect()
}

/// The argmax of embedding row `token`.
fn argmax(planes: &std::collections::BTreeMap<String, (Vec<u64>, Vec<f32>)>, token: u32) -> i32 {
    let row = &planes["embed"].1[(token * WIDTH) as usize..][..WIDTH as usize];
    (0..row.len())
        .max_by(|&a, &b| row[a].total_cmp(&row[b]))
        .expect("a row") as i32
}

/// The loaded miniature, its epilogue registered, and the slots a test's
/// guests take.
fn setup(tag: &str, lanes: u32) -> Option<(Loaded, engine::program::ProgramId, Slots, u32)> {
    setup_bound(tag, lanes, None)
}

/// [`setup`] on a device whose platform states `bound` on one GPU
/// submission ([`loaded_bound`]).
fn setup_bound(
    tag: &str,
    lanes: u32,
    bound: Option<std::time::Duration>,
) -> Option<(Loaded, engine::program::ProgramId, Slots, u32)> {
    let mut loaded = loaded_bound(tag, &Outputs, lanes, bound)?;
    let decls = output_decls(&loaded.caps);
    let program = loaded
        .engine
        .register_program(&registration_of(&decls, epilogue(), 0x0e7))
        .expect("the program registers");
    let chunks = &loaded.caps.caps.chunks;
    let slots = Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let t_input = loaded
        .trace
        .inputs
        .iter()
        .position(|input| input.name == "t")
        .expect("the plan reads `t`") as u32;
    Some((loaded, program, slots, t_input))
}

/// A bound program with guest state states no fire term: its state is its
/// instances' and its pair's, in `Instance` slots. 64 instances bound, the
/// span's rungs read as before any bound, and one fire of 64 lanes, each
/// its own instance's, serves every lane its own row: on Vulkan through
/// the one heap the pair's passes share, on wgpu through each instance's.
/// The binds and the passes reserve nothing on the device or the host: a
/// session's status and step records lie in its slots, and a pass stages
/// and reads back through room the load laid.
#[test]
fn a_bound_guest_states_no_fire_term_and_64_lanes_fire_in_the_rungs_span() {
    guest_states_no_fire_term_and_64_lanes_fire_in_the_rungs_span("guest-state-64", None);
}

/// [`a_bound_guest_states_no_fire_term_and_64_lanes_fire_in_the_rungs_span`]
/// on a device whose platform states a bound on one GPU submission (KGSL's
/// 2 s on Android, Windows' TDR delay): every serving step is estimated,
/// timed into its arm's timestamps, and a frame after an idle device
/// re-times the clock by the probe first (`engine_gpu::submit`). Every one
/// of them writes what the load laid: a fire reserves nothing still.
#[test]
fn under_a_stated_submit_bound_64_guest_lanes_fire_reserving_nothing() {
    guest_states_no_fire_term_and_64_lanes_fire_in_the_rungs_span(
        "guest-state-64-bound",
        Some(std::time::Duration::from_secs(2)),
    );
}

fn guest_states_no_fire_term_and_64_lanes_fire_in_the_rungs_span(
    tag: &str,
    submit_bound: Option<std::time::Duration>,
) {
    const LANES: u32 = 64;
    let Some((loaded, program, mut slots, t_input)) = setup_bound(tag, LANES, submit_bound) else {
        return;
    };
    // The bound stated is the one the load holds its steps to: the boot's
    // where it states one, else the platform's own (Windows' TDR, Android's
    // driver timeout; none elsewhere).
    let stated = submit_bound.or_else(engine_gpu::rates::stated_submit_bound);
    assert_eq!(
        loaded
            .engine
            .loaded()
            .expect("loaded")
            .submit_bound()
            .map(|held| held.bound_ns),
        stated.map(|b| b.as_nanos() as f64),
        "the load holds its steps to the bound its boot states"
    );
    let Loaded {
        mut engine,
        trace,
        planes,
        dir,
        ..
    } = loaded;
    assert_eq!(
        engine
            .fire_facts(program, &EXTENTS)
            .expect("the program's fire facts"),
        engine::FireFacts::default(),
        "a guest's state lies in no fire's span"
    );
    let state = engine
        .guest_state(program, &EXTENTS, GeometryClass::Host)
        .expect("the program's guest state");
    eprintln!("guest state: {state:?}");
    assert!(
        state.instance_bytes() + state.per_pair > 0,
        "the epilogue's heap is state of its instance or its pair"
    );
    let span = |engine: &backend::Engine| {
        engine
            .loaded()
            .expect("loaded")
            .pools()
            .chunk_facts()
            .act_chunks
    };
    let before = span(&engine);
    // Every buffer a guest session uses was laid at the load: binding 64
    // and firing them reserves nothing.
    let takes = |engine: &backend::Engine| {
        engine_gpu::backend::Context::books(engine.loaded().expect("loaded").device()).takes()
    };
    let loaded_takes = takes(&engine);

    let bound: Vec<(InstanceId, Vec<ChannelId>)> = (0..u64::from(LANES))
        .map(|k| {
            let (instance, channels, _) = bind(&mut engine, &mut slots, program, 1 + 3 * k);
            (instance, channels)
        })
        .collect();
    assert_eq!(span(&engine), before, "64 bound guests leave the span's rungs as they were");
    assert_eq!(takes(&engine), loaded_takes, "64 guest sessions bind with nothing reserved");

    let word = trace.facts.word(&Request::new(ROWS, false));
    // The first fire of 64 lanes may grow the load's readout seat, which
    // is not a guest's; the fire after it reserves nothing.
    let mut fired_takes = loaded_takes;
    for round in 0..2u32 {
        if round == 1 {
            fired_takes = takes(&engine);
        }
        let tokens: Vec<u32> = (0..LANES).map(|k| (k + round) % super::outputs::VOCAB).collect();
        let ids = fire(&mut engine, word, t_input, &bound, &tokens);
        let want: Vec<i32> = tokens.iter().map(|&token| argmax(&planes, token)).collect();
        assert_eq!(ids, want, "round {round}: each of 64 lanes reads its own row");
    }
    assert_eq!(span(&engine), before, "the fires leave the span's rungs as they were");
    assert_eq!(
        takes(&engine),
        fired_takes,
        "a fire of 64 guest sessions' passes reserves nothing"
    );
    // Under the bound, each frame here follows an idle device (each waited
    // for), so the probe re-timed it: into what the load laid too.
    let probed = engine.live().tallied(engine::live::Tally::Probed);
    assert_eq!(
        probed > 0,
        submit_bound.is_some(),
        "the probe re-times a frame after idle where a bound is stated, and only there: {probed}"
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A park and an unpark of an instance with guest state: the runtime gives
/// its state and its pair's back and takes fresh room, copying nothing, and
/// tells the engine where (`Rebind::Instance`). The fresh room holds
/// garbage and the old room is overwritten after the move: the passes after
/// read what they read before.
#[test]
fn an_unparked_guest_serves_on_from_fresh_room_with_nothing_copied() {
    let Some((loaded, program, mut slots, t_input)) = setup("guest-state-park", 4) else {
        return;
    };
    let Loaded {
        mut engine,
        trace,
        planes,
        dir,
        ..
    } = loaded;
    let (instance, channels, binding) = bind(&mut engine, &mut slots, program, 1);
    let bound = [(instance, channels)];
    let word = trace.facts.word(&Request::new(ROWS, false));
    let ids = fire(&mut engine, word, t_input, &bound, &[5]);
    assert_eq!(ids, [argmax(&planes, 5)], "the first pass reads its row");

    let state = engine
        .guest_state(program, &EXTENTS, GeometryClass::Host)
        .expect("the program's guest state");
    let scratch = slots.state(state.instance_bytes(), state.run);
    let pair = slots.state(state.per_pair, state.run);
    // Bytes `bytes` of `place` filled with `fill`, where it holds any.
    let fill = |engine: &backend::Engine, place: &ScratchPlace, bytes: u64, fill: u8| {
        if bytes == 0 {
            return;
        }
        let pools = engine.loaded().expect("loaded").pools();
        let (buffer, at) = match place {
            ScratchPlace::None => return,
            ScratchPlace::Run { first, .. } => pools
                .slot_place(engine_gpu::store::layout::INSTANCE, *first)
                .expect("the run's chunk"),
            ScratchPlace::Chunks { pieces, .. } => {
                pools.chunk_place(pieces[0]).expect("the run's chunks")
            }
        };
        buffer
            .write_shared(at, &vec![fill; bytes as usize])
            .expect("the room fills");
    };
    // Fresh room is dead: whatever lies there.
    fill(&engine, &scratch, state.instance_bytes(), 0x7f);
    fill(&engine, &pair, state.per_pair, 0x7f);
    engine
        .rebind_slots(Rebind::Instance {
            id: instance,
            instance: binding.instance_slots.clone(),
            rings: Vec::new(),
            adapter: Vec::new(),
            scratch: scratch.clone(),
            pair: pair.clone(),
            pair_generation: 0,
        })
        .expect("the unpark rebinds its state");
    // The old room is someone else's now.
    fill(&engine, &binding.scratch, state.instance_bytes(), 0x3c);
    fill(&engine, &binding.pair, state.per_pair, 0x3c);
    for token in [5, 9, 2] {
        let ids = fire(&mut engine, word, t_input, &bound, &[token]);
        assert_eq!(
            ids,
            [argmax(&planes, token)],
            "a pass after the unpark reads token {token}'s row"
        );
    }
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}

/// An evacuation under a kept park (`Rebind::Parked { kept: true }`):
/// instance A parks with its tables kept in place, and the chunk holding
/// them moves onto a free one. A's claim moves with them: an instance bound
/// at A's own refs (now the new bytes) is refused by name, and one bound in
/// the chunk now lying on A's old bytes binds and serves. A's unpark
/// rebinds it at the same refs and it reads its tables as before the move.
/// Unmoved, A's claim would stand on the old bytes: the first bind would
/// pass over A's tables and the second be refused on a stale claim.
#[test]
fn an_evacuation_moves_a_kept_park_s_claim_and_tables() {
    let Some((loaded, program, mut slots, t_input)) = setup("guest-state-kept-evac", 4) else {
        return;
    };
    let chunk_bytes = loaded.caps.caps.chunks.chunk_bytes;
    let Loaded {
        mut engine,
        trace,
        planes,
        dir,
        ..
    } = loaded;
    let state = engine
        .guest_state(program, &EXTENTS, GeometryClass::Host)
        .expect("the program's guest state");
    let word = trace.facts.word(&Request::new(ROWS, false));
    let per_chunk = slots.instance.expect("an Instance shape").per_chunk;
    // Room an instance's state and its pair's take, nobody's yet.
    let fresh = |slots: &mut Slots, pair_generation: u64, base: &InstanceBinding| InstanceBinding {
        scratch: slots.state(state.instance_bytes(), state.run),
        pair: slots.state(state.per_pair, state.run),
        pair_generation,
        ..base.clone()
    };

    let (a, a_channels, a_binding) = bind(&mut engine, &mut slots, program, 1);
    let a_bound = [(a, a_channels)];
    let ids = fire(&mut engine, word, t_input, &a_bound, &[5]);
    assert_eq!(ids, [argmax(&planes, 5)], "A's pass reads its row");

    engine
        .rebind_slots(Rebind::Parked { id: a, kept: true })
        .expect("A parks, its tables kept in place");
    // A's tables' chunk onto a chunk nothing holds: the chunk that lay
    // there takes the bytes A's tables left.
    let tables = a_binding.instance_slots[0] / per_chunk;
    let ScratchPlace::Chunks { pieces, .. } = slots.state(chunk_bytes, 1) else {
        panic!("a whole chunk of room");
    };
    let onto = pieces[0];
    assert_ne!(tables, onto);
    engine
        .evacuate(&Evacuation {
            moves: vec![(tables, onto)],
        })
        .expect("the chunk moves under the kept park");

    // At A's own refs, the new bytes: A's.
    let at_a = fresh(&mut slots, 4, &a_binding);
    let why = try_bind_in(&mut engine, &mut slots, program, 4, Some(&at_a))
        .expect_err("an instance at A's refs overlaps A's kept tables where they moved");
    assert!(
        why.to_string().contains("would overlap"),
        "refused by name as an overlap: {why}"
    );
    // In the chunk now lying on A's old bytes: free of A.
    let displaced = InstanceBinding {
        instance_slots: a_binding
            .instance_slots
            .iter()
            .map(|&slot| onto * per_chunk + slot % per_chunk)
            .collect(),
        ..fresh(&mut slots, 5, &a_binding)
    };
    let (b, b_channels, _) = try_bind_in(&mut engine, &mut slots, program, 7, Some(&displaced))
        .expect("B binds on the bytes A's tables left, no stale claim");
    let ids = fire(&mut engine, word, t_input, &[(b, b_channels)], &[9]);
    assert_eq!(ids, [argmax(&planes, 9)], "B serves from A's old bytes");

    let unparked = fresh(&mut slots, 6, &a_binding);
    engine
        .rebind_slots(Rebind::Instance {
            id: a,
            instance: a_binding.instance_slots.clone(),
            rings: Vec::new(),
            adapter: Vec::new(),
            scratch: unparked.scratch,
            pair: unparked.pair,
            pair_generation: unparked.pair_generation,
        })
        .expect("A's unpark claims its moved tables");
    for token in [5, 2] {
        let ids = fire(&mut engine, word, t_input, &a_bound, &[token]);
        assert_eq!(
            ids,
            [argmax(&planes, token)],
            "A reads its tables as before the move: token {token}'s row"
        );
    }
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Where `place` lies in the pool: its segment buffer and first byte.
fn located(
    engine: &backend::Engine,
    place: &ScratchPlace,
) -> Option<(backend::Buffer, u64)> {
    let pools = engine.loaded().expect("loaded").pools();
    let (buffer, at) = match place {
        ScratchPlace::None => return None,
        ScratchPlace::Run { first, .. } => pools
            .slot_place(engine_gpu::store::layout::INSTANCE, *first)
            .expect("the run's chunk"),
        ScratchPlace::Chunks { pieces, .. } => pools.chunk_place(pieces[0]).expect("the run's chunks"),
    };
    Some((buffer.clone(), at))
}

/// `bytes` of `place` filled with `fill`.
fn fill_room(engine: &backend::Engine, place: &ScratchPlace, bytes: u64, fill: u8) {
    if let (Some((buffer, at)), true) = (located(engine, place), bytes > 0) {
        buffer
            .write_shared(at, &vec![fill; bytes as usize])
            .expect("the room fills");
    }
}

/// `bytes` of `place` as they stand.
/// Fills `bytes` of `place` from its byte `from` with `fill`.
fn fill_room_from(engine: &backend::Engine, place: &ScratchPlace, from: u64, bytes: u64, fill: u8) {
    if let (Some((buffer, at)), true) = (located(engine, place), bytes > 0) {
        buffer
            .write_shared(at + from, &vec![fill; bytes as usize])
            .expect("the room fills");
    }
}

fn read_room(engine: &backend::Engine, place: &ScratchPlace, bytes: u64) -> Vec<u8> {
    let mut out = vec![0u8; bytes as usize];
    if let (Some((buffer, at)), true) = (located(engine, place), bytes > 0) {
        buffer.read(at, &mut out).expect("the room reads");
    }
    out
}

/// A guest's state changes owner clean: guest A runs over its heap, which
/// holds A's bytes when A closes; its slots are handed on to guest B, and
/// B's heap and pair room read zero, never A's bytes. A third instance on
/// B's live pair leaves the pair as it stands (a sibling's score rows), and
/// an unpark's fresh room reads zero though garbage lay there.
#[test]
fn a_guests_state_reads_zero_when_its_slots_change_owner() {
    let Some((loaded, program, mut slots, t_input)) = setup("guest-state-owner", 4) else {
        return;
    };
    let Loaded {
        mut engine,
        trace,
        planes,
        dir,
        ..
    } = loaded;
    let state = engine
        .guest_state(program, &EXTENTS, GeometryClass::Host)
        .expect("the program's guest state");
    assert!(state.instance_bytes() > 0, "the epilogue's heap is its instance's");
    let word = trace.facts.word(&Request::new(ROWS, false));

    let (a, a_channels, a_binding) = bind(&mut engine, &mut slots, program, 1);
    let ids = fire(&mut engine, word, t_input, &[(a, a_channels)], &[5]);
    assert_eq!(ids, [argmax(&planes, 5)], "A's pass reads its row");
    // Whatever A's launches left, A's bytes are in its room as it closes.
    fill_room(&engine, &a_binding.scratch, state.instance_bytes(), 0xa5);
    fill_room(&engine, &a_binding.pair, state.per_pair, 0xa5);
    engine.close_instance(a).expect("A closes");

    let (b, b_channels, b_binding) =
        bind_in(&mut engine, &mut slots, program, 4, Some(&a_binding));
    assert_eq!(b_binding.scratch, a_binding.scratch, "B takes A's room");
    assert!(
        read_room(&engine, &b_binding.scratch, state.instance_bytes())
            .iter()
            .all(|&byte| byte == 0),
        "B's heap holds none of A's bytes"
    );
    assert!(
        read_room(&engine, &b_binding.pair, state.per_pair)
            .iter()
            .all(|&byte| byte == 0),
        "B's pair room holds none of A's bytes"
    );
    let ids = fire(&mut engine, word, t_input, &[(b, b_channels.clone())], &[9]);
    assert_eq!(ids, [argmax(&planes, 9)], "B serves from its cleared room");

    // A sibling on B's live pair: the pair is not cleared under B.
    if state.per_pair > 0 {
        fill_room(&engine, &b_binding.pair, state.per_pair, 0x11);
        let sibling = InstanceBinding {
            scratch: slots.state(state.instance_bytes(), state.run),
            ..b_binding.clone()
        };
        let (c, _, _) = bind_in(&mut engine, &mut slots, program, 7, Some(&sibling));
        assert!(
            read_room(&engine, &b_binding.pair, state.per_pair)
                .iter()
                .all(|&byte| byte == 0x11),
            "a bind on a live taking leaves the pair as it stands"
        );
        engine.close_instance(c).expect("C closes");
    }

    // An unpark: fresh room holding garbage reads zero once B is there.
    let scratch = slots.state(state.instance_bytes(), state.run);
    let pair = slots.state(state.per_pair, state.run);
    fill_room(&engine, &scratch, state.instance_bytes(), 0x7f);
    fill_room(&engine, &pair, state.per_pair, 0x7f);
    engine
        .rebind_slots(Rebind::Instance {
            id: b,
            instance: b_binding.instance_slots.clone(),
            rings: Vec::new(),
            adapter: Vec::new(),
            scratch: scratch.clone(),
            pair: pair.clone(),
            pair_generation: 1,
        })
        .expect("the unpark rebinds its state");
    assert!(
        read_room(&engine, &scratch, state.instance_bytes())
            .iter()
            .all(|&byte| byte == 0),
        "the unparked heap holds none of the room's last bytes"
    );
    assert!(
        read_room(&engine, &pair, state.per_pair)
            .iter()
            .all(|&byte| byte == 0),
        "the unparked pair room holds none of the room's last bytes"
    );
    let ids = fire(&mut engine, word, t_input, &[(b, b_channels)], &[2]);
    assert_eq!(ids, [argmax(&planes, 2)], "B serves on after the unpark");
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Logical chunk `chunk`'s bytes as they stand, read through the chunk
/// table.
fn read_chunk(engine: &backend::Engine, chunk: u32) -> Vec<u8> {
    let pools = engine.loaded().expect("loaded").pools();
    let mut out = vec![0u8; pools.chunk_facts().chunk_bytes as usize];
    let (buffer, at) = pools.chunk_place(chunk).expect("the chunk's place");
    buffer.read(at, &mut out).expect("the chunk reads");
    out
}

/// `Engine::evacuate` is all or nothing. Instance A's state lies in a run
/// of two whole chunks, and an evacuation moves the run's head onto the
/// last chunk of segment 0, the chunk A's tables lie in: the pool takes
/// the move, and the rebind refuses it by name (the run would cross the
/// segment's end) after the copies were recorded. Nothing moved: the chunk
/// table, and the bytes of the run's head and of the tables' chunk, are as
/// they stood; A fires on from where it lay; and its claim stands (a park
/// and an unpark onto its own slots lay it again), after which it fires.
///
/// Negative control (run once on wgpu, reverted): with
/// `Shell::abort_evacuation` not restoring the table it replaced, this
/// failed at the chunk-table check (the refused move's two logical chunks
/// left swapped).
#[test]
fn an_evacuation_refused_at_its_rebind_moves_nothing() {
    let Some((loaded, program, mut slots, t_input)) = setup("guest-state-evacuate-refused", 4)
    else {
        return;
    };
    let Loaded {
        mut engine,
        trace,
        planes,
        dir,
        ..
    } = loaded;
    let state = engine
        .guest_state(program, &EXTENTS, GeometryClass::Host)
        .expect("the program's guest state");
    assert!(state.instance_bytes() > 0, "the epilogue's heap is its instance's");
    let word = trace.facts.word(&Request::new(ROWS, false));
    let top = engine
        .loaded()
        .expect("loaded")
        .pools()
        .chunk_facts()
        .segment_chunks;
    assert!(top >= 32, "segment 0 holds the test's chunks: {top}");
    let per = slots.instance.expect("an Instance shape").per_chunk;
    // A's tables lie in the slots' own chunk, segment 0's last; its state
    // in a run of two whole chunks well under every other state's.
    let tables = top - 1;
    let head = top - 24;

    let (a, a_channels, a_binding) = bind(&mut engine, &mut slots, program, 1);
    assert!(
        a_binding.instance_slots.iter().all(|&slot| slot / per == tables),
        "A's tables lie in chunk {tables}: {:?}",
        a_binding.instance_slots
    );
    for place in [&a_binding.scratch, &a_binding.pair] {
        if let ScratchPlace::Chunks { pieces, .. } = place {
            assert!(pieces[0] > head + 1, "a state's run above chunk {}: {place:?}", head + 1);
        }
    }
    let scratch = ScratchPlace::Chunks {
        run: 2,
        pieces: vec![head],
    };
    engine
        .rebind_slots(Rebind::Instance {
            id: a,
            instance: a_binding.instance_slots.clone(),
            rings: Vec::new(),
            adapter: Vec::new(),
            scratch: scratch.clone(),
            pair: a_binding.pair.clone(),
            pair_generation: a_binding.pair_generation,
        })
        .expect("A's state moves to a run of two chunks");
    let bound = [(a, a_channels.clone())];
    let ids = fire(&mut engine, word, t_input, &bound, &[5]);
    assert_eq!(ids, [argmax(&planes, 5)], "A reads its row from its run");

    let table = engine.chunk_table().expect("the engine's chunk table");
    let head_bytes = read_chunk(&engine, head);
    let table_bytes = read_chunk(&engine, tables);
    let onto = table.physical(tables).expect("the tables' chunk lies somewhere");
    let refused = engine
        .evacuate(&Evacuation {
            moves: vec![(head, onto)],
        })
        .expect_err("A's run would cross segment 0's end");
    eprintln!("refused: {refused}");
    assert!(
        matches!(&refused, engine::Error::Program(why) if why.contains("one segment")),
        "refused at the rebind, by the run's place: {refused}"
    );
    assert_eq!(
        engine.chunk_table().expect("the engine's chunk table"),
        table,
        "a refused evacuation leaves the chunk table as it stood"
    );
    assert!(
        read_chunk(&engine, tables) == table_bytes,
        "a refused evacuation copies nothing over A's tables"
    );
    assert!(
        read_chunk(&engine, head) == head_bytes,
        "a refused evacuation leaves A's run as it stood"
    );
    let ids = fire(&mut engine, word, t_input, &bound, &[2]);
    assert_eq!(ids, [argmax(&planes, 2)], "A fires on from where it lay");

    // A's claim stood: a park and an unpark onto its own slots lay it
    // again.
    engine
        .rebind_slots(Rebind::Parked { id: a, kept: true })
        .expect("A parks, its tables kept in place");
    engine
        .rebind_slots(Rebind::Instance {
            id: a,
            instance: a_binding.instance_slots.clone(),
            rings: Vec::new(),
            adapter: Vec::new(),
            scratch,
            pair: a_binding.pair.clone(),
            pair_generation: a_binding.pair_generation,
        })
        .expect("A unparks onto its own slots");
    let ids = fire(&mut engine, word, t_input, &bound, &[3]);
    assert_eq!(ids, [argmax(&planes, 3)], "A fires on after the refused evacuation");
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The miniature whose pair keeps state: Qwen3's, its fires capturing
/// attention scores into the score room of the pair a lane's program reads
/// them through (`guest::state`).
const SCORES_MODEL: &str = "qwen3-micro-text";
const SCORE_KEYS: &str = "attn.scores";
/// A pass's tokens on the scores miniature.
const SCORE_ROWS: u32 = 4;
const SCORE_EXTENTS: BindExtents = BindExtents {
    row_count: 1,
    token_count: SCORE_ROWS,
    sampled_rows: 1,
    query_len: SCORE_ROWS,
    key_len: 1,
};

/// The scores miniature loaded, a program that puts each pass's key row in
/// a channel the host reads registered, the test's chunks committed past
/// the load's, and the slots its guests take in them.
struct ScoresRig {
    engine: backend::Engine,
    trace: poem_ir::Trace,
    program: engine::program::ProgramId,
    slots: Slots,
    facts: engine::chunk::ChunkFacts,
    /// The key row's width: the load's context.
    width: u32,
}

fn scores_rig() -> Option<ScoresRig> {
    use engine::chunk::Growth;
    use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
    use eta_ir::container::{ChannelDecl, StageProgram, TraceContainer};
    use eta_ir::op::Op;
    use eta_ir::registry::Stage;
    use eta_ir::types::Shape;

    let boot = crate::gpu::device_boot();
    if backend::Context::bind(&boot, backend::ALLOWANCE).is_err() {
        eprintln!("not asked: no device");
        return None;
    }
    let fixture = crate::model_logits::fixture(SCORES_MODEL, None);
    assert!(
        fixture.trace.nodes.iter().any(|node| matches!(
            node.op,
            poem_ir::Operation::Attn(poem_ir::Attn::ScoreCapture { .. })
        )),
        "{SCORES_MODEL} captures scores"
    );
    let mut engine = backend::Engine::new(boot, super::outputs::contract_for);
    let loaded = engine
        .load(LoadRequest {
            trace: fixture.trace.clone(),
            checkpoint: Checkpoint::Path(fixture.container.clone()),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: 16,
                buckets: vec![16],
                max_adapters: 0,
                page_size: 16,
                max_context: 64,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency: Residency {
                device: crate::gpu::share(),
                ..Residency::default()
            },
            ordinal: 0,
            frames_in_flight: 1,
            steps: Default::default(),
        })
        .expect("the miniature loads");
    let decls = output_decls(&loaded);
    let decl = decls
        .iter()
        .find(|decl| decl.name == SCORE_KEYS)
        .unwrap_or_else(|| panic!("{SCORES_MODEL} declares `{SCORE_KEYS}`"));
    let rows = decl.layers.max(1);
    let width = decl.width;
    let program = TraceContainer {
        names: vec![SCORE_KEYS.to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::matrix(rows, width),
            dtype: ChanDType::Concrete(EtaDtype::F32),
            capacity: 2,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(rows, width),
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 0, value: 0 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    let program = engine
        .register_program(&registration_of(&decls, program, 0x5c0e))
        .expect("the program registers");

    // The test's chunks: past the seats' runs where the pool is laid whole
    // at its load, else in a segment committed past the load's.
    let facts = loaded.caps.chunks.clone();
    let segment = facts.segment_chunks;
    let need = 12;
    let first = if facts.segments.growth == Growth::Fixed {
        let seats = 1 + engine
            .loaded()
            .expect("the shell stands")
            .pool_layout()
            .seat_chunks();
        seats.div_ceil(segment) * segment
    } else {
        facts.segments.committed * segment
    };
    let commit = (first + need).div_ceil(segment);
    let most = if facts.segments.growth == Growth::Fixed {
        facts.segments.max - 1
    } else {
        facts.segments.max
    };
    assert!(most >= commit, "the test's chunks span {commit} segments of {most}");
    engine.commit_segments(commit).expect("the segments commit");
    let top = first + need;
    let slots = Slots::of(&facts, top - 1, top - 2);
    Some(ScoresRig {
        engine,
        trace: fixture.trace,
        program,
        slots,
        facts,
        width: rows * width,
    })
}

impl ScoresRig {
    /// An instance of the program over a channel of its own (id `channel`),
    /// its state and pair where `slots` takes them, or at `pair` (and its
    /// taking) where given.
    fn bind(
        &mut self,
        channel: u64,
        pair: Option<(&ScratchPlace, u64)>,
    ) -> (InstanceId, InstanceBinding) {
        let id = ChannelId(channel);
        self.slots.register(
            &mut self.engine,
            ChannelRegistration {
                id,
                shape: vec![1, self.width],
                dtype: ChanDType::Concrete(EtaDtype::F32),
                host_role: HostRole::Reader,
                seeded: false,
                extern_dir: None,
                capacity: 2,
                extern_name: Vec::new(),
                slots: Vec::new(),
            },
        );
        let binding = self.slots.binding(
            &self.engine,
            InstanceBinding {
                program: self.program,
                channels: vec![id],
                seeds: Vec::new(),
                geometry: GeometryClass::Host,
                extents: SCORE_EXTENTS,
                ..InstanceBinding::default()
            },
            &[id],
        );
        let binding = match pair {
            Some((pair, pair_generation)) => InstanceBinding {
                pair: pair.clone(),
                pair_generation,
                ..binding
            },
            None => binding,
        };
        let instance = self
            .engine
            .bind_instance(&binding)
            .expect("the instance binds")
            .id;
        (instance, binding)
    }

    /// One pass of `instance` over `tokens`, capturing its scores: the key
    /// row its program put.
    fn fire(&mut self, instance: InstanceId, tokens: &[u32]) -> Vec<f32> {
        let word = self.trace.facts.word(
            &Request::new(tokens.len() as u32, false).capturing_scores(true),
        );
        let mut ticket = self
            .engine
            .submit(&FrameSubmission::of(Step {
                lanes: vec![Lane {
                    slot: 0,
                    word,
                    tokens: tokens.to_vec(),
                    readout: Readout::Last,
                    kv_less: true,
                    ..Lane::default()
                }],
                attachments: vec![Attachment {
                    lane: 0,
                    lanes: 1,
                    instance,
                    adapter_only: false,
                    scratch: ScratchPlace::None,
                }],
                media: Vec::new(),
                voxels: Vec::new(),
            }))
            .expect("the frame fires");
        self.engine
            .settle_frame(&mut ticket)
            .expect("the frame settles");
        let cell = self
            .engine
            .take_channel(instance, 0)
            .expect("the channel reads")
            .expect("the program put the key row");
        let row: Vec<f32> = cell
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("an f32")))
            .collect();
        assert!(
            row.iter().all(|v| v.is_finite()) && row.iter().any(|&v| v != 0.0),
            "a key row of captured scores: {row:?}"
        );
        row
    }
}

/// The largest difference between two rows of one width.
fn row_apart(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "rows of one width");
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

/// A pair whose every member is parked while an evacuation moves its
/// state is laid where it lies now by an instance that joins it fresh
/// (engine-cuda's `a_pair_whose_members_are_all_parked_is_laid_again_where_
/// a_move_left_it`). Two instances of the scores program share a pair (its
/// state the score room their fires capture into) and fire; both park
/// (`Rebind::Parked`, `kept` as given); the pair's chunks move onto whole
/// chunks nothing holds, and the bytes they left are filled with garbage.
/// A third instance binds at the pair and its taking (a fresh joiner, no
/// member live) and fires: its key row is the first's, and the bytes the
/// pair left still hold the garbage (neither the bind's clear nor the
/// fire's capture went there). The two parked members come back at the
/// pair, now the joiner's live taking, and answer as before.
///
/// Each instance addresses the pair through its own session, which a bind
/// resolves from the pool as it stands (`guest::state::Placed::at`) and an
/// unpark retargets. Negative control (run once on Vulkan, reverted): a
/// bind that took a parked member's pre-move span for a pair at its place
/// and taking failed both variants here, at the joiner's check of the
/// vacated bytes (its clear and its captured rows landed on them).
fn a_fresh_joiner_of_an_all_parked_pair_lays_it_where_a_move_left_it(kept: bool) {
    let Some(mut rig) = scores_rig() else {
        return;
    };
    let state = rig
        .engine
        .guest_state(rig.program, &SCORE_EXTENTS, GeometryClass::Host)
        .expect("the program's guest state");
    assert!(state.per_pair > 0, "the pair keeps the score room");
    let chunk_bytes = rig.facts.chunk_bytes;
    let per_chunk = rig.slots.instance.expect("an Instance shape").per_chunk;
    let tokens: [Vec<u32>; 2] = [vec![3, 11, 29, 5], vec![7, 2, 41, 17]];

    let (a, a_binding) = rig.bind(1, None);
    let (b, b_binding) = rig.bind(2, None);
    assert_eq!(a_binding.pair, b_binding.pair, "A and B share the pair");
    assert_eq!(a_binding.pair_generation, b_binding.pair_generation);
    let pair = a_binding.pair.clone();
    let taking = a_binding.pair_generation;
    let want = [rig.fire(a, &tokens[0]), rig.fire(b, &tokens[1])];
    assert!(
        row_apart(&want[0], &want[1]) > 0.0,
        "the two passes' rows tell them apart"
    );

    for id in [a, b] {
        rig.engine
            .rebind_slots(Rebind::Parked { id, kept })
            .expect("the member parks");
    }
    // The pair's chunks onto whole chunks nothing holds: the chunks that
    // lay there take the bytes the pair left.
    let (from, run) = match &pair {
        &ScratchPlace::Run { first, .. } => (first / per_chunk, 1),
        ScratchPlace::Chunks { run, pieces } => (pieces[0], *run),
        ScratchPlace::None => unreachable!("the pair keeps state"),
    };
    let ScratchPlace::Chunks { pieces, .. } =
        rig.slots.state(u64::from(run) * chunk_bytes, run)
    else {
        panic!("a run of whole chunks");
    };
    let onto = pieces[0];
    assert!(
        onto + run <= from || from + run <= onto,
        "the run moves apart from itself"
    );
    rig.engine
        .evacuate(&Evacuation {
            moves: (0..run).map(|at| (from + at, onto + at)).collect(),
        })
        .expect("the pair's chunks move while every member is parked");
    let vacated = ScratchPlace::Chunks {
        run,
        pieces: vec![onto],
    };
    let vacated_bytes = u64::from(run) * chunk_bytes;
    fill_room(&rig.engine, &vacated, vacated_bytes, 0xee);

    // C joins the pair at its taking, no member live.
    let (c, _) = rig.bind(3, Some((&pair, taking)));
    let got = rig.fire(c, &tokens[0]);
    assert!(
        read_room(&rig.engine, &vacated, vacated_bytes)
            .iter()
            .all(|&byte| byte == 0xee),
        "the joiner's bind and fire leave the bytes the pair left as they stand"
    );
    let off = row_apart(&got, &want[0]);
    assert!(off < 1e-3, "the joiner's key row is A's: {off} apart");

    // A and B come back at the pair, the joiner's live taking.
    for (id, binding) in [(a, &a_binding), (b, &b_binding)] {
        let scratch = rig.slots.state(state.instance_bytes(), state.run);
        rig.engine
            .rebind_slots(Rebind::Instance {
                id,
                instance: binding.instance_slots.clone(),
                rings: Vec::new(),
                adapter: Vec::new(),
                scratch,
                pair: pair.clone(),
                pair_generation: taking,
            })
            .expect("the member comes back at the pair");
    }
    for (k, id) in [a, b].into_iter().enumerate() {
        let got = rig.fire(id, &tokens[k]);
        let off = row_apart(&got, &want[k]);
        assert!(off < 1e-3, "member {k} answers as before the move: {off} apart");
    }
    assert!(
        read_room(&rig.engine, &vacated, vacated_bytes)
            .iter()
            .all(|&byte| byte == 0xee),
        "the returning members leave the bytes the pair left as they stand"
    );
    for id in [a, b, c] {
        rig.engine.close_instance(id).expect("the instance closes");
    }
}

#[test]
fn a_fresh_joiner_of_a_pair_parked_kept_lays_it_where_a_move_left_it() {
    a_fresh_joiner_of_an_all_parked_pair_lays_it_where_a_move_left_it(true);
}

#[test]
fn a_fresh_joiner_of_a_pair_parked_shed_lays_it_where_a_move_left_it() {
    a_fresh_joiner_of_an_all_parked_pair_lays_it_where_a_move_left_it(false);
}

/// What binding and closing a guest costs, its state cleared at each bind:
/// `PIE_GUEST_BIND_ROUNDS` rounds (default 200) of one bind and one close.
#[test]
#[ignore = "a measurement: run on its own"]
fn guest_bind_close_rounds_timed() {
    let Some((loaded, program, mut slots, _)) = setup("guest-state-binds", 4) else {
        return;
    };
    let Loaded {
        mut engine, dir, ..
    } = loaded;
    let rounds: u64 = std::env::var("PIE_GUEST_BIND_ROUNDS")
        .ok()
        .and_then(|rounds| rounds.parse().ok())
        .unwrap_or(200);
    let state = engine
        .guest_state(program, &EXTENTS, GeometryClass::Host)
        .expect("the program's guest state");
    let (first, _, room) = bind(&mut engine, &mut slots, program, 1);
    engine.close_instance(first).expect("closes");
    let started = std::time::Instant::now();
    for round in 0..rounds {
        let (instance, _, _) =
            bind_in(&mut engine, &mut slots, program, 4 + 3 * round, Some(&room));
        engine.close_instance(instance).expect("closes");
    }
    let took = started.elapsed();
    eprintln!(
        "guest bind+close: {rounds} rounds, {:.1} us a round, state {} B an instance + {} B its pair",
        took.as_secs_f64() * 1e6 / rounds as f64,
        state.instance_bytes(),
        state.per_pair
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A beam group's state (`engine_gpu::beam::Resident`, first in the
/// instance's one place, `GuestState::resident`, its heaps past it) follows
/// a park into fresh room: laid and cleared at the bind, a search's lists
/// written there, the runtime's copy of the whole place into a fresh one
/// and the old handed on (overwritten), and the retarget the unpark's
/// `Rebind::Instance` makes: it now reads its lists from the fresh place,
/// the copied state standing while the heaps past it are cleared, and an
/// evacuation's rebind writes nothing.
#[test]
fn a_beam_groups_resident_state_survives_a_park_into_fresh_slots() {
    let Some((loaded, _program, mut slots, _)) = setup("beam-resident-park", 4) else {
        return;
    };
    let Loaded { engine, dir, .. } = loaded;
    let layout = engine_gpu::guest::state::BeamLayout::of(4, 300, 16, 2).expect("a depth");
    resident_survives_a_park(&engine, &mut slots, layout);
    // A search deep enough that its place passes a chunk: one run of whole
    // chunks, addressed as one span.
    let chunk = engine.loaded().expect("loaded").pools().layout().chunk_bytes;
    let mut end = 1024u32;
    let deep = loop {
        let layout = engine_gpu::guest::state::BeamLayout::of(4, end, 16, 2).expect("a depth");
        if layout.bytes() > chunk {
            break layout;
        }
        end *= 2;
    };
    resident_survives_a_park(&engine, &mut slots, deep);
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}

/// One group's state of `layout` through a park into fresh room
/// ([`a_beam_groups_resident_state_survives_a_park_into_fresh_slots`]).
fn resident_survives_a_park(engine: &backend::Engine, slots: &mut Slots, layout: engine_gpu::guest::state::BeamLayout) {
    let bytes = layout.bytes();
    // The instance's place: the state first, aligned, a heap past it.
    let guest = engine_gpu::guest::state::GuestLayout::of(&[64 << 10], 0).with_beam(Some(layout));
    let (head, place_bytes) = (guest.resident, guest.instance);
    assert!(head >= bytes && guest.stages[0].1 == head, "the heap lies past the state");
    let chunk = engine.loaded().expect("loaded").pools().layout().chunk_bytes;
    let run = u32::try_from(place_bytes.div_ceil(chunk)).expect("chunks");
    let parked = slots.state(place_bytes, run);
    let fresh = slots.state(place_bytes, run);
    assert_ne!(parked, fresh, "the unpark's place is another");
    let pools = engine.loaded().expect("loaded").pools();
    fill_room(&engine, &parked, place_bytes, 0x5a);
    let mut place = engine_gpu::guest::state::Placed::at(pools, &parked, place_bytes)
        .expect("the place lies in the pool")
        .keeping(head);
    place.clear().expect("the bind clears its place");
    let mut resident =
        engine_gpu::beam::Resident::bind(pools, &parked, layout).expect("the state binds");
    assert!(
        read_room(&engine, &parked, place_bytes).iter().all(|&b| b == 0),
        "the bind clears its place"
    );
    // A search's state as five fires left it: primed, its last list at
    // the fires' parity, beam `b`'s column `j` page `10 b + j`.
    let words = (bytes / 4) as usize;
    let mut state = vec![0u32; words];
    state[..4].copy_from_slice(&[1, 5, 40, 32]);
    let map = kernels_slang::runtime::beam::map_word(layout.lanes, layout.cap, 5) as usize;
    for b in 0..layout.lanes as usize {
        for j in 0..layout.cap as usize {
            state[map + b * layout.cap as usize + j] = (10 * b + j) as u32;
        }
    }
    let image: Vec<u8> = state.iter().flat_map(|w| w.to_le_bytes()).collect();
    let (buffer, at) = located(&engine, &parked).expect("the run lies in the pool");
    buffer.write_shared(at, &image).expect("the state lands");
    let want: Vec<u32> = (0..layout.cap).map(|j| 20 + j).collect();
    assert_eq!(resident.list(pools, 2).expect("beam 2's list"), want);

    // The heap's bytes as a fire left them: dead past it.
    fill_room_from(&engine, &parked, head, place_bytes - head, 0x11);
    // The park: the runtime copies the place's bytes wherever the instance
    // goes and back into a fresh place at the unpark; the old one is another
    // owner's.
    fill_room(&engine, &fresh, place_bytes, 0x7f);
    let copied_place = read_room(&engine, &parked, place_bytes);
    let copied = copied_place[..bytes as usize].to_vec();
    let (buffer, at) = located(&engine, &fresh).expect("the fresh place lies in the pool");
    buffer.write_shared(at, &copied_place).expect("the copy lands");
    fill_room(&engine, &parked, place_bytes, 0x3c);
    resident.retarget(pools, &fresh).expect("the unpark retargets");
    // The unpark clears the place past its resident head, as a session does.
    place.retarget(pools, &fresh).expect("the place retargets");
    place.clear_dead().expect("the heap clears");
    let landed = read_room(&engine, &fresh, place_bytes);
    assert_eq!(&landed[..head as usize], &copied_place[..head as usize], "the head stands");
    assert!(landed[head as usize..].iter().all(|&b| b == 0), "the heap past it is cleared");
    let first_word = match &fresh {
        &ScratchPlace::Run { first, .. } => pools
            .slot_word(engine_gpu::store::layout::INSTANCE, first)
            .expect("the fresh run's word"),
        ScratchPlace::Chunks { pieces, .. } => pools.chunk_word(pieces[0]).expect("the fresh run's word"),
        other => panic!("a beam state lies in one span: {other:?}"),
    };
    assert_eq!(
        resident.word(pools).expect("its word"),
        first_word,
        "the state's word is the fresh run's"
    );
    assert_eq!(
        read_room(&engine, &fresh, bytes),
        copied,
        "the retarget clears nothing: the copied bytes stand"
    );
    assert_eq!(
        resident.list(pools, 2).expect("beam 2's list"),
        want,
        "the lists read from the fresh run"
    );
    // An evacuation's rebind finds the run where it sits and writes nothing.
    resident.rebind(pools).expect("the rebind");
    assert_eq!(read_room(&engine, &fresh, bytes), copied, "a rebind writes nothing");
    assert_eq!(resident.list(pools, 3).expect("beam 3's list")[0], 30);
}
