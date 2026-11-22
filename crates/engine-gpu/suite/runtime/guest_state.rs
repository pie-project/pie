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
    let instance = engine
        .bind_instance(&binding)
        .expect("the instance binds")
        .id;
    assert!(
        engine
            .publish_channel(instance, 2, &T.to_le_bytes())
            .expect("the timestep publishes"),
        "the ring had room"
    );
    (instance, channels, binding)
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

/// `Engine::evacuate` is all or nothing. Instance A parks keeping its
/// tables where they lie (`Rebind::Parked { kept: true }`), and an
/// evacuation moves instance B's tables' chunk onto the physical chunk A's
/// lie on: B's tables resolve over A's kept run, and the claims refuse the
/// move by name at the rebind, after the pool took it. Nothing moved: the
/// chunk table, and the bytes of A's chunk and B's, are as they stood; B
/// fires on from where it lay; and A's claim stands (its unpark onto its
/// own slots lays it again beside B's), after which both fire.
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
    let word = trace.facts.word(&Request::new(ROWS, false));
    let top = engine
        .loaded()
        .expect("loaded")
        .pools()
        .chunk_facts()
        .segment_chunks;
    assert!(top >= 32, "segment 0 holds the test's chunks: {top}");
    let per = slots.instance.expect("an Instance shape").per_chunk;
    // A's tables lie in the slots' own chunk, at its first slots; B's at
    // the same slots of a chunk well under every state's run.
    let a_chunk = top - 1;
    let b_chunk = top - 24;

    let (a, a_channels, a_binding) = bind(&mut engine, &mut slots, program, 1);
    assert!(
        a_binding.instance_slots.iter().all(|&slot| slot / per == a_chunk),
        "A's tables lie in chunk {a_chunk}: {:?}",
        a_binding.instance_slots
    );
    let b_room = InstanceBinding {
        instance_slots: a_binding
            .instance_slots
            .iter()
            .map(|&slot| b_chunk * per + slot % per)
            .collect(),
        scratch: slots.state(state.instance_bytes(), state.run),
        ..a_binding.clone()
    };
    let (b, b_channels, b_binding) = bind_in(&mut engine, &mut slots, program, 4, Some(&b_room));
    for place in [&a_binding.scratch, &a_binding.pair, &b_binding.scratch] {
        if let ScratchPlace::Chunks { pieces, .. } = place {
            assert!(pieces[0] > b_chunk, "a state's run above chunk {b_chunk}: {place:?}");
        }
    }
    let both = [(a, a_channels.clone()), (b, b_channels.clone())];
    let ids = fire(&mut engine, word, t_input, &both, &[5, 9]);
    assert_eq!(ids, [argmax(&planes, 5), argmax(&planes, 9)], "A and B read their rows");

    engine
        .rebind_slots(Rebind::Parked { id: a, kept: true })
        .expect("A parks, its tables kept in place");
    let b_alone = [(b, b_channels.clone())];
    let ids = fire(&mut engine, word, t_input, &b_alone, &[7]);
    assert_eq!(ids, [argmax(&planes, 7)], "B reads its row with A parked");

    let table = engine.chunk_table().expect("the engine's chunk table");
    let a_bytes = read_chunk(&engine, a_chunk);
    let b_bytes = read_chunk(&engine, b_chunk);
    let onto = table.physical(a_chunk).expect("A's chunk lies somewhere");
    let refused = engine
        .evacuate(&Evacuation {
            moves: vec![(b_chunk, onto)],
        })
        .expect_err("B's tables would lie over A's kept ones");
    eprintln!("refused: {refused}");
    assert!(
        matches!(refused, engine::Error::Claimed(_)),
        "refused by the claims, at the rebind: {refused}"
    );
    assert_eq!(
        engine.chunk_table().expect("the engine's chunk table"),
        table,
        "a refused evacuation leaves the chunk table as it stood"
    );
    assert!(
        read_chunk(&engine, a_chunk) == a_bytes,
        "a refused evacuation copies nothing over A's kept tables"
    );
    assert!(
        read_chunk(&engine, b_chunk) == b_bytes,
        "a refused evacuation leaves B's tables as they stood"
    );
    let ids = fire(&mut engine, word, t_input, &b_alone, &[2]);
    assert_eq!(ids, [argmax(&planes, 2)], "B fires on from where it lay");

    // A's claim stood: its unpark onto its own slots claims them again
    // beside B's, whose claim stood too.
    engine
        .rebind_slots(Rebind::Instance {
            id: a,
            instance: a_binding.instance_slots.clone(),
            rings: Vec::new(),
            adapter: Vec::new(),
            scratch: a_binding.scratch.clone(),
            pair: a_binding.pair.clone(),
            pair_generation: a_binding.pair_generation,
        })
        .expect("A unparks onto its own slots");
    let ids = fire(&mut engine, word, t_input, &both, &[3, 6]);
    assert_eq!(
        ids,
        [argmax(&planes, 3), argmax(&planes, 6)],
        "A and B fire on after the refused evacuation"
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
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
