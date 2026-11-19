//! A guest program's state lies in its instance's and its pair's places in
//! the pool (`Engine::guest_state`, `engine_gpu::guest::state`), never in a
//! fire's span: the engine states no fire term for it, so 64 bound lanes
//! fire in the rung's span as it stood before any bound; and a park and an
//! unpark move the state to fresh room, nothing copied, the passes after
//! reading what they read before.

use engine::ChannelId;
use engine::Engine;
use engine::channel::ChannelRegistration;
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step};
use engine::program::{BindExtents, InstanceBinding, InstanceId, Rebind, ScratchPlace};
use engine_gpu::backend::Buffer as _;
use eta_ir::container::{ChanDType, HostRole};
use eta_ir::registry::GeometryClass;
use eta_ir::types::Dtype as EtaDtype;
use poem::Request;

use super::outputs::{
    LAYERS, Loaded, Outputs, ROWS, T, WIDTH, epilogue, loaded_lanes, output_decls,
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
    let mut loaded = loaded_lanes(tag, &Outputs, lanes)?;
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
#[test]
fn a_bound_guest_states_no_fire_term_and_64_lanes_fire_in_the_rungs_span() {
    const LANES: u32 = 64;
    let Some((loaded, program, mut slots, t_input)) = setup("guest-state-64", LANES) else {
        return;
    };
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

    let bound: Vec<(InstanceId, Vec<ChannelId>)> = (0..u64::from(LANES))
        .map(|k| {
            let (instance, channels, _) = bind(&mut engine, &mut slots, program, 1 + 3 * k);
            (instance, channels)
        })
        .collect();
    assert_eq!(span(&engine), before, "64 bound guests leave the span's rungs as they were");

    let word = trace.facts.word(&Request::new(ROWS, false));
    for round in 0..2u32 {
        let tokens: Vec<u32> = (0..LANES).map(|k| (k + round) % super::outputs::VOCAB).collect();
        let ids = fire(&mut engine, word, t_input, &bound, &tokens);
        let want: Vec<i32> = tokens.iter().map(|&token| argmax(&planes, token)).collect();
        assert_eq!(ids, want, "round {round}: each of 64 lanes reads its own row");
    }
    assert_eq!(span(&engine), before, "the fires leave the span's rungs as they were");
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
