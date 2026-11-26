//! A guest program's state lies in its instance's and its pair's places in
//! the pool (`Engine::guest_state`, `engine_gpu::guest::state`): an unpark
//! moves it to fresh room, nothing copied, the passes after reading what
//! they read before; and an evacuation is all or nothing, a move its
//! rebind refuses leaving every chunk as it stood.

use engine::chunk::Evacuation;
use engine::program::{Rebind, ScratchPlace};
use engine_gpu::backend::Buffer as _;
use eta_ir::registry::GeometryClass;

use super::outputs::{Bound, EXTENTS, Loaded, bound};
use crate::backend;

/// `bytes` of `place` filled with `fill`, where it holds any.
fn fill_room(engine: &backend::Engine, place: &ScratchPlace, bytes: u64, fill: u8) {
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
    if bytes > 0 {
        buffer
            .write_shared(at, &vec![fill; bytes as usize])
            .expect("the room fills");
    }
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

/// A park and an unpark: the runtime gives the state back and takes fresh
/// room, copying nothing, and tells the engine where (`Rebind::Instance`).
/// The fresh room holds garbage and the old room is overwritten after the
/// move: the passes after read what they read before.
#[test]
fn an_unparked_guest_serves_on_from_fresh_room_with_nothing_copied() {
    let Some(mut bound) = bound("guest-state-park") else {
        return;
    };
    let (instance, channels, binding) = bound.bind(1);
    assert_eq!(
        bound.pass(instance, &channels, 5).0,
        bound.argmax(5),
        "the first pass reads its row"
    );
    let Bound {
        loaded,
        program,
        slots,
        ..
    } = &mut bound;
    let state =
        engine::Programs::guest_state(&loaded.engine, *program, &EXTENTS, GeometryClass::Host)
            .expect("the program's guest state");
    let scratch = slots.state(state.instance_bytes(), state.run);
    let pair = slots.state(state.per_pair, state.run);
    // Fresh room is dead: whatever lies there.
    fill_room(&loaded.engine, &scratch, state.instance_bytes(), 0x7f);
    fill_room(&loaded.engine, &pair, state.per_pair, 0x7f);
    engine::Programs::rebind(
        &mut loaded.engine,
        Rebind::Instance {
            id: instance,
            instance: binding.instance_slots.clone(),
            rings: Vec::new(),
            adapter: Vec::new(),
            scratch,
            pair,
            pair_generation: 0,
        },
    )
    .expect("the unpark rebinds its state");
    // The old room is someone else's now.
    fill_room(
        &loaded.engine,
        &binding.scratch,
        state.instance_bytes(),
        0x3c,
    );
    fill_room(&loaded.engine, &binding.pair, state.per_pair, 0x3c);
    for token in [5, 9, 2] {
        let ids = bound.pass(instance, &channels, token).0;
        assert_eq!(
            ids,
            bound.argmax(token),
            "a pass after the unpark reads token {token}'s row"
        );
    }
    let _ = std::fs::remove_dir_all(&bound.loaded.dir);
}

/// `Engine::evacuate` is all or nothing. Instance A's state lies in a run
/// of two whole chunks, and an evacuation moves the run's head onto the
/// chunk A's tables lie in, segment 0's last: the pool takes the move, and
/// the rebind refuses it by name (the run would cross the segment's end)
/// after the copies were recorded. Nothing moved: the chunk table and the
/// bytes of both chunks are as they stood, A fires on, and its claim stands
/// (a park and an unpark onto its own slots lay it again).
#[test]
fn an_evacuation_refused_at_its_rebind_moves_nothing() {
    let Some(mut bound) = bound("guest-state-evacuate-refused") else {
        return;
    };
    let top = bound
        .loaded
        .engine
        .loaded()
        .expect("loaded")
        .pools()
        .chunk_facts()
        .segment_chunks;
    assert!(top >= 32, "segment 0 holds the test's chunks: {top}");
    let per = bound.slots.instance.expect("an Instance shape").per_chunk;
    let (tables, head) = (top - 1, top - 24);
    let (a, channels, binding) = bound.bind(1);
    assert!(
        binding
            .instance_slots
            .iter()
            .all(|&slot| slot / per == tables),
        "A's tables lie in chunk {tables}: {:?}",
        binding.instance_slots
    );
    let rebind = |engine: &mut backend::Engine, scratch: ScratchPlace| {
        engine::Programs::rebind(
            engine,
            Rebind::Instance {
                id: a,
                instance: binding.instance_slots.clone(),
                rings: Vec::new(),
                adapter: Vec::new(),
                scratch,
                pair: binding.pair.clone(),
                pair_generation: binding.pair_generation,
            },
        )
    };
    let run = ScratchPlace::Chunks {
        run: 2,
        pieces: vec![head],
    };
    rebind(&mut bound.loaded.engine, run.clone()).expect("A's state moves to a run of two chunks");
    assert_eq!(
        bound.pass(a, &channels, 5).0,
        bound.argmax(5),
        "A reads its row from its run"
    );

    let Loaded { engine, .. } = &mut bound.loaded;
    let table = engine::Pool::chunk_table(engine).expect("the engine's chunk table");
    let (head_bytes, table_bytes) = (read_chunk(engine, head), read_chunk(engine, tables));
    let onto = table
        .physical(tables)
        .expect("the tables' chunk lies somewhere");
    let refused = engine::Pool::evacuate(
        engine,
        &Evacuation {
            moves: vec![(head, onto)],
        },
    )
    .expect_err("A's run would cross segment 0's end");
    assert!(
        matches!(&refused, engine::Error::Program(why) if why.contains("one segment")),
        "refused at the rebind, by the run's place: {refused}"
    );
    assert_eq!(
        engine::Pool::chunk_table(engine).expect("the chunk table"),
        table,
        "the table stands"
    );
    assert!(
        read_chunk(engine, tables) == table_bytes,
        "nothing copied over A's tables"
    );
    assert!(read_chunk(engine, head) == head_bytes, "A's run stands");
    assert_eq!(
        bound.pass(a, &channels, 2).0,
        bound.argmax(2),
        "A fires on from where it lay"
    );
    let engine = &mut bound.loaded.engine;
    engine::Programs::rebind(engine, Rebind::Parked { id: a, kept: true })
        .expect("A parks, kept in place");
    rebind(engine, run).expect("A unparks onto its own slots");
    assert_eq!(
        bound.pass(a, &channels, 3).0,
        bound.argmax(3),
        "A fires on after the refusal"
    );
    let _ = std::fs::remove_dir_all(&bound.loaded.dir);
}
