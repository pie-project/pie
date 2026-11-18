//! A guest program's scratch as state (REDESIGN §6, `::engine::GuestState`):
//! each bound instance holds `Instance` slots for its lanes, one run in a
//! chunk where they fit a chunk's slots, else whole chunks
//! (`::engine::ScratchPlace`), and its (program, extents) pair holds its own
//! once while any instance of it is bound. Nothing of the activation span is a program's, so a one-lane
//! prefill reaches the top rung with a program bound; the state is counted
//! at the bind, which a pool short of it refuses by name; and a park gives
//! the scratch back and copies nothing, an unpark taking fresh chunks the
//! engine is told.

use std::sync::{Arc, Mutex};

use ::engine::chunk::{
    ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId as EngineShape, SlotKind,
    SlotRole, SlotShape,
};
use ::engine::{
    BindExtents, BoundInstance, Engine, FrameSubmission, FrameTicket, GuestState, InstanceBinding,
    InstanceId, LoadRequest, Loaded, ProgramFacts, ProgramId, Rebind, ScratchPlace,
};

use super::{
    Owner, Roles, RolesError, RolesHandle, bind_instance, bind_taken, learn_program, take_bind,
};
use crate::store::shared::{Shared, SharedPool};

const MIB: u64 = 1 << 20;
const CHUNK: u64 = 16 * MIB;
const LIST: EngineShape = EngineShape(0);
const INSTANCE: EngineShape = EngineShape(2);
/// An `Instance` slot: 4096 of them a chunk.
const SLOT: u64 = 4 << 10;
const SPC: u32 = 4096;
const PROGRAM: ProgramId = ProgramId(7);
/// A top-p sampler's lanes' scratch, a bound instance: a run of 709 slots.
const TOP_P: u64 = 2_900_000;
/// What the sampler's pair holds once: a run of 1536 slots.
const PAIR: u64 = 6 * MIB;

/// The `Instance` slots `bytes` take in one run.
fn slots(bytes: u64) -> u64 {
    bytes.div_ceil(SLOT)
}

/// The logical chunks `place` lies in (no evacuation here: a run's
/// chunks are its first's next logical ids too).
fn chunks(place: &ScratchPlace) -> Vec<u32> {
    match place {
        ScratchPlace::None => Vec::new(),
        ScratchPlace::Run { first, .. } => vec![first / SPC],
        ScratchPlace::Chunks { run, pieces } => pieces
            .iter()
            .flat_map(|&first| first..first + run)
            .collect(),
    }
}
/// The rungs: (16 tokens, 1 chunk), (256, 4), (2048, 12); the top rung's
/// span is 12 chunks.
const RUNGS: [(u32, u32); 3] = [(16, 1), (256, 4), (2048, 12)];

fn shape(id: EngineShape, kind: SlotKind, slot_bytes: u64, role: SlotRole) -> SlotShape {
    SlotShape {
        id,
        kind,
        slot_bytes,
        slots_per_chunk: u32::try_from(CHUNK / slot_bytes).unwrap(),
        sets_per_working_set: 1,
        rows: Vec::new(),
        pinned: None,
        role,
    }
}

/// `total` chunks of 16 MiB in one segment, a kv list and a 4 KiB
/// `Instance` shape, the rungs [`RUNGS`]; 8 host chunks.
fn facts(total: u32) -> ChunkFacts {
    facts_in(total, total)
}

/// [`facts`] in segments of `segment` chunks, all committed.
fn facts_in(total: u32, segment: u32) -> ChunkFacts {
    let segments = total.div_ceil(segment);
    facts_committed(total, segment, segments)
}

/// [`facts_in`] with the lowest `committed` segments committed at load
/// (and the top rung's span), the rest the pool's to commit as it grows.
fn facts_committed(total: u32, segment: u32, committed: u32) -> ChunkFacts {
    let segments = total.div_ceil(segment);
    let facts = ChunkFacts {
        chunk_bytes: CHUNK,
        total_chunks: total,
        segment_chunks: segment,
        segments: SegmentFacts {
            committed,
            max: segments,
            growth: if committed == segments {
                Growth::Fixed
            } else {
                Growth::Bottom
            },
        },
        reserved: vec![ReservedRun {
            what: Reserved::Null,
            first: 0,
            chunks: 1,
        }],
        shapes: vec![
            shape(
                LIST,
                SlotKind::List {
                    tokens: 16,
                    rate: 1,
                    window: None,
                },
                MIB,
                SlotRole::State,
            ),
            shape(
                INSTANCE,
                SlotKind::Fixed { pieces: 1 },
                SLOT,
                SlotRole::Instance,
            ),
        ],
        act_chunks: RUNGS.to_vec(),
        host_chunks: 8,
        ..ChunkFacts::default()
    };
    facts.validate().unwrap();
    facts
}

struct Pools {
    device: Shared,
    roles: RolesHandle,
}

fn pools(total: u32) -> Pools {
    pools_of(facts(total))
}

fn pools_of(facts: ChunkFacts) -> Pools {
    let device = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let host = Arc::new(Mutex::new(
        SharedPool::tier(&facts, facts.host_chunks)
            .unwrap()
            .unwrap(),
    ));
    let roles = Roles::new(&device, Some(&host)).expect("the pool states an Instance shape");
    Pools {
        device,
        roles: Arc::new(Mutex::new(roles)),
    }
}

/// The refusal `error` names: the room a bind found short, whether it
/// waits for room given back (`RolesError::BindShort`) or is refused.
fn short_of(error: &anyhow::Error) -> Option<&RolesError> {
    match error.downcast_ref::<RolesError>() {
        Some(RolesError::BindShort(short)) => Some(short),
        other => other,
    }
}

/// Bytes the pool's slots hold for live state.
fn held(pool: &Shared) -> u64 {
    pool.lock().unwrap().pool().demand().live
}

/// States `state` for every program, no tables; records its bindings and
/// rebinds.
struct GuestEngine {
    state: GuestState,
    next: u64,
    bindings: Vec<InstanceBinding>,
    rebinds: Vec<Rebind>,
}

impl GuestEngine {
    fn new(per_instance: u64, per_pair: u64) -> Self {
        Self {
            state: GuestState {
                per_instance,
                per_pair,
                run: 1,
            },
            next: 0,
            bindings: Vec::new(),
            rebinds: Vec::new(),
        }
    }
}

impl Engine for GuestEngine {
    fn kind(&self) -> &'static str {
        "fake-guests"
    }

    fn prepare_program(
        &mut self,
        _registration: &::engine::program::ProgramRegistration,
    ) -> ::engine::Result<::engine::ProgramCompile> {
        Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
    }

    fn finish_program(
        &mut self,
        _prepared: ::engine::Prepared,
    ) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "finish_program"))
    }

    fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "close_program"))
    }

    fn load(&mut self, _request: LoadRequest) -> ::engine::Result<Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(&mut self, _frame: &FrameSubmission) -> ::engine::Result<FrameTicket> {
        unreachable!("nothing fires here")
    }

    fn program_facts(&self, _program: ProgramId) -> ::engine::Result<ProgramFacts> {
        Ok(ProgramFacts::default())
    }

    fn guest_state(
        &self,
        _program: ProgramId,
        _extents: &BindExtents,
    ) -> ::engine::Result<GuestState> {
        Ok(self.state)
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> ::engine::Result<BoundInstance> {
        self.next += 1;
        self.bindings.push(binding.clone());
        Ok(BoundInstance {
            id: InstanceId(self.next),
            program: binding.program,
            geometry: binding.geometry,
            standing_guest_bytes: 0,
        })
    }

    fn rebind_slots(&mut self, rebind: Rebind) -> ::engine::Result<()> {
        self.rebinds.push(rebind);
        Ok(())
    }
}

fn holder(n: u128) -> super::Holder {
    uuid::Uuid::from_u128(n)
}

/// Binds an instance of [`PROGRAM`] at the default extents for `by`.
fn bind(roles: &RolesHandle, engine: &mut GuestEngine, by: u128) -> anyhow::Result<InstanceId> {
    learn_program(roles, engine, PROGRAM)?;
    let binding = InstanceBinding {
        program: PROGRAM,
        ..InstanceBinding::default()
    };
    Ok(bind_instance(Some(roles), engine, &binding, Some(holder(by)), |_| false)?.id)
}

/// A one-lane 2048-token prefill with a top-p-like program bound reaches
/// the top rung: the program's scratch (2.9 MB an instance, 6 MiB its
/// pair, each a run of slots in a chunk) is state, below the span, and the rung's activation is the
/// model's carve alone.
#[test]
fn a_one_lane_prefill_reaches_the_top_rung_with_a_program_bound() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(TOP_P, PAIR);
    bind(&roles, &mut engine, 1).unwrap();
    let binding = &engine.bindings[0];
    assert!(
        matches!(binding.scratch, ScratchPlace::Run { slots: 709, .. }),
        "2.9 MB: a run of 709 slots, {:?}",
        binding.scratch
    );
    assert!(
        matches!(binding.pair, ScratchPlace::Run { slots: 1536, .. }),
        "6 MiB: a run of 1536 slots, {:?}",
        binding.pair
    );
    {
        let pool = device.lock().unwrap();
        let limit = pool.pool().state_limit();
        assert_eq!(limit, 64 - 12, "the top rung's span lies above it");
        for chunk in chunks(&binding.scratch)
            .into_iter()
            .chain(chunks(&binding.pair))
        {
            assert!(
                pool.pool().physical(chunk).unwrap() < limit,
                "the program's state lies below the span"
            );
        }
        assert_eq!(pool.pool().act_chunks(2048), 12, "the model's carve alone");
        assert_eq!(pool.widest_fit(), Some(2048));
        assert_eq!(pool.fit_cap(), u32::MAX, "fires uncut");
        assert_eq!(pool.rung_for(2048), 2048, "a 2048-token fire takes r_top");
        assert!(pool.plan_raise(2048).is_ok(), "and is set aside");
    }
}

/// 256 bound instances of a program holding 11.2 MB each are counted at
/// their binds, a whole chunk each (past half a chunk's slots: a run of
/// them would need a chunk nearly empty) beside the pair's run, and take
/// state's room as any state does; the 257th
/// finds no chunk with the run free and is refused by name, nothing taken.
#[test]
fn two_hundred_fifty_six_instances_are_counted_at_bind_and_the_257th_is_refused() {
    // The null chunk, 256 scratch chunks, the pair's, and the top rung's
    // span of 12.
    let Pools { device, roles } = pools(1 + 256 + 1 + 12);
    let mut engine = GuestEngine::new(11_200_000, PAIR);
    let before = held(&device);
    for by in 1..=256 {
        bind(&roles, &mut engine, by).unwrap_or_else(|error| panic!("bind {by}: {error}"));
    }
    let state = 256 * CHUNK + slots(PAIR) * SLOT;
    assert_eq!(
        held(&device) - before,
        state,
        "256 scratch runs and the pair's"
    );
    assert_eq!(
        roles.lock().unwrap().device_bytes(),
        state,
        "state, counted"
    );
    let error = bind(&roles, &mut engine, 257).expect_err("the pool holds no more");
    let named = error.to_string();
    assert!(
        matches!(
            short_of(&error),
            Some(RolesError::GuestShort {
                bytes: 11_200_000,
                ..
            })
        ),
        "{named}"
    );
    assert!(
        named.contains("scratch") && named.contains("program 7"),
        "{named}"
    );
    assert_eq!(
        held(&device) - before,
        state,
        "the refused bind took nothing"
    );
    assert_eq!(engine.bindings.len(), 256, "the engine was not asked");
}

/// The pair's chunks (past a chunk's slots: whole chunks) are taken once,
/// at the pair's first bind: every instance of it is told the same ones.
/// They go back at the last close, each instance's scratch at its own.
#[test]
fn the_pair_s_chunks_are_taken_once_and_given_back_at_the_last_close() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(TOP_P, 2 * CHUNK);
    let before = held(&device);
    let ids: Vec<InstanceId> = (1..=3)
        .map(|by| bind(&roles, &mut engine, by).unwrap())
        .collect();
    let pair = engine.bindings[0].pair.clone();
    assert!(
        matches!(&pair, ScratchPlace::Chunks { run: 1, pieces } if pieces.len() == 2),
        "two whole chunks: {pair:?}"
    );
    assert!(engine.bindings.iter().all(|binding| binding.pair == pair));
    let run = slots(TOP_P) * SLOT;
    assert_eq!(held(&device) - before, 2 * CHUNK + 3 * run);
    roles.lock().unwrap().close_instance(ids[0]);
    roles.lock().unwrap().close_instance(ids[1]);
    assert_eq!(
        held(&device) - before,
        2 * CHUNK + run,
        "the pair stays with its last"
    );
    assert!(
        !roles
            .lock()
            .unwrap()
            .slots(Owner::Pair(super::FireKey {
                program: PROGRAM,
                extents: BindExtents::default(),
            }))
            .is_empty()
    );
    roles.lock().unwrap().close_instance(ids[2]);
    assert_eq!(held(&device), before, "the last close gives the pair back");
}

/// A park gives an instance's scratch (two whole chunks) and its pair's run
/// back and copies nothing of them (their contents are dead between
/// fires); the unpark takes fresh chunks and a fresh run, and the engine is
/// told them before the next fire.
#[test]
fn a_park_gives_the_scratch_back_whole_and_an_unpark_takes_fresh_chunks() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    let before = held(&device);
    let id = bind(&roles, &mut engine, 1).unwrap();
    let bound = engine.bindings[0].clone();
    assert_eq!(chunks(&bound.scratch).len(), 2);
    let pair = slots(PAIR) * SLOT;
    assert_eq!(held(&device) - before, 2 * CHUNK + pair);

    let park = roles
        .lock()
        .unwrap()
        .prepare_park(holder(1))
        .expect("the instance and its pair park");
    assert!(
        park.copies().is_empty(),
        "nothing copied: {:?}",
        park.copies()
    );
    roles.lock().unwrap().commit(park);
    assert_eq!(held(&device), before, "parked: its chunks given back whole");

    // The freed chunks are taken by another pair's instance meanwhile.
    let mut other = GuestEngine::new(3 * CHUNK, 0);
    // The engine's instance ids are one space.
    other.next = 100;
    learn_program(&roles, &mut other, ProgramId(8)).unwrap();
    bind_instance(
        Some(&roles),
        &mut other,
        &InstanceBinding {
            program: ProgramId(8),
            ..InstanceBinding::default()
        },
        Some(holder(2)),
        |_| false,
    )
    .unwrap();

    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .expect("it unparks");
    assert!(unpark.copies().is_empty(), "nothing copied back");
    roles.lock().unwrap().commit(unpark);
    assert_eq!(held(&device) - before, 3 * CHUNK + 2 * CHUNK + pair);
    let stale = roles.lock().unwrap().stale(holder(1));
    let [
        Rebind::Instance {
            id: rebound,
            scratch,
            pair: pair_place,
            ..
        },
    ] = stale.as_slice()
    else {
        panic!("one rebind of the instance: {stale:?}");
    };
    assert_eq!(*rebound, id);
    let fresh = chunks(scratch);
    assert!(
        matches!(scratch, ScratchPlace::Chunks { .. }) && fresh.len() == 2,
        "{scratch:?}"
    );
    assert!(
        matches!(pair_place, ScratchPlace::Run { slots: 1536, .. }),
        "a fresh run: {pair_place:?}"
    );
    let others = chunks(&other.bindings[0].scratch);
    assert!(
        fresh.iter().all(|chunk| !others.contains(chunk)),
        "none of another instance's: {fresh:?} {others:?}"
    );
    for rebind in stale {
        engine.rebind_slots(rebind).unwrap();
        roles.lock().unwrap().rebound(Owner::Instance(id));
    }
    assert!(roles.lock().unwrap().stale(holder(1)).is_empty());
}

/// The `Instance` slot refs `place` covers.
fn refs_of(place: &ScratchPlace) -> Vec<u32> {
    match place {
        ScratchPlace::None => Vec::new(),
        ScratchPlace::Run { first, slots } => (*first..first + slots).collect(),
        ScratchPlace::Chunks { .. } => chunks(place)
            .iter()
            .flat_map(|&chunk| chunk * SPC..(chunk + 1) * SPC)
            .collect(),
    }
}

/// Whether no slot lies in two of `places`: the first one that does,
/// named.
fn disjoint(places: &[&ScratchPlace]) -> Result<(), String> {
    let mut seen = std::collections::HashMap::new();
    for (at, place) in places.iter().enumerate() {
        for slot in refs_of(place) {
            if let Some(was) = seen.insert(slot, at) {
                return Err(format!("slot {slot} lies in places {was} and {at}"));
            }
        }
    }
    Ok(())
}

/// Two programs at their own pairs, three instances each, all of them in
/// one fire: a beam stage and a sampler. Every instance's scratch and each
/// pair's lie in places of their own, no slot in two of them, all below the
/// top rung's span (where a fire's readout and capture rows lie, beside its
/// carve); so too after one instance parks and unparks. What a fire lays
/// for one program never lies where another's lies, whatever the facts.
#[test]
fn two_programs_scratch_and_pairs_lie_apart_and_stay_apart_after_a_park() {
    let Pools { device, roles } = pools(64);
    // The sampler: a run an instance, chunks for its pair.
    let mut sampler = GuestEngine::new(TOP_P, 2 * CHUNK);
    // The beam: chunks an instance, a run for its pair.
    let mut beam = GuestEngine::new(2 * CHUNK, PAIR);
    beam.next = 100;
    for by in 1..=3 {
        bind(&roles, &mut sampler, by).unwrap();
        learn_program(&roles, &mut beam, ProgramId(8)).unwrap();
        bind_instance(
            Some(&roles),
            &mut beam,
            &InstanceBinding {
                program: ProgramId(8),
                ..InstanceBinding::default()
            },
            Some(holder(10 + by)),
            |_| false,
        )
        .unwrap();
    }
    let mut scratch: Vec<ScratchPlace> = sampler
        .bindings
        .iter()
        .chain(&beam.bindings)
        .map(|binding| binding.scratch.clone())
        .collect();
    let pairs = [
        sampler.bindings[0].pair.clone(),
        beam.bindings[0].pair.clone(),
    ];
    assert!(
        sampler
            .bindings
            .iter()
            .all(|binding| binding.pair == pairs[0])
    );
    assert!(beam.bindings.iter().all(|binding| binding.pair == pairs[1]));
    assert!(matches!(scratch[0], ScratchPlace::Run { .. }));
    assert!(matches!(scratch[3], ScratchPlace::Chunks { .. }));
    assert!(matches!(pairs[0], ScratchPlace::Chunks { .. }));
    assert!(matches!(pairs[1], ScratchPlace::Run { .. }));
    let limit = device.lock().unwrap().pool().state_limit();
    let check = |scratch: &[ScratchPlace]| {
        let places: Vec<&ScratchPlace> = scratch.iter().chain(&pairs).collect();
        disjoint(&places).unwrap();
        for place in places {
            assert!(
                chunks(place).iter().all(|&chunk| chunk < limit),
                "{place:?} lies below the span's {limit}"
            );
        }
    };
    check(&scratch);

    // The sampler's first instance parks alone (its pair stays with the
    // other two) and unparks into fresh room.
    let park = roles.lock().unwrap().prepare_park(holder(1)).unwrap();
    roles.lock().unwrap().commit(park);
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .unwrap();
    roles.lock().unwrap().commit(unpark);
    let stale = roles.lock().unwrap().stale(holder(1));
    let [
        Rebind::Instance {
            scratch: fresh,
            pair,
            ..
        },
    ] = stale.as_slice()
    else {
        panic!("one rebind: {stale:?}");
    };
    assert_eq!(*pair, pairs[0], "the pair did not move");
    scratch[0] = fresh.clone();
    check(&scratch);
}

/// The control: two instances told one run, or runs that overlap, are
/// caught; adjacent runs are not.
#[test]
fn two_instances_told_one_run_are_caught() {
    let run = ScratchPlace::Run {
        first: SPC,
        slots: 709,
    };
    let overlapping = ScratchPlace::Run {
        first: SPC + 708,
        slots: 2,
    };
    let adjacent = ScratchPlace::Run {
        first: SPC + 709,
        slots: 1,
    };
    assert!(disjoint(&[&run, &run]).is_err());
    assert!(disjoint(&[&run, &overlapping]).is_err());
    assert!(disjoint(&[&run, &adjacent]).is_ok());
}

/// A guest engine stating pieces of `run` chunks, `pieces` of them an
/// instance and no pair, for program `program`, its instance ids from
/// `next`.
fn runs_of(program: u32, run: u32, pieces: u32, next: u64) -> (ProgramId, GuestEngine) {
    let mut engine = GuestEngine::new(u64::from(run * pieces) * CHUNK, 0);
    engine.state.run = run;
    engine.next = next;
    (ProgramId(u64::from(program)), engine)
}

/// Binds an instance of `program` on `engine` for `by`.
fn bind_as(
    roles: &RolesHandle,
    engine: &mut GuestEngine,
    program: ProgramId,
    by: u128,
) -> anyhow::Result<InstanceId> {
    learn_program(roles, engine, program)?;
    let binding = InstanceBinding {
        program,
        ..InstanceBinding::default()
    };
    Ok(bind_instance(Some(roles), engine, &binding, Some(holder(by)), |_| false)?.id)
}

/// As [`bind_as`], for a bind whose refusal is final: its process gives
/// up, so it leaves the queue of guest binds waiting for room.
fn bind_or_leave(
    roles: &RolesHandle,
    engine: &mut GuestEngine,
    program: ProgramId,
    by: u128,
) -> anyhow::Result<InstanceId> {
    bind_as(roles, engine, program, by)
        .inspect_err(|_| roles.lock().unwrap().leave_guest_queue(holder(by)))
}

/// Each piece of `place`: its chunks' physical ids, and the segment each
/// lies in.
fn pieces_of(device: &Shared, place: &ScratchPlace) -> Vec<Vec<(u32, u32)>> {
    let ScratchPlace::Chunks { run, pieces } = place else {
        panic!("runs of whole chunks: {place:?}");
    };
    let pool = device.lock().unwrap();
    pieces
        .iter()
        .map(|&first| {
            let at = pool.pool().physical(first).expect("held");
            (at..at + run)
                .map(|physical| (physical, pool.pool().segments().of(physical)))
                .collect()
        })
        .collect()
}

/// A guest whose widest value takes 2.5 chunks lays its pieces in runs
/// of 3 (`GuestState::run`): each piece the bind takes is 3 adjacent
/// chunks inside one segment, the engine told each run's first. Two
/// instances' runs share no chunk.
#[test]
fn a_bind_needing_a_run_of_three_takes_three_adjacent_chunks_in_one_segment() {
    // Segments of 8 chunks: the null chunk at 0, the span at 52..64.
    let Pools { device, roles } = pools_of(facts_in(64, 8));
    let (program, mut engine) = runs_of(8, 3, 2, 0);
    let before = held(&device);
    let a = bind_as(&roles, &mut engine, program, 1).unwrap();
    let b = bind_as(&roles, &mut engine, program, 2).unwrap();
    assert_eq!(
        held(&device) - before,
        2 * 2 * 3 * CHUNK,
        "two runs of 3 an instance"
    );
    let mut seen = std::collections::BTreeSet::new();
    for binding in &engine.bindings {
        assert!(
            matches!(&binding.scratch, ScratchPlace::Chunks { run: 3, pieces } if pieces.len() == 2),
            "{:?}",
            binding.scratch
        );
        for piece in pieces_of(&device, &binding.scratch) {
            let (first, segment) = piece[0];
            assert_eq!(
                piece,
                (first..first + 3)
                    .map(|at| (at, segment))
                    .collect::<Vec<_>>(),
                "3 adjacent chunks inside segment {segment}"
            );
            for (at, _) in piece {
                assert!(seen.insert(at), "chunk {at} lies in two runs");
            }
        }
    }
    // Segment 0 holds the null chunk: runs 1..4 and 4..7, then 8..11 and
    // 11..14, never 7..10 across the segments' end.
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
    );
    roles.lock().unwrap().close_instance(a);
    roles.lock().unwrap().close_instance(b);
    assert_eq!(held(&device), before);
}

/// Closing an instance gives its runs back whole: the next bind of the
/// same runs takes the same chunks.
#[test]
fn release_then_rebind_reuses_the_run() {
    let Pools { device, roles } = pools_of(facts_in(64, 8));
    let (program, mut engine) = runs_of(8, 3, 1, 0);
    let before = held(&device);
    let id = bind_as(&roles, &mut engine, program, 1).unwrap();
    let first = engine.bindings[0].scratch.clone();
    roles.lock().unwrap().close_instance(id);
    assert_eq!(held(&device), before, "the whole run back");
    bind_as(&roles, &mut engine, program, 2).unwrap();
    assert_eq!(engine.bindings[1].scratch, first, "the same run again");
}

/// Fills `roles`' room with single-chunk instances and closes every other
/// one: the free chunks lie scattered, none beside another. The ids still
/// bound.
fn scatter(roles: &RolesHandle, device: &Shared) -> Vec<InstanceId> {
    let (program, mut singles) = runs_of(9, 1, 1, 1000);
    let mut ids = Vec::new();
    for by in 0.. {
        match bind_or_leave(roles, &mut singles, program, 500 + by) {
            Ok(id) => ids.push(id),
            Err(_) => break,
        }
    }
    assert!(ids.len() > 8, "{} singles", ids.len());
    let mut kept = Vec::new();
    for (k, id) in ids.into_iter().enumerate() {
        if k % 2 == 0 {
            roles.lock().unwrap().close_instance(id);
        } else {
            kept.push(id);
        }
    }
    assert_eq!(
        device.lock().unwrap().pool().free_chunk_runs(2),
        0,
        "none adjacent"
    );
    kept
}

/// Where only scattered single chunks are free, a bind needing a run of 3
/// is refused by name, stating the run and what was free: no such run,
/// and the free chunks there were. Nothing is taken.
#[test]
fn a_run_is_refused_by_name_where_only_scattered_chunks_are_free() {
    let Pools { device, roles } = pools_of(facts_in(64, 8));
    scatter(&roles, &device);
    let free = device.lock().unwrap().pool().free_chunks();
    assert!(
        free >= 3,
        "{free} free chunks: room enough, but not adjacent"
    );
    let before = held(&device);
    let (program, mut engine) = runs_of(8, 3, 1, 0);
    let error = bind_as(&roles, &mut engine, program, 1).expect_err("no run of 3 is free");
    let named = error.to_string();
    assert_eq!(
        short_of(&error),
        Some(&RolesError::GuestShort {
            what: format!(
                "an instance of {}'s scratch",
                super::FireKey {
                    program,
                    extents: BindExtents::default(),
                }
            ),
            bytes: 3 * CHUNK,
            room: super::GuestRoom::Runs {
                run: 3,
                pieces: 1,
                free_runs: 0,
                free_chunks: free,
                committable: 0,
            },
        }),
        "{named}"
    );
    assert!(
        named.contains("1 runs of 3 adjacent chunks")
            && named.contains("0 such runs were free")
            && named.contains("no segment is left to commit within the budget"),
        "{named}"
    );
    assert_eq!(held(&device), before, "the refused bind took nothing");
    assert!(engine.bindings.is_empty(), "the engine was not asked");
}

/// Many instances of runs 1, 2 and 3, bound and closed in a seeded random
/// order, then one bind of a run of 6. It lands, or is refused by name
/// with the free-run accounting, never a panic: where it is refused with
/// free chunks enough, the pool is fragmented, and only a compaction of
/// the bound runs would place it. The outcome is stated.
#[test]
fn mixed_runs_bound_and_closed_then_a_long_run_lands_or_is_refused_by_name() {
    let Pools { device, roles } = pools_of(facts_in(128, 8));
    let mut engines: Vec<(ProgramId, GuestEngine)> = (1..=3)
        .map(|run| runs_of(20 + run, run, 1, u64::from(run) * 100_000))
        .collect();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let mut bound: Vec<(InstanceId, u32)> = Vec::new();
    let (mut binds, mut refused) = (0u32, 0u32);
    for step in 0..2000u128 {
        if bound.is_empty() || next(3) != 0 {
            let which = next(3) as usize;
            let (program, engine) = &mut engines[which];
            match bind_or_leave(&roles, engine, *program, step) {
                Ok(id) => {
                    bound.push((id, which as u32 + 1));
                    binds += 1;
                }
                Err(error) => {
                    // Refused by name: a run of chunks, or (run 1, a
                    // chunk's slots) a run of slots, short.
                    assert!(
                        matches!(short_of(&error), Some(RolesError::GuestShort { .. })),
                        "{error}"
                    );
                    refused += 1;
                }
            }
        } else {
            let at = next(bound.len() as u64) as usize;
            let (id, _) = bound.swap_remove(at);
            roles.lock().unwrap().close_instance(id);
        }
    }
    // Half of what is bound closes, at random.
    for _ in 0..bound.len() / 2 {
        let at = next(bound.len() as u64) as usize;
        let (id, _) = bound.swap_remove(at);
        roles.lock().unwrap().close_instance(id);
    }
    let (free, runs) = {
        let pool = device.lock().unwrap();
        (pool.pool().free_chunks(), pool.pool().free_chunk_runs(6))
    };
    let (program, mut long) = runs_of(30, 6, 1, 900_000);
    let outcome = bind_as(&roles, &mut long, program, 9_999);
    match &outcome {
        Ok(_) => assert!(runs >= 1, "it landed in a free run of 6"),
        Err(error) => {
            let named = error.to_string();
            assert_eq!(
                short_of(&error).map(|error| match error {
                    RolesError::GuestShort { room, .. } => *room,
                    other => panic!("{other}"),
                }),
                Some(super::GuestRoom::Runs {
                    run: 6,
                    pieces: 1,
                    free_runs: 0,
                    free_chunks: free,
                    committable: 0,
                }),
                "{named}"
            );
            assert_eq!(runs, 0);
        }
    }
    eprintln!(
        "{binds} binds, {refused} refused by name, {} bound; then a run of 6: {} \
         ({free} free chunks, {runs} free runs of 6){}",
        bound.len(),
        if outcome.is_ok() {
            "landed"
        } else {
            "refused by name"
        },
        if outcome.is_err() && free >= 6 {
            ": fragmented, only a compaction would place it"
        } else {
            ""
        },
    );
}

/// A pool with one segment committed (and the top rung's span), filled,
/// and five segments left to commit: a guest bind needing a run of 3 finds
/// no free run and waits on a segment commit (`RolesError::GuestGrow`),
/// counted short in the pool so the walk's next step commits segment 1;
/// once it is committed the wait ends, the bind lands in it, and the count
/// is cleared. With no segment left to commit, the wait is refused by name.
#[tokio::test(flavor = "current_thread")]
async fn a_guest_bind_short_of_a_run_commits_the_next_segment_and_binds() {
    use crate::governor::PoolTarget;
    const ENGINE: usize = 61;
    let Pools { device, roles } = pools_of(facts_committed(64, 8, 1));
    assert_eq!(device.lock().unwrap().pool().prefix_segments(), 1);
    // Every committed chunk live state may take, a chunk an instance:
    // segment 0's beside the null chunk, and the top span's segment's
    // below the span.
    let (singles, mut single) = runs_of(9, 1, 1, 1000);
    for by in 0.. {
        if bind_or_leave(&roles, &mut single, singles, 500 + by).is_err() {
            break;
        }
    }
    assert_eq!(device.lock().unwrap().pool().free_chunk_runs(1), 0, "full");
    let live = device.lock().unwrap().pool().demand().live;
    let (program, mut engine) = runs_of(8, 3, 1, 0);
    let error = bind_as(&roles, &mut engine, program, 1).expect_err("no free run");
    let Some(RolesError::GuestGrow {
        room, prefix, run, ..
    }) = error.downcast_ref::<RolesError>()
    else {
        panic!("waits on a commit: {error}");
    };
    assert_eq!((*prefix, *run), (1, 3));
    assert!(
        matches!(room, super::GuestRoom::Runs { run: 3, pieces: 1, free_runs: 0, committable, .. } if *committable > 0),
        "{room:?}"
    );
    assert!(engine.bindings.is_empty(), "the engine was not asked");
    {
        let pool = device.lock().unwrap();
        let demand = pool.pool().demand();
        assert_eq!(demand.live, live, "nothing taken");
        assert_eq!(demand.wanted - demand.live, 3 * CHUNK, "counted short");
    }
    // The walk, the governor's target above what is mapped, commits the
    // next segment for it: the step a kv admission short of room waits on.
    let mut grow = crate::store::shared::SharedGrow::new(0, ENGINE, device.clone());
    let _ = grow.retarget(u64::from(u32::MAX));
    assert_eq!(
        grow.live_step_for_test(),
        Some(crate::store::shared::GrowStep::Commit { prefix: 2 })
    );
    let wait = super::grow_for_guest(&device, ENGINE, *prefix, *run);
    tokio::pin!(wait);
    assert!(
        futures::poll!(wait.as_mut()).is_pending(),
        "it waits on the commit"
    );
    // What the walk does once the engine backed it.
    device.lock().unwrap().committed(2);
    crate::store::shared::rung_changed(ENGINE);
    assert_eq!(wait.await, Ok(()));
    bind_as(&roles, &mut engine, program, 1).expect("it binds in the new segment");
    let piece = &pieces_of(&device, &engine.bindings[0].scratch)[0];
    assert_eq!(piece, &[(8, 1), (9, 1), (10, 1)], "segment 1's first run");
    let demand = device.lock().unwrap().pool().demand();
    assert_eq!(demand.wanted, demand.live, "no longer counted short");

    // Every segment committed and full: no segment is left to commit, and
    // the bind is refused by name.
    let Pools { device, roles } = pools_of(facts_in(64, 8));
    let (singles, mut single) = runs_of(9, 1, 1, 1000);
    for by in 0.. {
        if bind_or_leave(&roles, &mut single, singles, 500 + by).is_err() {
            break;
        }
    }
    let (program, mut engine) = runs_of(8, 3, 1, 0);
    let error = bind_as(&roles, &mut engine, program, 1).expect_err("no room left");
    let named = error.to_string();
    assert!(
        matches!(
            short_of(&error),
            Some(RolesError::GuestShort {
                room: super::GuestRoom::Runs { committable: 0, .. },
                ..
            })
        ),
        "{named}"
    );
    assert!(
        named.contains("no segment is left to commit within the budget"),
        "{named}"
    );
    let prefix = device.lock().unwrap().pool().prefix_segments();
    assert_eq!(
        super::grow_for_guest(&device, ENGINE, prefix, 3).await,
        Err(
            "no free run of 3 adjacent chunks, and no segment is left to commit within the \
             budget"
                .into()
        )
    );
}

/// More parked guest states come back than the device has room for: four
/// instances of 11.2 MB (past half a chunk's slots: a whole chunk each)
/// park, four of another program take their chunks, and every restore then
/// finds no room. Each unpark short of room the budget holds is a wait
/// (`RolesError::UnparkShort`), never a failure; as each
/// of the others completes, one waiting restore comes back, until all four
/// have.
#[test]
fn restores_past_the_free_room_wait_and_all_come_back() {
    let Pools { roles, .. } = pools(1 + 4 + 1 + 12);
    let mut engine = GuestEngine::new(11_200_000, PAIR);
    let mut ids = Vec::new();
    for by in 1..=4 {
        ids.push(bind(&roles, &mut engine, by).unwrap());
    }
    assert!(
        engine
            .bindings
            .iter()
            .all(|b| matches!(b.scratch, ScratchPlace::Chunks { run: 1, ref pieces } if pieces.len() == 1)),
        "a whole chunk each"
    );
    for by in 1..=4 {
        let park = roles
            .lock()
            .unwrap()
            .prepare_park(holder(by))
            .expect("it parks");
        roles.lock().unwrap().commit(park);
    }

    // Another program's four instances take the chunks meanwhile.
    let mut other = GuestEngine::new(11_200_000, 0);
    other.next = 100;
    learn_program(&roles, &mut other, ProgramId(8)).unwrap();
    let mut others = Vec::new();
    for by in 5..=8 {
        let bound = bind_instance(
            Some(&roles),
            &mut other,
            &InstanceBinding {
                program: ProgramId(8),
                ..InstanceBinding::default()
            },
            Some(holder(by)),
            |_| false,
        )
        .unwrap();
        others.push(bound.id);
    }
    let mut back = 0;
    for (n, by) in (1..=4).enumerate() {
        let short = roles
            .lock()
            .unwrap()
            .prepare_unpark(holder(by))
            .expect_err("no room: it waits");
        assert!(
            matches!(short, RolesError::UnparkShort(_)),
            "a wait, not a failure: {short}"
        );
        // One of the others completes: its chunk frees, and the waiting
        // restore comes back into it.
        roles.lock().unwrap().close_instance(others[n]);
        let unpark = roles
            .lock()
            .unwrap()
            .prepare_unpark(holder(by))
            .expect("room now")
            .expect("its items unpark");
        roles.lock().unwrap().commit(unpark);
        let stale = roles.lock().unwrap().stale(holder(by));
        for rebind in stale {
            roles.lock().unwrap().rebound(Owner::of(&rebind));
        }
        back += 1;
    }
    assert_eq!(back, 4, "every restore came back");
}

/// More binds than the device has room for, the pool whole: the bind past
/// the room is a wait (`RolesError::BindShort`), not a refusal, and a watch
/// taken before it ends once another instance closes; the bind then lands
/// in the room given back. A state no budget holds is still refused.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_past_the_room_waits_for_room_given_back() {
    let Pools { roles, .. } = pools(1 + 4 + 12);
    let mut engine = GuestEngine::new(11_200_000, 0);
    let mut ids = Vec::new();
    for by in 1..=4 {
        ids.push(bind(&roles, &mut engine, by).unwrap());
    }
    let watch = roles.lock().unwrap().room_watch();
    let short = bind(&roles, &mut engine, 5).expect_err("no room");
    assert!(
        matches!(
            short.downcast_ref::<RolesError>(),
            Some(RolesError::BindShort(_))
        ),
        "a wait, not a refusal: {short}"
    );
    let freed = watch.clone().freed();
    tokio::pin!(freed);
    assert!(
        futures::poll!(freed.as_mut()).is_pending(),
        "nothing given back yet"
    );
    roles.lock().unwrap().close_instance(ids[0]);
    freed.await;
    bind(&roles, &mut engine, 5).expect("room given back: it binds");

    // Twenty chunks of state: no budget of this pool holds it.
    let mut huge = GuestEngine::new(20 * CHUNK, 0);
    huge.next = 200;
    learn_program(&roles, &mut huge, ProgramId(9)).unwrap();
    let refused = bind_instance(
        Some(&roles),
        &mut huge,
        &InstanceBinding {
            program: ProgramId(9),
            ..InstanceBinding::default()
        },
        Some(holder(9)),
        |_| false,
    )
    .expect_err("refused");
    assert!(
        matches!(
            refused.downcast_ref::<RolesError>(),
            Some(RolesError::GuestShort { .. })
        ),
        "{refused}"
    );
}

/// Twelve processes, each binding two programs' instances in turn (the
/// first held while the second binds, as a prefill's sampler and a
/// decode's), under a pool whose room fits five instances. A bind short of
/// room its process holds part of suspends the process (its guest state
/// parked, given back); one holding none waits for room given back; a
/// parked process comes back once room is there, else waits. Every process
/// finishes, none is refused, and at the end the pool holds what it held
/// before: nothing a park, a restore or a close took stays held.
///
/// Each bind goes as the lane's register+bind does: its guest state taken
/// first, then its channels registered, then the instance bound. A bind
/// short of room is answered before it registers anything: no
/// registration is ever rolled back, however many binds park and come
/// back.
#[test]
fn processes_cycling_through_park_restore_and_finish_all_finish_and_give_back_all() {
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Stage {
        First,
        Second,
        Parked,
        Done,
    }
    let Pools { device, roles } = pools(1 + 5 + 12);
    let before = held(&device);
    let mut first = GuestEngine::new(11_200_000, 0);
    let mut second = GuestEngine::new(11_200_000, 0);
    second.next = 10_000;
    learn_program(&roles, &mut first, ProgramId(1)).unwrap();
    learn_program(&roles, &mut second, ProgramId(2)).unwrap();
    // Registrations made, and those rolled back because the bind after
    // them failed.
    let (registrations, rollbacks) = (std::cell::Cell::new(0), std::cell::Cell::new(0));
    let bind_of = |roles: &RolesHandle, engine: &mut GuestEngine, program: u64, by: u128| {
        let binding = InstanceBinding {
            program: ProgramId(program),
            ..InstanceBinding::default()
        };
        let taken = take_bind(Some(roles), engine, &binding, |_| false, Some(holder(by)))?;
        registrations.set(registrations.get() + 1);
        bind_taken(Some(roles), engine, &binding, Some(holder(by)), taken)
            .map(|bound| bound.id)
            .inspect_err(|_| rollbacks.set(rollbacks.get() + 1))
    };
    let n = 12;
    let mut stage = vec![Stage::First; n];
    let mut ids: Vec<Vec<InstanceId>> = vec![Vec::new(); n];
    let (mut parks, mut waits) = (0, 0);
    for _round in 0..10_000 {
        if stage.iter().all(|s| *s == Stage::Done) {
            break;
        }
        for p in 0..n {
            let by = p as u128 + 1;
            match stage[p] {
                Stage::Done => {}
                Stage::Parked => match {
                    let unparked = roles.lock().unwrap().prepare_unpark(holder(by));
                    unparked
                } {
                    Ok(Some(txn)) => {
                        roles.lock().unwrap().commit(txn);
                        let stale = roles.lock().unwrap().stale(holder(by));
                        for rebind in stale {
                            roles.lock().unwrap().rebound(Owner::of(&rebind));
                        }
                        stage[p] = Stage::Second;
                    }
                    Ok(None) => stage[p] = Stage::Second,
                    Err(RolesError::UnparkShort(_)) => waits += 1,
                    Err(other) => panic!("process {p}'s restore failed: {other}"),
                },
                Stage::First | Stage::Second => {
                    let (engine, program) = if stage[p] == Stage::First {
                        (&mut first, 1)
                    } else {
                        (&mut second, 2)
                    };
                    match bind_of(&roles, engine, program, by) {
                        Ok(id) => {
                            ids[p].push(id);
                            if stage[p] == Stage::First {
                                stage[p] = Stage::Second;
                            } else {
                                // Both bound: it runs to its end, closing both.
                                for id in ids[p].drain(..) {
                                    roles.lock().unwrap().close_instance(id);
                                }
                                stage[p] = Stage::Done;
                            }
                        }
                        Err(error) => {
                            // Short of room, or behind an earlier bind short
                            // of it: a wait either way.
                            assert!(
                                matches!(
                                    short_of_wait(&error),
                                    Some(RolesError::BindShort(_) | RolesError::GuestQueued { .. })
                                ) || matches!(
                                    error.downcast_ref::<RolesError>(),
                                    Some(RolesError::GuestQueued { .. })
                                ),
                                "process {p} refused: {error}"
                            );
                            if roles.lock().unwrap().holds_guest_state(holder(by)) {
                                // Suspended for room: it waits for its
                                // restore, not in the queue.
                                roles.lock().unwrap().leave_guest_queue(holder(by));
                                let park = roles
                                    .lock()
                                    .unwrap()
                                    .prepare_park(holder(by))
                                    .expect("its state parks");
                                roles.lock().unwrap().commit(park);
                                stage[p] = Stage::Parked;
                                parks += 1;
                            } else {
                                waits += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        stage.iter().all(|s| *s == Stage::Done),
        "every process finishes: {stage:?}"
    );
    assert!(
        parks > 0 && waits > 0,
        "the budget fit few: {parks} parks, {waits} waits"
    );
    assert_eq!(registrations.get(), 2 * n, "one registration a bind");
    assert_eq!(rollbacks.get(), 0, "no short bind registered anything");
    assert_eq!(held(&device), before, "all given back");
    assert_eq!(roles.lock().unwrap().device_bytes(), 0);
}

fn short_of_wait(error: &anyhow::Error) -> Option<&RolesError> {
    error.downcast_ref::<RolesError>()
}

/// The pair's place an instance's rebind names, from a restore's stale
/// list: the one rebind of `id`.
fn rebound_pair(roles: &RolesHandle, by: u128, id: InstanceId) -> ScratchPlace {
    let stale = roles.lock().unwrap().stale(holder(by));
    let found: Vec<ScratchPlace> = stale
        .iter()
        .filter_map(|rebind| match rebind {
            Rebind::Instance { id: at, pair, .. } if *at == id => Some(pair.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(found.len(), 1, "one rebind of {id:?}: {stale:?}");
    found[0].clone()
}

/// Parks and unparks `by`'s items, nothing copied.
fn park(roles: &RolesHandle, by: u128) {
    let park = roles
        .lock()
        .unwrap()
        .prepare_park(holder(by))
        .expect("it parks");
    roles.lock().unwrap().commit(park);
}

/// A pair parked with the instances that held it comes back for a new
/// bind of the same pair: its state is dead between fires, so it takes
/// fresh room on the device and the bind lands, never refused for it. The
/// parked instance, restored, is rebound to the pair where it lies now.
#[test]
fn a_new_bind_of_a_parked_pair_brings_the_pair_back_and_lands() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(TOP_P, 2 * CHUNK);
    let first = bind(&roles, &mut engine, 1).unwrap();
    park(&roles, 1);
    // Another process binds the same pair while it is parked.
    bind(&roles, &mut engine, 2).expect("the pair comes back for it, no refusal");
    let pair = engine.bindings[1].pair.clone();
    assert!(matches!(pair, ScratchPlace::Chunks { .. }), "{pair:?}");
    assert!(held(&device) > 0);
    // The parked instance comes back, rebound to the pair where it lies.
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .expect("it unparks");
    roles.lock().unwrap().commit(unpark);
    assert_eq!(rebound_pair(&roles, 1, first), pair);
}

/// A restore's rebind of an instance names its pair's place: where the
/// pair stayed resident (another process's instance held it), and where it
/// was parked too and came back with it. Two processes restoring a pair
/// parked with both: the second waits for the first's move to land, then
/// is rebound to it, never told the pair lies nowhere.
#[test]
fn a_restore_rebinds_an_instance_with_its_pair_s_place() {
    // The pair resident: process 2's instance holds it on the device.
    let Pools { roles, .. } = pools(64);
    let mut engine = GuestEngine::new(TOP_P, 2 * CHUNK);
    let first = bind(&roles, &mut engine, 1).unwrap();
    bind(&roles, &mut engine, 2).unwrap();
    let resident = engine.bindings[0].pair.clone();
    park(&roles, 1);
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .unwrap();
    roles.lock().unwrap().commit(unpark);
    assert_eq!(
        rebound_pair(&roles, 1, first),
        resident,
        "the resident pair"
    );

    // The pair parked with its only process, and back with it.
    let Pools { roles, .. } = pools(64);
    let mut engine = GuestEngine::new(TOP_P, 2 * CHUNK);
    let only = bind(&roles, &mut engine, 1).unwrap();
    park(&roles, 1);
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .unwrap();
    roles.lock().unwrap().commit(unpark);
    let back = rebound_pair(&roles, 1, only);
    assert!(matches!(back, ScratchPlace::Chunks { .. }), "{back:?}");

    // Parked with two processes: one restore moves it; the other waits.
    let Pools { roles, .. } = pools(64);
    let mut engine = GuestEngine::new(TOP_P, 2 * CHUNK);
    let a = bind(&roles, &mut engine, 1).unwrap();
    let b = bind(&roles, &mut engine, 2).unwrap();
    park(&roles, 2);
    park(&roles, 1);
    let first_back = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .unwrap();
    assert!(
        matches!(
            roles.lock().unwrap().prepare_unpark(holder(2)),
            Err(RolesError::PairParked(_))
        ),
        "its pair is moving with the other restore"
    );
    roles.lock().unwrap().commit(first_back);
    let pair = rebound_pair(&roles, 1, a);
    assert!(matches!(pair, ScratchPlace::Chunks { .. }), "{pair:?}");
    let second = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(2))
        .unwrap()
        .unwrap();
    roles.lock().unwrap().commit(second);
    assert_eq!(
        rebound_pair(&roles, 2, b),
        pair,
        "rebound to where it landed"
    );
}

/// Fills the committed room with single-chunk guest states for holders
/// from 500; the bind refused for want of room leaves the queue it joined.
fn fill_with_singles(roles: &RolesHandle) -> Vec<InstanceId> {
    let (singles, mut single) = runs_of(9, 1, 1, 1000);
    let mut ids = Vec::new();
    for by in 0.. {
        match bind_as(roles, &mut single, singles, 500 + by) {
            Ok(id) => ids.push(id),
            Err(_) => {
                roles.lock().unwrap().leave_guest_queue(holder(500 + by));
                break;
            }
        }
    }
    ids
}

/// The free chunks and the live demand: what a refused bind must leave as
/// it found it.
fn free_and_live(device: &Shared) -> (u32, u64) {
    let pool = device.lock().unwrap();
    (pool.pool().free_chunks(), pool.pool().demand().live)
}

/// guest_mix's warm-up on the M1 Max: a guest bind needing two runs of four
/// adjacent chunks waits on a segment commit while smaller guest binds keep
/// arriving. Before the queue, each smaller bind took the fresh segment's
/// chunks a few at a time, the long runs never stood free, and the long
/// bind waited for ever. Now it heads the queue: the smaller binds wait
/// behind it taking nothing (the free count does not drift), it binds in
/// the committed segment, and they follow in arrival order.
#[tokio::test(flavor = "current_thread")]
async fn a_long_guest_bind_heads_the_queue_and_binds_after_the_commit_while_smaller_binds_wait() {
    use crate::governor::PoolTarget;
    const ENGINE: usize = 62;
    let Pools { device, roles } = pools_of(facts_committed(64, 8, 1));
    fill_with_singles(&roles);
    assert_eq!(device.lock().unwrap().pool().free_chunk_runs(1), 0, "full");
    let before = free_and_live(&device);

    let (long, mut long_engine) = runs_of(8, 4, 2, 0);
    let error = bind_as(&roles, &mut long_engine, long, 1).expect_err("no run of 4");
    assert!(
        matches!(
            error.downcast_ref::<RolesError>(),
            Some(RolesError::GuestGrow { run: 4, .. })
        ),
        "waits on a commit: {error}"
    );
    assert_eq!(roles.lock().unwrap().guest_queue(), vec![holder(1)], "it heads the queue");
    let mut grow = crate::store::shared::SharedGrow::new(0, ENGINE, device.clone());
    let _ = grow.retarget(u64::from(u32::MAX));
    assert_eq!(
        grow.live_step_for_test(),
        Some(crate::store::shared::GrowStep::Commit { prefix: 2 }),
        "the walk commits for it"
    );

    // Smaller guest binds arriving meanwhile wait behind it, taking nothing.
    let (small, mut small_engine) = runs_of(10, 1, 1, 2000);
    for by in 0..5 {
        let error = bind_as(&roles, &mut small_engine, small, 100 + by).expect_err("behind it");
        assert!(
            matches!(
                error.downcast_ref::<RolesError>(),
                Some(RolesError::GuestQueued { head }) if *head == holder(1)
            ),
            "{error}"
        );
    }
    assert!(small_engine.bindings.is_empty(), "the engine was not asked");
    assert_eq!(free_and_live(&device), before, "the free count does not drift");

    // The commit: the head binds in the fresh segment, two runs of four.
    device.lock().unwrap().committed(2);
    bind_as(&roles, &mut long_engine, long, 1).expect("the head binds");
    let pieces = pieces_of(&device, &long_engine.bindings[0].scratch);
    assert_eq!(pieces.len(), 2);
    assert!(pieces.iter().all(|piece| piece.len() == 4 && piece.iter().all(|&(_, s)| s == 1)));
    assert_eq!(
        roles.lock().unwrap().guest_queue(),
        (0..5).map(|by| holder(100 + by)).collect::<Vec<_>>(),
        "the binds behind it, in arrival order"
    );

    // The next segment: they bind, in order, and the queue empties.
    device.lock().unwrap().committed(3);
    for by in 0..5 {
        bind_as(&roles, &mut small_engine, small, 100 + by).expect("in its turn");
    }
    assert!(roles.lock().unwrap().guest_queue().is_empty());
    let demand = device.lock().unwrap().pool().demand();
    assert_eq!(demand.wanted, demand.live, "no need is counted short any more");
}

/// Free chunks enough for the head's slots, but scattered by the binds
/// before it, so none of its runs stand free: the walk commits a segment
/// for its runs, where counting its slots alone (the commit-ahead margin
/// covered) it would commit none and the head would wait for ever.
#[test]
fn the_walk_commits_for_the_head_s_runs_where_scattered_free_chunks_cover_its_slots() {
    use crate::governor::PoolTarget;
    const ENGINE: usize = 63;
    let Pools { device, roles } = pools_of(facts_committed(64, 8, 4));
    let ids = fill_with_singles(&roles);
    for (k, id) in ids.into_iter().enumerate() {
        if k % 2 == 0 {
            roles.lock().unwrap().close_instance(id);
        }
    }
    let (free, runs) = {
        let pool = device.lock().unwrap();
        (pool.pool().free_chunks(), pool.pool().free_chunk_runs(4))
    };
    assert!(free >= 13, "{free} free chunks: the head's 8 and the 4 ahead");
    assert_eq!(runs, 0, "none of four adjacent");
    let mut grow = crate::store::shared::SharedGrow::new(0, ENGINE, device.clone());
    let _ = grow.retarget(u64::from(u32::MAX));
    assert_eq!(grow.live_step_for_test(), None, "the scattered room covers the ahead");

    let (long, mut long_engine) = runs_of(8, 4, 2, 0);
    let error = bind_as(&roles, &mut long_engine, long, 1).expect_err("no run of 4");
    assert!(
        matches!(error.downcast_ref::<RolesError>(), Some(RolesError::GuestGrow { .. })),
        "{error}"
    );
    assert_eq!(
        grow.live_step_for_test(),
        Some(crate::store::shared::GrowStep::Commit { prefix: 5 }),
        "the walk commits for the head's runs"
    );
    roles.lock().unwrap().leave_guest_queue(holder(1));
    assert_eq!(grow.live_step_for_test(), None, "with no head, nothing more");
}

/// The head's process ends while it waits: dropping its wait leaves the
/// queue (one place, however the wait ends), the bind behind it is told,
/// and binds in room that was free all along; a bind of the same process
/// queued again would find no stale head.
#[tokio::test(flavor = "current_thread")]
async fn a_head_whose_process_ends_while_waiting_lets_the_next_bind() {
    const ENGINE: usize = 64;
    let Pools { device, roles } = pools_of(facts_committed(64, 8, 1));
    let ids = fill_with_singles(&roles);
    // One single chunk free again: room for a small bind, not the long.
    roles.lock().unwrap().close_instance(ids[0]);
    let before = free_and_live(&device);

    let waits = |by: u128, program: ProgramId, engine: std::rc::Rc<std::cell::RefCell<GuestEngine>>| {
        let device = device.clone();
        super::bind_when_resident(
            holder(by),
            {
                let roles = roles.clone();
                move || {
                    let outcome = bind_as(&roles, &mut engine.borrow_mut(), program, by);
                    std::future::ready(outcome)
                }
            },
            || std::future::pending::<Result<(), String>>(),
            move |prefix, run| {
                let device = device.clone();
                async move { super::grow_for_guest(&device, ENGINE, prefix, run).await }
            },
            {
                let roles = roles.clone();
                move || Some(roles.lock().unwrap().room_watch())
            },
            |_, _| false,
            |_, _| false,
            |_, _| std::future::pending::<Result<(), String>>(),
            {
                let roles = roles.clone();
                move || {
                    let roles = roles.clone();
                    async move { super::guest_turn(&roles, holder(by)).await }
                }
            },
            {
                let roles = roles.clone();
                move || roles.lock().unwrap().leave_guest_queue(holder(by))
            },
        )
    };
    let (long, long_engine) = runs_of(8, 4, 2, 0);
    let (small, small_engine) = runs_of(10, 1, 1, 2000);
    let small_engine = std::rc::Rc::new(std::cell::RefCell::new(small_engine));
    let mut head = Box::pin(waits(1, long, std::rc::Rc::new(std::cell::RefCell::new(long_engine))));
    let next = waits(2, small, small_engine.clone());
    tokio::pin!(next);
    assert!(futures::poll!(head.as_mut()).is_pending(), "the head waits on a commit");
    assert!(futures::poll!(next.as_mut()).is_pending(), "the next waits behind it");
    assert_eq!(roles.lock().unwrap().guest_queue(), vec![holder(1), holder(2)]);
    assert_eq!(free_and_live(&device), before, "nothing taken");

    // The head's process ends: its wait is dropped.
    drop(head);
    assert_eq!(roles.lock().unwrap().guest_queue(), vec![holder(2)], "it left, once");
    next.await.expect("the next binds in the free chunk");
    assert_eq!(small_engine.borrow().bindings.len(), 1);
    assert!(roles.lock().unwrap().guest_queue().is_empty());
    let demand = device.lock().unwrap().pool().demand();
    assert_eq!(demand.wanted, demand.live, "the head's need is no longer counted");
}

/// While the head waits for runs of four, a bind taking whole runs at least
/// as long breaks none of them into shorter ones: it passes the head,
/// taking room where there is some, and where there is none it queues
/// behind it without touching the head's need.
#[test]
fn a_bind_of_runs_as_long_as_the_head_s_passes_it_and_leaves_its_need() {
    let Pools { device, roles } = pools_of(facts_committed(64, 8, 1));
    fill_with_singles(&roles);
    let (long, mut long_engine) = runs_of(8, 4, 2, 0);
    bind_as(&roles, &mut long_engine, long, 1).expect_err("the head waits");
    assert_eq!(device.lock().unwrap().guest_head(), Some((4, 2)));

    // As long, no room: it waits behind the head; the head's need stays.
    let (four, mut four_engine) = runs_of(11, 4, 1, 3000);
    let error = bind_as(&roles, &mut four_engine, four, 200).expect_err("no room");
    assert!(
        matches!(error.downcast_ref::<RolesError>(), Some(RolesError::GuestGrow { .. })),
        "it passed the head and found no room: {error}"
    );
    assert_eq!(device.lock().unwrap().guest_head(), Some((4, 2)), "the head's need stands");
    assert_eq!(roles.lock().unwrap().guest_queue(), vec![holder(1), holder(200)]);

    // Shorter pieces queue behind it, taking nothing.
    let (two, mut two_engine) = runs_of(12, 2, 1, 4000);
    let error = bind_as(&roles, &mut two_engine, two, 300).expect_err("behind the head");
    assert!(
        matches!(error.downcast_ref::<RolesError>(), Some(RolesError::GuestQueued { .. })),
        "{error}"
    );

    // A commit: the head binds in it, and the queue moves on.
    device.lock().unwrap().committed(2);
    bind_as(&roles, &mut long_engine, long, 1).expect("the head binds");
    assert_eq!(
        device.lock().unwrap().guest_head(),
        Some((4, 1)),
        "the next head's need counts at once"
    );
    assert_eq!(roles.lock().unwrap().guest_queue(), vec![holder(200), holder(300)]);
}

/// A bind reusing a pair's state another process's instance holds (taken
/// by `take_bind`, the roles' lock let go for the engine's bind) while that
/// process parks: the bind attaches to the pair, so it stays on the device
/// and is not parked or freed under the bind ([`super::Item::pending`],
/// counted from the take, let go at `bound` or `untake`); the bind lands
/// on it. Parked, it would have been refused at `bound` as moved off the
/// device.
#[test]
fn a_pair_a_bind_reuses_does_not_park_under_it() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(TOP_P, 2 * CHUNK);
    let first = bind(&roles, &mut engine, 1).unwrap();
    let pair = super::FireKey {
        program: PROGRAM,
        extents: BindExtents::default(),
    };
    let was = roles.lock().unwrap().slots(Owner::Pair(pair));
    assert!(!was.is_empty());
    let binding = InstanceBinding {
        program: PROGRAM,
        ..InstanceBinding::default()
    };
    let taken = take_bind(Some(&roles), &mut engine, &binding, |_| false, None).unwrap();
    // The pair's only holder parks while the bind is under way.
    let parked = roles.lock().unwrap().prepare_park(holder(1));
    if let Some(park) = parked {
        roles.lock().unwrap().commit(park);
    }
    assert_eq!(
        roles.lock().unwrap().slots(Owner::Pair(pair)),
        was,
        "the pair a bind reuses moved under it"
    );
    let second = bind_taken(Some(&roles), &mut engine, &binding, Some(holder(2)), taken)
        .expect("the bind lands on the pair it took")
        .id;
    assert_eq!(engine.bindings[1].pair, engine.bindings[0].pair);
    roles.lock().unwrap().close_instance(second);
    roles.lock().unwrap().close_instance(first);
    let _ = held(&device);
}

/// Gemma 4 E4B on the M5 under the fence: room a guest bind waits for comes
/// free only as the fire last naming it retires, and that retire told no
/// one, so with no segment to commit the bind waited for ever. The pool
/// now tells as the room actually comes free: an instance's run of four,
/// let go while a fire at epoch 5 names it, does not bind the waiter at the
/// close (it is fenced); `retire_through(5)` wakes it, and it binds there,
/// no segment committed.
#[tokio::test(flavor = "current_thread")]
async fn a_guest_bind_waiting_for_fenced_room_binds_as_the_fire_retires() {
    const ENGINE: usize = 65;
    let Pools { device, roles } = pools_of(facts_committed(64, 8, 1));
    // A run of four, the only one the bind could take, then every other
    // committed chunk filled.
    let (held_program, mut held_engine) = runs_of(13, 4, 1, 5000);
    let held = bind_as(&roles, &mut held_engine, held_program, 7).expect("a run of four");
    fill_with_singles(&roles);
    assert_eq!(device.lock().unwrap().pool().free_chunk_runs(4), 0);
    let prefix = device.lock().unwrap().pool().prefix_segments();

    let (long, long_engine) = runs_of(8, 4, 1, 0);
    let long_engine = std::rc::Rc::new(std::cell::RefCell::new(long_engine));
    let wait = {
        let (roles, device, engine) = (roles.clone(), device.clone(), long_engine.clone());
        super::bind_when_resident(
            holder(1),
            {
                let roles = roles.clone();
                move || std::future::ready(bind_as(&roles, &mut engine.borrow_mut(), long, 1))
            },
            || std::future::pending::<Result<(), String>>(),
            move |prefix, run| {
                let device = device.clone();
                async move { super::grow_for_guest(&device, ENGINE, prefix, run).await }
            },
            {
                let roles = roles.clone();
                move || Some(roles.lock().unwrap().room_watch())
            },
            {
                let roles = roles.clone();
                move |run, pieces| roles.lock().unwrap().free_runs(run) >= pieces
            },
            |_, _| false,
            |_, _| std::future::pending::<Result<(), String>>(),
            {
                let roles = roles.clone();
                move || {
                    let roles = roles.clone();
                    async move { super::guest_turn(&roles, holder(1)).await }
                }
            },
            {
                let roles = roles.clone();
                move || roles.lock().unwrap().leave_guest_queue(holder(1))
            },
        )
    };
    tokio::pin!(wait);
    assert!(futures::poll!(wait.as_mut()).is_pending(), "it waits for room");

    // A fire at epoch 5 names the held instance; it closes while that fire
    // is in flight: its run is fenced, not free.
    roles.lock().unwrap().touch(held, 5);
    roles.lock().unwrap().close_instance(held);
    assert_eq!(device.lock().unwrap().pool().free_chunk_runs(4), 0, "fenced");
    assert!(futures::poll!(wait.as_mut()).is_pending(), "the close does not bind it");
    assert!(long_engine.borrow().bindings.is_empty());

    // The fire retires: the run is free, the pool tells, and it binds there.
    device.lock().unwrap().pool_mut().retire_through(5);
    let std::task::Poll::Ready(bound) = futures::poll!(wait.as_mut()) else {
        panic!("the retire did not wake it: room came free and no one was told");
    };
    bound.expect("it binds as the fire retires");
    assert_eq!(long_engine.borrow().bindings.len(), 1);
    assert_eq!(
        device.lock().unwrap().pool().prefix_segments(),
        prefix,
        "no segment committed"
    );
    assert!(roles.lock().unwrap().guest_queue().is_empty());
}
