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
    BindExtents, BoundInstance, Engine, FrameSubmission, FrameTicket, GuestState,
    InstanceBinding, InstanceId, LoadRequest, Loaded, ProgramFacts, ProgramId, Rebind,
    ScratchPlace,
};

use super::{Owner, Roles, RolesError, RolesHandle, bind_instance, learn_program};
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
    let facts = ChunkFacts {
        chunk_bytes: CHUNK,
        total_chunks: total,
        segment_chunks: segment,
        segments: SegmentFacts {
            committed: segments,
            max: segments,
            growth: Growth::Fixed,
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

    fn load(&mut self, _request: LoadRequest) -> ::engine::Result<Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(&mut self, _frame: &FrameSubmission) -> ::engine::Result<FrameTicket> {
        unreachable!("nothing fires here")
    }

    fn program_facts(&self, _program: ProgramId) -> ::engine::Result<ProgramFacts> {
        Ok(ProgramFacts::default())
    }

    fn guest_state(&self, _program: ProgramId, _extents: &BindExtents) -> ::engine::Result<GuestState> {
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
/// model's carve alone. The negative control is the old per-fire term:
/// that program's bytes counted in the top rung's activation (12 chunks,
/// the whole span) leave no room for the 2048 rung, and its fires were cut
/// to 256.
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
        let limit = pool.pool().live_limit();
        assert_eq!(limit, 64 - 12, "the top rung's span lies above it");
        for chunk in chunks(&binding.scratch).into_iter().chain(chunks(&binding.pair)) {
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

    // Control: the old per-fire term, one member's `fixed + a_lane` added
    // to every rung's activation, as the pool counted it before.
    let mut pool = device.lock().unwrap();
    pool.standing_stated(PAIR + TOP_P);
    assert_eq!(pool.pool().act_chunks(2048), 12 + 1, "past the span");
    assert_eq!(pool.widest_fit(), Some(256), "r_top unreachable");
    assert_eq!(pool.rung_for(2048), 256, "the prefill cut to 256");
}

/// 256 bound instances of a program holding 11.2 MB each are counted at
/// their binds, a run of 2735 slots each (one a chunk: two do not fit one)
/// beside the pair's, and take state's room as any state does; the 257th
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
    let state = (256 * slots(11_200_000) + slots(PAIR)) * SLOT;
    assert_eq!(held(&device) - before, state, "256 scratch runs and the pair's");
    assert_eq!(roles.lock().unwrap().device_bytes(), state, "state, counted");
    let error = bind(&roles, &mut engine, 257).expect_err("the pool holds no more");
    let named = error.to_string();
    assert!(
        matches!(
            error.downcast_ref::<RolesError>(),
            Some(RolesError::GuestShort {
                bytes: 11_200_000,
                ..
            })
        ),
        "{named}"
    );
    assert!(named.contains("scratch") && named.contains("program 7"), "{named}");
    assert_eq!(held(&device) - before, state, "the refused bind took nothing");
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
    assert_eq!(held(&device) - before, 2 * CHUNK + run, "the pair stays with its last");
    assert!(!roles.lock().unwrap().slots(Owner::Pair(super::FireKey {
        program: PROGRAM,
        extents: BindExtents::default(),
    }))
    .is_empty());
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
    assert!(park.copies().is_empty(), "nothing copied: {:?}", park.copies());
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
    assert!(matches!(scratch, ScratchPlace::Chunks { .. }) && fresh.len() == 2, "{scratch:?}");
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
    assert!(sampler.bindings.iter().all(|binding| binding.pair == pairs[0]));
    assert!(beam.bindings.iter().all(|binding| binding.pair == pairs[1]));
    assert!(matches!(scratch[0], ScratchPlace::Run { .. }));
    assert!(matches!(scratch[3], ScratchPlace::Chunks { .. }));
    assert!(matches!(pairs[0], ScratchPlace::Chunks { .. }));
    assert!(matches!(pairs[1], ScratchPlace::Run { .. }));
    let limit = device.lock().unwrap().pool().live_limit();
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
        match bind_as(roles, &mut singles, program, 500 + by) {
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
        error.downcast_ref::<RolesError>(),
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
            },
        }),
        "{named}"
    );
    assert!(
        named.contains("1 runs of 3 adjacent chunks") && named.contains("0 such runs were free"),
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
            match bind_as(&roles, engine, *program, step) {
                Ok(id) => {
                    bound.push((id, which as u32 + 1));
                    binds += 1;
                }
                Err(error) => {
                    // Refused by name: a run of chunks, or (run 1, a
                    // chunk's slots) a run of slots, short.
                    assert!(
                        matches!(
                            error.downcast_ref::<RolesError>(),
                            Some(RolesError::GuestShort { .. })
                        ),
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
                error.downcast_ref::<RolesError>().map(|error| match error {
                    RolesError::GuestShort { room, .. } => *room,
                    other => panic!("{other}"),
                }),
                Some(super::GuestRoom::Runs {
                    run: 6,
                    pieces: 1,
                    free_runs: 0,
                    free_chunks: free,
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
