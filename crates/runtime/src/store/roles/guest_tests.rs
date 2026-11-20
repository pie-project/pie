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
                resident: per_instance,
                scratch: 0,
                scratch_arms: 1,
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
    /// A test engine's copies land inside the call: nothing to wait for.
    fn settle_copies(&mut self) -> ::engine::Result<()> {
        Ok(())
    }

    /// A test engine's copies complete inside the call: their sources are
    /// read by the time it returns.
    fn copy_slots_landing(
        &mut self,
        copy: &::engine::chunk::SlotCopy,
        _token: u64,
    ) -> ::engine::Result<bool> {
        self.copy_slots(copy)?;
        Ok(true)
    }

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
        _class: eta_ir::registry::GeometryClass,
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
/// up, so its need is no longer counted.
fn bind_or_leave(
    roles: &RolesHandle,
    engine: &mut GuestEngine,
    program: ProgramId,
    by: u128,
) -> anyhow::Result<InstanceId> {
    bind_as(roles, engine, program, by)
        .inspect_err(|_| roles.lock().unwrap().leave_guest_wait(holder(by)))
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
                            // Short of room: a wait.
                            assert!(
                                matches!(short_of_wait(&error), Some(RolesError::BindShort(_))),
                                "process {p} refused: {error}"
                            );
                            if roles.lock().unwrap().holds_guest_state(holder(by)) {
                                // Suspended for room: it waits for its
                                // restore, its need not counted.
                                roles.lock().unwrap().leave_guest_wait(holder(by));
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
/// from 500; the bind refused for want of room no longer counts its need.
fn fill_with_singles(roles: &RolesHandle) -> Vec<InstanceId> {
    let (singles, mut single) = runs_of(9, 1, 1, 1000);
    let mut ids = Vec::new();
    for by in 0.. {
        match bind_as(roles, &mut single, singles, 500 + by) {
            Ok(id) => ids.push(id),
            Err(_) => {
                roles.lock().unwrap().leave_guest_wait(holder(500 + by));
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
async fn a_long_guest_bind_s_need_drives_the_walk_while_smaller_binds_short_of_room_leave_it() {
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
    assert_eq!(device.lock().unwrap().guest_head(), Some((4, 2)), "its need is counted");
    let mut grow = crate::store::shared::SharedGrow::new(0, ENGINE, device.clone());
    let _ = grow.retarget(u64::from(u32::MAX));
    assert_eq!(
        grow.live_step_for_test(),
        Some(crate::store::shared::GrowStep::Commit { prefix: 2 }),
        "the walk commits for it"
    );

    // Smaller guest binds short of room meanwhile wait on the commit too,
    // taking nothing and leaving the long bind's need as it is.
    let (small, mut small_engine) = runs_of(10, 1, 1, 2000);
    for by in 0..5 {
        let error = bind_as(&roles, &mut small_engine, small, 100 + by).expect_err("no room");
        assert!(
            matches!(error.downcast_ref::<RolesError>(), Some(RolesError::GuestGrow { .. })),
            "{error}"
        );
    }
    assert!(small_engine.bindings.is_empty(), "the engine was not asked");
    assert_eq!(device.lock().unwrap().guest_head(), Some((4, 2)), "its need stands");

    // The commit: the long bind binds in the fresh segment, two runs of
    // four, and its need is no longer counted.
    device.lock().unwrap().committed(2);
    bind_as(&roles, &mut long_engine, long, 1).expect("it binds");
    let pieces = pieces_of(&device, &long_engine.bindings[0].scratch);
    assert_eq!(pieces.len(), 2);
    assert!(pieces.iter().all(|piece| piece.len() == 4 && piece.iter().all(|&(_, s)| s == 1)));
    assert_eq!(device.lock().unwrap().guest_head(), None);

    // The next segment: the small binds bind.
    device.lock().unwrap().committed(3);
    for by in 0..5 {
        bind_as(&roles, &mut small_engine, small, 100 + by).expect("room came");
    }
    let demand = device.lock().unwrap().pool().demand();
    assert_eq!(demand.wanted, demand.live, "no need is counted short any more");
    let _ = before;
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
    roles.lock().unwrap().leave_guest_wait(holder(1));
    assert_eq!(grow.live_step_for_test(), None, "with no head, nothing more");
}

/// A long bind waits on a commit while one chunk stands free: a small
/// bind takes it at once, waiting behind nothing (the queue that held it
/// back is gone). The long bind's process then ends: dropping its wait
/// counts its need no more (one place, however the wait ends).
#[tokio::test(flavor = "current_thread")]
async fn a_small_bind_takes_free_room_while_a_long_one_waits_and_a_dropped_wait_s_need_is_cleared() {
    const ENGINE: usize = 64;
    let Pools { device, roles } = pools_of(facts_committed(64, 8, 1));
    let ids = fill_with_singles(&roles);
    // One single chunk free again: room for a small bind, not the long.
    roles.lock().unwrap().close_instance(ids[0]);

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
                move || roles.lock().unwrap().leave_guest_wait(holder(by))
            },
            |_| {},
        )
    };
    let (long, long_engine) = runs_of(8, 4, 2, 0);
    let (small, small_engine) = runs_of(10, 1, 1, 2000);
    let small_engine = std::rc::Rc::new(std::cell::RefCell::new(small_engine));
    let mut long_wait = Box::pin(waits(1, long, std::rc::Rc::new(std::cell::RefCell::new(long_engine))));
    assert!(futures::poll!(long_wait.as_mut()).is_pending(), "the long bind waits on a commit");
    assert_eq!(device.lock().unwrap().guest_head(), Some((4, 2)));
    waits(2, small, small_engine.clone())
        .await
        .expect("the small bind takes the free chunk at once");
    assert_eq!(small_engine.borrow().bindings.len(), 1);

    // The long bind's process ends: its wait is dropped.
    drop(long_wait);
    assert_eq!(device.lock().unwrap().guest_head(), None, "its need is no longer counted");
    let demand = device.lock().unwrap().pool().demand();
    assert_eq!(demand.wanted, demand.live);
}

/// A guest bind's wait under `bind_when_resident` for `by`, binding
/// `program` on `engine`, the planner's serve played by `serve` (what the
/// planner would do for it: drop cache or park a resident), counting its
/// calls in `asked`.
fn waiting_bind<'a>(
    roles: &'a RolesHandle,
    device: &'a Shared,
    engine_id: usize,
    by: u128,
    program: ProgramId,
    engine: std::rc::Rc<std::cell::RefCell<GuestEngine>>,
    serve: std::rc::Rc<dyn Fn()>,
    asked: std::rc::Rc<std::cell::Cell<u32>>,
) -> impl std::future::Future<Output = anyhow::Result<InstanceId>> + 'a {
    super::bind_when_resident(
        holder(by),
        move || std::future::ready(bind_as(roles, &mut engine.borrow_mut(), program, by)),
        || std::future::pending::<Result<(), String>>(),
        move |prefix, run| super::grow_for_guest(device, engine_id, prefix, run),
        move || Some(roles.lock().unwrap().room_watch()),
        move |run, pieces| roles.lock().unwrap().free_runs(run) >= pieces,
        |_, _| false,
        move |_, _| {
            asked.set(asked.get() + 1);
            serve();
            std::future::ready(Ok::<(), String>(()))
        },
        move || roles.lock().unwrap().leave_guest_wait(holder(by)),
        |_| {},
    )
}

/// At the governor's target the walk commits nothing, however many
/// segments the pool's own limits leave uncommitted: a guest bind short of
/// runs is short of room the planner makes (BindShort), not a commit's to
/// wait on (GuestGrow). With the target past what is mapped, it waits on
/// the commit.
#[test]
fn at_the_target_a_guest_bind_short_of_runs_goes_to_the_planner() {
    use crate::governor::PoolTarget;
    const ENGINE: usize = 66;
    let Pools { device, roles } = pools_of(facts_committed(64, 8, 1));
    let mut grow = crate::store::shared::SharedGrow::new(0, ENGINE, device.clone());
    fill_with_singles(&roles);
    let (long, mut long_engine) = runs_of(8, 4, 2, 0);
    let error = bind_as(&roles, &mut long_engine, long, 1).expect_err("no room");
    assert!(
        matches!(short_of_wait(&error), Some(RolesError::BindShort(_))),
        "at the target it is the planner's: {error}"
    );
    roles.lock().unwrap().leave_guest_wait(holder(1));
    let _ = grow.retarget(u64::from(u32::MAX));
    let error = bind_as(&roles, &mut long_engine, long, 1).expect_err("no room yet");
    assert!(
        matches!(error.downcast_ref::<RolesError>(), Some(RolesError::GuestGrow { .. })),
        "past it, a commit's: {error}"
    );
}

/// GPU's 500-concurrent wedge: the device full at the governor's target, a
/// guest head short of runs, other guest binds queued behind it. The
/// planner is asked; its serve (a resident parked) frees the runs; the head
/// binds; the bind behind it follows. Before, the head waited on a commit
/// that could not come and no one asked the planner.
#[tokio::test(flavor = "current_thread")]
async fn a_guest_head_at_the_target_is_served_by_the_planner_and_the_next_bind_follows() {
    const ENGINE: usize = 67;
    let Pools { device, roles } = pools_of(facts_committed(64, 8, 1));
    let _grow = crate::store::shared::SharedGrow::new(0, ENGINE, device.clone());
    // A resident the planner may park, holding two runs of four, then
    // every other committed chunk filled.
    let (held_program, mut held_engine) = runs_of(13, 4, 2, 5000);
    let victim = bind_as(&roles, &mut held_engine, held_program, 7).expect("a resident");
    fill_with_singles(&roles);
    let asked = std::rc::Rc::new(std::cell::Cell::new(0u32));
    let parked: std::rc::Rc<dyn Fn()> = {
        let roles = roles.clone();
        let done = std::cell::Cell::new(false);
        std::rc::Rc::new(move || {
            if !done.replace(true) {
                roles.lock().unwrap().close_instance(victim);
            }
        })
    };
    let (long, long_engine) = runs_of(8, 4, 1, 0);
    let long_engine = std::rc::Rc::new(std::cell::RefCell::new(long_engine));
    let (small, small_engine) = runs_of(10, 1, 1, 2000);
    let small_engine = std::rc::Rc::new(std::cell::RefCell::new(small_engine));
    let head = waiting_bind(&roles, &device, ENGINE, 1, long, long_engine.clone(), parked.clone(), asked.clone());
    let behind = waiting_bind(&roles, &device, ENGINE, 2, small, small_engine.clone(), parked, asked.clone());
    tokio::pin!(head);
    tokio::pin!(behind);
    let first = futures::poll!(head.as_mut());
    assert!(asked.get() >= 1, "the planner is asked for the head");
    assert!(matches!(first, std::task::Poll::Ready(Ok(_))), "the head binds in the room it made");
    behind.await.expect("the bind behind it follows");
    assert_eq!(long_engine.borrow().bindings.len(), 1);
    assert_eq!(small_engine.borrow().bindings.len(), 1);
}

/// The head waits on a commit the target said could come, and the walk
/// makes none (its target came down after): the planner, asked beside the
/// commit, makes the room, and the head binds.
#[tokio::test(flavor = "current_thread")]
async fn a_guest_head_waiting_on_a_commit_that_does_not_come_is_served_by_the_planner() {
    use crate::governor::PoolTarget;
    const ENGINE: usize = 68;
    let Pools { device, roles } = pools_of(facts_committed(64, 8, 1));
    let mut grow = crate::store::shared::SharedGrow::new(0, ENGINE, device.clone());
    let _ = grow.retarget(u64::from(u32::MAX));
    let (held_program, mut held_engine) = runs_of(13, 4, 2, 5000);
    let victim = bind_as(&roles, &mut held_engine, held_program, 7).expect("a resident");
    fill_with_singles(&roles);
    let asked = std::rc::Rc::new(std::cell::Cell::new(0u32));
    let parked: std::rc::Rc<dyn Fn()> = {
        let roles = roles.clone();
        std::rc::Rc::new(move || roles.lock().unwrap().close_instance(victim))
    };
    let (long, long_engine) = runs_of(8, 4, 2, 0);
    let long_engine = std::rc::Rc::new(std::cell::RefCell::new(long_engine));
    let prefix = device.lock().unwrap().pool().prefix_segments();
    let head = waiting_bind(&roles, &device, ENGINE, 1, long, long_engine.clone(), parked, asked.clone());
    tokio::pin!(head);
    let polled = futures::poll!(head.as_mut());
    assert!(asked.get() >= 1, "the planner is asked beside the commit");
    assert!(matches!(polled, std::task::Poll::Ready(Ok(_))), "the head binds in the room it made");
    assert_eq!(device.lock().unwrap().pool().prefix_segments(), prefix, "no commit came");
}

/// The pools of `facts` with no host tier below the device (AMD's repro:
/// budget_host a few MiB, no host chunks).
fn pools_no_host(facts: ChunkFacts) -> Pools {
    let device = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let roles = Roles::new(&device, None).expect("the pool states an Instance shape");
    Pools {
        device,
        roles: Arc::new(Mutex::new(roles)),
    }
}

/// AMD's freeze, its root: with no host tier a park moved nothing, so a
/// parked process kept its guest scratch on the device and Instance chunks
/// filled with idle scratch no kv eviction could free. Scratch is dead
/// between fires: a park now gives it back whatever lies below (the live
/// tables stay where they lie), a new bind lands in the room it left, and
/// the restore takes fresh scratch in its class and is rebound to it
/// before its next fire.
#[test]
fn with_no_host_tier_a_park_gives_guest_scratch_back_and_a_restore_takes_it_fresh() {
    let Pools { device, roles } = pools_no_host(facts_in(64, 8));
    let (program, mut engine) = runs_of(13, 4, 1, 0);
    let mut bound = Vec::new();
    for by in 1.. {
        match bind_or_leave(&roles, &mut engine, program, by) {
            Ok(id) => bound.push((by, id)),
            Err(_) => break,
        }
    }
    assert!(bound.len() >= 2, "{} bound", bound.len());
    assert_eq!(device.lock().unwrap().pool().free_chunk_runs(4), 0, "no run of 4 left");
    let (other, mut other_engine) = runs_of(14, 4, 1, 9000);
    bind_or_leave(&roles, &mut other_engine, other, 999).expect_err("short of a run");

    // The first process parks, no host tier below: its scratch comes back.
    let (parked, id) = bound[0];
    park(&roles, parked);
    assert!(device.lock().unwrap().pool().free_chunk_runs(4) >= 1, "its run of 4 is free");
    assert!(!roles.lock().unwrap().holds_guest_state(holder(parked)));
    let landed = bind_as(&roles, &mut other_engine, other, 999).expect("the new bind lands");

    // The new bind leaves; the parked process restores, taking fresh
    // scratch in its class, and is told where before its next fire.
    roles.lock().unwrap().close_instance(landed);
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(parked))
        .expect("room for it")
        .expect("something comes back");
    roles.lock().unwrap().commit(unpark);
    let stale = roles.lock().unwrap().stale(holder(parked));
    let Some(Rebind::Instance { id: rebound, scratch, .. }) = stale
        .iter()
        .find(|rebind| matches!(rebind, Rebind::Instance { .. }))
        .cloned()
    else {
        panic!("its instance is rebound: {stale:?}");
    };
    assert_eq!(rebound, id);
    assert_ne!(scratch, ScratchPlace::None, "fresh scratch, laid");
    assert!(roles.lock().unwrap().holds_guest_state(holder(parked)));
}

/// The Instance chunks parking a process empties: per chunk, the slots of
/// its dead parts (guest scratch) on the device, its live tables never;
/// after its park, none.
#[test]
fn a_process_s_guest_chunk_slots_are_its_scratch_on_the_device() {
    let Pools { roles, .. } = pools_no_host(facts_in(64, 8));
    let (program, mut engine) = runs_of(13, 2, 1, 0);
    bind_as(&roles, &mut engine, program, 1).unwrap();
    bind_as(&roles, &mut engine, program, 2).unwrap();
    let slots = roles.lock().unwrap().guest_chunk_slots(&[holder(1)]);
    let total: u32 = slots.iter().map(|&(_, _, n)| n).sum();
    assert_eq!(total, 2 * SPC, "its run of two whole chunks");
    assert!(slots.iter().all(|&(_, by, _)| by == holder(1)));
    assert_eq!(slots.len(), 2, "two chunks");
    park(&roles, 1);
    assert!(roles.lock().unwrap().guest_chunk_slots(&[holder(1)]).is_empty());
}

/// The Spark's beam restores (dbe9c0e93): fragmented Instance chunks, a
/// parked guest's freed run filled by another bind, then its restore. The
/// unpark is short and waits, naming the runs of chunks it needs (what the
/// planner funds, `RolesError::unpark_runs`), never refused; once room
/// comes it lands.
#[test]
fn a_restore_short_of_its_run_waits_with_its_need_and_lands_once_room_comes() {
    let Pools { device, roles } = pools_no_host(facts_in(64, 8));
    let (program, mut engine) = runs_of(13, 4, 1, 0);
    let mut bound = Vec::new();
    for by in 1.. {
        match bind_or_leave(&roles, &mut engine, program, by) {
            Ok(id) => bound.push((by, id)),
            Err(_) => break,
        }
    }
    assert!(bound.len() >= 2, "{} bound", bound.len());
    let (parked, _) = bound[0];
    park(&roles, parked);
    let (other, mut other_engine) = runs_of(14, 4, 1, 9000);
    let landed = bind_as(&roles, &mut other_engine, other, 999).expect("the freed run is taken");
    assert_eq!(device.lock().unwrap().pool().free_chunk_runs(4), 0, "no run of 4 left");

    let error = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(parked))
        .expect_err("short of its run");
    assert!(matches!(error, RolesError::UnparkShort(_)), "it waits: {error}");
    assert_eq!(error.unpark_runs(), Some((4, 1)), "its need, named: {error}");

    roles.lock().unwrap().close_instance(landed);
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(parked))
        .expect("room came")
        .expect("something comes back");
    roles.lock().unwrap().commit(unpark);
    assert!(roles.lock().unwrap().holds_guest_state(holder(parked)));
}

/// A Channel item that must lie in one run, short of it at a restore,
/// waits too, its need one whole chunk (any free chunk holds the run).
#[test]
fn a_restore_short_of_a_channel_run_waits_for_one_chunk() {
    let short = RolesError::UnparkShort(Box::new(RolesError::NoRun {
        role: SlotRole::Channel,
        slots: 2,
    }));
    assert_eq!(short.unpark_runs(), Some((1, 1)));
    assert!(RolesError::NoRun { role: SlotRole::Channel, slots: 2 }.unpark_runs().is_none(), "only a waiting unpark names one");
}

/// A frame naming the last instance of a pair is on its way through the
/// engine when that instance closes: the pair's state (its batch tables,
/// which every grouped fire of its instances reads) and the instance's own
/// slots stay out of the free count until the frame lands, then come free.
/// Given back at the close, the pair's chunks went to another owner under
/// the fire (Metal's 0xa0, FUSED_GEOMETRY_MISMATCH).
#[test]
fn a_pair_s_state_is_given_back_only_once_the_fires_naming_it_land() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(TOP_P, 2 * CHUNK);
    let id = bind(&roles, &mut engine, 1).unwrap();
    let pair = super::FireKey {
        program: PROGRAM,
        extents: BindExtents::default(),
    };
    assert!(!roles.lock().unwrap().slots(Owner::Pair(pair)).is_empty());
    let free = || device.lock().unwrap().pool().free_chunks();
    let before = free();
    let named = roles.lock().unwrap().reading([id]);
    assert!(named.contains(&Owner::Pair(pair)), "the frame names the pair");
    roles.lock().unwrap().close_instance(id);
    assert_eq!(
        free(),
        before,
        "the pair's chunks, and its instance's, wait on the frame naming them"
    );
    roles.lock().unwrap().landed(&named);
    assert_eq!(
        free(),
        before + 3,
        "as the frame lands: the pair's two chunks, and the one its instance's scratch run lay in"
    );
    assert!(roles.lock().unwrap().slots(Owner::Pair(pair)).is_empty());
}

/// A pair moving with another process's instances holds a restore back
/// (`RolesError::PairParked`) until that move ends; an aborted move ends it
/// as a committed one does, and what waits on it is told (a bind's or a
/// restore's room watch). Told only at a commit, a move that failed left
/// them waiting for good. And a pair whose holders all closed (a failed
/// process's close included) is freed, nothing left of it on the device.
#[tokio::test(flavor = "current_thread")]
async fn an_aborted_pair_move_wakes_its_waiters_and_a_pair_of_closed_holders_is_freed() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    let before = held(&device);
    let first = bind(&roles, &mut engine, 1).unwrap();
    let second = bind(&roles, &mut engine, 2).unwrap();
    park(&roles, 1);
    // The second parks: its pair moves with it.
    let moving = roles
        .lock()
        .unwrap()
        .prepare_park(holder(2))
        .expect("the second and the pair park");
    // The first's restore meanwhile: its pair is moving.
    let watch = roles.lock().unwrap().room_watch();
    assert!(matches!(
        roles.lock().unwrap().prepare_unpark(holder(1)),
        Err(RolesError::PairParked(_))
    ));
    let woken = tokio::spawn(watch.freed());
    tokio::task::yield_now().await;
    assert!(!woken.is_finished(), "the move is under way");
    roles.lock().unwrap().abort(moving);
    tokio::time::timeout(std::time::Duration::from_secs(5), woken)
        .await
        .expect("an aborted move tells its waiters")
        .unwrap();
    let back = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .expect("the pair stands still now")
        .expect("it unparks");
    roles.lock().unwrap().commit(back);

    // Both holders close (a failed process's close included): nothing of
    // them or their pair is left.
    roles.lock().unwrap().close_instance(first);
    roles.lock().unwrap().close_instance(second);
    assert_eq!(held(&device), before, "the pair freed with its last holder");
}

/// Over-demand with a pair four processes share, every one parked: the
/// first restore brings the pair back with it, and while those copies are
/// on their way each other holder's restore finds the pair moving
/// (`pair_moving` names the instances it moves with) and waits once on
/// the move's end, told by the commit; then each comes back, the pair
/// where the first's brought it, and every holder closes leaving nothing.
#[tokio::test(flavor = "current_thread")]
async fn restores_of_a_pair_s_many_holders_wait_once_on_its_move_and_all_land() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    let before = held(&device);
    let ids: Vec<InstanceId> = (1..=4).map(|by| bind(&roles, &mut engine, by).unwrap()).collect();
    for by in 1..=4 {
        park(&roles, by);
    }
    let moving = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .expect("the first brings the pair back");
    let mut waits = 0;
    let mut watches = Vec::new();
    for by in 2..=4 {
        let watch = roles.lock().unwrap().room_watch();
        let unparked = roles.lock().unwrap().prepare_unpark(holder(by));
        match unparked {
            Err(RolesError::PairParked(pair)) => {
                let named = roles.lock().unwrap().pair_moving(pair).expect("moving");
                assert_eq!(named.len(), 4, "the instances it moves with: {named:?}");
                waits += 1;
                watches.push(tokio::spawn(watch.freed()));
            }
            other => panic!("the pair is moving: {other:?}"),
        }
    }
    assert_eq!(waits, 3, "one wait each");
    roles.lock().unwrap().commit(moving);
    for watch in watches {
        tokio::time::timeout(std::time::Duration::from_secs(5), watch)
            .await
            .expect("the move's end tells each")
            .unwrap();
    }
    for by in 2..=4 {
        let back = roles
            .lock()
            .unwrap()
            .prepare_unpark(holder(by))
            .expect("the pair stands still")
            .expect("it comes back");
        roles.lock().unwrap().commit(back);
    }
    for id in ids {
        roles.lock().unwrap().close_instance(id);
    }
    assert_eq!(held(&device), before);
}

/// The disk-only home's "KV restore failed: rebind Instance(...): a guest
/// item of 2508288 bytes takes one run of 613 Instance slot(s) ... and its
/// binding names []": two processes' instances share a pair; with no tier
/// below, each parks where it lies, its live parts kept and its scratch
/// given back ([`Item::shed`]); the pair, all dead, parks once both have.
/// One process's restore brings the pair back and rebinds its holders on
/// the device: the other process's instance, still parked where it lies,
/// was rebound too, its scratch emptied, and named no place. It is rebound
/// only with its own restore, which takes its scratch back; no rebind ever
/// names an emptied guest part.
#[test]
fn a_restore_of_a_shared_pair_leaves_another_s_instance_parked_where_it_lies_unbound() {
    let facts = facts(64);
    let device = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let roles = Arc::new(Mutex::new(
        Roles::new(&device, None).expect("the pool states an Instance shape"),
    ));
    let mut engine = GuestEngine::new(TOP_P, 2 * CHUNK);
    let a = bind(&roles, &mut engine, 1).unwrap();
    let b = bind(&roles, &mut engine, 2).unwrap();
    park(&roles, 2);
    park(&roles, 1);
    let named_nothing = |rebinds: &[Rebind]| {
        rebinds.iter().any(|rebind| {
            matches!(
                rebind,
                Rebind::Instance {
                    scratch: ScratchPlace::Run { slots: 0, .. },
                    ..
                } | Rebind::Instance {
                    pair: ScratchPlace::Run { slots: 0, .. },
                    ..
                }
            ) || matches!(rebind, Rebind::Instance { scratch: ScratchPlace::Chunks { pieces, .. }, .. } if pieces.is_empty())
        })
    };
    let back = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .expect("process 1's items come back");
    roles.lock().unwrap().commit(back);
    let stale = roles.lock().unwrap().stale(holder(1));
    assert!(
        stale.iter().any(|rebind| matches!(rebind, Rebind::Instance { id, .. } if *id == a)),
        "its own instance is rebound: {stale:?}"
    );
    assert!(
        !stale.iter().any(|rebind| matches!(rebind, Rebind::Instance { id, .. } if *id == b)),
        "the other's instance, parked where it lies, is not: {stale:?}"
    );
    assert!(!named_nothing(&stale), "no rebind names an emptied part: {stale:?}");
    for rebind in stale {
        roles.lock().unwrap().rebound(super::Owner::of(&rebind));
    }
    // The other's restore takes its scratch back and rebinds it.
    let back = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(2))
        .unwrap()
        .expect("process 2's items come back");
    roles.lock().unwrap().commit(back);
    let stale = roles.lock().unwrap().stale(holder(2));
    assert!(
        stale.iter().any(|rebind| matches!(rebind, Rebind::Instance { id, .. } if *id == b)),
        "rebound with its own restore: {stale:?}"
    );
    assert!(!named_nothing(&stale), "with its scratch: {stale:?}");
}

/// mx-2's cycle: a bind short of room suspended its process, the process
/// was restored, the bind was short again and suspended it again, every
/// ~25 ms. Suspended once for its bind's room and restored, a bind short
/// again waits parked in the planner for the room, and binds once served;
/// its process is suspended once only.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_suspended_once_for_room_waits_parked_when_short_again() {
    use std::cell::Cell;
    use std::rc::Rc;
    let short = || {
        anyhow::Error::new(RolesError::BindShort(Box::new(RolesError::Short {
            role: SlotRole::Instance,
            slots: 3,
        })))
    };
    let (binds, suspends, waits) = (Rc::new(Cell::new(0u32)), Rc::new(Cell::new(0u32)), Rc::new(Cell::new(0u32)));
    let bound = super::bind_when_resident(
        holder(7),
        {
            let binds = binds.clone();
            move || {
                binds.set(binds.get() + 1);
                // Short three times (before the suspend, after the restore,
                // and once more while it waits), then bound.
                std::future::ready(if binds.get() <= 3 { Err(short()) } else { Ok(()) })
            }
        },
        || std::future::ready(Ok::<(), String>(())),
        |_, _| std::future::ready(Ok::<(), String>(())),
        || None,
        |_, _| false,
        {
            let suspends = suspends.clone();
            move |_, _| {
                suspends.set(suspends.get() + 1);
                true
            }
        },
        {
            let waits = waits.clone();
            move |_, _| {
                waits.set(waits.get() + 1);
                std::future::ready(Ok::<(), String>(()))
            }
        },
        || {},
        |_| {},
    )
    .await;
    assert!(bound.is_ok(), "{bound:?}");
    assert_eq!(suspends.get(), 1, "suspended once for its room, never again");
    assert_eq!(waits.get(), 2, "short again after its restore, it waits parked for room");
    assert_eq!(binds.get(), 4);
}
