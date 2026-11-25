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
    /// Where each live instance's scratch lies, as a Vulkan or wgpu engine
    /// claims it (`engine_gpu::guest::claims`): a bind or rebind onto
    /// slots another live instance claims is refused by name, a park
    /// (`Rebind::Parked`) releases them. `None`: nothing claimed.
    claims: Option<std::collections::BTreeMap<InstanceId, Vec<u32>>>,
}

impl GuestEngine {
    /// Claims `place` for `id` in place of what it held, as
    /// `engine_gpu::guest::claims::Claims::claim` does.
    fn claim(&mut self, id: InstanceId, place: &ScratchPlace) -> ::engine::Result<()> {
        let Some(claims) = self.claims.as_mut() else {
            return Ok(());
        };
        let refs = refs_of(place);
        if let Some((holder, _)) = claims
            .iter()
            .find(|(holder, held)| **holder != id && held.iter().any(|slot| refs.contains(slot)))
        {
            return Err(::engine::Error::Program(format!(
                "guest::claims: Instance slots already hold instance {holder}'s run: instance \
                 {id}'s would overlap it"
            )));
        }
        claims.insert(id, refs);
        Ok(())
    }
}

impl GuestEngine {
    fn new(per_instance: u64, per_pair: u64) -> Self {
        Self {
            state: GuestState {
                resident: 0,
                scratch: per_instance,
                scratch_arms: 1,
                per_pair,
                run: 1,
            },
            next: 0,
            bindings: Vec::new(),
            rebinds: Vec::new(),
            claims: None,
        }
    }
}

impl ::engine::Engine for GuestEngine {
    fn committed_covers(&self) -> &'static [&'static str] {
        &[]
    }

    fn measured_at_load(&self) -> Vec<String> {

        Vec::new()

    }

    fn skips_measuring(&self) -> bool {
        false
    }

    fn kind(&self) -> &'static str {
        "fake-guests"
    }

    fn load(&mut self, _request: LoadRequest) -> ::engine::Result<Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(&mut self, _frame: &FrameSubmission) -> ::engine::Result<FrameTicket> {
        unreachable!("nothing fires here")
    }

    fn settle_frame(&mut self, _ticket: &mut ::engine::FrameTicket) -> ::engine::Result<()> {
        Ok(())
    }

    fn programs(&mut self) -> Option<&mut dyn ::engine::Programs> {
        Some(self)
    }

    fn programs_ref(&self) -> Option<&dyn ::engine::Programs> {
        Some(self)
    }

    fn pool(&mut self) -> Option<&mut dyn ::engine::Pool> {
        Some(self)
    }

    fn tiers(&mut self) -> Option<&mut dyn ::engine::Tiers> {
        Some(self)
    }
}

impl ::engine::Programs for GuestEngine {
    fn prepare(
        &mut self,
        _registration: &::engine::program::ProgramRegistration,
    ) -> ::engine::Result<::engine::ProgramCompile> {
        Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
    }

    fn finish(
        &mut self,
        _prepared: ::engine::Prepared,
    ) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "finish_program"))
    }

    fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "close_program"))
    }

    fn facts(&self, _program: ProgramId) -> ::engine::Result<ProgramFacts> {
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

    fn bind(&mut self, binding: &InstanceBinding) -> ::engine::Result<BoundInstance> {
        self.claim(InstanceId(self.next + 1), &binding.scratch)?;
        self.next += 1;
        self.bindings.push(binding.clone());
        Ok(BoundInstance {
            id: InstanceId(self.next),
            program: binding.program,
            geometry: binding.geometry,
            standing_guest_bytes: 0,
        })
    }

    fn rebind(&mut self, rebind: Rebind) -> ::engine::Result<()> {
        match &rebind {
            Rebind::Instance { id, scratch, .. } => self.claim(*id, scratch)?,
            Rebind::Parked { id, .. } => {
                if let Some(claims) = self.claims.as_mut() {
                    claims.remove(id);
                }
            }
            Rebind::Channel { .. } => {}
        }
        self.rebinds.push(rebind);
        Ok(())
    }

    fn register(&mut self, _registration: &::engine::ProgramRegistration) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "register_program"))
    }

    fn close(&mut self, _id: ::engine::InstanceId) -> ::engine::Result<u64> {
        Err(::engine::Error::unsupported(self.kind(), "close_instance"))
    }

    fn adapter(&mut self, _registration: &::engine::adapter::AdapterRegistration) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "register_adapter"))
    }
}

impl ::engine::Pool for GuestEngine {
    fn chunk_table(&self) -> ::engine::Result<::engine::chunk::ChunkTable> {
        Err(::engine::Error::unsupported(self.kind(), "chunk_table"))
    }

    fn commit(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "commit_segments"))
    }

    fn release(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "release_segments"))
    }

    fn evacuate(&mut self, _evacuation: &::engine::chunk::Evacuation) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "evacuate"))
    }
}

impl ::engine::Tiers for GuestEngine {
    /// A test engine's copies land inside the call: nothing to wait for.
    fn settle(&mut self) -> ::engine::Result<()> {
        Ok(())
    }

    /// A test engine's copies complete inside the call: their sources are
    /// read by the time it returns.
    fn copy_landing(
        &mut self,
        copy: &::engine::chunk::SlotCopy,
        _token: u64,
    ) -> ::engine::Result<bool> {
        ::engine::Tiers::copy(self, copy)?;
        Ok(true)
    }

    fn copy(&mut self, _copy: &::engine::chunk::SlotCopy) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "copy_slots"))
    }

    fn read(&mut self, _slots: &[(::engine::chunk::ShapeId, Vec<u32>)]) -> ::engine::Result<Vec<u8>> {
        Err(::engine::Error::unsupported(self.kind(), "read_tier_slots"))
    }

    fn write(&mut self, _slots: &[(::engine::chunk::ShapeId, Vec<u32>)], _bytes: &[u8]) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "write_tier_slots"))
    }

    fn copy_kv(&mut self, _copy: &::engine::KvCopy) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "copy_kv"))
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

/// A park's instances are said parked before its commit gives their
/// slots to anyone else, so an engine claiming where each instance lies
/// (Vulkan, wgpu) takes another process's bind there, and the parked
/// one's restore rebinds it to fresh slots none other claims. Untold, the
/// next bind on the freed slots is refused by name ("Instance slots ...
/// already hold instance N's run").
#[test]
fn a_park_is_said_to_the_engine_before_its_slots_are_another_s() {
    for told in [true, false] {
        let Pools { roles, .. } = pools(64);
        let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
        engine.claims = Some(Default::default());
        let id = bind(&roles, &mut engine, 1).unwrap();
        let parked_at = engine.bindings[0].scratch.clone();
        let park = roles
            .lock()
            .unwrap()
            .prepare_park(holder(1))
            .expect("the instance parks");
        assert_eq!(park.parked(), vec![Rebind::Parked { id, kept: false }], "the park names it");
        if told {
            for rebind in park.parked() {
                super::rebind(Some(&roles), &mut engine, rebind, |_, _, _| Ok(())).unwrap();
            }
            assert!(
                roles.lock().unwrap().stale(holder(1)).is_empty(),
                "a park rebinds nothing"
            );
        }
        roles.lock().unwrap().commit(park);
        // Another process's instance takes the freed chunks.
        let other = bind(&roles, &mut engine, 2);
        if !told {
            let error = other.expect_err("untold, the engine still claims the parked slots");
            assert!(error.to_string().contains("would overlap"), "{error:#}");
            continue;
        }
        other.expect("told, the freed slots are another's");
        assert_eq!(
            refs_of(&engine.bindings[1].scratch).first(),
            refs_of(&parked_at).first(),
            "the second bind lies where the parked one did"
        );
        let unpark = roles
            .lock()
            .unwrap()
            .prepare_unpark(holder(1))
            .unwrap()
            .expect("it unparks");
        assert!(unpark.parked().is_empty(), "an unpark names no park");
        roles.lock().unwrap().commit(unpark);
        let stale = roles.lock().unwrap().stale(holder(1));
        for rebind in stale {
            super::rebind(Some(&roles), &mut engine, rebind, |_, _, _| Ok(()))
                .expect("its fresh slots are claimed by none");
        }
        assert!(roles.lock().unwrap().stale(holder(1)).is_empty());
    }
}

/// A pair's run is its batch's while any member is on the device or on
/// its way: with one member's park in flight, another member's park leaves
/// the pair where it lies, so when the first park is taken back (the tier
/// below full) its member is resident on its pair, and another program's
/// bind never takes that run as fresh room (a device fault: one program's
/// pair run bound by another). Once
/// every member's park has landed, the run is given back.
#[test]
fn a_pair_run_is_never_another_program_s_while_a_member_may_still_fire() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    let before = held(&device);
    let a1 = bind(&roles, &mut engine, 1).unwrap();
    let _a2 = bind(&roles, &mut engine, 2).unwrap();
    let pair_at = engine.bindings[0].pair.clone();
    assert_eq!(pair_at, engine.bindings[1].pair, "one pair run for the batch");
    // The first member's park in flight; the second parks and lands.
    let first = roles.lock().unwrap().prepare_park(holder(1)).expect("it parks");
    let second = roles.lock().unwrap().prepare_park(holder(2)).expect("it parks");
    roles.lock().unwrap().commit(second);
    // The first's move is taken back: its member is resident again.
    roles.lock().unwrap().abort(first);
    assert!(roles.lock().unwrap().stale(holder(1)).is_empty(), "nothing moved of {a1:?}");
    // Another program binds: none of its room is the resident pair's.
    let mut other = GuestEngine::new(2 * CHUNK, PAIR);
    other.next = 100;
    learn_program(&roles, &mut other, ProgramId(8)).unwrap();
    bind_instance(
        Some(&roles),
        &mut other,
        &InstanceBinding {
            program: ProgramId(8),
            ..InstanceBinding::default()
        },
        Some(holder(3)),
        |_| false,
    )
    .unwrap();
    let theirs = &other.bindings[0];
    disjoint(&[&pair_at, &theirs.pair, &theirs.scratch, &engine.bindings[0].scratch])
        .expect("the resident member's pair run is its batch's alone");
    // The first member's park lands now: the pair goes back with it.
    let first = roles.lock().unwrap().prepare_park(holder(1)).expect("it parks");
    roles.lock().unwrap().commit(first);
    let mine = 2 * CHUNK + slots(PAIR) * SLOT;
    assert_eq!(
        held(&device) - before,
        mine,
        "only the other program's room is held: the batch's run went back"
    );
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
                singles: 0,
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

/// A guest bind needing two runs of four adjacent chunks waits on a segment
/// commit while smaller guest binds keep arriving. Unqueued, each smaller
/// bind would take the fresh segment's chunks a few at a time, the long
/// runs never stand free, and the long bind wait for ever. It heads the
/// queue: the smaller binds wait
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
        move |run, pieces, singles| roles.lock().unwrap().room_free(run, pieces, singles),
        |_, _, _| false,
        move |_, _, _| {
            asked.set(asked.get() + 1);
            serve();
            std::future::ready(Ok::<_, String>(crate::planner::BindRoomWait::Served))
        },
        move || roles.lock().unwrap().leave_guest_wait(holder(by)),
        |_| {},
        || std::future::ready(()),
    )
}

/// The device is full at the governor's target, a guest head short of runs,
/// other guest binds queued behind it. The planner is asked (the head waits on
/// no commit that cannot come); its serve (a resident parked) frees the runs;
/// the head binds; the bind behind it follows.
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

/// The pools of `facts` with no host tier below the device.
fn pools_no_host(facts: ChunkFacts) -> Pools {
    let device = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let roles = Roles::new(&device, None).expect("the pool states an Instance shape");
    Pools {
        device,
        roles: Arc::new(Mutex::new(roles)),
    }
}

/// With no host tier a park gives its guest scratch back whatever lies below
/// (scratch is dead between fires; the live tables stay where they lie): kept,
/// it would fill Instance chunks no kv eviction can free. A new bind lands in
/// the room it left, and the restore takes fresh scratch in its class and is
/// rebound to it before its next fire.
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

/// Beam restores on unified memory: fragmented Instance chunks, a
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

/// A move whose copies an engine could not settle (its `settle_copies`
/// failed): it ends all the same, each item where the move found it, so a
/// pair's other holders no longer wait on it moving, and are told; the
/// slots it took on the tier it was going to stay out of service for good,
/// counted as leaked by name, never given back nor reused. Nothing of the
/// pair's state is lost: dead between fires, it comes back fresh.
#[tokio::test(flavor = "current_thread")]
async fn an_abandoned_move_ends_leaks_its_slots_by_name_and_frees_its_waiters() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    let first = bind(&roles, &mut engine, 1).unwrap();
    let second = bind(&roles, &mut engine, 2).unwrap();
    park(&roles, 1);
    park(&roles, 2);
    // The first's restore brings its tables and the pair back: its copies
    // never settle.
    let moving = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .expect("it unparks");
    let taken: u64 = held(&device);
    let watch = roles.lock().unwrap().room_watch();
    let unparked = roles.lock().unwrap().prepare_unpark(holder(2));
    assert!(matches!(unparked, Err(RolesError::PairParked(_))), "the pair moves");
    let woken = tokio::spawn(watch.freed());
    let leaked = roles.lock().unwrap().abandon(moving);
    assert!(leaked > 0, "the slots it took on the device");
    assert_eq!(roles.lock().unwrap().leaked(), leaked, "counted by name");
    assert_eq!(held(&device), taken, "never given back");
    tokio::time::timeout(std::time::Duration::from_secs(5), woken)
        .await
        .expect("its waiters are told")
        .unwrap();
    // The move ended: both restores go, the pair's state taken fresh.
    for (by, id) in [(1, first), (2, second)] {
        let back = roles
            .lock()
            .unwrap()
            .prepare_unpark(holder(by))
            .expect("nothing moves now")
            .expect("it comes back");
        roles.lock().unwrap().commit(back);
        let stale = roles.lock().unwrap().stale(holder(by));
        for rebind in stale {
            <GuestEngine as ::engine::Programs>::rebind(&mut engine, rebind).unwrap();
            roles.lock().unwrap().rebound(Owner::Instance(id));
        }
    }
    roles.lock().unwrap().close_instance(first);
    roles.lock().unwrap().close_instance(second);
    assert_eq!(roles.lock().unwrap().leaked(), leaked, "still out of service");
}

/// A failed process's teardown (else pairs of dead processes are left on
/// the device and the host): the engine refuses its instances' closes
/// (a failed fire left them unusable), and the lane closes their books all
/// the same. Its slots go back once no step on its way names them (a step
/// still reads one when the close comes, and lands after), and the pair
/// whose last holders they were is freed: the pool holds exactly what it
/// held before the process bound.
#[test]
fn a_close_the_engine_refuses_still_frees_the_instance_and_its_pair() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    let before = held(&device);
    let first = bind(&roles, &mut engine, 1).unwrap();
    let second = bind(&roles, &mut engine, 1).unwrap();
    // A step naming the first is still on its way when the closes come.
    let reads = roles.lock().unwrap().reading([first]);
    let mut channels = crate::engine::channel::ChannelJoin::new();
    for id in [first, second] {
        crate::scheduler::lane::close_books(
            id,
            Err(anyhow::anyhow!("the backend refused a dispatch")),
            &mut channels,
            Some(&roles),
            None,
        );
    }
    assert!(
        held(&device) - before >= 2 * CHUNK + slots(PAIR) * SLOT,
        "the first's scratch and the pair wait for the step that names them"
    );
    roles.lock().unwrap().landed(&reads);
    assert_eq!(
        held(&device),
        before,
        "every slot back, the pair freed with its last holder"
    );
    assert!(roles.lock().unwrap().guest_chunk_slots(&[holder(1)]).is_empty());
}

/// A load with no host tier (unified memory): two processes' instances
/// share their pair's state. Each park leaves its instance's tables where
/// they lie (no room below) but gives back its scratch; once both have
/// parked, the pair's state (dead between fires) goes back too: an
/// instance parked where it lies counts as parked for what it shares.
#[test]
fn with_no_host_tier_a_shared_pair_s_state_goes_back_once_every_holder_parked() {
    let facts = facts(64);
    let device = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let roles: RolesHandle = Arc::new(Mutex::new(Roles::new(&device, None).unwrap()));
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    let before = held(&device);
    let first = bind(&roles, &mut engine, 1).unwrap();
    let second = bind(&roles, &mut engine, 2).unwrap();
    let pair = slots(PAIR) * SLOT;
    let bound = held(&device);
    let tables = (bound - before - 2 * 2 * CHUNK - pair) / 2;

    let park = roles.lock().unwrap().prepare_park(holder(1)).unwrap();
    roles.lock().unwrap().commit(park);
    assert_eq!(
        held(&device) - before,
        2 * tables + 2 * CHUNK + pair,
        "the first's scratch back, the pair still the second's"
    );
    let park = roles.lock().unwrap().prepare_park(holder(2)).unwrap();
    roles.lock().unwrap().commit(park);
    assert_eq!(
        held(&device) - before,
        2 * tables,
        "every scratch and the pair's state back, the tables where they lie"
    );
    for (by, id) in [(1, first), (2, second)] {
        let back = roles
            .lock()
            .unwrap()
            .prepare_unpark(holder(by))
            .unwrap()
            .expect("it comes back");
        roles.lock().unwrap().commit(back);
        let stale = roles.lock().unwrap().stale(holder(by));
        for rebind in stale {
            <GuestEngine as ::engine::Programs>::rebind(&mut engine, rebind).unwrap();
            roles.lock().unwrap().rebound(Owner::Instance(id));
        }
    }
    assert_eq!(held(&device), bound, "both back, with one pair");
    roles.lock().unwrap().close_instance(first);
    roles.lock().unwrap().close_instance(second);
    assert_eq!(held(&device), before);
}

/// A guest bind's run at the pool's limit (every segment committed, nothing left
/// to evict for it): free chunks scattered by the binds before it, enough
/// for its run of 4 but none adjacent. It is short (`BindShort`, to the
/// planner); the planner asks for a compaction, the walk moves chunks
/// until the run stands (each move's chunks closed until the fire that
/// copies it retires), and the bind lands. The singles moved keep their
/// instances, bound on their logical chunks.
#[test]
fn a_bind_short_of_a_run_among_scattered_free_chunks_lands_after_a_compaction() {
    let segments = 8;
    let Pools { device, roles } = pools_of(facts_committed(64, 8, segments));
    let ids = fill_with_singles(&roles);
    for (k, id) in ids.iter().enumerate() {
        if k % 2 == 0 {
            roles.lock().unwrap().close_instance(*id);
        }
    }
    {
        let pool = device.lock().unwrap();
        assert!(pool.pool().free_chunks() >= 4);
        assert_eq!(pool.pool().free_chunk_runs(4), 0, "none of four adjacent");
    }
    let (long, mut long_engine) = runs_of(8, 4, 1, 0);
    let error = bind_as(&roles, &mut long_engine, long, 1).expect_err("no run of 4");
    assert!(
        matches!(error.downcast_ref::<RolesError>(), Some(RolesError::BindShort(_))),
        "at the limit, short of a run: to the planner: {error}"
    );
    roles.lock().unwrap().leave_guest_wait(holder(1));

    // The planner's ask, then the walk's steps.
    assert!(
        !device.lock().unwrap().compaction_comes(4, 1),
        "control: no walk steers the pool, nothing would carry it out"
    );
    let _grow = crate::store::shared::SharedGrow::new(0, 74, device.clone());
    assert!(device.lock().unwrap().compaction_comes(4, 1));
    let mut epoch = 10;
    let mut compactions = 0;
    loop {
        let step = device.lock().unwrap().next_move(segments, epoch);
        match step {
            Some(crate::store::shared::SegmentMove::Compact {
                evacuation,
                applied,
                epoch: at,
            }) => {
                assert_eq!(at, epoch);
                assert_eq!(evacuation.moves.len(), applied.len());
                compactions += 1;
                let mut pool = device.lock().unwrap();
                pool.compaction_copied(at, at);
                assert_eq!(pool.pool().free_chunk_runs(4), 0, "closed until its fire retires");
                pool.pool_mut().retire_through(at);
                epoch += 1;
            }
            None => break,
            other => panic!("only compactions at the limit: {other:?}"),
        }
    }
    assert_eq!(compactions, 1, "one compaction stands the run");
    assert_eq!(device.lock().unwrap().pool().free_chunk_runs(4), 1);
    assert!(!device.lock().unwrap().compaction_comes(4, 1), "the run stands: nothing to move");
    bind_as(&roles, &mut long_engine, long, 1).expect("the run stands: it binds");
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

/// Bindings are never lost nor empty, as a property over seeded schedules:
/// four processes bind instances of one program (sharing its pair), then
/// park and restore in random order, with a host tier below or none (each
/// parking where it lies, its scratch given back). After every restore,
/// each rebind names its instance's tables and a place for its scratch and
/// its pair (never an emptied part), only that process's own instances
/// and those of a pair's holders that lie whole on the device are rebound,
/// and every instance of a process on the device is bound somewhere.
/// Failing seeds print their schedules to replay.
#[test]
fn rebinds_never_name_an_emptied_guest_part_over_random_park_and_restore_schedules() {
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            if n == 0 { 0 } else { self.next() % n }
        }
    }
    let mut failures = Vec::new();
    for seed in 0..300u64 {
        let mut rng = Rng(seed);
        let host_tier = rng.below(2) == 0;
        let facts = facts(64);
        let device = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
        let host = host_tier.then(|| {
            Arc::new(Mutex::new(
                SharedPool::tier(&facts, facts.host_chunks).unwrap().unwrap(),
            ))
        });
        let roles = Arc::new(Mutex::new(Roles::new(&device, host.as_ref()).unwrap()));
        let mut engine = GuestEngine::new(TOP_P, 2 * CHUNK);
        let procs = 2 + rng.below(3) as u128;
        let mut ids = Vec::new();
        for by in 1..=procs {
            ids.push((by, bind(&roles, &mut engine, by).unwrap()));
        }
        let mut parked = vec![false; procs as usize + 1];
        let mut schedule = vec![format!("host_tier={host_tier} procs={procs}")];
        let mut failed = None;
        for _ in 0..12 {
            let by = 1 + rng.below(procs as u64) as u128;
            if parked[by as usize] {
                let back = roles.lock().unwrap().prepare_unpark(holder(by));
                match back {
                    Ok(Some(txn)) => {
                        roles.lock().unwrap().commit(txn);
                        schedule.push(format!("restore {by}"));
                    }
                    Ok(None) => schedule.push(format!("restore {by} (nothing moved)")),
                    Err(error) => {
                        schedule.push(format!("restore {by} refused: {error}"));
                        continue;
                    }
                }
                parked[by as usize] = false;
                let stale = roles.lock().unwrap().stale(holder(by));
                for rebind in &stale {
                    if let Rebind::Instance {
                        id, scratch, pair, ..
                    } = rebind
                    {
                        let owner = ids.iter().find(|(_, at)| at == id).map(|(by, _)| *by);
                        if owner.is_some_and(|owner| parked[owner as usize]) {
                            failed = Some(format!("rebound {id:?} of parked process {owner:?}"));
                        } else if matches!(scratch, ScratchPlace::None)
                            || matches!(pair, ScratchPlace::None)
                        {
                            failed = Some(format!("{rebind:?} names an emptied part"));
                        }
                    }
                }
                for rebind in stale {
                    roles.lock().unwrap().rebound(super::Owner::of(&rebind));
                }
            } else {
                let park = roles.lock().unwrap().prepare_park(holder(by));
                if let Some(txn) = park {
                    roles.lock().unwrap().commit(txn);
                }
                parked[by as usize] = true;
                schedule.push(format!("park {by}"));
            }
            if failed.is_some() {
                break;
            }
        }
        if let Some(why) = failed {
            failures.push(format!("seed {seed}: {why}\n  schedule: {}", schedule.join(" -> ")));
            if failures.len() > 10 {
                break;
            }
        }
    }
    assert!(failures.is_empty(), "{} failing seeds:\n{}", failures.len(), failures.join("\n"));
}

/// Suspended once for its bind's room and restored, a process whose bind is
/// short again waits parked in the planner for the room, and binds once
/// served; it is suspended once only, never in a cycle of suspend, restore and
/// short.
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
        |_, _, _| false,
        {
            let suspends = suspends.clone();
            move |_, _, _| {
                suspends.set(suspends.get() + 1);
                true
            }
        },
        {
            let waits = waits.clone();
            move |_, _, _| {
                waits.set(waits.get() + 1);
                std::future::ready(Ok::<_, String>(crate::planner::BindRoomWait::Served))
            }
        },
        || {},
        |_| {},
        || std::future::ready(()),
    )
    .await;
    assert!(bound.is_ok(), "{bound:?}");
    assert_eq!(suspends.get(), 1, "suspended once for its room, never again");
    assert_eq!(waits.get(), 2, "short again after its restore, it waits parked for room");
    assert_eq!(binds.get(), 4);
}

/// Shape E's stall (Metal, unified, every segment committed to kv early):
/// kv pages fill every committed segment, so a bind needing a run of 3
/// whole `Instance` chunks finds none and waits for room
/// (`RolesError::BindShort`). Once the kv pages go (their owners evicted),
/// the chunks they emptied are free for any shape: the bind lands in
/// them, the planner's map and the pool's own count agree on what is free
/// (the same physical ids), and a run the map counts free is one the bind
/// takes.
#[test]
fn chunks_kv_filled_and_emptied_take_an_instance_run() {
    let Pools { device, roles } = pools_of(facts_in(64, 8));
    let list = device.lock().unwrap().list().expect("a kv list");
    let spc = u64::from(list.slots_per_chunk);
    // A chunk of kv pages at a time, until no chunk opens: every chunk
    // below the span held by kv.
    let mut kv: Vec<Vec<u32>> = Vec::new();
    while let Some(refs) = device.lock().unwrap().take_refs(list.pool, spc) {
        kv.push(refs);
    }
    assert!(!kv.is_empty());
    assert_eq!(device.lock().unwrap().pool().free_chunk_runs(1), 0, "kv holds every chunk");
    let (program, mut engine) = runs_of(8, 3, 1, 0);
    let error = bind_as(&roles, &mut engine, program, 1).expect_err("no free run");
    assert!(
        matches!(
            error.downcast_ref::<RolesError>(),
            Some(RolesError::BindShort(_))
        ),
        "short of room the pool holds: it waits, {error}"
    );
    roles.lock().unwrap().leave_guest_wait(holder(1));
    // Every kv owner evicted: its pages go back, and their chunks close.
    for refs in kv.drain(..) {
        device.lock().unwrap().give_refs(list.pool, &refs);
    }
    {
        let pool = device.lock().unwrap();
        let free = pool.pool().state_open_chunks();
        assert!(!free.is_empty());
        let runs = pool.pool().free_chunk_runs(3);
        assert!(runs >= 1, "the emptied chunks stand runs of 3");
        // The planner's map over the same pool: one notion of free, one
        // id space. Its cheapest run moves no one, and is one the pool
        // counts.
        let map = crate::planner::ChunkMap {
            held: Vec::new(),
            free,
            headroom: Some(pool.pool().headroom()),
            segment_chunks: pool.pool().segments().segment_chunks(),
        };
        assert_eq!(map.cheapest_runs(3, runs), Some(Vec::new()));
        assert_eq!(map.cheapest_runs(3, runs + 1), None, "the map counts no run the pool lacks");
    }
    bind_as(&roles, &mut engine, program, 1).expect("it lands in the chunks kv gave back");
    let piece = &pieces_of(&device, &engine.bindings[0].scratch)[0];
    assert_eq!(piece.len(), 3);
    assert!(piece.windows(2).all(|w| w[0].0 + 1 == w[1].0 && w[0].1 == w[1].1), "{piece:?}");
}

/// With no host tier (unified memory) two processes share a pair, both
/// parked where they lie (scratch and the pair's state given back). Their
/// restores run at once: while the first brings the pair back, the
/// second's must wait for that move (`PairParked`), as it does where a
/// host tier took the items. Let through, its instance came back stale
/// with its pair not yet on the device, so its restore named it no rebind
/// (`stale` skips an instance whose pair is away) and reported it
/// resident: its next fire ran on the scratch and pair places it held
/// before its park, given back and taken by others since.
#[test]
fn with_no_host_tier_a_second_restore_waits_for_the_pair_the_first_brings_back() {
    let facts = facts(64);
    let device = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let roles: RolesHandle = Arc::new(Mutex::new(Roles::new(&device, None).unwrap()));
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    let first = bind(&roles, &mut engine, 1).unwrap();
    let second = bind(&roles, &mut engine, 2).unwrap();
    park(&roles, 1);
    park(&roles, 2);
    let first_back = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .expect("the first brings the pair back");
    let concurrent = roles.lock().unwrap().prepare_unpark(holder(2));
    match concurrent {
        Err(RolesError::PairParked(_)) => {}
        Ok(Some(back)) => {
            roles.lock().unwrap().commit(back);
            let stale = roles.lock().unwrap().stale(holder(2));
            panic!(
                "the second restore went while its pair was moving; its rebinds: {stale:?} \
                 (instance {second:?} named: {})",
                stale
                    .iter()
                    .any(|rebind| matches!(rebind, Rebind::Instance { id, .. } if *id == second))
            );
        }
        other => panic!("the pair is moving: {other:?}"),
    }
    roles.lock().unwrap().commit(first_back);
    let pair = rebound_pair(&roles, 1, first);
    assert!(matches!(pair, ScratchPlace::Chunks { .. } | ScratchPlace::Run { .. }), "{pair:?}");
    // Once the pair has landed the second comes back, rebound to it.
    let back = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(2))
        .unwrap()
        .expect("it comes back");
    roles.lock().unwrap().commit(back);
    assert_eq!(rebound_pair(&roles, 2, second), pair, "rebound to where it landed");
    // Back but not yet rebound, it is no resident's: its restore waits.
    assert_eq!(roles.lock().unwrap().unbound(holder(2)), vec![second]);
    let stale = roles.lock().unwrap().stale(holder(2));
    for rebind in stale {
        roles.lock().unwrap().rebound(Owner::of(&rebind));
    }
    assert!(roles.lock().unwrap().unbound(holder(2)).is_empty(), "rebound: resident");
}

/// A process's instances parked where they lie (no host tier) are unbound
/// until its restore brings them back and the engine is told where: a
/// restore reports it resident only once none is, so no fire runs on the
/// places it gave back at its park. A process never parked has none.
#[test]
fn an_instance_is_unbound_from_its_park_until_its_rebind() {
    let Pools { roles, .. } = pools_no_host(facts_in(64, 8));
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    let id = bind(&roles, &mut engine, 1).unwrap();
    let other = bind(&roles, &mut engine, 2).unwrap();
    assert!(roles.lock().unwrap().unbound(holder(1)).is_empty(), "bound where it lies");
    park(&roles, 1);
    assert_eq!(roles.lock().unwrap().unbound(holder(1)), vec![id], "parked where it lies");
    assert!(roles.lock().unwrap().unbound(holder(2)).is_empty(), "the other stays bound");
    let back = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .expect("it comes back");
    assert_eq!(roles.lock().unwrap().unbound(holder(1)), vec![id], "moving");
    roles.lock().unwrap().commit(back);
    assert_eq!(roles.lock().unwrap().unbound(holder(1)), vec![id], "back, not rebound");
    let stale = roles.lock().unwrap().stale(holder(1));
    for rebind in stale {
        roles.lock().unwrap().rebound(Owner::of(&rebind));
    }
    assert!(roles.lock().unwrap().unbound(holder(1)).is_empty(), "rebound");
    roles.lock().unwrap().close_instance(id);
    roles.lock().unwrap().close_instance(other);
}

/// Where the engine fires instance `id` now: the scratch and pair its bind
/// named, then each rebind it was told since (the mock's own record).
fn fires_on(engine: &GuestEngine, id: InstanceId) -> (ScratchPlace, ScratchPlace) {
    let bound = &engine.bindings[usize::try_from(id.0 - 1).expect("an index")];
    let (mut scratch, mut pair) = (bound.scratch.clone(), bound.pair.clone());
    for rebind in &engine.rebinds {
        if let Rebind::Instance {
            id: at,
            scratch: to,
            pair: to_pair,
            ..
        } = rebind
            && *at == id
        {
            if *to != ScratchPlace::None {
                scratch = to.clone();
            }
            if *to_pair != ScratchPlace::None {
                pair = to_pair.clone();
            }
        }
    }
    (scratch, pair)
}

/// A mock engine's view: two processes share a pair on a pool with no host
/// tier, both park where they lie (scratch and the pair's state given back),
/// and a third binds into the room they gave back. Their restores run at once.
/// Whenever a process is fit to fire (no instance of its unbound,
/// `Roles::unbound`, what its restore waits on), the engine fires each of its
/// instances on the place it was rebound to: the roles' current place, apart
/// from every other live instance's. The second restore waits for the pair the
/// first brings back, and until its rebind lands it is not fit to fire: the
/// place its engine binding still names was given back at its park.
#[test]
fn with_no_host_tier_a_restored_instance_fires_on_its_rebound_place() {
    let Pools { roles, .. } = pools_no_host(facts_in(64, 8));
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    let a = bind(&roles, &mut engine, 1).unwrap();
    let b = bind(&roles, &mut engine, 2).unwrap();
    let parked_at = fires_on(&engine, b).0;
    park(&roles, 1);
    park(&roles, 2);
    // Another program's process binds in the room given back (one of the
    // same program would bring the pair back with its bind).
    let (other, mut other_engine) = runs_of(14, 1, 1, 9000);
    bind_as(&roles, &mut other_engine, other, 3).expect("the third binds in the room given back");
    let c_place = other_engine.bindings[0].scratch.clone();
    let lane = |roles: &RolesHandle, engine: &mut GuestEngine, by: u128| {
        let stale = roles.lock().unwrap().stale(holder(by));
        for rebind in stale {
            super::rebind(Some(roles), engine, rebind, |_, _, _| Ok(())).expect("rebound");
        }
    };
    let apart = |engine: &GuestEngine, live: &[InstanceId]| {
        let places: Vec<ScratchPlace> = live.iter().map(|&id| fires_on(engine, id).0).collect();
        let pair = fires_on(engine, live[0]).1;
        let mut all: Vec<&ScratchPlace> = places.iter().collect();
        all.push(&pair);
        all.push(&c_place);
        disjoint(&all)
    };

    let first = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .expect("the first brings the pair back");
    assert!(
        matches!(roles.lock().unwrap().prepare_unpark(holder(2)), Err(RolesError::PairParked(_))),
        "the second waits for the pair's move"
    );
    roles.lock().unwrap().commit(first);
    assert_eq!(roles.lock().unwrap().unbound(holder(1)), vec![a], "back, not yet rebound");
    lane(&roles, &mut engine, 1);
    assert!(roles.lock().unwrap().unbound(holder(1)).is_empty(), "the first may fire");
    apart(&engine, &[a]).expect("the first fires on its own place");

    let second = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(2))
        .unwrap()
        .expect("the second comes back once the pair has landed");
    roles.lock().unwrap().commit(second);
    assert_eq!(
        roles.lock().unwrap().unbound(holder(2)),
        vec![b],
        "back, its engine binding still where it parked: not fit to fire"
    );
    assert_eq!(fires_on(&engine, b).0, parked_at, "the engine still names its old place");
    lane(&roles, &mut engine, 2);
    assert!(roles.lock().unwrap().unbound(holder(2)).is_empty(), "the second may fire");
    assert_ne!(fires_on(&engine, b).0, parked_at, "it fires on fresh room");
    apart(&engine, &[a, b]).expect("each fires on its own place");
    assert_eq!(fires_on(&engine, a).1, fires_on(&engine, b).1, "one pair, where it landed");
}

/// A bind short of room asks for all it takes, not the part that came up
/// short first: an instance's scratch and its pair's state, each two whole
/// chunks, the pair's first bind, three chunks free. The scratch fits and
/// the pair does not, and the scratch goes back with the refusal; asked for
/// the pair's two alone, the restore its suspension waits on came back
/// once two stood, the bind took them for the scratch, came up short on
/// the pair, and was suspended again (bind, suspend, restore, short).
#[test]
fn a_bind_short_of_room_asks_for_its_scratch_and_its_pair_together() {
    let Pools { device, roles } = pools(64);
    let singles = fill_with_singles(&roles);
    for id in singles.iter().take(3) {
        roles.lock().unwrap().close_instance(*id);
    }
    assert_eq!(device.lock().unwrap().pool().free_chunks(), 3, "three chunks free");
    let mut engine = GuestEngine::new(2 * CHUNK, 2 * CHUNK);
    let error = bind(&roles, &mut engine, 1).expect_err("four chunks asked, three free");
    let short = error.downcast_ref::<RolesError>().expect("a roles error");
    assert!(matches!(short, RolesError::BindShort(_)), "a wait: {short}");
    assert_eq!(
        short.room_runs(),
        Some((1, 4, 0)),
        "its scratch's two chunks and its pair's two, together: {short}"
    );
}

/// A guest-only restore (kv 0, rs 0, asking 21 one-chunk pieces
/// against 22 state chunks): what an unpark lays fresh is what its parked
/// parts take, by shape, packed into chunks and into the free slots of the
/// chunks open for that shape, not a chunk a part. A process of many small
/// instances parked on the host asks exactly the chunks its unpark takes.
#[test]
fn an_unpark_asks_the_chunks_it_lays_not_a_chunk_a_part() {
    let Pools { device, roles } = pools(64);
    let (program, mut engine) = (PROGRAM, GuestEngine::new(SLOT, 0));
    let _ = program;
    // Many small instances of one process: each a few slots.
    for _ in 0..12 {
        bind(&roles, &mut engine, 1).unwrap();
    }
    park(&roles, 1);
    let free = || device.lock().unwrap().pool().free_chunks();
    let before = free();
    let asked: u32 = roles
        .lock()
        .unwrap()
        .unpark_need(holder(1))
        .iter()
        .map(|&(run, pieces)| run * pieces)
        .sum();
    let back = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .expect("it comes back");
    let laid = before - free();
    roles.lock().unwrap().commit(back);
    assert!(asked < 12, "not a chunk an instance: asked {asked}");
    assert_eq!(asked, laid, "asked what the unpark lays: {asked} vs {laid}");
}

/// A restore needing three runs of nine (open
/// stretches of 10, 10 and 9 in three segments, nothing in flight): where
/// the pool counts the runs a restore needs, the restore's own unpark
/// seats it there. The count and the take read the same runs
/// (`SlotPool::chunk_runs_for`).
#[test]
fn a_restore_needing_three_runs_of_nine_is_seated_in_stretches_of_ten_ten_and_nine() {
    let Pools { device, roles } = pools_no_host(facts_in(80, 16));
    let (program, mut engine) = runs_of(13, 9, 3, 0);
    bind_as(&roles, &mut engine, program, 1).expect("an empty pool seats it");
    park(&roles, 1);
    // Every free chunk taken by single-chunk states of others.
    let (singles, mut single) = runs_of(9, 1, 1, 1000);
    let mut held = Vec::new();
    for by in 0.. {
        match bind_as(&roles, &mut single, singles, 500 + by) {
            Ok(id) => held.push((id, single.bindings.last().unwrap().scratch.clone())),
            Err(_) => {
                roles.lock().unwrap().leave_guest_wait(holder(500 + by));
                break;
            }
        }
    }
    // Free stretches of 10 (segment 1), 10 (segment 2) and 9 (segment 3).
    let physical = |logical: u32| device.lock().unwrap().pool().physical(logical).unwrap();
    let stretch = |chunk: u32| (16..26).contains(&chunk) || (32..42).contains(&chunk) || (48..57).contains(&chunk);
    for (id, place) in &held {
        if chunks(place).iter().all(|&logical| stretch(physical(logical))) {
            roles.lock().unwrap().close_instance(*id);
        }
    }
    assert_eq!(
        device.lock().unwrap().pool().free_chunk_runs(9),
        3,
        "the pool counts three runs of nine"
    );
    let back = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .expect("seated where its runs stand")
        .expect("it comes back");
    roles.lock().unwrap().commit(back);
}

/// A run of two its need counted on is never refused: taking its parts in
/// its items' order, a run of slots would open the lowest free chunk,
/// splitting the only stretch the run of two stands in. An unpark lays its
/// runs of whole chunks first, the longest first, then its runs of slots:
/// free chunks `c`, `c + 1` and `d` hold a run of two and a run of slots,
/// and the unpark lands. (Its items lie in a hash map: five runs of slots
/// beside the run of two, four rounds, so the old order fails nearly
/// always.)
#[test]
fn an_unpark_lays_its_runs_of_chunks_before_its_runs_of_slots() {
    for round in 0..4u64 {
        let Pools { device, roles } = pools(64);
        // Its five runs of slots and its run of two chunks, then parked.
        let mut slots = GuestEngine::new(TOP_P, 0);
        slots.next = round * 10_000;
        for _ in 0..5 {
            bind(&roles, &mut slots, 1).unwrap();
        }
        let (program, mut two) = runs_of(13, 2, 1, round * 10_000 + 500);
        bind_as(&roles, &mut two, program, 1).unwrap();
        park(&roles, 1);
        // The device filled with single chunks, then three of them closed:
        // two adjacent at the lowest, one apart above.
        let (singles, mut single) = runs_of(9, 1, 1, round * 10_000 + 1000);
        let mut at: Vec<(u32, InstanceId)> = Vec::new();
        for by in 0.. {
            match bind_as(&roles, &mut single, singles, 500 + by) {
                Ok(id) => at.push((chunks(&single.bindings.last().unwrap().scratch)[0], id)),
                Err(_) => {
                    roles.lock().unwrap().leave_guest_wait(holder(500 + by));
                    break;
                }
            }
        }
        at.sort_unstable();
        let c = at
            .windows(2)
            .find(|pair| pair[1].0 == pair[0].0 + 1)
            .map(|pair| pair[0])
            .expect("two adjacent singles");
        let d = at
            .iter()
            .rev()
            .find(|(chunk, _)| *chunk > c.0 + 2)
            .copied()
            .expect("a single apart above");
        let next = at.iter().find(|(chunk, _)| *chunk == c.0 + 1).unwrap().1;
        for id in [c.1, next, d.1] {
            roles.lock().unwrap().close_instance(id);
        }
        assert_eq!(device.lock().unwrap().pool().free_chunks(), 3, "c, c + 1 and d free");
        let back = roles
            .lock()
            .unwrap()
            .prepare_unpark(holder(1))
            .expect("its run of two at c, its runs of slots at d: it lands")
            .expect("it comes back");
        roles.lock().unwrap().commit(back);
    }
}

/// A first-frame bind short of a run of three waits on a segment commit
/// (the segments left within the budget hold three more), and the walk
/// then has none left for it. A commit that will not come is no refusal: the
/// bind goes again, short of room the planner makes, and binds once it
/// has made it.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_whose_commit_will_not_come_waits_in_the_planner_and_binds() {
    use crate::governor::PoolTarget;
    const ENGINE: usize = 69;
    let Pools { device, roles } = pools_of(facts_committed(64, 8, 1));
    let grow = std::rc::Rc::new(std::cell::RefCell::new(crate::store::shared::SharedGrow::new(
        0,
        ENGINE,
        device.clone(),
    )));
    let _ = grow.borrow_mut().retarget(u64::from(u32::MAX));
    let (held_program, mut held_engine) = runs_of(13, 4, 2, 5000);
    let victim = bind_as(&roles, &mut held_engine, held_program, 7).expect("a resident");
    fill_with_singles(&roles);
    let (long, long_engine) = runs_of(8, 4, 2, 0);
    let mut long_engine = long_engine;
    let error = bind_as(&roles, &mut long_engine, long, 1).expect_err("no room yet");
    assert!(
        matches!(error.downcast_ref::<RolesError>(), Some(RolesError::GuestGrow { .. })),
        "a commit's to wait on first: {error}"
    );
    roles.lock().unwrap().leave_guest_wait(holder(1));
    let long_engine = std::rc::Rc::new(std::cell::RefCell::new(long_engine));
    let asked = std::rc::Rc::new(std::cell::Cell::new(0u32));
    let mapped = device.lock().unwrap().mapped_segments();
    let bound = super::bind_when_resident(
        holder(1),
        {
            let (roles, engine) = (roles.clone(), long_engine.clone());
            move || std::future::ready(bind_as(&roles, &mut engine.borrow_mut(), long, 1))
        },
        || std::future::pending::<Result<(), String>>(),
        {
            // The walk has no segment left for it: the target came down.
            let grow = grow.clone();
            move |_, _| {
                let _ = grow.borrow_mut().retarget(u64::from(mapped));
                std::future::ready(Err::<(), String>(
                    "no free run of 4 adjacent chunks, and no segment is left to commit within \
                     the budget"
                        .into(),
                ))
            }
        },
        {
            let roles = roles.clone();
            move || Some(roles.lock().unwrap().room_watch())
        },
        {
            let roles = roles.clone();
            move |run, pieces, singles| roles.lock().unwrap().room_free(run, pieces, singles)
        },
        |_, _, _| false,
        {
            // The planner: asked beside the commit, it makes room only once
            // the bind is short of room it makes (its second ask).
            let (roles, asked) = (roles.clone(), asked.clone());
            move |_, _, _| {
                asked.set(asked.get() + 1);
                let roles = roles.clone();
                let second = asked.get() >= 2;
                async move {
                    if !second {
                        return std::future::pending().await;
                    }
                    roles.lock().unwrap().close_instance(victim);
                    Ok::<_, String>(crate::planner::BindRoomWait::Served)
                }
            }
        },
        {
            let roles = roles.clone();
            move || roles.lock().unwrap().leave_guest_wait(holder(1))
        },
        |_| {},
        || std::future::ready(()),
    )
    .await;
    assert!(bound.is_ok(), "served, never refused: {bound:?}");
    assert_eq!(long_engine.borrow().bindings.len(), 1);
}

/// A disk-only load: a process's bind short of two runs of nine Instance
/// chunks, the process evicted to disk and restored, must not then wait
/// "for its instances' rebinds" until teardown, committed with an
/// instance whose rebind names a dead part no slots hold and nothing
/// making room for it. Such an instance's dead parts are its unpark's to take fresh, in
/// their class, before the restore commits: its need counts their runs, an
/// unpark short of them is refused (the restore waits queued for room the
/// planner makes, nothing committed), and once they stand it unparks, and
/// its rebind names them; nothing is left unbound.
#[test]
fn a_restore_takes_the_runs_its_rebinds_name_before_it_commits() {
    let Pools { device, roles } = pools_of(facts_in(64, 8));
    let (program, mut engine) = runs_of(8, 3, 2, 0);
    let id = bind_as(&roles, &mut engine, program, 1).expect("two runs of three");
    // Its scratch has no slots (given back, as a bind short of it left
    // it), the instance still bound on the device.
    {
        let mut roles = roles.lock().unwrap();
        let item = roles.items.get_mut(&Owner::Instance(id)).expect("bound");
        let dead: Vec<super::Part> = item.parts.iter().filter(|part| part.dead()).cloned().collect();
        assert!(!dead.is_empty(), "a scratch part");
        for part in item.parts.iter_mut().filter(|part| part.dead()) {
            part.refs.clear();
        }
        let device = std::sync::Arc::clone(&roles.device);
        roles.give(&device, &dead);
    }
    let singles = fill_with_singles(&roles);
    assert!(
        roles.lock().unwrap().unpark_need(holder(1)).contains(&(3, 2)),
        "its need counts the runs its rebind names: {:?}",
        roles.lock().unwrap().unpark_need(holder(1))
    );
    let short = roles.lock().unwrap().prepare_unpark(holder(1));
    assert!(
        matches!(short, Err(RolesError::UnparkShort(_))),
        "no two runs of three: refused, nothing committed: {short:?}"
    );
    assert_eq!(roles.lock().unwrap().unbound(holder(1)), vec![id], "not bound where it lies");
    assert_eq!(
        roles.lock().unwrap().unbound_after(holder(1), None),
        vec![id],
        "committed now, it would stay unbound"
    );
    // Room made (the planner's moves): it unparks and is rebound.
    for single in singles {
        roles.lock().unwrap().close_instance(single);
    }
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .expect("room stands")
        .expect("its scratch is taken");
    assert!(
        roles.lock().unwrap().unbound_after(holder(1), Some(&unpark)).is_empty(),
        "committed, every instance its rebinds name is placed"
    );
    roles.lock().unwrap().commit(unpark);
    let stale = roles.lock().unwrap().stale(holder(1));
    let [Rebind::Instance { id: rebound, scratch, .. }] = stale.as_slice() else {
        panic!("one rebind of the instance: {stale:?}");
    };
    assert_eq!(*rebound, id);
    assert_eq!(pieces_of(&device, scratch).len(), 2, "two runs: {scratch:?}");
    for rebind in stale {
        <GuestEngine as ::engine::Programs>::rebind(&mut engine, rebind).unwrap();
        roles.lock().unwrap().rebound(Owner::Instance(id));
    }
    assert!(roles.lock().unwrap().unbound(holder(1)).is_empty(), "nothing left unbound");
}

/// A restore's need must not read [(1, 1)] while its unpark is refused a
/// run of 465 slots: counted by the free slots partly filled chunks hold,
/// its run would ask no chunk, the map stand the need, and nothing move.
/// A restore's need
/// and its standing are the pool's own placement of every piece it takes:
/// a run of slots no partly filled chunk of its class holds asks a free
/// chunk, and the state its serve takes first is laid before it.
#[test]
fn a_restore_s_need_and_standing_are_the_placement_of_every_piece_it_takes() {
    let Pools { device, roles } = pools_of(facts_in(64, 8));
    let mut engine = GuestEngine::new(2 * MIB, 0);
    let id = bind(&roles, &mut engine, 1).expect("a run of slots");
    let shape = roles.lock().unwrap().shapes.of(SlotRole::Instance).expect("shape").pool;
    // Its scratch has no slots now (its rebind's to take).
    {
        let mut roles = roles.lock().unwrap();
        let item = roles.items.get_mut(&Owner::Instance(id)).expect("bound");
        let dead: Vec<super::Part> = item.parts.iter().filter(|part| part.dead()).cloned().collect();
        for part in item.parts.iter_mut().filter(|part| part.dead()) {
            part.refs.clear();
        }
        let device = std::sync::Arc::clone(&roles.device);
        roles.give(&device, &dead);
    }
    // A chunk of single slots, mostly free: its free slots outnumber the
    // run's, but a run of slots never lies among single slots.
    let singles_part = {
        let roles = roles.lock().unwrap();
        let device = std::sync::Arc::clone(&roles.device);
        roles.take(&device, SlotRole::Instance, 1000, false).expect("single slots")
    };
    let partial = device.lock().unwrap().pool().partial_free(shape);
    assert!(partial >= 512, "{partial} free single slots in a chunk");
    let singles = fill_with_singles(&roles);
    roles.lock().unwrap().close_instance(singles[0]);
    assert_eq!(device.lock().unwrap().pool().free_chunks(), 1, "one chunk free");
    assert_eq!(
        roles.lock().unwrap().unpark_need(holder(1)),
        vec![(1, 1)],
        "its run of slots asks the free chunk, whatever single slots stand free"
    );
    let placed = |takes: &[(crate::store::slots::ShapeId, u64)]| {
        roles.lock().unwrap().restore_placement(holder(1), takes).1
    };
    assert!(placed(&[]).stands(), "it stands");
    let chunk = u64::from(device.lock().unwrap().pool().shape(shape).unwrap().slots_per_chunk);
    assert!(
        !placed(&[(shape, chunk)]).stands(),
        "its serve's state takes that chunk first: it does not stand"
    );
    assert_eq!(placed(&[]).free_runs(1), 0, "nor with a pending bind's chunk beside it");
    drop(singles_part);
}

/// A beam pair's scratch (73 dead refs) whose first holder parked while
/// the pair was still another's, that other then ending: nothing else
/// moves the pair with the parked one. A move down for the parked holder
/// takes it, its dead parts given back.
#[test]
fn a_pair_left_behind_by_a_parked_holder_goes_with_its_move_down() {
    let facts = facts(64);
    let device = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let roles: RolesHandle = Arc::new(Mutex::new(Roles::new(&device, None).unwrap()));
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    let before = held(&device);
    let _first = bind(&roles, &mut engine, 1).unwrap();
    let second = bind(&roles, &mut engine, 2).unwrap();
    let park = roles.lock().unwrap().prepare_park(holder(1)).unwrap();
    roles.lock().unwrap().commit(park);
    // The other holder ends: the pair is the parked one's alone, on the device.
    roles.lock().unwrap().close_instance(second);
    let left = held(&device) - before;
    let down = roles.lock().unwrap().prepare_park_down(&[holder(1)], true);
    let txn = down.txn.expect("the pair left behind moves down with its parked holder");
    roles.lock().unwrap().commit(txn);
    assert!(
        held(&device) - before < left,
        "its dead slots given back: {} of {left} left",
        held(&device) - before
    );
}

/// A served restore's bind room must not be taken by another head between
/// its serve and its unpark (`PendingBindShort`). What its placement chose is held for
/// it: another holder's bind cannot take those chunks, its unpark takes
/// them, and its pending bind's run, held beside them, is its own bind's.
#[test]
fn a_served_restore_s_reserved_chunks_are_its_own_until_it_binds() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    bind(&roles, &mut engine, 1).unwrap();
    let park = roles.lock().unwrap().prepare_park(holder(1)).expect("it parks");
    roles.lock().unwrap().commit(park);
    // Held for its restore: its unpark's pieces and a pending bind's run
    // of one chunk.
    assert!(roles.lock().unwrap().reserve_restore(holder(1), &[(1, 1)]));
    let held_for = roles.lock().unwrap().claimed(holder(1));
    assert!(!held_for.is_empty(), "its chunks are held");
    // Another holder binds meanwhile: none of the held chunks are its.
    let mut other = GuestEngine::new(2 * CHUNK, 0);
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
    let theirs = chunks(&other.bindings[0].scratch);
    assert!(
        theirs.iter().all(|chunk| !held_for.contains(chunk)),
        "another's bind took none of them: {theirs:?} vs {held_for:?}"
    );
    // Its unpark takes from what was held; the bind's chunk stays held.
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .expect("it unparks");
    roles.lock().unwrap().commit(unpark);
    let left = roles.lock().unwrap().claimed(holder(1));
    assert_eq!(left.len(), 1, "its pending bind's chunk still held: {left:?}");
    // Its bind takes it, and the reservation is gone.
    bind(&roles, &mut engine, 1).unwrap();
    assert!(roles.lock().unwrap().claimed(holder(1)).is_empty(), "released once it bound");
    let _ = device;
}

/// A served restore's bind runs are held for it: a pending-bind check
/// counting free runs that left them out would find none every serve.
/// Checked by the one
/// placement with its own reservation open to it, its bind stands; to any
/// other take those chunks are not there.
#[test]
fn a_restore_whose_bind_runs_are_held_for_it_finds_them() {
    use crate::store::slots::OwnPiece;
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    bind(&roles, &mut engine, 1).unwrap();
    let park = roles.lock().unwrap().prepare_park(holder(1)).expect("it parks");
    roles.lock().unwrap().commit(park);
    assert!(roles.lock().unwrap().reserve_restore(holder(1), &[(1, 1)]));
    let unpark = roles.lock().unwrap().prepare_unpark(holder(1)).unwrap().expect("it unparks");
    roles.lock().unwrap().commit(unpark);
    // Every chunk free to others, and one more: only its own held chunk
    // makes the last.
    let others = device.lock().unwrap().pool().free_chunks();
    let bind_all = [OwnPiece::Chunks { run: 1, pieces: others + 1 }];
    assert!(
        !device.lock().unwrap().pool().place(&bind_all).stands(),
        "to any other take the held chunk is not there"
    );
    assert!(
        roles.lock().unwrap().place_claimed(holder(1), &bind_all).stands(),
        "to its own, it is"
    );
}

/// A bind taken on a pair its other members hold, whose last member parks
/// before the bind lands (the planner's moves run on other threads than
/// the lane's take and bind): the pair stays the bind's. Given back with
/// that park, its run would go to another program's bind while the new
/// member is bound on it (`guest::claims` refusing a bind over a live
/// pair's run).
#[test]
fn a_pair_a_bind_has_taken_stays_while_its_last_member_parks() {
    let Pools { device: _, roles } = pools(64);
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    bind(&roles, &mut engine, 1).unwrap();
    let pair_at = engine.bindings[0].pair.clone();
    learn_program(&roles, &mut engine, PROGRAM).unwrap();
    let binding = InstanceBinding {
        program: PROGRAM,
        ..InstanceBinding::default()
    };
    let taken = take_bind(Some(&roles), &mut engine, &binding, |_| false, Some(holder(2)), |_| true)
        .unwrap()
        .expect("slots taken");
    // The only member parks while the bind is on its way.
    park(&roles, 1);
    // Another program binds meanwhile: none of its room is the pair's.
    let mut other = GuestEngine::new(2 * CHUNK, PAIR);
    other.next = 100;
    learn_program(&roles, &mut other, ProgramId(8)).unwrap();
    bind_instance(
        Some(&roles),
        &mut other,
        &InstanceBinding {
            program: ProgramId(8),
            ..InstanceBinding::default()
        },
        Some(holder(3)),
        |_| false,
    )
    .unwrap();
    let theirs = &other.bindings[0];
    disjoint(&[&pair_at, &theirs.pair, &theirs.scratch])
        .expect("the pair the bind took is no other program's room");
    // The bind lands on the pair as it stands.
    bind_taken(Some(&roles), &mut engine, &binding, Some(holder(2)), Some(taken)).unwrap();
    assert_eq!(engine.bindings.last().unwrap().pair, pair_at, "bound on the pair it took");
}

/// A bind of a process that is not resident is refused before it takes
/// anything, by name, as one on a channel parked with it is: its bind
/// waits for its restore. Taken, its instance stood bound on the device
/// outside its process's park (the simulation's rescore on a process
/// evicted with its first program closed).
#[test]
fn a_bind_of_a_process_not_resident_waits_for_its_restore() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    learn_program(&roles, &mut engine, PROGRAM).unwrap();
    let before = held(&device);
    let binding = InstanceBinding {
        program: PROGRAM,
        ..InstanceBinding::default()
    };
    let error = take_bind(Some(&roles), &mut engine, &binding, |_| false, Some(holder(4)), |_| false)
        .expect_err("not resident: no take");
    assert!(
        matches!(short_of_wait(&error), Some(RolesError::HolderParked(by)) if *by == holder(4)),
        "{error:#}"
    );
    assert_eq!(held(&device), before, "nothing taken");
}

/// A bind taken for a process is on its way until the engine's bind lands
/// or it is given back: a park of that process waits for it
/// ([`Roles::binds_on_way`], the planner's eviction), so the instance parks
/// with the rest. Parked under it, the instance stood on the device outside
/// its process's park and its pair's run went to another program's bind
/// (the simulation with a yield between take and bind: `guest::claims`
/// refused binds over a live pair's run in nine seeds of 5000). A process
/// whose park began after the take's first read is refused past the take,
/// and the take goes back.
#[test]
fn a_bind_on_its_way_holds_its_process_s_park_until_it_lands() {
    let Pools { device, roles } = pools(64);
    let mut engine = GuestEngine::new(2 * CHUNK, PAIR);
    learn_program(&roles, &mut engine, PROGRAM).unwrap();
    let binding = InstanceBinding {
        program: PROGRAM,
        ..InstanceBinding::default()
    };
    let taken = take_bind(Some(&roles), &mut engine, &binding, |_| false, Some(holder(2)), |_| true)
        .unwrap()
        .expect("slots taken");
    assert!(roles.lock().unwrap().binds_on_way(holder(2)), "on its way");
    bind_taken(Some(&roles), &mut engine, &binding, Some(holder(2)), Some(taken)).unwrap();
    assert!(!roles.lock().unwrap().binds_on_way(holder(2)), "landed");
    // Resident at the first read, parked by the second: the take goes back.
    let before = held(&device);
    let reads = std::cell::Cell::new(0);
    let error = take_bind(Some(&roles), &mut engine, &binding, |_| false, Some(holder(3)), |_| {
        reads.set(reads.get() + 1);
        reads.get() == 1
    })
    .expect_err("parked past the take");
    assert!(
        matches!(short_of_wait(&error), Some(RolesError::HolderParked(by)) if *by == holder(3)),
        "{error:#}"
    );
    assert!(!roles.lock().unwrap().binds_on_way(holder(3)));
    assert_eq!(held(&device), before, "the take went back");
}
