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
    Owner, Roles, RolesError, RolesHandle, bind_instance, learn_program,
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
