//! REDESIGN §6: the registered programs' guest term in the rung arithmetic.
//! A rung of `r` tokens carries `min(r, max_lanes)` lanes; its activation
//! is the model's chunks plus the chunks the worst fire of those lanes
//! takes for the pairs open, each pair paying `fixed + lanes × a_lane` for
//! the members the fire gives it and nothing where it gives none
//! ([`::engine::GuestTerm`]). A bind of a program whose term raises the
//! span waits until the raise is funded and taken; the last instance's
//! close drops the pair, and the next lower gives the room back.

use std::sync::atomic::{AtomicU32, Ordering};

use ::engine::{FireFacts, InstanceId, ProgramId};
use tokio::sync::Notify;

use super::*;
use crate::store::shared::programs::{FireKey, Funder, Mover, ProgramOpen, fit_fire, fund_span};

const PROGRAM: ProgramId = ProgramId(7);
const LANES: u32 = 64;
/// The engine whose rung events these tests' raises tell.
const TEST_ENGINE: usize = 50;

/// `program` bound at the default extents.
fn key(program: ProgramId) -> FireKey {
    at(program, 1)
}

/// `program` bound at `rows` rows.
fn at(program: ProgramId, rows: u32) -> FireKey {
    FireKey {
        program,
        extents: ::engine::BindExtents {
            row_count: rows,
            ..::engine::BindExtents::default()
        },
    }
}

/// 36 MiB a fire, nothing a lane: 3 chunks at every rung, however many
/// instances are live.
fn fire_facts() -> FireFacts {
    FireFacts {
        fixed: 36 * MIB,
        ..FireFacts::default()
    }
}

/// An empty pool, every segment committed, at the 256 rung.
fn pool_at_256() -> (SharedPool, FakeEngine) {
    let mut pool = pool();
    pool.set_max_lanes(LANES);
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    pool.lower_rung(256).unwrap();
    assert_eq!(pool.pool().act_reserved(), 4);
    (pool, engine)
}

#[derive(Default)]
struct Planner {
    asked: AtomicU32,
    granted: Notify,
}

impl Funder for Arc<Planner> {
    async fn fund(&self, pages: u32) -> Result<bool, String> {
        self.asked.store(pages, Ordering::Release);
        self.granted.notified().await;
        Ok(true)
    }
}

/// A planner nothing should ask.
struct NoPlanner;

impl Funder for NoPlanner {
    async fn fund(&self, pages: u32) -> Result<bool, String> {
        panic!("asked for {pages} pages with nothing short");
    }
}

/// The engine's copy of a raise, on the fake engine.
struct Lane(Arc<Mutex<FakeEngine>>);

impl Mover for Lane {
    async fn evacuate(&self, evacuation: EngineEvacuation) -> Result<(), String> {
        self.0
            .lock()
            .unwrap()
            .evacuate(&evacuation)
            .map_err(|error| error.to_string())
    }

    async fn set_rung(&self, from: u32, to: u32) -> Result<(), String> {
        self.0
            .lock()
            .unwrap()
            .set_rung(from, to)
            .map_err(|error| error.to_string())
    }

    fn engine(&self) -> usize {
        TEST_ENGINE
    }

    fn epoch_now(&self) -> u64 {
        0
    }

    async fn check_table(&self, runtime: Vec<u32>) -> Result<(), String> {
        let engine = self.0.lock().unwrap();
        match crate::store::shared::programs::table_apart(engine.table.as_slice(), &runtime) {
            Some(apart) => Err(apart),
            None => Ok(()),
        }
    }
}

/// A term that leaves the smallest rung no room on the device refuses the
/// bind at once rather than waiting on room that never comes.
#[test]
fn a_fire_term_past_the_device_is_refused() {
    let (mut pool, _) = pool_at_256();
    pool.fire_facts_learned(
        key(PROGRAM),
        &FireFacts {
            fixed: u64::from(TOTAL) * CHUNK,
            ..FireFacts::default()
        },
    );
    let shared: Shared = Arc::new(Mutex::new(pool));
    assert!(ProgramOpen::new(&shared, key(PROGRAM), 0).is_err());
    assert_eq!(
        shared.lock().unwrap().pool().act_chunks(256),
        4,
        "a refused term counts nothing"
    );
}

/// 55 chunks a fire: the 2048 rung (12 + 55) no longer fits the 63 chunks
/// past the reserved one, the 256 rung (4 + 55) does.
fn wide_facts() -> FireFacts {
    FireFacts {
        fixed: 55 * CHUNK,
        ..FireFacts::default()
    }
}

/// A term the largest rung cannot hold with it cuts fires to the widest
/// rung that fits rather than refusing the bind: the wait funds that rung,
/// then holds until the walk has cut fires to it. Only where even the
/// smallest rung cannot fit is a bind refused.
#[tokio::test(flavor = "current_thread")]
async fn a_term_past_the_largest_rung_funds_the_widest_that_fits_and_cuts_fires_to_it() {
    let mut pool = pool();
    pool.set_max_lanes(LANES);
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(pool.rung(), 2048);
    pool.fire_facts_learned(key(PROGRAM), &wide_facts());
    let shared: Shared = Arc::new(Mutex::new(pool));
    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.widest_fit(), Some(256));
        assert_eq!(pool.rung_for(u32::MAX), 256, "fires are asked no wider");
        assert_eq!(pool.fit_cap(), 256);
    }
    // Fires are cut at the largest rung's width when the bind starts.
    let cap = Arc::new(AtomicU32::new(u32::MAX));
    let wait = tokio::spawn({
        let (shared, cap, engine) = (shared.clone(), cap.clone(), engine);
        async move {
            fund_span(
                &shared,
                move || cap.load(Ordering::Acquire),
                &NoPlanner,
                &Lane(Arc::new(Mutex::new(engine))),
            )
            .await
        }
    });
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        shared.lock().unwrap().pool().act_reserved(),
        4 + 55,
        "the 256 rung with its term is taken"
    );
    assert!(!wait.is_finished(), "fires are still cut wider than it");
    // The walk cuts them, whatever is asked.
    let step = RungWalk::default()
        .step(&shared.lock().unwrap(), u32::MAX, u32::MAX, u32::MAX, 1, 1)
        .unwrap();
    assert_eq!(step, RungStep::Cap(256));
    cap.store(256, Ordering::Release);
    crate::store::shared::rung_changed(TEST_ENGINE);
    wait.await.unwrap().unwrap();
    drop(open);

    // A term even the smallest rung cannot hold is refused.
    const HUGE: ProgramId = ProgramId(9);
    shared.lock().unwrap().fire_facts_learned(
        key(HUGE),
        &FireFacts {
            fixed: 63 * CHUNK,
            ..FireFacts::default()
        },
    );
    assert!(ProgramOpen::new(&shared, key(HUGE), 0).is_err());
    assert_eq!(shared.lock().unwrap().fit_cap(), u32::MAX, "no term counts");
}

/// A bind is never sent unfunded: the planner's refusal ends the wait with
/// an error; a raise the engine refuses is taken back, the span and the
/// state as they were, and refuses the bind too: nothing tries it again on
/// its own wake.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_whose_room_is_refused_fails_its_wait() {
    struct Refusing;
    impl Funder for Refusing {
        async fn fund(&self, _pages: u32) -> Result<bool, String> {
            Err("the planner refuses".into())
        }
    }
    struct NoCopy;
    impl Mover for NoCopy {
        async fn evacuate(&self, _evacuation: EngineEvacuation) -> Result<(), String> {
            Err("the engine refuses".into())
        }
        async fn set_rung(&self, _from: u32, _to: u32) -> Result<(), String> {
            Ok(())
        }

        fn engine(&self) -> usize {
            TEST_ENGINE
        }
        fn epoch_now(&self) -> u64 {
            0
        }
        // It keeps no table, and moves nothing.
        async fn check_table(&self, _runtime: Vec<u32>) -> Result<(), String> {
            Ok(())
        }
    }
    struct Granting;
    impl Funder for Granting {
        async fn fund(&self, _pages: u32) -> Result<bool, String> {
            Ok(true)
        }
    }

    let (mut pool, _) = pool_at_256();
    pool.fire_facts_learned(key(PROGRAM), &fire_facts());
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let _open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
    let error = fund_span(&shared, || 256, &Refusing, &NoCopy)
        .await
        .unwrap_err();
    assert!(error.contains("the planner refuses"));
    assert_eq!(shared.lock().unwrap().pool().act_reserved(), 4);

    // Room there, the copy refused: a refused raise, taken back, and the
    // bind refused (never sent), the span and the state as they were.
    let kept = all[all.len() / 2..].to_vec();
    {
        let mut pool = shared.lock().unwrap();
        pool.give_records(&all[..all.len() / 2]);
        pool.pool_mut().retire_through(u64::MAX);
    }
    let where_before: Vec<u32> = {
        let pool = shared.lock().unwrap();
        kept.iter()
            .flat_map(|&record| pool.record_slots(record))
            .map(|slot| pool.pool().physical(slot.chunk).unwrap())
            .collect()
    };
    let refused = fund_span(&shared, || 256, &Granting, &NoCopy)
        .await
        .unwrap_err();
    assert!(refused.contains("the engine refuses"), "{refused}");
    let pool = shared.lock().unwrap();
    assert_eq!(pool.pool().act_reserved(), 4, "the rung in hand");
    assert!(pool.copying().is_none());
    let where_after: Vec<u32> = kept
        .iter()
        .flat_map(|&record| pool.record_slots(record))
        .map(|slot| pool.pool().physical(slot.chunk).unwrap())
        .collect();
    assert_eq!(where_after, where_before, "the state where it lay");
    pool.pool().check();
}

/// One program at two extents is two pairs, and a fire that carries both
/// pays both `fixed`: 256 rows take 8 MiB fixed, 4096 rows 16x that.
fn rows_facts(rows: u32) -> FireFacts {
    FireFacts {
        fixed: u64::from(rows) * (32 << 10),
        ..FireFacts::default()
    }
}

/// A bind the engine refuses (`Error::Exhausted`: on an on-demand pool it
/// backs the newly opened pair's region at that bind, and the OS may refuse
/// it) is a refused bind. The binder's guard is all that counted the pair
/// (`SchedulerHandle::register_channels_bind` holds it across the bind and
/// returns the lane's error, dropping it; the lane counts an instance only
/// on a bind that succeeded): the term closes as if the bind never
/// happened, nothing is left counted, and a later bind of the same extents
/// opens it anew. That the widened chunks are free of state when the bind
/// reaches the engine is `a_larger_extent_bind_widens_the_span_before_it_is_sent`.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_the_engine_refuses_closes_its_term_as_if_it_never_happened() {
    struct Granting;
    impl Funder for Granting {
        async fn fund(&self, _pages: u32) -> Result<bool, String> {
            Ok(true)
        }
    }
    let (mut pool, engine) = pool_at_256();
    pool.fire_facts_learned(at(PROGRAM, 4096), &rows_facts(4096));
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));

    // The bind: its pair opened and funded, then the engine refuses it.
    let open = ProgramOpen::new(&shared, at(PROGRAM, 4096), 0).unwrap();
    fund_span(&shared, || 256, &Granting, &Lane(engine.clone()))
        .await
        .unwrap();
    assert_eq!(shared.lock().unwrap().pool().act_chunks(256), 4 + 8);
    let refused: Result<(), ::engine::Error> = Err(::engine::Error::Exhausted {
        resource: "placement heap",
        wanted: 128 << 20,
        available: 0,
    });
    // No instance is bound; the binder's error drops its guard.
    let error = refused.expect_err("the engine refused the bind");
    assert!(error.is_retryable());
    drop(open);
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.pool().act_chunks(256), 4, "the term closed");
        assert!(pool.pool().fire().is_zero(), "no term left counted");
        assert!(pool.knows_fire(&at(PROGRAM, 4096)), "its facts are kept");
        pool.pool().check();
    }
    // A later bind of the same extents opens the pair anew, and its close
    // leaves nothing behind either.
    let open = ProgramOpen::new(&shared, at(PROGRAM, 4096), 0).unwrap();
    shared
        .lock()
        .unwrap()
        .instance_bound(InstanceId(7), at(PROGRAM, 4096));
    drop(open);
    assert_eq!(shared.lock().unwrap().pool().act_chunks(256), 4 + 8);
    shared.lock().unwrap().instance_closed(InstanceId(7));
    fit_fire(&shared);
    assert!(shared.lock().unwrap().pool().fire().is_zero());
}

// ---- ruling 2: the rung the engine holds, and the span it states ----

/// Two chunks of batch tables a set: what [`SetEngine`] keeps standing for
/// every program with a live instance.
const SET: u64 = 2 * CHUNK;

/// An engine that keeps one set of tables standing a program, shared by
/// every live instance of it (Metal's batch tables), and states the total
/// as `Programs::standing_after`, `BoundInstance::standing_guest_bytes` and
/// `Programs::close` do.
#[derive(Default)]
struct SetEngine {
    /// The bytes of each program's set.
    sets: std::collections::HashMap<ProgramId, u64>,
    live: std::collections::HashMap<InstanceId, ProgramId>,
    next: u64,
    /// The binds the engine was asked for.
    binds: u32,
}

impl SetEngine {
    fn with(sets: &[(ProgramId, u64)]) -> Self {
        Self {
            sets: sets.iter().copied().collect(),
            ..Self::default()
        }
    }

    /// The total over the programs `live` holds an instance of, each set
    /// once.
    fn total(&self, live: impl Iterator<Item = ProgramId>) -> u64 {
        let programs: std::collections::HashSet<ProgramId> = live.collect();
        programs.iter().map(|program| self.sets[program]).sum()
    }
}

impl ::engine::Engine for SetEngine {
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
        "fake-sets"
    }

    fn load(&mut self, _request: ::engine::LoadRequest) -> ::engine::Result<::engine::Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(
        &mut self,
        _frame: &::engine::FrameSubmission,
    ) -> ::engine::Result<::engine::FrameTicket> {
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

impl ::engine::Programs for SetEngine {
    fn prepare(
        &mut self,
        _registration: &::engine::program::ProgramRegistration,
    ) -> ::engine::Result<::engine::ProgramCompile> {
        Err(::engine::Error::unsupported(
            ::engine::Engine::kind(self),
            "prepare_program",
        ))
    }

    fn finish(&mut self, _prepared: ::engine::Prepared) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(
            ::engine::Engine::kind(self),
            "finish_program",
        ))
    }

    fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(
            ::engine::Engine::kind(self),
            "close_program",
        ))
    }

    fn standing_after(&self, binding: &::engine::InstanceBinding) -> ::engine::Result<u64> {
        Ok(self.total(
            self.live
                .values()
                .copied()
                .chain(std::iter::once(binding.program)),
        ))
    }

    fn bind(
        &mut self,
        binding: &::engine::InstanceBinding,
    ) -> ::engine::Result<::engine::BoundInstance> {
        self.binds += 1;
        self.next += 1;
        let id = InstanceId(self.next);
        self.live.insert(id, binding.program);
        Ok(::engine::BoundInstance {
            id,
            program: binding.program,
            geometry: binding.geometry,
            standing_guest_bytes: self.total(self.live.values().copied()),
        })
    }

    fn close(&mut self, id: InstanceId) -> ::engine::Result<u64> {
        self.live.remove(&id);
        Ok(self.total(self.live.values().copied()))
    }

    fn register(
        &mut self,
        _registration: &::engine::ProgramRegistration,
    ) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(
            ::engine::Engine::kind(self),
            "register_program",
        ))
    }

    fn adapter(
        &mut self,
        _registration: &::engine::adapter::AdapterRegistration,
    ) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(
            ::engine::Engine::kind(self),
            "register_adapter",
        ))
    }
}

impl ::engine::Pool for SetEngine {
    fn chunk_table(&self) -> ::engine::Result<::engine::chunk::ChunkTable> {
        Err(::engine::Error::unsupported(
            ::engine::Engine::kind(self),
            "chunk_table",
        ))
    }

    fn commit(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(
            ::engine::Engine::kind(self),
            "commit_segments",
        ))
    }

    fn release(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(
            ::engine::Engine::kind(self),
            "release_segments",
        ))
    }

    fn evacuate(&mut self, _evacuation: &::engine::chunk::Evacuation) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(
            ::engine::Engine::kind(self),
            "evacuate",
        ))
    }
}

impl ::engine::Tiers for SetEngine {
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
        Err(::engine::Error::unsupported(
            ::engine::Engine::kind(self),
            "copy_slots",
        ))
    }

    fn read(
        &mut self,
        _slots: &[(::engine::chunk::ShapeId, Vec<u32>)],
    ) -> ::engine::Result<Vec<u8>> {
        Err(::engine::Error::unsupported(
            ::engine::Engine::kind(self),
            "read_tier_slots",
        ))
    }

    fn write(
        &mut self,
        _slots: &[(::engine::chunk::ShapeId, Vec<u32>)],
        _bytes: &[u8],
    ) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(
            ::engine::Engine::kind(self),
            "write_tier_slots",
        ))
    }

    fn copy_kv(&mut self, _copy: &::engine::KvCopy) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(
            ::engine::Engine::kind(self),
            "copy_kv",
        ))
    }
}

fn binding(program: ProgramId) -> ::engine::InstanceBinding {
    ::engine::InstanceBinding {
        program,
        extents: key(program).extents,
        ..::engine::InstanceBinding::default()
    }
}

/// A bind as the scheduler orders it: the engine asked the total it would
/// hold (on the lane), the pair and what the bind adds held, then the bind,
/// its total booked. Refused by name before the engine's bind where the
/// smallest rung cannot hold the total with the term.
fn bind_through(
    shared: &Shared,
    engine: &mut SetEngine,
    program: ProgramId,
) -> Result<InstanceId, String> {
    let total = <SetEngine as ::engine::Programs>::standing_after(engine, &binding(program))
        .map_err(|error| error.to_string())?;
    let added = shared.lock().unwrap().standing_added(total);
    let open = ProgramOpen::new(shared, key(program), added)?;
    let bound = <SetEngine as ::engine::Programs>::bind(engine, &binding(program))
        .map_err(|error| error.to_string())?;
    {
        let mut pool = shared.lock().unwrap();
        pool.instance_bound(bound.id, key(program));
        pool.standing_stated(bound.standing_guest_bytes);
    }
    drop(open);
    Ok(bound.id)
}

/// A bind's standing bytes are backed below the carve before the engine is
/// asked to bind, and a total no rung can hold refuses the bind by name
/// before the engine's bind is entered, the pool as it was.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_s_standing_bytes_are_backed_before_its_bind_and_a_total_past_the_span_is_refused() {
    let (pool, fake) = pool_at_256();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let mut engine = SetEngine::with(&[(PROGRAM, SET), (ProgramId(9), 63 * CHUNK)]);

    // Backed first: the span widens by the set while the bind is held.
    let total =
        <SetEngine as ::engine::Programs>::standing_after(&engine, &binding(PROGRAM)).unwrap();
    let added = shared.lock().unwrap().standing_added(total);
    assert_eq!(added, SET);
    let open = ProgramOpen::new(&shared, key(PROGRAM), added).unwrap();
    fund_span(
        &shared,
        || 256,
        &NoPlanner,
        &Lane(Arc::new(Mutex::new(fake))),
    )
    .await
    .unwrap();
    assert_eq!(shared.lock().unwrap().pool().act_reserved(), 4 + 2);
    assert_eq!(engine.binds, 0, "backed before the engine binds");
    let bound = <SetEngine as ::engine::Programs>::bind(&mut engine, &binding(PROGRAM)).unwrap();
    {
        let mut pool = shared.lock().unwrap();
        pool.instance_bound(bound.id, key(PROGRAM));
        pool.standing_stated(bound.standing_guest_bytes);
    }
    drop(open);
    assert_eq!(shared.lock().unwrap().pool().standing(), SET);

    // 63 chunks more: the smallest rung's one and the 65 standing are past
    // the 63 the device has.
    let why = bind_through(&shared, &mut engine, ProgramId(9))
        .expect_err("a total past the span is refused");
    assert!(why.contains("standing guest bytes"), "{why}");
    assert_eq!(engine.binds, 1, "the engine's bind was not entered");
    let pool = shared.lock().unwrap();
    assert_eq!(pool.pool().standing(), SET, "a refused bind holds nothing");
    assert_eq!(pool.pool().act_chunks(256), 4 + 2);
}

/// `program` bound with lanes that read out `rows` rows each.
fn reading(program: ProgramId, rows: u32) -> FireKey {
    FireKey {
        program,
        extents: ::engine::BindExtents {
            sampled_rows: rows,
            ..::engine::BindExtents::default()
        },
    }
}

/// A lane that alone reads out more rows than the engine's fixed readout
/// store holds is refused at its bind, by name, before anything counts; a
/// lane the store holds binds.
#[test]
fn a_lane_past_the_readout_store_is_refused_at_bind_by_name() {
    let (mut pool, _) = pool_at_256();
    pool.set_readouts(
        &::engine::ReadoutFacts {
            store_rows: Some(128),
            ..::engine::ReadoutFacts::default()
        },
        0,
        false,
    );
    for rows in [16, 128, 129, 4096] {
        pool.fire_facts_learned(reading(PROGRAM, rows), &FireFacts::default());
    }
    let shared: Shared = Arc::new(Mutex::new(pool));
    for rows in [129, 4096] {
        let refused = ProgramOpen::new(&shared, reading(PROGRAM, rows), 0)
            .err()
            .expect("a lane past the store is refused");
        assert!(
            refused.contains(&reading(PROGRAM, rows).to_string())
                && refused.contains(&format!("{rows} rows each"))
                && refused.contains("128 rows the engine's readout store holds"),
            "refused by name: {refused}"
        );
    }
    let _sixteen = ProgramOpen::new(&shared, reading(PROGRAM, 16), 0)
        .expect("sixteen rows a lane the store holds");
    let _whole = ProgramOpen::new(&shared, reading(PROGRAM, 128), 0)
        .expect("a lane of the store's own rows");
    // A float pass's bind: its rows land through the readout landing, not
    // the store, so its extents refuse nothing.
    let _float = ProgramOpen::float(&shared, reading(PROGRAM, 4096), 0)
        .expect("a float pass's bind is not held to the store");
}
