//! REDESIGN §6: the registered programs' guest term in the rung arithmetic.
//! A rung of `r` tokens carries `min(r, max_lanes)` lanes; its activation
//! is the model's chunks plus the chunks the worst fire of those lanes
//! takes for the pairs open, each pair paying `fixed + lanes × a_lane` for
//! the members the fire gives it and nothing where it gives none
//! ([`::engine::GuestTerm`]). A bind of a program whose term raises the
//! span waits until the raise is funded and taken; the last instance's
//! close drops the pair, and the next lower gives the room back.

use std::sync::atomic::{AtomicU32, Ordering};

use ::engine::{FireFacts, InstanceId, LaidLanes, ProgramId};
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

/// 20 MiB a fire and 256 KiB a lane: 16 lanes 24 MiB (2 chunks), 64 lanes
/// 36 MiB (3 chunks).
fn lane_facts() -> FireFacts {
    FireFacts {
        fixed: 20 * MIB,
        a_lane: 256 << 10,
        ..FireFacts::default()
    }
}

/// The engine closed instance `id` and the close committed: the lane drops
/// the count, the scheduler builds the term.
fn closed(pool: &mut SharedPool, id: u64) {
    pool.instance_closed(InstanceId(id));
    pool.refit();
}

/// Binds `n` instances of `key`, ids from `from`.
fn bind_live(pool: &mut SharedPool, key: FireKey, from: u64, n: u64) {
    for id in from..from + n {
        pool.instance_bound(InstanceId(id), key);
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

/// The planner as a raise's funding meets it: an ask parks until the test
/// has freed the room, then is granted.
/// A rise that cannot fire narrower (a piece no cut brings under the rung
/// held, a program's term), with its own process's kv in the span the
/// lowered rung lent and free chunks below: the rise moves that kv down and
/// binds at once. The requester's own pages are moved, never waited on,
/// and no planner is asked.
#[tokio::test(flavor = "current_thread")]
async fn a_rise_whose_own_kv_lies_in_the_lent_span_moves_it_and_binds() {
    let (mut pool, engine) = pool_at_256();
    let list = pool.list().expect("a kv list");
    let spc = u64::from(list.slots_per_chunk);
    // Others take every chunk below the span; the requester the lent ones.
    let span = pool.pool().total_chunks() - pool.pool().act_max();
    let below = u64::from(span - pool.pool().reserved_chunks()) * spc;
    let others = pool.take_refs(list.pool, below).expect("below the span");
    let lent = u64::from(pool.pool().act_max() - pool.pool().act_reserved()) * spc;
    let _own = pool.take_refs(list.pool, lent).expect("the lent span");
    assert_eq!(pool.pool().held_in_span(2048), 8);
    // Others give back as many chunks below as the requester holds above.
    pool.give_refs(list.pool, &others[..usize::try_from(lent).unwrap()]);
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        fund_span(&shared, || 2048, &NoPlanner, &Lane(engine.clone())),
    )
    .await
    .expect("no stall")
    .expect("the rise binds");
    let pool = shared.lock().unwrap();
    assert_eq!(pool.rung(), 2048);
    assert_eq!(pool.pool().held_in_span(2048), 0, "its kv moved below");
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

/// With no fire term, a program registered and bound leaves every rung's
/// activation and the walk as they were.
#[tokio::test(flavor = "current_thread")]
async fn a_zero_fire_term_leaves_the_rung_arithmetic_unchanged() {
    let (mut pool, engine) = pool_at_256();
    let before: Vec<u32> = [0, 16, 17, 256, 2048, u32::MAX]
        .iter()
        .map(|&rung| pool.pool().act_chunks(rung))
        .collect();
    assert_eq!(before, [1, 1, 4, 4, 12, 12]);
    pool.fire_facts_learned(key(PROGRAM), &FireFacts::default());
    pool.instance_bound(InstanceId(1), key(PROGRAM));
    let after: Vec<u32> = [0, 16, 17, 256, 2048, u32::MAX]
        .iter()
        .map(|&rung| pool.pool().act_chunks(rung))
        .collect();
    assert_eq!(after, before);
    assert_eq!(pool.rung(), 256);
    assert_eq!(pool.span_short(256), 0);
    // The span is the largest rung's; what the smallest leaves of it is
    // lent, the zero term taking none of it.
    let room = pool.room();
    assert_eq!((room.act_chunks, room.lent_chunks), (12, 11));
    let mut walk = RungWalk::default();
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 1, 1).unwrap(),
        RungStep::Wait,
        "at its rung: nothing moves"
    );
    // A bind waits on nothing, and asks no planner.
    let shared: Shared = Arc::new(Mutex::new(pool));
    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
    fund_span(
        &shared,
        || 256,
        &NoPlanner,
        &Lane(Arc::new(Mutex::new(engine))),
    )
    .await
    .unwrap();
    drop(open);
    assert_eq!(shared.lock().unwrap().pool().act_reserved(), 4);
}

/// A rung's activation is the model's chunks plus the term's at its lanes:
/// `min(r, FireLimits::max_lanes, the pair's max_lanes, its live
/// instances)`. Each bind grows it, each close shrinks it.
#[test]
fn the_fire_term_takes_its_chunks_at_every_rung_s_lanes() {
    let (mut pool, _) = pool_at_256();
    pool.fire_facts_learned(key(PROGRAM), &lane_facts());
    assert_eq!(pool.pool().act_chunks(256), 4, "registered, no instance");
    // One live instance: one lane at every rung, 20.25 MiB, 2 chunks.
    pool.instance_bound(InstanceId(1), key(PROGRAM));
    assert_eq!(pool.pool().act_chunks(16), 1 + 2);
    assert_eq!(
        pool.pool().act_chunks(2048),
        12 + 2,
        "one live instance: one lane"
    );
    // 64 live: 16 lanes at 16 (24 MiB, 2 chunks), 64 past it (36 MiB, 3).
    bind_live(&mut pool, key(PROGRAM), 2, 63);
    assert_eq!(pool.pool().act_chunks(16), 1 + 2);
    assert_eq!(pool.pool().act_chunks(256), 4 + 3);
    assert_eq!(pool.pool().act_chunks(2048), 12 + 3, "lanes capped at 64");
    // More live than the fire carries: still the fire's 64.
    bind_live(&mut pool, key(PROGRAM), 65, 10);
    assert_eq!(pool.pool().act_chunks(2048), 12 + 3);
    // The span set aside at 256 no longer holds the 256 rung.
    assert_eq!(pool.span_short(256), 3);
    assert_eq!(pool.rung(), 16, "4 chunks hold the 16 rung with its term");
    // Closes shrink it, a live instance a lane, to nothing with the last.
    for id in 1..=74 {
        closed(&mut pool, id);
    }
    assert_eq!(pool.pool().act_chunks(256), 4, "the last closed: gone");
    assert_eq!(pool.rung(), 256);
}

/// The term a rung backs for `n` live instances is what a fire of `n` lays,
/// for every `n` up to the fire's lanes and for each form an engine states
/// ([`LaidLanes`]): an engine laying the members themselves takes `n`
/// lanes' bytes, one laying the power of two over them takes 64 for 33
/// live. The lanes are the engine's own statement, read back here as the
/// engine lays them (`FireFacts::bytes`).
#[test]
fn the_term_a_rung_backs_for_n_live_is_what_a_fire_of_n_lays() {
    for laid in [LaidLanes::Exact, LaidLanes::PowerOfTwo] {
        let (mut pool, _) = pool_at_256();
        let facts = FireFacts {
            a_lane: MIB,
            laid,
            ..FireFacts::default()
        };
        pool.fire_facts_learned(key(PROGRAM), &facts);
        for n in 1..=u64::from(LANES) {
            pool.instance_bound(InstanceId(n), key(PROGRAM));
            let lays = facts.bytes(n);
            assert_eq!(
                u64::from(pool.pool().act_chunks(2048)),
                12 + lays.div_ceil(CHUNK),
                "{n} live, laid {laid:?}"
            );
        }
        let lanes = match laid {
            LaidLanes::Exact => 33,
            LaidLanes::PowerOfTwo => 64,
        };
        assert_eq!(facts.bytes(33), lanes * MIB, "33 members, laid {laid:?}");
    }
}

/// A pair's own cap holds over the lanes laid: 5 live of an engine laying
/// powers of two with flights of 4 take 4 lanes' bytes, not 8.
#[test]
fn a_pairs_own_cap_holds_over_the_lanes_laid() {
    let (mut pool, _) = pool_at_256();
    let facts = FireFacts {
        a_lane: MIB,
        max_lanes: 4,
        laid: LaidLanes::PowerOfTwo,
        ..FireFacts::default()
    };
    pool.fire_facts_learned(key(PROGRAM), &facts);
    for n in 1..=5 {
        pool.instance_bound(InstanceId(n), key(PROGRAM));
    }
    assert_eq!(
        u64::from(pool.pool().act_chunks(2048)),
        12 + (4 * MIB).div_ceil(CHUNK)
    );
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

/// The bind's wait: with state filling the pool, registering a program
/// whose term raises the 256 rung's span leaves the wait pending on the
/// planner; once the room is freed and granted, the raise moves the state
/// in the wider span down (the engine copying it) and takes the span 3
/// chunks wider. Closing the program drops the term, and the walk's next
/// lower gives the room back.
#[tokio::test(flavor = "current_thread")]
async fn a_program_s_fire_term_widens_the_span_before_its_bind_and_its_close_gives_it_back() {
    let (mut pool, engine) = pool_at_256();
    pool.fire_facts_learned(key(PROGRAM), &fire_facts());
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    let planner = Arc::new(Planner::default());

    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
    assert_eq!(shared.lock().unwrap().span_short(256), 3);
    let wait = tokio::spawn({
        let (shared, planner, engine) = (shared.clone(), planner.clone(), engine.clone());
        async move { fund_span(&shared, || 256, &planner, &Lane(engine)).await }
    });
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert!(!wait.is_finished(), "the bind waits on the raise's room");
    let asked = planner.asked.load(Ordering::Acquire);
    let (short, pages) = shared.lock().unwrap().raise_short(256);
    assert!(short > 0);
    assert_eq!(asked, pages, "the room a raise lacks, asked as a waiter");
    assert_eq!(
        shared.lock().unwrap().pool().act_reserved(),
        4,
        "nothing taken before it is funded"
    );

    // The planner frees room (half the state goes), then grants.
    {
        let mut pool = shared.lock().unwrap();
        pool.give_records(&all[..all.len() / 2]);
        pool.pool_mut().retire_through(u64::MAX);
    }
    planner.granted.notify_one();
    wait.await.unwrap().unwrap();
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.pool().act_reserved(), 4 + 3, "3 chunks wider");
        assert_eq!(pool.span_short(256), 0);
        assert_eq!(pool.rung(), 256);
        assert_eq!(pool.pool().state_limit(), TOTAL - 7);
        assert_eq!(
            engine.lock().unwrap().table.as_slice(),
            pool.pool().chunk_table(),
            "the engine copied what the raise moved"
        );
        assert!(pool.copying().is_none());
        // What the engine relies on when it backs its own span: no state
        // slot lies in the widened span once the wait is through, and the
        // copy that moved them is done (the wait awaited it).
        let in_span: Vec<u32> = all[all.len() / 2..]
            .iter()
            .flat_map(|&record| pool.record_slots(record))
            .map(|slot| pool.pool().physical(slot.chunk).unwrap())
            .filter(|&physical| physical >= TOTAL - 7)
            .collect();
        assert!(in_span.is_empty(), "state left in the span: {in_span:?}");
        assert!(
            engine
                .lock()
                .unwrap()
                .calls
                .iter()
                .any(|call| matches!(call, Call::Evacuate(moves) if !moves.is_empty())),
            "the raise moved state out through the engine's copy"
        );
    }

    // The bind went through: the instance holds the term past the guard.
    shared
        .lock()
        .unwrap()
        .instance_bound(InstanceId(1), key(PROGRAM));
    drop(open);
    assert_eq!(shared.lock().unwrap().pool().act_chunks(256), 7);

    // Its close drops the term; the walk lowers to give the room back once
    // the fires handed out by then retire.
    shared.lock().unwrap().instance_closed(InstanceId(1));
    fit_fire(&shared);
    let mut walk = RungWalk::default();
    let pool = shared.lock().unwrap();
    assert_eq!(pool.pool().act_chunks(256), 4);
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 3, 2).unwrap(),
        RungStep::Wait
    );
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 3, 3).unwrap(),
        RungStep::Lower(256)
    );
    drop(pool);
    shared.lock().unwrap().lower_rung(256).unwrap();
    let pool = shared.lock().unwrap();
    assert_eq!(pool.pool().act_reserved(), 4, "the room is back");
    assert_eq!(pool.pool().state_limit(), TOTAL - 4);
}

/// A program that flies at most 16 lanes a fire takes its term at 16 lanes
/// on every rung past 16 tokens, however many are live: `fixed + min(r,
/// max_lanes, program max_lanes, live) × a_lane`.
#[test]
fn a_program_s_own_lane_cap_bounds_its_term() {
    let (mut pool, _) = pool_at_256();
    pool.fire_facts_learned(
        key(PROGRAM),
        &FireFacts {
            max_lanes: 16,
            ..lane_facts()
        },
    );
    bind_live(&mut pool, key(PROGRAM), 1, 64);
    // 20 + 16 x 0.25 MiB = 24 MiB, 2 chunks, at 16, 256 and 2048 alike.
    assert_eq!(pool.pool().act_chunks(16), 1 + 2);
    assert_eq!(pool.pool().act_chunks(256), 4 + 2);
    assert_eq!(pool.pool().act_chunks(2048), 12 + 2);
    // Beside an uncapped program with 64 live: the two share one fire's 64
    // lanes, both `fixed` paid. The capped one lays no more than 16, so
    // the worst fire is 20 + 20 MiB and 64 lanes' 16 MiB whichever way
    // they fall: 56 MiB, 4 chunks (pair by pair it was 24 + 36 MiB).
    const OTHER: ProgramId = ProgramId(8);
    pool.fire_facts_learned(key(OTHER), &lane_facts());
    bind_live(&mut pool, key(OTHER), 100, 64);
    assert_eq!(pool.pool().act_chunks(256), 4 + 4);
    assert_eq!(pool.pool().fire().bytes(LANES), 56 * MIB);
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

/// A pool that states no rungs (Vulkan, wgpu, XLA, Metal before its span):
/// its activation is no part of it.
fn rungless_pool() -> (SharedPool, FakeEngine) {
    let mut facts = facts();
    facts.act_chunks = Vec::new();
    let mut pool = SharedPool::new(&facts, 256).unwrap();
    pool.set_max_lanes(LANES);
    assert!(pool.rungs().is_empty());
    (pool, FakeEngine::new())
}

/// The regression: on a pool without rungs every bind was refused, a zero
/// term included ("fires need 0 bytes ... 0 chunks of the 15173"). A zero
/// term never refuses, and the bind waits on nothing.
#[tokio::test(flavor = "current_thread")]
async fn a_zero_term_never_refuses_a_bind_on_a_pool_without_rungs() {
    let (mut pool, engine) = rungless_pool();
    pool.fire_facts_learned(key(PROGRAM), &FireFacts::default());
    let shared: Shared = Arc::new(Mutex::new(pool));
    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).expect("a zero term never refuses");
    fund_span(
        &shared,
        || u32::MAX,
        &NoPlanner,
        &Lane(Arc::new(Mutex::new(engine))),
    )
    .await
    .unwrap();
    drop(open);
}

/// On a pool without rungs a term, however large, is ignored: no bind is
/// refused for it and none waits on it.
#[tokio::test(flavor = "current_thread")]
async fn a_pool_without_rungs_skips_the_term_check() {
    let (mut pool, engine) = rungless_pool();
    pool.fire_facts_learned(key(PROGRAM), &fire_facts());
    pool.fire_facts_learned(
        key(ProgramId(9)),
        &FireFacts {
            fixed: u64::from(TOTAL) * CHUNK,
            ..FireFacts::default()
        },
    );
    let shared: Shared = Arc::new(Mutex::new(pool));
    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).expect("the term is ignored");
    let huge =
        ProgramOpen::new(&shared, key(ProgramId(9)), 0).expect("past the device, still ignored");
    {
        let pool = shared.lock().unwrap();
        assert_eq!(
            pool.pool().act_chunks(16),
            0,
            "no activation to fold it into"
        );
        assert_eq!(pool.span_short(u32::MAX), 0);
        assert_eq!(pool.fit_cap(), u32::MAX, "fires uncut");
    }
    fund_span(
        &shared,
        || u32::MAX,
        &NoPlanner,
        &Lane(Arc::new(Mutex::new(engine))),
    )
    .await
    .unwrap();
    drop((open, huge));
}

/// On a pool with rungs a zero term never refuses either.
#[test]
fn a_zero_term_never_refuses_a_bind_on_a_pool_with_rungs() {
    let (mut pool, _) = pool_at_256();
    pool.fire_facts_learned(key(PROGRAM), &FireFacts::default());
    let shared: Shared = Arc::new(Mutex::new(pool));
    assert!(ProgramOpen::new(&shared, key(PROGRAM), 0).is_ok());
}

/// One program at two extents is two pairs, and a fire that carries both
/// pays both `fixed`: 256 rows take 8 MiB fixed, 4096 rows 16x that.
fn rows_facts(rows: u32) -> FireFacts {
    FireFacts {
        fixed: u64::from(rows) * (32 << 10),
        ..FireFacts::default()
    }
}

#[test]
fn two_extents_of_one_program_sum_and_each_closes_with_its_last_bind() {
    let (mut pool, _) = pool_at_256();
    pool.fire_facts_learned(at(PROGRAM, 256), &rows_facts(256));
    pool.fire_facts_learned(at(PROGRAM, 4096), &rows_facts(4096));
    pool.instance_bound(InstanceId(1), at(PROGRAM, 256));
    // 8 MiB: 1 chunk.
    assert_eq!(pool.pool().act_chunks(256), 4 + 1);
    pool.instance_bound(InstanceId(2), at(PROGRAM, 4096));
    pool.instance_bound(InstanceId(3), at(PROGRAM, 256));
    // 8 + 128 MiB: 9 chunks.
    assert_eq!(pool.pool().act_chunks(256), 4 + 9, "the two pairs sum");
    closed(&mut pool, 2);
    assert_eq!(
        pool.pool().act_chunks(256),
        4 + 1,
        "4096 rows: its last bind closed"
    );
    closed(&mut pool, 1);
    assert_eq!(
        pool.pool().act_chunks(256),
        4 + 1,
        "256 rows: one bind left"
    );
    closed(&mut pool, 3);
    assert_eq!(pool.pool().act_chunks(256), 4, "every pair closed");
}

/// A bind at larger extents than the ones live widens the span before it is
/// sent: its pair's term is opened and funded, the state in the wider span
/// moved down by the engine's copy, while the smaller pair's stays.
#[tokio::test(flavor = "current_thread")]
async fn a_larger_extent_bind_widens_the_span_before_it_is_sent() {
    struct Granting;
    impl Funder for Granting {
        async fn fund(&self, _pages: u32) -> Result<bool, String> {
            Ok(true)
        }
    }
    let (mut pool, engine) = pool_at_256();
    pool.fire_facts_learned(at(PROGRAM, 256), &rows_facts(256));
    pool.fire_facts_learned(at(PROGRAM, 4096), &rows_facts(4096));
    // The 256-row pair live and funded: 5 chunks at the 256 rung.
    pool.instance_bound(InstanceId(1), at(PROGRAM, 256));
    let evacuation = pool.plan_raise(256).unwrap();
    pool.apply_rung(&evacuation, 0).unwrap();
    assert_eq!(pool.pool().act_reserved(), 5);
    // State right under the span.
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    pool.give_records(&all[..all.len() / 2]);
    pool.pool_mut().retire_through(u64::MAX);
    let kept = all[all.len() / 2..].to_vec();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));

    let open = ProgramOpen::new(&shared, at(PROGRAM, 4096), 0).unwrap();
    assert_eq!(shared.lock().unwrap().span_short(256), 8);
    fund_span(&shared, || 256, &Granting, &Lane(engine.clone()))
        .await
        .unwrap();
    // What the bind is sent after.
    let pool = shared.lock().unwrap();
    assert_eq!(pool.pool().act_reserved(), 4 + 9);
    let in_span = kept
        .iter()
        .flat_map(|&record| pool.record_slots(record))
        .map(|slot| pool.pool().physical(slot.chunk).unwrap())
        .filter(|&physical| physical >= TOTAL - 13)
        .count();
    assert_eq!(in_span, 0, "no state in the widened span");
    assert_eq!(
        engine.lock().unwrap().table.as_slice(),
        pool.pool().chunk_table()
    );
    drop(pool);
    drop(open);
}

/// What the CUDA engine relies on: for the same live pairs, the chunks the
/// terms take never fall as the rung rises (more lanes, each program's own
/// cap and the fire's taken), and neither does a rung's activation.
#[test]
fn the_fire_chunks_never_fall_as_the_rung_rises() {
    let mut facts = facts();
    // Every power of two from 1 to 2048.
    facts.act_chunks = (0..12).map(|k| (1u32 << k, 1 + k / 4)).collect();
    let mut pool = SharedPool::new(&facts, 256).unwrap();
    pool.set_max_lanes(LANES);
    let base: Vec<u32> = pool
        .rungs()
        .iter()
        .map(|&rung| pool.pool().act_chunks(rung))
        .collect();
    let terms = [
        (
            at(PROGRAM, 256),
            FireFacts {
                fixed: 3 * MIB,
                a_lane: 384 << 10,
                max_lanes: 0,
                ..FireFacts::default()
            },
        ),
        (
            at(PROGRAM, 4096),
            FireFacts {
                fixed: 20 * MIB,
                a_lane: 1 << 20,
                max_lanes: 8,
                ..FireFacts::default()
            },
        ),
        (
            key(ProgramId(8)),
            FireFacts {
                fixed: 0,
                a_lane: 100 << 10,
                max_lanes: 32,
                ..FireFacts::default()
            },
        ),
    ];
    for (n, (key, facts)) in terms.iter().enumerate() {
        pool.fire_facts_learned(*key, facts);
        pool.instance_bound(InstanceId(n as u64), *key);
        let (mut last_fire, mut last_act) = (0, 0);
        for (&rung, &base) in pool.rungs().iter().zip(&base) {
            let act = pool.pool().act_chunks(rung);
            let fire = act - base;
            assert!(fire >= last_fire, "rung {rung}: {fire} < {last_fire}");
            assert!(act >= last_act, "rung {rung}");
            (last_fire, last_act) = (fire, act);
        }
        assert!(last_fire > 0);
    }
}

/// What the CUDA engine relies on: a lower lends the chunks under the
/// smaller span to state only once every fire cut to the old rung has
/// retired, so the old carve and the program's groups are free before
/// state enters them. Here the span narrows as a pair's last bind closes.
#[test]
fn a_lower_lends_only_after_every_fire_cut_to_the_old_rung_has_retired() {
    let (mut pool, _) = pool_at_256();
    pool.fire_facts_learned(key(PROGRAM), &fire_facts());
    pool.instance_bound(InstanceId(1), key(PROGRAM));
    let evacuation = pool.plan_raise(256).unwrap();
    pool.apply_rung(&evacuation, 0).unwrap();
    assert_eq!(pool.pool().act_reserved(), 7);
    closed(&mut pool, 1);
    assert_eq!(pool.pool().act_chunks(256), 4);
    let mut walk = RungWalk::default();
    // A step wider than the cap is still in flight: wait for the grant.
    assert_eq!(
        walk.step(&pool, 256, 256, u32::MAX, 9, 9).unwrap(),
        RungStep::Wait
    );
    // Granted with fires through 12 handed out: wait until they retire.
    for retired in 9..12 {
        assert_eq!(
            walk.step(&pool, 256, 256, 256, 12, retired).unwrap(),
            RungStep::Wait,
            "fire 12 not retired at {retired}"
        );
        assert_eq!(pool.pool().state_limit(), TOTAL - 7, "nothing lent yet");
    }
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 12, 12).unwrap(),
        RungStep::Lower(256)
    );
    pool.lower_rung(256).unwrap();
    assert_eq!(pool.pool().state_limit(), TOTAL - 4, "lent once retired");
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

/// Each bind of a pair grows its term by a live instance, and widens the
/// span through fund_span before the bind is sent; each close shrinks it,
/// and the next lower gives the room back.
#[tokio::test(flavor = "current_thread")]
async fn each_bind_grows_the_term_and_each_close_shrinks_it() {
    let (mut pool, engine) = pool_at_256();
    // 8 MiB a lane: two live take 1 chunk, three take 2.
    pool.fire_facts_learned(
        key(PROGRAM),
        &FireFacts {
            a_lane: 8 * MIB,
            ..FireFacts::default()
        },
    );
    bind_live(&mut pool, key(PROGRAM), 1, 2);
    let evacuation = pool.plan_raise(256).unwrap();
    pool.apply_rung(&evacuation, 0).unwrap();
    assert_eq!(pool.pool().act_reserved(), 4 + 1);
    let shared: Shared = Arc::new(Mutex::new(pool));

    // A third bind: its guard grows the term before it is sent.
    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
    assert_eq!(
        shared.lock().unwrap().span_short(256),
        1,
        "three live: one chunk more"
    );
    fund_span(
        &shared,
        || 256,
        &NoPlanner,
        &Lane(Arc::new(Mutex::new(engine))),
    )
    .await
    .unwrap();
    assert_eq!(
        shared.lock().unwrap().pool().act_reserved(),
        4 + 2,
        "widened before the bind"
    );
    shared
        .lock()
        .unwrap()
        .instance_bound(InstanceId(3), key(PROGRAM));
    drop(open);
    assert_eq!(shared.lock().unwrap().pool().act_chunks(256), 4 + 2);

    // A close shrinks it; the walk lowers once fires cut to the old span
    // have retired.
    shared.lock().unwrap().instance_closed(InstanceId(3));
    fit_fire(&shared);
    let pool = shared.lock().unwrap();
    assert_eq!(pool.pool().act_chunks(256), 4 + 1, "two live again");
    let mut walk = RungWalk::default();
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 5, 4).unwrap(),
        RungStep::Wait
    );
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 5, 5).unwrap(),
        RungStep::Lower(256)
    );
    drop(pool);
    shared.lock().unwrap().lower_rung(256).unwrap();
    assert_eq!(
        shared.lock().unwrap().pool().act_reserved(),
        4 + 1,
        "the room is back"
    );
}

/// The term is a fire's, not a pair's: fourteen pairs of one sampler (one
/// program at fourteen extents), 64 instances bound of each, take what one
/// fire's 64 lanes take however they fall among the pairs, and each pair's
/// `fixed` once where the fire carries it. Pair by pair it was fourteen
/// times a fire's lanes.
#[test]
fn fourteen_pairs_of_one_sampler_count_as_one_fire_s_lanes() {
    for laid in [LaidLanes::Exact, LaidLanes::PowerOfTwo] {
        let (mut pool, _) = pool_at_256();
        let facts = FireFacts {
            fixed: 2 * MIB,
            a_lane: MIB,
            laid,
            ..FireFacts::default()
        };
        for pair in 0..14u32 {
            let key = at(PROGRAM, 256 + pair);
            pool.fire_facts_learned(key, &facts);
            bind_live(&mut pool, key, u64::from(pair) * 100, 64);
        }
        let by_pair = 14 * facts.bytes(64);
        assert_eq!(by_pair, 14 * 66 * MIB, "what the sum over pairs set aside");
        let (at_16, at_64) = match laid {
            // Every pair carried, the fire's lanes among them: 14 x 2 MiB
            // and 16 lanes, 14 x 2 MiB and 64.
            LaidLanes::Exact => (44 * MIB, 92 * MIB),
            // Laid by powers of two a split lays more than the fire's
            // lanes, under twice them: every pair carried, 16 members as
            // one pair's 3 and thirteen of one lay 17 lanes, 64 as 33, 17,
            // 3 and eleven of one lay 111.
            LaidLanes::PowerOfTwo => ((28 + 17) * MIB, (28 + 111) * MIB),
        };
        let term = pool.pool().fire();
        assert_eq!(term.bytes(16), at_16, "laid {laid:?}");
        assert_eq!(term.bytes(LANES), at_64, "laid {laid:?}");
        assert_eq!(
            u64::from(pool.pool().act_chunks(2048)),
            12 + at_64.div_ceil(CHUNK),
            "laid {laid:?}"
        );
        assert_eq!(
            u64::from(pool.pool().act_chunks(16)),
            1 + at_16.div_ceil(CHUNK),
            "laid {laid:?}"
        );
    }
}

/// Who builds the table, and when. A bind builds the one it would leave on
/// its own task and counts with it. The lane's bind takes over the count
/// the bind's guard held, so nothing changes there and nothing is built on
/// the lane; the guard then gives nothing back. A close drops the count on
/// the lane and leaves the term as it was, the instance still counted,
/// until the close's commit builds it.
#[test]
fn the_lane_s_bind_takes_over_the_guard_s_count_and_a_close_is_built_at_its_commit() {
    let (mut pool, _) = pool_at_256();
    // A chunk a live instance.
    pool.fire_facts_learned(
        key(PROGRAM),
        &FireFacts {
            a_lane: CHUNK,
            ..FireFacts::default()
        },
    );
    let shared: Shared = Arc::new(Mutex::new(pool));
    let act = || shared.lock().unwrap().pool().act_chunks(256);

    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
    assert_eq!(act(), 4 + 1, "one binding");
    shared
        .lock()
        .unwrap()
        .instance_bound(InstanceId(1), key(PROGRAM));
    assert_eq!(act(), 4 + 1, "the instance took the guard's count");
    drop(open);
    assert_eq!(act(), 4 + 1, "the guard gives nothing back");

    // Two binds under way, one bound, the other refused by the engine:
    // its guard's count goes, the bound one's stays.
    let (first, second) = (
        ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap(),
        ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap(),
    );
    assert_eq!(act(), 4 + 3);
    shared
        .lock()
        .unwrap()
        .instance_bound(InstanceId(2), key(PROGRAM));
    drop(first);
    drop(second);
    assert_eq!(act(), 4 + 2, "two bound, none binding");

    // A bind whose task was dropped before the lane reached it: no guard
    // is left, and the lane counts the instance itself.
    drop(ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap());
    assert_eq!(act(), 4 + 2);
    shared
        .lock()
        .unwrap()
        .instance_bound(InstanceId(3), key(PROGRAM));
    assert_eq!(act(), 4 + 3);

    // A close: counted until its commit builds the term.
    shared.lock().unwrap().instance_closed(InstanceId(3));
    assert_eq!(act(), 4 + 3, "the lane builds nothing");
    fit_fire(&shared);
    assert_eq!(act(), 4 + 2, "built at the commit");
    // A bind that follows a close not yet committed builds from what is
    // open: the closed instance is gone from its table.
    shared.lock().unwrap().instance_closed(InstanceId(2));
    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
    assert_eq!(act(), 4 + 2, "one bound, one binding");
    drop(open);
    shared.lock().unwrap().instance_closed(InstanceId(1));
    fit_fire(&shared);
    assert!(shared.lock().unwrap().pool().fire().is_zero());
}

/// A bind refused for the term it would leave leaves the pool as it found
/// it: the pair is not counted, the term of the pairs open stands, and the
/// cap fires may take never saw the refused one.
#[test]
fn a_refused_bind_leaves_the_term_of_the_pairs_open() {
    const HUGE: ProgramId = ProgramId(9);
    let (mut pool, _) = pool_at_256();
    pool.fire_facts_learned(key(PROGRAM), &fire_facts());
    pool.fire_facts_learned(
        key(HUGE),
        &FireFacts {
            fixed: u64::from(TOTAL) * CHUNK,
            ..FireFacts::default()
        },
    );
    pool.instance_bound(InstanceId(1), key(PROGRAM));
    let shared: Shared = Arc::new(Mutex::new(pool));
    let before = shared.lock().unwrap().pool().fire().clone();
    let refused = ProgramOpen::new(&shared, key(HUGE), 0)
        .err()
        .expect("past the device");
    assert!(
        refused.contains("the smallest rung would then take"),
        "{refused}"
    );
    let pool = shared.lock().unwrap();
    assert_eq!(pool.pool().fire(), &before);
    assert_eq!(pool.pool().act_chunks(256), 4 + 3);
    assert_eq!(pool.fit_cap(), u32::MAX);
}

// ---- ruling 2: the rung the engine holds, and the span it states ----

/// The fixture's pool with its span backed on demand, stating as Metal
/// does that its activation span covers the segments from the one the
/// largest rung's carve starts in (6 and 7: 16 chunks); as it loads, the
/// prefix through segment 5 committed. The engine holds the smallest rung.
fn on_demand_loaded() -> (SharedPool, FakeEngine) {
    let mut facts = facts();
    facts.span_backed_on_demand = true;
    facts.act_span_chunks = Some(2 * SEGMENT);
    let mut pool = SharedPool::new(&facts, 256).unwrap();
    pool.set_max_lanes(LANES);
    let mut engine = FakeEngine::on_demand();
    assert_eq!(
        walk_segments(&mut pool, 6, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(pool.told(), 16, "the load's rung");
    assert_eq!(pool.act_most(), 2 * SEGMENT);
    (pool, engine)
}

/// [`on_demand_loaded`], the engine told `rung` by a raise from the load's.
fn on_demand_at(rung: u32) -> (SharedPool, FakeEngine) {
    let (mut pool, engine) = on_demand_loaded();
    let evacuation = pool.plan_raise(rung).unwrap();
    assert_eq!(pool.raise_applied(&evacuation).unwrap(), Some((16, rung)));
    pool.raise_copied(0);
    assert_eq!(pool.told(), rung);
    (pool, engine)
}

/// Every `Engine::set_rung(from, to)` `engine` was sent.
fn notices(engine: &FakeEngine) -> Vec<(u32, u32)> {
    engine
        .calls
        .iter()
        .filter_map(|call| match call {
            Call::Rung(from, to) => Some((*from, *to)),
            _ => None,
        })
        .collect()
}

/// A planner that grants what it is asked.
struct Grants;

impl Funder for Grants {
    async fn fund(&self, _pages: u32) -> Result<bool, String> {
        Ok(true)
    }
}

/// `Engine::set_rung(from, to)` is sent only where the rung the engine
/// holds differs. Each of 64 binds at a standing rung grows the guest
/// term (twice by a chunk); the room is set aside before the bind is sent,
/// where the engine covers it inside the bind, with no notice, no trip
/// through the engine's lane, no fence and no `copying`: the binds do not
/// wait on the engine, nor on one another.
#[tokio::test(flavor = "current_thread")]
async fn binds_at_a_standing_rung_send_the_engine_no_notice() {
    let (mut pool, engine) = on_demand_at(256);
    pool.fire_facts_learned(key(PROGRAM), &lane_facts());
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    let before = engine.lock().unwrap().calls.clone();
    let mut reserved = Vec::new();
    for id in 1..=u64::from(LANES) {
        let open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
        {
            // A growth that moves nothing waits on nothing.
            let pool = shared.lock().unwrap();
            let planned = pool.plan_raise(256).unwrap();
            assert!(planned.moves.is_empty());
            assert!(!pool.raise_waits(&planned), "bind {id}");
        }
        fund_span(&shared, || 256, &NoPlanner, &Lane(engine.clone()))
            .await
            .unwrap();
        let mut pool = shared.lock().unwrap();
        assert!(pool.copying().is_none(), "bind {id}");
        assert_eq!(pool.told(), 256, "bind {id}");
        assert_eq!(pool.span_short(256), 0, "funded before bind {id} is sent");
        pool.instance_bound(InstanceId(id), key(PROGRAM));
        reserved.push(pool.pool().act_reserved());
        drop(pool);
        drop(open);
    }
    // 20 MiB and 256 KiB a lane: 2 chunks through 48 lanes, 3 past them.
    assert_eq!(reserved[0], 4 + 2, "the first bind's term");
    assert_eq!(reserved[47], 4 + 2);
    assert_eq!(reserved[48], 4 + 3, "the 49th's lane is a chunk more");
    assert_eq!(reserved[63], 4 + 3);
    assert_eq!(
        engine.lock().unwrap().calls,
        before,
        "64 binds at a standing rung: nothing reached the engine"
    );
    assert!(notices(&engine.lock().unwrap()).is_empty());
}

/// A growth at a standing rung that finds state in the chunks it takes
/// still sends no notice: the copy of that state goes as its own message,
/// before the bind, fenced as any raise's.
#[tokio::test(flavor = "current_thread")]
async fn a_growth_at_a_standing_rung_over_state_sends_the_copy_alone() {
    let (mut pool, mut engine) = on_demand_at(256);
    let below = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    // The 256 rung lends 48..60; span segment 7 backed, state enters 56..60.
    assert_eq!(
        walk_segments(&mut pool, 7, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(engine.calls.last(), Some(&Call::CommitSpan(1)));
    let lent = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    assert!(!lent.is_empty(), "state in the lent span");
    // Room below for what a wider span moves out.
    pool.give_records(&below[..below.len() / 2]);
    pool.pool_mut().retire_through(u64::MAX);
    pool.fire_facts_learned(key(PROGRAM), &fire_facts());
    let sent = engine.calls.len();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));

    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
    {
        let pool = shared.lock().unwrap();
        let planned = pool.plan_raise(256).unwrap();
        assert!(!planned.moves.is_empty(), "57..60 holds state");
        assert!(pool.raise_waits(&planned), "the copy is the engine's first");
    }
    fund_span(&shared, || 256, &Grants, &Lane(engine.clone()))
        .await
        .unwrap();
    {
        let engine = engine.lock().unwrap();
        assert!(
            matches!(&engine.calls[sent..], [Call::Evacuate(moves)] if !moves.is_empty()),
            "one message, the copy: {:?}",
            &engine.calls[sent..]
        );
        assert!(notices(&engine).is_empty());
        let pool = shared.lock().unwrap();
        assert_eq!(pool.pool().act_reserved(), 4 + 3);
        assert_eq!(pool.told(), 256);
        assert!(pool.copying().is_none());
        assert_eq!(engine.table.as_slice(), pool.pool().chunk_table());
        pool.pool().check();
    }
    drop(open);
}

/// A bind the engine cannot cover at the rung it holds (Metal's
/// `Pool::cover` past its span: a `Ceiling`, answered `Error::Impossible`)
/// is that bind's own error, the engine's text in it: the lane answers
/// the binder with it, and the binder's guard drops. The room set aside
/// for the bind returns through the walk's next lower, behind the retire
/// fence, and the engine is sent no notice either way: its rung stood.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_the_engine_cannot_cover_at_a_standing_rung_gives_its_room_back_unnoticed() {
    let (mut pool, engine) = on_demand_at(256);
    pool.fire_facts_learned(key(PROGRAM), &fire_facts());
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    let before = engine.lock().unwrap().calls.clone();

    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
    fund_span(&shared, || 256, &NoPlanner, &Lane(engine.clone()))
        .await
        .unwrap();
    assert_eq!(shared.lock().unwrap().pool().act_reserved(), 4 + 3);
    // The lane's bind: the engine covers the term inside it, and cannot.
    const CEILING: &str = "an activation carve of 9 chunks is past the 7 the span covers";
    let refused: Result<(), ::engine::Error> = Err(::engine::Error::Impossible(CEILING.into()));
    let error = refused.expect_err("the engine refused the bind");
    assert!(error.to_string().contains(CEILING), "named: {error}");
    assert!(!error.is_retryable());
    // No instance was bound: the binder's guard is all that counted it.
    drop(open);
    let mut pool = shared.lock().unwrap();
    assert!(pool.pool().fire().is_zero(), "the term closed");
    assert_eq!(
        pool.pool().act_reserved(),
        4 + 3,
        "set aside until the lower"
    );
    let mut walk = RungWalk::default();
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 3, 2).unwrap(),
        RungStep::Wait,
        "fire 3 may still take the wider span"
    );
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 3, 3).unwrap(),
        RungStep::Lower(256)
    );
    assert_eq!(pool.lower_rung(256).unwrap(), None, "the rung stood");
    assert_eq!(pool.pool().act_reserved(), 4, "the reservation is back");
    assert_eq!(pool.told(), 256);
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 4, 4).unwrap(),
        RungStep::Wait
    );
    assert_eq!(
        engine.lock().unwrap().calls,
        before,
        "nothing reached the engine"
    );
}

/// A raise never brings the engine down: with fires cut to 256 under an
/// engine that still holds 2048 (the walk's lower waits on its fence), a
/// bind's term is set aside at 2048, where the engine covers it, and the
/// engine is sent nothing.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_under_a_wider_rung_the_engine_holds_is_funded_at_that_rung() {
    let (mut pool, engine) = on_demand_at(2048);
    pool.fire_facts_learned(key(PROGRAM), &fire_facts());
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    let before = engine.lock().unwrap().calls.clone();
    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.widest_fit(), Some(2048), "12 + 3 of the span's 16");
        assert!(!pool.told_unfit());
        assert_eq!(pool.raise_at(256), Some(2048), "no narrower than it holds");
        assert_eq!(pool.raise_at(16), Some(2048));
    }
    fund_span(&shared, || 256, &NoPlanner, &Lane(engine.clone()))
        .await
        .unwrap();
    let pool = shared.lock().unwrap();
    assert_eq!(pool.pool().act_reserved(), 12 + 3, "the 2048 rung's");
    assert_eq!(pool.told(), 2048);
    assert_eq!(engine.lock().unwrap().calls, before);
    drop(pool);
    drop(open);
}

/// A term the rung the engine holds no longer fits with: the bind is not
/// sent, and nothing is set aside, until the walk has brought the engine
/// down to the widest rung that fits, as a lower is (fires cut, the grant,
/// the retire fence, then the notice). The term's room at that rung is
/// then a standing rung's: set aside with no notice. The engine is never
/// asked for a rung, nor to cover a bind at one, whose term was not held
/// against its span first.
#[tokio::test(flavor = "current_thread")]
async fn a_term_past_the_rung_the_engine_holds_brings_it_down_before_the_bind() {
    let (mut pool, engine) = on_demand_at(2048);
    assert_eq!(pool.pool().act_reserved(), 12);
    // 10 chunks a fire: 12 + 10 is past the span's 16, 4 + 10 is not.
    pool.fire_facts_learned(
        key(PROGRAM),
        &FireFacts {
            fixed: 10 * CHUNK,
            ..FireFacts::default()
        },
    );
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.widest_fit(), Some(256));
        assert!(pool.told_unfit());
        assert_eq!(pool.raise_at(256), None, "the walk's lower first");
        assert_eq!(pool.raise_at(2048), None);
    }
    let cap = Arc::new(AtomicU32::new(u32::MAX));
    let wait = tokio::spawn({
        let (shared, cap, engine) = (shared.clone(), cap.clone(), engine.clone());
        async move {
            fund_span(
                &shared,
                move || cap.load(Ordering::Acquire),
                &NoPlanner,
                &Lane(engine),
            )
            .await
        }
    });
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert!(!wait.is_finished(), "the bind waits on the engine's rung");
    assert_eq!(shared.lock().unwrap().pool().act_reserved(), 12);
    assert!(notices(&engine.lock().unwrap()).is_empty());

    // The walk: fires cut, the grant, the fence, the lower.
    let mut walk = RungWalk::default();
    let notice = {
        let mut pool = shared.lock().unwrap();
        assert_eq!(
            walk.step(&pool, u32::MAX, u32::MAX, u32::MAX, 5, 4)
                .unwrap(),
            RungStep::Cap(256)
        );
        cap.store(256, Ordering::Release);
        assert_eq!(
            walk.step(&pool, u32::MAX, 256, u32::MAX, 5, 4).unwrap(),
            RungStep::Wait,
            "a wider step is still granted"
        );
        assert_eq!(
            walk.step(&pool, u32::MAX, 256, 256, 5, 4).unwrap(),
            RungStep::Wait,
            "fire 5 may still take the 2048 carve"
        );
        assert_eq!(
            walk.step(&pool, u32::MAX, 256, 256, 5, 5).unwrap(),
            RungStep::Lower(256)
        );
        let notice = pool.lower_rung(256).unwrap();
        assert_eq!(pool.pool().act_reserved(), 12, "what is set aside stays");
        assert!(!pool.told_unfit());
        notice
    };
    assert_eq!(notice, Some((2048, 256)), "the engine is told, fenced");
    engine.lock().unwrap().set_rung(2048, 256).unwrap();
    crate::store::shared::rung_changed(TEST_ENGINE);
    wait.await.unwrap().unwrap();
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.pool().act_reserved(), 4 + 10, "the 256 rung's term");
        assert_eq!(pool.told(), 256);
        assert!(pool.copying().is_none());
        assert_eq!(
            walk.step(&pool, u32::MAX, 256, 256, 6, 6).unwrap(),
            RungStep::Wait
        );
    }
    assert_eq!(
        notices(&engine.lock().unwrap()),
        [(2048, 256)],
        "the lower's notice and no other"
    );
    drop(open);
}

/// A steady one-row decode moves nothing after its first step: the
/// instance's term is set aside once, at the rung the engine holds, and
/// every later step of the walk waits. No rung, no cap, no notice.
#[test]
fn a_steady_one_row_decode_moves_no_rung_and_no_cap() {
    let (mut pool, engine) = on_demand_loaded();
    let cap = pool.start_cap();
    assert_eq!(cap, 16, "fires start cut to the load's rung");
    pool.fire_facts_learned(key(PROGRAM), &lane_facts());
    pool.instance_bound(InstanceId(1), key(PROGRAM));
    let goal = pool.rung_for(1);
    assert_eq!(goal, 16);
    let mut walk = RungWalk::default();
    let RungStep::Raise { rung, evacuation } = walk.step(&pool, goal, cap, cap, 1, 1).unwrap()
    else {
        panic!("the first step sets the term's room aside");
    };
    assert_eq!(rung, 16);
    assert!(pool.stands_at(rung));
    assert_eq!(pool.raise_applied(&evacuation).unwrap(), None, "no notice");
    assert!(pool.copying().is_none());
    assert_eq!(pool.pool().act_reserved(), 1 + 2);
    for epoch in 2..200 {
        assert_eq!(
            walk.step(&pool, pool.rung_for(1), cap, cap, epoch, epoch)
                .unwrap(),
            RungStep::Wait,
            "step {epoch}"
        );
        assert_eq!(pool.fit_cap(), u32::MAX);
        assert_eq!(pool.told(), 16);
        assert_eq!(pool.pool().act_reserved(), 1 + 2);
    }
    assert!(notices(&engine).is_empty());
}

/// Mac Pro's serving case (M3): a wave of 256 lanes over 14 pairs of
/// 9 312 512 bytes a lane and 4 718 592 a fire wanted 1911 chunks at the
/// 256 rung, of a span of 1758 (3.3 GB): Metal's `Pool::cover` refused it
/// inside the bind. Held against the span the engine states, the 256 rung
/// does not fit and the 128 rung does: fires are cut to 128 (the wave
/// serves in more fires), every raise asks for 128, and no bind is
/// refused.
///
/// From its boot line: chunks of 1 901 568 bytes, 28 603 of them, in
/// segments of 35. The 256 rung's model carve (622) is the 1911 wanted
/// less the term's 1289. The other rungs' carves (60, 320, 1740) are
/// assumed: any largest carve of 1724..=1758 chunks starts in the segment
/// at chunk 26 845, which leaves the span 1758.
#[tokio::test(flavor = "current_thread")]
async fn a_wave_past_the_span_the_engine_states_is_cut_to_the_widest_rung_it_covers() {
    const CHUNK_BYTES: u64 = 1_901_568;
    const CHUNKS: u32 = 28_603;
    const SEGMENT_CHUNKS: u32 = 35;
    const PAIRS: u32 = 14;
    // What Metal states: from the segment the largest carve starts in.
    let top_first = (CHUNKS - 1740) / SEGMENT_CHUNKS * SEGMENT_CHUNKS;
    let span = CHUNKS - top_first;
    assert_eq!(span, 1758);
    assert_eq!(u64::from(span) * CHUNK_BYTES / 100_000_000, 33, "3.3 GB");
    let kind = SlotKind::List {
        tokens: 16,
        rate: 1,
        window: None,
    };
    let facts = ChunkFacts {
        chunk_bytes: CHUNK_BYTES,
        total_chunks: CHUNKS,
        segment_chunks: SEGMENT_CHUNKS,
        segments: SegmentFacts {
            committed: 2,
            max: CHUNKS.div_ceil(SEGMENT_CHUNKS),
            growth: SegmentGrowth::TwoEnded,
        },
        reserved: vec![ReservedRun {
            what: Reserved::Null,
            first: 0,
            chunks: 1,
        }],
        shapes: vec![Facts {
            slots_per_chunk: 8,
            ..shape(LIST, kind, CHUNK_BYTES / 8)
        }],
        act_chunks: vec![(16, 60), (128, 320), (256, 622), (2048, 1740)],
        span_backed_on_demand: true,
        act_span_chunks: Some(span),
        ..ChunkFacts::default()
    };
    facts.validate().unwrap();
    let mut pool = SharedPool::new(&facts, 256).unwrap();
    pool.set_max_lanes(256);
    assert_eq!(pool.told(), 16);
    let pair = FireFacts {
        a_lane: 9_312_512,
        fixed: 4_718_592,
        max_lanes: 0,
        laid: LaidLanes::Exact,
    };
    for at_pair in 0..PAIRS {
        let key = at(PROGRAM, 256 + at_pair);
        pool.fire_facts_learned(key, &pair);
        bind_live(&mut pool, key, u64::from(at_pair) * 100, 20);
    }
    // The term a 256-lane fire lays: every pair's fixed and 256 lanes.
    let term = pool.pool().fire();
    assert_eq!(term.bytes(256), 2_450_063_360);
    assert_eq!(term.bytes(256).div_ceil(CHUNK_BYTES), 1289);
    assert_eq!(term.bytes(128), 1_258_061_824);
    assert_eq!(term.bytes(128).div_ceil(CHUNK_BYTES), 662);
    assert_eq!(pool.pool().act_chunks(256), 1911, "what the wave wanted");
    assert_eq!(pool.pool().act_chunks(128), 320 + 662);
    assert_eq!(pool.pool().act_span(), Some(1758));
    assert_eq!(pool.act_most(), 1758, "the span, not the device's room");

    // The fit: 128, by construction the rung of every goal.
    assert_eq!(pool.widest_fit(), Some(128));
    assert_eq!(pool.fit_cap(), 128);
    assert_eq!(pool.rung_for(256), 128, "a wave of 256 rows asks 128");
    assert_eq!(pool.rung_for(u32::MAX), 128);
    assert!(!pool.told_unfit());
    let mut walk = RungWalk::default();
    assert_eq!(
        walk.step(&pool, u32::MAX, 256, 256, 1, 1).unwrap(),
        RungStep::Cap(128),
        "the wave is cut to 128"
    );
    let RungStep::Raise { rung, evacuation } = walk.step(&pool, u32::MAX, 128, 128, 1, 1).unwrap()
    else {
        panic!("the walk raises to the widest rung that fits");
    };
    assert_eq!(rung, 128);
    let notice = pool.raise_applied(&evacuation).unwrap();
    assert_eq!(notice, Some((16, 128)));
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(FakeEngine::on_demand()));
    crate::store::shared::programs::carry_raise(
        &shared,
        &evacuation,
        notice,
        &Lane(engine.clone()),
    )
    .await
    .unwrap();
    assert_eq!(
        engine.lock().unwrap().calls,
        [Call::Rung(16, 128)],
        "the engine is asked for 128 and nothing wider"
    );
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.pool().act_reserved(), 982);
        assert!(pool.pool().act_reserved() <= span);
        assert_eq!(pool.told(), 128);
        assert_eq!(
            walk.step(&pool, u32::MAX, 128, 128, 2, 2).unwrap(),
            RungStep::Wait
        );
    }
    // One more bind of a pair open is funded at 128 and waits on nothing.
    let open = ProgramOpen::new(&shared, at(PROGRAM, 256), 0).unwrap();
    fund_span(&shared, || 128, &NoPlanner, &Lane(engine.clone()))
        .await
        .unwrap();
    drop(open);
    assert_eq!(engine.lock().unwrap().calls, [Call::Rung(16, 128)]);

    // Only a term the smallest rung cannot hold is refused, the span named.
    const HUGE: ProgramId = ProgramId(9);
    shared.lock().unwrap().fire_facts_learned(
        key(HUGE),
        &FireFacts {
            fixed: u64::from(span) * CHUNK_BYTES,
            ..FireFacts::default()
        },
    );
    let why = ProgramOpen::new(&shared, key(HUGE), 0)
        .err()
        .expect("past the span at the smallest rung");
    assert!(
        why.contains("1758 chunks the engine's activation span covers"),
        "{why}"
    );
    assert_eq!(shared.lock().unwrap().widest_fit(), Some(128));

    // The span is the engine's statement in force, read at each fit: one
    // stated anew counts from the next, and none stated bounds nothing.
    let mut pool = shared.lock().unwrap();
    pool.pool_mut().set_act_span(Some(1911));
    assert_eq!(pool.widest_fit(), Some(256));
    assert_eq!(pool.rung_for(256), 256);
    pool.pool_mut().set_act_span(None);
    assert_eq!(pool.widest_fit(), Some(2048), "every lendable chunk");
    assert_eq!(pool.fit_cap(), u32::MAX);
}

/// Two chunks of batch tables a set: what [`SetEngine`] keeps standing for
/// every program with a live instance.
const SET: u64 = 2 * CHUNK;

/// An engine that keeps one set of tables standing a program, shared by
/// every live instance of it (Metal's batch tables), and states the total
/// as `Engine::standing_after`, `BoundInstance::standing_guest_bytes` and
/// `Engine::close_instance` do.
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

    fn skips_measuring(&self) -> bool {
        false
    }

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
        "fake-sets"
    }

    fn chunk_table(&self) -> ::engine::Result<::engine::chunk::ChunkTable> {
        Err(self.unsupported("chunk_table"))
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

    fn load(&mut self, _request: ::engine::LoadRequest) -> ::engine::Result<::engine::Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(
        &mut self,
        _frame: &::engine::FrameSubmission,
    ) -> ::engine::Result<::engine::FrameTicket> {
        unreachable!("nothing fires here")
    }

    fn standing_after(&self, binding: &::engine::InstanceBinding) -> ::engine::Result<u64> {
        Ok(self.total(
            self.live
                .values()
                .copied()
                .chain(std::iter::once(binding.program)),
        ))
    }

    fn bind_instance(
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

    fn close_instance(&mut self, id: InstanceId) -> ::engine::Result<u64> {
        self.live.remove(&id);
        Ok(self.total(self.live.values().copied()))
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
    use ::engine::Engine as _;
    let total = engine
        .standing_after(&binding(program))
        .map_err(|error| error.to_string())?;
    let added = shared.lock().unwrap().standing_added(total);
    let open = ProgramOpen::new(shared, key(program), added)?;
    let bound = engine
        .bind_instance(&binding(program))
        .map_err(|error| error.to_string())?;
    {
        let mut pool = shared.lock().unwrap();
        pool.instance_bound(bound.id, key(program));
        pool.standing_stated(bound.standing_guest_bytes);
    }
    drop(open);
    Ok(bound.id)
}

/// A close as the lane books it: the engine's total after it, then the
/// term built at the close's commit.
fn close_through(shared: &Shared, engine: &mut SetEngine, id: InstanceId) {
    use ::engine::Engine as _;
    let total = engine.close_instance(id).unwrap();
    {
        let mut pool = shared.lock().unwrap();
        pool.instance_closed(id);
        pool.standing_stated(total);
    }
    fit_fire(shared);
}

/// Instances of one program share its set: a second bind adds nothing. A
/// distinct program adds its own set, and the close that leaves no instance
/// of a set gives it back; one that leaves an instance gives nothing.
#[test]
fn a_set_shared_by_instances_stands_once_and_a_close_that_empties_it_gives_it_back() {
    const OTHER: ProgramId = ProgramId(8);
    let (pool, _) = pool_at_256();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let mut engine = SetEngine::with(&[(PROGRAM, SET), (OTHER, SET)]);
    let at_256 = |shared: &Shared| shared.lock().unwrap().pool().act_chunks(256);
    assert_eq!(at_256(&shared), 4);

    let first = bind_through(&shared, &mut engine, PROGRAM).unwrap();
    assert_eq!(shared.lock().unwrap().standing_booked(), SET);
    assert_eq!(at_256(&shared), 4 + 2);
    let second = bind_through(&shared, &mut engine, PROGRAM).unwrap();
    assert_eq!(shared.lock().unwrap().standing_booked(), SET, "one set");
    assert_eq!(at_256(&shared), 4 + 2);

    let other = bind_through(&shared, &mut engine, OTHER).unwrap();
    assert_eq!(shared.lock().unwrap().standing_booked(), 2 * SET);
    assert_eq!(at_256(&shared), 4 + 4);

    close_through(&shared, &mut engine, first);
    assert_eq!(at_256(&shared), 4 + 4, "an instance still shares the set");
    close_through(&shared, &mut engine, second);
    assert_eq!(shared.lock().unwrap().standing_booked(), SET);
    assert_eq!(at_256(&shared), 4 + 2);
    close_through(&shared, &mut engine, other);
    assert_eq!(shared.lock().unwrap().pool().standing(), 0);
    assert_eq!(at_256(&shared), 4);
}

/// At every rung the span backs the standing total and the fire term at
/// the rung's lanes, one ceil over the two; a total the largest rung cannot
/// hold with its carve lowers the widest fit rather than hand out a span
/// short of the tables.
#[test]
fn the_span_backs_the_standing_total_and_the_fire_term_at_every_rung() {
    let (mut pool, _) = pool_at_256();
    pool.fire_facts_learned(key(PROGRAM), &lane_facts());
    let shared: Shared = Arc::new(Mutex::new(pool));
    let mut engine = SetEngine::with(&[(PROGRAM, SET + MIB)]);
    for _ in 0..3 {
        bind_through(&shared, &mut engine, PROGRAM).unwrap();
    }
    {
        let pool = shared.lock().unwrap();
        // A fire at `rung` takes the rung at or above it: its model chunks
        // and the term at that rung's lanes.
        let model = [
            (0, 16, 1),
            (16, 16, 1),
            (17, 256, 4),
            (256, 256, 4),
            (2048, 2048, 12),
            (u32::MAX, 2048, 12),
        ];
        for (rung, tokens, chunks) in model {
            let fire = pool.pool().fire().bytes(u32::min(tokens, LANES));
            assert_eq!(
                u64::from(pool.pool().act_chunks(rung)),
                chunks + (fire + SET + MIB).div_ceil(CHUNK),
                "rung {rung}"
            );
        }
        assert_eq!(pool.widest_fit(), Some(2048));
    }

    // 52 chunks standing: 2048's 12 and its term no longer fit the 63
    // past the reserved one, 256's do.
    const WIDE: ProgramId = ProgramId(11);
    engine.sets.insert(WIDE, 52 * CHUNK - SET - MIB);
    shared
        .lock()
        .unwrap()
        .fire_facts_learned(key(WIDE), &FireFacts::default());
    bind_through(&shared, &mut engine, WIDE).unwrap();
    let pool = shared.lock().unwrap();
    assert_eq!(pool.widest_fit(), Some(256));
    assert_eq!(pool.rung_for(u32::MAX), 256, "fires are asked no wider");
    assert!(pool.pool().act_chunks(256) <= pool.act_most());
}

/// A bind's standing bytes are backed below the carve before the engine is
/// asked to bind, and a total no rung can hold refuses the bind by name
/// before the engine's bind is entered, the pool as it was.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_s_standing_bytes_are_backed_before_its_bind_and_a_total_past_the_span_is_refused() {
    use ::engine::Engine as _;
    let (pool, fake) = pool_at_256();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let mut engine = SetEngine::with(&[(PROGRAM, SET), (ProgramId(9), 63 * CHUNK)]);

    // Backed first: the span widens by the set while the bind is held.
    let total = engine.standing_after(&binding(PROGRAM)).unwrap();
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
    let bound = engine.bind_instance(&binding(PROGRAM)).unwrap();
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
        .err()
        .expect("a total past the span is refused");
    assert!(why.contains("standing guest bytes"), "{why}");
    assert_eq!(engine.binds, 1, "the engine's bind was not entered");
    let pool = shared.lock().unwrap();
    assert_eq!(pool.pool().standing(), SET, "a refused bind holds nothing");
    assert_eq!(pool.pool().act_chunks(256), 4 + 2);
}

/// Negative control: the same two sets stated as each pair's `fixed` (the
/// form before the standing total) are counted for the pair a fire carries
/// alone. With one lane a fire, the floor rung holds one set where the
/// engine lays both; stated as the standing total, it holds both.
#[test]
fn tables_stated_as_a_fires_fixed_back_one_set_where_two_lie() {
    const OTHER: ProgramId = ProgramId(8);
    let (mut pool, _) = pool_at_256();
    pool.set_max_lanes(1);
    for program in [PROGRAM, OTHER] {
        pool.fire_facts_learned(
            key(program),
            &FireFacts {
                fixed: SET,
                ..FireFacts::default()
            },
        );
        pool.instance_bound(InstanceId(program.0), key(program));
    }
    assert_eq!(pool.pool().act_chunks(16), 1 + 2, "one set of the two");

    let (mut pool, _) = pool_at_256();
    pool.set_max_lanes(1);
    for program in [PROGRAM, OTHER] {
        pool.fire_facts_learned(key(program), &FireFacts::default());
        pool.instance_bound(InstanceId(program.0), key(program));
    }
    pool.standing_stated(2 * SET);
    assert_eq!(pool.pool().act_chunks(16), 1 + 4, "both sets");
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

/// Readout rows that land beside the activation are counted at every rung
/// beside the guest term, as many as a fire of the rung reads out (its
/// tokens, under the store's rows); rows inside the span, or stated by no
/// one, leave every rung as it was.
#[test]
fn readout_rows_beside_the_activation_are_counted_at_every_rung() {
    let rungs = [0, 16, 17, 256, 2048, u32::MAX];
    let chunks = |pool: &SharedPool| -> Vec<u32> {
        rungs
            .iter()
            .map(|&rung| pool.pool().act_chunks(rung))
            .collect()
    };
    let (mut pool, _) = pool_at_256();
    let before = chunks(&pool);
    assert_eq!(before, [1, 1, 4, 4, 12, 12]);
    let widest = pool.widest_fit();
    for unchanged in [
        ::engine::ReadoutFacts::default(),
        ::engine::ReadoutFacts {
            landing: ::engine::ReadoutLanding::InSpan,
            store_rows: Some(128),
        },
    ] {
        pool.set_readouts(&unchanged, 2048, false);
        assert_eq!(chunks(&pool), before, "{unchanged:?}");
        assert_eq!(pool.widest_fit(), widest, "{unchanged:?}");
    }
    // A MiB a row, at most 128 a fire: the 16 rung's 16 rows one chunk,
    // the wider rungs' 128 rows eight.
    pool.set_readouts(
        &::engine::ReadoutFacts {
            landing: ::engine::ReadoutLanding::Beside { row_bytes: MIB },
            store_rows: Some(128),
        },
        2048,
        false,
    );
    assert_eq!(chunks(&pool), [2, 2, 12, 12, 20, 20]);
    // With float lanes a fire may read out every token through the
    // landing, past the store: the term counts the rung's tokens.
    pool.set_readouts(
        &::engine::ReadoutFacts {
            landing: ::engine::ReadoutLanding::Beside { row_bytes: MIB },
            store_rows: Some(128),
        },
        2048,
        true,
    );
    assert_eq!(chunks(&pool), [2, 2, 20, 20, 140, 140]);
}

/// A lower the walk takes tells the engine first and gives back what was
/// set aside only once the engine's narrow lands: until then the engine
/// holds the wider rung, laying its guests' tables and covering a bind's
/// term there. A bind funded in between is funded at the rung the engine
/// holds, so what it covers stays out of state's reach. (Given back at
/// once, a bind there was funded at the narrower rung while the engine
/// covered the wider one, over chunks state was lent: Metal's c64
/// epilogue faults.)
#[tokio::test(flavor = "current_thread")]
async fn a_bind_before_the_engine_narrows_is_funded_at_the_rung_it_holds() {
    let (mut pool, engine) = on_demand_at(2048);
    pool.fire_facts_learned(key(PROGRAM), &lane_facts());
    let held = pool.pool().act_reserved();
    let limit = pool.pool().state_limit();
    assert_eq!(
        pool.lower_notice(256).unwrap(),
        Some((2048, 256)),
        "the engine is told first"
    );
    assert_eq!(pool.told(), 2048, "it holds the wider rung until it lands");
    assert_eq!(pool.pool().act_reserved(), held, "nothing given back yet");
    assert_eq!(pool.pool().state_limit(), limit, "no state lent above");
    assert_eq!(
        pool.raise_at(256),
        Some(2048),
        "a term is set aside at the rung the engine holds"
    );

    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
    fund_span(&shared, || 256, &Grants, &Lane(engine.clone()))
        .await
        .unwrap();
    {
        let pool = shared.lock().unwrap();
        // The engine covers the bind's term at 2048, wider than 256's.
        assert!(pool.pool().act_chunks(2048) > pool.pool().act_chunks(256));
        assert!(
            pool.pool().act_reserved() >= pool.pool().act_chunks(2048),
            "the funded room covers the held rung's term: {} of {}",
            pool.pool().act_reserved(),
            pool.pool().act_chunks(2048)
        );
        assert!(
            pool.pool().state_limit() <= TOTAL - pool.pool().act_chunks(2048),
            "no state where the engine covers the term"
        );
    }
    let mut pool = shared.lock().unwrap();
    pool.instance_bound(InstanceId(1), key(PROGRAM));
    pool.lower_landed(256).unwrap();
    assert_eq!(pool.told(), 256, "the narrow landed");
    assert_eq!(
        pool.pool().act_reserved(),
        pool.pool().act_chunks(256),
        "what the wider rung held is given back once it landed"
    );
    drop(pool);
    drop(open);
}
