//! REDESIGN §6: a registered program's fire term in the rung arithmetic.
//! A rung of `r` tokens carries `min(r, max_lanes)` lanes; its activation
//! is the model's chunks plus the chunks `fixed + lanes × a_lane` bytes of
//! every program that can be in a fire take. A bind of a program whose term
//! raises the span waits until the raise is funded and taken; the last
//! instance's close drops the term, and the next lower gives the room back.

use std::sync::atomic::{AtomicU32, Ordering};

use ::engine::{FireFacts, InstanceId, ProgramId};
use tokio::sync::Notify;

use super::*;
use crate::store::shared::programs::{FireKey, Funder, Mover, ProgramOpen, fund_span};

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
    assert_eq!(pool.room().act_chunks, 1);
    let mut walk = RungWalk::default();
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 1, 1).unwrap(),
        RungStep::Wait,
        "at its rung: nothing moves"
    );
    // A bind waits on nothing, and asks no planner.
    let shared: Shared = Arc::new(Mutex::new(pool));
    let open = ProgramOpen::new(&shared, key(PROGRAM)).unwrap();
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
        pool.instance_closed(InstanceId(id));
    }
    assert_eq!(pool.pool().act_chunks(256), 4, "the last closed: gone");
    assert_eq!(pool.rung(), 256);
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
    assert!(ProgramOpen::new(&shared, key(PROGRAM)).is_err());
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

    let open = ProgramOpen::new(&shared, key(PROGRAM)).unwrap();
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
    // Beside an uncapped program with 64 live: each takes its own lanes.
    const OTHER: ProgramId = ProgramId(8);
    pool.fire_facts_learned(key(OTHER), &lane_facts());
    bind_live(&mut pool, key(OTHER), 100, 64);
    // 24 + 36 MiB = 60 MiB, 4 chunks.
    assert_eq!(pool.pool().act_chunks(256), 4 + 4);
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
    let open = ProgramOpen::new(&shared, key(PROGRAM)).unwrap();
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
    assert!(ProgramOpen::new(&shared, key(HUGE)).is_err());
    assert_eq!(shared.lock().unwrap().fit_cap(), u32::MAX, "no term counts");
}

/// A bind is never sent unfunded: the planner's refusal ends the wait with
/// an error; a raise the engine refuses is taken back and the wait holds,
/// the span and the state as they were.
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
    let _open = ProgramOpen::new(&shared, key(PROGRAM)).unwrap();
    let error = fund_span(&shared, || 256, &Refusing, &NoCopy)
        .await
        .unwrap_err();
    assert!(error.contains("the planner refuses"));
    assert_eq!(shared.lock().unwrap().pool().act_reserved(), 4);

    // Room there, the copy refused: a refused raise, taken back. The wait
    // holds the bind (never sent) as any waiter, the span and the state as
    // they were, and nothing faults.
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
    let waited = tokio::time::timeout(
        std::time::Duration::from_millis(50),
        fund_span(&shared, || 256, &Granting, &NoCopy),
    )
    .await;
    assert!(waited.is_err(), "the bind still waits");
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
    let open = ProgramOpen::new(&shared, key(PROGRAM)).expect("a zero term never refuses");
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
    let open = ProgramOpen::new(&shared, key(PROGRAM)).expect("the term is ignored");
    let huge =
        ProgramOpen::new(&shared, key(ProgramId(9))).expect("past the device, still ignored");
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
    assert!(ProgramOpen::new(&shared, key(PROGRAM)).is_ok());
}

/// One program at two extents is two terms, summed: 256 rows take 8 MiB
/// fixed, 4096 rows 16x that.
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
    pool.instance_closed(InstanceId(2));
    assert_eq!(
        pool.pool().act_chunks(256),
        4 + 1,
        "4096 rows: its last bind closed"
    );
    pool.instance_closed(InstanceId(1));
    assert_eq!(
        pool.pool().act_chunks(256),
        4 + 1,
        "256 rows: one bind left"
    );
    pool.instance_closed(InstanceId(3));
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

    let open = ProgramOpen::new(&shared, at(PROGRAM, 4096)).unwrap();
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
            },
        ),
        (
            at(PROGRAM, 4096),
            FireFacts {
                fixed: 20 * MIB,
                a_lane: 1 << 20,
                max_lanes: 8,
            },
        ),
        (
            key(ProgramId(8)),
            FireFacts {
                fixed: 0,
                a_lane: 100 << 10,
                max_lanes: 32,
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
    pool.instance_closed(InstanceId(1));
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
    let open = ProgramOpen::new(&shared, at(PROGRAM, 4096)).unwrap();
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
        assert!(pool.pool().fire().is_empty(), "no term left counted");
        assert!(pool.knows_fire(&at(PROGRAM, 4096)), "its facts are kept");
        pool.pool().check();
    }
    // A later bind of the same extents opens the pair anew, and its close
    // leaves nothing behind either.
    let open = ProgramOpen::new(&shared, at(PROGRAM, 4096)).unwrap();
    shared
        .lock()
        .unwrap()
        .instance_bound(InstanceId(7), at(PROGRAM, 4096));
    drop(open);
    assert_eq!(shared.lock().unwrap().pool().act_chunks(256), 4 + 8);
    shared.lock().unwrap().instance_closed(InstanceId(7));
    assert!(shared.lock().unwrap().pool().fire().is_empty());
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
    let open = ProgramOpen::new(&shared, key(PROGRAM)).unwrap();
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
