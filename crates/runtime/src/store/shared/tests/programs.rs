//! REDESIGN §6: a registered program's fire term in the rung arithmetic.
//! A rung of `r` tokens carries `min(r, max_lanes)` lanes; its activation
//! is the model's chunks plus the chunks `fixed + lanes × a_lane` bytes of
//! every program that can be in a fire take. A bind of a program whose term
//! raises the span waits until the raise is funded and taken; the last
//! instance's close drops the term, and the next lower gives the room back.

use std::sync::atomic::{AtomicU32, Ordering};

use ::engine::{InstanceId, ProgramFacts, ProgramId};
use tokio::sync::Notify;

use super::*;
use crate::store::shared::programs::{Funder, Mover, ProgramOpen, fund_span};

const PROGRAM: ProgramId = ProgramId(7);
const LANES: u32 = 64;

/// 20 MiB a fire and 256 KiB a lane: a rung of 16 tokens 24 MiB (2
/// chunks), one of 256 or more 64 lanes, 36 MiB (3 chunks).
fn fire_facts() -> ProgramFacts {
    ProgramFacts {
        fire_bytes_fixed: 20 * MIB,
        fire_bytes_a_lane: 256 << 10,
        ..ProgramFacts::default()
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
    pool.program_registered(PROGRAM, &ProgramFacts::default());
    pool.instance_bound(InstanceId(1), PROGRAM);
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
    let open = ProgramOpen::new(&shared, PROGRAM).unwrap();
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

/// A rung's activation is the model's chunks plus the term's at its lanes,
/// the lanes capped by the engine's `max_lanes`; only while an instance of
/// the program can be in a fire.
#[test]
fn the_fire_term_takes_its_chunks_at_every_rung_s_lanes() {
    let (mut pool, _) = pool_at_256();
    pool.program_registered(PROGRAM, &fire_facts());
    assert_eq!(pool.pool().act_chunks(256), 4, "registered, no instance");
    pool.instance_bound(InstanceId(1), PROGRAM);
    // 16 lanes: 20 + 4 MiB, 2 chunks; 64 lanes: 20 + 16 MiB, 3 chunks.
    assert_eq!(pool.pool().act_chunks(16), 1 + 2);
    assert_eq!(pool.pool().act_chunks(256), 4 + 3);
    assert_eq!(pool.pool().act_chunks(2048), 12 + 3, "lanes capped at 64");
    // Two instances count the program once.
    pool.instance_bound(InstanceId(2), PROGRAM);
    assert_eq!(pool.pool().act_chunks(256), 4 + 3);
    // The span set aside at 256 no longer holds the 256 rung.
    assert_eq!(pool.span_short(256), 3);
    assert_eq!(pool.rung(), 16, "4 chunks hold the 16 rung with its term");
    pool.instance_closed(InstanceId(1));
    assert_eq!(pool.pool().act_chunks(256), 4 + 3);
    pool.instance_closed(InstanceId(2));
    assert_eq!(pool.pool().act_chunks(256), 4, "the last closed: gone");
    assert_eq!(pool.rung(), 256);
}

/// A term that leaves the smallest rung no room on the device refuses the
/// bind at once rather than waiting on room that never comes.
#[test]
fn a_fire_term_past_the_device_is_refused() {
    let (mut pool, _) = pool_at_256();
    pool.program_registered(
        PROGRAM,
        &ProgramFacts {
            fire_bytes_fixed: u64::from(TOTAL) * CHUNK,
            ..ProgramFacts::default()
        },
    );
    let shared: Shared = Arc::new(Mutex::new(pool));
    assert!(ProgramOpen::new(&shared, PROGRAM).is_err());
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
    pool.program_registered(PROGRAM, &fire_facts());
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    let planner = Arc::new(Planner::default());

    let open = ProgramOpen::new(&shared, PROGRAM).unwrap();
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
        .instance_bound(InstanceId(1), PROGRAM);
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
/// on every rung past 16 tokens: `fixed + min(r, max_lanes, program
/// max_lanes) × a_lane`.
#[test]
fn a_program_s_own_lane_cap_bounds_its_term() {
    let (mut pool, _) = pool_at_256();
    pool.program_registered(
        PROGRAM,
        &ProgramFacts {
            max_lanes: 16,
            ..fire_facts()
        },
    );
    pool.instance_bound(InstanceId(1), PROGRAM);
    // 20 + 16 x 0.25 MiB = 24 MiB, 2 chunks, at 16, 256 and 2048 alike.
    assert_eq!(pool.pool().act_chunks(16), 1 + 2);
    assert_eq!(pool.pool().act_chunks(256), 4 + 2);
    assert_eq!(pool.pool().act_chunks(2048), 12 + 2);
    // Beside an uncapped program: each takes its own lanes.
    const OTHER: ProgramId = ProgramId(8);
    pool.program_registered(OTHER, &fire_facts());
    pool.instance_bound(InstanceId(2), OTHER);
    // 24 + 36 MiB = 60 MiB, 4 chunks.
    assert_eq!(pool.pool().act_chunks(256), 4 + 4);
}

/// 55 chunks a fire: the 2048 rung (12 + 55) no longer fits the 63 chunks
/// past the reserved one, the 256 rung (4 + 55) does.
fn wide_facts() -> ProgramFacts {
    ProgramFacts {
        fire_bytes_fixed: 55 * CHUNK,
        ..ProgramFacts::default()
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
    pool.program_registered(PROGRAM, &wide_facts());
    let shared: Shared = Arc::new(Mutex::new(pool));
    let open = ProgramOpen::new(&shared, PROGRAM).unwrap();
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
    wait.await.unwrap().unwrap();
    drop(open);

    // A term even the smallest rung cannot hold is refused.
    const HUGE: ProgramId = ProgramId(9);
    shared.lock().unwrap().program_registered(
        HUGE,
        &ProgramFacts {
            fire_bytes_fixed: 63 * CHUNK,
            ..ProgramFacts::default()
        },
    );
    assert!(ProgramOpen::new(&shared, HUGE).is_err());
    assert_eq!(shared.lock().unwrap().fit_cap(), u32::MAX, "no term counts");
}

/// A bind that cannot be funded is refused, never sent: the planner's
/// refusal and a refused copy both end the wait with an error, the span as
/// it was.
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
    pool.program_registered(PROGRAM, &fire_facts());
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let _open = ProgramOpen::new(&shared, PROGRAM).unwrap();
    let error = fund_span(&shared, || 256, &Refusing, &NoCopy)
        .await
        .unwrap_err();
    assert!(error.contains("the planner refuses"));
    assert_eq!(shared.lock().unwrap().pool().act_reserved(), 4);

    // Room there, the copy refused: the wait fails rather than let the
    // bind past state still in the span.
    {
        let mut pool = shared.lock().unwrap();
        pool.give_records(&all[..all.len() / 2]);
        pool.pool_mut().retire_through(u64::MAX);
    }
    let error = fund_span(&shared, || 256, &Granting, &NoCopy)
        .await
        .unwrap_err();
    assert!(error.contains("the engine refuses"), "{error}");
}
