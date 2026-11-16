//! REDESIGN §6: what bound programs ask of the rung arithmetic. A guest
//! program's scratch is state of its instance (`::engine::GuestState`,
//! `store::roles`): no fire takes anything of the span for it. A rung's
//! activation is the model's chunks, the rows a fire reads out where they
//! land beside it, and the engine's standing guest bytes (Metal's batch
//! tables): a bind whose standing bytes raise the span waits until the
//! raise is funded and taken, and one they fit at no rung is refused.

use std::sync::atomic::{AtomicU32, Ordering};

use ::engine::{InstanceId, ProgramId};

use super::*;
use crate::store::shared::programs::{FireKey, Mover, ProgramOpen, fund_span};

const PROGRAM: ProgramId = ProgramId(7);
const LANES: u32 = 64;
/// The engine whose rung events these tests' raises tell.
const TEST_ENGINE: usize = 50;
/// 36 MiB standing: 3 chunks at every rung.
const FIRE: u64 = 36 * MIB;

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

/// An empty pool, every segment committed, at the 256 rung.
fn pool_at_256() -> (SharedPool, FakeEngine) {
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    pool.lower_rung(256).unwrap();
    assert_eq!(pool.pool().act_reserved(), 4);
    (pool, engine)
}

/// The engine's side of a raise, on the fake engine.
struct Lane(Arc<Mutex<FakeEngine>>);

impl Mover for Lane {
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
}

/// A program bound with no standing bytes leaves every rung's activation
/// and the walk as they were: its scratch is its instance's state.
#[tokio::test(flavor = "current_thread")]
async fn a_bound_program_leaves_the_rung_arithmetic_unchanged() {
    let (pool, engine) = pool_at_256();
    let before: Vec<u32> = [0, 16, 17, 256, 2048, u32::MAX]
        .iter()
        .map(|&rung| pool.pool().act_chunks(rung))
        .collect();
    assert_eq!(before, [1, 1, 4, 4, 12, 12]);
    let after: Vec<u32> = [0, 16, 17, 256, 2048, u32::MAX]
        .iter()
        .map(|&rung| pool.pool().act_chunks(rung))
        .collect();
    assert_eq!(after, before);
    assert_eq!(pool.rung(), 256);
    assert_eq!(pool.span_short(256), 0);
    assert_eq!(
        pool.room().act_chunks,
        12,
        "the top rung's span: live state's room lies below it at every rung"
    );
    let mut walk = RungWalk::default();
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 1, 1).unwrap(),
        RungStep::Wait,
        "at its rung: nothing moves"
    );
    // A bind waits on nothing.
    let shared: Shared = Arc::new(Mutex::new(pool));
    let open = ProgramOpen::new(&shared, key(PROGRAM), 0).unwrap();
    fund_span(&shared, || 256, &Lane(Arc::new(Mutex::new(engine))))
        .await
        .unwrap();
    drop(open);
    assert_eq!(shared.lock().unwrap().pool().act_reserved(), 4);
}

/// Standing bytes that leave the smallest rung no room on the device refuse
/// the bind at once rather than waiting on room that never comes.
#[test]
fn standing_bytes_past_the_device_are_refused() {
    let (pool, _) = pool_at_256();
    let shared: Shared = Arc::new(Mutex::new(pool));
    assert!(ProgramOpen::new(&shared, key(PROGRAM), u64::from(TOTAL) * CHUNK).is_err());
    assert_eq!(
        shared.lock().unwrap().pool().act_chunks(256),
        4,
        "a refused bind counts nothing"
    )
}

/// Standing bytes are cut as the readout term is, against the top rung's
/// span (below which live state lies at every rung, so no raise ever moves
/// it): bytes that fit only under a narrower rung cut fires to the widest
/// rung whose activation fits the span, and the raise there moves nothing;
/// bytes that fit at no rung are refused at the bind, by name, the pool as
/// it was.
#[tokio::test(flavor = "current_thread")]
async fn standing_bytes_are_cut_to_the_top_rung_s_span_and_past_every_rung_refused() {
    const WIDE: ProgramId = ProgramId(71);
    const PAST: ProgramId = ProgramId(72);
    let (mut pool, _) = pool_at_256();
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    // 9 chunks a fire: the 16 rung takes 10 of the span's 12, 256 would
    // take 13.
    // 12 chunks a fire: even the 16 rung would take 13.
    let table = pool.pool().chunk_table().to_vec();
    let shared: Shared = Arc::new(Mutex::new(pool));

    let refused = ProgramOpen::new(&shared, key(PAST), 12 * CHUNK)
        .map(|_| ())
        .unwrap_err();
    assert!(refused.contains(&key(PAST).to_string()), "{refused}");
    assert!(refused.contains("of the top rung's span"), "{refused}");
    {
        let pool = shared.lock().unwrap();
        assert!(
            pool.pool().standing() == 0,
            "the refused bytes are never seen"
        );
        assert_eq!(pool.fit_cap(), u32::MAX);
    }

    let open = ProgramOpen::new(&shared, key(WIDE), 9 * CHUNK).unwrap();
    let mut walk = RungWalk::default();
    {
        let mut pool = shared.lock().unwrap();
        assert_eq!(pool.pool().act_chunks(16), 10);
        assert_eq!(pool.pool().act_chunks(256), 13);
        assert_eq!(pool.widest_fit(), Some(16), "the widest rung the span fits");
        assert_eq!(pool.fit_cap(), 16);
        assert_eq!(pool.rung_for(5000), 16, "fires cut to it");
        assert_eq!(
            pool.plan_raise(256).unwrap_err(),
            RungError::Pool(SlotError::PastSpan {
                rung: 256,
                chunks: 13,
                span: 12
            }),
            "never set aside past the span"
        );
        // The walk cuts fires to it, then takes its activation, the term's
        // included: nothing moves.
        assert_eq!(
            walk.step(&pool, 256, 256, 256, 1, 1).unwrap(),
            RungStep::Cap(16)
        );
        let RungStep::Raise { rung, raise } = walk.step(&pool, 256, 16, 16, 1, 1).unwrap() else {
            panic!("a raise to the cut rung");
        };
        assert_eq!(rung, 16);
        assert_eq!(pool.raise_applied(&raise).unwrap(), None);
        assert_eq!(pool.pool().act_reserved(), 10);
        assert_eq!(pool.pool().chunk_table(), table.as_slice(), "nothing moved");
        assert!(
            all.iter()
                .flat_map(|&record| pool.record_slots(record))
                .all(|slot| pool.pool().physical(slot.chunk).unwrap() < TOTAL - 12)
        );
        pool.pool().check();
    }
    drop(open);
}

/// A pool that states no rungs (a load whose activation lies outside the
/// pool): one rung of fires of any tokens whose carve takes no chunk, its
/// span the room the load backs, `room` chunks (`act_span_chunks`).
fn rungless_pool(room: Option<u32>) -> (SharedPool, FakeEngine) {
    let mut facts = facts();
    facts.act_chunks = Vec::new();
    facts.act_span_chunks = room;
    let pool = SharedPool::new(&facts, 256).unwrap();
    assert_eq!(pool.rungs(), [u32::MAX], "one rung");
    assert_eq!(pool.pool().act_max(), room.unwrap_or(0));
    (pool, FakeEngine::new())
}

/// The regression: on a pool without rungs every bind was refused, a zero
/// term included ("fires need 0 bytes ... 0 chunks of the 15173"). A bind
/// with nothing standing never refuses, and waits on nothing, with or
/// without room.
#[tokio::test(flavor = "current_thread")]
async fn a_zero_term_never_refuses_a_bind_on_a_pool_without_rungs() {
    for room in [None, Some(4)] {
        let (pool, engine) = rungless_pool(room);
        let shared: Shared = Arc::new(Mutex::new(pool));
        let open = ProgramOpen::new(&shared, key(PROGRAM), 0).expect("a zero term never refuses");
        fund_span(&shared, || u32::MAX, &Lane(Arc::new(Mutex::new(engine))))
            .await
            .unwrap();
        drop(open);
    }
}

/// A pool without rungs is the one-rung case of the same cut: standing
/// bytes (here 36 MiB, 3 chunks) are set aside in the room its load backs
/// (4 chunks at the pool's end, below which live state lies), the bind
/// funded there and its fires uncut; bytes past that room are refused at
/// the bind, by name, the pool as it was. A load that backs no room
/// refuses any.
#[tokio::test(flavor = "current_thread")]
async fn a_pool_without_rungs_cuts_and_refuses_standing_bytes_against_the_room_its_load_backs() {
    const PAST: ProgramId = ProgramId(9);
    let (mut pool, engine) = rungless_pool(Some(4));
    let records = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    let shared: Shared = Arc::new(Mutex::new(pool));

    let refused = ProgramOpen::new(&shared, key(PAST), 5 * CHUNK)
        .map(|_| ())
        .unwrap_err();
    assert!(refused.contains(&key(PAST).to_string()), "{refused}");
    assert!(
        refused.contains("4 chunks the engine's activation span covers"),
        "{refused}"
    );
    assert!(shared.lock().unwrap().pool().standing() == 0);

    let open = ProgramOpen::new(&shared, key(PROGRAM), FIRE).expect("3 chunks fit the room of 4");
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.pool().act_chunks(16), 3, "the term, no carve");
        assert_eq!(pool.widest_fit(), Some(u32::MAX));
        assert_eq!(pool.fit_cap(), u32::MAX, "fires uncut");
    }
    fund_span(&shared, || u32::MAX, &Lane(Arc::new(Mutex::new(engine))))
        .await
        .unwrap();
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.pool().act_reserved(), 3, "set aside for its fires");
        assert_eq!(pool.pool().live_limit(), TOTAL - 4);
        assert!(
            records
                .iter()
                .flat_map(|&record| pool.record_slots(record))
                .all(|slot| pool.pool().physical(slot.chunk).unwrap() < TOTAL - 4),
            "live state below the room"
        );
        pool.pool().check();
    }
    drop(open);

    let (bare, _) = rungless_pool(None);
    let bare: Shared = Arc::new(Mutex::new(bare));
    let refused = ProgramOpen::new(&bare, key(PROGRAM), FIRE)
        .map(|_| ())
        .unwrap_err();
    assert!(refused.contains(&key(PROGRAM).to_string()), "{refused}");
    assert!(refused.contains("of the top rung's span"), "{refused}");
}

/// On a pool with rungs a zero term never refuses either.
#[test]
fn a_zero_term_never_refuses_a_bind_on_a_pool_with_rungs() {
    let (pool, _) = pool_at_256();
    let shared: Shared = Arc::new(Mutex::new(pool));
    assert!(ProgramOpen::new(&shared, key(PROGRAM), 0).is_ok());
}

// ---- ruling 2: the rung the engine holds, and the span it states ----

/// The fixture's pool with its engine told rungs, stating as Metal does
/// that its activation span covers the segments from the one the largest
/// rung's carve starts in (6 and 7: 16 chunks); every segment committed.
/// The engine holds the smallest rung.
fn told_loaded() -> (SharedPool, FakeEngine) {
    let mut facts = facts();
    facts.tells_rungs = true;
    facts.act_span_chunks = Some(2 * SEGMENT);
    let mut pool = SharedPool::new(&facts, 256).unwrap();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(pool.told(), 16, "the load's rung");
    // The top rung's span bounds it, under the 16 the engine states.
    assert_eq!(pool.act_most(), 12);
    (pool, engine)
}

/// [`told_loaded`], the engine told `rung` by a raise from the load's.
fn told_at(rung: u32) -> (SharedPool, FakeEngine) {
    let (mut pool, engine) = told_loaded();
    let raise = pool.plan_raise(rung).unwrap();
    assert_eq!(pool.raise_applied(&raise).unwrap(), Some((16, rung)));
    pool.rung_told();
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

/// `Engine::set_rung(from, to)` is sent only where the rung the engine
/// holds differs. Each of 64 binds at a standing rung grows the standing
/// bytes (twice by a chunk); the room is set aside before the bind is
/// sent, where the engine covers it inside the bind, with no notice, no
/// trip through the engine's lane, no fence and no `copying`: the binds do
/// not wait on the engine, nor on one another.
#[tokio::test(flavor = "current_thread")]
async fn binds_at_a_standing_rung_send_the_engine_no_notice() {
    let (pool, engine) = told_at(256);
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    let before = engine.lock().unwrap().calls.clone();
    let mut reserved = Vec::new();
    for id in 1..=u64::from(LANES) {
        // 20 MiB, and 256 KiB more each bind.
        let total = 20 * MIB + id * (256 << 10);
        let added = shared.lock().unwrap().standing_added(total);
        let open = ProgramOpen::new(&shared, key(PROGRAM), added).unwrap();
        {
            // A growth that moves nothing waits on nothing.
            let pool = shared.lock().unwrap();
            let planned = pool.plan_raise(256).unwrap();
            assert!(!pool.raise_waits(&planned), "bind {id}");
        }
        fund_span(&shared, || 256, &Lane(engine.clone()))
            .await
            .unwrap();
        let mut pool = shared.lock().unwrap();
        assert!(pool.awaiting_notice().is_none(), "bind {id}");
        assert_eq!(pool.told(), 256, "bind {id}");
        assert_eq!(pool.span_short(256), 0, "funded before bind {id} is sent");
        pool.standing_stated(total);
        reserved.push(pool.pool().act_reserved());
        drop(pool);
        drop(open);
    }
    // 2 chunks through 48 binds, 3 past them.
    assert_eq!(reserved[0], 4 + 2, "the first bind's");
    assert_eq!(reserved[47], 4 + 2);
    assert_eq!(reserved[48], 4 + 3, "the 49th's a chunk more");
    assert_eq!(reserved[63], 4 + 3);
    assert_eq!(
        engine.lock().unwrap().calls,
        before,
        "64 binds at a standing rung: nothing reached the engine"
    );
    assert!(notices(&engine.lock().unwrap()).is_empty());
}

/// A growth at a standing rung that finds idle cache in the chunks it
/// takes (live state never lies in the top rung's span) sends no notice and
/// no copy: the cache is let go first, and the room set aside moves
/// nothing, before the bind.
#[tokio::test(flavor = "current_thread")]
async fn a_growth_at_a_standing_rung_over_the_span_s_cache_sends_nothing_once_it_is_cleared() {
    let (mut pool, engine) = told_at(256);
    let below = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    // The 256 rung leaves 52..60 to cache, backed from the load; the
    // weight cache takes it.
    assert_eq!(pool.records_available(), 0, "live state takes none of it");
    let cached = span_cache(&mut pool, 16);
    let sent = engine.calls.len();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));

    let open = ProgramOpen::new(&shared, key(PROGRAM), FIRE).unwrap();
    {
        let mut pool = shared.lock().unwrap();
        assert!(
            pool.plan_raise(256).is_err_and(|error| error.span_cached()),
            "57..60 holds cache"
        );

        clear_span_cache(&mut pool, &cached, 1);
        let planned = pool.plan_raise(256).unwrap();
        assert!(!pool.raise_waits(&planned), "no trip through the lane");
    }
    fund_span(&shared, || 256, &Lane(engine.clone()))
        .await
        .unwrap();
    {
        let engine = engine.lock().unwrap();
        assert!(
            engine.calls[sent..].is_empty(),
            "nothing sent: {:?}",
            &engine.calls[sent..]
        );
        let pool = shared.lock().unwrap();
        assert_eq!(pool.pool().act_reserved(), 4 + 3);
        assert_eq!(pool.told(), 256);
        assert!(pool.awaiting_notice().is_none());
        assert_eq!(engine.table.as_slice(), pool.pool().chunk_table());
        assert!(
            below
                .iter()
                .flat_map(|&record| pool.record_slots(record))
                .all(|slot| pool.pool().physical(slot.chunk).unwrap() < 52)
        );
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
    let (pool, engine) = told_at(256);
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    let before = engine.lock().unwrap().calls.clone();

    let open = ProgramOpen::new(&shared, key(PROGRAM), FIRE).unwrap();
    fund_span(&shared, || 256, &Lane(engine.clone()))
        .await
        .unwrap();
    assert_eq!(shared.lock().unwrap().pool().act_reserved(), 4 + 3);
    // The lane's bind: the engine covers the bytes inside it, and cannot.
    const CEILING: &str = "an activation carve of 9 chunks is past the 7 the span covers";
    let refused: Result<(), ::engine::Error> = Err(::engine::Error::Impossible(CEILING.into()));
    let error = refused.expect_err("the engine refused the bind");
    assert!(error.to_string().contains(CEILING), "named: {error}");
    assert!(!error.is_retryable());
    // No instance was bound: the binder's guard is all that counted it.
    drop(open);
    let mut pool = shared.lock().unwrap();
    assert!(pool.pool().standing() == 0, "the standing bytes closed");
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

/// A raise never brings the engine down: with fires cut to 16 under an
/// engine that still holds 256 (the walk's lower waits on its fence), a
/// bind's term is set aside at 256, where the engine covers it, and the
/// engine is sent nothing.
#[tokio::test(flavor = "current_thread")]
async fn a_bind_under_a_wider_rung_the_engine_holds_is_funded_at_that_rung() {
    let (pool, engine) = told_at(256);
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    let before = engine.lock().unwrap().calls.clone();
    let open = ProgramOpen::new(&shared, key(PROGRAM), FIRE).unwrap();
    {
        let pool = shared.lock().unwrap();
        assert_eq!(
            pool.widest_fit(),
            Some(256),
            "4 + 3 of the top rung's 12; 12 + 3 is past it"
        );
        assert!(!pool.told_unfit());
        assert_eq!(pool.raise_at(16), Some(256), "no narrower than it holds");
    }
    fund_span(&shared, || 16, &Lane(engine.clone()))
        .await
        .unwrap();
    let pool = shared.lock().unwrap();
    assert_eq!(pool.pool().act_reserved(), 4 + 3, "the 256 rung's");
    assert_eq!(pool.told(), 256);
    assert_eq!(engine.lock().unwrap().calls, before);
    drop(pool);
    drop(open);
}

/// A term the rung the engine holds no longer fits with: the bind is not
/// sent, and nothing is set aside, until the walk has brought the engine
/// down to the widest rung that fits, as a lower is (fires cut, the grant,
/// the retire fence, then the notice). The term's room at that rung lies
/// within the top rung's span, so the lower sets it aside. The engine is never
/// asked for a rung, nor to cover a bind at one, whose term was not held
/// against its span first.
#[tokio::test(flavor = "current_thread")]
async fn a_term_past_the_rung_the_engine_holds_brings_it_down_before_the_bind() {
    let (pool, engine) = told_at(2048);
    assert_eq!(pool.pool().act_reserved(), 12);
    // 7 chunks a fire: 12 + 7 is past the top rung's 12, 4 + 7 is not.
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    let open = ProgramOpen::new(&shared, key(PROGRAM), 7 * CHUNK).unwrap();
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
        async move { fund_span(&shared, move || cap.load(Ordering::Acquire), &Lane(engine)).await }
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
        assert_eq!(
            pool.pool().act_reserved(),
            4 + 7,
            "the 256 rung's, its term included: within the top rung's span"
        );
        assert!(!pool.told_unfit());
        notice
    };
    assert_eq!(notice, Some((2048, 256)), "the engine is told, fenced");
    engine.lock().unwrap().set_rung(2048, 256).unwrap();
    crate::store::shared::rung_changed(TEST_ENGINE);
    wait.await.unwrap().unwrap();
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.pool().act_reserved(), 4 + 7, "the 256 rung's term");
        assert_eq!(pool.told(), 256);
        assert!(pool.awaiting_notice().is_none());
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
/// standing bytes are set aside once, at the rung the engine holds, and
/// every later step of the walk waits. No rung, no cap, no notice.
#[test]
fn a_steady_one_row_decode_moves_no_rung_and_no_cap() {
    let (mut pool, engine) = told_loaded();
    let cap = pool.start_cap();
    assert_eq!(cap, 16, "fires start cut to the load's rung");
    pool.standing_stated(20 * MIB + (256 << 10));
    let goal = pool.rung_for(1);
    assert_eq!(goal, 16);
    let mut walk = RungWalk::default();
    let RungStep::Raise { rung, raise } = walk.step(&pool, goal, cap, cap, 1, 1).unwrap() else {
        panic!("the first step sets the room aside");
    };
    assert_eq!(rung, 16);
    assert!(pool.stands_at(rung));
    assert_eq!(pool.raise_applied(&raise).unwrap(), None, "no notice");
    assert!(pool.awaiting_notice().is_none());
    assert_eq!(pool.pool().act_reserved(), 1 + 2);
    for epoch in 2..200 {
        assert_eq!(
            walk.step(&pool, pool.rung_for(1), cap, cap, epoch, epoch)
                .unwrap(),
            RungStep::Wait,
            "step {epoch}"
        );
        // The bytes take 12 + 2 at 2048, past the top rung's span: fires
        // are cut to 256 at most, the cap here under it.
        assert_eq!(pool.fit_cap(), 256);
        assert_eq!(pool.told(), 16);
        assert_eq!(pool.pool().act_reserved(), 1 + 2);
    }
    assert!(notices(&engine).is_empty());
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
/// smallest rung cannot hold the total.
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
        pool.standing_stated(bound.standing_guest_bytes);
    }
    drop(open);
    Ok(bound.id)
}

/// A close as the lane books it: the engine's total after it.
fn close_through(shared: &Shared, engine: &mut SetEngine, id: InstanceId) {
    use ::engine::Engine as _;
    let total = engine.close_instance(id).unwrap();
    shared.lock().unwrap().standing_stated(total);
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
    fund_span(&shared, || 256, &Lane(Arc::new(Mutex::new(fake))))
        .await
        .unwrap();
    assert_eq!(shared.lock().unwrap().pool().act_reserved(), 4 + 2);
    assert_eq!(engine.binds, 0, "backed before the engine binds");
    let bound = engine.bind_instance(&binding(PROGRAM)).unwrap();
    {
        let mut pool = shared.lock().unwrap();
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

/// Readout rows that land beside the activation are counted at every rung,
/// as many as a fire of the rung reads out (its
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
