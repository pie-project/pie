//! The top rung's span, backed from the load on every engine; an engine
//! told rungs (`ChunkFacts::tells_rungs`, Metal) hears each one. Yield gives
//! span segments back under pressure, its last shed, below the activation
//! set aside (`release_span`), and the governor's grow backs them again,
//! whole, before a raise takes them (`commit_span`); a raise into them
//! waits on that grow, never on a fire.
//!
//! The fixture's span: segments 6 and 7 (chunks 48..64), the largest rung's
//! activation 52..64.

use super::*;
use crate::governor::{PATIENCE, PoolTarget};
use crate::store::shared::programs::{Mover, take_raise};

/// The engine's lane as a raise meets it: the fake engine, shared.
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
        49
    }
}

/// A pool whose engine is told rungs, as it loads: at the smallest rung,
/// its span backed.
fn loaded_told() -> (SharedPool, FakeEngine) {
    let mut facts = facts();
    facts.tells_rungs = true;
    (SharedPool::new(&facts, 256).unwrap(), FakeEngine::new())
}

/// [`loaded_told`] raised to the largest rung, as a first wide pass takes
/// it (the engine told), nothing held yet.
pub(super) fn told_pool() -> (SharedPool, FakeEngine) {
    let (mut pool, engine) = loaded_told();
    let raise = pool.plan_raise(2048).unwrap();
    assert_eq!(
        pool.raise_applied(&raise).unwrap(),
        Some((16, 2048)),
        "the engine is told the rung it did not hold"
    );
    pool.rung_told();
    assert_eq!(pool.told(), 2048);
    (pool, engine)
}

/// The physical chunks `slots` lie in, ascending.
fn chunks_of(pool: &SharedPool, slots: &[Slot]) -> Vec<u32> {
    let mut chunks: Vec<u32> = slots
        .iter()
        .map(|slot| pool.pool().physical(slot.chunk).unwrap())
        .collect();
    chunks.sort_unstable();
    chunks.dedup();
    chunks
}

fn state_chunks(pool: &SharedPool, records: &[u32]) -> Vec<u32> {
    let mut chunks: Vec<u32> = records
        .iter()
        .flat_map(|&record| pool.record_slots(record))
        .map(|slot| pool.pool().physical(slot.chunk).unwrap())
        .collect();
    chunks.sort_unstable();
    chunks.dedup();
    chunks
}

/// [`told_pool`] lowered to the 256 rung with every segment mapped, and
/// yield's pressure walked down to `target`: the idle prefix goes first,
/// then the span below the activation, its last shed.
pub(super) fn shed_pool(target: u32) -> (SharedPool, FakeEngine) {
    let (mut pool, mut engine) = told_pool();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    pool.lower_rung(256).unwrap();
    engine.calls.clear();
    // Past what yield can give back (the null chunk's segment stays): it
    // ends with no room, having shed all it could.
    assert!(matches!(
        walk_segments(&mut pool, target, 1, &mut engine),
        Retarget::NoRoom { .. }
    ));
    (pool, engine)
}

#[test]
fn the_span_is_backed_at_load_on_every_engine() {
    for tells in [false, true] {
        let mut facts = facts();
        facts.tells_rungs = tells;
        let pool = SharedPool::new(&facts, 256).unwrap();
        for segment in 6..8 {
            assert_eq!(
                pool.pool().segments().state(segment),
                SegmentState::Committed
            );
        }
        assert_eq!(pool.pool().span_committed(), 2);
        assert_eq!(pool.pool().span_segments(), 2);
        assert_eq!(pool.pool().span_to_regrow(), None);
        assert_eq!(pool.mapped_segments(), 4, "the loaded prefix and the span");
        assert_eq!(pool.rung(), if tells { 16 } else { 2048 });
        pool.pool().check();
    }
}

/// Yield sheds in PLEX order: the idle prefix first, then the span below
/// the activation set aside, its last; the activation's own segment stays.
/// A raise into what it gave back waits on the governor's grow, which backs
/// the span again first, whole, and the raise then moves nothing.
#[test]
fn yield_sheds_the_span_last_and_the_governor_s_grow_backs_it_before_a_raise() {
    let (mut pool, mut engine) = shed_pool(1);
    let sheds: Vec<&Call> = engine
        .calls
        .iter()
        .filter(|call| matches!(call, Call::Release(_) | Call::ReleaseSpan(_)))
        .collect();
    assert!(
        matches!(sheds.last(), Some(Call::ReleaseSpan(1))),
        "the span last: {:?}",
        engine.calls
    );
    assert!(
        sheds[..sheds.len() - 1]
            .iter()
            .all(|call| matches!(call, Call::Release(_))),
        "the prefix first"
    );
    assert_eq!(
        pool.pool().span_committed(),
        1,
        "the activation's own stays"
    );
    assert_eq!(pool.pool().segments().state(6), SegmentState::Uncommitted);
    assert_eq!(pool.pool().segments().state(7), SegmentState::Committed);
    pool.pool().check();

    // A raise into what yield gave back: refused by name, nothing moved.
    assert_eq!(
        pool.plan_raise(2048).unwrap_err(),
        RungError::Pool(SlotError::SpanShed {
            rung: 2048,
            segments: 2
        })
    );
    assert_eq!(pool.rung(), 256);

    // The pressure lifts: the grow backs the span first, whole.
    engine.calls.clear();
    assert_eq!(
        walk_segments(&mut pool, 8, 2, &mut engine),
        Retarget::Reached
    );
    assert_eq!(engine.calls.first(), Some(&Call::CommitSpan(2)));
    assert_eq!(pool.pool().span_committed(), 2);
    let table = pool.pool().chunk_table().to_vec();
    let raise = pool.plan_raise(2048).unwrap();
    pool.raise_applied(&raise).unwrap();
    pool.rung_told();
    assert_eq!(pool.rung(), 2048);
    assert_eq!(pool.pool().chunk_table(), table.as_slice(), "nothing moved");
    pool.pool().check();
}

/// Yield gives back no span segment state or cache holds a chunk in, nor
/// one under the activation set aside: at the top rung it sheds only the
/// prefix.
#[test]
fn yield_never_sheds_a_span_segment_in_use_or_under_the_activation() {
    // At the top rung the activation reaches into segment 6.
    let (mut pool, mut engine) = told_pool();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    engine.calls.clear();
    assert!(matches!(
        walk_segments(&mut pool, 1, 1, &mut engine),
        Retarget::NoRoom { .. }
    ));
    assert!(
        !engine
            .calls
            .iter()
            .any(|call| matches!(call, Call::ReleaseSpan(_))),
        "{:?}",
        engine.calls
    );
    assert_eq!(pool.pool().span_committed(), 2);
    // Lowered with cache in segment 6: it stays too.
    let (mut pool, mut engine) = told_pool();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    pool.lower_rung(256).unwrap();
    let cached = span_cache(&mut pool, 2);
    assert_eq!(chunks_of(&pool, &cached), [52]);
    engine.calls.clear();
    assert!(matches!(
        walk_segments(&mut pool, 1, 1, &mut engine),
        Retarget::NoRoom { .. }
    ));
    assert_eq!(pool.pool().span_committed(), 2, "cache holds segment 6");
    pool.pool().check();
}

#[test]
fn an_engine_not_told_rungs_starts_at_the_largest_rung_and_leaves_the_span_to_cache() {
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    assert_eq!(pool.pool().committed_chunks(), 4 * SEGMENT);
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    pool.lower_rung(16).unwrap();
    assert_eq!(pool.next_move(8, 1), None, "nothing to back or give back");
    assert!(
        engine
            .calls
            .iter()
            .all(|call| matches!(call, Call::Commit(_)))
    );
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    assert!(
        state_chunks(&pool, &all).iter().all(|&chunk| chunk < 52),
        "live state never opens in the top rung's span"
    );
    let cached = span_cache(&mut pool, 2);
    assert_eq!(
        chunks_of(&pool, &cached),
        [52],
        "the span the lower leaves opens for cache at once"
    );
    pool.pool().check();
}

/// A refused `commit_span` (the governor's grow of the span yield gave
/// back): the span stays as yield left it, a raise into it still waits, and
/// the grow is not asked again until room is freed. Nothing faults.
#[test]
fn a_refused_grow_of_the_span_keeps_the_raise_waiting() {
    let (mut pool, mut engine) = shed_pool(1);
    engine.refuse = true;
    assert_eq!(
        walk_segments(&mut pool, 8, 2, &mut engine),
        Retarget::Reached
    );
    assert_eq!(pool.pool().span_committed(), 1);
    assert!(pool.plan_raise(2048).is_err_and(|error| error.span_shed()));
    pool.block_span_commit(true);
    engine.refuse = false;
    assert!(
        !matches!(pool.next_move(8, 3), Some(SegmentMove::CommitSpan { .. })),
        "not asked again on the next wake"
    );
    pool.block_span_commit(false);
    assert!(matches!(
        pool.next_move(8, 3),
        Some(SegmentMove::CommitSpan { n: 2 })
    ));
}

/// On an engine told rungs a raise reaches it (`set_rung`) before any fire
/// of the wider rung; refused, the raise is taken back and fires stay at
/// the rung in hand. An engine not told rungs hears no raise.
#[tokio::test(flavor = "current_thread")]
async fn a_raise_reaches_an_engine_told_rungs_and_can_be_refused() {
    let (mut pool, mut engine) = told_pool();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(pool.lower_rung(256).unwrap(), Some((2048, 256)));
    engine.calls.clear();
    engine.refuse = true;
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    let lane = Lane(engine.clone());

    // Refused: taken back, and the refusal answered.
    let refused = take_raise(&shared, |_| 2048, &lane).await.unwrap_err();
    assert!(refused.contains("refused rung 2048's notice"), "{refused}");
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.rung(), 256, "the rung in hand");
        assert_eq!(pool.pool().act_reserved(), 4);
        assert!(pool.awaiting_notice().is_none());
        pool.pool().check();
    }
    assert!(engine.lock().unwrap().calls.is_empty());

    // Taken: the engine hears it before any fire of the wider rung.
    engine.lock().unwrap().refuse = false;
    let taken = take_raise(&shared, |_| 2048, &lane).await.unwrap();
    taken.expect("raised");
    assert_eq!(engine.lock().unwrap().calls, [Call::Rung(256, 2048)]);
    let pool = shared.lock().unwrap();
    assert_eq!(pool.rung(), 2048);
    assert!(pool.awaiting_notice().is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn an_engine_not_told_rungs_hears_no_raise() {
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(pool.lower_rung(256).unwrap(), None, "no lower notice");
    engine.calls.clear();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));
    let taken = take_raise(&shared, |_| 2048, &Lane(engine.clone()))
        .await
        .unwrap();
    assert!(taken.is_some());
    assert!(engine.lock().unwrap().calls.is_empty(), "no lane trip");
    assert_eq!(shared.lock().unwrap().rung(), 2048);
}

/// A lower on an on-demand pool is told the engine only past the walk's
/// retire fence: once every fire cut to the old rung has retired.
#[test]
fn a_lower_notice_comes_after_the_retire_fence() {
    let (mut pool, mut engine) = told_pool();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    let mut walk = RungWalk::default();
    assert_eq!(
        walk.step(&pool, 256, u32::MAX, u32::MAX, 5, 5).unwrap(),
        RungStep::Cap(256)
    );
    assert_eq!(
        walk.step(&pool, 256, 256, u32::MAX, 5, 5).unwrap(),
        RungStep::Wait,
        "a wider step still in flight"
    );
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 8, 5).unwrap(),
        RungStep::Wait
    );
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 8, 7).unwrap(),
        RungStep::Wait,
        "fire 8 not retired"
    );
    assert_eq!(
        walk.step(&pool, 256, 256, 256, 8, 8).unwrap(),
        RungStep::Lower(256)
    );
    // Only now is the rung lowered, and the notice it yields sent.
    assert_eq!(pool.lower_rung(256).unwrap(), Some((2048, 256)));
}

/// A pool whose engine is told rungs starts at the smallest rung, the one
/// the engine lays its fires by at load, fires cut to it. A first wider pass does not widen fires in
/// place: the raise reaches the engine (`set_rung`) first, and only then is
/// the cap lifted for the pass's fire.
#[tokio::test(flavor = "current_thread")]
async fn an_engine_told_rungs_starts_at_the_smallest_rung_and_raises_through_the_engine() {
    const ENGINE: usize = 56;
    let (pool, mut engine) = loaded_told();
    assert_eq!(pool.rung(), 16, "the floor rung");
    assert_eq!(pool.pool().act_reserved(), 1);
    assert_eq!(pool.start_cap(), 16);
    let shared: Shared = Arc::new(Mutex::new(pool));
    let _grow = SharedGrow::new(0, ENGINE, shared.clone());
    assert_eq!(fire_cap(ENGINE), 16, "fires start cut to it");
    {
        let mut pool = shared.lock().unwrap();
        assert_eq!(
            walk_segments(&mut pool, 6, 0, &mut engine),
            Retarget::Reached
        );
    }
    engine.calls.clear();
    let engine = Arc::new(Mutex::new(engine));

    // A pass of 2000 rows asks: nothing widens before the engine hears.
    raise_now(&shared, ENGINE, 2000);
    assert_eq!(shared.lock().unwrap().rung(), 16);
    assert_eq!(fire_cap(ENGINE), 16, "no fire wider yet");
    assert!(engine.lock().unwrap().calls.is_empty());

    // The raise reaches the engine, then the pass's fire may be wide.
    let taken = take_raise(&shared, |pool| pool.rung_for(2000), &Lane(engine.clone()))
        .await
        .unwrap();
    assert!(taken.is_some());
    assert_eq!(engine.lock().unwrap().calls, [Call::Rung(16, 2048)]);
    raise_now(&shared, ENGINE, 2000);
    assert_eq!(shared.lock().unwrap().rung(), 2048);
    assert_eq!(
        fire_cap(ENGINE),
        u32::MAX,
        "the cap lifted after the engine was told"
    );
}

/// An engine not told rungs starts at the largest rung, fires uncut.
#[test]
fn an_engine_not_told_rungs_starts_at_the_largest_rung() {
    const ENGINE: usize = 55;
    let pool = pool();
    assert_eq!(pool.rung(), 2048);
    assert_eq!(pool.start_cap(), u32::MAX);
    let _grow = SharedGrow::new(0, ENGINE, Arc::new(Mutex::new(pool)));
    assert_eq!(fire_cap(ENGINE), u32::MAX);
}

/// Every power of two from 1 to 2048, the engine told rungs.
fn fine_told() -> (SharedPool, FakeEngine) {
    let mut facts = facts();
    facts.tells_rungs = true;
    facts.act_chunks = (0..12).map(|k| (1u32 << k, 1 + k)).collect();
    (SharedPool::new(&facts, 256).unwrap(), FakeEngine::new())
}

/// Mac's footprint at rest: on an on-demand pool with one live lane, a
/// 5-token pass raises the rung to the smallest that covers it (8), not to
/// the engine's whole lane count; idle again, the pool comes back down
/// after the governor's patience, the carve held costing resident memory.
#[tokio::test(flavor = "current_thread")]
async fn a_small_pass_raises_to_the_rung_covering_it_and_an_idle_pool_lowers_after_patience() {
    const ENGINE: usize = 54;
    let (mut pool, mut engine) = fine_told();
    pool.set_max_lanes(512);
    assert_eq!(
        walk_segments(&mut pool, 6, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(pool.rung(), 1);
    engine.calls.clear();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let grow = SharedGrow::new(0, ENGINE, shared.clone()).with_live_lanes(|| 1);
    assert_eq!(fire_cap(ENGINE), 1);
    grant_rung(ENGINE, u32::MAX);
    // Idle with one live lane: the walk holds the floor, no raise.
    grow.walk_rung().await;
    assert_eq!(
        shared.lock().unwrap().rung(),
        1,
        "not the engine's 512 lanes"
    );

    // The pass: 5 rows ask the rung covering them, as every pass wider
    // than a row does whatever the cap (`pipeline::fire::ask_rung`).
    let want = want_rows(ENGINE, 5);
    let engine = Arc::new(Mutex::new(engine));
    let taken = take_raise(
        &shared,
        |pool| pool.rung_for(wanted_rows(ENGINE)),
        &Lane(engine.clone()),
    )
    .await
    .unwrap();
    assert!(taken.is_some());
    assert_eq!(engine.lock().unwrap().calls, [Call::Rung(1, 8)]);
    raise_now(&shared, ENGINE, 5);
    assert_eq!(
        shared.lock().unwrap().rung(),
        8,
        "the smallest rung covering 5"
    );
    assert_eq!(fire_cap(ENGINE), 8);

    // The pass is through: the pool stays up until the patience is out.
    drop(want);
    for _ in 0..PATIENCE {
        grow.ticked_for_test();
        grow.walk_rung().await;
        assert_eq!(fire_cap(ENGINE), 8);
    }
    grow.ticked_for_test();
    grow.walk_rung().await;
    assert_eq!(fire_cap(ENGINE), 1, "past the patience: fires cut back");
    grant_rung(ENGINE, 1);
    grow.walk_rung().await;
    grow.walk_rung().await;
    assert_eq!(shared.lock().unwrap().rung(), 1, "lowered back");
    assert_eq!(shared.lock().unwrap().pool().act_reserved(), 1);
}

/// The 8B c256 hang (decodes cut at a low cap, waiting on a raise nobody
/// funds) cannot come back: the walk never lowers the rung under the
/// decode fire the live lanes take, however long the pool idles, so a
/// decode batch of every live lane is never held for a raise; and a batch
/// past it is raised at once where nothing moves.
#[tokio::test(flavor = "current_thread")]
async fn the_walk_never_lowers_under_the_live_lanes_decode() {
    const ENGINE: usize = 53;
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    let shared: Shared = Arc::new(Mutex::new(pool));
    let grow = SharedGrow::new(0, ENGINE, shared.clone()).with_live_lanes(|| 200);
    assert_eq!(fire_cap(ENGINE), u32::MAX);
    for _ in 0..3 * PATIENCE {
        grow.ticked_for_test();
        grant_rung(ENGINE, fire_cap(ENGINE));
        grow.walk_rung().await;
        assert!(fire_cap(ENGINE) >= 200, "a 200-lane decode is never cut");
    }
    assert_eq!(
        shared.lock().unwrap().rung(),
        256,
        "down to the live lanes' rung"
    );
    // The batch of every live lane asks: nothing to raise, nothing held.
    raise_now(&shared, ENGINE, 200);
    assert_eq!(fire_cap(ENGINE), 256);
    // State takes the span the rung lent; a batch past the live lanes is
    // raised at once once nothing lies in its span.
    let all = {
        let mut pool = shared.lock().unwrap();
        let n = pool.records_available() as usize;
        pool.take_records(n).unwrap()
    };
    {
        let mut pool = shared.lock().unwrap();
        pool.give_records(&all);
        pool.pool_mut().retire_through(u64::MAX);
    }
    raise_now(&shared, ENGINE, 300);
    assert_eq!(shared.lock().unwrap().rung(), 2048);
    assert_eq!(fire_cap(ENGINE), u32::MAX);
}

/// `can_grow` is whether the walk's next step maps a segment, by the one
/// function both take ([`SharedPool::grow_step`]), over every pool state
/// the stall could stand in: the prefix whole or not, the span yield gave
/// back or whole, a span commit refused or not, the target above what is
/// mapped or at it, and a waiter short of kv, of state, or none. Spelled
/// out apart from the code: a live waiter grows only into the prefix,
/// under a target above it, and backing the span again is never its step.
#[test]
fn can_grow_is_whether_the_walk_s_next_step_maps_a_segment() {
    const ENGINE: usize = 46;
    let waiters = [None, Some(Short::Kv), Some(Short::State)];
    let mut cases = 0;
    for prefix_whole in [false, true] {
        for shed in [false, true] {
            for blocked in [false, true] {
                for below in [false, true] {
                    for waiter in waiters {
                        let case = format!(
                            "prefix_whole={prefix_whole} shed={shed} blocked={blocked} \
                             below={below} waiter={waiter:?}"
                        );
                        let mut pool = grow_pool(prefix_whole, shed);
                        assert_eq!(pool.pool().span_to_regrow().is_some(), shed, "{case}");
                        pool.block_span_commit(blocked);
                        if let Some(short) = waiter {
                            wait_short(&mut pool, short);
                        }
                        let mapped = pool.mapped_segments();
                        let shared: Shared = Arc::new(Mutex::new(pool));
                        let mut grow = SharedGrow::new(0, ENGINE, shared.clone());
                        let target = if below { u32::MAX } else { mapped };
                        let _ = grow.retarget(u64::from(target));
                        let can = grow.can_grow();
                        let goal = grow.goal_for_test();
                        let step = shared.lock().unwrap().next_move(goal, 1);
                        let maps = matches!(
                            step,
                            Some(SegmentMove::Commit { .. } | SegmentMove::CommitSpan { .. })
                        );
                        assert_eq!(can, grow.live_step_for_test().is_some(), "{case}");
                        if waiter.is_some() {
                            assert_eq!(can, maps, "{case}: can_grow is the walk's own step");
                            assert_eq!(can, below && !prefix_whole, "{case}");
                        } else {
                            let pool = shared.lock().unwrap();
                            assert_eq!(
                                maps,
                                pool.grow_step(goal, GrowFor::Target).is_some(),
                                "{case}: the walk takes the step for no waiter"
                            );
                        }
                        assert!(
                            waiter.is_none()
                                || !matches!(step, Some(SegmentMove::CommitSpan { .. })),
                            "{case}: no live waiter's step backs the span again"
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 48);
}

/// Mac's over-demand stall, on the span yield gave back: the prefix whole,
/// a live waiter short of room is told the pool does not grow, and the walk
/// backs no span for it (live state never opens there), so the planner goes
/// on to demote, sweep and evict rather than wait on a grow that never
/// comes.
#[test]
fn a_live_waiter_beside_the_span_yield_gave_back_is_told_the_pool_does_not_grow() {
    const ENGINE: usize = 48;
    for short in [Short::Kv, Short::State] {
        let mut pool = grow_pool(true, true);
        assert_eq!(pool.pool().span_to_regrow(), Some(2));
        wait_short(&mut pool, short);
        let mapped = pool.mapped_segments();
        let shared: Shared = Arc::new(Mutex::new(pool));
        let mut grow = SharedGrow::new(0, ENGINE, shared.clone());
        let _ = grow.retarget(u64::from(u32::MAX));
        assert!(!grow.can_grow(), "{short:?}: no step maps a segment");
        assert!(!grow.promise_grow(), "{short:?}: so nothing is promised");
        assert_eq!(grow.goal_for_test(), mapped, "{short:?}: the prefix");
        assert_eq!(
            shared.lock().unwrap().next_move(mapped, 1),
            None,
            "{short:?}: and the walk backs no span"
        );
    }
}

/// A grow promised a waiter that the walk does not make (here the governor
/// lowered its target since) is told as no grow when the walk ends, once;
/// a walk with nothing promised tells nothing.
#[tokio::test(flavor = "current_thread")]
async fn a_promised_grow_the_walk_does_not_make_is_told_as_no_grow() {
    const ENGINE: usize = 45;
    let mut pool = grow_pool(false, false);
    wait_short(&mut pool, Short::Kv);
    let mapped = pool.mapped_segments();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let told = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let counted = told.clone();
    let mut grow = SharedGrow::new(0, ENGINE, shared.clone()).with_no_grow(move || {
        counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
    let _ = grow.retarget(u64::from(u32::MAX));
    assert!(grow.promise_grow(), "the prefix grows");
    assert!(grow.grow_promised());
    let _ = grow.retarget(u64::from(mapped));
    assert!(!grow.can_grow(), "the target stands at what is mapped");
    grow.walk().await;
    assert_eq!(told.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert!(!grow.grow_promised(), "told once");
    assert_eq!(shared.lock().unwrap().mapped_segments(), mapped);
    grow.walk().await;
    assert_eq!(
        told.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "nothing promised: nothing told"
    );
}

/// An engine told rungs whose span is backed from the load (Metal with its
/// span at load, `tells_rungs` its own fact): it starts at the smallest
/// rung and hears every raise and lower (`Engine::set_rung`), and that is
/// all it hears: no span segment is committed or released, nothing is
/// backed by a raise. A guest term the top rung's span cannot seat with
/// it is cut to the widest rung that can, as on any engine.
#[tokio::test(flavor = "current_thread")]
async fn an_engine_told_rungs_on_a_span_backed_at_load_hears_each_rung_and_nothing_else() {
    let mut facts = facts();
    facts.tells_rungs = true;
    let mut pool = SharedPool::new(&facts, 256).unwrap();
    assert!(pool.tells_rungs(), "its own fact, the span backed at load");
    assert_eq!(pool.rung(), 16, "it starts at the smallest rung");
    assert_eq!(pool.told(), 16);
    assert_eq!(pool.start_cap(), 16, "fires start cut to it");
    let mut engine = FakeEngine::new();
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    let mapped = pool.mapped_segments();
    engine.calls.clear();
    let shared: Shared = Arc::new(Mutex::new(pool));
    let engine = Arc::new(Mutex::new(engine));

    // The raise: the engine hears it before any fire of the wider rung.
    let raised = take_raise(&shared, |_| 2048, &Lane(engine.clone()))
        .await
        .unwrap();
    assert!(raised.is_some());
    assert_eq!(engine.lock().unwrap().calls, [Call::Rung(16, 2048)]);
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.rung(), 2048);
        assert_eq!(pool.told(), 2048);
        assert!(pool.awaiting_notice().is_none(), "the notice was taken");
        assert_eq!(pool.mapped_segments(), mapped, "nothing backed by it");
    }

    // The lower: its notice, past the retire fence.
    let notice = shared.lock().unwrap().lower_rung(256).unwrap();
    assert_eq!(notice, Some((2048, 256)));
    engine.lock().unwrap().set_rung(2048, 256).unwrap();
    assert_eq!(
        engine.lock().unwrap().calls,
        [Call::Rung(16, 2048), Call::Rung(2048, 256)],
        "rung notices and nothing else: no span commit or release"
    );
    {
        let mut pool = shared.lock().unwrap();
        assert_eq!(
            pool.next_move(8, 1),
            None,
            "no span segment to back or give back"
        );
        assert_eq!(pool.mapped_segments(), mapped);
    }

    // A guest term the top rung's span cannot seat with the largest rung
    // (12 + 3 of 12): the widest rung that fits it, 256 (4 + 3).
    {
        let mut pool = shared.lock().unwrap();
        pool.pool_mut().set_standing(3 * CHUNK);
        assert_eq!(pool.widest_fit(), Some(256));
        assert_eq!(pool.fit_cap(), 256);
        assert_eq!(pool.rung_for(5000), 256);
    }
}
