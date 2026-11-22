//! The activation span backed on demand (`ChunkFacts::span_backed_on_demand`,
//! where a mapped heap is resident whether touched or not): no span segment
//! is committed at load; a span chunk a lowered rung lends is opened for
//! state only once `commit_span` backs its segment; a span segment goes back
//! (`release_span`) once the rung is raised past it and it holds nothing.
//! With the flag off, the span is backed from the load as before.
//!
//! The fixture's span: segments 6 and 7 (chunks 48..64), the largest rung's
//! activation 52..64.

use super::*;
use crate::governor::PoolTarget;
use crate::store::shared::programs::{Mover, carry_raise, take_raise};

/// The engine's lane as a raise meets it: the fake engine, shared.
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
        49
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

/// An on-demand pool as it loads: at the smallest rung.
fn loaded_on_demand() -> (SharedPool, FakeEngine) {
    let mut facts = facts();
    facts.span_backed_on_demand = true;
    (
        SharedPool::new(&facts, 256).unwrap(),
        FakeEngine::on_demand(),
    )
}

/// An on-demand pool raised to the largest rung, as a first wide pass
/// takes it (the engine told), nothing held yet.
pub(super) fn on_demand_pool() -> (SharedPool, FakeEngine) {
    let (mut pool, engine) = loaded_on_demand();
    let evacuation = pool.plan_raise(2048).unwrap();
    assert_eq!(
        pool.raise_applied(&evacuation).unwrap(),
        Some((16, 2048)),
        "the engine is told the rung it did not hold"
    );
    pool.raise_copied(0);
    assert_eq!(pool.told(), 2048);
    (pool, engine)
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

#[test]
fn no_span_segment_is_committed_at_load() {
    let (pool, _) = on_demand_pool();
    let segments = pool.pool().segments();
    assert_eq!(pool.pool().top_segment(), 6);
    for segment in 6..8 {
        assert_eq!(segments.state(segment), SegmentState::Uncommitted);
    }
    assert_eq!(pool.pool().span_committed(), 0);
    assert_eq!(pool.mapped_segments(), 2, "the committed prefix only");
    // The rung's activation still counts as room the engine backs itself.
    assert_eq!(pool.rung(), 2048);
    assert_eq!(pool.pool().committed_chunks(), 2 * SEGMENT + 12);
    pool.pool().check();
}

#[test]
fn a_lent_span_chunk_opens_only_once_its_segment_is_committed() {
    let (mut pool, mut engine) = on_demand_pool();
    // The whole prefix, no more: the span stays unbacked.
    assert_eq!(
        walk_segments(&mut pool, 6, 0, &mut engine),
        Retarget::Reached
    );
    assert!(
        !engine
            .calls
            .iter()
            .any(|call| matches!(call, Call::CommitSpan(_)))
    );
    // The 256 rung lends 48..60 to state; none of it is backed yet.
    pool.lower_rung(256).unwrap();
    assert_eq!(
        pool.pool().span_to_commit(),
        Some(1),
        "60.. down: segment 7 first"
    );
    let first = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    assert!(
        state_chunks(&pool, &first).iter().all(|&chunk| chunk < 48),
        "no state in an unbacked span segment"
    );
    // The governor wants one segment more: the walk backs span segment 7.
    assert_eq!(
        walk_segments(&mut pool, 7, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(engine.calls.last(), Some(&Call::CommitSpan(1)));
    assert_eq!(pool.pool().span_committed(), 1);
    let second = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    let lent = state_chunks(&pool, &second);
    assert!(lent.iter().any(|&chunk| (56..60).contains(&chunk)));
    assert!(
        lent.iter()
            .all(|&chunk| chunk < 48 || (56..60).contains(&chunk))
    );
    for &chunk in &lent {
        assert!(engine.backed(chunk), "state on unbacked chunk {chunk}");
    }
    pool.pool().check();
}

#[test]
fn a_span_segment_goes_back_once_the_rung_is_raised_past_it() {
    let (mut pool, mut engine) = on_demand_pool();
    assert_eq!(
        walk_segments(&mut pool, 6, 0, &mut engine),
        Retarget::Reached
    );
    pool.lower_rung(256).unwrap();
    assert_eq!(
        walk_segments(&mut pool, 7, 0, &mut engine),
        Retarget::Reached
    );
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    assert!(state_chunks(&pool, &all).iter().any(|&chunk| chunk >= 56));
    // Lent and holding state: it stays.
    assert_eq!(pool.pool().span_to_release(), None);
    // Room below, the rung raised past segment 7: its state moves down.
    pool.give_records(&all[..all.len() / 2]);
    pool.pool_mut().retire_through(u64::MAX);
    let evacuation = pool.plan_raise(2048).unwrap();
    assert!(!evacuation.moves.is_empty());
    engine
        .evacuate(&SharedPool::rung_moves(&evacuation))
        .unwrap();
    pool.apply_rung(&evacuation, 10).unwrap();
    assert!(
        state_chunks(&pool, &all[all.len() / 2..])
            .iter()
            .all(|&chunk| chunk < 52)
    );
    // Not while the fire carrying the copy may still read the old chunks.
    assert_eq!(pool.pool().span_to_release(), None);
    pool.pool_mut().retire_through(10);
    assert_eq!(pool.pool().span_to_release(), Some(0));
    assert_eq!(
        walk_segments(&mut pool, 6, 11, &mut engine),
        Retarget::Reached
    );
    assert_eq!(engine.calls.last(), Some(&Call::ReleaseSpan(0)));
    assert_eq!(pool.pool().span_committed(), 0);
    assert_eq!(pool.pool().segments().state(7), SegmentState::Uncommitted);
    pool.pool().check();
}

#[test]
fn with_the_flag_off_the_span_is_backed_from_the_load_as_before() {
    let mut pool = pool();
    let mut engine = FakeEngine::new();
    for segment in 6..8 {
        assert_eq!(
            pool.pool().segments().state(segment),
            SegmentState::Committed
        );
    }
    assert_eq!(pool.pool().committed_chunks(), 4 * SEGMENT);
    assert_eq!(
        walk_segments(&mut pool, 8, 0, &mut engine),
        Retarget::Reached
    );
    pool.lower_rung(16).unwrap();
    assert_eq!(pool.pool().span_to_commit(), None);
    assert_eq!(pool.pool().span_to_release(), None);
    assert_eq!(pool.pool().span_committed(), 0);
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
        state_chunks(&pool, &all).iter().any(|&chunk| chunk >= 48),
        "the lent span opens at once"
    );
    pool.pool().check();
}

/// A refused `commit_span` (the device has no room for the placement heap)
/// is a refused lend: the span chunks stay closed, the rung stays where it
/// is, and state goes below or waits as any waiter. Nothing faults; a later
/// walk may try again.
#[test]
fn a_refused_commit_span_lends_nothing() {
    let (mut pool, mut engine) = on_demand_pool();
    assert_eq!(
        walk_segments(&mut pool, 6, 0, &mut engine),
        Retarget::Reached
    );
    pool.lower_rung(256).unwrap();
    engine.refuse = true;
    assert_eq!(
        walk_segments(&mut pool, 7, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(pool.pool().span_committed(), 0);
    assert_eq!(pool.rung(), 256, "the rung stays");
    assert_eq!(pool.pool().span_to_commit(), Some(1), "asked again later");
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    assert!(state_chunks(&pool, &all).iter().all(|&chunk| chunk < 48));
    pool.pool().check();
    // The device has room again: the next walk lends it.
    engine.refuse = false;
    assert_eq!(
        walk_segments(&mut pool, 7, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(pool.pool().span_committed(), 1);
}

/// A raise whose copy the engine refuses is taken back: the state where it
/// lay (the engine's table unchanged), the activation the rung in hand's,
/// no fence left, and a later raise succeeds.
#[test]
fn a_refused_raise_is_taken_back() {
    let (mut pool, mut engine) = on_demand_pool();
    assert_eq!(
        walk_segments(&mut pool, 6, 0, &mut engine),
        Retarget::Reached
    );
    pool.lower_rung(256).unwrap();
    assert_eq!(
        walk_segments(&mut pool, 7, 0, &mut engine),
        Retarget::Reached
    );
    let all = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    pool.give_records(&all[..all.len() / 2]);
    pool.pool_mut().retire_through(u64::MAX);
    let kept = &all[all.len() / 2..];
    let before = state_chunks(&pool, kept);
    let evacuation = pool.plan_raise(2048).unwrap();
    assert!(!evacuation.moves.is_empty());
    pool.raise_applied(&evacuation).unwrap();
    assert_eq!(pool.copying(), Some(256));
    engine.refuse = true;
    assert!(
        engine
            .evacuate(&SharedPool::rung_moves(&evacuation))
            .is_err()
    );
    pool.raise_refused(&evacuation);
    assert_eq!(pool.rung(), 256, "the rung in hand");
    assert_eq!(pool.pool().act_reserved(), 4);
    assert!(pool.copying().is_none());
    assert_eq!(state_chunks(&pool, kept), before, "the state where it lay");
    assert_eq!(engine.table.as_slice(), pool.pool().chunk_table());
    pool.pool().check();
    // Tried again with the engine willing: it lands.
    engine.refuse = false;
    let evacuation = pool.plan_raise(2048).unwrap();
    pool.raise_applied(&evacuation).unwrap();
    engine
        .evacuate(&SharedPool::rung_moves(&evacuation))
        .unwrap();
    pool.raise_copied(1);
    assert_eq!(pool.rung(), 2048);
    assert_eq!(engine.table.as_slice(), pool.pool().chunk_table());
}

/// On an on-demand pool a raise that moves nothing still reaches the
/// engine (`set_rung`), which backs its carve there; refused, the raise is
/// taken back and fires stay at the rung in hand. With the span backed from
/// the load, the same raise makes no engine call.
#[tokio::test(flavor = "current_thread")]
async fn a_move_free_raise_reaches_an_on_demand_engine_and_can_be_refused() {
    let (mut pool, mut engine) = on_demand_pool();
    assert_eq!(
        walk_segments(&mut pool, 6, 0, &mut engine),
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
    assert!(refused.contains("refused the carve"), "{refused}");
    {
        let pool = shared.lock().unwrap();
        assert_eq!(pool.rung(), 256, "the rung in hand");
        assert_eq!(pool.pool().act_reserved(), 4);
        assert!(pool.copying().is_none());
        pool.pool().check();
    }
    assert!(engine.lock().unwrap().calls.is_empty());

    // Backed: the engine hears it before any fire of the wider rung.
    engine.lock().unwrap().refuse = false;
    let taken = take_raise(&shared, |_| 2048, &lane).await.unwrap();
    let evacuation = taken.expect("raised");
    assert!(evacuation.moves.is_empty(), "nothing to move");
    assert_eq!(engine.lock().unwrap().calls, [Call::Rung(256, 2048)]);
    let pool = shared.lock().unwrap();
    assert_eq!(pool.rung(), 2048);
    assert!(pool.copying().is_none());
}

#[tokio::test(flavor = "current_thread")]
async fn with_the_flag_off_a_move_free_raise_makes_no_engine_call() {
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
    let (mut pool, mut engine) = on_demand_pool();
    assert_eq!(
        walk_segments(&mut pool, 6, 0, &mut engine),
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

/// An on-demand pool starts at the smallest rung, the one the engine backs
/// at load, fires cut to it. A first wider pass does not widen fires in
/// place: the raise reaches the engine (`set_rung`) first, and only then is
/// the cap lifted for the pass's fire.
#[tokio::test(flavor = "current_thread")]
async fn an_on_demand_pool_starts_at_the_smallest_rung_and_raises_through_the_engine() {
    const ENGINE: usize = 56;
    let (pool, mut engine) = loaded_on_demand();
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
        "the cap lifted after the engine backed it"
    );
}

/// With the flag off, a pool starts at the largest rung, fires uncut.
#[test]
fn with_the_flag_off_a_pool_starts_at_the_largest_rung() {
    const ENGINE: usize = 55;
    let pool = pool();
    assert_eq!(pool.rung(), 2048);
    assert_eq!(pool.start_cap(), u32::MAX);
    let _grow = SharedGrow::new(0, ENGINE, Arc::new(Mutex::new(pool)));
    assert_eq!(fire_cap(ENGINE), u32::MAX);
}

/// Every power of two from 1 to 2048, the span backed on demand.
fn fine_on_demand() -> (SharedPool, FakeEngine) {
    let mut facts = facts();
    facts.span_backed_on_demand = true;
    facts.act_chunks = (0..12).map(|k| (1u32 << k, 1 + k)).collect();
    (
        SharedPool::new(&facts, 256).unwrap(),
        FakeEngine::on_demand(),
    )
}

/// Mac's footprint at rest: on an on-demand pool with one live lane, a
/// 5-token pass raises the rung to the smallest that covers it (8), not to
/// the engine's whole lane count; idle again, the pool stays up while
/// nothing is short, and comes back down once a waiter needs the chunks the
/// carve holds.
#[tokio::test(flavor = "current_thread")]
async fn a_small_pass_raises_to_the_rung_covering_it_and_an_idle_pool_lowers_once_needed() {
    const ENGINE: usize = 54;
    let (mut pool, mut engine) = fine_on_demand();
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

    // The pass is through: the pool stays up while nothing is short.
    drop(want);
    for _ in 0..64 {
        grow.walk_rung().await;
        assert_eq!(fire_cap(ENGINE), 8);
    }
    shared.lock().unwrap().pool_mut().wait(1, 0, 1);
    grow.walk_rung().await;
    assert_eq!(fire_cap(ENGINE), 1, "a waiter is short: fires cut back");
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
    // A waiter short of a slot: the walk lowers as far as it may.
    shared.lock().unwrap().pool_mut().wait(1, 0, 1);
    for _ in 0..6 {
        grant_rung(ENGINE, fire_cap(ENGINE));
        grow.walk_rung().await;
        assert!(fire_cap(ENGINE) >= 200, "a 200-lane decode is never cut");
    }
    assert_eq!(
        shared.lock().unwrap().rung(),
        256,
        "down to the live lanes' rung"
    );
    shared.lock().unwrap().pool_mut().unwait(1);
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

/// A refused commit_span is not asked again on the walk's next wake, only
/// once room is freed lifts the block.
#[test]
fn a_refused_commit_span_is_asked_again_only_once_room_is_freed() {
    let (mut pool, mut engine) = on_demand_pool();
    assert_eq!(
        walk_segments(&mut pool, 6, 0, &mut engine),
        Retarget::Reached
    );
    pool.lower_rung(256).unwrap();
    assert!(matches!(
        pool.next_move(7, 1),
        Some(SegmentMove::CommitSpan { n: 1 })
    ));
    // The engine refused it: blocked.
    pool.block_span_commit(true);
    assert!(
        !matches!(pool.next_move(7, 2), Some(SegmentMove::CommitSpan { .. })),
        "not asked again on the next wake"
    );
    // Room freed lifts it.
    pool.block_span_commit(false);
    assert!(matches!(
        pool.next_move(7, 3),
        Some(SegmentMove::CommitSpan { n: 1 })
    ));
}

/// `can_grow` is whether the walk's next step maps a segment, by the one
/// function both take ([`SharedPool::grow_step`]), over every pool state
/// the stall could stand in: the prefix whole or not, a span lent or not,
/// a span commit refused or not, the target above what is mapped or at it,
/// and a waiter short of kv, of state, or none. Spelled out apart from the
/// code: a live waiter grows into the prefix, then into the span the rung
/// lends (span lending), under a target above what is mapped, never while
/// a refused span commit waits for room.
#[test]
fn can_grow_is_whether_the_walk_s_next_step_maps_a_segment() {
    const ENGINE: usize = 46;
    let waiters = [None, Some(Short::Kv), Some(Short::State)];
    let mut cases = 0;
    for prefix_whole in [false, true] {
        for lent in [false, true] {
            for blocked in [false, true] {
                for below in [false, true] {
                    for waiter in waiters {
                        let case = format!(
                            "prefix_whole={prefix_whole} lent={lent} blocked={blocked} \
                             below={below} waiter={waiter:?}"
                        );
                        let mut pool = grow_pool(prefix_whole, lent);
                        assert_eq!(pool.pool().span_to_commit().is_some(), lent, "{case}");
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
                        assert_eq!(can, maps, "{case}: can_grow is the walk's own step");
                        if waiter.is_some() {
                            assert_eq!(
                                can,
                                below && (!prefix_whole || (lent && !blocked)),
                                "{case}"
                            );
                            assert_eq!(
                                matches!(step, Some(SegmentMove::CommitSpan { .. })),
                                below && prefix_whole && lent && !blocked,
                                "{case}: the lent span backed once the prefix is whole"
                            );
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 48);
}

/// Mac Pro at 2 GiB on 5120cbc3c (Metal, the span backed on demand): the
/// prefix whole, a rung of 32 chunks under a top span of 165, and the 126
/// chunks it lends unbacked (committable_one_chunk=0), a live waiter short
/// of room waited on evictions. Span lending: the walk backs the lent span
/// for it, the goal past the prefix, and the waiter's room opens there.
/// The commit is within the range reserved at load: the span's own
/// segments, the pool's total unchanged. Once every lent segment is
/// backed, no step is left and nothing more is promised.
#[test]
fn a_live_waiter_beside_an_unbacked_lent_span_is_funded_by_lending_it() {
    const ENGINE: usize = 48;
    for short in [Short::Kv, Short::State] {
        let mut pool = grow_pool(true, true);
        let total = pool.pool().total_chunks();
        let segments = pool.pool().segments().count();
        let top = pool.pool().top_segment();
        assert_eq!(pool.pool().span_to_commit(), Some(1));
        let (lent, backed, _) = pool.pool().span_lent();
        assert!(lent > 0 && backed < lent, "{short:?}: the span lent, unbacked");
        assert_eq!(pool.pool().free_chunks(), 0, "{short:?}: nothing open");
        assert!(pool.pool().committable_runs(1) > 0, "{short:?}: the lent span commits");
        wait_short(&mut pool, short);
        let mapped = pool.mapped_segments();
        let shared: Shared = Arc::new(Mutex::new(pool));
        let mut grow = SharedGrow::new(0, ENGINE, shared.clone());
        let _ = grow.retarget(u64::from(u32::MAX));
        assert!(grow.can_grow(), "{short:?}: the lent span is a step");
        assert!(grow.goal_for_test() > mapped, "{short:?}: past the prefix");
        let mut pool = shared.lock().unwrap();
        let mut steps = 0;
        while let Some(step) = pool.next_move(u32::MAX, 1) {
            let SegmentMove::CommitSpan { n } = step else {
                panic!("{short:?}: only the lent span is backed, got {step:?}");
            };
            assert!(top + n <= segments + n && n <= segments - top, "{short:?}: a span segment");
            pool.span_committed(n);
            steps += 1;
        }
        assert!(steps > 0, "{short:?}: the span is backed for the waiter");
        let (lent, backed, _) = pool.pool().span_lent();
        assert_eq!(backed, lent, "{short:?}: every lent chunk backed");
        assert!(pool.pool().free_chunks() > 0, "{short:?}: room opens in the span");
        assert_eq!(pool.pool().total_chunks(), total, "{short:?}: no allocation past load");
        assert_eq!(pool.pool().segments().count(), segments, "{short:?}: the range as loaded");
        drop(pool);
        assert!(!grow.can_grow(), "{short:?}: nothing left to back");
    }
}

/// A raise to a rung whose activation takes lent span chunks takes them
/// back: state lying there moves down into free chunks below where they
/// stand; where none do, the raise is short by the chunks to free (the
/// planner frees them as any head's room, its occupants evicted), and once
/// they are free the raise moves the state down and fits.
#[test]
fn a_rising_rung_reclaims_the_lent_span() {
    let mut pool = grow_pool(true, true);
    while let Some(n) = pool.pool().span_to_commit() {
        pool.span_committed(n);
    }
    // State takes the lent span: every chunk below is held already.
    let records = pool.records_available() as usize;
    let lent = pool.take_records(records).unwrap();
    let (_, _, reclaim) = pool.pool().span_lent();
    assert!(reclaim > 0, "state lies in the lent span");
    let top = 2048;
    let short = pool.pool().rung_short(top);
    assert!(short > 0, "no free chunk below: the raise is short");
    assert!(matches!(
        pool.rise_room(top),
        crate::planner::RiseRoom::Short { .. }
    ));
    // The planner frees room below (here: records that lay below go, as
    // their evictions would take them): the raise fits, and moves the
    // state lying in the span down into what they freed.
    let below: Vec<u32> = (0..u32::try_from(lent.len()).unwrap())
        .filter(|record| !lent.contains(record))
        .collect();
    assert_eq!(below.len(), lent.len(), "the records below are the older ids");
    pool.give_records(&below);
    pool.pool_mut().retire_through(u64::MAX);
    assert_eq!(pool.pool().rung_short(top), 0, "freed: the raise fits");
    let evacuation = pool.plan_raise(top).unwrap();
    assert!(!evacuation.moves.is_empty(), "the span's state moves down");
    assert!(pool.raise_applied(&evacuation).is_ok());
    assert_eq!(pool.pool().span_lent().0, 0, "the largest rung lends nothing");
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

/// Fires cut under a wide rung are still queued when the walk lowers the
/// rung and its notice lands; a pass then asks a width of a rung that
/// shares the lowered rung's chunk class. No queued fire reaches the
/// engine wider than the rung it was told: the lower narrows what the lane
/// lets through from the moment its notice goes, and the rung of the class
/// set aside is told the engine (a raise that moves nothing) before the
/// cap lifts to it.
#[tokio::test(flavor = "current_thread")]
async fn no_fire_wider_than_the_rung_the_engine_was_told_reaches_it() {
    const ENGINE: usize = 9212;
    let mut facts = facts();
    facts.span_backed_on_demand = true;
    // 256 and 512 one chunk class: the same span set aside for each.
    facts.act_chunks = vec![(16, 1), (256, 4), (512, 4), (2048, 12)];
    let mut pool = SharedPool::new(&facts, 256).unwrap();
    let evacuation = pool.plan_raise(2048).unwrap();
    assert_eq!(pool.raise_applied(&evacuation).unwrap(), Some((16, 2048)));
    pool.raise_copied(0);
    let shared: Shared = Arc::new(Mutex::new(pool));
    let _grow = SharedGrow::new(0, ENGINE, shared.clone());
    let engine = Arc::new(Mutex::new(FakeEngine::on_demand()));
    // The rung the engine was last told, as it lays a fire's inputs.
    let held = |engine: &Arc<Mutex<FakeEngine>>| {
        engine
            .lock()
            .unwrap()
            .calls
            .iter()
            .rev()
            .find_map(|call| match call {
                Call::Rung(_, to) => Some(*to),
                _ => None,
            })
            .unwrap_or(2048)
    };
    // What the scheduler hands the lane each turn: every queued fire the
    // cap covers; the lane lets through what the rung it holds covers.
    let mut queued = vec![1024u32, 297, 298];
    let mut reached = Vec::new();
    let turn = |queued: &mut Vec<u32>, reached: &mut Vec<u32>| {
        let cap = fire_cap(ENGINE);
        queued.retain(|&rows| {
            if rows > cap {
                return true;
            }
            assert!(
                rows <= engine_rows(ENGINE),
                "{rows} rows past the lane's rung"
            );
            assert!(rows <= held(&engine), "{rows} rows past the engine's rung");
            reached.push(rows);
            false
        });
    };
    assert_eq!(fire_cap(ENGINE), u32::MAX);

    // The walk cuts fires to 256 and, past the retire fence, lowers.
    let mut walk = RungWalk::default();
    let step = walk.step(&shared.lock().unwrap(), 256, u32::MAX, u32::MAX, 5, 5);
    assert_eq!(step.unwrap(), RungStep::Cap(256));
    set_fire_cap_for_test(ENGINE, 256);
    turn(&mut queued, &mut reached);
    let step = walk.step(&shared.lock().unwrap(), 256, 256, 256, 5, 5);
    assert_eq!(step.unwrap(), RungStep::Wait);
    let step = walk.step(&shared.lock().unwrap(), 256, 256, 256, 5, 5);
    assert_eq!(step.unwrap(), RungStep::Lower(256));
    let notice = shared.lock().unwrap().lower_notice(256).unwrap();
    assert_eq!(notice, Some((2048, 256)));
    assert_eq!(engine_rows(ENGINE), 256, "narrowed as the notice goes");
    turn(&mut queued, &mut reached);
    engine.lock().unwrap().set_rung(2048, 256).unwrap();
    shared.lock().unwrap().lower_landed(256).unwrap();
    turn(&mut queued, &mut reached);
    assert!(reached.is_empty(), "every queued fire is wider than 256");

    // A pass of 297 rows asks: 512 is of the class set aside, yet the
    // engine holds 256, so the cap stays until the engine hears 512.
    raise_now(&shared, ENGINE, 297);
    assert_eq!(
        fire_cap(ENGINE),
        256,
        "no fire past the rung the engine holds"
    );
    turn(&mut queued, &mut reached);
    let step = walk
        .step(&shared.lock().unwrap(), 512, 256, 256, 6, 6)
        .unwrap();
    let RungStep::Raise { rung, evacuation } = step else {
        panic!("a rung of the class set aside is a raise: {step:?}");
    };
    assert_eq!(rung, 512);
    assert!(evacuation.moves.is_empty(), "it moves nothing");
    let notice = shared.lock().unwrap().raise_applied(&evacuation).unwrap();
    assert_eq!(notice, Some((256, 512)));
    turn(&mut queued, &mut reached);
    carry_raise(&shared, &evacuation, notice, &Lane(engine.clone()))
        .await
        .unwrap();
    raise_now(&shared, ENGINE, 297);
    assert_eq!(fire_cap(ENGINE), 512);
    turn(&mut queued, &mut reached);
    assert_eq!(reached, [297, 298]);
    assert_eq!(queued, [1024], "held until the engine holds its rung");
}

/// Mac Pro at 5 GiB on 5aa45805b: an eviction's copy of state in a lent
/// span chunk. State lent the span (the engine told the lower) is evicted:
/// while the eviction's copy reads its chunks, a raise wanting them waits
/// (never lays its activation over state a copy reads); once the copy has
/// landed, the raise moves what is left there down and takes the span back.
#[test]
fn a_raise_waits_for_a_copy_reading_a_lent_chunk_then_takes_it_back() {
    let mut pool = grow_pool(true, true);
    while let Some(n) = pool.pool().span_to_commit() {
        pool.span_committed(n);
    }
    let lent = pool.take_records(pool.records_available() as usize).unwrap();
    let chunks = state_chunks(&pool, &lent);
    let (span_lent, _, _) = pool.pool().span_lent();
    assert!(span_lent > 0 && !chunks.is_empty(), "state in the lent span");
    // Room below for them: older records go.
    let below: Vec<u32> = (0..u32::try_from(lent.len()).unwrap()).collect();
    pool.give_records(&below);
    pool.pool_mut().retire_through(u64::MAX);
    // The eviction's copy reads the lent chunks.
    pool.pool_mut().moving_out(&chunks);
    assert!(
        matches!(pool.plan_raise(2048), Err(RungError::Pool(SlotError::Occupied { .. }))),
        "a copy reads them: the raise waits"
    );
    // The copy landed (the eviction moved them off, or was taken back).
    pool.pool_mut().moved_out(&chunks);
    let evacuation = pool.plan_raise(2048).expect("the copy landed: it raises");
    assert!(!evacuation.moves.is_empty(), "what lies in the span moves down");
    pool.raise_applied(&evacuation).unwrap();
    pool.pool().check();
}

/// Mac Pro at 2 GiB on 6c21ebca3: a forward refused "state chunk 231
/// (physical 433) lies inside the activation span's backed carve" and
/// three beams were lost: a raise told the engine its wider carve before
/// it moved the state lying there, and a fire in between read it inside
/// the carve. State in a lent chunk, then a raise covering it: the state
/// moves down first, and only then does the engine back the carve.
#[tokio::test(flavor = "current_thread")]
async fn a_raise_moves_the_state_down_before_the_engine_backs_its_carve() {
    let mut pool = grow_pool(true, true);
    while let Some(n) = pool.pool().span_to_commit() {
        pool.span_committed(n);
    }
    let lent = pool.take_records(pool.records_available() as usize).unwrap();
    let below: Vec<u32> = (0..u32::try_from(lent.len()).unwrap()).collect();
    pool.give_records(&below);
    pool.pool_mut().retire_through(u64::MAX);
    let evacuation = pool.plan_raise(2048).expect("room below");
    assert!(!evacuation.moves.is_empty(), "state lies in the span it raises over");
    let notice = pool.raise_applied(&evacuation).unwrap();
    assert!(notice.is_some(), "the engine is told the wider rung");
    let mut engine = FakeEngine::on_demand();
    engine.prefix = pool.pool().prefix_segments();
    engine.calls.clear();
    let engine = Arc::new(Mutex::new(engine));
    let shared: Shared = Arc::new(Mutex::new(pool));
    carry_raise(&shared, &evacuation, notice, &Lane(engine.clone()))
        .await
        .expect("carried");
    let calls = engine.lock().unwrap().calls.clone();
    let evacuated = calls.iter().position(|call| matches!(call, Call::Evacuate(_)));
    let told = calls.iter().position(|call| matches!(call, Call::Rung(..)));
    assert!(
        evacuated.is_some() && told.is_some() && evacuated < told,
        "moved down, then the carve: {calls:?}"
    );
}

/// A fire refused for the state in a chunk (`engine::Error::Placement`):
/// the walk's next step moves that chunk down to the lowest free chunk
/// below the state limit, as a compaction is carried, and its fires wait
/// until the engine has taken that copy.
#[test]
fn a_refused_chunk_moves_down_and_its_fires_wait_for_the_copy() {
    let mut pool = grow_pool(true, true);
    while let Some(n) = pool.pool().span_to_commit() {
        pool.span_committed(n);
    }
    let lent = pool.take_records(pool.records_available() as usize).unwrap();
    let below: Vec<u32> = (0..u32::try_from(lent.len()).unwrap()).collect();
    pool.give_records(&below);
    pool.pool_mut().retire_through(u64::MAX);
    let physical = state_chunks(&pool, &lent);
    let logical: Vec<u32> = physical
        .iter()
        .map(|&chunk| pool.pool().chunk_table().iter().position(|&p| p == chunk).unwrap() as u32)
        .collect();
    pool.refuse_chunks(&logical);
    assert!(pool.refusals_pending(&logical), "waits for its move");
    let step = pool.next_move(pool.mapped_segments(), 1);
    let Some(SegmentMove::Compact { applied, epoch, .. }) = step else {
        panic!("the refused chunks move down: {step:?}");
    };
    assert!(!applied.is_empty(), "the refused chunks move");
    assert!(
        applied.iter().all(|step| step.to < step.from),
        "down: {applied:?}"
    );
    let moved: Vec<u32> = applied.iter().map(|step| step.logical).collect();
    let unmoved: Vec<u32> = logical.iter().copied().filter(|chunk| !moved.contains(chunk)).collect();
    assert!(pool.refusals_pending(&moved), "until the engine took the copy");
    pool.compaction_copied(epoch, 2);
    assert!(!pool.refusals_pending(&moved), "the copy taken: their fires go");
    assert_eq!(
        pool.refusals_pending(&unmoved),
        !unmoved.is_empty(),
        "those with no free chunk below wait for room"
    );
    pool.pool().check();
}

/// A refused chunk with no free chunk below for it: its move waits, and
/// the pool says how many free chunks it lacks, which the scheduler asks
/// the planner to make (a need like any other, never a silent wait).
#[test]
fn a_refused_chunk_with_no_room_below_says_the_chunks_it_lacks() {
    let mut pool = grow_pool(true, false);
    assert!(pool.pool().open_chunks().is_empty(), "no free chunk open for state");
    let records = state_chunks(&pool, &[0]);
    let logical: Vec<u32> = records
        .iter()
        .map(|&chunk| pool.pool().chunk_table().iter().position(|&p| p == chunk).unwrap() as u32)
        .collect();
    pool.refuse_chunks(&logical);
    assert!(
        !matches!(pool.next_move(pool.mapped_segments(), 1), Some(SegmentMove::Compact { .. })),
        "no move without a free chunk below"
    );
    assert!(pool.refusals_pending(&logical), "it still waits");
    assert_eq!(
        pool.refusals_stuck(&logical),
        u32::try_from(logical.len()).unwrap(),
        "the chunks it lacks"
    );
}
