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
use crate::governor::PATIENCE;
use crate::store::shared::programs::{Mover, take_raise};

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
fn on_demand_pool() -> (SharedPool, FakeEngine) {
    let (mut pool, engine) = loaded_on_demand();
    pool.pool_mut().set_rung(2048).unwrap();
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
/// the engine's whole lane count; idle again, the pool comes back down
/// after the governor's patience, the carve held costing resident memory.
#[tokio::test(flavor = "current_thread")]
async fn a_small_pass_raises_to_the_rung_covering_it_and_an_idle_pool_lowers_after_patience() {
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

    // The pass: 5 rows, cut narrower by the cap, ask the rung covering them.
    assert!(pass_asks(5, 5, fire_cap(ENGINE)));
    assert!(!pass_asks(5, 5, 8), "a cap that covers it: no ask");
    let want = want_rows(ENGINE, 5);
    let engine = Arc::new(Mutex::new(engine));
    let taken = take_raise(
        &shared,
        |pool| rung_goal(pool, ENGINE, wanted_rows(ENGINE)),
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

/// One process decoding 64 sequences, at rest between its steps (nothing
/// queued, no pass asking): its 64 lanes are live, each a seat its working
/// set holds, so the walk's goal stays the rung of 64 however long, and the
/// next step's fire is not cut. Once the set is released the lanes are gone
/// and the idle pool comes down after the patience.
#[tokio::test(flavor = "current_thread")]
async fn one_process_of_64_lanes_at_rest_keeps_the_rung_of_64() {
    use crate::store::chunks::Seats;
    use crate::store::genmap::GenMap;
    use crate::store::kv::page_table::WorkingSetId;
    const ENGINE: usize = 64;
    let (mut pool, mut engine) = fine_on_demand();
    pool.set_max_lanes(512);
    assert_eq!(
        walk_segments(&mut pool, 6, 0, &mut engine),
        Retarget::Reached
    );
    let shared: Shared = Arc::new(Mutex::new(pool));
    // The engine's own count: the seats, not a stand-in.
    let grow = SharedGrow::new(0, ENGINE, shared.clone());
    grant_rung(ENGINE, u32::MAX);
    let seats = Arc::new(Mutex::new(Seats::new(256)));
    seats.lock().unwrap().count_for(ENGINE);
    let ws: WorkingSetId = GenMap::<_, ()>::new().insert(());

    // The one process's wave: 64 lanes seated, the rung raised for them.
    seats.lock().unwrap().seats(ws, 64, 0).unwrap();
    assert_eq!(live_lanes(ENGINE), 64);
    let want = want_rows(ENGINE, 64);
    let engine = Arc::new(Mutex::new(engine));
    take_raise(
        &shared,
        |pool| rung_goal(pool, ENGINE, asked_rows(ENGINE)),
        &Lane(engine.clone()),
    )
    .await
    .unwrap();
    raise_now(&shared, ENGINE, 64);
    assert_eq!(shared.lock().unwrap().rung(), 64);
    assert_eq!(fire_cap(ENGINE), 64);

    // At rest: no ask stands, one process is all there is.
    drop(want);
    assert_eq!(wanted_rows(ENGINE), 0);
    assert_eq!(grow.walk_rows(), 64, "its lanes, not its one process");
    for _ in 0..PATIENCE + 2 {
        grow.ticked_for_test();
        grow.walk_rung().await;
        assert_eq!(fire_cap(ENGINE), 64, "the next step is not cut");
        assert_eq!(shared.lock().unwrap().rung(), 64);
    }

    // The sequences end: no lane is live, and the idle pool comes down.
    seats.lock().unwrap().release(ws);
    assert_eq!(grow.walk_rows(), 0);
    for _ in 0..=PATIENCE {
        grow.ticked_for_test();
        grow.walk_rung().await;
    }
    assert_eq!(fire_cap(ENGINE), 1, "past the patience: fires cut back");
    grant_rung(ENGINE, 1);
    grow.walk_rung().await;
    grow.walk_rung().await;
    assert_eq!(shared.lock().unwrap().rung(), 1, "lowered back");
}

/// One lane decoding by fires of 2 rows the device places (never cut), back
/// to back with a settle between, the pool resting at its prompt's rung of
/// 32, on a ladder where the rungs of 1 and 2 rows take different chunks.
/// Each fire's own ask goes when it settles, but the lane counts at the 2
/// rows it stages next step, so the walk's goal stays the rung of 2 between
/// fires: the cap is cut once, from the prompt's rung to the lane's, never
/// under the lane's fire, and the lowering to the lane's rung completes.
#[tokio::test(flavor = "current_thread")]
async fn a_lane_decoding_two_rows_a_fire_keeps_its_rung_between_fires() {
    use crate::store::chunks::Seats;
    use crate::store::genmap::GenMap;
    use crate::store::kv::page_table::WorkingSetId;
    const ENGINE: usize = 67;
    let (mut pool, mut engine) = fine_on_demand();
    pool.set_max_lanes(512);
    assert_eq!(
        walk_segments(&mut pool, 6, 0, &mut engine),
        Retarget::Reached
    );
    assert_ne!(
        pool.pool().act_chunks(1),
        pool.pool().act_chunks(2),
        "no chunk class hides a goal of 1 behind a rung of 2"
    );
    let shared: Shared = Arc::new(Mutex::new(pool));
    let grow = SharedGrow::new(0, ENGINE, shared.clone());
    grant_rung(ENGINE, u32::MAX);
    let seats = Arc::new(Mutex::new(Seats::new(256)));
    seats.lock().unwrap().count_for(ENGINE);
    let ws: WorkingSetId = GenMap::<_, ()>::new().insert(());

    // The prompt: the lane seated, its pass asking 32 rows a fire.
    seats.lock().unwrap().seats(ws, 1, 0).unwrap();
    let want = want_rows(ENGINE, 32);
    let engine = Arc::new(Mutex::new(engine));
    take_raise(
        &shared,
        |pool| rung_seatable(pool, ENGINE, asked_rows(ENGINE)),
        &Lane(engine.clone()),
    )
    .await
    .unwrap();
    raise_now(&shared, ENGINE, 32);
    assert_eq!(shared.lock().unwrap().rung(), 32);
    assert_eq!(fire_cap(ENGINE), 32);
    drop(want);

    // The decode. Every change of the cap is kept.
    let mut caps = vec![fire_cap(ENGINE)];
    let note = |caps: &mut Vec<u32>| {
        let cap = fire_cap(ENGINE);
        if caps.last() != Some(&cap) {
            caps.push(cap);
        }
    };
    for _ in 0..4 * PATIENCE {
        // A fire: its seat stamped with the rows it stages whole, and its
        // own ask, held until it settles.
        seats.lock().unwrap().seats(ws, 1, 2).unwrap();
        let want = want_rows(ENGINE, 2);
        raise_now(&shared, ENGINE, 2);
        note(&mut caps);
        assert!(fire_cap(ENGINE) >= 2, "the fire is not held for a raise");
        grow.walk_rung().await;
        note(&mut caps);
        // It settles: its ask goes, the lane's rows stand.
        drop(want);
        assert_eq!(wanted_rows(ENGINE), 0);
        assert_eq!(grow.walk_rows(), 2, "the rows the lane stages next step");
        grow.walk_rung().await;
        note(&mut caps);
        // The scheduler's turn publishes what it has in flight; a tick.
        grant_rung(ENGINE, fire_cap(ENGINE));
        grow.ticked_for_test();
    }
    assert_eq!(caps, [32, 2], "one cut, to the lane's rung, and no flip");
    assert_eq!(
        shared.lock().unwrap().rung(),
        2,
        "the lowering to the lane's rung completes"
    );

    // The sequence ends: nothing of it counts.
    seats.lock().unwrap().release(ws);
    assert_eq!(grow.walk_rows(), 0);
}

/// The patience runs from the last moment an ask stood above the goal, for
/// each goal. A pool resting at its prompt's rung of 32 with one live lane,
/// and asks of 2 rows made and dropped fire after fire, so the rows asked
/// flip between 2 and 1, both under where the pool rests, on a ladder where
/// the rungs of 1 and 2 take different chunks. Fires are cut to the rung
/// of 2 once no ask above it has stood for the patience, and never to the
/// rung of 1 while an ask of 2 stood within it: the cap does not flip, and
/// the lowering to the rung of 2, once begun, is not begun again by a look
/// that finds the ask gone. PATIENCE ticks after the last ask, and not a
/// tick sooner, exactly one cut: to the rung of the lane.
#[tokio::test(flavor = "current_thread")]
async fn the_patience_runs_from_the_last_ask_above_the_goal() {
    const ENGINE: usize = 68;
    const FIRES: u64 = 4 * PATIENCE as u64;
    /// One look of the walk at `tick`; a move of the cap or of the rung
    /// is kept with the tick it came at.
    async fn look(grow: &SharedGrow, tick: u64, moves: &mut Vec<(u64, u32, u32)>) {
        grow.walk_rung().await;
        let now = (fire_cap(ENGINE), grow.shared().lock().unwrap().rung());
        if moves.last().map(|&(_, cap, rung)| (cap, rung)) != Some(now) {
            moves.push((tick, now.0, now.1));
        }
    }
    let (mut pool, mut engine) = fine_on_demand();
    pool.set_max_lanes(512);
    assert_eq!(
        walk_segments(&mut pool, 6, 0, &mut engine),
        Retarget::Reached
    );
    assert_ne!(
        pool.pool().act_chunks(1),
        pool.pool().act_chunks(2),
        "no chunk class hides a goal of 1 behind a rung of 2"
    );
    let shared: Shared = Arc::new(Mutex::new(pool));
    let grow = SharedGrow::new(0, ENGINE, shared.clone()).with_live_lanes(|| 1);
    grant_rung(ENGINE, u32::MAX);
    let engine = Arc::new(Mutex::new(engine));
    let mut moves = Vec::new();
    let mut tick = 0u64;

    // The prompt: its pass asks 32 rows a fire, and is done at tick 0.
    let want = want_rows(ENGINE, 32);
    take_raise(
        &shared,
        |pool| rung_seatable(pool, ENGINE, asked_rows(ENGINE)),
        &Lane(engine.clone()),
    )
    .await
    .unwrap();
    raise_now(&shared, ENGINE, 32);
    look(&grow, tick, &mut moves).await;
    drop(want);
    look(&grow, tick, &mut moves).await;
    assert_eq!(moves, [(0, 32, 32)]);

    // The decode: a fire a tick, each asking 2 rows until it settles. The
    // walk looks while the ask stands and again once it is gone.
    for _ in 0..FIRES {
        grow.ticked_for_test();
        tick += 1;
        let want = want_rows(ENGINE, 2);
        raise_now(&shared, ENGINE, 2);
        look(&grow, tick, &mut moves).await;
        drop(want);
        assert_eq!(grow.walk_rows(), 1, "the ask is gone: the lane's row");
        look(&grow, tick, &mut moves).await;
        grant_rung(ENGINE, fire_cap(ENGINE));
    }
    let patience = u64::from(PATIENCE);
    let caps = |moves: &[(u64, u32, u32)]| {
        let mut caps: Vec<(u64, u32)> = Vec::new();
        for &(tick, cap, _) in moves {
            if caps.last().map(|&(_, last)| last) != Some(cap) {
                caps.push((tick, cap));
            }
        }
        caps
    };
    assert_eq!(
        caps(&moves),
        [(0, 32), (patience, 2)],
        "one cut, PATIENCE ticks after the prompt's ask, to the rung still asked"
    );
    assert_eq!(
        shared.lock().unwrap().rung(),
        2,
        "the lowering to the rung of 2 completes between the asks"
    );

    // No ask from here: the last stood at tick FIRES.
    for _ in 1..PATIENCE {
        grow.ticked_for_test();
        tick += 1;
        look(&grow, tick, &mut moves).await;
        grant_rung(ENGINE, fire_cap(ENGINE));
        assert_eq!(fire_cap(ENGINE), 2, "inside the patience of the last ask");
    }
    grow.ticked_for_test();
    tick += 1;
    assert_eq!(tick, FIRES + patience);
    for _ in 0..3 {
        look(&grow, tick, &mut moves).await;
        grant_rung(ENGINE, fire_cap(ENGINE));
    }
    assert_eq!(
        caps(&moves),
        [(0, 32), (patience, 2), (FIRES + patience, 1)],
        "exactly one cut, PATIENCE ticks after the last ask"
    );
    let rungs: Vec<u32> = moves.iter().map(|&(_, _, rung)| rung).collect();
    let mut steps = rungs.clone();
    steps.dedup();
    assert_eq!(steps, [32, 2, 1], "each lowering begun once and completed");
}

/// A lift of fires' cap (a pass's, at its ask) and a cut (the walk's) are
/// one move of the cap, logged once, when the cap changes, and at one
/// level: info where fires move to a rung of another chunk class, debug
/// within a class. A ladder whose small rungs share their activation's
/// chunks (1 to 16 take 1, 32 to 128 take 2).
#[tokio::test(flavor = "current_thread")]
async fn a_lift_and_a_cut_of_the_cap_are_each_logged_once_at_one_level() {
    use tracing::Level;
    const ENGINE: usize = 69;
    const LIFTED: &str = "fires lifted to the activation rung";
    const CUT: &str = "fires cut to the activation rung";
    let facts = ChunkFacts {
        act_chunks: vec![
            (1, 1),
            (2, 1),
            (4, 1),
            (8, 1),
            (16, 1),
            (32, 2),
            (64, 2),
            (128, 2),
            (256, 4),
            (512, 4),
            (1024, 8),
            (2048, 12),
        ],
        ..facts()
    };
    facts.validate().unwrap();
    let pool = SharedPool::new(&facts, 256).unwrap();
    assert_eq!(pool.rung(), 2048);
    // The line and the level of each move: a lift and a cut between the
    // same two caps are logged at the same level.
    for (from, tokens, level) in [
        (u32::MAX, 16, Level::INFO),
        (16, 128, Level::INFO),
        (128, 32, Level::DEBUG),
        (32, 64, Level::DEBUG),
        (2048, u32::MAX, Level::DEBUG),
        (1, 2048, Level::INFO),
    ] {
        let (down, up) = if tokens < from {
            (CUT, LIFTED)
        } else {
            (LIFTED, CUT)
        };
        assert_eq!(cap_move(&pool, from, tokens), Some((down, level)));
        assert_eq!(cap_move(&pool, tokens, from), Some((up, level)));
        assert_eq!(cap_move(&pool, tokens, tokens), None, "no change: no line");
    }

    // The walk's cut and a pass's lift both move the cap through it.
    let shared: Shared = Arc::new(Mutex::new(pool));
    let grow = SharedGrow::new(0, ENGINE, shared.clone());
    assert_eq!(fire_cap(ENGINE), u32::MAX);
    // Nothing asked: past the patience the walk cuts fires to the widest
    // rung of the smallest class.
    for _ in 0..=PATIENCE {
        grow.walk_rung().await;
        grow.ticked_for_test();
    }
    grow.walk_rung().await;
    assert_eq!(fire_cap(ENGINE), 16);
    // A pass asks 64 rows: lifted to the widest rung of their class, which
    // the rung in hand holds.
    raise_now(&shared, ENGINE, 64);
    assert_eq!(fire_cap(ENGINE), 128);
    {
        let pool = shared.lock().unwrap();
        move_fire_cap(&pool, ENGINE, 32);
        assert_eq!(fire_cap(ENGINE), 32);
        assert_eq!(cap_move(&pool, fire_cap(ENGINE), 32), None);
    }
}

/// A process waiting for room contributes no rows to what the walk holds
/// the rung for: 64 lanes decoding beside a pass of 32 lanes that waits on
/// the planner's grant hold the rung of 64, not of 96, so the span past it
/// can be lent to the state the waiter needs. Served, its lanes count
/// again.
#[test]
fn a_process_waiting_for_room_holds_no_rows_of_the_rung() {
    use crate::store::chunks::{Seats, WaitsForRoom};
    use crate::store::genmap::GenMap;
    use crate::store::kv::page_table::WorkingSetId;
    const ENGINE: usize = 65;
    let (mut pool, _) = fine_on_demand();
    pool.set_max_lanes(512);
    let shared: Shared = Arc::new(Mutex::new(pool));
    let grow = SharedGrow::new(0, ENGINE, shared.clone());
    let seats = Arc::new(Mutex::new(Seats::new(256)));
    seats.lock().unwrap().count_for(ENGINE);
    let mut sets = GenMap::<_, ()>::new();
    let (decoding, waiter): (WorkingSetId, WorkingSetId) = (sets.insert(()), sets.insert(()));
    seats.lock().unwrap().seats(decoding, 64, 0).unwrap();
    seats.lock().unwrap().seats(waiter, 32, 0).unwrap();
    let goal = || rung_goal(&shared.lock().unwrap(), ENGINE, grow.walk_rows());
    assert_eq!(grow.walk_rows(), 96);
    assert_eq!(goal(), 128, "both live: the rung covering 96");

    let waits = WaitsForRoom::mark(&seats, [waiter]);
    assert_eq!(live_lanes(ENGINE), 64);
    assert_eq!(grow.walk_rows(), 64, "the waiter's 32 lanes stage no row");
    assert_eq!(goal(), 64);
    // The one count: the scheduler's ask and a pass's read the same lanes.
    assert_eq!(asked_rows(ENGINE), 64);

    drop(waits);
    assert_eq!(grow.walk_rows(), 96, "served: its lanes stage rows again");
    assert_eq!(goal(), 128);
}

/// Metal's rungs (Qwen3-0.6B on an M1 Max) over a pool at rung 1 as a
/// wave of 160 prompts found it: 2365 chunks committed against the 2927 a
/// raise to rung 8192 needs, the state the lent span holds with no free
/// chunk under it and no tier to move it to. The state's slot refs with it.
fn second_wave_pool() -> (SharedPool, Vec<u32>) {
    let mut facts = facts();
    facts.span_backed_on_demand = true;
    facts.total_chunks = 4096;
    facts.segment_chunks = 1;
    facts.segments = SegmentFacts {
        committed: 670,
        max: 4096,
        growth: SegmentGrowth::TwoEnded,
    };
    facts.act_chunks = vec![
        (1, 3),
        (2, 5),
        (4, 10),
        (8, 20),
        (16, 39),
        (32, 78),
        (64, 156),
        (128, 311),
        (256, 622),
        (512, 1244),
        (1024, 1334),
        (2048, 1514),
        (4096, 1629),
        (8192, 1695),
        (10240, 1729),
    ];
    facts.weights = Vec::new();
    facts.host_chunks = 0;
    facts.disk_chunks = 0;
    facts.validate().unwrap();
    let mut pool = SharedPool::new(&facts, 256).unwrap();
    assert_eq!(pool.rung(), 1);
    // The span's top 1000 chunks backed (lent at rung 1, committed as state
    // took them); the state fills the prefix and 562 of them.
    pool.pool_mut().commit_span(1000);
    let list = pool.list().unwrap().pool;
    let per_chunk = u64::from(pool.pool().shape(list).unwrap().slots_per_chunk);
    let chunks = u64::from(1231 - pool.pool().state_chunks());
    let refs = pool.take_refs(list, chunks * per_chunk).unwrap();
    (pool, refs)
}

/// A wave's prompts summed ask a rung the pool cannot seat (Mac's wave of
/// 160: rung 8192 for 7151 rows, 2927 chunks against 2365 committed,
/// nothing movable). The goal is cut at the ask to the widest rung that
/// fits: the raise to it is taken and fires are cut to it; the walk never
/// takes the refused-raise path (the only one that logs "the activation
/// rung waits" and holds raises until room is freed), however often it
/// looks while the ask stands; and the cut rises once room is freed.
#[tokio::test(flavor = "current_thread")]
async fn an_ask_past_what_is_committed_takes_the_widest_rung_that_fits() {
    use crate::scheduler::worker::{LaneStandIn, PoolOp};
    const ENGINE: usize = 66;
    let (pool, refs) = second_wave_pool();
    assert_eq!(
        pool.plan_raise(8192).err(),
        Some(RungError::Pool(SlotError::Budget {
            needed: 2927,
            committed: 2365
        }))
    );
    // Nothing movable: the 562 chunks of state in the lent span have no
    // free chunk under them, and the pool has no tier under the device.
    assert_eq!(pool.raise_short(8192).0, 562);
    // 128 fits (its 311 chunks are committed and free); 256 does not.
    assert_eq!(pool.seatable(8192), 128);
    assert!(pool.plan_raise(128).is_ok());
    assert!(pool.plan_raise(256).is_err());
    assert_eq!(pool.seatable(64), 64, "an ask that fits is not cut");
    assert_eq!(rung_goal(&pool, ENGINE, 7151), 8192, "the rung asked");
    assert_eq!(rung_seatable(&pool, ENGINE, 7151), 128, "the goal");

    let list = pool.list().unwrap().pool;
    let shared: Shared = Arc::new(Mutex::new(pool));
    let mut lane = LaneStandIn::install(ENGINE);
    let grow = SharedGrow::new(0, ENGINE, shared.clone());
    assert_eq!(fire_cap(ENGINE), 1);
    grant_rung(ENGINE, u32::MAX);

    // The wave's ask, as the scheduler makes it for its prompts summed.
    let _want = want_rows(ENGINE, 7151);
    raise_now(&shared, ENGINE, 7151);
    {
        let mut walk = std::pin::pin!(grow.walk_rung());
        assert!(
            futures::poll!(walk.as_mut()).is_pending(),
            "the raise is the engine's to back"
        );
        let Some((PoolOp::Rung { from: 1, to: 128 }, answer)) = lane.pool_op() else {
            panic!("the raise to the widest rung that fits is at the lane");
        };
        answer.send(Ok(())).unwrap();
        walk.await;
    }
    assert_eq!(shared.lock().unwrap().rung(), 128);
    assert_eq!(fire_cap(ENGINE), 128, "fires take the rung that fits");
    assert!(!grow.raise_blocked_for_test(), "no wait state entered");

    // The ask stands (7151 rows) through every later look: nothing is
    // tried, nothing is refused, nothing waits.
    for _ in 0..3 {
        grow.ticked_for_test();
        grow.walk_rung().await;
        assert!(lane.pool_op().is_none());
        assert!(!grow.raise_blocked_for_test());
        assert_eq!(shared.lock().unwrap().rung(), 128);
        assert_eq!(fire_cap(ENGINE), 128);
    }

    // Room is freed (the state goes): the cut rises at the walk's next
    // look, to the rung asked.
    {
        let mut pool = shared.lock().unwrap();
        pool.give_refs(list, &refs);
        pool.pool_mut().retire_through(u64::MAX);
        assert_eq!(rung_seatable(&pool, ENGINE, 7151), 8192);
    }
    grow.room_freed();
    {
        let mut walk = std::pin::pin!(grow.walk_rung());
        assert!(futures::poll!(walk.as_mut()).is_pending());
        let Some((
            PoolOp::Rung {
                from: 128,
                to: 8192,
            },
            answer,
        )) = lane.pool_op()
        else {
            panic!("the raise to the rung asked, once it fits");
        };
        answer.send(Ok(())).unwrap();
        walk.await;
    }
    assert_eq!(shared.lock().unwrap().rung(), 8192);
    assert_eq!(fire_cap(ENGINE), 8192);
    assert!(!grow.raise_blocked_for_test());
}

/// A wave arriving at a pool resting at rung 1, the span backed on demand:
/// the scheduler's ask for its 64 one-row launches cannot widen fires in
/// the call (the engine backs the carve first), so it wakes the walk. The
/// raise reaches the lane with no governor tick and no room freed between,
/// and once the engine has it a step takes the 64 rows.
#[tokio::test(flavor = "current_thread")]
async fn an_ask_the_call_cannot_take_wakes_the_walk_at_the_ask() {
    use crate::scheduler::worker::{LaneStandIn, PoolOp};
    const ENGINE: usize = 58;
    let (mut pool, mut engine) = fine_on_demand();
    pool.set_max_lanes(512);
    assert_eq!(
        walk_segments(&mut pool, 6, 0, &mut engine),
        Retarget::Reached
    );
    assert_eq!(pool.rung(), 1);
    let shared: Shared = Arc::new(Mutex::new(pool));
    let mut lane = LaneStandIn::install(ENGINE);
    let grow = SharedGrow::new(0, ENGINE, shared.clone()).with_live_lanes(|| 1);
    assert_eq!(
        fire_cap(ENGINE),
        1,
        "at rest: fires cut to the smallest rung"
    );
    grant_rung(ENGINE, 1);
    grow.start();
    let settle = || async {
        for _ in 0..20 {
            tokio::task::yield_now().await;
        }
    };
    settle().await;
    assert_eq!(shared.lock().unwrap().rung(), 1);
    assert!(lane.pool_op().is_none(), "nothing asked: nothing sent");

    // The batch's ask, as the scheduler makes it: its rows wanted, and the
    // raise tried in the call.
    let _want = want_rows(ENGINE, 64);
    raise_now(&shared, ENGINE, 64);
    assert_eq!(
        fire_cap(ENGINE),
        1,
        "not in the call: the engine comes first"
    );
    // No tick and no room freed: the ask alone woke the walk.
    settle().await;
    let Some((PoolOp::Rung { from: 1, to: 64 }, answer)) = lane.pool_op() else {
        panic!("the raise is at the lane, carried at the ask");
    };
    assert_eq!(fire_cap(ENGINE), 1, "cut until the engine has the carve");
    answer.send(Ok(())).unwrap();
    settle().await;
    assert_eq!(shared.lock().unwrap().rung(), 64);
    // A step's width is the cap's (`scheduler::worker`): 64 rows now.
    assert_eq!(fire_cap(ENGINE), 64, "the next step takes the 64 rows");
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
