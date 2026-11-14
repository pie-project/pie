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

    // Refused: taken back.
    let taken = take_raise(&shared, |_| 2048, &lane).await.unwrap();
    assert!(taken.is_none());
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
