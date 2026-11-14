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

fn on_demand_pool() -> (SharedPool, FakeEngine) {
    let mut facts = facts();
    facts.span_backed_on_demand = true;
    (
        SharedPool::new(&facts, 256).unwrap(),
        FakeEngine::on_demand(),
    )
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
