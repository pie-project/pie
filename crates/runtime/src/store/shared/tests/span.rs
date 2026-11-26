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
use crate::store::shared::programs::{Mover, carry_raise};

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

/// The walk never lowers the rung under the decode fire the live lanes take,
/// however long the pool idles, so a decode batch of every live lane is never
/// held for a raise nobody funds; and a batch past it is raised at once where
/// nothing moves.
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

/// `can_grow` is whether the walk's next step maps a segment, by the one
/// function both take ([`SharedPool::grow_step`]), over every pool state
/// the stall could stand in: the prefix whole or not, a span lent or not,
/// a span commit refused or not, the target above what is mapped or at it,
/// and a waiter short of kv, of state, or none. Spelled out apart from the
/// code: a live waiter grows into the prefix, then into the span the rung
/// lends (span lending), under a target above what is mapped, never while
/// a refused commit waits for an event that can change the device's answer.
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
                        if let Some(short) = waiter {
                            wait_short(&mut pool, short);
                        }
                        let mapped = pool.mapped_segments();
                        let shared: Shared = Arc::new(Mutex::new(pool));
                        let mut grow = SharedGrow::new(0, ENGINE, shared.clone());
                        let target = if below { u32::MAX } else { mapped };
                        let _ = grow.retarget(u64::from(target));
                        // A commit refused at this target.
                        if blocked {
                            shared.lock().unwrap().block_commit(target);
                        }
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
                            assert_eq!(can, below && !blocked && (!prefix_whole || lent), "{case}");
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
    let lent = pool
        .take_records(pool.records_available() as usize)
        .unwrap();
    let below: Vec<u32> = (0..u32::try_from(lent.len()).unwrap()).collect();
    pool.give_records(&below);
    pool.pool_mut().retire_through(u64::MAX);
    let physical = state_chunks(&pool, &lent);
    let logical: Vec<u32> = physical
        .iter()
        .map(|&chunk| {
            pool.pool()
                .chunk_table()
                .iter()
                .position(|&p| p == chunk)
                .unwrap() as u32
        })
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
    let unmoved: Vec<u32> = logical
        .iter()
        .copied()
        .filter(|chunk| !moved.contains(chunk))
        .collect();
    assert!(
        pool.refusals_pending(&moved),
        "until the engine took the copy"
    );
    pool.compaction_copied(epoch, 2);
    assert!(
        !pool.refusals_pending(&moved),
        "the copy taken: their fires go"
    );
    assert_eq!(
        pool.refusals_pending(&unmoved),
        !unmoved.is_empty(),
        "those with no free chunk below wait for room"
    );
    pool.pool().check();
}
