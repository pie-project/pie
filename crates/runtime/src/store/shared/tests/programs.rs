//! REDESIGN §6: the registered programs' guest term in the rung arithmetic.
//! A rung of `r` tokens carries `min(r, max_lanes)` lanes; its activation
//! is the model's chunks plus the chunks the worst fire of those lanes
//! takes for the pairs open, each pair paying `fixed + lanes × a_lane` for
//! the members the fire gives it and nothing where it gives none
//! ([`::engine::GuestTerm`]). A bind of a program whose term raises the
//! span waits until the raise is funded and taken; the last instance's
//! close drops the pair, and the next lower gives the room back.

use std::sync::atomic::{AtomicU32, Ordering};

use ::engine::{FireFacts, ProgramId};
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

// ---- ruling 2: the rung the engine holds, and the span it states ----

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
