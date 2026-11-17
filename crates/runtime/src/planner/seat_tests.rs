//! The state pool's seats: one fixed set a process, none kept back for the
//! lanes a fire may run; a fork's copy-on-write set is asked for by the fire
//! that writes it, and seats no one else while it is held.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use futures::FutureExt;

use crate::inferlet::process::residency::{ProcessResidency, register_residency};
use crate::store::kv::page_table::PhysicalKvPageId;
use crate::store::registry::Stores;
use crate::store::rs::{RsGeometry, RsUnits, RsWorkingSetId};

use super::acquire::Acquired;
use super::grant::Demand;
use super::pool::{ChunkStats, PoolPort, RegistryPool};
use super::{PlannerError, ProcessId, ResidencyPlanner};

const GEOMETRY: RsGeometry = RsGeometry {
    state_size: 64,
    buffer_page_tokens: 4,
};

/// A working set whose state holds a record of its own.
fn written(stores: &Stores) -> RsWorkingSetId {
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(GEOMETRY);
    let prepared = rs.prepare_group(ws, true, false, None, None, None).unwrap();
    let published = rs.publish_prepared(prepared).unwrap();
    rs.settle(published);
    ws
}

/// Lets `ws` go as its handle's drop does: the slots retire once no fire
/// on its way reads them.
fn release(stores: &Stores, ws: RsWorkingSetId) {
    let mut rs = stores.rs.lock().unwrap();
    rs.release_working_set(ws);
    rs.retire_idle();
}

/// A process of `model` whose residency names `sets`, registered with
/// `planner`; the residency is kept so the sets stay its own.
fn process(
    planner: &ResidencyPlanner,
    model: usize,
    sets: &[RsWorkingSetId],
) -> (ProcessId, Arc<std::sync::Mutex<ProcessResidency>>) {
    let pid = ProcessId::new_v4();
    let residency = Arc::new(std::sync::Mutex::new(ProcessResidency::default()));
    residency
        .lock()
        .unwrap()
        .rs_working_sets
        .extend(sets.iter().map(|&ws| (model, 0, ws)));
    register_residency(pid, Arc::downgrade(&residency));
    planner.register(pid);
    (pid, residency)
}

fn one_slot() -> Demand {
    Demand {
        kv_pages: 0,
        rs: RsUnits {
            slots: 1,
            ..RsUnits::default()
        },
    }
}

/// With no process forking, every set of the pool is a seat, however many
/// lanes a fire runs: none is held back for copies that may never be made.
#[test]
fn every_set_seats_a_process_when_none_forks() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let model = crate::store::registry::register_model(16, &[4], &[4]);
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RegistryPool::new(
        model, 0, false,
    ))));
    assert_eq!(planner.state_seats(), Some(4));
    let procs: Vec<_> = (0..5).map(|_| process(&planner, model, &[])).collect();
    for (pid, _) in &procs[..4] {
        assert!(
            planner.admit(*pid).now_or_never().is_some(),
            "each of the pool's four sets seats a process"
        );
    }
    let mut fifth = Box::pin(planner.admit(procs[4].0));
    assert!(fifth.as_mut().now_or_never().is_none(), "a fifth waits");
    planner.unregister(procs[0].0);
    assert!(
        fifth.as_mut().now_or_never().is_some(),
        "and is seated once one leaves"
    );
}

/// A fork's copy-on-write set is asked for by the fire that writes it: short,
/// it parks as any demand does and is served once a set is let go. Held, it
/// seats no one else, and a process waiting at admission is seated once the
/// fork lets it go.
#[tokio::test(flavor = "current_thread")]
async fn a_fork_waits_for_its_copy_and_holds_a_seat_while_it_keeps_it() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[4], &[3]);
    let stores = crate::store::registry::get(model, 0);
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RegistryPool::new(
        model, 0, false,
    ))));
    let (a, b, c) = (written(&stores), written(&stores), written(&stores));
    let child = stores.rs.lock().unwrap().fork(a).unwrap();
    let (pa, _ra) = process(&planner, model, &[a, child]);
    let (pb, _rb) = process(&planner, model, &[b]);
    let (pc, _rc) = process(&planner, model, &[c]);
    for pid in [pa, pb, pc] {
        assert!(planner.admit(pid).now_or_never().is_some());
        planner.note_admitted(pid);
    }
    assert_eq!(
        planner.state_seats(),
        Some(3),
        "a fork shares its parent's record until it writes"
    );

    // The pool is held whole: the fork's write parks for its copy.
    let p = planner.clone();
    let parked = crate::rt::spawn(async move { p.acquire(pa, pa, one_slot()).await });
    for _ in 0..50 {
        crate::rt::yield_now().await;
    }
    assert!(!parked.is_finished(), "no set free: the fork's copy waits");

    // One process leaves and lets its set go: the fork is served.
    release(&stores, c);
    planner.unregister(pc);
    planner.pages_freed();
    for _ in 0..50 {
        if parked.is_finished() {
            break;
        }
        crate::rt::yield_now().await;
    }
    assert!(parked.is_finished(), "the freed set serves the fork's copy");
    let Ok(Acquired::Granted(mut grant)) = parked.await.expect("the parked task") else {
        panic!("the freed set serves the fork's copy");
    };
    {
        let mut rs = stores.rs.lock().unwrap();
        let prepared = rs
            .prepare_group(child, true, false, None, None, Some(grant.lend_rs()))
            .unwrap();
        assert!(
            prepared
                .state
                .as_ref()
                .is_some_and(|s| s.copy_from.is_some()),
            "the fork writes a copy of the record its parent still reads"
        );
        let published = rs.publish_prepared(prepared).unwrap();
        rs.settle(published);
    }
    assert_eq!(
        planner.state_seats(),
        Some(2),
        "the copy a fork holds seats no one else"
    );

    // Two charged, two seats: a newcomer waits until the fork lets go.
    let (pd, _rd) = process(&planner, model, &[]);
    let mut newcomer = Box::pin(planner.admit(pd));
    assert!(newcomer.as_mut().now_or_never().is_none());
    release(&stores, child);
    planner.pages_freed();
    assert_eq!(planner.state_seats(), Some(3));
    assert!(
        newcomer.as_mut().now_or_never().is_some(),
        "the fork's copy let go seats the newcomer, with no process leaving"
    );
    let _ = pb;
}

/// A fork whose copy the pool can never hold beside what its process holds
/// is refused by name, not left waiting.
#[tokio::test(flavor = "current_thread")]
async fn a_fork_the_pool_can_never_copy_is_refused_by_name() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[4], &[1]);
    let stores = crate::store::registry::get(model, 0);
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RegistryPool::new(
        model, 0, false,
    ))));
    let a = written(&stores);
    let child = stores.rs.lock().unwrap().fork(a).unwrap();
    let (pa, _ra) = process(&planner, model, &[a, child]);
    assert!(planner.admit(pa).now_or_never().is_some());
    planner.note_admitted(pa);
    let error = crate::rt::time::timeout(
        std::time::Duration::from_secs(60),
        planner.acquire(pa, pa, one_slot()),
    )
    .await
    .expect("refused, not left waiting")
    .err()
    .expect("a one-set pool its parent holds cannot copy the fork");
    assert!(
        matches!(
            error,
            PlannerError::RsHog { need, held, total }
                if need.slots == 1 && held.slots == 1 && total.slots == 1
        ),
        "{error}"
    );
}

/// A state pool whose backed slots rise as the governor's walk maps more,
/// with no set of its own held.
struct GrowingPool {
    slots: AtomicU32,
}

impl PoolPort for GrowingPool {
    fn chunk_stats(&self) -> ChunkStats {
        let slots = self.slots.load(Ordering::SeqCst);
        let units = RsUnits {
            slots,
            ..RsUnits::default()
        };
        ChunkStats {
            pages: (0, 0),
            rs: (units, units),
        }
    }
    fn host_stats(&self) -> (u32, u32) {
        (0, 0)
    }
    fn disk_stats(&self) -> (u32, u32) {
        (0, 0)
    }
    fn rs_host_stats(&self) -> (RsUnits, RsUnits) {
        Default::default()
    }
    fn reclaim_idle(&self) -> u32 {
        0
    }
    fn reclaimable_idle(&self) -> u32 {
        0
    }
    fn idle_below_device(&self) -> (u32, u32) {
        (0, 0)
    }
    fn suspend_capable(&self) -> bool {
        false
    }
    fn locus(&self) -> (usize, usize) {
        (usize::MAX, 0)
    }
    fn reserve_device(&self, _count: u32) -> Option<Vec<PhysicalKvPageId>> {
        None
    }
    fn reserve_device_up_to(&self, _count: u32) -> Vec<PhysicalKvPageId> {
        Vec::new()
    }
    fn release_device(&self, _pages: Vec<PhysicalKvPageId>) {}
    fn reserve_rs(&self, _need: RsUnits) -> Option<crate::store::rs::RsGrant> {
        None
    }
    fn release_rs(&self, _grant: crate::store::rs::RsGrant) {}
    fn reclaim_sharing(&self, _working_sets: &[RsWorkingSetId]) -> usize {
        0
    }
    fn prepare_sweep(
        &self,
        _keep: &std::collections::HashSet<crate::store::kv::page_table::WorkingSetId>,
        _room: usize,
        _order: &[(crate::store::kv::page_table::WorkingSetId, u64)],
        _wanted: usize,
    ) -> Option<crate::store::kv::KvResidencyTxn> {
        None
    }
    fn prepare_demote_now(&self) -> Option<crate::store::kv::KvResidencyTxn> {
        None
    }
    fn drop_disallowed(&self) -> usize {
        0
    }
    fn prepare_demote(&self, _room: usize) -> Option<crate::store::kv::KvResidencyTxn> {
        None
    }
}

/// A process parked at admission is seated once the pool backs more slots,
/// with no process leaving.
#[test]
fn a_process_waiting_for_a_seat_is_seated_when_the_pool_grows() {
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_read();
    let pool = Arc::new(GrowingPool {
        slots: AtomicU32::new(1),
    });
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let (first, second) = (ProcessId::new_v4(), ProcessId::new_v4());
    planner.register(first);
    planner.register(second);
    assert!(planner.admit(first).now_or_never().is_some());
    let mut waiting = Box::pin(planner.admit(second));
    assert!(
        waiting.as_mut().now_or_never().is_none(),
        "one slot, one seat"
    );
    pool.slots.store(2, Ordering::SeqCst);
    planner.pages_freed();
    assert!(
        waiting.as_mut().now_or_never().is_some(),
        "the slot the pool backed seats it"
    );
}
