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
    /// Tiers sweeps that reached the store.
    sweeps: AtomicU32,
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
    fn reclaim_idle(&self, _: super::pool::IdleShort) -> super::pool::IdleReclaim {
        super::pool::IdleReclaim::default()
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
        self.sweeps.fetch_add(1, Ordering::SeqCst);
        0
    }
    fn prepare_demote(&self, _room: usize) -> Option<crate::store::kv::KvResidencyTxn> {
        None
    }
}

/// A tiers sweep asked off the runtime (a table installed on the policy
/// port's thread) is not lost: the next plan, on the runtime, runs it.
#[test]
fn a_tiers_sweep_asked_off_the_runtime_runs_at_the_next_plan() {
    use std::collections::HashMap;
    let _schedule = crate::policy::schedule::TEST_LOCK.blocking_write();
    let pool = Arc::new(GrowingPool {
        slots: AtomicU32::new(1),
        sweeps: AtomicU32::new(0),
    });
    let planner = Arc::new(ResidencyPlanner::new(pool.clone() as Arc<dyn PoolPort>));
    let owner = ProcessId::new_v4();
    crate::policy::retention::install(Some(
        crate::policy::retention::Table::new(HashMap::new(), None)
            .with_tiers(HashMap::from([(owner, vec![(None, 0b010)])]), None),
    ));
    planner.sweep_tiers();
    assert_eq!(
        pool.sweeps.load(Ordering::SeqCst),
        0,
        "no runtime: nothing moved yet"
    );
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { planner.plan() });
    assert_eq!(pool.sweeps.load(Ordering::SeqCst), 1, "the plan ran it");
    crate::policy::retention::install(None);
}

/// A device-only pool (no host tier, no disk tier) serves: a process is
/// seated and given pages.
#[tokio::test(flavor = "current_thread")]
async fn a_pool_with_no_tier_below_the_device_serves() {
    let _schedule = crate::policy::schedule::TEST_LOCK.read().await;
    let model = crate::store::registry::register_model(16, &[8], &[2]);
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RegistryPool::new(
        model, 0, false,
    ))));
    let (pid, _) = process(&planner, model, &[]);
    assert!(planner.admit(pid).now_or_never().is_some(), "seated");
    let demand = super::Demand {
        kv_pages: 4,
        rs: Default::default(),
    };
    let acquired = planner
        .acquire(pid, pid, demand)
        .now_or_never()
        .expect("pages stand free")
        .expect("served");
    assert!(
        matches!(acquired, super::Acquired::Granted(_)),
        "granted on the device alone"
    );
}
