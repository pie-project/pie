//! The state pool's seats: one fixed set a process, none kept back for the
//! lanes a fire may run; a fork's copy-on-write set is asked for by the fire
//! that writes it, and seats no one else while it is held.

use std::sync::Arc;

use futures::FutureExt;

use crate::inferlet::process::residency::{ProcessResidency, register_residency};
use crate::store::registry::Stores;
use crate::store::rs::{RsGeometry, RsUnits, RsWorkingSetId};

use super::grant::Demand;
use super::pool::RegistryPool;
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

/// A device-only pool (no host tier, no disk tier) serves: a process is
/// seated and given pages. Persisting a run there is refused by name: there
/// is no disk tier for a copy to outlive the run in.
#[tokio::test(flavor = "current_thread")]
async fn a_pool_with_no_tier_below_the_device_serves_and_refuses_persisting_by_name() {
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
    assert!(matches!(acquired, super::Acquired::Granted(_)), "granted on the device alone");
    let refused = super::exec::persist(model, 0, Vec::new(), 1)
        .await
        .expect_err("no disk tier");
    assert!(refused.contains("no disk tier"), "{refused}");
}
