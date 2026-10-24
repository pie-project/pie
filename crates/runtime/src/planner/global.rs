use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use super::{ProcessId, ResidencyPlanner};

type PlannerMap = HashMap<(usize, usize), Arc<ResidencyPlanner>>;

static PRIMARY: OnceLock<Arc<ResidencyPlanner>> = OnceLock::new();
static PLANNERS: OnceLock<RwLock<PlannerMap>> = OnceLock::new();

pub fn init_planner(model: usize, engine: usize, planner: ResidencyPlanner) {
    let planner = Arc::new(planner);
    planner.arm_drain_task();
    if model == 0 && engine == 0 {
        let _ = PRIMARY.set(planner.clone());
    }
    PLANNERS
        .get_or_init(Default::default)
        .write()
        .unwrap()
        .insert((model, engine), planner);
}

/// Persists the sealed run `keys` (from token 0) on the disk tier of the
/// engine that holds most of it, as far as it is held there on the device:
/// the pages stay where they are, and a restart takes the copies back. They
/// are kept under the cluster `retention` key: swap takes the lowest first.
/// Answers the pages persisted.
pub async fn persist_run(
    keys: Vec<crate::store::kv::hash::Hash256>,
    retention: i64,
) -> Result<usize, String> {
    let stores = crate::store::registry::all_for_model(0);
    let engine = stores
        .iter()
        .enumerate()
        .map(|(engine, stores)| {
            let held = crate::store::registry::with_kv_lock(&stores.kv, "persist", |kv| {
                kv.cached_run_any_tier(&keys).len()
            });
            (held, engine)
        })
        .max()
        .filter(|&(held, _)| held > 0)
        .map(|(_, engine)| engine)
        .ok_or("the run is not held here")?;
    super::exec::persist(0, engine, keys, retention).await
}

pub fn planner_for(model: usize, engine: usize) -> Option<Arc<ResidencyPlanner>> {
    if model == 0 && engine == 0 {
        return PRIMARY.get().cloned();
    }
    PLANNERS
        .get()?
        .read()
        .unwrap()
        .get(&(model, engine))
        .cloned()
}

/// A message for `pid` is on its way: if it was parked, its restore starts
/// now rather than when it reads the message.
pub fn wake(pid: ProcessId) {
    if let Some(planner) = planner() {
        planner.wake(pid);
    }
}

pub fn planner() -> Option<&'static Arc<ResidencyPlanner>> {
    PRIMARY.get()
}
