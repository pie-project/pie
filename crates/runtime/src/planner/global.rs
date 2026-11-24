use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use super::{ProcessId, ResidencyPlanner};

type PlannerMap = HashMap<(usize, usize), Arc<ResidencyPlanner>>;

static PRIMARY: crate::slot::Slot<Arc<ResidencyPlanner>> = crate::slot::Slot::new();
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

/// Puts `planner` as `(model, engine)`'s, with no drain task: a test plans
/// its passes itself. `None` takes it off.
#[cfg(test)]
pub(crate) fn set_test_planner(
    model: usize,
    engine: usize,
    planner: Option<Arc<ResidencyPlanner>>,
) {
    let mut planners = PLANNERS.get_or_init(Default::default).write().unwrap();
    match planner {
        Some(planner) => planners.insert((model, engine), planner),
        None => planners.remove(&(model, engine)),
    };
}

pub fn planner_for(model: usize, engine: usize) -> Option<Arc<ResidencyPlanner>> {
    if model == 0 && engine == 0 {
        return PRIMARY.get();
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
    if let Some(planner) = planner_of(pid) {
        planner.wake(pid);
    }
}

pub fn planner() -> Option<Arc<ResidencyPlanner>> {
    PRIMARY.get()
}

/// The engine each live process runs on, where a model runs on several (the
/// ranks of an expert-parallel worker); a process not here runs on engine 0.
static ENGINE_OF: OnceLock<RwLock<HashMap<ProcessId, usize>>> = OnceLock::new();

fn engines() -> &'static RwLock<HashMap<ProcessId, usize>> {
    ENGINE_OF.get_or_init(Default::default)
}

/// The engines of model 0 that have a planner, ascending.
fn planned() -> Vec<usize> {
    let mut out: Vec<usize> = PLANNERS
        .get()
        .map(|p| {
            p.read()
                .unwrap()
                .keys()
                .filter(|(m, _)| *m == 0)
                .map(|(_, e)| *e)
                .collect()
        })
        .unwrap_or_default();
    out.sort_unstable();
    out
}

/// Places `pid` on an engine: its parent's where it has one (a restart or a
/// fork keeps its kv where it was), else the engine running the fewest
/// processes. With one engine, engine 0.
pub fn assign(pid: ProcessId, parent: Option<ProcessId>) -> usize {
    let engines_of = engines();
    let mut held = engines_of.write().unwrap();
    let engine = match parent.and_then(|p| held.get(&p).copied()) {
        Some(engine) => engine,
        None => fewest(&planned(), held.values().copied()),
    };
    if engine != 0 || planned().len() > 1 {
        held.insert(pid, engine);
    }
    engine
}

/// Of `engines`, the one running the fewest of `running` (each a process's
/// engine), the lowest on a tie; engine 0 where there are none.
fn fewest(engines: &[usize], running: impl Iterator<Item = usize> + Clone) -> usize {
    engines
        .iter()
        .copied()
        .min_by_key(|e| (running.clone().filter(|x| x == e).count(), *e))
        .unwrap_or(0)
}

/// Puts `pid` on `engine`, where its working sets already are.
pub fn place(pid: ProcessId, engine: usize) {
    engines().write().unwrap().insert(pid, engine);
}

/// The engine `pid` runs on.
pub fn engine_of(pid: ProcessId) -> usize {
    ENGINE_OF
        .get()
        .and_then(|held| held.read().unwrap().get(&pid).copied())
        .unwrap_or(0)
}

/// Forgets where `pid` ran, once it is gone.
pub fn release(pid: ProcessId) {
    if let Some(held) = ENGINE_OF.get() {
        held.write().unwrap().remove(&pid);
    }
}

/// The planner of the engine `pid` runs on.
pub fn planner_of(pid: ProcessId) -> Option<Arc<ResidencyPlanner>> {
    planner_for(0, engine_of(pid))
}

/// Reads holders' retention keys again on every planner: those `only`
/// names (by PLEX id), or all.
pub fn rekey_holders(only: Option<&std::collections::HashSet<ProcessId>>) {
    for planner in planners() {
        planner.rekey_holders(only);
    }
}

/// The worker's `pie/lanes-to-end` and `pie/starts-spread`: the lanes
/// every engine's pools hold to their end at once, summed (`None` until
/// learned), and whether starting together runs short: every engine's pool
/// has no host tier, so a lane run short restarts its process rather than
/// park below. Runtime facts, read from the pools and what finished; the
/// engines state no flag for them.
pub fn to_end_facts() -> (Option<u64>, bool) {
    let facts: Vec<(Option<u64>, bool)> = planners().iter().map(|p| p.to_end_facts()).collect();
    if facts.is_empty() {
        return (None, false);
    }
    let lanes = facts
        .iter()
        .map(|(lanes, _)| *lanes)
        .try_fold(0u64, |sum, lanes| lanes.map(|lanes| sum.saturating_add(lanes)));
    (lanes, facts.iter().all(|&(_, spread)| spread))
}

/// Forgets the boot's planners and where its processes ran, for
/// [`crate::bootstrap::reset`].
pub(crate) fn reset() {
    PRIMARY.clear();
    if let Some(planners) = PLANNERS.get() {
        planners.write().unwrap().clear();
    }
    if let Some(held) = ENGINE_OF.get() {
        held.write().unwrap().clear();
    }
}

/// Every engine's planner, engine 0's first: for what frees pages or
/// changes the order on all of them.
pub fn planners() -> Vec<Arc<ResidencyPlanner>> {
    planned()
        .into_iter()
        .filter_map(|e| planner_for(0, e))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::fewest;

    #[test]
    fn a_process_goes_to_the_engine_running_the_fewest() {
        assert_eq!(fewest(&[], [].into_iter()), 0);
        assert_eq!(fewest(&[0], [0, 0].into_iter()), 0);
        assert_eq!(fewest(&[0, 1, 2], [].into_iter()), 0);
        assert_eq!(fewest(&[0, 1, 2], [0, 1].into_iter()), 2);
        assert_eq!(fewest(&[0, 1, 2], [0, 2, 0].into_iter()), 1);
    }
}
