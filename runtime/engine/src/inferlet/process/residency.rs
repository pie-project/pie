//! Process-owned membership inventory for KV/RS working sets and fire
//! queues, plus the pid-keyed registry the planner probes through (victim
//! quoting, eviction targeting, restore sizing).

use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex, RwLock, Weak};

use crate::pipeline::fire::{PendingFireQueue, PendingFires};
use crate::store::kv::page_table::{ReclaimQuote, WorkingSetId};
use crate::store::kv::working_set::KvSuspendHandle;
use crate::store::rs::RsWorkingSetId;

type WeakPendingFires = Weak<PendingFireQueue>;

pub(crate) struct ResidentPipeline {
    pub(crate) scope: crate::store::PipelineScope,
    pub(crate) fires: WeakPendingFires,
}

#[derive(Default)]
pub(crate) struct ProcessResidency {
    /// Every live KV working set, keyed by locus — the value is the
    /// planner's weak suspend handle (fence + quiescence).
    pub(crate) kv_working_sets:
        HashMap<(usize, crate::driver::DriverId, WorkingSetId), KvSuspendHandle>,
    pub(crate) rs_working_sets: HashSet<(usize, crate::driver::DriverId, RsWorkingSetId)>,
    pub(crate) pipelines: Vec<ResidentPipeline>,
}

#[derive(Clone)]
pub(crate) struct ResidencySnapshot {
    pub pipelines: Vec<PendingFires>,
    pub departed_pipeline_ids: Vec<uuid::Uuid>,
}

impl ProcessResidency {
    /// The live pipeline queues (pruning entries whose queue is gone).
    pub(crate) fn pipelines(&mut self) -> Vec<PendingFires> {
        let pipelines: Vec<_> = self
            .pipelines
            .iter()
            .filter_map(|pipeline| pipeline.fires.upgrade())
            .collect();
        self.pipelines
            .retain(|pipeline| pipeline.fires.strong_count() > 0);
        pipelines
    }

    pub(crate) fn teardown_snapshot(&mut self) -> ResidencySnapshot {
        let departed_pipeline_ids = self
            .pipelines
            .iter()
            // `close` is the side effect: it claims the departure exactly
            // once, so a second teardown snapshot reports nothing.
            .filter(|pipeline| pipeline.scope.close())
            .map(|pipeline| pipeline.scope.scheduler_id())
            .collect();
        ResidencySnapshot {
            pipelines: self.pipelines(),
            departed_pipeline_ids,
        }
    }
}

/// pid → residency, for cross-layer probes that only know a process id
/// (planner victim sizing, eviction execution). Weak entries; pruned on
/// unregister and on probe misses.
static RESIDENCIES: LazyLock<RwLock<HashMap<uuid::Uuid, Weak<Mutex<ProcessResidency>>>>> =
    LazyLock::new(Default::default);

pub(crate) fn register_residency(pid: uuid::Uuid, residency: Weak<Mutex<ProcessResidency>>) {
    RESIDENCIES.write().unwrap().insert(pid, residency);
}

pub(crate) fn unregister_residency(pid: uuid::Uuid) {
    RESIDENCIES.write().unwrap().remove(&pid);
}

fn with_residency<R: Default>(pid: uuid::Uuid, f: impl FnOnce(&mut ProcessResidency) -> R) -> R {
    let residency = {
        let residencies = RESIDENCIES.read().unwrap();
        residencies.get(&pid).and_then(Weak::upgrade)
    };
    match residency {
        Some(residency) => {
            let mut residency = residency.lock().unwrap();
            f(&mut residency)
        }
        None => R::default(),
    }
}

/// The process's live KV working-set ids on `(model, driver)`.
pub(crate) fn kv_working_set_ids(
    pid: uuid::Uuid,
    model: usize,
    driver: usize,
) -> HashSet<WorkingSetId> {
    with_residency(pid, |residency| {
        residency
            .kv_working_sets
            .keys()
            .filter_map(|&(m, d, ws)| (m == model && d as usize == driver).then_some(ws))
            .collect()
    })
}

/// The planner's suspend handles for the process's KV working sets on
/// `(model, driver)`.
pub(crate) fn kv_suspend_handles(
    pid: uuid::Uuid,
    model: usize,
    driver: usize,
) -> Vec<KvSuspendHandle> {
    with_residency(pid, |residency| {
        residency
            .kv_working_sets
            .iter()
            .filter(|((m, d, _), _)| *m == model && *d as usize == driver)
            .map(|(_, handle)| handle.clone())
            .collect()
    })
}

/// The process's live pipeline FIFOs (for the planner's detachable drain).
pub(crate) fn pipelines_of(pid: uuid::Uuid) -> Vec<PendingFires> {
    with_residency(pid, |residency| residency.pipelines())
}

/// Whether every KV working set of `pid` on `(model, driver)` holds ZERO
/// fire leases right now — a racy snapshot, used only as a victim-selection
/// PREFERENCE: evicting a lease-quiescent process skips the measured
/// ~2-frame lease drain (the p50 9.1 ms of the evict chain,
/// CONTENTION_FOLLOWUP.md §17), so the planner prefers such victims among
/// the equally-eligible. Never a correctness gate — the eviction's own
/// fence + quiesce remains the seal.
pub(crate) fn kv_lease_quiescent(pid: uuid::Uuid, model: usize, driver: usize) -> bool {
    with_residency(pid, |residency| {
        residency
            .kv_working_sets
            .iter()
            .filter(|((m, d, _), _)| *m == model && *d as usize == driver)
            .all(|(_, handle)| handle.active_leases() == 0)
    })
}

/// What suspending each candidate would ACTUALLY free on `(model, driver)`,
/// computed for the whole candidate set under ONE store lock. This is the
/// same question `prepare_suspend` answers, so victim selection and eviction
/// execution cannot disagree; a zero answer carries the reason.
///
/// The per-process KV working sets on `(model, driver)`, gathered WITHOUT
/// touching the KV store lock.
///
/// Split out of [`kv_reclaim_quotes`] so a caller that needs an atomic
/// decision can gather here first and then take the store lock itself:
/// `RESIDENCIES` is acquired *before* the KV lock everywhere in the tree,
/// and taking them the other way round would invert that order.
pub(crate) fn kv_working_sets_for(
    pids: &[uuid::Uuid],
    model: usize,
    driver: usize,
) -> Vec<Option<HashSet<WorkingSetId>>> {
    let residencies = RESIDENCIES.read().unwrap();
    pids.iter()
        .map(|pid| {
            let residency = residencies.get(pid)?.upgrade()?;
            let residency = residency.lock().unwrap();
            Some(
                residency
                    .kv_working_sets
                    .keys()
                    .filter_map(|&(m, d, ws)| (m == model && d as usize == driver).then_some(ws))
                    .collect(),
            )
        })
        .collect()
}

/// Quote `working_sets` against an ALREADY-LOCKED store, keeping the result
/// positional with the `pids` the sets came from.
///
/// `budget` stops the quoting once the answers cover that many pages; the
/// positions past the cut come back `None` ("no opinion"), which is what a
/// caller that consumes the list in preference order until its deficit is
/// covered would have ignored anyway. Pass `u32::MAX` to quote every pid.
pub(crate) fn quote_locked(
    kv: &crate::store::kv::KvStore,
    working_sets: Vec<Option<HashSet<WorkingSetId>>>,
    budget: u32,
) -> Vec<Option<ReclaimQuote>> {
    let known: Vec<HashSet<WorkingSetId>> = working_sets.iter().flatten().cloned().collect();
    let mut quotes = kv.reclaim_quotes(&known, budget).into_iter();
    // `and_then`, not `and`: `Option::and` evaluates its argument eagerly, so
    // a `None` entry would still consume a quote and shift every later
    // position by one. Only known processes were quoted, so only they may
    // take from the iterator.
    working_sets
        .into_iter()
        .map(|entry| entry.and_then(|_| quotes.next()))
        .collect()
}

/// `None` for a process that is unknown or already tearing down — that is
/// "no opinion", distinct from a quote of "frees nothing".
pub(crate) fn kv_reclaim_quotes(
    pids: &[uuid::Uuid],
    model: usize,
    driver: usize,
    budget: u32,
) -> Vec<Option<ReclaimQuote>> {
    let working_sets = kv_working_sets_for(pids, model, driver);
    let Some(stores) = crate::store::registry::try_get(model, driver) else {
        return vec![None; pids.len()];
    };
    crate::store::registry::with_kv_lock(&stores.kv, "planner-quotes", |kv| {
        quote_locked(kv, working_sets, budget)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{ProcessResidency, ResidentPipeline};

    #[test]
    fn teardown_closes_each_orphan_pipeline_once() {
        let pipeline = crate::pipeline::Pipeline::new();
        let pipeline_id = pipeline.scope.scheduler_id();
        let mut residency = ProcessResidency::default();
        residency.pipelines.push(ResidentPipeline {
            scope: pipeline.scope.clone(),
            fires: Arc::downgrade(&pipeline.fires),
        });

        let first = residency.teardown_snapshot();
        assert_eq!(first.departed_pipeline_ids, vec![pipeline_id]);
        assert!(pipeline.scope.is_closed());

        let second = residency.teardown_snapshot();
        assert!(second.departed_pipeline_ids.is_empty());
    }
}
