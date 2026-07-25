//! Process-owned membership inventory for KV/RS working sets and fire queues.

use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex, RwLock, Weak};

use crate::pipeline::fire::{PendingFireQueue, PendingFires};
use crate::store::kv::page_table::WorkingSetId;
use crate::store::rs::RsWorkingSetId;

type WeakPendingFires = Weak<PendingFireQueue>;

pub(crate) struct ResidentPipeline {
    pub(crate) scope: crate::store::PipelineScope,
    pub(crate) fires: WeakPendingFires,
}

#[derive(Default)]
pub(crate) struct ProcessResidency {
    pub(crate) kv_working_sets: HashSet<(usize, crate::driver::DriverId, WorkingSetId)>,
    pub(crate) rs_working_sets: HashSet<(usize, crate::driver::DriverId, RsWorkingSetId)>,
    pub(crate) pipelines: Vec<ResidentPipeline>,
}

#[derive(Clone)]
pub(crate) struct ResidencySnapshot {
    pub kv_working_sets: HashSet<(usize, crate::driver::DriverId, WorkingSetId)>,
    pub pipelines: Vec<PendingFires>,
    pub departed_pipeline_ids: Vec<uuid::Uuid>,
}

impl ProcessResidency {
    /// Just the live pipeline queues — the hot-path subset of
    /// [`Self::snapshot`], skipping the working-set clones.
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

    pub(crate) fn snapshot(&mut self) -> ResidencySnapshot {
        ResidencySnapshot {
            pipelines: self.pipelines(),
            kv_working_sets: self.kv_working_sets.clone(),
            departed_pipeline_ids: Vec::new(),
        }
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
        let mut snapshot = self.snapshot();
        snapshot.departed_pipeline_ids = departed_pipeline_ids;
        snapshot
    }
}

/// pid → residency, for cross-layer probes that only know a process id
/// (victim footprint sizing). Weak entries; pruned on unregister and on
/// probe misses.
static RESIDENCIES: LazyLock<RwLock<HashMap<uuid::Uuid, Weak<Mutex<ProcessResidency>>>>> =
    LazyLock::new(Default::default);

pub(crate) fn register_residency(pid: uuid::Uuid, residency: Weak<Mutex<ProcessResidency>>) {
    RESIDENCIES.write().unwrap().insert(pid, residency);
}

pub(crate) fn unregister_residency(pid: uuid::Uuid) {
    RESIDENCIES.write().unwrap().remove(&pid);
}

/// Pages only each candidate's working sets can free on `(model, driver)`
/// — the contention ladder's victim-cost figures (D6 smallest-cover),
/// computed for the whole candidate set under ONE store lock. `None` for a
/// process that is unknown or already tearing down.
pub(crate) fn kv_exclusive_footprints(
    pids: &[uuid::Uuid],
    model: usize,
    driver: usize,
) -> Vec<Option<u32>> {
    let working_sets: Vec<Option<Vec<WorkingSetId>>> = {
        let residencies = RESIDENCIES.read().unwrap();
        pids.iter()
            .map(|pid| {
                let residency = residencies.get(pid)?.upgrade()?;
                let residency = residency.lock().unwrap();
                Some(
                    residency
                        .kv_working_sets
                        .iter()
                        .filter_map(|&(m, d, ws)| (m == model && d == driver).then_some(ws))
                        .collect(),
                )
            })
            .collect()
    };
    let Some(stores) = crate::store::registry::try_get(model, driver) else {
        return vec![None; pids.len()];
    };
    crate::store::registry::with_kv_lock(&stores.kv, "reclaim", |kv| {
        working_sets
            .into_iter()
            .map(|working_sets| {
                let working_sets = working_sets?;
                let total: u64 = working_sets
                    .iter()
                    .map(|&ws| kv.exclusive_footprint(ws).unwrap_or(0))
                    .sum();
                u32::try_from(total).ok()
            })
            .collect()
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
