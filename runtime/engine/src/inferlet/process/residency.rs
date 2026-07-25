//! Process-owned membership inventory for KV/RS working sets and fire queues.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex, RwLock, Weak};

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
    pub rs_working_sets: HashSet<(usize, crate::driver::DriverId, RsWorkingSetId)>,
    pub pipelines: Vec<PendingFires>,
    pub departed_pipeline_ids: Vec<uuid::Uuid>,
}

impl ProcessResidency {
    pub(crate) fn snapshot(&mut self) -> ResidencySnapshot {
        let pipelines: Vec<_> = self
            .pipelines
            .iter()
            .filter_map(|pipeline| pipeline.fires.upgrade())
            .collect();
        self.pipelines
            .retain(|pipeline| pipeline.fires.strong_count() > 0);
        ResidencySnapshot {
            kv_working_sets: self.kv_working_sets.clone(),
            rs_working_sets: self.rs_working_sets.clone(),
            pipelines,
            departed_pipeline_ids: Vec::new(),
        }
    }

    pub(crate) fn teardown_snapshot(&mut self) -> ResidencySnapshot {
        let departed_pipeline_ids = self
            .pipelines
            .iter()
            .filter_map(|pipeline| {
                pipeline
                    .scope
                    .close()
                    .then(|| pipeline.scope.scheduler_id())
            })
            .collect();
        let mut snapshot = self.snapshot();
        snapshot.departed_pipeline_ids = departed_pipeline_ids;
        snapshot
    }
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

/// Pages only `pid`'s working sets can free on `(model, driver)` — the
/// contention ladder's victim-cost figure (D6 smallest-cover). `None` when
/// the process is unknown or already tearing down.
pub(crate) fn kv_exclusive_footprint(pid: uuid::Uuid, model: usize, driver: usize) -> Option<u32> {
    let residency = RESIDENCIES.read().unwrap().get(&pid)?.upgrade()?;
    let working_sets: Vec<WorkingSetId> = {
        let residency = residency.lock().unwrap();
        residency
            .kv_working_sets
            .iter()
            .filter_map(|&(m, d, ws)| (m == model && d as usize == driver).then_some(ws))
            .collect()
    };
    if working_sets.is_empty() {
        return Some(0);
    }
    let stores = crate::store::registry::try_get(model, driver)?;
    let total = crate::store::registry::with_kv_lock(&stores.kv, "reclaim", |kv| {
        working_sets
            .iter()
            .map(|&ws| kv.exclusive_footprint(ws).unwrap_or(0))
            .sum::<u64>()
    });
    u32::try_from(total).ok()
}
