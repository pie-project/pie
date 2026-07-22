//! Process-owned membership inventory for KV/RS working sets and fire queues.

use std::collections::HashSet;
use std::sync::Weak;

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
