//! Process-owned membership inventory for KV/RS working sets and fire queues.

use std::collections::{HashSet, VecDeque};
use std::sync::{Mutex, Weak};

use crate::pipeline::fire::{PendingFires, PendingOp};
use crate::store::kv::page_table::WorkingSetId;
use crate::store::rs::RsWorkingSetId;

type WeakPendingFires = Weak<Mutex<VecDeque<PendingOp>>>;

#[derive(Default)]
pub(crate) struct ProcessResidency {
    pub(crate) kv_working_sets: HashSet<(usize, crate::driver::DriverId, WorkingSetId)>,
    pub(crate) rs_working_sets: HashSet<(usize, crate::driver::DriverId, RsWorkingSetId)>,
    pub(crate) pipelines: Vec<WeakPendingFires>,
}

#[derive(Clone)]
pub(crate) struct ResidencySnapshot {
    pub kv_working_sets: HashSet<(usize, crate::driver::DriverId, WorkingSetId)>,
    pub rs_working_sets: HashSet<(usize, crate::driver::DriverId, RsWorkingSetId)>,
    pub pipelines: Vec<PendingFires>,
}

impl ProcessResidency {
    pub(crate) fn snapshot(&mut self) -> ResidencySnapshot {
        let pipelines: Vec<_> = self.pipelines.iter().filter_map(Weak::upgrade).collect();
        self.pipelines
            .retain(|pipeline| pipeline.strong_count() > 0);
        ResidencySnapshot {
            kv_working_sets: self.kv_working_sets.clone(),
            rs_working_sets: self.rs_working_sets.clone(),
            pipelines,
        }
    }
}
