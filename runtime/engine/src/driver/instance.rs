//! Instance binding: the plan a scheduler sends to bind an instance
//! ([`InstanceBindingPlan`]), the backend's owned handle to it
//! ([`BoundInstance`]), and the wait-slot/completion-lease lifecycle that
//! keeps its pacing wait id alive until every in-flight completion lease
//! has dropped.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pie_driver_abi::{GeometryClass, PieInstanceBinding};

use super::channel::ChannelValue;

pub type ProgramId = u64;
pub type InstanceId = u64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceBindingPlan {
    pub driver_id: usize,
    pub program_id: ProgramId,
    pub requested_instance_id: InstanceId,
    pub pacing_wait_id: u64,
    pub channel_ids: Vec<u64>,
    pub seed_values: Vec<ChannelValue>,
    pub geometry_class: GeometryClass,
}

impl InstanceBindingPlan {
    pub(crate) fn validate_binding(&self, binding: &PieInstanceBinding) -> anyhow::Result<()> {
        pie_driver_abi::validate_instance_binding(binding)
            .map_err(|err| anyhow::anyhow!("invalid native instance binding: {err}"))?;
        if self.requested_instance_id != 0 {
            anyhow::ensure!(
                binding.instance_id == self.requested_instance_id,
                "native binding returned instance {} for requested {}",
                binding.instance_id,
                self.requested_instance_id
            );
        }
        anyhow::ensure!(
            binding.geometry_class == self.geometry_class as u32,
            "native binding acknowledged geometry class {}, expected {:?}",
            binding.geometry_class,
            self.geometry_class
        );
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct BoundWaitSlots {
    pacing_wait_id: u64,
    completion_wait_ids: Mutex<Vec<u64>>,
    close_requested: AtomicBool,
    freed: AtomicBool,
    active_leases: AtomicUsize,
}

impl BoundWaitSlots {
    fn new(pacing_wait_id: u64) -> Self {
        Self {
            pacing_wait_id,
            completion_wait_ids: Mutex::new(Vec::new()),
            close_requested: AtomicBool::new(false),
            freed: AtomicBool::new(false),
            active_leases: AtomicUsize::new(0),
        }
    }

    fn acquire_completion_lease(
        this: &Arc<Self>,
        completion_wait_id: u64,
    ) -> Arc<dyn crate::driver::completion::CompletionLease> {
        if this.close_requested.load(Ordering::Acquire) {
            return Arc::new(BoundWaitLease {
                slots: Arc::clone(this),
                completion_wait_id,
                active: false,
            });
        }
        this.completion_wait_ids
            .lock()
            .unwrap()
            .push(completion_wait_id);
        this.active_leases.fetch_add(1, Ordering::AcqRel);
        if this.close_requested.load(Ordering::Acquire) {
            this.release_completion_lease_for(completion_wait_id);
            return Arc::new(BoundWaitLease {
                slots: Arc::clone(this),
                completion_wait_id,
                active: false,
            });
        }
        Arc::new(BoundWaitLease {
            slots: Arc::clone(this),
            completion_wait_id,
            active: true,
        })
    }

    pub(crate) fn close(&self) {
        if !self.close_requested.swap(true, Ordering::AcqRel) {
            pie_waker::WakerTable::global().sweep(&self.wait_ids());
            let completion_wait_ids = self.completion_wait_ids.lock().unwrap().clone();
            pie_waker::WakerTable::global().sweep(&completion_wait_ids);
            self.maybe_finalize();
        }
    }

    fn release_completion_lease_for(&self, completion_wait_id: u64) {
        self.completion_wait_ids
            .lock()
            .unwrap()
            .retain(|&id| id != completion_wait_id);
        let prev = self.active_leases.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prev > 0);
        if prev == 1 {
            self.maybe_finalize();
        }
    }

    fn maybe_finalize(&self) {
        if !self.close_requested.load(Ordering::Acquire)
            || self.active_leases.load(Ordering::Acquire) != 0
            || self.freed.swap(true, Ordering::AcqRel)
        {
            return;
        }
        let table = pie_waker::WakerTable::global();
        for id in self.wait_ids() {
            table.deregister(id);
            table.free(id);
        }
    }

    fn wait_ids(&self) -> Vec<u64> {
        vec![self.pacing_wait_id]
    }

    fn is_closed(&self) -> bool {
        self.close_requested.load(Ordering::Acquire)
    }
}

impl crate::driver::completion::CompletionLease for BoundWaitLease {
    fn is_closed(&self) -> bool {
        self.slots.is_closed()
    }
}

#[derive(Debug)]
struct BoundWaitLease {
    slots: Arc<BoundWaitSlots>,
    completion_wait_id: u64,
    active: bool,
}

impl Drop for BoundWaitLease {
    fn drop(&mut self) {
        if self.active {
            self.slots
                .release_completion_lease_for(self.completion_wait_id);
        }
    }
}

#[derive(Debug)]
pub struct BoundInstance {
    pub driver_id: usize,
    pub program_id: ProgramId,
    pub instance_id: InstanceId,
    pub pacing_wait_id: u64,
    pub geometry_class: GeometryClass,
    wait_slots: Arc<BoundWaitSlots>,
}

impl BoundInstance {
    pub fn new(
        driver_id: usize,
        program_id: ProgramId,
        binding: PieInstanceBinding,
        pacing_wait_id: u64,
    ) -> Self {
        let wait_slots = Arc::new(BoundWaitSlots::new(pacing_wait_id));
        Self {
            driver_id,
            program_id,
            instance_id: binding.instance_id,
            pacing_wait_id,
            geometry_class: GeometryClass::try_from(binding.geometry_class)
                .expect("validated geometry class"),
            wait_slots,
        }
    }

    pub fn reserve_completion(&self) -> crate::driver::completion::WorkItemCompletion {
        let wait_id = pie_waker::WakerTable::global().alloc();
        crate::driver::completion::WorkItemCompletion::with_guard(
            wait_id,
            0,
            BoundWaitSlots::acquire_completion_lease(&self.wait_slots, wait_id),
        )
    }

    pub(crate) fn wait_slots(&self) -> Arc<BoundWaitSlots> {
        Arc::clone(&self.wait_slots)
    }

    pub fn close_wait_slots(&self) {
        self.wait_slots.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding_plan(requested_instance_id: u64) -> InstanceBindingPlan {
        InstanceBindingPlan {
            driver_id: 0,
            program_id: 1,
            requested_instance_id,
            pacing_wait_id: 11,
            channel_ids: vec![101],
            seed_values: Vec::new(),
            geometry_class: GeometryClass::Host,
        }
    }

    #[test]
    fn accepts_driver_or_requested_identity() {
        binding_plan(0)
            .validate_binding(&PieInstanceBinding {
                instance_id: 9,
                ..PieInstanceBinding::default()
            })
            .unwrap();
        binding_plan(7)
            .validate_binding(&PieInstanceBinding {
                instance_id: 7,
                ..PieInstanceBinding::default()
            })
            .unwrap();
    }

    #[test]
    fn rejects_zero_or_mismatched_identity() {
        assert!(
            binding_plan(0)
                .validate_binding(&PieInstanceBinding::default())
                .is_err()
        );
        assert!(
            binding_plan(7)
                .validate_binding(&PieInstanceBinding {
                    instance_id: 8,
                    ..PieInstanceBinding::default()
                })
                .is_err()
        );
    }

    #[test]
    fn rejects_mismatched_geometry_ack() {
        let mut plan = binding_plan(0);
        plan.geometry_class = GeometryClass::DecodeEnvelope;
        assert!(
            plan.validate_binding(&PieInstanceBinding {
                instance_id: 9,
                geometry_class: GeometryClass::Host as u32,
                reserved0: 0,
            })
            .is_err()
        );
    }
}
