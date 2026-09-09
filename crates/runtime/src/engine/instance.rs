use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use engine::channel::ChannelSeed;
use engine::program::{BindExtents, InstanceBinding};
use eta_ir::registry::GeometryClass;

pub type ProgramId = engine::ProgramId;
pub type InstanceId = engine::InstanceId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceBindingPlan {
    pub engine_id: usize,
    pub pacing_wait_id: u64,
    pub binding: InstanceBinding,
}

impl InstanceBindingPlan {
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "the contract's five fields plus the runtime's two; every one of \
                  them is stated by a different party at the bind site and none \
                  has a group it obviously belongs to"
    )]
    pub fn new(
        engine_id: usize,
        pacing_wait_id: u64,
        program: ProgramId,
        channels: Vec<u64>,
        seeds: Vec<ChannelSeed>,
        geometry: GeometryClass,
        extents: BindExtents,
    ) -> Self {
        Self {
            engine_id,
            pacing_wait_id,
            binding: InstanceBinding {
                program,
                channels,
                seeds,
                geometry,
                extents,
            },
        }
    }

    #[must_use]
    pub fn program_id(&self) -> ProgramId {
        self.binding.program
    }

    #[must_use]
    pub fn geometry_class(&self) -> GeometryClass {
        self.binding.geometry
    }

    pub fn validate_binding(&self, bound: &engine::BoundInstance) -> anyhow::Result<()> {
        anyhow::ensure!(
            bound.geometry == self.binding.geometry,
            "engine acknowledged geometry class {:?} for a binding that asked for {:?}",
            bound.geometry,
            self.binding.geometry
        );
        Ok(())
    }
}

#[derive(Debug)]
pub struct BoundWaitSlots {
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
    ) -> Arc<dyn super::completion::CompletionLease> {
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

    pub fn close(&self) {
        if !self.close_requested.swap(true, Ordering::AcqRel) {
            waker::WakerTable::global().sweep(&self.wait_ids());
            let completion_wait_ids = self.completion_wait_ids.lock().unwrap().clone();
            waker::WakerTable::global().sweep(&completion_wait_ids);
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
        let table = waker::WakerTable::global();
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

impl super::completion::CompletionLease for BoundWaitLease {
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
    pub engine_id: usize,
    pub program_id: ProgramId,
    pub instance_id: InstanceId,
    pub pacing_wait_id: u64,
    pub geometry_class: GeometryClass,
    wait_slots: Arc<BoundWaitSlots>,
}

impl BoundInstance {
    #[must_use]
    pub fn new(engine_id: usize, bound: &engine::BoundInstance, pacing_wait_id: u64) -> Self {
        Self {
            engine_id,
            program_id: bound.program,
            instance_id: bound.id,
            pacing_wait_id,
            geometry_class: bound.geometry,
            wait_slots: Arc::new(BoundWaitSlots::new(pacing_wait_id)),
        }
    }

    pub fn reserve_completion(&self) -> super::completion::WorkItemCompletion {
        let wait_id = waker::WakerTable::global().alloc();
        super::completion::WorkItemCompletion::with_guard(
            wait_id,
            0,
            BoundWaitSlots::acquire_completion_lease(&self.wait_slots, wait_id),
        )
    }

    pub fn wait_slots(&self) -> Arc<BoundWaitSlots> {
        Arc::clone(&self.wait_slots)
    }

    pub fn close_wait_slots(&self) {
        self.wait_slots.close();
    }
}
