//! Bound instances and their wait slots — the other half of the broker lift.
//!
//! **THIS CODE IS A LIFT, NOT A REWRITE**, on the same terms as
//! [`completion`](crate::driver::completion): it stood in
//! `driver-api::instance` and it is engine bookkeeping, not a statement about
//! what a driver is. It arrived comment-stripped and stays that way; what is
//! documented is what this wave changed.
//!
//! # What changed on the way over
//!
//! * [`InstanceBindingPlan`] is now the engine's THREE fields plus the
//!   contract's [`InstanceBinding`](driver_api::InstanceBinding). The old
//!   struct carried `driver_id`, `pacing_wait_id` and
//!   `requested_instance_id` *through* the driver so they could come back
//!   unchanged; the driver mints the id and the engine keeps its own tables
//!   (`driver-api::program`'s note on `InstanceBinding`). So the plan holds
//!   the engine's half and hands the driver only `binding`.
//! * `validate_binding` no longer re-checks a native struct's fields. The
//!   driver answers a typed
//!   [`BoundInstance`](driver_api::BoundInstance) whose geometry class is an
//!   enum; the one thing left to say is whether it acknowledged the class
//!   that was asked for.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use driver_api::channel::ChannelSeed;
use driver_api::program::{BindExtents, InstanceBinding};
use tensor_ir::registry::GeometryClass;

/// A registered program's id, as the driver minted it.
pub type ProgramId = driver_api::ProgramId;
/// A bound instance's id, as the driver minted it.
pub type InstanceId = driver_api::InstanceId;

/// One instance binding, engine bookkeeping and contract argument together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstanceBindingPlan {
    /// Which driver in the registry.
    pub driver_id: usize,
    /// The wait slot this instance's pacing parks on.
    pub pacing_wait_id: u64,
    /// What the driver is asked to bind.
    pub binding: InstanceBinding,
}

impl InstanceBindingPlan {
    /// The plan that binds `program` to `channels`.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "the contract's five fields plus the engine's two; every one of \
                  them is stated by a different party at the bind site and none \
                  has a group it obviously belongs to"
    )]
    pub fn new(
        driver_id: usize,
        pacing_wait_id: u64,
        program: ProgramId,
        channels: Vec<u64>,
        seeds: Vec<ChannelSeed>,
        geometry: GeometryClass,
        extents: BindExtents,
    ) -> Self {
        Self {
            driver_id,
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

    /// Which program this instantiates.
    #[must_use]
    pub fn program_id(&self) -> ProgramId {
        self.binding.program
    }

    /// The class this binding asked for.
    #[must_use]
    pub fn geometry_class(&self) -> GeometryClass {
        self.binding.geometry
    }

    /// Did the driver acknowledge the class that was asked for?
    ///
    /// # Errors
    ///
    /// When it bound a different one — which means the driver resolves a
    /// different amount of the fire geometry on the device than the engine
    /// staged for, and every fire after would read a descriptor nobody wrote.
    pub fn validate_binding(&self, bound: &driver_api::BoundInstance) -> anyhow::Result<()> {
        anyhow::ensure!(
            bound.geometry == self.binding.geometry,
            "driver acknowledged geometry class {:?} for a binding that asked for {:?}",
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
    pub driver_id: usize,
    pub program_id: ProgramId,
    pub instance_id: InstanceId,
    pub pacing_wait_id: u64,
    pub geometry_class: GeometryClass,
    wait_slots: Arc<BoundWaitSlots>,
}

impl BoundInstance {
    /// Wrap the driver's answer in the engine's bookkeeping.
    ///
    /// Takes the contract's [`BoundInstance`](driver_api::BoundInstance)
    /// whole, where the lifted version took a `#[repr(C)]` binding struct and
    /// re-derived a geometry class out of a `u32` it had already validated.
    #[must_use]
    pub fn new(
        driver_id: usize,
        bound: &driver_api::BoundInstance,
        pacing_wait_id: u64,
    ) -> Self {
        Self {
            driver_id,
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
