use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use super::page_table::WorkingSetId;
use crate::engine::EngineId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FireLeaseError {
    Released,
    Fenced,
}

impl std::fmt::Display for FireLeaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FireLeaseError::Released => f.write_str("working set release already requested"),
            FireLeaseError::Fenced => f.write_str("working set is suspend-fenced"),
        }
    }
}

#[derive(Debug)]
struct KvLifecycle {
    released: AtomicBool,
    release_requested: AtomicBool,
    active_fire_leases: AtomicUsize,
    suspend_fence: AtomicBool,
    quiesced: tokio::sync::Notify,
    model: usize,
    engine: EngineId,
    id: WorkingSetId,
    pipeline_scope: Mutex<Option<crate::store::PipelineScope>>,
}

impl KvLifecycle {
    fn release(&self) {
        self.release_requested.store(true, Ordering::Release);
        self.maybe_release();
    }

    fn maybe_release(&self) {
        if !self.release_requested.load(Ordering::Acquire)
            || self.active_fire_leases.load(Ordering::Acquire) != 0
            || self.released.swap(true, Ordering::AcqRel)
        {
            return;
        }
        let stores = crate::store::registry::get(self.model, self.engine);
        crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            let epoch = kv.current_epoch();
            kv.release_working_set(self.id, epoch);
            kv.retire_idle();
        });
        stores.seats.lock().unwrap().release(self.id);
        stores.seats_freed.notify_waiters();
        if let Some(planner) = crate::planner::planner_for(self.model, self.engine) {
            planner.working_set_released(self.id);
            planner.pages_freed();
        }
    }

    fn acquire_fire_lease(this: &Arc<Self>) -> Result<KvFireLease, FireLeaseError> {
        if this.release_requested.load(Ordering::Acquire) {
            return Err(FireLeaseError::Released);
        }
        this.active_fire_leases.fetch_add(1, Ordering::SeqCst);
        let refused = if this.release_requested.load(Ordering::Acquire) {
            Some(FireLeaseError::Released)
        } else if this.suspend_fence.load(Ordering::SeqCst) {
            Some(FireLeaseError::Fenced)
        } else {
            None
        };
        if let Some(refusal) = refused {
            let previous = this.active_fire_leases.fetch_sub(1, Ordering::SeqCst);
            debug_assert!(previous > 0);
            this.maybe_release();
            if previous == 1 {
                this.quiesced.notify_waiters();
            }
            return Err(refusal);
        }
        Ok(KvFireLease {
            lifecycle: Arc::clone(this),
        })
    }
}

pub struct KvFireLease {
    lifecycle: Arc<KvLifecycle>,
}

impl Drop for KvFireLease {
    fn drop(&mut self) {
        let previous = self
            .lifecycle
            .active_fire_leases
            .fetch_sub(1, Ordering::SeqCst);
        debug_assert!(previous > 0);
        if previous == 1 {
            self.lifecycle.maybe_release();
            self.lifecycle.quiesced.notify_waiters();
        }
    }
}

#[derive(Clone)]
pub struct KvSuspendHandle {
    lifecycle: Weak<KvLifecycle>,
}

impl KvSuspendHandle {
    pub fn fence(&self) {
        if let Some(lifecycle) = self.lifecycle.upgrade() {
            lifecycle.suspend_fence.store(true, Ordering::SeqCst);
        }
    }

    pub fn unfence(&self) {
        if let Some(lifecycle) = self.lifecycle.upgrade() {
            lifecycle.suspend_fence.store(false, Ordering::SeqCst);
        }
    }

    pub fn active_leases(&self) -> usize {
        self.lifecycle.upgrade().map_or(0, |lifecycle| {
            lifecycle.active_fire_leases.load(Ordering::SeqCst)
        })
    }

    pub async fn quiesce(&self) {
        loop {
            let Some(lifecycle) = self.lifecycle.upgrade() else {
                return;
            };
            if lifecycle.active_fire_leases.load(Ordering::SeqCst) == 0 {
                return;
            }
            let notified = lifecycle.quiesced.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if lifecycle.active_fire_leases.load(Ordering::SeqCst) == 0 {
                return;
            }
            notified.await;
        }
    }
}

impl Drop for KvLifecycle {
    fn drop(&mut self) {
        self.release();
    }
}

#[derive(Debug, Clone)]
pub struct KvWorkingSet {
    pub model: usize,
    pub engine: EngineId,
    pub id: WorkingSetId,
    pub page_size: u32,
    lifecycle: Arc<KvLifecycle>,
}

impl KvWorkingSet {
    pub fn new(model: usize, engine: EngineId, id: WorkingSetId, page_size: u32) -> Self {
        Self::new_with_scope(model, engine, id, page_size, None)
    }

    fn new_with_scope(
        model: usize,
        engine: EngineId,
        id: WorkingSetId,
        page_size: u32,
        pipeline_scope: Option<crate::store::PipelineScope>,
    ) -> Self {
        KvWorkingSet {
            model,
            engine,
            id,
            page_size,
            lifecycle: Arc::new(KvLifecycle {
                released: AtomicBool::new(false),
                release_requested: AtomicBool::new(false),
                active_fire_leases: AtomicUsize::new(0),
                suspend_fence: AtomicBool::new(false),
                quiesced: tokio::sync::Notify::new(),
                model,
                engine,
                id,
                pipeline_scope: Mutex::new(pipeline_scope),
            }),
        }
    }

    pub fn forked(&self, id: WorkingSetId) -> Self {
        let scope = self.lifecycle.pipeline_scope.lock().unwrap().clone();
        Self::new_with_scope(self.model, self.engine, id, self.page_size, scope)
    }

    pub fn claim_pipeline_scope(
        &self,
        scope: &crate::store::PipelineScope,
    ) -> Result<(), crate::store::PipelineScopeId> {
        scope.claim(&self.lifecycle.pipeline_scope)
    }

    pub fn fire_lease(&self) -> Result<KvFireLease, FireLeaseError> {
        KvLifecycle::acquire_fire_lease(&self.lifecycle)
    }

    pub fn suspend_handle(&self) -> KvSuspendHandle {
        KvSuspendHandle {
            lifecycle: Arc::downgrade(&self.lifecycle),
        }
    }

    pub fn release(&self) {
        self.lifecycle.release();
    }
}
