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
    /// Its owner's hold on the planner past its exit, given back once the
    /// pages are: the planner forgets a process only after its last set.
    departure: Mutex<Option<Arc<crate::planner::Departure>>>,
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
            // Every fire of the set has settled (its leases drop only once
            // the engine has run it), and a page another set still reads is
            // not freed with it: no fire in flight reads what this lets go,
            // so it waits on no epoch. Held behind the newest fire's, the
            // pages of each process that finished stood off the pool while
            // unrelated fires ran, and the prompt that took its place waited
            // that long to be seated (Qwen3-8B c256: 0.8 to 2 s, main 6 ms).
            let epoch = kv.retired_through();
            kv.release_working_set(self.id, epoch);
            kv.retire_idle();
        });
        stores.seats.lock().unwrap().release(self.id);
        let departure = self.departure.lock().unwrap().take();
        drop(departure);
        if let Some(planner) = crate::planner::planner_for(self.model, self.engine) {
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
                departure: Mutex::new(None),
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

    /// Why the pipeline this set last fired on failed, if it did.
    pub fn pipeline_failure(&self) -> Option<String> {
        let scope = self.lifecycle.pipeline_scope.lock().unwrap();
        scope
            .as_ref()
            .and_then(crate::store::PipelineScope::failure)
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

    /// This set leaves its owner alive: the owner's hold on the planner, and
    /// the seats that fed its fires, go back. A seat keeps nothing between
    /// fires, so the next fire takes any free one; none may be in flight.
    pub fn leave_owner(&self) {
        debug_assert_eq!(self.lifecycle.active_fire_leases.load(Ordering::Acquire), 0);
        drop(self.lifecycle.departure.lock().unwrap().take());
        let stores = crate::store::registry::get(self.model, self.engine);
        stores.seats.lock().unwrap().release(self.id);
    }

    /// Holds `departure` until this set is released: its owner's pages are
    /// not back before then. A set released already holds nothing.
    pub fn hold_departure(&self, departure: Arc<crate::planner::Departure>) {
        let mut held = self.lifecycle.departure.lock().unwrap();
        if !self.lifecycle.released.load(Ordering::Acquire) {
            *held = Some(departure);
        }
    }
}

#[cfg(test)]
mod pipeline_failure_tests {
    use super::*;
    use crate::store::PipelineScope;

    #[test]
    fn a_set_answers_for_the_pipeline_it_last_fired_on() {
        let model = crate::store::registry::register_model(16, &[8], &[4]);
        let stores = crate::store::registry::get(model, 0);
        let id =
            crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| kv.create_working_set());
        let ws = KvWorkingSet::new(model, 0, id, 16);
        assert_eq!(ws.pipeline_failure(), None, "no pipeline yet");

        let failure = crate::pipeline::fire::PipelineFailure::default();
        let scope = PipelineScope::failing_as(failure.clone(), || true);
        ws.claim_pipeline_scope(&scope).unwrap();
        assert_eq!(ws.pipeline_failure(), None);

        *failure.lock().unwrap() = Some("the fire failed".to_string());
        assert_eq!(ws.pipeline_failure().as_deref(), Some("the fire failed"));

        let other = PipelineScope::failing_as(Default::default(), || true);
        let unrelated = KvWorkingSet::new(model, 0, id, 16);
        unrelated.claim_pipeline_scope(&other).unwrap();
        assert_eq!(unrelated.pipeline_failure(), None, "it is not process-wide");
    }
}
