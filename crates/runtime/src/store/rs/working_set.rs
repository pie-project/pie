//! Thin WIT/resource handle for `rs-working-set`. All substantive operations delegate to the owning `RsStore`, resolved through `store::registry` by `(model, engine)`.
//! [`RsWorkingSet`] is `Clone`, not `Copy`; every clone shares one [`Arc<RsLifecycle>`], whose idempotent release only runs when the last outstanding clone drops. The explicit WIT `drop` path calls [`RsWorkingSet::release`] synchronously and marks it done, so a teardown that bypasses that glue still releases via `Drop`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::{RsGeometry, RsWorkingSetId};
use crate::engine::EngineId;

/// Idempotent release fallback shared by every clone of one [`RsWorkingSet`] value; runs `release_working_set`/`retire_idle` exactly once, via this type's `Drop`.
#[derive(Debug)]
struct RsLifecycle {
    released: AtomicBool,
    model: usize,
    engine: EngineId,
    id: RsWorkingSetId,
    pipeline_scope: Mutex<Option<crate::store::PipelineScope>>,
}

impl RsLifecycle {
    fn release(&self) {
        if self.released.swap(true, Ordering::AcqRel) {
            return;
        }
        let stores = crate::store::registry::get(self.model, self.engine);
        let mut rs = stores.rs.lock().unwrap();
        let epoch = rs.current_epoch();
        rs.release_working_set(self.id, epoch);
        rs.retire_idle();
    }
}

impl Drop for RsLifecycle {
    /// The process-teardown fallback: runs only when the last `Arc` clone drops. No-ops if [`RsWorkingSet::release`] already ran.
    fn drop(&mut self) {
        self.release();
    }
}

/// Host resource state behind the `pie:inferlet/working-set.rs-working-set` WIT resource. `Clone`, not `Copy`; every clone shares one lifecycle.
#[derive(Debug, Clone)]
pub struct RsWorkingSet {
    pub model: usize,
    pub engine: EngineId,
    pub id: RsWorkingSetId,
    /// Model RS geometry (cached from model caps at construction).
    pub geom: RsGeometry,
    lifecycle: Arc<RsLifecycle>,
}

impl RsWorkingSet {
    /// A fresh handle for a newly minted working-set `id` (a `create`/`fork` result — never an already-live id).
    pub fn new(model: usize, engine: EngineId, id: RsWorkingSetId, geom: RsGeometry) -> Self {
        RsWorkingSet {
            model,
            engine,
            id,
            geom,
            lifecycle: Arc::new(RsLifecycle {
                released: AtomicBool::new(false),
                model,
                engine,
                id,
                pipeline_scope: Mutex::new(None),
            }),
        }
    }

    /// Explicit release (the WIT `drop` path): runs now and marks it done, so every clone's eventual `Arc` drop is a no-op.
    pub fn release(&self) {
        self.lifecycle.release();
    }

    pub fn claim_pipeline_scope(
        &self,
        scope: &crate::store::PipelineScope,
    ) -> Result<(), crate::store::PipelineScopeId> {
        let mut owner = self.lifecycle.pipeline_scope.lock().unwrap();
        match owner.as_ref() {
            Some(existing) if existing.id() == scope.id() => Ok(()),
            Some(existing) if !scope.is_closed() && existing.is_releasable() => {
                *owner = Some(scope.clone());
                Ok(())
            }
            Some(existing) => Err(existing.id()),
            None if scope.is_closed() => Err(scope.id()),
            None => {
                *owner = Some(scope.clone());
                Ok(())
            }
        }
    }

    /// Whether [`Self::release`] (or the `Drop` fallback) has already run.
    /// Test/diagnostic use.
    #[cfg(test)]
    pub fn is_released(&self) -> bool {
        self.lifecycle.released.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::registry;

    fn geom() -> RsGeometry {
        RsGeometry {
            state_size: 4096,
            buffer_page_tokens: 4,
            fold_granularity: 4,
        }
    }

    /// A fresh single-engine model registration with a `capacity`-slot RS
    /// pool, isolated from every other test (`register_model` mints a new
    /// model index each call).
    fn fresh_model(capacity: usize) -> usize {
        registry::register_model(16, &[0], &[capacity])
    }

    /// Prepare + publish a fresh folded-state write (mirrors
    /// `store::rs::tests::write_state`): consumes exactly one RS pool slot.
    fn commit_state_write(model: usize, id: RsWorkingSetId, _epoch: u64) {
        let stores = registry::get(model, 0);
        let mut rs = stores.rs.lock().unwrap();
        let prepared = rs.prepare_write(id, true, None).unwrap();
        let published = rs.publish_prepared(prepared).unwrap();
        rs.settle(published);
    }

    #[test]
    fn working_set_is_scoped_to_one_pipeline_fifo() {
        let model = fresh_model(1);
        let stores = registry::get(model, 0);
        let id = stores.rs.lock().unwrap().create_working_set(geom());
        let ws = RsWorkingSet::new(model, 0, id, geom());

        let drained = Arc::new(AtomicBool::new(false));
        let drained_probe = Arc::clone(&drained);
        let first = crate::store::PipelineScope::new(move || drained_probe.load(Ordering::Acquire));
        let other = crate::store::PipelineScope::new(|| true);
        assert_eq!(ws.claim_pipeline_scope(&first), Ok(()));
        assert_eq!(ws.claim_pipeline_scope(&first), Ok(()));
        assert_eq!(ws.claim_pipeline_scope(&other), Err(first.id()));
        first.close();
        assert_eq!(ws.claim_pipeline_scope(&other), Err(first.id()));
        drained.store(true, Ordering::Release);
        assert_eq!(ws.claim_pipeline_scope(&other), Ok(()));
    }
}
