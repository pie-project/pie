//! Thin WIT/resource handle for `rs-working-set` (kv_refact.md,
//! `store/rs/working_set.rs`). All substantive operations delegate to the
//! owning `RsStore`, resolved through `store::registry` by `(model, driver)`.
//!
//! [`RsWorkingSet`] mirrors [`crate::store::kv::working_set::KvWorkingSet`]'s
//! clone-safe lifecycle: it is `Clone`, not `Copy`, and every clone shares
//! one [`Arc<RsLifecycle>`]. Cloning a value out of the `ResourceTable` (to
//! read its fields without holding the table borrow across a lock) is safe
//! — [`RsLifecycle`]'s idempotent release only runs when the LAST
//! outstanding `Arc` clone drops. `HostRsWorkingSet::drop` (the explicit WIT
//! path, `inferlet::host::rs_working_set`) calls [`RsWorkingSet::release`]
//! synchronously right away and marks it done, so the eventual `Arc` drop
//! (when the table's own clone is deleted) is a no-op; a `ResourceTable`/
//! `ProcessCtx` teardown that bypasses the WIT `drop` glue leaves the
//! table's clone as the last reference, whose `Drop` then performs the
//! release — the process-teardown fallback.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::{RsGeometry, RsWorkingSetId};
use crate::driver::DriverId;

/// Idempotent release fallback shared by every clone of one [`RsWorkingSet`]
/// value. Performs the exact `release_working_set` / `current_epoch` /
/// `retire_idle` sequence the explicit WIT `drop` used to inline; shared so
/// a bypassed WIT drop still runs it exactly once, via this type's own
/// `Drop`.
#[derive(Debug)]
struct RsLifecycle {
    released: AtomicBool,
    model: usize,
    driver: DriverId,
    id: RsWorkingSetId,
    pipeline_scope: Mutex<Option<usize>>,
}

impl RsLifecycle {
    fn release(&self) {
        if self.released.swap(true, Ordering::AcqRel) {
            return;
        }
        let stores = crate::store::registry::get(self.model, self.driver as usize);
        let mut rs = stores.rs.lock().unwrap();
        let epoch = rs.current_epoch();
        rs.release_working_set(self.id, epoch);
        rs.retire_idle();
    }
}

impl Drop for RsLifecycle {
    /// The process-teardown fallback: runs only when the LAST `Arc` clone of
    /// an [`RsWorkingSet`]'s lifecycle drops. No-ops if
    /// [`RsWorkingSet::release`] already ran explicitly. Never panics —
    /// `release` only takes a lock and calls infallible store methods.
    fn drop(&mut self) {
        self.release();
    }
}

/// Host resource state behind the `pie:inferlet/working-set.rs-working-set`
/// WIT resource. `Clone`, not `Copy` (see module docs): every clone shares
/// one lifecycle, so pulling a value out of the `ResourceTable` for field
/// access never triggers an early release.
#[derive(Debug, Clone)]
pub struct RsWorkingSet {
    pub model: usize,
    pub driver: DriverId,
    pub id: RsWorkingSetId,
    /// Model RS geometry (cached from model caps at construction).
    pub geom: RsGeometry,
    lifecycle: Arc<RsLifecycle>,
}

impl RsWorkingSet {
    /// A fresh handle for a NEWLY minted working-set `id` (a `create` or
    /// `fork` result — never an ALREADY-live id, which would wrongly share
    /// this fresh lifecycle with another handle's).
    pub fn new(model: usize, driver: DriverId, id: RsWorkingSetId, geom: RsGeometry) -> Self {
        RsWorkingSet {
            model,
            driver,
            id,
            geom,
            lifecycle: Arc::new(RsLifecycle {
                released: AtomicBool::new(false),
                model,
                driver,
                id,
                pipeline_scope: Mutex::new(None),
            }),
        }
    }

    /// Explicit release (the WIT `drop` path): runs `release_working_set` +
    /// `retire_idle` NOW and marks it done, so this handle's (and any other
    /// outstanding clone's, e.g. the `ResourceTable`'s own) eventual `Arc`
    /// drop is a no-op.
    pub fn release(&self) {
        self.lifecycle.release();
    }

    pub fn claim_pipeline_scope(&self, scope: usize) -> Result<(), usize> {
        let mut owner = self.lifecycle.pipeline_scope.lock().unwrap();
        match *owner {
            Some(existing) if existing != scope => Err(existing),
            Some(_) => Ok(()),
            None => {
                *owner = Some(scope);
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
    //! Process-teardown fallback coverage (thrust-3), mirroring
    //! `store::kv::working_set::tests`: an `RsWorkingSet` dropped directly
    //! (never routed through `HostRsWorkingSet::drop`'s explicit `release()`)
    //! must still return its slots to the pool.
    use super::*;
    use crate::store::registry;

    fn geom() -> RsGeometry {
        RsGeometry {
            state_size: 4096,
            buffer_page_tokens: 4,
            fold_granularity: 4,
        }
    }

    /// A fresh single-driver model registration with a `capacity`-slot RS
    /// pool, isolated from every other test (`register_model` mints a new
    /// model index each call).
    fn fresh_model(capacity: usize) -> usize {
        registry::register_model(16, &[0], &[capacity])
    }

    /// Prepare + commit a fresh folded-state write (mirrors
    /// `store::rs::tests::write_state`): consumes exactly one RS pool slot.
    fn commit_state_write(model: usize, id: RsWorkingSetId, epoch: u64) {
        let stores = registry::get(model, 0);
        let mut rs = stores.rs.lock().unwrap();
        let prepared = rs.prepare_write(id, true, None).unwrap();
        rs.commit(prepared, epoch).unwrap();
    }

    #[test]
    fn drop_without_explicit_release_reclaims_pool_capacity() {
        let model = fresh_model(1);
        let stores = registry::get(model, 0);
        let id = stores.rs.lock().unwrap().create_working_set(geom());
        commit_state_write(model, id, 1);
        assert_eq!(
            stores.rs.lock().unwrap().available_slots(),
            0,
            "the lane's fresh state write exhausts the 1-slot pool"
        );

        // Simulate a `ResourceTable`/`ProcessCtx` teardown dropping the
        // resource value directly — `HostRsWorkingSet::drop`/`release` is
        // never called.
        let ws = RsWorkingSet::new(model, 0, id, geom());
        drop(ws);

        assert_eq!(
            stores.rs.lock().unwrap().available_slots(),
            1,
            "the Drop fallback released the working set's slot back to the pool"
        );
    }

    #[test]
    fn explicit_release_is_idempotent_and_the_drop_fallback_does_not_double_free() {
        let model = fresh_model(1);
        let stores = registry::get(model, 0);
        let id = stores.rs.lock().unwrap().create_working_set(geom());
        commit_state_write(model, id, 1);

        let ws = RsWorkingSet::new(model, 0, id, geom());
        assert!(!ws.is_released());
        ws.release();
        assert!(ws.is_released());
        assert_eq!(stores.rs.lock().unwrap().available_slots(), 1);

        // A second explicit release must not re-run against an
        // already-torn-down id.
        ws.release();
        assert_eq!(stores.rs.lock().unwrap().available_slots(), 1);

        // The value's own `Drop` fires next — also a no-op.
        drop(ws);
        assert_eq!(
            stores.rs.lock().unwrap().available_slots(),
            1,
            "no double release/free after the explicit release already ran"
        );
    }

    #[test]
    fn a_temporary_clone_dropping_first_does_not_release_early() {
        let model = fresh_model(1);
        let stores = registry::get(model, 0);
        let id = stores.rs.lock().unwrap().create_working_set(geom());
        commit_state_write(model, id, 1);

        let table_owned = RsWorkingSet::new(model, 0, id, geom());
        let temporary_clone = table_owned.clone();
        drop(temporary_clone);
        assert_eq!(
            stores.rs.lock().unwrap().available_slots(),
            0,
            "a non-last clone's drop must not trigger release"
        );
        assert!(!table_owned.is_released());

        drop(table_owned);
        assert_eq!(
            stores.rs.lock().unwrap().available_slots(),
            1,
            "the last clone's drop runs the release fallback"
        );
    }

    #[test]
    fn fork_mints_an_independent_lifecycle_not_a_shared_clone() {
        let model = fresh_model(2);
        let stores = registry::get(model, 0);
        let parent_id = stores.rs.lock().unwrap().create_working_set(geom());
        commit_state_write(model, parent_id, 1);
        let child_id = stores.rs.lock().unwrap().fork(parent_id).unwrap();

        let parent = RsWorkingSet::new(model, 0, parent_id, geom());
        let child = RsWorkingSet::new(model, 0, child_id, geom());

        // Releasing the child must not mark the parent released, nor
        // release the parent's slot (a shared-Arc bug would conflate the
        // two ids under one lifecycle).
        child.release();
        assert!(!parent.is_released());
        assert_eq!(
            stores.rs.lock().unwrap().available_slots(),
            1,
            "the fork shares the parent's folded slot; releasing the child \
             alone doesn't reclaim it"
        );

        drop(parent);
        assert_eq!(stores.rs.lock().unwrap().available_slots(), 2);
    }

    #[test]
    fn working_set_is_scoped_to_one_pipeline_fifo() {
        let model = fresh_model(1);
        let stores = registry::get(model, 0);
        let id = stores.rs.lock().unwrap().create_working_set(geom());
        let ws = RsWorkingSet::new(model, 0, id, geom());

        assert_eq!(ws.claim_pipeline_scope(0x11), Ok(()));
        assert_eq!(ws.claim_pipeline_scope(0x11), Ok(()));
        assert_eq!(ws.claim_pipeline_scope(0x22), Err(0x11));
    }
}
