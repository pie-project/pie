//! Thin WIT/resource handle for `kv-working-set` (kv_refact.md,
//! `store/kv/working_set.rs`). All substantive operations delegate to the
//! owning `KvStore`, resolved through `store::registry` by `(model, driver)`.
//!
//! [`KvWorkingSet`] is a lightweight `Clone` (not `Copy`) handle: every clone
//! shares one [`Arc<KvLifecycle>`]. Reading a clone out of the
//! `ResourceTable` (to inspect its fields, or resolve the owning store,
//! without holding the table borrow across an await/lock) is safe and free
//! of side effects — [`KvLifecycle`]'s idempotent release only runs when the
//! LAST outstanding `Arc` clone drops (ordinary Rust reference counting), so
//! a temporary clone dropped at the end of a host function never triggers an
//! early release while the table's own clone is still alive.
//!
//! `HostKvWorkingSet::drop` (the explicit WIT path,
//! `inferlet::host::kv_working_set`) additionally calls
//! [`KvWorkingSet::release`] synchronously right away, which both performs
//! the release NOW and marks it done — so the eventual `Arc` drop (when the
//! table's clone is deleted and goes out of scope) is a no-op. If instead a
//! `ResourceTable`/`ProcessCtx` is torn down directly and the WIT `drop`
//! glue is bypassed entirely, the table's clone is the only reference left
//! and ITS `Drop` performs the release — the process-teardown fallback.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use super::page_table::WorkingSetId;
use crate::driver::DriverId;

/// Why a fire lease was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FireLeaseError {
    /// The working set's release has been requested — a permanent refusal.
    Released,
    /// The planner is suspending (or has suspended) this working set — a
    /// transient refusal; the fire parks at the residency gate and retries
    /// after the restore.
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

/// Idempotent release fallback shared by every clone of one [`KvWorkingSet`].
#[derive(Debug)]
struct KvLifecycle {
    released: AtomicBool,
    release_requested: AtomicBool,
    active_fire_leases: AtomicUsize,
    /// The planner's suspend fence (Project Rainer). Dekker-paired with
    /// `active_fire_leases`: a fire increments the count then loads the
    /// fence; the planner stores the fence then loads the count (both
    /// SeqCst) — so either the fire sees the fence and backs out, or the
    /// planner sees the fire's lease and waits for it. While fenced the
    /// count is monotone non-increasing, which makes `quiesce` a finite
    /// wait and the D2H copy race-free against launches.
    suspend_fence: AtomicBool,
    /// Notified whenever the lease count drops to zero under the fence.
    quiesced: tokio::sync::Notify,
    model: usize,
    driver: DriverId,
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
        let stores = crate::store::registry::get(self.model, self.driver as usize);
        crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
            let epoch = kv.current_epoch();
            kv.release_working_set(self.id, epoch);
            kv.retire_idle();
        }); // store lock released before the planner's drain re-locks pools.
        // Freed pool space may unblock a parked ask.
        if let Some(planner) = crate::planner::planner() {
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
            // The planner's quiesce wait (if any) is over.
            self.lifecycle.quiesced.notify_waiters();
        }
    }
}

/// The planner's weak handle onto a working set's lifecycle — the suspend
/// fence and quiescence wait live here, never keeping the set alive.
#[derive(Clone)]
pub struct KvSuspendHandle {
    lifecycle: Weak<KvLifecycle>,
}

impl KvSuspendHandle {
    /// Raise the suspend fence: every later [`KvWorkingSet::fire_lease`]
    /// refuses with [`FireLeaseError::Fenced`].
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

    /// The current fire-lease count (0 when the lifecycle is gone). Two
    /// consumers: the eviction executor's quiesce trace, and the planner's
    /// victim-selection PREFERENCE (`kv_lease_quiescent` — a racy snapshot
    /// that only reorders equally-eligible candidates; the fence + quiesce
    /// wait remains the correctness seal). The starvation predicate
    /// deliberately does NOT read lease snapshots: a guest between two
    /// fires holds none for an instant but is very much alive — see
    /// `check_starvation`.
    pub fn active_leases(&self) -> usize {
        self.lifecycle.upgrade().map_or(0, |lifecycle| {
            lifecycle.active_fire_leases.load(Ordering::SeqCst)
        })
    }

    /// Await zero fire leases. Meaningful only under a raised fence, where
    /// the count is monotone non-increasing. Returns immediately when the
    /// lifecycle is already gone (released — nothing left to protect).
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
    /// The process-teardown fallback: runs only when the LAST `Arc` clone of
    /// a [`KvWorkingSet`]'s lifecycle drops. No-ops if [`KvWorkingSet::release`]
    /// already ran explicitly. Never panics — `release` only takes locks and
    /// calls infallible store methods.
    fn drop(&mut self) {
        self.release();
    }
}

/// Host resource state behind the `pie:inferlet/working-set.kv-working-set`
/// WIT resource. `Clone`, not `Copy` (see module docs): every clone shares
/// one lifecycle, so pulling a value out of the `ResourceTable` for field
/// access never triggers an early release.
#[derive(Debug, Clone)]
pub struct KvWorkingSet {
    pub model: usize,
    pub driver: DriverId,
    pub id: WorkingSetId,
    /// Tokens per KV page (cached from the store registry at construction).
    pub page_size: u32,
    translation: Arc<crate::store::kv::KvTranslation>,
    lifecycle: Arc<KvLifecycle>,
}

impl KvWorkingSet {
    /// A fresh handle for a NEWLY minted working-set `id` (a `create`,
    /// `fork`, or `slice` result — never an ALREADY-live id, which would
    /// wrongly share this fresh lifecycle with another handle's).
    pub fn new(model: usize, driver: DriverId, id: WorkingSetId, page_size: u32) -> Self {
        Self::new_with_scope(model, driver, id, page_size, None)
    }

    fn new_with_scope(
        model: usize,
        driver: DriverId,
        id: WorkingSetId,
        page_size: u32,
        pipeline_scope: Option<crate::store::PipelineScope>,
    ) -> Self {
        let stores = crate::store::registry::get(model, driver as usize);
        let translation =
            crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
                kv.translation(id)
                    .expect("new working set has a translation state")
            });
        KvWorkingSet {
            model,
            driver,
            id,
            page_size,
            translation,
            lifecycle: Arc::new(KvLifecycle {
                released: AtomicBool::new(false),
                release_requested: AtomicBool::new(false),
                active_fire_leases: AtomicUsize::new(0),
                suspend_fence: AtomicBool::new(false),
                quiesced: tokio::sync::Notify::new(),
                model,
                driver,
                id,
                pipeline_scope: Mutex::new(pipeline_scope),
            }),
        }
    }

    pub fn forked(&self, id: WorkingSetId) -> Self {
        let scope = self.lifecycle.pipeline_scope.lock().unwrap().clone();
        Self::new_with_scope(self.model, self.driver, id, self.page_size, scope)
    }

    pub fn claim_pipeline_scope(
        &self,
        scope: &crate::store::PipelineScope,
    ) -> Result<(), crate::store::PipelineScopeId> {
        let mut owner = self.lifecycle.pipeline_scope.lock().unwrap();
        match owner.as_ref() {
            Some(existing) if existing.id() == scope.id() => Ok(()),
            // Succession, on exactly the RS rule (`rs::WorkingSet`): a scope
            // that is closed AND drained has no fire left that could still
            // reference this mapping, so handing the working set to a live
            // pipeline races with nothing. Without this a context is
            // single-use — the guest's second `generate` on the same pool
            // opens a new pipeline (one per generate) and would be refused,
            // which is every multi-turn conversation.
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

    pub fn fire_lease(&self) -> Result<KvFireLease, FireLeaseError> {
        KvLifecycle::acquire_fire_lease(&self.lifecycle)
    }

    /// The planner's weak handle for suspend fencing and quiescence.
    pub fn suspend_handle(&self) -> KvSuspendHandle {
        KvSuspendHandle {
            lifecycle: Arc::downgrade(&self.lifecycle),
        }
    }

    /// Whether no submitted fire still holds this WorkingSet's mapping.
    pub fn is_settled(&self) -> bool {
        self.lifecycle.active_fire_leases.load(Ordering::Acquire) == 0
            && !self.lifecycle.release_requested.load(Ordering::Acquire)
    }

    pub fn translation(&self) -> Result<(u64, Arc<[u32]>), &'static str> {
        self.translation.snapshot()
    }

    /// Explicit release (the WIT `drop` path): releases resource ownership,
    /// drains contention, and marks it
    /// done, so this handle's (and any other outstanding clone's, e.g. the
    /// `ResourceTable`'s own) eventual `Arc` drop is a no-op.
    pub fn release(&self) {
        self.lifecycle.release();
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
    //! Process-teardown fallback coverage (thrust-3): a `KvWorkingSet` value
    //! dropped directly — never routed through `HostKvWorkingSet::drop`'s
    //! explicit `release()` — must still return its reserved/committed pages
    //! to the pool. This is the host-level analog of the CUDA-contention
    //! "solo lane exhausts an 8-page pool" repro: an exhausted pool that
    //! isn't reclaimed on teardown compounds across every later process.
    use super::*;
    use crate::store::kv::write::PageCommit;
    use crate::store::registry;

    /// A fresh single-driver model registration with a `capacity`-page pool,
    /// isolated from every other test (`register_model` mints a new model
    /// index each call).
    fn fresh_model(capacity: usize) -> usize {
        registry::register_model(16, &[capacity], &[0])
    }

    /// Reserve + prepare + commit `n` FRESH pages onto `id` (mirrors
    /// `store::kv::tests::commit_fresh`): the exact shape a solo forward
    /// lane uses to actually consume physical pool capacity (plain
    /// `reserve` alone is logical-only and never touches the pool).
    fn commit_fresh_pages(model: usize, id: WorkingSetId, n: u64, _epoch: u64) {
        let stores = registry::get(model, 0);
        let mut kv = stores.kv.lock();
        let start = kv.page_len(id).unwrap();
        kv.reserve(id, n).unwrap();
        let indexes: Vec<u64> = (start..start + n).collect();
        let prepared = kv.prepare_write(id, &indexes).unwrap();
        let commits: Vec<PageCommit> = (0..n)
            .map(|_| PageCommit {
                token_hashes: Vec::new(),
                page_hash: None,
            })
            .collect();
        let (seq, intents) = kv.publish_prepared(prepared, &commits).unwrap();
        kv.settle(intents, true);
        kv.retire_through(seq);
    }

    #[test]
    fn drop_without_explicit_release_reclaims_pool_capacity() {
        let model = fresh_model(4);
        let stores = registry::get(model, 0);
        let id = stores.kv.lock().create_working_set();
        commit_fresh_pages(model, id, 4, 1);
        assert_eq!(
            stores.kv.lock().available_pages(),
            0,
            "the lane's fresh commit exhausts the 4-page pool"
        );

        // Simulate a `ResourceTable`/`ProcessCtx` teardown dropping the
        // resource value directly — `HostKvWorkingSet::drop`/`release` is
        // never called.
        let ws = KvWorkingSet::new(model, 0, id, 16);
        drop(ws);

        assert_eq!(
            stores.kv.lock().available_pages(),
            4,
            "the Drop fallback released the working set's pages back to the pool"
        );
    }

    #[test]
    fn explicit_release_is_idempotent_and_the_drop_fallback_does_not_double_free() {
        let model = fresh_model(4);
        let stores = registry::get(model, 0);
        let id = stores.kv.lock().create_working_set();
        commit_fresh_pages(model, id, 4, 1);

        let ws = KvWorkingSet::new(model, 0, id, 16);
        assert!(!ws.is_released());
        ws.release();
        assert!(ws.is_released());
        assert_eq!(
            stores.kv.lock().available_pages(),
            4,
            "the explicit release reclaimed the pool"
        );

        // A second explicit release (e.g. a defensive extra call) must not
        // re-run the release logic against an already-torn-down id.
        ws.release();
        assert_eq!(stores.kv.lock().available_pages(), 4);

        // The value's own `Drop` (its lifecycle `Arc`'s last reference)
        // fires next — also a no-op, since `released` is already set.
        drop(ws);
        assert_eq!(
            stores.kv.lock().available_pages(),
            4,
            "no double release/free after the explicit release already ran"
        );
    }

    #[test]
    fn a_temporary_clone_dropping_first_does_not_release_early() {
        // Mirrors the `let ws = table.get(&this)?.clone();` pattern used
        // throughout the host glue to read fields without holding the table
        // borrow: the temporary clone must NOT release when it drops — only
        // the LAST clone (here, the "table's own" original) may.
        let model = fresh_model(4);
        let stores = registry::get(model, 0);
        let id = stores.kv.lock().create_working_set();
        commit_fresh_pages(model, id, 4, 1);

        let table_owned = KvWorkingSet::new(model, 0, id, 16);
        let temporary_clone = table_owned.clone();
        drop(temporary_clone);
        assert_eq!(
            stores.kv.lock().available_pages(),
            0,
            "a non-last clone's drop must not trigger release"
        );
        assert!(!table_owned.is_released());

        drop(table_owned);
        assert_eq!(
            stores.kv.lock().available_pages(),
            4,
            "the last clone's drop runs the release fallback"
        );
    }

    #[test]
    fn fork_mints_an_independent_lifecycle_not_a_shared_clone() {
        let model = fresh_model(4);
        let stores = registry::get(model, 0);
        let parent_id = stores.kv.lock().create_working_set();
        commit_fresh_pages(model, parent_id, 2, 1);
        let child_id = stores
            .kv
            .lock()
            .fork(parent_id, Default::default())
            .unwrap();

        let parent = KvWorkingSet::new(model, 0, parent_id, 16);
        let child = KvWorkingSet::new(model, 0, child_id, 16);

        // Releasing the child must not mark the parent released, nor
        // release the parent's pages (a shared-Arc bug would conflate the
        // two ids under one lifecycle).
        child.release();
        assert!(!parent.is_released());
        assert_eq!(
            stores.kv.lock().available_pages(),
            2,
            "the fork shares the parent's 2 committed pages; releasing the \
             child alone doesn't reclaim them"
        );

        drop(parent);
        assert_eq!(stores.kv.lock().available_pages(), 4);
    }

    #[test]
    fn pipeline_scope_succeeds_only_once_drained_and_is_inherited_by_forks() {
        let model = fresh_model(4);
        let stores = registry::get(model, 0);
        let parent_id = stores.kv.lock().create_working_set();
        let child_id = stores
            .kv
            .lock()
            .fork(parent_id, Default::default())
            .unwrap();
        let parent = KvWorkingSet::new(model, 0, parent_id, 16);
        let drained = Arc::new(AtomicBool::new(false));
        let drained_probe = Arc::clone(&drained);
        let first = crate::store::PipelineScope::new(move || drained_probe.load(Ordering::Acquire));
        let second = crate::store::PipelineScope::new(|| true);

        parent.claim_pipeline_scope(&first).unwrap();
        // A live scope owns the mapping outright.
        assert_eq!(parent.claim_pipeline_scope(&second), Err(first.id()));
        first.close();
        // Closed is not enough: a fire of the old pipeline may still be in
        // flight against this mapping.
        assert_eq!(parent.claim_pipeline_scope(&second), Err(first.id()));

        // Drained: the working set is reusable, which is what lets a guest
        // run a second `generate` (a new pipeline) on the same context.
        drained.store(true, Ordering::Release);
        assert_eq!(parent.claim_pipeline_scope(&second), Ok(()));

        // A fork inherits the parent's current scope rather than starting free.
        let child = parent.forked(child_id);
        let third = crate::store::PipelineScope::new(|| true);
        assert_eq!(child.claim_pipeline_scope(&third), Err(second.id()));
        child.release();
        parent.release();
    }
}
