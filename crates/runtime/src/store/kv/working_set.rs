//! Host resource handle for `kv-working-set`. All substantive operations
//! delegate to the owning `KvStore`, resolved via `store::registry`.
//!
//! [`KvWorkingSet`] is `Clone` (not `Copy`): every clone shares one
//! [`Arc<KvLifecycle>`], and release only runs when the last clone drops
//! (ordinary refcounting), so a temporary clone read out of the
//! `ResourceTable` never triggers an early release.
//!
//! `HostKvWorkingSet::drop` (the WIT path) calls [`KvWorkingSet::release`]
//! synchronously and marks it done, so the later `Arc` drop is a no-op.
//! If the `ResourceTable` is torn down directly instead, that `Arc`'s
//! `Drop` performs the release as a fallback.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use super::page_table::WorkingSetId;
use crate::engine::EngineId;

/// Why a fire lease was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FireLeaseError {
    /// The working set's release has been requested — a permanent refusal.
    Released,
    /// The planner is suspending this working set; the fire parks at the
    /// residency gate and retries after the restore.
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
    /// The planner's suspend fence. Dekker-paired with `active_fire_leases`
    /// (both SeqCst): a fire increments the count then loads the fence; the
    /// planner stores the fence then loads the count. While fenced the count
    /// is monotone non-increasing, making `quiesce` a finite wait.
    suspend_fence: AtomicBool,
    /// Notified whenever the lease count drops to zero under the fence.
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
        // A working set's seat shares its lifetime with its pages; unlike a
        // page it carries no bytes an in-flight fire could read, so it needs
        // no epoch delay.
        stores.seats.lock().unwrap().release(self.id);
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

    /// The current fire-lease count (0 when the lifecycle is gone). A racy
    /// snapshot used only as a victim-selection preference; the fence +
    /// quiesce wait is the correctness seal, not this count.
    pub fn active_leases(&self) -> usize {
        self.lifecycle.upgrade().map_or(0, |lifecycle| {
            lifecycle.active_fire_leases.load(Ordering::SeqCst)
        })
    }

    /// Await zero fire leases. Meaningful only under a raised fence.
    /// Returns immediately if the lifecycle is already gone.
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
    /// Process-teardown fallback; no-ops if [`KvWorkingSet::release`] already
    /// ran.
    fn drop(&mut self) {
        self.release();
    }
}

/// Host resource state behind the `kv-working-set` WIT resource. `Clone`,
/// not `Copy` (see module docs).
#[derive(Debug, Clone)]
pub struct KvWorkingSet {
    pub model: usize,
    pub engine: EngineId,
    pub id: WorkingSetId,
    /// Tokens per KV page (cached from the store registry at construction).
    pub page_size: u32,
    /// Lock-free mirror of this WorkingSet's logical page extent; read via
    /// [`Self::page_len`].
    page_len: Arc<AtomicU64>,
    translation: Arc<crate::store::kv::KvTranslation>,
    lifecycle: Arc<KvLifecycle>,
}

impl KvWorkingSet {
    /// A fresh handle for a newly minted working-set `id` (never an
    /// already-live one, which would wrongly share this fresh lifecycle).
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
        let stores = crate::store::registry::get(model, engine);
        let (translation, page_len) =
            crate::store::registry::with_kv_lock(&stores.kv, "host-working-set", |kv| {
                (
                    kv.translation(id)
                        .expect("new working set has a translation state"),
                    kv.page_len_mirror(id)
                        .expect("new working set has a page-length mirror"),
                )
            });
        KvWorkingSet {
            model,
            engine,
            id,
            page_size,
            page_len,
            translation,
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

    /// This WorkingSet's logical page extent, read without the global KV
    /// mutex: the extent moves only under this process's own writes, so the
    /// reader never races its own writer.
    pub fn page_len(&self) -> Result<u64, crate::store::kv::KvTableError> {
        match self.page_len.load(Ordering::Acquire) {
            u64::MAX => Err(crate::store::kv::KvTableError::UnknownWorkingSet),
            page_len => Ok(page_len),
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
        let mut owner = self.lifecycle.pipeline_scope.lock().unwrap();
        match owner.as_ref() {
            Some(existing) if existing.id() == scope.id() => Ok(()),
            // A closed and drained scope has no fire left referencing this
            // mapping, so it may be handed to a new pipeline (else a second
            // `generate` on the same context would always be refused).
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

    /// Explicit release (the WIT `drop` path): releases ownership and marks
    /// it done, so every clone's eventual `Arc` drop is a no-op.
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

