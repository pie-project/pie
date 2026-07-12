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
use std::sync::{Arc, Mutex};

use super::page_table::WorkingSetId;
use crate::driver::DriverId;

/// Auto-retained prefix-cache root cap (`PIE_KV_CACHE_ROOTS_MAX`, default
/// 256; `0` disables retention on release). The contention ladder's rung 1
/// reclaims retained roots the moment memory is needed, so the cap bounds
/// metadata, not pressure behavior. Read once (env lookup cached process-
/// wide) and baked into each [`KvWorkingSet`]'s lifecycle at construction, so
/// the release fallback — which may run long after construction, from
/// process teardown — never needs to consult the environment itself.
pub fn cache_roots_max() -> usize {
    static MAX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("PIE_KV_CACHE_ROOTS_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
    })
}

/// Idempotent release fallback shared by every clone of one [`KvWorkingSet`]
/// value. Performs the exact `release_working_set_cached` /
/// `current_epoch` / `retire_idle` + contention `on_blocks_freed` sequence
/// the explicit WIT `drop` used to inline; now shared so a bypassed WIT drop
/// (direct `ResourceTable`/`ProcessCtx` teardown) still runs it exactly
/// once, via this type's own `Drop`.
#[derive(Debug)]
struct KvLifecycle {
    released: AtomicBool,
    release_requested: AtomicBool,
    active_fire_leases: AtomicUsize,
    model: usize,
    driver: DriverId,
    id: WorkingSetId,
    cache_roots_max: usize,
    pipeline_scope: Mutex<Option<u128>>,
    submitted_cursor: Mutex<SubmittedCursor>,
}

#[derive(Debug, Default)]
struct SubmittedCursor {
    next_token: u64,
    generation: u64,
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
        {
            // Retain canonical paths as prefix-cache roots (bounded FIFO;
            // reclaimed by the contention ladder's rung 1 under pressure).
            let mut kv = stores.kv.lock().unwrap();
            let epoch = kv.current_epoch();
            kv.release_working_set_cached(self.id, epoch, self.cache_roots_max);
            kv.retire_idle();
        } // store lock released before the contention drain re-locks pools.
        // Freed pool space may unblock a preempted inferlet.
        if let Some(orchestrator) = crate::store::reclaim::contention() {
            orchestrator.on_blocks_freed();
        }
    }

    fn acquire_fire_lease(this: &Arc<Self>) -> Result<KvFireLease, &'static str> {
        if this.release_requested.load(Ordering::Acquire) {
            return Err("working set release already requested");
        }
        this.active_fire_leases.fetch_add(1, Ordering::AcqRel);
        if this.release_requested.load(Ordering::Acquire) {
            let previous = this.active_fire_leases.fetch_sub(1, Ordering::AcqRel);
            debug_assert!(previous > 0);
            this.maybe_release();
            return Err("working set release raced fire preparation");
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
            .fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
        if previous == 1 {
            self.lifecycle.maybe_release();
        }
    }
}

/// Submit-order token extent reserved on one working set.
///
/// Mapping allocation remains dispatch-time work, but every later submit must
/// immediately see this extent. A reservation rolls back only while it is
/// still the newest reservation; once a successor exists, failure poisons the
/// owning pipeline and the cursor must not rewind underneath that successor.
pub struct KvTokenExtentReservation {
    lifecycle: Arc<KvLifecycle>,
    start_token: u64,
    end_token: u64,
    previous_next_token: u64,
    generation: u64,
    committed: bool,
}

impl KvTokenExtentReservation {
    pub fn start_token(&self) -> u64 {
        self.start_token
    }

    pub fn commit(mut self) {
        self.committed = true;
    }
}

impl Drop for KvTokenExtentReservation {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let mut cursor = self.lifecycle.submitted_cursor.lock().unwrap();
        if cursor.generation == self.generation && cursor.next_token == self.end_token {
            cursor.next_token = self.previous_next_token;
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
    lifecycle: Arc<KvLifecycle>,
}

impl KvWorkingSet {
    /// A fresh handle for a NEWLY minted working-set `id` (a `create`,
    /// `fork`, or `slice` result — never an ALREADY-live id, which would
    /// wrongly share this fresh lifecycle with another handle's). Bakes in
    /// `cache_roots_max` now (see [`cache_roots_max`]) so the release
    /// fallback need not read the environment at drop time.
    pub fn new(model: usize, driver: DriverId, id: WorkingSetId, page_size: u32) -> Self {
        let cache_roots_max = cache_roots_max();
        KvWorkingSet {
            model,
            driver,
            id,
            page_size,
            lifecycle: Arc::new(KvLifecycle {
                released: AtomicBool::new(false),
                release_requested: AtomicBool::new(false),
                active_fire_leases: AtomicUsize::new(0),
                model,
                driver,
                id,
                cache_roots_max,
                pipeline_scope: Mutex::new(None),
                submitted_cursor: Mutex::new(SubmittedCursor::default()),
            }),
        }
    }

    pub fn claim_pipeline_scope(&self, scope: u128) -> Result<(), u128> {
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

    pub fn fire_lease(&self) -> Result<KvFireLease, &'static str> {
        KvLifecycle::acquire_fire_lease(&self.lifecycle)
    }

    pub fn reserve_token_extent(
        &self,
        pass_cursor: u64,
        committed_floor: u64,
        token_count: u64,
    ) -> Result<KvTokenExtentReservation, &'static str> {
        if token_count == 0 {
            return Err("token extent must be non-empty");
        }
        if self.lifecycle.release_requested.load(Ordering::Acquire) {
            return Err("working set release already requested");
        }
        let mut cursor = self.lifecycle.submitted_cursor.lock().unwrap();
        let previous_next_token = cursor.next_token;
        let start_token = pass_cursor.max(committed_floor).max(cursor.next_token);
        let end_token = start_token
            .checked_add(token_count)
            .ok_or("working set token extent overflow")?;
        cursor.generation = cursor.generation.wrapping_add(1);
        cursor.next_token = end_token;
        Ok(KvTokenExtentReservation {
            lifecycle: Arc::clone(&self.lifecycle),
            start_token,
            end_token,
            previous_next_token,
            generation: cursor.generation,
            committed: false,
        })
    }

    /// Explicit release (the WIT `drop` path): runs
    /// `release_working_set_cached` + contention drain NOW and marks it
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
    fn commit_fresh_pages(model: usize, id: WorkingSetId, n: u64, epoch: u64) {
        let stores = registry::get(model, 0);
        let mut kv = stores.kv.lock().unwrap();
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
        kv.commit(prepared, &commits, epoch).unwrap();
    }

    #[test]
    fn drop_without_explicit_release_reclaims_pool_capacity() {
        let model = fresh_model(4);
        let stores = registry::get(model, 0);
        let id = stores.kv.lock().unwrap().create_working_set();
        commit_fresh_pages(model, id, 4, 1);
        assert_eq!(
            stores.kv.lock().unwrap().available_pages(),
            0,
            "the lane's fresh commit exhausts the 4-page pool"
        );

        // Simulate a `ResourceTable`/`ProcessCtx` teardown dropping the
        // resource value directly — `HostKvWorkingSet::drop`/`release` is
        // never called.
        let ws = KvWorkingSet::new(model, 0, id, 16);
        drop(ws);

        assert_eq!(
            stores.kv.lock().unwrap().available_pages(),
            4,
            "the Drop fallback released the working set's pages back to the pool"
        );
    }

    #[test]
    fn explicit_release_is_idempotent_and_the_drop_fallback_does_not_double_free() {
        let model = fresh_model(4);
        let stores = registry::get(model, 0);
        let id = stores.kv.lock().unwrap().create_working_set();
        commit_fresh_pages(model, id, 4, 1);

        let ws = KvWorkingSet::new(model, 0, id, 16);
        assert!(!ws.is_released());
        ws.release();
        assert!(ws.is_released());
        assert_eq!(
            stores.kv.lock().unwrap().available_pages(),
            4,
            "the explicit release reclaimed the pool"
        );

        // A second explicit release (e.g. a defensive extra call) must not
        // re-run the release logic against an already-torn-down id.
        ws.release();
        assert_eq!(stores.kv.lock().unwrap().available_pages(), 4);

        // The value's own `Drop` (its lifecycle `Arc`'s last reference)
        // fires next — also a no-op, since `released` is already set.
        drop(ws);
        assert_eq!(
            stores.kv.lock().unwrap().available_pages(),
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
        let id = stores.kv.lock().unwrap().create_working_set();
        commit_fresh_pages(model, id, 4, 1);

        let table_owned = KvWorkingSet::new(model, 0, id, 16);
        let temporary_clone = table_owned.clone();
        drop(temporary_clone);
        assert_eq!(
            stores.kv.lock().unwrap().available_pages(),
            0,
            "a non-last clone's drop must not trigger release"
        );
        assert!(!table_owned.is_released());

        drop(table_owned);
        assert_eq!(
            stores.kv.lock().unwrap().available_pages(),
            4,
            "the last clone's drop runs the release fallback"
        );
    }

    #[test]
    fn fork_mints_an_independent_lifecycle_not_a_shared_clone() {
        let model = fresh_model(4);
        let stores = registry::get(model, 0);
        let parent_id = stores.kv.lock().unwrap().create_working_set();
        commit_fresh_pages(model, parent_id, 2, 1);
        let child_id = stores.kv.lock().unwrap().fork(parent_id).unwrap();

        let parent = KvWorkingSet::new(model, 0, parent_id, 16);
        let child = KvWorkingSet::new(model, 0, child_id, 16);

        // Releasing the child must not mark the parent released, nor
        // release the parent's pages (a shared-Arc bug would conflate the
        // two ids under one lifecycle).
        child.release();
        assert!(!parent.is_released());
        assert_eq!(
            stores.kv.lock().unwrap().available_pages(),
            2,
            "the fork shares the parent's 2 committed pages; releasing the \
             child alone doesn't reclaim them"
        );

        drop(parent);
        assert_eq!(stores.kv.lock().unwrap().available_pages(), 4);
    }

    #[test]
    fn submitted_token_extents_are_visible_across_clones_before_prepare() {
        let model = fresh_model(4);
        let stores = registry::get(model, 0);
        let id = stores.kv.lock().unwrap().create_working_set();
        let ws = KvWorkingSet::new(model, 0, id, 16);
        let other_pass = ws.clone();

        let first = ws.reserve_token_extent(0, 0, 4).unwrap();
        assert_eq!(first.start_token(), 0);
        first.commit();

        let second = other_pass.reserve_token_extent(0, 0, 1).unwrap();
        assert_eq!(
            second.start_token(),
            4,
            "a later pass sees the earlier submit before mapping preparation"
        );
        second.commit();
    }

    #[test]
    fn newest_uncommitted_token_extent_rolls_back_without_rewinding_successors() {
        let model = fresh_model(4);
        let stores = registry::get(model, 0);
        let id = stores.kv.lock().unwrap().create_working_set();
        let ws = KvWorkingSet::new(model, 0, id, 16);

        drop(ws.reserve_token_extent(0, 0, 4).unwrap());
        let replacement = ws.reserve_token_extent(0, 0, 2).unwrap();
        assert_eq!(replacement.start_token(), 0);
        replacement.commit();

        let earlier = ws.reserve_token_extent(0, 0, 3).unwrap();
        let later = ws.reserve_token_extent(0, 0, 1).unwrap();
        drop(earlier);
        assert_eq!(later.start_token(), 5);
        later.commit();

        let next = ws.reserve_token_extent(0, 0, 1).unwrap();
        assert_eq!(next.start_token(), 6);
    }
}
