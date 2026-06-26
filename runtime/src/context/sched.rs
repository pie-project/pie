//! Scheduling — FCFS Contention and Eviction.
//!
//! This module handles GPU page contention under first-come-first-served
//! (FCFS) ordering keyed on each process's monotonic launch sequence:
//!
//! - **Contention**: `when_allocated` — the universal GPU page contention
//!   primitive. Handles free-pool allocation, eviction loops, priority gates,
//!   and deferred-op stashing for suspended contexts.
//! - **Eviction**: `find_eviction_victim` — evicts the most-recently-launched
//!   (highest `launch_seq`) GPU-resident context; an older context keeps its
//!   pages.
//! - **Suspension**: `suspend` — stashes pages to CPU, releases GPU refcounts.
//! - **Queue draining**: `drain_queues` — serves `alloc_queue` (FIFO) and
//!   `restore_queue` (oldest-launched first) as pages become available.
//!
//! ## Effective Pages
//!
//! Contexts sharing a KV-cache prefix split its physical cost equally.
//! `effective_pages(i)` = unique pages + Σ shared_segment_pages / refcount,
//! used for placement (`best_driver_for`).

use std::fmt;
use std::time::Instant;

use crate::driver::{self, DriverId};
use crate::process::ProcessId;

use super::pagestore::PhysicalPageId;
use super::{Context, ContextId, ContextManager, RestoreEntry, State};

// =============================================================================
// ProcessEntry — Ownership + FCFS launch order
// =============================================================================

/// Per-process state: owned contexts + FCFS launch sequence.
#[derive(Debug)]
pub(crate) struct ProcessEntry {
    /// Context IDs owned by this process.
    pub context_ids: Vec<ContextId>,
    /// FCFS launch sequence (monotonic, assigned at `process::spawn`).
    /// Eviction targets the highest (newest-launched); the restore queue
    /// serves the lowest (oldest-launched) first.
    pub launch_seq: u64,
}

impl ProcessEntry {
    pub(crate) fn new(launch_seq: u64) -> Self {
        ProcessEntry {
            context_ids: Vec::new(),
            launch_seq,
        }
    }
}

// =============================================================================
// PendingAlloc — Deferred GPU Page Operation
// =============================================================================

/// A deferred GPU page operation stored on `ctx.deferred_ops`.
///
/// On success, `on_alloc` is called with pre-allocated pages.
/// On cancellation (destroy), the struct is dropped — dropping the closure
/// drops the captured `oneshot::Sender`, closing the channel.
pub(crate) struct PendingAlloc {
    pub driver: usize,
    pub num_pages: usize,
    pub needs_rs_slot: bool,
    /// Callback invoked with allocated pages on success.
    pub on_alloc: Box<dyn FnOnce(&mut ContextManager, Vec<PhysicalPageId>) + Send>,
}

impl fmt::Debug for PendingAlloc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingAlloc")
            .field("driver", &self.driver)
            .field("num_pages", &self.num_pages)
            .field("needs_rs_slot", &self.needs_rs_slot)
            .field("on_alloc", &"<closure>")
            .finish()
    }
}

// =============================================================================
// Process Lifecycle
// =============================================================================

impl ContextManager {
    /// Register a process with its FCFS launch sequence.
    /// Called once per process lifetime (from `InstanceState::new`).
    ///
    /// Panics on double-registration (indicates a bug in the caller).
    pub(crate) fn register_process(
        &mut self,
        pid: ProcessId,
        launch_seq: u64,
    ) -> anyhow::Result<()> {
        assert!(
            !self.processes.contains_key(&pid),
            "register_process: process {pid} already registered"
        );
        self.processes.insert(pid, ProcessEntry::new(launch_seq));
        Ok(())
    }

    /// Unregister a process: destroy all owned contexts and remove the process entry.
    /// Called on WASM instance drop for automatic cleanup.
    pub(crate) fn unregister_process(&mut self, pid: ProcessId) {
        let t_start = Instant::now();
        let proc = match self.processes.remove(&pid) {
            Some(p) => p,
            None => return,
        };

        // Drop this process's contexts from alloc_queue.
        let ctx_ids: std::collections::HashSet<ContextId> =
            proc.context_ids.iter().copied().collect();
        self.alloc_queue.retain(|ctx_id| !ctx_ids.contains(ctx_id));

        // restore_queue: lazy deletion — stale entries filtered on pop in drain_queues.

        let t_queues = t_start.elapsed();

        // Destroy all owned contexts
        for ctx_id in &proc.context_ids {
            crate::inference::invalidate_speculation_for_ctx(*ctx_id);
            if let Some(ctx) = self.contexts.remove(ctx_id) {
                let driver_idx = ctx.driver.unwrap_or(0) as usize;
                if let Some(slot) = ctx.rs_state.resident_slot() {
                    if let Some(store) = self.rs_stores.get_mut(driver_idx) {
                        store.free(slot);
                    }
                }
                if !ctx.committed_hashes.is_empty() && !ctx.is_off_gpu() {
                    self.gpu_stores[driver_idx].release(&ctx.committed_hashes);
                }
                if !ctx.working_pages.is_empty() {
                    self.gpu_stores[driver_idx].free(&ctx.working_pages);
                }
                if !ctx.cpu_working_pages.is_empty() {
                    self.cpu_stores[driver_idx].free(&ctx.cpu_working_pages);
                }
                if ctx.is_off_gpu() && !ctx.committed_hashes.is_empty() {
                    self.cpu_stores[driver_idx].release(&ctx.committed_hashes);
                }
                self.remove_context_caches(*ctx_id);
            }
        }

        // Single pass: remove all snapshot entries pointing to this process's contexts.
        self.snapshots.retain(|_, v| !ctx_ids.contains(v));

        let t_destroy = t_start.elapsed();

        let t_pre_drain = t_start.elapsed();
        // Early-exit: skip drain_queues if both queues are empty.
        if !self.restore_queue.is_empty() || !self.alloc_queue.is_empty() {
            self.drain_queues();
        }
        let t_total = t_start.elapsed();

        self.sched_counters.unreg_queues_us += t_queues.as_micros() as u64;
        self.sched_counters.unreg_destroy_us += (t_destroy - t_queues).as_micros() as u64;
        self.sched_counters.unreg_drain_us += (t_total - t_pre_drain).as_micros() as u64;
    }

    /// Get a mutable reference to a registered process's entry.
    /// Panics if the process is not registered — callers must ensure
    /// `register_process()` was called first.
    pub(crate) fn process_entry(&mut self, pid: ProcessId) -> &mut ProcessEntry {
        self.processes
            .get_mut(&pid)
            .expect("process_entry: process not registered (missing register_process call)")
    }

    // =========================================================================
    // Best Driver — Per-Context Shapley Placement (§4.3)
    // =========================================================================

    /// Evaluate the cheapest driver for a single context by page cost.
    ///
    /// Picks the driver minimizing `effective_pages(ctx, d) + migration_cost`,
    /// favoring prefix reuse and avoiding cross-driver migration.
    ///
    /// Called from `drain_queues` before each restore.
    pub(crate) fn best_driver_for(&self, ctx: &Context) -> usize {
        let num_drivers = self.gpu_stores.len();
        let current_dev = ctx.driver.unwrap_or(0) as usize;
        if num_drivers <= 1 {
            return current_dev;
        }

        let hashes = &ctx.committed_hashes;
        if hashes.is_empty() {
            return current_dev;
        }

        let working_count = if ctx.is_off_gpu() {
            ctx.suspended_working_count as f64
        } else {
            ctx.working_pages.len() as f64
        };

        let mut best_dev = current_dev;
        let mut best_cost = f64::MAX;

        for d in 0..num_drivers {
            let shared = self.gpu_stores[d].prefix_len(hashes);
            let shared_eff = if shared > 0 {
                self.gpu_stores[d].effective_pages(&hashes[..shared])
            } else {
                0.0
            };
            let unique_eff = (hashes.len() - shared) as f64 + working_count;
            let eff = shared_eff + unique_eff;

            let migration_cost = if d != current_dev {
                hashes.len() as f64
            } else {
                0.0
            };

            let total_cost = eff + migration_cost;
            if total_cost < best_cost {
                best_cost = total_cost;
                best_dev = d;
            }
        }

        best_dev
    }

    // =========================================================================
    // GPU Page Contention
    // =========================================================================

    /// Pending-aware operation helper (no pages needed).
    ///
    /// If the owning context is Suspended, defers `on_ready` as a zero-page
    /// `PendingAlloc`. Otherwise calls `on_ready` immediately. Used for
    /// operations like `pin` that don't need page allocation but must respect
    /// context suspension.
    pub(crate) fn when_active(
        &mut self,
        ctx_id: ContextId,
        on_ready: impl FnOnce(&mut ContextManager) + Send + 'static,
    ) {
        self.when_allocated(ctx_id, 0, 0, move |mgr, _pages| on_ready(mgr));
    }

    /// Universal GPU page contention primitive.
    ///
    /// Attempts to allocate `num_pages` GPU pages on `driver_idx` for context `ctx_id`.
    /// Goes through: Suspended check → num_pages==0 fast-path →
    /// priority gate → free pool → eviction loop → self-suspend.
    ///
    /// On success, invokes `on_alloc` with the allocated pages.
    /// On deferral, the operation is stashed on `ctx.deferred_ops`
    /// and will be replayed when pages become available.
    pub(crate) fn when_allocated(
        &mut self,
        ctx_id: ContextId,
        driver_idx: usize,
        num_pages: usize,
        on_alloc: impl FnOnce(&mut ContextManager, Vec<PhysicalPageId>) + Send + 'static,
    ) {
        self.when_allocated_inner(ctx_id, driver_idx, num_pages, false, on_alloc);
    }

    /// Variant for operations such as fork/take that can operate from an
    /// off-GPU source context and replay directly into the destination.
    pub(crate) fn when_allocated_allow_off_gpu(
        &mut self,
        ctx_id: ContextId,
        driver_idx: usize,
        num_pages: usize,
        on_alloc: impl FnOnce(&mut ContextManager, Vec<PhysicalPageId>) + Send + 'static,
    ) {
        self.when_allocated_inner(ctx_id, driver_idx, num_pages, true, on_alloc);
    }

    fn when_allocated_inner(
        &mut self,
        ctx_id: ContextId,
        driver_idx: usize,
        num_pages: usize,
        allow_off_gpu: bool,
        on_alloc: impl FnOnce(&mut ContextManager, Vec<PhysicalPageId>) + Send + 'static,
    ) {
        // RESIDENCY/BUSY CHECK: If the context is off GPU or already pinned
        // by a forward/replay, store the operation and let unpin or
        // replay_complete fire it once the context is active again.
        let requester_off_gpu = if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            let off_gpu = ctx.is_off_gpu();
            if (off_gpu && !allow_off_gpu) || ctx.is_pinned() {
                let pending = PendingAlloc {
                    driver: driver_idx,
                    num_pages,
                    needs_rs_slot: false,
                    on_alloc: Box::new(on_alloc),
                };
                ctx.deferred_ops.push(pending);
                return;
            }
            off_gpu
        } else {
            tracing::error!("when_allocated: context not found: {}", ctx_id);
            return;
        };

        // Short-circuit: no pages needed.
        if num_pages == 0 {
            (on_alloc)(self, Vec::new());
            return;
        }

        // Compute requester's FCFS launch sequence.
        let requester_seq = self.contexts[&ctx_id].launch_seq;

        // Step 2: PRIORITY GATE — compare requester vs restore_queue head.
        // If a strictly older-launched Suspended context is waiting, the
        // newer requester yields (FCFS: oldest gets the GPU first).
        if let Some(top_id) = self.oldest_in_restore_queue() {
            let top_seq = self.contexts[&top_id].launch_seq;
            if requester_seq > top_seq {
                self.sched_counters.priority_gate_suspends += 1;
                let pending = PendingAlloc {
                    driver: driver_idx,
                    num_pages,
                    needs_rs_slot: false,
                    on_alloc: Box::new(on_alloc),
                };
                self.contexts
                    .get_mut(&ctx_id)
                    .unwrap()
                    .deferred_ops
                    .push(pending);
                if requester_off_gpu {
                    self.alloc_queue.push_back(ctx_id);
                } else {
                    self.suspend(ctx_id);
                    self.enqueue_restore(ctx_id);
                }
                self.drain_queues();
                return;
            }
        }

        // Step 3: TRY ALLOCATE from free pool.
        if let Some(pages) = self.gpu_stores[driver_idx].alloc(num_pages) {
            (on_alloc)(self, pages);
            return;
        }

        // Step 4: EVICTION LOOP — with deferred page tracking.
        let mut deferred_pages: usize = 0;
        let mut has_deferred = false;
        let mut alloc_result: Option<Vec<PhysicalPageId>> = None;

        loop {
            match self.find_eviction_victim(driver_idx, requester_seq, Some(ctx_id)) {
                Some(victim_ctx_id) => {
                    self.sched_counters.eviction_suspends += 1;
                    let victim = self.contexts.get(&victim_ctx_id).unwrap();

                    if victim.is_pinned() {
                        // Pinned victim: set pending_suspend, pages freed later on unpin.
                        let reclaimable = victim.working_pages.len()
                            + self.gpu_stores[driver_idx]
                                .count_reclaimable(&victim.committed_hashes);
                        self.contexts
                            .get_mut(&victim_ctx_id)
                            .unwrap()
                            .pending_suspend = true;
                        deferred_pages += reclaimable;
                        has_deferred = true;
                    } else {
                        // Active victim: suspend immediately.
                        self.suspend(victim_ctx_id);
                        self.enqueue_restore(victim_ctx_id);
                    }

                    // Retry alloc after victim suspension freed pages.
                    if let Some(pages) = self.gpu_stores[driver_idx].alloc(num_pages) {
                        alloc_result = Some(pages);
                        break;
                    }

                    if has_deferred {
                        let free_now = self.gpu_stores[driver_idx].available();
                        if free_now + deferred_pages >= num_pages {
                            break;
                        }
                    }
                }
                None => break,
            }
        }

        // Post-loop: handle eviction results.
        if let Some(pages) = alloc_result {
            (on_alloc)(self, pages);
            self.drain_queues();
        } else {
            // No pages available — defer the operation.
            let pending = PendingAlloc {
                driver: driver_idx,
                num_pages,
                needs_rs_slot: false,
                on_alloc: Box::new(on_alloc),
            };
            self.contexts
                .get_mut(&ctx_id)
                .unwrap()
                .deferred_ops
                .push(pending);

            if has_deferred {
                // Step 5: Deferred pages from Pinned contexts will cover the gap.
                self.alloc_queue.push_back(ctx_id);
            } else if requester_off_gpu {
                // The requester is already off-GPU and this operation knows
                // how to replay from metadata; wait for capacity without
                // restoring the source context first.
                self.alloc_queue.push_back(ctx_id);
            } else {
                // Step 6: NO VICTIM — requester self-suspends.
                self.sched_counters.no_victim_suspends += 1;
                self.suspend(ctx_id);
                self.enqueue_restore(ctx_id);
                self.drain_queues();
            }
        }
    }

    // =========================================================================
    // Eviction
    // =========================================================================

    /// Find the best eviction victim context on a driver.
    ///
    /// FCFS: evicts the most-recently-launched (highest `launch_seq`)
    /// GPU-resident context on the driver. Only contexts launched no earlier
    /// than the requester are eligible — an older context keeps its pages.
    pub(crate) fn find_eviction_victim(
        &self,
        driver_idx: usize,
        requester_seq: u64,
        requester: Option<ContextId>,
    ) -> Option<ContextId> {
        // (launch_seq, ctx_id) — best victim has the highest launch_seq
        // (most recently launched).
        let mut best: Option<(u64, ContextId)> = None;

        for (&ctx_id, ctx) in &self.contexts {
            if requester == Some(ctx_id) {
                continue;
            }
            if ctx.is_off_gpu() {
                continue;
            }
            let ctx_driver = ctx.driver.unwrap_or(0) as usize;
            if ctx_driver != driver_idx {
                continue;
            }
            let pages = ctx.committed_hashes.len() + ctx.working_pages.len();
            if pages == 0 {
                continue;
            }
            if ctx.pending_suspend {
                continue;
            }

            // Only evict contexts launched no earlier than the requester.
            if ctx.launch_seq < requester_seq {
                continue;
            }

            let dominated = if let Some((best_seq, best_id)) = best {
                ctx.launch_seq > best_seq || (ctx.launch_seq == best_seq && ctx_id > best_id)
            } else {
                true
            };
            if dominated {
                best = Some((ctx.launch_seq, ctx_id));
            }
        }

        best.map(|(_, ctx_id)| ctx_id)
    }

    /// Helper: enqueue a context for restoration.
    pub(crate) fn enqueue_restore(&mut self, ctx_id: ContextId) {
        let launch_seq = self
            .contexts
            .get(&ctx_id)
            .map(|c| c.launch_seq)
            .unwrap_or(u64::MAX);
        self.restore_queue.push(RestoreEntry { ctx_id, launch_seq });
    }

    /// Peek at the oldest-launched context in the restore queue (the FCFS
    /// restore head). Returns None if the queue is empty.
    /// Applies the same lazy-deletion convention as `drain_queues`: entries
    /// whose owning context was already removed from `self.contexts` (e.g.
    /// its process unregistered before the entry was popped/restored) are
    /// stale and are popped/skipped here, so the caller never indexes
    /// `self.contexts` with a dangling `ContextId`.
    pub(crate) fn oldest_in_restore_queue(&mut self) -> Option<ContextId> {
        while let Some(entry) = self.restore_queue.peek() {
            if self.contexts.contains_key(&entry.ctx_id) {
                return Some(entry.ctx_id);
            }
            self.restore_queue.pop();
        }
        None
    }

    // =========================================================================
    // CPU Eviction — tier-boundary contention
    // =========================================================================

    /// Find the best CPU eviction victim on a driver.
    ///
    /// Iterates all **Stashed** contexts (CPU-resident pages) on the driver
    /// and returns the most-recently-launched (highest `launch_seq`) under
    /// FCFS. The requester is excluded; only contexts launched no earlier than
    /// the requester are eligible.
    fn find_cpu_eviction_victim(
        &self,
        driver_idx: usize,
        requester_seq: u64,
        requester: Option<ContextId>,
    ) -> Option<ContextId> {
        let mut best: Option<(u64, ContextId)> = None;

        for (&ctx_id, ctx) in &self.contexts {
            if requester == Some(ctx_id) {
                continue;
            }
            if !ctx.is_stashed() {
                continue;
            }
            let ctx_driver = ctx.driver.unwrap_or(0) as usize;
            if ctx_driver != driver_idx {
                continue;
            }

            // Must have CPU-resident pages (working stash or committed stash).
            let has_cpu_working = !ctx.cpu_working_pages.is_empty();
            let has_cpu_committed = !ctx.committed_hashes.is_empty()
                && self.cpu_stores[driver_idx].prefix_len(&ctx.committed_hashes) > 0;
            if !has_cpu_working && !has_cpu_committed {
                continue;
            }

            // Only evict contexts launched no earlier than the requester.
            if ctx.launch_seq < requester_seq {
                continue;
            }

            let dominated = if let Some((best_seq, best_id)) = best {
                ctx.launch_seq > best_seq || (ctx.launch_seq == best_seq && ctx_id > best_id)
            } else {
                true
            };
            if dominated {
                best = Some((ctx.launch_seq, ctx_id));
            }
        }

        best.map(|(_, ctx_id)| ctx_id)
    }

    /// Evict a Stashed context's pages from CPU to recompute.
    ///
    /// Releases committed pages from the CPU FlatPageStore (rc--, free at
    /// rc=0) and frees working page stash from the CPU pool. Transitions
    /// the context from Stashed to Suspended (no CPU cache, full recompute
    /// on restore).
    fn evict_from_cpu(&mut self, ctx_id: ContextId) {
        let (driver_idx, committed_hashes, cpu_working) = match self.contexts.get(&ctx_id) {
            Some(ctx) if ctx.is_stashed() => (
                ctx.driver.unwrap_or(0) as usize,
                ctx.committed_hashes.clone(),
                ctx.cpu_working_pages.clone(),
            ),
            _ => return,
        };

        // Release committed pages from CPU store.
        if !committed_hashes.is_empty() {
            self.cpu_stores[driver_idx].release(&committed_hashes);
        }

        // Free working page stash from CPU pool.
        if !cpu_working.is_empty() {
            self.cpu_stores[driver_idx].free(&cpu_working);
            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.cpu_working_pages.clear();
            }
        }

        // Transition Stashed → Suspended (no longer has CPU pages).
        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            ctx.state = State::Suspended;
        }
    }

    // =========================================================================
    // Suspension
    // =========================================================================

    /// Suspend a single Active context: stash pages to CPU, then release GPU.
    ///
    /// Uses `would_free()` to identify which committed pages will reach rc=0
    /// on release, and stashes only those to CPU via FlatPageStore. Shared
    /// prefix pages (rc > 1) stay on GPU.
    ///
    /// When the CPU pool is full, runs an eviction loop to free CPU pages
    /// from the newest-launched suspended context before falling through to
    /// recompute.
    pub(crate) fn suspend(&mut self, ctx_id: ContextId) {
        let (driver_idx, working, committed_hashes) = match self.contexts.get(&ctx_id) {
            Some(ctx) if ctx.is_active() || ctx.is_pinned() => (
                ctx.driver.unwrap_or(0) as usize,
                ctx.working_pages.clone(),
                ctx.committed_hashes.clone(),
            ),
            _ => return,
        };

        // Recurrent state is volatile GPU residency. Releasing it here
        // makes rs_cache contention obey the same suspend/restore path
        // as KV pages; restore will replay lineage into a fresh slot.
        self.release_rs_slot_for_context(ctx_id);

        // Compute total CPU pages needed upfront for a single eviction pass.
        let requester_seq = self.contexts.get(&ctx_id).map(|c| c.launch_seq).unwrap_or(0);
        let working_cpu_needed = working.len();
        let evictable_hashes = if !committed_hashes.is_empty() {
            self.gpu_stores[driver_idx].would_free(&committed_hashes)
        } else {
            Vec::new()
        };
        let committed_cpu_needed = evictable_hashes.len();
        let total_cpu_needed = working_cpu_needed + committed_cpu_needed;

        // Single eviction pass: free enough CPU pages for both phases.
        if total_cpu_needed > 0 && self.cpu_stores[driver_idx].available() < total_cpu_needed {
            while self.cpu_stores[driver_idx].available() < total_cpu_needed {
                match self.find_cpu_eviction_victim(driver_idx, requester_seq, Some(ctx_id)) {
                    Some(victim_id) => {
                        self.evict_from_cpu(victim_id);
                    }
                    None => break,
                }
            }
        }

        // All-or-nothing: only stash to CPU if enough space for the full
        // request. Partial stashing (working on CPU, committed dropped) would
        // waste CPU capacity on a context that still needs full recompute.
        let cpu_offload =
            total_cpu_needed > 0 && self.cpu_stores[driver_idx].available() >= total_cpu_needed;

        // Phase 1: Stash working pages to CPU.
        // Working pages are exclusive — just D2H copy to CPU pool.
        if !working.is_empty() {
            let ctx = self.contexts.get_mut(&ctx_id).unwrap();
            ctx.suspended_working_count = ctx.working_pages.len();
            ctx.working_pages.clear();

            if cpu_offload {
                if let Some(cpu_pages) = self.cpu_stores[driver_idx].alloc(working.len()) {
                    let _ = driver::copy_d2h(driver_idx as DriverId, &working, &cpu_pages);
                    let ctx = self.contexts.get_mut(&ctx_id).unwrap();
                    ctx.cpu_working_pages = cpu_pages;
                }
            }

            self.gpu_stores[driver_idx].free(&working);
        }

        // Phase 2: Stash evictable committed pages to CPU.
        // Only pages with rc=1 (will reach rc=0 on release) need stashing.
        // Shared prefix pages (rc > 1) stay on GPU for other contexts.
        if cpu_offload && !evictable_hashes.is_empty() {
            let gpu_phys = self.gpu_stores[driver_idx].physical_ids(&evictable_hashes);
            if let Some(cpu_pages) = self.cpu_stores[driver_idx].alloc(gpu_phys.len()) {
                let _ = driver::copy_d2h(driver_idx as DriverId, &gpu_phys, &cpu_pages);
                self.cpu_stores[driver_idx].insert(&evictable_hashes, &cpu_pages);
            }
        }

        // Phase 3: Release committed chain refcounts from GPU trie.
        if !committed_hashes.is_empty() {
            if let Some(dev) = self.gpu_stores.get_mut(driver_idx) {
                dev.release(&committed_hashes);
            }
        }

        // Phase 4: Mark stashed or suspended.
        let ctx = self.contexts.get_mut(&ctx_id).unwrap();
        ctx.state = if cpu_offload {
            State::Stashed
        } else {
            State::Suspended
        };
        ctx.pending_suspend = false;

        // Remove this context from alloc_queue (can't serve while suspended;
        // deferred_ops are already on the context and will replay on restore).
        self.alloc_queue.retain(|&id| id != ctx_id);
    }

    /// Voluntarily suspend a context (program-initiated).
    pub(crate) fn voluntary_suspend(&mut self, id: ContextId) -> anyhow::Result<()> {
        match self.contexts.get(&id) {
            Some(ctx) if ctx.is_active() => {
                self.suspend(id);
                self.enqueue_restore(id);
                self.drain_queues();
                Ok(())
            }
            Some(ctx) if ctx.is_off_gpu() => Ok(()), // already off GPU
            Some(_) => Err(anyhow::anyhow!(
                "Context {id} is pinned, cannot voluntarily suspend"
            )),
            None => Err(anyhow::anyhow!("Context {id} not found")),
        }
    }
}
