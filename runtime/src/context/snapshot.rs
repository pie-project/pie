//! Snapshots — Named Context Save, Load, Fork, Take.
//!
//! Provides named persistence points for contexts. Snapshots are namespace-scoped
//! by `(username, name)` and stored as context IDs in `ContextManager.snapshots`.
//!
//! Key operations:
//! - **save**: clone a context's committed chain + lineage into a new snapshot.
//! - **fork**: create a new context sharing the snapshot's committed pages (retain).
//! - **take**: fork + delete in one step (transfers ownership).
//! - **delete**: release committed chain refcounts and remove the snapshot.

use std::time::Instant;

use anyhow::Result;
use tokio::sync::oneshot;

use super::rs_cache::RsState;
use super::{Context, ContextId, ContextManager, State};

use crate::driver::{self, DriverId};
use crate::process::ProcessId;

fn snapshot_pages(ctx: &Context) -> usize {
    ctx.committed_hashes.len()
        .saturating_add(ctx.working_pages.len())
        .saturating_add(ctx.suspended_working_count)
}

// =============================================================================
// Persistence methods on ContextManager
// =============================================================================

impl ContextManager {
    /// Contention-aware fork: estimates GPU page requirement and delegates
    /// to `when_allocated` for Pending deferral and contention resolution.
    ///
    /// Clones the committed chain (refcount bump), copies working pages using
    /// pre-allocated GPU pages, and creates a new Active context.
    ///
    /// When the source is Suspended with partially-evicted committed pages,
    /// allocates suffix GPU pages and spawns replay forward passes to restore
    /// them (child starts as Pinned until replay completes).
    pub(crate) fn fork(
        &mut self,
        id: ContextId,
        owner: ProcessId,
        response: oneshot::Sender<Result<ContextId>>,
    ) {
        let (needed, driver_idx) = match self.contexts.get(&id) {
            Some(ctx) => {
                let driver_idx = ctx.driver.unwrap_or(0) as usize;
                let working = ctx.working_pages.len();
                let prefix = if ctx.is_off_gpu() && !ctx.committed_hashes.is_empty() {
                    self.gpu_stores[driver_idx].prefix_len(&ctx.committed_hashes)
                } else {
                    ctx.committed_hashes.len()
                };
                let suffix = if ctx.is_off_gpu() && !ctx.committed_hashes.is_empty() {
                    ctx.committed_hashes.len() - prefix
                } else {
                    0
                };
                let rs_scratch = if ctx.rs_state.is_missing()
                    && self.driver_uses_rs_cache(driver_idx)
                    && self.context_token_len(ctx) > 0
                {
                    prefix
                } else {
                    0
                };
                (working + suffix + rs_scratch, driver_idx)
            }
            None => {
                let _ = response.send(Err(anyhow::anyhow!("Context not found")));
                return;
            }
        };

        // Create a temporary context ID for the fork operation.
        // We use the source context for contention — the fork inherits
        // the source's scheduling state.
        self.when_allocated_allow_off_gpu(id, driver_idx, needed, move |mgr, gpu_pages| {
            let result = (|| -> Result<ContextId> {
                let ctx = mgr
                    .contexts
                    .get(&id)
                    .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
                let driver_idx = ctx.driver.unwrap_or(0) as usize;
                let driver_id = driver_idx as DriverId;
                let src_on_gpu = !ctx.is_off_gpu();

                // Snapshot source state.
                let committed_hashes = ctx.committed_hashes.clone();
                let max_pos = ctx.max_committed_position;
                let lineage = ctx.lineage.clone();
                let forked_tokens = ctx.working_page_tokens.clone();
                let next_forward_id = ctx.next_forward_id;
                let src = ctx.working_pages.clone();
                let src_rs_state = ctx.rs_state;
                let source_has_state =
                    ctx.committed_len() * mgr.page_size + ctx.working_page_tokens.len() > 0;
                let working_count = src.len();

                // Determine committed chain restoration needs.
                let (prefix_len, suffix_count) = if !committed_hashes.is_empty() && !src_on_gpu {
                    let prefix = mgr.gpu_stores[driver_idx].prefix_len(&committed_hashes);
                    (prefix, committed_hashes.len() - prefix)
                } else {
                    (committed_hashes.len(), 0)
                };

                // Validate pre-allocated page count against actual needs.
                let rs_replay_needed = mgr.driver_uses_rs_cache(driver_idx)
                    && src_rs_state.is_missing()
                    && source_has_state;
                let rs_scratch_count = if rs_replay_needed { prefix_len } else { 0 };
                let total_needed = working_count + suffix_count + rs_scratch_count;
                if gpu_pages.len() < total_needed {
                    mgr.gpu_stores[driver_idx].free(&gpu_pages);
                    anyhow::bail!(
                        "fork: insufficient GPU pages (got {}, need {total_needed})",
                        gpu_pages.len()
                    );
                }

                // Split pre-allocated pages: working | suffix | rs scratch | surplus.
                let working_pages = gpu_pages[..working_count].to_vec();
                let suffix_pages = gpu_pages[working_count..working_count + suffix_count].to_vec();
                let scratch_pages = gpu_pages[working_count + suffix_count..total_needed].to_vec();
                let surplus = gpu_pages[total_needed..].to_vec();
                if !surplus.is_empty() {
                    mgr.gpu_stores[driver_idx].free(&surplus);
                }

                // Copy source working → child working using pre-allocated pages.
                if !src.is_empty() && !working_pages.is_empty() {
                    let _ = if src_on_gpu {
                        driver::copy_d2d(driver_id, &src, &working_pages)
                    } else {
                        driver::copy_h2d(driver_id, &working_pages, &src)
                    };
                }

                // Retain committed prefix. For Active sources, retain the full chain.
                // For Suspended sources, retain only the GPU-resident prefix.
                if prefix_len > 0 {
                    mgr.gpu_stores[driver_idx].fork(&committed_hashes[..prefix_len]);
                }

                // Create the child context (state set below after replay check).
                let new_id = mgr.next_id();
                let mut child_rs_state = if mgr.driver_uses_rs_cache(driver_idx) {
                    if let Some(src_slot) = src_rs_state.resident_slot() {
                        let dst_slot = mgr.alloc_rs_slot_now_with_eviction(id, driver_idx)?;
                        driver::copy_rs_d2d(driver_id, &[src_slot], &[dst_slot])?;
                        RsState::Resident(dst_slot)
                    } else if src_rs_state.is_missing() && source_has_state {
                        RsState::Missing
                    } else {
                        RsState::Empty
                    }
                } else {
                    RsState::Unsupported
                };
                let rs_replay_slot = if child_rs_state.is_missing() {
                    let slot = mgr.alloc_rs_slot_now_with_eviction(id, driver_idx)?;
                    child_rs_state = RsState::Resident(slot);
                    Some(slot)
                } else {
                    None
                };
                mgr.contexts.insert(
                    new_id,
                    Context {
                        owner: Some(owner),
                        driver: Some(driver_id),
                        working_pages,
                        suspended_working_count: 0,
                        committed_hashes: committed_hashes.clone(),
                        max_committed_position: max_pos,
                        lineage,
                        rs_state: child_rs_state,
                        working_page_tokens: forked_tokens,
                        driver_repaired_spec_tail: 0,
                        next_forward_id,
                        state: State::Active, // may become Pinned below
                        pending_suspend: false,
                        last_access: Instant::now(),
                        bid: 0.0,
                        cpu_working_pages: Vec::new(),
                        deferred_ops: Vec::new(),
                        pending_replay: false,
                        defaulted: false,
                        cached_effective_pages: 0.0,
                    },
                );

                // Spawn replay for committed suffix if needed.
                let has_replay = if let Some(slot) = rs_replay_slot {
                    mgr.spawn_full_rs_replay_pass(
                        new_id,
                        driver_idx,
                        slot,
                        prefix_len,
                        scratch_pages,
                        suffix_pages,
                    )?
                } else if suffix_count > 0 {
                    if !scratch_pages.is_empty() {
                        mgr.gpu_stores[driver_idx].free(&scratch_pages);
                    }
                    mgr.spawn_replay_passes(new_id, driver_idx, prefix_len, suffix_pages)?
                } else {
                    if !scratch_pages.is_empty() {
                        mgr.gpu_stores[driver_idx].free(&scratch_pages);
                    }
                    false
                };

                if has_replay {
                    if let Some(ctx) = mgr.contexts.get_mut(&new_id) {
                        ctx.state = State::Pinned;
                        ctx.pending_replay = true;
                    }
                }

                mgr.process_entry(owner).context_ids.push(new_id);
                mgr.publish_context_counts(new_id);

                Ok(new_id)
            })();
            let _ = response.send(result);
        });
    }

    /// Save/snapshot a context. If `name` is None, auto-generates a snapshot name.
    /// Returns the name used (Some only when auto-generated).
    pub(crate) fn save(
        &mut self,
        id: ContextId,
        username: String,
        name: Option<String>,
    ) -> Result<Option<String>> {
        let (name, auto_generated) = match name {
            Some(n) => (n, false),
            None => (format!("__snapshot_{}", self.next_id()), true),
        };

        if self
            .snapshots
            .contains_key(&(username.clone(), name.clone()))
        {
            anyhow::bail!("Snapshot name already exists: {}", name);
        }

        let ctx = self
            .contexts
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let driver_idx = ctx.driver.unwrap_or(0) as usize;
        let committed_hashes = ctx.committed_hashes.clone();
        let lineage = ctx.lineage.clone();
        let src_working = ctx.working_pages.clone();

        let max_pos = ctx.max_committed_position;
        let snapshot_filled = ctx.working_page_tokens.clone();
        let next_forward_id = ctx.next_forward_id;
        let src_rs_state = ctx.rs_state;
        let source_has_state =
            ctx.committed_len() * self.page_size + ctx.working_page_tokens.len() > 0;

        let reserved_rs_slot = if self.driver_uses_rs_cache(driver_idx) {
            if src_rs_state.is_missing() && source_has_state {
                anyhow::bail!("save: source context has no resident rs_cache state to snapshot");
            }
            if src_rs_state.resident_slot().is_some() {
                Some(self.alloc_rs_slot_now_with_eviction(id, driver_idx)?)
            } else {
                None
            }
        } else {
            None
        };

        if !committed_hashes.is_empty() {
            self.gpu_stores[driver_idx].fork(&committed_hashes);
        }

        // Snapshot working pages: try GPU-first, fall back to CPU swap pool.
        let (snapshot_working, snapshot_state) = if !src_working.is_empty() {
            let n = src_working.len();
            if let Some(dst_pages) = self.gpu_stores[driver_idx].alloc(n) {
                // GPU → GPU copy
                let _ = driver::copy_d2d(driver_idx as DriverId, &src_working, &dst_pages);
                (dst_pages, State::Active)
            } else if let Some(cpu_pages) = self.cpu_stores[driver_idx].alloc(n) {
                // Fallback: GPU → CPU copy (source GPU pages stay intact)
                let _ = driver::copy_d2h(driver_idx as DriverId, &src_working, &cpu_pages);
                (cpu_pages, State::Stashed)
            } else {
                eprintln!("SNAPSHOT_PAGE_COPY_FAIL ctx={id}: no GPU or CPU pages available");
                (Vec::new(), State::Active)
            }
        } else {
            (Vec::new(), State::Active)
        };

        let snapshot_id = self.next_id();
        let snapshot_rs_state = if snapshot_state == State::Stashed {
            if let Some(slot) = reserved_rs_slot {
                self.rs_stores[driver_idx].free(slot);
            }
            if self.driver_uses_rs_cache(driver_idx) && source_has_state {
                RsState::Missing
            } else {
                self.initial_rs_state(driver_idx)
            }
        } else if let (Some(src_slot), Some(dst_slot)) =
            (src_rs_state.resident_slot(), reserved_rs_slot)
        {
            driver::copy_rs_d2d(driver_idx as DriverId, &[src_slot], &[dst_slot])?;
            RsState::Resident(dst_slot)
        } else {
            self.initial_rs_state(driver_idx)
        };
        self.contexts.insert(
            snapshot_id,
            Context {
                owner: None,
                driver: Some(driver_idx as DriverId),
                working_pages: snapshot_working,
                suspended_working_count: 0,
                committed_hashes: committed_hashes.clone(),
                lineage,
                rs_state: snapshot_rs_state,
                working_page_tokens: snapshot_filled,
                driver_repaired_spec_tail: 0,
                next_forward_id,
                max_committed_position: max_pos,
                state: snapshot_state,
                pending_suspend: false,
                last_access: Instant::now(),
                bid: 0.0,
                cpu_working_pages: Vec::new(),
                deferred_ops: Vec::new(),
                pending_replay: false,
                defaulted: false,
                cached_effective_pages: 0.0,
            },
        );
        self.snapshots.insert((username, name.clone()), snapshot_id);

        // If snapshot ended up Suspended (CPU fallback for working pages),
        // release the refcounts we acquired. Suspended invariant: no held refcounts.
        // On open/fork, retain will be called to re-acquire.
        if snapshot_state == State::Stashed && !committed_hashes.is_empty() {
            self.gpu_stores[driver_idx].release(&committed_hashes);
        }

        Ok(if auto_generated { Some(name) } else { None })
    }

    pub(crate) fn delete_snapshot_key(&mut self, key: &(String, String)) -> Result<()> {
        let snapshot_id = self.snapshots.remove(key)
            .ok_or_else(|| anyhow::anyhow!("Snapshot not found"))?;
        self.active_snapshots
            .retain(|(u, name, _), _| u != &key.0 || name != &key.1);

        if let Some(ctx) = self.contexts.remove(&snapshot_id) {
            let driver_idx = ctx.driver.unwrap_or(0) as usize;
            if let Some(slot) = ctx.rs_state.resident_slot() {
                if let Some(store) = self.rs_stores.get_mut(driver_idx) {
                    store.free(slot);
                }
            }
            if !ctx.committed_hashes.is_empty() && !ctx.is_off_gpu() {
                self.gpu_stores[driver_idx].release(&ctx.committed_hashes);
            }
            // Free snapshot working pages (GPU or CPU depending on state)
            if ctx.is_stashed() {
                self.cpu_stores[driver_idx].free(&ctx.working_pages);
            } else {
                self.gpu_stores[driver_idx].free(&ctx.working_pages);
            }
        }

        Ok(())
    }

    pub(crate) fn delete(&mut self, username: String, name: String) -> Result<()> {
        self.delete_snapshot_key(&(username, name))
    }

    pub(crate) fn retain_snapshot(
        &mut self,
        username: String,
        name: String,
        owner: crate::process::ProcessId,
    ) -> Result<()> {
        if !self.snapshots.contains_key(&(username.clone(), name.clone())) {
            anyhow::bail!("Snapshot not found");
        }
        *self.active_snapshots.entry((username, name, owner)).or_insert(0) += 1;
        Ok(())
    }

    pub(crate) fn release_snapshot(
        &mut self,
        username: String,
        name: String,
        owner: crate::process::ProcessId,
    ) -> Result<()> {
        let key = (username, name, owner);
        match self.active_snapshots.get_mut(&key) {
            Some(count) if *count > 1 => *count -= 1,
            Some(_) => {
                self.active_snapshots.remove(&key);
            }
            None => {}
        }
        Ok(())
    }

    fn is_snapshot_active(&self, key: &(String, String)) -> bool {
        self.active_snapshots
            .keys()
            .any(|(u, name, _)| u == &key.0 && name == &key.1)
    }

    pub(crate) fn enforce_snapshot_retention(
        &mut self,
        username: String,
        name_prefix: String,
        current_name: String,
        budget: super::SnapshotRetentionBudget,
    ) -> super::SnapshotRetentionReport {
        self.enforce_snapshot_retention_with_delete(
            username,
            name_prefix,
            current_name,
            budget,
            |mgr, key| mgr.delete_snapshot_key(key),
        )
    }

    pub(crate) fn enforce_snapshot_retention_with_delete<F>(
        &mut self,
        username: String,
        name_prefix: String,
        current_name: String,
        budget: super::SnapshotRetentionBudget,
        mut delete: F,
    ) -> super::SnapshotRetentionReport
    where
        F: FnMut(&mut Self, &(String, String)) -> Result<()>,
    {
        let scoped: Vec<((String, String), ContextId)> = self.snapshots
            .iter()
            .filter(|((u, name), _)| u == &username && name.starts_with(&name_prefix))
            .map(|(key, id)| (key.clone(), *id))
            .collect();
        let retained_snapshot_count = scoped.len() as u32;
        if !budget.coherent() {
            return super::SnapshotRetentionReport {
                retained_snapshot_count,
                reason: super::SnapshotRetentionReason::SkippedUncertainAccounting,
                ..Default::default()
            };
        }

        let current_key = (username.clone(), current_name);
        let mut protected_active_pages = 0u32;
        for (key, id) in &scoped {
            let protected = key == &current_key || self.is_snapshot_active(key);
            if protected {
                if let Some(ctx) = self.contexts.get(id) {
                    protected_active_pages =
                        protected_active_pages.saturating_add(snapshot_pages(ctx) as u32);
                }
            }
        }

        let target = budget.evict_pages();
        if budget.kv_pages_used <= budget.soft_pages() {
            return super::SnapshotRetentionReport {
                retained_snapshot_count,
                protected_active_pages,
                reason: super::SnapshotRetentionReason::RetainedBelowSoftLimit,
                ..Default::default()
            };
        }
        if budget.kv_pages_used <= target {
            return super::SnapshotRetentionReport {
                retained_snapshot_count,
                protected_active_pages,
                reason: super::SnapshotRetentionReason::RetainedBelowEvictionLimit,
                ..Default::default()
            };
        }

        let mut candidates: Vec<((String, String), usize, Instant)> = scoped
            .into_iter()
            .filter(|(key, _)| key != &current_key && !self.is_snapshot_active(key))
            .filter_map(|(key, id)| {
                self.contexts.get(&id).map(|ctx| (key, snapshot_pages(ctx), ctx.last_access))
            })
            .collect();
        candidates.sort_by(|(a_key, _, a_last), (b_key, _, b_last)| {
            a_last
                .cmp(b_last)
                .then_with(|| a_key.1.cmp(&b_key.1))
        });

        let mut projected = budget.kv_pages_used;
        let mut evicted_names = Vec::new();
        let mut pages_reclaimed = 0u32;
        let mut delete_failed_count = 0u32;
        for (key, pages, _) in candidates {
            if projected <= target {
                break;
            }
            match delete(self, &key) {
                Ok(()) => {
                    let pages = pages as u32;
                    projected = projected.saturating_sub(pages);
                    pages_reclaimed = pages_reclaimed.saturating_add(pages);
                    evicted_names.push(key.1);
                }
                Err(e) => {
                    delete_failed_count = delete_failed_count.saturating_add(1);
                    tracing::warn!(
                        "snapshot retention delete failed for user={} name={}: {e:#}",
                        key.0,
                        key.1
                    );
                }
            }
        }

        let retained_snapshot_count = self.snapshots
            .keys()
            .filter(|(u, name)| u == &username && name.starts_with(&name_prefix))
            .count() as u32;
        let reason = if delete_failed_count > 0 && projected > target {
            super::SnapshotRetentionReason::RetentionDeleteFailed
        } else if !evicted_names.is_empty() {
            if projected > budget.hard_pages() {
                super::SnapshotRetentionReason::HardCapStillExceeded
            } else {
                super::SnapshotRetentionReason::EvictedPressure
            }
        } else if protected_active_pages > 0 {
            super::SnapshotRetentionReason::ProtectedActive
        } else {
            super::SnapshotRetentionReason::NoInactiveSnapshots
        };

        super::SnapshotRetentionReport {
            evicted_names,
            pages_reclaimed,
            protected_active_pages,
            retained_snapshot_count,
            delete_failed_count,
            reason,
        }
    }


    /// Contention-aware take: estimates GPU page requirement and delegates
    /// to `when_allocated` for Pending deferral and contention resolution.
    ///
    /// Takes ownership of a saved snapshot. The snapshot is deleted and its
    /// resources are transferred to the new context. GPU working pages are
    /// moved directly (no D2D copy needed). CPU working pages use pre-allocated
    /// GPU pages for H2D copy. Committed pages are ref-bumped (shared via CAS).
    ///
    /// When the snapshot is Suspended with partially-evicted committed pages,
    /// allocates suffix GPU pages and spawns replay forward passes to restore
    /// them (context starts as Pinned until replay completes).
    pub(crate) fn take(
        &mut self,
        username: String,
        name: String,
        owner: ProcessId,
        response: oneshot::Sender<Result<ContextId>>,
    ) {
        // Estimate pages needed: working swap-in + committed suffix restoration.
        let key = (username.clone(), name.clone());
        let (needed, driver_idx, snap_ctx_id) = match self.snapshots.get(&key) {
            Some(&snap_id) => match self.contexts.get(&snap_id) {
                Some(snap) if snap.is_off_gpu() => {
                    let driver_idx = snap.driver.unwrap_or(0) as usize;
                    let working = snap.working_pages.len();
                    let prefix = if !snap.committed_hashes.is_empty() {
                        self.gpu_stores[driver_idx].prefix_len(&snap.committed_hashes)
                    } else {
                        0
                    };
                    let suffix = if !snap.committed_hashes.is_empty() {
                        snap.committed_hashes.len() - prefix
                    } else {
                        0
                    };
                    let rs_scratch = if snap.rs_state.is_missing()
                        && self.driver_uses_rs_cache(driver_idx)
                        && self.context_token_len(snap) > 0
                    {
                        prefix
                    } else {
                        0
                    };
                    (working + suffix + rs_scratch, driver_idx, snap_id)
                }
                Some(snap) => (0, snap.driver.unwrap_or(0) as usize, snap_id),
                None => {
                    let _ = response.send(Err(anyhow::anyhow!("Snapshot context missing")));
                    return;
                }
            },
            None => {
                let _ = response.send(Err(anyhow::anyhow!("Snapshot not found")));
                return;
            }
        };

        // For Take, we need pages but the snapshot context doesn't participate
        // in scheduling (no owner). We create the new context first and use it
        // for contention. But since we need to allocate before creating, we
        // use the snapshot context ID. If pages == 0, allocation is a no-op.
        if needed == 0 {
            // No GPU pages needed — do the take directly.
            let result = self.take_inner(username, name, owner, Vec::new());
            let _ = response.send(result);
            return;
        }

        // Allocate directly (no contention for Take — it's always a new context).
        if let Some(pages) = self.gpu_stores[driver_idx].alloc(needed) {
            let result = self.take_inner(username, name, owner, pages);
            let _ = response.send(result);
            return;
        }

        // Not enough free pages — defer on the snapshot context.
        let pending = super::sched::PendingAlloc {
            driver: driver_idx,
            num_pages: needed,
            needs_rs_slot: false,
            on_alloc: Box::new(move |mgr, pages| {
                let result = mgr.take_inner(username, name, owner, pages);
                let _ = response.send(result);
            }),
        };
        if let Some(ctx) = self.contexts.get_mut(&snap_ctx_id) {
            ctx.deferred_ops.push(pending);
        }
        self.alloc_queue.push_back(snap_ctx_id);
    }

    /// Inner take implementation, called with pre-allocated GPU pages.
    fn take_inner(
        &mut self,
        username: String,
        name: String,
        owner: ProcessId,
        gpu_pages: Vec<super::pagestore::PhysicalPageId>,
    ) -> Result<ContextId> {
        let key = (username, name);
        let snapshot_id = *self
            .snapshots
            .get(&key)
            .ok_or_else(|| anyhow::anyhow!("Snapshot not found"))?;

        let snap = self
            .contexts
            .get(&snapshot_id)
            .ok_or_else(|| anyhow::anyhow!("Snapshot context missing"))?;

        let driver_idx = snap.driver.unwrap_or(0) as usize;
        let was_off_gpu = snap.is_off_gpu();

        // Determine committed chain restoration needs.
        let (prefix_len, suffix_count) = if was_off_gpu && !snap.committed_hashes.is_empty() {
            let prefix = self.gpu_stores[driver_idx].prefix_len(&snap.committed_hashes);
            (prefix, snap.committed_hashes.len() - prefix)
        } else {
            (snap.committed_hashes.len(), 0)
        };

        let working_count = snap.working_pages.len();
        let snap_rs_state = snap.rs_state;
        let source_has_state =
            snap.committed_len() * self.page_size + snap.working_page_tokens.len() > 0;
        let rs_replay_needed =
            self.driver_uses_rs_cache(driver_idx) && snap_rs_state.is_missing() && source_has_state;
        let rs_scratch_count = if rs_replay_needed { prefix_len } else { 0 };
        let total_needed = working_count + suffix_count + rs_scratch_count;

        // Validate pre-allocated page count against actual needs.
        if gpu_pages.len() < total_needed {
            self.gpu_stores[driver_idx].free(&gpu_pages);
            anyhow::bail!(
                "take: insufficient GPU pages (got {}, need {total_needed})",
                gpu_pages.len()
            );
        }

        let new_rs_state = if self.driver_uses_rs_cache(driver_idx) {
            if let Some(slot) = snap_rs_state.resident_slot() {
                RsState::Resident(slot)
            } else if snap_rs_state.is_missing() && source_has_state {
                let slot = self.alloc_rs_slot_now_with_eviction(snapshot_id, driver_idx)?;
                RsState::Resident(slot)
            } else {
                RsState::Empty
            }
        } else {
            RsState::Unsupported
        };
        let rs_replay_slot = if rs_replay_needed {
            new_rs_state.resident_slot()
        } else {
            None
        };

        // Validation passed — consume the snapshot (point of no return).
        let snap = self.contexts.remove(&snapshot_id).unwrap();
        self.snapshots.remove(&key);

        // Split pre-allocated pages: working | suffix | rs scratch | surplus.
        let (working_gpu, rest) = gpu_pages.split_at(working_count);
        let suffix_pages = rest[..suffix_count].to_vec();
        let scratch_pages = rest[suffix_count..suffix_count + rs_scratch_count].to_vec();
        let surplus = rest[suffix_count + rs_scratch_count..].to_vec();
        let working_gpu = working_gpu.to_vec();
        if !surplus.is_empty() {
            self.gpu_stores[driver_idx].free(&surplus);
        }

        // Working pages: GPU pages transfer directly, CPU pages use pre-allocated pages.
        let new_working = if snap.working_pages.is_empty() {
            Vec::new()
        } else if was_off_gpu {
            let _ = driver::copy_h2d(driver_idx as DriverId, &working_gpu, &snap.working_pages);
            self.cpu_stores[driver_idx].free(&snap.working_pages);
            working_gpu
        } else {
            // GPU → new context: direct ownership transfer (zero-copy).
            if !working_gpu.is_empty() {
                self.gpu_stores[driver_idx].free(&working_gpu);
            }
            snap.working_pages
        };

        // Committed chain refcount handling.
        if was_off_gpu && prefix_len > 0 {
            self.gpu_stores[driver_idx].fork(&snap.committed_hashes[..prefix_len]);
        }

        let new_id = self.next_id();
        self.contexts.insert(
            new_id,
            Context {
                owner: Some(owner),
                driver: Some(driver_idx as DriverId),
                working_pages: new_working,
                suspended_working_count: 0,
                committed_hashes: snap.committed_hashes,
                lineage: snap.lineage,
                rs_state: new_rs_state,
                working_page_tokens: snap.working_page_tokens,
                driver_repaired_spec_tail: 0,
                next_forward_id: snap.next_forward_id,
                max_committed_position: snap.max_committed_position,
                state: State::Active,
                pending_suspend: false,
                last_access: Instant::now(),
                bid: 0.0,
                cpu_working_pages: Vec::new(),
                deferred_ops: Vec::new(),
                pending_replay: false,
                defaulted: false,
                cached_effective_pages: 0.0,
            },
        );

        // Spawn replay for committed suffix if needed.
        let has_replay = if let Some(slot) = rs_replay_slot {
            self.spawn_full_rs_replay_pass(
                new_id,
                driver_idx,
                slot,
                prefix_len,
                scratch_pages,
                suffix_pages,
            )?
        } else if suffix_count > 0 {
            if !scratch_pages.is_empty() {
                self.gpu_stores[driver_idx].free(&scratch_pages);
            }
            self.spawn_replay_passes(new_id, driver_idx, prefix_len, suffix_pages)?
        } else {
            if !scratch_pages.is_empty() {
                self.gpu_stores[driver_idx].free(&scratch_pages);
            }
            false
        };

        if has_replay {
            if let Some(ctx) = self.contexts.get_mut(&new_id) {
                ctx.state = State::Pinned;
                ctx.pending_replay = true;
            }
        }

        self.process_entry(owner).context_ids.push(new_id);
        self.publish_context_counts(new_id);

        Ok(new_id)
    }
}

#[cfg(test)]
mod retention_tests {
    use super::*;
    use crate::context::{Context, SnapshotRetentionBudget, SnapshotRetentionReason};

    fn manager() -> ContextManager {
        ContextManager::new(0, 16, &[32], &[32], 10, &[0], &[false], 16, None, 10000.0, 0.85)
    }

    fn add_snapshot(mgr: &mut ContextManager, username: &str, name: &str, pages: usize) {
        let id = mgr.next_id();
        let mut ctx = Context::new(None);
        ctx.driver = Some(0);
        ctx.working_pages = mgr.gpu_stores[0].alloc(pages).expect("snapshot pages");
        mgr.contexts.insert(id, ctx);
        mgr.snapshots.insert((username.to_string(), name.to_string()), id);
    }

    #[test]
    fn retention_lru_sees_snapshots_saved_before_the_current_request() {
        let mut mgr = manager();
        add_snapshot(&mut mgr, "u", "apc/old", 15);
        add_snapshot(&mut mgr, "u", "apc/recent", 15);

        mgr.retain_snapshot(
            "u".to_string(),
            "apc/recent".to_string(),
            crate::process::ProcessId::new_v4(),
        )
        .unwrap();
        let report = mgr.enforce_snapshot_retention(
            "u".to_string(),
            "apc/".to_string(),
            "apc/current".to_string(),
            SnapshotRetentionBudget {
                kv_pages_used: 95,
                kv_pages_total: 100,
                soft_percent: 70,
                evict_percent: 80,
                hard_percent: 95,
            },
        );

        assert_eq!(report.evicted_names, vec!["apc/old"]);
        assert_eq!(report.pages_reclaimed, 15);
        assert!(mgr.snapshots.contains_key(&("u".to_string(), "apc/recent".to_string())));
        assert!(!mgr.snapshots.contains_key(&("u".to_string(), "apc/old".to_string())));
    }

    #[test]
    fn retention_counts_only_successful_deletes_and_keeps_failed_candidates() {
        let mut mgr = manager();
        add_snapshot(&mut mgr, "u", "apc/old", 15);
        add_snapshot(&mut mgr, "u", "apc/new", 15);

        let report = mgr.enforce_snapshot_retention_with_delete(
            "u".to_string(),
            "apc/".to_string(),
            "apc/current".to_string(),
            SnapshotRetentionBudget {
                kv_pages_used: 95,
                kv_pages_total: 100,
                soft_percent: 70,
                evict_percent: 80,
                hard_percent: 95,
            },
            |mgr, key| {
                if key.1 == "apc/old" {
                    anyhow::bail!("injected delete failure")
                }
                mgr.delete_snapshot_key(key)
            },
        );

        assert_eq!(report.evicted_names, vec!["apc/new"]);
        assert_eq!(report.pages_reclaimed, 15);
        assert_eq!(report.delete_failed_count, 1);
        assert_eq!(report.reason, SnapshotRetentionReason::EvictedPressure);
        assert!(mgr.snapshots.contains_key(&("u".to_string(), "apc/old".to_string())));
        assert!(!mgr.snapshots.contains_key(&("u".to_string(), "apc/new".to_string())));
    }

    #[test]
    fn retention_reports_delete_failure_when_all_candidates_fail() {
        let mut mgr = manager();
        add_snapshot(&mut mgr, "u", "apc/old", 15);
        add_snapshot(&mut mgr, "u", "apc/new", 15);

        let report = mgr.enforce_snapshot_retention_with_delete(
            "u".to_string(),
            "apc/".to_string(),
            "apc/current".to_string(),
            SnapshotRetentionBudget {
                kv_pages_used: 95,
                kv_pages_total: 100,
                soft_percent: 70,
                evict_percent: 80,
                hard_percent: 95,
            },
            |_mgr, _key| anyhow::bail!("injected delete failure"),
        );

        assert!(report.evicted_names.is_empty());
        assert_eq!(report.pages_reclaimed, 0);
        assert_eq!(report.delete_failed_count, 2);
        assert_eq!(report.reason, SnapshotRetentionReason::RetentionDeleteFailed);
        assert!(mgr.snapshots.contains_key(&("u".to_string(), "apc/old".to_string())));
        assert!(mgr.snapshots.contains_key(&("u".to_string(), "apc/new".to_string())));
    }
}
