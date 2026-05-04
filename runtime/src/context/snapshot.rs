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

use super::{
    Context, ContextId, ContextManager, State,
};

use crate::device::{self, DeviceId};
use crate::process::ProcessId;

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
        let (needed, dev_idx) = match self.contexts.get(&id) {
            Some(ctx) => {
                let dev_idx = ctx.device.unwrap_or(0) as usize;
                let working = ctx.working_pages.len();
                let suffix = if ctx.is_off_gpu() && !ctx.committed_hashes.is_empty() {
                    let prefix = self.gpu_stores[dev_idx].prefix_len(&ctx.committed_hashes);
                    ctx.committed_hashes.len() - prefix
                } else {
                    0
                };
                (working + suffix, dev_idx)
            }
            None => {
                let _ = response.send(Err(anyhow::anyhow!("Context not found")));
                return;
            }
        };

        // Create a temporary context ID for the fork operation.
        // We use the source context for contention — the fork inherits
        // the source's scheduling state.
        self.when_allocated(id, dev_idx, needed, move |mgr, gpu_pages| {
            let result = (|| -> Result<ContextId> {
                let ctx = mgr.contexts.get(&id)
                    .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
                let dev_idx = ctx.device.unwrap_or(0) as usize;
                let device = dev_idx as DeviceId;
                let src_on_gpu = !ctx.is_off_gpu();

                // Snapshot source state.
                let committed_hashes = ctx.committed_hashes.clone();
                let max_pos = ctx.max_committed_position;
                let lineage = ctx.lineage.clone();
                let forked_tokens = ctx.working_page_tokens.clone();
                let src = ctx.working_pages.clone();
                let working_count = src.len();

                // Determine committed chain restoration needs.
                let (prefix_len, suffix_count) = if !committed_hashes.is_empty() && !src_on_gpu {
                    let prefix = mgr.gpu_stores[dev_idx].prefix_len(&committed_hashes);
                    (prefix, committed_hashes.len() - prefix)
                } else {
                    (committed_hashes.len(), 0)
                };

                // Validate pre-allocated page count against actual needs.
                let total_needed = working_count + suffix_count;
                if gpu_pages.len() < total_needed {
                    mgr.gpu_stores[dev_idx].free(&gpu_pages);
                    anyhow::bail!(
                        "fork: insufficient GPU pages (got {}, need {total_needed})",
                        gpu_pages.len()
                    );
                }

                // Split pre-allocated pages: working | suffix | surplus.
                let working_pages = gpu_pages[..working_count].to_vec();
                let suffix_pages = gpu_pages[working_count..total_needed].to_vec();
                let surplus = gpu_pages[total_needed..].to_vec();
                if !surplus.is_empty() {
                    mgr.gpu_stores[dev_idx].free(&surplus);
                }

                // Copy source working → child working using pre-allocated pages.
                if !src.is_empty() && !working_pages.is_empty() {
                    let _ = if src_on_gpu {
                        device::copy_d2d(device, &src, &working_pages)
                    } else {
                        device::copy_h2d(device, &working_pages, &src)
                    };
                }

                // Retain committed prefix. For Active sources, retain the full chain.
                // For Suspended sources, retain only the GPU-resident prefix.
                if prefix_len > 0 {
                    mgr.gpu_stores[dev_idx].fork(&committed_hashes[..prefix_len]);
                }

                // Create the child context (state set below after replay check).
                let new_id = mgr.next_id();
                mgr.contexts.insert(new_id, Context {
                    owner: Some(owner),
                    device: Some(device),
                    working_pages,
                    suspended_working_count: 0,
                    committed_hashes: committed_hashes.clone(),
                    max_committed_position: max_pos,
                    lineage,
                    working_page_tokens: forked_tokens,
                    state: State::Active, // may become Pinned below
                    pending_suspend: false,
                    last_access: Instant::now(),
                    bid: 0.0,
                    cpu_working_pages: Vec::new(),
                    deferred_ops: Vec::new(),
                    pending_replay: false,
                    defaulted: false,
                    cached_effective_pages: 0.0,
                });

                // Spawn replay for committed suffix if needed.
                let has_replay = if suffix_count > 0 {
                    mgr.spawn_replay_passes(new_id, dev_idx, prefix_len, suffix_pages)?
                } else {
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
    pub(crate) fn save(&mut self, id: ContextId, username: String, name: Option<String>) -> Result<Option<String>> {
        let (name, auto_generated) = match name {
            Some(n) => (n, false),
            None => (format!("__snapshot_{}", self.next_id()), true),
        };

        if self.snapshots.contains_key(&(username.clone(), name.clone())) {
            anyhow::bail!("Snapshot name already exists: {}", name);
        }

        let ctx = self.contexts.get(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let committed_hashes = ctx.committed_hashes.clone();
        let lineage = ctx.lineage.clone();
        let src_working = ctx.working_pages.clone();

        let max_pos = ctx.max_committed_position;
        let snapshot_filled = ctx.working_page_tokens.clone();

        if !committed_hashes.is_empty() {
            self.gpu_stores[dev_idx].fork(&committed_hashes);
        }

        // Snapshot working pages: try GPU-first, fall back to CPU swap pool.
        let (snapshot_working, snapshot_state) = if !src_working.is_empty() {
            let n = src_working.len();
            if let Some(dst_pages) = self.gpu_stores[dev_idx].alloc(n) {
                // GPU → GPU copy
                let _ = device::copy_d2d(dev_idx as DeviceId, &src_working, &dst_pages);
                (dst_pages, State::Active)
            } else if let Some(cpu_pages) = self.cpu_stores[dev_idx].alloc(n) {
                // Fallback: GPU → CPU copy (source GPU pages stay intact)
                let _ = device::copy_d2h(dev_idx as DeviceId, &src_working, &cpu_pages);
                (cpu_pages, State::Stashed)
            } else {
                eprintln!("SNAPSHOT_PAGE_COPY_FAIL ctx={id}: no GPU or CPU pages available");
                (Vec::new(), State::Active)
            }
        } else {
            (Vec::new(), State::Active)
        };

        let snapshot_id = self.next_id();
        self.contexts.insert(snapshot_id, Context {
            owner: None,
            device: Some(dev_idx as DeviceId),
            working_pages: snapshot_working,
            suspended_working_count: 0,
            committed_hashes: committed_hashes.clone(),
            lineage,
            working_page_tokens: snapshot_filled,
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
        });
        self.snapshots.insert((username, name.clone()), snapshot_id);

        // If snapshot ended up Suspended (CPU fallback for working pages),
        // release the refcounts we acquired. Suspended invariant: no held refcounts.
        // On open/fork, retain will be called to re-acquire.
        if snapshot_state == State::Stashed && !committed_hashes.is_empty() {
            self.gpu_stores[dev_idx].release(&committed_hashes);
        }

        Ok(if auto_generated { Some(name) } else { None })
    }

    pub(crate) fn delete(&mut self, username: String, name: String) -> Result<()> {
        let snapshot_id = self.snapshots.remove(&(username, name))
            .ok_or_else(|| anyhow::anyhow!("Snapshot not found"))?;

        if let Some(ctx) = self.contexts.remove(&snapshot_id) {
            let dev_idx = ctx.device.unwrap_or(0) as usize;
            if !ctx.committed_hashes.is_empty() && !ctx.is_off_gpu() {
                self.gpu_stores[dev_idx].release(&ctx.committed_hashes);
            }
            // Free snapshot working pages (GPU or CPU depending on state)
            if ctx.is_stashed() {
                self.cpu_stores[dev_idx].free(&ctx.working_pages);
            } else {
                self.gpu_stores[dev_idx].free(&ctx.working_pages);
            }
        }

        Ok(())
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
        let (needed, dev_idx, snap_ctx_id) = match self.snapshots.get(&key) {
            Some(&snap_id) => match self.contexts.get(&snap_id) {
                Some(snap) if snap.is_off_gpu() => {
                    let dev_idx = snap.device.unwrap_or(0) as usize;
                    let working = snap.working_pages.len();
                    let suffix = if !snap.committed_hashes.is_empty() {
                        let prefix = self.gpu_stores[dev_idx].prefix_len(&snap.committed_hashes);
                        snap.committed_hashes.len() - prefix
                    } else {
                        0
                    };
                    (working + suffix, dev_idx, snap_id)
                }
                Some(snap) => (0, snap.device.unwrap_or(0) as usize, snap_id),
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
        if let Some(pages) = self.gpu_stores[dev_idx].alloc(needed) {
            let result = self.take_inner(username, name, owner, pages);
            let _ = response.send(result);
            return;
        }

        // Not enough free pages — defer on the snapshot context.
        let pending = super::sched::PendingAlloc {
            device: dev_idx,
            num_pages: needed,
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
        let snapshot_id = *self.snapshots.get(&key)
            .ok_or_else(|| anyhow::anyhow!("Snapshot not found"))?;

        let snap = self.contexts.get(&snapshot_id)
            .ok_or_else(|| anyhow::anyhow!("Snapshot context missing"))?;

        let dev_idx = snap.device.unwrap_or(0) as usize;
        let was_off_gpu = snap.is_off_gpu();

        // Determine committed chain restoration needs.
        let (prefix_len, suffix_count) = if was_off_gpu && !snap.committed_hashes.is_empty() {
            let prefix = self.gpu_stores[dev_idx].prefix_len(&snap.committed_hashes);
            (prefix, snap.committed_hashes.len() - prefix)
        } else {
            (snap.committed_hashes.len(), 0)
        };

        let working_count = snap.working_pages.len();
        let total_needed = working_count + suffix_count;

        // Validate pre-allocated page count against actual needs.
        if gpu_pages.len() < total_needed {
            self.gpu_stores[dev_idx].free(&gpu_pages);
            anyhow::bail!(
                "take: insufficient GPU pages (got {}, need {total_needed})",
                gpu_pages.len()
            );
        }

        // Validation passed — consume the snapshot (point of no return).
        let snap = self.contexts.remove(&snapshot_id).unwrap();
        self.snapshots.remove(&key);

        // Split pre-allocated pages: working | suffix | surplus.
        let (working_gpu, rest) = gpu_pages.split_at(working_count);
        let suffix_pages = rest[..suffix_count].to_vec();
        let surplus = rest[suffix_count..].to_vec();
        let working_gpu = working_gpu.to_vec();
        if !surplus.is_empty() {
            self.gpu_stores[dev_idx].free(&surplus);
        }

        // Working pages: GPU pages transfer directly, CPU pages use pre-allocated pages.
        let new_working = if snap.working_pages.is_empty() {
            Vec::new()
        } else if was_off_gpu {
            let _ = device::copy_h2d(dev_idx as DeviceId, &working_gpu, &snap.working_pages);
            self.cpu_stores[dev_idx].free(&snap.working_pages);
            working_gpu
        } else {
            // GPU → new context: direct ownership transfer (zero-copy).
            if !working_gpu.is_empty() {
                self.gpu_stores[dev_idx].free(&working_gpu);
            }
            snap.working_pages
        };

        // Committed chain refcount handling.
        if was_off_gpu && prefix_len > 0 {
            self.gpu_stores[dev_idx].fork(&snap.committed_hashes[..prefix_len]);
        }

        let new_id = self.next_id();
        self.contexts.insert(new_id, Context {
            owner: Some(owner),
            device: Some(dev_idx as DeviceId),
            working_pages: new_working,
            suspended_working_count: 0,
            committed_hashes: snap.committed_hashes,
            lineage: snap.lineage,
            working_page_tokens: snap.working_page_tokens,
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
        });

        // Spawn replay for committed suffix if needed.
        let has_replay = if suffix_count > 0 {
            self.spawn_replay_passes(new_id, dev_idx, prefix_len, suffix_pages)?
        } else {
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
