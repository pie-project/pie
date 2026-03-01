//! Suspension, eviction, and residency restoration for contexts.
//!
//! This module handles the lifecycle of GPU residency: suspending contexts
//! to CPU, restoring them from CPU, and replaying committed chains.

use std::collections::HashMap;
use std::time::Instant;
use anyhow::Result;
use serde::Serialize;

use crate::adapter::AdapterId;
use crate::process::ProcessId;
use crate::device::{self, DeviceId};
use crate::inference::brle::Brle;

use super::{CONTEXTS, Context, ContextId, ContextState, Record, ReplayFill};
use super::kvcache::{self, PhysicalPageId, PageHash};
use super::manager::ContextManager;
use super::waitqueue::WaitNeeded;

// =============================================================================
// ReplayPlan — intermediate data for ensure_resident decomposition
// =============================================================================

/// Computed by `build_replay_plan`, consumed by `execute_replay`.
struct ReplayPlan {
    all_tokens: Vec<u32>,
    all_positions: Vec<u32>,
    all_masks: Vec<Brle>,
    adapters: Vec<(usize, Option<AdapterId>)>,
    matched_tokens: usize,
    kv_so_far: u32,
}

// =============================================================================
// Eviction & Suspension
// =============================================================================

impl ContextManager {
    /// Find the cheapest active context on a given device whose group
    /// utility is below `floor_utility`, or equal with a later creation time
    /// (FCFS tiebreaker: older processes are harder to evict).
    ///
    /// Tie-breaking among candidates:
    ///   1. Lowest priority first
    ///   2. At equal priority: newer process first (FCFS — older wins)
    ///   3. At equal priority & age: most GPU pages first (spread eviction)
    ///   4. Within same node: oldest context access (LRU)
    ///
    /// Unowned contexts (snapshots, orphans) have 0 priority via the
    /// fallback heuristic (no arbiter node), so they are naturally
    /// the cheapest victims — no separate tier needed.
    pub(crate) fn find_cheapest_victim(
        &self, dev_idx: usize, floor_utility: f64, requester: Option<ProcessId>,
    ) -> Option<ContextId> {
        let requester_birth = requester
            .and_then(|pid| self.arbiter.node_created_at(&pid));

        // (id, priority, birth, node_pages, last_access)
        let mut best: Option<(ContextId, f64, Option<Instant>, usize, Instant)> = None;

        for entry in CONTEXTS.iter() {
            let &(model_idx, ctx_id) = entry.key();
            if model_idx != self.model_idx { continue; }
            let ctx = entry.value();
            if ctx.state != ContextState::Active { continue; }
            if !ctx.has_gpu_pages() { continue; }
            if ctx.device != Some(dev_idx as DeviceId) { continue; }
            // Skip contexts with in-flight working pages when CPU swap
            // cannot offload them — eviction would lose uncommitted KV data.
            if !ctx.working_pages.is_empty()
                && !self.devices[dev_idx].can_swap_working(ctx.working_pages.len())
            {
                continue;
            }

            let (priority, node_pages) = ctx.owner
                .map(|pid| {
                    (self.arbiter.priority(&pid, dev_idx), self.arbiter.node_pages(&pid, dev_idx))
                })
                .unwrap_or((0.0, 0));

            // Skip victims with strictly higher priority.
            // Equal priority victims ARE evictable — the anti-thrashing
            // invariant still holds: after R evicts V, R has p_R+n pages
            // while V has 0. V's post-allocation floor = w*(0+n) ≤ w*(p_R+n)
            // = R's new priority, so V cannot evict R back.
            if priority > floor_utility + 1e-9 { continue; }

            let victim_birth = ctx.owner
                .and_then(|pid| self.arbiter.node_created_at(&pid));

            let dominated = match &best {
                None => true,
                Some((_, best_u, best_birth, best_gp, best_t)) => {
                    if priority < *best_u {
                        true
                    } else if (priority - *best_u).abs() < 1e-9 {
                        // Among equal-priority candidates, prefer newest (most recently created)
                        match (victim_birth, best_birth) {
                            (Some(v), Some(b)) if v > *b => true,
                            (Some(v), Some(b)) if v < *b => false,
                            _ => {
                                // Same birth or unknown: fall back to page count / LRU
                                if node_pages > *best_gp {
                                    true
                                } else if node_pages == *best_gp {
                                    ctx.last_access < *best_t
                                } else {
                                    false
                                }
                            }
                        }
                    } else {
                        false
                    }
                }
            };

            if dominated {
                best = Some((ctx_id, priority, victim_birth, node_pages, ctx.last_access));
            }
        }

        best.map(|(id, _, _, _, _)| id)
    }

    pub(crate) async fn suspend_context(&mut self, id: ContextId) {
        let (working, tip, dev_idx) = {
            let ctx = match CONTEXTS.get(&(self.model_idx, id)) {
                Some(ctx) => ctx, None => return,
            };
            // Skip if context transitioned to InFlight between victim selection and now.
            if ctx.state == ContextState::InFlight { return; }
            if !ctx.has_gpu_pages() { return; }
            (ctx.working_pages.clone(), ctx.committed_tip, ctx.device.unwrap_or(0) as usize)
        };

        // Phase 1: Swap working pages to CPU (or discard if unavailable)
        let swap_ops = if !working.is_empty() {
            match self.devices[dev_idx].swap_out_working(&working) {
                Ok(ops) => ops,
                Err(e) => {
                    tracing::warn!("Cannot swap working pages for {id}, discarding (will replay): {e}");
                    self.devices[dev_idx].free_working(&working);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        // Phase 2: Fire-and-forget D2H copy RPC — actor doesn't block.
        // If the copy fails, restore will fall back to replay.
        if !swap_ops.is_empty() {
            #[derive(Serialize)]
            struct SwapOutRequest { phys_ids: Vec<u32>, slots: Vec<PhysicalPageId> }
            let request = SwapOutRequest {
                phys_ids: swap_ops.iter().map(|op| op.gpu_phys).collect(),
                slots: swap_ops.iter().map(|op| op.cpu_slot).collect(),
            };
            let dev = dev_idx as DeviceId;
            tokio::spawn(async move {
                let _: Result<(), _> = device::call(dev, "swap_out_pages", &request).await;
            });
        }

        // Phase 3: Release committed chain refcounts
        if let Some(tip_hash) = tip {
            let dev = &mut self.devices[dev_idx];
            dev.release_chain(tip_hash);
            dev.remove_index_cache(tip_hash);
            dev.evict_unreferenced();
        }

        // Update context state
        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
            ctx.working_cpu_slots = swap_ops.iter().map(|op| op.cpu_slot).collect();
            ctx.working_pages.clear();
            ctx.state = ContextState::Suspended;
            ctx.last_access = Instant::now();

            let owner = ctx.owner;
            let committed_len = ctx.committed_len;
            drop(ctx);
            if let Some(pid) = owner {
                self.arbiter.suspend(pid, dev_idx, committed_len, swap_ops.len());
            }
        }

        tracing::info!("Suspended context {} on model {}", id, self.model_idx);
    }

    /// Resolve physical page IDs and atomically pin the context as InFlight.
    ///
    /// This runs inside the actor, serialized with eviction operations.
    /// By setting InFlight here, we guarantee no eviction can happen between
    /// page resolution and the forward pass completing (cleared by clear_in_flight).
    pub(crate) fn get_physical_page_ids(&mut self, id: ContextId) -> Result<HashMap<DeviceId, Vec<PhysicalPageId>>> {
        let ctx = self.ctx(id)?;
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let mut phys = if let Some(tip) = ctx.committed_tip {
            drop(ctx);
            self.devices[dev_idx].resolve_physical(tip)
        } else {
            drop(ctx);
            Vec::new()
        };

        if let Some(ctx) = CONTEXTS.get(&(self.model_idx, id)) {
            phys.extend_from_slice(&ctx.working_pages);
        }

        let mut result = HashMap::new();
        if !phys.is_empty() {
            result.insert(dev_idx as DeviceId, phys);
        }
        Ok(result)
    }

    pub(crate) async fn ensure_resident(&mut self, id: ContextId) -> Result<Option<Vec<ReplayFill>>, WaitNeeded> {
        let ctx = self.ctx(id).map_err(WaitNeeded::Fatal)?;
        let tip = ctx.committed_tip;
        let working_cpu = ctx.working_cpu_slots.clone();
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let was_suspended = ctx.state == ContextState::Suspended;
        let owner = ctx.owner;
        drop(ctx);

        // Fast path: not suspended, no work to do
        if !was_suspended && working_cpu.is_empty() {
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                ctx.last_access = Instant::now();
            }
            return Ok(None);
        }

        // Phase 1: Swap working pages from CPU back to GPU
        if !working_cpu.is_empty() {
            self.restore_working_pages(id, dev_idx, &working_cpu, owner).await?;
        }

        // Phase 2: Ensure committed chain is resident
        if let Some(tip_hash) = tip {
            let (_, discarded) = self.devices[dev_idx].classify_chain(tip_hash);

            if discarded.is_empty() {
                if was_suspended {
                    self.devices[dev_idx].acquire_chain(tip_hash);
                    // Restore arbiter accounting for committed pages (Bug 2 fix).
                    if let Some(pid) = owner {
                        let committed_len = CONTEXTS.get(&(self.model_idx, id))
                            .map(|c| c.committed_len).unwrap_or(0);
                        self.arbiter.restore(pid, dev_idx, committed_len, 0);
                    }
                }
                if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                    ctx.state = ContextState::Active;
                    ctx.last_access = Instant::now();
                }
                return Ok(None);
            }

            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                ctx.state = ContextState::Restoring;
            }

            if let Some((plan, prefix_hashes)) = self.build_replay_plan(id, dev_idx) {
                // execute_replay is self-cleaning: on failure it frees
                // partially-allocated working pages and resets to Suspended.
                let chunks = self.execute_replay(id, dev_idx, plan, owner).await?;

                // Replay succeeded — now commit the restore atomically:
                // acquire refcounts for the prefix-matched pages and
                // restore arbiter accounting.
                if !prefix_hashes.is_empty() {
                    self.devices[dev_idx].longest_prefix_match(&prefix_hashes);
                }
                if let Some(pid) = owner {
                    let matched_committed = CONTEXTS.get(&(self.model_idx, id))
                        .map(|c| c.committed_len).unwrap_or(0);
                    self.arbiter.restore(pid, dev_idx, matched_committed, 0);
                }

                if chunks.is_empty() {
                    if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                        ctx.state = ContextState::Active;
                    }
                    return Ok(None);
                }
                return Ok(Some(chunks));
            }

            // All pages prefix-matched, no replay needed. (Bug 3 fix)
            if let Some(pid) = owner {
                let committed_len = CONTEXTS.get(&(self.model_idx, id))
                    .map(|c| c.committed_len).unwrap_or(0);
                self.arbiter.restore(pid, dev_idx, committed_len, 0);
            }
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                ctx.state = ContextState::InFlight;
            }
            return Ok(None);
        }

        // No committed chain
        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
            ctx.state = ContextState::Active;
            ctx.last_access = Instant::now();
        }
        Ok(None)
    }

    /// Phase 1: swap working pages from CPU back to GPU.
    async fn restore_working_pages(
        &mut self, id: ContextId, dev_idx: usize,
        working_cpu: &[PhysicalPageId], owner: Option<ProcessId>,
    ) -> Result<(), WaitNeeded> {
        let swap_ops = match self.devices[dev_idx].swap_in_working(working_cpu) {
            Ok(ops) => ops,
            Err(_) => {
                let needed = working_cpu.len();
                let gpu_pages = self.allocate_working_with_suspension(dev_idx, needed, owner).await?;
                self.devices[dev_idx].free_working(&gpu_pages);
                self.devices[dev_idx].swap_in_working(working_cpu)?
            }
        };

        if !swap_ops.is_empty() {
            #[derive(Serialize)]
            struct SwapInRequest { phys_ids: Vec<u32>, slots: Vec<PhysicalPageId> }
            let request = SwapInRequest {
                phys_ids: swap_ops.iter().map(|op| op.gpu_phys).collect(),
                slots: swap_ops.iter().map(|op| op.cpu_slot).collect(),
            };
            let _: () = device::call(dev_idx as DeviceId, "swap_in_pages", &request).await
                .map_err(|e| anyhow::anyhow!("swap_in_pages RPC failed: {e}"))?;

            let new_working: Vec<_> = swap_ops.iter().map(|op| op.gpu_phys).collect();
            let swapped_count = new_working.len();
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                ctx.working_pages = new_working;
                ctx.working_cpu_slots.clear();
            }
            if let Some(pid) = owner {
                self.arbiter.add_working(pid, dev_idx, swapped_count);
            }
        }
        Ok(())
    }

    /// Phase 2a: flatten lineage, compute hashes, find prefix.
    ///
    /// Returns `(ReplayPlan, prefix_hashes)`. The prefix hashes are returned
    /// so the caller can acquire refcounts AFTER successful replay (transactional).
    /// `build_replay_plan` itself uses `longest_prefix_length` (read-only) to
    /// avoid refcount side-effects that would need rollback on failure.
    fn build_replay_plan(&mut self, id: ContextId, dev_idx: usize) -> Option<(ReplayPlan, Vec<PageHash>)> {
        let lineage = CONTEXTS.get(&(self.model_idx, id))
            .map(|ctx| ctx.lineage.clone()).unwrap_or_default();

        let mut all_tokens = Vec::new();
        let mut all_positions = Vec::new();
        let mut all_masks = Vec::new();
        let mut adapters = Vec::new();

        for record in &lineage {
            match record {
                Record::Fill { tokens, positions, mask, adapter } => {
                    adapters.push((all_tokens.len(), *adapter));
                    all_tokens.extend_from_slice(tokens);
                    all_positions.extend_from_slice(positions);
                    all_masks.extend_from_slice(mask);
                }
            }
        }

        let page_aligned = (all_tokens.len() / self.page_size) * self.page_size;
        let (matched_pages, all_hashes) = if page_aligned > 0 {
            let hashes = self.devices[dev_idx].compute_page_hashes(
                &all_tokens[..page_aligned],
                &all_positions[..page_aligned],
                &all_masks[..page_aligned],
                0,
            );
            // Read-only prefix check — no refcount side-effects.
            let matched = self.devices[dev_idx].longest_prefix_length(&hashes);
            (matched, hashes)
        } else {
            (0, Vec::new())
        };

        let matched_tokens = matched_pages * self.page_size;
        let kv_so_far = matched_tokens as u32;

        // Prefix hashes for deferred refcount acquisition (caller acquires
        // after successful replay to keep the restore transactional).
        let prefix_hashes: Vec<PageHash> = all_hashes[..matched_pages].to_vec();

        if matched_pages > 0 {
            let prefix_hash_slice = &all_hashes[..matched_pages];
            let Some(&new_tip) = prefix_hash_slice.last() else { return None; };
            // Populate index_cache for the prefix tip (side-effect of resolve_physical).
            let _phys = self.devices[dev_idx].resolve_physical(new_tip);
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                ctx.committed_tip = Some(new_tip);
                ctx.committed_len = matched_pages;
                // Reset max_committed_position to match the prefix we recovered.
                // Without this, replay's commit_pages rejects positions that are
                // <= the pre-eviction max_committed_position.
                ctx.max_committed_position = all_positions[..matched_tokens].iter().copied().max();
            }
        } else {
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                ctx.committed_tip = None;
                ctx.committed_len = 0;
                ctx.max_committed_position = None;
            }
        }

        if matched_tokens >= all_tokens.len() {
            return None;
        }

        Some((ReplayPlan { all_tokens, all_positions, all_masks, adapters, matched_tokens, kv_so_far }, prefix_hashes))
    }

    /// Phase 2b: build replay chunks for tokens after the matched prefix.
    ///
    /// Self-cleaning on failure: if allocation fails midway, frees any
    /// working pages allocated during this call and resets state to
    /// `Suspended` for a clean retry.
    async fn execute_replay(
        &mut self, id: ContextId, dev_idx: usize,
        plan: ReplayPlan, owner: Option<ProcessId>,
    ) -> Result<Vec<ReplayFill>, WaitNeeded> {
        let ReplayPlan { all_tokens, all_positions, all_masks, adapters, matched_tokens, mut kv_so_far } = plan;

        // Track how many working pages existed before this call so we
        // can rollback exactly the pages WE allocated on failure.
        let initial_working_len = CONTEXTS.get(&(self.model_idx, id))
            .map(|c| c.working_pages.len()).unwrap_or(0);

        let mut chunks = Vec::new();
        let mut offset = matched_tokens;
        let mut pages_added_to_arbiter = 0;

        while offset < all_tokens.len() {
            let adapter = adapters.iter().rev()
                .find(|(start, _)| *start <= offset)
                .and_then(|(_, a)| *a);

            let next_adapter_start = adapters.iter()
                .find(|(start, _)| *start > offset)
                .map(|(start, _)| *start)
                .unwrap_or(all_tokens.len());

            let chunk_end = next_adapter_start;
            let chunk_tokens = &all_tokens[offset..chunk_end];
            let chunk_positions = &all_positions[offset..chunk_end];
            let chunk_masks = &all_masks[offset..chunk_end];

            let num_pages = (chunk_tokens.len() + self.page_size - 1) / self.page_size;
            let new_pages = match self.allocate_working_with_suspension(dev_idx, num_pages, owner).await {
                Ok(pages) => pages,
                Err(e) => {
                    // Rollback: free working pages we allocated in THIS call.
                    self.rollback_replay(id, dev_idx, owner, initial_working_len, pages_added_to_arbiter);
                    return Err(e);
                }
            };

            {
                let mut ctx = CONTEXTS.get_mut(&(self.model_idx, id))
                    .ok_or_else(|| anyhow::anyhow!("Context lost during replay"))?;
                ctx.working_pages.extend(&new_pages);
                // NOTE: Do NOT touch tokens_buffered or tokens_filled here.
                // The inferlet may have pending data in these buffers from an
                // in-flight operation. Replay data goes directly into ReplayFill.
            }
            if let Some(pid) = owner {
                self.arbiter.add_working(pid, dev_idx, num_pages);
            }
            pages_added_to_arbiter += num_pages;

            let phys_ids = {
                let mut all = if let Some(tip) = CONTEXTS.get(&(self.model_idx, id))
                    .and_then(|c| c.committed_tip) {
                    self.devices[dev_idx].resolve_physical(tip)
                } else { Vec::new() };
                if let Some(ctx) = CONTEXTS.get(&(self.model_idx, id)) {
                    all.extend_from_slice(&ctx.working_pages);
                }
                all
            };

            let num_input = chunk_tokens.len() as u32;
            let total_kv = kv_so_far + num_input;
            let total_pages = phys_ids.len() as u32;
            let last_page_len = kvcache::compute_last_page_len(total_kv, total_pages, self.page_size as u32);

            chunks.push(ReplayFill {
                tokens: chunk_tokens.to_vec(),
                positions: chunk_positions.to_vec(),
                masks: chunk_masks.to_vec(),
                adapter,
                physical_page_ids: phys_ids,
                device_id: dev_idx as DeviceId,
                kv_len: kv_so_far,
                last_page_len,
                num_pages: num_pages as u32,
            });
            kv_so_far += num_input;
            offset = chunk_end;
        }

        Ok(chunks)
    }

    /// Rollback working pages allocated during a failed `execute_replay`.
    /// Frees the pages, undoes arbiter accounting, and resets state to Suspended.
    fn rollback_replay(
        &mut self, id: ContextId, dev_idx: usize,
        owner: Option<ProcessId>, initial_working_len: usize, arbiter_pages: usize,
    ) {
        let to_free = {
            let mut ctx = match CONTEXTS.get_mut(&(self.model_idx, id)) {
                Some(ctx) => ctx,
                None => return,
            };
            let current_len = ctx.working_pages.len();
            if current_len <= initial_working_len {
                ctx.state = ContextState::Suspended;
                return;
            }
            let pages: Vec<PhysicalPageId> = ctx.working_pages.drain(initial_working_len..).collect();
            ctx.state = ContextState::Suspended;
            pages
        };

        self.devices[dev_idx].free_working(&to_free);
        if let Some(pid) = owner {
            self.arbiter.remove_working(pid, dev_idx, arbiter_pages);
        }
    }

    /// Commit a replay chunk using provided data directly, without touching tokens_filled.
    /// This avoids the race where replay overwrites the inferlet's in-flight buffer data.
    pub(crate) fn commit_replay_chunk(
        &mut self, id: ContextId, num_pages: u32,
        tokens: Vec<u32>, positions: Vec<u32>, masks: Vec<Brle>, adapter: Option<AdapterId>,
    ) -> Result<()> {
        if num_pages == 0 { return Ok(()); }

        let page_size = self.page_size;
        let ctx = self.ctx(id)?;

        let prev_hash = ctx.committed_tip.unwrap_or(0);
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let old_tip = ctx.committed_tip;

        let working_phys: Vec<PhysicalPageId> = (0..num_pages as usize)
            .map(|i| ctx.working_pages[i])
            .collect();

        drop(ctx);

        let hashes = self.devices[dev_idx].compute_page_hashes(&tokens, &positions, &masks, prev_hash);

        let mut new_phys = Vec::new();
        let mut running_prev = prev_hash;
        for (i, &hash) in hashes.iter().enumerate() {
            let (phys, _freed) = self.devices[dev_idx].commit_working_page(hash, running_prev, working_phys[i]);
            new_phys.push(phys);
            running_prev = hash;
        }

        let new_tip = *hashes.last()
            .ok_or_else(|| anyhow::anyhow!("No page hashes computed during replay commit"))?;
        self.devices[dev_idx].update_index_cache(new_tip, old_tip, &new_phys);

        let owner = {
            let mut ctx = self.ctx_mut(id)
                .map_err(|_| anyhow::anyhow!("Context lost during replay commit"))?;

            // Remove working pages that were committed
            let mut sorted_indices: Vec<usize> = (0..num_pages as usize).collect();
            sorted_indices.sort_unstable();
            for &idx in sorted_indices.iter().rev() {
                ctx.working_pages.remove(idx);
            }

            ctx.committed_tip = Some(new_tip);
            ctx.committed_len += num_pages as usize;
            ctx.max_committed_position = positions.iter().copied().max()
                .or(ctx.max_committed_position);

            // NOTE: Do NOT append to lineage here. The replay data already
            // exists in the lineage — that's where build_replay_plan read it
            // from. Appending again would duplicate it on every evict/restore cycle.

            ctx.owner
        };

        if let Some(pid) = owner {
            self.arbiter.commit(pid, dev_idx, num_pages as usize);
        }

        Ok(())
    }

    pub(crate) fn finish_restore(&mut self, id: ContextId) {
        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
            // Restoring → InFlight: skip the Active window entirely.
            // The process will call clear_in_flight after inference::submit.
            ctx.state = ContextState::InFlight;
        }
    }
}
