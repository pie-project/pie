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
    /// utility is at or below `floor_utility`.
    ///
    /// Tie-breaking for equal-priority nodes:
    ///   1. Prefer the node holding the **most GPU pages** (spread eviction)
    ///   2. Within the same node, prefer the **oldest** context (LRU)
    ///
    /// Unowned contexts (snapshots, orphans) have 0 priority via the
    /// fallback heuristic (no arbiter node), so they are naturally
    /// the cheapest victims — no separate tier needed.
    pub(crate) fn find_cheapest_victim(&self, dev_idx: usize, floor_utility: f64) -> Option<ContextId> {
        // (id, priority, node_pages, last_access)
        let mut best: Option<(ContextId, f64, usize, Instant)> = None;

        for entry in CONTEXTS.iter() {
            let &(model_idx, ctx_id) = entry.key();
            if model_idx != self.model_idx { continue; }
            let ctx = entry.value();
            if ctx.state != ContextState::Active { continue; }
            if !ctx.has_gpu_pages() { continue; }
            if ctx.device != Some(dev_idx as DeviceId) { continue; }

            let (priority, node_pages) = ctx.owner
                .map(|pid| {
                    (self.arbiter.priority(&pid, dev_idx), self.arbiter.node_pages(&pid, dev_idx))
                })
                .unwrap_or((0.0, 0));

            if priority >= floor_utility { continue; }

            let dominated = match &best {
                None => true,
                Some((_, best_u, best_gp, best_t)) => {
                    if priority < *best_u {
                        true
                    } else if (priority - *best_u).abs() < 1e-9 {
                        if node_pages > *best_gp {
                            true
                        } else if node_pages == *best_gp {
                            ctx.last_access < *best_t
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }
            };

            if dominated {
                best = Some((ctx_id, priority, node_pages, ctx.last_access));
            }
        }

        best.map(|(id, _, _, _)| id)
    }

    pub(crate) async fn suspend_context(&mut self, id: ContextId) {
        let (working, tip, dev_idx) = {
            let ctx = match CONTEXTS.get(&(self.model_idx, id)) {
                Some(ctx) => ctx, None => return,
            };
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

    // ==================== Residency Restoration ====================

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

            if let Some(plan) = self.build_replay_plan(id, dev_idx) {
                // Restore arbiter accounting for prefix-matched committed pages
                // (Bug 1 fix: only prefix-matched, not the full committed_len).
                // build_replay_plan has set ctx.committed_len = matched_pages.
                if let Some(pid) = owner {
                    let matched_committed = CONTEXTS.get(&(self.model_idx, id))
                        .map(|c| c.committed_len).unwrap_or(0);
                    self.arbiter.restore(pid, dev_idx, matched_committed, 0);
                }

                let chunks = self.execute_replay(id, dev_idx, plan, owner).await?;
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
                ctx.state = ContextState::Active;
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
    fn build_replay_plan(&mut self, id: ContextId, dev_idx: usize) -> Option<ReplayPlan> {
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
        let matched_pages = if page_aligned > 0 {
            let hashes = self.devices[dev_idx].compute_page_hashes(
                &all_tokens[..page_aligned],
                &all_positions[..page_aligned],
                &all_masks[..page_aligned],
                0,
            );
            self.devices[dev_idx].longest_prefix_match(&hashes)
        } else {
            0
        };

        let matched_tokens = matched_pages * self.page_size;
        let kv_so_far = matched_tokens as u32;

        if matched_pages > 0 {
            let hashes = self.devices[dev_idx].compute_page_hashes(
                &all_tokens[..matched_tokens],
                &all_positions[..matched_tokens],
                &all_masks[..matched_tokens],
                0,
            );
            let new_tip = *hashes.last().unwrap();
            let _phys = self.devices[dev_idx].resolve_physical(new_tip);
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                ctx.committed_tip = Some(new_tip);
                ctx.committed_len = matched_pages;
            }
        } else {
            if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
                ctx.committed_tip = None;
                ctx.committed_len = 0;
            }
        }

        if matched_tokens >= all_tokens.len() {
            return None;
        }

        Some(ReplayPlan { all_tokens, all_positions, all_masks, adapters, matched_tokens, kv_so_far })
    }

    /// Phase 2b: build replay chunks for tokens after the matched prefix.
    async fn execute_replay(
        &mut self, id: ContextId, dev_idx: usize,
        plan: ReplayPlan, owner: Option<ProcessId>,
    ) -> Result<Vec<ReplayFill>, WaitNeeded> {
        let ReplayPlan { all_tokens, all_positions, all_masks, adapters, matched_tokens, mut kv_so_far } = plan;

        let mut chunks = Vec::new();
        let mut offset = matched_tokens;

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
            let new_pages = self.allocate_working_with_suspension(dev_idx, num_pages, owner).await?;

            {
                let mut ctx = CONTEXTS.get_mut(&(self.model_idx, id))
                    .ok_or_else(|| anyhow::anyhow!("Context lost during replay"))?;
                ctx.working_pages.extend(&new_pages);
                ctx.tokens_buffered = chunk_tokens.to_vec();
                ctx.fill(chunk_tokens.len(), chunk_positions.to_vec(), chunk_masks.to_vec(), adapter)?;
            }
            if let Some(pid) = owner {
                self.arbiter.add_working(pid, dev_idx, num_pages);
            }

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

    pub(crate) fn commit_replay_chunk(&mut self, id: ContextId, num_pages: u32) -> Result<()> {
        if num_pages == 0 { return Ok(()); }
        let page_indices: Vec<u32> = (0..num_pages).collect();
        self.commit_pages(id, page_indices)
    }

    pub(crate) fn finish_restore(&mut self, id: ContextId) {
        if let Some(mut ctx) = CONTEXTS.get_mut(&(self.model_idx, id)) {
            ctx.state = ContextState::Active;
            // Arbiter accounting is handled by ensure_resident — no call here.
        }
    }
}
