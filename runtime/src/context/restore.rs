//! Restoration — Context Recovery and Replay Planning.
//!
//! When a Suspended context wins the `restore_queue`, this module brings it
//! back to Active state.
//!
//! ## `restore(ctx_id)` — Single Context Recovery
//!
//! Three phases:
//! 1. **Working re-alloc**: alloc GPU pages for working data (recomputed via replay).
//! 2. **Prefix match**: `prefix_len(committed_hashes)` → acquire refcounts
//!    for GPU-resident prefix via `retain`.
//! 3. **Replay suffix**: alloc fresh GPU pages for missing suffix, register
//!    in PageStore via `extend`, spawn replay forward passes. Context
//!    stays Pinned until replay completes (`replay_complete`).
//!
//! **Invariant**: restoration never evicts. The admission check in
//! `can_restore` guarantees sufficient pages before `restore` is called.
//!
//! ## `replay_complete(ctx_id)` — Post-Replay Transition
//!
//! Pinned → Active. Once the replay pass completes, deferred ops fire.
//! If the context was re-suspended mid-replay (`pending_suspend`), it is
//! re-suspended instead.

use super::{
    ContextId, ContextManager, State, SERVICES,
    Record,
};
use super::pagestore::{PhysicalPageId, compute_last_page_len};
use crate::inference;
use crate::inference::request::ForwardPassRequest;
use crate::device::{self, DeviceId};

// =============================================================================
// Restore methods on ContextManager
// =============================================================================


impl ContextManager {

    /// Admission check: can this context be restored?
    /// Checks that the device has enough free GPU pages for the context's
    /// working pages (recomputed) plus replay pages plus deferred alloc requirements.
    pub(crate) fn can_restore(&mut self, ctx_id: ContextId) -> bool {
        let ctx = match self.contexts.get(&ctx_id) {
            Some(c) if c.is_off_gpu() => c,
            _ => return false,
        };

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let mut required = ctx.suspended_working_count;

        // Pages needing replay: check prefix match, count missing suffix
        if !ctx.committed_hashes.is_empty() {
            let prefix_len = self.gpu_stores[dev_idx].prefix_len(&ctx.committed_hashes);
            let replay_pages = ctx.committed_hashes.len().saturating_sub(prefix_len);
            required += replay_pages;
        }

        // Include deferred ops: these fire immediately after restore via
        // fire_deferred_ops. If the pool can't satisfy them, the context
        // would re-suspend immediately — wasting the restore work.
        let deferred_pages: usize = ctx.deferred_ops.iter().map(|op| op.num_pages).sum();
        required += deferred_pages;

        if self.gpu_stores[dev_idx].available() < required {
            return false;
        }

        true
    }

    /// Restore a single suspended context.
    ///
    /// Phase 1: Allocate fresh GPU pages for working data (recomputed via replay)
    /// Phase 2: Acquire refcounts for GPU-resident committed prefix
    /// Phase 3: Eagerly promote suffix pages and spawn replay forward passes
    ///          (includes working token replay after committed suffix)
    ///
    /// After this function returns, the context is fully restored from the
    /// metadata perspective. Forward passes fill KV data in the background
    /// while the context is Pinned.
    ///
    /// If no replay is needed, deferred ops fire immediately.
    pub(crate) fn restore(&mut self, ctx_id: ContextId) -> anyhow::Result<()> {
        let ctx = self.contexts.get(&ctx_id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;

        if !ctx.is_off_gpu() {
            return Ok(());
        }

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let working_count = ctx.suspended_working_count;
        let committed_hashes = ctx.committed_hashes.clone();
        let cpu_working_pages = ctx.cpu_working_pages.clone();

        // Phase 1: Restore working pages.
        // If CPU-stashed, H2D copy; otherwise allocate fresh for replay.
        if working_count > 0 {
            let gpu_pages = self.gpu_stores[dev_idx].alloc(working_count)
                .ok_or_else(|| anyhow::anyhow!("No free GPU pages for working re-alloc"))?;

            if !cpu_working_pages.is_empty() && cpu_working_pages.len() == working_count {
                // H2D copy from CPU stash.
                let _ = device::copy_h2d(dev_idx as DeviceId, &gpu_pages, &cpu_working_pages);
                self.cpu_stores[dev_idx].free(&cpu_working_pages);
            }

            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.working_pages = gpu_pages;
                ctx.suspended_working_count = 0;
                ctx.cpu_working_pages.clear();
            }
        }

        // Phase 2: Acquire refcounts for GPU-resident committed prefix.
        let prefix_len = if !committed_hashes.is_empty() {
            let dev = &mut self.gpu_stores[dev_idx];
            let prefix_len = dev.prefix_len(&committed_hashes);
            if prefix_len > 0 {
                dev.fork(&committed_hashes[..prefix_len]);
            }
            prefix_len
        } else {
            0
        };

        // Phase 3: Suffix restoration.
        // Check CPU store for suffix pages before falling back to replay.
        let suffix_count = committed_hashes.len().saturating_sub(prefix_len);
        let suffix_hashes = &committed_hashes[prefix_len..];

        let has_replay = if suffix_count > 0 {
            // Check how many suffix pages are on CPU.
            let cpu_prefix = self.cpu_stores[dev_idx].prefix_len(suffix_hashes);

            if cpu_prefix > 0 {
                // CPU-warm restore: H2D copy for CPU-resident portion.
                let cpu_hashes = &suffix_hashes[..cpu_prefix];
                let cpu_phys = self.cpu_stores[dev_idx].physical_ids(cpu_hashes);

                let gpu_pages = self.gpu_stores[dev_idx].alloc(cpu_prefix)
                    .ok_or_else(|| anyhow::anyhow!("No GPU pages for CPU-cache restore"))?;

                let _ = device::copy_h2d(dev_idx as DeviceId, &gpu_pages, &cpu_phys);

                // Register in GPU trie.
                let prefix = &committed_hashes[..prefix_len];
                self.gpu_stores[dev_idx].extend(prefix, cpu_hashes, &gpu_pages);

                // Release from CPU store (rc--, free at rc=0).
                self.cpu_stores[dev_idx].release(cpu_hashes);

                // Remaining suffix (if any) needs replay.
                let remaining = suffix_count - cpu_prefix;
                if remaining > 0 {
                    let replay_pages = self.gpu_stores[dev_idx].alloc(remaining)
                        .ok_or_else(|| anyhow::anyhow!("No GPU pages for replay suffix"))?;
                    self.spawn_replay_passes(ctx_id, dev_idx, prefix_len + cpu_prefix, replay_pages)?
                } else {
                    // All suffix restored from CPU. Still need replay for working pages.
                    self.spawn_replay_passes(ctx_id, dev_idx, committed_hashes.len(), Vec::new())?
                }
            } else {
                // Cold restore: no CPU pages, allocate and replay everything.
                let suffix_pages = self.gpu_stores[dev_idx].alloc(suffix_count)
                    .ok_or_else(|| anyhow::anyhow!("No GPU pages for replay but admission check passed"))?;
                self.spawn_replay_passes(ctx_id, dev_idx, prefix_len, suffix_pages)?
            }
        } else {
            // No suffix to restore — only working pages need replay.
            self.spawn_replay_passes(ctx_id, dev_idx, committed_hashes.len(), Vec::new())?
        };

        // Set context state.
        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            if has_replay {
                ctx.state = State::Pinned;
                ctx.pending_replay = true;
            } else {
                ctx.state = State::Active;
            }
            // Refresh cached effective_pages now that committed chain is on GPU.
            ctx.cached_effective_pages = self.gpu_stores[dev_idx].effective_pages(&ctx.committed_hashes);
        }

        // If no replays needed, fire deferred ops immediately.
        if !has_replay {
            self.fire_deferred_ops(ctx_id);
        }

        self.publish_context_counts(ctx_id);
        Ok(())
    }

    /// Promote pre-allocated suffix pages and spawn replay forward passes.
    ///
    /// Registers the provided `suffix_pages` in the page store for the suffix
    /// (pages not GPU-resident), then spawns a tokio task to fill KV data.
    /// Also includes a final replay chunk for working page tokens if present.
    /// Returns `true` if any passes were spawned.
    ///
    /// All GPU page allocation must be done by the caller.
    pub(crate) fn spawn_replay_passes(
        &mut self,
        ctx_id: ContextId,
        dev_idx: usize,
        prefix_len: usize,
        suffix_pages: Vec<PhysicalPageId>,
    ) -> anyhow::Result<bool> {
        let ctx = self.contexts.get(&ctx_id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let lineage = ctx.lineage.clone();
        let committed_hashes = ctx.committed_hashes.clone();
        let committed_len = committed_hashes.len();
        let working_page_tokens = ctx.working_page_tokens.clone();
        let working_pages = ctx.working_pages.clone();
        let page_size = self.page_size;

        if prefix_len >= committed_len && working_page_tokens.is_empty() {
            return Ok(false);
        }

        // Register pre-allocated suffix pages in the page store.
        // The forward pass will fill KV data; until then the context is Pinned.
        let suffix_count = committed_len - prefix_len;
        if suffix_count > 0 {
            let suffix_hashes = &committed_hashes[prefix_len..];
            anyhow::ensure!(suffix_pages.len() == suffix_count,
                "suffix page count mismatch: got {}, need {suffix_count}", suffix_pages.len());
            let suffix_phys = suffix_pages;

            // Register suffix pages in PageStore — navigate through the retained
            // prefix so the trie correctly chains the suffix as children.
            self.gpu_stores[dev_idx].extend(
                &committed_hashes[..prefix_len],
                suffix_hashes,
                &suffix_phys,
            );
        }

        // Build the full physical page table (prefix + suffix).
        let full_committed_phys = self.gpu_stores[dev_idx].physical_ids(&committed_hashes);

        // Build forward pass requests from the lineage (for tokens/positions/masks).
        let prefix_tokens = prefix_len * page_size;
        let committed_tokens = committed_len * page_size;
        let mut kv_so_far = prefix_tokens as u32;

        let mut requests: Vec<(ForwardPassRequest, Vec<PhysicalPageId>, u32)> = Vec::new();
        let mut token_offset = 0usize;
        let mut pages_emitted = prefix_len;

        for record in &lineage {
            match record {
                Record::Fill { tokens, positions, mask, adapter, adapter_seed } => {
                    let record_end = token_offset + tokens.len();

                    if record_end <= prefix_tokens {
                        token_offset = record_end;
                        continue;
                    }
                    if token_offset >= committed_tokens {
                        break;
                    }

                    let start_in_record = prefix_tokens.saturating_sub(token_offset);
                    let end_in_record = (committed_tokens - token_offset).min(tokens.len());
                    let suffix_tokens = &tokens[start_in_record..end_in_record];
                    let suffix_positions = &positions[start_in_record..end_in_record];
                    let suffix_masks = &mask[start_in_record..end_in_record];

                    let num_pages = suffix_tokens.len() / page_size;
                    if num_pages == 0 {
                        token_offset = record_end;
                        continue;
                    }

                    let aligned_len = num_pages * page_size;
                    pages_emitted += num_pages;

                    // Page table for this chunk: all pages up to current position
                    let phys_ids = full_committed_phys[..pages_emitted].to_vec();

                    let num_input = aligned_len as u32;
                    let total_kv = kv_so_far + num_input;
                    let total_pages_for_fwd = phys_ids.len() as u32;
                    let last_page_len = compute_last_page_len(total_kv, total_pages_for_fwd, page_size as u32);

                    let fwd_req = ForwardPassRequest {
                        context_id: 0,
                        tokens: suffix_tokens[..aligned_len].to_vec(),
                        positions: suffix_positions[..aligned_len].to_vec(),
                        speculative_tokens: Vec::new(),
                        speculative_positions: Vec::new(),
                        output_speculative_tokens: false,
                        masks: suffix_masks[..aligned_len].to_vec(),
                        // Replay reproduces a previously-executed prefix whose
                        // masks were already in the lineage. We can't
                        // distinguish user-supplied from synthesized in the
                        // lineage, so we conservatively force the prefill
                        // kernel (which honors `custom_mask`) to preserve the
                        // original semantics regardless.
                        has_user_mask: true,
                        logit_mask: None,
                        sampling_indices: Vec::new(),
                        samplers: Vec::new(),
                        adapter_id: *adapter,
                        adapter_seed: *adapter_seed,
                    };

                    requests.push((fwd_req, phys_ids, last_page_len));
                    kv_so_far += num_input;
                    token_offset = record_end;
                }
            }
        }

        // Append working page tokens to the replay.
        if !working_page_tokens.is_empty() && !working_pages.is_empty() {
            let recomputable = (working_page_tokens.len() + page_size - 1) / page_size;
            let num_replay_pages = recomputable.min(working_pages.len());
            let num_replay_tokens = working_page_tokens.len().min(num_replay_pages * page_size);

            if num_replay_tokens > 0 {
                let mut tokens = Vec::with_capacity(num_replay_tokens);
                let mut positions = Vec::with_capacity(num_replay_tokens);
                let mut masks = Vec::with_capacity(num_replay_tokens);
                let adapter = working_page_tokens[0].adapter;
                let adapter_seed = working_page_tokens[0].adapter_seed;

                for info in &working_page_tokens[..num_replay_tokens] {
                    tokens.push(info.token);
                    positions.push(info.position);
                    masks.push(info.mask.clone());
                }

                // Try to merge into the last committed suffix request.
                let merged = if let Some((last_req, last_phys, _last_page_len)) = requests.last_mut() {
                    if last_req.adapter_id == adapter && last_req.adapter_seed == adapter_seed {
                        last_req.tokens.extend_from_slice(&tokens);
                        last_req.positions.extend_from_slice(&positions);
                        last_req.masks.extend_from_slice(&masks);
                        last_phys.extend_from_slice(&working_pages[..num_replay_pages]);

                        let num_input = num_replay_tokens as u32;
                        kv_so_far += num_input;
                        let total_pages_for_fwd = last_phys.len() as u32;
                        *_last_page_len = compute_last_page_len(kv_so_far, total_pages_for_fwd, page_size as u32);

                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                if !merged {
                    let mut phys_ids = full_committed_phys.clone();
                    phys_ids.extend_from_slice(&working_pages[..num_replay_pages]);

                    let num_input = num_replay_tokens as u32;
                    let total_kv = kv_so_far + num_input;
                    let total_pages_for_fwd = phys_ids.len() as u32;
                    let last_page_len = compute_last_page_len(total_kv, total_pages_for_fwd, page_size as u32);

                    let fwd_req = ForwardPassRequest {
                        context_id: 0,
                        tokens,
                        positions,
                        speculative_tokens: Vec::new(),
                        speculative_positions: Vec::new(),
                        output_speculative_tokens: false,
                        masks,
                        // See restore.rs head: replay forces prefill kernel
                        // because the lineage doesn't tell us whether masks
                        // were originally user-supplied or synthesized.
                        has_user_mask: true,
                        logit_mask: None,
                        sampling_indices: Vec::new(),
                        samplers: Vec::new(),
                        adapter_id: adapter,
                        adapter_seed,
                    };

                    requests.push((fwd_req, phys_ids, last_page_len));
                }
            }
        }

        if requests.is_empty() {
            return Ok(false);
        }

        // Spawn a task that submits forward passes sequentially, then
        // sends ReplayComplete to unpin the context.
        let model_idx = self.model_idx;
        let device_id = dev_idx;

        tokio::spawn(async move {
            for (fwd_req, phys_ids, last_page_len) in requests {
                let result = inference::submit(
                    model_idx, fwd_req,
                    device_id, phys_ids,
                    last_page_len,
                ).await;

                if let Err(e) = result {
                    tracing::error!(ctx = ctx_id, device = device_id, "replay forward pass failed: {e:#}");
                    break; // Later chunks depend on this one's KV data
                }
            }

            // Unpin after all chunks complete (or first failure)
            let _ = SERVICES.send(model_idx, super::Message::ReplayComplete { id: ctx_id });
        });

        Ok(true)
    }

    /// Fire all of a context's deferred operations.
    /// Called when replay completes or when the context is restored without replay.
    pub(crate) fn fire_deferred_ops(&mut self, ctx_id: ContextId) {
        let mut ops = self.contexts.get_mut(&ctx_id)
            .map(|c| std::mem::take(&mut c.deferred_ops))
            .unwrap_or_default();
        while !ops.is_empty() {
            let num_pages = ops[0].num_pages;
            let device = ops[0].device;
            if num_pages == 0 {
                let op = ops.remove(0);
                (op.on_alloc)(self, Vec::new());
            } else if let Some(pages) = self.gpu_stores[device].alloc(num_pages) {
                let op = ops.remove(0);
                (op.on_alloc)(self, pages);
            } else {
                // Stalled — put all remaining ops back on deferred_ops.
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.deferred_ops = ops;
                }
                self.alloc_queue.push_back(ctx_id);
                return;
            }
        }
    }

    /// Handle a completed replay forward pass: transition context out of
    /// Pinned, then fire deferred ops.
    ///
    /// Called by the actor when a ReplayComplete message arrives.
    pub(crate) fn replay_complete(&mut self, id: ContextId) {
        let pending = match self.contexts.get(&id) {
            Some(ctx) if ctx.is_pinned() && ctx.pending_replay => ctx.pending_suspend,
            _ => return,
        };

        if pending {
            // Re-suspension was requested while replay was in-flight.
            if let Some(ctx) = self.contexts.get_mut(&id) {
                ctx.pending_replay = false;
            }
            self.suspend(id);
            // Re-enqueue for restoration.
            self.enqueue_restore(id);
            return;
        }

        // Normal path: Pinned → Active, fire deferred ops.
        if let Some(ctx) = self.contexts.get_mut(&id) {
            ctx.state = State::Active;
            ctx.pending_replay = false;
        }

        self.fire_deferred_ops(id);
    }
}
