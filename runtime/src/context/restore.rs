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

use super::pagestore::{PhysicalPageId, compute_last_page_len};
use super::rs_cache::RsState;
use super::{
    ContextId, ContextManager, Record, ReplayPageRegistration, SERVICE, State,
    materialize_lineage_mask,
};
use crate::adapter::AdapterId;
use crate::driver::{self, DriverId};
use crate::inference;

// =============================================================================
// Restore methods on ContextManager
// =============================================================================

impl ContextManager {
    /// Admission check: can this context be restored?
    /// Checks that the driver has enough free GPU pages for the context's
    /// working pages (recomputed) plus replay pages plus deferred alloc requirements.
    pub(crate) fn can_restore(&mut self, ctx_id: ContextId) -> bool {
        let ctx = match self.contexts.get(&ctx_id) {
            Some(c) if c.is_off_gpu() => c,
            _ => return false,
        };

        let driver_idx = ctx.driver.unwrap_or(0) as usize;
        let mut required = ctx.suspended_working_count;

        // Pages needing replay: check prefix match, count missing suffix
        let mut prefix_len = 0usize;
        if !ctx.committed_hashes.is_empty() {
            prefix_len = self.gpu_stores[driver_idx].prefix_len(&ctx.committed_hashes);
            let replay_pages = ctx.committed_hashes.len().saturating_sub(prefix_len);
            required += replay_pages;
        }

        // Include deferred ops: these fire immediately after restore via
        // fire_deferred_ops. If the pool can't satisfy them, the context
        // would re-suspend immediately — wasting the restore work.
        let deferred_pages: usize = ctx.deferred_ops.iter().map(|op| op.num_pages).sum();
        required += deferred_pages;

        if ctx.rs_state.is_missing() && self.driver_uses_rs_cache(driver_idx) {
            // RS replay must walk the full token lineage. Existing GPU
            // prefix pages may be shared, so replay uses scratch pages for
            // that prefix and writes only suffix/working pages in place.
            required += prefix_len;
            if self.rs_stores[driver_idx].available() == 0 {
                return false;
            }
        }
        if self.gpu_stores[driver_idx].available() < required {
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
        let ctx = self
            .contexts
            .get(&ctx_id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;

        if !ctx.is_off_gpu() {
            return Ok(());
        }

        let driver_idx = ctx.driver.unwrap_or(0) as usize;
        let working_count = ctx.suspended_working_count;
        let committed_hashes = ctx.committed_hashes.clone();
        let cpu_working_pages = ctx.cpu_working_pages.clone();
        let rs_missing = ctx.rs_state.is_missing() && self.driver_uses_rs_cache(driver_idx);

        // Phase 1: Restore working pages.
        // If CPU-stashed, H2D copy; otherwise allocate fresh for replay.
        if working_count > 0 {
            let gpu_pages = self.gpu_stores[driver_idx]
                .alloc(working_count)
                .ok_or_else(|| anyhow::anyhow!("No free GPU pages for working re-alloc"))?;

            if !cpu_working_pages.is_empty() && cpu_working_pages.len() == working_count {
                // H2D copy from CPU stash.
                let _ = driver::copy_h2d(driver_idx as DriverId, &gpu_pages, &cpu_working_pages);
                self.cpu_stores[driver_idx].free(&cpu_working_pages);
            }

            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.working_pages = gpu_pages;
                ctx.suspended_working_count = 0;
                ctx.cpu_working_pages.clear();
            }
        }

        // Phase 2: Acquire refcounts for GPU-resident committed prefix.
        let prefix_len = if !committed_hashes.is_empty() {
            let dev = &mut self.gpu_stores[driver_idx];
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

        let has_replay = if rs_missing {
            let mut suffix_replay_pages = Vec::with_capacity(suffix_count);
            if suffix_count > 0 {
                let suffix_hashes = &committed_hashes[prefix_len..];
                let cpu_prefix = self.cpu_stores[driver_idx].prefix_len(suffix_hashes);
                if cpu_prefix > 0 {
                    let cpu_hashes = &suffix_hashes[..cpu_prefix];
                    let cpu_phys = self.cpu_stores[driver_idx].physical_ids(cpu_hashes);
                    let gpu_pages = self.gpu_stores[driver_idx]
                        .alloc(cpu_prefix)
                        .ok_or_else(|| anyhow::anyhow!("No GPU pages for CPU-cache restore"))?;
                    let _ = driver::copy_h2d(driver_idx as DriverId, &gpu_pages, &cpu_phys);
                    self.cpu_stores[driver_idx].release(cpu_hashes);
                    suffix_replay_pages.extend_from_slice(&gpu_pages);
                }
                let remaining = suffix_count.saturating_sub(suffix_replay_pages.len());
                if remaining > 0 {
                    let replay_pages = self.gpu_stores[driver_idx]
                        .alloc(remaining)
                        .ok_or_else(|| anyhow::anyhow!("No GPU pages for replay suffix"))?;
                    suffix_replay_pages.extend_from_slice(&replay_pages);
                }
            }

            let scratch_prefix_pages = if prefix_len > 0 {
                self.gpu_stores[driver_idx]
                    .alloc(prefix_len)
                    .ok_or_else(|| anyhow::anyhow!("No GPU pages for rs_cache replay scratch"))?
            } else {
                Vec::new()
            };
            let slot = self.alloc_rs_slot_now_with_eviction(ctx_id, driver_idx)?;
            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.rs_state = RsState::Resident(slot);
            }
            self.spawn_full_rs_replay_pass(
                ctx_id,
                driver_idx,
                slot,
                prefix_len,
                scratch_prefix_pages,
                suffix_replay_pages,
            )?
        } else if suffix_count > 0 {
            // Check how many suffix pages are on CPU.
            let cpu_prefix = self.cpu_stores[driver_idx].prefix_len(suffix_hashes);

            if cpu_prefix > 0 {
                // CPU-warm restore: H2D copy for CPU-resident portion.
                let cpu_hashes = &suffix_hashes[..cpu_prefix];
                let cpu_phys = self.cpu_stores[driver_idx].physical_ids(cpu_hashes);

                let gpu_pages = self.gpu_stores[driver_idx]
                    .alloc(cpu_prefix)
                    .ok_or_else(|| anyhow::anyhow!("No GPU pages for CPU-cache restore"))?;

                let _ = driver::copy_h2d(driver_idx as DriverId, &gpu_pages, &cpu_phys);

                // Register in GPU trie.
                let prefix = &committed_hashes[..prefix_len];
                self.gpu_stores[driver_idx].extend(prefix, cpu_hashes, &gpu_pages);

                // Release from CPU store (rc--, free at rc=0).
                self.cpu_stores[driver_idx].release(cpu_hashes);

                // Remaining suffix (if any) needs replay.
                let remaining = suffix_count - cpu_prefix;
                if remaining > 0 {
                    let replay_pages = self.gpu_stores[driver_idx]
                        .alloc(remaining)
                        .ok_or_else(|| anyhow::anyhow!("No GPU pages for replay suffix"))?;
                    self.spawn_replay_passes(
                        ctx_id,
                        driver_idx,
                        prefix_len + cpu_prefix,
                        replay_pages,
                    )?
                } else {
                    // All suffix restored from CPU. Still need replay for working pages.
                    self.spawn_replay_passes(
                        ctx_id,
                        driver_idx,
                        committed_hashes.len(),
                        Vec::new(),
                    )?
                }
            } else {
                // Cold restore: no CPU pages, allocate and replay everything.
                let suffix_pages =
                    self.gpu_stores[driver_idx]
                        .alloc(suffix_count)
                        .ok_or_else(|| {
                            anyhow::anyhow!("No GPU pages for replay but admission check passed")
                        })?;
                self.spawn_replay_passes(ctx_id, driver_idx, prefix_len, suffix_pages)?
            }
        } else {
            // No suffix to restore — only working pages need replay.
            self.spawn_replay_passes(ctx_id, driver_idx, committed_hashes.len(), Vec::new())?
        };

        // Set context state.
        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            if has_replay {
                ctx.state = State::Pinned;
                ctx.pending_replay = true;
            } else {
                ctx.state = State::Active;
            }
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
    pub(crate) fn spawn_full_rs_replay_pass(
        &mut self,
        ctx_id: ContextId,
        driver_idx: usize,
        rs_slot: super::rs_cache::RsSlotId,
        scratch_prefix_len: usize,
        scratch_prefix_pages: Vec<PhysicalPageId>,
        suffix_replay_pages: Vec<PhysicalPageId>,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(
            scratch_prefix_pages.len() == scratch_prefix_len,
            "rs_cache replay scratch page count mismatch: got {}, need {scratch_prefix_len}",
            scratch_prefix_pages.len()
        );
        let ctx = self
            .contexts
            .get(&ctx_id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let lineage = ctx.lineage.clone();
        let committed_hashes = ctx.committed_hashes.clone();
        let working_page_tokens = ctx.working_page_tokens.clone();
        let working_pages = ctx.working_pages.clone();
        let page_size = self.page_size;
        let suffix_count = committed_hashes.len().saturating_sub(scratch_prefix_len);
        anyhow::ensure!(
            suffix_replay_pages.len() == suffix_count,
            "rs_cache replay suffix page count mismatch: got {}, need {suffix_count}",
            suffix_replay_pages.len()
        );

        let mut all_phys = scratch_prefix_pages.clone();
        all_phys.extend_from_slice(&suffix_replay_pages);
        all_phys.extend_from_slice(&working_pages);
        if all_phys.is_empty() {
            if !scratch_prefix_pages.is_empty() {
                self.gpu_stores[driver_idx].free(&scratch_prefix_pages);
            }
            if !suffix_replay_pages.is_empty() {
                self.gpu_stores[driver_idx].free(&suffix_replay_pages);
            }
            return Ok(false);
        }
        let registration = if suffix_replay_pages.is_empty() {
            None
        } else {
            Some(ReplayPageRegistration {
                driver: driver_idx,
                prefix: committed_hashes[..scratch_prefix_len].to_vec(),
                hashes: committed_hashes[scratch_prefix_len..].to_vec(),
                pages: suffix_replay_pages,
            })
        };

        #[derive(Clone)]
        struct ReplayToken {
            token: u32,
            position: u32,
            mask: pie_driver_abi::Brle,
            adapter: Option<AdapterId>,
            adapter_seed: Option<i64>,
            forward_id: u64,
        }

        let total_replay_tokens: usize = lineage
            .iter()
            .map(|record| match record {
                Record::Fill { tokens, .. } => tokens.len(),
            })
            .sum::<usize>()
            + working_page_tokens.len();
        let mut replay_tokens = Vec::with_capacity(total_replay_tokens);
        for record in &lineage {
            match record {
                Record::Fill {
                    tokens,
                    positions,
                    mask,
                    adapter,
                    adapter_seed,
                    forward_id,
                } => {
                    for (i, &token) in tokens.iter().enumerate() {
                        replay_tokens.push(ReplayToken {
                            token,
                            position: positions[i],
                            mask: mask
                                .get(i)
                                .cloned()
                                .unwrap_or_else(|| pie_driver_abi::Brle::new(0)),
                            adapter: *adapter,
                            adapter_seed: *adapter_seed,
                            forward_id: *forward_id,
                        });
                    }
                }
            }
        }
        replay_tokens.extend(working_page_tokens.iter().map(|info| ReplayToken {
            token: info.token,
            position: info.position,
            mask: info.mask.clone(),
            adapter: info.adapter,
            adapter_seed: info.adapter_seed,
            forward_id: info.forward_id,
        }));

        let mut requests: Vec<(pie_driver_abi::ForwardRequest, Vec<PhysicalPageId>, u32)> = Vec::new();
        let mut kv_so_far = 0usize;
        let mut first = true;
        let mut start = 0usize;
        while start < replay_tokens.len() {
            let forward_id = replay_tokens[start].forward_id;
            let adapter = replay_tokens[start].adapter;
            let adapter_seed = replay_tokens[start].adapter_seed;
            let mut end = start + 1;
            while end < replay_tokens.len()
                && replay_tokens[end].forward_id == forward_id
                && replay_tokens[end].adapter == adapter
                && replay_tokens[end].adapter_seed == adapter_seed
            {
                end += 1;
            }
            let group = &replay_tokens[start..end];
            let total_after = kv_so_far + group.len();
            let pages_needed = total_after.div_ceil(page_size);
            anyhow::ensure!(
                pages_needed <= all_phys.len(),
                "rs_cache replay page table too short: need {pages_needed} pages, have {}",
                all_phys.len()
            );
            if pages_needed > 0 {
                let tokens = group.iter().map(|info| info.token).collect();
                let positions = group.iter().map(|info| info.position).collect();
                let masks = group.iter().map(|info| info.mask.clone()).collect();
                let last_page_len = compute_last_page_len(
                    total_after as u32,
                    pages_needed as u32,
                    page_size as u32,
                );
                let mut fwd_req = crate::inference::request::new_per_request(
                    ctx_id,
                    tokens,
                    positions,
                    masks,
                    true,
                    None,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    false,
                    adapter,
                    adapter_seed,
                );
                fwd_req.rs_slot_ids = vec![rs_slot];
                fwd_req.rs_slot_flags = vec![if first {
                    super::rs_cache::RS_FLAG_RESET
                } else {
                    0
                }];
                first = false;
                requests.push((fwd_req, all_phys[..pages_needed].to_vec(), last_page_len));
                kv_so_far = total_after;
            }
            start = end;
        }

        if requests.is_empty() {
            if !scratch_prefix_pages.is_empty() {
                self.gpu_stores[driver_idx].free(&scratch_prefix_pages);
            }
            if let Some(registration) = &registration {
                self.gpu_stores[registration.driver].free(&registration.pages);
            }
            return Ok(false);
        }

        let driver_id = driver_idx;
        tokio::spawn(async move {
            for (fwd_req, phys_ids, last_page_len) in requests {
                let result = inference::submit(
                    fwd_req,
                    driver_id,
                    phys_ids,
                    Vec::new(),
                    last_page_len,
                )
                .await;
                if let Err(e) = result {
                    tracing::error!(
                        ctx = ctx_id,
                        driver = driver_id,
                        "rs_cache replay forward pass failed: {e:#}"
                    );
                    break;
                }
            }
            let _ = SERVICE.send(
                super::Message::ReplayComplete {
                    id: ctx_id,
                    scratch_driver: driver_id,
                    scratch_pages: scratch_prefix_pages,
                    registration,
                },
            );
        });

        Ok(true)
    }

    pub(crate) fn spawn_replay_passes(
        &mut self,
        ctx_id: ContextId,
        driver_idx: usize,
        prefix_len: usize,
        suffix_pages: Vec<PhysicalPageId>,
    ) -> anyhow::Result<bool> {
        let ctx = self
            .contexts
            .get(&ctx_id)
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
            anyhow::ensure!(
                suffix_pages.len() == suffix_count,
                "suffix page count mismatch: got {}, need {suffix_count}",
                suffix_pages.len()
            );
            let suffix_phys = suffix_pages;

            // Register suffix pages in PageStore — navigate through the retained
            // prefix so the trie correctly chains the suffix as children.
            self.gpu_stores[driver_idx].extend(
                &committed_hashes[..prefix_len],
                suffix_hashes,
                &suffix_phys,
            );
        }

        // Build the full physical page table (prefix + suffix).
        let full_committed_phys = self.gpu_stores[driver_idx].physical_ids(&committed_hashes);

        // Build forward pass requests from the lineage (for tokens/positions/masks).
        let prefix_tokens = prefix_len * page_size;
        let committed_tokens = committed_len * page_size;
        let mut kv_so_far = prefix_tokens as u32;

        let mut requests: Vec<(pie_driver_abi::ForwardRequest, Vec<PhysicalPageId>, u32)> = Vec::new();
        let mut token_offset = 0usize;
        let mut pages_emitted = prefix_len;

        for record in &lineage {
            match record {
                Record::Fill {
                    tokens,
                    positions,
                    mask,
                    adapter,
                    adapter_seed,
                    forward_id: _,
                } => {
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
                    let last_page_len =
                        compute_last_page_len(total_kv, total_pages_for_fwd, page_size as u32);

                    // Replay reproduces a previously-executed prefix whose
                    // masks were already in the lineage. We can't
                    // distinguish user-supplied from synthesized in the
                    // lineage, so we conservatively force the prefill
                    // kernel (which honors `custom_mask`) to preserve the
                    // original semantics regardless.
                    let fwd_req = crate::inference::request::new_per_request(
                        0,
                        suffix_tokens[..aligned_len].to_vec(),
                        suffix_positions[..aligned_len].to_vec(),
                        suffix_masks[..aligned_len]
                            .iter()
                            .zip(&suffix_positions[..aligned_len])
                            .map(|(mask, &position)| materialize_lineage_mask(mask, position))
                            .collect(),
                        true, // has_user_mask
                        None,
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        false,
                        *adapter,
                        *adapter_seed,
                    );

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
                    masks.push(materialize_lineage_mask(&info.mask, info.position));
                }

                // Try to merge into the last committed suffix request.
                let adapter_i64 = adapter.map(|id| id as i64).unwrap_or(-1);
                let adapter_seed_i64 = adapter_seed.unwrap_or(-1);
                let merged = if let Some((last_req, last_phys, _last_page_len)) =
                    requests.last_mut()
                {
                    let last_binding = &last_req.adapter_bindings[0];
                    if last_binding.adapter_id == adapter_i64
                        && last_binding.seed == adapter_seed_i64
                    {
                        // Extend tokens, positions, masks. Update per-request
                        // indptr tails so the value stays a valid per-request
                        // ForwardRequest.
                        last_req.token_ids.extend_from_slice(&tokens);
                        last_req.position_ids.extend_from_slice(&positions);
                        last_req.masks.extend_from_slice(&masks);
                        *last_req.qo_indptr.last_mut().unwrap() = last_req.token_ids.len() as u32;
                        *last_req.mask_indptr.last_mut().unwrap() = last_req.masks.len() as u32;
                        last_phys.extend_from_slice(&working_pages[..num_replay_pages]);

                        let num_input = num_replay_tokens as u32;
                        kv_so_far += num_input;
                        let total_pages_for_fwd = last_phys.len() as u32;
                        *_last_page_len =
                            compute_last_page_len(kv_so_far, total_pages_for_fwd, page_size as u32);

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
                    let last_page_len =
                        compute_last_page_len(total_kv, total_pages_for_fwd, page_size as u32);

                    // See restore.rs head: replay forces prefill kernel
                    // because the lineage doesn't tell us whether masks
                    // were originally user-supplied or synthesized.
                    let fwd_req = crate::inference::request::new_per_request(
                        0,
                        tokens,
                        positions,
                        masks,
                        true, // has_user_mask
                        None,
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        false,
                        adapter,
                        adapter_seed,
                    );

                    requests.push((fwd_req, phys_ids, last_page_len));
                }
            }
        }

        if requests.is_empty() {
            return Ok(false);
        }

        // Spawn a task that submits forward passes sequentially, then
        // sends ReplayComplete to unpin the context.
        let driver_id = driver_idx;

        tokio::spawn(async move {
            for (fwd_req, phys_ids, last_page_len) in requests {
                let result = inference::submit(
                    fwd_req,
                    driver_id,
                    phys_ids,
                    Vec::new(),
                    last_page_len,
                )
                .await;

                if let Err(e) = result {
                    tracing::error!(
                        ctx = ctx_id,
                        driver = driver_id,
                        "replay forward pass failed: {e:#}"
                    );
                    break; // Later chunks depend on this one's KV data
                }
            }

            // Unpin after all chunks complete (or first failure)
            let _ = SERVICE.send(
                super::Message::ReplayComplete {
                    id: ctx_id,
                    scratch_driver: driver_id,
                    scratch_pages: Vec::new(),
                    registration: None,
                },
            );
        });

        Ok(true)
    }

    /// Fire all of a context's deferred operations.
    /// Called when replay completes or when the context is restored without replay.
    pub(crate) fn fire_deferred_ops(&mut self, ctx_id: ContextId) {
        let mut ops = self
            .contexts
            .get_mut(&ctx_id)
            .map(|c| std::mem::take(&mut c.deferred_ops))
            .unwrap_or_default();
        while !ops.is_empty() {
            let num_pages = ops[0].num_pages;
            let driver_idx = ops[0].driver;
            if ops[0].needs_rs_slot && self.rs_stores[driver_idx].available() == 0 {
                if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                    ctx.deferred_ops = ops;
                }
                self.alloc_queue.push_back(ctx_id);
                return;
            }
            if num_pages == 0 {
                let op = ops.remove(0);
                (op.on_alloc)(self, Vec::new());
            } else if let Some(pages) = self.gpu_stores[driver_idx].alloc(num_pages) {
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
