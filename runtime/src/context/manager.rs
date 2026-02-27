//! ContextManager — lifecycle and page management.

use std::collections::{BinaryHeap, HashMap};
use std::time::Instant;
use anyhow::Result;

use super::arbiter::Arbiter;
use crate::adapter::AdapterId;
use crate::process::ProcessId;
use crate::device::DeviceId;

use super::{CONTEXTS, Context, ContextId, ContextState, Record, ReplayFill};
use super::kvcache::{DevicePageCache, PhysicalPageId, PageHash};
use super::waitqueue::{PageWaiter, WaitNeeded};

// =============================================================================
// ContextPin — snapshot of context fields for the drop-and-reacquire pattern
// =============================================================================

/// Fields extracted from a Context before dropping the DashMap guard,
/// allowing safe mutation of ContextManager fields (devices, arbiter).
pub(super) struct ContextPin {
    pub owner: Option<ProcessId>,
    pub dev_idx: usize,
    pub tip: Option<PageHash>,
    pub committed_len: usize,
}

impl ContextPin {
    pub(super) fn from_ctx(ctx: &Context) -> Self {
        ContextPin {
            owner: ctx.owner,
            dev_idx: ctx.device.unwrap_or(0) as usize,
            tip: ctx.committed_tip,
            committed_len: ctx.committed_len,
        }
    }
}

// =============================================================================
// ContextManager
// =============================================================================

#[derive(Debug)]
pub(crate) struct ContextManager {
    pub(crate) devices: Vec<DevicePageCache>,
    pub(crate) page_size: usize,
    pub(crate) model_idx: usize,
    pub(crate) name_to_id: HashMap<(String, String), ContextId>,
    next_id: u64,
    pub(crate) arbiter: Arbiter,
    /// Per-device wait queues. Waiters only compete against others on the
    /// same device, so free events on device `d` only wake device-`d` waiters.
    pub(crate) wait_queues: Vec<BinaryHeap<PageWaiter>>,
}

impl ContextManager {
    pub(crate) fn new(model_idx: usize, page_size: usize, num_gpu_pages: &[usize], num_cpu_pages: &[usize]) -> Self {
        let devices: Vec<_> = num_gpu_pages.iter().zip(num_cpu_pages.iter())
            .map(|(&gpu, &cpu)| DevicePageCache::new(page_size, gpu, cpu))
            .collect();
        let arbiter = Arbiter::new();
        let num_devices = devices.len();
        ContextManager {
            devices, page_size, model_idx,
            name_to_id: HashMap::new(), next_id: 1, arbiter,
            wait_queues: (0..num_devices).map(|_| BinaryHeap::new()).collect(),
        }
    }

    // ==================== DashMap Helpers ====================

    /// Get a read reference to a context. Caller must drop before re-entering CONTEXTS.
    pub(super) fn ctx(&self, id: ContextId) -> Result<dashmap::mapref::one::Ref<'_, (usize, ContextId), Context>> {
        CONTEXTS.get(&(self.model_idx, id))
            .ok_or_else(|| anyhow::anyhow!("Context {} not found", id))
    }

    /// Get a mutable reference to a context. Caller must drop before re-entering CONTEXTS.
    pub(super) fn ctx_mut(&self, id: ContextId) -> Result<dashmap::mapref::one::RefMut<'_, (usize, ContextId), Context>> {
        CONTEXTS.get_mut(&(self.model_idx, id))
            .ok_or_else(|| anyhow::anyhow!("Context {} not found", id))
    }

    fn next_id(&mut self) -> ContextId {
        let id = self.next_id; self.next_id += 1; id
    }

    fn select_device_for_context(&self, ctx: &Context) -> usize {
        if let Some(dev) = ctx.device { return dev as usize; }
        self.devices.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.pressure().partial_cmp(&b.pressure()).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }

    // ==================== Core Operations ====================

    pub(crate) fn create(&mut self, owner: Option<ProcessId>) -> Result<ContextId> {
        let id = self.next_id();
        if let Some(pid) = owner {
            self.arbiter.activate(pid);
        }
        let mut ctx = Context::new(owner);
        let dev = self.select_device_for_context(&ctx);
        ctx.device = Some(dev as DeviceId);
        CONTEXTS.insert((self.model_idx, id), ctx);
        Ok(id)
    }

    pub(crate) fn save(&mut self, id: ContextId, username: String, name: String) -> Result<()> {
        let source = self.ctx(id)?;
        if self.name_to_id.contains_key(&(username.clone(), name.clone())) {
            anyhow::bail!("Snapshot name already exists: {}", name);
        }

        let pin = ContextPin::from_ctx(&source);
        let lineage = source.lineage.clone();
        let max_pos = source.max_committed_position;

        let mut snapshot_buffered: Vec<u32> = source.tokens_filled.iter().map(|t| t.token).collect();
        snapshot_buffered.extend_from_slice(&source.tokens_buffered);
        drop(source);

        if let Some(tip_hash) = pin.tip {
            if let Some(dev) = self.devices.get_mut(pin.dev_idx) {
                dev.acquire_chain(tip_hash);
            }
        }

        let snapshot_id = self.next_id();
        let snapshot_ctx = Context {
            owner: None,
            device: Some(pin.dev_idx as DeviceId),
            working_pages: Vec::new(),
            working_cpu_slots: Vec::new(),
            committed_tip: pin.tip,
            committed_len: pin.committed_len,
            tokens_filled: Vec::new(),
            tokens_buffered: snapshot_buffered,
            lineage,
            max_committed_position: max_pos,
            state: ContextState::Active,
            last_access: Instant::now(),
        };
        CONTEXTS.insert((self.model_idx, snapshot_id), snapshot_ctx);
        self.name_to_id.insert((username, name), snapshot_id);
        Ok(())
    }

    pub(crate) fn snapshot(&mut self, id: ContextId, username: String) -> Result<String> {
        let name = format!("__snapshot_{}", self.next_id());
        self.save(id, username, name.clone())?;
        Ok(name)
    }

    pub(crate) fn delete(&mut self, username: String, name: String) -> Result<()> {
        let snapshot_id = self.name_to_id.remove(&(username, name))
            .ok_or_else(|| anyhow::anyhow!("Snapshot not found"))?;
        self.destroy(snapshot_id, true)
    }

    pub(crate) fn destroy(&mut self, id: ContextId, _force: bool) -> Result<()> {
        let ctx = self.ctx(id)?;
        let pin = ContextPin::from_ctx(&ctx);
        let working = ctx.working_pages.clone();
        let working_cpu = ctx.working_cpu_slots.clone();
        let working_len = ctx.working_pages.len();
        drop(ctx);

        CONTEXTS.remove(&(self.model_idx, id));

        if let Some(pid) = pin.owner {
            self.arbiter.deactivate(pid);
            self.arbiter.uncommit(pid, pin.dev_idx, pin.committed_len);
            self.arbiter.remove_working(pid, pin.dev_idx, working_len);
        }

        if let Some(tip_hash) = pin.tip {
            if let Some(dev) = self.devices.get_mut(pin.dev_idx) {
                dev.release_chain(tip_hash);
                dev.remove_index_cache(tip_hash);
            }
        }

        if let Some(dev) = self.devices.get_mut(pin.dev_idx) {
            dev.free_working(&working);
        }
        if !working_cpu.is_empty() {
            if let Some(dev) = self.devices.get_mut(pin.dev_idx) {
                dev.free_cpu_slots(&working_cpu);
            }
        }

        self.name_to_id.retain(|_, v| *v != id);
        Ok(())
    }

    pub(crate) fn fork(&mut self, id: ContextId) -> Result<ContextId> {
        let source = self.ctx(id)?;

        if !source.tokens_filled.is_empty() {
            let base = source.max_committed_position.map(|p| p + 1).unwrap_or(0);
            for (i, info) in source.tokens_filled.iter().enumerate() {
                if info.position != base + i as u32 {
                    anyhow::bail!("Cannot fork: non-sequential filled positions");
                }
            }
        }

        let pin = ContextPin::from_ctx(&source);
        let lineage = source.lineage.clone();
        let max_pos = source.max_committed_position;

        let mut new_buffered: Vec<u32> = source.tokens_filled.iter().map(|t| t.token).collect();
        new_buffered.extend_from_slice(&source.tokens_buffered);
        drop(source);

        if let Some(tip_hash) = pin.tip {
            if let Some(dev) = self.devices.get_mut(pin.dev_idx) {
                dev.acquire_chain(tip_hash);
            }
        }

        let new_id = self.next_id();
        let source_owner = CONTEXTS.get(&(self.model_idx, id)).map(|c| c.owner).flatten();
        if let Some(pid) = source_owner {
            self.arbiter.activate(pid);
        }
        let new_ctx = Context {
            owner: source_owner,
            device: Some(pin.dev_idx as DeviceId),
            working_pages: Vec::new(),
            working_cpu_slots: Vec::new(),
            committed_tip: pin.tip,
            committed_len: pin.committed_len,
            tokens_filled: Vec::new(),
            tokens_buffered: new_buffered,
            lineage,
            max_committed_position: max_pos,
            state: ContextState::Active,
            last_access: Instant::now(),
        };
        CONTEXTS.insert((self.model_idx, new_id), new_ctx);
        Ok(new_id)
    }

    // ==================== Page Management ====================

    pub(crate) async fn reserve_pages(&mut self, id: ContextId, num_pages: u32) -> Result<(), WaitNeeded> {
        if num_pages == 0 { return Ok(()); }

        let ctx = self.ctx(id).map_err(WaitNeeded::Fatal)?;
        let current_working = ctx.working_pages.len() as u32;
        let dev_idx = self.select_device_for_context(&ctx);
        let owner = ctx.owner;
        drop(ctx);

        // Only allocate the additional pages needed beyond what we already have
        let additional = num_pages.saturating_sub(current_working);
        if additional == 0 { return Ok(()); }

        let new_pages = self.allocate_working_with_suspension(dev_idx, additional as usize, owner).await?;

        {
            let mut ctx = self.ctx_mut(id).map_err(WaitNeeded::Fatal)?;
            ctx.working_pages.extend(new_pages);
            ctx.device = Some(dev_idx as DeviceId);
        }

        if let Some(pid) = owner {
            self.arbiter.add_working(pid, dev_idx, additional as usize);
        }
        Ok(())
    }

    /// Compute the post-allocation eviction floor for a requester on a device.
    pub(crate) fn requester_floor(&self, requester: Option<ProcessId>, dev_idx: usize, num_pages: usize) -> f64 {
        requester
            .map(|pid| {
                let current = self.arbiter.node_pages(&pid, dev_idx);
                self.arbiter.priority_at(&pid, dev_idx, current + num_pages)
            })
            .unwrap_or(0.0)
    }

    pub(crate) async fn allocate_working_with_suspension(
        &mut self,
        dev_idx: usize,
        num_pages: usize,
        requester: Option<ProcessId>,
    ) -> Result<Vec<PhysicalPageId>, WaitNeeded> {
        if let Ok(pages) = self.devices[dev_idx].allocate_working(num_pages) {
            return Ok(pages);
        }

        loop {
            let requester_floor = self.requester_floor(requester, dev_idx, num_pages);
            match self.find_cheapest_victim(dev_idx, requester_floor) {
                Some(vid) => {
                    self.suspend_context(vid).await;
                    if let Ok(pages) = self.devices[dev_idx].allocate_working(num_pages) {
                        return Ok(pages);
                    }
                }
                None => break,
            }
        }

        Err(WaitNeeded::NeedPages)
    }

    pub(crate) fn free_pages(&mut self, id: ContextId, num_pages: u32) -> Result<()> {
        let mut ctx = self.ctx_mut(id)?;

        let n = (num_pages as usize).min(ctx.working_pages.len());
        if n == 0 { return Ok(()); }

        let start = ctx.working_pages.len() - n;
        let to_free: Vec<PhysicalPageId> = ctx.working_pages.drain(start..).collect();
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let owner = ctx.owner;

        let tokens_to_remove = n * self.page_size;
        let tokens_len = ctx.tokens_filled.len();
        if tokens_to_remove > 0 && tokens_len > 0 {
            ctx.tokens_filled.truncate(tokens_len.saturating_sub(tokens_to_remove));
        }
        drop(ctx);

        self.devices[dev_idx].free_working(&to_free);

        if let Some(pid) = owner {
            self.arbiter.remove_working(pid, dev_idx, n);
        }
        Ok(())
    }

    pub(crate) fn commit_pages(&mut self, id: ContextId, indices: Vec<u32>) -> Result<()> {
        let page_size = self.page_size;
        let ctx = self.ctx(id)?;

        for &idx in &indices {
            if idx as usize >= ctx.working_pages.len() {
                anyhow::bail!("Invalid page index: {}", idx);
            }
        }

        let mut tokens = Vec::new();
        let mut positions = Vec::new();
        let mut masks = Vec::new();
        let mut all_positions_for_validation = Vec::new();

        for &idx in &indices {
            let start = idx as usize * page_size;
            let end = start + page_size;
            if end > ctx.tokens_filled.len() {
                anyhow::bail!("Page {} not fully filled: need {} tokens but only have {}", idx, end, ctx.tokens_filled.len());
            }
            for i in start..end {
                let info = &ctx.tokens_filled[i];
                tokens.push(info.token);
                positions.push(info.position);
                masks.push(info.mask.clone());
                all_positions_for_validation.push(info.position);
            }
        }

        if let Some(max_committed) = ctx.max_committed_position {
            for &pos in &all_positions_for_validation {
                if pos <= max_committed {
                    anyhow::bail!("Position {} must be > max committed position {}", pos, max_committed);
                }
            }
        }

        let prev_hash = ctx.committed_tip.unwrap_or(0);
        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let old_tip = ctx.committed_tip;
        let lineage_adapter = ctx.tokens_filled.first().and_then(|t| t.adapter);

        // Collect working page physical IDs for the pages being committed
        let working_phys: Vec<PhysicalPageId> = indices.iter()
            .map(|&idx| ctx.working_pages[idx as usize])
            .collect();

        drop(ctx);

        let dev = &mut self.devices[dev_idx];
        let hashes = dev.compute_page_hashes(&tokens, &positions, &masks, prev_hash);

        let mut new_phys = Vec::new();
        let mut running_prev = prev_hash;
        for (i, &hash) in hashes.iter().enumerate() {
            let (phys, _freed) = dev.commit_working_page(hash, running_prev, working_phys[i]);
            new_phys.push(phys);
            running_prev = hash;
        }

        let new_tip = *hashes.last().unwrap();
        dev.update_index_cache(new_tip, old_tip, &new_phys);

        let owner = {
            let mut ctx = self.ctx_mut(id)
                .map_err(|_| anyhow::anyhow!("Context lost during commit"))?;

            let mut sorted_indices: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
            sorted_indices.sort_unstable();
            for &idx in sorted_indices.iter().rev() {
                ctx.working_pages.remove(idx);
            }
            for &idx in sorted_indices.iter().rev() {
                let start = idx * page_size;
                let end = ((idx + 1) * page_size).min(ctx.tokens_filled.len());
                ctx.tokens_filled.drain(start..end);
            }

            ctx.committed_tip = Some(new_tip);
            ctx.committed_len += indices.len();
            ctx.max_committed_position = all_positions_for_validation.iter().copied().max()
                .or(ctx.max_committed_position);

            if let Some(Record::Fill { tokens: t, positions: p, mask: m, adapter: a }) = ctx.lineage.last_mut() {
                if *a == lineage_adapter {
                    t.extend_from_slice(&tokens);
                    p.extend_from_slice(&positions);
                    m.extend_from_slice(&masks);
                } else {
                    ctx.lineage.push(Record::Fill { tokens, positions, mask: masks, adapter: lineage_adapter });
                }
            } else {
                ctx.lineage.push(Record::Fill { tokens, positions, mask: masks, adapter: lineage_adapter });
            }

            ctx.owner
        };

        if let Some(pid) = owner {
            self.arbiter.commit(pid, dev_idx, indices.len());
        }

        Ok(())
    }
}
