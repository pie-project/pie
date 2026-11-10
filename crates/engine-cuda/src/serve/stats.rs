use poem_compiler::Budget;
use poem_ir::Trace;

use super::{Graphs, Shell};

use crate::record;
use crate::store::kv::Paging;

impl Shell {
    #[must_use]
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    /// The inputs this load's guest binds, and the lane words reading each.
    #[must_use]
    pub fn inputs(&self) -> Vec<poem_ir::InputDecl> {
        self.feeds.decls(&self.trace, &self.compiled)
    }

    #[must_use]
    pub fn classes(&self) -> poem_ir::LaneClasses {
        poem_exec::feeds::lane_classes(&self.trace, &self.compiled, &self.exports.outputs)
    }

    /// The regions whose attention stages at a prefill's width, and the
    /// width (`crate::window::Staging`).
    pub(crate) fn staging(&self) -> crate::window::Staging<'_> {
        crate::window::Staging {
            regions: &self.staging,
            from: self.device.ctx().tuning().compressed_prefill_from,
        }
    }

    /// The kernel decisions this load runs under.
    #[must_use]
    pub fn tuning(&self) -> kernels_cuda::tuning::Tuning {
        self.device.ctx().tuning()
    }

    #[must_use]
    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    #[must_use]
    pub fn paging(&self) -> Paging {
        self.pools.paging()
    }

    #[must_use]
    pub fn ordinal(&self) -> i32 {
        self.device.ordinal()
    }

    #[must_use]
    pub fn sms(&self) -> u32 {
        self.device.device().num_sm
    }

    #[must_use]
    pub fn patch_element(&self) -> Option<poem_ir::Dtype> {
        self.patch_seat.map(|seat| seat.dtype)
    }

    #[must_use]
    pub fn voxel_element(&self) -> Option<poem_ir::Dtype> {
        self.voxels.as_ref().map(|store| store.seat().dtype)
    }

    #[must_use]
    pub fn held(&self, slot: u32) -> u32 {
        self.held.get(slot as usize).copied().unwrap_or(0)
    }

    #[must_use]
    pub fn banks(&self) -> Vec<(&str, u32, u64)> {
        self.weights.banks()
    }

    #[must_use]
    pub fn weights_resident(&self) -> bool {
        self.weights.all_resident()
    }

    #[must_use]
    pub fn rotation(&self) -> Option<(crate::rotate::Observed, u32, u64, u64)> {
        let rotor = self.weights.rotor()?;
        Some((
            rotor.observed(),
            rotor.rotation().slots(),
            rotor.rotation().arena(),
            rotor.rotation().rotating(),
        ))
    }

    #[must_use]
    pub fn buffer_bytes(&self) -> u64 {
        // The rs buffer's rows are the records', in the state pool.
        0
    }

    #[must_use]
    pub fn buffer_capacity_bytes(&self) -> u64 {
        0
    }

    #[must_use]
    pub fn mode(&self) -> Graphs {
        self.graphs
    }

    #[must_use]
    pub fn copying(&self) -> bool {
        self.copies
    }

    #[must_use]
    pub fn body_stats(&self) -> record::BodyStats {
        self.cache.body_stats()
    }

    #[must_use]
    pub fn streams(&self) -> (u32, u32, usize, usize) {
        (
            self.compiled.streams.streams,
            self.compiled.streams.events,
            self.compiled
                .regions
                .iter()
                .filter(|r| r.stream != 0)
                .count(),
            self.device.lanes(),
        )
    }

    #[must_use]
    pub fn footprint(&self) -> (u64, u64, u64, u64) {
        (
            self.weights.bytes(),
            self.arena.bytes(),
            self.pools.bytes(),
            self.inputs.bytes(),
        )
    }

    #[must_use]
    pub fn accounting(&self) -> crate::store::Accounting {
        self.accounting
    }

    /// See `Engine::commit_segments`: off a fire's path.
    pub fn commit_segments(&mut self, segments: u32) -> crate::Result<()> {
        self.pools.commit_segments(segments)
    }

    /// See `Engine::release_segments`. The runtime has waited out every
    /// frame that named the segments' chunks; what the stream still holds
    /// (an evacuation's copy out of them) lands first.
    pub fn release_segments(&mut self, segments: u32) -> crate::Result<()> {
        self.device.synchronize()?;
        self.pools.release_segments(segments)
    }

    /// See `Engine::evacuate`: the copy and the table change are ordered on
    /// the stream behind every fire enqueued so far and before the next.
    pub fn evacuate(&mut self, evacuation: &engine::chunk::Evacuation) -> crate::Result<()> {
        self.pools.evacuate(self.device.stream(), evacuation)
    }

    /// The state pool as the runtime counts it
    /// ([`crate::store::Pools::chunk_facts`]).
    pub fn chunk_facts(&mut self) -> crate::Result<engine::chunk::ChunkFacts> {
        let max_tokens = self.budget().max_tokens;
        self.pools.chunk_facts(max_tokens)
    }

    /// Names the recurrent record whose pieces lie in the slot refs
    /// `pieces`: the slot a lane then names it by (its first).
    pub fn name_record(&mut self, pieces: &[u32]) -> crate::Result<u32> {
        self.pools.name_record(pieces)
    }

    /// See `Engine::weight_residency`: stream-ordered before the next fire.
    pub fn weight_residency(
        &mut self,
        residency: &engine::chunk::WeightResidency,
    ) -> crate::Result<()> {
        if self
            .pools
            .weight_residency(self.device.stream(), residency)?
        {
            // A recorded body keeps the select it chose for where a bank's
            // experts lay: the lattice is recorded again over where they
            // lie now.
            self.device.synchronize()?;
            self.cache.forget_bodies();
            self.arm_bodies()?;
        }
        // A set held whole for the load's life is read from the pool alone:
        // its pinned host copy goes.
        let held = self.pools.held_weights();
        if !held.is_empty()
            && let Some(tier) = self.weights.experts_mut()
        {
            self.device.synchronize()?;
            let freed = tier.release_host(&held);
            if freed > 0 {
                self.pools.drop_weight_hosts();
                eprintln!(
                    "engine-cuda: the weights the pool holds whole are read from it alone; \
                     {} MiB of their pinned host copy freed",
                    freed >> 20
                );
            }
        }
        Ok(())
    }

    /// The row-streamed tables' blocks named a slot, and all of them: the
    /// rest a fire reads from the host copy.
    #[must_use]
    pub fn row_blocks(&self) -> (u64, u64) {
        self.pools.row_blocks()
    }

    /// See `Engine::weight_uses`.
    pub fn weight_uses(&mut self) -> crate::Result<engine::chunk::WeightUses> {
        self.pools.weight_uses(self.device.stream())
    }

    /// From here on lanes name their slots by slot ref.
    pub fn take_refs(&mut self) {
        self.pools.take_refs();
    }

    /// Copies a working set's recurrent record from slot `src` onto `dst`,
    /// on the stream ahead of the fire that writes `dst`.
    pub fn fork_state(&mut self, src: u32, dst: u32) -> crate::Result<()> {
        match self.rs_ahead.get(&src).copied() {
            Some(at) => self.rs_ahead.insert(dst, at),
            None => self.rs_ahead.remove(&dst),
        };
        self.pools.copy_slot(self.device.stream(), src, dst)
    }

    /// Copies whole slots of `shape`, `(src, dst)`, on the stream ahead of
    /// the fire that reads or writes `dst`.
    pub fn copy_whole_slots(
        &mut self,
        shape: engine::chunk::ShapeId,
        pairs: &[[u32; 2]],
    ) -> crate::Result<()> {
        self.pools
            .copy_whole_slots(self.device.stream(), shape, pairs)
    }

    /// Copies whole slots and token runs of a kv shape (`windowed`: the windowed
    /// rows') within the state pool.
    pub fn copy_token_slots(
        &mut self,
        windowed: bool,
        runs: &[engine::transfer::KvRun],
    ) -> crate::Result<()> {
        self.pools.fence_all_swaps(self.device.stream())?;
        let (list, windowed): (&[_], &[_]) = if windowed { (&[], runs) } else { (runs, &[]) };
        self.pools.copy_kv(self.device.stream(), list, windowed)
    }

    /// See `Engine::commit_pool`.
    pub fn commit_pool(&mut self, kv_pages: u32) -> crate::Result<()> {
        let state_slots = if self.pools.has_state() {
            self.pools.paging().slots
        } else {
            0
        };
        self.pools.commit(engine::frame::Demand {
            kv_pages,
            state_slots,
            workspace: 0,
        })
    }

    #[must_use]
    pub fn elastic(&self) -> (u64, u64, u64, u64) {
        (
            self.pools.committed_bytes(),
            self.pools.high_water_bytes(),
            self.pools.elastic_page_bytes(),
            self.pools.elastic_budget_pages(),
        )
    }

    #[must_use]
    pub fn airborne_steps(&self) -> u64 {
        self.airborne.count()
    }

    #[must_use]
    pub fn weights_from_cache(&self) -> bool {
        self.weights.from_cache()
    }
}
