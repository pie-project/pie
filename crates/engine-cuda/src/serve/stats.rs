use poem_compiler::{Budget, CompiledModel};
use poem_ir::Trace;

use super::{Graphs, Shell};

use crate::record;
use crate::store::kv::Paging;
use crate::store::rs::Buffers;

impl Shell {
    #[must_use]
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    #[must_use]
    pub fn compiled_model(&self) -> &CompiledModel {
        &self.compiled
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
    pub fn expert_residency(&self) -> Vec<crate::experts::BankResidency> {
        self.weights
            .experts()
            .map_or_else(Vec::new, crate::experts::Tier::residency)
    }

    #[must_use]
    pub fn expert_motion(&self) -> (u64, u64, u64) {
        self.weights
            .experts()
            .map_or((0, 0, 0), crate::experts::Tier::motion)
    }

    #[must_use]
    pub fn buffer_bytes(&self) -> u64 {
        self.buffers.as_ref().map_or(0, Buffers::bytes)
    }

    #[must_use]
    pub fn buffer_capacity_bytes(&self) -> u64 {
        self.buffers.as_ref().map_or(0, Buffers::capacity_bytes)
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

    pub fn keep_graphs(&mut self, keep: bool) {
        self.cache.keep_graphs(keep);
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
