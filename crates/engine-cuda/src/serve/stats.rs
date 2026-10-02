use model_compiler::{Budget, CompiledModel};
use model_ir::Trace;

use super::{FireCost, Graphs, Shell};
use crate::error::Result;
use crate::record;
use crate::store::kv::{self, Paging};
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

    pub fn out_width(&self) -> Result<u64> {
        let out = self
            .exports
            .out
            .ok_or_else(|| crate::error::Fault::Unbound {
                what: "a plan with no `out` seam has no vocabulary width".to_string(),
            })?;
        kv::width_of(&self.trace, out)
    }

    #[must_use]
    pub fn velocity_width(&self) -> Option<u32> {
        let export = self.exports.velocity.as_ref()?;
        kv::width_of(&self.trace, export.value)
            .ok()
            .and_then(|width| u32::try_from(width).ok())
    }

    #[must_use]
    pub fn pixels_facts(&self) -> (bool, u32) {
        let mut widths = self.exports.pixels_widths(&self.trace);
        let Some(first) = widths.next() else {
            return (false, 0);
        };
        (
            true,
            if widths.all(|width| width == first) {
                first
            } else {
                0
            },
        )
    }

    #[must_use]
    pub fn readout_seam(&self) -> Option<(engine::fire::ReadoutSeam, u32)> {
        let readout = self.exports.readout()?;
        let width = kv::width_of(&self.trace, readout.value)
            .ok()
            .and_then(|width| u32::try_from(width).ok())?;
        Some((readout.seam, width))
    }

    #[must_use]
    pub fn drafts(&self) -> bool {
        self.exports.mtp.is_some()
    }

    #[must_use]
    pub fn mtp_depth(&self) -> u32 {
        self.exports.drafts_depth
    }

    #[must_use]
    pub fn captures_scores(&self) -> bool {
        !self.exports.scores.is_empty()
    }

    #[must_use]
    pub fn patch_element(&self) -> Option<model_ir::Dtype> {
        self.patch_seat.map(|seat| seat.dtype)
    }

    #[must_use]
    pub fn voxel_element(&self) -> Option<model_ir::Dtype> {
        self.voxels.as_ref().map(|store| store.seat().dtype)
    }

    #[must_use]
    pub fn observes_scores(&self) -> bool {
        self.scores.is_some()
    }

    #[must_use]
    pub fn score_planes(&self) -> u32 {
        self.scores
            .as_ref()
            .map_or(0, crate::scores::Scores::planes)
    }

    #[must_use]
    pub fn score_heads(&self) -> u32 {
        self.scores.as_ref().map_or(0, crate::scores::Scores::heads)
    }

    pub fn observed(&self, lane: u32) -> crate::error::Result<Option<Vec<f32>>> {
        self.scores
            .as_ref()
            .map(|scores| scores.read_lane(lane))
            .transpose()
    }

    #[must_use]
    pub fn score_layers(&self) -> Vec<u32> {
        self.exports.scores.iter().map(|e| e.layer).collect()
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
    pub fn group_ladder(&self) -> (u64, u64, u64) {
        self.weights
            .experts()
            .map_or((0, 0, 0), crate::experts::Tier::ladder)
    }

    pub fn promote_group(
        &mut self,
        name: &str,
    ) -> Result<Option<(crate::experts::Held, crate::experts::Held)>> {
        let (compute, notify) = (self.device.stream(), self.device.notify_stream());
        match self.weights.experts_mut() {
            None => Ok(None),
            Some(tier) => tier.promote_now(name, compute, notify),
        }
    }

    pub fn settle_tier_refill(&mut self) -> Result<bool> {
        let (compute, notify) = (self.device.stream(), self.device.notify_stream());
        match self.weights.experts_mut() {
            None => Ok(false),
            Some(tier) => tier.settle_refill(compute, notify),
        }
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

    pub fn set_mode(&mut self, graphs: Graphs) {
        self.graphs = graphs;
    }

    #[must_use]
    pub fn copying(&self) -> bool {
        self.copies
    }

    pub fn set_copies(&mut self, copies: bool) {
        self.copies = copies;
    }

    #[must_use]
    pub fn bodying(&self) -> bool {
        self.bodies
    }

    pub fn set_bodies(&mut self, bodies: bool) {
        self.bodies = bodies;
    }

    #[must_use]
    pub fn last_fire_cost(&self) -> FireCost {
        self.last
    }

    #[must_use]
    pub fn body_stats(&self) -> record::BodyStats {
        self.cache.body_stats()
    }

    pub fn keep_graphs(&mut self, keep: bool) {
        self.cache.keep_graphs(keep);
    }

    #[must_use]
    pub fn kept_graphs(&self) -> &[(record::BodyKey, crate::device::Graph)] {
        self.cache.kept()
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
    pub fn pool_bases(&self) -> Vec<u64> {
        self.pools.bases()
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

    pub fn weight_digest(&self) -> Result<u64> {
        self.weights.digest()
    }

    #[must_use]
    pub fn envelopes_resolved() -> u64 {
        crate::program::ports::resolved()
    }
}
