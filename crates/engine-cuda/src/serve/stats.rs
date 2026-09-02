//! Read-back surface for `Shell` — accessors, counters, and mode toggles.
//!
//! Lives here rather than in `serve.rs` because these are `Shell`'s own
//! methods over `Shell`'s own private fields; nothing here binds a device,
//! enqueues a launch, or decides anything.

use model_compiler::{Budget, CompiledModel};
use model_ir::Trace;

use super::{FireCost, Graphs, Shell};
use crate::error::Result;
use crate::record;
use crate::store::kv::{self, Paging};
use crate::store::rs::Buffers;

impl Shell {
    /// The trace this shell serves.
    #[must_use]
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    /// The artifact it was baked into.
    #[must_use]
    pub fn compiled_model(&self) -> &CompiledModel {
        &self.compiled
    }

    /// The ceilings it was baked against.
    #[must_use]
    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    /// How its pools hand pages out.
    #[must_use]
    pub fn paging(&self) -> Paging {
        self.pools.paging()
    }

    /// Which device it bound.
    #[must_use]
    pub fn ordinal(&self) -> i32 {
        self.device.ordinal()
    }

    /// That device's parallel width, probed once at bind.
    #[must_use]
    pub fn sms(&self) -> u32 {
        self.device.device().num_sm
    }

    /// The `out` seam's row width — the vocabulary; errors on a symbolic width.
    pub fn out_width(&self) -> Result<u64> {
        kv::width_of(&self.trace, self.exports.out)
    }

    /// Does this load's model text declare a draft head — gates `IntrinsicId::MtpLogits`.
    #[must_use]
    pub fn drafts(&self) -> bool {
        self.exports.mtp.is_some()
    }

    /// Does this load's model text declare a capture arm for attention scores?
    #[must_use]
    pub fn captures_scores(&self) -> bool {
        !self.exports.scores.is_empty()
    }

    /// The element type this plan's patch rows are written in, or `None` if none are declared.
    #[must_use]
    pub fn patch_element(&self) -> Option<model_ir::Dtype> {
        self.patch_seat.map(|seat| seat.dtype)
    }

    /// Can this load serve `IntrinsicId::AttnScore`? Only if the artifact declares a capture column AND the slab was carved.
    #[must_use]
    pub fn observes_scores(&self) -> bool {
        self.scores.is_some()
    }

    /// Planes per lane in the slab — exported attention layers × query heads; `0` if unobserved.
    #[must_use]
    pub fn score_planes(&self) -> u32 {
        self.scores.as_ref().map_or(0, crate::scores::Scores::planes)
    }

    /// Query heads each exported layer contributes to the slab; `0` if unobserved.
    #[must_use]
    pub fn score_heads(&self) -> u32 {
        self.scores.as_ref().map_or(0, crate::scores::Scores::heads)
    }

    /// One lane's block of score planes: `score_planes()` rows of F32 each, or `None` if unobserved; errors on a lane past the slab.
    pub fn observed(&self, lane: u32) -> crate::error::Result<Option<Vec<f32>>> {
        self.scores
            .as_ref()
            .map(|scores| scores.read_lane(lane))
            .transpose()
    }

    /// The attention layers this load exports a capture column for, in the plan's own order.
    #[must_use]
    pub fn score_layers(&self) -> Vec<u32> {
        self.exports.scores.iter().map(|e| e.layer).collect()
    }

    /// How many kv tokens a slot holds.
    #[must_use]
    pub fn held(&self, slot: u32) -> u32 {
        self.held.get(slot as usize).copied().unwrap_or(0)
    }

    /// The banks this load declared: name, capacity, bytes per adapter slot.
    #[must_use]
    pub fn banks(&self) -> Vec<(&str, u32, u64)> {
        self.weights.banks()
    }

    /// Is the whole weight table on the device? `false` means it opened the routed-expert tier.
    #[must_use]
    pub fn weights_resident(&self) -> bool {
        self.weights.all_resident()
    }

    /// What the rotating dense pump has done, or `None` if this load armed none; `late` is a stall, never a wrong answer.
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

    /// Which expert is in which slot of each streamed bank and how often routed to; empty if fully resident, and hit counts lag the newest fire.
    #[must_use]
    pub fn expert_residency(&self) -> Vec<crate::experts::BankResidency> {
        self.weights
            .experts()
            .map_or_else(Vec::new, crate::experts::Tier::residency)
    }

    /// `(experts promoted, experts demoted, rounds skipped)` since load; zeros if fully resident.
    #[must_use]
    pub fn expert_motion(&self) -> (u64, u64, u64) {
        self.weights
            .experts()
            .map_or((0, 0, 0), crate::experts::Tier::motion)
    }

    /// `(groups promoted, groups demoted, gaps held back)` since load — the packed ladder's motion.
    #[must_use]
    pub fn group_ladder(&self) -> (u64, u64, u64) {
        self.weights
            .experts()
            .map_or((0, 0, 0), crate::experts::Tier::ladder)
    }

    /// Take one rung of the packed ladder for `name` now; nothing on the serving path calls it. `None` if no berth or no tier.
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

    /// Close the deferred seat's window now, joining the fill thread synchronously; `false` if never deferred, already promoted, or fill failed.
    pub fn settle_tier_refill(&mut self) -> Result<bool> {
        let (compute, notify) = (self.device.stream(), self.device.notify_stream());
        match self.weights.experts_mut() {
            None => Ok(false),
            Some(tier) => tier.settle_refill(compute, notify),
        }
    }

    /// Bytes the buffered-activation pool holds; `0` if there's no chunked recurrence to buffer.
    #[must_use]
    pub fn buffer_bytes(&self) -> u64 {
        self.buffers.as_ref().map_or(0, Buffers::bytes)
    }

    /// Which mode it is firing in.
    #[must_use]
    pub fn mode(&self) -> Graphs {
        self.graphs
    }

    /// Change the mode between fires; one load so two residencies can't differ for unrelated reasons. Already-captured execs stay cached.
    pub fn set_mode(&mut self, graphs: Graphs) {
        self.graphs = graphs;
    }

    /// Does this shell serve `Fallback::Copy`?
    #[must_use]
    pub fn copying(&self) -> bool {
        self.copies
    }

    /// Turn the copy path on or off. Unlike [`Shell::set_mode`], NOT safe between fires once a copy-row load has armed — costly until the load restarts.
    pub fn set_copies(&mut self, copies: bool) {
        self.copies = copies;
    }

    /// Does this shell serve fires from a recorded body?
    #[must_use]
    pub fn bodying(&self) -> bool {
        self.bodies
    }

    /// Turn the bodies path on or off; takes effect at the next `prepare`, not the next launch.
    pub fn set_bodies(&mut self, bodies: bool) {
        self.bodies = bodies;
    }

    /// What the last fire's window table cost.
    #[must_use]
    pub fn last_fire_cost(&self) -> FireCost {
        self.last
    }

    /// This load's graph-cache census; [`record::BodyStats`] groups its numbers by lifetime — tally, last capture, and current census.
    #[must_use]
    pub fn body_stats(&self) -> record::BodyStats {
        self.cache.body_stats()
    }

    /// Probe seam, off by default: ask captures to keep their `cudaGraph_t` so a probe can walk the recorded kernel nodes.
    pub fn keep_graphs(&mut self, keep: bool) {
        self.cache.keep_graphs(keep);
    }

    /// Graphs kept by [`Shell::keep_graphs`], each beside the [`record::BodyKey`] its body was captured for.
    #[must_use]
    pub fn kept_graphs(&self) -> &[(record::BodyKey, crate::device::Graph)] {
        self.cache.kept()
    }

    /// `(streams, events, forked regions, side streams open)` — the only way to tell a fork from a sequential graph from outside.
    #[must_use]
    pub fn streams(&self) -> (u32, u32, usize, usize) {
        (
            self.compiled.streams.streams,
            self.compiled.streams.events,
            self.compiled.regions.iter().filter(|r| r.stream != 0).count(),
            self.device.lanes(),
        )
    }

    /// What this load holds: `(weights, arena, pools, inputs)`; the pool figure is the reserved ceiling, not what's mapped now.
    #[must_use]
    pub fn footprint(&self) -> (u64, u64, u64, u64) {
        (
            self.weights.bytes(),
            self.arena.bytes(),
            self.pools.bytes(),
            self.inputs.bytes(),
        )
    }

    /// The accounting this load was admitted under: weight tiers + elastic pool + safety floor; [`Shell::elastic`] is what it actually took.
    #[must_use]
    pub fn accounting(&self) -> crate::store::Accounting {
        self.accounting
    }

    /// Every pool arena's base address; fixed before one byte is mapped and never moves for the life of the load.
    #[must_use]
    pub fn pool_bases(&self) -> Vec<u64> {
        self.pools.bases()
    }

    /// What the elastic supply holds: `(committed bytes, high-water bytes, page bytes, budget pages)`.
    #[must_use]
    pub fn elastic(&self) -> (u64, u64, u64, u64) {
        (
            self.pools.committed_bytes(),
            self.pools.high_water_bytes(),
            self.pools.elastic_page_bytes(),
            self.pools.elastic_budget_pages(),
        )
    }

    /// Steps registered a settlement for with no callback seen yet — the run-ahead. Read by the saturation gates only.
    #[must_use]
    pub fn airborne_steps(&self) -> u64 {
        self.airborne.count()
    }

    /// Did this load's weight table come off the warm-boot artifact? `true` means the transform pipeline never ran.
    #[must_use]
    pub fn weights_from_cache(&self) -> bool {
        self.weights.from_cache()
    }

    /// The digest of the weight bytes actually resident on this device.
    pub fn weight_digest(&self) -> Result<u64> {
        self.weights.digest()
    }

    /// Descriptor-port envelopes resolved off guest device rings, process-global.
    #[must_use]
    pub fn envelopes_resolved() -> u64 {
        crate::program::ports::resolved()
    }

}
