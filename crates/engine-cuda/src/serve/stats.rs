//! **WHAT A CALLER CAN ASK A LOADED SHELL** — the read-back surface, and the
//! five words that flip between fires.
//!
//! A child module of [`serve`](super) rather than a sibling, because these
//! are `Shell`'s own methods reading `Shell`'s own private fields: what moved
//! is the TEXT, not the visibility. Nothing here binds a device, enqueues a
//! launch or decides anything — every function is one of three shapes:
//!
//! * **an accessor**, handing back a fact the load already computed (the
//!   trace, the bake, the paging, the export axes);
//! * **a counter**, handing back what the last fire or the whole process
//!   measured (the graph cache's stats, the fold's motion, the pools'
//!   committed bytes, the weight cache's hits);
//! * **a toggle**, moving one word between two fires so an A/B is ONE load
//!   with one thing changed — [`Shell::set_mode`]'s argument, made five
//!   times.
//!
//! It sat in `serve.rs` under a header saying that file has no logic and is
//! the call order top to bottom, and thirty-seven methods of read-back is
//! neither (alto survey §2 debt 6, wave P). The call order is next door; this
//! is what a gate asks afterwards.

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

    /// The `out` seam's row width — the vocabulary, for a plan whose out seam
    /// is logits.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`](crate::error::Fault::Unbound) for an out value
    /// whose width is symbolic.
    pub fn out_width(&self) -> Result<u64> {
        kv::width_of(&self.trace, self.exports.out)
    }

    /// Does this load's model text declare a draft head (design §8's MTP
    /// row, palo C3)?
    ///
    /// What `engine_cuda::api::profile` answers `ModelProfile::has_mtp_logits`
    /// with, and therefore what decides whether a guest program may declare
    /// `IntrinsicId::MtpLogits` at all. A bind-time contract has to be true at
    /// the FIRST fire, and it is true exactly when the plan states the export
    /// this shell binds the intrinsic at.
    #[must_use]
    pub fn drafts(&self) -> bool {
        self.exports.mtp.is_some()
    }

    /// Does this load's model text declare a capture arm (design §9, palo C4)?
    ///
    /// Empty means a `Lane::captures_scores` has nowhere to go, and the fire
    /// says so by name rather than answering with an uncaptured continuation.
    #[must_use]
    pub fn captures_scores(&self) -> bool {
        !self.exports.scores.is_empty()
    }

    /// The attention layers this load exports a capture column for, in the
    /// plan's own order.
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

    /// **Is the whole weight table on the device?** (alto design §7.)
    ///
    /// What [`LoadFacts::weights_resident`](engine::engine_api::load::LoadFacts)
    /// reports. `false` says this load opened the routed-expert tier, and
    /// [`Shell::expert_residency`] is what says how much of it is where.
    #[must_use]
    pub fn weights_resident(&self) -> bool {
        self.weights.all_resident()
    }

    /// **Which expert is in which slot of each streamed bank, and how often
    /// each has been routed to** — the promotion's only observable (alto
    /// design §7, wave D2).
    ///
    /// Empty for a fully-resident load, which has no tier to report on. The
    /// hit counts are the last settled readback's, so they lag the newest
    /// airborne fire by design: they are carried out asynchronously on the
    /// notify stream and nothing waits for them.
    #[must_use]
    pub fn expert_residency(&self) -> Vec<crate::experts::BankResidency> {
        self.weights
            .experts()
            .map_or_else(Vec::new, crate::experts::Tier::residency)
    }

    /// `(experts promoted, experts demoted, promotion rounds skipped because
    /// the previous one was still moving)` since load, or all zeros for a
    /// fully-resident load.
    #[must_use]
    pub fn expert_motion(&self) -> (u64, u64, u64) {
        self.weights
            .experts()
            .map_or((0, 0, 0), crate::experts::Tier::motion)
    }

    /// How many bytes the buffered-activation pool holds, and `0` for a plan
    /// with no chunked recurrence to buffer.
    #[must_use]
    pub fn buffer_bytes(&self) -> u64 {
        self.buffers.as_ref().map_or(0, Buffers::bytes)
    }

    /// Which mode it is firing in.
    #[must_use]
    pub fn mode(&self) -> Graphs {
        self.graphs
    }

    /// Change the mode between fires.
    ///
    /// **THE A/B IS ONE LOAD, NOT TWO**: 1.7 GB of weights landed twice would
    /// be two residencies, two arenas and two tuner histories, and a
    /// difference between the runs could be any of those. One shell, one set
    /// of addresses, one word changed — then the tokens either match or the
    /// graph is wrong.
    ///
    /// Execs already captured stay cached: their key still means what it
    /// meant, and going Off and back On is a policy change, not an
    /// invalidation.
    pub fn set_mode(&mut self, graphs: Graphs) {
        self.graphs = graphs;
    }

    /// Does this shell serve `Fallback::Copy`? See [`Shell::copies`]'s field.
    #[must_use]
    pub fn copying(&self) -> bool {
        self.copies
    }

    /// Turn the copy path on or off between fires — the other A/B, and the
    /// one whose oracle is free.
    ///
    /// A copy and a split compute the same numbers over the same rows by
    /// construction (a gather moves bytes), so flipping this word between two
    /// otherwise identical fires and diffing the logits is a complete test of
    /// the claim. One shell, for `set_mode`'s reason: two loads would be two
    /// residencies and two tuner histories, and a difference could be either.
    pub fn set_copies(&mut self, copies: bool) {
        // The graph cache is keyed on this (`record::Key`), so flipping it
        // misses rather than replaying a body recorded under the other policy.
        self.copies = copies;
    }

    /// Does this shell fold the composition axis? See [`Shell::fold`]'s field.
    #[must_use]
    pub fn folding(&self) -> bool {
        self.fold
    }

    /// Turn the fold on or off between fires — the third A/B, and it is one
    /// load for [`Shell::set_mode`]'s reason: two loads would be two
    /// residencies and two tuner histories, and a difference could be either.
    /// Buckets already armed stay armed; turning the fold off simply stops
    /// routing fires through them, exactly as `set_mode(Off)` leaves keyed
    /// execs resident.
    pub fn set_fold(&mut self, fold: bool) {
        self.fold = fold;
    }

    /// Turn the fold's pipeline on or off between fires — the twin exec and
    /// the ahead-of-sync prebind ([`Knobs::pipeline`](super::Knobs::pipeline)).
    /// Off is step 4's fold exactly, which is what the pipelined revisit
    /// gate diffs against; one load, for [`Shell::set_mode`]'s reason.
    pub fn set_pipeline(&mut self, pipeline: bool) {
        self.cache.set_pipeline(pipeline);
    }

    /// Is the fold's pipeline on?
    #[must_use]
    pub fn pipelining(&self) -> bool {
        self.cache.pipelined()
    }

    /// Choose the fold's disable policy between fires
    /// ([`Knobs::fold_disable_library`](super::Knobs::fold_disable_library)):
    /// `false`
    /// disables every absent-window node, `true` keeps pie windowed nodes
    /// enabled at fitted zero rows and disables only the library residue.
    pub fn set_fold_library(&mut self, library: bool) {
        self.cache.set_fold_library(library);
    }

    /// What the last fire's window table cost. See [`FireCost`].
    #[must_use]
    pub fn last_fire_cost(&self) -> FireCost {
        self.last
    }

    /// What this load's graph cache has done.
    #[must_use]
    pub fn graph_stats(&self) -> record::Stats {
        self.cache.stats()
    }

    /// What this load's fold has done. See [`record::FoldStats`].
    #[must_use]
    pub fn fold_stats(&self) -> record::FoldStats {
        self.cache.fold_stats()
    }

    /// **PROBE SEAM (`palo cuda-abi` wave), off by default.** Ask this load's
    /// captures to keep their `cudaGraph_t` so a probe can walk the recorded
    /// kernel nodes. The fire path does not read it.
    pub fn keep_graphs(&mut self, keep: bool) {
        self.cache.keep_graphs(keep);
    }

    /// The graphs kept by [`Shell::keep_graphs`], in capture order.
    #[must_use]
    pub fn kept_graphs(&self) -> &[(record::Key, crate::device::Graph)] {
        self.cache.kept()
    }

    /// **WHAT P6 BAKED FOR THIS LOAD, AND WHAT THIS SHELL OPENED FOR IT**:
    /// `(streams, events, forked regions, side streams open)`.
    ///
    /// The one observable of a fork from outside. A recorded graph does not
    /// carry its event points as NODES — stream capture turns a
    /// `cudaEventRecord` and the `cudaStreamWaitEvent` behind it into an edge
    /// between the launches on either side, which is exactly what one wants
    /// and exactly what makes `cudaGraphGetNodes` unable to tell a forked
    /// graph from a sequential one. So a measurement that wants to say its two
    /// arms are two different artifacts asks here.
    #[must_use]
    pub fn streams(&self) -> (u32, u32, usize, usize) {
        (
            self.compiled.streams.streams,
            self.compiled.streams.events,
            self.compiled.regions.iter().filter(|r| r.stream != 0).count(),
            self.device.lanes(),
        )
    }

    /// What this load holds on the device: `(weights, arena, pools, inputs)`.
    ///
    /// The pool figure is the CEILING its address space was reserved at, which
    /// is what this tuple has always meant and what a caller sizing a machine
    /// wants. What is mapped right now is [`Shell::elastic`], and the two are
    /// different numbers on purpose.
    #[must_use]
    pub fn footprint(&self) -> (u64, u64, u64, u64) {
        (
            self.weights.bytes(),
            self.arena.bytes(),
            self.pools.bytes(),
            self.inputs.bytes(),
        )
    }

    /// **Every pool arena's base address**, in row-then-plane order.
    ///
    /// The addresses a recorded graph reads. They are answered before one
    /// byte is mapped and do not move for the load (article 7), which is the
    /// property the whole elastic shape stands on and the one a gate should
    /// be able to check.
    #[must_use]
    pub fn pool_bases(&self) -> Vec<u64> {
        self.pools.bases()
    }

    /// **What the elastic supply is actually holding**: `(committed bytes,
    /// high-water bytes, page bytes, budget pages)`.
    ///
    /// **ONE NUMBER, ONE OWNER** (article 8). The engine owns physical commit
    /// and trim, so the engine is what can be asked how much is committed and
    /// how much has ever been — rather than the runtime re-deriving a high
    /// water by scanning its own free list, which is what the 10-second
    /// resize poll did and what died with it.
    #[must_use]
    pub fn elastic(&self) -> (u64, u64, u64, u64) {
        (
            self.pools.committed_bytes(),
            self.pools.high_water_bytes(),
            self.pools.elastic_page_bytes(),
            self.pools.elastic_budget_pages(),
        )
    }

    /// How many steps this shell has registered a settlement for and not yet
    /// seen a callback from — the run-ahead, as the shell sees it. Read by the
    /// saturation gates and by nothing on the fire path.
    #[must_use]
    pub fn airborne_steps(&self) -> u64 {
        self.airborne.count()
    }

    /// **Did this load's weight table come off the warm-boot artifact?**
    /// (alto design §7.)
    ///
    /// `true` says the host-side transform pipeline never ran. The per-load
    /// half of [`Shell::weight_cache_observed`], and what
    /// `LoadFacts::weights_from_cache` carries home to the caller.
    #[must_use]
    pub fn weights_from_cache(&self) -> bool {
        self.weights.from_cache()
    }

    /// The digest of the weight bytes actually resident on this device.
    ///
    /// What a gate compares between a cold load and a warm one.
    ///
    /// # Errors
    ///
    /// A device failure reading the store back.
    pub fn weight_digest(&self) -> Result<u64> {
        self.weights.digest()
    }

    /// The weight artifact cache's process-global census — restored, missed,
    /// stored, corrupt, declined.
    ///
    /// Process-global for the same reason [`Shell::fold_observed`] is: a gate
    /// at the runtime level holds the engine behind `Box<dyn Engine>` on a
    /// lane thread and cannot ask a shell instance anything. See
    /// [`crate::weight_cache::observed`].
    #[must_use]
    pub fn weight_cache_observed() -> crate::weight_cache::Observed {
        crate::weight_cache::observed()
    }

    /// How many descriptor-port envelopes have been resolved off guest device
    /// rings in this process. See [`crate::program::ports::resolved`], which
    /// is where the counter lives and why it is process-global.
    #[must_use]
    pub fn envelopes_resolved() -> u64 {
        crate::program::ports::resolved()
    }

    /// The fold's process-global motion mirror —
    /// `(folds, rebinds, rebind_us, swaps, prebinds, prebind_us, twins)` —
    /// for a caller that cannot reach a shell instance: the serving runtime's
    /// gates, which hold the engine behind `Box<dyn Engine>` on a lane
    /// thread. See [`record::fold_observed`] for what is published, where,
    /// and why process-global is the honest scope. An instance in hand
    /// should ask [`Shell::fold_stats`] instead — it answers the full
    /// census.
    #[must_use]
    pub fn fold_observed() -> (u64, u64, u64, u64, u64, u64, u64) {
        record::fold_observed()
    }
}
