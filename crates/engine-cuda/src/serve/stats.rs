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
        // A load with no staging attention never asks its width.
        let stages = self.staging.contains(&true);
        crate::window::Staging {
            regions: &self.staging,
            from: if stages {
                self.device.ctx().tuning().compressed_prefill_from()
            } else {
                0
            },
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

    /// What the runtime's host cover depends on in this load
    /// (`engine::Engine::cover_fingerprint`): the device and its kernel
    /// libraries' code ([`crate::stored::Keys`]; the device a part of its
    /// own, as `pie tune export` finds a device's covers by), the guest emitter's
    /// version, the decode products' kernels as picked (their routes'
    /// table) and the lanes and tokens a fire spans.
    #[must_use]
    pub fn cover_fingerprint(&self) -> String {
        let routes = self
            .device
            .ctx()
            .routes()
            .map_or(0, |routes| routes.fingerprint());
        format!(
            "cuda|{}|kernels {}|emitter {}|routes {routes:#010x}|lanes {} tokens {}",
            profile::fire::Now::on(&self.keys),
            self.keys.codes_said(),
            eta_compiler::codegen::cuda::CUDA_GENERATED_EMITTER_VERSION,
            self.budget.max_lanes,
            self.budget.max_tokens,
        )
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
    pub fn buffer_bytes(&self) -> u64 {
        // The rs buffer's rows are the records', in the state pool.
        0
    }

    #[must_use]
    pub fn mode(&self) -> Graphs {
        self.graphs
    }

    /// The rungs this load weighed its top by and the estimate it weighed
    /// them with: what a check of the estimate reads.
    #[must_use]
    pub fn weighed(&self) -> Option<&crate::top::Weighed> {
        self.weighed.as_ref()
    }

    /// What the device's free bytes paid for the bodies the load armed,
    /// the figure a load with no record measures (0 where none armed).
    #[must_use]
    pub const fn bodies_took(&self) -> u64 {
        self.bodies_took
    }

    #[must_use]
    pub fn body_stats(&self) -> record::BodyStats {
        self.cache.body_stats()
    }

    /// The key of the body the last fire replayed (`None` till one has).
    #[must_use]
    pub fn replayed_body(&self) -> Option<&record::BodyKey> {
        self.cache.replayed()
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
        self.evacuate_with(evacuation, &[]).map(drop)
    }

    /// [`Self::evacuate`], all or nothing: every check it makes (the table
    /// it leaves, each guest's state where it then lies and its claims
    /// there, each channel span of `channels`, `(slots, bytes)`) is made
    /// before anything moves, and an `Err` leaves the pool, the claims and
    /// every ring as they were. Past the copy's enqueue nothing fails but
    /// the device. Answers each channel's address after.
    pub fn evacuate_with(
        &mut self,
        evacuation: &engine::chunk::Evacuation,
        channels: &[(Vec<u32>, u64)],
    ) -> crate::Result<Vec<u64>> {
        let stream = self.device.stream();
        self.pools.fence_all_swaps(stream)?;
        let mut evacuated = self.pools.prepare_evacuation(evacuation)?;
        let programs = &self.programs;
        let (rebase, at) = self.pools.as_if(&mut evacuated, |pools| {
            // A guest's rings bind their slots' chunks by place.
            let rebase = programs.plan_rebase(&|item, bytes| pools.guest_item(item, bytes))?;
            let at = channels
                .iter()
                .map(|(slots, bytes)| {
                    pools.guest_span(engine::chunk::SlotRole::Channel, slots, *bytes)
                })
                .collect::<crate::Result<Vec<u64>>>()?;
            crate::Result::Ok((rebase, at))
        })?;
        let before = self.programs.claim_rebase(&rebase)?;
        if let Err(fault) = self.pools.move_chunks(stream, evacuated) {
            self.programs.unclaim_rebase(before);
            return Err(fault);
        }
        drop(before);
        self.programs.apply_rebase(rebase);
        self.pools.restage_after_move(stream)?;
        Ok(at)
    }

    /// The state pool's chunk table as it stands (`Engine::chunk_table`).
    #[must_use]
    pub fn chunk_table(&self) -> &engine::chunk::ChunkTable {
        self.pools.chunk_table()
    }

    /// The state pool as the runtime counts it
    /// ([`crate::store::Pools::chunk_facts`]).
    pub fn chunk_facts(&mut self) -> crate::Result<engine::chunk::ChunkFacts> {
        let max_tokens = self.budget().max_tokens;
        let rungs = self.arena.rungs(self.compiled.arena_forked.as_ref());
        let mut facts = self.pools.chunk_facts(max_tokens, &rungs)?;
        (facts.held_bytes, facts.outside_bytes) = self.held_and_outside();
        // Each rung's replay scratch, from the extended run's reserve
        // (`store::rs::Buffers::reserve`): `RsScratch::need(rows_ext, row)`,
        // `(rows_ext + 64) * row + 4 MiB`, laid for the widest bucket's
        // `rows_ext(max_tokens)` whatever the fire's bucket, so the rows a
        // fire's extended run takes, its own and its replayed (a decoding
        // lane's none: its decode step reads them as a linear attention, so
        // its scratch is always compact), are at most that many.
        if let Some(buffers) = &self.buffers {
            let row = buffers.ext_row_bytes();
            let rows = u64::from(crate::store::rs::Buffers::rows_ext(max_tokens));
            facts.act_replay = facts
                .act_chunks
                .iter()
                .map(|&(tokens, _)| engine::RungReplay {
                    own_max: Some(tokens),
                    carried_max: Some(max_tokens),
                    scratch: engine::ScratchBytes {
                        slot_bytes: vec![row],
                        lane_bytes: 0,
                        room_bytes: rows.saturating_mul(row),
                        pad: 1,
                    },
                })
                .collect();
            facts.rs_compact = true;
            facts.rs_scratch_per_fire = false;
        }
        Ok(facts)
    }

    /// Names the recurrent record whose pieces lie in the slot refs
    /// `pieces`: the slot a lane then names it by (its first).
    pub fn name_record(&mut self, pieces: &[u32]) -> crate::Result<u32> {
        self.pools.name_record(pieces)
    }

    /// Seats the weights of a held load, before it arms: in pool slots of
    /// the load's own choosing, filled from the serving file (`filed`) or
    /// the pinned copy it keeps, where the pool holds them all; else a filed
    /// load's go on a pinned copy the fires read in place. The runtime's
    /// first residency then says where each lies.
    /// `copied`: the weights are on a pageable host copy no fire may read
    /// (`Tier::open`'s held copy); where the pool seats none of them (it
    /// cannot hold them), the runtime streams them from a pinned copy made
    /// of it, as from the file's.
    pub(crate) fn preseat_weights(&mut self, filed: bool, copied: bool) -> crate::Result<()> {
        self.preseated_as = Some((filed, copied));
        if filed {
            self.pools.file_weights();
        }
        match self.pools.preseat_plan() {
            Some(plan) => {
                let chunks = self.pools.weight_chunks(&plan);
                self.pools.keep_clear(chunks);
                self.pools.preseat_weights(self.device.stream(), &plan)?;
            }
            None if filed || copied => self.pools.pin_weight_hosts()?,
            None => {}
        }
        self.device.synchronize()?;
        self.pools.release_file_pages();
        Ok(())
    }

    /// See `Engine::weight_residency`: stream-ordered before the next fire.
    /// Answers what the fit holds back of the chunks stated
    /// ([`Self::held_and_outside`]) once the set is taken.
    pub fn weight_residency(
        &mut self,
        residency: &engine::chunk::WeightResidency,
    ) -> crate::Result<u64> {
        // The pages this shell seats for its lanes itself stay clear of the
        // chunks the runtime lays weights in. One already there is let go
        // with every other, which only a shell holding no lane's rows may.
        let chunks = self.pools.weight_chunks(residency);
        if self.pools.seated_in(&chunks) {
            if let Some(slot) = self.held.iter().position(|&held| held > 0) {
                return Err(crate::error::Fault::Unbound {
                    what: format!(
                        "a weight slot in a chunk this shell seated lane pages in, while \
                         slot {slot} holds rows: the pages cannot move"
                    ),
                });
            }
            self.device.synchronize()?;
        }
        self.pools.keep_clear(chunks);
        let flipped = self
            .pools
            .weight_residency(self.device.stream(), residency)?;
        if self.arm_deferred {
            // The load streams its weights and armed nothing at load: the
            // lattice arms here, over the slots this residency names.
            if self.pools.by_refs() {
                return Err(crate::error::Fault::Unbound {
                    what: "the starting weight residency of a load that arms at it came after \
                           lanes named slot refs: send it before the first fire"
                        .to_string(),
                });
            }
            self.arm_deferred = false;
            self.device.synchronize()?;
            // The atoms the residency leaves on the host copy are read
            // through a slot the pool holds while the lattice arms: over the
            // copy every prefill would read its experts across the bus.
            self.pools.stand_in(true)?;
            let armed = self.arm_bodies();
            self.device.synchronize()?;
            self.pools.stand_in(false)?;
            armed?;
            // Every composition has fired and the bodies hold what they
            // take: the fit the load left unsettled settles, once, and the
            // answer below says what it still holds back.
            self.settle_fit()?;
        } else if flipped {
            if self.pools.by_refs() {
                // The lattice fires lanes this shell seats itself, which a
                // pool whose slots the runtime hands out no longer has.
                if self.pools.weights_located() {
                    // The recorded bodies keep the plain-load select they
                    // chose, which reads a located table as well.
                    tracing::info!(
                        "a held weight set arrived after lanes named slot refs; \
                         the recorded bodies keep the per-route select (send the set before \
                         the first fire)"
                    );
                } else {
                    // The bodies chose selects for experts that all lay on
                    // the device, and one now lies on the host copy.
                    return Err(crate::error::Fault::Unbound {
                        what: "a weight residency that streams, after lanes named slot refs, \
                               on a load whose bodies were recorded over held weights: send \
                               the first residency before the first fire"
                            .to_string(),
                    });
                }
            } else {
                // A recorded body keeps the select it chose for where a
                // bank's experts lay: the lattice is recorded again over
                // where they lie now.
                self.device.synchronize()?;
                self.rearm_bodies()?;
            }
        }
        if self.pools.copied_from_file() {
            self.device.synchronize()?;
            self.pools.release_file_pages();
        }
        // A set held whole for the load's life is read from the pool alone:
        // its host copy goes.
        let held = self.pools.held_weights();
        if !held.is_empty()
            && let Some(tier) = self.weights.experts_mut()
        {
            self.device.synchronize()?;
            let freed = tier.release_host(&held);
            if freed > 0 {
                self.pools.drop_weight_hosts();
                tracing::info!(
                    "the weights the pool holds whole are read from it alone; \
                     {} MiB of their host copy freed",
                    freed >> 20
                );
            }
        }
        Ok(self.pools.kept_back_bytes())
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

    /// The record a lane bound to none reads and writes, where lanes name
    /// refs and the load holds state: the null record, ref 0.
    #[must_use]
    pub fn null_record(&self) -> Option<u32> {
        (self.pools.by_refs() && self.pools.has_record()).then_some(0)
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

    /// Copies whole slots of `shape` between tiers (CONTRACT-BC 22), on
    /// the stream behind the fires enqueued and ahead of the next.
    pub fn copy_tier_slots(
        &mut self,
        shape: engine::chunk::ShapeId,
        from: crate::store::Tier,
        to: crate::store::Tier,
        pairs: &[(u32, u32)],
    ) -> crate::Result<()> {
        // A record that leaves the device, or arrives on it, moves when the
        // runtime's count of its folds has caught up (it moves no set with
        // a device-counted fold unsettled): what a fold left ahead of that
        // count belongs to neither the slot it left nor the one it lands
        // in, and a later record seated there must not inherit it.
        if shape == crate::store::layout::FIXED {
            for &(src, dst) in pairs {
                if from == crate::store::Tier::Device {
                    self.rs_ahead.remove(&src);
                }
                if to == crate::store::Tier::Device {
                    self.rs_ahead.remove(&dst);
                }
            }
        }
        let swap = (!super::diag::on().tier_inline).then(|| self.device.swap_stream());
        self.pools
            .copy_tier_slots(self.device.stream(), swap, shape, from, to, pairs)
    }

    /// Waits, on the host, until every move between the device and the host
    /// tier enqueued so far has landed.
    pub fn settle_moves(&mut self) -> crate::Result<()> {
        self.pools.settle_swaps()
    }

    /// Where the next move between the host tier and the slot file is told
    /// how it ended ([`crate::store::Pools::defer_disk_tell`]); returns a
    /// tell no such move took.
    pub fn defer_disk_tell(
        &mut self,
        told: Option<super::landing::Told>,
    ) -> Option<super::landing::Told> {
        self.pools.defer_disk_tell(told)
    }

    /// Tells `landed` how every move between the device and the host tier
    /// enqueued so far ended (landed, or failed with its fault), from a
    /// waiter of its own: nothing is enqueued on the swap stream for it, so
    /// no later move waits on it. `false` (and `landed` dropped) where the
    /// moves ride no swap stream, so each has landed by the time it returned.
    pub fn tell_swaps_landed(&mut self, landed: super::landing::Told) -> crate::Result<bool> {
        if super::diag::on().tier_inline {
            return Ok(false);
        }
        super::landing::tell_when_landed(self.device.swap_stream(), landed)?;
        Ok(true)
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

    /// The pool's bytes with memory behind them now, and the most that ever
    /// had at once.
    #[must_use]
    pub fn elastic(&self) -> (u64, u64) {
        (self.pools.committed_bytes(), self.pools.high_water_bytes())
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
