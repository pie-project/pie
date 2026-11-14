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
        self.pools.evacuate(self.device.stream(), evacuation)?;
        // A guest's rings bind their slots' chunks by place.
        self.rebase_programs()
    }

    /// The state pool as the runtime counts it
    /// ([`crate::store::Pools::chunk_facts`]).
    pub fn chunk_facts(&mut self) -> crate::Result<engine::chunk::ChunkFacts> {
        let max_tokens = self.budget().max_tokens;
        let rungs = self.arena.rungs(self.compiled.arena_forked.as_ref());
        self.pools.chunk_facts(max_tokens, &rungs)
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
    pub fn weight_residency(
        &mut self,
        residency: &engine::chunk::WeightResidency,
    ) -> crate::Result<()> {
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
        } else if flipped {
            if self.pools.by_refs() {
                // The lattice fires lanes this shell seats itself, which a
                // pool whose slots the runtime hands out no longer has.
                if self.pools.weights_located() {
                    // The recorded bodies keep the plain-load select they
                    // chose, which reads a located table as well.
                    eprintln!(
                        "engine-cuda: a held weight set arrived after lanes named slot refs; \
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
                self.cache.forget_bodies();
                self.arm_bodies()?;
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
                eprintln!(
                    "engine-cuda: the weights the pool holds whole are read from it alone; \
                     {} MiB of their host copy freed",
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

    /// Copies whole slots of `shape` between tiers (CONTRACT-BC 22), on
    /// the stream behind the fires enqueued and ahead of the next.
    pub fn copy_tier_slots(
        &mut self,
        shape: engine::chunk::ShapeId,
        from: crate::store::Tier,
        to: crate::store::Tier,
        pairs: &[(u32, u32)],
    ) -> crate::Result<()> {
        let swap = (!super::diag::on().tier_inline).then(|| self.device.swap_stream());
        self.pools
            .copy_tier_slots(self.device.stream(), swap, shape, from, to, pairs)
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
