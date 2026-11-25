//! The shell's pool residency: the activation rung and its span, evacuation
//! and repointing, slot and record copies, and the host and disk tiers.

use super::*;

impl Shell {

    /// See `Engine::set_rung`: a raise backs the carve of `to` before any
    /// fire of it, refused as [`Fault::Refused`]; a lower gives back the
    /// carve above it. Either way, where the rung's staged regions move,
    /// every frame drains first: the new rung's regions lie over the old
    /// rung's carve.
    #[allow(unsafe_code)]
    pub fn set_rung(&mut self, from: u32, to: u32) -> Result<()> {
        let reach = self.span_for(to)?;
        if crate::diag::on().tier_trace {
            eprintln!(
                "residency: rung {from} -> {to}, the carve {reach} bytes from the span's top \
                 (span backed {})",
                self.pools.span_held_bytes()
            );
        }
        if to >= from {
            self.pools.cover(reach)?;
        }
        let index = self.arena.index(to);
        let moved = index != self.arena.current();
        if moved || to < from {
            self.drain()?;
        }
        // A refused move leaves the rung as it was: the runtime keeps `from`
        // on a refusal (`lower_refused`), so a switch made before a step that
        // failed is undone, or Metal would stage in `to`'s regions while the
        // runtime cuts fires to `from` (the M1 stress: "this fire wants 2
        // token rows ... the load reserved 1" at rung 2).
        let left = self.arena.current();
        if moved {
            if let Err(fault) = host_rung(
                &self.pools,
                &mut self.inputs,
                &mut self.readout,
                left,
                false,
            ) {
                self.rehost(left);
                return Err(fault);
            }
            self.arena.set_current(to);
            if let Err(fault) = host_rung(
                &self.pools,
                &mut self.inputs,
                &mut self.readout,
                index,
                true,
            ) {
                self.unswitch(from, index, left);
                return Err(fault);
            }
        }
        if to >= from {
            self.rung = to;
            return Ok(());
        }
        // SAFETY: drained just above, on the one thread that submits; the
        // runtime lowers only once every fire wider than `to` retired, and
        // the rung's regions lie inside what stays.
        if let Err(fault) = unsafe { self.pools.narrow(reach) } {
            if moved {
                self.unswitch(from, index, left);
            }
            return Err(fault);
        }
        self.rung = to;
        Ok(())
    }

    /// Hosts rung `slot`'s regions again after a move that failed; a failure
    /// here leaves the load unable to stage, and says so.
    pub(super) fn rehost(&mut self, slot: usize) {
        if let Err(fault) = host_rung(&self.pools, &mut self.inputs, &mut self.readout, slot, true) {
            tracing::error!(%fault, "metal: a refused rung move could not host the rung it kept");
        }
    }

    /// Undoes a rung switch to `index` whose next step failed: `index`'s
    /// regions let go, the arena back on `from` and its regions (`left`)
    /// hosted again.
    pub(super) fn unswitch(&mut self, from: u32, index: usize, left: usize) {
        if let Err(fault) = host_rung(&self.pools, &mut self.inputs, &mut self.readout, index, false) {
            tracing::error!(%fault, "metal: a refused rung move could not let go of the rung it tried");
        }
        self.arena.set_current(from);
        self.rehost(left);
    }

    /// The span's bytes from its end a rung of `to` tokens takes, as the
    /// runtime counts it: the model's chunks. Nothing guest lies in the
    /// span: a guest program's state is its instances' and its pair's, in
    /// `Instance` slots (`engine::Engine::guest_state`).
    pub(super) fn span_for(&self, to: u32) -> Result<u64> {
        let chunk = self.pools.layout().chunk_bytes.max(1);
        Ok(self.arena.rung_reach(to).div_ceil(chunk) * chunk)
    }

    /// The bytes the span backs from the range's end now: the rung's
    /// carve, and nothing guest beside it.
    #[must_use]
    pub fn span_covered(&self) -> u64 {
        self.pools.covered()
    }

    /// See `Engine::guest_state`: what an instance of `program_id` at
    /// `extents` and its pair hold in `Instance` slots.
    pub fn guest_state(
        &self,
        program_id: engine::ProgramId,
        extents: &eta_exec::Extents,
    ) -> Result<engine::GuestState> {
        self.programs.guest_state(&self.pools, program_id, extents)
    }

    /// Raises the rung to the widest, backing its carve: what a caller that
    /// fires this shell itself, with no runtime raising the rung, asks
    /// first (tuning, the tests).
    pub fn raise_whole(&mut self) -> Result<()> {
        self.set_rung(0, u32::MAX)
    }

    /// See `Engine::release_span`: every frame drains first.
    #[allow(unsafe_code)]
    pub fn release_span(&mut self, segments: u32) -> Result<()> {
        self.drain()?;
        // SAFETY: drained just above, on the one thread that submits; the
        // runtime holds no state chunk in what goes back.
        unsafe { self.pools.release_span(segments) }
    }

    /// See `Engine::release_segments`: every frame drains first.
    #[allow(unsafe_code)]
    pub fn release_segments(&mut self, segments: u32) -> Result<()> {
        self.drain()?;
        // SAFETY: drained just above, on the one thread that submits; the
        // runtime evacuated the segments' chunks before asking.
        unsafe { self.pools.release_segments(segments) }
    }

    /// The state pool's chunk table as it stands (`Engine::chunk_table`).
    #[must_use]
    pub fn chunk_table(&self) -> &engine::chunk::ChunkTable {
        self.pools.chunk_table()
    }

    /// See `Engine::evacuate`: the moved chunks' bytes land, and the chunk
    /// table changes, before any later fire is encoded.
    pub fn evacuate(&mut self, evacuation: &engine::chunk::Evacuation) -> Result<()> {
        if evacuation.moves.is_empty() {
            return Ok(());
        }
        self.drain()?;
        let before = self.pools.chunk_table().clone();
        let mut frame = self.device.frame()?;
        self.pools.evacuate(&mut frame, evacuation)?;
        frame.commit()?;
        // Guest tables, rings and landed adapters in moved chunks: the refs
        // stand, their addresses do not.
        let Err(refused) = self.repoint_after_move() else {
            return Ok(());
        };
        // Refused past the copies (a holder whose refs this table cannot
        // resolve): the evacuation is taken back whole, so the refusal
        // means nothing moved (`Engine::evacuate`), the runtime takes its
        // raise back too, and the two keep one map. Answered as half done,
        // the runtime's map and this one would part, and a later hand-out
        // could land on bytes the engine had moved a holder onto.
        tracing::warn!(
            %refused,
            moves = evacuation.moves.len(),
            "metal: an evacuation's rebind was refused; its moves are taken back"
        );
        let mut frame = self.device.frame()?;
        self.pools.evacuate_back(&mut frame, evacuation, before)?;
        frame.commit()?;
        self.repoint_after_move().map_err(|again| {
            Fault::program(
                "serve::evacuate",
                format!(
                    "an evacuation refused ({refused}) and taken back could not repoint where its \
                     holders lay before it ({again}): a bug"
                ),
            )
        })?;
        Err(refused)
    }

    /// Every landing, guest table, ring and lay pointed where the chunk
    /// table puts their refs now ([`Self::evacuate`]).
    pub(super) fn repoint_after_move(&mut self) -> Result<()> {
        let landings: Vec<Landing> = self.landings.values().cloned().collect();
        for landing in &landings {
            self.point_landing(landing)?;
        }
        self.programs.rebind(&self.pools)?;
        #[cfg(feature = "fault-injection")]
        if FAIL_NEXT_REPOINT.with(|next| next.replace(false)) {
            return Err(Fault::program(
                "serve::evacuate",
                "a refusal injected by `fail_next_repoint`, past the copies and the rebind",
            ));
        }
        Ok(())
    }

    /// Whole-slot copies a fire states (`Lane::slot_copies`), each
    /// `[src, dst]` of its shape: they land before any lane of the fire
    /// reads or writes.
    pub fn copy_slot_pairs(
        &mut self,
        copies: &[(engine::chunk::ShapeId, Vec<[u32; 2]>)],
    ) -> Result<()> {
        if copies.iter().all(|(_, pairs)| pairs.is_empty()) {
            return Ok(());
        }
        self.drain()?;
        let mut frame = self.device.frame()?;
        for (shape, pairs) in copies {
            if *shape == crate::store::layout::FIXED {
                for &[src, dst] in pairs {
                    match self.frozen_ahead.get(&src).copied() {
                        Some(at) => self.frozen_ahead.insert(dst, at),
                        None => self.frozen_ahead.remove(&dst),
                    };
                }
            }
            let moves: Vec<(u32, u32)> = pairs.iter().map(|&[src, dst]| (src, dst)).collect();
            self.pools.copy_shape(&mut frame, *shape, &moves)?;
        }
        self.grafted = Some(frame.commit_async(None)?);
        Ok(())
    }

    /// See `Engine::copy_slots`.
    pub fn copy_slots(&mut self, copy: &engine::chunk::SlotCopy) -> Result<()> {
        if let Some(frame) = self.encode_copy(copy)? {
            self.grafted = Some(frame.commit_async(None)?);
        }
        Ok(())
    }

    /// `copy` drained behind every frame and encoded (a host copy to or
    /// from the slot files done here); the frame to
    /// commit, `None` for an empty copy.
    pub(super) fn encode_copy(
        &mut self,
        copy: &engine::chunk::SlotCopy,
    ) -> Result<Option<Frame>> {
        if copy.slots.is_empty() && copy.runs.is_empty() {
            return Ok(None);
        }
        self.drain()?;
        // Every fold has landed and been told once drained: a record moved
        // between tiers carries no length ahead of the runtime's, and the
        // slot it lands in starts from what its verb states, never from
        // a count its last occupant left.
        if copy.shape == crate::store::layout::FIXED && copy.src != copy.dst {
            for &(src, dst) in &copy.slots {
                self.frozen_ahead.remove(&src);
                self.frozen_ahead.remove(&dst);
            }
        }
        let mut frame = self.device.frame()?;
        self.pools.copy_slots(&mut frame, copy)?;
        Ok(Some(frame))
    }

    /// See `Engine::copy_slots_landing`: whether `copy`'s sources may go
    /// back now. A copy on the device is encoded into a command buffer the
    /// device may still be running when this returns: `false`, and
    /// `landed`'s sink is called with its token from that buffer's
    /// completion; with no sink installed (the runtime installs it before
    /// any copy) it is refused by name, before anything is encoded. A move
    /// to or from the slot files is a host copy done here,
    /// past the drain: `true`.
    pub fn copy_slots_landing(
        &mut self,
        copy: &engine::chunk::SlotCopy,
        landed: Option<(engine::LandingSink, u64)>,
    ) -> Result<bool> {
        let device = engine::MemoryDomain::Device { ordinal: 0, reach: engine::DeviceReach::HostShared };
        if copy.src != device || copy.dst != device {
            self.copy_slots(copy)?;
            return Ok(true);
        }
        let Some((sink, token)) = landed else {
            return Err(Fault::Unbound {
                what: "a copy on the device with no landing sink installed \
                       (Engine::on_landed comes before any copy_slots_landing)"
                    .into(),
            });
        };
        let Some(frame) = self.encode_copy(copy)? else {
            return Ok(true);
        };
        // Told once the buffer retires, faulted or not: a faulted copy has
        // stopped reading its sources too, and its fault reaches the next
        // drain.
        self.grafted = Some(frame.commit_async(Some(Box::new(move |_| sink(token, engine::Landing::Landed))))?);
        Ok(false)
    }

    /// Slot ref `slot` of shape `id`, read whole on the host once every
    /// copy has settled.
    pub fn read_slot_bytes(&mut self, id: engine::chunk::ShapeId, slot: u32) -> Result<Vec<u8>> {
        self.settle_copies()?;
        self.pools.read_slot_bytes(id, slot)
    }

    /// Writes one slot's `bytes` into slot ref `slot` of shape `id` once
    /// every copy has settled.
    pub fn write_slot_bytes(
        &mut self,
        id: engine::chunk::ShapeId,
        slot: u32,
        bytes: &[u8],
    ) -> Result<()> {
        self.settle_copies()?;
        self.pools.write_slot_bytes(id, slot, bytes)
    }

    /// The bytes one disk chunk takes.
    #[must_use]
    pub fn disk_chunk_bytes(&self) -> u64 {
        self.pools.disk_chunk_bytes()
    }

    /// Test builds (`test-slot-fill`): the runtime's given-back slots
    /// poisoned before this engine's next verb ([`engine::chunk::PoisonQueue`]):
    /// a slot is given back only once every fire that touched it retired
    /// and every copy reading it landed, and its next owner writes nothing
    /// before a verb of its own, so a host fill here races neither. The
    /// ranges poisoned.
    #[cfg(feature = "test-slot-fill")]
    pub fn poison_given_back(&self, ranges: &[(u64, u64)]) -> usize {
        self.pools.poison_ranges(ranges)
    }

    /// See `Engine::settle_copies`: the last copy's command buffer retires.
    pub fn settle_copies(&mut self) -> Result<()> {
        if let Some(grafted) = self.grafted.take() {
            grafted.wait()?;
        }
        Ok(())
    }

}
