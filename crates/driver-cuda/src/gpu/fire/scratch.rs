//! The fire's pooled buffers — one of the two names the cross-backend
//! dictionary settles.
//!
//! CUDA called this `serve::state::FireArrays` and justified it as
//! *"pooled so a capture can outlive the fire"*; Metal called it
//! `metal::Scratch` and justified it as *"so a fire's addresses are the
//! same as the last fire's"*. Those are one design named twice, and
//! `Scratch` is the name both crates now use.


/// The per-fire device arrays, POOLED across fires.
///
/// They used to be allocated and dropped every launch — `step_impl`'s own
/// comment said "KV pools (persistent), fire arrays (per launch)" — and
/// that is the one thing standing between the supergraph and the live
/// path. A captured exec bakes the addresses it recorded, so an arena
/// freed at the end of its fire can never be replayed into.
///
/// So they are kept and reused, and grown when a fire needs more than the
/// last one did. Growth MOVES a base address, which invalidates every
/// capture that recorded it — hence [`Self::epoch`], which is the
/// `PlanEpoch` `model::supergraph::Recordings` keys its execs on. A
/// bump means stale, and stale means recapture rather than a wrong
/// answer.
#[derive(Default)]
pub(crate) struct Scratch {
    pub(crate) arena: Option<crate::gpu::device::DeviceBuffer>,
    /// The unconditional attention-score sink — see [`Self::score`].
    pub(crate) score: Option<crate::gpu::device::DeviceBuffer>,
    /// The unconditional custom attention mask — see [`Self::mask`].
    pub(crate) mask: Option<crate::gpu::device::DeviceBuffer>,
    /// The driver-owned attention landing buffer — see [`Self::attn_out`].
    pub(crate) attn_out: Option<crate::gpu::device::DeviceBuffer>,
    pub(crate) named:
        std::collections::BTreeMap<model_compiler::trace::ValueId, crate::gpu::device::DeviceBuffer>,
    /// The small per-fire u32 descriptor arrays, by slot.
    pub(crate) slots: Vec<Option<crate::gpu::device::DeviceBuffer>>,
    /// PINNED host staging for those uploads, ONE PER SLOT.
    ///
    /// A `cudaMemcpyAsync` out of PAGEABLE host memory is SYNCHRONOUS — it
    /// blocks until the copy lands — and there are eight of these per fire.
    /// A `Vec` here therefore drained the stream eight times inside
    /// `pie_cuda_launch`, which is the same trap the logits D2H fell into.
    ///
    /// Per slot and not one shared buffer, because pinning is what makes the
    /// copy genuinely ASYNCHRONOUS: a single buffer would be overwritten by
    /// the next slot's upload while the previous copy was still queued. Two
    /// fires cannot collide on one slot either — the next launch waits on the
    /// previous fire's event before it reclaims anything (`InFlight`).
    pub(crate) staging: Vec<Option<crate::gpu::device::PinnedBuf>>,
    pub(crate) epoch: u64,
}

impl Scratch {

    /// Grow a pooled slot if it is too small, and bump the generation IF
    /// it moved.
    ///
    /// **THE ONLY PLACE THIS STRUCT INCREMENTS `epoch`.** It used to be
    /// six identical three-line blocks — one per pooled buffer — and a
    /// seventh pooled buffer added without its bump would have been a
    /// captured graph replaying against a freed address, which is a
    /// wrong answer rather than a fault.
    ///
    /// The bump is not removed by this, it is RELOCATED: the
    /// reallocation path owns it instead of every caller. That
    /// relocation is the whole value — `epoch` is now impossible to
    /// forget, because growing is the only way to get a buffer and
    /// growing is what bumps.
    ///
    /// Returns whether it grew, for the two callers that must also
    /// refresh contents.
    fn grow(
        slot: &mut Option<crate::gpu::device::DeviceBuffer>,
        epoch: &mut u64,
        alloc: &crate::gpu::device::Allocator,
        what: &'static str,
        bytes: usize,
    ) -> crate::Result<()> {
        if slot.as_ref().is_none_or(|b| b.len() < bytes) {
            // THE SIZE IS THE DIAGNOSIS. An engine deciding whether to
            // evict, shrink the batch or refuse needs the figure, and
            // `Error::Exhausted` carries it precisely so this does not
            // become "something ran out".
            *slot = Some(
                alloc.alloc(bytes).map_err(|_| crate::Error::exhausted(what, bytes))?,
            );
            *epoch += 1;
        }
        Ok(())
    }
    /// The activation arena, at least `bytes` wide.
    pub(crate) fn arena(
        &mut self,
        alloc: &crate::gpu::device::Allocator,
        bytes: usize,
    ) -> crate::Result<*mut std::ffi::c_void> {
        Self::grow(&mut self.arena, &mut self.epoch, alloc, "fire arena", bytes)?;
        Ok(self.arena.as_ref().expect("just grown").as_ptr())
    }

    /// The score sink, at least `bytes` wide — the same pooling discipline
    /// the arena gets, for the same reason: a captured exec bakes the
    /// address, so the sink outlives its fire and grows rather than moving
    /// every launch. See [`crate::gpu::fire::attn_score::plan_score_sink`].
    ///
    /// Returns the base and writes the CSR into its own carved span, so a
    /// replay whose KV lengths moved still scores against this fire's
    /// truth rather than the recording fire's.
    pub(crate) fn score(
        &mut self,
        alloc: &crate::gpu::device::Allocator,
        plan: &crate::gpu::fire::attn_score::ScoreSinkPlan,
        stream: crate::gpu::device::StreamRef<'_>,
    ) -> crate::Result<*mut std::ffi::c_void> {
        Self::grow(&mut self.score, &mut self.epoch, alloc, "attention score sink", plan.bytes)?;
        let b = self.score.as_mut().expect("just grown");
        let mut csr = Vec::with_capacity(plan.indptr.len() * 4);
        for v in &plan.indptr {
            csr.extend_from_slice(&v.to_le_bytes());
        }
        b.write_at(plan.indptr_offset, &csr, stream)
            ?;
        Ok(b.as_ptr())
    }

    /// The attention output slot, for a fire whose op join names none.
    ///
    /// `AttnCtx::o_out` is driver-owned by design -- a guard region's
    /// launches record no SSA output of their own -- so this is where that
    /// design finally has storage instead of a refusal. Pooled like the
    /// arena, because a capture bakes the address.
    pub(crate) fn attn_out(
        &mut self,
        alloc: &crate::gpu::device::Allocator,
        bytes: usize,
    ) -> crate::Result<*mut std::ffi::c_void> {
        let bytes = bytes.max(64);
        Self::grow(&mut self.attn_out, &mut self.epoch, alloc, "attention landing", bytes)?;
        Ok(self.attn_out.as_ref().expect("just grown").as_ptr())
    }

    /// The custom attention mask, pooled like the score sink and for the
    /// same reason. See [`crate::gpu::fire::page_mask::element_mask`].
    pub(crate) fn mask(
        &mut self,
        alloc: &crate::gpu::device::Allocator,
        plan: &crate::gpu::fire::page_mask::element_mask::ElementMaskPlan,
        stream: crate::gpu::device::StreamRef<'_>,
    ) -> crate::Result<*mut std::ffi::c_void> {
        Self::grow(&mut self.mask, &mut self.epoch, alloc, "attention mask", plan.bytes)?;
        let b = self.mask.as_mut().expect("just grown");
        b.write_at(0, &plan.mask, stream)?;
        let mut csr = Vec::with_capacity(plan.indptr.len() * 4);
        for v in &plan.indptr {
            csr.extend_from_slice(&v.to_le_bytes());
        }
        b.write_at(plan.indptr_offset, &csr, stream)
            ?;
        Ok(b.as_ptr())
    }

    /// One per-fire u32 descriptor array, by SLOT.
    ///
    /// The same discipline the arena gets, for the small arrays: the
    /// buffer is kept and its CONTENTS refreshed, so a capture that
    /// recorded the address keeps addressing something real. Slots are
    /// positional because these are a fixed list — see the constants
    /// beside the call site.
    ///
    /// Returns the device pointer rather than the buffer, so a caller
    /// holds no borrow and the next slot can be uploaded on the next line.
    pub(crate) fn upload_u32(
        &mut self,
        alloc: &crate::gpu::device::Allocator,
        slot: usize,
        vals: &[u32],
        stream: crate::gpu::device::StreamRef<'_>,
    ) -> crate::Result<*const u32> {
        if self.slots.len() <= slot {
            self.slots.resize_with(slot + 1, || None);
        }
        if self.staging.len() <= slot {
            self.staging.resize_with(slot + 1, || None);
        }
        let live = vals.len() * 4;
        let need = live.max(4);
        if self.staging[slot].as_ref().is_none_or(|p| p.len() < need) {
            // PINNED HOST memory, whose address no graph bakes — so it
            // grows without a generation bump, which is why it does not
            // go through `grow`.
            self.staging[slot] = Some(
                crate::gpu::device::PinnedBuf::new(need)
                    .map_err(|_| crate::Error::exhausted("fire staging", need))?,
            );
        }
        let pin = self.staging[slot].as_mut().expect("just sized");
        for (dst, v) in pin.as_mut_slice()[..live].chunks_exact_mut(4).zip(vals) {
            dst.copy_from_slice(&v.to_le_bytes());
        }
        Self::grow(&mut self.slots[slot], &mut self.epoch, alloc, "fire descriptor array", need)?;
        let src = &self.staging[slot].as_ref().expect("just sized").as_slice()[..live];
        let b = self.slots[slot].as_mut().expect("just grown");
        b.copy_from_host(src, stream)?;
        Ok(b.as_ptr().cast_const().cast::<u32>())
    }

    /// One named seam buffer, at least `bytes` wide, zeroed.
    ///
    /// Zeroed on every fire rather than only on allocation: the pin is
    /// per-fire state whatever its storage is, and a reused buffer still
    /// holds the last fire's values.
    pub(crate) fn named(
        &mut self,
        alloc: &crate::gpu::device::Allocator,
        v: model_compiler::trace::ValueId,
        bytes: usize,
        stream: crate::gpu::device::StreamRef<'_>,
    ) -> crate::Result<()> {
        // A `BTreeMap` entry rather than an `Option`, so it takes the
        // same path through a temporary: the invariant is that growing
        // is what bumps, and an entry that grew is still a grow.
        let mut held = self.named.remove(&v);
        Self::grow(&mut held, &mut self.epoch, alloc, "named seam buffer", bytes)?;
        let b = self.named.entry(v).or_insert(held.expect("just grown"));
        b.memset(0, stream)
    }
}
