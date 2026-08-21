//! The fire's pooled buffers.
//!
//! Pooled so a capture can outlive the fire: a recorded graph bakes the
//! addresses it saw.

/// The per-fire device arrays, pooled across fires.
///
/// A captured exec bakes the addresses it recorded, so buffers are grown, not
/// freed. Growth moves a base address and bumps [`Self::epoch`], the key
/// `Recordings` files execs under: stale means recapture, not a wrong answer.
#[derive(Default)]
pub(crate) struct Scratch {
    pub(crate) arena: Option<crate::device::DeviceBuffer>,
    /// The unconditional attention-score sink — see [`Self::score`].
    pub(crate) score: Option<crate::device::DeviceBuffer>,
    /// The unconditional custom attention mask — see [`Self::mask`].
    pub(crate) mask: Option<crate::device::DeviceBuffer>,
    /// The driver-owned attention landing buffer — see [`Self::attn_out`].
    pub(crate) attn_out: Option<crate::device::DeviceBuffer>,
    /// The attention log-sum-exp — see [`Self::lse`].
    pub(crate) lse: Option<crate::device::DeviceBuffer>,
    /// The per-row live flags — see [`Self::row_valid`].
    pub(crate) row_valid: Option<crate::device::DeviceBuffer>,
    pub(crate) named:
        std::collections::BTreeMap<model_ir::trace::ValueId, crate::device::DeviceBuffer>,
    /// The small per-fire u32 descriptor arrays, by slot — see [`slot`].
    pub(crate) slots: Vec<Option<crate::device::DeviceBuffer>>,
    /// Pinned host staging for those uploads, one per slot.
    ///
    /// Pageable-source `cudaMemcpyAsync` is synchronous; pinning makes it
    /// async. One per slot, since a shared buffer would be overwritten while
    /// the previous slot's copy was still queued.
    pub(crate) staging: Vec<Option<crate::device::PinnedBuf>>,
    pub(crate) epoch: crate::fire::recordings::PlanEpoch,
}

/// Which slot holds what. The numbers are the interface: two slots claiming one
/// index silently makes the second upload overwrite the first.
pub(crate) mod slot {
    pub(crate) const IDS: usize = 0;
    pub(crate) const POS: usize = 1;
    pub(crate) const KV_INDICES: usize = 2;
    pub(crate) const KV_INDPTR: usize = 3;
    pub(crate) const KV_LENS: usize = 4;
    pub(crate) const QO: usize = 5;
    pub(crate) const W_PAGE: usize = 6;
    pub(crate) const W_OFF: usize = 7;
    pub(crate) const SAMPLED: usize = 8;
    // A peel's tail needs its own: the prefix reads the fire's CSRs after the
    // tail has uploaded, so the two must not share a slot.
    pub(crate) const TAIL_INDPTR: usize = 9;
    pub(crate) const TAIL_LENS: usize = 10;
    pub(crate) const TAIL_INDICES: usize = 11;
    pub(crate) const TAIL_QO: usize = 12;
    /// `"request_of_token"` — the runtime stream a text may name; staged only
    /// when the fire's plan names it.
    pub(crate) const REQ_OF_TOKEN: usize = 13;
}

impl Scratch {
    /// Grow a pooled slot if it is too small, bumping `epoch` if it moved. The
    /// only place `epoch` is incremented, so a capture can never replay against
    /// a freed address without going stale first.
    fn grow(
        slot: &mut Option<crate::device::DeviceBuffer>,
        epoch: &mut crate::fire::recordings::PlanEpoch,
        alloc: &crate::device::Allocator,
        what: &'static str,
        bytes: usize,
    ) -> crate::Result<()> {
        if slot.as_ref().is_none_or(|b| b.len() < bytes) {
            // `Exhausted` carries the byte figure the engine needs to choose
            // between evicting, shrinking the batch and refusing.
            *slot = Some(alloc.alloc(bytes).map_err(|_| crate::Error::exhausted(what, bytes))?);
            epoch.bump();
        }
        Ok(())
    }
    /// The activation arena, at least `bytes` wide.
    pub(crate) fn arena(
        &mut self,
        alloc: &crate::device::Allocator,
        bytes: usize,
    ) -> crate::Result<*mut std::ffi::c_void> {
        Self::grow(&mut self.arena, &mut self.epoch, alloc, "fire arena", bytes)?;
        Ok(self.arena.as_ref().expect("just grown").as_ptr())
    }

    /// The score sink, at least `plan.bytes` wide. The CSR goes in its own
    /// carved span, so a replay whose KV lengths moved scores against this
    /// fire's truth.
    pub(crate) fn score(
        &mut self,
        alloc: &crate::device::Allocator,
        plan: &crate::fire::attn_score::ScoreSinkPlan,
        stream: crate::device::StreamRef<'_>,
    ) -> crate::Result<*mut std::ffi::c_void> {
        Self::grow(&mut self.score, &mut self.epoch, alloc, "attention score sink", plan.bytes)?;
        let b = self.score.as_mut().expect("just grown");
        let mut csr = Vec::with_capacity(plan.indptr.len() * 4);
        for v in &plan.indptr {
            csr.extend_from_slice(&v.to_le_bytes());
        }
        b.write_at(plan.indptr_offset, &csr, stream)?;
        Ok(b.as_ptr())
    }

    /// The attention output slot, for a fire whose op join names none:
    /// `AttnCtx::o_out` is driver-owned, since a guard region's launches record
    /// no SSA output of their own.
    pub(crate) fn attn_out(
        &mut self,
        alloc: &crate::device::Allocator,
        bytes: usize,
    ) -> crate::Result<*mut std::ffi::c_void> {
        let bytes = bytes.max(64);
        Self::grow(&mut self.attn_out, &mut self.epoch, alloc, "attention landing", bytes)?;
        Ok(self.attn_out.as_ref().expect("just grown").as_ptr())
    }

    /// The log-sum-exp the attention dispatches write beside their output.
    pub(crate) fn lse(
        &mut self,
        alloc: &crate::device::Allocator,
        bytes: usize,
    ) -> crate::Result<*mut std::ffi::c_void> {
        Self::grow(&mut self.lse, &mut self.epoch, alloc, "attention lse", bytes.max(64))?;
        Ok(self.lse.as_ref().expect("just grown").as_ptr())
    }

    /// Which rows of the fire are live, for the KV write descriptors.
    pub(crate) fn row_valid(
        &mut self,
        alloc: &crate::device::Allocator,
        rows: usize,
        stream: crate::device::StreamRef<'_>,
    ) -> crate::Result<*mut std::ffi::c_void> {
        Self::grow(&mut self.row_valid, &mut self.epoch, alloc, "row valid", rows.max(64))?;
        let b = self.row_valid.as_mut().expect("just grown");
        b.memset(1, stream)?;
        Ok(b.as_ptr())
    }

    /// The custom attention mask. See [`crate::fire::page_mask::element_mask`].
    pub(crate) fn mask(
        &mut self,
        alloc: &crate::device::Allocator,
        plan: &crate::fire::page_mask::element_mask::ElementMaskPlan,
        stream: crate::device::StreamRef<'_>,
    ) -> crate::Result<*mut std::ffi::c_void> {
        Self::grow(&mut self.mask, &mut self.epoch, alloc, "attention mask", plan.bytes)?;
        let b = self.mask.as_mut().expect("just grown");
        b.write_at(0, &plan.mask, stream)?;
        let mut csr = Vec::with_capacity(plan.indptr.len() * 4);
        for v in &plan.indptr {
            csr.extend_from_slice(&v.to_le_bytes());
        }
        b.write_at(plan.indptr_offset, &csr, stream)?;
        Ok(b.as_ptr())
    }

    /// One per-fire u32 descriptor array, by slot. Returns the pointer, not the
    /// buffer, so the caller holds no borrow and the next slot can upload.
    pub(crate) fn upload_u32(
        &mut self,
        alloc: &crate::device::Allocator,
        slot: usize,
        vals: &[u32],
        stream: crate::device::StreamRef<'_>,
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
            // Pinned host memory: no graph bakes its address, so it grows
            // without an epoch bump and does not go through `grow`.
            self.staging[slot] = Some(
                crate::device::PinnedBuf::new(need)
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

    /// One named seam buffer, at least `bytes` wide, zeroed every fire — a
    /// reused buffer still holds the last fire's values.
    pub(crate) fn named(
        &mut self,
        alloc: &crate::device::Allocator,
        v: model_ir::trace::ValueId,
        bytes: usize,
        stream: crate::device::StreamRef<'_>,
    ) -> crate::Result<()> {
        // The failure path must put the buffer back: `grow` allocates before it
        // assigns, so an `Exhausted` here would drop a buffer whose address a
        // capture baked, with no epoch bump to mark it stale.
        let mut held = self.named.remove(&v);
        let grown = Self::grow(&mut held, &mut self.epoch, alloc, "named seam buffer", bytes);
        if let Some(back) = held {
            self.named.insert(v, back);
        }
        grown?;
        let b = self.named.get_mut(&v).expect("just grown");
        b.memset(0, stream)
    }
}
