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
    /// The adapter correction's xAᵀ staging — see [`Self::lora_gate`].
    pub(crate) lora_gate: Option<crate::device::DeviceBuffer>,
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

/// The granularity every pooled slot is rounded up to. Small enough that the
/// tiny descriptor arrays (`kv_indices` is four bytes per page) do not carry a
/// meaningful pool cost, large enough that a slot which grows by one entry per
/// page does not reallocate for sixty-four of them.
const GROW_ALIGN: usize = 256;

/// How big a slot that held `held` bytes should be made when it is asked for
/// `bytes`: at least what was asked, at least double what it had, rounded up to
/// [`GROW_ALIGN`].
///
/// Split out from the two callers so the arithmetic is testable without a
/// device, and so both the device slot and its pinned staging round the same
/// way — a staging buffer that grew on a different schedule from the slot it
/// feeds would reallocate on the fires the slot did not.
const fn headroom(held: usize, bytes: usize) -> usize {
    let doubled = match held.checked_mul(2) {
        Some(n) => n,
        None => usize::MAX,
    };
    let want = if bytes > doubled { bytes } else { doubled };
    match want.checked_next_multiple_of(GROW_ALIGN) {
        Some(n) => n,
        // Rounding up overflowed, so `want` is within `GROW_ALIGN` of the
        // address space; the ask itself is what has to be honoured.
        None => bytes,
    }
}

impl Scratch {
    /// Grow a pooled slot if it is too small, bumping `epoch` if it moved. The
    /// only place `epoch` is incremented, so a capture can never replay against
    /// a freed address without going stale first.
    ///
    /// # Why a slot is given more than it asked for
    ///
    /// The epoch is the graph cache's staleness key, and a bump invalidates
    /// EVERY captured exec, not just the one that touched this slot. Recapture
    /// is not cheap: a Qwen3-0.6B decode records 535 launches, measured at
    /// **3.0–4.8 seconds** per capture on a 4090, against **12 ms** for the
    /// replay it replaces.
    ///
    /// Sizing a slot to exactly what this fire asked for makes that bump
    /// happen on a schedule set by the context length. `kv_indices` is one u32
    /// per KV page, so it needs four more bytes every time the sequence
    /// crosses a page boundary; at an exact fit, those four bytes are a fresh
    /// allocation, a moved base and a stale cache. Measured on `pie run` with
    /// the naive-baseline inferlet: a recapture every three or four decode
    /// steps, GPU utilisation at 0 %, and **~1.25 s per token** on a model
    /// whose replay costs twelve milliseconds.
    ///
    /// So growth is geometric with a floor: at least double what the slot
    /// already held, rounded up to [`GROW_ALIGN`]. A slot that grows by one
    /// entry per page now bumps the epoch a logarithmic number of times over a
    /// whole sequence instead of once per page, and the steady state — the
    /// state the cache exists for — is reached and stays reached.
    ///
    /// Over-allocating is safe at every use: `copy_from_host` bounds-checks
    /// against the buffer and copies the SOURCE's length, `memset` zeroes the
    /// whole buffer (so the extra bytes are never stale), and no caller reads
    /// a pooled slot's `len()` as a count of anything.
    ///
    /// The doubled ask is an optimisation, not a requirement, so exhaustion on
    /// it falls back to the exact figure. The arena can be hundreds of
    /// megabytes and doubling it may genuinely not fit; refusing a fire that
    /// the driver could have run, because the headroom did not fit, would
    /// trade a latency defect for a correctness one.
    fn grow(
        slot: &mut Option<crate::device::DeviceBuffer>,
        epoch: &mut crate::fire::recordings::PlanEpoch,
        alloc: &crate::device::Allocator,
        what: &'static str,
        bytes: usize,
    ) -> crate::Result<()> {
        if slot.as_ref().is_none_or(|b| b.len() < bytes) {
            let want = headroom(slot.as_ref().map_or(0, crate::device::DeviceBuffer::len), bytes);
            // `Exhausted` carries the byte figure the engine needs to choose
            // between evicting, shrinking the batch and refusing — the exact
            // one, since that is what the fire could not have without.
            *slot = Some(
                alloc
                    .alloc(want)
                    .or_else(|_| alloc.alloc(bytes))
                    .map_err(|_| crate::Error::exhausted(what, bytes))?,
            );
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
        Self::grow(
            &mut self.score,
            &mut self.epoch,
            alloc,
            "attention score sink",
            plan.bytes,
        )?;
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
        Self::grow(
            &mut self.attn_out,
            &mut self.epoch,
            alloc,
            "attention landing",
            bytes,
        )?;
        Ok(self.attn_out.as_ref().expect("just grown").as_ptr())
    }

    /// The adapter correction's xAᵀ staging: rank-wide rows the low-rank GEMM
    /// writes and then reads back through B.
    ///
    /// # Why this is not [`Self::attn_out`]
    ///
    /// It was. The adapter phase asked `attn_out` for its gate, and on any
    /// family whose `attn_output` is `DriverPinned` -- qwen3 among them --
    /// that is the SAME buffer the attention dispatch lands its output in. The
    /// correction then wrote xAᵀ over the attention output, and whether the
    /// answer survived depended on which of the two touched the bytes last.
    ///
    /// # What that cost
    ///
    /// `lora-probe` at `adapter_scale: 0.0` -- a zero-B adapter, whose
    /// correction is EXACTLY zero and whose answer must therefore be the base
    /// model's, byte for byte -- answered the base model on some runs and
    /// something else on others, in the same process, with the same seed. The
    /// correction's OUTPUT was zero every time and correctly so; it was the
    /// intermediate, which is not zero because A is not zero, that landed on
    /// somebody else's rows.
    ///
    /// The fixture's whole reason to exist is the §5.1 claim that no adapter
    /// means no difference, and the aliasing falsified it nondeterministically,
    /// which is the one failure mode a parity gate cannot be written against.
    pub(crate) fn lora_gate(
        &mut self,
        alloc: &crate::device::Allocator,
        bytes: usize,
    ) -> crate::Result<*mut std::ffi::c_void> {
        let bytes = bytes.max(64);
        Self::grow(
            &mut self.lora_gate,
            &mut self.epoch,
            alloc,
            "lora gate",
            bytes,
        )?;
        Ok(self.lora_gate.as_ref().expect("just grown").as_ptr())
    }

    /// The log-sum-exp the attention dispatches write beside their output.
    pub(crate) fn lse(
        &mut self,
        alloc: &crate::device::Allocator,
        bytes: usize,
    ) -> crate::Result<*mut std::ffi::c_void> {
        Self::grow(
            &mut self.lse,
            &mut self.epoch,
            alloc,
            "attention lse",
            bytes.max(64),
        )?;
        Ok(self.lse.as_ref().expect("just grown").as_ptr())
    }

    /// Which rows of the fire are live, for the KV write descriptors.
    pub(crate) fn row_valid(
        &mut self,
        alloc: &crate::device::Allocator,
        rows: usize,
        stream: crate::device::StreamRef<'_>,
    ) -> crate::Result<*mut std::ffi::c_void> {
        Self::grow(
            &mut self.row_valid,
            &mut self.epoch,
            alloc,
            "row valid",
            rows.max(64),
        )?;
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
        Self::grow(
            &mut self.mask,
            &mut self.epoch,
            alloc,
            "attention mask",
            plan.bytes,
        )?;
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
            // without an epoch bump and does not go through `grow`. It gets
            // the same headroom anyway, for its own reason: `cudaHostAlloc`
            // synchronizes the device, so reallocating it once per page is a
            // stall on the fire's critical path even where nothing goes stale.
            let want = headroom(self.staging[slot].as_ref().map_or(0, |p| p.len()), need);
            self.staging[slot] = Some(
                crate::device::PinnedBuf::new(want)
                    .or_else(|_| crate::device::PinnedBuf::new(need))
                    .map_err(|_| crate::Error::exhausted("fire staging", need))?,
            );
        }
        let pin = self.staging[slot].as_mut().expect("just sized");
        for (dst, v) in pin.as_mut_slice()[..live].chunks_exact_mut(4).zip(vals) {
            dst.copy_from_slice(&v.to_le_bytes());
        }
        Self::grow(
            &mut self.slots[slot],
            &mut self.epoch,
            alloc,
            "fire descriptor array",
            need,
        )?;
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
        let grown = Self::grow(
            &mut held,
            &mut self.epoch,
            alloc,
            "named seam buffer",
            bytes,
        );
        if let Some(back) = held {
            self.named.insert(v, back);
        }
        grown?;
        let b = self.named.get_mut(&v).expect("just grown");
        b.memset(0, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::{GROW_ALIGN, headroom};

    /// Whatever the headroom does, it cannot hand back less than the fire
    /// asked for: the ask is the width a kernel is about to write.
    #[test]
    fn a_slot_is_never_given_less_than_it_asked_for() {
        for held in [0, 1, 4, 255, 256, 4096, usize::MAX / 2] {
            for bytes in [1, 4, 255, 256, 257, 1 << 20] {
                assert!(
                    headroom(held, bytes) >= bytes,
                    "held={held} bytes={bytes} came back short"
                );
            }
        }
        assert_eq!(headroom(usize::MAX, usize::MAX), usize::MAX, "no room to round");
    }

    /// The defect this exists for: `kv_indices` is four bytes per KV page, so
    /// an exactly-fitted slot reallocated — and bumped the epoch, and stranded
    /// every captured graph — once per page. Growing one entry at a time from
    /// cold must now cross the allocator a handful of times, not sixty-four.
    #[test]
    fn a_slot_that_grows_one_entry_per_page_stops_reallocating() {
        let mut have = 0usize;
        let mut moves = 0;
        for pages in 1..=64usize {
            let need = (pages * 4).max(4);
            if have < need {
                have = headroom(have, need);
                moves += 1;
            }
        }
        assert_eq!(
            moves, 1,
            "sixty-four pages should fit in one {GROW_ALIGN}-byte slot, took {moves} allocations"
        );
        assert_eq!(have, GROW_ALIGN);
    }

    /// Past the first block the growth has to stay geometric, or a long
    /// context walks back into a reallocation per page.
    #[test]
    fn growth_past_the_first_block_at_least_doubles() {
        assert_eq!(headroom(GROW_ALIGN, GROW_ALIGN + 4), GROW_ALIGN * 2);
        assert_eq!(headroom(GROW_ALIGN * 2, GROW_ALIGN * 2 + 4), GROW_ALIGN * 4);
        // A step bigger than the doubling is honoured as asked, rounded up.
        assert_eq!(headroom(GROW_ALIGN, 10 * GROW_ALIGN + 1), 11 * GROW_ALIGN);
    }

    /// One buffer holding twenty thousand pages must be reached in a
    /// logarithmic number of moves, since each one strands the graph cache.
    #[test]
    fn a_long_context_costs_a_logarithmic_number_of_epoch_bumps() {
        let mut have = 0usize;
        let mut moves = 0;
        for pages in 1..=20_000usize {
            let need = (pages * 4).max(4);
            if have < need {
                have = headroom(have, need);
                moves += 1;
            }
        }
        assert!(moves <= 10, "twenty thousand pages took {moves} allocations");
    }
}
