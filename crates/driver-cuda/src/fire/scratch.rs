//! The fire's pooled buffers.
//!
//! Pooled, not per-fire: growth is what a pool avoids, and `cudaFree`
//! synchronizes the device.

/// Monotonic count of pooled-buffer rewrites.
///
/// It lived in `fire::recordings` and outlived it. Its old job was the graph
/// cache's staleness key — a captured exec bakes the addresses it saw, so a
/// pool that grew and moved a base had to make every recorded exec a miss.
/// There is no capture to invalidate, and the counter STAYS because it is the
/// one honest record of "a pooled base moved", which is what a future capture
/// of the eager walk will key on and what a leak hunt reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct PlanEpoch(u64);

impl PlanEpoch {
    /// The only way the epoch changes, so nothing else can manufacture one.
    pub(crate) fn bump(&mut self) {
        self.0 += 1;
    }
}

/// The per-fire device arrays, pooled across fires.
///
/// Buffers are grown, not freed. Growth moves a base address and bumps
/// [`Self::epoch`].
#[derive(Default)]
pub(crate) struct Scratch {
    /// The walk's activation arena, `rows * program.row_pitch` — see
    /// [`Self::baker_arena`].
    pub(crate) baker: Option<crate::device::DeviceBuffer>,
    /// Where the walk repitches its `out` seam, and where the delivery reads
    /// it — see [`Self::logits`].
    pub(crate) logits: Option<crate::device::DeviceBuffer>,
    /// The unconditional attention-score sink — see [`Self::score`].
    pub(crate) score: Option<crate::device::DeviceBuffer>,
    /// The unconditional custom attention mask — see [`Self::mask`].
    pub(crate) mask: Option<crate::device::DeviceBuffer>,
    /// The per-row live flags — see [`Self::row_valid`].
    pub(crate) row_valid: Option<crate::device::DeviceBuffer>,
    /// The small per-fire u32 descriptor arrays, by slot — see [`slot`].
    pub(crate) slots: Vec<Option<crate::device::DeviceBuffer>>,
    /// Pinned host staging for those uploads, one per slot.
    ///
    /// Pageable-source `cudaMemcpyAsync` is synchronous; pinning makes it
    /// async. One per slot, since a shared buffer would be overwritten while
    /// the previous slot's copy was still queued.
    pub(crate) staging: Vec<Option<crate::device::PinnedBuf>>,
    pub(crate) epoch: PlanEpoch,
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
    // 9..=12 STOOD FOR a peel tail's own four CSR slots, so the prefix could
    // read the fire's after the tail had uploaded. There is no peel. The
    // numbers are NOT renumbered: they are an interface, and a gap costs
    // nothing while a shift would silently repoint every slot above it.
    /// `"request_of_token"` — the runtime stream a lane may name; staged only
    /// when the lane's slots name it.
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
        epoch: &mut PlanEpoch,
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
    /// The walk's activation arena, at least `bytes` wide.
    ///
    /// Grow-only through the same [`Self::grow`] every pooled buffer uses:
    /// never freed, and growth bumps the epoch. The walk is eager and
    /// captures nothing today, so the epoch bump is precaution rather than
    /// necessity — and it is the cheap kind, since an arena that has stopped
    /// growing bumps nothing.
    pub(crate) fn baker_arena(
        &mut self,
        alloc: &crate::device::Allocator,
        bytes: usize,
    ) -> crate::Result<*mut std::ffi::c_void> {
        Self::grow(
            &mut self.baker,
            &mut self.epoch,
            alloc,
            "baker arena",
            bytes,
        )?;
        Ok(self.baker.as_ref().expect("just grown").as_ptr())
    }

    /// The logits landing, at least `bytes` wide.
    ///
    /// DRIVER-OWNED AND NOT A VALUE ANY TEXT STATES, which is the change the
    /// legacy purge made here. The delivery used to read whichever
    /// `Arg::Named` buffer the last lowered launch wrote, discovered by
    /// walking the launch list backwards; the walk's `out` seam is repitched
    /// into this instead, so one buffer answers `deliver_logits` and
    /// `run_sampling_programs` and nothing has to agree about which.
    pub(crate) fn logits(
        &mut self,
        alloc: &crate::device::Allocator,
        bytes: usize,
    ) -> crate::Result<()> {
        Self::grow(
            &mut self.logits,
            &mut self.epoch,
            alloc,
            "logits landing",
            bytes.max(64),
        )
    }

    /// The logits landing, if one has been grown.
    pub(crate) fn logits_buf(&self) -> Option<&crate::device::DeviceBuffer> {
        self.logits.as_ref()
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

    // `attn_out`, `lora_gate` and `lse` STOOD HERE. All three were operands
    // the LEGACY dispatch took loose, off `AttnCtx`/`DispatchCtx`: the
    // driver-owned attention landing for a fire whose op join named no
    // output slot, the adapter correction's xAᵀ staging, and the
    // log-sum-exp the decode dispatch wrote beside its output. A `Program`'s
    // statements state their own results, so a landing is an arena slot; the
    // adapter arm is deleted; and `baker::staging` passes `None` for the LSE
    // because the lane it fires states one attention leg and nothing merges
    // partials across it.

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

    // `named` STOOD HERE — one pooled buffer per `Arg::Named` the legacy
    // lowering placed, zeroed every fire. See `fire::launch`'s
    // `publish_seam_pins` for what replaced the walk that filled it.
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
