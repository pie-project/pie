//! The fire's ATTENTION MASK: one bit per `(query, kv)` pair, planned on the
//! host and staged into the mask view every fire carries.
//!
//! # What left
//!
//! `FirePageMask`, `AttentionMaskSink`, `MaskSlotLayout`,
//! `prepare_page_mask_capture` and `MaskError` STOOD HERE — 800 lines of the
//! PAGE-mask hook: a guest PTIR program declaring an `attn_page_mask` sink,
//! the driver carving its slot out of the sideband arena at `OnAttnProj`, and
//! the legacy capture arming it. Every one of those was reached through
//! `fire::stage_hooks` and `bind::dispatch`, and both are deleted.
//!
//! THE HOOK'S VERDICT: it dies with the machinery. A PTIR sink is written by
//! a guest program mid-forward, and the thing that gave a guest program a
//! place to stand inside the forward was the lowered launch list — a
//! `Program`'s statements are the lane's, and there is no per-layer callback
//! seam for a guest to be spliced into. Reviving it is a design (a point that
//! declares an observation output), not a re-wiring.
//!
//! # What stayed, and why it is not the same thing
//!
//! `element_mask` below. It is the driver's own mask — causal by default, the
//! caller's when a point that reads one exists — staged into
//! `kernels_cuda::views::
//! MaskView` and answered as the `"attention_mask"` runtime object. A claim
//! body reads it through the view; nothing writes into it mid-fire. That is
//! why it survives the hook that died beside it.
//!
//! BOTH PLANNERS FIRE NOW. `fire::launch::publish_seam_pins` runs
//! `from_words` for a frame that carries a mask and `plan_causal` for one
//! that does not, and `attention.masked` is a claim body that reads the
//! result through the raise door. What is refused is a text with no `masked`
//! fact — one attention arm and it is causal — and that refusal is
//! `baker::word_of`'s, where the lane is picked.

/// The element mask the custom-mask attention dispatch reads — element-, not
/// page-granularity like everything above.
///
/// # The layout is BIT-packed, and this file used to think otherwise
///
/// Both CUDA kernels that read a custom mask index it the same way, and both
/// index it by BIT:
///
/// ```text
///   // kernels/flashinfer/attention/variants.cuh
///   mask &= ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
///   // kernels/attn/attention_naive_paged.cuh
///   const long long bit  = qo_off * kv_total + kv_idx;
///   const long long byte = mask_indptr[request_idx] + (bit >> 3);
///   return ((mask[byte] >> (bit & 7)) & 1) != 0;
/// ```
///
/// So `mask_d` holds `(q, kv)` as one BIT, `mask_indptr_d` counts BYTES, and a
/// request's mask begins on a byte boundary. This module published one byte per
/// pair instead, with the CSR counting pairs. The kernel then read pair `8i` as
/// the whole byte for pairs `8i..8i+8` and took bit `k % 8` of a byte that is
/// only ever `0` or `1`, so seven of every eight positions were forced closed
/// and the eighth answered for its neighbours.
///
/// Nothing caught it, because nothing READS the mask on the arm that is
/// exercised. The causal plan below is published unconditionally so the
/// unmasked arm can still be RECORDED under `GuardMode::Union` — which captures
/// both arms and aborts if either's mask was never built — but that arm's
/// kernel is compiled without custom-mask support and never dereferences it.
/// Only a fire that actually supplies a mask reaches the reading form, and the
/// one curated fixture that does (`tart-masked`) was wedged on an unrelated
/// channel-cursor defect for as long as this was here. Its answer with a mask
/// whose numerics are exactly causal was `" wore of of of.. the."`; without it,
/// `"<think>\nOkay, the user is asking"`.
pub mod element_mask {
    /// A mask this large is refused rather than published — the extent (`sum_r
    /// qo_len[r] * kv_len[r]`) grows with the context.
    const MAX_MASK_BYTES: u64 = 1 << 30;

    /// Bytes a request of `cells` `(q, kv)` pairs occupies, one bit each.
    const fn packed_len(cells: u64) -> usize {
        cells.div_ceil(8) as usize
    }

    /// Set pair `index` of the request whose mask starts at byte `base`.
    fn set_bit(mask: &mut [u8], base: usize, index: usize) {
        mask[base + (index >> 3)] |= 1 << (index & 7);
    }

    /// One fire's element mask, planned but not allocated.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ElementMaskPlan {
        /// Bytes of `mask_d`.
        pub mask_bytes: usize,
        /// Byte offset of the `num_requests + 1` i32 CSR.
        pub indptr_offset: usize,
        /// Total bytes to allocate.
        pub bytes: usize,
        /// The CSR, in mask BYTES — what the kernels add `bit >> 3` to.
        pub indptr: Vec<i32>,
        /// The mask bits themselves, one `(q, kv)` pair each, request by
        /// request, every request starting on a byte boundary.
        pub mask: Vec<u8>,
    }

    /// Plan and fill a causal element mask for the fire's geometry. `None` when there
    /// is nothing to mask or it would exceed [`MAX_MASK_BYTES`]; the custom arm
    /// declines in that case.
    #[must_use]
    pub fn plan_causal(
        qo_indptr_h: &[u32],
        kv_page_indptr_h: &[u32],
        kv_last_page_lens_h: &[u32],
        page_size: i32,
    ) -> Option<ElementMaskPlan> {
        let requests = qo_indptr_h.len().checked_sub(1)?;
        if requests == 0 || kv_page_indptr_h.len() < requests + 1 {
            return None;
        }
        let page = u32::try_from(page_size.max(0)).unwrap_or(0);
        let mut indptr = vec![0i32; requests + 1];
        let mut extents = Vec::with_capacity(requests);
        let mut total: u64 = 0;
        let mut pairs: u64 = 0;
        for r in 0..requests {
            let qo = qo_indptr_h[r + 1].saturating_sub(qo_indptr_h[r]);
            let pages = kv_page_indptr_h[r + 1].saturating_sub(kv_page_indptr_h[r]);
            let kv = if pages == 0 {
                0
            } else {
                (pages - 1) * page + kv_last_page_lens_h.get(r).copied().unwrap_or(0)
            };
            indptr[r] = i32::try_from(total).ok()?;
            extents.push((qo, kv));
            let cells = u64::from(qo) * u64::from(kv);
            pairs += cells;
            // Byte-aligned per request: the kernel adds `bit >> 3` to
            // `mask_indptr[r]`, so a request that began mid-byte would read its
            // neighbour's tail.
            total += packed_len(cells) as u64;
        }
        indptr[requests] = i32::try_from(total).ok()?;
        if pairs == 0 || total > MAX_MASK_BYTES {
            return None;
        }
        let mask_bytes = usize::try_from(total).ok()?;
        let mut mask = vec![0u8; mask_bytes];
        let mut at = 0usize;
        for &(qo, kv) in &extents {
            // Local row `qi` sits at absolute position `kv - qo + qi`, so it attends
            // every key at or before that.
            for qi in 0..qo {
                let last = kv.saturating_sub(qo) + qi;
                for ki in 0..kv {
                    if ki <= last {
                        set_bit(&mut mask, at, (qi * kv + ki) as usize);
                    }
                }
            }
            at += packed_len(u64::from(qo) * u64::from(kv));
        }
        let indptr_offset = mask_bytes.next_multiple_of(4);
        Some(ElementMaskPlan {
            mask_bytes,
            indptr_offset,
            bytes: indptr_offset + (requests + 1) * 4,
            indptr,
            mask,
        })
    }

    /// The engine's mask, repacked into the bits the launcher reads: the engine
    /// gives one bitset per query row, bit `i` whether that row attends KV
    /// position `i`, each row starting at bit 0 of its own words; the kernel
    /// wants one contiguous `qo * kv` bitset per REQUEST, byte-aligned, with
    /// the same CSR as [`plan_causal`].
    /// `None` when the fire's and table's shapes disagree — a REFUSAL, not a fallback,
    /// since serving causally would look exactly right.
    #[must_use]
    pub fn from_words(
        qo_indptr_h: &[u32],
        kv_page_indptr_h: &[u32],
        kv_last_page_lens_h: &[u32],
        page_size: i32,
        request_indptr: &[u32],
        word_indptr: &[u32],
        words: &[u32],
    ) -> Option<ElementMaskPlan> {
        let requests = qo_indptr_h.len().checked_sub(1)?;
        if requests == 0 || request_indptr.len() < requests + 1 {
            return None;
        }
        let page = u32::try_from(page_size.max(0)).unwrap_or(0);
        let mut indptr = vec![0i32; requests + 1];
        let mut mask: Vec<u8> = Vec::new();
        let mut total: u64 = 0;
        let mut pairs: u64 = 0;
        for r in 0..requests {
            let qo = qo_indptr_h[r + 1].saturating_sub(qo_indptr_h[r]) as usize;
            let pages = kv_page_indptr_h[r + 1].saturating_sub(kv_page_indptr_h[r]);
            let kv = if pages == 0 {
                0
            } else {
                (pages - 1) * page + kv_last_page_lens_h.get(r).copied().unwrap_or(0)
            } as usize;
            indptr[r] = i32::try_from(total).ok()?;
            // One mask per query row; the count must match or the table describes a different fire.
            let (lo, hi) = (request_indptr[r] as usize, request_indptr[r + 1] as usize);
            if hi.saturating_sub(lo) != qo || hi > word_indptr.len().saturating_sub(1) {
                return None;
            }
            let base = mask.len();
            mask.resize(base + packed_len((qo * kv) as u64), 0);
            for (qi, m) in (lo..hi).enumerate() {
                let (wlo, whi) = (word_indptr[m] as usize, word_indptr[m + 1] as usize);
                let row = words.get(wlo..whi)?;
                // A mask shorter than the row's KV extent can't say what the tail
                // attends, and guessing is what this refuses.
                if row.len() * 32 < kv {
                    return None;
                }
                for k in 0..kv {
                    if row[k / 32] >> (k % 32) & 1 == 1 {
                        set_bit(&mut mask, base, qi * kv + k);
                    }
                }
            }
            pairs += (qo * kv) as u64;
            total += packed_len((qo * kv) as u64) as u64;
        }
        indptr[requests] = i32::try_from(total).ok()?;
        if pairs == 0 || total > MAX_MASK_BYTES {
            return None;
        }
        let mask_bytes = usize::try_from(total).ok()?;
        debug_assert_eq!(mask.len(), mask_bytes);
        let indptr_offset = mask_bytes.next_multiple_of(4);
        Some(ElementMaskPlan {
            mask_bytes,
            indptr_offset,
            bytes: indptr_offset + (requests + 1) * 4,
            indptr,
            mask,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The kernels' own read, spelled out once so every expectation below is
        /// checked against the thing that consumes it rather than against a
        /// restatement of the packer.
        ///
        /// `attention_naive_paged.cuh`:
        /// ```text
        ///   bit  = qo_off * kv_total + kv_idx;
        ///   byte = mask_indptr[request_idx] + (bit >> 3);
        ///   ((mask[byte] >> (bit & 7)) & 1) != 0
        /// ```
        fn kernel_reads(
            p: &ElementMaskPlan,
            request: usize,
            qi: usize,
            ki: usize,
            kv: usize,
        ) -> bool {
            let bit = qi * kv + ki;
            let byte = p.indptr[request] as usize + (bit >> 3);
            (p.mask[byte] >> (bit & 7)) & 1 != 0
        }

        /// Every pair of a request, as the kernel would read them.
        fn read_all(p: &ElementMaskPlan, request: usize, qo: usize, kv: usize) -> Vec<bool> {
            (0..qo)
                .flat_map(|qi| (0..kv).map(move |ki| (qi, ki)))
                .map(|(qi, ki)| kernel_reads(p, request, qi, ki, kv))
                .collect()
        }

        #[test]
        fn a_decode_row_attends_its_whole_context() {
            let p = plan_causal(&[0, 1], &[0, 1], &[3], 16).expect("planned");
            // Three pairs is three BITS, so one byte, and the CSR counts bytes.
            assert_eq!(p.indptr, vec![0, 1]);
            assert_eq!(p.mask, vec![0b111]);
            assert_eq!(read_all(&p, 0, 1, 3), vec![true; 3]);
        }

        #[test]
        fn a_prefill_row_attends_no_further_than_itself() {
            // 3 query rows against a 3-long context: the plain lower triangle.
            let p = plan_causal(&[0, 3], &[0, 1], &[3], 16).expect("planned");
            assert_eq!(
                read_all(&p, 0, 3, 3),
                vec![true, false, false, true, true, false, true, true, true]
            );
            // Nine pairs is two bytes, and the ninth is the low bit of the second.
            assert_eq!(p.mask, vec![0b1101_1001, 0b1]);
        }

        #[test]
        fn a_continuation_attends_the_prefix_it_did_not_write() {
            // 2 new rows onto a 5-long context: rows 3 and 4 are the new ones.
            let p = plan_causal(&[0, 2], &[0, 1], &[5], 16).expect("planned");
            assert_eq!(
                read_all(&p, 0, 2, 5),
                vec![true, true, true, true, false, true, true, true, true, true]
            );
        }

        #[test]
        fn two_requests_get_their_own_bases() {
            // Two pairs and four pairs: one byte each, because a request has to
            // START on a byte boundary or the kernel's `bit >> 3` walks into its
            // neighbour's tail.
            let p = plan_causal(&[0, 1, 3], &[0, 1, 2], &[2, 2], 16).expect("planned");
            assert_eq!(p.indptr, vec![0, 1, 2]);
            assert_eq!(read_all(&p, 0, 1, 2), vec![true, true]);
            assert_eq!(read_all(&p, 1, 2, 2), vec![true, false, true, true]);
        }

        /// The engine's bitset, repacked: one decode row attending 3 KV positions
        /// is three set bits in one byte.
        #[test]
        fn a_set_bit_survives_the_repack() {
            let p = from_words(&[0, 1], &[0, 1], &[3], 16, &[0, 1], &[0, 1], &[0b111])
                .expect("decoded");
            assert_eq!(p.mask, vec![0b111]);
            assert_eq!(p.indptr, vec![0, 1]);
            assert_eq!(read_all(&p, 0, 1, 3), vec![true; 3]);
        }

        /// A CLEARED bit is a position the kernel skips — the whole point of a
        /// caller's mask, and what a causal fallback would silently undo.
        #[test]
        fn a_cleared_bit_survives_the_repack() {
            let p = from_words(&[0, 1], &[0, 1], &[4], 16, &[0, 1], &[0, 1], &[0b1011])
                .expect("decoded");
            assert_eq!(read_all(&p, 0, 1, 4), vec![true, true, false, true]);
        }

        /// A prefill's rows are its own masks; the repack CONCATENATES them into
        /// one `qo * kv` bitset, because the engine's rows each start at bit 0 of
        /// their own words and the kernel's do not.
        #[test]
        fn each_query_row_brings_its_own_mask() {
            let p = from_words(
                &[0, 2],
                &[0, 1],
                &[3],
                16,
                &[0, 2],
                &[0, 1, 2],
                &[0b001, 0b011],
            )
            .expect("decoded");
            assert_eq!(
                read_all(&p, 0, 2, 3),
                vec![true, false, false, true, true, false]
            );
            let causal = plan_causal(&[0, 2], &[0, 1], &[3], 16).expect("causal");
            assert_eq!(p.indptr, causal.indptr, "same geometry, same CSR");
        }

        /// The defect this layout was changed for: a mask whose numerics ARE
        /// causal has to read back identical to the causal plan, pair for pair.
        /// It did not — the packer wrote a byte per pair while both kernels read
        /// a bit per pair, so seven of every eight positions were forced closed.
        #[test]
        fn a_causal_custom_mask_reads_back_as_the_causal_plan() {
            // 24 query rows over 24 keys, which is `tart-masked`'s prefill: two
            // bytes' worth of row and a row length that is not a multiple of 8,
            // so every misalignment this could have shows up.
            let (qo, kv) = (24usize, 24usize);
            let words: Vec<u32> = (0..qo)
                .flat_map(|qi| {
                    let row: u32 = (0..kv).filter(|&ki| ki <= qi).map(|ki| 1u32 << ki).sum();
                    [row]
                })
                .collect();
            let word_indptr: Vec<u32> = (0..=qo as u32).collect();
            let user = from_words(
                &[0, qo as u32],
                &[0, 2],
                &[8],
                16,
                &[0, qo as u32],
                &word_indptr,
                &words,
            )
            .expect("decoded");
            let causal = plan_causal(&[0, qo as u32], &[0, 2], &[8], 16).expect("causal");
            assert_eq!(user.mask, causal.mask);
            assert_eq!(user.indptr, causal.indptr);
            assert_eq!(user.mask.len(), (qo * kv).div_ceil(8));
        }

        /// A `set` never spills into the neighbouring pair, which one byte per
        /// pair could not get wrong and one bit per pair can.
        #[test]
        fn a_single_open_position_opens_exactly_one() {
            let p = from_words(&[0, 1], &[0, 1], &[9], 16, &[0, 1], &[0, 1], &[1 << 8])
                .expect("decoded");
            assert_eq!(
                read_all(&p, 0, 1, 9),
                vec![false, false, false, false, false, false, false, false, true]
            );
        }

        #[test]
        fn an_empty_fire_publishes_nothing() {
            assert!(plan_causal(&[0], &[0], &[], 16).is_none());
            assert!(plan_causal(&[0, 1], &[0, 0], &[0], 16).is_none());
        }
    }
}
