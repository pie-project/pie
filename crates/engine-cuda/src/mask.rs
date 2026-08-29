//! The `masked` axis's one piece of arithmetic: a lane's run-length mask,
//! expanded into the bits `attention.masked` actually reads.
//!
//! **THE TWO FORMS ARE NOT THE SAME SHAPE, AND THAT IS THE WHOLE FILE.** A
//! submission states a lane's masking as run-length encodings over that
//! lane's readable extent —
//! [`Mask`] is a run-length encoding over `total` KV positions, because a
//! prefix drop, a retained-block set and a sliding window are each a handful
//! of runs over thousands of positions. The kernel reads something else: the
//! fa2 custom-mask arm indexes `qo_idx * kv_len + kv_idx`, one bit per
//! (query row, key position) PAIR, LSB-first inside each byte, with a
//! per-request byte offset in a span table beside it. So the shell owns one
//! expansion, and it is here rather than inside the fire path because it is
//! the only place in this crate where a wrong bit is silently a different
//! sentence rather than a fault.
//!
//! # The submission has TWO mask shapes and the expansion has one
//!
//! [`Masking::Extent`] is the mask above: one restriction of the lane's
//! readable extent, re-applied to every query row. [`Masking::Rows`] is one
//! restriction PER query row — the windowed prefill, where row `i` keeps
//! `[i - w, i]` and row `i + 1` keeps `[i + 1 - w, i + 1]`, two sets that are
//! not nested and that therefore no single extent mask is either of. They are
//! two shapes of SUBMISSION and one shape of output: the slab this file
//! builds is a `rows x kv` rectangle of bits either way, and the attention
//! dispatch below it cannot tell which shape produced it. That is the whole
//! reason the per-row form needs no kernel: `fa2` already reads a bit per
//! (query row, key position) pair.
//!
//! What differs is only the WALK. An extent mask is walked ONCE and applied
//! to every row, because a prefill lane's mask is the same runs at every row
//! and only the causal bound moves. A per-row mask is walked once per row,
//! because that is what "per row" means. Neither is a special case of the
//! other in code, and collapsing them would cost the common form its single
//! pass over a handful of runs.
//!
//! # Causality is folded IN, and the sliding window is not
//!
//! The custom arm applies no causal bound of its own —
//! `flashinfer::attention::variants`' `LogitsMask` under `use_custom_mask`
//! reads the bit and nothing else, where the causal arm computes
//! `kv_idx + qo_len > kv_len + q_idx`. A lane that carried more than one
//! query row and got its mask verbatim on every row would let row 0 attend
//! row 4's key. So the expansion is
//!
//! ```text
//! keep(q, k) = row q's runs keep k   ∧   k <= have + q
//! ```
//!
//! — where "row q's runs" are the lane's single set under
//! [`Masking::Extent`] and row `q`'s own under [`Masking::Rows`] — which is
//! exactly `Lane::mask`'s own words — "which readable extent of a
//! slot a lane's attention may reach" — intersected with the order the cache
//! is written in. On a one-row lane the second term is vacuous and the two
//! readings coincide, which is why a decode-shaped custom mask needs no
//! special case.
//!
//! The SLIDING window is deliberately not folded in, and the reason is that
//! it is not a property of the lane. `Attention::Masked` states its window
//! per NODE — gemma's text alternates `Some(512)` and `None` down its layer
//! stack — so one slab of bits cannot serve both, and a fold would have to
//! be per layer. `fa2::prefill_custom_arm` instantiates `Custom` and
//! `CustomSoftcap` and no windowed variant, so a masked op that states a
//! window has no kernel to run on and is refused by name at dispatch rather
//! than silently attending the whole prefix.
//!
//! # A mask may be LONGER than the extent, and the surplus is clipped
//!
//! **A GUEST STATES ITS MASK OVER THE PAGES IT RESERVED, NOT OVER THE TOKENS
//! IT HAS.** A sliding-window inferlet reserves 3 pages of 16 for a 23-token
//! lane and builds its `[rows, keys]` bool rectangle at the POOL's width, 48
//! — the number it knows at reserve time, and the number that does not move
//! as the sequence grows a token per fire. The extent this fire reads is 23.
//! Every guest in `tests/inferlets` that carries a mask does this, and the
//! old C++ engine took them exactly so: `brle::decode` walked `for key in
//! 0..kv_len` and read `packed_bit` past the row's stored words as FALSE, so
//! a longer mask was clipped and a shorter one padded with masked-out.
//!
//! So the extent rule here is an INEQUALITY, and only one direction is a
//! fault:
//!
//! ```text
//! total >= extent    accepted; positions past the extent are dropped
//! total <  extent    Fault::Mask
//! ```
//!
//! **THE SURPLUS IS NOT AMBIGUOUS AND THAT IS WHY IT MAY BE DROPPED.** A
//! position past `have + rows` is one this fire has not written, so the
//! causal bound `k <= have + q` masks it out for every query row whatever its
//! bit says — the clip removes nothing the expansion would have kept. A SHORT
//! mask is the opposite: its missing tail expands to zero, zero is
//! MASKED-OUT, and the lane's attention is silently truncated to the stated
//! prefix. That one stays refused, in both forms, and neither direction is
//! repaired into the other.
//!
//! The clip costs no code: both walks already stop the runs at `kv`
//! (`.min(kv)` and the `end == kv` break), because that bound was always the
//! rectangle's width. Only the CHECK moved.
//!
//! # The span table holds BYTES
//!
//! `variants.cuh` reads `maybe_custom_mask + maybe_mask_indptr[batch_idx]`
//! as pointer arithmetic on a `uint8_t*` — a BYTE offset, not the bit offset
//! upstream flashinfer carries — so every lane's region starts on a byte
//! boundary and the table's entries are byte counts. The bits slab is
//! fire-wide and its offsets are ABSOLUTE, which is what lets a windowed
//! consumer take a sliced span table and the whole slab (the same shape
//! `GeomKind::Indices` and its bounds vector already have).

use engine::engine_api::fire::{Mask, Masking};

use crate::error::{Fault, Result};

/// **A DEVICE-RESOLVED DENSE RECTANGLE, AS THE RESTRICTIONS THIS FILE ALREADY
/// EXPANDS.**
///
/// A guest whose ancestry mask is device data states it as a `[rows, keys]`
/// rectangle of bools on a channel, and the descriptor-port plane reads that
/// cell (`program::ports`). What the attention arm reads is the packed slab
/// [`stage`] builds, and the ONE thing that must be true of a device mask is
/// that it packs into the very same slab a host-stated mask of the same bools
/// packs into — otherwise a guest that moved its mask onto the device would
/// silently change which keys its beams reach.
///
/// So the dense form is not given a second walk. It is run-length encoded
/// here, one [`Mask`] per row over the rectangle's own key width, and handed
/// to [`stage`] — which is the host path, byte for byte, and is why the gate
/// for this is an EQUALITY against a host-stated control rather than a smoke
/// test.
///
/// `total` is the rectangle's own stride and not the lane's extent, which is
/// deliberate: a guest builds its rectangle at the width of the POOL it
/// reserved and the pool does not shrink as the extent grows, so the surplus
/// is exactly the "a mask may be LONGER" case the module doc argues and it
/// takes the same clip.
#[must_use]
pub fn from_dense(cells: &[bool], stride: usize) -> Masking {
    let rows = if stride == 0 { 0 } else { cells.len() / stride };
    Masking::Rows(
        (0..rows)
            .map(|row| {
                let mut runs: Vec<u32> = Vec::new();
                // Alternating lengths, MASKED-OUT FIRST — `Mask`'s own
                // encoding, so a row that opens with a kept key opens with a
                // zero-length dropped run.
                let mut keeping = false;
                let mut run = 0u32;
                for &kept in &cells[row * stride..(row + 1) * stride] {
                    if kept == keeping {
                        run += 1;
                        continue;
                    }
                    runs.push(run);
                    keeping = kept;
                    run = 1;
                }
                if run != 0 {
                    runs.push(run);
                }
                Mask::new(runs, stride as u64)
            })
            .collect(),
    )
}

/// One lane's mask, with the geometry that says what shape it expands to.
#[derive(Debug, Clone, Copy)]
pub struct LaneMask<'a> {
    /// The lane's masking — one restriction over its extent or one per query
    /// row — or `None` for a lane that carries none.
    pub mask: Option<&'a Masking>,
    /// How many KV tokens the slot held BEFORE this fire.
    pub have: u32,
    /// How many token rows this fire feeds it.
    pub rows: u32,
}

/// A fire's mask bits and their per-lane span table, ready to stage.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Staged {
    /// The packed bits, every lane's `rows x kv` rectangle end to end, each
    /// starting on a byte boundary.
    pub bits: Vec<u8>,
    /// `[lanes + 1]`: each lane's byte offset into [`bits`](Staged::bits).
    /// The last entry is the total, so a reader can bound the final lane.
    pub indptr: Vec<i32>,
}

/// Expand a fire's lane masks, in fire (seriated) row order.
///
/// `Ok(None)` is a fire no lane put a mask on — the shell then binds neither
/// seat, and a masked consumer gets `attn::masked`'s own typed refusal
/// instead of a rectangle of zeros that would blank every logit.
///
/// # Errors
///
/// [`Fault::Mask`] for a mask SHORTER than the lane's post-append KV length —
/// checked for EVERY mask a lane states, because a [`Masking::Rows`] row
/// covers the same extent its neighbours do (the causal bound, not the mask,
/// is what keeps row `i` off row `i + 1`'s key). Short is the dangerous
/// direction: the missing positions read as MASKED-OUT and quietly truncate
/// the attention. LONGER is accepted and clipped — a guest states its mask
/// over the pages it reserved, and a position past the extent is one no query
/// row could attend anyway (see the module doc).
///
/// [`Fault::MaskRows`] for a [`Masking::Rows`] whose row count is not the
/// lane's row count: some query row would have no mask of its own, and the
/// only thing to do with such a row is invent one, which is the silent
/// row-zero substitution the per-row form exists to end.
pub fn stage(lanes: &[LaneMask<'_>]) -> Result<Option<Staged>> {
    if lanes.iter().all(|lane| lane.mask.is_none()) {
        return Ok(None);
    }
    let mut bits: Vec<u8> = Vec::new();
    let mut indptr: Vec<i32> = Vec::with_capacity(lanes.len() + 1);
    indptr.push(0);
    for (at, lane) in lanes.iter().enumerate() {
        let kv = u64::from(lane.have) + u64::from(lane.rows);
        if let Some(masking) = lane.mask {
            // ONE RULE FOR BOTH SHAPES, CHECKED FIRST. `Masking::masks` is
            // the lane's restrictions in row order — one for an extent mask,
            // one per row otherwise — and every one of them covers the lane's
            // post-append extent.
            // ONE DIRECTION IS A FAULT AND THE OTHER IS THE ECOSYSTEM'S
            // ORDINARY SHAPE. See the module doc's "A mask may be LONGER".
            let stated: &[Mask] = masking.masks();
            if let Some(bad) = stated.iter().find(|mask| mask.total < kv) {
                return Err(Fault::Mask {
                    lane: at as u32,
                    stated: bad.total,
                    extent: kv,
                });
            }
            if let Some(stated) = masking.stated_rows()
                && stated as u64 != u64::from(lane.rows)
            {
                return Err(Fault::MaskRows {
                    lane: at as u32,
                    stated: stated as u64,
                    rows: lane.rows,
                });
            }
            let cells = u64::from(lane.rows) * kv;
            let bytes = usize::try_from(cells.div_ceil(8)).unwrap_or(usize::MAX);
            let base = bits.len();
            bits.resize(base + bytes, 0);
            let mut set = |cell: u64| {
                let byte = base + usize::try_from(cell / 8).unwrap_or(usize::MAX);
                if let Some(word) = bits.get_mut(byte) {
                    *word |= 1 << (cell % 8);
                }
            };
            match masking {
                // The runs, walked once per lane and applied to every query
                // row — one pass over the encoding rather than `rows` passes,
                // because an extent mask IS the same runs at every row and
                // only the causal bound moves.
                Masking::Extent(mask) => {
                    let mut at_position = 0u64;
                    for (index, &run) in mask.runs.iter().enumerate() {
                        let end = at_position.saturating_add(u64::from(run)).min(kv);
                        if index % 2 == 1 {
                            for key in at_position..end {
                                // Causal: query row `q` stands at absolute
                                // position `have + q`, and a key past it has
                                // not been written.
                                let first = key.saturating_sub(u64::from(lane.have));
                                for q in first..u64::from(lane.rows) {
                                    set(q * kv + key);
                                }
                            }
                        }
                        if end == kv {
                            break;
                        }
                        at_position = end;
                    }
                }
                // One walk per row, over that row's own runs — and under that
                // row's own causal bound, which is the half a mask may not
                // widen: row `q` stands at `have + q` and the keys past it
                // are positions this fire has not written. A row's runs
                // choose among the keys causality already allows it; they
                // never add one.
                Masking::Rows(rows) => {
                    for (q, mask) in rows.iter().enumerate() {
                        let q = q as u64;
                        let bound = u64::from(lane.have) + q;
                        let mut at_position = 0u64;
                        for (index, &run) in mask.runs.iter().enumerate() {
                            let end = at_position.saturating_add(u64::from(run)).min(kv);
                            if index % 2 == 1 {
                                for key in at_position..end.min(bound + 1) {
                                    set(q * kv + key);
                                }
                            }
                            if end == kv {
                                break;
                            }
                            at_position = end;
                        }
                    }
                }
            }
        }
        indptr.push(i32::try_from(bits.len()).unwrap_or(i32::MAX));
    }
    Ok(Some(Staged { bits, indptr }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Is `(q, k)` kept, read the way the device text reads it?
    fn keeps(staged: &Staged, lane: usize, kv: u64, q: u64, k: u64) -> bool {
        let base = staged.indptr[lane] as usize;
        let cell = q * kv + k;
        (staged.bits[base + (cell / 8) as usize] >> (cell % 8)) & 1 == 1
    }

    /// A one-row lane is the decode-shaped custom mask, and its bits are the
    /// runs verbatim: the causal bound at `have + 0` keeps every key the
    /// cache holds.
    #[test]
    fn a_decode_lane_expands_to_its_runs() {
        // 7 held + 1 new = 8 positions; drop the first three, keep the rest.
        let mask = Masking::Extent(Mask::new(vec![3, 5], 8));
        let staged = stage(&[LaneMask {
            mask: Some(&mask),
            have: 7,
            rows: 1,
        }])
        .expect("the mask covers the lane's extent")
        .expect("a masked fire stages bits");
        assert_eq!(staged.indptr, vec![0, 1], "8 cells is one byte");
        for k in 0..8 {
            assert_eq!(
                keeps(&staged, 0, 8, 0, k),
                k >= 3,
                "key {k} of a three-position prefix drop"
            );
        }
    }

    /// A multi-row lane gets the causal bound the custom arm does not apply.
    #[test]
    fn a_prefill_lane_gets_its_causal_bound() {
        // A fresh 4-token prompt, nothing masked out by the runs.
        let mask = Masking::Extent(Mask::new(vec![0, 4], 4));
        let staged = stage(&[LaneMask {
            mask: Some(&mask),
            have: 0,
            rows: 4,
        }])
        .expect("the mask covers the lane's extent")
        .expect("a masked fire stages bits");
        assert_eq!(staged.indptr, vec![0, 2], "16 cells is two bytes");
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(
                    keeps(&staged, 0, 4, q, k),
                    k <= q,
                    "row {q} must not reach key {k}"
                );
            }
        }
    }

    /// The runs and the causal bound are ANDed, not chosen between.
    #[test]
    fn the_runs_and_the_causal_bound_intersect() {
        // 2 held + 3 new = 5; keep positions 1..4 (drop 0 and 4).
        let mask = Masking::Extent(Mask::new(vec![1, 3, 1], 5));
        let staged = stage(&[LaneMask {
            mask: Some(&mask),
            have: 2,
            rows: 3,
        }])
        .expect("the mask covers the lane's extent")
        .expect("a masked fire stages bits");
        for q in 0..3u64 {
            for k in 0..5u64 {
                let want = (1..4).contains(&k) && k <= 2 + q;
                assert_eq!(keeps(&staged, 0, 5, q, k), want, "cell ({q}, {k})");
            }
        }
    }

    /// **THE SHAPE THE EXTENT FORM CANNOT SAY: A SLIDING WINDOW.**
    ///
    /// Row `q` keeps `[q - 1, q]` over a fresh six-token prompt — six
    /// restrictions, no two of them nested, so no `Masking::Extent` is any of
    /// them and the one this used to be lowered as was row ZERO's (keep key
    /// 0, and nothing else, forever). Each row's own runs, under each row's
    /// own causal bound.
    #[test]
    fn a_windowed_prefill_expands_row_by_row() {
        const N: u64 = 6;
        let rows: Vec<Mask> = (0..N)
            .map(|q| {
                let front = q.saturating_sub(1);
                Mask::new(vec![front as u32, (q + 1 - front) as u32], N)
            })
            .collect();
        let mask = Masking::Rows(rows);
        let staged = stage(&[LaneMask {
            mask: Some(&mask),
            have: 0,
            rows: N as u32,
        }])
        .expect("every row covers the lane's extent")
        .expect("a masked fire stages bits");
        assert_eq!(staged.indptr, vec![0, 5], "36 cells is five bytes");
        for q in 0..N {
            for k in 0..N {
                assert_eq!(
                    keeps(&staged, 0, N, q, k),
                    k + 1 >= q && k <= q,
                    "cell ({q}, {k}) of a two-wide sliding window"
                );
            }
        }
    }

    /// **A ROW'S RUNS CHOOSE AMONG THE KEYS CAUSALITY ALLOWS; THEY NEVER ADD
    /// ONE.** Every row here states "keep everything", including the
    /// positions later rows write, and every row still stops at its own
    /// bound. This is the property that makes the per-row form safe to hand
    /// a kernel that applies no causal bound of its own.
    #[test]
    fn a_per_row_mask_may_not_widen_causality() {
        // 2 held + 3 new = 5, and every row asks for all five.
        let mask = Masking::Rows(vec![
            Mask::new(vec![0, 5], 5),
            Mask::new(vec![0, 5], 5),
            Mask::new(vec![0, 5], 5),
        ]);
        let staged = stage(&[LaneMask {
            mask: Some(&mask),
            have: 2,
            rows: 3,
        }])
        .expect("every row covers the lane's extent")
        .expect("a masked fire stages bits");
        for q in 0..3u64 {
            for k in 0..5u64 {
                assert_eq!(
                    keeps(&staged, 0, 5, q, k),
                    k <= 2 + q,
                    "cell ({q}, {k}): an all-keeping row is still causal"
                );
            }
        }
    }

    /// The two forms AGREE when they say the same thing: a `Rows` whose every
    /// row is one restriction stages the identical slab an `Extent` of it
    /// does. One walk or `rows` walks is a cost, not a meaning.
    #[test]
    fn a_rows_form_that_repeats_one_restriction_is_the_extent_form() {
        let one = Mask::new(vec![1, 3, 1], 5);
        let extent = Masking::Extent(one.clone());
        let repeated = Masking::Rows(vec![one.clone(), one.clone(), one]);
        let lane = |mask| LaneMask {
            mask: Some(mask),
            have: 2,
            rows: 3,
        };
        assert_eq!(
            stage(&[lane(&extent)]).expect("the extent form stages"),
            stage(&[lane(&repeated)]).expect("the row form stages"),
        );
    }

    /// A per-row mask with the wrong number of rows is refused by name: the
    /// alternatives are inventing a row's mask or blanking its softmax.
    #[test]
    fn a_per_row_mask_that_is_not_parallel_to_the_rows_is_refused() {
        let mask = Masking::Rows(vec![Mask::new(vec![0, 4], 4), Mask::new(vec![0, 4], 4)]);
        let refused = stage(&[LaneMask {
            mask: Some(&mask),
            have: 0,
            rows: 4,
        }]);
        assert!(
            matches!(
                refused,
                Err(Fault::MaskRows {
                    lane: 0,
                    stated: 2,
                    rows: 4
                })
            ),
            "two masks for four rows is refused: {refused:?}"
        );
    }

    /// **THE PAGE-PADDED MASK IS THE ORDINARY ONE, AND IT STAGES THE SAME
    /// SLAB.** A guest reserves three 16-token pages for a 23-token lane and
    /// builds its rectangle 48 keys wide; the fire reads 23. Both forms of
    /// masking are handed the padded width and must produce, byte for byte,
    /// what the exact-width mask produces — the surplus is positions no query
    /// row could attend, so clipping it removes nothing.
    #[test]
    fn a_page_padded_mask_stages_what_the_exact_one_does() {
        // 20 held + 3 new = 23 readable; the pool is 3 pages of 16 = 48.
        const HAVE: u32 = 20;
        const ROWS: u32 = 3;
        const KV: u64 = 23;
        const POOL: u64 = 48;
        // A sliding window over the last eight keys: drop 0..15, keep 15..23,
        // and then — in the padded spelling — the pool's tail past the extent,
        // whose bits are set exactly so that a reader that did NOT clip would
        // stage something different.
        let exact = Mask::new(vec![15, 8], KV);
        let padded = Mask::new(vec![15, 8, 0, 25], POOL);
        let lane = |mask| LaneMask {
            mask: Some(mask),
            have: HAVE,
            rows: ROWS,
        };

        let extent_exact = Masking::Extent(exact.clone());
        let extent_padded = Masking::Extent(padded.clone());
        assert_eq!(
            stage(&[lane(&extent_padded)]).expect("a padded extent mask is accepted"),
            stage(&[lane(&extent_exact)]).expect("the exact one stages"),
        );

        let rows_exact = Masking::Rows(vec![exact.clone(), exact.clone(), exact]);
        let rows_padded = Masking::Rows(vec![padded.clone(), padded.clone(), padded]);
        assert_eq!(
            stage(&[lane(&rows_padded)]).expect("a padded per-row mask is accepted"),
            stage(&[lane(&rows_exact)]).expect("the exact one stages"),
        );

        // And the slab is the EXTENT's rectangle, not the pool's.
        let staged = stage(&[lane(&extent_padded)])
            .expect("a padded mask is accepted")
            .expect("a masked fire stages bits");
        assert_eq!(
            staged.indptr,
            vec![0, (u64::from(ROWS) * KV).div_ceil(8) as i32],
            "3 x 23 cells, not 3 x 48"
        );
        for q in 0..u64::from(ROWS) {
            for k in 0..KV {
                assert_eq!(
                    keeps(&staged, 0, KV, q, k),
                    (15..23).contains(&k) && k <= u64::from(HAVE) + q,
                    "cell ({q}, {k}) of a page-padded window"
                );
            }
        }
    }

    /// Lanes without a mask occupy no bytes, and the ones beside them keep
    /// their own offsets — the span table is what a windowed consumer slices.
    #[test]
    fn unmasked_lanes_occupy_no_bytes_and_the_table_still_bounds_them() {
        let mask = Masking::Extent(Mask::new(vec![0, 8], 8));
        let staged = stage(&[
            LaneMask {
                mask: None,
                have: 3,
                rows: 1,
            },
            LaneMask {
                mask: Some(&mask),
                have: 7,
                rows: 1,
            },
            LaneMask {
                mask: None,
                have: 1,
                rows: 1,
            },
        ])
        .expect("the mask covers its lane's extent")
        .expect("a masked fire stages bits");
        assert_eq!(staged.indptr, vec![0, 0, 1, 1]);
        assert_eq!(staged.bits, vec![0xff]);
    }

    /// A fire nobody masked binds nothing, so a masked consumer refuses in
    /// its own name rather than reading a blanked rectangle.
    #[test]
    fn a_fire_with_no_masks_stages_nothing() {
        let staged = stage(&[LaneMask {
            mask: None,
            have: 4,
            rows: 1,
        }])
        .expect("no mask is no error");
        assert_eq!(staged, None);
    }

    /// A mask SHORTER than its lane is refused by name: the missing tail
    /// would expand to zeros, and zero is masked-out, so the attention would
    /// be silently truncated to the stated prefix. (Longer is the padded
    /// shape above and is clipped.)
    #[test]
    fn a_mask_short_of_its_lanes_extent_is_refused() {
        let mask = Masking::Extent(Mask::new(vec![0, 4], 4));
        let refused = stage(&[LaneMask {
            mask: Some(&mask),
            have: 7,
            rows: 1,
        }]);
        assert!(
            matches!(
                refused,
                Err(Fault::Mask {
                    lane: 0,
                    stated: 4,
                    extent: 8
                })
            ),
            "a 4-position mask on an 8-position lane is refused: {refused:?}"
        );
    }
}
