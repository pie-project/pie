//! The `masked` axis's one piece of arithmetic: a lane's run-length mask,
//! expanded into the bits `attention.masked` actually reads.
//!
//! **THE TWO FORMS ARE NOT THE SAME SHAPE, AND THAT IS THE WHOLE FILE.** A
//! submission states one mask per lane over that lane's readable extent —
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
//! keep(q, k) = the lane's runs keep k   ∧   k <= have + q
//! ```
//!
//! which is exactly `Lane::mask`'s own words — "which readable extent of a
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
//! # The span table holds BYTES
//!
//! `variants.cuh` reads `maybe_custom_mask + maybe_mask_indptr[batch_idx]`
//! as pointer arithmetic on a `uint8_t*` — a BYTE offset, not the bit offset
//! upstream flashinfer carries — so every lane's region starts on a byte
//! boundary and the table's entries are byte counts. The bits slab is
//! fire-wide and its offsets are ABSOLUTE, which is what lets a windowed
//! consumer take a sliced span table and the whole slab (the same shape
//! `GeomKind::Indices` and its bounds vector already have).

use engine::engine_api::fire::Mask;

use crate::error::{Fault, Result};

/// One lane's mask, with the geometry that says what shape it expands to.
#[derive(Debug, Clone, Copy)]
pub struct LaneMask<'a> {
    /// The lane's mask, or `None` for a lane that carries none.
    pub mask: Option<&'a Mask>,
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
/// [`Fault::Mask`] for a mask whose extent is not the lane's post-append KV
/// length. Short is the dangerous direction — the missing positions would
/// read as MASKED-OUT and quietly truncate the attention — so neither
/// direction is repaired.
pub fn stage(lanes: &[LaneMask<'_>]) -> Result<Option<Staged>> {
    if lanes.iter().all(|lane| lane.mask.is_none()) {
        return Ok(None);
    }
    let mut bits: Vec<u8> = Vec::new();
    let mut indptr: Vec<i32> = Vec::with_capacity(lanes.len() + 1);
    indptr.push(0);
    for (at, lane) in lanes.iter().enumerate() {
        let kv = u64::from(lane.have) + u64::from(lane.rows);
        if let Some(mask) = lane.mask {
            if mask.total != kv {
                return Err(Fault::Mask {
                    lane: at as u32,
                    stated: mask.total,
                    extent: kv,
                });
            }
            let cells = u64::from(lane.rows) * kv;
            let bytes = usize::try_from(cells.div_ceil(8)).unwrap_or(usize::MAX);
            let base = bits.len();
            bits.resize(base + bytes, 0);
            // The runs, walked once per lane and applied to every query row —
            // one pass over the encoding rather than `rows` passes, because a
            // prefill lane's mask is the same runs at every row and only the
            // causal bound moves.
            let mut at_position = 0u64;
            for (index, &run) in mask.runs.iter().enumerate() {
                let end = at_position.saturating_add(u64::from(run)).min(kv);
                if index % 2 == 1 {
                    for key in at_position..end {
                        // Causal: query row `q` stands at absolute position
                        // `have + q`, and a key past it has not been written.
                        let first = key.saturating_sub(u64::from(lane.have));
                        for q in first..u64::from(lane.rows) {
                            let cell = q * kv + key;
                            let byte = base + usize::try_from(cell / 8).unwrap_or(usize::MAX);
                            if let Some(word) = bits.get_mut(byte) {
                                *word |= 1 << (cell % 8);
                            }
                        }
                    }
                }
                if end == kv {
                    break;
                }
                at_position = end;
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
        let mask = Mask::new(vec![3, 5], 8);
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
        let mask = Mask::new(vec![0, 4], 4);
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
        let mask = Mask::new(vec![1, 3, 1], 5);
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

    /// Lanes without a mask occupy no bytes, and the ones beside them keep
    /// their own offsets — the span table is what a windowed consumer slices.
    #[test]
    fn unmasked_lanes_occupy_no_bytes_and_the_table_still_bounds_them() {
        let mask = Mask::new(vec![0, 8], 8);
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

    /// A mask that covers a different extent than the lane is refused by
    /// name. Short would truncate the attention silently.
    #[test]
    fn a_mask_of_the_wrong_extent_is_refused() {
        let mask = Mask::new(vec![0, 4], 4);
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
