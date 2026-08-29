//! The `masked` axis's one piece of arithmetic: a lane's run-length mask,
//! expanded into the table `attention.masked` actually reads.
//!
//! **THE TWO FORMS ARE NOT THE SAME SHAPE, AND THAT IS THE WHOLE FILE.** A
//! submission states one mask per lane over that lane's readable extent —
//! [`Mask`] is a run-length encoding over `total` KV positions, because a
//! prefix drop, a retained-block set and a sliding window are each a handful
//! of runs over thousands of positions. The shader reads something else:
//! `attn/sdpa_paged.metal` and its mma twin index
//! `attention_mask[row * attention_mask_stride + kp]`, one `uchar` per
//! (fire token row, key position) PAIR, with `attention_mask_enabled[row]`
//! gating the row. So the shell owns one expansion, and it is here rather
//! than inside the fire path because it is the only place in this crate
//! where a wrong cell is silently a different sentence rather than a fault.
//!
//! # Where this diverges from the CUDA shell, and why it must
//!
//! The sibling stages PACKED BITS with a per-lane BYTE-OFFSET span table,
//! because flashinfer's fa2 custom-mask arm reads `qo_idx * kv_len + kv_idx`
//! LSB-first inside each byte and takes `maybe_custom_mask +
//! maybe_mask_indptr[batch_idx]` as pointer arithmetic on a `uint8_t*`.
//! Neither half of that survives the crossing:
//!
//! ```text
//! CUDA   one BIT per pair, LSB-first     Metal   one BYTE per pair, != 0
//! CUDA   per-lane byte offsets, indptr   Metal   one uniform row STRIDE
//! CUDA   rows are lane-relative          Metal   rows are ABSOLUTE fire rows
//! ```
//!
//! The stride is not a shell preference: the shaders take
//! `attention_mask_stride` as a scalar (`buffer(13)`), address the plane by
//! the absolute token row they are already resolving `position_ids` and
//! `req_of_token` at, and treat `kp >= attention_mask_stride` as MASKED OUT.
//! So there is no lane base to hand them and no place to put one — the
//! rectangle is `[fire rows][stride]`, dense, and the stride must cover the
//! widest masked lane in the fire or that lane's tail silently drops. A
//! packed-bit slab with a span table would be a table nothing on this plane
//! can bind; porting the CUDA layout verbatim is the one way to get this
//! file wrong.
//!
//! The cost of the byte form is real and is the platform's: a masked prefill
//! of `n` rows over `n` keys is `n²` bytes here against `n²/8` there. It is
//! paid rather than argued with, because the alternative is a shader change.
//!
//! # Causality is folded IN, and the sliding window is not
//!
//! `keep(q, k) = the lane's runs keep k ∧ k <= have + q`, which is exactly
//! `Lane::mask`'s own words — "which readable extent of a slot a lane's
//! attention may reach" — intersected with the order the cache is written
//! in. On a one-row lane the second term is vacuous and the two readings
//! coincide, which is why a decode-shaped mask needs no special case.
//!
//! **THE FOLD IS REDUNDANT ON THIS PLANE AND IS KEPT ANYWAY.** Unlike the
//! fa2 custom arm — which reads the bit and nothing else, so a multi-row
//! lane given its mask verbatim would let row 0 attend row 4's key — the
//! metal shaders apply the causal bound themselves: `sdpa_paged.metal` walks
//! `kp` up to `q_pos` and no further, and the mma arm spells
//! `keep = kp <= q_pos && kp >= my_start` before it consults the mask at
//! all. Folding it in costs nothing (the plane is written row by row
//! regardless), makes the staged table MEAN the same thing on both planes,
//! and keeps this file correct against whichever sdpa arm the dispatcher
//! picks rather than against the two that exist today. An intersection with
//! a bound the shader also applies is idempotent.
//!
//! The SLIDING window is deliberately not folded in, and the reason is that
//! it is not a property of the lane. `Attention::Masked` states its window
//! per NODE — gemma's text alternates `Some(512)` and `None` down its layer
//! stack — so one slab cannot serve both, and a fold would have to be per
//! layer. The metal shaders take the window as their own scalar argument and
//! apply it beside the mask (`my_start`), which is the right place for a
//! per-node fact and the reason it must not be baked into a per-lane table.
//!
//! # The per-row form is REFUSED, by name
//!
//! `Masking` has two shapes: `Extent`, one restriction of the lane's readable
//! extent re-applied to every query row, and `Rows`, one restriction PER
//! query row — the windowed prefill, where row `i` keeps `[i - w, i]` and no
//! single extent mask is any of them. The CUDA shell expands both (one walk
//! for the first, one walk per row for the second, each under that row's own
//! causal bound). This plane expands the first and refuses the second by name
//! ([`Fault::MaskRows`]).
//!
//! Nothing about the metal ABI makes it impossible: the plane written here is
//! already a dense `[fire rows][stride]` rectangle addressed by absolute row,
//! so a per-row walk would write into exactly the same cells the extent walk
//! writes into. What is missing is the wave — metal parity for the per-row
//! form was explicitly out of scope where the form landed — and a refusal
//! that names the SHAPE is what keeps that from being discovered as a wrong
//! continuation. Serving `Rows` as `Extent` would be row zero's mask on every
//! row, which is the silent substitution the form exists to end.
//!
//! # `enabled` is per ROW, not per lane
//!
//! `kernels_metal::attn`'s field doc calls `mask_enabled` "one per request",
//! and the shaders index it `attention_mask_enabled[row]` at the absolute
//! token row. The shaders are what runs, so this file stages one flag per
//! fire ROW — every row of a masked lane set, every row of an unmasked one
//! clear. A lane's rows all carry the same flag, so the two readings agree
//! for every fire the composition can build; stating it per row is what
//! makes them agree by construction rather than by coincidence.

use engine::engine_api::fire::{Mask, Masking};

use crate::error::{Fault, Result};

/// One lane's mask, with the geometry that says what shape it expands to.
#[derive(Debug, Clone, Copy)]
pub struct LaneMask<'a> {
    /// The lane's masking, or `None` for a lane that carries none.
    ///
    /// **ONLY `Masking::Extent` EXPANDS ON THIS PLANE.** The per-row form is
    /// refused by name ([`Fault::MaskRows`]) — see the module doc's last
    /// section.
    pub mask: Option<&'a Masking>,
    /// How many KV tokens the slot held BEFORE this fire.
    pub have: u32,
    /// How many token rows this fire feeds it.
    pub rows: u32,
}

/// A fire's mask plane and the flags that gate it, ready to write into the
/// input reservation.
///
/// One dense `[rows][stride]` rectangle in fire row order, which is the
/// order the shaders address it in — no lane offsets, because there is
/// nowhere to state one (see the module doc).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Staged {
    /// `[rows * stride]`: 1 keeps the pair, 0 drops it. Row `r` of the fire
    /// starts at `r * stride`.
    pub bytes: Vec<u8>,
    /// `[rows]`: whether that row's plane is consulted at all. Rows of a
    /// lane that stated no mask are 0, and the shaders then read neither
    /// the plane nor the stride bound.
    pub enabled: Vec<u8>,
    /// Key positions from one row's plane to the next — the widest masked
    /// lane's post-append KV length, and the bound the shaders themselves
    /// enforce with `kp >= attention_mask_stride`.
    pub stride: u32,
}

/// Expand a fire's lane masks, in fire (seriated) row order.
///
/// `Ok(None)` is a fire no lane put a mask on — the shell then binds neither
/// seat, and a masked consumer gets `attn::masked`'s own typed refusal
/// instead of a rectangle of zeros that would blank every logit.
///
/// # Errors
///
/// [`Fault::Mask`] for a mask SHORTER than the lane's post-append KV length.
/// Short is the dangerous direction — the missing positions read as
/// MASKED-OUT and quietly truncate the attention — so it is refused rather
/// than padded. A LONGER mask is accepted and clipped to the extent (the
/// module doc's page-padded shape).
///
/// [`Fault::MaskRows`] for a [`Masking::Rows`]: the per-row form has no
/// expansion on this plane.
///
/// [`Fault::Ceiling`] for a fire whose widest masked lane needs more key
/// positions than a `u32` stride can name, or whose whole plane does not fit
/// in host memory this process can address.
pub fn stage(lanes: &[LaneMask<'_>]) -> Result<Option<Staged>> {
    if lanes.iter().all(|lane| lane.mask.is_none()) {
        return Ok(None);
    }

    // Pass one: check every stated extent, and find the stride. The widest
    // MASKED lane sets it — an unmasked lane's rows are never consulted, so
    // a long unmasked neighbour must not make every masked row pay for it.
    let mut widest = 0u64;
    for (at, lane) in lanes.iter().enumerate() {
        let kv = u64::from(lane.have) + u64::from(lane.rows);
        let Some(masking) = lane.mask else {
            continue;
        };
        let mask = extent_of(masking, at as u32)?;
        // SHORT IS THE FAULT; LONGER IS THE ECOSYSTEM'S ORDINARY SHAPE and is
        // clipped, exactly as the CUDA sibling takes it — a guest states its
        // mask over the pages it RESERVED (48 keys for a 3-page pool holding
        // 23 tokens), and a position past the extent is one no query row can
        // attend under the causal bound anyway. The stride below is the
        // EXTENT's, not the mask's, so the surplus never reaches the plane.
        if mask.total < kv {
            return Err(Fault::Mask {
                lane: at as u32,
                stated: mask.total,
                extent: kv,
            });
        }
        widest = widest.max(kv);
    }
    let stride = u32::try_from(widest).map_err(|_| Fault::Ceiling {
        what: "key positions in one mask row",
        need: widest,
        have: u64::from(u32::MAX),
    })?;

    let rows: u64 = lanes.iter().map(|lane| u64::from(lane.rows)).sum();
    let cells = rows.saturating_mul(u64::from(stride));
    let size = usize::try_from(cells).map_err(|_| Fault::Ceiling {
        what: "bytes of one fire's mask plane",
        need: cells,
        have: usize::MAX as u64,
    })?;
    let mut out = Staged {
        bytes: vec![0; size],
        enabled: vec![0; rows as usize],
        stride,
    };

    // Pass two: the runs, walked once per lane and applied to every query row
    // — one pass over the encoding rather than `rows` passes, because a
    // prefill lane's mask is the same runs at every row and only the causal
    // bound moves.
    let mut row = 0usize;
    for (at, lane) in lanes.iter().enumerate() {
        let Some(masking) = lane.mask else {
            row += lane.rows as usize;
            continue;
        };
        let mask = extent_of(masking, at as u32)?;
        let kv = u64::from(lane.have) + u64::from(lane.rows);
        for q in 0..lane.rows as usize {
            out.enabled[row + q] = 1;
        }
        let mut at_position = 0u64;
        for (index, &run) in mask.runs.iter().enumerate() {
            let end = at_position.saturating_add(u64::from(run)).min(kv);
            if index % 2 == 1 {
                for key in at_position..end {
                    // Causal: query row `q` stands at absolute position
                    // `have + q`, and a key past it has not been written.
                    let first = key.saturating_sub(u64::from(lane.have));
                    for q in first..u64::from(lane.rows) {
                        let cell = (row as u64 + q) * u64::from(stride) + key;
                        if let Some(word) = out.bytes.get_mut(cell as usize) {
                            *word = 1;
                        }
                    }
                }
            }
            if end == kv {
                break;
            }
            at_position = end;
        }
        row += lane.rows as usize;
    }
    Ok(Some(out))
}

/// The one restriction this plane can expand, or the refusal that names the
/// form it cannot.
///
/// Both passes of [`stage`] go through it so that a per-row mask is refused
/// on the FIRST pass — before a stride is chosen and a plane is allocated for
/// a fire that will not run.
fn extent_of(masking: &Masking, lane: u32) -> Result<&Mask> {
    match masking {
        Masking::Extent(mask) => Ok(mask),
        Masking::Rows(_) => Err(Fault::MaskRows { lane }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Is `(row, k)` kept, read the way the shader reads it?
    fn keeps(staged: &Staged, row: u64, k: u64) -> bool {
        staged.enabled[row as usize] != 0
            && k < u64::from(staged.stride)
            && staged.bytes[(row * u64::from(staged.stride) + k) as usize] != 0
    }

    /// A one-row lane is the decode-shaped mask, and its plane is the runs
    /// verbatim: the causal bound at `have + 0` keeps every key the cache
    /// holds.
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
        .expect("a masked fire stages a plane");
        assert_eq!(staged.stride, 8, "one lane of 8 keys is an 8-wide row");
        assert_eq!(staged.enabled, vec![1]);
        assert_eq!(staged.bytes.len(), 8, "one row, one byte per key");
        for k in 0..8 {
            assert_eq!(
                keeps(&staged, 0, k),
                k >= 3,
                "key {k} of a three-position prefix drop"
            );
        }
    }

    /// A multi-row lane gets the causal bound, which the shaders also apply
    /// — the intersection is the point, not the redundancy.
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
        .expect("a masked fire stages a plane");
        assert_eq!(staged.stride, 4);
        assert_eq!(staged.enabled, vec![1, 1, 1, 1]);
        assert_eq!(staged.bytes.len(), 16, "4 rows of 4 keys");
        for q in 0..4 {
            for k in 0..4 {
                assert_eq!(keeps(&staged, q, k), k <= q, "row {q} must not reach key {k}");
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
        .expect("a masked fire stages a plane");
        for q in 0..3u64 {
            for k in 0..5u64 {
                let want = (1..4).contains(&k) && k <= 2 + q;
                assert_eq!(keeps(&staged, q, k), want, "cell ({q}, {k})");
            }
        }
    }

    /// Lanes without a mask still own their rows of the plane — the rows are
    /// ABSOLUTE fire rows on this plane — and their flag is what turns the
    /// plane off for them.
    #[test]
    fn unmasked_lanes_own_their_rows_and_their_flag_is_clear() {
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
        .expect("a masked fire stages a plane");
        assert_eq!(staged.stride, 8, "the masked lane sets the stride alone");
        assert_eq!(staged.enabled, vec![0, 1, 0]);
        assert_eq!(staged.bytes.len(), 24, "3 fire rows of 8");
        assert!(staged.bytes[..8].iter().all(|&cell| cell == 0));
        assert!(staged.bytes[8..16].iter().all(|&cell| cell == 1));
        assert!(staged.bytes[16..].iter().all(|&cell| cell == 0));
    }

    /// **THE PER-ROW FORM IS REFUSED BY NAME, ON THE FIRST PASS.** A windowed
    /// prefill states one restriction per query row; this plane expands only
    /// `Masking::Extent`, and the alternative to naming the shape is serving
    /// row zero's mask on every row.
    #[test]
    fn a_per_row_mask_is_refused_by_the_name_of_its_form() {
        let windowed = Masking::Rows(vec![
            Mask::new(vec![0, 1, 2], 3),
            Mask::new(vec![0, 2, 1], 3),
            Mask::new(vec![1, 2], 3),
        ]);
        let refused = stage(&[LaneMask {
            mask: Some(&windowed),
            have: 0,
            rows: 3,
        }]);
        assert!(
            matches!(refused, Err(Fault::MaskRows { lane: 0 })),
            "a `Masking::Rows` must be refused by its own name on this plane: \
             {refused:?}"
        );
        let said = Fault::MaskRows { lane: 0 }.to_string();
        assert!(
            said.contains("Masking::Rows"),
            "the refusal names the form it cannot serve: {said}"
        );
    }

    /// **THE PAGE-PADDED MASK IS THE ORDINARY ONE, AND IT STAGES THE SAME
    /// PLANE.** A guest reserves three 16-token pages for a 23-token lane and
    /// builds its rectangle 48 keys wide; the fire reads 23. The surplus is
    /// clipped — including out of the STRIDE, which is the extent's and not
    /// the mask's — so the padded spelling and the exact one are one plane.
    #[test]
    fn a_page_padded_mask_stages_what_the_exact_one_does() {
        const HAVE: u32 = 20;
        const ROWS: u32 = 3;
        const KV: u64 = 23;
        // Keep keys 15..23; the padded spelling adds a kept run past the
        // extent, out to the 48-key pool width.
        let exact = Masking::Extent(Mask::new(vec![15, 8], KV));
        let padded = Masking::Extent(Mask::new(vec![15, 8, 0, 25], 48));
        let lane = |mask| LaneMask {
            mask: Some(mask),
            have: HAVE,
            rows: ROWS,
        };
        let staged = stage(&[lane(&padded)]).expect("a padded mask is accepted");
        assert_eq!(staged, stage(&[lane(&exact)]).expect("the exact one stages"));
        let staged = staged.expect("a masked fire stages a plane");
        assert_eq!(staged.stride, KV as u32, "the stride is the EXTENT's");
        for q in 0..u64::from(ROWS) {
            for k in 0..KV {
                assert_eq!(
                    keeps(&staged, q, k),
                    (15..23).contains(&k) && k <= u64::from(HAVE) + q,
                    "cell ({q}, {k}) of a page-padded window"
                );
            }
        }
    }

    /// The stride is the widest MASKED lane's extent, and a shorter lane's
    /// tail past its own KV length stays masked out — which is what the
    /// shaders' own `kp >= stride` bound would say anyway.
    #[test]
    fn the_stride_is_the_widest_masked_lanes_extent() {
        let short = Masking::Extent(Mask::new(vec![0, 2], 2));
        let long = Masking::Extent(Mask::new(vec![0, 6], 6));
        let staged = stage(&[
            LaneMask {
                mask: Some(&short),
                have: 1,
                rows: 1,
            },
            LaneMask {
                mask: Some(&long),
                have: 5,
                rows: 1,
            },
        ])
        .expect("both masks cover their lanes")
        .expect("a masked fire stages a plane");
        assert_eq!(staged.stride, 6);
        assert_eq!(staged.enabled, vec![1, 1]);
        for k in 0..6 {
            assert_eq!(keeps(&staged, 0, k), k < 2, "the short lane holds 2 keys");
            assert!(keeps(&staged, 1, k), "the long lane holds all 6");
        }
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
