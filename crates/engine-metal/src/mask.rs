//! Expands a lane's run-length mask ([`Mask`]) into the dense table
//! `attention.masked` reads: one byte per (fire token row, key position)
//! pair, rows are absolute fire rows, and `attention_mask_enabled[row]`
//! gates the row. Differs from the CUDA sibling (which packs bits with a
//! per-lane byte-offset span table): this plane's shaders take one scalar
//! stride and treat `kp >= stride` as masked out, so there's no per-lane
//! offset to put.
//!
//! Causality is folded into the expansion even though the shaders also
//! apply their own causal bound (redundant, but keeps the staged table
//! meaning the same regardless of which sdpa arm runs). The sliding window
//! is NOT folded in: `Attention::Masked` states it per node, so the shaders
//! take it as their own scalar argument.
//!
//! `Masking::Extent` (one restriction re-applied to every row) and
//! `Masking::Rows` (one restriction per row) both land in the same dense
//! plane. A per-row mask may only choose among keys causality already
//! allows, never add one; a mismatched row count is refused
//! ([`Fault::MaskRows`]) rather than silently reused.

use engine::fire::{Mask, Masking};

use crate::error::{Fault, Result};

/// A device-resolved dense `[rows, keys]` rectangle of bools, run-length
/// encoded here and handed to [`stage`]. `stride` is the rectangle's own
/// key width (the pool's, not the lane's extent), clipped the same as any
/// longer mask.
#[must_use]
pub fn from_dense(cells: &[bool], stride: usize) -> Masking {
    let rows = cells.len().checked_div(stride).unwrap_or(0);
    Masking::Rows(
        (0..rows)
            .map(|row| {
                let mut runs: Vec<u32> = Vec::new();
                // Alternating lengths, masked-out first — `Mask`'s own
                // encoding, so a row that opens kept opens with a
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
    /// The lane's masking, or `None` for a lane that carries none.
    /// `Masking::Rows` must state exactly [`rows`](LaneMask::rows) of them,
    /// or it's refused by name ([`Fault::MaskRows`]).
    pub mask: Option<&'a Masking>,
    /// How many KV tokens the slot held BEFORE this fire.
    pub have: u32,
    /// How many token rows this fire feeds it.
    pub rows: u32,
}

/// A fire's mask plane and the flags that gate it: one dense
/// `[rows][stride]` rectangle in fire row order (no lane offsets).
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

/// Expand a fire's lane masks, in fire (seriated) row order. `Ok(None)`
/// means no lane stated a mask, so the shell binds neither seat.
///
/// # Errors
///
/// [`Fault::Mask`]: a mask shorter than the lane's post-append KV length
/// (refused, not padded, since the missing positions would read as masked
/// out); a longer mask is accepted and clipped. [`Fault::MaskRows`]: a
/// [`Masking::Rows`] with a different row count than the lane feeds.
/// [`Fault::Ceiling`]: more key positions than a `u32` stride can name, or a
/// plane too large for host memory.
pub fn stage(lanes: &[LaneMask<'_>]) -> Result<Option<Staged>> {
    if lanes.iter().all(|lane| lane.mask.is_none()) {
        return Ok(None);
    }

    // Pass one: check every stated extent, and find the stride (the widest
    // masked lane's; an unmasked neighbour must not make every masked row
    // pay for it).
    let mut widest = 0u64;
    for (at, lane) in lanes.iter().enumerate() {
        let kv = u64::from(lane.have) + u64::from(lane.rows);
        let Some(masking) = lane.mask else {
            continue;
        };
        // The stride is the extent's, not the mask's, so a longer mask's
        // surplus is clipped. Every stated restriction is checked, not just
        // the first, since a short row truncates like a short extent.
        for mask in masking.masks() {
            if mask.total < kv {
                return Err(Fault::Mask {
                    lane: at as u32,
                    stated: mask.total,
                    extent: kv,
                });
            }
        }
        // Row count, checked after the extents and before the allocation,
        // same order as the CUDA sibling.
        if let Some(stated) = masking.stated_rows()
            && stated != lane.rows as usize
        {
            return Err(Fault::MaskRows {
                lane: at as u32,
                stated: stated as u64,
                rows: lane.rows,
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

    // Pass two: the runs. `Extent` is walked once per lane and applied to
    // every row (only the causal bound moves); `Rows` walks once per row,
    // each under its own bound.
    let mut row = 0usize;
    for lane in lanes {
        let Some(masking) = lane.mask else {
            row += lane.rows as usize;
            continue;
        };
        let kv = u64::from(lane.have) + u64::from(lane.rows);
        for q in 0..lane.rows as usize {
            out.enabled[row + q] = 1;
        }
        let base = row as u64;
        match masking {
            Masking::Extent(mask) => {
                let mut at_position = 0u64;
                for (index, &run) in mask.runs.iter().enumerate() {
                    let end = at_position.saturating_add(u64::from(run)).min(kv);
                    if index % 2 == 1 {
                        for key in at_position..end {
                            // Causal, read from the key's side: the first
                            // row that may see key `key` is `key - have`.
                            let first = key.saturating_sub(u64::from(lane.have));
                            for q in first..u64::from(lane.rows) {
                                keep(&mut out.bytes, stride, base + q, key);
                            }
                        }
                    }
                    if end == kv {
                        break;
                    }
                    at_position = end;
                }
            }
            Masking::Rows(masks) => {
                for (q, mask) in masks.iter().enumerate() {
                    let q = q as u64;
                    // Same bound, read from the row's side, inclusive: row
                    // `q` stands at `have + q` and may reach its own key.
                    let bound = u64::from(lane.have) + q;
                    let mut at_position = 0u64;
                    for (index, &run) in mask.runs.iter().enumerate() {
                        let end = at_position.saturating_add(u64::from(run)).min(kv);
                        if index % 2 == 1 {
                            for key in at_position..end.min(bound + 1) {
                                keep(&mut out.bytes, stride, base + q, key);
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
        row += lane.rows as usize;
    }
    Ok(Some(out))
}

/// One cell of the plane, kept. A free function so both walks above write
/// through the same address (`row * stride + key`, absolute fire row); a
/// bounds miss is a no-op rather than a panic, since stride and row count
/// come from the same `have + rows` the callers walk.
fn keep(bytes: &mut [u8], stride: u32, row: u64, key: u64) {
    let cell = row * u64::from(stride) + key;
    if let Some(word) = bytes.get_mut(cell as usize) {
        *word = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row count that is not the lane's is refused by name.
    #[test]
    fn a_per_row_mask_of_the_wrong_height_is_refused() {
        let short = Masking::Rows(vec![Mask::new(vec![0, 3], 3), Mask::new(vec![0, 3], 3)]);
        let refused = stage(&[LaneMask {
            mask: Some(&short),
            have: 0,
            rows: 3,
        }]);
        assert!(
            matches!(
                refused,
                Err(Fault::MaskRows {
                    lane: 0,
                    stated: 2,
                    rows: 3
                })
            ),
            "a two-row masking on a three-row lane is refused: {refused:?}"
        );
        let said = Fault::MaskRows {
            lane: 0,
            stated: 2,
            rows: 3,
        }
        .to_string();
        assert!(
            said.contains("Masking::Rows"),
            "the refusal names the form it is about: {said}"
        );
    }

    /// A per-row masking short of the extent is refused for the extent
    /// first, before the row count is checked.
    #[test]
    fn a_short_per_row_mask_is_refused_for_its_extent_first() {
        let short = Masking::Rows(vec![Mask::new(vec![0, 2], 2), Mask::new(vec![0, 2], 2)]);
        let refused = stage(&[LaneMask {
            mask: Some(&short),
            have: 1,
            rows: 3,
        }]);
        assert!(
            matches!(
                refused,
                Err(Fault::Mask {
                    lane: 0,
                    stated: 2,
                    extent: 4
                })
            ),
            "a mask that is both short and miscounted names its extent: {refused:?}"
        );
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

    /// A mask short of its lane's extent is refused by name.
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
