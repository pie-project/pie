//! Expands a lane's run-length [`Masking`] (per-extent or per-row) into the packed bit slab `attention.masked`'s custom-mask kernel reads: one bit per (query row, key) at `qo_idx * kv_len + kv_idx`, LSB-first, each lane starting on a byte boundary, ANDed with the causal bound the kernel does not otherwise apply — unless the lane is [`LaneMask::bidirectional`], the denoiser's reading, where every row keeps every key its runs keep. A mask longer than the lane's extent is clipped; shorter is `Fault::Mask`.

use engine::fire::{Mask, Masking};

use crate::error::{Fault, Result};

/// Encodes a dense `[rows, keys]` bool rectangle into per-row run-length
/// masks. `stride` is the rectangle's own key width (e.g. a reserved pool),
/// not necessarily the lane's extent.
#[must_use]
pub fn from_dense(cells: &[bool], stride: usize) -> Masking {
    let rows = if stride == 0 { 0 } else { cells.len() / stride };
    Masking::Rows(
        (0..rows)
            .map(|row| {
                let mut runs: Vec<u32> = Vec::new();
                // Alternating lengths, masked-out first (`Mask`'s encoding).
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
    /// Lift the causal bound: a row keeps a key past its own position too.
    pub bidirectional: bool,
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

/// Expand a fire's lane masks, in fire (seriated) row order. `Ok(None)` is a fire no lane put a mask on.
/// Errs [`Fault::Mask`] for a mask shorter than the lane's post-append KV length (longer is accepted and clipped); [`Fault::MaskRows`] for a [`Masking::Rows`] whose row count doesn't match the lane's.
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
                // Walked once; the same runs apply to every row.
                Masking::Extent(mask) => {
                    let mut at_position = 0u64;
                    for (index, &run) in mask.runs.iter().enumerate() {
                        let end = at_position.saturating_add(u64::from(run)).min(kv);
                        if index % 2 == 1 {
                            for key in at_position..end {
                                // query row q stands at absolute position have + q; a key past it is unwritten — unless the lane reads bidirectionally, when every row of the fire's own keys is written before any row attends.
                                let first = if lane.bidirectional {
                                    0
                                } else {
                                    key.saturating_sub(u64::from(lane.have))
                                };
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
                // Walked once per row, under that row's own causal bound;
                // a row's runs may only narrow what causality already
                // allows, never widen it.
                Masking::Rows(rows) => {
                    for (q, mask) in rows.iter().enumerate() {
                        let q = q as u64;
                        let bound = if lane.bidirectional {
                            kv
                        } else {
                            u64::from(lane.have) + q
                        };
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

    /// The runs and the causal bound are ANDed, not chosen between.
    #[test]
    fn the_runs_and_the_causal_bound_intersect() {
        // 2 held + 3 new = 5; keep positions 1..4 (drop 0 and 4).
        let mask = Masking::Extent(Mask::new(vec![1, 3, 1], 5));
        let staged = stage(&[LaneMask {
            mask: Some(&mask),
            have: 2,
            rows: 3,
            bidirectional: false,
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

    /// A sliding window (row q keeps `[q-1, q]`) is not any single
    /// `Masking::Extent`, so it needs the per-row form.
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
            bidirectional: false,
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

    #[test]
    fn a_mask_short_of_its_lanes_extent_is_refused() {
        let mask = Masking::Extent(Mask::new(vec![0, 4], 4));
        let refused = stage(&[LaneMask {
            mask: Some(&mask),
            have: 7,
            rows: 1,
            bidirectional: false,
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
