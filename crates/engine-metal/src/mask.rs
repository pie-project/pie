use engine::fire::Masking;

use crate::error::{Fault, Result};

#[derive(Debug, Clone, Copy)]
pub struct LaneMask<'a> {
    pub mask: Option<&'a Masking>,
    pub have: u32,
    /// The lane's committed boundary (`KvDelta::folded`, at most `have`): its
    /// rows read every key below it with no mask bit, and its mask's column
    /// `c` is key `folded + c`.
    pub folded: u32,
    pub rows: u32,
    /// Attends both ways: each row reads every key the lane holds; enable flag 2, every bit kept
    /// unless a mask is stated.
    pub wide: bool,
}

impl LaneMask<'_> {
    /// The columns the lane's rows read through the plane: its buffer, then
    /// its own rows.
    fn extent(&self) -> u64 {
        u64::from(self.have - self.folded.min(self.have)) + u64::from(self.rows)
    }

    /// Where the lane's own rows start among its columns.
    fn buffer(&self) -> u64 {
        u64::from(self.have - self.folded.min(self.have))
    }
}

/// A fire's mask plane: a row of `stride` columns each masked row reads from
/// its lane's committed boundary on, and a word a row packing its enable
/// flag (bits 0-1: 1 masked, 2 wide) and that boundary (bits 2-31), the key
/// its column 0 is.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Staged {
    pub bytes: Vec<u8>,
    pub words: Vec<u32>,
    pub stride: u32,
}

/// The most a boundary can be in a row's word.
const ORIGIN_MAX: u32 = u32::MAX >> 2;

pub fn stage(lanes: &[LaneMask<'_>]) -> Result<Option<Staged>> {
    if lanes.iter().all(|lane| lane.mask.is_none() && !lane.wide) {
        return Ok(None);
    }

    let mut widest = 0u64;
    for (at, lane) in lanes.iter().enumerate() {
        let extent = lane.extent();
        if lane.folded > ORIGIN_MAX {
            return Err(Fault::Ceiling {
                what: "a masked lane's committed boundary in its row's mask word",
                need: u64::from(lane.folded),
                have: u64::from(ORIGIN_MAX),
            });
        }
        let Some(masking) = lane.mask else {
            if lane.wide {
                widest = widest.max(extent);
            }
            continue;
        };

        for mask in masking.masks() {
            if mask.total < extent {
                return Err(Fault::Mask {
                    lane: at as u32,
                    stated: mask.total,
                    extent,
                });
            }
        }

        if let Some(stated) = masking.stated_rows()
            && stated != lane.rows as usize
        {
            return Err(Fault::MaskRows {
                lane: at as u32,
                stated: stated as u64,
                rows: lane.rows,
            });
        }
        widest = widest.max(extent);
    }
    // A whole number of 4-byte words a row: a copied window gathers the
    // plane's rows as words (`dispatch::copy`). Columns past a lane's extent
    // are no key of it, and read as hidden.
    let widest = widest.next_multiple_of(4);
    let stride = u32::try_from(widest).map_err(|_| Fault::Ceiling {
        what: "columns in one mask row",
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
        words: vec![0; rows as usize],
        stride,
    };

    let mut row = 0usize;
    for lane in lanes {
        let extent = lane.extent();
        let buffer = lane.buffer();
        let word = (if lane.wide { 2 } else { 1 }) | (lane.folded.min(lane.have) << 2);
        let Some(masking) = lane.mask else {
            if lane.wide {
                for q in 0..u64::from(lane.rows) {
                    out.words[row + q as usize] = word;
                    for column in 0..extent {
                        keep(&mut out.bytes, stride, row as u64 + q, column);
                    }
                }
            }
            row += lane.rows as usize;
            continue;
        };
        for q in 0..lane.rows as usize {
            out.words[row + q] = word;
        }
        let base = row as u64;
        // A row reads no own row after it, whatever its bits say: the
        // bound is row order, past its buffer.
        let bound = |q: u64| if lane.wide { extent } else { buffer + q + 1 };
        match masking {
            Masking::Extent(mask) => {
                for span in mask.admitted(0..extent) {
                    for column in span {
                        let first = if lane.wide {
                            0
                        } else {
                            column.saturating_sub(buffer)
                        };
                        for q in first..u64::from(lane.rows) {
                            keep(&mut out.bytes, stride, base + q, column);
                        }
                    }
                }
            }
            Masking::Rows(masks) => {
                for (q, mask) in masks.iter().enumerate() {
                    let q = q as u64;
                    for span in mask.admitted(0..bound(q)) {
                        for column in span {
                            keep(&mut out.bytes, stride, base + q, column);
                        }
                    }
                }
            }
        }
        row += lane.rows as usize;
    }
    Ok(Some(out))
}

fn keep(bytes: &mut [u8], stride: u32, row: u64, column: u64) {
    let cell = row * u64::from(stride) + column;
    if let Some(word) = bytes.get_mut(cell as usize) {
        *word = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::fire::Mask;

    #[test]
    fn mask_every_case() {
        a_wide_lane_keeps_the_keys_after_the_row();
        a_per_row_mask_of_the_wrong_height_is_refused();
        a_short_per_row_mask_is_refused_for_its_extent_first();
        a_fire_with_no_masks_stages_nothing();
        a_mask_short_of_its_lanes_extent_is_refused();
        a_mask_speaks_of_the_keys_past_the_boundary_alone();
    }

    /// A lane with ten keys committed and two buffered, writing two rows:
    /// its plane is four columns (the buffer, then its rows), its words carry
    /// the boundary, and a row reads no own row after it. No column of the
    /// plane is a committed key, so a long committed part grows no plane.
    fn a_mask_speaks_of_the_keys_past_the_boundary_alone() {
        // Hides the first buffered key, admits the rest.
        let mask = Masking::Extent(Mask::new(vec![1, 3], 4));
        let staged = stage(&[LaneMask {
            mask: Some(&mask),
            have: 12,
            folded: 10,
            rows: 2,
            wide: false,
        }])
        .unwrap()
        .unwrap();
        assert_eq!(staged.stride, 4, "buffer and rows, not the context");
        assert_eq!(staged.words, vec![1 | (10 << 2), 1 | (10 << 2)]);
        assert_eq!(staged.bytes, vec![0, 1, 1, 0, 0, 1, 1, 1]);
        // Its extent is its buffer and its rows, not its keys.
        let short = Masking::Extent(Mask::new(vec![0, 3], 3));
        let refused = stage(&[LaneMask {
            mask: Some(&short),
            have: 12,
            folded: 10,
            rows: 2,
            wide: false,
        }]);
        assert!(
            matches!(
                refused,
                Err(Fault::Mask {
                    stated: 3,
                    extent: 4,
                    ..
                })
            ),
            "{refused:?}"
        );
    }

    fn a_wide_lane_keeps_the_keys_after_the_row() {
        let all = Masking::Extent(Mask::new(vec![0, 3], 3));
        let causal = stage(&[LaneMask {
            mask: Some(&all),
            have: 1,
            folded: 0,
            rows: 2,
            wide: false,
        }])
        .unwrap()
        .unwrap();
        assert_eq!(causal.words, vec![1, 1]);
        // Three columns a row, padded to a whole word (`stage`): the fourth
        // is no key and reads as hidden.
        assert_eq!(causal.stride, 4);
        assert_eq!(causal.bytes, vec![1, 1, 0, 0, 1, 1, 1, 0]);
        let wide = stage(&[LaneMask {
            mask: Some(&all),
            have: 1,
            folded: 0,
            rows: 2,
            wide: true,
        }])
        .unwrap()
        .unwrap();
        assert_eq!(wide.words, vec![2, 2]);
        assert_eq!(wide.bytes, vec![1, 1, 1, 0, 1, 1, 1, 0]);
        let rows = Masking::Rows(vec![Mask::new(vec![0, 3], 3), Mask::new(vec![1, 2], 3)]);
        let wide = stage(&[LaneMask {
            mask: Some(&rows),
            have: 1,
            folded: 0,
            rows: 2,
            wide: true,
        }])
        .unwrap()
        .unwrap();
        assert_eq!(wide.bytes, vec![1, 1, 1, 0, 0, 1, 1, 0]);
    }

    fn a_per_row_mask_of_the_wrong_height_is_refused() {
        let short = Masking::Rows(vec![Mask::new(vec![0, 3], 3), Mask::new(vec![0, 3], 3)]);
        let refused = stage(&[LaneMask {
            mask: Some(&short),
            have: 0,
            folded: 0,
            rows: 3,
            wide: false,
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

    fn a_short_per_row_mask_is_refused_for_its_extent_first() {
        let short = Masking::Rows(vec![Mask::new(vec![0, 2], 2), Mask::new(vec![0, 2], 2)]);
        let refused = stage(&[LaneMask {
            mask: Some(&short),
            have: 1,
            folded: 0,
            rows: 3,
            wide: false,
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

    fn a_fire_with_no_masks_stages_nothing() {
        let staged = stage(&[LaneMask {
            mask: None,
            have: 4,
            folded: 0,
            rows: 1,
            wide: false,
        }])
        .expect("no mask is no error");
        assert_eq!(staged, None);
    }

    fn a_mask_short_of_its_lanes_extent_is_refused() {
        let mask = Masking::Extent(Mask::new(vec![0, 4], 4));
        let refused = stage(&[LaneMask {
            mask: Some(&mask),
            have: 7,
            folded: 0,
            rows: 1,
            wide: false,
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
