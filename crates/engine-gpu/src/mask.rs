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
}

impl LaneMask<'_> {
    /// Where the lane's own rows start among its columns: its buffer, the
    /// keys from its boundary to `have`.
    fn buffer(&self) -> u64 {
        u64::from(self.have - self.folded.min(self.have))
    }

    /// The columns the lane's rows read through the plane: its buffer, then
    /// its own rows.
    fn extent(&self) -> u64 {
        self.buffer() + u64::from(self.rows)
    }
}

/// A fire's mask plane: a row of `stride` columns each masked row reads from
/// its lane's committed boundary on, and a word a row packing its enable
/// flag (bits 0-1: 1 masked) and that boundary (bits 2-31), the key its
/// column 0 is.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Staged {
    pub bytes: Vec<u8>,

    pub words: Vec<u32>,

    pub stride: u32,
}

/// The most a boundary can be in a row's word.
const ORIGIN_MAX: u32 = u32::MAX >> 2;

pub fn stage(lanes: &[LaneMask<'_>]) -> Result<Option<Staged>> {
    if lanes.iter().all(|lane| lane.mask.is_none()) {
        return Ok(None);
    }

    let mut widest = 0u64;
    for (at, lane) in lanes.iter().enumerate() {
        let Some(masking) = lane.mask else {
            continue;
        };
        let extent = lane.extent();
        if lane.folded.min(lane.have) > ORIGIN_MAX {
            return Err(Fault::Ceiling {
                what: "a masked lane's committed boundary in its row's mask word",
                need: u64::from(lane.folded),
                have: u64::from(ORIGIN_MAX),
            });
        }

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
        let Some(masking) = lane.mask else {
            row += lane.rows as usize;
            continue;
        };
        let extent = lane.extent();
        let buffer = lane.buffer();
        let word = 1 | (lane.folded.min(lane.have) << 2);
        for q in 0..lane.rows as usize {
            out.words[row + q] = word;
        }
        let base = row as u64;
        match masking {
            Masking::Extent(mask) => {
                for span in mask.admitted(0..extent) {
                    for column in span {
                        // A row reads no own row after it, whatever its bits
                        // say: the bound is row order, past its buffer.
                        for q in column.saturating_sub(buffer)..u64::from(lane.rows) {
                            keep(&mut out.bytes, stride, base + q, column);
                        }
                    }
                }
            }
            Masking::Rows(masks) => {
                for (q, mask) in masks.iter().enumerate() {
                    let q = q as u64;
                    for span in mask.admitted(0..buffer + q + 1) {
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
    if let Some(byte) = bytes.get_mut(cell as usize) {
        *byte = 1;
    }
}

#[cfg(test)]
mod tests {
    use engine::fire::{Mask, Masking};

    use super::{LaneMask, stage};
    use crate::error::Fault;

    /// A lane with ten keys committed and two buffered, writing two rows:
    /// its plane is four columns (the buffer, then its rows), its words carry
    /// the boundary, and a row reads no own row after it. No column of the
    /// plane is a committed key, so a long committed part grows no plane.
    #[test]
    fn a_mask_speaks_of_the_keys_past_the_boundary_alone() {
        // Hides the first buffered key, admits the rest.
        let mask = Masking::Extent(Mask::new(vec![1, 3], 4));
        let lane = |mask| LaneMask {
            mask: Some(mask),
            have: 12,
            folded: 10,
            rows: 2,
        };
        let staged = stage(&[lane(&mask)]).unwrap().unwrap();
        assert_eq!(staged.stride, 4, "buffer and rows, not the context");
        assert_eq!(staged.words, vec![1 | (10 << 2), 1 | (10 << 2)]);
        assert_eq!(staged.bytes, vec![0, 1, 1, 0, 0, 1, 1, 1]);
        // The same per row, its bits above the diagonal not read.
        let rows = Masking::Rows(vec![Mask::new(vec![1, 3], 4), Mask::new(vec![1, 3], 4)]);
        let staged = stage(&[lane(&rows)]).unwrap().unwrap();
        assert_eq!(staged.bytes, vec![0, 1, 1, 0, 0, 1, 1, 1]);
        // Its extent is its buffer and its rows, not its keys.
        let short = Masking::Extent(Mask::new(vec![0, 3], 3));
        let refused = stage(&[lane(&short)]);
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

    /// A lane with no mask beside a masked one takes no word and no bits;
    /// a fire with no masks stages nothing.
    #[test]
    fn an_unmasked_lane_beside_a_masked_one_takes_no_word() {
        let mask = Masking::Extent(Mask::new(vec![0, 3], 3));
        let staged = stage(&[
            LaneMask {
                mask: None,
                have: 7,
                folded: 3,
                rows: 1,
            },
            LaneMask {
                mask: Some(&mask),
                have: 5,
                folded: 3,
                rows: 1,
            },
        ])
        .unwrap()
        .unwrap();
        assert_eq!(staged.stride, 3);
        assert_eq!(staged.words, vec![0, 1 | (3 << 2)]);
        assert_eq!(staged.bytes, vec![0, 0, 0, 1, 1, 1]);
        let none = stage(&[LaneMask {
            mask: None,
            have: 4,
            folded: 0,
            rows: 1,
        }]);
        assert_eq!(none.unwrap(), None);
    }
}
