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

/// A fire's masks, per-row data: each masked row's columns, a byte each from
/// its lane's committed boundary through its own key (the keys it may read
/// past the boundary), packed row after row; and two words a row, the key
/// its column 0 is (the boundary, or [`NONE`] where its lane states no mask)
/// and the byte its columns start at. A row stating none reads no plane, and
/// a fire's kernels are the same whichever of its rows state one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Staged {
    pub bytes: Vec<u8>,

    pub words: Vec<u32>,
}

/// The boundary of a row stating no mask: past every key.
pub const NONE: u32 = kernels_slang::attn::sdpa::MASK_NONE;

/// The most a boundary can be in a row's word: one short of [`NONE`].
const ORIGIN_MAX: u32 = NONE - 1;

/// Every row's entry stating no mask: what a fire none of whose lanes
/// states one binds.
#[must_use]
pub fn none(rows: usize) -> Vec<u32> {
    [NONE, 0].repeat(rows)
}

pub fn stage(lanes: &[LaneMask<'_>]) -> Result<Option<Staged>> {
    if lanes.iter().all(|lane| lane.mask.is_none()) {
        return Ok(None);
    }

    let rows: usize = lanes.iter().map(|lane| lane.rows as usize).sum();
    let mut out = Staged {
        bytes: Vec::new(),
        words: Vec::with_capacity(2 * rows),
    };
    for (at, lane) in lanes.iter().enumerate() {
        let Some(masking) = lane.mask else {
            out.words.extend(none(lane.rows as usize));
            continue;
        };
        let extent = lane.extent();
        let origin = lane.folded.min(lane.have);
        if origin > ORIGIN_MAX {
            return Err(Fault::Ceiling {
                what: "a masked lane's committed boundary in its row's mask word",
                need: u64::from(origin),
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

        let buffer = lane.buffer();
        for q in 0..lane.rows as usize {
            // A row reads no own row after it, whatever its bits say: its
            // columns end at its own key.
            let columns = buffer + q as u64 + 1;
            let offset = u32::try_from(out.bytes.len()).map_err(|_| Fault::Ceiling {
                what: "bytes of one fire's mask plane",
                need: out.bytes.len() as u64,
                have: u64::from(u32::MAX),
            })?;
            out.words.extend([origin, offset]);
            let base = out.bytes.len();
            out.bytes.resize(base + columns as usize, 0);
            let mask = match masking {
                Masking::Extent(mask) => mask,
                Masking::Rows(masks) => &masks[q],
            };
            for span in mask.admitted(0..columns) {
                for column in span {
                    out.bytes[base + column as usize] = 1;
                }
            }
        }
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use engine::fire::{Mask, Masking};

    use super::{LaneMask, NONE, stage};
    use crate::error::Fault;

    /// A lane with ten keys committed and two buffered, writing two rows:
    /// each row's columns are the buffer through its own key (three, then
    /// four), its words carry the boundary and where its columns start, and
    /// a row reads no own row after it. No column of the plane is a
    /// committed key, so a long committed part grows no plane.
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
        assert_eq!(staged.words, vec![10, 0, 10, 3]);
        assert_eq!(staged.bytes, vec![0, 1, 1, 0, 1, 1, 1]);
        // The same per row, its bits above the diagonal not read.
        let rows = Masking::Rows(vec![Mask::new(vec![1, 3], 4), Mask::new(vec![1, 3], 4)]);
        let staged = stage(&[lane(&rows)]).unwrap().unwrap();
        assert_eq!(staged.bytes, vec![0, 1, 1, 0, 1, 1, 1]);
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

    /// A lane with no mask beside a masked one takes the word stating none
    /// and no bytes; a fire with no masks stages nothing.
    #[test]
    fn an_unmasked_lane_beside_a_masked_one_takes_the_word_stating_none() {
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
        assert_eq!(staged.words, vec![NONE, 0, 3, 0]);
        assert_eq!(staged.bytes, vec![1, 1, 1]);
        let none = stage(&[LaneMask {
            mask: None,
            have: 4,
            folded: 0,
            rows: 1,
        }]);
        assert_eq!(none.unwrap(), None);
    }
}
