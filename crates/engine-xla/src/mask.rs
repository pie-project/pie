use engine::fire::Masking;

use crate::error::{Fault, Result};

#[derive(Debug, Clone, Copy)]
pub struct LaneMask<'a> {
    pub mask: Option<&'a Masking>,

    pub have: u32,

    /// The lane's set's committed boundary (`KvDelta::folded`, at most
    /// `have`): mask column `c` is key `folded + c`; keys below it are read
    /// under the causal bound and no mask bit.
    pub folded: u32,

    pub rows: u32,

    /// The class attends both ways: a stated mask's rows are unclipped and
    /// flagged 2. With none, the lane stages nothing: its non-causal ops
    /// read every key it holds (the plan's `kv_len`), its causal ops stay
    /// causal.
    pub wide: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Staged {
    pub bytes: Vec<u8>,

    pub enabled: Vec<u8>,

    pub stride: u32,
}

/// The fire's byte plane: a row a lane row, a byte a key position (`stride`
/// the load's: `width`, a power of two of 256 and more past every key a lane
/// can hold, so the plane's shape is the load's and not the fire's). A lane's
/// mask speaks of its keys from `folded` on: its committed keys (`0..folded`)
/// are admitted under the causal bound, and mask column `c` lands at key
/// `folded + c`. Among the lane's own rows the bound is the row order.
pub fn stage(lanes: &[LaneMask<'_>], width: u32) -> Result<Option<Staged>> {
    if lanes.iter().all(|lane| lane.mask.is_none()) {
        return Ok(None);
    }

    let mut widest = 0u64;
    for (at, lane) in lanes.iter().enumerate() {
        let kv = u64::from(lane.have) + u64::from(lane.rows);
        let Some(masking) = lane.mask else {
            continue;
        };
        // The mask's extent: the buffer past the boundary, then the rows.
        let extent = kv - u64::from(lane.folded.min(lane.have));
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
        widest = widest.max(kv);
    }
    if widest > u64::from(width) {
        return Err(Fault::Ceiling {
            what: "key positions in one mask row",
            need: widest,
            have: u64::from(width),
        });
    }
    let stride = width;

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

    let mut row = 0usize;
    for lane in lanes {
        let folded = u64::from(lane.folded.min(lane.have));
        let kv = u64::from(lane.have) + u64::from(lane.rows);
        // The mask's columns: `0..extent`, column `c` key `folded + c`.
        let extent = kv - folded;
        let flag = if lane.wide { 2 } else { 1 };
        let Some(masking) = lane.mask else {
            row += lane.rows as usize;
            continue;
        };
        for q in 0..lane.rows as usize {
            out.enabled[row + q] = flag;
        }
        let base = row as u64;
        // Every row reads the committed keys under the causal bound alone.
        for q in 0..u64::from(lane.rows) {
            let at = ((base + q) * u64::from(stride)) as usize;
            if let Some(cells) = out.bytes.get_mut(at..at + folded as usize) {
                cells.fill(1);
            }
        }
        match masking {
            Masking::Extent(mask) => {
                let mut at_position = 0u64;
                for (index, &run) in mask.runs.iter().enumerate() {
                    let end = at_position.saturating_add(u64::from(run)).min(extent);
                    if index % 2 == 1 {
                        for key in (folded + at_position)..(folded + end) {
                            let first = if lane.wide {
                                0
                            } else {
                                key.saturating_sub(u64::from(lane.have))
                            };
                            for q in first..u64::from(lane.rows) {
                                keep(&mut out.bytes, stride, base + q, key);
                            }
                        }
                    }
                    if end == extent {
                        break;
                    }
                    at_position = end;
                }
            }
            Masking::Rows(masks) => {
                for (q, mask) in masks.iter().enumerate() {
                    let q = q as u64;

                    let bound = if lane.wide {
                        kv
                    } else {
                        u64::from(lane.have) + q
                    };
                    let mut at_position = 0u64;
                    for (index, &run) in mask.runs.iter().enumerate() {
                        let end = at_position.saturating_add(u64::from(run)).min(extent);
                        if index % 2 == 1 {
                            for key in (folded + at_position)..(folded + end).min(bound + 1) {
                                keep(&mut out.bytes, stride, base + q, key);
                            }
                        }
                        if end == extent {
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

fn keep(bytes: &mut [u8], stride: u32, row: u64, key: u64) {
    let cell = row * u64::from(stride) + key;
    if let Some(word) = bytes.get_mut(cell as usize) {
        *word = 1;
    }
}

