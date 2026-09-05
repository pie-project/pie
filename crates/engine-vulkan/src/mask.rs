use engine::fire::{Mask, Masking};

use crate::error::{Fault, Result};

#[must_use]
pub fn from_dense(cells: &[bool], stride: usize) -> Masking {
    let rows = cells.len().checked_div(stride).unwrap_or(0);
    Masking::Rows(
        (0..rows)
            .map(|row| {
                let mut runs: Vec<u32> = Vec::new();

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

#[derive(Debug, Clone, Copy)]
pub struct LaneMask<'a> {
    pub mask: Option<&'a Masking>,

    pub have: u32,

    pub rows: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Staged {
    pub bytes: Vec<u8>,

    pub enabled: Vec<u8>,

    pub stride: u32,
}

pub fn stage(lanes: &[LaneMask<'_>]) -> Result<Option<Staged>> {
    if lanes.iter().all(|lane| lane.mask.is_none()) {
        return Ok(None);
    }

    let mut widest = 0u64;
    for (at, lane) in lanes.iter().enumerate() {
        let kv = u64::from(lane.have) + u64::from(lane.rows);
        let Some(masking) = lane.mask else {
            continue;
        };

        for mask in masking.masks() {
            if mask.total < kv {
                return Err(Fault::Mask {
                    lane: at as u32,
                    stated: mask.total,
                    extent: kv,
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

fn keep(bytes: &mut [u8], stride: u32, row: u64, key: u64) {
    let cell = row * u64::from(stride) + key;
    if let Some(word) = bytes.get_mut(cell as usize) {
        *word = 1;
    }
}
