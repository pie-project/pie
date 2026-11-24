use engine::fire::{Mask, Masking};

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
    /// Attends both ways: no causal bound, and with no mask every held key is kept.
    pub wide: bool,
}

/// A fire's masks: each lane's plane of `rows x (have - folded + rows)`
/// bits from byte `indptr[lane]`, and its word (`origins[lane]`, the
/// kernels' `maybe_mask_origin`): its committed boundary in bits 0-30 and
/// its wide bit in bit 31. A key below the boundary is read under the
/// model's own bounds; past it the plane speaks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Staged {
    pub bits: Vec<u8>,
    pub indptr: Vec<i64>,
    pub origins: Vec<u32>,
}

/// The most a boundary can be in a lane's word, past every key a lane holds.
const ORIGIN_MAX: u32 = u32::MAX >> 1;
const WIDE: u32 = 1 << 31;
/// The word of a lane stating no mask: its boundary past every key, so its
/// rows read no bit.
pub const NONE: u32 = ORIGIN_MAX;

/// The offset a lane's mask ends at, as the kernels read it: 64-bit, so
/// any fire's masks have one.
fn mask_offset(bytes: usize, lane: usize) -> Result<i64> {
    i64::try_from(bytes).map_err(|_| {
        Fault::program(
            "mask",
            format!("the fire's attention masks reach {bytes} bytes by lane {lane}"),
        )
    })
}

fn word(origin: u64, wide: bool, lane: usize) -> Result<u32> {
    let origin = u32::try_from(origin)
        .ok()
        .filter(|&origin| origin < ORIGIN_MAX)
        .ok_or_else(|| {
            Fault::program(
                "mask",
                format!(
                    "lane {lane}'s committed boundary {origin} is past the {ORIGIN_MAX} a \
                     mask word states"
                ),
            )
        })?;
    Ok(origin | if wide { WIDE } else { 0 })
}

/// A fire's masks, where some lane's class reads them (`reads`) or a lane
/// attends both ways: every lane states its entry, a lane with no mask the
/// word [`NONE`].
pub fn stage(lanes: &[LaneMask<'_>], reads: bool) -> Result<Option<Staged>> {
    if !reads && lanes.iter().all(|lane| lane.mask.is_none() && !lane.wide) {
        return Ok(None);
    }
    let mut bits: Vec<u8> = Vec::new();
    let mut indptr: Vec<i64> = Vec::with_capacity(lanes.len() + 1);
    let mut origins: Vec<u32> = Vec::with_capacity(lanes.len());
    indptr.push(0);
    for (at, lane) in lanes.iter().enumerate() {
        let folded = lane.folded.min(lane.have);
        let buffer = u64::from(lane.have - folded);
        // The columns the lane's rows read through the plane: its buffer,
        // then its own rows.
        let extent = buffer + u64::from(lane.rows);
        match lane.mask {
            // A lane with no mask reads every key it holds under the causal
            // bound: all of them committed, whatever it holds.
            None if !lane.wide => origins.push(NONE),
            None => {
                origins.push(word(u64::from(folded), true, at)?);
                let cells = u64::from(lane.rows) * extent;
                let bytes = usize::try_from(cells.div_ceil(8)).unwrap_or(usize::MAX);
                let base = bits.len();
                bits.resize(base + bytes, 0);
                for cell in 0..cells {
                    bits[base + (cell / 8) as usize] |= 1 << (cell % 8);
                }
            }
            Some(masking) => {
                origins.push(word(u64::from(folded), lane.wide, at)?);
                let stated: &[Mask] = masking.masks();
                if let Some(bad) = stated.iter().find(|mask| mask.total < extent) {
                    return Err(Fault::Mask {
                        lane: at as u32,
                        stated: bad.total,
                        extent,
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
                let cells = u64::from(lane.rows) * extent;
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
                    Masking::Extent(mask) => {
                        for span in mask.admitted(0..extent) {
                            for column in span {
                                // A row reads no own row after it: the bound
                                // is row order, past its buffer.
                                let first = if lane.wide {
                                    0
                                } else {
                                    column.saturating_sub(buffer)
                                };
                                for q in first..u64::from(lane.rows) {
                                    set(q * extent + column);
                                }
                            }
                        }
                    }
                    Masking::Rows(rows) => {
                        for (q, mask) in rows.iter().enumerate() {
                            let q = q as u64;
                            let bound = if lane.wide { extent } else { buffer + q + 1 };
                            for span in mask.admitted(0..bound) {
                                for column in span {
                                    set(q * extent + column);
                                }
                            }
                        }
                    }
                }
            }
        }
        indptr.push(mask_offset(bits.len(), at)?);
    }
    Ok(Some(Staged {
        bits,
        indptr,
        origins,
    }))
}
