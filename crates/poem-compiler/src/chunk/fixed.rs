//! A working set's fixed record laid into slots of one size, as the engines
//! address a row: `pieces` equal pieces of `atoms_per_piece` atoms each,
//! every piece at one `offset` in its own slot.
//!
//! A row no larger than a slot stays whole: one piece, packed first-fit with
//! other whole rows into a slot. A row larger is cut on its atom axis into
//! the fewest equal pieces a slot holds (the last padded); rows cut into as
//! many pieces share a band of that many slots, one piece a slot, each row
//! at its own offset common to the band.

use super::census::FixedRow;
use super::{FixedRun, MIN_PIECE, align_up};

/// A band of `n` slots from record slot `first`, holding cut rows of `n`
/// pieces each.
struct Band {
    first: u32,
    n: u32,
}

/// Lays `rows` into slots of `slot` bytes: the slots a working set's record
/// takes, and each row's run when `build`. `None` where a row that may not be
/// cut is larger than `slot`, or where a cut piece would hold fewer than
/// `least` bytes.
pub fn pack(
    rows: &[FixedRow],
    slot: u64,
    align: u64,
    least: u64,
    build: bool,
) -> Option<(u32, Vec<FixedRun>)> {
    if rows.is_empty() {
        return None;
    }
    let mut used: Vec<u64> = Vec::new();
    let mut bands: Vec<Band> = Vec::new();
    let mut runs = Vec::new();
    for row in rows {
        if row.bytes <= slot || !cuttable(row) {
            if row.bytes > slot {
                return None;
            }
            let at = used
                .iter()
                .position(|&u| align_up(u, align) + row.bytes <= slot);
            let at = at.unwrap_or_else(|| {
                used.push(0);
                used.len() - 1
            });
            let offset = align_up(used[at], align);
            used[at] = offset + row.bytes;
            if build {
                runs.push(FixedRun {
                    row: row.id,
                    pieces: 1,
                    atoms_per_piece: row.atom.extent,
                    offset,
                    slots: vec![u32::try_from(at).ok()?],
                });
            }
            continue;
        }
        let (n, atoms) = cut(row, slot, least)?;
        let bytes = atoms * row.atom.bytes;
        let top = |band: &Band, used: &[u64]| {
            let first = band.first as usize;
            align_up(
                used[first..first + band.n as usize]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0),
                align,
            )
        };
        let band = match bands
            .iter()
            .position(|b| b.n == n && top(b, &used) + bytes <= slot)
        {
            Some(band) => band,
            None => {
                bands.push(Band {
                    first: u32::try_from(used.len()).ok()?,
                    n,
                });
                used.extend(std::iter::repeat_n(0, n as usize));
                bands.len() - 1
            }
        };
        let band = &bands[band];
        let offset = top(band, &used);
        let first = band.first as usize;
        for u in &mut used[first..first + n as usize] {
            *u = offset + bytes;
        }
        if build {
            runs.push(FixedRun {
                row: row.id,
                pieces: n,
                atoms_per_piece: atoms,
                offset,
                slots: (band.first..band.first + n).collect(),
            });
        }
    }
    Some((u32::try_from(used.len()).ok()?, runs))
}

/// The fewest equal pieces of whole grains `row` cuts into for each to fit
/// `slot`, and the atoms a piece holds; `None` where a grain outgrows the
/// slot or a piece would hold fewer than `least` bytes.
pub fn cut(row: &FixedRow, slot: u64, least: u64) -> Option<(u32, u64)> {
    let atom = row.atom;
    let unit = atom.bytes * atom.grain;
    let per = slot / unit.max(1);
    if per == 0 {
        return None;
    }
    let units = atom.extent.div_ceil(atom.grain);
    let n = units.div_ceil(per);
    let atoms = units.div_ceil(n) * atom.grain;
    if atoms * atom.bytes < least {
        return None;
    }
    Some((u32::try_from(n).ok()?, atoms))
}

/// A row a piece may cut: one its readers cut on an axis, no smaller than a
/// piece may be.
#[must_use]
pub fn cuttable(row: &FixedRow) -> bool {
    row.atom.extent > 1 && row.bytes >= MIN_PIECE
}
