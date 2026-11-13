//! A working set's fixed record laid into slots of one size, as the engines
//! address a row: `pieces` equal pieces of `atoms_per_piece` atoms each,
//! every piece at one `offset` in its own slot. The record holds state only;
//! the rs buffer is its own list ([`super::RsLayout`]).
//!
//! A row no larger than a slot stays whole: one piece, packed first-fit with
//! other whole rows into a slot. A row larger is cut on its atom axis into
//! the fewest equal pieces a slot holds (the last padded), laid in a window
//! of as many slots of a band whose slots that many divide (its own band
//! where none has room), one piece a slot, at an offset common to the window.

use super::census::FixedRow;
use super::{FixedRun, MIN_PIECE, align_up};

/// A band of `n` slots from record slot `first`, holding cut rows of `n`
/// pieces each, or of a divisor of `n` in windows of its slots.
struct Band {
    first: u32,
    n: u32,
}

/// How [`pack`] lays a record: slots of `slot` bytes, a cut piece no smaller
/// than `least`.
#[derive(Debug, Clone, Copy)]
pub struct Packing {
    pub slot: u64,
    pub align: u64,
    pub least: u64,
}

/// Lays `rows` as `how` says: the slots a working set's record takes, and
/// each row's run (in `rows`' order) when `build`. `None` where a row that
/// may not be cut is larger than a slot, or where a cut piece would hold
/// fewer than `least` bytes.
pub fn pack(rows: &[FixedRow], how: Packing, build: bool) -> Option<(u32, Vec<FixedRun>)> {
    let Packing { slot, align, least } = how;
    if rows.is_empty() {
        return None;
    }
    let mut used: Vec<u64> = Vec::new();
    let mut bands: Vec<Band> = Vec::new();
    let mut runs: Vec<Option<FixedRun>> = Vec::new();
    if build {
        runs.resize(rows.len(), None);
    }
    for (i, row) in rows.iter().enumerate() {
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
                runs[i] = Some(FixedRun {
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
        let top = |first: usize, used: &[u64]| {
            align_up(
                used[first..first + n as usize]
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0),
                align,
            )
        };
        // The first window of `n` slots of a band `n` divides with room.
        let window = bands
            .iter()
            .filter(|b| b.n % n == 0)
            .flat_map(|b| (0..b.n / n).map(move |k| (b.first + k * n) as usize))
            .find(|&first| top(first, &used) + bytes <= slot);
        let first = match window {
            Some(first) => first,
            None => {
                let first = used.len();
                bands.push(Band {
                    first: u32::try_from(first).ok()?,
                    n,
                });
                used.extend(std::iter::repeat_n(0, n as usize));
                first
            }
        };
        let offset = top(first, &used);
        for u in &mut used[first..first + n as usize] {
            *u = offset + bytes;
        }
        if build {
            let first = u32::try_from(first).ok()?;
            runs[i] = Some(FixedRun {
                row: row.id,
                pieces: n,
                atoms_per_piece: atoms,
                offset,
                slots: (first..first + n).collect(),
            });
        }
    }
    Some((
        u32::try_from(used.len()).ok()?,
        runs.into_iter().flatten().collect(),
    ))
}

/// The fewest equal pieces of whole grains `row` cuts into for each to fit
/// `cap`, and the atoms a piece holds; `None` where a grain outgrows the
/// cap or a piece would hold fewer than `least` bytes.
pub fn cut(row: &FixedRow, cap: u64, least: u64) -> Option<(u32, u64)> {
    let atom = row.atom;
    let unit = atom.bytes * atom.grain;
    let per = cap / unit.max(1);
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
