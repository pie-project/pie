//! A working set's fixed record laid into slots of one size, as the engines
//! address a row: `pieces` equal pieces of `atoms_per_piece` atoms each,
//! every piece at one `offset` in its own slot.
//!
//! The rs rows go first, together: an engine reads a lane's rs buffer
//! through one page table, so every rs row lies in the same record slots,
//! one after another. They stay whole in slot 0 unless the plan lets them
//! be cut (they outgrow every chunk the device would take); then every one
//! is cut alike on its tokens into the fewest equal pieces whose share of
//! each fits a slot, piece `p` of every rs row in slot `p`.
//!
//! Then the other rows. A row no larger than a slot stays whole: one piece,
//! packed first-fit with other whole rows into a slot. A row larger is cut
//! on its atom axis into the fewest equal pieces a slot holds (the last
//! padded); rows cut into as many pieces share a band of that many slots,
//! one piece a slot, each row at its own offset common to the band.

use super::census::FixedRow;
use super::{FixedRun, MIN_PIECE, RowId, align_up};

/// A band of `n` slots from record slot `first`, holding cut rows of `n`
/// pieces each.
struct Band {
    first: u32,
    n: u32,
}

/// Whether `row` is one of the rs buffer's.
#[must_use]
pub fn is_rs(row: &FixedRow) -> bool {
    matches!(row.id, RowId::Rs(_))
}

/// Lays `rows` into slots of `slot` bytes: the slots a working set's record
/// takes, and each row's run (in `rows`' order) when `build`. `None` where a
/// row that may not be cut is larger than `slot` (the rs rows together,
/// unless `rs_cut`), or where a cut piece would hold fewer than `least`
/// bytes.
pub fn pack(
    rows: &[FixedRow],
    slot: u64,
    align: u64,
    least: u64,
    rs_cut: bool,
    build: bool,
) -> Option<(u32, Vec<FixedRun>)> {
    if rows.is_empty() {
        return None;
    }
    let mut used: Vec<u64> = Vec::new();
    let mut bands: Vec<Band> = Vec::new();
    let mut runs: Vec<Option<FixedRun>> = Vec::new();
    if build {
        runs.resize(rows.len(), None);
    }
    let rs: Vec<&FixedRow> = rows.iter().filter(|r| is_rs(r)).collect();
    if !rs.is_empty() {
        let (n, atoms) = rs_pieces(&rs, slot, align, rs_cut)?;
        let mut at = 0;
        for (i, row) in rows.iter().enumerate().filter(|(_, r)| is_rs(r)) {
            let offset = align_up(at, align);
            at = offset + atoms * row.atom.bytes;
            if build {
                runs[i] = Some(FixedRun {
                    row: row.id,
                    pieces: n,
                    atoms_per_piece: atoms,
                    offset,
                    slots: (0..n).collect(),
                });
            }
        }
        used.extend(std::iter::repeat_n(at, n as usize));
        if n > 1 {
            bands.push(Band { first: 0, n });
        }
    }
    for (i, row) in rows.iter().enumerate() {
        if is_rs(row) {
            continue;
        }
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
            runs[i] = Some(FixedRun {
                row: row.id,
                pieces: n,
                atoms_per_piece: atoms,
                offset,
                slots: (band.first..band.first + n).collect(),
            });
        }
    }
    Some((
        u32::try_from(used.len()).ok()?,
        runs.into_iter().flatten().collect(),
    ))
}

/// The bytes a slot holds of the `rs` rows, one after another, at `atoms`
/// tokens each.
fn rs_bytes(rs: &[&FixedRow], atoms: u64, align: u64) -> u64 {
    rs.iter()
        .fold(0, |at, r| align_up(at, align) + atoms * r.atom.bytes)
}

/// The pieces every rs row is cut into alike and the tokens a piece holds:
/// whole where they fit `slot` together; else, where `cut`, the fewest equal
/// pieces whose share of every row fits. A share may hold less than a cut
/// state's least piece: the rs rows are a lane's buffer, read a token at a
/// time.
fn rs_pieces(rs: &[&FixedRow], slot: u64, align: u64, cut: bool) -> Option<(u32, u64)> {
    let extent = rs.iter().map(|r| r.atom.extent).max().unwrap_or(1).max(1);
    if rs_bytes(rs, extent, align) <= slot {
        return Some((1, extent));
    }
    if !cut {
        return None;
    }
    let atoms = (1..extent)
        .rev()
        .find(|&atoms| rs_bytes(rs, atoms, align) <= slot)?;
    // The fewest pieces of `atoms`, then as even as they cut.
    let atoms = extent.div_ceil(extent.div_ceil(atoms));
    let n = extent.div_ceil(atoms);
    Some((u32::try_from(n).ok()?, atoms))
}

/// The slots of one piece the rs rows cut `n` ways take: their share of
/// each, aligned; `None` where they are not cut `n` ways.
#[must_use]
pub fn rs_piece(rs: &[&FixedRow], n: u64, align: u64) -> Option<u64> {
    let extent = rs.iter().map(|r| r.atom.extent).max()?.max(1);
    let atoms = extent.div_ceil(n.max(1));
    (extent.div_ceil(atoms) == n).then(|| align_up(rs_bytes(rs, atoms, align), align))
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
