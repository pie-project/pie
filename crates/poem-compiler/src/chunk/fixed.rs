//! A working set's fixed record laid into pieces of one slot size: rows in
//! order, each whole where it fits, a row past the piece's room cut on its
//! atoms into the next pieces.

use super::census::FixedRow;
use super::{MIN_PIECE, Piece, align_up};

/// Lays `rows` into pieces of `slot` bytes: the pieces a working set takes,
/// and where each row's spans land when `build`. `None` where a row that
/// may not be cut is larger than `slot`.
pub fn pack(rows: &[FixedRow], slot: u64, align: u64, build: bool) -> Option<(u32, Vec<Piece>)> {
    let mut pieces = Vec::new();
    let mut piece = 0u32;
    let mut used = 0u64;
    let mut any = false;
    for row in rows {
        any = true;
        let start = align_up(used, align);
        if start + row.bytes <= slot {
            if build {
                pieces.push(Piece {
                    row: row.id,
                    span: 0..row.atom.extent,
                    piece,
                    offset: start,
                });
            }
            used = start + row.bytes;
            continue;
        }
        if !cuttable(row) {
            if row.bytes > slot {
                return None;
            }
            piece += 1;
            if build {
                pieces.push(Piece {
                    row: row.id,
                    span: 0..row.atom.extent,
                    piece,
                    offset: 0,
                });
            }
            used = row.bytes;
            continue;
        }
        let unit = row.atom.bytes * row.atom.grain;
        let mut at = 0u64;
        loop {
            let start = align_up(used, align);
            let room = slot.saturating_sub(start);
            let take = ((room / unit) * row.atom.grain).min(row.atom.extent - at);
            if take == 0 {
                if used == 0 {
                    return None;
                }
                piece += 1;
                used = 0;
                continue;
            }
            if build {
                pieces.push(Piece {
                    row: row.id,
                    span: at..at + take,
                    piece,
                    offset: start,
                });
            }
            used = start + take * row.atom.bytes;
            at += take;
            if at == row.atom.extent {
                break;
            }
            piece += 1;
            used = 0;
        }
    }
    any.then_some((piece + 1, pieces))
}

/// A row a piece may cut: one its readers cut on an axis, no smaller than a
/// piece may be.
#[must_use]
pub fn cuttable(row: &FixedRow) -> bool {
    row.atom.extent > 1 && row.bytes >= MIN_PIECE
}
