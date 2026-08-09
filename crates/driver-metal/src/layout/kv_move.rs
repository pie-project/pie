//! Moving KV rows between pages of the paged pool: the offset arithmetic.
//!
//! A fork or a compaction asks for token rows to move — `(src page, src
//! row)` to `(dst page, dst row)` — and the engine sends the request across
//! the ABI as a list of cells. The C++ validated and executed them inside
//! `copy_kv_cells`; here the validation and the arithmetic are split out,
//! because they are the part with a history of being wrong and they need no
//! device to be tested.
//!
//! One plan serves every buffer: the same `(src, dst, bytes)` offsets apply
//! to the K pages and the V pages of every full-attention layer, because
//! the pool is page-major `[page, row]` at one stride everywhere. The
//! device half walks the layers and runs each copy with `Region::copy`,
//! whose memmove semantics are not incidental — a compaction slides rows
//! toward the front of the pool, and source and destination overlap.

/// One row move. Field order is the wire order (`KvMoveCell`):
/// destination first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvMoveCell {
    /// The page the row lands in.
    pub dst_page_id: u32,
    /// The row within that page.
    pub dst_token_offset: u32,
    /// The page the row comes from.
    pub src_page_id: u32,
    /// The row within that page.
    pub src_token_offset: u32,
}

/// The paged pool's shape, as the move plan needs it.
#[derive(Clone, Copy, Debug)]
pub struct PoolGrid {
    /// Physical pages in the pool.
    pub total_pages: u32,
    /// Rows per page.
    pub page_size: u32,
    /// Bytes one row occupies (`n_kv_heads * head_dim * act_bytes`).
    pub row_bytes: u64,
}

/// One byte-range copy, valid for every K/V buffer of the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellCopy {
    /// Source offset within the layer buffer.
    pub src_off: u64,
    /// Destination offset within the layer buffer.
    pub dst_off: u64,
    /// The row width.
    pub bytes: u64,
}

/// The validated move list, plus what the elastic pool must be grown to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellMovePlan {
    /// The copies, in request order.
    pub copies: Vec<CellCopy>,
    /// One past the highest page any cell touches, for the elastic ensure;
    /// zero when there are no cells.
    pub pages_touched: u32,
}

/// A cell names a page or row the pool does not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellOutOfRange {
    /// The offending cell's index in the request.
    pub index: usize,
    /// The cell itself.
    pub cell: KvMoveCell,
    /// The pool's page count.
    pub total_pages: u32,
    /// The pool's rows per page.
    pub page_size: u32,
}

impl std::fmt::Display for CellOutOfRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "kv move: cell {} names page {}->{} row {}->{} outside a pool of \
             {} pages x {} rows",
            self.index,
            self.cell.src_page_id,
            self.cell.dst_page_id,
            self.cell.src_token_offset,
            self.cell.dst_token_offset,
            self.total_pages,
            self.page_size
        )
    }
}

impl std::error::Error for CellOutOfRange {}

/// Validate every cell, then lay out the copies.
///
/// Validation runs to completion BEFORE the first offset is produced, for
/// the same reason every rewrite in this crate refuses up front: a move
/// list that fails halfway leaves the pool in a state nobody asked for,
/// and the caller cannot tell which rows moved.
///
/// # Errors
///
/// [`CellOutOfRange`] naming the first offending cell.
pub fn plan_cell_moves(
    cells: &[KvMoveCell],
    grid: PoolGrid,
) -> Result<CellMovePlan, CellOutOfRange> {
    let mut pages_touched = 0u32;
    for (index, cell) in cells.iter().enumerate() {
        if cell.src_page_id >= grid.total_pages
            || cell.dst_page_id >= grid.total_pages
            || cell.src_token_offset >= grid.page_size
            || cell.dst_token_offset >= grid.page_size
        {
            return Err(CellOutOfRange {
                index,
                cell: *cell,
                total_pages: grid.total_pages,
                page_size: grid.page_size,
            });
        }
        pages_touched = pages_touched
            .max(cell.src_page_id + 1)
            .max(cell.dst_page_id + 1);
    }
    let page_bytes = u64::from(grid.page_size) * grid.row_bytes;
    let copies = cells
        .iter()
        .map(|cell| CellCopy {
            src_off: u64::from(cell.src_page_id) * page_bytes
                + u64::from(cell.src_token_offset) * grid.row_bytes,
            dst_off: u64::from(cell.dst_page_id) * page_bytes
                + u64::from(cell.dst_token_offset) * grid.row_bytes,
            bytes: grid.row_bytes,
        })
        .collect();
    Ok(CellMovePlan {
        copies,
        pages_touched,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> PoolGrid {
        PoolGrid {
            total_pages: 8,
            page_size: 32,
            row_bytes: 1024,
        }
    }

    #[test]
    fn offsets_are_page_major_rows_at_one_stride() {
        let plan = plan_cell_moves(
            &[KvMoveCell {
                dst_page_id: 2,
                dst_token_offset: 5,
                src_page_id: 7,
                src_token_offset: 31,
            }],
            grid(),
        )
        .expect("in range");
        assert_eq!(
            plan.copies,
            [CellCopy {
                src_off: 7 * 32 * 1024 + 31 * 1024,
                dst_off: 2 * 32 * 1024 + 5 * 1024,
                bytes: 1024,
            }]
        );
        assert_eq!(plan.pages_touched, 8);
    }

    #[test]
    fn a_compactions_overlapping_slide_is_a_plan_not_a_special_case() {
        // Slide page 1's rows down one row within the same page: source and
        // destination ranges overlap, which Region::copy's memmove
        // semantics carry; the plan just states the offsets.
        let cells: Vec<_> = (1..4)
            .map(|row| KvMoveCell {
                dst_page_id: 1,
                dst_token_offset: row - 1,
                src_page_id: 1,
                src_token_offset: row,
            })
            .collect();
        let plan = plan_cell_moves(&cells, grid()).expect("in range");
        assert_eq!(plan.copies.len(), 3);
        assert_eq!(plan.pages_touched, 2);
    }

    #[test]
    fn every_cell_is_validated_before_any_offset_exists() {
        let cells = [
            KvMoveCell {
                dst_page_id: 0,
                dst_token_offset: 0,
                src_page_id: 1,
                src_token_offset: 0,
            },
            KvMoveCell {
                dst_page_id: 0,
                dst_token_offset: 32, // == page_size: one past the last row
                src_page_id: 1,
                src_token_offset: 0,
            },
        ];
        let err = plan_cell_moves(&cells, grid()).expect_err("row 32 of 32");
        assert_eq!(err.index, 1);
        assert_eq!(err.page_size, 32);
    }

    #[test]
    fn an_empty_request_touches_no_pages() {
        let plan = plan_cell_moves(&[], grid()).expect("empty is fine");
        assert!(plan.copies.is_empty());
        assert_eq!(plan.pages_touched, 0);
    }
}
