//! A windowed lane's composed page table and its fold (//! `chunks::MetalRing`). A windowed layer's folded rows lie in the lane's
//! ring, `R` rows laid as pages shaped exactly as a window-buffer slot (`P`
//! rows of every windowed layer), row `r` at ring row `r mod R`; its unfolded
//! rows lie in the lane's window-buffer slots, entry `i` the rows
//! `[base + i*P, base + (i+1)*P)`. The attention kernels read both through
//! one page table, logical page `n` holding rows `[n*P, (n+1)*P)`:
//! - a page a buffer slot holds reads that slot;
//! - every other page reads its ring page, `n mod (R/P)`: rows below the
//!   buffer's base are folded, and rows past the buffer's last slot are the
//!   fire's own rows that take no slot (they fold in the same fire), which
//!   the fire writes straight into their ring rows, where a fold would put
//!   them.
//!
//! `R >= W + P` keeps every page a window reads distinct: a ring page's
//! earlier rows are a whole ring behind, past every window. A fold copies the
//! rows it advances over that lie in buffer slots into their ring rows, at
//! the same row of the page (`R` a multiple of `P`).
//!
//! A fire writes every row before its queries read, so its own rows go
//! straight into the ring only where none of them overwrites a row a query of
//! the fire still reads ([`direct_allowed`]): a solo lane (no beam sibling,
//! no prefix-cache entry sharing the record before its copy) of at most
//! `R - W + 1` rows. Otherwise its own rows lie in activation pages
//! ([`WindowedLane::own`]), each filled first from its ring page where it
//! holds rows folded before the fire ([`WindowedLane::own_fills`]), and the
//! fold copies them into the ring at the fire's end.

use crate::error::{Fault, Result};

/// A lane's windowed rows as one fire reads them.
#[derive(Debug, Clone, Copy)]
pub struct WindowedLane<'a> {
    /// Rows a page holds: the buffer slot's `T`, the ring page's rows.
    pub page: u32,
    /// The ring's pages (`R / P`), page ids at the window slot's stride.
    pub ring: &'a [u32],
    /// The row entry 0 of the buffer holds, a multiple of `page`.
    pub base: u32,
    /// The buffer's slots, page ids at the same stride.
    pub buffer: &'a [u32],
    /// Where the fire's own rows past the buffer lie when they may not go
    /// straight into the ring ([`direct_allowed`]): activation pages at the
    /// same stride, entry `i` the logical page `own_page + i`. Empty: they go
    /// into the ring.
    pub own: &'a [u32],
    /// The logical page `own`'s entry 0 holds.
    pub own_page: u32,
}

/// Whether a fire's own rows may go straight into the lane's ring: a solo
/// lane's (`shared` false: no beam sibling, no copy-on-first-fold pending)
/// fire of at most `R - W + 1` rows, so no row it writes overwrites a row
/// one of its queries reads (the first query reads back `W - 1` rows, the
/// last row lands `rows - 1` past it, and the ring holds `R`).
#[must_use]
pub fn direct_allowed(shared: bool, rows: u32, ring_rows: u32, window: u32) -> bool {
    !shared && u64::from(rows) + u64::from(window) <= u64::from(ring_rows) + 1
}

fn refused(why: String) -> Fault {
    Fault::program("store::windowed", why)
}

impl WindowedLane<'_> {
    /// The ring's rows, `R`.
    #[must_use]
    pub fn ring_rows(&self) -> u32 {
        u32::try_from(self.ring.len()).unwrap_or(u32::MAX).saturating_mul(self.page)
    }

    /// The page logical page `n` reads.
    pub fn page_of(&self, n: u32) -> Result<u32> {
        let page = self.page.max(1);
        if self.base % page != 0 {
            return Err(refused(format!(
                "a window buffer based at row {} of {page}-row pages",
                self.base
            )));
        }
        let first = u64::from(n) * u64::from(page);
        let base = u64::from(self.base);
        let held = base + self.buffer.len() as u64 * u64::from(page);
        if first >= base && first < held {
            return Ok(self.buffer[usize::try_from((first - base) / u64::from(page)).unwrap_or(usize::MAX)]);
        }
        if n >= self.own_page && ((n - self.own_page) as usize) < self.own.len() {
            return Ok(self.own[(n - self.own_page) as usize]);
        }
        if self.ring.is_empty() {
            return Err(refused(format!(
                "page {n} lies past the window buffer's rows [{base}, {held}) and the lane holds no ring"
            )));
        }
        Ok(self.ring[n as usize % self.ring.len()])
    }

    /// The pages logical pages `[0, ceil(end / P))` read: a table for rows
    /// `[0, end)`. Pages a window never reads (behind `end - R`) name their
    /// ring page too; the kernels bound their reads by the window.
    pub fn pages(&self, end: u32) -> Result<Vec<u32>> {
        let count = end.div_ceil(self.page.max(1));
        (0..count).map(|n| self.page_of(n)).collect()
    }

    /// The copies a fold from row `from` to row `to` makes: rows lying in a
    /// buffer slot, each run of one page `(source page, ring page, first row
    /// in the page, rows)`. Rows past the buffer are the fire's own, already
    /// in their ring rows.
    pub fn fold_copies(&self, from: u32, to: u32) -> Result<Vec<(u32, u32, u32, u32)>> {
        let page = self.page.max(1);
        if self.ring.is_empty() && to > from {
            return Err(refused("a fold over a lane that holds no ring".to_string()));
        }
        if u64::from(self.ring_rows()) < u64::from(page) * 2 && to > from {
            return Err(refused(format!(
                "a ring of {} rows over {page}-row pages: it holds a window and a page past it",
                self.ring_rows()
            )));
        }
        // Rows the fold moves: the buffer's, then the activation pages'.
        let held = if self.own.is_empty() {
            u64::from(self.base) + self.buffer.len() as u64 * u64::from(page)
        } else {
            (u64::from(self.own_page) + self.own.len() as u64) * u64::from(page)
        };
        let mut out = Vec::new();
        // Rows below the buffer's base lie in the ring already, unless the
        // fire's own pages hold them.
        let first = if self.own.is_empty() {
            from.max(self.base)
        } else {
            from.max(self.base.min(self.own_page.saturating_mul(page)))
        };
        let mut row = u64::from(first);
        let end = u64::from(to).min(held);
        while row < end {
            let n = u32::try_from(row / u64::from(page)).unwrap_or(u32::MAX);
            let in_page = u32::try_from(row % u64::from(page)).unwrap_or(0);
            let page_end = (u64::from(n) + 1) * u64::from(page);
            let rows = u32::try_from(page_end.min(end) - row).unwrap_or(0);
            let source = self.page_of(n)?;
            let ring = self.ring[n as usize % self.ring.len()];
            if source != ring {
                out.push((source, ring, in_page, rows));
            }
            row += u64::from(rows);
        }
        Ok(out)
    }

    /// The ring pages to copy into the activation pages before the fire:
    /// an activation page whose logical page holds rows folded before the
    /// fire (below `folded`) reads them from its ring page, as the ring page
    /// would have held them. `(ring page, activation page)`.
    #[must_use]
    pub fn own_fills(&self, folded: u32) -> Vec<(u32, u32)> {
        let page = self.page.max(1);
        if self.ring.is_empty() {
            return Vec::new();
        }
        self.own
            .iter()
            .enumerate()
            .filter_map(|(i, &activation)| {
                let n = self.own_page + u32::try_from(i).unwrap_or(u32::MAX);
                (u64::from(n) * u64::from(page) < u64::from(folded))
                    .then(|| (self.ring[n as usize % self.ring.len()], activation))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u32 = 16;

    /// A model of the lane's rows: where each row's bytes lie, (page id, row
    /// in page), as the fire writes them and the fold moves them.
    fn place(lane: &WindowedLane<'_>, row: u32) -> (u32, u32) {
        (lane.page_of(row / P).expect("a page"), row % P)
    }

    /// Every row a window of `w` at position `q` reads lies, through the
    /// composed table, where its bytes are: in its buffer slot while
    /// unfolded, in its ring row once folded, and no two rows of one window
    /// share a place (`R >= W + P`).
    #[test]
    fn a_window_reads_each_row_where_it_lies() {
        let w = 40u32;
        let ring_rows = crate::chunks::MetalRing::rows(w, P);
        assert_eq!(ring_rows, 64, "ceil(40 / 16) * 16 + 16");
        let ring: Vec<u32> = (100..100 + ring_rows / P).collect();
        for folded in [0u32, 16, 33, 48, 130] {
            let base = folded / P * P;
            let buffer: Vec<u32> = (500..503).collect();
            let lane = WindowedLane { page: P, ring: &ring, base, buffer: &buffer, own: &[], own_page: 0 };
            let end = base + 3 * P;
            for q in folded..end {
                let lo = (q + 1).saturating_sub(w);
                let mut seen = std::collections::HashSet::new();
                for kp in lo..=q {
                    let (page, at) = place(&lane, kp);
                    let in_buffer = kp >= base && kp < base + 3 * P;
                    if in_buffer {
                        assert_eq!(page, buffer[((kp - base) / P) as usize], "row {kp} in its slot");
                    } else {
                        assert_eq!(page, ring[((kp / P) % (ring_rows / P)) as usize], "row {kp} in the ring");
                    }
                    assert_eq!(at, kp % P);
                    assert!(seen.insert((page, at)), "rows of one window share no place: {kp} at q {q}");
                }
            }
        }
    }

    /// A fold copies exactly the rows it advances over that lie in buffer
    /// slots, each to its ring row at the same row of the page, split at
    /// page ends; rows past the buffer (the fire's own) are not copied.
    #[test]
    fn a_fold_copies_the_buffered_rows_into_their_ring_rows() {
        let ring: Vec<u32> = (100..104).collect();
        let buffer = [500u32, 501];
        let lane = WindowedLane { page: P, ring: &ring, base: 32, buffer: &buffer, own: &[], own_page: 0 };
        // Rows 35..70: 35..48 in slot 500 (page 2), 48..64 in 501 (page 3),
        // 64..70 the fire's own (page 4, its ring page 100).
        let copies = lane.fold_copies(35, 70).expect("a fold");
        assert_eq!(copies, vec![(500, 102, 3, 13), (501, 103, 0, 16)]);
        assert!(lane.fold_copies(70, 70).expect("an empty fold").is_empty());
        assert_eq!(lane.page_of(4).expect("page 4"), 100, "the fire's own rows read their ring page");
    }

    /// A fire's own rows go straight into the ring only for a solo lane of
    /// at most `R - W + 1` rows; past that, or shared, they lie in
    /// activation pages, the one holding rows folded before the fire filled
    /// from its ring page first, and the fold copies them into the ring.
    #[test]
    fn a_long_or_shared_fire_writes_its_own_rows_beside_the_ring() {
        let (w, ring_rows) = (40u32, crate::chunks::MetalRing::rows(40, P));
        assert!(direct_allowed(false, ring_rows - w + 1, ring_rows, w));
        assert!(!direct_allowed(false, ring_rows - w + 2, ring_rows, w), "a row past R - W + 1");
        assert!(!direct_allowed(true, 1, ring_rows, w), "a shared record");
        // Folded through row 37 (page 2 part folded), no buffer: a 30-row
        // fire's own rows 37..67 lie in activation pages for pages 2..4.
        let ring: Vec<u32> = (100..104).collect();
        let own = [900u32, 901, 902];
        let lane = WindowedLane { page: P, ring: &ring, base: 32, buffer: &[], own: &own, own_page: 2 };
        assert_eq!(lane.page_of(2).expect("page 2"), 900);
        assert_eq!(lane.page_of(1).expect("page 1"), 101, "a page wholly folded reads its ring page");
        assert_eq!(lane.own_fills(37), vec![(102, 900)], "page 2 holds rows folded before the fire");
        assert_eq!(
            lane.fold_copies(37, 67).expect("a fold"),
            vec![(900, 102, 5, 11), (901, 103, 0, 16), (902, 100, 0, 3)]
        );
    }

    /// A long fire past `R - W + 1` rows whose base is the folded index
    /// after it: its own pages hold every row it writes below the base too,
    /// and the fold copies them all into the ring.
    #[test]
    fn a_long_fire_folds_its_own_rows_below_the_base() {
        let ring: Vec<u32> = (100..104).collect();
        let own = [900u32, 901, 902, 903];
        // have 2, 62 rows, base 64: own pages for logical pages 0..4.
        let lane = WindowedLane { page: P, ring: &ring, base: 64, buffer: &[], own: &own, own_page: 0 };
        assert_eq!(lane.page_of(1).expect("page 1"), 901);
        assert_eq!(lane.own_fills(2), vec![(100, 900)], "page 0 holds rows folded before the fire");
        assert_eq!(
            lane.fold_copies(2, 64).expect("a fold"),
            vec![(900, 100, 2, 14), (901, 101, 0, 16), (902, 102, 0, 16), (903, 103, 0, 16)]
        );
    }

    #[test]
    fn a_base_off_the_page_grid_is_refused() {
        let ring = [1u32, 2, 3];
        let lane = WindowedLane { page: P, ring: &ring, base: 5, buffer: &[], own: &[], own_page: 0 };
        assert!(lane.page_of(0).is_err());
    }
}
