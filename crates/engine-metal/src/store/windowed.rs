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
        u32::try_from(self.ring.len())
            .unwrap_or(u32::MAX)
            .saturating_mul(self.page)
    }

    /// The page logical page `n` reads.
    pub fn page_of(&self, n: u32) -> Result<u32> {
        let page = self.page.max(1);
        if !self.base.is_multiple_of(page) {
            return Err(refused(format!(
                "a window buffer based at row {} of {page}-row pages",
                self.base
            )));
        }
        let first = u64::from(n) * u64::from(page);
        let base = u64::from(self.base);
        let held = base + self.buffer.len() as u64 * u64::from(page);
        if first >= base && first < held {
            return Ok(self.buffer
                [usize::try_from((first - base) / u64::from(page)).unwrap_or(usize::MAX)]);
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

    #[test]
    fn a_base_off_the_page_grid_is_refused() {
        let ring = [1u32, 2, 3];
        let lane = WindowedLane {
            page: P,
            ring: &ring,
            base: 5,
            buffer: &[],
            own: &[],
            own_page: 0,
        };
        assert!(lane.page_of(0).is_err());
    }
}
