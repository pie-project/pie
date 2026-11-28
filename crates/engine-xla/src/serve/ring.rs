//! A windowed lane's page table as ring + buffer: the
//! attention kernels read a window's rows by one page table, so each fire
//! composes a lane's from three kinds of pages, every one laid as a
//! window-buffer slot and named by a grain index of the window buffer's
//! grain (engine-cuda `serve/ring.rs`):
//!
//! - the ring pages of the lane's record, for the rows below its buffer's
//!   `base` (all folded): page `n` on ring page `n mod R`;
//! - the lane's window-buffer slots, from `base / T` on;
//! - past the buffer, the rows the fire folds itself (they take no buffer
//!   slot): the ring pages again where the lane writes its ring alone and
//!   the pages its fire reads and writes span no more than the ring
//!   (`direct`), else own-row pages (`Pools::own_grains`), the first filled
//!   first with the folded rows its page holds below the fire.
//!
//! A page wholly behind the window of the fire's first row reads the null
//! page. A buffer slot whose page holds folded rows below the lane's first
//! buffered row is filled from its ring page by the fire that first writes
//! into it (the lane buffered nothing before), whole: its other rows are
//! not yet written. After the fire the fold copies, whole, each page holding a row
//! the next fire's window still reads (`[folded' - W, folded')`, at most
//! `ceil(W / T) + 1 = R` pages, each on a ring page of its own) from its
//! buffer or own-row page into its ring page: the rows a page holds below
//! the fold are the ring's own already, and those past it lie where no
//! read reaches before a later fold writes them.

use crate::error::{Fault, Result};

/// One lane's inputs to its composed table, pages as grain indices.
pub struct RingLane<'a> {
    /// The ring pages of the record the lane writes.
    pub ring: &'a [u32],
    /// The lane's window-buffer slots from `base`.
    pub buffer: &'a [u32],
    /// The row the buffer's first slot holds.
    pub base: u32,
    /// The rows before the fire, and the fire's.
    pub have: u32,
    pub rows: u32,
    /// The rows the fire folds: `[folded, folded + fold)`.
    pub folded: u32,
    pub fold: u32,
    /// Whether the lane writes its ring alone this fire (no other lane of
    /// the fire names its record).
    pub alone: bool,
}

/// A lane's composed table and the page copies around its fire.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Composed {
    /// The page ids, one a page of `[0, have + rows)`.
    pub table: Vec<u32>,
    /// Whole-page copies before the fire: (to, from).
    pub fills: Vec<(u32, u32)>,
    /// Whole-page copies after it, each into its ring page: (to, from).
    pub folds: Vec<(u32, u32)>,
    /// The own-row pages it took.
    pub own: u32,
}

/// Composes `lane`'s table. `page` rows a page, `window` the widest
/// window, `null` the null page; `own` hands out own-row pages.
pub fn compose(
    lane: &RingLane<'_>,
    page: u32,
    window: u32,
    null: u32,
    own: &mut dyn FnMut() -> Result<u32>,
) -> Result<Composed> {
    let refuse = |why: String| Fault::Program {
        at: "serve::ring",
        why,
    };
    let ring_pages = lane.ring.len() as u32;
    if ring_pages == 0 {
        return Err(refuse("a windowed lane whose record holds no ring".into()));
    }
    let page = page.max(1);
    let end = lane.have + lane.rows;
    if !lane.base.is_multiple_of(page) || lane.base > (lane.folded + lane.fold).max(lane.folded) {
        return Err(refuse(format!(
            "a window buffer from row {} under a fold of [{}, {}) on pages of {page}",
            lane.base,
            lane.folded,
            lane.folded + lane.fold
        )));
    }
    let pages = end.div_ceil(page).max(1);
    // the window of the fire's first row: pages wholly behind it read null
    let first = (lane.have + 1).saturating_sub(window) / page;
    let base_page = lane.base / page;
    let buffered_end = base_page + lane.buffer.len() as u32;
    let past = buffered_end * page;
    if past < end && lane.folded + lane.fold < end {
        return Err(refuse(format!(
            "rows [{past}, {end}) past the lane's window buffer, which its fire folds only to {}",
            lane.folded + lane.fold
        )));
    }
    let direct = lane.alone && pages.saturating_sub(first) <= ring_pages;
    let mut out = Composed {
        table: Vec::with_capacity(pages as usize),
        ..Composed::default()
    };
    // each page's source for the fold: `None` the ring itself
    let mut sources: Vec<Option<u32>> = Vec::with_capacity(pages as usize);
    for n in 0..pages {
        let ring = lane.ring[(n % ring_pages) as usize];
        // a page of rows all held before the fire, below the buffer: the
        // ring's (or behind the window: the null page)
        let held = (n + 1) * page <= lane.have;
        if n < base_page && held {
            out.table.push(if n < first { null } else { ring });
            sources.push(None);
        } else if n >= base_page && n < buffered_end {
            let slot = lane.buffer[(n - base_page) as usize];
            // The fire that first writes into a slot whose page holds folded
            // rows below it (a lane buffering none before: base below the
            // folded index, its rows `[base, folded)` the ring's alone) fills
            // it whole from its ring page first; later fires find them there.
            if n * page < lane.have && lane.have == lane.folded {
                out.fills.push((slot, ring));
            }
            out.table.push(slot);
            sources.push(Some(slot));
        } else if direct {
            // a page of the fire's own rows, folded by it: its ring page
            out.table.push(ring);
            sources.push(None);
        } else {
            let at = own()?;
            out.own += 1;
            // the page's rows below the fire are folded: the ring's
            if n * page < lane.have {
                out.fills.push((at, ring));
            }
            out.table.push(at);
            sources.push(Some(at));
        }
    }
    let to = (lane.folded + lane.fold).min(end);
    let from = lane.folded.max(to.saturating_sub(window));
    if from < to {
        for n in from / page..=(to - 1) / page {
            if let Some(Some(at)) = sources.get(n as usize) {
                out.folds.push((lane.ring[(n % ring_pages) as usize], *at));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(pages: u32) -> Vec<u32> {
        (0..pages).map(|p| 100 + p).collect()
    }

    #[test]
    fn a_decode_writes_its_row_into_the_ring_and_folds_nothing() {
        let ring = ring(3);
        let lane = RingLane {
            ring: &ring,
            buffer: &[],
            base: 32,
            have: 40,
            rows: 1,
            folded: 40,
            fold: 1,
            alone: true,
        };
        let mut own = || panic!("no own page");
        let c = compose(&lane, 16, 20, 0, &mut own).unwrap();
        // rows [21, 41): pages 1 and 2 read, page 0 null
        assert_eq!(c.table, vec![0, 101, 102]);
        assert!(c.fills.is_empty() && c.folds.is_empty());
    }

    #[test]
    fn a_long_prefill_takes_own_pages_and_folds_the_pages_its_window_reads_next() {
        let ring = ring(3);
        let lane = RingLane {
            ring: &ring,
            buffer: &[],
            base: 0,
            have: 8,
            rows: 100,
            folded: 8,
            fold: 100,
            alone: true,
        };
        let mut next = 0u32;
        let mut own = || {
            next += 1;
            Ok(500 + next)
        };
        let c = compose(&lane, 16, 20, 0, &mut own).unwrap();
        assert_eq!(c.own, 7);
        // page 0 holds 8 folded rows: filled from ring page 0 first
        assert_eq!(c.fills, vec![(501, 100)]);
        // the next window [88, 108): pages 5 and 6, to ring pages 2 and 0
        assert_eq!(c.folds, vec![(102, 506), (100, 507)]);
    }

    #[test]
    fn a_lagged_lane_reads_its_buffer_and_folds_from_it() {
        let ring = ring(3);
        let buffer = [7, 8];
        let lane = RingLane {
            ring: &ring,
            buffer: &buffer,
            base: 32,
            have: 50,
            rows: 2,
            folded: 36,
            fold: 4,
            alone: false,
        };
        let mut own = || panic!("no own page");
        let c = compose(&lane, 16, 20, 0, &mut own).unwrap();
        assert_eq!(c.table, vec![0, 101, 7, 8]);
        // rows [36, 40): page 2, the buffer's first slot, to ring page 2
        assert_eq!(c.folds, vec![(102, 7)]);
    }

    #[test]
    fn a_lane_buffering_nothing_starts_its_buffer_past_the_rows_it_folds() {
        // a prompt folding every row: base the folded index after it, no slot
        let ring = ring(3);
        let lane = RingLane {
            ring: &ring,
            buffer: &[],
            base: 48,
            have: 20,
            rows: 30,
            folded: 20,
            fold: 30,
            alone: true,
        };
        let mut next = 0u32;
        let mut own = || {
            next += 1;
            Ok(500 + next)
        };
        let c = compose(&lane, 16, 20, 0, &mut own).unwrap();
        // pages 0..=3 read [0, 50): the fire's span (pages 0 to 3) is past
        // the ring's three pages, so pages 1, 2 and 3 (its rows) are own pages,
        // page 1 filled from the ring first (rows 16..20 held)
        assert_eq!(c.table, vec![100, 501, 502, 503]);
        assert_eq!(c.fills, vec![(501, 101)]);
        // the next window [30, 50): pages 1, 2, 3
        assert_eq!(c.folds, vec![(101, 501), (102, 502), (100, 503)]);
    }

    #[test]
    fn a_lane_starting_its_buffer_mid_page_fills_its_first_slot_from_the_ring() {
        // a 37-row prompt folded itself; the decode buffers its row
        let ring = ring(3);
        let buffer = [9];
        let lane = RingLane {
            ring: &ring,
            buffer: &buffer,
            base: 32,
            have: 37,
            rows: 1,
            folded: 37,
            fold: 0,
            alone: true,
        };
        let mut own = || panic!("no own page");
        let c = compose(&lane, 16, 20, 0, &mut own).unwrap();
        assert_eq!(c.table, vec![0, 101, 9]);
        // rows [32, 37) the ring's: the slot takes its ring page first
        assert_eq!(c.fills, vec![(9, 102)]);
        assert!(c.folds.is_empty());
        // the next fire carries the buffered row: the slot holds them all
        let lane = RingLane {
            have: 38,
            folded: 37,
            fold: 2,
            ..lane
        };
        let c = compose(&lane, 16, 20, 0, &mut own).unwrap();
        assert!(c.fills.is_empty());
        assert_eq!(c.folds, vec![(102, 9)]);
    }

    #[test]
    fn the_pages_a_fold_copies_lie_on_ring_pages_of_their_own() {
        // every fold of a wide prefill: its pages' ring pages are distinct
        let ring = ring(3);
        for have in 0..40 {
            for rows in 1..90 {
                let lane = RingLane {
                    ring: &ring,
                    buffer: &[],
                    base: 0,
                    have,
                    rows,
                    folded: have,
                    fold: rows,
                    alone: true,
                };
                let mut next = 0u32;
                let mut own = || {
                    next += 1;
                    Ok(500 + next)
                };
                let c = compose(&lane, 16, 20, 0, &mut own).unwrap();
                let mut to: Vec<u32> = c.folds.iter().map(|f| f.0).collect();
                to.sort_unstable();
                to.dedup();
                assert_eq!(
                    to.len(),
                    c.folds.len(),
                    "have {have} rows {rows}: {:?}",
                    c.folds
                );
            }
        }
    }
}
