//! A windowed lane's page table as ring + buffer (CONTRACT-BC 19, 21): the
//! attention kernels read a window's rows by one page table, so each fire
//! composes a lane's from three kinds of pages, every one laid as a
//! window-buffer slot and named by a page id of the buffer's grain:
//!
//! - the ring pages of the lane's record, for the rows below its buffer's
//!   `base` (all folded): page `n` on ring page `n mod R` (`R` ring pages);
//! - the lane's window-buffer slots, from `base / P` on;
//! - past the buffer, the rows the fire folds itself (they take no buffer
//!   slot): the ring pages again where the lane writes its ring alone and
//!   its fire's pages and the window behind them span no more than the ring
//!   (`direct`), else own-row pages at the activation's top, the first one
//!   filled first with the folded rows its page holds below the fire.
//!
//! A page wholly behind the window of the fire's first row reads the null
//! page. At the fire's end the fold copies the rows it folds from buffer or
//! own-row pages into their ring pages (`Fold`).

/// One lane's inputs to its composed table.
pub struct RingLane<'a> {
    /// The ring pages of the record the lane writes: (page id, address).
    pub ring: &'a [(i32, u64)],
    /// The lane's window-buffer slots from `base`: (page id, address).
    pub buffer: &'a [(i32, u64)],
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

/// Where a page's rows lie for the fold to read them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Already in the ring (a page the fire wrote there itself).
    Ring,
    /// At this address: a buffer slot or an own-row page.
    At(u64),
}

/// A lane's composed table and what its fire's ends do around it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Composed {
    /// The page ids, one a page of `[0, have + rows)`.
    pub table: Vec<i32>,
    /// Whole-page copies before the fire: (to, from) addresses.
    pub fills: Vec<(u64, u64)>,
    /// The fold's row copies, a page of rows at a time: (ring page address,
    /// source, first row in the page, rows).
    pub folds: Vec<(u64, Source, u32, u32)>,
    /// The own-row pages it took.
    pub own: u32,
}

/// The widest window's rows below `row` the fire's first row reads from:
/// a page wholly below `have + 1 - window` reads the null page.
fn first_read_page(have: u32, window: u32, page: u32) -> u32 {
    (have + 1).saturating_sub(window) / page
}

/// Composes `lane`'s table. `page` rows a page, `window` the widest window,
/// `null` the null page's id; `own` hands out own-row pages, (id, address)
/// of the next.
pub fn compose(
    lane: &RingLane<'_>,
    page: u32,
    window: u32,
    null: i32,
    own: &mut dyn FnMut() -> crate::error::Result<(i32, u64)>,
) -> crate::error::Result<Composed> {
    use crate::error::Fault;
    let ring_pages = lane.ring.len() as u32;
    if ring_pages == 0 {
        return Err(Fault::program(
            "serve::ring",
            "a windowed lane whose record holds no ring",
        ));
    }
    if lane.base % page != 0 || lane.base > lane.folded {
        return Err(Fault::program(
            "serve::ring",
            format!(
                "a window buffer from row {} under a fold at {} on pages of {page}",
                lane.base, lane.folded
            ),
        ));
    }
    let end = lane.have + lane.rows;
    let pages = end.div_ceil(page).max(1);
    let first = first_read_page(lane.have, window, page);
    let base_page = lane.base / page;
    let buffered_end = base_page + lane.buffer.len() as u32;
    // Rows past the buffer the fire folds itself: they must reach the fold.
    let past = buffered_end * page;
    if past < end && lane.folded + lane.fold < end {
        return Err(Fault::program(
            "serve::ring",
            format!(
                "rows [{past}, {end}) past the lane's window buffer, which its fire folds \
                 only to {}",
                lane.folded + lane.fold
            ),
        ));
    }
    // The fire's own pages past the buffer go to the ring where the pages
    // read and written span no more than it.
    let direct = lane.alone && pages.saturating_sub(first) <= ring_pages;
    let mut out = Composed {
        table: Vec::with_capacity(pages as usize),
        ..Composed::default()
    };
    let mut sources: Vec<Source> = Vec::with_capacity(pages as usize);
    for n in 0..pages {
        let ring = lane.ring[(n % ring_pages) as usize];
        if n < first.min(base_page) {
            out.table.push(null);
            sources.push(Source::Ring);
        } else if n < base_page {
            out.table.push(ring.0);
            sources.push(Source::Ring);
        } else if n < buffered_end {
            let slot = lane.buffer[(n - base_page) as usize];
            out.table.push(slot.0);
            sources.push(Source::At(slot.1));
        } else if direct {
            out.table.push(ring.0);
            sources.push(Source::Ring);
        } else {
            let (id, at) = own()?;
            out.own += 1;
            // The page's rows below the fire are folded: the ring holds
            // them; the page starts as the ring's.
            if n * page < lane.have {
                out.fills.push((at, ring.1));
            }
            out.table.push(id);
            sources.push(Source::At(at));
        }
    }
    // The fold: rows `[folded, folded + fold)`, only the last ring's worth
    // of them (the rest the ring drops at once), each page's to its ring
    // page.
    let to = (lane.folded + lane.fold).min(end);
    let from = lane.folded.max(to.saturating_sub(ring_pages * page));
    let mut row = from;
    while row < to {
        let n = row / page;
        let stop = ((n + 1) * page).min(to);
        if let Source::At(at) = sources[n as usize] {
            let ring = lane.ring[(n % ring_pages) as usize];
            out.folds.push((ring.1, Source::At(at), row % page, stop - row));
        }
        row = stop;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(pages: u32) -> Vec<(i32, u64)> {
        (0..pages).map(|p| (100 + p as i32, 1000 + u64::from(p))).collect()
    }

    #[test]
    fn a_decode_writes_its_row_into_the_ring_and_folds_nothing_more() {
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
        // Rows [21, 41): pages 1 and 2 read; page 0 null.
        assert_eq!(c.table, vec![0, 101, 102]);
        assert!(c.fills.is_empty() && c.folds.is_empty());
    }

    #[test]
    fn a_long_prefill_takes_own_pages_and_folds_its_last_ring_of_rows() {
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
            Ok((500 + next as i32, 5000 + u64::from(next)))
        };
        let c = compose(&lane, 16, 20, 0, &mut own).unwrap();
        assert_eq!(c.own, 7);
        // Page 0 holds 8 folded rows: filled from ring page 0 first.
        assert_eq!(c.fills, vec![(5001, 1000)]);
        // The last 48 rows [60, 108): pages 3..=6, to ring pages 0, 1, 2, 0.
        let rows: u32 = c.folds.iter().map(|f| f.3).sum();
        assert_eq!(rows, 48);
        assert_eq!(c.folds[0], (1000, Source::At(5004), 12, 4));
    }

    #[test]
    fn a_lagged_lane_reads_its_buffer_and_folds_from_it() {
        let ring = ring(3);
        let buffer = [(7, 70u64), (8, 80)];
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
        assert_eq!(c.folds, vec![(1002, Source::At(70), 4, 4)]);
    }
}
