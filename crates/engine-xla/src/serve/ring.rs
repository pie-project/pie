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
    if lane.base % page != 0 || lane.base > (lane.folded + lane.fold).max(lane.folded) {
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

