//! A windowed lane's page table as ring + buffer: the
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
    /// The rows after which the fire lands a snapshot of the lane's rings
    /// (`Lane::rs_snapshots`, a fire's rows from its first), each onto the
    /// ring pages of the snapshot's record.
    pub snapshots: &'a [(u32, &'a [(i32, u64)])],
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
    /// Its own pages written on the ring directly.
    pub direct: u32,
    /// Ring pages it reads that its fold then writes.
    pub read_then_folded: u32,
    /// The snapshots' row copies, before the fold: (snapshot ring page
    /// address, source address, first row in the page, rows).
    pub snaps: Vec<(u64, u64, u32, u32)>,
    /// Row copies before the fire onto buffer slots from the ring: the
    /// folded rows `[base, folded)` a buffer's first slot holds, which an
    /// earlier fire folded without a slot (its own rows past its ring): (slot
    /// address, ring page address, first row in the page, rows).
    pub refills: Vec<(u64, u64, u32, u32)>,
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
    // The buffer starts at or below the fold the fire leaves: past the rows
    // a fire folds itself where it buffers none earlier.
    if !lane.base.is_multiple_of(page) || lane.base > lane.folded + lane.fold {
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
    // A fire that lands a snapshot keeps its ring as it found it until its
    // fold, so the snapshot reads every row where it lies.
    let direct =
        lane.alone && lane.snapshots.is_empty() && pages.saturating_sub(first) <= ring_pages;
    let mut out = Composed {
        table: Vec::with_capacity(pages as usize),
        ..Composed::default()
    };
    let mut sources: Vec<Source> = Vec::with_capacity(pages as usize);
    // Rows of the fire's own below its buffer's base take no slot (the
    // runtime starts the buffer past the rows a fire folds itself): they
    // must fold here, as the rows past the buffer must.
    if lane.base > lane.have && lane.folded + lane.fold < lane.base.min(end) {
        return Err(Fault::program(
            "serve::ring",
            format!(
                "rows [{}, {}) below the lane's window buffer, which its fire folds only to {}",
                lane.have,
                lane.base.min(end),
                lane.folded + lane.fold
            ),
        ));
    }
    for n in 0..pages {
        let ring = lane.ring[(n % ring_pages) as usize];
        // A page of the fire's own rows with no buffer slot: on the ring
        // where the lane writes it alone, else an own-row page, filled
        // first with the folded rows below the fire it holds.
        let mut own_page = |out: &mut Composed, sources: &mut Vec<Source>| {
            if direct {
                out.direct += 1;
                out.table.push(ring.0);
                sources.push(Source::Ring);
                return Ok(());
            }
            let (id, at) = own()?;
            out.own += 1;
            if n * page < lane.have {
                out.fills.push((at, ring.1));
            }
            out.table.push(id);
            sources.push(Source::At(at));
            Ok::<(), crate::error::Fault>(())
        };
        // A page wholly behind the window of the fire's first row reads the
        // null page, wherever its rows lie (an attention that reads every
        // key a page holds, a masked one, reads none of them), and its rows
        // stay where they lie for the fold and the snapshots.
        if n < first && n >= base_page && n < buffered_end {
            let slot = lane.buffer[(n - base_page) as usize];
            out.table.push(null);
            sources.push(Source::At(slot.1));
        } else if n < first.min(base_page) {
            out.table.push(null);
            sources.push(Source::Ring);
        } else if n < base_page && (n + 1) * page <= lane.have {
            out.table.push(ring.0);
            sources.push(Source::Ring);
        } else if n < base_page {
            own_page(&mut out, &mut sources)?;
        } else if n < buffered_end {
            let slot = lane.buffer[(n - base_page) as usize];
            out.table.push(slot.0);
            sources.push(Source::At(slot.1));
        } else {
            own_page(&mut out, &mut sources)?;
        }
    }
    // The buffer's slots hold rows from `base`; those below `folded` are
    // folded, and the ring holds them, but an earlier fire that folded them
    // itself wrote them to no slot. They land on their slots from the ring
    // before the fire reads them there.
    let mut row = lane.base.max(first * page);
    let held = lane.folded.min(lane.have).min(buffered_end * page);
    while row < held {
        let n = row / page;
        let stop = ((n + 1) * page).min(held);
        if n >= base_page {
            let slot = lane.buffer[(n - base_page) as usize];
            out.refills.push((
                slot.1,
                lane.ring[(n % ring_pages) as usize].1,
                row % page,
                stop - row,
            ));
        }
        row = stop;
    }
    // Each snapshot: the last ring's worth of the rows before its row, each
    // page's from where it lies now (the ring as the fire found it, a
    // buffer slot, an own-row page) to the snapshot's ring page.
    for &(at, rings) in lane.snapshots {
        if rings.len() != ring_pages as usize {
            return Err(Fault::program(
                "serve::ring",
                format!(
                    "a snapshot record of {} ring pages beside a lane's of {ring_pages}",
                    rings.len()
                ),
            ));
        }
        let to = (lane.have + at).min(end);
        let mut row = to.saturating_sub(ring_pages * page);
        while row < to {
            let n = row / page;
            let stop = ((n + 1) * page).min(to);
            let from = match sources[n as usize] {
                Source::At(from) => from,
                Source::Ring => lane.ring[(n % ring_pages) as usize].1,
            };
            let dst = rings[(n % ring_pages) as usize].1;
            out.snaps.push((dst, from, row % page, stop - row));
            row = stop;
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
            if out
                .table
                .iter()
                .take(base_page as usize)
                .any(|&id| id == ring.0)
            {
                out.read_then_folded += 1;
            }
            out.folds
                .push((ring.1, Source::At(at), row % page, stop - row));
        }
        row = stop;
    }
    Ok(out)
}

/// A fire's ring work: whole pages filled before it, row runs folded
/// after it, (to, from, bytes) each.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Work {
    pub page_bytes: u64,
    pub fills: Vec<(u64, u64)>,
    /// Row runs onto buffer slots from the ring, before the fire.
    pub refills: Vec<(u64, u64, u64)>,
    /// The snapshots' row runs, landed after the forward and before the
    /// fold (which may write the ring pages they read).
    pub snaps: Vec<(u64, u64, u64)>,
    pub folds: Vec<(u64, u64, u64)>,
}

impl Work {
    /// Adds `composed`'s work: its fills, and its folds as row runs of each
    /// windowed row's planes, `planes` each (offset into a page, bytes a
    /// row) — a row's key plane and its value plane.
    pub fn add(&mut self, composed: &Composed, planes: &[(u64, u64)]) {
        self.fills.extend_from_slice(&composed.fills);
        for &(dst, from, first, rows) in &composed.refills {
            for &(offset, row_bytes) in planes {
                let at = offset + u64::from(first) * row_bytes;
                self.refills
                    .push((dst + at, from + at, u64::from(rows) * row_bytes));
            }
        }
        for &(dst, from, first, rows) in &composed.snaps {
            for &(offset, row_bytes) in planes {
                let at = offset + u64::from(first) * row_bytes;
                self.snaps
                    .push((dst + at, from + at, u64::from(rows) * row_bytes));
            }
        }
        for &(ring, source, first, rows) in &composed.folds {
            let Source::At(from) = source else { continue };
            for &(offset, row_bytes) in planes {
                let at = offset + u64::from(first) * row_bytes;
                self.folds
                    .push((ring + at, from + at, u64::from(rows) * row_bytes));
            }
        }
    }
}

/// Two lanes that read one pool of rows as separate hypotheses: lanes of
/// one working set (or one record) each telling its own rows apart by a
/// mask of its own (a beam group laid in one shared pool). A window cuts
/// such a pool by its slots, not by each hypothesis's own positions, so its
/// lanes would read too few of their own rows: refused by name until
/// windowed beams lie in a page list a lane. `lanes` is each lane's
/// (working set, record, whether it states a mask).
#[must_use]
pub fn shared_pool(lanes: &[(u64, u32, bool)]) -> Option<(usize, usize)> {
    for (at, &(set, record, masked)) in lanes.iter().enumerate() {
        if !masked {
            continue;
        }
        if let Some(other) = lanes[..at]
            .iter()
            .position(|&(held, named, its)| its && (held == set || named == record))
        {
            return Some((other, at));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_at(pages: u32, at: u64) -> Vec<(i32, u64)> {
        (0..pages)
            .map(|p| (900 + p as i32, at + u64::from(p)))
            .collect()
    }

    fn ring(pages: u32) -> Vec<(i32, u64)> {
        (0..pages)
            .map(|p| (100 + p as i32, 1000 + u64::from(p)))
            .collect()
    }

    #[test]
    fn hypotheses_sharing_one_pool_of_window_rows_are_found() {
        // A beam group laid in one pool: one working set, a mask a lane.
        assert_eq!(shared_pool(&[(5, 1, true), (5, 2, true)]), Some((0, 1)));
        // Or one record.
        assert_eq!(shared_pool(&[(5, 1, true), (6, 1, true)]), Some((0, 1)));
        // Unmasked lanes of one set (a plain fork), masked lanes of their
        // own sets and records, share no pool.
        assert_eq!(shared_pool(&[(5, 1, false), (5, 1, false)]), None);
        assert_eq!(shared_pool(&[(5, 1, true), (6, 2, true)]), None);
        assert_eq!(shared_pool(&[(5, 1, true), (5, 1, false)]), None);
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
            snapshots: &[],
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
            snapshots: &[],
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
    fn a_snapshot_lands_the_rows_before_its_row_from_where_they_lie() {
        let ring = ring(3);
        let snap = ring_at(3, 9000);
        let lane = RingLane {
            ring: &ring,
            buffer: &[],
            base: 32,
            have: 40,
            rows: 20,
            folded: 40,
            fold: 20,
            alone: true,
            snapshots: &[(5, &snap)],
        };
        let mut next = 0u32;
        let mut own = || {
            next += 1;
            Ok((500 + next as i32, 5000 + u64::from(next)))
        };
        let c = compose(&lane, 16, 20, 0, &mut own).unwrap();
        // A snapshot keeps the fire off its ring: its own rows on own pages.
        assert_eq!(c.direct, 0);
        // Rows [0, 45): page 0 and 1 from the ring as found, page 2's rows
        // [32, 40) from the ring and [40, 45) from its own page (the page is
        // the own page, filled from the ring): one run a page.
        let rows: u32 = c.snaps.iter().map(|s| s.3).sum();
        assert_eq!(rows, 45);
        assert_eq!(c.snaps[0], (9000, 1000, 0, 16));
        assert_eq!(c.snaps[2], (9002, 5001, 0, 13));
    }

    #[test]
    fn a_buffered_page_behind_the_window_reads_the_null_page() {
        let ring = ring(3);
        let buffer: Vec<(i32, u64)> = (0..7).map(|p| (10 + p, 100 + p as u64)).collect();
        // Nothing folded: the whole context in the buffer, past the window.
        let lane = RingLane {
            ring: &ring,
            buffer: &buffer,
            base: 0,
            have: 100,
            rows: 1,
            folded: 0,
            fold: 0,
            alone: true,
            snapshots: &[],
        };
        let mut own = || panic!("no own page");
        let c = compose(&lane, 16, 20, 0, &mut own).unwrap();
        // Rows [81, 101) are read: pages 0..=4 lie wholly behind them.
        assert_eq!(c.table, vec![0, 0, 0, 0, 0, 15, 16]);
    }

    #[test]
    fn a_prefill_whose_buffer_starts_past_its_folded_rows_keeps_them_off_its_ring() {
        // The runtime starts the buffer at the fold after the fire (rows
        // [0, 192) folded, [192, 200) buffered): the fire's 12 pages below
        // the buffer, past a ring of 3, go to own-row pages.
        let ring = ring(3);
        let buffer = [(7, 70u64)];
        let lane = RingLane {
            ring: &ring,
            buffer: &buffer,
            base: 192,
            have: 0,
            rows: 200,
            folded: 0,
            fold: 192,
            alone: true,
            snapshots: &[],
        };
        let mut next = 0u32;
        let mut own = || {
            next += 1;
            Ok((500 + next as i32, 5000 + u64::from(next)))
        };
        let c = compose(&lane, 16, 20, 0, &mut own).unwrap();
        assert_eq!(c.own, 12);
        assert_eq!(c.direct, 0);
        assert_eq!(c.table[12], 7);
        // The fold lands the last ring's worth of [0, 192) from own pages.
        let rows: u32 = c.folds.iter().map(|f| f.3).sum();
        assert_eq!(rows, 48);
    }

    #[test]
    fn a_buffer_slot_takes_its_folded_rows_from_the_ring() {
        // Folded to 200 by a fire that took no slot; the buffer from 192.
        let ring = ring(3);
        let buffer = [(7, 70u64)];
        let lane = RingLane {
            ring: &ring,
            buffer: &buffer,
            base: 192,
            have: 200,
            rows: 1,
            folded: 200,
            fold: 0,
            alone: true,
            snapshots: &[],
        };
        let mut own = || panic!("no own page");
        let c = compose(&lane, 16, 20, 0, &mut own).unwrap();
        // Rows [192, 200) onto the slot from ring page 12 mod 3.
        assert_eq!(c.refills, vec![(70, 1000, 0, 8)]);
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
            snapshots: &[],
        };
        let mut own = || panic!("no own page");
        let c = compose(&lane, 16, 20, 0, &mut own).unwrap();
        assert_eq!(c.table, vec![0, 101, 7, 8]);
        assert_eq!(c.folds, vec![(1002, Source::At(70), 4, 4)]);
    }
}
