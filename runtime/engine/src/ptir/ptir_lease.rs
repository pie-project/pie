//! PTIR device-geometry page leasing (plan W3.3).
//!
//! A device-geometry pass (its `pages`/`w_slot` descriptor ports are produced by
//! the program on-device — the generalized §6.2 beam) needs the runtime to grant
//! PHYSICAL pages the device writes new-token K/V into, since the host no longer
//! knows the per-fire geometry. Physical-space page ids from the start (§3.2):
//! the runtime seeds fire 0's pages and grants `B` fresh physical ids per submit;
//! the whole loop stays physical and the driver needs no slot→physical table.
//!
//! Lifecycle:
//!   * fire 0: `B` pages granted as the seed (one live page per lane).
//!   * each submit: `B` fresh pages granted, delivered to the program as a
//!     host-put on its fresh-page input channel (D1-coalesced).
//!   * after a fire commits: the harvested `w_cont` (per-lane "continued a shared
//!     tail in place" vs "forked a fresh page") says which fresh grants were
//!     UNUSED (a continuing heir did not consume its fresh page) — reclaim them.
//!   * pass drop / failure: reclaim everything.
//!
//! Pin float is bounded by `(run-ahead depth) × B` pages (each in-flight fire
//! holds at most its `B` fresh grants until it commits + reclaims).
//!
//! This module is the PURE bookkeeping half (grant / reclaim / free-list),
//! unit-tested here; the working-set page allocation + the host-put wiring live
//! in `ptir_host` (they need the arena / ws, which need a device).

/// A per-device-geometry-pass physical page lease. Tracks the pages granted to
/// each in-flight fire (FIFO) so unused fresh grants are reclaimed as fires
/// commit, and everything is reclaimed on drop.
#[derive(Debug, Default)]
pub struct PageLease {
    /// Beam / lane width `B` — the number of fresh pages granted per fire.
    pub b: usize,
    /// The reusable free-list of physical page ids reclaimed from prior fires
    /// (drawn from before allocating anew, bounding total page use).
    free: Vec<u32>,
    /// Per-in-flight-fire grants, submission order (FIFO). `pending[i]` are the
    /// `B` fresh page ids granted to the i-th oldest un-reclaimed fire.
    pending: std::collections::VecDeque<Vec<u32>>,
    /// The live seed pages (fire 0's one-page-per-lane grant), reclaimed only on
    /// pass drop.
    seed_pages: Vec<u32>,
}

impl PageLease {
    /// A fresh lease for `b` lanes.
    pub fn new(b: usize) -> Self {
        PageLease {
            b,
            free: Vec::new(),
            pending: std::collections::VecDeque::new(),
            seed_pages: Vec::new(),
        }
    }

    /// Record the fire-0 seed pages (one live page per lane); reclaimed on drop.
    pub fn seed(&mut self, pages: Vec<u32>) {
        self.seed_pages = pages;
    }

    /// Draw `B` fresh physical page ids for a submit: reuse the free-list first,
    /// then mint fresh ids via `alloc` (a monotonic page-id source or ws-backed
    /// allocator). Records the grant on the pending FIFO so it can be reclaimed
    /// after the fire commits.
    pub fn grant<F: FnMut() -> u32>(&mut self, mut alloc: F) -> Vec<u32> {
        let mut pages = Vec::with_capacity(self.b);
        for _ in 0..self.b {
            pages.push(self.free.pop().unwrap_or_else(&mut alloc));
        }
        self.pending.push_back(pages.clone());
        pages
    }

    /// Reclaim the UNUSED fresh grants of the oldest in-flight fire after it
    /// commits, using its harvested per-lane `w_cont` (`true` = the lane
    /// CONTINUED a shared tail in place, so its fresh page went unused →
    /// reclaimable; `false` = the lane FORKED onto its fresh page → keep it live).
    /// Returns the reclaimed ids (re-added to the free-list). No-op if no fire is
    /// pending.
    pub fn reclaim_after_fire(&mut self, w_cont: &[bool]) -> Vec<u32> {
        let Some(grant) = self.pending.pop_front() else {
            return Vec::new();
        };
        let mut reclaimed = Vec::new();
        for (lane, page) in grant.into_iter().enumerate() {
            // A continuing heir (w_cont true) didn't consume its fresh page.
            if w_cont.get(lane).copied().unwrap_or(false) {
                self.free.push(page);
                reclaimed.push(page);
            }
            // else: the lane forked onto `page` — it's now a live page (tracked
            // by the program's geometry, not the lease's pending set).
        }
        reclaimed
    }

    /// Reclaim EVERY page still held by the lease (pass drop / failure): all
    /// pending fires' grants + the seed pages. Returns the freed ids.
    pub fn reclaim_all(&mut self) -> Vec<u32> {
        let mut all = Vec::new();
        while let Some(grant) = self.pending.pop_front() {
            all.extend(grant);
        }
        all.extend(std::mem::take(&mut self.seed_pages));
        all.append(&mut self.free);
        all
    }

    /// Number of in-flight (un-reclaimed) fires — the pin-float depth × B bound.
    pub fn in_flight(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A monotonic page-id allocator for tests (mints 1000, 1001, …).
    fn allocator() -> impl FnMut() -> u32 {
        let mut next = 1000u32;
        move || {
            let id = next;
            next += 1;
            id
        }
    }

    #[test]
    fn grant_mints_b_pages_and_tracks_in_flight() {
        let mut lease = PageLease::new(2);
        let mut alloc = allocator();
        let g0 = lease.grant(&mut alloc);
        assert_eq!(g0, vec![1000, 1001], "B=2 fresh pages");
        assert_eq!(lease.in_flight(), 1);
        let g1 = lease.grant(&mut alloc);
        assert_eq!(g1, vec![1002, 1003], "next fire mints the next B");
        assert_eq!(lease.in_flight(), 2, "run-ahead: two fires in flight");
    }

    #[test]
    fn reclaim_returns_only_continued_lanes() {
        let mut lease = PageLease::new(2);
        let mut alloc = allocator();
        let _g = lease.grant(&mut alloc); // [1000, 1001]
        // lane 0 continued (heir, fresh unused → reclaim), lane 1 forked (keep).
        let reclaimed = lease.reclaim_after_fire(&[true, false]);
        assert_eq!(
            reclaimed,
            vec![1000],
            "only the continuing lane's fresh page"
        );
        assert_eq!(lease.in_flight(), 0, "the oldest fire is retired");
    }

    #[test]
    fn reclaimed_pages_are_reused_before_minting() {
        let mut lease = PageLease::new(2);
        let mut alloc = allocator();
        let _g0 = lease.grant(&mut alloc); // [1000, 1001]
        lease.reclaim_after_fire(&[true, true]); // both continued → 1000,1001 freed
        // Next grant reuses the free-list (LIFO) before minting.
        let g1 = lease.grant(&mut alloc);
        assert_eq!(
            g1,
            vec![1001, 1000],
            "reused freed pages, none newly minted"
        );
    }

    #[test]
    fn reclaim_all_frees_pending_and_seed() {
        let mut lease = PageLease::new(2);
        let mut alloc = allocator();
        lease.seed(vec![500, 501]);
        let _g0 = lease.grant(&mut alloc); // [1000,1001]
        let _g1 = lease.grant(&mut alloc); // [1002,1003]
        let mut all = lease.reclaim_all();
        all.sort();
        assert_eq!(
            all,
            vec![500, 501, 1000, 1001, 1002, 1003],
            "everything freed on drop"
        );
        assert_eq!(lease.in_flight(), 0);
    }

    #[test]
    fn reclaim_with_no_pending_is_noop() {
        let mut lease = PageLease::new(2);
        assert!(lease.reclaim_after_fire(&[true, true]).is_empty());
    }
}
