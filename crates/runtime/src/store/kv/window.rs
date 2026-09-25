use std::collections::{HashMap, HashSet};

use super::page_table::{PhysicalKvPageId, WorkingSetId};
use crate::store::pool::{Pool, PoolId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WindowPageId(u32);

impl PoolId for WindowPageId {
    fn from_index(index: u32) -> Self {
        Self(index)
    }
    fn index(self) -> u32 {
        self.0
    }
}

/// Each lane's windowed ids aligned with its pages, and the windowed page
/// copies `(src, dst)` the fire runs first.
pub type Claimed = (Vec<Vec<u32>>, Vec<(u32, u32)>);

/// Why a claim was refused: the grant is short of what the fire needs now
/// (its demand went stale), or a lane reads a page already given back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimError {
    Short(String),
    Rewound(String),
}

impl std::fmt::Display for ClaimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClaimError::Short(why) | ClaimError::Rewound(why) => f.write_str(why),
        }
    }
}

/// Who holds a claim on windowed pages: a working set, or an index entry
/// that a later working set is seated on.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Holder {
    Ws(WorkingSetId),
    Index(Vec<u8>),
}

/// One lane of a fire as the windowed pool sees it: its pages, the span of
/// them its rows read (`first..end`), and from which page on a page holds
/// no token written before this fire.
pub struct WindowLane<'a> {
    pub pages: &'a [u32],
    pub first: usize,
    pub end: usize,
    pub fresh_from: usize,
}

impl<'a> WindowLane<'a> {
    /// The lane of a fire that appends `rows` to the `held` tokens its
    /// `pages` hold, read through a window of `tokens`.
    #[must_use]
    pub fn of(pages: &'a [u32], held: u32, rows: u32, tokens: u32, page_size: u32) -> Self {
        let page_size = page_size.max(1);
        let end = ((held + rows).div_ceil(page_size).max(1) as usize).min(pages.len());
        WindowLane {
            pages,
            first: ((held.saturating_sub(tokens) / page_size) as usize).min(end),
            end,
            fresh_from: held.div_ceil(page_size) as usize,
        }
    }
}

/// A windowed page's id, how many holders claim it, and the epoch of the
/// last fire that read it, after which it may be handed out again.
struct Held {
    id: u32,
    holders: u32,
    read: u64,
}

/// The windowed kv spaces' own pages. A kv page holds one only while some
/// holder's window reads it: each fire claims what its lanes read and gives
/// back what fell behind, so a sequence costs its window here, not its
/// length. Id 0 is the engine's null page, never handed out. Free ids are
/// taken by the residency planner, as kv pages are, and handed to `claim`.
pub struct WindowPool {
    pub tokens: u32,
    page_size: u32,
    pool: Pool<WindowPageId>,
    of: HashMap<PhysicalKvPageId, Held>,
    claims: HashMap<Holder, Vec<PhysicalKvPageId>>,
}

impl WindowPool {
    #[must_use]
    pub fn new(tokens: u32, page_size: u32, pages: u32) -> Self {
        WindowPool {
            tokens,
            page_size,
            pool: Pool::new_range(1, pages.saturating_sub(1)),
            of: HashMap::new(),
            claims: HashMap::new(),
        }
    }

    #[must_use]
    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    #[must_use]
    pub fn available(&self) -> usize {
        self.pool.available()
    }

    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.pool.capacity()
    }

    pub fn reserve(&mut self, count: usize) -> Option<Vec<PhysicalKvPageId>> {
        let ids = self.pool.try_alloc_n(count)?;
        Some(ids.into_iter().map(|id| PhysicalKvPageId(id.0)).collect())
    }

    pub fn release_reserved(&mut self, ids: Vec<PhysicalKvPageId>) {
        self.pool
            .release_reserved(ids.into_iter().map(|id| WindowPageId(id.0)).collect());
    }

    #[must_use]
    pub fn backs(&self, page: PhysicalKvPageId) -> bool {
        self.of.contains_key(&page)
    }

    /// Gives back what `holder` claims outside `keep`: the pages its next
    /// fire's windows have moved past.
    pub fn keep(&mut self, holder: &Holder, keep: &HashSet<PhysicalKvPageId>) {
        let Some(claim) = self.claims.get_mut(holder) else {
            return;
        };
        let (kept, behind) = claim.drain(..).partition(|page| keep.contains(page));
        *claim = kept;
        self.give_back(behind);
    }

    /// Claims for `holder` the windowed pages `lanes` read in the fire of
    /// `epoch`, backing the fresh ones and the `copies` destinations whose
    /// source has one from `granted`, and gives back the pages its last
    /// claim held that none reads now. Returns each lane's ids aligned with
    /// its pages, and the windowed copies that mirror `copies`.
    pub fn claim(
        &mut self,
        holder: Holder,
        lanes: &[WindowLane<'_>],
        copies: &[(u32, u32)],
        granted: &mut Vec<PhysicalKvPageId>,
        epoch: u64,
    ) -> Result<Claimed, ClaimError> {
        let read: HashSet<u32> = lanes
            .iter()
            .flat_map(|lane| &lane.pages[lane.first..lane.end])
            .copied()
            .collect();
        let copies: Vec<(u32, u32)> = copies
            .iter()
            .copied()
            .filter(|&(src, dst)| read.contains(&dst) && self.backs(PhysicalKvPageId(src)))
            .collect();
        let copied = |page: u32| copies.iter().any(|&(_, dst)| dst == page);
        for lane in lanes {
            for at in lane.first..lane.fresh_from.min(lane.end) {
                let page = lane.pages[at];
                if !self.backs(PhysicalKvPageId(page)) && !copied(page) {
                    return Err(ClaimError::Rewound(format!(
                        "windowed kv page {at} (page {page}) holds tokens this lane's window \
                         reads and was given back when the sequence moved past it; a \
                         sequence cannot be rewound past its window"
                    )));
                }
            }
        }
        let fresh = read
            .iter()
            .filter(|&&page| !self.backs(PhysicalKvPageId(page)))
            .count();
        if fresh > granted.len() {
            return Err(ClaimError::Short(format!(
                "this fire needs {fresh} windowed kv page(s) and was granted {}",
                granted.len()
            )));
        }
        let mut held: Vec<PhysicalKvPageId> = Vec::new();
        let mut seen: HashSet<PhysicalKvPageId> = HashSet::new();
        let mut window_copies = Vec::with_capacity(copies.len());
        for &(src, dst) in &copies {
            let from = self.of[&PhysicalKvPageId(src)].id;
            window_copies.push((from, self.back(PhysicalKvPageId(dst), granted)));
        }
        let mut ids = Vec::with_capacity(lanes.len());
        for lane in lanes {
            let mut row = vec![0u32; lane.pages.len()];
            for (id, &page) in row[lane.first..lane.end]
                .iter_mut()
                .zip(&lane.pages[lane.first..lane.end])
            {
                let page = PhysicalKvPageId(page);
                *id = self.back(page, granted);
                if seen.insert(page) {
                    held.push(page);
                }
            }
            ids.push(row);
        }
        for page in &held {
            if let Some(entry) = self.of.get_mut(page) {
                entry.holders += 1;
                entry.read = entry.read.max(epoch);
            }
        }
        let previous = self.claims.insert(holder, held).unwrap_or_default();
        self.give_back(previous);
        Ok((ids, window_copies))
    }

    /// Seats `to` on what `from` holds, as a fork, slice or index entry
    /// reads the pages its source's window reads.
    pub fn share(&mut self, from: &Holder, to: Holder) {
        let pages = self.claims.get(from).cloned().unwrap_or_default();
        for page in &pages {
            if let Some(entry) = self.of.get_mut(page) {
                entry.holders += 1;
            }
        }
        let previous = self.claims.insert(to, pages).unwrap_or_default();
        self.give_back(previous);
    }

    pub fn release(&mut self, holder: &Holder) {
        let previous = self.claims.remove(holder).unwrap_or_default();
        self.give_back(previous);
    }

    pub fn retire_through(&mut self, epoch: u64) {
        self.pool.retire_through(epoch);
    }

    /// The page's windowed id, taken from `granted` when it has none;
    /// `claim` counted the ids this may take before calling it.
    fn back(&mut self, page: PhysicalKvPageId, granted: &mut Vec<PhysicalKvPageId>) -> u32 {
        if let Some(entry) = self.of.get(&page) {
            return entry.id;
        }
        let id = granted.pop().map_or(0, |id| id.0);
        self.of.insert(
            page,
            Held {
                id,
                holders: 0,
                read: 0,
            },
        );
        id
    }

    /// Drops a claim on `pages`; a page no holder claims goes back to the
    /// free list once the last fire that read it has retired.
    fn give_back(&mut self, pages: Vec<PhysicalKvPageId>) {
        for page in pages {
            let Some(entry) = self.of.get_mut(&page) else {
                continue;
            };
            entry.holders = entry.holders.saturating_sub(1);
            if entry.holders == 0 {
                let (id, read) = (entry.id, entry.read);
                self.of.remove(&page);
                self.pool.recycle_after_epoch(vec![WindowPageId(id)], read);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #699: a sequence costs its window in the windowed pool — a page
    // comes back once no holder's window reads it — while a fork keeps
    // reading what its parent has left behind.
    #[test]
    fn a_page_behind_every_window_is_given_back() {
        let mut table = super::super::page_table::KvPageTable::new();
        let (parent, child) = (table.create_working_set(), table.create_working_set());
        let ws = |n: u32| Holder::Ws(if n == 1 { parent } else { child });
        let mut pool = WindowPool::new(32, 16, 10);
        let pages: Vec<u32> = (100..110).collect();
        let forked_pages: Vec<u32> = [100, 101, 102, 103, 200, 201].into();
        let fire = |pool: &mut WindowPool, holder: Holder, pages: &[u32], held: u32, rows: u32| {
            let mut granted = pool.reserve(pool.available()).expect("the free ids");
            let lane = WindowLane::of(pages, held, rows, 32, 16);
            let claimed = pool.claim(holder, &[lane], &[], &mut granted, 0);
            pool.release_reserved(granted);
            claimed.map(|(ids, _)| ids.into_iter().next().expect("one lane"))
        };
        let prefill = fire(&mut pool, ws(1), &pages, 0, 64).expect("four fresh pages");
        assert_eq!(prefill.iter().filter(|&&id| id != 0).count(), 4);
        pool.share(&ws(1), ws(2));
        fire(&mut pool, ws(1), &pages, 64, 32).expect("two more");
        let decode = fire(&mut pool, ws(1), &pages, 96, 1).expect("the window moves on");
        assert_eq!(&decode[..4], &[0, 0, 0, 0], "tokens 64 on sit from page 4");
        pool.retire_through(0);
        assert_eq!(pool.available(), 2, "the fork still holds pages 0..4");
        let forked = fire(&mut pool, ws(2), &forked_pages, 64, 1).expect("the fork's window");
        assert_eq!(
            &forked[2..4],
            &prefill[2..4],
            "reads what its parent left behind"
        );
        pool.release(&ws(2));
        pool.retire_through(0);
        assert_eq!(
            pool.available(),
            6,
            "the parent's window of three pages is all it holds"
        );
        assert!(
            fire(&mut pool, ws(1), &pages, 40, 1).is_err(),
            "a rewind into pages given back is refused, never read stale"
        );
    }
}
