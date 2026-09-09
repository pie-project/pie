#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashSet;
use smallvec::{SmallVec, smallvec};

use super::hash::{self, Hash256};
use crate::store::genmap::{GenKey, GenMap};
use crate::store::pool::PoolId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WsMarker {}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeMarker {}

pub type WorkingSetId = GenKey<WsMarker>;
pub type NodeId = GenKey<NodeMarker>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysicalKvPageId(pub u32);

impl PoolId for PhysicalKvPageId {
    fn from_index(index: u32) -> Self {
        Self(index)
    }
    fn index(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HostKvSlotId(pub u32);

impl PoolId for HostKvSlotId {
    fn from_index(index: u32) -> Self {
        Self(index)
    }
    fn index(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvPageBacking {
    Resident(PhysicalKvPageId),
    Swapped(HostKvSlotId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReclaimQuote {
    Pages(u32),
    Nothing(NoReclaim),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoReclaim {
    HoldsNothing,
    AllShared,
    AllSwapped,
    Pinned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TriePageLocation {
    pub node: NodeId,
    pub local: u64,
}

type Runs = SmallVec<[Range<u32>; 2]>;

#[derive(Debug, Clone)]
pub struct PublishedPage {
    pub id: PhysicalKvPageId,
    pub token_hashes: Vec<Option<Hash256>>,
    pub page_hash: Option<Hash256>,
}

enum Pages {
    Owned {
        backings: Vec<KvPageBacking>,
        token_hashes: Vec<Vec<Option<Hash256>>>,
        page_hashes: Vec<Option<Hash256>>,
    },
    ParentSelection {
        runs: Runs,
    },
}

struct KvTrieNode {
    parent: Option<NodeId>,
    children: SmallVec<[NodeId; 2]>,
    pages: Pages,
    cached_path_hash: Option<Hash256>,
    exact_anchors: u32,
    path_anchors: u32,
}

struct WorkingSetEntry {
    terminal: Option<NodeId>,
    page_len: u64,
    page_len_mirror: Arc<AtomicU64>,
    mapped_len: u64,
    chain_state: Option<Hash256>,
}

impl WorkingSetEntry {
    const TORN_DOWN: u64 = u64::MAX;

    fn new(
        terminal: Option<NodeId>,
        page_len: u64,
        mapped_len: u64,
        chain_state: Option<Hash256>,
    ) -> Self {
        Self {
            terminal,
            page_len,
            page_len_mirror: Arc::new(AtomicU64::new(page_len)),
            mapped_len,
            chain_state,
        }
    }

    fn set_page_len(&mut self, page_len: u64) {
        self.page_len = page_len;
        self.page_len_mirror.store(page_len, Ordering::Release);
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct IndexedWorkingSet {
    pub(super) terminal: Option<NodeId>,
    page_len: u64,
    mapped_len: u64,
    chain_state: Option<Hash256>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KvTableError {
    #[error("unknown working set")]
    UnknownWorkingSet,
    #[error("page index {index} out of range (page_len {page_len})")]
    IndexOutOfRange { index: u64, page_len: u64 },
    #[error("page index {index} is reserved but unwritten (mapped {mapped_len})")]
    Unwritten { index: u64, mapped_len: u64 },
    #[error("range {start}..{end} invalid (mapped {mapped_len}, page_len {page_len})")]
    BadRange {
        start: u64,
        end: u64,
        mapped_len: u64,
        page_len: u64,
    },
    #[error(
        "publishing {count} pages exceeds the reservation (mapped {mapped_len}, page_len {page_len})"
    )]
    PublishExceedsReservation {
        count: u64,
        mapped_len: u64,
        page_len: u64,
    },
    #[error("interior discard on a shared path is rejected (growth-boundary invariant)")]
    SharedInteriorDiscard,
    #[error("working set page {index} is swapped out")]
    NonResident { index: u64 },
    #[error("page backing changed while a residency transaction was in flight")]
    BackingChanged,
    #[error(
        "working set cannot be indexed with an unmapped logical tail \
         (mapped {mapped_len}, page_len {page_len})"
    )]
    UnmappedTail { mapped_len: u64, page_len: u64 },
}

#[derive(Debug, Clone, Copy)]
struct Segment {
    node: NodeId,
    start: i64,
    len: u64,
}

impl Segment {
    fn end(&self) -> i64 {
        self.start + self.len as i64
    }
}

#[derive(Default)]
pub struct KvPageTable {
    working_sets: GenMap<WsMarker, WorkingSetEntry>,
    nodes: GenMap<NodeMarker, KvTrieNode>,
    cache_roots: HashMap<NodeId, u32>,
    pins: HashMap<NodeId, u32>,
    swap_locations: HashMap<TriePageLocation, u32>,
}

impl KvPageTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_working_set(&mut self) -> WorkingSetId {
        self.working_sets
            .insert(WorkingSetEntry::new(None, 0, 0, None))
    }

    pub fn fork(&mut self, ws: WorkingSetId) -> Result<WorkingSetId, KvTableError> {
        let entry = self.entry(ws)?;
        let (terminal, page_len, mapped_len, chain_state) = (
            entry.terminal,
            entry.page_len,
            entry.mapped_len,
            entry.chain_state,
        );
        let child = self.working_sets.insert(WorkingSetEntry::new(
            terminal,
            page_len,
            mapped_len,
            chain_state,
        ));
        if let Some(terminal) = terminal {
            self.add_anchor(terminal);
        }
        Ok(child)
    }

    pub(super) fn index_snapshot(
        &self,
        ws: WorkingSetId,
    ) -> Result<IndexedWorkingSet, KvTableError> {
        let entry = self.entry(ws)?;
        if entry.page_len != entry.mapped_len {
            return Err(KvTableError::UnmappedTail {
                mapped_len: entry.mapped_len,
                page_len: entry.page_len,
            });
        }
        Ok(IndexedWorkingSet {
            terminal: entry.terminal,
            page_len: entry.page_len,
            mapped_len: entry.mapped_len,
            chain_state: entry.chain_state,
        })
    }

    #[allow(
        clippy::wrong_self_convention,
        reason = "not a conversion: it mutates the table to seat a new working set, \
                  and is named for the `from-index` WIT call it serves (see \
                  `KvStore::from_index`, its only caller)"
    )]
    pub(super) fn from_index_snapshot(&mut self, snapshot: IndexedWorkingSet) -> WorkingSetId {
        let ws = self.working_sets.insert(WorkingSetEntry::new(
            snapshot.terminal,
            snapshot.page_len,
            snapshot.mapped_len,
            snapshot.chain_state,
        ));
        if let Some(terminal) = snapshot.terminal {
            self.add_anchor(terminal);
        }
        ws
    }

    pub fn slice(
        &mut self,
        ws: WorkingSetId,
        range: Range<u64>,
    ) -> Result<WorkingSetId, KvTableError> {
        let entry = self.entry(ws)?;
        let (terminal, page_len, mapped_len) = (entry.terminal, entry.page_len, entry.mapped_len);
        if range.start > range.end || range.end > mapped_len {
            return Err(KvTableError::BadRange {
                start: range.start,
                end: range.end,
                mapped_len,
                page_len,
            });
        }
        let len = range.end - range.start;
        let child_terminal = if len == 0 {
            None
        } else {
            let segs = self.segments(terminal, mapped_len);
            Some(self.boundary_terminal(&segs, range.end))
        };
        let child = self
            .working_sets
            .insert(WorkingSetEntry::new(child_terminal, len, len, None));
        if let Some(terminal) = child_terminal {
            self.add_anchor(terminal);
        }
        Ok(child)
    }

    pub fn reserve(&mut self, ws: WorkingSetId, pages: u64) -> Result<Range<u64>, KvTableError> {
        let entry = self.entry_mut(ws)?;
        let start = entry.page_len;
        entry.set_page_len(entry.page_len + pages);
        Ok(start..entry.page_len)
    }

    pub fn publish_appended(
        &mut self,
        ws: WorkingSetId,
        pages: Vec<PublishedPage>,
    ) -> Result<(), KvTableError> {
        let entry = self.entry(ws)?;
        let (terminal, page_len, mapped_len) = (entry.terminal, entry.page_len, entry.mapped_len);
        let count = pages.len() as u64;
        if mapped_len + count > page_len {
            return Err(KvTableError::PublishExceedsReservation {
                count,
                mapped_len,
                page_len,
            });
        }
        if count == 0 {
            return Ok(());
        }

        let mut backings = Vec::with_capacity(pages.len());
        let mut token_hashes = Vec::with_capacity(pages.len());
        let mut page_hashes = Vec::with_capacity(pages.len());
        for page in pages {
            backings.push(KvPageBacking::Resident(page.id));
            token_hashes.push(page.token_hashes);
            page_hashes.push(page.page_hash);
        }

        let new_terminal = match terminal {
            Some(t) if self.can_extend_in_place(ws, t) => {
                let node = self.nodes.get_mut(t).expect("live terminal");
                match &mut node.pages {
                    Pages::Owned {
                        backings: node_backings,
                        token_hashes: node_tokens,
                        page_hashes: node_pages,
                    } => {
                        node_backings.extend(backings);
                        node_tokens.extend(token_hashes);
                        node_pages.extend(page_hashes);
                    }
                    Pages::ParentSelection { .. } => unreachable!("checked owned"),
                }
                self.invalidate_subtree(t);
                t
            }
            _ => {
                let node = self.nodes.insert(KvTrieNode {
                    parent: terminal,
                    children: SmallVec::new(),
                    pages: Pages::Owned {
                        backings,
                        token_hashes,
                        page_hashes,
                    },
                    cached_path_hash: None,
                    exact_anchors: 0,
                    path_anchors: 0,
                });
                if let Some(t) = terminal {
                    self.nodes
                        .get_mut(t)
                        .expect("live terminal")
                        .children
                        .push(node);
                }
                node
            }
        };

        let mapped_len = self.entry(ws)?.mapped_len;
        let freed = self.move_terminal(ws, Some(new_terminal))?;
        debug_assert!(freed.is_empty(), "append descendants retain the old path");
        self.entry_mut(ws)?.mapped_len = mapped_len + count;
        Ok(())
    }

    pub fn replace_tail(
        &mut self,
        ws: WorkingSetId,
        from: u64,
        pages: Vec<PublishedPage>,
    ) -> Result<Vec<KvPageBacking>, KvTableError> {
        let entry = self.entry(ws)?;
        let (terminal, page_len, mapped_len) = (entry.terminal, entry.page_len, entry.mapped_len);
        if from > mapped_len {
            return Err(KvTableError::BadRange {
                start: from,
                end: from,
                mapped_len,
                page_len,
            });
        }
        let count = pages.len() as u64;
        if from.saturating_add(count) > page_len {
            return Err(KvTableError::PublishExceedsReservation {
                count,
                mapped_len: from,
                page_len,
            });
        }
        if from < mapped_len {
            let segs = self.segments(terminal, mapped_len);
            let new_terminal = if from == 0 {
                None
            } else {
                Some(self.boundary_terminal(&segs, from))
            };
            let freed = self.move_terminal(ws, new_terminal)?;
            self.entry_mut(ws)?.mapped_len = from;
            self.publish_appended(ws, pages)?;
            return Ok(freed);
        }
        self.publish_appended(ws, pages)?;
        Ok(Vec::new())
    }

    pub fn commit_in_place(
        &mut self,
        ws: WorkingSetId,
        index: u64,
        token_hashes: Vec<Option<Hash256>>,
        page_hash: Option<Hash256>,
    ) -> Result<(), KvTableError> {
        let entry = self.entry(ws)?;
        if index >= entry.mapped_len {
            return Err(KvTableError::Unwritten {
                index,
                mapped_len: entry.mapped_len,
            });
        }
        let segs = self.segments(entry.terminal, entry.mapped_len);
        let seg = segs
            .iter()
            .find(|s| (index as i64) >= s.start)
            .copied()
            .expect("published mapping covers [0, mapped_len)");
        let local = (index as i64 - seg.start) as u64;
        let (owner, owner_local) = self.resolve_owner(seg.node, local);
        let node = self.nodes.get_mut(owner).expect("live node");
        match &mut node.pages {
            Pages::Owned {
                token_hashes: node_tokens,
                page_hashes: node_pages,
                ..
            } => {
                node_tokens[owner_local as usize] = token_hashes;
                node_pages[owner_local as usize] = page_hash;
            }
            Pages::ParentSelection { .. } => unreachable!("resolved to owner"),
        }
        self.invalidate_subtree(owner);
        Ok(())
    }

    pub fn privately_writable(&self, ws: WorkingSetId, index: u64) -> Result<bool, KvTableError> {
        let entry = self.entry(ws)?;
        if index >= entry.mapped_len {
            return Err(KvTableError::Unwritten {
                index,
                mapped_len: entry.mapped_len,
            });
        }
        let segs = self.segments(entry.terminal, entry.mapped_len);
        let seg = segs
            .iter()
            .find(|s| (index as i64) >= s.start)
            .copied()
            .expect("published mapping covers [0, mapped_len)");
        let local = (index as i64 - seg.start) as u64;
        let (owner, _) = self.resolve_owner(seg.node, local);
        let mut targets = HashSet::new();
        targets.insert(owner);
        Ok(self.is_private_to(ws, &targets))
    }

    pub fn discard(
        &mut self,
        ws: WorkingSetId,
        ranges: &[Range<u64>],
    ) -> Result<Vec<KvPageBacking>, KvTableError> {
        let entry = self.entry(ws)?;
        let (terminal, page_len, mapped_len) = (entry.terminal, entry.page_len, entry.mapped_len);

        let mut norm: Vec<Range<u64>> =
            ranges.iter().filter(|r| r.start < r.end).cloned().collect();
        norm.sort_by_key(|r| r.start);
        let mut merged: Vec<Range<u64>> = Vec::with_capacity(norm.len());
        for r in norm {
            if r.end > page_len {
                return Err(KvTableError::BadRange {
                    start: r.start,
                    end: r.end,
                    mapped_len,
                    page_len,
                });
            }
            match merged.last_mut() {
                Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
                _ => merged.push(r),
            }
        }
        merged.reverse();

        self.classify_discard(ws, terminal, mapped_len, &merged)?;

        let mut freed = Vec::new();
        for r in &merged {
            self.apply_discard_range(ws, r.clone(), &mut freed)?;
        }
        Ok(freed)
    }

    pub fn release_working_set(&mut self, ws: WorkingSetId) -> Vec<PhysicalKvPageId> {
        self.release_working_set_backings(ws)
            .into_iter()
            .filter_map(|backing| match backing {
                KvPageBacking::Resident(id) => Some(id),
                KvPageBacking::Swapped(_) => {
                    debug_assert!(false, "swapped backing must be released through KvStore");
                    None
                }
            })
            .collect()
    }

    pub fn release_working_set_backings(&mut self, ws: WorkingSetId) -> Vec<KvPageBacking> {
        let Some(entry) = self.working_sets.remove(ws) else {
            return Vec::new();
        };
        entry
            .page_len_mirror
            .store(WorkingSetEntry::TORN_DOWN, Ordering::Release);
        entry
            .terminal
            .map_or_else(Vec::new, |terminal| self.remove_anchor(terminal))
    }

    pub fn lookup(&self, ws: WorkingSetId, index: u64) -> Result<PhysicalKvPageId, KvTableError> {
        let entry = self.entry(ws)?;
        if index >= entry.page_len {
            return Err(KvTableError::IndexOutOfRange {
                index,
                page_len: entry.page_len,
            });
        }
        if index >= entry.mapped_len {
            return Err(KvTableError::Unwritten {
                index,
                mapped_len: entry.mapped_len,
            });
        }
        let segs = self.segments(entry.terminal, entry.mapped_len);
        for seg in &segs {
            if (index as i64) >= seg.start {
                let local = (index as i64 - seg.start) as u64;
                return self
                    .resolve_id(seg.node, local)
                    .ok_or(KvTableError::NonResident { index });
            }
        }
        unreachable!("published mapping covers [0, mapped_len)");
    }

    pub fn flatten(&self, ws: WorkingSetId) -> Result<Vec<PhysicalKvPageId>, KvTableError> {
        let entry = self.entry(ws)?;
        let mapped_len = entry.mapped_len;
        let mut out = Vec::with_capacity(mapped_len as usize);
        let segs = self.segments(entry.terminal, mapped_len);
        for seg in segs.iter().rev() {
            let from = (-seg.start).max(0) as u64;
            for local in from..seg.len {
                out.push(
                    self.resolve_id(seg.node, local)
                        .ok_or(KvTableError::NonResident {
                            index: out.len() as u64,
                        })?,
                );
            }
        }
        debug_assert_eq!(out.len() as u64, mapped_len);
        Ok(out)
    }

    pub fn chain_state(&self, ws: WorkingSetId) -> Result<Option<Hash256>, KvTableError> {
        Ok(self.entry(ws)?.chain_state)
    }

    pub fn set_chain_state(
        &mut self,
        ws: WorkingSetId,
        state: Option<Hash256>,
    ) -> Result<(), KvTableError> {
        self.entry_mut(ws)?.chain_state = state;
        Ok(())
    }

    pub fn page_token_hashes(
        &self,
        ws: WorkingSetId,
        index: u64,
    ) -> Result<Vec<Option<Hash256>>, KvTableError> {
        let entry = self.entry(ws)?;
        if index >= entry.mapped_len {
            return Err(KvTableError::Unwritten {
                index,
                mapped_len: entry.mapped_len,
            });
        }
        let segs = self.segments(entry.terminal, entry.mapped_len);
        for seg in &segs {
            if (index as i64) >= seg.start {
                let local = (index as i64 - seg.start) as u64;
                let (owner, owner_local) = self.resolve_owner(seg.node, local);
                match &self.nodes.get(owner).expect("live node").pages {
                    Pages::Owned { token_hashes, .. } => {
                        return Ok(token_hashes[owner_local as usize].clone());
                    }
                    Pages::ParentSelection { .. } => unreachable!("resolved to owner"),
                }
            }
        }
        unreachable!("published mapping covers [0, mapped_len)");
    }

    pub fn visible_page_identities(
        &self,
        ws: WorkingSetId,
    ) -> Result<Vec<Option<Hash256>>, KvTableError> {
        let entry = self.entry(ws)?;
        let mut out = Vec::with_capacity(entry.mapped_len as usize);
        let segs = self.segments(entry.terminal, entry.mapped_len);
        for seg in segs.iter().rev() {
            let from = (-seg.start).max(0) as u64;
            for local in from..seg.len {
                let (owner, owner_local) = self.resolve_owner(seg.node, local);
                match &self.nodes.get(owner).expect("live node").pages {
                    Pages::Owned {
                        token_hashes,
                        page_hashes,
                        ..
                    } => {
                        let i = owner_local as usize;
                        out.push(match page_hashes[i] {
                            Some(h) => Some(h),
                            None if !token_hashes[i].is_empty() => {
                                Some(hash::page_hash(&token_hashes[i]))
                            }
                            None => None,
                        });
                    }
                    Pages::ParentSelection { .. } => unreachable!("resolved to owner"),
                }
            }
        }
        Ok(out)
    }

    pub fn locate_page(&self, ws: WorkingSetId, index: u64) -> Result<(NodeId, u64), KvTableError> {
        let entry = self.entry(ws)?;
        if index >= entry.mapped_len {
            return Err(KvTableError::Unwritten {
                index,
                mapped_len: entry.mapped_len,
            });
        }
        let segs = self.segments(entry.terminal, entry.mapped_len);
        for seg in &segs {
            if (index as i64) >= seg.start {
                let local = (index as i64 - seg.start) as u64;
                return Ok(self.resolve_owner(seg.node, local));
            }
        }
        unreachable!("published mapping covers [0, mapped_len)");
    }

    pub fn page_location_alive(&self, node: NodeId, local: u64) -> bool {
        match self.nodes.get(node).map(|n| &n.pages) {
            Some(Pages::Owned { backings, .. }) => (local as usize) < backings.len(),
            _ => false,
        }
    }

    pub fn path_prefix_len(&self, node: NodeId, local: u64) -> u64 {
        let mut full = local + 1;
        let mut cursor = self.predecessor(node);
        while let Some(n) = cursor {
            full += self.contribution_len(n);
            cursor = self.predecessor(n);
        }
        full
    }

    pub fn adopt_path_prefix(
        &mut self,
        ws: WorkingSetId,
        node: NodeId,
        local: u64,
    ) -> Result<u64, KvTableError> {
        let node_len = self.contribution_len(node);
        debug_assert!(local < node_len);
        let end = self.path_prefix_len(node, local);
        let reserved_len = {
            let entry = self.entry(ws)?;
            if entry.terminal.is_some()
                || entry.mapped_len != 0
                || (entry.page_len != 0 && entry.page_len < end)
            {
                return Err(KvTableError::BadRange {
                    start: 0,
                    end,
                    mapped_len: entry.mapped_len,
                    page_len: entry.page_len,
                });
            }
            entry.page_len
        };
        let full = end + (node_len - (local + 1));
        let segs = self.segments(Some(node), full);
        let terminal = self.boundary_terminal(&segs, end);
        let freed = self.move_terminal(ws, Some(terminal))?;
        debug_assert!(freed.is_empty(), "adoption starts without an old terminal");
        let entry = self.entry_mut(ws)?;
        entry.set_page_len(reserved_len.max(end));
        entry.mapped_len = end;
        Ok(end)
    }

    pub fn is_cache_root(&self, node: NodeId) -> bool {
        self.cache_roots.contains_key(&node)
    }

    pub fn node_page_last_slot_hash(&self, node: NodeId, local: u64) -> Option<Hash256> {
        match self.nodes.get(node).map(|n| &n.pages) {
            Some(Pages::Owned { token_hashes, .. }) => token_hashes
                .get(local as usize)?
                .iter()
                .rev()
                .find_map(|h| *h),
            _ => None,
        }
    }

    pub fn page_hash_at(
        &self,
        ws: WorkingSetId,
        index: u64,
    ) -> Result<Option<Hash256>, KvTableError> {
        let entry = self.entry(ws)?;
        if index >= entry.mapped_len {
            return Err(KvTableError::Unwritten {
                index,
                mapped_len: entry.mapped_len,
            });
        }
        let segs = self.segments(entry.terminal, entry.mapped_len);
        for seg in &segs {
            if (index as i64) >= seg.start {
                let local = (index as i64 - seg.start) as u64;
                let (owner, owner_local) = self.resolve_owner(seg.node, local);
                let node = self.nodes.get(owner).expect("live node");
                match &node.pages {
                    Pages::Owned { page_hashes, .. } => {
                        return Ok(page_hashes[owner_local as usize]);
                    }
                    Pages::ParentSelection { .. } => unreachable!("resolved to owner"),
                }
            }
        }
        unreachable!("published mapping covers [0, mapped_len)");
    }

    pub fn terminal_path_hash(
        &mut self,
        ws: WorkingSetId,
    ) -> Result<Option<Hash256>, KvTableError> {
        let terminal = self.entry(ws)?.terminal;
        Ok(match terminal {
            Some(node) => self.node_path_hash(node),
            None => None,
        })
    }

    pub fn lease_cache_root(&mut self, node: NodeId) {
        let first = !self.cache_roots.contains_key(&node);
        *self.cache_roots.entry(node).or_insert(0) += 1;
        if first {
            self.add_anchor(node);
        }
    }

    pub fn release_cache_root(&mut self, node: NodeId) -> Vec<KvPageBacking> {
        let mut remove = false;
        if let Some(count) = self.cache_roots.get_mut(&node) {
            *count -= 1;
            if *count == 0 {
                remove = true;
            }
        }
        if !remove {
            return Vec::new();
        }
        self.cache_roots.remove(&node);
        self.remove_anchor(node)
    }

    pub fn pin(&mut self, node: NodeId) {
        let first = !self.pins.contains_key(&node);
        *self.pins.entry(node).or_insert(0) += 1;
        if first {
            self.add_anchor(node);
        }
    }

    pub fn unpin(&mut self, node: NodeId) -> Vec<KvPageBacking> {
        let mut remove = false;
        if let Some(count) = self.pins.get_mut(&node) {
            *count -= 1;
            if *count == 0 {
                remove = true;
            }
        }
        if !remove {
            return Vec::new();
        }
        self.pins.remove(&node);
        self.remove_anchor(node)
    }

    pub fn terminal(&self, ws: WorkingSetId) -> Result<Option<NodeId>, KvTableError> {
        Ok(self.entry(ws)?.terminal)
    }

    pub fn page_len(&self, ws: WorkingSetId) -> Result<u64, KvTableError> {
        let entry = self.entry(ws)?;
        debug_assert_eq!(
            entry.page_len_mirror.load(Ordering::Relaxed),
            entry.page_len,
            "page_len mirror drifted: a write bypassed set_page_len"
        );
        Ok(entry.page_len)
    }

    pub fn page_len_mirror(&self, ws: WorkingSetId) -> Result<Arc<AtomicU64>, KvTableError> {
        Ok(Arc::clone(&self.entry(ws)?.page_len_mirror))
    }

    pub fn mapped_len(&self, ws: WorkingSetId) -> Result<u64, KvTableError> {
        Ok(self.entry(ws)?.mapped_len)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn node_parent(&self, node: NodeId) -> Option<NodeId> {
        self.nodes.get(node).and_then(|n| n.parent)
    }

    pub fn node_is_selection(&self, node: NodeId) -> bool {
        matches!(
            self.nodes.get(node).map(|n| &n.pages),
            Some(Pages::ParentSelection { .. })
        )
    }

    pub fn node_len(&self, node: NodeId) -> u64 {
        self.contribution_len(node)
    }

    #[cfg(test)]
    pub(super) fn assert_liveness_consistent(&self) {
        let mut exact = HashMap::<NodeId, u32>::new();
        for (_, entry) in self.working_sets.iter() {
            if let Some(node) = entry.terminal {
                *exact.entry(node).or_default() += 1;
            }
        }
        for &node in self.cache_roots.keys() {
            *exact.entry(node).or_default() += 1;
        }
        for &node in self.pins.keys() {
            *exact.entry(node).or_default() += 1;
        }

        let mut path = HashMap::<NodeId, u32>::new();
        for (&node, &count) in &exact {
            let mut cursor = Some(node);
            while let Some(current) = cursor {
                *path.entry(current).or_default() += count;
                cursor = self.nodes.get(current).expect("anchor path is live").parent;
            }
        }
        assert_eq!(path.len(), self.nodes.len());
        for (node, entry) in self.nodes.iter() {
            assert_eq!(entry.exact_anchors, exact.get(&node).copied().unwrap_or(0));
            assert_eq!(entry.path_anchors, path.get(&node).copied().unwrap_or(0));
            assert!(entry.path_anchors > 0);
            assert_eq!(
                entry.path_anchors,
                entry.exact_anchors
                    + entry
                        .children
                        .iter()
                        .map(|child| self.nodes.get(*child).expect("child is live").path_anchors)
                        .sum::<u32>()
            );
        }
    }

    fn entry(&self, ws: WorkingSetId) -> Result<&WorkingSetEntry, KvTableError> {
        self.working_sets
            .get(ws)
            .ok_or(KvTableError::UnknownWorkingSet)
    }

    fn entry_mut(&mut self, ws: WorkingSetId) -> Result<&mut WorkingSetEntry, KvTableError> {
        self.working_sets
            .get_mut(ws)
            .ok_or(KvTableError::UnknownWorkingSet)
    }

    fn move_terminal(
        &mut self,
        ws: WorkingSetId,
        terminal: Option<NodeId>,
    ) -> Result<Vec<KvPageBacking>, KvTableError> {
        let previous = self.entry(ws)?.terminal;
        if previous == terminal {
            return Ok(Vec::new());
        }
        if let Some(node) = terminal {
            self.add_anchor(node);
        }
        self.entry_mut(ws)?.terminal = terminal;
        Ok(previous.map_or_else(Vec::new, |node| self.remove_anchor(node)))
    }

    fn add_anchor(&mut self, node: NodeId) {
        self.nodes
            .get_mut(node)
            .expect("anchor node is live")
            .exact_anchors += 1;
        let mut cursor = Some(node);
        while let Some(current) = cursor {
            let entry = self.nodes.get_mut(current).expect("anchor path is live");
            entry.path_anchors += 1;
            cursor = entry.parent;
        }
    }

    fn remove_anchor(&mut self, node: NodeId) -> Vec<KvPageBacking> {
        let exact = &mut self
            .nodes
            .get_mut(node)
            .expect("anchor node is live")
            .exact_anchors;
        debug_assert!(*exact > 0);
        *exact -= 1;

        let mut path = Vec::new();
        let mut cursor = Some(node);
        while let Some(current) = cursor {
            let entry = self.nodes.get_mut(current).expect("anchor path is live");
            debug_assert!(entry.path_anchors > 0);
            entry.path_anchors -= 1;
            path.push(current);
            cursor = entry.parent;
        }

        let mut freed = Vec::new();
        for &current in &path {
            freed.extend(self.detach_unreferenced(current));
        }
        for current in path {
            freed.extend(self.compact_owner(current));
        }
        freed
    }

    fn detach_unreferenced(&mut self, node: NodeId) -> Vec<KvPageBacking> {
        let Some(entry) = self.nodes.get(node) else {
            return Vec::new();
        };
        if entry.path_anchors != 0 {
            return Vec::new();
        }
        let children = entry.children.clone();
        let parent = entry.parent;
        let mut freed = Vec::new();
        for child in children {
            debug_assert_eq!(
                self.nodes
                    .get(child)
                    .expect("attached child is live")
                    .path_anchors,
                0
            );
            freed.extend(self.detach_unreferenced(child));
        }
        if let Some(parent) = parent
            && let Some(parent) = self.nodes.get_mut(parent)
        {
            parent.children.retain(|child| *child != node);
        }
        let removed = self.nodes.remove(node).expect("unreferenced node is live");
        if let Pages::Owned { backings, .. } = removed.pages {
            freed.extend(backings);
        }
        freed
    }

    fn contribution_len(&self, node: NodeId) -> u64 {
        match &self.nodes.get(node).expect("live node").pages {
            Pages::Owned { backings, .. } => backings.len() as u64,
            Pages::ParentSelection { runs } => runs_len(runs),
        }
    }

    fn predecessor(&self, node: NodeId) -> Option<NodeId> {
        let n = self.nodes.get(node).expect("live node");
        match &n.pages {
            Pages::ParentSelection { .. } => {
                let owner = n.parent.expect("selection has owner");
                self.nodes.get(owner).expect("live owner").parent
            }
            Pages::Owned { .. } => n.parent,
        }
    }

    fn segments(&self, terminal: Option<NodeId>, anchor: u64) -> Vec<Segment> {
        let mut out = Vec::new();
        let mut end = anchor as i64;
        let mut cursor = terminal;
        while let Some(node) = cursor {
            let len = self.contribution_len(node);
            let start = end - len as i64;
            out.push(Segment { node, start, len });
            if start <= 0 {
                break;
            }
            cursor = self.predecessor(node);
            end = start;
        }
        debug_assert!(
            anchor == 0 || out.last().map(|s| s.start <= 0).unwrap_or(false),
            "published mapping must cover [0, anchor)"
        );
        out
    }

    fn resolve_owner(&self, node: NodeId, local: u64) -> (NodeId, u64) {
        let n = self.nodes.get(node).expect("live node");
        match &n.pages {
            Pages::Owned { .. } => (node, local),
            Pages::ParentSelection { runs } => {
                let owner = n.parent.expect("selection has owner");
                debug_assert!(!self.node_is_selection(owner), "owner must be owned");
                (owner, runs_offset(runs, local) as u64)
            }
        }
    }

    fn resolve_location(&self, node: NodeId, local: u64) -> TriePageLocation {
        let (owner, owner_local) = self.resolve_owner(node, local);
        TriePageLocation {
            node: owner,
            local: owner_local,
        }
    }

    fn resolve_backing(&self, node: NodeId, local: u64) -> KvPageBacking {
        let location = self.resolve_location(node, local);
        match &self.nodes.get(location.node).expect("live node").pages {
            Pages::Owned { backings, .. } => backings[location.local as usize],
            Pages::ParentSelection { .. } => unreachable!("resolved to owner"),
        }
    }

    fn resolve_id(&self, node: NodeId, local: u64) -> Option<PhysicalKvPageId> {
        match self.resolve_backing(node, local) {
            KvPageBacking::Resident(id) => Some(id),
            KvPageBacking::Swapped(_) => None,
        }
    }

    fn boundary_terminal(&mut self, segs: &[Segment], boundary: u64) -> NodeId {
        let b = boundary as i64;
        debug_assert!(boundary > 0);
        let seg = segs
            .iter()
            .find(|s| s.start < b && b <= s.end())
            .copied()
            .expect("boundary within published mapping");
        if b == seg.end() {
            return seg.node;
        }
        let off = (b - seg.start) as u64;
        self.make_prefix_selection(seg.node, off)
    }

    fn make_prefix_selection(&mut self, node: NodeId, off: u64) -> NodeId {
        let (owner, runs) = {
            let n = self.nodes.get(node).expect("live node");
            match &n.pages {
                Pages::Owned { backings, .. } => {
                    debug_assert!(off < backings.len() as u64);
                    let runs: Runs = smallvec![0..off as u32];
                    (node, runs)
                }
                Pages::ParentSelection { runs } => {
                    let owner = n.parent.expect("selection has owner");
                    (owner, runs_slice(runs, 0..off))
                }
            }
        };
        self.insert_selection(owner, runs)
    }

    fn selection_excluding(&mut self, node: NodeId, a: u64, b: u64) -> NodeId {
        let (owner, runs) = {
            let n = self.nodes.get(node).expect("live node");
            match &n.pages {
                Pages::Owned { backings, .. } => {
                    let full: Runs = smallvec![0..backings.len() as u32];
                    (node, runs_remove(&full, a, b))
                }
                Pages::ParentSelection { runs } => {
                    let owner = n.parent.expect("selection has owner");
                    (owner, runs_remove(runs, a, b))
                }
            }
        };
        self.insert_selection(owner, runs)
    }

    fn insert_selection(&mut self, owner: NodeId, runs: Runs) -> NodeId {
        debug_assert!(!self.node_is_selection(owner), "owner must be owned");
        let node = self.nodes.insert(KvTrieNode {
            parent: Some(owner),
            children: SmallVec::new(),
            pages: Pages::ParentSelection { runs },
            cached_path_hash: None,
            exact_anchors: 0,
            path_anchors: 0,
        });
        self.nodes
            .get_mut(owner)
            .expect("live owner")
            .children
            .push(node);
        node
    }

    fn is_private_to(&self, ws: WorkingSetId, targets: &HashSet<NodeId>) -> bool {
        let Some(terminal) = self.entry(ws).ok().and_then(|entry| entry.terminal) else {
            return false;
        };
        self.chain_touches(terminal, targets)
            && targets.iter().all(|node| {
                self.nodes
                    .get(*node)
                    .is_some_and(|entry| entry.path_anchors == 1)
            })
    }

    fn chain_touches(&self, start: NodeId, targets: &HashSet<NodeId>) -> bool {
        let mut cursor = Some(start);
        while let Some(node) = cursor {
            if targets.contains(&node) {
                return true;
            }
            cursor = self.nodes.get(node).expect("live node").parent;
        }
        false
    }

    fn can_extend_in_place(&self, ws: WorkingSetId, terminal: NodeId) -> bool {
        let node = self.nodes.get(terminal).expect("live terminal");
        if !matches!(node.pages, Pages::Owned { .. }) || !node.children.is_empty() {
            return false;
        }
        self.entry(ws)
            .is_ok_and(|entry| entry.terminal == Some(terminal))
            && node.exact_anchors == 1
            && node.path_anchors == 1
    }

    fn invalidate_subtree(&mut self, node: NodeId) {
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            let node = self.nodes.get_mut(n).expect("live node");
            node.cached_path_hash = None;
            stack.extend(node.children.iter().copied());
        }
    }

    fn drain_node(&mut self, node: NodeId, a: u64, b: u64) -> Vec<KvPageBacking> {
        let (owner, mut freed) = {
            let n = self.nodes.get_mut(node).expect("live node");
            match &mut n.pages {
                Pages::Owned {
                    backings,
                    token_hashes,
                    page_hashes,
                } => {
                    let freed: Vec<_> = backings.drain(a as usize..b as usize).collect();
                    token_hashes.drain(a as usize..b as usize);
                    page_hashes.drain(a as usize..b as usize);
                    (None, freed)
                }
                Pages::ParentSelection { runs } => {
                    let owner = n.parent.expect("selection has owner");
                    *runs = runs_remove(runs, a, b);
                    (Some(owner), Vec::new())
                }
            }
        };
        if let Some(owner) = owner {
            freed.extend(self.compact_owner(owner));
        }
        freed
    }

    fn classify_discard(
        &self,
        ws: WorkingSetId,
        terminal: Option<NodeId>,
        mapped_len: u64,
        ranges: &[Range<u64>],
    ) -> Result<(), KvTableError> {
        let segs = self.segments(terminal, mapped_len);
        let mut sim_mapped = mapped_len;
        let mut sim_term_start: i64 = segs.first().map(|s| s.start).unwrap_or(0);
        for r in ranges {
            if r.start >= sim_mapped {
                continue;
            }
            let m = r.start..r.end.min(sim_mapped);
            let m_len = m.end - m.start;
            let affected: HashSet<NodeId> = segs
                .iter()
                .filter(|s| s.start < m.end as i64 && (m.start as i64) < s.end())
                .map(|s| s.node)
                .collect();
            if self.is_private_to(ws, &affected) {
                let removed_below = (m.end as i64)
                    .min(sim_term_start)
                    .saturating_sub(m.start as i64)
                    .max(0);
                sim_term_start -= removed_below;
                sim_mapped -= m_len;
            } else if (m.start as i64) >= sim_term_start {
                sim_mapped -= m_len;
            } else if m.end == sim_mapped {
                if m.start == 0 {
                    sim_mapped = 0;
                    sim_term_start = 0;
                } else {
                    let b = m.start as i64;
                    let seg = segs
                        .iter()
                        .find(|s| s.start < b && b <= s.end())
                        .expect("boundary within published mapping");
                    sim_term_start = seg.start;
                    sim_mapped = m.start;
                }
            } else if m.start == 0 {
                sim_mapped -= m_len;
            } else {
                return Err(KvTableError::SharedInteriorDiscard);
            }
        }
        Ok(())
    }

    fn apply_discard_range(
        &mut self,
        ws: WorkingSetId,
        r: Range<u64>,
        freed: &mut Vec<KvPageBacking>,
    ) -> Result<(), KvTableError> {
        let entry = self.entry(ws)?;
        let (terminal, mapped_len) = (entry.terminal, entry.mapped_len);
        let page_reduction = r.end - r.start;

        if r.start >= mapped_len {
            let entry = self.entry_mut(ws)?;
            entry.set_page_len(entry.page_len - page_reduction);
            return Ok(());
        }
        let m = r.start..r.end.min(mapped_len);
        let m_len = m.end - m.start;
        let segs = self.segments(terminal, mapped_len);
        let affected: Vec<Segment> = segs
            .iter()
            .filter(|s| s.start < m.end as i64 && (m.start as i64) < s.end())
            .copied()
            .collect();
        let affected_set: HashSet<NodeId> = affected.iter().map(|s| s.node).collect();

        if self.is_private_to(ws, &affected_set) {
            for seg in &affected {
                let lo = seg.start.max(m.start as i64);
                let hi = seg.end().min(m.end as i64);
                let a = (lo - seg.start) as u64;
                let b = (hi - seg.start) as u64;
                freed.extend(self.drain_node(seg.node, a, b));
            }
            let shallowest = affected.last().expect("nonempty affected").node;
            self.invalidate_subtree(shallowest);
            let entry = self.entry_mut(ws)?;
            entry.mapped_len -= m_len;
            entry.set_page_len(entry.page_len - page_reduction);
        } else if (m.start as i64) >= segs[0].start {
            let t = segs[0].node;
            let a = (m.start as i64 - segs[0].start) as u64;
            let b = (m.end as i64 - segs[0].start) as u64;
            let selection = self.selection_excluding(t, a, b);
            freed.extend(self.move_terminal(ws, Some(selection))?);
            let entry = self.entry_mut(ws)?;
            entry.mapped_len -= m_len;
            entry.set_page_len(entry.page_len - page_reduction);
        } else if m.end == mapped_len {
            let new_terminal = if m.start == 0 {
                None
            } else {
                Some(self.boundary_terminal(&segs, m.start))
            };
            freed.extend(self.move_terminal(ws, new_terminal)?);
            let entry = self.entry_mut(ws)?;
            entry.mapped_len = m.start;
            entry.set_page_len(entry.page_len - page_reduction);
        } else if m.start == 0 {
            let entry = self.entry_mut(ws)?;
            entry.mapped_len -= m_len;
            entry.set_page_len(entry.page_len - page_reduction);
        } else {
            debug_assert!(false, "discard apply diverged from legality pre-pass");
            return Err(KvTableError::SharedInteriorDiscard);
        }
        Ok(())
    }

    fn mark_chains(&self, anchors: impl IntoIterator<Item = NodeId>) -> HashSet<NodeId> {
        let mut marked: HashSet<NodeId> = HashSet::new();
        for anchor in anchors {
            let mut cursor = Some(anchor);
            while let Some(node) = cursor {
                if !marked.insert(node) {
                    break;
                }
                cursor = self.nodes.get(node).expect("live node").parent;
            }
        }
        marked
    }

    pub fn drop_unused_cache_leases(&mut self) -> (usize, Vec<KvPageBacking>) {
        if self.cache_roots.is_empty() {
            return (0, Vec::new());
        }
        let live = self.mark_chains(
            self.working_sets
                .iter()
                .filter_map(|(_, e)| e.terminal)
                .chain(self.pins.keys().copied()),
        );
        let dropped: Vec<NodeId> = self
            .cache_roots
            .keys()
            .copied()
            .filter(|node| !live.contains(node))
            .collect();
        let mut freed = Vec::new();
        for node in &dropped {
            self.cache_roots.remove(node);
            freed.extend(self.remove_anchor(*node));
        }
        (dropped.len(), freed)
    }

    pub fn reclaim_quotes(
        &self,
        groups: &[HashSet<WorkingSetId>],
        budget: u32,
    ) -> Vec<ReclaimQuote> {
        let mut quotes = Vec::with_capacity(groups.len());
        let mut covered = 0u32;
        let mut mine: FxHashSet<TriePageLocation> = FxHashSet::default();
        for group in groups {
            if covered >= budget {
                break;
            }
            mine.clear();
            for &ws in group {
                let _ = self.visit_working_set_locations(ws, |location| {
                    mine.insert(location);
                });
            }
            let quote = self.quote_group(group, &mine);
            if let ReclaimQuote::Pages(pages) = quote {
                covered = covered.saturating_add(pages);
            }
            quotes.push(quote);
        }
        quotes
    }

    fn quote_group(
        &self,
        working_sets: &HashSet<WorkingSetId>,
        mine: &FxHashSet<TriePageLocation>,
    ) -> ReclaimQuote {
        if mine.is_empty() {
            return ReclaimQuote::Nothing(NoReclaim::HoldsNothing);
        }
        if !self.locations_held_by_pins(mine).is_empty() {
            return ReclaimQuote::Nothing(NoReclaim::Pinned);
        }
        let shared_ws = match self.locations_shared_outward(working_sets, mine) {
            Ok(shared) => shared,
            Err(_) => return ReclaimQuote::Nothing(NoReclaim::AllShared),
        };
        let cache_held = self.locations_held_by_cache_roots(mine);
        let (mut pages, mut shared, mut swapped) = (0u32, 0usize, 0usize);
        for location in mine {
            if cache_held.contains(location) || shared_ws.contains(location) {
                shared += 1;
                continue;
            }
            match self.backing_at(location) {
                Ok(KvPageBacking::Resident(_)) => pages += 1,
                _ => swapped += 1,
            }
        }
        match (pages, shared, swapped) {
            (0, 0, 0) => ReclaimQuote::Nothing(NoReclaim::HoldsNothing),
            (0, shared, _) if shared > 0 => ReclaimQuote::Nothing(NoReclaim::AllShared),
            (0, _, _) => ReclaimQuote::Nothing(NoReclaim::AllSwapped),
            (pages, _, _) => ReclaimQuote::Pages(pages),
        }
    }

    pub fn private_resident_pages(
        &self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<(Vec<(TriePageLocation, PhysicalKvPageId)>, bool), KvTableError> {
        let mut target = FxHashSet::default();
        for &ws in working_sets {
            self.visit_working_set_locations(ws, |location| {
                target.insert(location);
            })?;
        }

        if !self.locations_held_by_pins(&target).is_empty() {
            return Ok((Vec::new(), true));
        }

        let shared_ws_set = self.locations_shared_outward(working_sets, &target)?;
        let cache_only = self.locations_held_by_cache_roots(&target);

        let pages: Vec<_> = target
            .into_iter()
            .filter(|location| !shared_ws_set.contains(location) && !cache_only.contains(location))
            .filter_map(|location| match self.backing_at(&location).ok()? {
                KvPageBacking::Resident(id) => Some((location, id)),
                KvPageBacking::Swapped(_) => None,
            })
            .collect();
        Ok((pages, false))
    }

    pub fn post_drain_private_resident_pages(
        &self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<Vec<(TriePageLocation, PhysicalKvPageId)>, KvTableError> {
        let mut target = FxHashSet::default();
        for &ws in working_sets {
            self.visit_working_set_locations(ws, |location| {
                target.insert(location);
            })?;
        }
        let shared_ws_set = self.locations_shared_outward(working_sets, &target)?;
        let cache_only = self.locations_held_by_cache_roots(&target);
        Ok(target
            .into_iter()
            .filter(|location| !shared_ws_set.contains(location) && !cache_only.contains(location))
            .filter_map(|location| match self.backing_at(&location).ok()? {
                KvPageBacking::Resident(id) => Some((location, id)),
                KvPageBacking::Swapped(_) => None,
            })
            .collect())
    }

    fn locations_shared_outward(
        &self,
        working_sets: &HashSet<WorkingSetId>,
        target: &FxHashSet<TriePageLocation>,
    ) -> Result<FxHashSet<TriePageLocation>, KvTableError> {
        let mut shared = FxHashSet::default();
        if target.is_empty() {
            return Ok(shared);
        }
        for (ws, _) in self.working_sets.iter() {
            if shared.len() == target.len() {
                break;
            }
            if working_sets.contains(&ws) {
                continue;
            }
            self.visit_working_set_locations(ws, |location| {
                if target.contains(&location) {
                    shared.insert(location);
                }
            })?;
        }
        Ok(shared)
    }

    fn locations_held_by_cache_roots(
        &self,
        target: &FxHashSet<TriePageLocation>,
    ) -> FxHashSet<TriePageLocation> {
        let mut held = FxHashSet::default();
        if target.is_empty() {
            return held;
        }
        for &terminal in self.cache_roots.keys() {
            if held.len() == target.len() {
                break;
            }
            self.visit_anchor_locations(terminal, |location| {
                if target.contains(&location) {
                    held.insert(location);
                }
            });
        }
        held
    }

    fn locations_held_by_pins(
        &self,
        target: &FxHashSet<TriePageLocation>,
    ) -> FxHashSet<TriePageLocation> {
        let mut held = FxHashSet::default();
        if target.is_empty() {
            return held;
        }
        for &terminal in self.pins.keys() {
            if held.len() == target.len() {
                break;
            }
            self.visit_anchor_locations(terminal, |location| {
                if target.contains(&location) {
                    held.insert(location);
                }
            });
        }
        held
    }

    pub fn swapped_pages(
        &self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<Vec<(TriePageLocation, HostKvSlotId)>, KvTableError> {
        let mut locations = HashSet::new();
        for &ws in working_sets {
            locations.extend(self.working_set_locations(ws)?);
        }
        Ok(locations
            .into_iter()
            .filter_map(|location| match self.backing_at(&location).ok()? {
                KvPageBacking::Swapped(slot) => Some((location, slot)),
                KvPageBacking::Resident(_) => None,
            })
            .collect())
    }

    pub fn held_pages(&self, working_sets: &HashSet<WorkingSetId>) -> Result<usize, KvTableError> {
        let mut locations = HashSet::new();
        for &ws in working_sets {
            locations.extend(self.working_set_locations(ws)?);
        }
        Ok(locations.len())
    }

    pub fn pin_working_sets(
        &mut self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<Vec<NodeId>, KvTableError> {
        let mut terminals = HashSet::new();
        for &ws in working_sets {
            if let Some(terminal) = self.entry(ws)?.terminal {
                terminals.insert(terminal);
            }
        }
        for &terminal in &terminals {
            self.pin(terminal);
        }
        Ok(terminals.into_iter().collect())
    }

    pub fn unpin_terminals(&mut self, terminals: &[NodeId]) -> Vec<KvPageBacking> {
        let mut freed = Vec::new();
        for &terminal in terminals {
            freed.extend(self.unpin(terminal));
        }
        freed
    }

    pub fn backing_at(&self, location: &TriePageLocation) -> Result<KvPageBacking, KvTableError> {
        let node = self
            .nodes
            .get(location.node)
            .ok_or(KvTableError::BackingChanged)?;
        match &node.pages {
            Pages::Owned { backings, .. } => backings
                .get(location.local as usize)
                .copied()
                .ok_or(KvTableError::BackingChanged),
            Pages::ParentSelection { .. } => Err(KvTableError::BackingChanged),
        }
    }

    pub fn page_location_pinned(&self, location: TriePageLocation) -> bool {
        self.swap_locations.contains_key(&location)
    }

    pub fn pin_swap_locations<I: IntoIterator<Item = TriePageLocation>>(&mut self, locations: I) {
        for location in locations {
            *self.swap_locations.entry(location).or_insert(0) += 1;
        }
    }

    pub fn unpin_swap_locations<I: IntoIterator<Item = TriePageLocation>>(
        &mut self,
        locations: I,
    ) -> Vec<KvPageBacking> {
        let mut owners = HashSet::new();
        for location in locations {
            if let Some(count) = self.swap_locations.get_mut(&location) {
                *count -= 1;
                if *count == 0 {
                    self.swap_locations.remove(&location);
                    owners.insert(location.node);
                }
            }
        }
        let mut freed = Vec::new();
        for owner in owners {
            freed.extend(self.compact_owner(owner));
        }
        freed
    }

    pub fn replace_backings(
        &mut self,
        replacements: &[(TriePageLocation, KvPageBacking, KvPageBacking)],
    ) -> Result<(), KvTableError> {
        for (location, expected, _) in replacements {
            if self.backing_at(location)? != *expected {
                return Err(KvTableError::BackingChanged);
            }
        }
        for (location, _, replacement) in replacements {
            let node = self
                .nodes
                .get_mut(location.node)
                .ok_or(KvTableError::BackingChanged)?;
            match &mut node.pages {
                Pages::Owned { backings, .. } => {
                    backings[location.local as usize] = *replacement;
                }
                Pages::ParentSelection { .. } => return Err(KvTableError::BackingChanged),
            }
        }
        Ok(())
    }

    pub fn backing_counts(&self) -> (usize, usize) {
        self.nodes
            .iter()
            .fold((0, 0), |(resident, swapped), (_, node)| match &node.pages {
                Pages::Owned { backings, .. } => {
                    backings
                        .iter()
                        .fold(
                            (resident, swapped),
                            |(resident, swapped), backing| match backing {
                                KvPageBacking::Resident(_) => (resident + 1, swapped),
                                KvPageBacking::Swapped(_) => (resident, swapped + 1),
                            },
                        )
                }
                Pages::ParentSelection { .. } => (resident, swapped),
            })
    }

    fn working_set_locations(
        &self,
        ws: WorkingSetId,
    ) -> Result<HashSet<TriePageLocation>, KvTableError> {
        let mut locations = HashSet::new();
        self.visit_working_set_locations(ws, |location| {
            locations.insert(location);
        })?;
        Ok(locations)
    }

    fn visit_working_set_locations(
        &self,
        ws: WorkingSetId,
        mut visit: impl FnMut(TriePageLocation),
    ) -> Result<(), KvTableError> {
        let entry = self.entry(ws)?;
        for segment in self.segments(entry.terminal, entry.mapped_len).iter().rev() {
            let from = (-segment.start).max(0) as u64;
            for local in from..segment.len {
                visit(self.resolve_location(segment.node, local));
            }
        }
        Ok(())
    }

    fn anchor_locations(&self, terminal: NodeId) -> HashSet<TriePageLocation> {
        let mut locations = HashSet::new();
        self.visit_anchor_locations(terminal, |location| {
            locations.insert(location);
        });
        locations
    }

    fn visit_anchor_locations(&self, terminal: NodeId, mut visit: impl FnMut(TriePageLocation)) {
        let mut cursor = Some(terminal);
        while let Some(node) = cursor {
            let trie_node = self.nodes.get(node).expect("live anchor");
            match &trie_node.pages {
                Pages::Owned { backings, .. } => {
                    for local in 0..backings.len() as u64 {
                        visit(TriePageLocation { node, local });
                    }
                }
                Pages::ParentSelection { runs } => {
                    let owner = trie_node.parent.expect("selection has owner");
                    for local in runs.iter().flat_map(|run| run.clone()) {
                        visit(TriePageLocation {
                            node: owner,
                            local: u64::from(local),
                        });
                    }
                }
            }
            cursor = self.predecessor(node);
        }
    }

    fn compact_owner(&mut self, owner: NodeId) -> Vec<KvPageBacking> {
        let Some(node) = self.nodes.get(owner) else {
            return Vec::new();
        };
        if node.exact_anchors != 0
            || self
                .swap_locations
                .keys()
                .any(|location| location.node == owner)
        {
            return Vec::new();
        }
        let (sole_child, runs) = {
            let n = self.nodes.get(owner).expect("live node");
            if !matches!(n.pages, Pages::Owned { .. }) || n.children.len() != 1 {
                return Vec::new();
            }
            let child = n.children[0];
            match &self.nodes.get(child).expect("live child").pages {
                Pages::ParentSelection { runs } => (child, runs.clone()),
                Pages::Owned { .. } => return Vec::new(),
            }
        };

        let kept: Vec<u32> = runs.iter().flat_map(|r| r.clone()).collect();
        let node = self.nodes.get_mut(owner).expect("live node");
        let (backings, token_hashes, page_hashes) = match &mut node.pages {
            Pages::Owned {
                backings,
                token_hashes,
                page_hashes,
            } => (backings, token_hashes, page_hashes),
            Pages::ParentSelection { .. } => unreachable!("checked owned"),
        };
        if kept.len() == backings.len() {
            return Vec::new();
        }

        let keep_set: HashSet<u32> = kept.iter().copied().collect();
        let mut freed = Vec::new();
        let mut new_backings = Vec::with_capacity(kept.len());
        let mut new_tokens = Vec::with_capacity(kept.len());
        let mut new_pages = Vec::with_capacity(kept.len());
        for &index in &kept {
            new_backings.push(backings[index as usize]);
            new_tokens.push(std::mem::take(&mut token_hashes[index as usize]));
            new_pages.push(page_hashes[index as usize]);
        }
        for (index, backing) in backings.iter().enumerate() {
            if !keep_set.contains(&(index as u32)) {
                freed.push(*backing);
            }
        }
        *backings = new_backings;
        *token_hashes = new_tokens;
        *page_hashes = new_pages;
        node.cached_path_hash = None;

        let kept_len = kept.len() as u32;
        match &mut self.nodes.get_mut(sole_child).expect("live child").pages {
            Pages::ParentSelection { runs } => *runs = smallvec![0..kept_len],
            Pages::Owned { .. } => unreachable!("checked selection"),
        }
        freed
    }

    fn node_path_hash(&mut self, node: NodeId) -> Option<Hash256> {
        let mut chain = Vec::new();
        let mut base: Option<Hash256> = None;
        let mut cursor = Some(node);
        while let Some(n) = cursor {
            if let Some(hash) = self.nodes.get(n).expect("live node").cached_path_hash {
                base = Some(hash);
                break;
            }
            chain.push(n);
            cursor = self.predecessor(n);
        }
        let mut acc = base;
        for &n in chain.iter().rev() {
            let pages = self.contribution_page_hashes(n)?;
            acc = hash::fold_path_hash(acc, &pages);
            if let Some(hash) = acc {
                self.nodes.get_mut(n).expect("live node").cached_path_hash = Some(hash);
            }
        }
        acc
    }

    fn contribution_page_hashes(&self, node: NodeId) -> Option<Vec<Hash256>> {
        let n = self.nodes.get(node).expect("live node");
        match &n.pages {
            Pages::Owned { page_hashes, .. } => page_hashes.iter().copied().collect(),
            Pages::ParentSelection { runs } => {
                let owner = n.parent.expect("selection has owner");
                match &self.nodes.get(owner).expect("live owner").pages {
                    Pages::Owned { page_hashes, .. } => runs
                        .iter()
                        .flat_map(|r| r.clone())
                        .map(|i| page_hashes[i as usize])
                        .collect(),
                    Pages::ParentSelection { .. } => unreachable!("owner must be owned"),
                }
            }
        }
    }
}

fn runs_len(runs: &Runs) -> u64 {
    runs.iter().map(|r| (r.end - r.start) as u64).sum()
}

fn runs_offset(runs: &Runs, mut i: u64) -> u32 {
    for r in runs {
        let len = (r.end - r.start) as u64;
        if i < len {
            return r.start + i as u32;
        }
        i -= len;
    }
    unreachable!("offset within runs");
}

fn runs_slice(runs: &Runs, sel: Range<u64>) -> Runs {
    let mut out: Runs = SmallVec::new();
    let mut pos: u64 = 0;
    for r in runs {
        let len = (r.end - r.start) as u64;
        let lo = sel.start.max(pos);
        let hi = sel.end.min(pos + len);
        if lo < hi {
            let start = r.start + (lo - pos) as u32;
            let end = r.start + (hi - pos) as u32;
            push_coalesced(&mut out, start..end);
        }
        pos += len;
    }
    out
}

fn runs_remove(runs: &Runs, a: u64, b: u64) -> Runs {
    let total = runs_len(runs);
    let mut out = runs_slice(runs, 0..a);
    for r in runs_slice(runs, b..total) {
        push_coalesced(&mut out, r);
    }
    out
}

fn push_coalesced(out: &mut Runs, range: Range<u32>) {
    if let Some(last) = out.last_mut()
        && last.end == range.start
    {
        last.end = range.end;
        return;
    }
    out.push(range);
}
