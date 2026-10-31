use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::Range;

use rustc_hash::FxHashSet;
use smallvec::{SmallVec, smallvec};

use super::hash::Hash256;
use crate::inferlet::process::ProcessId;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DiskKvSlotId(pub u32);

impl PoolId for DiskKvSlotId {
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
    OnDisk(DiskKvSlotId),
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
    exact_anchors: u32,
    path_anchors: u32,
}

struct WorkingSetEntry {
    terminal: Option<NodeId>,
    page_len: u64,
    mapped_len: u64,
    suspended: bool,
    folded: u64,
    /// Past them a decode wrote, and the cache lets that go first.
    prompt_pages: u64,
    owner: Option<ProcessId>,
    supplied: u64,
    /// As far as every write since is known.
    logged: Vec<u32>,
    /// One that never settles leaves the log cut there.
    pending: BTreeMap<u64, u64>,
    /// Pages later rows no longer read.
    forgotten: BTreeSet<u64>,
}

impl WorkingSetEntry {
    fn new(terminal: Option<NodeId>, page_len: u64, mapped_len: u64) -> Self {
        Self {
            terminal,
            page_len,
            mapped_len,
            suspended: false,
            folded: 0,
            prompt_pages: 0,
            owner: None,
            supplied: 0,
            logged: Vec::new(),
            pending: BTreeMap::new(),
            forgotten: BTreeSet::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct KeptPath {
    pub(super) terminal: NodeId,
    pub(super) pages: u64,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum KvTableError {
    #[error("unknown working set")]
    UnknownWorkingSet,
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
    #[error("working set page {index} is swapped out")]
    NonResident { index: u64 },
    #[error("page backing changed while a residency transaction was in flight")]
    BackingChanged,
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
    swap_locations: HashSet<TriePageLocation>,
    node_starts: HashMap<Hash256, SmallVec<[NodeId; 2]>>,
}

/// Pages sealed apart are copies, and one may be off the device.
const NODE_COPIES: usize = 4;

impl KvPageTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_working_set(&mut self) -> WorkingSetId {
        self.working_sets.insert(WorkingSetEntry::new(None, 0, 0))
    }

    pub fn fork(&mut self, ws: WorkingSetId) -> Result<WorkingSetId, KvTableError> {
        let entry = self.entry(ws)?;
        let (terminal, page_len, mapped_len) = (entry.terminal, entry.page_len, entry.mapped_len);
        // The child holds what the parent's log knows, short of what the
        // parent's fires on their way may still change, and what it froze.
        let mut logged = entry.logged.clone();
        if let Some(&cut) = entry.pending.values().min() {
            logged.truncate(cut as usize);
        }
        let (folded, prompt_pages) = (entry.folded, entry.prompt_pages);
        let forgotten = entry.forgotten.clone();
        let mut child_entry = WorkingSetEntry::new(terminal, page_len, mapped_len);
        child_entry.forgotten = forgotten;
        child_entry.logged = logged;
        child_entry.folded = folded;
        child_entry.prompt_pages = prompt_pages;
        let child = self.working_sets.insert(child_entry);
        if let Some(terminal) = terminal {
            self.add_anchor(terminal);
        }
        Ok(child)
    }

    /// Rows written after this read less than their tokens say, so the log
    /// stops at `start`: they must not publish as the prompt's own state.
    pub fn forget(
        &mut self,
        ws: WorkingSetId,
        pages: Range<u64>,
        start: u64,
    ) -> Result<(), KvTableError> {
        let entry = self.entry_mut(ws)?;
        entry.forgotten.extend(pages);
        entry.logged.truncate(start as usize);
        Ok(())
    }

    pub fn forgotten(&self, ws: WorkingSetId) -> Result<&BTreeSet<u64>, KvTableError> {
        Ok(&self.entry(ws)?.forgotten)
    }

    /// What follows `start` is dropped first; logged only where the log reaches
    /// `start`.
    pub fn log_rows(
        &mut self,
        ws: WorkingSetId,
        start: u64,
        tokens: &[u32],
    ) -> Result<(), KvTableError> {
        let logged = &mut self.entry_mut(ws)?.logged;
        logged.truncate(start as usize);
        if logged.len() as u64 == start {
            logged.extend_from_slice(tokens);
        }
        Ok(())
    }

    pub fn log_begin(&mut self, ws: WorkingSetId, seq: u64, cut: u64) -> Result<(), KvTableError> {
        self.entry_mut(ws)?.pending.insert(seq, cut);
        Ok(())
    }

    pub fn log_end(
        &mut self,
        ws: WorkingSetId,
        seq: u64,
        rows: Option<(u64, &[u32])>,
    ) -> Result<(), KvTableError> {
        let entry = self.entry_mut(ws)?;
        let Some(cut) = entry.pending.remove(&seq) else {
            return Ok(());
        };
        match rows {
            Some((start, tokens)) => self.log_rows(ws, start, tokens),
            None => {
                entry.logged.truncate(cut as usize);
                Ok(())
            }
        }
    }

    pub fn logged(&self, ws: WorkingSetId) -> Result<&[u32], KvTableError> {
        let entry = self.entry(ws)?;
        let end = entry
            .pending
            .values()
            .min()
            .map_or(entry.logged.len(), |&cut| {
                entry.logged.len().min(cut as usize)
            });
        Ok(&entry.logged[..end])
    }

    pub fn sealed_prefix(&self, ws: WorkingSetId) -> Result<u64, KvTableError> {
        self.prefix_while(ws, |location| self.sealed_at(location))
    }

    fn prefix_while(
        &self,
        ws: WorkingSetId,
        kept: impl Fn(TriePageLocation) -> bool,
    ) -> Result<u64, KvTableError> {
        let (mut pages, mut open) = (0, false);
        self.visit_working_set_locations(ws, |location| {
            open = open || !kept(location);
            pages += u64::from(!open);
        })?;
        Ok(pages)
    }

    fn sealed_at(&self, location: TriePageLocation) -> bool {
        matches!(
            self.nodes.get(location.node).map(|n| &n.pages),
            Some(Pages::Owned { page_hashes, .. })
                if page_hashes.get(location.local as usize).is_some_and(Option::is_some)
        )
    }

    fn settled_at(&self, location: &TriePageLocation) -> bool {
        !self.swap_locations.contains(location)
            && matches!(self.backing_at(location), Ok(KvPageBacking::Resident(_)))
    }

    pub fn reserve(&mut self, ws: WorkingSetId, pages: u64) -> Result<Range<u64>, KvTableError> {
        let entry = self.entry_mut(ws)?;
        let start = entry.page_len;
        entry.page_len += pages;
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
        let (owner, owner_local) = self.page_owner(ws, index)?;
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
        Ok(())
    }

    pub fn folded(&self, ws: WorkingSetId) -> Result<u64, KvTableError> {
        Ok(self.entry(ws)?.folded)
    }

    /// What is frozen stays frozen.
    pub fn freeze(&mut self, ws: WorkingSetId, to: u64) -> Result<u64, KvTableError> {
        let entry = self.entry_mut(ws)?;
        entry.folded = entry.folded.max(to);
        Ok(entry.folded)
    }

    pub fn prompt_pages(&self, ws: WorkingSetId) -> Result<u64, KvTableError> {
        Ok(self.entry(ws)?.prompt_pages)
    }

    pub fn reach_prompt(&mut self, ws: WorkingSetId, pages: u64) -> Result<(), KvTableError> {
        let entry = self.entry_mut(ws)?;
        entry.prompt_pages = entry.prompt_pages.max(pages);
        Ok(())
    }

    /// Node by node: the copy of a node start that runs furthest, then the node
    /// its last key ends or branches at.
    pub fn sealed_run(&self, keys: &[Hash256], resident: bool) -> Vec<TriePageLocation> {
        let mut run = Vec::new();
        while let Some(copies) = keys
            .get(run.len())
            .and_then(|key| self.node_starts.get(key))
        {
            let from = run.len();
            let Some((node, matched)) = copies
                .iter()
                .map(|&node| (node, self.node_run(node, &keys[from..], resident)))
                .max_by_key(|&(_, matched)| matched)
                .filter(|&(_, matched)| matched > 0)
            else {
                break;
            };
            run.extend((0..matched as u64).map(|local| TriePageLocation { node, local }));
        }
        run
    }

    fn node_run(&self, node: NodeId, keys: &[Hash256], resident: bool) -> usize {
        let Some(Pages::Owned {
            page_hashes,
            backings,
            ..
        }) = self.nodes.get(node).map(|n| &n.pages)
        else {
            return 0;
        };
        page_hashes
            .iter()
            .zip(backings)
            .zip(keys)
            .enumerate()
            .take_while(|&(local, ((hash, backing), key))| {
                *hash == Some(*key)
                    && (!resident || matches!(backing, KvPageBacking::Resident(_)))
                    && !self.swap_locations.contains(&TriePageLocation {
                        node,
                        local: local as u64,
                    })
            })
            .count()
    }

    #[cfg(test)]
    pub fn sealed_page_at(
        &self,
        location: TriePageLocation,
        key: Hash256,
    ) -> Option<PhysicalKvPageId> {
        let Some(Pages::Owned {
            page_hashes,
            backings,
            ..
        }) = self.nodes.get(location.node).map(|n| &n.pages)
        else {
            return None;
        };
        let local = location.local as usize;
        match (page_hashes.get(local), backings.get(local)) {
            (Some(Some(hash)), Some(KvPageBacking::Resident(id))) if *hash == key => Some(*id),
            _ => None,
        }
    }

    /// Whole, computed and keyed, so it never changes in place.
    pub fn sealed(&self, ws: WorkingSetId, index: u64) -> Result<bool, KvTableError> {
        Ok(self.page_hash_at(ws, index)?.is_some())
    }

    pub fn commit_hashes(
        &mut self,
        ws: WorkingSetId,
        first: u64,
        pages: Vec<(Vec<Option<Hash256>>, Option<Hash256>)>,
    ) -> Result<(), KvTableError> {
        let entry = self.entry(ws)?;
        let end = first + pages.len() as u64;
        if end > entry.mapped_len {
            return Err(KvTableError::Unwritten {
                index: end - 1,
                mapped_len: entry.mapped_len,
            });
        }
        let segs = self.segments(entry.terminal, entry.mapped_len);
        for (offset, (tokens, page)) in pages.into_iter().enumerate() {
            let index = first + offset as u64;
            let seg = segs
                .iter()
                .find(|s| (index as i64) >= s.start)
                .copied()
                .expect("published mapping covers [0, mapped_len)");
            let (owner, local) = self.resolve_owner(seg.node, (index as i64 - seg.start) as u64);
            match &mut self.nodes.get_mut(owner).expect("live node").pages {
                Pages::Owned {
                    token_hashes,
                    page_hashes,
                    ..
                } => {
                    token_hashes[local as usize] = tokens;
                    page_hashes[local as usize] = page;
                }
                Pages::ParentSelection { .. } => unreachable!("resolved to owner"),
            }
            if let Some(key) = page.filter(|_| local == 0) {
                self.index_node_start(key, owner);
            }
        }
        Ok(())
    }

    /// The latest few copies that still start with it.
    fn index_node_start(&mut self, key: Hash256, node: NodeId) {
        let mut copies = self.node_starts.remove(&key).unwrap_or_default();
        copies.retain(|copy| {
            *copy != node
                && matches!(
                    self.nodes.get(*copy).map(|n| &n.pages),
                    Some(Pages::Owned { page_hashes, .. }) if page_hashes.first() == Some(&Some(key))
                )
        });
        copies.push(node);
        if copies.len() > NODE_COPIES {
            copies.remove(0);
        }
        self.node_starts.insert(key, copies);
    }

    pub fn privately_writable(&self, ws: WorkingSetId, index: u64) -> Result<bool, KvTableError> {
        let (owner, local) = self.page_owner(ws, index)?;
        Ok(self.page_readers(owner, local) == 1)
    }

    /// An owned child takes all of them, a selection its runs.
    fn page_readers(&self, owner: NodeId, local: u64) -> u32 {
        let node = self.nodes.get(owner).expect("live node");
        let mut readers = node.exact_anchors;
        for &child in &node.children {
            let child = self.nodes.get(child).expect("live child");
            let takes = match &child.pages {
                Pages::Owned { .. } => true,
                Pages::ParentSelection { runs } => {
                    runs.iter().any(|run| run.contains(&(local as u32)))
                }
            };
            if takes {
                readers += child.path_anchors;
            }
        }
        readers
    }

    #[cfg(test)]
    pub fn release_working_set(&mut self, ws: WorkingSetId) -> Vec<PhysicalKvPageId> {
        self.release_working_set_backings(ws)
            .into_iter()
            .filter_map(|backing| match backing {
                KvPageBacking::Resident(id) => Some(id),
                KvPageBacking::Swapped(_) | KvPageBacking::OnDisk(_) => {
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
            .terminal
            .map_or_else(Vec::new, |terminal| self.remove_anchor(terminal))
    }

    #[cfg(test)]
    pub fn lookup(&self, ws: WorkingSetId, index: u64) -> Result<PhysicalKvPageId, KvTableError> {
        let (owner, local) = self.page_owner(ws, index)?;
        self.resolve_id(owner, local)
            .ok_or(KvTableError::NonResident { index })
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

    pub fn page_token_hashes(
        &self,
        ws: WorkingSetId,
        index: u64,
    ) -> Result<Vec<Option<Hash256>>, KvTableError> {
        let (owner, local) = self.page_owner(ws, index)?;
        match &self.nodes.get(owner).expect("live node").pages {
            Pages::Owned { token_hashes, .. } => Ok(token_hashes[local as usize].clone()),
            Pages::ParentSelection { .. } => unreachable!("resolved to owner"),
        }
    }

    pub fn path_locations(&self, node: NodeId, local: u64) -> Vec<TriePageLocation> {
        let end = self.path_prefix_len(node, local);
        let full = end + (self.contribution_len(node) - (local + 1));
        let segs = self.segments(Some(node), full);
        let mut out = Vec::with_capacity(end as usize);
        for seg in segs.iter().rev() {
            for at in seg.start.max(0)..seg.end().min(end as i64) {
                out.push(self.resolve_location(seg.node, (at - seg.start) as u64));
            }
        }
        out
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

    /// The pages `ws` holds are the path's own, and none past them is written.
    pub fn attach(
        &mut self,
        ws: WorkingSetId,
        node: NodeId,
        local: u64,
    ) -> Result<u64, KvTableError> {
        let node_len = self.contribution_len(node);
        debug_assert!(local < node_len);
        let end = self.path_prefix_len(node, local);
        let full = end + (node_len - (local + 1));
        let segs = self.segments(Some(node), full);
        let reserved_len = {
            let entry = self.entry(ws)?;
            let held = entry.mapped_len;
            let refused = KvTableError::BadRange {
                start: held,
                end,
                mapped_len: entry.mapped_len,
                page_len: entry.page_len,
            };
            if held >= end {
                return Err(refused);
            }
            if held > 0 {
                let mut mine = Vec::with_capacity(held as usize);
                self.visit_working_set_locations(ws, |location| mine.push(location))?;
                let mut path = Vec::with_capacity(held as usize);
                for segment in segs.iter().rev() {
                    let from = (-segment.start).max(0) as u64;
                    for at in from..segment.len {
                        if path.len() as u64 == held {
                            break;
                        }
                        path.push(self.resolve_location(segment.node, at));
                    }
                }
                if mine != path {
                    return Err(refused);
                }
            }
            entry.page_len
        };
        let terminal = self.boundary_terminal(&segs, end);
        let freed = self.move_terminal(ws, Some(terminal))?;
        debug_assert!(freed.is_empty(), "the old path is a prefix of the new one");
        let entry = self.entry_mut(ws)?;
        entry.page_len = reserved_len.max(end);
        entry.mapped_len = end;
        Ok(end)
    }

    pub fn resident_run(&self, run: &[TriePageLocation]) -> usize {
        run.iter()
            .take_while(|location| self.settled_at(location))
            .count()
    }

    pub fn sealed_resident_prefix(&self, ws: WorkingSetId) -> Result<u64, KvTableError> {
        self.prefix_while(ws, |location| {
            self.sealed_at(location) && self.settled_at(&location)
        })
    }

    pub(super) fn keep_prefix_of(&mut self, path: KeptPath, pages: u64) -> KeptPath {
        let segs = self.segments(Some(path.terminal), path.pages);
        let terminal = self.boundary_terminal(&segs, pages);
        KeptPath { terminal, pages }
    }

    pub(super) fn keep_prefix(
        &mut self,
        ws: WorkingSetId,
        pages: u64,
    ) -> Result<KeptPath, KvTableError> {
        let entry = self.entry(ws)?;
        let segs = self.segments(entry.terminal, entry.mapped_len);
        let terminal = self.boundary_terminal(&segs, pages);
        Ok(KeptPath { terminal, pages })
    }

    pub fn set_owner(&mut self, ws: WorkingSetId, owner: ProcessId) -> Result<(), KvTableError> {
        self.entry_mut(ws)?.owner = Some(owner);
        Ok(())
    }

    pub fn owner(&self, ws: WorkingSetId) -> Result<Option<ProcessId>, KvTableError> {
        Ok(self.entry(ws)?.owner)
    }

    pub fn add_supplied(&mut self, ws: WorkingSetId, tokens: u64) -> Result<u64, KvTableError> {
        let entry = self.entry_mut(ws)?;
        entry.supplied += tokens;
        Ok(entry.supplied)
    }

    pub fn supplied(&self, ws: WorkingSetId) -> Result<u64, KvTableError> {
        Ok(self.entry(ws)?.supplied)
    }

    pub fn page_hash_at(
        &self,
        ws: WorkingSetId,
        index: u64,
    ) -> Result<Option<Hash256>, KvTableError> {
        let (owner, local) = self.page_owner(ws, index)?;
        match &self.nodes.get(owner).expect("live node").pages {
            Pages::Owned { page_hashes, .. } => Ok(page_hashes[local as usize]),
            Pages::ParentSelection { .. } => unreachable!("resolved to owner"),
        }
    }

    /// Pages on the path to `node`'s last page, from the first.
    pub fn path_len(&self, node: NodeId) -> u64 {
        match self.contribution_len(node) {
            0 => 0,
            len => self.path_prefix_len(node, len - 1),
        }
    }

    /// The keys of the path to `terminal`'s last page, from the first page;
    /// `None` where a page on it is not sealed.
    pub fn path_keys(&self, terminal: NodeId) -> Option<Vec<Hash256>> {
        let len = self.contribution_len(terminal);
        if len == 0 {
            return Some(Vec::new());
        }
        self.path_locations(terminal, len - 1)
            .into_iter()
            .map(|at| match &self.nodes.get(at.node)?.pages {
                Pages::Owned { page_hashes, .. } => page_hashes.get(at.local as usize).copied()?,
                Pages::ParentSelection { .. } => None,
            })
            .collect()
    }

    /// Every sealed page's key, on any tier: what this worker can give a
    /// peer that asks by key.
    pub fn sealed_keys(&self) -> std::collections::HashSet<Hash256> {
        let mut keys = std::collections::HashSet::new();
        for (_, node) in self.nodes.iter() {
            if let Pages::Owned { page_hashes, .. } = &node.pages {
                keys.extend(page_hashes.iter().flatten().copied());
            }
        }
        keys
    }

    /// A node of sealed pages that came from elsewhere. Grafted `after` the
    /// page it extends, a pass that maps the prefix maps it too: attaching
    /// takes the trie path to a run's last page. With no `after` it starts a
    /// path of its own.
    pub fn insert_cached(
        &mut self,
        after: Option<TriePageLocation>,
        backings: Vec<KvPageBacking>,
        token_hashes: Vec<Vec<Option<Hash256>>>,
        page_hashes: Vec<Hash256>,
    ) -> NodeId {
        let parent = after.map(|at| {
            let pages = match &self.nodes.get(at.node).expect("live node").pages {
                Pages::Owned { backings, .. } => backings.len() as u64,
                Pages::ParentSelection { .. } => unreachable!("a run names owned pages"),
            };
            if at.local + 1 == pages {
                at.node
            } else {
                self.make_prefix_selection(at.node, at.local + 1)
            }
        });
        let first = page_hashes.first().copied();
        let node = self.nodes.insert(KvTrieNode {
            parent,
            children: SmallVec::new(),
            pages: Pages::Owned {
                backings,
                token_hashes,
                page_hashes: page_hashes.into_iter().map(Some).collect(),
            },
            exact_anchors: 0,
            path_anchors: 0,
        });
        if let Some(parent) = parent {
            self.nodes
                .get_mut(parent)
                .expect("live parent")
                .children
                .push(node);
        }
        if let Some(key) = first {
            self.index_node_start(key, node);
        }
        node
    }

    pub fn lease_cache_root(&mut self, node: NodeId) {
        if hold(&mut self.cache_roots, node) {
            self.add_anchor(node);
        }
    }

    pub fn release_cache_root(&mut self, node: NodeId) -> Vec<KvPageBacking> {
        if !let_go(&mut self.cache_roots, node) {
            return Vec::new();
        }
        self.remove_anchor(node)
    }

    pub fn pin(&mut self, node: NodeId) {
        if hold(&mut self.pins, node) {
            self.add_anchor(node);
        }
    }

    pub fn unpin(&mut self, node: NodeId) -> Vec<KvPageBacking> {
        if !let_go(&mut self.pins, node) {
            return Vec::new();
        }
        self.remove_anchor(node)
    }

    #[cfg(test)]
    pub fn terminal(&self, ws: WorkingSetId) -> Result<Option<NodeId>, KvTableError> {
        Ok(self.entry(ws)?.terminal)
    }

    pub fn page_len(&self, ws: WorkingSetId) -> Result<u64, KvTableError> {
        Ok(self.entry(ws)?.page_len)
    }

    pub fn mapped_len(&self, ws: WorkingSetId) -> Result<u64, KvTableError> {
        Ok(self.entry(ws)?.mapped_len)
    }

    pub fn within(&self, root: NodeId, location: TriePageLocation, len: u64) -> bool {
        let Some(node) = self.nodes.get(root) else {
            return false;
        };
        let (owner, last) = match &node.pages {
            Pages::Owned { backings, .. } => (root, backings.len() as u64),
            Pages::ParentSelection { runs } => (
                node.parent.expect("selection has owner"),
                runs.iter().map(|run| u64::from(run.end)).max().unwrap_or(0),
            ),
        };
        last > 0
            && self.descends_from(location.node, owner)
            && (owner != location.node || last <= location.local + 1)
            && self.path_prefix_len(owner, last - 1) <= len
    }

    pub fn descends_from(&self, node: NodeId, ancestor: NodeId) -> bool {
        let mut cursor = Some(node);
        while let Some(current) = cursor {
            if current == ancestor {
                return true;
            }
            cursor = self.nodes.get(current).and_then(|n| n.parent);
        }
        false
    }

    #[cfg(test)]
    pub fn node_count(&self) -> usize {
        self.nodes.iter().count()
    }

    #[cfg(test)]
    pub fn node_parent(&self, node: NodeId) -> Option<NodeId> {
        self.nodes.get(node).and_then(|n| n.parent)
    }

    pub fn node_is_selection(&self, node: NodeId) -> bool {
        matches!(
            self.nodes.get(node).map(|n| &n.pages),
            Some(Pages::ParentSelection { .. })
        )
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

    fn page_owner(&self, ws: WorkingSetId, index: u64) -> Result<(NodeId, u64), KvTableError> {
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
            .expect("published mapping covers [0, mapped_len)");
        Ok(self.resolve_owner(seg.node, (index as i64 - seg.start) as u64))
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
        if let Pages::Owned { page_hashes, .. } = &self.nodes.get(node).expect("live").pages
            && let Some(&Some(first)) = page_hashes.first()
            && let Some(copies) = self.node_starts.get_mut(&first)
        {
            copies.retain(|copy| *copy != node);
            if copies.is_empty() {
                self.node_starts.remove(&first);
            }
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
            KvPageBacking::Swapped(_) | KvPageBacking::OnDisk(_) => None,
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

    fn insert_selection(&mut self, owner: NodeId, runs: Runs) -> NodeId {
        debug_assert!(!self.node_is_selection(owner), "owner must be owned");
        let node = self.nodes.insert(KvTrieNode {
            parent: Some(owner),
            children: SmallVec::new(),
            pages: Pages::ParentSelection { runs },
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

    /// `None` for all of it, else the runs their selections take of it.
    fn live_reads(&self) -> HashMap<NodeId, Option<Vec<Range<u32>>>> {
        let anchors = self
            .working_sets
            .iter()
            .filter_map(|(_, e)| e.terminal)
            .chain(self.pins.keys().copied());
        let mut reads: HashMap<NodeId, Option<Vec<Range<u32>>>> = HashMap::new();
        for anchor in anchors {
            let (mut cursor, mut read) = (anchor, None);
            loop {
                let seen = reads.contains_key(&cursor);
                let entry = reads.entry(cursor).or_insert_with(|| Some(Vec::new()));
                match (entry, read) {
                    (Some(runs), Some(more)) => runs.extend(more),
                    (entry, None) => *entry = None,
                    (None, Some(_)) => {}
                }
                // What a node passes up to its parent depends on the node
                // alone, so a chain already walked from here is done.
                if seen {
                    break;
                }
                let node = self.nodes.get(cursor).expect("live node");
                read = match &node.pages {
                    Pages::ParentSelection { runs } => Some(runs.to_vec()),
                    Pages::Owned { .. } => None,
                };
                match node.parent {
                    Some(parent) => cursor = parent,
                    None => break,
                }
            }
        }
        reads
    }

    /// A prefix a live working set published is a selection beside its chain:
    /// dropping it would cost the key and free none of its pages.
    pub(super) fn unused_cache_roots(&self) -> Vec<NodeId> {
        if self.cache_roots.is_empty() {
            return Vec::new();
        }
        let reads = self.live_reads();
        self.cache_roots
            .keys()
            .copied()
            .filter(|&root| {
                let node = self.nodes.get(root).expect("live node");
                let (source, runs) = match &node.pages {
                    Pages::ParentSelection { runs } => {
                        (node.parent.expect("selection has owner"), runs.clone())
                    }
                    Pages::Owned { backings, .. } => (root, smallvec![0..backings.len() as u32]),
                };
                match reads.get(&source) {
                    None => true,
                    Some(None) => false,
                    Some(Some(read)) => !covers(read, &runs),
                }
            })
            .collect()
    }

    pub fn unused_cache_lease_pages(&self) -> u32 {
        self.device_pages_freed(&self.unused_cache_roots())
    }

    #[cfg(test)]
    pub fn drop_unused_cache_leases(&mut self) -> (usize, Vec<KvPageBacking>) {
        let dropped = self.unused_cache_roots();
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
        let staying = match self.staying(working_sets, mine) {
            Ok(staying) => staying,
            Err(_) => return ReclaimQuote::Nothing(NoReclaim::AllShared),
        };
        let (mut pages, mut shared, mut swapped) = (0u32, 0usize, 0usize);
        for location in mine {
            if staying.contains(location) {
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

    /// pie cannot compute a dropped page again, so what a suspended process
    /// reads must stay somewhere, and the device is for what runs.
    pub fn movable_resident_pages(
        &self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<Vec<(TriePageLocation, PhysicalKvPageId)>, KvTableError> {
        let target = self.locations(working_sets)?;
        let staying = self.staying(working_sets, &target)?;
        Ok(target
            .into_iter()
            .filter(|location| !staying.contains(location))
            .filter_map(|location| match self.backing_at(&location).ok()? {
                KvPageBacking::Resident(id) => Some((location, id)),
                KvPageBacking::Swapped(_) | KvPageBacking::OnDisk(_) => None,
            })
            .collect())
    }

    /// Only those pages stay; the rest of `working_sets` can still go.
    fn staying(
        &self,
        working_sets: &HashSet<WorkingSetId>,
        target: &FxHashSet<TriePageLocation>,
    ) -> Result<FxHashSet<TriePageLocation>, KvTableError> {
        let mut staying = self.locations_read_by_running(working_sets, target)?;
        staying.extend(self.locations_held_by_pins(target));
        Ok(staying)
    }

    /// A running reader kept them on the device and has gone since.
    pub fn stranded_pages(
        &self,
        keep: &HashSet<WorkingSetId>,
    ) -> Vec<(TriePageLocation, PhysicalKvPageId)> {
        let mut running = FxHashSet::default();
        for (ws, entry) in self.working_sets.iter() {
            if !entry.suspended || keep.contains(&ws) {
                let _ = self.visit_working_set_locations(ws, |location| {
                    running.insert(location);
                });
            }
        }
        for &terminal in self.pins.keys() {
            self.visit_anchor_locations(terminal, |location| {
                running.insert(location);
            });
        }
        // Reader by reader, so a sweep short of room moves whole readers.
        let (mut seen, mut pages) = (FxHashSet::default(), Vec::new());
        for (ws, entry) in self.working_sets.iter() {
            if !entry.suspended {
                continue;
            }
            let _ = self.visit_working_set_locations(ws, |location| {
                if running.contains(&location) || !seen.insert(location) {
                    return;
                }
                if let Ok(KvPageBacking::Resident(id)) = self.backing_at(&location) {
                    pages.push((location, id));
                }
            });
        }
        pages
    }

    /// What the cache would drop, it can keep below the device instead.
    pub fn cache_only_pages(
        &self,
        roots: &[NodeId],
        above: &[NodeId],
    ) -> Vec<(TriePageLocation, PhysicalKvPageId)> {
        if roots.is_empty() {
            return Vec::new();
        }
        let mut read = FxHashSet::default();
        for (ws, _) in self.working_sets.iter() {
            let _ = self.visit_working_set_locations(ws, |location| {
                read.insert(location);
            });
        }
        for &terminal in self.pins.keys().chain(above) {
            self.visit_anchor_locations(terminal, |location| {
                read.insert(location);
            });
        }
        let (mut seen, mut pages) = (FxHashSet::default(), Vec::new());
        for &root in roots {
            if !self.nodes.contains(root) {
                continue;
            }
            self.visit_anchor_locations(root, |location| {
                if read.contains(&location)
                    || self.swap_locations.contains(&location)
                    || !seen.insert(location)
                {
                    return;
                }
                if let Ok(KvPageBacking::Resident(id)) = self.backing_at(&location) {
                    pages.push((location, id));
                }
            });
        }
        pages
    }

    fn path_nodes(&self, root: NodeId) -> Vec<NodeId> {
        let mut nodes = Vec::new();
        let mut cursor = self.nodes.contains(root).then_some(root);
        while let Some(node) = cursor {
            nodes.push(node);
            cursor = self.nodes.get(node).and_then(|n| n.parent);
        }
        nodes
    }

    pub fn holds_resident(&self, root: NodeId) -> bool {
        self.path_nodes(root).into_iter().any(|node| {
            matches!(
                self.nodes.get(node).map(|n| &n.pages),
                Some(Pages::Owned { backings, .. })
                    if backings.iter().any(|backing| matches!(backing, KvPageBacking::Resident(_)))
            )
        })
    }

    /// A root whose pages all went below the device, or that shares them with a
    /// live reader, frees none.
    pub fn device_pages_freed(&self, roots: &[NodeId]) -> u32 {
        let (through, exact) = self.anchors(roots);
        through
            .iter()
            .map(|(&node, &dropped)| self.node_pages_freed(node, dropped, &through, &exact))
            .sum()
    }

    /// Whether dropping `roots` frees any device page: `device_pages_freed`
    /// read leaf first, stopping at the first node that frees one, so a prefix
    /// every cached prompt shares is not walked child by child for each group.
    pub fn frees_device_pages(&self, roots: &[NodeId]) -> bool {
        let (through, exact) = self.anchors(roots);
        let mut seen = FxHashSet::default();
        roots
            .iter()
            .flat_map(|&root| self.path_nodes(root))
            .any(|node| {
                seen.insert(node)
                    && self.node_pages_freed(node, through[&node], &through, &exact) > 0
            })
    }

    fn anchors(&self, roots: &[NodeId]) -> (HashMap<NodeId, u32>, HashMap<NodeId, u32>) {
        let mut through: HashMap<NodeId, u32> = HashMap::new();
        let mut exact: HashMap<NodeId, u32> = HashMap::new();
        for &root in roots {
            *exact.entry(root).or_insert(0) += 1;
            for node in self.path_nodes(root) {
                *through.entry(node).or_insert(0) += 1;
            }
        }
        (through, exact)
    }

    fn node_pages_freed(
        &self,
        node: NodeId,
        dropped: u32,
        through: &HashMap<NodeId, u32>,
        exact: &HashMap<NodeId, u32>,
    ) -> u32 {
        let Some(entry) = self.nodes.get(node) else {
            return 0;
        };
        let Pages::Owned { backings, .. } = &entry.pages else {
            return 0;
        };
        let resident = |index: &usize| matches!(backings[*index], KvPageBacking::Resident(_));
        if entry.path_anchors == dropped {
            return (0..backings.len()).filter(resident).count() as u32;
        }
        // Only selections still read it: it keeps the pages they take.
        if entry.exact_anchors != exact.get(&node).copied().unwrap_or(0)
            || self
                .swap_locations
                .iter()
                .any(|location| location.node == node)
        {
            return 0;
        }
        let mut taken = Vec::new();
        for child in &entry.children {
            let child_entry = self.nodes.get(*child).expect("live child");
            if child_entry.path_anchors == through.get(child).copied().unwrap_or(0) {
                continue;
            }
            match &child_entry.pages {
                Pages::ParentSelection { runs } => taken.extend(runs.iter().cloned()),
                Pages::Owned { .. } => return 0,
            }
        }
        let kept: HashSet<u32> = union(taken).into_iter().flatten().collect();
        (0..backings.len())
            .filter(|index| resident(index) && !kept.contains(&(*index as u32)))
            .count() as u32
    }

    /// Dropping what holds it would change pages under the move.
    pub fn in_move_at(&self, root: NodeId) -> bool {
        if self.swap_locations.is_empty() {
            return false;
        }
        let path: FxHashSet<NodeId> = self.path_nodes(root).into_iter().collect();
        self.swap_locations
            .iter()
            .any(|location| path.contains(&location.node))
    }

    pub fn suspended_readers(
        &self,
        pages: &[(TriePageLocation, PhysicalKvPageId)],
    ) -> HashSet<WorkingSetId> {
        let pages: FxHashSet<TriePageLocation> = pages.iter().map(|(at, _)| *at).collect();
        let mut readers = HashSet::new();
        for (ws, entry) in self.working_sets.iter() {
            if !entry.suspended {
                continue;
            }
            let mut reads = false;
            let _ = self.visit_working_set_locations(ws, |location| {
                reads |= pages.contains(&location);
            });
            if reads {
                readers.insert(ws);
            }
        }
        readers
    }

    pub fn set_suspended(&mut self, working_sets: &HashSet<WorkingSetId>, suspended: bool) {
        for &ws in working_sets {
            if let Ok(entry) = self.entry_mut(ws) {
                entry.suspended = suspended;
            }
        }
    }

    pub fn in_move(&self, working_sets: &HashSet<WorkingSetId>) -> Result<bool, KvTableError> {
        Ok(self
            .locations(working_sets)?
            .iter()
            .any(|location| self.swap_locations.contains(location)))
    }

    pub(super) fn settled_on_device(&self, terminal: NodeId) -> bool {
        let mut settled = true;
        self.visit_anchor_locations(terminal, |location| {
            settled &= self.settled_at(&location);
        });
        settled
    }

    fn locations_read_by_running(
        &self,
        working_sets: &HashSet<WorkingSetId>,
        target: &FxHashSet<TriePageLocation>,
    ) -> Result<FxHashSet<TriePageLocation>, KvTableError> {
        let mut shared = FxHashSet::default();
        if target.is_empty() {
            return Ok(shared);
        }
        for (ws, entry) in self.working_sets.iter() {
            if shared.len() == target.len() {
                break;
            }
            if entry.suspended || working_sets.contains(&ws) {
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
    ) -> Result<Vec<(TriePageLocation, KvPageBacking)>, KvTableError> {
        Ok(self
            .locations(working_sets)?
            .into_iter()
            .filter_map(|location| match self.backing_at(&location).ok()? {
                KvPageBacking::Resident(_) => None,
                swapped => Some((location, swapped)),
            })
            .collect())
    }

    pub fn held_pages(&self, working_sets: &HashSet<WorkingSetId>) -> Result<usize, KvTableError> {
        Ok(self.locations(working_sets)?.len())
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

    /// The tokens' hashes of the page at `location`.
    pub fn token_hashes_at(&self, location: &TriePageLocation) -> Option<&[Option<Hash256>]> {
        match &self.nodes.get(location.node)?.pages {
            Pages::Owned { token_hashes, .. } => {
                token_hashes.get(location.local as usize).map(Vec::as_slice)
            }
            Pages::ParentSelection { .. } => None,
        }
    }

    /// The key of the sealed page at `location`.
    pub fn page_key_at(&self, location: &TriePageLocation) -> Option<Hash256> {
        match &self.nodes.get(location.node)?.pages {
            Pages::Owned { page_hashes, .. } => *page_hashes.get(location.local as usize)?,
            Pages::ParentSelection { .. } => None,
        }
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
        self.swap_locations.contains(&location)
    }

    /// One move has a page at a time: every move leaves out, or waits for, the
    /// pages another holds.
    pub fn pin_swap_locations<I: IntoIterator<Item = TriePageLocation>>(&mut self, locations: I) {
        for location in locations {
            let fresh = self.swap_locations.insert(location);
            debug_assert!(fresh, "a page in two residency moves");
        }
    }

    pub fn unpin_swap_locations<I: IntoIterator<Item = TriePageLocation>>(
        &mut self,
        locations: I,
    ) -> Vec<KvPageBacking> {
        let mut owners = HashSet::new();
        for location in locations {
            if self.swap_locations.remove(&location) {
                owners.insert(location.node);
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

    fn locations(
        &self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<FxHashSet<TriePageLocation>, KvTableError> {
        let mut locations = FxHashSet::default();
        for &ws in working_sets {
            self.visit_working_set_locations(ws, |location| {
                locations.insert(location);
            })?;
        }
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
                .iter()
                .any(|location| location.node == owner)
        {
            return Vec::new();
        }
        // Below a node nothing anchors, only selections read it: it keeps
        // the pages some selection takes, and its other pages go.
        let (children, kept) = {
            let n = self.nodes.get(owner).expect("live node");
            if !matches!(n.pages, Pages::Owned { .. }) || n.children.is_empty() {
                return Vec::new();
            }
            let mut taken = Vec::new();
            for &child in &n.children {
                match &self.nodes.get(child).expect("live child").pages {
                    Pages::ParentSelection { runs } => taken.extend(runs.iter().cloned()),
                    Pages::Owned { .. } => return Vec::new(),
                }
            }
            (n.children.clone(), union(taken))
        };

        let kept: Vec<u32> = kept.iter().flat_map(|r| r.clone()).collect();
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

        // A kept page's new index is its rank among the kept, so a run of
        // consecutive pages stays one.
        for child in children {
            match &mut self.nodes.get_mut(child).expect("live child").pages {
                Pages::ParentSelection { runs } => {
                    let mut moved: Runs = SmallVec::new();
                    for r in runs.iter() {
                        let start = kept.partition_point(|&index| index < r.start) as u32;
                        push_coalesced(&mut moved, start..start + (r.end - r.start));
                    }
                    *runs = moved;
                }
                Pages::Owned { .. } => unreachable!("checked selections"),
            }
        }
        freed
    }
}

fn hold(holds: &mut HashMap<NodeId, u32>, node: NodeId) -> bool {
    let count = holds.entry(node).or_insert(0);
    *count += 1;
    *count == 1
}

fn let_go(holds: &mut HashMap<NodeId, u32>, node: NodeId) -> bool {
    let Some(count) = holds.get_mut(&node) else {
        return false;
    };
    *count -= 1;
    if *count > 0 {
        return false;
    }
    holds.remove(&node);
    true
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

fn union(mut runs: Vec<Range<u32>>) -> Vec<Range<u32>> {
    runs.retain(|r| !r.is_empty());
    runs.sort_unstable_by_key(|r| r.start);
    let mut merged: Vec<Range<u32>> = Vec::new();
    for r in runs {
        match merged.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => merged.push(r),
        }
    }
    merged
}

fn covers(read: &[Range<u32>], runs: &Runs) -> bool {
    let read = union(read.to_vec());
    runs.iter()
        .filter(|r| !r.is_empty())
        .all(|r| read.iter().any(|m| m.start <= r.start && r.end <= m.end))
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
