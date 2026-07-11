//! `KvPageTable`: the hash-labeled, radix-compressed KV mapping trie
//! (kv_refact.md, "Minimal WorkingSet Specification").
//!
//! The table owns the trie and the WorkingSet terminal registry. It does not
//! allocate `PhysicalKvPageId`s or call driver APIs: `KvStore` passes freshly
//! allocated ids in, and freed ids are returned to the caller for
//! epoch-delayed recycling. There is no reverse hash index and no per-page
//! refcount; lifetime is reachability from WorkingSet terminals, cache roots,
//! and in-flight snapshot pins.
//!
//! Index model: lookup anchors at the terminal. A WorkingSet's published
//! mapping covers `[0, mapped_len)`; walking backward from the terminal, each
//! node's contribution start may go negative, which is how front truncation
//! (front discard, non-prefix slice) works without a structural node.
//! `page_len >= mapped_len`; the difference is logically reserved,
//! not-yet-published space (`reserve` is purely logical).
//!
//! Growth-boundary invariant: a `Pages::ParentSelection` is created only at
//! the growth boundary of some WorkingSet mapping, and everything below a
//! selection is created after it. Pre-existing shared nodes are never
//! re-parented, so an interior discard on a shared path is rejected.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use smallvec::{SmallVec, smallvec};

use super::hash::{self, Hash256};
use crate::store::genmap::{GenKey, GenMap};
use crate::store::pool::PoolId;

/// Marker for WorkingSet ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WsMarker {}
/// Marker for trie node ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeMarker {}

pub type WorkingSetId = GenKey<WsMarker>;
pub type NodeId = GenKey<NodeMarker>;

/// A stable pool offset. Never renumbered while live; the kernel address is
/// `kv_pool_base + id * kv_page_bytes`.
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

type Runs = SmallVec<[Range<u32>; 2]>;

/// One committed page being published into the mapping.
#[derive(Debug, Clone)]
pub struct PublishedPage {
    pub id: PhysicalKvPageId,
    /// One entry per token slot of the page; `None` = unwritten/invalid.
    pub token_hashes: Vec<Option<Hash256>>,
    /// `None` while the page hash is not yet valid/committed.
    pub page_hash: Option<Hash256>,
}

enum Pages {
    Owned {
        ids: Vec<PhysicalKvPageId>,
        token_hashes: Vec<Vec<Option<Hash256>>>,
        page_hashes: Vec<Option<Hash256>>,
    },
    /// Ordered runs selecting from the parent's aligned vectors. The selected
    /// entries replace the parent's local contribution on this branch. A
    /// selection's parent is always an owned node (selection-of-selection
    /// composes runs and becomes another child of the owner).
    ParentSelection { runs: Runs },
}

struct KvTrieNode {
    parent: Option<NodeId>,
    children: SmallVec<[NodeId; 2]>,
    pages: Pages,
    cached_path_hash: Option<Hash256>,
}

/// Registry entry for one WorkingSet.
struct WorkingSetEntry {
    terminal: Option<NodeId>,
    /// Logical extent including pending (reserved, unpublished) space.
    page_len: u64,
    /// Exclusive end of the published mapping; the lookup anchor.
    ///
    /// kv_refact.md anchors lookup at `page_len`; with purely logical
    /// `reserve` the anchor must exclude reserved-unpublished space, hence
    /// this second length. (Doc to be updated.)
    mapped_len: u64,
    /// The token-slot hash the NEXT appended slot chains from: the identity
    /// of the visible content so far. `None` = empty mapping (chain start).
    /// Maintained by `KvStore` — the last committed slot hash after an
    /// append, or a recomputed visible-content identity after surgery that
    /// edits the prefix (so post-surgery appends never impersonate the
    /// unedited continuation).
    chain_state: Option<Hash256>,
    // The device-shared flattened table handle attaches here with the
    // KvStore/driver integration; the pure flatten lives in `flatten()`.
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
    #[error("publishing {count} pages exceeds the reservation (mapped {mapped_len}, page_len {page_len})")]
    PublishExceedsReservation {
        count: u64,
        mapped_len: u64,
        page_len: u64,
    },
    #[error("interior discard on a shared path is rejected (growth-boundary invariant)")]
    SharedInteriorDiscard,
}

#[derive(Debug, Clone, Copy)]
struct Segment {
    node: NodeId,
    /// Contribution start in WorkingSet coordinates; negative when `page_len`
    /// truncation hides the front of the path.
    start: i64,
    len: u64,
}

impl Segment {
    fn end(&self) -> i64 {
        self.start + self.len as i64
    }
}

/// The mapping trie and WorkingSet terminal registry.
#[derive(Default)]
pub struct KvPageTable {
    working_sets: GenMap<WsMarker, WorkingSetEntry>,
    nodes: GenMap<NodeMarker, KvTrieNode>,
    /// Cache/checkpoint lease roots (lease counts). A lease keeps an
    /// otherwise-unused subtree's path alive.
    cache_roots: HashMap<NodeId, u32>,
    /// In-flight snapshot pins (pin counts) from fires binding a terminal.
    pins: HashMap<NodeId, u32>,
}

impl KvPageTable {
    pub fn new() -> Self {
        Self::default()
    }

    // ------------------------------------------------------------------
    // WorkingSet lifecycle
    // ------------------------------------------------------------------

    pub fn create_working_set(&mut self) -> WorkingSetId {
        self.working_sets.insert(WorkingSetEntry {
            terminal: None,
            page_len: 0,
            mapped_len: 0,
            chain_state: None,
        })
    }

    /// `fork`: O(1) child over the complete logical address space. Parent and
    /// child point at the same terminal until they diverge.
    pub fn fork(&mut self, ws: WorkingSetId) -> Result<WorkingSetId, KvTableError> {
        let entry = self.entry(ws)?;
        let (terminal, page_len, mapped_len, chain_state) = (
            entry.terminal,
            entry.page_len,
            entry.mapped_len,
            entry.chain_state,
        );
        Ok(self.working_sets.insert(WorkingSetEntry {
            terminal,
            page_len,
            mapped_len,
            chain_state,
        }))
    }

    /// `slice`: structurally shared child over `range`, rebased to page zero.
    /// At most one prefix selection is created at the end boundary; the front
    /// cut is implicit in the child's smaller mapped extent.
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
        Ok(self.working_sets.insert(WorkingSetEntry {
            terminal: child_terminal,
            page_len: len,
            mapped_len: len,
            // The caller (`KvStore::slice`) derives the child's chain state:
            // inherit on a full-range slice, recompute otherwise.
            chain_state: None,
        }))
    }

    /// Purely logical reservation: extends the index space, allocates nothing.
    pub fn reserve(&mut self, ws: WorkingSetId, pages: u64) -> Result<Range<u64>, KvTableError> {
        let entry = self.entry_mut(ws)?;
        let start = entry.page_len;
        entry.page_len += pages;
        Ok(start..entry.page_len)
    }

    /// Publish committed pages at the end of the mapping (`[mapped_len,
    /// mapped_len + k)`). Extends the terminal node in place when it is
    /// private and unobserved; otherwise attaches a fresh owned child.
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

        let mut ids = Vec::with_capacity(pages.len());
        let mut token_hashes = Vec::with_capacity(pages.len());
        let mut page_hashes = Vec::with_capacity(pages.len());
        for page in pages {
            ids.push(page.id);
            token_hashes.push(page.token_hashes);
            page_hashes.push(page.page_hash);
        }

        let new_terminal = match terminal {
            Some(t) if self.can_extend_in_place(ws, t) => {
                let node = self.nodes.get_mut(t).expect("live terminal");
                match &mut node.pages {
                    Pages::Owned {
                        ids: node_ids,
                        token_hashes: node_tokens,
                        page_hashes: node_pages,
                    } => {
                        node_ids.extend(ids);
                        node_tokens.extend(token_hashes);
                        node_pages.extend(page_hashes);
                    }
                    Pages::ParentSelection { .. } => unreachable!("checked owned"),
                }
                // The node's contribution changed; its path hash (and any
                // descendants', vacuously none here) is stale.
                self.invalidate_subtree(t);
                t
            }
            _ => {
                let node = self.nodes.insert(KvTrieNode {
                    parent: terminal,
                    children: SmallVec::new(),
                    pages: Pages::Owned {
                        ids,
                        token_hashes,
                        page_hashes,
                    },
                    cached_path_hash: None,
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

        let entry = self.entry_mut(ws)?;
        entry.terminal = Some(new_terminal);
        entry.mapped_len += count;
        Ok(())
    }

    /// CoW mapping publication: rebase the mapping tail from `from` onward
    /// onto `pages` (copied tail pages plus fresh appends). The old tail
    /// remains owned by its (shared) node; `page_len` is untouched, so the
    /// fresh portion must have been reserved. This is a growth-boundary edit:
    /// the terminal moves to the boundary and new owned growth attaches below.
    pub fn replace_tail(
        &mut self,
        ws: WorkingSetId,
        from: u64,
        pages: Vec<PublishedPage>,
    ) -> Result<(), KvTableError> {
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
        if from < mapped_len {
            let segs = self.segments(terminal, mapped_len);
            let new_terminal = if from == 0 {
                None
            } else {
                Some(self.boundary_terminal(&segs, from))
            };
            let entry = self.entry_mut(ws)?;
            entry.terminal = new_terminal;
            entry.mapped_len = from;
        }
        self.publish_appended(ws, pages)
    }

    /// Hash-lifecycle step 4: commit an in-place write to a committed page.
    /// The physical id is unchanged; only the affected page's token hashes and
    /// page hash are replaced, and cached path hashes at and below the owning
    /// node are invalidated. The caller (KvStore prepare) is responsible for
    /// having classified the page as privately writable.
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

    /// Whether the committed page at `index` may be written in place: its
    /// owning node is observed by nothing but `ws` itself (no other terminal,
    /// cache root, or in-flight snapshot pin reaches it).
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

    /// `discard`: remove `ranges` (pre-discard indexes, applied atomically)
    /// from the mapping. Returns freed physical ids (caller recycles them
    /// after the appropriate epoch). See kv_refact.md "Discard and Residency"
    /// for the private / shared case analysis; an interior range on a shared
    /// path is rejected before any mutation.
    pub fn discard(
        &mut self,
        ws: WorkingSetId,
        ranges: &[Range<u64>],
    ) -> Result<Vec<PhysicalKvPageId>, KvTableError> {
        let entry = self.entry(ws)?;
        let (terminal, page_len, mapped_len) = (entry.terminal, entry.page_len, entry.mapped_len);

        // Normalize: sort descending by start, merge overlaps, validate.
        let mut norm: Vec<Range<u64>> = ranges
            .iter()
            .filter(|r| r.start < r.end)
            .cloned()
            .collect();
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
        merged.reverse(); // process back to front so shifts never move pending ranges

        // Legality pre-pass: reject SharedInteriorDiscard before any mutation.
        self.classify_discard(ws, terminal, mapped_len, &merged)?;

        let mut freed = Vec::new();
        for r in &merged {
            self.apply_discard_range(ws, r.clone(), &mut freed)?;
        }
        freed.extend(self.collect());
        Ok(freed)
    }

    /// Drop a WorkingSet and reclaim everything only it kept alive.
    pub fn release_working_set(&mut self, ws: WorkingSetId) -> Vec<PhysicalKvPageId> {
        if self.working_sets.remove(ws).is_none() {
            return Vec::new();
        }
        self.collect()
    }

    // ------------------------------------------------------------------
    // Lookup
    // ------------------------------------------------------------------

    pub fn lookup(
        &self,
        ws: WorkingSetId,
        index: u64,
    ) -> Result<PhysicalKvPageId, KvTableError> {
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
                return Ok(self.resolve_id(seg.node, local));
            }
        }
        unreachable!("published mapping covers [0, mapped_len)");
    }

    /// The full logical-to-physical mapping, `mapped_len` entries. This is
    /// the source for the device-shared flattened table; the runtime performs
    /// this walk once per mapping change, kernels only read the result.
    pub fn flatten(&self, ws: WorkingSetId) -> Result<Vec<PhysicalKvPageId>, KvTableError> {
        let entry = self.entry(ws)?;
        let mapped_len = entry.mapped_len;
        let mut out = Vec::with_capacity(mapped_len as usize);
        let segs = self.segments(entry.terminal, mapped_len);
        for seg in segs.iter().rev() {
            let from = (-seg.start).max(0) as u64; // clip hidden front
            for local in from..seg.len {
                out.push(self.resolve_id(seg.node, local));
            }
        }
        debug_assert_eq!(out.len() as u64, mapped_len);
        Ok(out)
    }

    /// The chain state the next appended slot hashes from (see
    /// [`WorkingSetEntry::chain_state`]).
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

    /// The committed token-slot hashes of the page at `index` (one entry per
    /// slot, `None` = unwritten).
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

    /// Per-visible-page identity, mapping order: the committed page hash,
    /// else a fold of the page's recorded token-slot hashes, else `None`
    /// (nothing recorded — the caller substitutes an opaque draw). Feeds the
    /// post-surgery chain-state recompute.
    pub fn visible_page_identities(
        &self,
        ws: WorkingSetId,
    ) -> Result<Vec<Option<Hash256>>, KvTableError> {
        let entry = self.entry(ws)?;
        let mut out = Vec::with_capacity(entry.mapped_len as usize);
        let segs = self.segments(entry.terminal, entry.mapped_len);
        for seg in segs.iter().rev() {
            let from = (-seg.start).max(0) as u64; // clip hidden front
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

    /// Locate the owning node + node-local index of the page at `index` (CAS
    /// index bookkeeping).
    pub fn locate_page(
        &self,
        ws: WorkingSetId,
        index: u64,
    ) -> Result<(NodeId, u64), KvTableError> {
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

    /// Whether a CAS-index entry still points at a live owned page slot.
    pub fn page_location_alive(&self, node: NodeId, local: u64) -> bool {
        match self.nodes.get(node).map(|n| &n.pages) {
            Some(Pages::Owned { ids, .. }) => (local as usize) < ids.len(),
            _ => false,
        }
    }

    /// Pages on the structural path from the trie root through page `local`
    /// of `node`, inclusive (selection predecessors skip their owners, like
    /// every walk).
    pub fn path_prefix_len(&self, node: NodeId, local: u64) -> u64 {
        let mut full = local + 1;
        let mut cursor = self.predecessor(node);
        while let Some(n) = cursor {
            full += self.contribution_len(n);
            cursor = self.predecessor(n);
        }
        full
    }

    /// Prefix-cache graft: adopt the structural path prefix ending at page
    /// `local` of `node` as the complete mapping of the EMPTY WorkingSet
    /// `ws`. The path becomes the child's visible mapping (structurally
    /// shared — writes CoW like any shared path); a mid-node boundary gets
    /// one prefix selection, exactly slice's end-boundary mechanism. Returns
    /// the adopted page count. The caller owns chain-state bookkeeping.
    pub fn adopt_path_prefix(
        &mut self,
        ws: WorkingSetId,
        node: NodeId,
        local: u64,
    ) -> Result<u64, KvTableError> {
        {
            let entry = self.entry(ws)?;
            if entry.mapped_len != 0 || entry.page_len != 0 {
                return Err(KvTableError::BadRange {
                    start: 0,
                    end: 0,
                    mapped_len: entry.mapped_len,
                    page_len: entry.page_len,
                });
            }
        }
        let node_len = self.contribution_len(node);
        debug_assert!(local < node_len);
        let end = self.path_prefix_len(node, local);
        let full = end + (node_len - (local + 1));
        let segs = self.segments(Some(node), full);
        let terminal = self.boundary_terminal(&segs, end);
        let entry = self.entry_mut(ws)?;
        entry.terminal = Some(terminal);
        entry.page_len = end;
        entry.mapped_len = end;
        Ok(end)
    }

    /// Whether `node` currently holds a cache-root lease.
    pub fn is_cache_root(&self, node: NodeId) -> bool {
        self.cache_roots.contains_key(&node)
    }

    /// The last committed token-slot hash of an owned page, addressed by node
    /// + node-local index (CAS lookup validation: owner compaction can shift
    /// locals, so an index entry must re-prove its content before use).
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

    /// Page hash of the page at `index`, if valid/committed.
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

    /// Cached path hash at the WorkingSet's terminal: folds all visible page
    /// hashes from the trie root through the terminal's contribution,
    /// independent of node boundaries. `None` when any contributing page hash
    /// is not yet valid (or the path is empty). Lazily computed and cached.
    pub fn terminal_path_hash(&mut self, ws: WorkingSetId) -> Result<Option<Hash256>, KvTableError> {
        let terminal = self.entry(ws)?.terminal;
        Ok(match terminal {
            Some(node) => self.node_path_hash(node),
            None => None,
        })
    }

    // ------------------------------------------------------------------
    // Anchors: cache roots and in-flight snapshot pins
    // ------------------------------------------------------------------

    pub fn lease_cache_root(&mut self, node: NodeId) {
        *self.cache_roots.entry(node).or_insert(0) += 1;
    }

    /// Caller should run [`Self::collect`] afterwards to reclaim.
    pub fn release_cache_root(&mut self, node: NodeId) {
        if let Some(count) = self.cache_roots.get_mut(&node) {
            *count -= 1;
            if *count == 0 {
                self.cache_roots.remove(&node);
            }
        }
    }

    pub fn pin(&mut self, node: NodeId) {
        *self.pins.entry(node).or_insert(0) += 1;
    }

    /// Caller should run [`Self::collect`] afterwards to reclaim.
    pub fn unpin(&mut self, node: NodeId) {
        if let Some(count) = self.pins.get_mut(&node) {
            *count -= 1;
            if *count == 0 {
                self.pins.remove(&node);
            }
        }
    }

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------

    pub fn terminal(&self, ws: WorkingSetId) -> Result<Option<NodeId>, KvTableError> {
        Ok(self.entry(ws)?.terminal)
    }

    pub fn page_len(&self, ws: WorkingSetId) -> Result<u64, KvTableError> {
        Ok(self.entry(ws)?.page_len)
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

    // ------------------------------------------------------------------
    // Internal: path walk and resolution
    // ------------------------------------------------------------------

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

    fn contribution_len(&self, node: NodeId) -> u64 {
        match &self.nodes.get(node).expect("live node").pages {
            Pages::Owned { ids, .. } => ids.len() as u64,
            Pages::ParentSelection { runs } => runs_len(runs),
        }
    }

    /// The logical predecessor of a node's contribution: a selection replaces
    /// its parent's local contribution, so it skips past the parent.
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

    /// Backward walk from the terminal, anchored so the terminal's
    /// contribution ends at `anchor`. Stops once coverage reaches index 0.
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

    /// Resolve a node-local offset to the owning node and its local index.
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

    fn resolve_id(&self, node: NodeId, local: u64) -> PhysicalKvPageId {
        let (owner, owner_local) = self.resolve_owner(node, local);
        match &self.nodes.get(owner).expect("live node").pages {
            Pages::Owned { ids, .. } => ids[owner_local as usize],
            Pages::ParentSelection { .. } => unreachable!("resolved to owner"),
        }
    }

    /// Terminal for a boundary at index `boundary` (> 0): the node whose
    /// contribution ends there, or a prefix selection of the node containing
    /// it (composed against the owner when that node is itself a selection).
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

    /// A selection over the first `off` entries of `node`'s contribution.
    fn make_prefix_selection(&mut self, node: NodeId, off: u64) -> NodeId {
        let (owner, runs) = {
            let n = self.nodes.get(node).expect("live node");
            match &n.pages {
                Pages::Owned { ids, .. } => {
                    debug_assert!(off < ids.len() as u64);
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

    /// A selection over `node`'s contribution excluding local `[a, b)`.
    fn selection_excluding(&mut self, node: NodeId, a: u64, b: u64) -> NodeId {
        let (owner, runs) = {
            let n = self.nodes.get(node).expect("live node");
            match &n.pages {
                Pages::Owned { ids, .. } => {
                    let full: Runs = smallvec![0..ids.len() as u32];
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
        });
        self.nodes
            .get_mut(owner)
            .expect("live owner")
            .children
            .push(node);
        node
    }

    // ------------------------------------------------------------------
    // Internal: privacy and mutation
    // ------------------------------------------------------------------

    /// True when no anchor other than `ws` itself reaches any node in
    /// `targets`. An anchor reaches a node when the node is on the anchor's
    /// trie-parent chain (which retains selection owners transitively).
    fn is_private_to(&self, ws: WorkingSetId, targets: &HashSet<NodeId>) -> bool {
        for &anchor in self.cache_roots.keys().chain(self.pins.keys()) {
            if self.chain_touches(anchor, targets) {
                return false;
            }
        }
        for (id, entry) in self.working_sets.iter() {
            if id == ws {
                continue;
            }
            if let Some(t) = entry.terminal {
                if self.chain_touches(t, targets) {
                    return false;
                }
            }
        }
        true
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

    /// In-place terminal extension is allowed only when nothing but `ws`
    /// observes the node: owned, childless, no lease/pin, no other terminal.
    fn can_extend_in_place(&self, ws: WorkingSetId, terminal: NodeId) -> bool {
        let node = self.nodes.get(terminal).expect("live terminal");
        if !matches!(node.pages, Pages::Owned { .. }) || !node.children.is_empty() {
            return false;
        }
        if self.cache_roots.contains_key(&terminal) || self.pins.contains_key(&terminal) {
            return false;
        }
        !self
            .working_sets
            .iter()
            .any(|(id, entry)| id != ws && entry.terminal == Some(terminal))
    }

    /// Clear cached path hashes on `node` and its whole subtree.
    fn invalidate_subtree(&mut self, node: NodeId) {
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            let node = self.nodes.get_mut(n).expect("live node");
            node.cached_path_hash = None;
            stack.extend(node.children.iter().copied());
        }
    }

    /// Drain local `[a, b)` from a node's contribution. Owned nodes free
    /// their slots; selections just rewrite runs.
    fn drain_node(&mut self, node: NodeId, a: u64, b: u64) -> Vec<PhysicalKvPageId> {
        let n = self.nodes.get_mut(node).expect("live node");
        match &mut n.pages {
            Pages::Owned {
                ids,
                token_hashes,
                page_hashes,
            } => {
                let freed: Vec<_> = ids.drain(a as usize..b as usize).collect();
                token_hashes.drain(a as usize..b as usize);
                page_hashes.drain(a as usize..b as usize);
                freed
            }
            Pages::ParentSelection { runs } => {
                *runs = runs_remove(runs, a, b);
                Vec::new()
            }
        }
    }

    // ------------------------------------------------------------------
    // Internal: discard
    // ------------------------------------------------------------------

    /// Legality pre-pass over `ranges` (descending, disjoint): simulates only
    /// `(mapped_len, terminal-span start)` and rejects any range that would
    /// need to reroute a shared suffix. Runs before any mutation so `discard`
    /// is atomic.
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
                continue; // logical-only
            }
            let m = r.start..r.end.min(sim_mapped);
            let m_len = m.end - m.start;
            let affected: HashSet<NodeId> = segs
                .iter()
                .filter(|s| s.start < m.end as i64 && (m.start as i64) < s.end())
                .map(|s| s.node)
                .collect();
            if self.is_private_to(ws, &affected) {
                let removed_below =
                    (m.end as i64).min(sim_term_start).saturating_sub(m.start as i64).max(0);
                sim_term_start -= removed_below;
                sim_mapped -= m_len;
            } else if (m.start as i64) >= sim_term_start {
                sim_mapped -= m_len; // within terminal contribution
            } else if m.end == sim_mapped {
                // tail-reaching: terminal moves up to the boundary
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
                sim_mapped -= m_len; // front truncation
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
        freed: &mut Vec<PhysicalKvPageId>,
    ) -> Result<(), KvTableError> {
        let entry = self.entry(ws)?;
        let (terminal, page_len, mapped_len) = (entry.terminal, entry.page_len, entry.mapped_len);
        let page_reduction = r.end - r.start;

        if r.start >= mapped_len {
            self.entry_mut(ws)?.page_len -= page_reduction;
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
            // Content below the shallowest affected node shifted; all cached
            // path hashes at and below it are stale.
            let shallowest = affected.last().expect("nonempty affected").node;
            self.invalidate_subtree(shallowest);
            let entry = self.entry_mut(ws)?;
            entry.mapped_len -= m_len;
            entry.page_len -= page_reduction;
        } else if (m.start as i64) >= segs[0].start {
            // Within the terminal node's contribution: replace the terminal
            // with a selection excluding the range.
            let t = segs[0].node;
            let a = (m.start as i64 - segs[0].start) as u64;
            let b = (m.end as i64 - segs[0].start) as u64;
            let selection = self.selection_excluding(t, a, b);
            let entry = self.entry_mut(ws)?;
            entry.terminal = Some(selection);
            entry.mapped_len -= m_len;
            entry.page_len -= page_reduction;
        } else if m.end == mapped_len {
            // Tail-reaching above the terminal: move the terminal up.
            let new_terminal = if m.start == 0 {
                None
            } else {
                Some(self.boundary_terminal(&segs, m.start))
            };
            let entry = self.entry_mut(ws)?;
            entry.terminal = new_terminal;
            entry.mapped_len = m.start;
            entry.page_len -= page_reduction;
        } else if m.start == 0 {
            // Front-reaching: pure truncation; the anchor shift re-bases all
            // surviving indexes. Excluded pages stay on the ancestor path.
            let entry = self.entry_mut(ws)?;
            entry.mapped_len -= m_len;
            entry.page_len -= page_reduction;
        } else {
            debug_assert!(false, "discard apply diverged from legality pre-pass");
            return Err(KvTableError::SharedInteriorDiscard);
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Reachability collection and owner compaction
    // ------------------------------------------------------------------

    /// Mark every node on the parent chain of each anchor (anchors retain
    /// their whole prefix path; selection owners are parents, so they are
    /// retained transitively).
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

    /// Contention-ladder rung 1 (kv_refact.md, Scheduler): drop cache-root
    /// leases on prefixes no WorkingSet terminal or in-flight pin reaches —
    /// they are retained ONLY by the lease, so reclaiming them loses no work.
    /// Returns the number of lease roots dropped; the caller runs
    /// [`Self::collect`] to free their pages.
    pub fn drop_unused_cache_leases(&mut self) -> usize {
        if self.cache_roots.is_empty() {
            return 0;
        }
        let live = self.mark_chains(
            self.working_sets
                .iter()
                .filter_map(|(_, e)| e.terminal)
                .chain(self.pins.keys().copied()),
        );
        let before = self.cache_roots.len();
        self.cache_roots.retain(|node, _| live.contains(node));
        before - self.cache_roots.len()
    }

    /// FCFS victim sizing (kv_refact.md, Scheduler): the pages reachable ONLY
    /// from `ws`'s terminal — its private suffix. Shared prefixes are
    /// excluded because preempting one sharer never frees them; a terminal
    /// pinned by an in-flight fire counts as shared (nothing frees until the
    /// fire finalizes).
    pub fn exclusive_footprint(&self, ws: WorkingSetId) -> Result<u64, KvTableError> {
        let Some(terminal) = self.entry(ws)?.terminal else {
            return Ok(0);
        };
        let others = self.mark_chains(
            self.working_sets
                .iter()
                .filter_map(|(id, e)| if id == ws { None } else { e.terminal })
                .chain(self.cache_roots.keys().copied())
                .chain(self.pins.keys().copied()),
        );
        let mut pages = 0u64;
        let mut cursor = Some(terminal);
        while let Some(node) = cursor {
            if others.contains(&node) {
                break; // everything above here is shared
            }
            if let Pages::Owned { ids, .. } = &self.nodes.get(node).expect("live node").pages {
                pages += ids.len() as u64;
            }
            cursor = self.nodes.get(node).expect("live node").parent;
        }
        Ok(pages)
    }

    /// Mark-and-sweep from all anchors (WorkingSet terminals, cache roots,
    /// pins), then apply the owner-compaction rule. Returns freed ids.
    pub fn collect(&mut self) -> Vec<PhysicalKvPageId> {
        let mut exact_anchors: HashSet<NodeId> = HashSet::new();
        for (_, entry) in self.working_sets.iter() {
            if let Some(t) = entry.terminal {
                exact_anchors.insert(t);
            }
        }
        exact_anchors.extend(self.cache_roots.keys().copied());
        exact_anchors.extend(self.pins.keys().copied());

        let marked = self.mark_chains(exact_anchors.iter().copied());

        let unmarked: Vec<NodeId> = self.nodes.keys().filter(|n| !marked.contains(n)).collect();
        let mut freed = Vec::new();
        for node in unmarked {
            let removed = self.nodes.remove(node).expect("live node");
            if let Pages::Owned { ids, .. } = removed.pages {
                freed.extend(ids);
            }
        }
        // Prune dangling child edges on survivors.
        let survivors: Vec<NodeId> = self.nodes.keys().collect();
        for node in &survivors {
            let live: SmallVec<[NodeId; 2]> = {
                let n = self.nodes.get(*node).expect("live node");
                n.children
                    .iter()
                    .copied()
                    .filter(|c| marked.contains(c))
                    .collect()
            };
            self.nodes.get_mut(*node).expect("live node").children = live;
        }

        // Owner compaction: an owned node whose node-local pages are consumed
        // only by its sole direct selection donates the selected entries and
        // frees the excluded slots. The selection's runs become the identity;
        // node identities are not merged (unary merge is a separate, later
        // concern).
        for node in survivors {
            if !self.nodes.contains(node) {
                continue;
            }
            freed.extend(self.try_compact_owner(node, &exact_anchors));
        }
        freed
    }

    fn try_compact_owner(
        &mut self,
        owner: NodeId,
        exact_anchors: &HashSet<NodeId>,
    ) -> Vec<PhysicalKvPageId> {
        if exact_anchors.contains(&owner) {
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
        let (ids, token_hashes, page_hashes) = match &mut node.pages {
            Pages::Owned {
                ids,
                token_hashes,
                page_hashes,
            } => (ids, token_hashes, page_hashes),
            Pages::ParentSelection { .. } => unreachable!("checked owned"),
        };
        if kept.len() == ids.len() {
            return Vec::new(); // full-coverage selection: nothing to free
        }

        let keep_set: HashSet<u32> = kept.iter().copied().collect();
        let mut freed = Vec::new();
        let mut new_ids = Vec::with_capacity(kept.len());
        let mut new_tokens = Vec::with_capacity(kept.len());
        let mut new_pages = Vec::with_capacity(kept.len());
        for &index in &kept {
            new_ids.push(ids[index as usize]);
            new_tokens.push(std::mem::take(&mut token_hashes[index as usize]));
            new_pages.push(page_hashes[index as usize]);
        }
        for (index, id) in ids.iter().enumerate() {
            if !keep_set.contains(&(index as u32)) {
                freed.push(*id);
            }
        }
        *ids = new_ids;
        *token_hashes = new_tokens;
        *page_hashes = new_pages;
        node.cached_path_hash = None; // full contribution changed (unobserved)

        let kept_len = kept.len() as u32;
        match &mut self.nodes.get_mut(sole_child).expect("live child").pages {
            Pages::ParentSelection { runs } => *runs = smallvec![0..kept_len],
            Pages::Owned { .. } => unreachable!("checked selection"),
        }
        freed
    }

    // ------------------------------------------------------------------
    // Internal: path hashes
    // ------------------------------------------------------------------

    fn node_path_hash(&mut self, node: NodeId) -> Option<Hash256> {
        // Collect the uncached chain (terminal-ward to root-ward), then fold
        // forward. Iterative to keep deep paths off the call stack.
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

    /// The node's contribution as page hashes; `None` if any is invalid.
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

// ----------------------------------------------------------------------
// Runs helpers (selection-space arithmetic over ordered ranges)
// ----------------------------------------------------------------------

fn runs_len(runs: &Runs) -> u64 {
    runs.iter().map(|r| (r.end - r.start) as u64).sum()
}

/// Owner index of selection-space offset `i`.
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

/// Selection-space slice `[sel.start, sel.end)` of `runs`.
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

/// `runs` with selection-space `[a, b)` removed.
fn runs_remove(runs: &Runs, a: u64, b: u64) -> Runs {
    let total = runs_len(runs);
    let mut out = runs_slice(runs, 0..a);
    for r in runs_slice(runs, b..total) {
        push_coalesced(&mut out, r);
    }
    out
}

fn push_coalesced(out: &mut Runs, range: Range<u32>) {
    if let Some(last) = out.last_mut() {
        if last.end == range.start {
            last.end = range.end;
            return;
        }
    }
    out.push(range);
}
