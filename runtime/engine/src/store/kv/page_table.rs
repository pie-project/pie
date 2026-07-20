//! `KvPageTable`: the hash-labeled, radix-compressed KV mapping trie
//! (kv_refact.md, "Minimal WorkingSet Specification").
//!
//! The table owns the trie and the WorkingSet terminal registry. It does not
//! allocate `PhysicalKvPageId`s or call driver APIs: `KvStore` passes freshly
//! allocated ids in, and freed ids are returned to the caller for
//! epoch-delayed recycling. There is no reverse hash index and no per-page
//! refcount; lifetime is reachability from WorkingSet terminals, cache roots,
//! and residency-transaction terminal pins.
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
//!
//! Complete typed-store API (kv_refact.md): some methods here are not yet
//! called by the live single-model fire path (only a subset of the typed
//! store surface is currently wired) but are exercised by this module's
//! own unit test suite and reserved for upcoming increments (contention/
//! reclaim expansion, RS buffer-write paths, etc.) — kept rather than
//! deleted, allowed rather than silently masked.
#![allow(dead_code)]

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

/// A host-pinned slot in the driver's KV swap pool.
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

/// Physical backing for one committed logical KV page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvPageBacking {
    Resident(PhysicalKvPageId),
    Swapped(HostKvSlotId),
}

/// Stable location of an owned page while a suspend/restore transaction pins
/// the process's trie terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TriePageLocation {
    pub node: NodeId,
    pub local: u64,
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
        backings: Vec<KvPageBacking>,
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
    /// Number of WorkingSet terminals plus the presence of cache-root and
    /// snapshot-pin keys anchored exactly at this node.
    exact_anchors: u32,
    /// Number of exact anchors whose structural parent chain includes this
    /// node. A zero transition makes the node unreachable immediately.
    path_anchors: u32,
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

/// Opaque-index snapshot of one fully mapped WorkingSet entry. The index owns
/// a separate cache-root lease for `terminal`; materializing the snapshot as a
/// WorkingSet adds a normal terminal anchor.
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
    /// Terminal pins held by asynchronous suspend/restore transactions.
    pins: HashMap<NodeId, u32>,
    /// Page locations whose backings are mid-swap (suspend/restore copy in
    /// flight). Unlike `pins` — collection anchors keyed by terminal — this
    /// set is PRECISE: only the locations a swap transaction is actually
    /// replacing. Its sole consumer is adoption (`page_location_pinned`):
    /// a resident page not being moved stays adoptable even while some
    /// working set on the same path is suspending.
    swap_locations: HashMap<TriePageLocation, u32>,
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
        let child = self.working_sets.insert(WorkingSetEntry {
            terminal,
            page_len,
            mapped_len,
            chain_state,
        });
        if let Some(terminal) = terminal {
            self.add_anchor(terminal);
        }
        Ok(child)
    }

    /// Capture the complete visible mapping for an opaque index. Logical
    /// reservations without backing are rejected because `from-index` must
    /// reconstruct an immediately usable WorkingSet without token semantics.
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

    /// Create a fresh WorkingSet terminal from an opaque-index snapshot.
    pub(super) fn from_index_snapshot(&mut self, snapshot: IndexedWorkingSet) -> WorkingSetId {
        let ws = self.working_sets.insert(WorkingSetEntry {
            terminal: snapshot.terminal,
            page_len: snapshot.page_len,
            mapped_len: snapshot.mapped_len,
            chain_state: snapshot.chain_state,
        });
        if let Some(terminal) = snapshot.terminal {
            self.add_anchor(terminal);
        }
        ws
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
        let child = self.working_sets.insert(WorkingSetEntry {
            terminal: child_terminal,
            page_len: len,
            mapped_len: len,
            // The caller (`KvStore::slice`) derives the child's chain state:
            // inherit on a full-range slice, recompute otherwise.
            chain_state: None,
        });
        if let Some(terminal) = child_terminal {
            self.add_anchor(terminal);
        }
        Ok(child)
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
    /// cache root, or residency pin reaches it).
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
    ) -> Result<Vec<KvPageBacking>, KvTableError> {
        let entry = self.entry(ws)?;
        let (terminal, page_len, mapped_len) = (entry.terminal, entry.page_len, entry.mapped_len);

        // Normalize: sort descending by start, merge overlaps, validate.
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
        merged.reverse(); // process back to front so shifts never move pending ranges

        // Legality pre-pass: reject SharedInteriorDiscard before any mutation.
        self.classify_discard(ws, terminal, mapped_len, &merged)?;

        let mut freed = Vec::new();
        for r in &merged {
            self.apply_discard_range(ws, r.clone(), &mut freed)?;
        }
        Ok(freed)
    }

    /// Drop a WorkingSet and reclaim everything only it kept alive.
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
            .terminal
            .map_or_else(Vec::new, |terminal| self.remove_anchor(terminal))
    }

    // ------------------------------------------------------------------
    // Lookup
    // ------------------------------------------------------------------

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

    /// Whether a CAS-index entry still points at a live owned page slot.
    pub fn page_location_alive(&self, node: NodeId, local: u64) -> bool {
        match self.nodes.get(node).map(|n| &n.pages) {
            Some(Pages::Owned { backings, .. }) => (local as usize) < backings.len(),
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
    /// `local` of `node` as the mapping prefix of an unmapped WorkingSet
    /// `ws`. Existing reserved logical tail capacity is preserved. The path
    /// becomes the child's visible mapping (structurally
    /// shared — writes CoW like any shared path); a mid-node boundary gets
    /// one prefix selection, exactly slice's end-boundary mechanism. Returns
    /// the adopted page count. The caller owns chain-state bookkeeping.
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
        entry.page_len = reserved_len.max(end);
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

    // ------------------------------------------------------------------
    // Anchors: cache roots and residency pins
    // ------------------------------------------------------------------

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

    /// A selection over `node`'s contribution excluding local `[a, b)`.
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

    // ------------------------------------------------------------------
    // Internal: privacy and mutation
    // ------------------------------------------------------------------

    /// True when no anchor other than `ws` itself reaches any node in
    /// `targets`. An anchor reaches a node when the node is on the anchor's
    /// trie-parent chain (which retains selection owners transitively).
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

    /// In-place terminal extension is allowed only when nothing but `ws`
    /// observes the node: owned, childless, no lease/pin, no other terminal.
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
                let removed_below = (m.end as i64)
                    .min(sim_term_start)
                    .saturating_sub(m.start as i64)
                    .max(0);
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
        freed: &mut Vec<KvPageBacking>,
    ) -> Result<(), KvTableError> {
        let entry = self.entry(ws)?;
        let (terminal, mapped_len) = (entry.terminal, entry.mapped_len);
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
            freed.extend(self.move_terminal(ws, Some(selection))?);
            let entry = self.entry_mut(ws)?;
            entry.mapped_len -= m_len;
            entry.page_len -= page_reduction;
        } else if m.end == mapped_len {
            // Tail-reaching above the terminal: move the terminal up.
            let new_terminal = if m.start == 0 {
                None
            } else {
                Some(self.boundary_terminal(&segs, m.start))
            };
            freed.extend(self.move_terminal(ws, new_terminal)?);
            let entry = self.entry_mut(ws)?;
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
            if let Pages::Owned { backings, .. } = &self.nodes.get(node).expect("live node").pages {
                pages += backings.len() as u64;
            }
            cursor = self.nodes.get(node).expect("live node").parent;
        }
        Ok(pages)
    }

    /// Exact private resident pages reachable by `working_sets`. Sharing
    /// between members of the set is counted once; pages visible from any
    /// outside WorkingSet, cache root, or in-flight pin are excluded.
    ///
    /// The boolean reports whether an in-flight pin overlaps the target
    /// residency. Callers defer the whole suspend transaction in that case.
    pub fn private_resident_pages(
        &self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<(Vec<(TriePageLocation, PhysicalKvPageId)>, bool), KvTableError> {
        let mut target = HashSet::new();
        for &ws in working_sets {
            target.extend(self.working_set_locations(ws)?);
        }

        let mut pinned = HashSet::new();
        for &terminal in self.pins.keys() {
            pinned.extend(self.anchor_locations(terminal));
        }
        if target.iter().any(|location| pinned.contains(location)) {
            return Ok((Vec::new(), true));
        }

        let mut external = HashSet::new();
        for (ws, _) in self.working_sets.iter() {
            if !working_sets.contains(&ws) {
                external.extend(self.working_set_locations(ws)?);
            }
        }
        for &terminal in self.cache_roots.keys() {
            external.extend(self.anchor_locations(terminal));
        }

        let pages = target
            .into_iter()
            .filter(|location| !external.contains(location))
            .filter_map(|location| match self.backing_at(&location).ok()? {
                KvPageBacking::Resident(id) => Some((location, id)),
                KvPageBacking::Swapped(_) => None,
            })
            .collect();
        Ok((pages, false))
    }

    /// Private resident pages after this process drains its own fire pins.
    /// External WorkingSets and cache roots still exclude shared pages.
    pub fn post_drain_private_resident_pages(
        &self,
        working_sets: &HashSet<WorkingSetId>,
    ) -> Result<Vec<(TriePageLocation, PhysicalKvPageId)>, KvTableError> {
        let mut target = HashSet::new();
        for &ws in working_sets {
            target.extend(self.working_set_locations(ws)?);
        }
        let mut external = HashSet::new();
        for (ws, _) in self.working_sets.iter() {
            if !working_sets.contains(&ws) {
                external.extend(self.working_set_locations(ws)?);
            }
        }
        for &terminal in self.cache_roots.keys() {
            external.extend(self.anchor_locations(terminal));
        }
        Ok(target
            .into_iter()
            .filter(|location| !external.contains(location))
            .filter_map(|location| match self.backing_at(&location).ok()? {
                KvPageBacking::Resident(id) => Some((location, id)),
                KvPageBacking::Swapped(_) => None,
            })
            .collect())
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

    /// Mark the exact locations a swap transaction is replacing; overlapping
    /// transactions stack (counted).
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
        let entry = self.entry(ws)?;
        let mut locations = HashSet::new();
        for segment in self.segments(entry.terminal, entry.mapped_len).iter().rev() {
            let from = (-segment.start).max(0) as u64;
            for local in from..segment.len {
                locations.insert(self.resolve_location(segment.node, local));
            }
        }
        Ok(locations)
    }

    fn anchor_locations(&self, terminal: NodeId) -> HashSet<TriePageLocation> {
        let mut locations = HashSet::new();
        let mut cursor = Some(terminal);
        while let Some(node) = cursor {
            let trie_node = self.nodes.get(node).expect("live anchor");
            match &trie_node.pages {
                Pages::Owned { backings, .. } => {
                    locations.extend(
                        (0..backings.len() as u64).map(|local| TriePageLocation { node, local }),
                    );
                }
                Pages::ParentSelection { runs } => {
                    let owner = trie_node.parent.expect("selection has owner");
                    locations.extend(runs.iter().flat_map(|run| run.clone()).map(|local| {
                        TriePageLocation {
                            node: owner,
                            local: u64::from(local),
                        }
                    }));
                }
            }
            cursor = self.predecessor(node);
        }
        locations
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
            return Vec::new(); // full-coverage selection: nothing to free
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
