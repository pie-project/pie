//! PageStore — Per-Device CAS Cache + Physical Page Pools (Radix Trie v2).
//!
//! Content-addressed storage for KV cache pages using a compressed Radix
//! Trie (Patricia Trie) with **path-inclusive refcounting**.
//!
//! `refcount` on each node means "how many contexts include this node in
//! their committed chain." This correctly handles:
//! - **Incremental commits**: `extend(prefix, suffix, phys)` navigates
//!   the trie through the existing prefix before inserting new suffix pages.
//! - **Dedup**: when two contexts commit the same chain, rc is bumped on all
//!   shared nodes; suspending one doesn't evict pages needed by the other.
//! - **O(depth) operations**: all operations traverse the trie path once.
//!
//! Each model device gets its own `PageStore` — no cross-device coordination.
//! Owned exclusively by the `ContextManager` actor (no interior mutability).

use std::hash::{Hash, Hasher};

use ahash::AHasher;
use rustc_hash::FxHashMap;

use crate::inference::brle::Brle;

// =============================================================================
// Types
// =============================================================================

/// Content hash of a KV cache page, chained to its predecessor.
pub type PageHash = u64;

/// Physical page index in GPU or CPU memory.
pub type PhysicalPageId = u32;

// =============================================================================
// PagePool (bulk alloc via split_off)
// =============================================================================

use rustc_hash::FxHashSet;

#[derive(Debug)]
struct PagePool {
    total: usize,
    free: Vec<PhysicalPageId>,
    alloc_total: usize,
    free_total: usize,
    free_set: FxHashSet<PhysicalPageId>,
    // Per-tag counters
    free_reclaim: usize,
    free_release: usize,
    free_working: usize,
    over_logged: bool,
}

impl PagePool {
    fn new(total: usize) -> Self {
        PagePool {
            total,
            free: (0..total as PhysicalPageId).collect(),
            alloc_total: 0,
            free_total: 0,
            free_set: (0..total as PhysicalPageId).collect(),
            free_reclaim: 0,
            free_release: 0,
            free_working: 0,
            over_logged: false,
        }
    }

    fn alloc_n(&mut self, n: usize) -> Option<Vec<PhysicalPageId>> {
        if self.free.len() < n { return None; }
        let start = self.free.len() - n;
        self.alloc_total += n;
        let pages = self.free.split_off(start);
        for &p in &pages { self.free_set.remove(&p); }
        Some(pages)
    }

    fn free_tagged(&mut self, ids: &[PhysicalPageId], tag: &str) {
        self.free_total += ids.len();
        // Track per-tag
        match tag {
            "reclaim" => self.free_reclaim += ids.len(),
            "free_gpu_pages" => self.free_working += ids.len(),
            _ if tag.starts_with("release_") => self.free_release += ids.len(),
            _ => {}
        }
        for &id in ids {
            if !self.free_set.insert(id) {
                tracing::error!("[DOUBLE_FREE] page={} tag={} free.len={} total={}",
                    id, tag, self.free.len(), self.total);
            }
            self.free.push(id);
        }
        // Log the first time free pool exceeds total
        if !self.over_logged && self.free.len() > self.total {
            self.over_logged = true;
            let last_ids: Vec<_> = ids.iter().take(5).collect();
            tracing::error!("[FIRST_OVER] tag={} free.len={} total={} ids={:?} | reclaim={} release={} working={} alloc={}",
                tag, self.free.len(), self.total, last_ids,
                self.free_reclaim, self.free_release, self.free_working, self.alloc_total);
        }
    }
    fn free(&mut self, id: PhysicalPageId) { self.free_tagged(&[id], "single"); }
    fn free_batch(&mut self, ids: &[PhysicalPageId]) { self.free_tagged(ids, "batch"); }
    fn used(&self) -> usize { self.total - self.free.len() }
    fn available(&self) -> usize { self.free.len() }
    fn pool_stats(&self) -> (usize, usize) { (self.alloc_total, self.free_total) }
    fn tag_stats(&self) -> (usize, usize, usize) { (self.free_reclaim, self.free_release, self.free_working) }
}

// =============================================================================
// TrieNode — path-inclusive refcounting
// =============================================================================

/// A node in the Radix Trie representing an unbroken chunk of cached pages.
///
/// Invariants:
/// - `prefix.len() == prefix_hashes.len()`
/// - `refcount` = number of contexts whose committed chain **includes** this node
///   (passes through OR terminates at it)
/// - A node is evictable iff `refcount == 0` — no structural protection needed
#[derive(Debug)]
struct TrieNode {
    /// Physical page IDs for this contiguous chunk.
    prefix: Vec<PhysicalPageId>,
    /// Corresponding Merkle PageHashes for each page.
    prefix_hashes: Vec<PageHash>,
    /// Path-inclusive refcount: contexts traversing or ending at this node.
    refcount: usize,
    /// Children keyed by the first PageHash of their prefix_hashes.
    children: FxHashMap<PageHash, TrieNode>,
}

impl TrieNode {
    /// Create an empty root node (no pages, no hashes).
    fn new_root() -> Self {
        TrieNode {
            prefix: Vec::new(),
            prefix_hashes: Vec::new(),
            refcount: 0,
            children: FxHashMap::default(),
        }
    }

    /// Recursive trie statistics: (total_pages, total_rc_sum, node_count, rc0_with_children)
    fn trie_stats(&self) -> (usize, usize, usize, usize) {
        let mut pages = self.prefix.len();
        let mut rc_sum = self.refcount;
        let mut nodes = if self.prefix_hashes.is_empty() { 0 } else { 1 };
        let mut rc0_interior = if self.refcount == 0 && !self.children.is_empty() && !self.prefix_hashes.is_empty() { 1 } else { 0 };
        for child in self.children.values() {
            let (p, r, n, z) = child.trie_stats();
            pages += p;
            rc_sum += r;
            nodes += n;
            rc0_interior += z;
        }
        (pages, rc_sum, nodes, rc0_interior)
    }

    // =========================================================================
    // Core helpers
    // =========================================================================

    /// Match `hashes` against this node's prefix_hashes, comparing actual values.
    /// Returns the number of leading hashes that match. Used by queries,
    /// release, and fork where correctness requires hash verification.
    #[inline]
    fn match_len(&self, hashes: &[PageHash]) -> usize {
        common_prefix_len(&self.prefix_hashes, hashes)
    }

    /// Split this node at `index`, keeping `0..index` in `self` and moving
    /// `index..` into a new child node with the same rc and all children.
    /// Returns the key (first hash) of the split-off child.
    fn split_at(&mut self, index: usize) -> PageHash {
        debug_assert!(index < self.prefix_hashes.len());
        let tail_hashes = self.prefix_hashes.split_off(index);
        let tail_phys = self.prefix.split_off(index);
        let key = tail_hashes[0];
        let tail = TrieNode {
            prefix_hashes: tail_hashes,
            prefix: tail_phys,
            refcount: self.refcount,
            children: std::mem::take(&mut self.children),
        };
        self.children.insert(key, tail);
        key
    }

    // =========================================================================
    // Extend: navigate prefix WITHOUT rc bump, insert suffix, eager merge
    // =========================================================================
    //
    // Used by `extend()` — the caller already owns the prefix (via prior
    // commit or fork). No new reference is created, so rc is NOT bumped.
    // After insert, `try_merge_child` compresses single-child nodes when
    // parent.rc == child.rc (which fires correctly because rc is not inflated).

    /// Navigate prefix (no rc bump), insert suffix, merge if possible.
    fn navigate_and_extend(
        &mut self,
        prefix: &[PageHash],
        suffix: &[PageHash],
        phys: &[PhysicalPageId],
        reclaimed_pages: &mut Vec<PhysicalPageId>,
    ) {
        if suffix.is_empty() { return; }

        if prefix.is_empty() {
            self.insert_suffix(suffix, phys, reclaimed_pages);
            return;
        }

        let first = prefix[0];
        if let Some(child) = self.children.get_mut(&first) {
            let node_len = child.prefix_hashes.len();
            let nav = prefix.len().min(node_len);

            if nav < node_len {
                // Prefix ends inside this node — split, then insert suffix.
                child.split_at(nav);
                // NO rc bump — structural split for navigation.
                child.insert_suffix(suffix, phys, reclaimed_pages);
            } else if prefix.len() == node_len {
                // Prefix exactly matches this node.
                child.insert_suffix(suffix, phys, reclaimed_pages);
                self.try_merge_child(first);
            } else {
                // Prefix extends beyond this node — recurse with remaining.
                child.navigate_and_extend(&prefix[node_len..], suffix, phys, reclaimed_pages);
                self.try_merge_child(first);
            }
        } else {
            // Prefix not in trie — broken invariant. The caller claims to own
            // this prefix, but it doesn't exist. Log and bail in release,
            // panic in debug.
            debug_assert!(false,
                "navigate_and_extend: prefix not in trie (first hash={first})");
            tracing::error!("BUG: navigate_and_extend called with prefix not in trie");
        }
    }

    /// Insert suffix hashes as a new child (rc=1) or bump rc on dedup.
    /// Uses `common_prefix_len` for splits (divergent chains need exact match).
    fn insert_suffix(
        &mut self,
        suffix: &[PageHash],
        phys: &[PhysicalPageId],
        reclaimed_pages: &mut Vec<PhysicalPageId>,
    ) {
        debug_assert!(!suffix.is_empty());
        debug_assert_eq!(suffix.len(), phys.len());

        let first = suffix[0];
        if let Some(child) = self.children.get_mut(&first) {
            let common = common_prefix_len(&child.prefix_hashes, suffix);

            if common < child.prefix_hashes.len() {
                // Partial match — split and insert divergent suffix.
                child.split_at(common);
                child.refcount += 1;
                // The common-prefix physical pages are duplicates of what's
                // already in the trie — reclaim them.
                reclaimed_pages.extend_from_slice(&phys[..common]);
                if common < suffix.len() {
                    child.children.insert(suffix[common], TrieNode {
                        prefix_hashes: suffix[common..].to_vec(),
                        prefix: phys[common..].to_vec(),
                        refcount: 1,
                        children: FxHashMap::default(),
                    });
                }
            } else if common < suffix.len() {
                // Full prefix match — the common portion's physical pages are
                // duplicates already stored in the trie node; reclaim them.
                child.refcount += 1;
                reclaimed_pages.extend_from_slice(&phys[..common]);
                child.insert_suffix(&suffix[common..], &phys[common..], reclaimed_pages);
            } else {
                // Exact match — dedup.
                child.refcount += 1;
                reclaimed_pages.extend_from_slice(phys);
            }
        } else {
            self.children.insert(first, TrieNode {
                prefix_hashes: suffix.to_vec(),
                prefix: phys.to_vec(),
                refcount: 1,
                children: FxHashMap::default(),
            });
        }
    }

    /// Merge child with its sole grandchild when they have the same rc.
    fn try_merge_child(&mut self, child_key: PageHash) {
        let child = match self.children.get_mut(&child_key) {
            Some(c) if c.children.len() == 1 => c,
            _ => return,
        };
        // Safety: we just checked children.len() == 1.
        let sole_gc = child.children.values().next().unwrap();
        if sole_gc.refcount != child.refcount { return; }
        let (_, sole) = child.children.drain().next().unwrap();
        child.prefix_hashes.extend(sole.prefix_hashes);
        child.prefix.extend(sole.prefix);
        child.children = sole.children;
    }

    // =========================================================================
    // Release — path-inclusive, bottom-up eviction
    // =========================================================================

    /// Decrement rc on every node along the path matching `hashes`.
    /// Evict nodes that reach rc=0 and have no children. Cascade upward.
    /// Returns (freed_page_count, should_remove_self).
    fn release_path(
        &mut self,
        hashes: &[PageHash],
        pool: &mut PagePool,
    ) -> (usize, bool) {
        if hashes.is_empty() { return (0, false); }

        let first = hashes[0];
        let (child_freed, remove_child) = match self.children.get_mut(&first) {
            Some(child) => {
                let matched = child.match_len(hashes);
                let node_len = child.prefix_hashes.len();

                if matched < node_len {
                    // Hash mismatch within node — this context's chain diverges.
                    // Do NOT decrement rc; these are different chains sharing
                    // a common prefix shorter than the node's compressed prefix.
                    (0, false)
                } else if hashes.len() > node_len {
                    // Full match of node — decrement rc, recurse.
                    child.refcount = child.refcount.saturating_sub(1);
                    let (deeper_freed, _) = child.release_path(
                        &hashes[node_len..], pool,
                    );
                    let evictable = child.refcount == 0 && child.children.is_empty();
                    let freed = if evictable {
                        pool.free_tagged(&child.prefix, "release_child_evict");
                        child.prefix.len() + deeper_freed
                    } else {
                        deeper_freed
                    };
                    (freed, evictable)
                } else {
                    // Exact match (tip) — decrement rc.
                    child.refcount = child.refcount.saturating_sub(1);
                    let evictable = child.refcount == 0 && child.children.is_empty();
                    let freed = if evictable {
                        pool.free_tagged(&child.prefix, "release_tip_evict");
                        child.prefix.len()
                    } else {
                        0
                    };
                    (freed, evictable)
                }
            }
            None => (0, false),
        };

        if remove_child {
            self.children.remove(&first);
        }

        // Merge: compress single-child internal nodes.
        if self.children.len() == 1 && self.refcount == 0 && !self.prefix_hashes.is_empty() {
            let (_, sole) = self.children.drain().next().unwrap();
            self.prefix.extend(sole.prefix);
            self.prefix_hashes.extend(sole.prefix_hashes);
            self.refcount = sole.refcount;
            self.children = sole.children;
        }

        let remove_self = self.refcount == 0
            && self.children.is_empty()
            && !self.prefix_hashes.is_empty();

        // NOTE: Do NOT free self.prefix here even if remove_self is true.
        // The CALLER is responsible for freeing our pages (via the
        // release_child_evict / release_tip_evict path). Freeing here would
        // cause a double-free because the caller also frees child.prefix when
        // it sees evictable=true.

        (child_freed, remove_self)
    }

    // =========================================================================
    // Lookup (hash-based matching)
    // =========================================================================

    fn prefix_match_len(&self, hashes: &[PageHash]) -> usize {
        if hashes.is_empty() { return 0; }
        match self.children.get(&hashes[0]) {
            Some(child) => {
                let matched = child.match_len(hashes);
                let node_len = child.prefix_hashes.len();
                if matched < node_len {
                    // Hash mismatch within node — only matched this many.
                    matched
                } else if hashes.len() > node_len {
                    node_len + child.prefix_match_len(&hashes[node_len..])
                } else {
                    matched // == hashes.len()
                }
            }
            None => 0,
        }
    }

    fn collect_physical(&self, hashes: &[PageHash], out: &mut Vec<PhysicalPageId>) {
        if hashes.is_empty() { return; }
        if let Some(child) = self.children.get(&hashes[0]) {
            let consumed = child.match_len(hashes);
            out.extend_from_slice(&child.prefix[..consumed]);
            if consumed == child.prefix_hashes.len() && consumed < hashes.len() {
                child.collect_physical(&hashes[consumed..], out);
            }
        }
    }

    // =========================================================================
    // Estimate reclaimable (length-based — safe for own-chain queries)
    // =========================================================================

    fn count_reclaimable_path(&self, hashes: &[PageHash]) -> usize {
        if hashes.is_empty() { return 0; }
        match self.children.get(&hashes[0]) {
            Some(child) => {
                let node_len = child.prefix_hashes.len();
                if hashes.len() < node_len {
                    0
                } else if hashes.len() > node_len {
                    let deeper = child.count_reclaimable_path(&hashes[node_len..]);
                    if child.refcount <= 1 && child.children.is_empty() {
                        child.prefix.len() + deeper
                    } else {
                        deeper
                    }
                } else {
                    if child.refcount <= 1 && child.children.is_empty() {
                        child.prefix.len()
                    } else {
                        0
                    }
                }
            }
            None => 0,
        }
    }

    /// Like `count_reclaimable_path`, but returns the actual hashes that would
    /// be freed (those with rc=1 that will reach rc=0 on release).
    /// Returns `true` if this subtree is fully reclaimable (node + all children
    /// along the path would be freed), so parent nodes can include themselves.
    fn would_free_hashes(&self, hashes: &[PageHash], out: &mut Vec<PageHash>) -> bool {
        if hashes.is_empty() { return true; }
        if let Some(child) = self.children.get(&hashes[0]) {
            let node_len = child.prefix_hashes.len();
            if hashes.len() < node_len {
                // Partial match — would not free.
                return false;
            }
            // Recurse into deeper nodes.
            let subtree_reclaimable = if hashes.len() > node_len {
                child.would_free_hashes(&hashes[node_len..], out)
            } else {
                true // Exact match at this node — no deeper children on this path.
            };
            // This node is reclaimable if rc=1 AND either:
            //   - it's a leaf (no children), OR
            //   - it has children but the child along our path is fully reclaimable
            //     AND it has no other children (so it becomes childless after release).
            let reclaimable = child.refcount <= 1
                && (child.children.is_empty()
                    || (subtree_reclaimable && child.children.len() == 1));
            if reclaimable {
                out.extend_from_slice(&child.prefix_hashes);
            }
            reclaimable
        } else {
            false // Hash not found in trie.
        }
    }

    /// Shapley effective pages along a chain: sum of `prefix.len() / refcount`
    /// per trie node touched by the path.
    fn effective_pages_path(&self, hashes: &[PageHash]) -> f64 {
        if hashes.is_empty() { return 0.0; }
        match self.children.get(&hashes[0]) {
            Some(child) => {
                let matched = child.match_len(hashes);
                let node_len = child.prefix_hashes.len();
                let rc = child.refcount.max(1) as f64;

                if matched < node_len {
                    // Partial match — only the matched portion counts.
                    matched as f64 / rc
                } else if hashes.len() > node_len {
                    // Full match — this node's cost + recurse.
                    (node_len as f64 / rc) + child.effective_pages_path(&hashes[node_len..])
                } else {
                    // Exact match (tip).
                    node_len as f64 / rc
                }
            }
            None => 0.0,
        }
    }

    /// Bump rc along the path (fork). Split-on-fork for partial hash matches.
    fn fork_path(&mut self, hashes: &[PageHash]) {
        if hashes.is_empty() { return; }
        if let Some(child) = self.children.get_mut(&hashes[0]) {
            let matched = child.match_len(hashes);
            let node_len = child.prefix_hashes.len();

            if matched < node_len {
                // Hash mismatch or partial match — split at the divergence point.
                child.split_at(matched);
                child.refcount += 1;
            } else {
                child.refcount += 1;
                if hashes.len() > node_len {
                    child.fork_path(&hashes[node_len..]);
                }
            }
        }
    }

    fn collect_phys_with_info(&self, out: &mut Vec<(PhysicalPageId, usize, usize, usize)>, depth: usize) {
        for &page in &self.prefix {
            out.push((page, self.refcount, self.prefix.len(), depth));
        }
        for child in self.children.values() {
            child.collect_phys_with_info(out, depth + 1);
        }
    }
}

/// Count how many leading elements of `a` and `b` are equal.
/// Uses an indexed loop for better auto-vectorization with u64 elements.
#[inline]
fn common_prefix_len(a: &[PageHash], b: &[PageHash]) -> usize {
    let len = a.len().min(b.len());
    for i in 0..len {
        if a[i] != b[i] { return i; }
    }
    len
}

// =============================================================================
// PageStore
// =============================================================================

/// Per-device page cache backed by a Radix Trie with path-inclusive refcounting.
///
/// Manages a single physical page pool (GPU or CPU — the PageStore is tier-agnostic)
/// and a CAS trie for content-addressed committed page sharing.
#[derive(Debug)]
pub struct PageStore {
    page_size: usize,
    root: TrieNode,
    pool: PagePool,
}

impl PageStore {
    pub fn new(page_size: usize, num_pages: usize) -> Self {
        PageStore {
            page_size,
            root: TrieNode::new_root(),
            pool: PagePool::new(num_pages),
        }
    }

    pub fn page_size(&self) -> usize { self.page_size }

    pub fn stats(&self) -> (usize, usize) {
        (self.pool.used(), self.pool.total)
    }

    pub fn available(&self) -> usize { self.pool.available() }

    /// Returns (total_pages_in_trie, total_rc_sum, node_count, rc0_interior_nodes)
    pub fn trie_stats(&self) -> (usize, usize, usize, usize) {
        self.root.trie_stats()
    }

    pub fn total_pages(&self) -> usize { self.pool.total }
    pub fn pool_stats(&self) -> (usize, usize) { self.pool.pool_stats() }
    pub fn tag_stats(&self) -> (usize, usize, usize) { self.pool.tag_stats() }

    /// Audit: count pages that exist in both trie AND the free set.
    /// These are phantom pages — they've been freed to the pool but still appear
    /// in trie nodes (double-counted). Returns (phantom_count, first_few_ids).
    pub fn phantom_audit(&self) -> (usize, String) {
        let mut all_pages = Vec::new();
        self.root.collect_phys_with_info(&mut all_pages, 0);
        let mut phantoms = Vec::new();
        for &(page, rc, prefix_len, depth) in &all_pages {
            if self.pool.free_set.contains(&page) {
                phantoms.push((page, rc, prefix_len, depth));
            }
        }
        let desc: Vec<String> = phantoms.iter().take(10)
            .map(|(p, rc, plen, d)| format!("{}(rc={},n={},d={})", p, rc, plen, d))
            .collect();
        (phantoms.len(), desc.join(" "))
    }

    // =========================================================================
    // Page allocation (bulk alloc via split_off)
    // =========================================================================

    pub fn alloc(&mut self, n: usize) -> Option<Vec<PhysicalPageId>> {
        self.pool.alloc_n(n)
    }

    pub fn free(&mut self, pages: &[PhysicalPageId]) {
        self.pool.free_tagged(pages, "free_gpu_pages");
    }

    // =========================================================================
    // Extend: append pages to an existing chain (no rc bump on prefix)
    // =========================================================================

    /// Extend an existing committed chain with new pages.
    ///
    /// Navigates through `prefix` WITHOUT bumping rc (caller already owns it),
    /// then inserts `new_hashes`/`new_phys` as new children. Eagerly merges
    /// single-child nodes when parent.rc == child.rc (trie compression).
    ///
    /// On dedup hit (suffix already exists), bumps rc and frees duplicates.
    pub fn extend(
        &mut self,
        prefix: &[PageHash],
        new_hashes: &[PageHash],
        new_phys: &[PhysicalPageId],
    ) {
        debug_assert_eq!(new_hashes.len(), new_phys.len());
        if new_hashes.is_empty() { return; }

        let mut reclaimed_pages = Vec::new();
        self.root.navigate_and_extend(prefix, new_hashes, new_phys, &mut reclaimed_pages);

        if !reclaimed_pages.is_empty() {
            self.pool.free_tagged(&reclaimed_pages, "reclaim");
        }
    }

    /// First commit — no existing prefix.
    pub fn commit_batch(&mut self, hashes: &[PageHash], phys: &[PhysicalPageId]) {
        self.extend(&[], hashes, phys);
    }

    /// Commit a single page.
    pub fn commit(&mut self, hash: PageHash, phys: PhysicalPageId) {
        self.commit_batch(&[hash], &[phys]);
    }

    // =========================================================================
    // Fork: bump rc along path (for prefix sharing)
    // =========================================================================

    /// Bump rc on every node along the path (path-inclusive).
    /// Used when a new context shares an existing prefix (restore, dedup).
    /// This is the ONLY operation that increases rc on existing nodes.
    pub fn fork(&mut self, hashes: &[PageHash]) {
        if hashes.is_empty() { return; }
        self.root.fork_path(hashes);
    }

    /// Decrement refcount along the entire path. Evict nodes that reach rc=0
    /// and have no children. Returns the number of GPU pages freed.
    pub fn release(&mut self, hashes: &[PageHash]) -> usize {
        if hashes.is_empty() { return 0; }
        let (freed, _) = self.root.release_path(hashes, &mut self.pool);
        freed
    }

    /// Estimate how many GPU pages would be freed by `release(hashes)`.
    pub fn count_reclaimable(&self, hashes: &[PageHash]) -> usize {
        if hashes.is_empty() { return 0; }
        self.root.count_reclaimable_path(hashes)
    }

    /// Return the hashes that would be freed by `release(hashes)` — i.e., pages
    /// with rc=1 and no children (will reach rc=0 on release).
    ///
    /// Read-only, O(depth). Used to identify which pages to D2H copy to CPU
    /// before calling `release()`, solving the D2H-before-release ordering problem.
    pub fn would_free(&self, hashes: &[PageHash]) -> Vec<PageHash> {
        if hashes.is_empty() { return Vec::new(); }
        let mut out = Vec::new();
        self.root.would_free_hashes(hashes, &mut out);
        out
    }

    // =========================================================================
    // Lookup
    // =========================================================================

    pub fn prefix_len(&self, hashes: &[PageHash]) -> usize {
        self.root.prefix_match_len(hashes)
    }

    pub fn physical_ids(&self, hashes: &[PageHash]) -> Vec<PhysicalPageId> {
        let mut out = Vec::with_capacity(hashes.len());
        self.root.collect_physical(hashes, &mut out);
        out
    }



    pub fn debug_info(&self, hashes: &[PageHash]) -> String {
        let prefix = self.prefix_len(hashes);
        format!("trie_v2: prefix_len={prefix}/{}", hashes.len())
    }

    /// Shapley cost-sharing: effective page count for a context's committed chain.
    ///
    /// Walks the path through the trie matching `hashes` and sums
    /// `node.pages / node.refcount` per node. Shared prefix segments are split
    /// equally among all contexts sharing them (the Shapley value for additive
    /// cost games). Unique suffix segments count at full cost.
    ///
    /// Returns the fractional effective page count.
    pub fn effective_pages(&self, hashes: &[PageHash]) -> f64 {
        if hashes.is_empty() { return 0.0; }
        self.root.effective_pages_path(hashes)
    }
}

// =============================================================================
// FlatPageStore — FlatMap-based CPU tier storage
// =============================================================================

/// CPU-tier page store using a flat hash map with refcounting.
///
/// Unlike the GPU `PageStore` (CAS Trie), this supports out-of-order insertion:
/// a suffix can be stashed before its prefix, which is the normal case when
/// contexts are suspended at different times but share a prefix.
///
/// The pool is shared across committed page stash, working page stash, and
/// snapshot swap storage.
#[derive(Debug)]
pub struct FlatPageStore {
    /// Hash → (physical CPU page ID, refcount).
    map: FxHashMap<PageHash, (PhysicalPageId, usize)>,
    /// Physical CPU page allocation pool.
    pub(crate) pool: PagePool,
}

impl FlatPageStore {
    pub fn new(num_pages: usize) -> Self {
        FlatPageStore {
            map: FxHashMap::default(),
            pool: PagePool::new(num_pages),
        }
    }

    /// Insert pages into the CPU store. On dedup hit (hash already present),
    /// bumps refcount and frees the duplicate physical page back to the pool.
    pub fn insert(&mut self, hashes: &[PageHash], phys_ids: &[PhysicalPageId]) {
        debug_assert_eq!(hashes.len(), phys_ids.len());
        for (&hash, &phys) in hashes.iter().zip(phys_ids.iter()) {
            match self.map.get_mut(&hash) {
                Some(entry) => {
                    // Dedup hit: bump rc, free the duplicate page.
                    entry.1 += 1;
                    self.pool.free(phys);
                }
                None => {
                    self.map.insert(hash, (phys, 1));
                }
            }
        }
    }

    /// Decrement refcount for each hash. Free pages reaching rc=0.
    /// Returns the number of pages freed.
    pub fn release(&mut self, hashes: &[PageHash]) -> usize {
        let mut freed = 0;
        for hash in hashes {
            if let Some(entry) = self.map.get_mut(hash) {
                entry.1 = entry.1.saturating_sub(1);
                if entry.1 == 0 {
                    let (phys, _) = self.map.remove(hash).unwrap();
                    self.pool.free(phys);
                    freed += 1;
                }
            }
        }
        freed
    }

    /// Look up physical page IDs for hashes present in the store.
    /// Returns a vec of page IDs for hashes that are found; missing hashes are skipped.
    pub fn physical_ids(&self, hashes: &[PageHash]) -> Vec<PhysicalPageId> {
        hashes.iter().filter_map(|h| self.map.get(h).map(|&(p, _)| p)).collect()
    }

    /// Count consecutive hashes (from the start) present in the CPU store.
    pub fn prefix_len(&self, hashes: &[PageHash]) -> usize {
        for (i, h) in hashes.iter().enumerate() {
            if !self.map.contains_key(h) { return i; }
        }
        hashes.len()
    }

    /// Shapley effective pages: sum of `1/rc` for each hash present in the map.
    pub fn effective_pages(&self, hashes: &[PageHash]) -> f64 {
        let mut eff = 0.0;
        for h in hashes {
            if let Some(&(_, rc)) = self.map.get(h) {
                eff += 1.0 / rc.max(1) as f64;
            }
        }
        eff
    }

    /// Check if a specific hash is present on CPU.
    pub fn contains(&self, hash: &PageHash) -> bool {
        self.map.contains_key(hash)
    }

    /// Allocate CPU pages from the shared pool.
    pub fn alloc(&mut self, n: usize) -> Option<Vec<PhysicalPageId>> {
        self.pool.alloc_n(n)
    }

    /// Free CPU pages back to the shared pool.
    pub fn free(&mut self, pages: &[PhysicalPageId]) {
        self.pool.free_batch(pages);
    }

    /// Available pages in the CPU pool.
    pub fn available(&self) -> usize {
        self.pool.available()
    }

    /// Total pages in the CPU pool.
    pub fn total(&self) -> usize {
        self.pool.total
    }

    /// Number of entries (unique hashes) in the map.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Pool stats (alloc_total, free_total).
    pub fn pool_stats(&self) -> (usize, usize) {
        self.pool.pool_stats()
    }
}

// =============================================================================
// Utility
// =============================================================================

pub fn compute_last_page_len(total_kv: u32, num_pages: u32, page_size: u32) -> u32 {
    if num_pages == 0 { 0 }
    else {
        let r = total_kv % page_size;
        if r == 0 { page_size } else { r }
    }
}

pub fn compute_page_hashes(
    page_size: usize,
    tokens: &[u32],
    positions: &[u32],
    masks: &[Brle],
    prev_hash: PageHash,
    adapter_seed: Option<i64>,
) -> Vec<PageHash> {
    let mut hashes = Vec::new();
    let mut running = prev_hash;

    for (chunk_idx, chunk) in tokens.chunks(page_size).enumerate() {
        let start = chunk_idx * page_size;
        let end = start + chunk.len();
        let chunk_pos = &positions[start..end];
        let chunk_masks = &masks[start..end];

        let mut hasher = AHasher::default();
        chunk.hash(&mut hasher);
        for pos in chunk_pos { pos.hash(&mut hasher); }
        for mask in chunk_masks { mask.hash(&mut hasher); }
        // Include adapter_seed so ZO-perturbed pages with different seeds
        // produce different hashes and are not incorrectly shared via dedup.
        adapter_seed.hash(&mut hasher);
        let content_hash = hasher.finish();

        let mut chain_hasher = AHasher::default();
        content_hash.hash(&mut chain_hasher);
        running.hash(&mut chain_hasher);
        let page_hash = chain_hasher.finish();

        hashes.push(page_hash);
        running = page_hash;
    }

    hashes
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn h(v: u64) -> PageHash { v }

    // =========================================================================
    // PagePool
    // =========================================================================

    #[test]
    fn pool_alloc_and_free() {
        let mut pool = PagePool::new(4);
        assert_eq!(pool.available(), 4);
        let pages = pool.alloc_n(2).unwrap();
        assert_eq!(pages.len(), 2);
        assert_eq!(pool.available(), 2);
        pool.free_batch(&pages);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn pool_exhaustion() {
        let mut pool = PagePool::new(2);
        let _ = pool.alloc_n(2).unwrap();
        assert!(pool.alloc_n(1).is_none());
    }

    // =========================================================================
    // Basic commit & lookup
    // =========================================================================

    #[test]
    fn commit_and_lookup_single() {
        let mut store = PageStore::new(16, 100);
        store.commit(h(1), 10);
        assert_eq!(store.prefix_len(&[h(1)]), 1);
        assert_eq!(store.physical_ids(&[h(1)]), vec![10]);
    }

    #[test]
    fn commit_batch_and_lookup() {
        let mut store = PageStore::new(16, 100);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3)]), 3);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(3)]), vec![10, 20, 30]);
    }

    // =========================================================================
    // Incremental commits (the key fix)
    // =========================================================================

    #[test]
    fn incremental_commit_via_append() {
        let mut store = PageStore::new(16, 100);
        // First commit: [H1, H2]
        store.commit_batch(&[h(1), h(2)], &[10, 20]);
        // Incremental commit: append H3 after existing [H1, H2]
        store.extend(&[h(1), h(2)], &[h(3)], &[30]);

        // physical_ids should find all three via chain traversal
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3)]), 3);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(3)]), vec![10, 20, 30]);
    }

    #[test]
    fn multiple_incremental_commits() {
        let mut store = PageStore::new(16, 100);
        store.commit_batch(&[h(1)], &[10]);
        store.extend(&[h(1)], &[h(2)], &[20]);
        store.extend(&[h(1), h(2)], &[h(3)], &[30]);
        store.extend(&[h(1), h(2), h(3)], &[h(4)], &[40]);

        assert_eq!(store.physical_ids(&[h(1), h(2), h(3), h(4)]), vec![10, 20, 30, 40]);
    }

    // =========================================================================
    // Dedup refcounting
    // =========================================================================

    #[test]
    fn dedup_survives_one_release() {
        let mut store = PageStore::new(16, 100);
        // Context A commits [H1, H2]
        store.commit_batch(&[h(1), h(2)], &[10, 20]);
        // Context B commits same chain — dedup hit
        let phys_b = store.alloc(2).unwrap();
        store.commit_batch(&[h(1), h(2)], &phys_b);

        // B's pages should be freed back (dedup)
        // rc should be 2 on each node

        // Release A's chain — should NOT evict (B still holds refs)
        let freed = store.release(&[h(1), h(2)]);
        assert_eq!(freed, 0, "should not evict — B still holds refs");
        assert_eq!(store.prefix_len(&[h(1), h(2)]), 2);
        assert_eq!(store.physical_ids(&[h(1), h(2)]), vec![10, 20]);

        // Release B's chain — NOW evict
        let freed = store.release(&[h(1), h(2)]);
        assert_eq!(freed, 2);
        assert_eq!(store.prefix_len(&[h(1), h(2)]), 0);
    }

    #[test]
    fn dedup_frees_duplicate_pages() {
        let mut store = PageStore::new(16, 10);
        assert_eq!(store.available(), 10);

        // Commit 2 pages (properly allocated from pool)
        let first_pages = store.alloc(2).unwrap();
        assert_eq!(store.available(), 8);
        store.commit_batch(&[h(1), h(2)], &first_pages);

        // Commit same chain with new pages — should free the duplicates
        let dup_pages = store.alloc(2).unwrap();
        assert_eq!(store.available(), 6);
        store.commit_batch(&[h(1), h(2)], &dup_pages);
        assert_eq!(store.available(), 8); // 6 + 2 freed
    }

    // =========================================================================
    // Path-inclusive release
    // =========================================================================

    #[test]
    fn release_decrements_entire_path() {
        let mut store = PageStore::new(16, 100);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        // Extend with divergent suffix
        store.extend(&[h(1), h(2)], &[h(4)], &[40]);

        // Release [H1, H2, H3]: decrements rc on all 3 nodes
        let freed = store.release(&[h(1), h(2), h(3)]);
        assert_eq!(freed, 1); // only H3 leaf evicted (H1,H2 still used by H4 path)

        // [H1, H2] should still be accessible (held by H4 path)
        assert_eq!(store.prefix_len(&[h(1), h(2), h(4)]), 3);

        // Release [H1, H2, H4]: now everything evicts
        let freed = store.release(&[h(1), h(2), h(4)]);
        assert!(freed >= 3);
        assert_eq!(store.prefix_len(&[h(1), h(2)]), 0);
    }

    // =========================================================================
    // Split preserves rc
    // =========================================================================

    #[test]
    fn split_preserves_existing_rc() {
        let mut store = PageStore::new(16, 100);
        // A commits [H1, H2, H3]
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        // B commits [H1, H2, H4] — causes split at H2
        store.commit_batch(&[h(1), h(2), h(4)], &[10, 20, 40]);

        // Both chains should be fully accessible
        assert_eq!(store.physical_ids(&[h(1), h(2), h(3)]), vec![10, 20, 30]);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(4)]), vec![10, 20, 40]);

        // Release A — should only evict H3 leaf
        let freed = store.release(&[h(1), h(2), h(3)]);
        assert_eq!(freed, 1);
        assert_eq!(store.prefix_len(&[h(1), h(2), h(4)]), 3);

        // Release B — now evict everything
        let freed = store.release(&[h(1), h(2), h(4)]);
        assert!(freed >= 3);
    }

    // =========================================================================
    // Retain (path-inclusive)
    // =========================================================================

    #[test]
    fn retain_bumps_rc_along_path() {
        let mut store = PageStore::new(16, 100);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        // Retain prefix [H1, H2]
        store.fork(&[h(1), h(2)]);

        // Release full chain — H3 evicted, but H1,H2 survive (retain holds them)
        let freed = store.release(&[h(1), h(2), h(3)]);
        assert_eq!(freed, 1);
        assert_eq!(store.prefix_len(&[h(1), h(2)]), 2);
    }

    // =========================================================================
    // Restore protocol: retain + extend
    // =========================================================================

    #[test]
    fn restore_protocol() {
        let mut store = PageStore::new(16, 100);
        // Initial commit
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);

        // Simulate suspend: release
        store.release(&[h(1), h(2), h(3)]);
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3)]), 0);

        // Simulate restore: prefix_len=0, so alloc + commit full chain
        let new_phys = store.alloc(3).unwrap();
        store.commit_batch(&[h(1), h(2), h(3)], &new_phys);
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3)]), 3);
    }

    #[test]
    fn restore_with_shared_prefix() {
        let mut store = PageStore::new(16, 100);
        // A and B both commit [H1, H2] (different suffixes)
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        store.commit_batch(&[h(1), h(2), h(4)], &[10, 20, 40]);

        // Suspend A
        store.release(&[h(1), h(2), h(3)]);

        // Restore A: prefix_len should be 2 (shared with B)
        let prefix = store.prefix_len(&[h(1), h(2), h(3)]);
        assert_eq!(prefix, 2);

        // Retain prefix + commit suffix
        store.fork(&[h(1), h(2)]);
        let suffix_phys = store.alloc(1).unwrap();
        store.extend(&[h(1), h(2)], &[h(3)], &suffix_phys);

        // Full chain accessible again
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3)]), 3);
    }

    // =========================================================================
    // GPU pool integration
    // =========================================================================

    #[test]
    fn eviction_returns_pages_to_pool() {
        let mut store = PageStore::new(16, 10);
        let phys = store.alloc(3).unwrap();
        assert_eq!(store.available(), 7);
        store.commit_batch(&[h(1), h(2), h(3)], &phys);
        store.release(&[h(1), h(2), h(3)]);
        assert_eq!(store.available(), 10);
    }

    // =========================================================================
    // count_reclaimable
    // =========================================================================

    #[test]
    fn count_reclaimable_single_holder() {
        let mut store = PageStore::new(16, 100);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        // Single holder — all pages reclaimable
        assert_eq!(store.count_reclaimable(&[h(1), h(2), h(3)]), 3);
    }

    #[test]
    fn count_reclaimable_shared_prefix() {
        let mut store = PageStore::new(16, 100);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        store.commit_batch(&[h(1), h(2), h(4)], &[10, 20, 40]);
        // Shared prefix — only the leaf is reclaimable
        assert_eq!(store.count_reclaimable(&[h(1), h(2), h(3)]), 1);
    }

    // =========================================================================
    // KV invariant: physical_ids must never return placeholder 0s
    // =========================================================================

    /// Simulates the real workload: many contexts share a prompt prefix
    /// (via dedup), each extends with unique output tokens, physical_ids
    /// must return valid page IDs for every page.
    #[test]
    fn kv_invariant_shared_prefix_incremental() {
        let mut store = PageStore::new(16, 200);
        let prompt = [h(1), h(2)]; // shared prompt prefix (2 pages)

        // Context A: commit prompt incrementally
        let pa = store.alloc(2).unwrap();
        store.commit_batch(&prompt[..1], &pa[..1]);
        store.extend(&prompt[..1], &prompt[1..2], &pa[1..2]);

        // Context B: commit same prompt (dedup)
        let pb = store.alloc(2).unwrap();
        store.commit_batch(&prompt, &pb);

        // Context A extends with unique output
        let pa3 = store.alloc(1).unwrap();
        store.extend(&prompt, &[h(100)], &pa3);

        // Context B extends with different output
        let pb3 = store.alloc(1).unwrap();
        store.extend(&prompt, &[h(200)], &pb3);

        // Verify physical_ids — NO placeholder 0s allowed
        let phys_a = store.physical_ids(&[h(1), h(2), h(100)]);
        let phys_b = store.physical_ids(&[h(1), h(2), h(200)]);

        assert_eq!(phys_a.len(), 3, "Context A should have 3 pages");
        assert_eq!(phys_b.len(), 3, "Context B should have 3 pages");

        // All physical IDs must be valid pool-allocated pages (non-zero unless
        // pool legitimately allocated page 0, which it can).
        // More importantly, the page counts must match.
        assert_eq!(phys_a[2], pa3[0], "Context A's 3rd page should be its allocated page");
        assert_eq!(phys_b[2], pb3[0], "Context B's 3rd page should be its allocated page");

        // Release context A — context B's pages must survive
        store.release(&[h(1), h(2), h(100)]);
        let phys_b_after = store.physical_ids(&[h(1), h(2), h(200)]);
        assert_eq!(phys_b_after.len(), 3, "Context B should still have 3 pages after A released");
        assert_eq!(phys_b_after[2], pb3[0]);
    }

    /// Simulates many incremental commits from same context, then verifies
    /// physical_ids returns correct count (catches the merge-corruption bug).
    #[test]
    fn kv_invariant_incremental_then_lookup() {
        let mut store = PageStore::new(16, 200);

        // Build a chain of 10 pages incrementally
        let pages = store.alloc(10).unwrap();
        let hashes: Vec<u64> = (1..=10).map(h).collect();

        store.commit_batch(&hashes[..1], &pages[..1]);
        for i in 1..10 {
            store.extend(&hashes[..i], &hashes[i..i+1], &pages[i..i+1]);
        }

        // Verify full chain lookup
        let phys = store.physical_ids(&hashes);
        assert_eq!(phys.len(), 10, "Should have 10 physical pages");
        // Verify each page matches what was committed
        for i in 0..10 {
            assert_eq!(phys[i], pages[i], "Page {} mismatch", i);
        }

        // Verify prefix_len
        assert_eq!(store.prefix_len(&hashes), 10);
    }

    // =========================================================================
    // Stress tests: simulate production workload patterns
    // =========================================================================

    /// Simulate N contexts sharing a prompt, each extending with unique output.
    /// Verifies physical_ids correctness for every context at every step.
    #[test]
    fn stress_shared_prefix_many_contexts() {
        let mut store = PageStore::new(16, 500);
        let n_contexts = 16;
        let prompt = [h(1), h(2)];

        // All contexts commit the same prompt (dedup)
        for _ in 0..n_contexts {
            let pages = store.alloc(2).unwrap();
            store.commit_batch(&prompt, &pages);
        }

        // Each context extends with unique output (3 tokens each)
        let mut handles: Vec<(Vec<u64>, Vec<u32>)> = Vec::new();
        for ctx in 0..n_contexts {
            let mut chain = prompt.to_vec();
            let mut phys_chain = store.physical_ids(&prompt);
            for tok in 0..3 {
                let hash = h(1000 + ctx * 100 + tok);
                let page = store.alloc(1).unwrap();
                store.extend(&chain, &[hash], &page);
                chain.push(hash);
                phys_chain.push(page[0]);
            }
            handles.push((chain, phys_chain));
        }

        // Verify every context's physical_ids
        for (i, (chain, expected_phys)) in handles.iter().enumerate() {
            let actual = store.physical_ids(chain);
            assert_eq!(actual.len(), chain.len(),
                "Context {} physical_ids length mismatch: expected {}, got {}",
                i, chain.len(), actual.len());
            // The prompt phys might differ (dedup), but suffix phys must match
            for j in 2..chain.len() {
                assert_eq!(actual[j], expected_phys[j],
                    "Context {} page {} mismatch", i, j);
            }
        }
    }

    /// Simulate suspend/restore cycle: retain prefix, release full chain,
    /// then retain + extend for restore. Verifies no pages are lost.
    #[test]
    fn stress_suspend_restore_cycle() {
        let mut store = PageStore::new(16, 500);
        let prompt = [h(1), h(2)];

        // Context A: commit prompt + 3 output tokens
        let pa = store.alloc(5).unwrap();
        store.commit_batch(&prompt, &pa[..2]);
        store.extend(&prompt, &[h(10)], &pa[2..3]);
        let chain_a1 = [h(1), h(2), h(10)];
        store.extend(&chain_a1, &[h(11)], &pa[3..4]);
        let chain_a2 = [h(1), h(2), h(10), h(11)];
        store.extend(&chain_a2, &[h(12)], &pa[4..5]);
        let full_chain_a = vec![h(1), h(2), h(10), h(11), h(12)];

        // Context B: shares prompt
        let pb = store.alloc(2).unwrap();
        store.commit_batch(&prompt, &pb);
        let pb_ext = store.alloc(1).unwrap();
        store.extend(&prompt, &[h(20)], &pb_ext);
        let chain_b = vec![h(1), h(2), h(20)];

        // Verify A's chain before suspend
        assert_eq!(store.physical_ids(&full_chain_a).len(), 5);

        // SUSPEND A: retain prefix, release full chain
        store.fork(&prompt);  // fork the prefix
        store.release(&full_chain_a);

        // B should still be alive
        assert_eq!(store.physical_ids(&chain_b).len(), 3,
            "Context B should survive after A suspended");

        // RESTORE A: retain prefix again (re-join), extend with suffix
        store.fork(&prompt);
        let suffix_pages = store.alloc(3).unwrap();
        store.extend(&prompt, &[h(10)], &suffix_pages[..1]);
        let chain_r1 = [h(1), h(2), h(10)];
        store.extend(&chain_r1, &[h(11)], &suffix_pages[1..2]);
        let chain_r2 = [h(1), h(2), h(10), h(11)];
        store.extend(&chain_r2, &[h(12)], &suffix_pages[2..3]);

        // Verify restored chain
        let phys_a = store.physical_ids(&full_chain_a);
        assert_eq!(phys_a.len(), 5,
            "Restored A should have 5 pages, got {}: {:?}", phys_a.len(), phys_a);
    }

    /// Simulate the exact pattern that causes the bug:
    /// release a context such that post-release merge fires,
    /// then extend from another context and verify physical_ids.
    #[test]
    fn stress_release_merge_then_extend() {
        let mut store = PageStore::new(16, 500);
        let prompt = [h(1), h(2)];

        // A and B share prompt
        let pa = store.alloc(2).unwrap();
        store.commit_batch(&prompt, &pa);
        let pb = store.alloc(2).unwrap();
        store.commit_batch(&prompt, &pb);

        // A extends
        let pa3 = store.alloc(1).unwrap();
        store.extend(&prompt, &[h(10)], &pa3);

        // B extends
        let pb3 = store.alloc(1).unwrap();
        store.extend(&prompt, &[h(20)], &pb3);

        // A extends further
        let pa4 = store.alloc(1).unwrap();
        store.extend(&[h(1), h(2), h(10)], &[h(11)], &pa4);
        let pa5 = store.alloc(1).unwrap();
        store.extend(&[h(1), h(2), h(10), h(11)], &[h(12)], &pa5);

        // Release A — this should evict A's suffix, leave B intact
        let chain_a = [h(1), h(2), h(10), h(11), h(12)];
        store.release(&chain_a);

        // B extends further
        let pb4 = store.alloc(1).unwrap();
        store.extend(&[h(1), h(2), h(20)], &[h(21)], &pb4);
        let pb5 = store.alloc(1).unwrap();
        store.extend(&[h(1), h(2), h(20), h(21)], &[h(22)], &pb5);

        // Verify B's full chain
        let chain_b = [h(1), h(2), h(20), h(21), h(22)];
        let phys_b = store.physical_ids(&chain_b);
        assert_eq!(phys_b.len(), 5,
            "Context B should have 5 pages after A released, got {}: {:?}",
            phys_b.len(), phys_b);
    }

    /// Simulate many suspend/restore cycles on overlapping contexts.
    #[test]
    fn stress_many_suspend_restore() {
        let mut store = PageStore::new(16, 1000);
        let prompt = [h(1), h(2)];

        // Create 8 contexts sharing prompt
        for _ in 0..8 {
            let p = store.alloc(2).unwrap();
            store.commit_batch(&prompt, &p);
        }

        // Each extends with 5 unique tokens
        let mut chains: Vec<Vec<u64>> = Vec::new();
        for ctx in 0..8u64 {
            let mut chain = prompt.to_vec();
            for tok in 0..5 {
                let hash = h(100 * ctx + tok + 1);
                let page = store.alloc(1).unwrap();
                store.extend(&chain, &[hash], &page);
                chain.push(hash);
            }
            chains.push(chain);
        }

        // Verify all chains
        for (i, chain) in chains.iter().enumerate() {
            let phys = store.physical_ids(chain);
            assert_eq!(phys.len(), 7, "Chain {} should have 7 pages", i);
        }

        // Suspend first 4 contexts: retain prefix + release full chain
        for ctx in 0..4 {
            store.fork(&prompt);
            store.release(&chains[ctx]);
        }

        // Remaining 4 should still work
        for ctx in 4..8 {
            let phys = store.physical_ids(&chains[ctx]);
            assert_eq!(phys.len(), 7,
                "Chain {} should have 7 pages after suspensions, got {}: {:?}",
                ctx, phys.len(), phys);
        }

        // Restore the first 4: retain prefix + extend with new pages
        for ctx in 0..4 {
            store.fork(&prompt);
            let suffix = &chains[ctx][2..]; // original suffix hashes
            let mut prefix = prompt.to_vec();
            for &hash in suffix {
                let page = store.alloc(1).unwrap();
                store.extend(&prefix, &[hash], &page);
                prefix.push(hash);
            }
        }

        // Now all 8 chains should be accessible (possibly with new phys for restored ones,
        // but the LENGTH must be correct)
        for (i, chain) in chains.iter().enumerate() {
            let phys = store.physical_ids(chain);
            assert_eq!(phys.len(), 7,
                "After restore, chain {} should have 7 pages, got {}: {:?}",
                i, phys.len(), phys);
        }
    }

    /// REGRESSION: hash-mismatch-after-merge.
    /// After try_merge_child merges [H1]+[H2] into [H1,H2], a different
    /// context releasing [H1, H_OTHER] must NOT decrement rc on [H1,H2].
    #[test]
    fn hash_mismatch_after_merge_release() {
        let mut store = PageStore::new(16, 200);

        // Context A: commit [H1], extend to [H1, H2]
        let pa = store.alloc(1).unwrap();
        store.commit_batch(&[h(1)], &pa);
        let pa2 = store.alloc(1).unwrap();
        store.extend(&[h(1)], &[h(2)], &pa2);
        // After this, try_merge_child merges: [H1,H2] rc=1

        // Context B: commit [H1] (dedup → rc=2), extend to [H1, H99]
        let pb = store.alloc(1).unwrap();
        store.commit_batch(&[h(1)], &pb);
        // Now: insert_suffix sees existing [H1,H2] rc=1, H1 matches.
        // common_prefix_len([H1,H2], [H1]) = 1 < node_len=2 → split.
        // Result: [H1] rc=2, children: {[H2] rc=1, ...}

        // B extends with different hash
        let pb2 = store.alloc(1).unwrap();
        store.extend(&[h(1)], &[h(99)], &pb2);
        // Now: [H1] rc=2, children: {[H2] rc=1, [H99] rc=1}

        // Verify A's chain
        let phys_a = store.physical_ids(&[h(1), h(2)]);
        assert_eq!(phys_a.len(), 2, "A should have 2 pages");

        // Verify B's chain
        let phys_b = store.physical_ids(&[h(1), h(99)]);
        assert_eq!(phys_b.len(), 2, "B should have 2 pages");

        // Release B: [H1, H99]
        store.release(&[h(1), h(99)]);

        // A's pages must survive! Release of [H1, H99] should NOT affect [H1, H2]
        let phys_a2 = store.physical_ids(&[h(1), h(2)]);
        assert_eq!(phys_a2.len(), 2,
            "A should still have 2 pages after B released, got {}: {:?}",
            phys_a2.len(), phys_a2);

        // Release A
        store.release(&[h(1), h(2)]);
    }

    /// REGRESSION: release of a chain that diverges from a merged node
    /// mid-path should not corrupt the trie.
    #[test]
    fn hash_mismatch_deep_merge_release() {
        let mut store = PageStore::new(16, 200);

        // Context A: single chain [H1] → [H1,H2] → [H1,H2,H3] (all merged, rc=1)
        let pa = store.alloc(1).unwrap();
        store.commit_batch(&[h(1)], &pa);
        let pa2 = store.alloc(1).unwrap();
        store.extend(&[h(1)], &[h(2)], &pa2);
        let pa3 = store.alloc(1).unwrap();
        store.extend(&[h(1), h(2)], &[h(3)], &pa3);
        // Merged: [H1,H2,H3] rc=1

        // Context C: commit [H1] (dedup, splits merged node)
        // Then extend [H1, H50, H60] — completely different branch
        let pc = store.alloc(1).unwrap();
        store.commit_batch(&[h(1)], &pc);
        let pc2 = store.alloc(1).unwrap();
        store.extend(&[h(1)], &[h(50)], &pc2);
        let pc3 = store.alloc(1).unwrap();
        store.extend(&[h(1), h(50)], &[h(60)], &pc3);

        // Verify both chains
        assert_eq!(store.physical_ids(&[h(1), h(2), h(3)]).len(), 3, "A chain len");
        assert_eq!(store.physical_ids(&[h(1), h(50), h(60)]).len(), 3, "C chain len");

        // Release C: should NOT affect A
        store.release(&[h(1), h(50), h(60)]);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(3)]).len(), 3,
            "A should survive after C released");

        // Release A
        store.release(&[h(1), h(2), h(3)]);
    }

    /// REGRESSION: prefix_match_len should stop at hash divergence.
    #[test]
    fn prefix_match_len_stops_at_divergence() {
        let mut store = PageStore::new(16, 200);

        // Build merged node [H1, H2, H3] rc=1
        let p = store.alloc(1).unwrap();
        store.commit_batch(&[h(1)], &p);
        let p2 = store.alloc(1).unwrap();
        store.extend(&[h(1)], &[h(2)], &p2);
        let p3 = store.alloc(1).unwrap();
        store.extend(&[h(1), h(2)], &[h(3)], &p3);

        // Query with different second hash: [H1, H99, ...]
        // Should match only 1 (H1), not 3
        assert_eq!(store.prefix_len(&[h(1), h(99), h(999)]), 1,
            "prefix_len should stop at H99 divergence");

        // Full match
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3)]), 3);

        // Partial match, same prefix then diverge
        assert_eq!(store.prefix_len(&[h(1), h(2), h(99)]), 2,
            "prefix_len should stop at H99 divergence after matching H1,H2");
    }

    // =========================================================================
    // would_free
    // =========================================================================

    #[test]
    fn would_free_returns_rc1_only() {
        let mut store = PageStore::new(16, 100);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        // All rc=1, leaf → everything would be freed.
        let wf = store.would_free(&[h(1), h(2), h(3)]);
        assert_eq!(wf.len(), 3, "all 3 hashes should be would-free");
    }

    #[test]
    fn would_free_empty_on_shared() {
        let mut store = PageStore::new(16, 100);
        // A and B share [H1, H2]
        store.commit_batch(&[h(1), h(2)], &[10, 20]);
        store.commit_batch(&[h(1), h(2)], &[30, 40]); // dedup, rc=2
        // All rc=2 → nothing would be freed.
        let wf = store.would_free(&[h(1), h(2)]);
        assert!(wf.is_empty(), "shared pages should not be in would_free");
    }

    #[test]
    fn would_free_partial_shared() {
        let mut store = PageStore::new(16, 100);
        // A commits [H1, H2, H3], B commits [H1, H2, H4]
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        store.commit_batch(&[h(1), h(2), h(4)], &[10, 20, 40]);
        // For chain A: [H1, H2] shared (rc=2), [H3] unique (rc=1, leaf)
        let wf = store.would_free(&[h(1), h(2), h(3)]);
        assert_eq!(wf, vec![h(3)], "only H3 should be in would_free");
    }

    #[test]
    fn would_free_interior_nodes() {
        let mut store = PageStore::new(16, 100);
        // Build chain incrementally: [H1,H2] then extend → [H1,H2,H3]
        // This creates interior node [H1,H2](rc=1) → child [H3](rc=1, leaf).
        store.commit_batch(&[h(1), h(2)], &[10, 20]);
        store.extend(&[h(1), h(2)], &[h(3)], &[30]);
        // Sole holder → all 3 pages should be would-free, including interior [H1,H2].
        let wf = store.would_free(&[h(1), h(2), h(3)]);
        assert_eq!(wf.len(), 3, "interior + leaf should both be would-free: got {:?}", wf);
        assert!(wf.contains(&h(3)), "leaf H3 must be present");
        assert!(wf.contains(&h(1)), "interior H1 must be present");
        assert!(wf.contains(&h(2)), "interior H2 must be present");
    }

    // =========================================================================
    // FlatPageStore
    // =========================================================================

    #[test]
    fn cpu_store_insert_and_lookup() {
        let mut cpu = FlatPageStore::new(100);
        cpu.insert(&[h(1), h(2), h(3)], &[10, 20, 30]);
        assert_eq!(cpu.physical_ids(&[h(1), h(2), h(3)]), vec![10, 20, 30]);
        assert_eq!(cpu.prefix_len(&[h(1), h(2), h(3)]), 3);
        assert_eq!(cpu.len(), 3);
    }

    #[test]
    fn cpu_store_dedup_bumps_rc() {
        let mut cpu = FlatPageStore::new(100);
        let p1 = cpu.alloc(3).unwrap();
        cpu.insert(&[h(1), h(2), h(3)], &p1);
        assert_eq!(cpu.available(), 97);
        // Insert same hashes again with new pages → dedup, frees new pages.
        let p2 = cpu.alloc(3).unwrap();
        assert_eq!(cpu.available(), 94);
        cpu.insert(&[h(1), h(2), h(3)], &p2);
        // Dedup freed the 3 duplicate pages back.
        assert_eq!(cpu.available(), 97);
        // Release once — rc drops to 1, pages stay.
        let freed = cpu.release(&[h(1), h(2), h(3)]);
        assert_eq!(freed, 0);
        // Release again — rc drops to 0, pages freed.
        let freed = cpu.release(&[h(1), h(2), h(3)]);
        assert_eq!(freed, 3);
        assert_eq!(cpu.available(), 100);
    }

    #[test]
    fn cpu_store_release_frees_at_zero() {
        let mut cpu = FlatPageStore::new(10);
        let p = cpu.alloc(2).unwrap();
        cpu.insert(&[h(1), h(2)], &p);
        assert_eq!(cpu.available(), 8);
        let freed = cpu.release(&[h(1), h(2)]);
        assert_eq!(freed, 2);
        assert_eq!(cpu.available(), 10);
        assert_eq!(cpu.prefix_len(&[h(1), h(2)]), 0);
    }

    #[test]
    fn cpu_store_out_of_order_insert() {
        let mut cpu = FlatPageStore::new(100);
        // Insert suffix first [H50..H99], then prefix [H0..H49].
        // This is the key scenario from the design discussion.
        let suffix: Vec<PageHash> = (50..100).map(|i| h(i)).collect();
        let suffix_phys: Vec<PhysicalPageId> = (50..100).map(|i| i as u32).collect();
        cpu.insert(&suffix, &suffix_phys);

        let prefix: Vec<PageHash> = (0..50).map(|i| h(i)).collect();
        let prefix_phys: Vec<PhysicalPageId> = (0..50).map(|i| i as u32).collect();
        cpu.insert(&prefix, &prefix_phys);

        // Full chain [H0..H99] should be findable.
        let full_chain: Vec<PageHash> = (0..100).map(|i| h(i)).collect();
        assert_eq!(cpu.prefix_len(&full_chain), 100);
        assert_eq!(cpu.physical_ids(&full_chain).len(), 100);
    }

    #[test]
    fn cpu_store_effective_pages() {
        let mut cpu = FlatPageStore::new(100);
        // A inserts [H1, H2, H3], all rc=1
        let p = cpu.alloc(3).unwrap();
        cpu.insert(&[h(1), h(2), h(3)], &p);
        assert!((cpu.effective_pages(&[h(1), h(2), h(3)]) - 3.0).abs() < 0.001);

        // B inserts same → rc=2 each
        let p2 = cpu.alloc(3).unwrap();
        cpu.insert(&[h(1), h(2), h(3)], &p2);
        assert!((cpu.effective_pages(&[h(1), h(2), h(3)]) - 1.5).abs() < 0.001);
    }

    #[test]
    fn cpu_store_prefix_len_gap() {
        let mut cpu = FlatPageStore::new(100);
        // Insert H1 and H3 but not H2 — prefix_len should be 1.
        cpu.insert(&[h(1)], &[10]);
        cpu.insert(&[h(3)], &[30]);
        assert_eq!(cpu.prefix_len(&[h(1), h(2), h(3)]), 1);
    }

    #[test]
    fn cpu_store_contains() {
        let mut cpu = FlatPageStore::new(100);
        cpu.insert(&[h(42)], &[10]);
        assert!(cpu.contains(&h(42)));
        assert!(!cpu.contains(&h(99)));
    }
}
