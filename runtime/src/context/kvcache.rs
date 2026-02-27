//! KV Cache — Content-Addressable Storage for KV cache pages.
//!
//! Pages are identified by a chained content hash (`PageHash`).
//! Sharing across contexts is structural: two contexts with the same
//! prefix automatically share physical pages.
//!
//! Each model device gets its own `DevicePageCache` — no cross-device
//! coordination for page identity. The cache is exclusively owned by
//! a `ContextManager` actor, so no interior mutability or locking is needed.

use std::hash::{Hash, Hasher};

use anyhow::Result;
use rustc_hash::{FxHashMap, FxHasher};

use crate::inference::brle::Brle;

// =============================================================================
// Types
// =============================================================================

/// Content hash of a KV cache page, chained to its predecessor.
pub type PageHash = u64;

/// Physical page index in GPU or CPU memory.
pub type PhysicalPageId = u32;

/// Copy coordinates for GPU↔CPU swap, used by the RPC layer.
#[derive(Debug, Clone)]
pub struct SwapOp {
    pub gpu_phys: PhysicalPageId,
    pub cpu_slot: PhysicalPageId,
}

// =============================================================================
// PhysicalPagePool
// =============================================================================

/// Manages a pool of physical page slots (GPU or CPU) for a single device.
#[derive(Debug)]
struct PhysicalPagePool {
    total: usize,
    free: Vec<PhysicalPageId>,
}

impl PhysicalPagePool {
    fn new(total: usize) -> Self {
        PhysicalPagePool {
            total,
            free: (0..total as PhysicalPageId).collect(),
        }
    }

    fn alloc(&mut self) -> Option<PhysicalPageId> {
        self.free.pop()
    }

    fn free(&mut self, id: PhysicalPageId) {
        self.free.push(id);
    }

    fn used(&self) -> usize {
        self.total - self.free.len()
    }

    fn available(&self) -> usize {
        self.free.len()
    }
}

// =============================================================================
// DevicePageCache
// =============================================================================

/// Per-device page cache. Manages the content-addressed KV cache for one GPU.
///
/// Data structures:
/// - `pages`:       hash → GPU physical page (always present for GPU-resident pages)
/// - `chain`:       hash → prev_hash (backward links for traversal)
/// - `refcount`:    hash → number of contexts including this hash in their chain
/// - `index_cache`: tip_hash → flattened `Vec<PhysicalPageId>` from root to tip
///                  Sparse: only cached for active `committed_tip`s.
///
/// Committed pages that are evicted from GPU are simply discarded. They can be
/// replayed from lineage when the context needs to be restored. CPU memory is
/// used exclusively for working page swap (uncommitted pages during suspension).
#[derive(Debug)]
pub struct DevicePageCache {
    page_size: usize,

    /// hash → GPU physical page ID. Present for every GPU-resident committed page.
    pages: FxHashMap<PageHash, PhysicalPageId>,

    /// hash → prev_hash for backward chain traversal. 0 = root (no predecessor).
    chain: FxHashMap<PageHash, PageHash>,

    /// hash → number of active contexts whose committed chain includes this hash.
    refcount: FxHashMap<PageHash, usize>,

    /// Traversal cache: tip_hash → ordered physical page IDs from root to tip.
    /// Sparse: only maintained for hashes that are an active context's `committed_tip`.
    index_cache: FxHashMap<PageHash, Vec<PhysicalPageId>>,

    /// GPU physical page pool.
    gpu: PhysicalPagePool,

    /// CPU physical page pool (exclusively for working page swap).
    cpu: PhysicalPagePool,
}

impl DevicePageCache {
    pub fn new(page_size: usize, num_gpu_pages: usize, num_cpu_pages: usize) -> Self {
        DevicePageCache {
            page_size,
            pages: FxHashMap::default(),
            chain: FxHashMap::default(),
            refcount: FxHashMap::default(),
            index_cache: FxHashMap::default(),
            gpu: PhysicalPagePool::new(num_gpu_pages),
            cpu: PhysicalPagePool::new(num_cpu_pages),
        }
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Returns (used_gpu_pages, total_gpu_pages).
    pub fn stats(&self) -> (usize, usize) {
        (self.gpu.used(), self.gpu.total)
    }

    /// GPU memory pressure as a fraction [0.0, 1.0].
    pub fn pressure(&self) -> f64 {
        if self.gpu.total == 0 { 0.0 }
        else { self.gpu.used() as f64 / self.gpu.total as f64 }
    }

    /// Number of available GPU pages (free pool + evictable LRU pages).
    pub fn available_gpu_pages(&self) -> usize {
        self.gpu.available() + self.evictable_count()
    }

    // =========================================================================
    // Hash computation
    // =========================================================================

    /// Compute chained hashes for a sequence of page-aligned token chunks.
    /// Each hash depends on (tokens, positions, masks, prev_hash).
    pub fn compute_page_hashes(
        &self,
        tokens: &[u32],
        positions: &[u32],
        masks: &[Brle],
        prev_hash: PageHash,
    ) -> Vec<PageHash> {
        let mut hashes = Vec::new();
        let mut running = prev_hash;

        for (chunk_idx, chunk) in tokens.chunks(self.page_size).enumerate() {
            let start = chunk_idx * self.page_size;
            let end = start + chunk.len();
            let chunk_pos = &positions[start..end];
            let chunk_masks = &masks[start..end];

            let mut hasher = FxHasher::default();
            chunk.hash(&mut hasher);
            for pos in chunk_pos { pos.hash(&mut hasher); }
            for mask in chunk_masks { mask.hash(&mut hasher); }
            let content_hash = hasher.finish();

            let mut chain_hasher = FxHasher::default();
            content_hash.hash(&mut chain_hasher);
            running.hash(&mut chain_hasher);
            let page_hash = chain_hasher.finish();

            hashes.push(page_hash);
            running = page_hash;
        }

        hashes
    }

    // =========================================================================
    // Commit
    // =========================================================================


    /// Promote a working page to a committed page.
    ///
    /// Like `commit_page`, but reuses the working page's physical ID instead
    /// of allocating a new one. This is essential because the KV cache data
    /// was written to the working page during the forward pass.
    ///
    /// If the hash is a dedup hit, the working page is freed back to the pool
    /// since the existing physical page already has the correct data.
    ///
    /// Returns `(physical_page_id, working_page_freed)`.
    pub fn commit_working_page(
        &mut self,
        hash: PageHash,
        prev_hash: PageHash,
        working_phys: PhysicalPageId,
    ) -> (PhysicalPageId, bool) {
        // Dedup: page already exists on GPU
        if let Some(&phys) = self.pages.get(&hash) {
            *self.refcount.entry(hash).or_insert(0) += 1;
            // Free the working page since we'll use the existing committed copy
            self.gpu.free(working_phys);
            return (phys, true);
        }

        // Promote: register the working page's physical ID as the committed page
        self.pages.insert(hash, working_phys);
        self.chain.insert(hash, prev_hash);
        *self.refcount.entry(hash).or_insert(0) += 1;

        (working_phys, false)
    }

    /// Update the index_cache for a context's new committed_tip.
    ///
    /// Move semantics: if `old_tip` is provided and has an index_cache entry,
    /// take it, append the new physical pages, and store under `new_tip`.
    /// Otherwise rebuild from chain + pages.
    pub fn update_index_cache(
        &mut self,
        new_tip: PageHash,
        old_tip: Option<PageHash>,
        new_phys_pages: &[PhysicalPageId],
    ) {
        // Try to reuse old tip's cached page table
        let mut page_table = match old_tip {
            Some(old) => self.index_cache.remove(&old).unwrap_or_default(),
            None => Vec::new(),
        };

        // If empty (cache miss or first commit), rebuild from chain
        if page_table.is_empty() && old_tip.is_some() {
            page_table = self.rebuild_page_table(old_tip.unwrap());
        }

        page_table.extend_from_slice(new_phys_pages);
        self.index_cache.insert(new_tip, page_table);
    }

    /// Remove an index_cache entry (e.g., when context is destroyed or suspended).
    pub fn remove_index_cache(&mut self, tip: PageHash) {
        self.index_cache.remove(&tip);
    }

    // =========================================================================
    // Chain traversal & refcounting
    // =========================================================================

    /// Increment refcount for every page reachable from `tip`.
    pub fn acquire_chain(&mut self, tip: PageHash) {
        let mut h = tip;
        loop {
            *self.refcount.entry(h).or_insert(0) += 1;
            match self.chain.get(&h) {
                Some(&prev) if prev != 0 => h = prev,
                _ => break,
            }
        }
    }

    /// Decrement refcount for every page reachable from `tip`.
    /// Returns hashes that hit refcount=0 (candidates for eviction).
    pub fn release_chain(&mut self, tip: PageHash) -> Vec<PageHash> {
        let mut evictable = Vec::new();
        let mut h = tip;
        loop {
            if let Some(rc) = self.refcount.get_mut(&h) {
                *rc = rc.saturating_sub(1);
                if *rc == 0 {
                    evictable.push(h);
                }
            }
            match self.chain.get(&h) {
                Some(&prev) if prev != 0 => h = prev,
                _ => break,
            }
        }
        evictable
    }

    /// Walk the hash chain backward from `tip` to root, returning hashes in
    /// root-to-tip order.
    pub fn walk_chain(&self, tip: PageHash) -> Vec<PageHash> {
        let mut hashes = Vec::new();
        let mut h = tip;
        loop {
            hashes.push(h);
            match self.chain.get(&h) {
                Some(&prev) if prev != 0 => h = prev,
                _ => break,
            }
        }
        hashes.reverse();
        hashes
    }

    /// Get the refcount for a specific hash.
    pub fn refcount(&self, hash: PageHash) -> usize {
        self.refcount.get(&hash).copied().unwrap_or(0)
    }

    // =========================================================================
    // Physical page resolution
    // =========================================================================

    /// Resolve the ordered list of GPU physical page IDs for a chain tip.
    /// O(1) if the tip is in index_cache; otherwise rebuilds from chain + pages.
    pub fn resolve_physical(&mut self, tip: PageHash) -> Vec<PhysicalPageId> {
        if let Some(cached) = self.index_cache.get(&tip) {
            return cached.clone();
        }
        let table = self.rebuild_page_table(tip);
        // Cache it for future lookups
        self.index_cache.insert(tip, table.clone());
        table
    }

    /// Rebuild a page table from chain links + per-hash physical pages.
    fn rebuild_page_table(&self, tip: PageHash) -> Vec<PhysicalPageId> {
        let chain = self.walk_chain(tip);
        chain.iter()
            .filter_map(|h| self.pages.get(h).copied())
            .collect()
    }

    /// Check if a specific hash has a GPU-resident physical page.
    pub fn is_gpu_resident(&self, hash: PageHash) -> bool {
        self.pages.contains_key(&hash)
    }

    /// Count committed pages cached on this device (GPU-resident) in a chain.
    pub fn chain_resident_count(&self, tip: PageHash) -> usize {
        let chain = self.walk_chain(tip);
        chain.iter().filter(|h| self.pages.contains_key(h)).count()
    }

    // =========================================================================
    // Working pages (uncommitted, mutable, exclusive to one context)
    // =========================================================================

    /// Allocate `n` mutable GPU pages. Returns physical page IDs.
    /// Evicts unreferenced committed pages from LRU if needed.
    pub fn allocate_working(&mut self, n: usize) -> Result<Vec<PhysicalPageId>> {
        let mut allocated = Vec::with_capacity(n);

        for _ in 0..n {
            let phys = loop {
                if let Some(p) = self.gpu.alloc() {
                    break p;
                }
                // Evict one unreferenced committed page
                if !self.evict_one() {
                    // Rollback already allocated
                    for p in allocated {
                        self.gpu.free(p);
                    }
                    anyhow::bail!("No free GPU pages for working allocation");
                }
            };
            allocated.push(phys);
        }

        Ok(allocated)
    }

    /// Free working pages back to the GPU pool.
    pub fn free_working(&mut self, pages: &[PhysicalPageId]) {
        for &p in pages {
            self.gpu.free(p);
        }
    }

    /// Free CPU slots back to the CPU pool (for suspended working pages).
    pub fn free_cpu_slots(&mut self, slots: &[PhysicalPageId]) {
        for &s in slots {
            self.cpu.free(s);
        }
    }

    /// Swap working pages from GPU to CPU.
    /// Returns swap operations for the RPC layer.
    pub fn swap_out_working(&mut self, gpu_pages: &[PhysicalPageId]) -> Result<Vec<SwapOp>> {
        let mut ops = Vec::with_capacity(gpu_pages.len());

        for &gpu_phys in gpu_pages {
            let cpu_slot = self.cpu.alloc()
                .ok_or_else(|| anyhow::anyhow!("No free CPU pages for working swap-out"))?;

            ops.push(SwapOp { gpu_phys, cpu_slot });
            self.gpu.free(gpu_phys);
        }

        Ok(ops)
    }

    /// Swap working pages from CPU back to GPU.
    /// Returns swap operations for the RPC layer.
    pub fn swap_in_working(&mut self, cpu_slots: &[PhysicalPageId]) -> Result<Vec<SwapOp>> {
        let mut ops = Vec::with_capacity(cpu_slots.len());

        for &cpu_slot in cpu_slots {
            let gpu_phys = self.gpu.alloc()
                .ok_or_else(|| anyhow::anyhow!("No free GPU pages for working swap-in"))?;

            ops.push(SwapOp { gpu_phys, cpu_slot });
            self.cpu.free(cpu_slot);
        }

        Ok(ops)
    }

    // =========================================================================
    // Eviction
    // =========================================================================

    /// Number of committed GPU pages with refcount=0 (evictable).
    fn evictable_count(&self) -> usize {
        self.pages.keys()
            .filter(|h| self.refcount.get(h).copied().unwrap_or(0) == 0)
            .count()
    }

    /// Evict one unreferenced committed page from GPU.
    /// The page is discarded — it can be replayed from lineage.
    /// Returns true if a page was evicted.
    fn evict_one(&mut self) -> bool {
        // Find an unreferenced GPU page
        let victim = self.pages.iter()
            .find(|(h, _)| self.refcount.get(h).copied().unwrap_or(0) == 0)
            .map(|(&h, &p)| (h, p));

        let (hash, gpu_phys) = match victim {
            Some(v) => v,
            None => return false,
        };

        // Remove from GPU and free the slot
        self.pages.remove(&hash);
        self.gpu.free(gpu_phys);

        // Clean up refcount (keep chain links for future walk_chain)
        self.refcount.remove(&hash);

        true
    }

    /// Evict all unreferenced committed pages from GPU.
    /// Returns the number of GPU pages freed.
    pub fn evict_unreferenced(&mut self) -> usize {
        let mut freed = 0;
        while self.evict_one() {
            freed += 1;
        }
        freed
    }

    // =========================================================================
    // Residency check (for ensure_resident)
    // =========================================================================

    /// Classify pages in a committed chain by their GPU residency.
    ///
    /// Returns `(gpu_resident, discarded)` — each a Vec of PageHash
    /// in root-to-tip order. Discarded pages must be replayed from lineage.
    pub fn classify_chain(
        &self,
        tip: PageHash,
    ) -> (Vec<PageHash>, Vec<PageHash>) {
        let chain = self.walk_chain(tip);
        let mut gpu = Vec::new();
        let mut discarded = Vec::new();

        for h in chain {
            if self.pages.contains_key(&h) {
                gpu.push(h);
            } else {
                discarded.push(h);
            }
        }

        (gpu, discarded)
    }

    // =========================================================================
    // Prefix lookup (for restore from lineage)
    // =========================================================================

    /// Find the longest prefix of `hashes` that exists on GPU.
    /// Increments refcount for matched pages.
    /// Returns the number of matched pages (from the start).
    pub fn longest_prefix_match(&mut self, hashes: &[PageHash]) -> usize {
        let mut matched = 0;
        for &h in hashes {
            if self.pages.contains_key(&h) {
                *self.refcount.entry(h).or_insert(0) += 1;
                matched += 1;
            } else {
                break;
            }
        }
        matched
    }
}

// =============================================================================
// Utility
// =============================================================================

/// Compute FlashInfer's `last_page_len` from total KV tokens and page geometry.
///
/// FlashInfer equation: `seq_len = (num_pages - 1) * page_size + last_page_len`
pub fn compute_last_page_len(total_kv: u32, num_pages: u32, page_size: u32) -> u32 {
    if num_pages == 0 {
        0
    } else {
        let r = total_kv % page_size;
        if r == 0 { page_size } else { r }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_brle(n: usize) -> Vec<Brle> {
        (0..n).map(|_| Brle::new(0)).collect()
    }

    /// Test helper: allocate a working page, then promote it via commit_working_page.
    fn alloc_and_commit(cache: &mut DevicePageCache, hash: PageHash, prev: PageHash) -> PhysicalPageId {
        let working = cache.allocate_working(1).unwrap()[0];
        let (phys, _freed) = cache.commit_working_page(hash, prev, working);
        phys
    }

    #[test]
    fn test_commit_and_resolve() {
        let mut cache = DevicePageCache::new(4, 100, 10);

        // Compute hashes for 8 tokens (2 pages)
        let tokens: Vec<u32> = (0..8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let masks = make_brle(8);
        let hashes = cache.compute_page_hashes(&tokens, &positions, &masks, 0);
        assert_eq!(hashes.len(), 2);

        // Commit page 1
        let phys1 = alloc_and_commit(&mut cache, hashes[0], 0);
        assert!(cache.is_gpu_resident(hashes[0]));

        // Commit page 2
        let phys2 = alloc_and_commit(&mut cache, hashes[1], hashes[0]);

        // Update index cache
        cache.update_index_cache(hashes[1], None, &[phys1, phys2]);

        // Resolve physical pages
        let resolved = cache.resolve_physical(hashes[1]);
        assert_eq!(resolved, vec![phys1, phys2]);
    }

    #[test]
    fn test_dedup_on_commit() {
        let mut cache = DevicePageCache::new(4, 100, 10);

        let tokens: Vec<u32> = (0..4).collect();
        let positions: Vec<u32> = (0..4).collect();
        let masks = make_brle(4);
        let hashes = cache.compute_page_hashes(&tokens, &positions, &masks, 0);

        // First commit
        let phys1 = alloc_and_commit(&mut cache, hashes[0], 0);
        assert_eq!(cache.refcount(hashes[0]), 1);

        // Second commit of same hash (dedup) — working page should be freed
        let working2 = cache.allocate_working(1).unwrap()[0];
        let (phys2, freed) = cache.commit_working_page(hashes[0], 0, working2);
        assert!(freed); // dedup hit frees the working page
        assert_eq!(phys1, phys2); // Same physical page
        assert_eq!(cache.refcount(hashes[0]), 2);
    }

    #[test]
    fn test_acquire_release_chain() {
        let mut cache = DevicePageCache::new(4, 100, 10);

        let tokens: Vec<u32> = (0..8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let masks = make_brle(8);
        let hashes = cache.compute_page_hashes(&tokens, &positions, &masks, 0);

        alloc_and_commit(&mut cache, hashes[0], 0);
        alloc_and_commit(&mut cache, hashes[1], hashes[0]);

        // Initial refcount: 1 each (from commit)
        assert_eq!(cache.refcount(hashes[0]), 1);
        assert_eq!(cache.refcount(hashes[1]), 1);

        // Acquire (simulating fork)
        cache.acquire_chain(hashes[1]);
        assert_eq!(cache.refcount(hashes[0]), 2);
        assert_eq!(cache.refcount(hashes[1]), 2);

        // Release (simulating destroy of fork)
        let evictable = cache.release_chain(hashes[1]);
        assert!(evictable.is_empty()); // Still held by original
        assert_eq!(cache.refcount(hashes[0]), 1);
        assert_eq!(cache.refcount(hashes[1]), 1);

        // Release again (original destroyed)
        let evictable = cache.release_chain(hashes[1]);
        assert_eq!(evictable.len(), 2); // Both now evictable
        assert_eq!(cache.refcount(hashes[0]), 0);
        assert_eq!(cache.refcount(hashes[1]), 0);
    }

    #[test]
    fn test_shared_pages_survive_eviction() {
        let mut cache = DevicePageCache::new(4, 10, 5);

        let tokens: Vec<u32> = (0..8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let masks = make_brle(8);
        let hashes = cache.compute_page_hashes(&tokens, &positions, &masks, 0);

        alloc_and_commit(&mut cache, hashes[0], 0);
        alloc_and_commit(&mut cache, hashes[1], hashes[0]);

        // Simulate fork: acquire chain
        cache.acquire_chain(hashes[1]);
        // refcount: h0=2, h1=2

        // Release one context's chain
        cache.release_chain(hashes[1]);
        // refcount: h0=1, h1=1

        // Try to evict — nothing should be evicted (refcount > 0)
        let freed = cache.evict_unreferenced();
        assert_eq!(freed, 0);
        assert!(cache.is_gpu_resident(hashes[0]));
        assert!(cache.is_gpu_resident(hashes[1]));
    }

    #[test]
    fn test_evict_unreferenced() {
        let mut cache = DevicePageCache::new(4, 10, 5);

        let tokens: Vec<u32> = (0..4).collect();
        let positions: Vec<u32> = (0..4).collect();
        let masks = make_brle(4);
        let hashes = cache.compute_page_hashes(&tokens, &positions, &masks, 0);

        alloc_and_commit(&mut cache, hashes[0], 0);

        // Release chain → refcount=0
        cache.release_chain(hashes[0]);

        // Evict
        let freed = cache.evict_unreferenced();
        assert_eq!(freed, 1);
        assert!(!cache.is_gpu_resident(hashes[0])); // Discarded entirely
    }

    #[test]
    fn test_working_pages_alloc_free() {
        let mut cache = DevicePageCache::new(4, 10, 5);

        let pages = cache.allocate_working(3).unwrap();
        assert_eq!(pages.len(), 3);
        assert_eq!(cache.gpu.used(), 3);

        cache.free_working(&pages);
        assert_eq!(cache.gpu.used(), 0);
    }

    #[test]
    fn test_working_pages_swap() {
        let mut cache = DevicePageCache::new(4, 10, 5);

        let pages = cache.allocate_working(2).unwrap();
        let swap_ops = cache.swap_out_working(&pages).unwrap();
        assert_eq!(swap_ops.len(), 2);
        assert_eq!(cache.gpu.used(), 0); // GPU freed
        assert_eq!(cache.cpu.used(), 2); // CPU used

        let cpu_slots: Vec<_> = swap_ops.iter().map(|op| op.cpu_slot).collect();
        let swap_in_ops = cache.swap_in_working(&cpu_slots).unwrap();
        assert_eq!(swap_in_ops.len(), 2);
        assert_eq!(cache.gpu.used(), 2); // GPU allocated
        assert_eq!(cache.cpu.used(), 0); // CPU freed
    }

    #[test]
    fn test_classify_chain() {
        let mut cache = DevicePageCache::new(4, 10, 5);

        let tokens: Vec<u32> = (0..8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let masks = make_brle(8);
        let hashes = cache.compute_page_hashes(&tokens, &positions, &masks, 0);

        alloc_and_commit(&mut cache, hashes[0], 0);
        alloc_and_commit(&mut cache, hashes[1], hashes[0]);

        // Both on GPU
        let (gpu, disc) = cache.classify_chain(hashes[1]);
        assert_eq!(gpu.len(), 2);
        assert_eq!(disc.len(), 0);

        // Release and evict page 0
        cache.release_chain(hashes[0]); // Only decrements h0
        cache.evict_unreferenced();

        let (gpu, disc) = cache.classify_chain(hashes[1]);
        assert_eq!(gpu.len(), 1); // Only h1 on GPU
        assert_eq!(disc.len(), 1); // h0 discarded
    }

    #[test]
    fn test_longest_prefix_match() {
        let mut cache = DevicePageCache::new(4, 100, 10);

        let tokens: Vec<u32> = (0..12).collect();
        let positions: Vec<u32> = (0..12).collect();
        let masks = make_brle(12);
        let hashes = cache.compute_page_hashes(&tokens, &positions, &masks, 0);
        assert_eq!(hashes.len(), 3);

        // Commit first 2 pages
        alloc_and_commit(&mut cache, hashes[0], 0);
        alloc_and_commit(&mut cache, hashes[1], hashes[0]);

        // Lookup all 3 — should match first 2
        let matched = cache.longest_prefix_match(&hashes);
        assert_eq!(matched, 2);
        // Refcount bumped for matched pages
        assert_eq!(cache.refcount(hashes[0]), 2); // 1 from commit + 1 from match
        assert_eq!(cache.refcount(hashes[1]), 2);
    }

    #[test]
    fn test_index_cache_move_on_commit() {
        let mut cache = DevicePageCache::new(4, 100, 10);

        let tokens: Vec<u32> = (0..8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let masks = make_brle(8);
        let hashes = cache.compute_page_hashes(&tokens, &positions, &masks, 0);

        let phys1 = alloc_and_commit(&mut cache, hashes[0], 0);
        cache.update_index_cache(hashes[0], None, &[phys1]);

        // Now commit page 2 — index_cache should move from h0 to h1
        let phys2 = alloc_and_commit(&mut cache, hashes[1], hashes[0]);
        cache.update_index_cache(hashes[1], Some(hashes[0]), &[phys2]);

        // h0 entry should be gone
        assert!(!cache.index_cache.contains_key(&hashes[0]));
        // h1 entry should have both pages
        let resolved = cache.resolve_physical(hashes[1]);
        assert_eq!(resolved, vec![phys1, phys2]);
    }
}
