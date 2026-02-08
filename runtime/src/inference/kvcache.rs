//! KV Cache - Content-Addressable Storage (CAS) for KV cache pages
//!
//! This module provides the PageStore for managing KV cache pages
//! using content-addressable storage. Pages are identified by their content hash,
//! enabling efficient deduplication and sharing across contexts.
//!
//! `PageStore` uses interior mutability via `Arc<Inner>`, so it is `Clone` and
//! all public methods take `&self`.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};
use rustc_hash::{FxHashMap, FxHasher};
use anyhow::Result;

use crate::inference::brle::Brle;

/// Hash key for content-addressable pages
pub type PageHash = u64;

pub type PageId = usize;

/// Physical page index in GPU memory
pub type PhysicalPageId = u32;

/// Device identifier
pub type DeviceId = u8;

/// Load threshold for spillover (fraction of capacity)
const LOAD_THRESHOLD: f64 = 0.8;


#[derive(Debug)]
pub enum Mapping {
    Single(DeviceId, PhysicalPageId),
    Replicated(Vec<(DeviceId, PhysicalPageId)>),
}

#[derive(Debug)]
pub struct Page {
    pub mapping: Mapping,
    pub hash: Option<PageHash>,  // None if mutable (not yet committed)
    pub refcount: u16,
    pub max_position: Option<u32>,  // Highest position ID in this page (None if uncommitted)
}

impl Page {
    fn is_mutable(&self) -> bool {
        self.hash.is_none()
    }

    fn device_id(&self) -> DeviceId {
        match &self.mapping {
            Mapping::Single(device_id, _) => *device_id,
            Mapping::Replicated(mappings) => mappings[0].0,
        }
    }

    /// Add mapping from another page, deduplicating by device.
    /// Returns redundant physical pages that should be freed.
    fn add_mapping(&mut self, other: Mapping) -> Vec<(DeviceId, PhysicalPageId)> {
        let new_entries = match other {
            Mapping::Single(n, p) => vec![(n, p)],
            Mapping::Replicated(v) => v,
        };

        let mut all_entries = match &self.mapping {
            Mapping::Single(n, p) => vec![(*n, *p)],
            Mapping::Replicated(v) => v.clone(),
        };

        let mut redundant = Vec::new();

        // Deduplicate by device: keep first occurrence, mark rest as redundant
        for (new_device, new_phys) in new_entries {
            if all_entries.iter().any(|(n, _)| *n == new_device) {
                redundant.push((new_device, new_phys));
            } else {
                all_entries.push((new_device, new_phys));
            }
        }

        // Update mapping
        self.mapping = if all_entries.len() == 1 {
            Mapping::Single(all_entries[0].0, all_entries[0].1)
        } else {
            Mapping::Replicated(all_entries)
        };

        redundant
    }
}

// ============================================================================
// PageStore: Centralized logical page storage with interior mutability
// ============================================================================

/// Handle to the shared page store. Cheap to clone.
#[derive(Clone)]
pub struct PageStore {
    inner: Arc<Inner>,
}

struct Inner {
    /// Tokens per page (immutable after construction).
    page_size: usize,
    /// Logical page metadata (pages, index, free list, LRU).
    page_table: RwLock<PageTable>,
    /// Per-device physical page storage.
    phys_page_tables: boxcar::Vec<Mutex<PhysicalPageStore>>,
}

struct PageTable {
    pages: Vec<Option<Page>>,
    index: FxHashMap<PageHash, PageId>,
    free_page_ids: Vec<PageId>,
    lru: VecDeque<PageId>,
}

impl std::fmt::Debug for PageStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageStore")
            .field("page_size", &self.inner.page_size)
            .finish()
    }
}

// Per-device physical page storage
#[derive(Debug)]
struct PhysicalPageStore {
    total_pages: usize,
    free_pages: Vec<PhysicalPageId>,
}

impl PhysicalPageStore {
    fn new(total_pages: usize) -> Self {
        PhysicalPageStore {
            total_pages,
            free_pages: (0..total_pages as PhysicalPageId).collect(),
        }
    }

    fn load_factor(&self) -> f64 {
        if self.total_pages == 0 {
            1.0
        } else {
            (self.total_pages - self.free_pages.len()) as f64 / self.total_pages as f64
        }
    }

    #[allow(dead_code)]
    fn is_overloaded(&self) -> bool {
        self.load_factor() >= LOAD_THRESHOLD
    }
}


impl PageStore {
    pub async fn new(page_size: usize, device_indices: &[usize]) -> Self {
        let phys_page_tables = boxcar::Vec::new();
        for &idx in device_indices {
            let info = crate::device::get_info(idx).await
                .unwrap_or_else(|e| panic!("Failed to get device info for index {idx}: {e}"));
            phys_page_tables.push(Mutex::new(PhysicalPageStore::new(info.num_kv_pages)));
        }
        PageStore {
            inner: Arc::new(Inner {
                page_size,
                page_table: RwLock::new(PageTable {
                    pages: Vec::new(),
                    index: FxHashMap::default(),
                    free_page_ids: Vec::new(),
                    lru: VecDeque::new(),
                }),
                phys_page_tables,
            }),
        }
    }

    /// Get the page size (number of tokens per page).
    pub fn page_size(&self) -> usize {
        self.inner.page_size
    }

    /// Compute hash chain from tokens.
    fn compute_hashes(&self, tokens: &[u32]) -> Vec<PageHash> {
        let page_size = self.inner.page_size;
        let mut hashes = Vec::new();
        let mut prev_hash: PageHash = 0;

        for chunk in tokens.chunks(page_size) {
            let mut hasher = FxHasher::default();
            chunk.hash(&mut hasher);
            let content_hash = hasher.finish();

            let mut chain_hasher = FxHasher::default();
            content_hash.hash(&mut chain_hasher);
            prev_hash.hash(&mut chain_hasher);
            let page_hash = chain_hasher.finish();

            hashes.push(page_hash);
            prev_hash = page_hash;
        }

        hashes
    }

    /// Lookup cached pages by token prefix (longest prefix match).
    /// Automatically increments refcount for matched pages.
    pub fn lookup(&self, tokens: &[u32]) -> Option<Vec<PageId>> {
        if tokens.is_empty() {
            return None;
        }

        let hashes = self.compute_hashes(tokens);
        let mut pt = self.inner.page_table.write().unwrap();
        let mut matched_pages = Vec::new();

        for hash in &hashes {
            if let Some(&page_id) = pt.index.get(hash) {
                // Check refcount via shared ref first
                let needs_lru_remove = pt.pages[page_id]
                    .as_ref()
                    .map_or(false, |p| p.refcount == 0);
                if needs_lru_remove {
                    pt.lru.retain(|&id| id != page_id);
                }
                if let Some(page) = pt.pages[page_id].as_mut() {
                    page.refcount += 1;
                    matched_pages.push(page_id);
                }
            } else {
                break;
            }
        }

        if matched_pages.is_empty() {
            None
        } else {
            Some(matched_pages)
        }
    }

    /// Commit pages to the cache, making them immutable and shareable.
    /// Hash chain includes tokens, positions, and masks.
    /// Returns (final_page_ids, final_hash) for continued chaining.
    pub fn commit(
        &self,
        page_ids: &[PageId],
        tokens: &[u32],
        positions: &[u32],
        masks: &[Brle],
        prev_hash: PageHash,
    ) -> Result<(Vec<PageId>, PageHash)> {
        if tokens.len() != positions.len() || tokens.len() != masks.len() {
            anyhow::bail!(
                "Length mismatch: {} tokens, {} positions, {} masks",
                tokens.len(),
                positions.len(),
                masks.len()
            );
        }

        let page_size = self.inner.page_size;

        // Compute hashes with position and mask info
        let mut hashes = Vec::new();
        let mut max_positions = Vec::new();
        let mut running_hash = prev_hash;

        for (chunk_idx, chunk) in tokens.chunks(page_size).enumerate() {
            let start = chunk_idx * page_size;
            let end = start + chunk.len();
            let chunk_positions = &positions[start..end];
            let chunk_masks = &masks[start..end];

            let mut hasher = FxHasher::default();
            chunk.hash(&mut hasher);
            for pos in chunk_positions {
                pos.hash(&mut hasher);
            }
            for mask in chunk_masks {
                mask.hash(&mut hasher);
            }
            let content_hash = hasher.finish();

            let mut chain_hasher = FxHasher::default();
            content_hash.hash(&mut chain_hasher);
            running_hash.hash(&mut chain_hasher);
            let page_hash = chain_hasher.finish();

            hashes.push(page_hash);
            running_hash = page_hash;

            let max_pos = chunk_positions.iter().copied().max();
            max_positions.push(max_pos);
        }

        if page_ids.len() != hashes.len() {
            anyhow::bail!(
                "Page count mismatch: {} pages but {} hashes",
                page_ids.len(),
                hashes.len()
            );
        }

        let mut pt = self.inner.page_table.write().unwrap();
        let mut final_page_ids = Vec::with_capacity(page_ids.len());

        for (i, (&page_id, &hash)) in page_ids.iter().zip(hashes.iter()).enumerate() {
            let max_pos = max_positions[i];

            // Check for existing canonical page
            if let Some(&canonical_id) = pt.index.get(&hash) {
                if canonical_id != page_id {
                    // Deduplicate: merge current page into canonical
                    if let Some(page) = pt.pages[page_id].take() {
                        // Check canonical page refcount before mutable borrow
                        let needs_lru_remove = pt.pages[canonical_id]
                            .as_ref()
                            .map_or(false, |p| p.refcount == 0);
                        if needs_lru_remove {
                            pt.lru.retain(|&id| id != canonical_id);
                        }
                        if let Some(canonical_page) = pt.pages[canonical_id].as_mut() {
                            let redundant = canonical_page.add_mapping(page.mapping);
                            for (device_id, phys_id) in redundant {
                                self.inner.phys_page_tables[device_id as usize]
                                    .lock().unwrap()
                                    .free_pages.push(phys_id);
                            }
                            canonical_page.refcount += 1;
                        }
                    }
                    pt.free_page_ids.push(page_id);
                    final_page_ids.push(canonical_id);
                    continue;
                }
            }

            // Normal commit
            let page = pt.pages[page_id].as_mut().unwrap();
            debug_assert!(page.is_mutable(), "Page {} is already committed", page_id);
            page.hash = Some(hash);
            page.max_position = max_pos;
            pt.index.insert(hash, page_id);
            final_page_ids.push(page_id);
        }

        Ok((final_page_ids, running_hash))
    }

    /// Select the best device for allocation.
    /// If affinity is specified, use that device. Otherwise, use the device with lowest load.
    ///
    /// Caller must hold the page_table lock (passed as `pt`).
    fn select_device(&self, pt: &PageTable, affinity: Option<PageId>) -> Option<DeviceId> {
        if self.inner.phys_page_tables.count() == 0 {
            return None;
        }

        if let Some(affinity_page_id) = affinity {
            if let Some(page) = pt.pages.get(affinity_page_id).and_then(|p| p.as_ref()) {
                return Some(page.device_id());
            }
        }

        // Select device with lowest load
        let mut best: Option<(DeviceId, f64)> = None;
        for (idx, dev) in self.inner.phys_page_tables.iter() {
            let load = dev.lock().unwrap().load_factor();
            if best.is_none() || load < best.unwrap().1 {
                best = Some((idx as DeviceId, load));
            }
        }
        best.map(|(id, _)| id)
    }

    /// Try to evict one page from the LRU (must have refcount=0).
    /// Returns true if a page was evicted.
    ///
    /// Caller must hold the page_table write lock (passed as `pt`).
    fn evict_one(&self, pt: &mut PageTable, target_device: DeviceId) -> bool {
        let mut evict_idx = None;
        for (i, &page_id) in pt.lru.iter().enumerate() {
            if let Some(page) = pt.pages[page_id].as_ref() {
                if page.device_id() == target_device && page.refcount == 0 {
                    evict_idx = Some(i);
                    break;
                }
            }
        }

        if let Some(idx) = evict_idx {
            let page_id = pt.lru.remove(idx).unwrap();
            self.free_page(pt, page_id);
            true
        } else {
            false
        }
    }

    /// Free a page, returning its physical page to the device.
    ///
    /// Caller must hold the page_table write lock (passed as `pt`).
    fn free_page(&self, pt: &mut PageTable, page_id: PageId) {
        if let Some(page) = pt.pages[page_id].take() {
            // Remove from index if committed
            if let Some(hash) = page.hash {
                if let Some(&id) = pt.index.get(&hash) {
                    if id == page_id {
                        pt.index.remove(&hash);
                    }
                }
            }

            // Return physical page to device
            match page.mapping {
                Mapping::Single(device_id, phys_id) => {
                    self.inner.phys_page_tables[device_id as usize]
                        .lock().unwrap()
                        .free_pages.push(phys_id);
                }
                Mapping::Replicated(mappings) => {
                    for (device_id, phys_id) in mappings {
                        self.inner.phys_page_tables[device_id as usize]
                            .lock().unwrap()
                            .free_pages.push(phys_id);
                    }
                }
            }

            // Recycle the page ID
            pt.free_page_ids.push(page_id);
        }
    }

    /// Allocate pages on a specific device.
    pub fn allocate(&self, num_pages: usize, affinity: Option<PageId>) -> Result<Vec<PageId>> {
        let mut pt = self.inner.page_table.write().unwrap();

        let device_id = self.select_device(&pt, affinity)
            .ok_or_else(|| anyhow::anyhow!("No devices registered"))?;

        let mut allocated = Vec::with_capacity(num_pages);

        for _ in 0..num_pages {
            let phys_id = loop {
                let mut dev = self.inner.phys_page_tables[device_id as usize].lock().unwrap();
                if let Some(phys_id) = dev.free_pages.pop() {
                    break phys_id;
                }
                drop(dev); // release device lock before eviction
                if !self.evict_one(&mut pt, device_id) {
                    for page_id in allocated {
                        self.free_page(&mut pt, page_id);
                    }
                    anyhow::bail!("Out of memory on device {}", device_id);
                }
            };

            let page_id = if let Some(id) = pt.free_page_ids.pop() {
                id
            } else {
                let id = pt.pages.len();
                pt.pages.push(None);
                id
            };

            pt.pages[page_id] = Some(Page {
                mapping: Mapping::Single(device_id, phys_id),
                hash: None,
                refcount: 1,
                max_position: None,
            });

            allocated.push(page_id);
        }

        Ok(allocated)
    }

    /// Release pages (decrease refcount).
    pub fn release(&self, page_ids: &[PageId]) {
        let mut pt = self.inner.page_table.write().unwrap();
        for &page_id in page_ids {
            // Check if the page needs freeing before calling free_page
            let should_free = if let Some(page) = pt.pages[page_id].as_mut() {
                page.refcount = page.refcount.saturating_sub(1);
                if page.refcount == 0 {
                    if page.is_mutable() {
                        true // will free below
                    } else {
                        pt.lru.push_back(page_id);
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };
            if should_free {
                self.free_page(&mut pt, page_id);
            }
        }
    }

    /// Acquire pages (increase refcount).
    /// Used when forking contexts to share committed pages.
    pub fn acquire(&self, page_ids: &[PageId]) {
        let mut pt = self.inner.page_table.write().unwrap();
        for &page_id in page_ids {
            let needs_lru_remove = pt.pages[page_id]
                .as_ref()
                .map_or(false, |p| p.refcount == 0);
            if needs_lru_remove {
                pt.lru.retain(|&id| id != page_id);
            }
            if let Some(page) = pt.pages[page_id].as_mut() {
                page.refcount += 1;
            }
        }
    }

    /// Get physical page mappings for a logical page.
    /// Returns the (DeviceId, PhysicalPageId) pairs for a page.
    pub fn get_physical_mappings(&self, page_id: PageId) -> Vec<(DeviceId, PhysicalPageId)> {
        let pt = self.inner.page_table.read().unwrap();
        if let Some(Some(page)) = pt.pages.get(page_id) {
            match &page.mapping {
                Mapping::Single(device_id, phys_id) => vec![(*device_id, *phys_id)],
                Mapping::Replicated(mappings) => mappings.clone(),
            }
        } else {
            Vec::new()
        }
    }
}
