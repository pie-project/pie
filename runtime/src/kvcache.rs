//! KV Cache - Content-Addressable Storage (CAS) for KV cache pages
//!
//! This module provides the PageStore for managing KV cache pages
//! using content-addressable storage. Pages are identified by their content hash,
//! enabling efficient deduplication and sharing across contexts.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
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
        let mut new_entries = match other {
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
            if let Some(_) = all_entries.iter().find(|(n, _)| *n == new_device) {
                // Already have a page on this device, mark new one as redundant
                redundant.push((new_device, new_phys));
            } else {
                // New device, add it
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
// PageStore: Centralized logical page storage
// ============================================================================

#[derive(Debug)]
pub struct PageStore {
    page_size: usize,
    pages: Vec<Option<Page>>, // the index of page is the page id
    index: FxHashMap<PageHash, PageId>,
    free_page_ids: Vec<PageId>, // recycled page IDs
    lru: VecDeque<PageId>, // pages with refcount=0, front=oldest

    /// Per-device storage
    devices: Vec<PhysicalPageStore>,
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
    pub fn new(page_size: usize) -> Self {
        PageStore {
            page_size,
            pages: Vec::new(),
            index: FxHashMap::default(),
            free_page_ids: Vec::new(),
            lru: VecDeque::new(),
            devices: Vec::new(),
        }
    }

    /// Get the page size (number of tokens per page)
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Register a new device and return its ID
    pub fn register_device(&mut self, total_pages: usize) -> DeviceId {
        let device_id = self.devices.len() as DeviceId;
        self.devices.push(PhysicalPageStore::new(total_pages));
        device_id
    }

    /// Compute hash chain from tokens.
    /// Returns a list of (page_index, page_hash) pairs.
    fn compute_hashes(&self, tokens: &[u32]) -> Vec<PageHash> {
        let mut hashes = Vec::new();
        let mut prev_hash: PageHash = 0;

        for chunk in tokens.chunks(self.page_size) {
            // Hash the token content
            let mut hasher = FxHasher::default();
            chunk.hash(&mut hasher);
            let content_hash = hasher.finish();

            // Chain with previous hash
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
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<Vec<PageId>> {
        if tokens.is_empty() {
            return None;
        }

        let hashes = self.compute_hashes(tokens);
        let mut matched_pages = Vec::new();

        for hash in &hashes {
            if let Some(&page_id) = self.index.get(hash) {
                // Increment refcount
                if let Some(page) = self.pages[page_id].as_mut() {
                    if page.refcount == 0 {
                        // Remove from LRU since it's now in use
                        self.lru.retain(|&id| id != page_id);
                    }
                    page.refcount += 1;
                    matched_pages.push(page_id);
                }
            } else {
                // No match, stop here (prefix matching)
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
        &mut self,
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

        // Compute hashes with position and mask info
        let mut hashes = Vec::new();
        let mut max_positions = Vec::new();
        let mut running_hash = prev_hash;

        for (chunk_idx, chunk) in tokens.chunks(self.page_size).enumerate() {
            let start = chunk_idx * self.page_size;
            let end = start + chunk.len();
            let chunk_positions = &positions[start..end];
            let chunk_masks = &masks[start..end];

            // Hash tokens
            let mut hasher = FxHasher::default();
            chunk.hash(&mut hasher);
            
            // Hash positions
            for pos in chunk_positions {
                pos.hash(&mut hasher);
            }
            
            // Hash masks
            for mask in chunk_masks {
                mask.hash(&mut hasher);
            }
            
            let content_hash = hasher.finish();

            // Chain with previous hash
            let mut chain_hasher = FxHasher::default();
            content_hash.hash(&mut chain_hasher);
            running_hash.hash(&mut chain_hasher);
            let page_hash = chain_hasher.finish();

            hashes.push(page_hash);
            running_hash = page_hash;

            // Find max position for this page
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

        let mut final_page_ids = Vec::with_capacity(page_ids.len());

        for (i, (&page_id, &hash)) in page_ids.iter().zip(hashes.iter()).enumerate() {
            let max_pos = max_positions[i];

            // Check for existing canonical page
            if let Some(&canonical_id) = self.index.get(&hash) {
                if canonical_id != page_id {
                    // Deduplicate: merge current page into canonical
                    if let Some(page) = self.pages[page_id].take() {
                        if let Some(canonical_page) = self.pages[canonical_id].as_mut() {
                            if canonical_page.refcount == 0 {
                                self.lru.retain(|&id| id != canonical_id);
                            }
                            let redundant = canonical_page.add_mapping(page.mapping);
                            for (device_id, phys_id) in redundant {
                                self.devices[device_id as usize].free_pages.push(phys_id);
                            }
                            canonical_page.refcount += 1;
                        }
                    }
                    self.free_page_ids.push(page_id);
                    final_page_ids.push(canonical_id);
                    continue;
                }
            }

            // Normal commit
            let page = self.pages[page_id].as_mut().unwrap();
            debug_assert!(page.is_mutable(), "Page {} is already committed", page_id);
            page.hash = Some(hash);
            page.max_position = max_pos;
            self.index.insert(hash, page_id);
            final_page_ids.push(page_id);
        }

        Ok((final_page_ids, running_hash))
    }

    /// Select the best device for allocation.
    /// If affinity is specified, use that device. Otherwise, use the device with lowest load.
    fn select_device(&self, affinity: Option<PageId>) -> Option<DeviceId> {
        if self.devices.is_empty() {
            return None;
        }

        if let Some(affinity_page_id) = affinity {
            if let Some(page) = self.pages.get(affinity_page_id).and_then(|p| p.as_ref()) {
                return Some(page.device_id());
            }
        }

        // Select device with lowest load
        self.devices
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.load_factor()
                    .partial_cmp(&b.load_factor())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx as DeviceId)
    }

    /// Try to evict one page from the LRU (must have refcount=0).
    /// Returns true if a page was evicted.
    fn evict_one(&mut self, target_device: DeviceId) -> bool {
        // Find a page in LRU that belongs to the target device
        let mut evict_idx = None;
        for (i, &page_id) in self.lru.iter().enumerate() {
            if let Some(page) = self.pages[page_id].as_ref() {
                if page.device_id() == target_device && page.refcount == 0 {
                    evict_idx = Some(i);
                    break;
                }
            }
        }

        if let Some(idx) = evict_idx {
            let page_id = self.lru.remove(idx).unwrap();
            self.free_page(page_id);
            true
        } else {
            false
        }
    }

    /// Free a page, returning its physical page to the node.
    fn free_page(&mut self, page_id: PageId) {
        if let Some(page) = self.pages[page_id].take() {
            // Remove from index if committed
            if let Some(hash) = page.hash {
                // Only remove from index if it points to this specific page
                if let Some(&id) = self.index.get(&hash) {
                    if id == page_id {
                        self.index.remove(&hash);
                    }
                }
            }

            // Return physical page to device
            match page.mapping {
                Mapping::Single(device_id, phys_id) => {
                    self.devices[device_id as usize].free_pages.push(phys_id);
                }
                Mapping::Replicated(mappings) => {
                    for (device_id, phys_id) in mappings {
                        self.devices[device_id as usize].free_pages.push(phys_id);
                    }
                }
            }

            // Recycle the page ID
            self.free_page_ids.push(page_id);
        }
    }

    /// Allocate pages on a specific device.
    pub fn allocate(&mut self, num_pages: usize, affinity: Option<PageId>) -> Result<Vec<PageId>> {
        let device_id = self.select_device(affinity)
            .ok_or_else(|| anyhow::anyhow!("No devices registered"))?;

        // Try to allocate, evicting if necessary
        let mut allocated = Vec::with_capacity(num_pages);
        
        for _ in 0..num_pages {
            // Try to get a free physical page, evicting if necessary
            let phys_id = loop {
                if let Some(phys_id) = self.devices[device_id as usize].free_pages.pop() {
                    break phys_id;
                }
                // Try to evict a page from this device
                if !self.evict_one(device_id) {
                    // Failed to evict, rollback allocations
                    for page_id in allocated {
                        self.free_page(page_id);
                    }
                    anyhow::bail!("Out of memory on device {}", device_id);
                }
            };

            // Get or create a page ID
            let page_id = if let Some(id) = self.free_page_ids.pop() {
                id
            } else {
                let id = self.pages.len();
                self.pages.push(None);
                id
            };

            // Create the page
            self.pages[page_id] = Some(Page {
                mapping: Mapping::Single(device_id, phys_id),
                hash: None, // mutable
                refcount: 1,
                max_position: None,
            });

            allocated.push(page_id);
        }

        Ok(allocated)
    }

    /// Release pages (decrease refcount).
    pub fn release(&mut self, page_ids: &[PageId]) {
        for &page_id in page_ids {
            if let Some(page) = self.pages[page_id].as_mut() {
                page.refcount = page.refcount.saturating_sub(1);
                if page.refcount == 0 {
                    if page.is_mutable() {
                        // Mutable pages with refcount=0 are immediately freed
                        self.free_page(page_id);
                    } else {
                        // Committed pages go to LRU for potential reuse
                        self.lru.push_back(page_id);
                    }
                }
            }
        }
    }

    /// Acquire pages (increase refcount).
    /// Used when forking contexts to share committed pages.
    pub fn acquire(&mut self, page_ids: &[PageId]) {
        for &page_id in page_ids {
            if let Some(page) = self.pages[page_id].as_mut() {
                if page.refcount == 0 {
                    // Remove from LRU since it's now in use
                    self.lru.retain(|&id| id != page_id);
                }
                page.refcount += 1;
            }
        }
    }

    /// Get physical page mappings for a logical page.
    /// Returns the (DeviceId, PhysicalPageId) pairs for a page.
    pub fn get_physical_mappings(&self, page_id: PageId) -> Vec<(DeviceId, PhysicalPageId)> {
        if let Some(Some(page)) = self.pages.get(page_id) {
            match &page.mapping {
                Mapping::Single(device_id, phys_id) => vec![(*device_id, *phys_id)],
                Mapping::Replicated(mappings) => mappings.clone(),
            }
        } else {
            Vec::new()
        }
    }
}
