//! KV Cache - Content-Addressable Storage (CAS) for KV cache pages
//!
//! This module provides a model-specific actor for managing KV cache pages
//! using content-addressable storage. Pages are identified by their content hash,
//! enabling efficient deduplication and sharing across contexts.
//!
//! ## Hierarchical Hashing (Merkle-style)
//!
//! Uses fixed power-of-2 prefix boundaries for O(log N) cache lookups:
//! - Prefix hashes at boundaries: 16, 64, 256, 1024, 4096 pages
//! - Tail hashes for pages beyond the largest matched boundary
//!
//! This enables fast "short-circuit" lookups for shared prefixes like system prompts.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use tokio::sync::oneshot;
use anyhow::Result;

use crate::actor::{Handle, Actors, SendError};

/// Hash key for content-addressable pages
pub type PageHash = u64;

/// Physical page index in GPU memory
pub type PhysicalPageId = usize;

/// Node identifier
pub type NodeId = usize;

/// Load threshold for spillover (fraction of capacity)
const LOAD_THRESHOLD: f64 = 0.8;

/// Fixed prefix boundaries (power-of-2 aligned)
const PREFIX_BOUNDARIES: &[usize] = &[16, 64, 256, 1024, 4096];

/// Global registry for KV cache actors.
static ACTOR: LazyLock<Actors<Message>> = LazyLock::new(Actors::new);

/// Spawns a new KV cache actor.
pub(crate) fn spawn() -> usize {
    ACTOR.spawn::<KvCacheActor>()
}

// ============================================================================
// HashTree: Merkle-style hierarchical hashing
// ============================================================================

/// Hierarchical hash structure for efficient prefix matching.
///
/// Combines coarse prefix hashes (at fixed boundaries) with fine-grained
/// tail hashes for pages beyond the largest boundary.
#[derive(Debug, Clone)]
pub struct HashTree {
    /// Prefix hashes at power-of-2 boundaries: (boundary, hash)
    /// e.g., [(16, H(0..16)), (64, H(0..64)), ...]
    prefix_hashes: Vec<(usize, PageHash)>,
    
    /// Individual page hashes for the tail (beyond largest prefix boundary)
    tail_hashes: Vec<PageHash>,
    
    /// Starting page index for tail_hashes
    tail_start: usize,
    
    /// Total number of pages
    total_pages: usize,
}

impl HashTree {
    /// Build a HashTree from individual page hashes.
    ///
    /// Each page hash is H(tokens[0..page_end]) for that page.
    pub fn from_page_hashes(page_hashes: &[PageHash]) -> Self {
        let total_pages = page_hashes.len();
        let mut prefix_hashes = Vec::new();
        let mut largest_boundary = 0;
        
        // Build prefix hashes at each boundary
        for &boundary in PREFIX_BOUNDARIES {
            if total_pages >= boundary {
                // Merkle-style: hash the prefix hashes together
                let prefix_hash = Self::merkle_hash(&page_hashes[..boundary]);
                prefix_hashes.push((boundary, prefix_hash));
                largest_boundary = boundary;
            }
        }
        
        // Tail = individual hashes beyond the largest boundary
        let tail_start = largest_boundary;
        let tail_hashes = if tail_start < total_pages {
            page_hashes[tail_start..].to_vec()
        } else {
            Vec::new()
        };
        
        HashTree {
            prefix_hashes,
            tail_hashes,
            tail_start,
            total_pages,
        }
    }
    
    /// Compute Merkle hash of a slice of page hashes.
    fn merkle_hash(hashes: &[PageHash]) -> PageHash {
        let mut hasher = DefaultHasher::new();
        for h in hashes {
            h.hash(&mut hasher);
        }
        hasher.finish()
    }
    
    /// Get the largest prefix hash (most coarse).
    pub fn largest_prefix(&self) -> Option<(usize, PageHash)> {
        self.prefix_hashes.last().copied()
    }
    
    /// Find the longest cached prefix in a store.
    ///
    /// Returns the number of pages that are cached (as a contiguous prefix).
    pub fn find_cached_prefix(&self, store: &PageStore) -> usize {
        let mut cached_pages = 0;
        
        // 1. Check prefix hashes (coarse, fast) - from smallest to largest
        for &(boundary, hash) in &self.prefix_hashes {
            if store.has_prefix_hash(hash) {
                cached_pages = boundary;
            } else {
                // Prefix not found, no point checking larger ones
                break;
            }
        }
        
        // 2. Check tail hashes individually (fine, but small)
        for (i, &hash) in self.tail_hashes.iter().enumerate() {
            let page_idx = self.tail_start + i;
            if store.has_page_hash(hash) {
                cached_pages = page_idx + 1;
            } else {
                break;
            }
        }
        
        cached_pages
    }
    
    /// Get all hashes that need to be stored (prefix + tail).
    pub fn all_hashes(&self) -> impl Iterator<Item = PageHash> + '_ {
        self.prefix_hashes.iter().map(|(_, h)| *h)
            .chain(self.tail_hashes.iter().copied())
    }
    
    pub fn total_pages(&self) -> usize {
        self.total_pages
    }
}

// ============================================================================
// Messages
// ============================================================================

/// Messages for the KV cache actor.
#[derive(Debug)]
pub enum Message {
    /// Adds a new node/GPU to the cache pool
    AddNode {
        total_pages: usize,
        response: oneshot::Sender<NodeId>,
    },

    /// Lookup cached prefix using HashTree, returns (node_id, cached_pages)
    GetCachedPrefix {
        tree: HashTree,
        response: oneshot::Sender<Option<(NodeId, usize)>>,
    },

    /// Store pages using HashTree for routing
    Put {
        tree: HashTree,
        response: oneshot::Sender<Vec<(NodeId, PhysicalPageId)>>,
    },

    /// Decrement reference count for pages by hash
    Unref {
        hashes: Vec<PageHash>,
        node_hint: Option<NodeId>,
    },
}

impl Message {
    pub fn send(self, model_idx: usize) -> Result<(), SendError> {
        ACTOR.send(model_idx, self)
    }
}

// ============================================================================
// PageStore: Per-node storage
// ============================================================================

/// Per-node page storage
#[derive(Debug)]
struct PageStore {
    node_id: NodeId,
    total_pages: usize,
    
    /// Individual page hashes -> physical page ID
    page_hash_to_id: HashMap<PageHash, PhysicalPageId>,
    page_id_to_hash: HashMap<PhysicalPageId, PageHash>,
    ref_counts: HashMap<PhysicalPageId, usize>,
    free_pages: Vec<PhysicalPageId>,
    
    /// Prefix hashes (Merkle interior nodes) - just need existence check
    prefix_hashes: HashSet<PageHash>,
}

impl PageStore {
    fn new(node_id: NodeId, total_pages: usize) -> Self {
        PageStore {
            node_id,
            total_pages,
            page_hash_to_id: HashMap::new(),
            page_id_to_hash: HashMap::new(),
            ref_counts: HashMap::new(),
            free_pages: (0..total_pages).collect(),
            prefix_hashes: HashSet::new(),
        }
    }

    fn load_factor(&self) -> f64 {
        if self.total_pages == 0 {
            1.0
        } else {
            (self.total_pages - self.free_pages.len()) as f64 / self.total_pages as f64
        }
    }

    fn is_overloaded(&self) -> bool {
        self.load_factor() >= LOAD_THRESHOLD
    }

    /// Check if a prefix hash exists (Merkle interior node)
    fn has_prefix_hash(&self, hash: PageHash) -> bool {
        self.prefix_hashes.contains(&hash)
    }
    
    /// Check if an individual page hash exists
    fn has_page_hash(&self, hash: PageHash) -> bool {
        self.page_hash_to_id.contains_key(&hash)
    }

    /// Store a prefix hash (Merkle interior node)
    fn put_prefix_hash(&mut self, hash: PageHash) {
        self.prefix_hashes.insert(hash);
    }

    /// Store an individual page hash, returns physical page ID
    fn put_page(&mut self, hash: PageHash) -> Option<PhysicalPageId> {
        // Dedup: if already exists, bump ref count
        if let Some(&page_id) = self.page_hash_to_id.get(&hash) {
            *self.ref_counts.entry(page_id).or_insert(1) += 1;
            return Some(page_id);
        }
        // Allocate new page
        if let Some(page_id) = self.free_pages.pop() {
            self.page_hash_to_id.insert(hash, page_id);
            self.page_id_to_hash.insert(page_id, hash);
            self.ref_counts.insert(page_id, 1);
            Some(page_id)
        } else {
            None
        }
    }

    fn unref(&mut self, hash: &PageHash) -> bool {
        if let Some(&page_id) = self.page_hash_to_id.get(hash) {
            if let Some(count) = self.ref_counts.get_mut(&page_id) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.ref_counts.remove(&page_id);
                    self.page_id_to_hash.remove(&page_id);
                    self.page_hash_to_id.remove(hash);
                    self.free_pages.push(page_id);
                }
                return true;
            }
        }
        false
    }
}

// ============================================================================
// KvCacheActor
// ============================================================================

/// The KV cache actor manages multiple page stores with routing.
#[derive(Debug)]
struct KvCacheActor {
    stores: Vec<PageStore>,
}

impl KvCacheActor {
    /// Hash-Affinity + Spillover routing using the tree's largest prefix
    fn route(&self, tree: &HashTree) -> Option<NodeId> {
        if self.stores.is_empty() {
            return None;
        }

        // Use largest prefix hash for affinity
        let affinity_hash = tree.largest_prefix()
            .map(|(_, h)| h)
            .unwrap_or(0);
        
        let primary = (affinity_hash as usize) % self.stores.len();

        // 1. Primary not overloaded → use it
        if !self.stores[primary].is_overloaded() {
            return Some(primary);
        }

        // 2. Find node that has the largest prefix cached
        for &(boundary, hash) in tree.prefix_hashes.iter().rev() {
            let cached = self.stores.iter()
                .filter(|s| !s.is_overloaded() && s.has_prefix_hash(hash))
                .min_by(|a, b| a.load_factor().partial_cmp(&b.load_factor()).unwrap());
            if let Some(store) = cached {
                return Some(store.node_id);
            }
        }

        // 3. Spillover to least-loaded
        self.stores.iter()
            .min_by(|a, b| a.load_factor().partial_cmp(&b.load_factor()).unwrap())
            .map(|s| s.node_id)
    }
}

impl Handle for KvCacheActor {
    type Message = Message;

    fn new() -> Self {
        KvCacheActor { stores: Vec::new() }
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::AddNode { total_pages, response } => {
                let node_id = self.stores.len();
                self.stores.push(PageStore::new(node_id, total_pages));
                let _ = response.send(node_id);
            }

            Message::GetCachedPrefix { tree, response } => {
                // Find the best node with cached prefix
                let mut best: Option<(NodeId, usize)> = None;
                
                for store in &self.stores {
                    let cached = tree.find_cached_prefix(store);
                    if cached > best.map(|(_, c)| c).unwrap_or(0) {
                        best = Some((store.node_id, cached));
                    }
                }
                
                let _ = response.send(best);
            }

            Message::Put { tree, response } => {
                let mut results = Vec::new();
                
                // Route to best node
                if let Some(node_id) = self.route(&tree) {
                    let store = &mut self.stores[node_id];
                    
                    // Store prefix hashes (Merkle interior nodes)
                    for &(_, hash) in &tree.prefix_hashes {
                        store.put_prefix_hash(hash);
                    }
                    
                    // Store tail page hashes
                    for &hash in &tree.tail_hashes {
                        if let Some(page_id) = store.put_page(hash) {
                            results.push((node_id, page_id));
                        }
                    }
                }
                
                let _ = response.send(results);
            }

            Message::Unref { hashes, node_hint } => {
                for hash in hashes {
                    let found = if let Some(node_id) = node_hint {
                        self.stores.get_mut(node_id)
                            .map(|s| s.unref(&hash))
                            .unwrap_or(false)
                    } else {
                        false
                    };
                    
                    if !found {
                        for store in &mut self.stores {
                            if store.unref(&hash) {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}
