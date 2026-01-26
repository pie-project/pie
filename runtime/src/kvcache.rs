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
//! - Use `GetCachedPrefix` with `HashTree` for fast routing discovery
//! - Use `Put` with flat `Vec<PageHash>` for robust allocation
//! - Use `Unref` with `HashTree` to clean up both pages and prefix markers

use std::collections::HashMap;
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
#[derive(Debug, Clone)]
pub struct HashTree {
    /// Prefix hashes at power-of-2 boundaries: (boundary, hash)
    prefix_hashes: Vec<(usize, PageHash)>,
    
    /// Full list of page hashes (needed for Unref correct counting)
    all_hashes: Vec<PageHash>,
}

impl HashTree {
    /// Build a HashTree from individual page hashes.
    pub fn from_page_hashes(page_hashes: &[PageHash]) -> Self {
        let total_pages = page_hashes.len();
        let mut prefix_hashes = Vec::new();
        
        // Build prefix hashes at each boundary
        for &boundary in PREFIX_BOUNDARIES {
            if total_pages >= boundary {
                let prefix_hash = Self::merkle_hash(&page_hashes[..boundary]);
                prefix_hashes.push((boundary, prefix_hash));
            }
        }
        
        HashTree {
            prefix_hashes,
            all_hashes: page_hashes.to_vec(),
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
    
    /// Get the largest prefix hash for routing.
    pub fn largest_prefix(&self) -> Option<(usize, PageHash)> {
        self.prefix_hashes.last().copied()
    }
    
    /// Find the longest cached prefix in a store.
    pub fn find_cached_prefix(&self, store: &PageStore) -> usize {
        let mut cached_pages = 0;
        
        // 1. Check prefix hashes (coarse, fast)
        for &(boundary, hash) in &self.prefix_hashes {
            if store.has_prefix_hash(hash) {
                cached_pages = boundary;
            } else {
                break;
            }
        }
        
        // 2. Check remaining individual pages
        for (i, &hash) in self.all_hashes.iter().enumerate().skip(cached_pages) {
            if store.has_page_hash(hash) {
                cached_pages = i + 1;
            } else {
                break;
            }
        }
        
        cached_pages
    }
}


// how about "hot pages" as in beam search or spec dec?
// maybe commit the cold ones, and let inferlets control the workspace kvcache?


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
    Get {
        tree: HashTree,
        response: oneshot::Sender<Option<(NodeId, usize)>>,
    },

    /// Store pages using flat hashes (safe, constructs prefixes internally)
    Put {
        hashes: Vec<PageHash>,
        response: oneshot::Sender<Vec<(NodeId, PhysicalPageId)>>,
    },

    /// Decrement ref counts using Tree (cleans up prefixes and pages)
    Unref {
        tree: HashTree,
        node_hint: Option<NodeId>,
    },

    Allocate {
        num_pages: usize,
        node: NodeId,
        response: oneshot::Sender<Vec<(NodeId, PhysicalPageId)>>,
    },

    Free {
        page_ids: Vec<(NodeId, PhysicalPageId)>,
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
    
    /// Reference counts for pages
    ref_counts: HashMap<PhysicalPageId, usize>,
    free_pages: Vec<PhysicalPageId>,
    
    /// Reference counts for prefix hashes (Merkle nodes)
    prefix_ref_counts: HashMap<PageHash, usize>,
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
            prefix_ref_counts: HashMap::new(),
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

    /// Check if a prefix hash exists and has refs
    fn has_prefix_hash(&self, hash: PageHash) -> bool {
        self.prefix_ref_counts.contains_key(&hash)
    }
    
    /// Check if an individual page hash exists
    fn has_page_hash(&self, hash: PageHash) -> bool {
        self.page_hash_to_id.contains_key(&hash)
    }

    /// Register a prefix hash (increment ref count)
    fn ref_prefix(&mut self, hash: PageHash) {
        *self.prefix_ref_counts.entry(hash).or_insert(0) += 1;
    }

    /// Deregister a prefix hash (decrement ref count)
    fn unref_prefix(&mut self, hash: PageHash) {
        if let Some(count) = self.prefix_ref_counts.get_mut(&hash) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.prefix_ref_counts.remove(&hash);
            }
        }
    }

    /// Store individual page, returns physical page ID
    fn put_page(&mut self, hash: PageHash) -> Option<PhysicalPageId> {
        if let Some(&page_id) = self.page_hash_to_id.get(&hash) {
            *self.ref_counts.entry(page_id).or_insert(1) += 1;
            return Some(page_id);
        }
        if let Some(page_id) = self.free_pages.pop() {
            self.page_hash_to_id.insert(hash, page_id);
            self.page_id_to_hash.insert(page_id, hash);
            self.ref_counts.insert(page_id, 1);
            Some(page_id)
        } else {
            None
        }
    }

    /// Unref individual page
    fn unref_page(&mut self, hash: &PageHash) -> bool {
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

#[derive(Debug)]
struct KvCacheActor {
    stores: Vec<PageStore>,
}

impl KvCacheActor {
    /// Use HashTree (built locally) to route `Put` requests
    fn route(&self, tree: &HashTree) -> Option<NodeId> {
        if self.stores.is_empty() { return None; }

        let affinity_hash = tree.largest_prefix().map(|(_, h)| h).unwrap_or(0);
        let primary = (affinity_hash as usize) % self.stores.len();

        if !self.stores[primary].is_overloaded() {
            return Some(primary);
        }

        // Search for cache hits using prefixes
        for &(boundary, hash) in tree.prefix_hashes.iter().rev() {
            let cached = self.stores.iter()
                .filter(|s| !s.is_overloaded() && s.has_prefix_hash(hash))
                .min_by(|a, b| a.load_factor().partial_cmp(&b.load_factor()).unwrap());
            if let Some(store) = cached {
                return Some(store.node_id);
            }
        }

        // Spillover
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

            Message::Get { tree, response } => {
                let mut best: Option<(NodeId, usize)> = None;
                for store in &self.stores {
                    let cached = tree.find_cached_prefix(store);
                    if cached > best.map(|(_, c)| c).unwrap_or(0) {
                        best = Some((store.node_id, cached));
                    }
                }
                let _ = response.send(best);
            }

            Message::Put { hashes, response } => {
                // Build tree locally to determine routing and update prefixes
                let tree = HashTree::from_page_hashes(&hashes);
                let mut results = Vec::new();

                if let Some(node_id) = self.route(&tree) {
                    let store = &mut self.stores[node_id];
                    
                    // 1. Ref count the prefixes (Merkle nodes)
                    for &(_, hash) in &tree.prefix_hashes {
                        store.ref_prefix(hash);
                    }

                    // 2. Alloc/Ref the actual pages
                    for &hash in &hashes {
                        if let Some(page_id) = store.put_page(hash) {
                            results.push((node_id, page_id));
                        }
                    }
                }
                let _ = response.send(results);
            }

            Message::Unref { tree, node_hint } => {
                // Remove prefixes if hinted
                if let Some(node_id) = node_hint {
                     if let Some(store) = self.stores.get_mut(node_id) {
                         for &(_, hash) in &tree.prefix_hashes {
                             store.unref_prefix(hash);
                         }
                     }
                }

                // Remove pages (search everywhere)
                for hash in tree.all_hashes {
                    let found = if let Some(node_id) = node_hint {
                        self.stores.get_mut(node_id).map(|s| s.unref_page(&hash)).unwrap_or(false)
                    } else {
                        false
                    };

                    if !found {
                        for store in &mut self.stores {
                            if store.unref_page(&hash) {
                                break; 
                            }
                        }
                    }
                }
            }
        }
    }
}
