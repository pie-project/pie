//! KV Cache - Content-Addressable Storage (CAS) for KV cache pages
//!
//! This module provides a model-specific actor for managing KV cache pages
//! using content-addressable storage. Pages are identified by their content hash,
//! enabling efficient deduplication and sharing across contexts.

use std::collections::HashMap;
use std::sync::LazyLock;
use anyhow::Result;

use crate::actor::{Handle, Actors, SendError};

/// Hash key for content-addressable pages
pub type PageHash = u64;

/// Physical page index in GPU memory
pub type PhysicalPageId = usize;

/// Global registry for KV cache actors.
static ACTOR: LazyLock<Actors<Message>> = LazyLock::new(Actors::new);

/// Spawns a new KV cache actor.
pub(crate) fn spawn() -> usize {
    ACTOR.spawn::<KvCacheActor>()
}

/// Messages for the KV cache actor.
#[derive(Debug)]
pub enum Message {
    /// Translates a list of content hashes to physical page IDs.
    Translate {
        hashes: Vec<PageHash>,
        response: tokio::sync::oneshot::Sender<Result<Vec<PhysicalPageId>>>,
    },

    /// Allocates new physical pages and returns their IDs.
    Allocate {
        count: usize,
        response: tokio::sync::oneshot::Sender<Result<Vec<PhysicalPageId>>>,
    },

    /// Deallocates physical pages, decrementing their reference counts.
    Deallocate {
        page_ids: Vec<PhysicalPageId>,
        response: tokio::sync::oneshot::Sender<Result<()>>,
    },

    /// Registers content for a physical page, making it addressable by hash.
    Register {
        page_id: PhysicalPageId,
        hash: PageHash,
        response: tokio::sync::oneshot::Sender<Result<()>>,
    },

    /// Increments reference count for pages (used when forking contexts).
    AddRef {
        page_ids: Vec<PhysicalPageId>,
        response: tokio::sync::oneshot::Sender<Result<()>>,
    },

    /// Gets statistics about the KV cache.
    Stats {
        response: tokio::sync::oneshot::Sender<KvCacheStats>,
    },
}

impl Message {
    /// Sends this message to the KV cache actor for the given model.
    pub fn send(self, model_idx: usize) -> Result<(), SendError> {
        ACTOR.send(model_idx, self)
    }
}

/// Statistics about the KV cache.
#[derive(Debug, Clone, Default)]
pub struct KvCacheStats {
    pub total_pages: usize,
    pub used_pages: usize,
    pub unique_hashes: usize,
    pub total_refs: usize,
}

/// Internal representation of a physical page.
#[derive(Debug, Clone)]
struct PhysicalPage {
    hash: Option<PageHash>,
    ref_count: usize,
    gpu_ptr: usize,
}

impl PhysicalPage {
    fn new(gpu_ptr: usize) -> Self {
        PhysicalPage {
            hash: None,
            ref_count: 1,
            gpu_ptr,
        }
    }
}

/// The KV cache actor manages content-addressable KV cache pages.
#[derive(Debug)]
struct KvCacheActor {
    pages: HashMap<PhysicalPageId, PhysicalPage>,
    hash_to_page: HashMap<PageHash, PhysicalPageId>,
    next_page_id: PhysicalPageId,
    next_gpu_ptr: usize,
    page_size: usize,
}

impl Handle for KvCacheActor {
    type Message = Message;

    fn new() -> Self {
        KvCacheActor {
            pages: HashMap::new(),
            hash_to_page: HashMap::new(),
            next_page_id: 1,
            next_gpu_ptr: 0x1000,
            page_size: 16,
        }
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Translate { hashes, response } => {
                let mut page_ids = Vec::with_capacity(hashes.len());
                
                for hash in hashes {
                    let page_id = if let Some(&existing_id) = self.hash_to_page.get(&hash) {
                        if let Some(page) = self.pages.get_mut(&existing_id) {
                            page.ref_count += 1;
                        }
                        existing_id
                    } else {
                        let new_id = self.allocate_page();
                        if let Some(page) = self.pages.get_mut(&new_id) {
                            page.hash = Some(hash);
                        }
                        self.hash_to_page.insert(hash, new_id);
                        new_id
                    };
                    page_ids.push(page_id);
                }
                
                let _ = response.send(Ok(page_ids));
            }

            Message::Allocate { count, response } => {
                let mut page_ids = Vec::with_capacity(count);
                for _ in 0..count {
                    page_ids.push(self.allocate_page());
                }
                let _ = response.send(Ok(page_ids));
            }

            Message::Deallocate { page_ids, response } => {
                for page_id in page_ids {
                    if let Some(page) = self.pages.get_mut(&page_id) {
                        page.ref_count = page.ref_count.saturating_sub(1);
                        if page.ref_count == 0 {
                            if let Some(hash) = page.hash {
                                self.hash_to_page.remove(&hash);
                            }
                        }
                    }
                }
                let _ = response.send(Ok(()));
            }

            Message::Register { page_id, hash, response } => {
                let result = if let Some(page) = self.pages.get_mut(&page_id) {
                    if let Some(&existing_id) = self.hash_to_page.get(&hash) {
                        if existing_id != page_id {
                            Err(anyhow::anyhow!("Hash already registered to page {}", existing_id))
                        } else {
                            Ok(())
                        }
                    } else {
                        page.hash = Some(hash);
                        self.hash_to_page.insert(hash, page_id);
                        Ok(())
                    }
                } else {
                    Err(anyhow::anyhow!("Page {} not found", page_id))
                };
                let _ = response.send(result);
            }

            Message::AddRef { page_ids, response } => {
                for page_id in page_ids {
                    if let Some(page) = self.pages.get_mut(&page_id) {
                        page.ref_count += 1;
                    }
                }
                let _ = response.send(Ok(()));
            }

            Message::Stats { response } => {
                let stats = KvCacheStats {
                    total_pages: self.pages.len(),
                    used_pages: self.pages.values().filter(|p| p.ref_count > 0).count(),
                    unique_hashes: self.hash_to_page.len(),
                    total_refs: self.pages.values().map(|p| p.ref_count).sum(),
                };
                let _ = response.send(stats);
            }
        }
    }
}

impl KvCacheActor {
    fn allocate_page(&mut self) -> PhysicalPageId {
        let page_id = self.next_page_id;
        self.next_page_id += 1;
        
        let gpu_ptr = self.next_gpu_ptr;
        self.next_gpu_ptr += self.page_size * 1024;
        
        let page = PhysicalPage::new(gpu_ptr);
        self.pages.insert(page_id, page);
        
        page_id
    }
}
