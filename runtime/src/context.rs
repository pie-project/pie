//! Context Service - Execution context management with KV cache state
//!
//! This module provides a model-specific actor for managing named execution
//! contexts with support for forking, joining, locking, and capacity management.

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot;
use anyhow::Result;

use crate::actor::{Handle, Actors, SendError};
use crate::adapter::AdapterId;
use crate::inference::brle::Brle;
use crate::kvcache::{PageId, PageStore, NodeId, PhysicalPageId, PageHash};

/// Unique identifier for a context.
pub type ContextId = u64;
pub type LockId = u64;

/// Global table of context actors.
static ACTOR: LazyLock<Actors<Message>> = LazyLock::new(Actors::new);

/// Spawns a new context actor.
pub(crate) fn spawn() -> usize {
    ACTOR.spawn::<ContextManagerActor>()
}

/// Messages for the context actor.
#[derive(Debug)]
pub enum Message {
    /// Creates a new context with the given name.
    Create {
        user_id: u32,
        name: String,
        fill: Option<Vec<u32>>,
        response: oneshot::Sender<Result<ContextId>>,
    },

    /// Destroys a context.
    Destroy {
        id: ContextId,
        lock_id: LockId,
        response: oneshot::Sender<Result<()>>,
    },

    /// Retrieves an existing context by name.
    Lookup {
        user_id: u32,
        name: String,
        response: oneshot::Sender<Option<ContextId>>,
    },

    /// Forks a context into a new one with the given name.
    Fork {
        id: ContextId,
        user_id: u32,
        new_name: String,
        response: oneshot::Sender<Result<ContextId>>,
    },

    /// Acquires a lock on the context.
    AcquireLock {
        id: ContextId,
        response: oneshot::Sender<LockId>,
    },

    /// Releases the lock on the context.
    ReleaseLock {
        id: ContextId,
        lock_id: LockId,
    },

    /// Get tokens per page
    TokensPerPage {
        id: ContextId,
        response: oneshot::Sender<u32>,
    },

    /// Get number of committed pages
    CommittedPageCount {
        id: ContextId,
        response: oneshot::Sender<u32>,
    },

    /// Commit pages to the context
    CommitPages {
        id: ContextId,
        lock_id: LockId,
        page_indices: Vec<u32>,
        response: oneshot::Sender<Result<()>>,
    },

    /// Reserve pages for the context
    ReservePages {
        id: ContextId,
        lock_id: LockId,
        num_pages: u32,
        response: oneshot::Sender<Result<()>>,
    },

    /// Release pages from the context
    ReleasePages {
        id: ContextId,
        lock_id: LockId,
        num_pages: u32,
    },

    /// Get the cursor position
    GetCursor {
        id: ContextId,
        lock_id: LockId,
        response: oneshot::Sender<u32>,
    },

    /// Set the cursor position
    SetCursor {
        id: ContextId,
        lock_id: LockId,
        cursor: u32,
    },

    /// Get buffered tokens
    GetBufferedTokens {
        id: ContextId,
        lock_id: LockId,
        response: oneshot::Sender<Vec<u32>>,
    },

    /// Set buffered tokens
    SetBufferedTokens {
        id: ContextId,
        lock_id: LockId,
        tokens: Vec<u32>,
    },

    /// Append buffered tokens (avoids get+set roundtrip)
    AppendBufferedTokens {
        id: ContextId,
        lock_id: LockId,
        tokens: Vec<u32>,
    },
}

impl Message {
    /// Sends this message to the context actor for the given model.
    pub fn send(self, model_idx: usize) -> Result<(), SendError> {
        ACTOR.send(model_idx, self)
    }
}

/// Record of operations on a context (for lineage tracking).
#[derive(Debug, Clone)]
enum Record {
    Fill {
        tokens: Vec<u32>,
        positions: Vec<u32>,
        mask: Vec<Brle>,
        adapter: Option<AdapterId>,
    },
}

#[derive(Debug, Clone)]
pub struct TokenInfo {
    pub token: u32,
    pub position: u32,
    pub mask: Brle,
    pub adapter: Option<AdapterId>,
}

// Lineage created during commit_pages()

/// Internal representation of a context.
#[derive(Debug, Clone)]
struct Context {
    lineage: Vec<Record>,
    pages_committed: Vec<PageId>,
    pages_uncommitted: Vec<PageId>,
    tokens_uncommitted: Vec<TokenInfo>,

    pointer: usize,
    mutex: Option<LockId>,
    
    /// Highest position ID among all committed pages
    max_committed_position: Option<u32>,
    /// Hash of the last committed page (for hash chaining)
    last_committed_hash: PageHash,
}


impl Context {
    fn new() -> Self {
        Context {
            lineage: Vec::new(),
            pages_committed: Vec::new(),
            pages_uncommitted: Vec::new(),
            tokens_uncommitted: Vec::new(),
            pointer: 0,
            mutex: None,
            max_committed_position: None,
            last_committed_hash: 0,
        }
    }
}

/// The context manager handles all context operations.
/// This is the core business logic, separate from the actor message handling.
#[derive(Debug)]
pub struct ContextManager {
    pub page_store: Arc<RwLock<PageStore>>,
    contexts: DashMap<ContextId, Context>,
    name_to_id: DashMap<(u32, String), ContextId>,
    next_id: AtomicU64,
    pub page_size: usize,
}

impl ContextManager {
    pub fn new(page_store: Arc<RwLock<PageStore>>, page_size: usize) -> Self {
        ContextManager {
            page_store,
            contexts: DashMap::new(),
            name_to_id: DashMap::new(),
            next_id: AtomicU64::new(1),
            page_size,
        }
    }

    fn next_id(&self) -> ContextId {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    // ==================== Core Operations ====================

    pub fn create(&mut self, user_id: u32, name: String, fill: Option<Vec<u32>>) -> Result<ContextId> {
        let id = self.next_id();
        let mut ctx = Context::new();

        // Process fill tokens if provided
        if let Some(tokens) = fill {
            // Add lineage record
            ctx.lineage.push(Record::Fill {
                tokens: tokens.clone(),
                positions: Vec::new(),
                mask: Vec::new(),
                adapter: None,
            });

            let mut page_store = self.page_store.write().unwrap();
            
            // Lookup existing cached pages (longest prefix match)
            let matched_pages = page_store.lookup(&tokens);
            
            if let Some(pages) = matched_pages {
                // Calculate how many tokens are covered by matched pages
                let tokens_matched = pages.len() * page_store.page_size();
                
                // Acquire refcount for matched pages
                page_store.acquire(&pages);
                
                // Place matched pages in pages_committed
                ctx.pages_committed = pages.clone();
                
                // Set pointer to end of committed tokens
                ctx.pointer = tokens_matched;
                
                // Remaining tokens become tokens_uncommitted
                if tokens_matched < tokens.len() {
                    let remaining_tokens = tokens.len() - tokens_matched;
                    let page_size = page_store.page_size();
                    let pages_needed = (remaining_tokens + page_size - 1) / page_size;
                    
                    // Allocate uncommitted pages
                    let affinity = pages.first().copied();
                    ctx.pages_uncommitted = page_store.allocate(pages_needed, affinity)?;
                    
                    ctx.tokens_uncommitted = tokens[tokens_matched..]
                        .iter()
                        .enumerate()
                        .map(|(i, &token)| TokenInfo { 
                            token, 
                            position: (tokens_matched + i) as u32, 
                            mask: Brle::new(0),
                            adapter: None,
                        })
                        .collect();
                }
            } else {
                // No match: all tokens become tokens_uncommitted
                let page_size = page_store.page_size();
                let pages_needed = (tokens.len() + page_size - 1) / page_size;
                
                // Allocate uncommitted pages (no affinity)
                if pages_needed > 0 {
                    ctx.pages_uncommitted = page_store.allocate(pages_needed, None)?;
                }
                
                ctx.tokens_uncommitted = tokens
                    .into_iter()
                    .enumerate()
                    .map(|(i, token)| TokenInfo { 
                        token, 
                        position: i as u32, 
                        mask: Brle::new(0),
                        adapter: None,
                    })
                    .collect();
            }
        }

        self.contexts.insert(id, ctx);
        self.name_to_id.insert((user_id, name), id);
        Ok(id)
    }

    pub fn destroy(&mut self, id: ContextId, lock_id: LockId) -> Result<()> {
        if let Some(ctx) = self.contexts.get(&id) {
            if ctx.mutex == Some(lock_id) {
                drop(ctx);
                self.contexts.remove(&id);
                Ok(())
            } else {
                Err(anyhow::anyhow!("Context not locked by this lock_id"))
            }
        } else {
            Err(anyhow::anyhow!("Context not found"))
        }
    }

    pub fn get(&self, user_id: u32, name: String) -> Option<ContextId> {
        self.name_to_id.get(&(user_id, name)).map(|v| *v.value())
    }

    pub fn fork(&mut self, id: ContextId, user_id: u32, new_name: String) -> Result<ContextId> {
        // Check if name already exists
        if self.name_to_id.contains_key(&(user_id, new_name.clone())) {
            anyhow::bail!("Context name already exists: {}", new_name);
        }

        // Get the source context
        let source_ctx = self.contexts.get(&id)
            .ok_or_else(|| anyhow::anyhow!("Source context not found"))?;

        // Clone committed pages and collect info we need
        let pages_committed = source_ctx.pages_committed.clone();
        let num_uncommitted = source_ctx.pages_uncommitted.len();
        let tokens_uncommitted = source_ctx.tokens_uncommitted.clone();
        let lineage = source_ctx.lineage.clone();
        let affinity = pages_committed.first().copied();
        let source_max_position = source_ctx.max_committed_position;
        let source_last_hash = source_ctx.last_committed_hash;
        
        drop(source_ctx); // Release the borrow

        let mut page_store = self.page_store.write().unwrap();
        
        // Increase refcount for committed pages
        page_store.acquire(&pages_committed);

        // Allocate new uncommitted pages
        let new_uncommitted = if num_uncommitted > 0 {
            page_store.allocate(num_uncommitted, affinity)?
        } else {
            Vec::new()
        };

        // Calculate pointer
        let pointer = pages_committed.len() * page_store.page_size();

        // Create the new context
        let new_id = self.next_id();
        let new_ctx = Context {
            lineage,
            pages_committed,
            pages_uncommitted: new_uncommitted,
            tokens_uncommitted,
            pointer,
            mutex: None,
            max_committed_position: source_max_position,
            last_committed_hash: source_last_hash,
        };

        self.contexts.insert(new_id, new_ctx);
        self.name_to_id.insert((user_id, new_name), new_id);

        Ok(new_id)
    }

    // ==================== Locking ====================

    pub fn lock(&self, id: ContextId) -> LockId {
        let lock_id = self.next_id();
        if let Some(mut ctx) = self.contexts.get_mut(&id) {
            if ctx.mutex.is_none() {
                ctx.mutex = Some(lock_id);
                return lock_id;
            }
        }
        0 // Lock failed
    }

    pub fn unlock(&self, id: ContextId, lock_id: LockId) {
        if let Some(mut ctx) = self.contexts.get_mut(&id) {
            if ctx.mutex == Some(lock_id) {
                ctx.mutex = None;
            }
        }
    }

    // ==================== Queries ====================

    pub fn page_size(&self) -> u32 {
        self.page_size as u32
    }

    pub fn num_total_pages(&self, id: ContextId) -> u32 {
        self.contexts.get(&id)
            .map(|ctx| ctx.pages_committed.len() as u32)
            .unwrap_or(0)
    }

    pub fn num_total_tokens(&self, id: ContextId) -> u32 {
        self.contexts.get(&id)
            .map(|ctx| ctx.pointer as u32 + ctx.tokens_uncommitted.len() as u32)
            .unwrap_or(0)
    }

    pub fn get_pointer(&self, id: ContextId) -> u32 {
        self.contexts.get(&id)
            .map(|ctx| ctx.pointer as u32)
            .unwrap_or(0)
    }

    pub fn set_pointer(&self, id: ContextId, pointer: u32) -> Result<()> {
        if let Some(mut ctx) = self.contexts.get_mut(&id) {
            ctx.pointer = pointer as usize;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Context not found"))
        }
    }

    pub fn get_buffered_tokens(&self, id: ContextId) -> Vec<u32> {
        self.contexts.get(&id)
            .map(|ctx| ctx.tokens_uncommitted.iter().map(|t| t.token).collect())
            .unwrap_or_default()
    }

    pub fn set_buffered_tokens(&self, id: ContextId, tokens: Vec<u32>) -> Result<()> {
        if let Some(mut ctx) = self.contexts.get_mut(&id) {
            ctx.tokens_uncommitted = tokens.into_iter().enumerate().map(|(i, token)| {
                TokenInfo {
                    token,
                    position: i as u32,
                    mask: Brle::new(0),
                    adapter: None,
                }
            }).collect();
            Ok(())
        } else {
            Err(anyhow::anyhow!("Context not found"))
        }
    }

    pub fn append_buffered_tokens(&self, id: ContextId, tokens: Vec<u32>) -> Result<()> {
        if let Some(mut ctx) = self.contexts.get_mut(&id) {
            let start_pos = ctx.tokens_uncommitted.len();
            ctx.tokens_uncommitted.extend(tokens.into_iter().enumerate().map(|(i, token)| {
                TokenInfo {
                    token,
                    position: (start_pos + i) as u32,
                    mask: Brle::new(0),
                    adapter: None,
                }
            }));
            Ok(())
        } else {
            Err(anyhow::anyhow!("Context not found"))
        }
    }

    pub fn get_physical_page_ids(&self, id: ContextId) -> HashMap<NodeId, Vec<PhysicalPageId>> {
        let mut result: HashMap<NodeId, Vec<PhysicalPageId>> = HashMap::new();
        
        if let Some(ctx) = self.contexts.get(&id) {
            let page_store = self.page_store.read().unwrap();
            
            // Get all committed pages and collect their physical mappings
            for &page_id in &ctx.pages_committed {
                for (node_id, phys_id) in page_store.get_physical_mappings(page_id) {
                    result.entry(node_id).or_default().push(phys_id);
                }
            }
            // Also include uncommitted pages
            for &page_id in &ctx.pages_uncommitted {
                for (node_id, phys_id) in page_store.get_physical_mappings(page_id) {
                    result.entry(node_id).or_default().push(phys_id);
                }
            }
        }
        
        result
    }

    // ==================== Page Management ====================

    pub fn allocate_pages(&mut self, id: ContextId, lock_id: LockId, num_pages: u32) -> Result<()> {
        let ctx = self.contexts.get(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        
        // Verify lock
        if ctx.mutex != Some(lock_id) {
            anyhow::bail!("Invalid lock");
        }
        
        // Get affinity from existing pages (committed first, then uncommitted)
        let affinity = ctx.pages_committed.first()
            .or_else(|| ctx.pages_uncommitted.first())
            .copied();
        
        drop(ctx);
        
        // Allocate pages from page store
        let new_pages = self.page_store.write().unwrap().allocate(num_pages as usize, affinity)?;
        
        // Add to uncommitted pages
        let mut ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        ctx.pages_uncommitted.extend(new_pages);
        
        Ok(())
    }

    pub fn free_pages(&mut self, id: ContextId, lock_id: LockId, num_pages: u32) -> Result<()> {
        let mut ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        
        // Verify lock
        if ctx.mutex != Some(lock_id) {
            anyhow::bail!("Invalid lock");
        }
        
        let num_to_free = (num_pages as usize).min(ctx.pages_uncommitted.len());
        if num_to_free == 0 {
            return Ok(());
        }
        
        // Get the pages to free (from the end)
        let start_idx = ctx.pages_uncommitted.len() - num_to_free;
        let pages_to_free: Vec<PageId> = ctx.pages_uncommitted[start_idx..].to_vec();
        
        // Remove from uncommitted pages
        ctx.pages_uncommitted.truncate(start_idx);
        
        // Remove corresponding tokens from the end
        let tokens_to_remove = num_to_free * self.page_size;
        let tokens_len = ctx.tokens_uncommitted.len();
        if tokens_to_remove > 0 && tokens_len > 0 {
            let new_tokens_len = tokens_len.saturating_sub(tokens_to_remove);
            ctx.tokens_uncommitted.truncate(new_tokens_len);
        }
        
        drop(ctx);
        
        // Release pages back to page store
        self.page_store.write().unwrap().release(&pages_to_free);
        
        Ok(())
    }

    pub fn commit_pages(&mut self, id: ContextId, lock_id: LockId, indices: Vec<u32>) -> Result<()> {
        let page_size = self.page_size;
        
        let mut ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        
        // Verify lock
        if ctx.mutex != Some(lock_id) {
            anyhow::bail!("Context not locked by this lock_id");
        }

        // Validate indices
        for &idx in &indices {
            if idx as usize >= ctx.pages_uncommitted.len() {
                anyhow::bail!("Invalid page index: {}", idx);
            }
        }

        // Gather pages and their corresponding tokens
        let mut pages_to_commit = Vec::new();
        let mut tokens_to_commit = Vec::new();
        let mut positions_to_commit = Vec::new();
        let mut masks_to_commit = Vec::new();
        let mut all_positions = Vec::new();

        for &idx in &indices {
            let idx = idx as usize;
            let page_id = ctx.pages_uncommitted[idx];
            pages_to_commit.push(page_id);

            // Get tokens for this page (index offset mapping)
            let token_start = idx * page_size;
            let token_end = (idx + 1) * page_size;

            // Validate page is fully filled
            if token_end > ctx.tokens_uncommitted.len() {
                anyhow::bail!(
                    "Page {} not fully filled: need {} tokens but only have {}",
                    idx, token_end, ctx.tokens_uncommitted.len()
                );
            }

            // Collect token info
            for i in token_start..token_end {
                let info = &ctx.tokens_uncommitted[i];
                tokens_to_commit.push(info.token);
                positions_to_commit.push(info.position);
                masks_to_commit.push(info.mask.clone());
                
                all_positions.push((info.position, idx));
            }
        }

        // Validate position uniqueness
        let mut pos_set = std::collections::HashSet::new();
        for &(pos, _) in &all_positions {
            if !pos_set.insert(pos) {
                anyhow::bail!("Duplicate position ID: {}", pos);
            }
        }

        // Validate all positions > max_committed_position
        if let Some(max_committed) = ctx.max_committed_position {
            for &(pos, _) in &all_positions {
                if pos <= max_committed {
                    anyhow::bail!(
                        "Position {} must be greater than max committed position {}",
                        pos, max_committed
                    );
                }
            }
        }

        // Validate pages are orderable by max position (no overlaps)
        // Group positions by page index and find max per page
        let mut page_max_positions: Vec<(usize, u32)> = Vec::new();
        for &page_idx in indices.iter().map(|i| *i as usize).collect::<Vec<_>>().iter() {
            let max_pos = all_positions.iter()
                .filter(|(_, idx)| *idx == page_idx)
                .map(|(pos, _)| *pos)
                .max();
            if let Some(max) = max_pos {
                page_max_positions.push((page_idx, max));
            }
        }
        
        // Check ordering: each page's max position should be > previous page's max
        for i in 1..page_max_positions.len() {
            if page_max_positions[i].1 <= page_max_positions[i-1].1 {
                anyhow::bail!(
                    "Pages have overlapping position ranges: page {} max={} <= page {} max={}",
                    page_max_positions[i].0, page_max_positions[i].1,
                    page_max_positions[i-1].0, page_max_positions[i-1].1
                );
            }
        }

        // Collect values before dropping ctx
        let prev_hash = ctx.last_committed_hash;
        let indices_set: std::collections::HashSet<usize> = indices.iter().map(|&i| i as usize).collect();
        
        // Prepare lineage record
        let lineage_tokens = tokens_to_commit.clone();
        let lineage_positions = positions_to_commit.clone();
        let lineage_masks = masks_to_commit.clone();

        drop(ctx); // Release borrow for page_store access

        // Commit to page store
        let (final_page_ids, final_hash) = self.page_store.write().unwrap().commit(
            &pages_to_commit,
            &tokens_to_commit,
            &positions_to_commit,
            &masks_to_commit,
            prev_hash,
        )?;

        // Re-acquire context and update state
        let mut ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context lost during commit"))?;

        // Add committed pages
        ctx.pages_committed.extend(final_page_ids.iter().cloned());

        // Remove committed pages from uncommitted (in reverse order to preserve indices)
        let mut sorted_indices: Vec<usize> = indices_set.into_iter().collect();
        sorted_indices.sort_by(|a, b| b.cmp(a)); // Reverse sort
        for idx in sorted_indices {
            ctx.pages_uncommitted.remove(idx);
            // Remove corresponding tokens
            let token_start = idx * page_size;
            let token_end = (idx + 1) * page_size;
            let tokens_len = ctx.tokens_uncommitted.len();
            ctx.tokens_uncommitted.drain(token_start..token_end.min(tokens_len));
        }

        // Update max_committed_position
        let new_max = page_max_positions.iter().map(|(_, max)| *max).max();
        if let Some(max) = new_max {
            ctx.max_committed_position = Some(max);
        }

        // Update hash chain
        ctx.last_committed_hash = final_hash;

        // Update pointer
        ctx.pointer = ctx.pages_committed.len() * page_size;

        // Add lineage record
        ctx.lineage.push(Record::Fill {
            tokens: lineage_tokens,
            positions: lineage_positions,
            mask: lineage_masks,
            adapter: None,
        });

        Ok(())
    }
}

/// The context actor wraps ContextManager for async message handling.
#[derive(Debug)]
struct ContextManagerActor {
    service: ContextManager,
}

impl Handle for ContextManagerActor {
    type Message = Message;

    fn new() -> Self {
        let page_size = 64; // Default page size
        let page_store = Arc::new(RwLock::new(PageStore::new(page_size)));
        ContextManagerActor {
            service: ContextManager::new(page_store, page_size),
        }
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Create { user_id, name, fill, response } => {
                let result = self.service.create(user_id, name, fill);
                let _ = response.send(result);
            }
            Message::Destroy { id, lock_id, response } => {
                let result = self.service.destroy(id, lock_id);
                let _ = response.send(result);
            }
            Message::Lookup { user_id, name, response } => {
                let id = self.service.get(user_id, name);
                let _ = response.send(id);
            }
            Message::Fork { id, user_id, new_name, response } => {
                let result = self.service.fork(id, user_id, new_name);
                let _ = response.send(result);
            }
            Message::AcquireLock { id, response } => {
                let lock_id = self.service.lock(id);
                let _ = response.send(lock_id);
            }
            Message::ReleaseLock { id, lock_id } => {
                self.service.unlock(id, lock_id);
            }
            Message::TokensPerPage { id: _, response } => {
                let size = self.service.page_size();
                let _ = response.send(size);
            }
            Message::CommittedPageCount { id, response } => {
                let count = self.service.num_total_pages(id);
                let _ = response.send(count);
            }
            Message::CommitPages { id, lock_id, page_indices, response } => {
                let result = self.service.commit_pages(id, lock_id, page_indices);
                let _ = response.send(result);
            }
            Message::ReservePages { id, lock_id, num_pages, response } => {
                let result = self.service.allocate_pages(id, lock_id, num_pages);
                let _ = response.send(result);
            }
            Message::ReleasePages { id, lock_id, num_pages } => {
                let _ = self.service.free_pages(id, lock_id, num_pages);
            }
            Message::GetCursor { id, lock_id: _, response } => {
                let cursor = self.service.get_pointer(id);
                let _ = response.send(cursor);
            }
            Message::SetCursor { id, lock_id: _, cursor } => {
                let _ = self.service.set_pointer(id, cursor);
            }
            Message::GetBufferedTokens { id, lock_id: _, response } => {
                let tokens = self.service.get_buffered_tokens(id);
                let _ = response.send(tokens);
            }
            Message::SetBufferedTokens { id, lock_id: _, tokens } => {
                let _ = self.service.set_buffered_tokens(id, tokens);
            }
            Message::AppendBufferedTokens { id, lock_id: _, tokens } => {
                let _ = self.service.append_buffered_tokens(id, tokens);
            }
        }
    }
}
