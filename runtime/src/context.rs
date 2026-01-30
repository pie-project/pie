//! Context Service - Execution context management with KV cache state
//!
//! This module provides a model-specific actor for managing named execution
//! contexts with support for forking, joining, locking, and capacity management.

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot;
use anyhow::Result;

use crate::actor::{Handle, Actors, SendError};
use crate::brle::Brle;
use crate::kvcache::{PageId, PageStore, NodeId, PhysicalPageId, PageHash};

/// Unique identifier for a context.
pub type ContextId = u64;
pub type LockId = u64;

/// Global table of context actors.
static ACTOR: LazyLock<Actors<Message>> = LazyLock::new(Actors::new);

/// Spawns a new context actor.
pub(crate) fn spawn() -> usize {
    ACTOR.spawn::<ContextActor>()
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
    Get {
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
    Lock {
        id: ContextId,
        response: oneshot::Sender<LockId>,
    },

    /// Releases the lock on the context.
    Unlock {
        id: ContextId,
        lock_id: LockId,
    },

    /// Get page size
    PageSize {
        id: ContextId,
        response: oneshot::Sender<u32>,
    },

    /// Get total number of committed pages
    NumTotalPages {
        id: ContextId,
        response: oneshot::Sender<u32>,
    },

    /// Get total number of tokens
    NumTotalTokens {
        id: ContextId,
        lock_id: LockId,
        response: oneshot::Sender<u32>,
    },

    /// Commit pages to the context
    CommitPages {
        id: ContextId,
        lock_id: LockId,
        indices: Vec<u32>,
        response: oneshot::Sender<Result<()>>,
    },

    /// Allocate pages for the context
    AllocatePages {
        id: ContextId,
        lock_id: LockId,
        num_pages: u32,
        response: oneshot::Sender<Result<()>>,
    },

    /// Free pages from the context
    FreePages {
        id: ContextId,
        lock_id: LockId,
        num_pages: u32,
        response: oneshot::Sender<Result<()>>,
    },


    GetPhysicalPageIds {
        id: ContextId,
        lock_id: LockId,
        response: oneshot::Sender<HashMap<NodeId, Vec<PhysicalPageId>>>,
    },

    /// Get the pointer position
    GetPointer {
        id: ContextId,
        lock_id: LockId,
        response: oneshot::Sender<u32>,
    },

    /// Set the pointer position
    SetPointer {
        id: ContextId,
        lock_id: LockId,
        pointer: u32,
        response: oneshot::Sender<Result<()>>,
    },

    /// Get uncommitted tokens
    GetUncommittedTokens {
        id: ContextId,
        lock_id: LockId,
        response: oneshot::Sender<Vec<TokenInfo>>,
    },

    /// Set uncommitted tokens
    SetUncommittedTokens {
        id: ContextId,
        lock_id: LockId,
        tokens: Vec<TokenInfo>,
        response: oneshot::Sender<Result<()>>,
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
    },
}

#[derive(Debug, Clone)]
pub struct TokenInfo {
    pub token: u32,
    pub position: u32,
    pub mask: Brle,
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
pub struct ContextService {
    pub page_store: PageStore,
    contexts: DashMap<ContextId, Context>,
    name_to_id: DashMap<(u32, String), ContextId>,
    next_id: AtomicU64,
    pub page_size: usize,
}

impl ContextService {
    pub fn new(page_size: usize) -> Self {
        ContextService {
            page_store: PageStore::new(page_size),
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
            });

            // Lookup existing cached pages (longest prefix match)
            let matched_pages = self.page_store.lookup(&tokens);
            
            if let Some(pages) = matched_pages {
                // Calculate how many tokens are covered by matched pages
                let tokens_matched = pages.len() * self.page_store.page_size();
                
                // Acquire refcount for matched pages
                self.page_store.acquire(&pages);
                
                // Place matched pages in pages_committed
                ctx.pages_committed = pages.clone();
                
                // Set pointer to end of committed tokens
                ctx.pointer = tokens_matched;
                
                // Remaining tokens become tokens_uncommitted
                if tokens_matched < tokens.len() {
                    let remaining_tokens = tokens.len() - tokens_matched;
                    let page_size = self.page_store.page_size();
                    let pages_needed = (remaining_tokens + page_size - 1) / page_size;
                    
                    // Allocate uncommitted pages
                    let affinity = pages.first().copied();
                    ctx.pages_uncommitted = self.page_store.allocate(pages_needed, affinity)?;
                    
                    ctx.tokens_uncommitted = tokens[tokens_matched..]
                        .iter()
                        .enumerate()
                        .map(|(i, &token)| TokenInfo { 
                            token, 
                            position: (tokens_matched + i) as u32, 
                            mask: Brle::new(0) 
                        })
                        .collect();
                }
            } else {
                // No match: all tokens become tokens_uncommitted
                let page_size = self.page_store.page_size();
                let pages_needed = (tokens.len() + page_size - 1) / page_size;
                
                // Allocate uncommitted pages (no affinity)
                if pages_needed > 0 {
                    ctx.pages_uncommitted = self.page_store.allocate(pages_needed, None)?;
                }
                
                ctx.tokens_uncommitted = tokens
                    .into_iter()
                    .enumerate()
                    .map(|(i, token)| TokenInfo { 
                        token, 
                        position: i as u32, 
                        mask: Brle::new(0) 
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

        // Increase refcount for committed pages
        self.page_store.acquire(&pages_committed);

        // Allocate new uncommitted pages
        let new_uncommitted = if num_uncommitted > 0 {
            self.page_store.allocate(num_uncommitted, affinity)?
        } else {
            Vec::new()
        };

        // Calculate pointer
        let pointer = pages_committed.len() * self.page_store.page_size();

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

    pub fn get_uncommitted_tokens(&self, id: ContextId) -> Vec<TokenInfo> {
        self.contexts.get(&id)
            .map(|ctx| ctx.tokens_uncommitted.clone())
            .unwrap_or_default()
    }

    pub fn set_uncommitted_tokens(&self, id: ContextId, tokens: Vec<TokenInfo>) -> Result<()> {
        if let Some(mut ctx) = self.contexts.get_mut(&id) {
            ctx.tokens_uncommitted = tokens;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Context not found"))
        }
    }

    pub fn get_physical_page_ids(&self, _id: ContextId) -> HashMap<NodeId, Vec<PhysicalPageId>> {
        // TODO: Implement physical page ID retrieval from PageStore
        HashMap::new()
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
        let new_pages = self.page_store.allocate(num_pages as usize, affinity)?;
        
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
        self.page_store.release(&pages_to_free);
        
        Ok(())
    }

    pub fn commit_pages(&mut self, id: ContextId, lock_id: LockId, indices: Vec<u32>) -> Result<()> {
        let page_size = self.page_store.page_size();
        
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
        let (final_page_ids, final_hash) = self.page_store.commit(
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
        });

        Ok(())
    }
}

/// The context actor wraps ContextService for async message handling.
#[derive(Debug)]
struct ContextActor {
    manager: ContextService,
}

impl Handle for ContextActor {
    type Message = Message;

    fn new() -> Self {
        ContextActor {
            manager: ContextService::new(64), // Default page size
        }
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Create { user_id, name, fill, response } => {
                let result = self.manager.create(user_id, name, fill);
                let _ = response.send(result);
            }
            Message::Destroy { id, lock_id, response } => {
                let result = self.manager.destroy(id, lock_id);
                let _ = response.send(result);
            }
            Message::Get { user_id, name, response } => {
                let id = self.manager.get(user_id, name);
                let _ = response.send(id);
            }
            Message::Fork { id, user_id, new_name, response } => {
                let result = self.manager.fork(id, user_id, new_name);
                let _ = response.send(result);
            }
            Message::Lock { id, response } => {
                let lock_id = self.manager.lock(id);
                let _ = response.send(lock_id);
            }
            Message::Unlock { id, lock_id } => {
                self.manager.unlock(id, lock_id);
            }
            Message::PageSize { id: _, response } => {
                let size = self.manager.page_size();
                let _ = response.send(size);
            }
            Message::NumTotalPages { id, response } => {
                let count = self.manager.num_total_pages(id);
                let _ = response.send(count);
            }
            Message::NumTotalTokens { id, lock_id: _, response } => {
                let count = self.manager.num_total_tokens(id);
                let _ = response.send(count);
            }
            Message::CommitPages { id, lock_id, indices, response } => {
                let result = self.manager.commit_pages(id, lock_id, indices);
                let _ = response.send(result);
            }
            Message::AllocatePages { id, lock_id, num_pages, response } => {
                let result = self.manager.allocate_pages(id, lock_id, num_pages);
                let _ = response.send(result);
            }
            Message::FreePages { id, lock_id, num_pages, response } => {
                let result = self.manager.free_pages(id, lock_id, num_pages);
                let _ = response.send(result);
            }
            Message::GetPointer { id, lock_id: _, response } => {
                let pointer = self.manager.get_pointer(id);
                let _ = response.send(pointer);
            }
            Message::SetPointer { id, lock_id: _, pointer, response } => {
                let result = self.manager.set_pointer(id, pointer);
                let _ = response.send(result);
            }
            Message::GetUncommittedTokens { id, lock_id: _, response } => {
                let tokens = self.manager.get_uncommitted_tokens(id);
                let _ = response.send(tokens);
            }
            Message::SetUncommittedTokens { id, lock_id: _, tokens, response } => {
                let result = self.manager.set_uncommitted_tokens(id, tokens);
                let _ = response.send(result);
            }
            Message::GetPhysicalPageIds { id, lock_id: _, response } => {
                let page_ids = self.manager.get_physical_page_ids(id);
                let _ = response.send(page_ids);
            }
        }
    }
}
