//! # Context Module
//!
//! Manages named execution contexts with KV cache state for model inference.
//!
//! Each model gets a dedicated ContextManager actor that handles:
//! - Context creation, destruction, and forking
//! - Lock acquisition for exclusive access
//! - Page management (commit, reserve, release)
//! - Token buffering and cursor tracking
//!
//! Contexts are stored per-model via a ServiceArray, accessed by model index.
pub mod kvcache;

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::{LazyLock};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::oneshot;
use anyhow::Result;

use crate::service::{ServiceHandler, ServiceArray};
use crate::adapter::AdapterId;
use crate::brle::Brle;
use kvcache::{PageId, PageStore, PhysicalPageId, PageHash};
use crate::device::DeviceId;

// =============================================================================
// Public Types
// =============================================================================

/// Unique identifier for a context.
pub type ContextId = u64;
pub type LockId = u64;

// =============================================================================
// Globals
// =============================================================================

static SERVICES: LazyLock<ServiceArray<Message>> = LazyLock::new(ServiceArray::new);
static CONTEXTS: LazyLock<DashMap<(usize, ContextId), Context>> = LazyLock::new(DashMap::new);
static NEXT_LOCK_ID: AtomicU64 = AtomicU64::new(1);
static PAGE_SIZES: LazyLock<boxcar::Vec<usize>> = LazyLock::new(boxcar::Vec::new);

// =============================================================================
// Public API
// =============================================================================

/// Spawns a new context manager for a model.
pub fn spawn(page_size: usize, num_kv_pages: Vec<usize>) -> usize {
    let model_idx = PAGE_SIZES.push(page_size);
    let idx = SERVICES.spawn(move || ContextManager::new(model_idx, page_size, &num_kv_pages)).expect("Failed to spawn context manager");
    assert_eq!(idx, model_idx);
    idx
}

// ---------- Actor-routed (needs PageStore) ----------

/// Looks up a context by name.
pub async fn lookup(model_idx: usize, username: String, name: String) -> Option<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Lookup { username, name, response: tx }).ok()?;
    rx.await.ok().flatten()
}

/// Creates a new context with the given name.
pub async fn create(model_idx: usize, username: String, name: String, fill: Option<Vec<u32>>) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Create { username, name, fill, response: tx })?;
    rx.await?
}

/// Destroys a context.
pub async fn destroy(model_idx: usize, id: ContextId, lock_id: LockId) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Destroy { id, lock_id, response: tx })?;
    rx.await?
}

/// Forks a context into a new one.
pub async fn fork(model_idx: usize, id: ContextId, username: String, new_name: String) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Fork { id, username, new_name, response: tx })?;
    rx.await?
}

/// Commits pages to the context.
pub async fn commit_pages(model_idx: usize, id: ContextId, lock_id: LockId, page_indices: Vec<u32>) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::CommitPages { id, lock_id, page_indices, response: tx })?;
    rx.await?
}

/// Reserves pages for the context.
pub async fn reserve_pages(model_idx: usize, id: ContextId, lock_id: LockId, num_pages: u32) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::ReservePages { id, lock_id, num_pages, response: tx })?;
    rx.await?
}

/// Releases pages from the context.
pub fn release_pages(model_idx: usize, id: ContextId, lock_id: LockId, num_pages: u32) -> Result<()> {
    SERVICES.send(model_idx, Message::ReleasePages { id, lock_id, num_pages })
}

/// Gets physical page IDs grouped by device, plus the last page length.
pub async fn get_physical_page_ids(model_idx: usize, id: ContextId) -> Result<(HashMap<DeviceId, Vec<PhysicalPageId>>, u32)> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::GetPhysicalPageIds { id, response: tx })?;
    Ok(rx.await?)
}



// ---------- Direct (no actor, uses global CONTEXTS DashMap) ----------


/// Acquires a lock on the context.
pub fn acquire_lock(model_idx: usize, id: ContextId) -> LockId {
    let lock_id = NEXT_LOCK_ID.fetch_add(1, Ordering::Relaxed);
    if let Some(mut ctx) = CONTEXTS.get_mut(&(model_idx, id)) {
        if ctx.acquire_lock(lock_id) {
            return lock_id;
        }
    }
    0
}

/// Releases the lock on the context.
pub fn release_lock(model_idx: usize, id: ContextId, lock_id: LockId) -> Result<()> {
    if let Some(mut ctx) = CONTEXTS.get_mut(&(model_idx, id)) {
        ctx.release_lock(lock_id);
    }
    Ok(())
}

pub fn tokens_per_page(model_idx: usize, _id: ContextId) -> u32 {
    PAGE_SIZES.get(model_idx).map(|v| *v as u32).unwrap_or(0)
}

pub fn committed_page_count(model_idx: usize, id: ContextId) -> u32 {
    CONTEXTS.get(&(model_idx, id)).map(|ctx| ctx.committed_page_count()).unwrap_or(0)
}

pub fn get_cursor(model_idx: usize, id: ContextId) -> u32 {
    CONTEXTS.get(&(model_idx, id)).map(|ctx| ctx.cursor()).unwrap_or(0)
}

pub fn set_cursor(model_idx: usize, id: ContextId, lock_id: LockId, cursor: u32) -> Result<()> {
    let mut ctx = CONTEXTS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    ctx.set_cursor(lock_id, cursor)
}

pub fn last_position(model_idx: usize, id: ContextId) -> Option<u32> {
    CONTEXTS.get(&(model_idx, id)).and_then(|ctx| ctx.last_position())
}

pub fn get_buffered_tokens(model_idx: usize, id: ContextId) -> Vec<u32> {
    CONTEXTS.get(&(model_idx, id)).map(|ctx| ctx.buffered_tokens()).unwrap_or_default()
}

pub fn set_buffered_tokens(model_idx: usize, id: ContextId, lock_id: LockId, tokens: Vec<u32>) -> Result<()> {
    let mut ctx = CONTEXTS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    ctx.set_buffered_tokens(lock_id, tokens)
}

pub fn append_buffered_tokens(model_idx: usize, id: ContextId, lock_id: LockId, tokens: Vec<u32>) -> Result<()> {
    let mut ctx = CONTEXTS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    ctx.append_buffered_tokens(lock_id, tokens)
}

/// Marks the first `n` buffered tokens as forwarded, moving them to filled state.
/// Called by the inference service after a forward pass completes.
pub fn fill(
    model_idx: usize,
    id: ContextId,
    n: usize,
    positions: Vec<u32>,
    masks: Vec<Brle>,
    adapter: Option<AdapterId>,
) -> Result<()> {
    let mut ctx = CONTEXTS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    ctx.fill(n, positions, masks, adapter)
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
    /// Tokens with KV already computed (forwarded), awaiting commit.
    tokens_filled: Vec<TokenInfo>,
    /// Token IDs queued for the next forward pass, no KV yet.
    tokens_buffered: Vec<u32>,

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
            tokens_filled: Vec::new(),
            tokens_buffered: Vec::new(),
            mutex: None,
            max_committed_position: None,
            last_committed_hash: 0,
        }
    }

    /// Check if the given lock_id is valid for this context.
    /// Allows lock_id=0 when the context is unlocked (no mutex held).
    fn check_lock(&self, lock_id: LockId) -> Result<()> {
        match (self.mutex, lock_id) {
            (None, 0) => Ok(()),         // Unlocked context, no lock requested
            (Some(held), id) if held == id => Ok(()), // Lock matches
            (None, _) => anyhow::bail!("Context is not locked"),
            (Some(_), 0) => anyhow::bail!("Context is locked; lock_id required"),
            (Some(_), _) => anyhow::bail!("Invalid lock_id"),
        }
    }

    fn acquire_lock(&mut self, lock_id: LockId) -> bool {
        if self.mutex.is_none() {
            self.mutex = Some(lock_id);
            true
        } else {
            false
        }
    }

    fn release_lock(&mut self, lock_id: LockId) {
        if self.mutex == Some(lock_id) {
            self.mutex = None;
        }
    }

    fn committed_page_count(&self) -> u32 {
        self.pages_committed.len() as u32
    }

    /// Cursor is derived from the number of filled (forwarded) tokens.
    fn cursor(&self) -> u32 {
        self.tokens_filled.len() as u32
    }

    /// Validates range 0..=tokens_filled.len() and truncates tokens_filled to n.
    fn set_cursor(&mut self, lock_id: LockId, cursor: u32) -> Result<()> {
        self.check_lock(lock_id)?;
        let max = self.tokens_filled.len();
        if cursor as usize > max {
            anyhow::bail!("cursor {} out of range 0..={}", cursor, max);
        }
        self.tokens_filled.truncate(cursor as usize);
        Ok(())
    }

    fn last_position(&self) -> Option<u32> {
        let max_filled = self.tokens_filled.iter()
            .map(|t| t.position)
            .max();
        match (self.max_committed_position, max_filled) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    fn buffered_tokens(&self) -> Vec<u32> {
        self.tokens_buffered.clone()
    }

    fn set_buffered_tokens(&mut self, lock_id: LockId, tokens: Vec<u32>) -> Result<()> {
        self.check_lock(lock_id)?;
        self.tokens_buffered = tokens;
        Ok(())
    }

    fn append_buffered_tokens(&mut self, lock_id: LockId, tokens: Vec<u32>) -> Result<()> {
        self.check_lock(lock_id)?;
        self.tokens_buffered.extend(tokens);
        Ok(())
    }

    /// Marks the first `n` buffered tokens as forwarded, moving them to tokens_filled
    /// with the given positions, masks, and adapter.
    fn fill(
        &mut self,
        n: usize,
        positions: Vec<u32>,
        masks: Vec<Brle>,
        adapter: Option<AdapterId>,
    ) -> Result<()> {
        if n > self.tokens_buffered.len() {
            anyhow::bail!(
                "cannot mark {} tokens as forwarded, only {} buffered",
                n, self.tokens_buffered.len()
            );
        }
        if positions.len() != n {
            anyhow::bail!("positions length {} != n {}", positions.len(), n);
        }
        if !masks.is_empty() && masks.len() != n {
            anyhow::bail!("masks length {} != n {}", masks.len(), n);
        }

        let tokens: Vec<u32> = self.tokens_buffered.drain(..n).collect();
        for (i, token) in tokens.into_iter().enumerate() {
            self.tokens_filled.push(TokenInfo {
                token,
                position: positions[i],
                mask: if masks.is_empty() { Brle::new(0) } else { masks[i].clone() },
                adapter,
            });
        }
        Ok(())
    }
}

/// The context manager handles page-store operations.
/// Context metadata lives in global CONTEXTS DashMap.
#[derive(Debug)]
struct ContextManager {
    page_store: PageStore,
    page_size: usize,
    model_idx: usize,
    name_to_id: HashMap<(String, String), ContextId>,
    next_id: u64,
}

impl ContextManager {
    pub fn new(model_idx: usize, page_size: usize, num_kv_pages: &[usize]) -> Self {
        ContextManager {
            page_store: PageStore::new(page_size, num_kv_pages),
            page_size,
            model_idx,
            name_to_id: HashMap::new(),
            next_id: 1,
        }
    }

    fn next_id(&mut self) -> ContextId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // ==================== Core Operations ====================

    pub fn create(&mut self, username: String, name: String, fill: Option<Vec<u32>>) -> Result<ContextId> {
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

            let page_store = &mut self.page_store;
            
            // Lookup existing cached pages (longest prefix match)
            let matched_pages = page_store.lookup(&tokens);
            
            if let Some(pages) = matched_pages {
                let tokens_matched = pages.len() * page_store.page_size();
                page_store.acquire(&pages);
                
                // Place matched pages in pages_committed
                ctx.pages_committed = pages.clone();
                
                // Remaining tokens become buffered (need forward pass)
                if tokens_matched < tokens.len() {
                    let remaining_tokens = tokens.len() - tokens_matched;
                    let page_size = page_store.page_size();
                    let pages_needed = (remaining_tokens + page_size - 1) / page_size;
                    
                    // Allocate uncommitted pages
                    let affinity = pages.first().copied();
                    ctx.pages_uncommitted = page_store.allocate(pages_needed, affinity)?;
                    
                    ctx.tokens_buffered = tokens[tokens_matched..].to_vec();
                }
            } else {
                // No match: all tokens become buffered (need forward pass)
                let page_size = page_store.page_size();
                let pages_needed = (tokens.len() + page_size - 1) / page_size;
                
                // Allocate uncommitted pages (no affinity)
                if pages_needed > 0 {
                    ctx.pages_uncommitted = page_store.allocate(pages_needed, None)?;
                }
                
                ctx.tokens_buffered = tokens;
            }
        }

        CONTEXTS.insert((self.model_idx, id), ctx);
        self.name_to_id.insert((username, name), id);
        Ok(id)
    }

    pub fn destroy(&mut self, id: ContextId, lock_id: LockId) -> Result<()> {
        if let Some(ctx) = CONTEXTS.get(&(self.model_idx, id)) {
            ctx.check_lock(lock_id)?;
            drop(ctx);
            CONTEXTS.remove(&(self.model_idx, id));
            // Clean up name_to_id reverse mapping
            self.name_to_id.retain(|_, v| *v != id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Context not found"))
        }
    }

    pub fn fork(&mut self, id: ContextId, username: String, new_name: String) -> Result<ContextId> {
        // Check for name collision
        if self.name_to_id.contains_key(&(username.clone(), new_name.clone())) {
            anyhow::bail!("Context name already exists: {}", new_name);
        }

        // Get the source context
        let source_ctx = CONTEXTS.get(&(self.model_idx, id))
            .ok_or_else(|| anyhow::anyhow!("Source context not found"))?;

        // Validate filled token positions are sequential from max_committed_position
        if !source_ctx.tokens_filled.is_empty() {
            let base = source_ctx.max_committed_position.map(|p| p + 1).unwrap_or(0);
            for (i, info) in source_ctx.tokens_filled.iter().enumerate() {
                let expected = base + i as u32;
                if info.position != expected {
                    anyhow::bail!(
                        "Cannot fork: filled token {} has position {}, expected {}",
                        i, info.position, expected
                    );
                }
            }
        }

        // Collect info from source — only committed pages are cloned
        let pages_committed = source_ctx.pages_committed.clone();
        let lineage = source_ctx.lineage.clone();
        let affinity = pages_committed.first().copied();
        let source_max_position = source_ctx.max_committed_position;
        let source_last_hash = source_ctx.last_committed_hash;

        // Convert filled tokens back to buffered (prepend to existing buffered)
        let filled_as_buffered: Vec<u32> = source_ctx.tokens_filled.iter().map(|t| t.token).collect();
        let mut new_buffered = filled_as_buffered;
        new_buffered.extend_from_slice(&source_ctx.tokens_buffered);
        
        drop(source_ctx); // Release the borrow

        let page_store = &mut self.page_store;
        
        // Increase refcount for committed pages
        page_store.acquire(&pages_committed);

        // Preallocate pages to hold all buffered tokens
        let page_size = page_store.page_size();
        let pages_needed = (new_buffered.len() + page_size - 1) / page_size;
        let new_uncommitted = if pages_needed > 0 {
            page_store.allocate(pages_needed, affinity)?
        } else {
            Vec::new()
        };

        // Create the new context — tokens_filled is empty
        let new_id = self.next_id();
        let new_ctx = Context {
            lineage,
            pages_committed,
            pages_uncommitted: new_uncommitted,
            tokens_filled: Vec::new(),
            tokens_buffered: new_buffered,
            mutex: None,
            max_committed_position: source_max_position,
            last_committed_hash: source_last_hash,
        };

        CONTEXTS.insert((self.model_idx, new_id), new_ctx);
        self.name_to_id.insert((username, new_name), new_id);

        Ok(new_id)
    }

    // ==================== Page Management ====================

    pub fn allocate_pages(&mut self, id: ContextId, lock_id: LockId, num_pages: u32) -> Result<()> {
        let ctx = CONTEXTS.get(&(self.model_idx, id))
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        
        ctx.check_lock(lock_id)?;
        
        // Only allocate the deficit: pages needed minus pages already uncommitted
        let existing_uncommitted = ctx.pages_uncommitted.len() as u32;
        let pages_to_allocate = num_pages.saturating_sub(existing_uncommitted);
        
        if pages_to_allocate == 0 {
            return Ok(()); // Already have enough uncommitted pages
        }
        
        // Get affinity from existing pages (committed first, then uncommitted)
        let affinity = ctx.pages_committed.first()
            .or_else(|| ctx.pages_uncommitted.first())
            .copied();
        
        drop(ctx);
        
        // Allocate pages from page store
        let new_pages = self.page_store.allocate(pages_to_allocate as usize, affinity)?;
        
        // Add to uncommitted pages
        let mut ctx = CONTEXTS.get_mut(&(self.model_idx, id))
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        ctx.pages_uncommitted.extend(new_pages);
        
        Ok(())
    }

    pub fn free_pages(&mut self, id: ContextId, lock_id: LockId, num_pages: u32) -> Result<()> {
        let mut ctx = CONTEXTS.get_mut(&(self.model_idx, id))
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        
        ctx.check_lock(lock_id)?;
        
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
        // Remove corresponding filled tokens from the end
        let tokens_to_remove = num_to_free * self.page_size;
        let tokens_len = ctx.tokens_filled.len();
        if tokens_to_remove > 0 && tokens_len > 0 {
            let new_tokens_len = tokens_len.saturating_sub(tokens_to_remove);
            ctx.tokens_filled.truncate(new_tokens_len);
        }
        
        drop(ctx);
        
        // Release pages back to page store
        self.page_store.release(&pages_to_free);
        
        Ok(())
    }

    pub fn commit_pages(&mut self, id: ContextId, lock_id: LockId, indices: Vec<u32>) -> Result<()> {
        let page_size = self.page_size;
        
        let mut ctx = CONTEXTS.get_mut(&(self.model_idx, id))
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        
        ctx.check_lock(lock_id)?;

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
            if token_end > ctx.tokens_filled.len() {
                anyhow::bail!(
                    "Page {} not fully filled: need {} tokens but only have {}",
                    idx, token_end, ctx.tokens_filled.len()
                );
            }

            // Collect token info from filled tokens
            for i in token_start..token_end {
                let info = &ctx.tokens_filled[i];
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
        let mut ctx = CONTEXTS.get_mut(&(self.model_idx, id))
            .ok_or_else(|| anyhow::anyhow!("Context lost during commit"))?;

        // Add committed pages
        ctx.pages_committed.extend(final_page_ids.iter().cloned());

        // Remove committed pages from uncommitted (in reverse order to preserve indices)
        let mut sorted_indices: Vec<usize> = indices_set.into_iter().collect();
        sorted_indices.sort_by(|a, b| b.cmp(a)); // Reverse sort
        for idx in sorted_indices {
            ctx.pages_uncommitted.remove(idx);
            // Remove corresponding filled tokens
            let token_start = idx * page_size;
            let token_end = (idx + 1) * page_size;
            let tokens_len = ctx.tokens_filled.len();
            ctx.tokens_filled.drain(token_start..token_end.min(tokens_len));
        }

        // Update max_committed_position
        let new_max = page_max_positions.iter().map(|(_, max)| *max).max();
        if let Some(max) = new_max {
            ctx.max_committed_position = Some(max);
        }

        // Update hash chain
        ctx.last_committed_hash = final_hash;

        // Add lineage record
        ctx.lineage.push(Record::Fill {
            tokens: lineage_tokens,
            positions: lineage_positions,
            mask: lineage_masks,
            adapter: None,
        });

        Ok(())
    }

    /// Returns physical page IDs and the number of tokens already written to the KV cache.
    ///
    /// The second element `kv_len` = committed_pages * page_size + tokens_filled.len().
    /// This does NOT include input tokens for the current forward pass.
    /// The caller must add input_tokens to compute FlashInfer's `last_page_len`.
    pub fn get_physical_page_ids(&self, id: ContextId) -> (HashMap<DeviceId, Vec<PhysicalPageId>>, u32) {
        let mut result: HashMap<DeviceId, Vec<PhysicalPageId>> = HashMap::new();
        
        if let Some(ctx) = CONTEXTS.get(&(self.model_idx, id)) {
            let page_store = &self.page_store;
            
            // Get all committed pages and collect their physical mappings
            for &page_id in &ctx.pages_committed {
                for (device_id, phys_id) in page_store.get_physical_mappings(page_id) {
                    result.entry(device_id as DeviceId).or_default().push(phys_id);
                }
            }
            // Also include uncommitted pages
            for &page_id in &ctx.pages_uncommitted {
                for (device_id, phys_id) in page_store.get_physical_mappings(page_id) {
                    result.entry(device_id as DeviceId).or_default().push(phys_id);
                }
            }

            // kv_len = total tokens already written to KV cache
            let kv_len = (ctx.pages_committed.len() * self.page_size + ctx.tokens_filled.len()) as u32;

            (result, kv_len)
        } else {
            (result, 0)
        }
    }
}

// =============================================================================
// ServiceHandler Implementation
// =============================================================================

/// Messages handled by ContextManager.
#[derive(Debug)]
enum Message {
    Lookup { username: String, name: String, response: oneshot::Sender<Option<ContextId>> },
    Create { username: String, name: String, fill: Option<Vec<u32>>, response: oneshot::Sender<Result<ContextId>> },
    Destroy { id: ContextId, lock_id: LockId, response: oneshot::Sender<Result<()>> },
    Fork { id: ContextId, username: String, new_name: String, response: oneshot::Sender<Result<ContextId>> },
    CommitPages { id: ContextId, lock_id: LockId, page_indices: Vec<u32>, response: oneshot::Sender<Result<()>> },
    ReservePages { id: ContextId, lock_id: LockId, num_pages: u32, response: oneshot::Sender<Result<()>> },
    ReleasePages { id: ContextId, lock_id: LockId, num_pages: u32 },
    GetPhysicalPageIds { id: ContextId, response: oneshot::Sender<(HashMap<DeviceId, Vec<PhysicalPageId>>, u32)> },
}


impl ServiceHandler for ContextManager {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Lookup { username, name, response } => {
                let _ = response.send(self.name_to_id.get(&(username, name)).copied());
            }
            Message::Create { username, name, fill, response } => {
                let _ = response.send(self.create(username, name, fill));
            }
            Message::Destroy { id, lock_id, response } => {
                let _ = response.send(self.destroy(id, lock_id));
            }
            Message::Fork { id, username, new_name, response } => {
                let _ = response.send(self.fork(id, username, new_name));
            }
            Message::CommitPages { id, lock_id, page_indices, response } => {
                let _ = response.send(self.commit_pages(id, lock_id, page_indices));
            }
            Message::ReservePages { id, lock_id, num_pages, response } => {
                let _ = response.send(self.allocate_pages(id, lock_id, num_pages));
            }
            Message::ReleasePages { id, lock_id, num_pages } => {
                let _ = self.free_pages(id, lock_id, num_pages);
            }
            Message::GetPhysicalPageIds { id, response } => {
                let _ = response.send(self.get_physical_page_ids(id));
            }
        }
    }
}
