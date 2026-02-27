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
mod arbiter;
mod manager;
mod waitqueue;
mod residency;

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::Instant;
use tokio::sync::oneshot;
use anyhow::Result;
use serde::Serialize;

use crate::service::{ServiceArray, ServiceHandler};
use crate::adapter::AdapterId;
use crate::inference::brle::Brle;
use crate::process::ProcessId;

use kvcache::{PhysicalPageId, PageHash};
use crate::device::DeviceId;
use manager::ContextManager;
use waitqueue::{PageWaiter, WaitNeeded};



// =============================================================================
// Public Types
// =============================================================================

pub type ContextId = u64;

// =============================================================================
// Globals
// =============================================================================

static SERVICES: LazyLock<ServiceArray<Message>> = LazyLock::new(ServiceArray::new);
static CONTEXTS: LazyLock<DashMap<(usize, ContextId), Context>> = LazyLock::new(DashMap::new);
static PAGE_SIZES: LazyLock<boxcar::Vec<usize>> = LazyLock::new(boxcar::Vec::new);

// =============================================================================
// Public API
// =============================================================================

/// Spawns a new context manager for a model.
pub fn spawn(page_size: usize, num_gpu_pages: Vec<usize>, num_cpu_pages: Vec<usize>) -> usize {
    PAGE_SIZES.push(page_size);
    SERVICES.spawn(move || manager::ContextManager::new(
        SERVICES.len().saturating_sub(1), page_size, &num_gpu_pages, &num_cpu_pages,
    )).expect("Failed to spawn context manager")
}

// ---------- Actor-routed (needs DevicePageCache) ----------

pub async fn open(model_idx: usize, username: String, name: String) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Open { username, name, response: tx })?;
    rx.await?
}

pub async fn create(model_idx: usize) -> Result<ContextId> {
    create_owned(model_idx, None).await
}

pub async fn create_owned(model_idx: usize, owner: Option<ProcessId>) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Create { owner, response: tx })?;
    rx.await?
}

pub async fn save(model_idx: usize, id: ContextId, username: String, name: String) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Save { id, username, name, response: tx })?;
    rx.await?
}

pub async fn snapshot(model_idx: usize, id: ContextId, username: String) -> Result<String> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Snapshot { id, username, response: tx })?;
    rx.await?
}

pub async fn delete(model_idx: usize, username: String, name: String) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Delete { username, name, response: tx })?;
    rx.await?
}

pub async fn destroy(model_idx: usize, id: ContextId, force: bool) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Destroy { id, force, response: tx })?;
    rx.await?
}

pub async fn fork(model_idx: usize, id: ContextId) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Fork { id, response: tx })?;
    rx.await?
}

pub async fn commit_pages(model_idx: usize, id: ContextId, page_indices: Vec<u32>) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::CommitPages { id, page_indices, response: tx })?;
    rx.await?
}

pub async fn reserve_pages(model_idx: usize, id: ContextId, num_pages: u32) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::ReservePages { id, num_pages, response: tx })?;
    rx.await?
}

pub fn release_pages(model_idx: usize, id: ContextId, num_pages: u32) -> Result<()> {
    SERVICES.send(model_idx, Message::ReleasePages { id, num_pages })
}

pub async fn get_physical_page_ids(model_idx: usize, id: ContextId) -> Result<HashMap<DeviceId, Vec<PhysicalPageId>>> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::GetPhysicalPageIds { id, response: tx })?;
    rx.await?
}

pub async fn ensure_resident(model_idx: usize, id: ContextId) -> Result<Option<Vec<ReplayFill>>> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::EnsureResident { id, response: tx })?;
    rx.await?
}

pub async fn commit_replay_chunk(model_idx: usize, id: ContextId, num_pages: u32) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::CommitReplayFill { id, num_pages, response: tx })?;
    rx.await?
}

pub fn finish_restore(model_idx: usize, id: ContextId) -> Result<()> {
    SERVICES.send(model_idx, Message::FinishRestore { id })
}

pub async fn get_stats(model_idx: usize) -> Vec<(usize, usize)> {
    let (tx, rx) = oneshot::channel();
    let _ = SERVICES.send(model_idx, Message::GetStats { response: tx });
    rx.await.unwrap_or_default()
}

// ---------- Arbiter policy (broadcast to all models) ----------

/// Set DAG-derived weights for processes in all model arbiters.
/// Called by the workflow actor when DAG topology changes.
pub fn set_dag_weights(
    weight: f64,
    pid_values: HashMap<ProcessId, f64>,
) {
    for model_idx in 0..SERVICES.len() {
        let _ = SERVICES.send(model_idx, Message::SetDagWeights {
            weight,
            pid_values: pid_values.clone(),
        });
    }
}

// ---------- Direct (no actor, uses global CONTEXTS DashMap) ----------

pub fn tokens_per_page(model_idx: usize, _id: ContextId) -> u32 {
    PAGE_SIZES.get(model_idx).map(|v| *v as u32).unwrap_or(0)
}

pub fn committed_page_count(model_idx: usize, id: ContextId) -> u32 {
    CONTEXTS.get(&(model_idx, id)).map(|ctx| ctx.committed_len as u32).unwrap_or(0)
}

pub fn kv_len(model_idx: usize, id: ContextId) -> u32 {
    let page_size = PAGE_SIZES.get(model_idx).copied().unwrap_or(0);
    CONTEXTS.get(&(model_idx, id))
        .map(|ctx| (ctx.committed_len * page_size + ctx.tokens_filled.len()) as u32)
        .unwrap_or(0)
}

pub fn get_cursor(model_idx: usize, id: ContextId) -> u32 {
    CONTEXTS.get(&(model_idx, id)).map(|ctx| ctx.cursor()).unwrap_or(0)
}

pub fn is_active(model_idx: usize, id: ContextId) -> bool {
    CONTEXTS.get(&(model_idx, id)).map(|ctx| ctx.state == ContextState::Active).unwrap_or(false)
}

pub fn set_cursor(model_idx: usize, id: ContextId, cursor: u32) -> Result<()> {
    let mut ctx = CONTEXTS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    ctx.set_cursor(cursor)
}

pub fn last_position(model_idx: usize, id: ContextId) -> Option<u32> {
    CONTEXTS.get(&(model_idx, id)).and_then(|ctx| ctx.last_position())
}

pub fn get_buffered_tokens(model_idx: usize, id: ContextId) -> Vec<u32> {
    CONTEXTS.get(&(model_idx, id)).map(|ctx| ctx.tokens_buffered.clone()).unwrap_or_default()
}

pub fn set_buffered_tokens(model_idx: usize, id: ContextId, tokens: Vec<u32>) -> Result<()> {
    let mut ctx = CONTEXTS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    ctx.tokens_buffered = tokens;
    Ok(())
}

pub fn append_buffered_tokens(model_idx: usize, id: ContextId, tokens: Vec<u32>) -> Result<()> {
    let mut ctx = CONTEXTS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    ctx.tokens_buffered.extend(tokens);
    Ok(())
}

pub fn fill(
    model_idx: usize, id: ContextId, n: usize,
    positions: Vec<u32>, masks: Vec<Brle>, adapter: Option<AdapterId>,
) -> Result<()> {
    let mut ctx = CONTEXTS.get_mut(&(model_idx, id))
        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
    ctx.fill(n, positions, masks, adapter)
}

// =============================================================================
// Internal Types
// =============================================================================

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
pub struct ReplayFill {
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub masks: Vec<Brle>,
    pub adapter: Option<AdapterId>,
    pub physical_page_ids: Vec<PhysicalPageId>,
    pub device_id: DeviceId,
    pub kv_len: u32,
    pub last_page_len: u32,
    pub num_pages: u32,
}

#[derive(Debug, Clone)]
pub struct TokenInfo {
    pub token: u32,
    pub position: u32,
    pub mask: Brle,
    pub adapter: Option<AdapterId>,
}

// =============================================================================
// Context
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextState {
    /// Active on GPU, ready for inference.
    Active,
    /// Committed chain refcounts released, working pages on CPU.
    Suspended,
    /// Being restored — replay in progress.
    Restoring,
}

#[derive(Debug, Clone)]
struct Context {
    /// Process that owns this context (None for named snapshots).
    owner: Option<ProcessId>,
    /// Device this context is on (None if fully evicted).
    device: Option<DeviceId>,
    /// Physical page IDs for uncommitted (working) pages on GPU.
    working_pages: Vec<PhysicalPageId>,
    /// CPU slots for working pages when suspended.
    working_cpu_slots: Vec<PhysicalPageId>,
    /// Tip of the committed hash chain (None if no commits yet).
    committed_tip: Option<PageHash>,
    /// Number of committed pages.
    committed_len: usize,

    // Token state
    tokens_filled: Vec<TokenInfo>,
    tokens_buffered: Vec<u32>,
    lineage: Vec<Record>,

    // Scheduling
    max_committed_position: Option<u32>,
    state: ContextState,
    last_access: Instant,
}

impl Context {
    fn new(owner: Option<ProcessId>) -> Self {
        Context {
            owner,
            device: None,
            working_pages: Vec::new(),
            working_cpu_slots: Vec::new(),
            committed_tip: None,
            committed_len: 0,
            tokens_filled: Vec::new(),
            tokens_buffered: Vec::new(),
            lineage: Vec::new(),
            max_committed_position: None,
            state: ContextState::Active,
            last_access: Instant::now(),
        }
    }

    fn cursor(&self) -> u32 { self.tokens_filled.len() as u32 }

    fn set_cursor(&mut self, cursor: u32) -> Result<()> {
        let max = self.tokens_filled.len();
        if cursor as usize > max { anyhow::bail!("cursor {} out of range 0..={}", cursor, max); }
        self.tokens_filled.truncate(cursor as usize);
        Ok(())
    }

    fn last_position(&self) -> Option<u32> {
        let max_filled = self.tokens_filled.iter().map(|t| t.position).max();
        match (self.max_committed_position, max_filled) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        }
    }

    fn num_uncommitted(&self) -> usize { self.working_pages.len() }

    fn has_gpu_pages(&self) -> bool {
        self.state == ContextState::Active && (!self.working_pages.is_empty() || self.committed_tip.is_some())
    }

    fn fill(&mut self, n: usize, positions: Vec<u32>, masks: Vec<Brle>, adapter: Option<AdapterId>) -> Result<()> {
        if n > self.tokens_buffered.len() {
            anyhow::bail!("fill: n ({}) > tokens_buffered ({})", n, self.tokens_buffered.len());
        }
        if positions.len() != n { anyhow::bail!("positions length {} != n {}", positions.len(), n); }
        if !masks.is_empty() && masks.len() != n { anyhow::bail!("masks length {} != n {}", masks.len(), n); }

        let tokens: Vec<u32> = self.tokens_buffered.drain(..n).collect();
        for (i, token) in tokens.into_iter().enumerate() {
            self.tokens_filled.push(TokenInfo {
                token, position: positions[i],
                mask: if masks.is_empty() { Brle::new(0) } else { masks[i].clone() },
                adapter,
            });
        }
        Ok(())
    }
}

// =============================================================================
// Message & ServiceHandler
// =============================================================================

#[derive(Debug)]
pub(crate) enum Message {
    Open { username: String, name: String, response: oneshot::Sender<Result<ContextId>> },
    Create { owner: Option<ProcessId>, response: oneshot::Sender<Result<ContextId>> },
    Save { id: ContextId, username: String, name: String, response: oneshot::Sender<Result<()>> },
    Snapshot { id: ContextId, username: String, response: oneshot::Sender<Result<String>> },
    Delete { username: String, name: String, response: oneshot::Sender<Result<()>> },
    Destroy { id: ContextId, force: bool, response: oneshot::Sender<Result<()>> },
    Fork { id: ContextId, response: oneshot::Sender<Result<ContextId>> },
    CommitPages { id: ContextId, page_indices: Vec<u32>, response: oneshot::Sender<Result<()>> },
    ReservePages { id: ContextId, num_pages: u32, response: oneshot::Sender<Result<()>> },
    ReleasePages { id: ContextId, num_pages: u32 },
    GetPhysicalPageIds { id: ContextId, response: oneshot::Sender<Result<HashMap<DeviceId, Vec<PhysicalPageId>>>> },
    EnsureResident { id: ContextId, response: oneshot::Sender<Result<Option<Vec<ReplayFill>>>> },
    CommitReplayFill { id: ContextId, num_pages: u32, response: oneshot::Sender<Result<()>> },
    FinishRestore { id: ContextId },
    GetStats { response: oneshot::Sender<Vec<(usize, usize)>> },
    SetDagWeights { weight: f64, pid_values: HashMap<ProcessId, f64> },
}

impl ServiceHandler for ContextManager {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Open { username, name, response } => {
                let result = match self.name_to_id.get(&(username, name)) {
                    Some(&snapshot_id) => self.fork(snapshot_id),
                    None => Err(anyhow::anyhow!("Snapshot not found")),
                };
                let _ = response.send(result);
            }
            Message::Create { owner, response } => { let _ = response.send(self.create(owner)); }
            Message::Save { id, username, name, response } => {
                let _ = response.send(self.save(id, username, name));
            }
            Message::Snapshot { id, username, response } => {
                let _ = response.send(self.snapshot(id, username));
            }
            Message::Delete { username, name, response } => {
                let _ = response.send(self.delete(username, name));
            }
            Message::Destroy { id, force, response } => {
                let dev = CONTEXTS.get(&(self.model_idx, id))
                    .map(|c| c.device.unwrap_or(0) as usize);
                let _ = response.send(self.destroy(id, force));
                if let Some(d) = dev {
                    self.try_serve_waiters(Some(d)).await;
                }
            }
            Message::Fork { id, response } => { let _ = response.send(self.fork(id)); }
            Message::CommitPages { id, page_indices, response } => {
                let dev = CONTEXTS.get(&(self.model_idx, id))
                    .map(|c| c.device.unwrap_or(0) as usize);
                let _ = response.send(self.commit_pages(id, page_indices));
                if let Some(d) = dev {
                    self.try_serve_waiters(Some(d)).await;
                }
            }
            Message::ReservePages { id, num_pages, response } => {
                let ctx = CONTEXTS.get(&(self.model_idx, id));
                let (dev, owner) = ctx.as_ref()
                    .map(|c| (c.device.unwrap_or(0), c.owner))
                    .unwrap_or((0, None));
                drop(ctx);
                let dev_idx = dev as usize;
                let floor = self.requester_floor(owner, dev_idx, num_pages as usize);

                let should_queue = self.wait_queues[dev_idx].peek()
                    .is_some_and(|top| floor < top.effective_floor());

                if should_queue {
                    tracing::info!("Enqueuing ReservePages waiter for ctx {id} on dev {dev_idx} behind queue (floor={floor:.1})");
                    self.wait_queues[dev_idx].push(PageWaiter::Allocate {
                        context_id: id, device: dev,
                        num_pages: num_pages as usize, requester: owner,
                        priority_floor: floor, enqueued_at: Instant::now(), response,
                    });
                } else {
                    match self.reserve_pages(id, num_pages).await {
                        Ok(()) => { let _ = response.send(Ok(())); }
                        Err(WaitNeeded::NeedPages) => {
                            tracing::info!("Enqueuing ReservePages waiter for ctx {id} on dev {dev_idx} (floor={floor:.1})");
                            self.wait_queues[dev_idx].push(PageWaiter::Allocate {
                                context_id: id, device: dev,
                                num_pages: num_pages as usize, requester: owner,
                                priority_floor: floor, enqueued_at: Instant::now(), response,
                            });
                        }
                        Err(WaitNeeded::Fatal(e)) => { let _ = response.send(Err(e)); }
                    }
                }
            }
            Message::ReleasePages { id, num_pages } => {
                let dev = CONTEXTS.get(&(self.model_idx, id))
                    .map(|c| c.device.unwrap_or(0) as usize);
                let _ = self.free_pages(id, num_pages);
                if let Some(d) = dev {
                    self.try_serve_waiters(Some(d)).await;
                }
            }
            Message::GetPhysicalPageIds { id, response } => {
                let _ = response.send(self.get_physical_page_ids(id));
            }
            Message::EnsureResident { id, response } => {
                let ctx = CONTEXTS.get(&(self.model_idx, id));
                let (dev, owner) = ctx.as_ref()
                    .map(|c| (c.device.unwrap_or(0), c.owner))
                    .unwrap_or((0, None));
                drop(ctx);
                let dev_idx = dev as usize;
                let floor = self.requester_floor(owner, dev_idx, 1);

                let should_queue = self.wait_queues[dev_idx].peek()
                    .is_some_and(|top| floor < top.effective_floor());

                if should_queue {
                    tracing::info!("Enqueuing EnsureResident waiter for ctx {id} on dev {dev_idx} behind queue (floor={floor:.1})");
                    self.wait_queues[dev_idx].push(PageWaiter::Restore {
                        context_id: id, device: dev,
                        requester: owner, priority_floor: floor,
                        enqueued_at: Instant::now(), response,
                    });
                } else {
                    match self.ensure_resident(id).await {
                        Ok(result) => { let _ = response.send(Ok(result)); }
                        Err(WaitNeeded::NeedPages) => {
                            tracing::info!("Enqueuing EnsureResident waiter for ctx {id} on dev {dev_idx} (floor={floor:.1})");
                            self.wait_queues[dev_idx].push(PageWaiter::Restore {
                                context_id: id, device: dev,
                                requester: owner, priority_floor: floor,
                                enqueued_at: Instant::now(), response,
                            });
                        }
                        Err(WaitNeeded::Fatal(e)) => { let _ = response.send(Err(e)); }
                    }
                }
            }
            Message::CommitReplayFill { id, num_pages, response } => {
                let _ = response.send(self.commit_replay_chunk(id, num_pages));
            }
            Message::FinishRestore { id } => {
                let dev = CONTEXTS.get(&(self.model_idx, id))
                    .map(|c| c.device.unwrap_or(0) as usize);
                self.finish_restore(id);
                if let Some(d) = dev {
                    self.try_serve_waiters(Some(d)).await;
                }
            }
            Message::GetStats { response } => {
                let stats: Vec<_> = self.devices.iter().map(|d| d.stats()).collect();
                let _ = response.send(stats);
            }
            Message::SetDagWeights { weight, pid_values } => {
                for (pid, value) in pid_values {
                    self.arbiter.set_node_weight(pid, weight * value);
                }
                self.try_serve_waiters(None).await;
            }
        }
    }
}
