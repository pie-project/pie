//! # Context Module — KV Cache Management
//!
//! Manages execution contexts with KV cache state for model inference.
//! Each model gets a dedicated `ContextManager` actor. All state lives
//! actor-locally — no interior mutability, no cross-actor locks.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                 Context Actor (per model)                     │
//! │                                                              │
//! │  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
//! │  │PageStore  │  │ ProcessEntry│  │alloc_queue│  │restore_queue│
//! │  │(per device)│  │(scheduling)│  │  (FIFO)   │  │(pri heap) │ │
//! │  └──────────┘  └───────────┘  └───────────┘  └───────────┘ │
//! └──────────────────────────────────────────────────────────────┘
//!         ▲                                      │
//!         │  reserve / commit / destroy          │ suspend / restore
//!         │  pin / unpin                         ▼
//! ┌──────────────────────────────────────────────────────────────┐
//! │  Process (inferlet)                                          │
//! │  Single-threaded WASM — blocked on response channel when     │
//! │  enqueued. Cannot make other WIT calls until response.       │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## State Model
//!
//! **Process** (2 states):
//! - **Running** — all contexts Active, inferlet executing.
//! - **Pending** — contexts Suspended, blocked on restore_queue.
//!
//! **Context** (3 states):
//!
//! | State         | Working Pages | Committed Chain      | Evictable?           |
//! |---------------|---------------|----------------------|----------------------|
//! | **Active**    | On GPU        | Refcounted in trie   | Yes                  |
//! | **Pinned**    | On GPU        | Refcounted in trie   | Deferred (`pending_suspend`) |
//! | **Suspended** | On CPU (swap) | Released (metadata)  | Nothing to evict     |
//!
//! ## Page Types
//!
//! **Working pages** (`working_pages: Vec<PhysicalPageId>`):
//! - GPU-exclusive, mutable, owned by one context.
//! - On suspend: freed from GPU. On restore: recomputed via replay
//!   using `working_page_tokens` (the token metadata survives suspension).
//! - Pages with no corresponding tokens are empty capacity reservations
//!   and are simply re-allocated fresh on restore.
//!
//! **Committed pages** (content-addressed via chained `PageHash`):
//! - Shared across contexts via refcount in PageStore (Radix Trie).
//! - `committed_hashes: Vec<PageHash>` — ordered root-to-tip chain.
//! - On suspend: refcounts decremented; hashes kept as metadata for restore.
//! - On restore: longest prefix match → replay missing suffix.
//!
//! ## Module Structure
//!
//! - `context.rs` — Public API, `Message` enum, `ServiceHandler`, core ops.
//! - `pagestore.rs` — `PageStore`: Radix Trie CAS cache + physical page pools.
//! - `sched.rs` — Scheduling economics, eviction, suspension, contention.
//! - `restore.rs` — Restoration, replay planning, `restore_queue`.
//! - `snapshot.rs` — Named snapshot save/load/fork/take.
//!
//! ## WIT Host Function Pattern
//!
//! Every WIT function that touches pages goes through the actor:
//! ```text
//! async fn wit_reserve_pages(pid, ctx_id, n) -> Result<()> {
//!     context::reserve_pages(model_idx, ctx_id, n).await
//! }
//! ```
//! No `wait_if_pending` needed. The process is single-threaded WASM —
//! when `reserve_pages` gets enqueued, the process blocks on `.await`.
//! When the actor serves the request (from alloc_queue or after restore),
//! it sends Ok and the process resumes, already Running.
// Radix Trie with path-inclusive refcounting — handles incremental
// commits and dedup correctly, all operations O(depth).
pub mod pagestore;
pub(crate) mod sched;
mod snapshot;
mod restore;

use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::sync::LazyLock;
use std::time::Instant;
use dashmap::DashMap;
use tokio::sync::oneshot;
use anyhow::{Result, Context as _};


use crate::service::{ServiceArray, ServiceHandler};
use crate::adapter::AdapterId;
use crate::inference::brle::Brle;
use crate::process::ProcessId;
use crate::device::{self, DeviceId};

use pagestore::{PhysicalPageId, PageHash, PageStore, FlatPageStore};
use sched::{AuctionResult, ProcessEntry, PendingAlloc};

// =============================================================================
// Public Types
// =============================================================================

pub type ContextId = u64;

/// Entry in the restore priority queue (BinaryHeap).
///
/// Ordering: non-defaulted before defaulted, then highest bid first.
/// Uses snapshot of `bid` and `defaulted` at enqueue time. The heap
/// provides O(log N) push/pop vs the previous O(N) full scan.
/// Stale entries (context already restored/destroyed) are lazily
/// filtered on pop.
#[derive(Debug, Clone)]
pub struct RestoreEntry {
    pub(crate) ctx_id: ContextId,
    pub(crate) bid: f64,
    pub(crate) defaulted: bool,
}

impl PartialEq for RestoreEntry {
    fn eq(&self, other: &Self) -> bool {
        self.ctx_id == other.ctx_id
    }
}
impl Eq for RestoreEntry {}

impl PartialOrd for RestoreEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RestoreEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Non-defaulted (false=0) sorts HIGHER than defaulted (true=1).
        // In a max-heap, we want non-defaulted first.
        other.defaulted.cmp(&self.defaulted)
            .then_with(|| self.bid.partial_cmp(&other.bid).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// =============================================================================
// Globals
// =============================================================================

pub(crate) static SERVICES: LazyLock<ServiceArray<Message>> = LazyLock::new(ServiceArray::new);
static PAGE_SIZES: LazyLock<boxcar::Vec<usize>> = LazyLock::new(boxcar::Vec::new);

// ---------------------------------------------------------------------------
// Lock-free read caches — written by the actor, read directly by callers.
// ---------------------------------------------------------------------------

/// Per-context snapshot: device + page counts.  Single DashMap lookup
/// instead of four separate maps.
pub(crate) static CACHED_CONTEXT_INFO: LazyLock<DashMap<(usize, ContextId), CachedContextInfo>> =
    LazyLock::new(DashMap::new);

/// Per-model market data: clearing prices, dividend rate, balances.
/// Indexed by `model_idx` (the spawn order).
pub(crate) static MARKET: LazyLock<boxcar::Vec<Market>> =
    LazyLock::new(boxcar::Vec::new);

/// Real-time pinned context count per device (max 8 devices).
/// Updated atomically on every pin/unpin — readable without actor overhead.
static PINNED_COUNTS: [std::sync::atomic::AtomicUsize; 8] = [
    std::sync::atomic::AtomicUsize::new(0), std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0), std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0), std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0), std::sync::atomic::AtomicUsize::new(0),
];

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CachedContextInfo {
    pub device: usize,
    pub working_pages: u32,
    pub committed_pages: u32,
    pub working_tokens: u32,
}

pub(crate) struct Market {
    /// Per-device clearing prices.  Key = device ordinal.
    pub clearing_prices: DashMap<usize, f64>,
    /// Per-device tick latency EWA (seconds, α=0.1).  Key = device ordinal.
    pub tick_latency_ewa: DashMap<usize, f64>,
    /// Sum of `dividend_per_endowment` across all devices.
    pub dividend_rate: std::sync::atomic::AtomicU64,  // f64 bits via to/from_bits
    /// Per-process credit balances (market wallet, unit: pages).
    pub balances: DashMap<ProcessId, f64>,
    /// Per-process endowments (fixed at creation, unit: pages).
    pub endowments: DashMap<ProcessId, f64>,
    /// Per-process remaining token budget (compute wallet, unit: tokens).
    /// `None` = unlimited, no cap.
    pub tokens_remaining: DashMap<ProcessId, Option<usize>>,
    /// Default credit endowment (pages) for new processes.
    pub default_credit: usize,
    /// Per-device GPU-resident Active context count (updated each tick).
    pub gpu_active: DashMap<usize, usize>,
    /// Per-device GPU-resident Pinned context count (updated each tick).
    pub gpu_pinned: DashMap<usize, usize>,
    /// Per-device count of contexts charged rent this tick (= batch size).
    pub gpu_charged: DashMap<usize, usize>,
}

impl Market {
    pub(crate) fn new(default_credit: usize) -> Self {
        Market {
            clearing_prices: DashMap::new(),
            tick_latency_ewa: DashMap::new(),
            dividend_rate: std::sync::atomic::AtomicU64::new(0),
            balances: DashMap::new(),
            endowments: DashMap::new(),
            tokens_remaining: DashMap::new(),
            default_credit,
            gpu_active: DashMap::new(),
            gpu_pinned: DashMap::new(),
            gpu_charged: DashMap::new(),
        }
    }

    pub(crate) fn get_dividend_rate(&self) -> f64 {
        f64::from_bits(self.dividend_rate.load(std::sync::atomic::Ordering::Relaxed))
    }

    pub(crate) fn set_dividend_rate(&self, rate: f64) {
        self.dividend_rate.store(rate.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
}


// =============================================================================
// Public API
// =============================================================================

/// Spawns a new context manager for a model.
///
/// - `default_endowment_pages`: market weight assigned to processes that
///   don't declare an explicit `token_budget` at admission.
/// - `default_tokens_remaining`: compute-wallet cap for the same;
///   `None` means unlimited (the system-wide default).
pub fn spawn(
    page_size: usize,
    num_gpu_pages: Vec<usize>,
    num_cpu_pages: Vec<usize>,
    default_endowment_pages: usize,
    default_tokens_remaining: Option<usize>,
    oversubscription_factor: f64,
) -> usize {
    PAGE_SIZES.push(page_size);
    MARKET.push(Market::new(default_endowment_pages));
    SERVICES.spawn(move || ContextManager::new(
        SERVICES.len().saturating_sub(1),
        page_size,
        &num_gpu_pages,
        &num_cpu_pages,
        default_endowment_pages,
        default_tokens_remaining,
        oversubscription_factor,
    )).expect("Failed to spawn context manager")
}

// ---------- Actor-routed ----------

pub async fn lookup(model_idx: usize, username: String, name: String) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Lookup { username, name, response: tx })?;
    rx.await.context("context::lookup: actor dropped response")?
}

pub async fn take(model_idx: usize, username: String, name: String, owner: ProcessId) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Take { username, name, owner, response: tx })?;
    rx.await.context("context::take: actor dropped response")?
}

pub async fn create(model_idx: usize, owner: ProcessId) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Create { owner, response: tx })?;
    rx.await.context("context::create: actor dropped response")?
}

/// Save a context under a name. If `name` is None, auto-generates a snapshot name.
/// Returns the name used (only meaningful when auto-generated).
pub async fn save(model_idx: usize, id: ContextId, username: String, name: Option<String>) -> Result<Option<String>> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Save { id, username, name, response: tx })?;
    rx.await.context("context::save: actor dropped response")?
}

pub async fn delete(model_idx: usize, username: String, name: String) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Delete { username, name, response: tx })?;
    rx.await.context("context::delete: actor dropped response")?
}

pub async fn destroy(model_idx: usize, id: ContextId) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Destroy { id, response: tx })?;
    rx.await.context("context::destroy: actor dropped response")?
}

/// Register a process across all models.
/// Called from `InstanceState::new` before any context operations.
///
/// Fails fast if any model's admission gate would refuse the request (the
/// `Σ endowment ≤ capacity × oversubscription_factor` invariant).
/// On partial failure — e.g., model 0 admits but model 1 refuses — the
/// successful registrations are rolled back so no orphan state remains.
pub async fn register_process(pid: ProcessId, token_budget: Option<usize>) -> Result<()> {
    let mut admitted: Vec<usize> = Vec::new();
    for model_idx in 0..SERVICES.len() {
        let (tx, rx) = oneshot::channel();
        SERVICES.send(model_idx, Message::RegisterProcess {
            pid, token_budget, response: tx,
        })?;
        match rx.await.context("register_process: actor dropped response")? {
            Ok(()) => admitted.push(model_idx),
            Err(e) => {
                for m in admitted {
                    let _ = SERVICES.send(m, Message::UnregisterProcess { pid });
                }
                return Err(e.context(format!(
                    "register_process failed on model {model_idx}"
                )));
            }
        }
    }
    Ok(())
}

/// Unregister a process: destroy all contexts and remove the process entry.
/// Called on WASM instance drop for automatic cleanup.
pub fn unregister_process(pid: ProcessId) {
    for model_idx in 0..SERVICES.len() {
        let _ = SERVICES.send(model_idx, Message::UnregisterProcess { pid });
    }
}

pub async fn fork(model_idx: usize, id: ContextId, owner: ProcessId) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Fork { id, owner, response: tx })?;
    rx.await.context("context::fork: actor dropped response")?
}

pub async fn commit_working_pages(model_idx: usize, id: ContextId, num_pages: usize) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::CommitWorkingPages { id, num_pages, response: tx })?;
    rx.await.context("context::commit_working_pages: actor dropped response")?
}

pub async fn reserve_working_pages(model_idx: usize, id: ContextId, num_pages: usize) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::ReserveWorkingPages { id, num_pages, response: tx })?;
    rx.await.context("context::reserve_working_pages: actor dropped response")?
}

pub fn release_working_pages(model_idx: usize, id: ContextId, num_pages: usize) -> Result<()> {
    SERVICES.send(model_idx, Message::ReleaseWorkingPages { id, num_pages })
}

/// Pin context for a forward pass: Active → Pinned.
/// Returns a PinnedContext with physical page IDs, kv_len, and last_page_len.
/// The context is non-evictable until `unpin`.
pub async fn pin(model_idx: usize, id: ContextId, num_input_tokens: u32) -> Result<PinnedContext> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Pin { id, num_input_tokens, response: tx })?;
    rx.await.context("context::pin: actor dropped response")?
}

/// Unpin context: Pinned → Active. Fire-and-forget actor message.
/// Also executes deferred suspension if `pending_suspend` was set.
pub fn unpin(model_idx: usize, id: ContextId) {
    let _ = SERVICES.send(model_idx, Message::Unpin { id });
}

pub async fn get_stats(model_idx: usize) -> Vec<(usize, usize)> {
    let (tx, rx) = oneshot::channel();
    let _ = SERVICES.send(model_idx, Message::GetStats { response: tx });
    rx.await.unwrap_or_default()
}

pub async fn debug_context_state(model_idx: usize, id: ContextId) -> String {
    let (tx, rx) = oneshot::channel();
    let _ = SERVICES.send(model_idx, Message::DebugState { id, response: tx });
    rx.await.unwrap_or_else(|_| "MISSING".to_string())
}


// ---------- Market (broadcast to all models) ----------

/// Execute one market tick on all models for a specific device.
/// Called per batch completion from the inference scheduler.
/// `batch_ctx_ids` lists the context IDs that were in the just-completed batch;
/// only these contexts are charged rent (prevents stale-unpin overcollection).
pub fn tick(device: usize, latency_secs: f64, batch_ctx_ids: Vec<ContextId>) {
    for model_idx in 0..SERVICES.len() {
        let _ = SERVICES.send(model_idx, Message::Tick { device, latency_secs, batch_ctx_ids: batch_ctx_ids.clone() });
    }
}

/// Count GPU-resident contexts on a device: (active, pinned).
/// Reads from `MARKET` cache published each tick — lock-free, O(1).
pub fn resident_count(device: usize) -> (usize, usize) {
    // Sum across all models (usually just one).
    let mut active = 0usize;
    let mut pinned = 0usize;
    for model_idx in 0..MARKET.count() {
        if let Some(market) = MARKET.get(model_idx) {
            active += market.gpu_active.get(&device).map(|v| *v).unwrap_or(0);
            pinned += market.gpu_pinned.get(&device).map(|v| *v).unwrap_or(0);
        }
    }
    (active, pinned)
}

/// Real-time pinned context count for a device.
/// Updated atomically on every pin/unpin — no actor overhead.
pub fn pinned_count(device: usize) -> usize {
    if device < PINNED_COUNTS.len() {
        PINNED_COUNTS[device].load(std::sync::atomic::Ordering::Relaxed)
    } else {
        0
    }
}

/// Get clearing price for a device from a specific model's context manager.
/// Reads directly from the lock-free cache (zero actor overhead).
pub fn get_clearing_price(model_idx: usize, device: usize) -> f64 {
    MARKET.get(model_idx)
        .and_then(|m| m.clearing_prices.get(&device).map(|v| *v))
        .unwrap_or(0.0)
}

/// Get EWA-smoothed tick latency (seconds, α=0.1) for a device.
/// Reads directly from the lock-free cache (zero actor overhead).
pub fn get_tick_latency(model_idx: usize, device: usize) -> f64 {
    MARKET.get(model_idx)
        .and_then(|m| m.tick_latency_ewa.get(&device).map(|v| *v))
        .unwrap_or(0.0)
}

/// Get dividend rate from a specific model's context manager.
pub fn get_dividend_rate(model_idx: usize) -> f64 {
    MARKET.get(model_idx)
        .map(|m| m.get_dividend_rate())
        .unwrap_or(0.0)
}

/// Get a process's credit balance.
pub fn get_balance(model_idx: usize, pid: ProcessId) -> f64 {
    MARKET.get(model_idx)
        .and_then(|m| m.balances.get(&pid).map(|v| *v))
        .unwrap_or(0.0)
}

/// Get a process's endowment (fixed at creation, used for dividend weighting).
pub fn get_endowment(model_idx: usize, pid: ProcessId) -> f64 {
    MARKET.get(model_idx)
        .and_then(|m| m.endowments.get(&pid).map(|v| *v))
        .unwrap_or(0.0)
}

/// Get a process's remaining token budget (compute wallet).
/// Returns `None` for unknown processes or processes with no cap.
/// `Some(n)` means the process is capped and has `n` tokens left.
pub fn get_tokens_remaining(model_idx: usize, pid: ProcessId) -> Option<usize> {
    MARKET.get(model_idx)
        .and_then(|m| m.tokens_remaining.get(&pid).map(|v| *v))
        .flatten()
}

/// Get the device index assigned to a specific context.
pub fn get_device(model_idx: usize, id: ContextId) -> usize {
    CACHED_CONTEXT_INFO.get(&(model_idx, id)).map(|v| v.device).unwrap_or(0)
}

/// Set a context's bid (willingness to pay per page per step).
pub async fn bid(model_idx: usize, id: ContextId, bid: f64) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Bid { id, bid, response: tx })?;
    rx.await.context("context::bid: actor dropped response")?
}

/// Suspend a context (program-initiated).
pub async fn suspend(model_idx: usize, id: ContextId) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Suspend { id, response: tx })?;
    rx.await.context("context::suspend: actor dropped response")?
}

// ---------- Direct (no actor) ----------

pub fn tokens_per_page(model_idx: usize) -> u32 {
    PAGE_SIZES.get(model_idx).map(|v| *v as u32).unwrap_or(0)
}

/// Default endowment (in pages) for new processes on a model.
pub fn default_endowment(model_idx: usize) -> f64 {
    MARKET.get(model_idx).map(|m| m.default_credit as f64).unwrap_or(0.0)
}

// ---------- DashMap-cached reads (zero actor overhead) ----------

pub fn committed_page_count(model_idx: usize, id: ContextId) -> u32 {
    CACHED_CONTEXT_INFO.get(&(model_idx, id)).map(|v| v.committed_pages).unwrap_or(0)
}

pub fn working_page_count(model_idx: usize, id: ContextId) -> u32 {
    CACHED_CONTEXT_INFO.get(&(model_idx, id)).map(|v| v.working_pages).unwrap_or(0)
}

pub fn working_page_token_count(model_idx: usize, id: ContextId) -> u32 {
    CACHED_CONTEXT_INFO.get(&(model_idx, id)).map(|v| v.working_tokens).unwrap_or(0)
}

pub async fn truncate_working_page_tokens(model_idx: usize, id: ContextId, count: u32) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::TruncateWorkingPageTokens { id, count, response: tx })?;
    rx.await.context("context::truncate_working_page_tokens: actor dropped response")?
}

pub async fn append_working_page_tokens(
    model_idx: usize, id: ContextId, tokens: Vec<u32>,
    positions: Vec<u32>, masks: Vec<Brle>, adapter: Option<AdapterId>,
    adapter_seed: Option<i64>,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::AppendWorkingPageTokens {
        id, tokens, positions, masks, adapter, adapter_seed, response: tx,
    })?;
    rx.await.context("context::append_working_page_tokens: actor dropped response")?
}

// =============================================================================
// PinnedContext — returned by pin()
// =============================================================================

#[derive(Debug)]
pub struct PinnedContext {
    pub device: DeviceId,
    pub pages: Vec<PhysicalPageId>,
    pub kv_len: u32,
    pub last_page_len: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct TokenInfo {
    pub token: u32,
    pub position: u32,
    pub mask: Brle,
    pub adapter: Option<AdapterId>,
    pub adapter_seed: Option<i64>,
}

// =============================================================================
// Internal Types
// =============================================================================

#[derive(Debug, Clone)]
pub(crate) enum Record {
    Fill {
        tokens: Vec<u32>,
        positions: Vec<u32>,
        mask: Vec<Brle>,
        adapter: Option<AdapterId>,
        adapter_seed: Option<i64>,
    },
}


// =============================================================================
// Context — lives in local HashMap on ContextManager (actor-only)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum State {
    /// Active on GPU, ready for inference. Evictable.
    Active,
    /// Active on GPU, forward pass in progress — NOT immediately evictable.
    /// Eviction is deferred via `pending_suspend` flag.
    Pinned,
    /// Off GPU, pages cached on CPU. Warm restore via H2D copy.
    Stashed,
    /// Off GPU, no pages anywhere. Cold restore via full recompute.
    Suspended,
}

#[derive(Debug)]
pub(crate) struct Context {
    /// Process that owns this context (None for named snapshots).
    pub owner: Option<ProcessId>,
    /// Device this context is on (None if fully evicted).
    pub device: Option<DeviceId>,
    /// Physical page IDs for uncommitted (working) pages.
    /// On GPU when Active/Pinned, empty when Suspended.
    pub working_pages: Vec<PhysicalPageId>,
    /// Number of working pages at suspension time. Used during restore
    /// to re-allocate the right number of fresh GPU pages (since
    /// `working_pages` is cleared on suspend instead of swapped to CPU).
    pub suspended_working_count: usize,
    /// Ordered committed page hashes (root-to-tip). Replaces committed_tip + committed_len.
    pub committed_hashes: Vec<PageHash>,
    /// Full token lineage for replay after eviction.
    pub lineage: Vec<Record>,

    // Token-level data (previously in ContextTokens / BUFFERS DashMap)
    /// Tokens that have been forwarded but not yet committed to a page.
    pub working_page_tokens: Vec<TokenInfo>,

    /// Maximum position value across all committed tokens. Need to check the validity of committed tokens
    pub max_committed_position: Option<u32>,

    // Scheduling
    pub state: State,
    /// Deferred suspension flag: set when context is Pinned and selected as victim.
    /// Actual suspension happens on clear_pinned.
    pub pending_suspend: bool,
    pub last_access: Instant,
    /// Program-declared bid (willingness to pay per page per step).
    pub bid: f64,
    /// CPU page IDs holding stashed working pages (exclusive, not in FlatPageStore).
    /// Committed pages are tracked by the FlatPageStore FlatMap, not here.
    pub cpu_working_pages: Vec<PhysicalPageId>,
    /// Operations deferred while this context is Suspended.
    /// Replayed after restoration completes.
    pub deferred_ops: Vec<PendingAlloc>,
    /// True while replay forward passes are in-flight after restoration.
    pub pending_replay: bool,
    /// True while a fork/save d2d copy is in-flight on the device.
    /// While set, the context is held in `State::Pinned` and the
    /// scheduler must not include it in `fire_batch` — the device
    /// pages backing this context have not yet been written by the
    /// async copy_d2d kernel and reading them would give stale KV.
    /// Cleared on `Message::D2DComplete` from the spawned awaiter.
    /// See `project_pie_kv_bleed_d2d_fork_race.md`.
    pub pending_d2d: bool,
    /// True when the owning process couldn't afford full rent last tick.
    /// Defaulted contexts are evicted first regardless of bid.
    /// Recomputed each tick — not sticky.
    pub defaulted: bool,
    /// Cached Shapley effective page count for committed pages.
    /// Updated on commit_working_pages and restore. Read by tick.pass2
    /// to avoid per-tick trie traversal.
    pub cached_effective_pages: f64,
}

impl Context {
    pub(crate) fn new(owner: Option<ProcessId>) -> Self {
        Context {
            owner,
            device: None,
            working_pages: Vec::new(),
            suspended_working_count: 0,
            committed_hashes: Vec::new(),
            lineage: Vec::new(),
            working_page_tokens: Vec::new(),
            max_committed_position: None,
            state: State::Active,
            pending_suspend: false,
            last_access: Instant::now(),
            bid: 0.0,
            cpu_working_pages: Vec::new(),
            deferred_ops: Vec::new(),
            pending_replay: false,
            pending_d2d: false,
            defaulted: false,
            cached_effective_pages: 0.0,
        }
    }

    pub fn is_active(&self) -> bool { self.state == State::Active }
    pub fn is_suspended(&self) -> bool { self.state == State::Suspended }
    pub fn is_stashed(&self) -> bool { self.state == State::Stashed }
    pub fn is_pinned(&self) -> bool { self.state == State::Pinned }
    /// True when the context is off GPU (Stashed or Suspended).
    pub fn is_off_gpu(&self) -> bool { matches!(self.state, State::Stashed | State::Suspended) }

    /// Tip of the committed hash chain (last element), or None if empty.
    pub fn committed_tip(&self) -> Option<PageHash> { self.committed_hashes.last().copied() }
    /// Number of committed pages.
    pub fn committed_len(&self) -> usize { self.committed_hashes.len() }

    /// Number of working pages whose KV data can be recomputed from
    /// `working_page_tokens` via a replay forward pass.
    /// Remaining pages (if any) are empty capacity reservations.
    pub fn recomputable_working_pages(&self, page_size: usize) -> usize {
        if page_size == 0 { return 0; }
        (self.working_page_tokens.len() + page_size - 1) / page_size
    }
}

// =============================================================================
// ContextManager
// =============================================================================

/// Lightweight diagnostic counters for scheduler health monitoring.
/// Reset after each summary dump.
#[derive(Debug, Default)]
pub(crate) struct SchedCounters {
    /// Number of tick() calls since last dump.
    pub ticks: u64,
    /// Contexts suspended due to contention (eviction victims).
    pub eviction_suspends: u64,
    /// Contexts self-suspended due to priority gate (lower bid than restore_queue head).
    pub priority_gate_suspends: u64,
    /// Contexts self-suspended because no eviction victim found.
    pub no_victim_suspends: u64,
    /// Contexts successfully restored from restore_queue.
    pub restores: u64,
    /// Contexts rejected from restore by can_restore (insufficient pages or credit).
    pub restore_rejections: u64,
    /// Contexts flagged as defaulted (can't pay rent) in a tick.
    pub defaults_flagged: u64,
    /// Total eviction victim searches.
    pub eviction_searches: u64,

    // --- Per-message-type cumulative timing (microseconds) ---
    pub tick_us: u64,
    pub tick_count: u64,
    pub pin_us: u64,
    pub pin_count: u64,
    pub unpin_us: u64,
    pub unpin_count: u64,
    pub reserve_us: u64,
    pub reserve_count: u64,
    pub release_us: u64,
    pub release_count: u64,
    pub commit_us: u64,
    pub commit_count: u64,
    pub bid_us: u64,
    pub bid_count: u64,
    pub replay_us: u64,
    pub replay_count: u64,
    pub register_us: u64,
    pub register_count: u64,
    pub unregister_us: u64,
    pub unregister_count: u64,
    pub destroy_us: u64,
    pub destroy_count: u64,
    pub append_us: u64,
    pub append_count: u64,

    // --- tick() sub-operation timing ---
    pub tick_pass1_us: u64,
    pub tick_pass2_us: u64,
    pub tick_pass3_us: u64,
    pub tick_publish_us: u64,

    // --- unregister_process() sub-operation timing ---
    pub unreg_queues_us: u64,
    pub unreg_destroy_us: u64,
    pub unreg_drain_us: u64,

    // --- drain_queues timing ---
    pub drain_queues_us: u64,
    pub drain_queues_count: u64,
}

#[derive(Debug)]
pub(crate) struct ContextManager {
    /// Per-device GPU page stores (radix trie). Indexed by device ordinal.
    /// Manages committed KV-cache pages with refcounted prefix sharing.
    pub(crate) gpu_stores: Vec<PageStore>,
    /// Per-device CPU page stores (flat map). Indexed by device ordinal.
    /// Holds stashed pages for suspended contexts (D2H copies).
    pub(crate) cpu_stores: Vec<FlatPageStore>,
    /// Tokens per KV-cache page. Used to convert token budgets to credit endowments.
    pub(crate) page_size: usize,
    /// Index of the model this manager serves. Used for routing messages.
    pub(crate) model_idx: usize,
    /// Named snapshots: (username, name) → snapshot context ID.
    pub(crate) snapshots: HashMap<(String, String), ContextId>,
    /// Monotonically increasing context ID generator.
    next_id: u64,
    /// Per-process state: credit balance, endowment, owned context IDs.
    pub(crate) processes: HashMap<ProcessId, ProcessEntry>,
    /// Per-context state: pages, device, bid, lineage, suspension info.
    pub(crate) contexts: HashMap<ContextId, Context>,
    /// FIFO queue: contexts with pending deferred allocs waiting for free GPU pages.
    pub(crate) alloc_queue: VecDeque<ContextId>,
    /// Restore queue: suspended contexts waiting for restoration, served by highest bid.
    /// Uses a max-heap with lazy deletion — stale entries filtered on pop.
    pub(crate) restore_queue: BinaryHeap<RestoreEntry>,
    /// Per-device auction results from the last tick (clearing price, revenue, dividend rate).
    pub(crate) auction_results: Vec<AuctionResult>,
    /// Default credit endowment (pages) for new processes that do not
    /// declare an explicit token_budget at admission. Pure market weight —
    /// independent of the token wallet.
    pub(crate) default_endowment: f64,
    /// Default compute-wallet cap for new processes that do not declare
    /// an explicit token_budget at admission.
    /// `None` = unlimited (the system-wide default); `Some(n)` = cap at `n`.
    pub(crate) default_tokens_remaining: Option<usize>,
    /// Admission cap: `Σ endowment ≤ total_gpu_capacity × oversubscription_factor`.
    /// At 1.0 strictly bound by physical capacity; > 1.0 allows overbook.
    pub(crate) oversubscription_factor: f64,
    /// Diagnostic counters for scheduler health.
    pub(crate) sched_counters: SchedCounters,
}

impl ContextManager {
    /// Hard admission control threshold: don't restore when any device
    /// exceeds this page utilization fraction. Prevents the
    /// evict→restore→re-evict thrashing cascade.
    const RESTORE_UTILIZATION_CAP: f64 = 0.85;

    pub(crate) fn new(
        model_idx: usize,
        page_size: usize,
        num_gpu_pages: &[usize],
        num_cpu_pages: &[usize],
        default_endowment_pages: usize,
        default_tokens_remaining: Option<usize>,
        oversubscription_factor: f64,
    ) -> Self {
        let gpu_stores: Vec<_> = num_gpu_pages.iter()
            .map(|&n| PageStore::new(page_size, n))
            .collect();
        let cpu_stores: Vec<_> = num_cpu_pages.iter()
            .map(|&n| FlatPageStore::new(n))
            .collect();
        ContextManager {
            gpu_stores, cpu_stores, page_size, model_idx,
            snapshots: HashMap::new(), next_id: 1,
            processes: HashMap::new(),
            contexts: HashMap::new(),
            alloc_queue: VecDeque::new(),
            restore_queue: BinaryHeap::new(),
            auction_results: vec![AuctionResult::default(); num_gpu_pages.len()],
            default_endowment: default_endowment_pages as f64,
            default_tokens_remaining,
            oversubscription_factor,
            sched_counters: SchedCounters::default(),
        }
    }

    /// Comprehensive GPU page audit: accounts for every allocated page.
    /// free + trie_pages + working_active + working_pinned = total (if balanced).
    /// Any gap = leaked pages.
    #[allow(dead_code, unused_variables)]
    pub(crate) fn page_audit(&self) {
        for (dev_idx, dev) in self.gpu_stores.iter().enumerate() {
            let free = dev.available();
            let total = dev.total_pages();
            let (trie_pages, trie_rc, trie_nodes, rc0_interior) = dev.trie_stats();

            let mut working_active = 0usize;
            let mut working_pinned = 0usize;
            let mut working_suspended = 0usize; // should be 0 (suspended = CPU)
            let mut committed_active = 0usize;  // committed hashes held by active contexts
            let mut committed_pinned = 0usize;
            let mut n_active = 0usize;
            let mut n_pinned = 0usize;
            let mut n_suspended = 0usize;

            for ctx in self.contexts.values() {
                let ctx_dev = ctx.device.unwrap_or(0) as usize;
                if ctx_dev != dev_idx { continue; }
                match ctx.state {
                    State::Active => {
                        working_active += ctx.working_pages.len();
                        committed_active += ctx.committed_hashes.len();
                        n_active += 1;
                    }
                    State::Pinned => {
                        working_pinned += ctx.working_pages.len();
                        committed_pinned += ctx.committed_hashes.len();
                        n_pinned += 1;
                    }
                    State::Suspended | State::Stashed => {
                        working_suspended += ctx.working_pages.len();
                        n_suspended += 1;
                    }
                }
            }

            let accounted = free + trie_pages + working_active + working_pinned;
            let leaked = if total > accounted { total - accounted } else { 0 };
            let over = if accounted > total { accounted - total } else { 0 };
            let (alloc_tot, free_tot) = dev.pool_stats();
            let (f_reclaim, f_release, f_working) = dev.tag_stats();

            // tracing::debug!(
            //     "[PAGE_AUDIT] dev={} total={} | free={} trie={} work_active={} work_pinned={} | accounted={} LEAKED={} OVER={} | \\
            //      pool: alloc={} free={} net={} | tags: reclaim={} release={} working={} | \\
            //      ctxs: active={} pinned={} suspended={} | commit_active={} commit_pinned={} | \\
            //      trie: rc={} nodes={} rc0_int={} | rq={} aq={}",
            //     dev_idx, total, free, trie_pages, working_active, working_pinned,
            //     accounted, leaked, over,
            //     alloc_tot, free_tot, alloc_tot as isize - free_tot as isize,
            //     f_reclaim, f_release, f_working,
            //     n_active, n_pinned, n_suspended,
            //     committed_active, committed_pinned,
            //     trie_rc, trie_nodes, rc0_interior,
            //     self.restore_queue.len(), self.alloc_queue.len(),
            // );

            // When overcounting is detected, identify phantom pages
            if over > 0 {
                let (phantom_count, phantom_desc) = dev.phantom_audit();
                if phantom_count > 0 {
                    tracing::error!("[PHANTOM_AUDIT] dev={} phantom_count={} pages=[{}]",
                        dev_idx, phantom_count, phantom_desc);
                }
            }
        }
    }

    fn next_id(&mut self) -> ContextId {
        let id = self.next_id; self.next_id += 1; id
    }

    fn least_loaded_device(&self) -> usize {
        self.gpu_stores.iter().enumerate()
            .max_by_key(|(_, d)| d.available())
            .map(|(i, _)| i).unwrap_or(0)
    }

    // ==================== Core Operations ====================

    pub(crate) fn create(&mut self, owner: ProcessId) -> Result<ContextId> {
        let id = self.next_id();
        let mut ctx = Context::new(Some(owner));
        ctx.device = Some(self.least_loaded_device());

        self.contexts.insert(id, ctx);

        let proc = self.process_entry(owner);
        proc.context_ids.push(id);

        self.publish_context_counts(id);
        Ok(id)
    }

    pub(crate) fn destroy(&mut self, id: ContextId) -> Result<()> {
        let ctx = self.contexts.remove(&id)
            .ok_or_else(|| anyhow::anyhow!("Context {id} not found"))?;

        let dev_idx = ctx.device.unwrap_or(0) as usize;

        if let Some(pid) = ctx.owner {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.context_ids.retain(|&c| c != id);
            }
        }

        // Drop this context from alloc_queue.
        self.alloc_queue.retain(|&ctx_id| ctx_id != id);

        // restore_queue: lazy deletion — stale entries filtered on pop in drain_queues.

        // Release committed chain (skip if already released during suspension)
        if !ctx.committed_hashes.is_empty() && !ctx.is_off_gpu() {
            self.gpu_stores[dev_idx].release(&ctx.committed_hashes);
        }

        // Free working pages (GPU when Active/Pinned; empty when Suspended).
        if !ctx.working_pages.is_empty() {
            self.gpu_stores[dev_idx].free(&ctx.working_pages);
        }

        // Free CPU working pages stash (if suspended with CPU stash).
        if !ctx.cpu_working_pages.is_empty() {
            self.cpu_stores[dev_idx].free(&ctx.cpu_working_pages);
        }

        // Release CPU-resident committed pages (suspended contexts may have
        // had their committed pages stashed to CPU via would_free + cpu.insert).
        if ctx.is_off_gpu() && !ctx.committed_hashes.is_empty() {
            self.cpu_stores[dev_idx].release(&ctx.committed_hashes);
        }

        self.snapshots.retain(|_, v| *v != id);

        self.remove_context_caches(id);
        self.drain_queues();
        Ok(())
    }

    

    

    // ==================== Page Management ====================

    /// Handle a ReserveWorkingPages message per DESIGN.md §4.
    /// Delegates to the universal `when_allocated` contention primitive.
    pub(crate) fn reserve_working_pages(
        &mut self,
        id: ContextId,
        num_pages: usize,
        response: oneshot::Sender<anyhow::Result<()>>,
    ) {
        if num_pages == 0 {
            let _ = response.send(Ok(()));
            return;
        }

        let ctx = match self.contexts.get(&id) {
            Some(c) => c,
            None => { let _ = response.send(Err(anyhow::anyhow!("Context not found"))); return; }
        };
        let dev_idx = ctx.device.unwrap_or(0) as usize;

        self.when_allocated(id, dev_idx, num_pages, move |mgr, pages| {
            if let Some(ctx) = mgr.contexts.get_mut(&id) {
                ctx.working_pages.extend_from_slice(&pages);
                ctx.device = Some(dev_idx);
            }
            mgr.publish_context_counts(id);
            let _ = response.send(Ok(()));
        });
    }

    pub(crate) fn release_working_pages(&mut self, id: ContextId, num_pages: usize) -> Result<()> {
        let ctx = self.contexts.get_mut(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        if num_pages == 0 { return Ok(()); }
        if num_pages > ctx.working_pages.len() {
            anyhow::bail!("release: requested {num_pages}, have {}", ctx.working_pages.len());
        }

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let start = ctx.working_pages.len() - num_pages;
        let to_free: Vec<_> = ctx.working_pages.drain(start..).collect();

        // Truncate working page tokens
        let tokens_to_remove = num_pages * self.page_size;
        let len = ctx.working_page_tokens.len();
        ctx.working_page_tokens.truncate(len.saturating_sub(tokens_to_remove));

        if !ctx.is_off_gpu() {
            self.gpu_stores[dev_idx].free(&to_free);
        }
        self.publish_context_counts(id);
        self.drain_queues();
        Ok(())
    }

    pub(crate) fn commit_working_pages(&mut self, id: ContextId, num_pages: usize) -> Result<()> {
        let page_size = self.page_size;
        let ctx = self.contexts.get(&id).ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        if num_pages == 0 { return Ok(()); }

        // Suspended contexts have empty working_pages but track the count
        // in suspended_working_count (working pages are recomputed on restore).
        let available_pages = if ctx.is_off_gpu() {
            ctx.suspended_working_count
        } else {
            ctx.working_pages.len()
        };
        if num_pages > available_pages {
            anyhow::bail!("commit: requested {num_pages}, have {}", available_pages);
        }

        let total_tokens = num_pages * page_size;
        if total_tokens > ctx.working_page_tokens.len() {
            anyhow::bail!("commit: need {total_tokens} tokens, have {}", ctx.working_page_tokens.len());
        }

        let dev_idx = ctx.device.unwrap_or(0) as usize;
        let prev_hash = ctx.committed_tip().unwrap_or(0);
        let pages = ctx.working_pages.get(..num_pages).map(|s| s.to_vec()).unwrap_or_default();

        // Extract token data for the pages being committed.
        let mut tokens = Vec::with_capacity(total_tokens);
        let mut positions = Vec::with_capacity(total_tokens);
        let mut masks = Vec::with_capacity(total_tokens);
        for info in &ctx.working_page_tokens[..total_tokens] {
            tokens.push(info.token);
            positions.push(info.position);
            masks.push(info.mask.clone());
        }

        // Validate positions are strictly after any previously committed position.
        if let Some(max_committed) = ctx.max_committed_position {
            for &pos in &positions {
                if pos <= max_committed {
                    anyhow::bail!("Position {} must be > max committed position {}", pos, max_committed);
                }
            }
        }
        let lineage_adapter = ctx.working_page_tokens.first().and_then(|t| t.adapter);
        let lineage_adapter_seed = ctx.working_page_tokens.first().and_then(|t| t.adapter_seed);

        // Compute content-based hashes (includes adapter_seed so ZO-perturbed
        // pages are not shared with unperturbed or differently-seeded pages).
        let hashes = pagestore::compute_page_hashes(page_size, &tokens, &positions, &masks, prev_hash, lineage_adapter_seed);
        let existing_prefix = ctx.committed_hashes.clone();
        let dev = &mut self.gpu_stores[dev_idx];

        // Commit: physical (GPU promotion + dedup) or logical (metadata only).
        if ctx.is_off_gpu() {
            // Suspended: no physical pages to promote (working pages were freed
            // on suspend). Metadata-only commit — on restore, the committed chain
            // replay will regenerate these pages.
        } else {
            // Use extend to navigate the trie through the existing
            // committed chain before inserting new pages as children.
            dev.extend(&existing_prefix, &hashes, &pages);
        }

        // Update context state.
        let ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context lost during commit"))?;
        if ctx.is_off_gpu() {
            ctx.suspended_working_count = ctx.suspended_working_count.saturating_sub(num_pages);
        } else {
            ctx.working_pages.drain(..num_pages);
        }
        ctx.working_page_tokens.drain(..total_tokens);
        ctx.committed_hashes.extend_from_slice(&hashes);
        ctx.max_committed_position = positions.iter().copied().max()
            .or(ctx.max_committed_position);

        // Refresh cached effective_pages after chain extension.
        if !ctx.is_off_gpu() {
            ctx.cached_effective_pages = self.gpu_stores[dev_idx].effective_pages(&ctx.committed_hashes);
        }

        // Append to lineage (merge with last record if same adapter AND seed).
        if let Some(Record::Fill { tokens: t, positions: p, mask: m, adapter: a, adapter_seed: s }) = ctx.lineage.last_mut() {
            if *a == lineage_adapter && *s == lineage_adapter_seed {
                t.extend_from_slice(&tokens);
                p.extend_from_slice(&positions);
                m.extend_from_slice(&masks);
            } else {
                ctx.lineage.push(Record::Fill { tokens, positions, mask: masks, adapter: lineage_adapter, adapter_seed: lineage_adapter_seed });
            }
        } else {
            ctx.lineage.push(Record::Fill { tokens, positions, mask: masks, adapter: lineage_adapter, adapter_seed: lineage_adapter_seed });
        }

        self.drain_queues();
        self.publish_context_counts(id);
        Ok(())
    }

    /// Handle a Pin request: transition an Active context to Pinned.
    ///
    /// Resolves physical page IDs for committed + working pages, computes
    /// `last_page_len`, and returns `PinnedContext` for the inference forward pass.
    /// If the context is Suspended, defers the operation until restoration.
    pub(crate) fn pin(
        &mut self,
        id: ContextId,
        num_input_tokens: u32,
        response: oneshot::Sender<Result<PinnedContext>>,
    ) {
        // Defer pin if the context is Pinned with d2d/replay in flight:
        // pin returns page IDs that fire_batch reads, so we MUST wait
        // for the working pages to actually contain valid data. Other
        // content-independent ops (Reserve/Commit/Append) can proceed.
        // See `project_pie_kv_bleed_d2d_fork_race.md`.
        if let Some(ctx) = self.contexts.get_mut(&id) {
            if ctx.is_pinned() && (ctx.pending_replay || ctx.pending_d2d) {
                let on_alloc = move |mgr: &mut ContextManager, _pages: Vec<pagestore::PhysicalPageId>| {
                    mgr.pin(id, num_input_tokens, response);
                };
                ctx.deferred_ops.push(PendingAlloc {
                    device: ctx.device.unwrap_or(0) as usize,
                    num_pages: 0,
                    on_alloc: Box::new(on_alloc),
                });
                return;
            }
        }
        self.when_active(id, move |mgr| {
            let result = (|| -> Result<PinnedContext> {
                // Token-budget gate: refuse to pin for a forward pass when
                // the owning process has exhausted its compute budget. The
                // actual debit happens on append_working_page_tokens after
                // the pass completes; this pre-check prevents overdraft.
                if num_input_tokens > 0
                    && !mgr.has_token_budget(id, num_input_tokens as usize)
                {
                    anyhow::bail!(
                        "pin: token budget exhausted for context {id} \
                         (requested {num_input_tokens} tokens)"
                    );
                }

                let ctx = mgr.contexts.get_mut(&id)
                    .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
                if ctx.is_off_gpu() {
                    anyhow::bail!("pin: context is suspended (cannot pin)");
                }
                let dev_idx = ctx.device.unwrap_or(0) as usize;
                let committed_hashes = ctx.committed_hashes.clone();
                let working = ctx.working_pages.clone();
                let kv_len = (ctx.committed_len() * mgr.page_size
                    + ctx.working_page_tokens.len()) as u32;
                ctx.state = State::Pinned;
                PINNED_COUNTS[dev_idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                let mut page_ids = Vec::new();
                if !committed_hashes.is_empty() {
                    page_ids.extend(mgr.gpu_stores[dev_idx].physical_ids(&committed_hashes));
                }
                page_ids.extend(&working);

                let num_pages = page_ids.len() as u32;
                let page_size = mgr.page_size as u32;
                let total_kv = kv_len + num_input_tokens;
                let last_page_len = pagestore::compute_last_page_len(
                    total_kv, num_pages, page_size,
                );
                Ok(PinnedContext {
                    device: dev_idx as DeviceId, pages: page_ids,
                    kv_len, last_page_len,
                })
            })();
            let _ = response.send(result);
        });
    }

    /// Handle Unpin message: Pinned → Active, then check deferred suspension.
    /// If `pending_suspend` was set (and this isn't a replay context),
    /// executes the deferred suspension and enqueues for restoration.
    pub(crate) fn unpin(&mut self, id: ContextId) {
        let (pending, replay, d2d, dev) = match self.contexts.get(&id) {
            Some(ctx) if ctx.is_pinned() => (
                ctx.pending_suspend,
                ctx.pending_replay,
                ctx.pending_d2d,
                ctx.device.unwrap_or(0) as usize,
            ),
            _ => return,
        };

        // Decrement real-time pinned counter (all paths leaving Pinned).
        if dev < PINNED_COUNTS.len() {
            PINNED_COUNTS[dev].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }

        // If this is a replay or d2d-pending context, the corresponding
        // *_complete handler owns the Pinned → Active transition.
        if replay || d2d {
            return;
        }

        if pending {
            // Deferred suspension: suspend the context and enqueue for restoration.
            self.suspend(id);
            self.enqueue_restore(id);

            // Deferred suspension may have freed pages — drain queues.
            self.drain_queues();
            return;
        }

        // No deferred suspension — normal Pinned → Active transition.
        if let Some(ctx) = self.contexts.get_mut(&id) {
            ctx.state = State::Active;
        }
    }


    /// Central queue drain: called after any event that frees GPU pages.
    ///
    /// Phase 1: alloc_queue (FIFO) — invoke deferred GPU operation callbacks.
    /// Phase 2: restore_queue (priority heap) — restore highest-bid Suspended
    ///          context, with per-restore placement evaluation (§4.3).
    pub(crate) fn drain_queues(&mut self) {
        let t0 = Instant::now();
        // Phase 1: alloc_queue FIFO — serve deferred ops for head context.
        while let Some(&front_ctx_id) = self.alloc_queue.front() {
            // Skip stale entries (destroyed contexts or empty deferred_ops).
            let front_op = match self.contexts.get(&front_ctx_id)
                .and_then(|c| c.deferred_ops.first())
            {
                Some(op) => op,
                None => { self.alloc_queue.pop_front(); continue; }
            };
            let (dev_idx, n) = (front_op.device, front_op.num_pages);
            if n > 0 && self.gpu_stores[dev_idx].available() < n {
                break;
            }
            let ctx_id = self.alloc_queue.pop_front().unwrap();
            self.fire_deferred_ops(ctx_id);
        }

        // Phase 2: restore_queue — pop highest-bid Suspended context from heap.
        // Only proceed if alloc_queue is empty (allocs have strict priority).
        if !self.alloc_queue.is_empty() {
            return;
        }

        // Hard admission control: don't restore when any device is near
        // page capacity. This prevents over-admission that causes the
        // thrashing cascade (evict→restore→re-evict cycle). Processes
        // wait here until running processes complete and free pages,
        // naturally dropping utilization below the threshold.
        let over_capacity = self.gpu_stores.iter().any(|d| {
            let (used, total) = d.stats();
            let utilization = used as f64 / total.max(1) as f64;
            utilization > Self::RESTORE_UTILIZATION_CAP
        });
        if over_capacity {
            return;
        }

        let num_devices = self.gpu_stores.len();
        let max_rejections = self.restore_queue.len();
        let mut rejections = 0;
        let mut re_enqueue = Vec::new();

        while let Some(entry) = self.restore_queue.pop() {
            let ctx_id = entry.ctx_id;

            // Lazy deletion: skip stale entries (destroyed or already restored).
            let is_off_gpu = self.contexts.get(&ctx_id)
                .map(|c| c.is_off_gpu()).unwrap_or(false);
            if !is_off_gpu {
                continue;
            }

            // Per-restore placement evaluation (§4.3): check if a different
            // device would be cheaper before restoring.
            if num_devices > 1 {
                if let Some(ctx) = self.contexts.get(&ctx_id) {
                    let best = self.best_device_for(ctx);
                    if best != ctx.device.unwrap_or(0) as usize {
                        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                            ctx.device = Some(best);
                        }
                    }
                }
            }

            // Admission check: enough free pages for this context?
            if !self.can_restore(ctx_id) {
                self.sched_counters.restore_rejections += 1;
                re_enqueue.push(ctx_id);
                rejections += 1;
                if rejections >= max_rejections {
                    break;
                }
                continue;
            }

            self.sched_counters.restores += 1;
            if let Err(e) = self.restore(ctx_id) {
                tracing::error!(ctx = ctx_id, "restore failed: {e:#}");
            }
        }

        // Re-enqueue rejected contexts.
        for ctx_id in re_enqueue {
            self.enqueue_restore(ctx_id);
        }

        self.sched_counters.drain_queues_us += t0.elapsed().as_micros() as u64;
        self.sched_counters.drain_queues_count += 1;
    }

    // ==================== Extracted handle() helpers ====================


    pub(crate) fn stats(&self) -> Vec<(usize, usize)> {
        self.gpu_stores.iter().map(|d| d.stats()).collect()
    }

    /// Publish the per-context counts + device to the global cache.
    /// Called after any mutation that changes working pages, committed pages,
    /// working page tokens, or device assignment.
    pub(crate) fn publish_context_counts(&self, id: ContextId) {
        if let Some(ctx) = self.contexts.get(&id) {
            CACHED_CONTEXT_INFO.insert((self.model_idx, id), CachedContextInfo {
                device: ctx.device.unwrap_or(0) as usize,
                working_pages: ctx.working_pages.len() as u32,
                committed_pages: ctx.committed_len() as u32,
                working_tokens: ctx.working_page_tokens.len() as u32,
            });
        }
    }

    /// Remove cached entry for a context (on destroy).
    pub(crate) fn remove_context_caches(&self, id: ContextId) {
        CACHED_CONTEXT_INFO.remove(&(self.model_idx, id));
    }

    pub(crate) fn truncate_working_page_tokens(&mut self, id: ContextId, count: u32) -> Result<()> {
        let ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let max = ctx.working_page_tokens.len();
        if count as usize > max {
            anyhow::bail!("truncate count {} out of range 0..={}", count, max);
        }
        ctx.working_page_tokens.truncate(count as usize);
        Ok(())
    }

    pub(crate) fn append_working_page_tokens(
        &mut self, id: ContextId, tokens: Vec<u32>, positions: Vec<u32>,
        masks: Vec<Brle>, adapter: Option<AdapterId>, adapter_seed: Option<i64>,
    ) -> Result<()> {
        let n = tokens.len();
        if positions.len() != n {
            anyhow::bail!("positions length {} != n {}", positions.len(), n);
        }
        if !masks.is_empty() && masks.len() != n {
            anyhow::bail!("masks length {} != n {}", masks.len(), n);
        }
        let ctx = self.contexts.get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let owner = ctx.owner;
        for (i, token) in tokens.into_iter().enumerate() {
            ctx.working_page_tokens.push(TokenInfo {
                token, position: positions[i],
                mask: if masks.is_empty() { Brle::new(0) } else { masks[i].clone() },
                adapter,
                adapter_seed,
            });
        }
        // Debit the token wallet for the work just completed. Snapshots
        // (owner=None) bypass billing.
        if let Some(pid) = owner {
            self.debit_tokens(pid, n);
        }
        Ok(())
    }

    pub(crate) fn debug_state(&self, id: ContextId) -> String {
        match self.contexts.get(&id) {
            Some(ctx) => format!("{ctx:?}"),
            None => "NOT_FOUND".to_string(),
        }
    }
}

// =============================================================================
// Message & ServiceHandler
// =============================================================================

#[derive(Debug)]
pub(crate) enum Message {
    Lookup { username: String, name: String, response: oneshot::Sender<Result<ContextId>> },
    Create { owner: ProcessId, response: oneshot::Sender<Result<ContextId>> },
    Save { id: ContextId, username: String, name: Option<String>, response: oneshot::Sender<Result<Option<String>>> },
    Delete { username: String, name: String, response: oneshot::Sender<Result<()>> },
    Destroy { id: ContextId, response: oneshot::Sender<Result<()>> },
    Fork { id: ContextId, owner: ProcessId, response: oneshot::Sender<Result<ContextId>> },
    Take { username: String, name: String, owner: ProcessId, response: oneshot::Sender<Result<ContextId>> },
    CommitWorkingPages { id: ContextId, num_pages: usize, response: oneshot::Sender<Result<()>> },
    ReserveWorkingPages { id: ContextId, num_pages: usize, response: oneshot::Sender<Result<()>> },
    ReleaseWorkingPages { id: ContextId, num_pages: usize },
    Pin { id: ContextId, num_input_tokens: u32, response: oneshot::Sender<Result<PinnedContext>> },
    Unpin { id: ContextId },
    ReplayComplete { id: ContextId },
    /// Sent by the d2d awaiter task after `device::copy_d2d_async`
    /// resolves. Clears `pending_d2d` and, if `pending_replay` is
    /// also clear, transitions Pinned → Active. Mirrors the
    /// ReplayComplete pattern. See `project_pie_kv_bleed_d2d_fork_race.md`.
    D2DComplete { id: ContextId },
    GetStats { response: oneshot::Sender<Vec<(usize, usize)>> },

    // Actor-routed write APIs
    TruncateWorkingPageTokens { id: ContextId, count: u32, response: oneshot::Sender<Result<()>> },
    AppendWorkingPageTokens {
        id: ContextId, tokens: Vec<u32>, positions: Vec<u32>,
        masks: Vec<Brle>, adapter: Option<AdapterId>, adapter_seed: Option<i64>,
        response: oneshot::Sender<Result<()>>,
    },

    DebugState { id: ContextId, response: oneshot::Sender<String> },

    RegisterProcess {
        pid: ProcessId,
        token_budget: Option<usize>,
        response: oneshot::Sender<Result<()>>,
    },
    UnregisterProcess { pid: ProcessId },

    // ── Market messages ────────────────────────────────────────
    /// Execute one market tick for a single device.
    /// `batch_ctx_ids` = context IDs from the just-completed batch.
    Tick { device: usize, latency_secs: f64, batch_ctx_ids: Vec<ContextId> },
    /// Set a context's bid (willingness to pay per page per step).
    Bid { id: ContextId, bid: f64, response: oneshot::Sender<Result<()>> },
    /// Suspend a context (program-initiated).
    Suspend { id: ContextId, response: oneshot::Sender<Result<()>> },
}

impl ServiceHandler for ContextManager {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        let _t = Instant::now();
        match msg {
            Message::Lookup { username, name, response } => {
                let result = self.snapshots.get(&(username, name))
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("Snapshot not found"));
                let _ = response.send(result);
            }
            Message::Take { username, name, owner, response } => {
                self.take(username, name, owner, response);
            }
            Message::Create { owner, response } => {
                let _ = response.send(self.create(owner));
            }
            Message::Save { id, username, name, response } => {
                let _ = response.send(self.save(id, username, name));
            }
            Message::Delete { username, name, response } => {
                let _ = response.send(self.delete(username, name));
            }
            Message::Destroy { id, response } => {
                let t0 = Instant::now();
                let _ = response.send(self.destroy(id));
                self.sched_counters.destroy_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.destroy_count += 1;
            }
            Message::Fork { id, owner, response } => {
                self.fork(id, owner, response);
            }
            Message::CommitWorkingPages { id, num_pages, response } => {
                let t0 = Instant::now();
                let _ = response.send(self.commit_working_pages(id, num_pages));
                self.sched_counters.commit_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.commit_count += 1;
            }
            Message::ReserveWorkingPages { id, num_pages, response } => {
                let t0 = Instant::now();
                self.reserve_working_pages(id, num_pages, response);
                self.sched_counters.reserve_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.reserve_count += 1;
            }
            Message::ReleaseWorkingPages { id, num_pages } => {
                let t0 = Instant::now();
                let _ = self.release_working_pages(id, num_pages);
                self.sched_counters.release_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.release_count += 1;
            }
            Message::Pin { id, num_input_tokens, response } => {
                let t0 = Instant::now();
                self.pin(id, num_input_tokens, response);
                self.sched_counters.pin_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.pin_count += 1;
            }
            Message::Unpin { id } => {
                let t0 = Instant::now();
                self.unpin(id);
                self.sched_counters.unpin_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.unpin_count += 1;
            }
            Message::ReplayComplete { id } => {
                let t0 = Instant::now();
                self.replay_complete(id);
                self.sched_counters.replay_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.replay_count += 1;
            }
            Message::D2DComplete { id } => {
                self.d2d_complete(id);
            }
            Message::GetStats { response } => {
                let _ = response.send(self.stats());
            }
            Message::TruncateWorkingPageTokens { id, count, response } => {
                let _ = response.send(self.truncate_working_page_tokens(id, count));
                self.publish_context_counts(id);
            }
            Message::AppendWorkingPageTokens { id, tokens, positions, masks, adapter, adapter_seed, response } => {
                let t0 = Instant::now();
                let _ = response.send(self.append_working_page_tokens(id, tokens, positions, masks, adapter, adapter_seed));
                self.publish_context_counts(id);
                self.sched_counters.append_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.append_count += 1;
            }
            Message::DebugState { id, response } => {
                let _ = response.send(self.debug_state(id));
            }
            Message::RegisterProcess { pid, token_budget, response } => {
                let t0 = Instant::now();
                let result = self.register_process(pid, token_budget);
                let _ = response.send(result);
                self.sched_counters.register_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.register_count += 1;
            }
            Message::UnregisterProcess { pid } => {
                let t0 = Instant::now();
                self.unregister_process(pid);
                self.sched_counters.unregister_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.unregister_count += 1;
            }

            // ── Market handlers ────────────────────────────────────
            Message::Tick { device, latency_secs, batch_ctx_ids } => {
                let t0 = Instant::now();
                self.tick(device, latency_secs, &batch_ctx_ids);
                self.sched_counters.tick_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.tick_count += 1;
                self.sched_counters.ticks += 1;
                // Diagnostic logging (uncomment for debugging):
                // if self.sched_counters.ticks % 1000 == 0 { ... }
            }
            Message::Bid { id, bid, response } => {
                let t0 = Instant::now();
                let _ = response.send(self.bid(id, bid));
                self.sched_counters.bid_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.bid_count += 1;
            }
            Message::Suspend { id, response } => {
                let _ = response.send(self.voluntary_suspend(id));
            }
        }
    }
}
