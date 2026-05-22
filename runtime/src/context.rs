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
//! │  │(per driver)│  │(scheduling)│  │  (FIFO)   │  │(pri heap) │ │
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
mod restore;
mod rs_cache;
pub(crate) mod sched;
mod snapshot;

use anyhow::{Context as _, Result};
use dashmap::DashMap;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::LazyLock;
use std::time::Instant;
use tokio::sync::{Notify, oneshot};

use crate::adapter::AdapterId;
use crate::driver::DriverId;
use crate::process::ProcessId;
use crate::service::{ServiceArray, ServiceHandler};
use pie_bridge::Brle;

use pagestore::{FlatPageStore, PageHash, PageStore, PhysicalPageId};
use rs_cache::{RS_FLAG_RESET, RsSlotId, RsState, RsStore};
use sched::{AuctionResult, PendingAlloc, ProcessEntry};

// =============================================================================
// Public Types
// =============================================================================

/// Stable per-context identifier. Survives swap / restore — backends that
/// maintain per-context scratch (n-gram drafter history, etc.) should key on
/// this rather than on volatile KV-page indices. Pie-internal — the wire
/// schema carries it as plain `[uint64]` in `ForwardRequest.context_ids`.
pub type ContextId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RsPinSlot {
    Ready(Option<RsSlotId>, u8),
    Deferred,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RsSlotAlloc {
    Ready(RsSlotId),
    Deferred,
}

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
        other.defaulted.cmp(&self.defaulted).then_with(|| {
            self.bid
                .partial_cmp(&other.bid)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
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

/// Per-context snapshot: driver + page counts.  Single DashMap lookup
/// instead of four separate maps.
pub(crate) static CACHED_CONTEXT_INFO: LazyLock<DashMap<(usize, ContextId), CachedContextInfo>> =
    LazyLock::new(DashMap::new);

/// Per-model market data: clearing prices, dividend rate, balances.
/// Indexed by `model_idx` (the spawn order).
pub(crate) static MARKET: LazyLock<boxcar::Vec<Market>> = LazyLock::new(boxcar::Vec::new);
static ADMISSION_NOTIFY: LazyLock<Notify> = LazyLock::new(Notify::new);

/// Real-time pinned context count per driver (max 8 drivers).
/// Updated atomically on every pin/unpin — readable without actor overhead.
static PINNED_COUNTS: [std::sync::atomic::AtomicUsize; 8] = [
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
];

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CachedContextInfo {
    pub driver: usize,
    /// Visible working pages — what the SDK's `working_page_count()` returns.
    pub working_pages: u32,
    pub committed_pages: u32,
    /// Visible working-page tokens — what the SDK's
    /// `working_page_token_count()` returns.
    pub working_tokens: u32,
}

pub(crate) struct Market {
    /// Per-driver clearing prices.  Key = driver ordinal.
    pub clearing_prices: DashMap<usize, f64>,
    /// Per-driver tick latency EWA (seconds, α=0.1).  Key = driver ordinal.
    pub tick_latency_ewa: DashMap<usize, f64>,
    /// Sum of `dividend_per_endowment` across all drivers.
    pub dividend_rate: std::sync::atomic::AtomicU64, // f64 bits via to/from_bits
    /// Per-process credit balances (market wallet, unit: pages).
    pub balances: DashMap<ProcessId, f64>,
    /// Per-process endowments (fixed at creation, unit: pages).
    pub endowments: DashMap<ProcessId, f64>,
    /// Per-process remaining token budget (compute wallet, unit: tokens).
    /// `None` = unlimited, no cap.
    pub tokens_remaining: DashMap<ProcessId, Option<usize>>,
    /// Default credit endowment (pages) for new processes.
    pub default_credit: usize,
    /// Per-driver GPU-resident Active context count (updated each tick).
    pub gpu_active: DashMap<usize, usize>,
    /// Per-driver GPU-resident Pinned context count (updated each tick).
    pub gpu_pinned: DashMap<usize, usize>,
    /// Per-driver count of contexts charged rent this tick (= batch size).
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
        f64::from_bits(
            self.dividend_rate
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    pub(crate) fn set_dividend_rate(&self, rate: f64) {
        self.dividend_rate
            .store(rate.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Spawns a new context manager for a model.
///
/// - `default_endowment_pages`: market weight assigned to processes that
///   don't declare an explicit token limit at admission.
/// - `default_token_limit`: compute-wallet cap for the same;
///   `None` means unlimited (the system-wide default).
/// - `admission_oversubscription_factor`: admission gate — `Σ admission_claim`
///   must not exceed `total_pages × factor`.
/// - `restore_pause_at_utilization`: the restore loop pauses when any
///   driver's GPU page utilization exceeds this fraction.
pub fn spawn(
    page_size: usize,
    num_gpu_pages: Vec<usize>,
    num_cpu_pages: Vec<usize>,
    max_forward_requests: usize,
    num_rs_slots: Vec<usize>,
    default_endowment_pages: usize,
    default_token_limit: Option<usize>,
    admission_oversubscription_factor: f64,
    restore_pause_at_utilization: f64,
) -> usize {
    let model_idx = SERVICES.len();
    PAGE_SIZES.push(page_size);
    MARKET.push(Market::new(default_endowment_pages));
    SERVICES
        .spawn(move || {
            ContextManager::new(
                model_idx,
                page_size,
                &num_gpu_pages,
                &num_cpu_pages,
                max_forward_requests,
                &num_rs_slots,
                default_endowment_pages,
                default_token_limit,
                admission_oversubscription_factor,
                restore_pause_at_utilization,
            )
        })
        .expect("Failed to spawn context manager")
}

// ---------- Actor-routed ----------

pub async fn lookup(model_idx: usize, username: String, name: String) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        model_idx,
        Message::Lookup {
            username,
            name,
            response: tx,
        },
    )?;
    rx.await
        .context("context::lookup: actor dropped response")?
}

pub async fn take(
    model_idx: usize,
    username: String,
    name: String,
    owner: ProcessId,
) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        model_idx,
        Message::Take {
            username,
            name,
            owner,
            response: tx,
        },
    )?;
    rx.await.context("context::take: actor dropped response")?
}

pub async fn create(model_idx: usize, owner: ProcessId) -> Result<ContextId> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        model_idx,
        Message::Create {
            owner,
            response: tx,
        },
    )?;
    rx.await
        .context("context::create: actor dropped response")?
}

/// Save a context under a name. If `name` is None, auto-generates a snapshot name.
/// Returns the name used (only meaningful when auto-generated).
pub async fn save(
    model_idx: usize,
    id: ContextId,
    username: String,
    name: Option<String>,
) -> Result<Option<String>> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        model_idx,
        Message::Save {
            id,
            username,
            name,
            response: tx,
        },
    )?;
    rx.await.context("context::save: actor dropped response")?
}

pub async fn delete(model_idx: usize, username: String, name: String) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        model_idx,
        Message::Delete {
            username,
            name,
            response: tx,
        },
    )?;
    rx.await
        .context("context::delete: actor dropped response")?
}

pub async fn destroy(model_idx: usize, id: ContextId) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Destroy { id, response: tx })?;
    rx.await
        .context("context::destroy: actor dropped response")?
}

/// Register a process across all models.
/// Called from `InstanceState::new` before any context operations.
///
/// Waits if a model's admission gate is temporarily full (the
/// `Σ admission_claim ≤ capacity × admission_oversubscription_factor` invariant).
/// On partial failure — e.g., model 0 admits but model 1 refuses — the
/// successful registrations are rolled back so no orphan state remains.
pub async fn register_process(pid: ProcessId, token_budget: Option<usize>) -> Result<()> {
    loop {
        match try_register_process(pid, token_budget).await {
            Ok(()) => return Ok(()),
            Err(e) => {
                let Some(admission) = e.downcast_ref::<sched::AdmissionDenied>() else {
                    return Err(e);
                };
                if admission.admission_pages > admission.cap {
                    return Err(e);
                }
                tokio::select! {
                    _ = ADMISSION_NOTIFY.notified() => {}
                    _ = tokio::time::sleep(std::time::Duration::from_millis(1)) => {}
                }
            }
        }
    }
}

async fn try_register_process(pid: ProcessId, token_budget: Option<usize>) -> Result<()> {
    let mut admitted: Vec<usize> = Vec::new();
    for model_idx in 0..SERVICES.len() {
        let (tx, rx) = oneshot::channel();
        SERVICES.send(
            model_idx,
            Message::RegisterProcess {
                pid,
                token_budget,
                response: tx,
            },
        )?;
        match rx
            .await
            .context("register_process: actor dropped response")?
        {
            Ok(()) => admitted.push(model_idx),
            Err(e) => {
                for m in admitted {
                    let _ = SERVICES.send(m, Message::UnregisterProcess { pid });
                }
                return Err(e.context(format!("register_process failed on model {model_idx}")));
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
    SERVICES.send(
        model_idx,
        Message::Fork {
            id,
            owner,
            response: tx,
        },
    )?;
    rx.await.context("context::fork: actor dropped response")?
}

pub async fn commit_working_pages(model_idx: usize, id: ContextId, num_pages: usize) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        model_idx,
        Message::CommitWorkingPages {
            id,
            num_pages,
            response: tx,
        },
    )?;
    rx.await
        .context("context::commit_working_pages: actor dropped response")?
}

pub async fn reserve_working_pages(
    model_idx: usize,
    id: ContextId,
    num_pages: usize,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        model_idx,
        Message::ReserveWorkingPages {
            id,
            num_pages,
            response: tx,
        },
    )?;
    rx.await
        .context("context::reserve_working_pages: actor dropped response")?
}

pub fn release_working_pages(model_idx: usize, id: ContextId, num_pages: usize) -> Result<()> {
    SERVICES.send(model_idx, Message::ReleaseWorkingPages { id, num_pages })
}

/// Pin context for a forward pass: Active → Pinned.
/// Returns a PinnedContext with physical page IDs, kv_len, and last_page_len.
/// The context is non-evictable until `unpin`.
pub async fn pin(model_idx: usize, id: ContextId, num_input_tokens: u32) -> Result<PinnedContext> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        model_idx,
        Message::Pin {
            id,
            num_input_tokens,
            response: tx,
        },
    )?;
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

/// Execute one market tick on all models for a specific driver.
/// Called per batch completion from the inference scheduler.
/// `batch_ctx_ids` lists the context IDs that were in the just-completed batch;
/// only these contexts are charged rent (prevents stale-unpin overcollection).
pub fn tick(driver_idx: usize, latency_secs: f64, batch_ctx_ids: Vec<ContextId>) {
    for model_idx in 0..SERVICES.len() {
        let _ = SERVICES.send(
            model_idx,
            Message::Tick {
                driver_idx,
                latency_secs,
                batch_ctx_ids: batch_ctx_ids.clone(),
            },
        );
    }
}

/// Count GPU-resident contexts on a driver: (active, pinned).
/// Reads from `MARKET` cache published each tick — lock-free, O(1).
pub fn resident_count(driver_idx: usize) -> (usize, usize) {
    // Sum across all models (usually just one).
    let mut active = 0usize;
    let mut pinned = 0usize;
    for model_idx in 0..MARKET.count() {
        if let Some(market) = MARKET.get(model_idx) {
            active += market.gpu_active.get(&driver_idx).map(|v| *v).unwrap_or(0);
            pinned += market.gpu_pinned.get(&driver_idx).map(|v| *v).unwrap_or(0);
        }
    }
    (active, pinned)
}

/// Real-time pinned context count for a driver.
/// Updated atomically on every pin/unpin — no actor overhead.
pub fn pinned_count(driver_idx: usize) -> usize {
    if driver_idx < PINNED_COUNTS.len() {
        PINNED_COUNTS[driver_idx].load(std::sync::atomic::Ordering::Relaxed)
    } else {
        0
    }
}

/// Get clearing price for a driver from a specific model's context manager.
/// Reads directly from the lock-free cache (zero actor overhead).
pub fn get_clearing_price(model_idx: usize, driver_idx: usize) -> f64 {
    MARKET
        .get(model_idx)
        .and_then(|m| m.clearing_prices.get(&driver_idx).map(|v| *v))
        .unwrap_or(0.0)
}

/// Get EWA-smoothed tick latency (seconds, α=0.1) for a driver.
/// Reads directly from the lock-free cache (zero actor overhead).
pub fn get_tick_latency(model_idx: usize, driver_idx: usize) -> f64 {
    MARKET
        .get(model_idx)
        .and_then(|m| m.tick_latency_ewa.get(&driver_idx).map(|v| *v))
        .unwrap_or(0.0)
}

/// Get dividend rate from a specific model's context manager.
pub fn get_dividend_rate(model_idx: usize) -> f64 {
    MARKET
        .get(model_idx)
        .map(|m| m.get_dividend_rate())
        .unwrap_or(0.0)
}

/// Get a process's credit balance.
pub fn get_balance(model_idx: usize, pid: ProcessId) -> f64 {
    MARKET
        .get(model_idx)
        .and_then(|m| m.balances.get(&pid).map(|v| *v))
        .unwrap_or(0.0)
}

/// Get a process's endowment (fixed at creation, used for dividend weighting).
pub fn get_endowment(model_idx: usize, pid: ProcessId) -> f64 {
    MARKET
        .get(model_idx)
        .and_then(|m| m.endowments.get(&pid).map(|v| *v))
        .unwrap_or(0.0)
}

/// Get a process's remaining token budget (compute wallet).
/// Returns `None` for unknown processes or processes with no cap.
/// `Some(n)` means the process is capped and has `n` tokens left.
pub fn get_tokens_remaining(model_idx: usize, pid: ProcessId) -> Option<usize> {
    MARKET
        .get(model_idx)
        .and_then(|m| m.tokens_remaining.get(&pid).map(|v| *v))
        .flatten()
}

/// Get the driver index assigned to a specific context.
pub fn get_driver(model_idx: usize, id: ContextId) -> usize {
    CACHED_CONTEXT_INFO
        .get(&(model_idx, id))
        .map(|v| v.driver)
        .unwrap_or(0)
}

/// Set a context's bid (willingness to pay per page per step).
pub async fn bid(model_idx: usize, id: ContextId, bid: f64) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        model_idx,
        Message::Bid {
            id,
            bid,
            response: tx,
        },
    )?;
    rx.await.context("context::bid: actor dropped response")?
}

/// Suspend a context (program-initiated).
pub async fn suspend(model_idx: usize, id: ContextId) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Suspend { id, response: tx })?;
    rx.await
        .context("context::suspend: actor dropped response")?
}

// ---------- Direct (no actor) ----------

pub fn tokens_per_page(model_idx: usize) -> u32 {
    PAGE_SIZES.get(model_idx).map(|v| *v as u32).unwrap_or(0)
}

/// Default endowment (in pages) for new processes on a model.
pub fn default_endowment(model_idx: usize) -> f64 {
    MARKET
        .get(model_idx)
        .map(|m| m.default_credit as f64)
        .unwrap_or(0.0)
}

// ---------- DashMap-cached reads (zero actor overhead) ----------

pub fn committed_page_count(model_idx: usize, id: ContextId) -> u32 {
    CACHED_CONTEXT_INFO
        .get(&(model_idx, id))
        .map(|v| v.committed_pages)
        .unwrap_or(0)
}

pub fn working_page_count(model_idx: usize, id: ContextId) -> u32 {
    CACHED_CONTEXT_INFO
        .get(&(model_idx, id))
        .map(|v| v.working_pages)
        .unwrap_or(0)
}

pub fn working_page_token_count(model_idx: usize, id: ContextId) -> u32 {
    CACHED_CONTEXT_INFO
        .get(&(model_idx, id))
        .map(|v| v.working_tokens)
        .unwrap_or(0)
}

/// Look up the driver index a ctx is bound to (the device in pre-bridge
/// terminology). Returns 0 if the ctx isn't in the cache — used by the
/// speculator to resolve its per-(model, driver) deque.
pub fn get_device(model_idx: usize, id: ContextId) -> usize {
    CACHED_CONTEXT_INFO
        .get(&(model_idx, id))
        .map(|v| v.driver)
        .unwrap_or(0)
}

pub async fn truncate_working_page_tokens(
    model_idx: usize,
    id: ContextId,
    count: u32,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        model_idx,
        Message::TruncateWorkingPageTokens {
            id,
            count,
            response: tx,
        },
    )?;
    rx.await
        .context("context::truncate_working_page_tokens: actor dropped response")?
}

/// Fire-and-forget: queues a lineage append on the context actor's
/// mailbox and returns immediately. Errors (e.g., ctx already
/// destroyed mid-flight) are logged inside the handler rather than
/// propagated. Subsequent ops on the same ctx (`pin`, `commit`,
/// `truncate`, `destroy`, eviction) traverse the same mpsc and so
/// are naturally ordered behind this append — the state remains
/// consistent for every actor-side reader.
pub fn append_working_page_tokens(
    model_idx: usize,
    id: ContextId,
    tokens: Vec<u32>,
    positions: Vec<u32>,
    masks: Vec<Brle>,
    adapter: Option<AdapterId>,
    adapter_seed: Option<i64>,
) {
    let _ = SERVICES.send(
        model_idx,
        Message::AppendWorkingPageTokens {
            id,
            tokens,
            positions,
            masks,
            adapter,
            adapter_seed,
        },
    );
}

// =============================================================================
// PinnedContext — returned by pin()
// =============================================================================

#[derive(Debug)]
pub struct PinnedContext {
    pub driver: DriverId,
    /// Active pages: committed + the working prefix needed to cover
    /// `kv_len + num_input_tokens`. The kernel reads
    /// `total_kv = (pages.len()-1)*page_size + last_page_len`, so
    /// pages must contain only the active prefix for the math to
    /// be correct.
    pub pages: Vec<PhysicalPageId>,
    /// Working pages that the ctx has reserved beyond the active
    /// prefix in `pages`. Available to the speculator's chain
    /// extender for cross-page extension without re-allocating.
    pub extra_pages: Vec<PhysicalPageId>,
    pub kv_len: u32,
    pub last_page_len: u32,
    pub rs_slot: Option<RsSlotId>,
    pub rs_flags: u8,
}

#[derive(Debug, Clone)]
pub(crate) struct ReplayPageRegistration {
    pub driver: usize,
    pub prefix: Vec<PageHash>,
    pub hashes: Vec<PageHash>,
    pub pages: Vec<PhysicalPageId>,
}

#[derive(Debug, Clone)]
pub(crate) struct TokenInfo {
    pub token: u32,
    pub position: u32,
    pub mask: Brle,
    pub adapter: Option<AdapterId>,
    pub adapter_seed: Option<i64>,
    pub forward_id: u64,
}

pub(super) fn materialize_lineage_mask(mask: &Brle, position: u32) -> Brle {
    if mask.buffer.is_empty() && mask.total_size == 0 {
        Brle::all_true((position + 1) as usize)
    } else {
        mask.clone()
    }
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
        /// Original forward-pass id. KV replay may ignore this and merge
        /// adjacent fills, but rs_cache replay uses it to reconstruct the
        /// recurrent state with the same forward boundaries.
        forward_id: u64,
    },
}

fn record_from_token_infos(infos: &[TokenInfo]) -> Record {
    let adapter = infos.first().and_then(|t| t.adapter);
    let adapter_seed = infos.first().and_then(|t| t.adapter_seed);
    let forward_id = infos.first().map(|t| t.forward_id).unwrap_or(0);
    Record::Fill {
        tokens: infos.iter().map(|t| t.token).collect(),
        positions: infos.iter().map(|t| t.position).collect(),
        mask: infos.iter().map(|t| t.mask.clone()).collect(),
        adapter,
        adapter_seed,
        forward_id,
    }
}

fn append_fill_record(lineage: &mut Vec<Record>, record: Record, allow_merge: bool) {
    if allow_merge {
        if let (
            Some(Record::Fill {
                tokens: existing_tokens,
                positions: existing_positions,
                mask: existing_masks,
                adapter: existing_adapter,
                adapter_seed: existing_seed,
                forward_id: _,
            }),
            Record::Fill {
                tokens,
                positions,
                mask,
                adapter,
                adapter_seed,
                forward_id: _,
            },
        ) = (lineage.last_mut(), &record)
        {
            if *existing_adapter == *adapter && *existing_seed == *adapter_seed {
                existing_tokens.extend_from_slice(tokens);
                existing_positions.extend_from_slice(positions);
                existing_masks.extend_from_slice(mask);
                return;
            }
        }
    }
    lineage.push(record);
}

fn append_token_infos_to_lineage(
    lineage: &mut Vec<Record>,
    infos: &[TokenInfo],
    preserve_forward_boundaries: bool,
) {
    if infos.is_empty() {
        return;
    }
    if !preserve_forward_boundaries {
        append_fill_record(lineage, record_from_token_infos(infos), true);
        return;
    }

    let mut start = 0usize;
    while start < infos.len() {
        let forward_id = infos[start].forward_id;
        let mut end = start + 1;
        while end < infos.len() && infos[end].forward_id == forward_id {
            end += 1;
        }
        append_fill_record(lineage, record_from_token_infos(&infos[start..end]), false);
        start = end;
    }
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
    /// Driver this context is on (None if fully evicted).
    pub driver: Option<DriverId>,
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

    /// Runtime-owned recurrent-state cache lifecycle for linear-attention
    /// models. This is kept as an enum so invalid combinations such as
    /// "resident and missing" cannot be represented.
    pub rs_state: RsState,

    // Token-level data (previously in ContextTokens / BUFFERS DashMap)
    /// Tokens that have been forwarded but not yet committed to a page.
    pub working_page_tokens: Vec<TokenInfo>,
    /// Monotonic id for preserving forward-pass boundaries in rs_cache replay.
    pub next_forward_id: u64,

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
            driver: None,
            working_pages: Vec::new(),
            suspended_working_count: 0,
            committed_hashes: Vec::new(),
            lineage: Vec::new(),
            rs_state: RsState::Unsupported,
            working_page_tokens: Vec::new(),
            next_forward_id: 0,
            max_committed_position: None,
            state: State::Active,
            pending_suspend: false,
            last_access: Instant::now(),
            bid: 0.0,
            cpu_working_pages: Vec::new(),
            deferred_ops: Vec::new(),
            pending_replay: false,
            defaulted: false,
            cached_effective_pages: 0.0,
        }
    }

    pub fn is_active(&self) -> bool {
        self.state == State::Active
    }
    pub fn is_suspended(&self) -> bool {
        self.state == State::Suspended
    }
    pub fn is_stashed(&self) -> bool {
        self.state == State::Stashed
    }
    pub fn is_pinned(&self) -> bool {
        self.state == State::Pinned
    }
    /// True when the context is off GPU (Stashed or Suspended).
    pub fn is_off_gpu(&self) -> bool {
        matches!(self.state, State::Stashed | State::Suspended)
    }

    /// Tip of the committed hash chain (last element), or None if empty.
    pub fn committed_tip(&self) -> Option<PageHash> {
        self.committed_hashes.last().copied()
    }
    /// Number of committed pages.
    pub fn committed_len(&self) -> usize {
        self.committed_hashes.len()
    }

    /// Number of working pages whose KV data can be recomputed from
    /// `working_page_tokens` via a replay forward pass.
    /// Remaining pages (if any) are empty capacity reservations.
    pub fn recomputable_working_pages(&self, page_size: usize) -> usize {
        if page_size == 0 {
            return 0;
        }
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
    /// Per-driver GPU page stores (radix trie). Indexed by driver ordinal.
    /// Manages committed KV-cache pages with refcounted prefix sharing.
    pub(crate) gpu_stores: Vec<PageStore>,
    /// Per-driver CPU page stores (flat map). Indexed by driver ordinal.
    /// Holds stashed pages for suspended contexts (D2H copies).
    pub(crate) cpu_stores: Vec<FlatPageStore>,
    /// Per-driver recurrent-state slot pools. Empty stores mean this
    /// driver/model does not require rs_cache.
    pub(crate) rs_stores: Vec<RsStore>,
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
    /// Per-context state: pages, driver, bid, lineage, suspension info.
    pub(crate) contexts: HashMap<ContextId, Context>,
    /// FIFO queue: contexts with pending deferred allocs waiting for free GPU pages.
    pub(crate) alloc_queue: VecDeque<ContextId>,
    /// Restore queue: suspended contexts waiting for restoration, served by highest bid.
    /// Uses a max-heap with lazy deletion — stale entries filtered on pop.
    pub(crate) restore_queue: BinaryHeap<RestoreEntry>,
    /// Per-driver auction results from the last tick (clearing price, revenue, dividend rate).
    pub(crate) auction_results: Vec<AuctionResult>,
    /// Default credit endowment (pages) for new processes that do not
    /// declare an explicit token_budget at admission. Pure market weight —
    /// independent of the token wallet.
    pub(crate) default_endowment: f64,
    /// Default compute-wallet cap for new processes that do not declare
    /// an explicit token_budget at admission.
    /// `None` = unlimited (the system-wide default); `Some(n)` = cap at `n`.
    pub(crate) default_token_limit: Option<usize>,
    /// Admission cap: `Σ admission_claim ≤ total_gpu_capacity × admission_oversubscription_factor`.
    /// At 1.0 claims are strictly bound by physical capacity; > 1.0 allows overbook.
    pub(crate) admission_oversubscription_factor: f64,
    /// Hard admission gate for the restore loop: pause restoring suspended
    /// contexts when any driver's page utilization exceeds this fraction.
    /// Prevents the evict→restore→re-evict thrash cascade.
    pub(crate) restore_pause_at_utilization: f64,
    /// Total driver-reported forward request slots across this model's
    /// registered drivers. Admission uses this as the largest useful launch
    /// wave and as the upper bound on page-rounding slack.
    pub(crate) admission_wave_requests: usize,
    /// Set after an admission denial caused by a full endowment pool. While
    /// active, new admissions wait until enough endowment has drained to fit
    /// the next full wave that can live inside the effective admission cap.
    pub(crate) admission_drain_barrier: bool,
    /// Diagnostic counters for scheduler health.
    pub(crate) sched_counters: SchedCounters,
    /// Round-robin counter for new-context driver assignment. Used when
    /// `least_loaded_driver()` would otherwise tie (which it does for
    /// every burst-arriving context until at least one of them allocates
    /// pages — `available()` doesn't reflect in-flight allocations). Without
    /// this, every burst pins to the same driver and DP collapses to DP=1.
    next_driver_rr: usize,
}

impl ContextManager {
    pub(crate) fn new(
        model_idx: usize,
        page_size: usize,
        num_gpu_pages: &[usize],
        num_cpu_pages: &[usize],
        max_forward_requests: usize,
        num_rs_slots: &[usize],
        default_endowment_pages: usize,
        default_token_limit: Option<usize>,
        admission_oversubscription_factor: f64,
        restore_pause_at_utilization: f64,
    ) -> Self {
        let gpu_stores: Vec<_> = num_gpu_pages
            .iter()
            .map(|&n| PageStore::new(page_size, n))
            .collect();
        let cpu_stores: Vec<_> = num_cpu_pages
            .iter()
            .map(|&n| FlatPageStore::new(n))
            .collect();
        let rs_stores: Vec<_> = num_rs_slots.iter().map(|&n| RsStore::new(n)).collect();
        ContextManager {
            gpu_stores,
            cpu_stores,
            rs_stores,
            page_size,
            model_idx,
            snapshots: HashMap::new(),
            next_id: 1,
            processes: HashMap::new(),
            contexts: HashMap::new(),
            alloc_queue: VecDeque::new(),
            restore_queue: BinaryHeap::new(),
            auction_results: vec![AuctionResult::default(); num_gpu_pages.len()],
            default_endowment: default_endowment_pages as f64,
            default_token_limit,
            admission_oversubscription_factor,
            restore_pause_at_utilization,
            admission_wave_requests: max_forward_requests.max(1),
            admission_drain_barrier: false,
            sched_counters: SchedCounters::default(),
            next_driver_rr: 0,
        }
    }

    /// Comprehensive GPU page audit: accounts for every allocated page.
    /// free + trie_pages + working_active + working_pinned = total (if balanced).
    /// Any gap = leaked pages.
    #[allow(dead_code, unused_variables)]
    pub(crate) fn page_audit(&self) {
        for (driver_idx, dev) in self.gpu_stores.iter().enumerate() {
            let free = dev.available();
            let total = dev.total_pages();
            let (trie_pages, trie_rc, trie_nodes, rc0_interior) = dev.trie_stats();

            let mut working_active = 0usize;
            let mut working_pinned = 0usize;
            let mut working_suspended = 0usize; // should be 0 (suspended = CPU)
            let mut committed_active = 0usize; // committed hashes held by active contexts
            let mut committed_pinned = 0usize;
            let mut n_active = 0usize;
            let mut n_pinned = 0usize;
            let mut n_suspended = 0usize;

            for ctx in self.contexts.values() {
                let ctx_driver = ctx.driver.unwrap_or(0) as usize;
                if ctx_driver != driver_idx {
                    continue;
                }
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
            let leaked = if total > accounted {
                total - accounted
            } else {
                0
            };
            let over = if accounted > total {
                accounted - total
            } else {
                0
            };
            let (alloc_tot, free_tot) = dev.pool_stats();
            let (f_reclaim, f_release, f_working) = dev.tag_stats();

            // tracing::debug!(
            //     "[PAGE_AUDIT] dev={} total={} | free={} trie={} work_active={} work_pinned={} | accounted={} LEAKED={} OVER={} | \\
            //      pool: alloc={} free={} net={} | tags: reclaim={} release={} working={} | \\
            //      ctxs: active={} pinned={} suspended={} | commit_active={} commit_pinned={} | \\
            //      trie: rc={} nodes={} rc0_int={} | rq={} aq={}",
            //     driver_idx, total, free, trie_pages, working_active, working_pinned,
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
                    tracing::error!(
                        "[PHANTOM_AUDIT] dev={} phantom_count={} pages=[{}]",
                        driver_idx,
                        phantom_count,
                        phantom_desc
                    );
                }
            }
        }
    }

    fn next_id(&mut self) -> ContextId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Pick a driver for a brand-new context.
    ///
    /// `available()` reflects only *allocated* pages, not pages claimed by
    /// concurrently-arriving contexts that haven't allocated yet. For a
    /// burst of N contexts admitted back-to-back (e.g. a benchmark firing
    /// `launch_process` async tasks in parallel), every call sees the same
    /// `available()` and `max_by_key` deterministically returns the same
    /// driver — collapsing DP to a single driver. We tiebreak with a
    /// round-robin counter so bursty arrivals spread across drivers.
    fn least_loaded_driver(&mut self) -> usize {
        // Find the maximum `available()` and tally how many drivers share it.
        let max_avail = self
            .gpu_stores
            .iter()
            .map(|d| d.available())
            .max()
            .unwrap_or(0);
        let tied: Vec<usize> = self
            .gpu_stores
            .iter()
            .enumerate()
            .filter(|(_, d)| d.available() == max_avail)
            .map(|(i, _)| i)
            .collect();
        if tied.is_empty() {
            return 0;
        }
        if tied.len() == 1 {
            return tied[0];
        }
        // On ties, round-robin among the tied drivers.
        let pick = tied[self.next_driver_rr % tied.len()];
        self.next_driver_rr = self.next_driver_rr.wrapping_add(1);
        pick
    }

    fn driver_uses_rs_cache(&self, driver_idx: usize) -> bool {
        self.rs_stores
            .get(driver_idx)
            .map(|s| s.total_slots() > 0)
            .unwrap_or(false)
    }

    fn initial_rs_state(&self, driver_idx: usize) -> RsState {
        if self.driver_uses_rs_cache(driver_idx) {
            RsState::Empty
        } else {
            RsState::Unsupported
        }
    }

    fn context_token_len(&self, ctx: &Context) -> usize {
        ctx.committed_len() * self.page_size + ctx.working_page_tokens.len()
    }

    fn alloc_rs_slot_with_eviction(
        &mut self,
        _ctx_id: ContextId,
        driver_idx: usize,
    ) -> Result<RsSlotAlloc> {
        if let Some(slot) = self.rs_stores[driver_idx].alloc() {
            return Ok(RsSlotAlloc::Ready(slot));
        }

        // RS state is compact but not paged/content-addressed like KV. Under
        // slot-only pressure, preempting an otherwise resident recurrent state
        // forces full replay and can perturb generation. Keep the slot lease
        // non-preemptive and let waiters resume when a context is destroyed or
        // suspended by the normal KV/page contention path.
        if self.rs_stores[driver_idx].total_slots() > 0 {
            return Ok(RsSlotAlloc::Deferred);
        }

        anyhow::bail!(
            "rs_cache exhausted on driver {driver_idx}: no evictable recurrent-state slot"
        );
    }

    fn alloc_rs_slot_now_with_eviction(
        &mut self,
        ctx_id: ContextId,
        driver_idx: usize,
    ) -> Result<RsSlotId> {
        match self.alloc_rs_slot_with_eviction(ctx_id, driver_idx)? {
            RsSlotAlloc::Ready(slot) => Ok(slot),
            RsSlotAlloc::Deferred => {
                anyhow::bail!("rs_cache slot unavailable until pinned contexts unpin")
            }
        }
    }

    fn release_rs_slot_for_context(&mut self, ctx_id: ContextId) {
        let (driver_idx, state, has_state) = match self.contexts.get(&ctx_id) {
            Some(ctx) => (
                ctx.driver.unwrap_or(0) as usize,
                ctx.rs_state,
                self.context_token_len(ctx) > 0,
            ),
            None => return,
        };
        if let Some(slot) = state.resident_slot() {
            if let Some(store) = self.rs_stores.get_mut(driver_idx) {
                store.free(slot);
            }
        }
        let uses_rs_cache = self.driver_uses_rs_cache(driver_idx);
        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            ctx.rs_state = if !uses_rs_cache {
                RsState::Unsupported
            } else if has_state {
                RsState::Missing
            } else {
                RsState::Empty
            };
        }
    }

    fn ensure_rs_slot_for_pin(
        &mut self,
        ctx_id: ContextId,
        driver_idx: usize,
    ) -> Result<RsPinSlot> {
        if !self.driver_uses_rs_cache(driver_idx) {
            return Ok(RsPinSlot::Ready(None, 0));
        }

        let (state, token_len) = {
            let ctx = self
                .contexts
                .get(&ctx_id)
                .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
            (ctx.rs_state, self.context_token_len(ctx))
        };
        match state {
            RsState::Unsupported => return Ok(RsPinSlot::Ready(None, 0)),
            RsState::Resident(slot) => return Ok(RsPinSlot::Ready(Some(slot), 0)),
            RsState::Missing if token_len > 0 => {
                anyhow::bail!(
                    "pin: context {ctx_id} has missing rs_cache state; restore must replay it first"
                );
            }
            RsState::Missing | RsState::Empty => {}
        }

        let slot = match self.alloc_rs_slot_with_eviction(ctx_id, driver_idx)? {
            RsSlotAlloc::Ready(slot) => slot,
            RsSlotAlloc::Deferred => return Ok(RsPinSlot::Deferred),
        };
        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            ctx.rs_state = RsState::Resident(slot);
        }
        Ok(RsPinSlot::Ready(Some(slot), RS_FLAG_RESET))
    }

    // ==================== Core Operations ====================

    pub(crate) fn create(&mut self, owner: ProcessId) -> Result<ContextId> {
        let id = self.next_id();
        let mut ctx = Context::new(Some(owner));
        let driver_idx = self.least_loaded_driver();
        ctx.driver = Some(driver_idx);
        ctx.rs_state = self.initial_rs_state(driver_idx);

        self.contexts.insert(id, ctx);

        let proc = self.process_entry(owner);
        proc.context_ids.push(id);

        self.publish_context_counts(id);
        Ok(id)
    }

    pub(crate) fn destroy(&mut self, id: ContextId) -> Result<()> {
        let ctx = self
            .contexts
            .remove(&id)
            .ok_or_else(|| anyhow::anyhow!("Context {id} not found"))?;

        let driver_idx = ctx.driver.unwrap_or(0) as usize;

        if let Some(pid) = ctx.owner {
            if let Some(proc) = self.processes.get_mut(&pid) {
                proc.context_ids.retain(|&c| c != id);
            }
        }

        // Drop this context from alloc_queue.
        self.alloc_queue.retain(|&ctx_id| ctx_id != id);

        // restore_queue: lazy deletion — stale entries filtered on pop in drain_queues.

        if let Some(slot) = ctx.rs_state.resident_slot() {
            if let Some(store) = self.rs_stores.get_mut(driver_idx) {
                store.free(slot);
            }
        }

        // Release committed chain (skip if already released during suspension)
        if !ctx.committed_hashes.is_empty() && !ctx.is_off_gpu() {
            self.gpu_stores[driver_idx].release(&ctx.committed_hashes);
        }

        // Free working pages (GPU when Active/Pinned; empty when Suspended).
        if !ctx.working_pages.is_empty() {
            self.gpu_stores[driver_idx].free(&ctx.working_pages);
        }

        // Free CPU working pages stash (if suspended with CPU stash).
        if !ctx.cpu_working_pages.is_empty() {
            self.cpu_stores[driver_idx].free(&ctx.cpu_working_pages);
        }

        // Release CPU-resident committed pages (suspended contexts may have
        // had their committed pages stashed to CPU via would_free + cpu.insert).
        if ctx.is_off_gpu() && !ctx.committed_hashes.is_empty() {
            self.cpu_stores[driver_idx].release(&ctx.committed_hashes);
        }

        self.snapshots.retain(|_, v| *v != id);

        self.remove_context_caches(id);
        self.drain_queues();
        Ok(())
    }

    // ==================== Page Management ====================

    /// Idempotent allocator: ensure ctx has at least `target` visible
    /// working pages. Three cases:
    ///
    /// 1. `target <= visible`: no-op.
    /// 2. `target <= visible + predicted_extra`: promote
    ///    `target - visible` pages from predicted-extra into visible.
    /// 3. Otherwise: promote all predicted-extra, then allocate the
    ///    remainder from the device pool (may evict / defer).
    ///
    /// See SPECULATIVE_EXECUTION_DESIGN.md §"Promotion: how
    /// predicted_extra → visible".
    pub(crate) fn ensure_working_pages(
        &mut self,
        id: ContextId,
        target: usize,
        response: oneshot::Sender<anyhow::Result<()>>,
    ) {
        let ctx = match self.contexts.get(&id) {
            Some(c) => c,
            None => {
                let _ = response.send(Err(anyhow::anyhow!("Context not found")));
                return;
            }
        };
        let physical = ctx.working_pages.len();

        if target <= physical {
            let _ = response.send(Ok(()));
            return;
        }

        let needed = target - physical;
        let driver_idx = ctx.driver.unwrap_or(0) as usize;

        self.when_allocated(id, driver_idx, needed, move |mgr, pages| {
            if let Some(ctx) = mgr.contexts.get_mut(&id) {
                ctx.working_pages.extend_from_slice(&pages);
                ctx.driver = Some(driver_idx);
            }
            mgr.publish_context_counts(id);
            let _ = response.send(Ok(()));
        });
    }

    pub(crate) fn release_working_pages(&mut self, id: ContextId, num_pages: usize) -> Result<()> {
        let ctx = self
            .contexts
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        if num_pages == 0 {
            return Ok(());
        }
        if num_pages > ctx.working_pages.len() {
            anyhow::bail!(
                "release: requested {num_pages}, have {}",
                ctx.working_pages.len()
            );
        }

        let driver_idx = ctx.driver.unwrap_or(0) as usize;
        let start = ctx.working_pages.len() - num_pages;
        let to_free: Vec<_> = ctx.working_pages.drain(start..).collect();

        // Truncate working page tokens
        let tokens_to_remove = num_pages * self.page_size;
        let len = ctx.working_page_tokens.len();
        ctx.working_page_tokens
            .truncate(len.saturating_sub(tokens_to_remove));

        if !ctx.is_off_gpu() {
            self.gpu_stores[driver_idx].free(&to_free);
        }
        self.publish_context_counts(id);
        self.drain_queues();
        Ok(())
    }

    pub(crate) fn commit_working_pages(&mut self, id: ContextId, num_pages: usize) -> Result<()> {
        let page_size = self.page_size;
        let ctx = self
            .contexts
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        if num_pages == 0 {
            return Ok(());
        }

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
            anyhow::bail!(
                "commit: need {total_tokens} tokens, have {}",
                ctx.working_page_tokens.len()
            );
        }

        let driver_idx = ctx.driver.unwrap_or(0) as usize;
        let prev_hash = ctx.committed_tip().unwrap_or(0);
        let pages = ctx
            .working_pages
            .get(..num_pages)
            .map(|s| s.to_vec())
            .unwrap_or_default();

        // Extract token data for the pages being committed.
        let committed_token_infos = ctx.working_page_tokens[..total_tokens].to_vec();
        let mut tokens = Vec::with_capacity(total_tokens);
        let mut positions = Vec::with_capacity(total_tokens);
        let mut masks = Vec::with_capacity(total_tokens);
        for info in &committed_token_infos {
            tokens.push(info.token);
            positions.push(info.position);
            masks.push(materialize_lineage_mask(&info.mask, info.position));
        }

        // Validate positions are strictly after any previously committed position.
        if let Some(max_committed) = ctx.max_committed_position {
            for &pos in &positions {
                if pos <= max_committed {
                    anyhow::bail!(
                        "Position {} must be > max committed position {}",
                        pos,
                        max_committed
                    );
                }
            }
        }
        let lineage_adapter_seed = committed_token_infos.first().and_then(|t| t.adapter_seed);

        // Compute content-based hashes (includes adapter_seed so ZO-perturbed
        // pages are not shared with unperturbed or differently-seeded pages).
        let hashes = pagestore::compute_page_hashes(
            page_size,
            &tokens,
            &positions,
            &masks,
            prev_hash,
            lineage_adapter_seed,
        );
        let existing_prefix = ctx.committed_hashes.clone();
        let dev = &mut self.gpu_stores[driver_idx];

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
        let ctx = self
            .contexts
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context lost during commit"))?;
        if ctx.is_off_gpu() {
            ctx.suspended_working_count = ctx.suspended_working_count.saturating_sub(num_pages);
        } else {
            ctx.working_pages.drain(..num_pages);
        }
        ctx.working_page_tokens.drain(..total_tokens);
        ctx.committed_hashes.extend_from_slice(&hashes);
        ctx.max_committed_position = positions
            .iter()
            .copied()
            .max()
            .or(ctx.max_committed_position);

        // Refresh cached effective_pages after chain extension.
        if !ctx.is_off_gpu() {
            ctx.cached_effective_pages =
                self.gpu_stores[driver_idx].effective_pages(&ctx.committed_hashes);
        }

        // KV-only replay can merge adjacent fills. Recurrent-state replay
        // preserves original forward boundaries to rebuild the same state path.
        let preserve_forward_boundaries = ctx.rs_state != RsState::Unsupported;
        append_token_infos_to_lineage(
            &mut ctx.lineage,
            &committed_token_infos,
            preserve_forward_boundaries,
        );

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
        let mut response = Some(response);
        self.when_active(id, move |mgr| {
            let result = (|| -> Result<Option<PinnedContext>> {
                // Token-budget gate: refuse to pin for a forward pass when
                // the owning process has exhausted its compute budget. The
                // actual debit happens on append_working_page_tokens after
                // the pass completes; this pre-check prevents overdraft.
                if num_input_tokens > 0 && !mgr.has_token_budget(id, num_input_tokens as usize) {
                    anyhow::bail!(
                        "pin: token budget exhausted for context {id} \
                         (requested {num_input_tokens} tokens)"
                    );
                }

                let driver_idx = {
                    let ctx = mgr
                        .contexts
                        .get(&id)
                        .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
                    if ctx.is_off_gpu() {
                        anyhow::bail!("pin: context is suspended (cannot pin)");
                    }
                    ctx.driver.unwrap_or(0) as usize
                };
                let (rs_slot, rs_flags) = match mgr.ensure_rs_slot_for_pin(id, driver_idx)? {
                    RsPinSlot::Ready(slot, flags) => (slot, flags),
                    RsPinSlot::Deferred => {
                        let pending_response = response
                            .take()
                            .ok_or_else(|| anyhow::anyhow!("pin response already consumed"))?;
                        let pending = PendingAlloc {
                            driver: driver_idx,
                            num_pages: 0,
                            needs_rs_slot: true,
                            on_alloc: Box::new(move |mgr, _pages| {
                                mgr.pin(id, num_input_tokens, pending_response);
                            }),
                        };
                        if let Some(ctx) = mgr.contexts.get_mut(&id) {
                            ctx.deferred_ops.push(pending);
                        }
                        mgr.alloc_queue.push_back(id);
                        return Ok(None);
                    }
                };
                let ctx = mgr
                    .contexts
                    .get_mut(&id)
                    .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
                let committed_hashes = ctx.committed_hashes.clone();
                let working = ctx.working_pages.clone();
                let kv_len =
                    (ctx.committed_len() * mgr.page_size + ctx.working_page_tokens.len()) as u32;
                ctx.state = State::Pinned;
                PINNED_COUNTS[driver_idx].fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                let mut committed_pages = Vec::new();
                if !committed_hashes.is_empty() {
                    committed_pages
                        .extend(mgr.gpu_stores[driver_idx].physical_ids(&committed_hashes));
                }

                let page_size = mgr.page_size as u32;
                let total_kv = kv_len + num_input_tokens;
                // Active page count: how many pages are needed to cover
                // `total_kv` tokens. The rest of `working` is held in
                // reserve for the speculator's chain extender.
                let active_pages_total = total_kv.div_ceil(page_size) as usize;
                let active_working = active_pages_total
                    .saturating_sub(committed_pages.len())
                    .min(working.len());
                let mut pages = committed_pages;
                pages.extend(working[..active_working].iter().copied());
                let extra_pages: Vec<PhysicalPageId> = working[active_working..].to_vec();

                let num_pages = pages.len() as u32;
                let last_page_len =
                    pagestore::compute_last_page_len(total_kv, num_pages, page_size);
                Ok(Some(PinnedContext {
                    driver: driver_idx as DriverId,
                    pages,
                    extra_pages,
                    kv_len,
                    last_page_len,
                    rs_slot,
                    rs_flags,
                }))
            })();
            if let Some(response) = response.take() {
                match result {
                    Ok(Some(pinned)) => {
                        let _ = response.send(Ok(pinned));
                    }
                    Ok(None) => {}
                    Err(err) => {
                        let _ = response.send(Err(err));
                    }
                }
            }
        });
    }

    /// Handle Unpin message: Pinned → Active, then check deferred suspension.
    /// If `pending_suspend` was set (and this isn't a replay context),
    /// executes the deferred suspension and enqueues for restoration.
    pub(crate) fn unpin(&mut self, id: ContextId) {
        let (pending, replay, dev) = match self.contexts.get(&id) {
            Some(ctx) if ctx.is_pinned() => (
                ctx.pending_suspend,
                ctx.pending_replay,
                ctx.driver.unwrap_or(0) as usize,
            ),
            _ => return,
        };

        // Decrement real-time pinned counter (all paths leaving Pinned).
        if dev < PINNED_COUNTS.len() {
            PINNED_COUNTS[dev].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }

        // If this is a replay context, replay_complete handles the transition.
        if replay {
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
        self.fire_deferred_ops(id);
        self.drain_queues();
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
            let front_op = match self
                .contexts
                .get(&front_ctx_id)
                .and_then(|c| c.deferred_ops.first())
            {
                Some(op) => op,
                None => {
                    self.alloc_queue.pop_front();
                    continue;
                }
            };
            let (driver_idx, n, needs_rs_slot) =
                (front_op.driver, front_op.num_pages, front_op.needs_rs_slot);
            if n > 0 && self.gpu_stores[driver_idx].available() < n {
                break;
            }
            if needs_rs_slot && self.rs_stores[driver_idx].available() == 0 {
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

        // Hard admission control: don't restore when any driver is near
        // page capacity. This prevents over-admission that causes the
        // thrashing cascade (evict→restore→re-evict cycle). Processes
        // wait here until running processes complete and free pages,
        // naturally dropping utilization below the threshold.
        let over_capacity = self.gpu_stores.iter().any(|d| {
            let (used, total) = d.stats();
            let utilization = used as f64 / total.max(1) as f64;
            utilization > self.restore_pause_at_utilization
        });
        if over_capacity {
            return;
        }

        let num_drivers = self.gpu_stores.len();
        let max_rejections = self.restore_queue.len();
        let mut rejections = 0;
        let mut re_enqueue = Vec::new();

        while let Some(entry) = self.restore_queue.pop() {
            let ctx_id = entry.ctx_id;

            // Lazy deletion: skip stale entries (destroyed or already restored).
            let is_off_gpu = self
                .contexts
                .get(&ctx_id)
                .map(|c| c.is_off_gpu())
                .unwrap_or(false);
            if !is_off_gpu {
                continue;
            }

            // Per-restore placement evaluation (§4.3): check if a different
            // driver would be cheaper before restoring.
            if num_drivers > 1 {
                if let Some(ctx) = self.contexts.get(&ctx_id) {
                    let best = self.best_driver_for(ctx);
                    if best != ctx.driver.unwrap_or(0) as usize {
                        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                            ctx.driver = Some(best);
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

    /// Publish the per-context counts + driver to the global cache.
    /// Called after any mutation that changes working pages, committed pages,
    /// working page tokens, or driver assignment.
    pub(crate) fn publish_context_counts(&self, id: ContextId) {
        if let Some(ctx) = self.contexts.get(&id) {
            CACHED_CONTEXT_INFO.insert(
                (self.model_idx, id),
                CachedContextInfo {
                    driver: ctx.driver.unwrap_or(0) as usize,
                    working_pages: ctx.working_pages.len() as u32,
                    committed_pages: ctx.committed_len() as u32,
                    working_tokens: ctx.working_page_tokens.len() as u32,
                },
            );
        }
    }

    pub(crate) fn publish_working_token_count(&self, id: ContextId) {
        let Some(ctx) = self.contexts.get(&id) else {
            return;
        };
        if let Some(mut info) = CACHED_CONTEXT_INFO.get_mut(&(self.model_idx, id)) {
            info.working_tokens = ctx.working_page_tokens.len() as u32;
        } else {
            self.publish_context_counts(id);
        }
    }

    /// Remove cached entry for a context (on destroy).
    pub(crate) fn remove_context_caches(&self, id: ContextId) {
        CACHED_CONTEXT_INFO.remove(&(self.model_idx, id));
    }

    pub(crate) fn truncate_working_page_tokens(&mut self, id: ContextId, count: u32) -> Result<()> {
        let ctx = self
            .contexts
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let max = ctx.working_page_tokens.len();
        if count as usize > max {
            anyhow::bail!("truncate count {} out of range 0..={}", count, max);
        }
        ctx.working_page_tokens.truncate(count as usize);
        Ok(())
    }

    pub(crate) fn append_working_page_tokens(
        &mut self,
        id: ContextId,
        tokens: Vec<u32>,
        positions: Vec<u32>,
        masks: Vec<Brle>,
        adapter: Option<AdapterId>,
        adapter_seed: Option<i64>,
    ) -> Result<()> {
        let n = tokens.len();
        if positions.len() != n {
            anyhow::bail!("positions length {} != n {}", positions.len(), n);
        }
        if !masks.is_empty() && masks.len() != n {
            anyhow::bail!("masks length {} != n {}", masks.len(), n);
        }
        let ctx = self
            .contexts
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("Context not found"))?;
        let owner = ctx.owner;
        let forward_id = ctx.next_forward_id;
        ctx.next_forward_id = ctx.next_forward_id.wrapping_add(1);
        for (i, token) in tokens.into_iter().enumerate() {
            ctx.working_page_tokens.push(TokenInfo {
                token,
                position: positions[i],
                mask: if masks.is_empty() {
                    Brle::new(0)
                } else {
                    masks[i].clone()
                },
                adapter,
                adapter_seed,
                forward_id,
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
    Lookup {
        username: String,
        name: String,
        response: oneshot::Sender<Result<ContextId>>,
    },
    Create {
        owner: ProcessId,
        response: oneshot::Sender<Result<ContextId>>,
    },
    Save {
        id: ContextId,
        username: String,
        name: Option<String>,
        response: oneshot::Sender<Result<Option<String>>>,
    },
    Delete {
        username: String,
        name: String,
        response: oneshot::Sender<Result<()>>,
    },
    Destroy {
        id: ContextId,
        response: oneshot::Sender<Result<()>>,
    },
    Fork {
        id: ContextId,
        owner: ProcessId,
        response: oneshot::Sender<Result<ContextId>>,
    },
    Take {
        username: String,
        name: String,
        owner: ProcessId,
        response: oneshot::Sender<Result<ContextId>>,
    },
    CommitWorkingPages {
        id: ContextId,
        num_pages: usize,
        response: oneshot::Sender<Result<()>>,
    },
    ReserveWorkingPages {
        id: ContextId,
        num_pages: usize,
        response: oneshot::Sender<Result<()>>,
    },
    ReleaseWorkingPages {
        id: ContextId,
        num_pages: usize,
    },
    Pin {
        id: ContextId,
        num_input_tokens: u32,
        response: oneshot::Sender<Result<PinnedContext>>,
    },
    Unpin {
        id: ContextId,
    },
    ReplayComplete {
        id: ContextId,
        scratch_driver: usize,
        scratch_pages: Vec<PhysicalPageId>,
        registration: Option<ReplayPageRegistration>,
    },
    GetStats {
        response: oneshot::Sender<Vec<(usize, usize)>>,
    },

    // Actor-routed write APIs
    TruncateWorkingPageTokens {
        id: ContextId,
        count: u32,
        response: oneshot::Sender<Result<()>>,
    },
    /// Fire-and-forget lineage append. Errors are logged at the
    /// handler, not propagated — see `context::append_working_page_tokens`.
    AppendWorkingPageTokens {
        id: ContextId,
        tokens: Vec<u32>,
        positions: Vec<u32>,
        masks: Vec<Brle>,
        adapter: Option<AdapterId>,
        adapter_seed: Option<i64>,
    },

    DebugState {
        id: ContextId,
        response: oneshot::Sender<String>,
    },

    RegisterProcess {
        pid: ProcessId,
        token_budget: Option<usize>,
        response: oneshot::Sender<Result<()>>,
    },
    UnregisterProcess {
        pid: ProcessId,
    },

    // ── Market messages ────────────────────────────────────────
    /// Execute one market tick for a single driver.
    /// `batch_ctx_ids` = context IDs from the just-completed batch.
    Tick {
        driver_idx: usize,
        latency_secs: f64,
        batch_ctx_ids: Vec<ContextId>,
    },
    /// Set a context's bid (willingness to pay per page per step).
    Bid {
        id: ContextId,
        bid: f64,
        response: oneshot::Sender<Result<()>>,
    },
    /// Suspend a context (program-initiated).
    Suspend {
        id: ContextId,
        response: oneshot::Sender<Result<()>>,
    },
}

impl ServiceHandler for ContextManager {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        let _t = Instant::now();
        match msg {
            Message::Lookup {
                username,
                name,
                response,
            } => {
                let result = self
                    .snapshots
                    .get(&(username, name))
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("Snapshot not found"));
                let _ = response.send(result);
            }
            Message::Take {
                username,
                name,
                owner,
                response,
            } => {
                self.take(username, name, owner, response);
            }
            Message::Create { owner, response } => {
                let _ = response.send(self.create(owner));
            }
            Message::Save {
                id,
                username,
                name,
                response,
            } => {
                let _ = response.send(self.save(id, username, name));
            }
            Message::Delete {
                username,
                name,
                response,
            } => {
                let _ = response.send(self.delete(username, name));
            }
            Message::Destroy { id, response } => {
                let t0 = Instant::now();
                let _ = response.send(self.destroy(id));
                self.sched_counters.destroy_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.destroy_count += 1;
            }
            Message::Fork {
                id,
                owner,
                response,
            } => {
                self.fork(id, owner, response);
            }
            Message::CommitWorkingPages {
                id,
                num_pages,
                response,
            } => {
                let t0 = Instant::now();
                let _ = response.send(self.commit_working_pages(id, num_pages));
                self.sched_counters.commit_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.commit_count += 1;
            }
            Message::ReserveWorkingPages {
                id,
                num_pages,
                response,
            } => {
                let t0 = Instant::now();
                // WIT semantics: "reserve N additional pages." Dispatch
                // through `ensure_working_pages` with target =
                // physical + N.
                let target = match self.contexts.get(&id) {
                    Some(ctx) => ctx.working_pages.len() + num_pages,
                    None => {
                        let _ = response.send(Err(anyhow::anyhow!("Context not found")));
                        self.sched_counters.reserve_us += t0.elapsed().as_micros() as u64;
                        self.sched_counters.reserve_count += 1;
                        return;
                    }
                };
                self.ensure_working_pages(id, target, response);
                self.sched_counters.reserve_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.reserve_count += 1;
            }
            Message::ReleaseWorkingPages { id, num_pages } => {
                let t0 = Instant::now();
                let _ = self.release_working_pages(id, num_pages);
                self.sched_counters.release_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.release_count += 1;
            }
            Message::Pin {
                id,
                num_input_tokens,
                response,
            } => {
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
            Message::ReplayComplete {
                id,
                scratch_driver,
                scratch_pages,
                registration,
            } => {
                let t0 = Instant::now();
                if !scratch_pages.is_empty() {
                    if let Some(store) = self.gpu_stores.get_mut(scratch_driver) {
                        store.free(&scratch_pages);
                    }
                }
                if let Some(registration) = registration {
                    if self.contexts.contains_key(&id) {
                        if !registration.hashes.is_empty() {
                            self.gpu_stores[registration.driver].extend(
                                &registration.prefix,
                                &registration.hashes,
                                &registration.pages,
                            );
                        }
                    } else if let Some(store) = self.gpu_stores.get_mut(registration.driver) {
                        store.free(&registration.pages);
                    }
                }
                self.replay_complete(id);
                self.sched_counters.replay_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.replay_count += 1;
            }
            Message::GetStats { response } => {
                let _ = response.send(self.stats());
            }
            Message::TruncateWorkingPageTokens {
                id,
                count,
                response,
            } => {
                let _ = response.send(self.truncate_working_page_tokens(id, count));
                self.publish_context_counts(id);
            }
            Message::AppendWorkingPageTokens {
                id,
                tokens,
                positions,
                masks,
                adapter,
                adapter_seed,
            } => {
                let t0 = Instant::now();
                if let Err(e) = self.append_working_page_tokens(
                    id,
                    tokens,
                    positions,
                    masks,
                    adapter,
                    adapter_seed,
                ) {
                    tracing::warn!("append_working_page_tokens for ctx {id}: {e:#}");
                }
                self.publish_working_token_count(id);
                self.sched_counters.append_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.append_count += 1;
            }
            Message::DebugState { id, response } => {
                let _ = response.send(self.debug_state(id));
            }
            Message::RegisterProcess {
                pid,
                token_budget,
                response,
            } => {
                let t0 = Instant::now();
                let result = self.register_process(pid, token_budget);
                let _ = response.send(result);
                self.sched_counters.register_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.register_count += 1;
            }
            Message::UnregisterProcess { pid } => {
                let t0 = Instant::now();
                self.unregister_process(pid);
                ADMISSION_NOTIFY.notify_waiters();
                self.sched_counters.unregister_us += t0.elapsed().as_micros() as u64;
                self.sched_counters.unregister_count += 1;
            }

            // ── Market handlers ────────────────────────────────────
            Message::Tick {
                driver_idx,
                latency_secs,
                batch_ctx_ids,
            } => {
                let t0 = Instant::now();
                self.tick(driver_idx, latency_secs, &batch_ctx_ids);
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
