//! Scheduling — Market Tick, Contention, and Eviction.
//!
//! This module handles all scheduling economics and GPU page contention:
//!
//! - **Market tick**: Per-device price discovery (clearing price = min bid),
//!   Shapley cost-sharing, rent payments, default flagging, and dividends.
//! - **Contention**: `when_allocated` — the universal GPU page contention
//!   primitive. Handles free-pool allocation, eviction loops, priority gates,
//!   and deferred-op stashing for suspended contexts.
//! - **Eviction**: `find_eviction_victim` — selects the best eviction target
//!   based on `(defaulted DESC, bid ASC, spawn_time DESC)`.
//! - **Suspension**: `suspend` — stashes pages to CPU, releases GPU refcounts.
//! - **Queue draining**: `drain_queues` — serves `alloc_queue` (FIFO) and
//!   `restore_queue` (highest bid) as pages become available.
//!
//! ## Shapley Cost-Sharing
//!
//! Contexts sharing a KV-cache prefix split its physical cost equally via
//! Shapley values. `effective_pages(i)` = unique pages + Σ shared_segment_pages / refcount.
//! The auction uses effective pages for packing, payment, and sorting.
//!
//! ## Conservation Invariant
//!
//! At all times: `Σ balance_i = Σ endowment_i − M(t)`, where M(t) is cumulative
//! make cost. Rent revenue is exactly redistributed as dividends.

use std::cmp::Ordering;
use std::fmt;
use std::time::Instant;

use crate::device::{self, DeviceId};
use crate::process::ProcessId;

use super::pagestore::PhysicalPageId;
use super::{Context, ContextId, ContextManager, RestoreEntry, State, MARKET};

// =============================================================================
// ProcessEntry — Wallet + Ownership
// =============================================================================

/// Per-process state. Two disjoint wallets + ownership record.
///
/// **Token wallet (`tokens_remaining`)**: optional compute/billing cap.
/// `None` = no cap (default), the process may compute indefinitely.
/// `Some(n)` = monotonically destructive cap; decremented per successful
/// forward pass, and when it saturates to zero further passes are rejected.
/// No market semantics in either case.
///
/// **Credit wallet (`balance`)**: market/scheduling currency.
/// Conserved — flows between processes via rent payments and endowment-
/// weighted dividends. Never created or destroyed after admission.
/// Drives auction allocation, never bounds compute.
///
/// The two wallets never exchange. Exhausting credits evicts the process
/// from GPU but does not terminate it. Exhausting tokens terminates forward
/// progress but does not affect market standing.
#[derive(Debug)]
pub(crate) struct ProcessEntry {
    /// Credit balance (market wallet, unit: pages). Initialized to `endowment`
    /// at admission; drifts via rent payments and dividends. Conserved.
    pub balance: f64,
    /// Remaining compute budget (token wallet, unit: tokens).
    /// `None` = unlimited (no cap); `Some(n)` = hard cap that decrements
    /// on forward-pass completion. Saturates to `Some(0)`, never wraps.
    pub tokens_remaining: Option<usize>,
    /// Context IDs owned by this process.
    pub context_ids: Vec<ContextId>,
    /// Birth timestamp — used as FCFS tiebreaker at equal bid.
    pub created_at: Instant,
    /// Share of long-run GPU capacity this process is entitled to (unit: pages).
    /// Fixed at admission. One endowment unit = one KV page of long-run
    /// residency under contention. Used to weight dividend distribution.
    pub endowment: f64,
}

impl ProcessEntry {
    pub(crate) fn new() -> Self {
        ProcessEntry {
            balance: 0.0,
            tokens_remaining: None,
            context_ids: Vec::new(),
            created_at: Instant::now(),
            endowment: 0.0,
        }
    }
}

// =============================================================================
// Price Curves
// =============================================================================



// =============================================================================
// AuctionResult
// =============================================================================

/// Per-device auction outcome, computed each tick.
#[derive(Debug, Clone, Default)]
pub(crate) struct AuctionResult {
    /// GPU clearing price: the marginal context's bid (highest excluded bid).
    /// Zero when device is not fully packed.
    pub clearing_price: f64,
    /// CPU clearing price: for the CPU-tier cache auction.
    pub cpu_clearing_price: f64,
    /// Total revenue collected from critical-value payments this step (GPU + CPU).
    pub total_revenue: f64,
    /// Dividend rate for this device: device_revenue / total_endowment.
    /// Multiplied by a process's endowment to get its per-device dividend.
    pub dividend_per_endowment: f64,
}

// =============================================================================
// PendingAlloc — Deferred GPU Page Operation
// =============================================================================

/// A deferred GPU page operation stored on `ctx.deferred_ops`.
///
/// On success, `on_alloc` is called with pre-allocated pages.
/// On cancellation (destroy), the struct is dropped — dropping the closure
/// drops the captured `oneshot::Sender`, closing the channel.
pub(crate) struct PendingAlloc {
    pub device: usize,
    pub num_pages: usize,
    /// Callback invoked with allocated pages on success.
    pub on_alloc: Box<dyn FnOnce(&mut ContextManager, Vec<PhysicalPageId>) + Send>,
}

impl fmt::Debug for PendingAlloc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingAlloc")
            .field("device", &self.device)
            .field("num_pages", &self.num_pages)
            .field("on_alloc", &"<closure>")
            .finish()
    }
}

// =============================================================================
// Process Lifecycle
// =============================================================================

impl ContextManager {

    /// Explicitly register a process with its token budget.
    /// Creates a `ProcessEntry` with the correct endowment.
    /// Called once per process lifetime (from `InstanceState::new`).
    ///
    /// Panics on double-registration (indicates a bug in the caller).
    pub(crate) fn register_process(
        &mut self, pid: ProcessId, token_budget: Option<usize>
    ) -> anyhow::Result<()> {
        assert!(
            !self.processes.contains_key(&pid),
            "register_process: process {pid} already registered"
        );
        // Two wallets, derived independently.
        //
        // Token wallet: Some(n) caps compute at n tokens; None = uncapped.
        //   - If the caller passes Some, use it verbatim (explicit opt-in).
        //   - If the caller passes None, inherit the config default, which
        //     itself is Option<usize> (None = unlimited by default).
        //
        // Credit wallet (balance/endowment): pages entitled under long-run
        // contention. Derived from the explicit token cap when present
        // (⌈T / page_size⌉), or from the configured default endowment when
        // the cap is unlimited.
        let tokens_remaining = token_budget.or(self.default_tokens_remaining);
        let endowment_pages = match token_budget {
            Some(t) => t.div_ceil(self.page_size.max(1)) as f64,
            None => self.default_endowment,
        };

        // Admission gate: Σ endowment ≤ total_capacity × oversubscription_factor.
        // Each endowment unit is a claim on one page of long-run GPU residency;
        // selling more than capacity × factor would overcommit beyond what
        // duty-cycle averaging can absorb.
        let sigma_e: f64 = self.processes.values().map(|p| p.endowment).sum();
        let total_capacity: f64 = self.gpu_stores.iter()
            .map(|s| s.total_pages() as f64).sum();
        let cap = total_capacity * self.oversubscription_factor;
        if sigma_e + endowment_pages > cap {
            anyhow::bail!(
                "admission denied: Σ endowment ({sigma_e} + {endowment_pages} = \
                 {}) would exceed capacity × factor ({total_capacity} × \
                 {} = {cap})",
                sigma_e + endowment_pages,
                self.oversubscription_factor,
            );
        }

        let mut entry = ProcessEntry::new();
        entry.tokens_remaining = tokens_remaining;
        entry.balance = endowment_pages;
        entry.endowment = endowment_pages;
        self.processes.insert(pid, entry);
        if let Some(market) = MARKET.get(self.model_idx) {
            market.balances.insert(pid, endowment_pages);
            market.endowments.insert(pid, endowment_pages);
            market.tokens_remaining.insert(pid, tokens_remaining);
        }
        Ok(())
    }

    /// Unregister a process: destroy all owned contexts and remove the process entry.
    /// Called on WASM instance drop for automatic cleanup.
    pub(crate) fn unregister_process(&mut self, pid: ProcessId) {
        let t_start = Instant::now();
        let proc = match self.processes.remove(&pid) {
            Some(p) => p,
            None => return,
        };

        // Drop this process's contexts from alloc_queue.
        let ctx_ids: std::collections::HashSet<ContextId> = proc.context_ids.iter().copied().collect();
        self.alloc_queue.retain(|ctx_id| !ctx_ids.contains(ctx_id));

        // restore_queue: lazy deletion — stale entries filtered on pop in drain_queues.

        let t_queues = t_start.elapsed();

        // Destroy all owned contexts
        for ctx_id in &proc.context_ids {
            if let Some(ctx) = self.contexts.remove(ctx_id) {
                let dev_idx = ctx.device.unwrap_or(0) as usize;
                if !ctx.committed_hashes.is_empty() && !ctx.is_off_gpu() {
                    self.gpu_stores[dev_idx].release(&ctx.committed_hashes);
                }
                if !ctx.working_pages.is_empty() {
                    self.gpu_stores[dev_idx].free(&ctx.working_pages);
                }
                if !ctx.cpu_working_pages.is_empty() {
                    self.cpu_stores[dev_idx].free(&ctx.cpu_working_pages);
                }
                if ctx.is_off_gpu() && !ctx.committed_hashes.is_empty() {
                    self.cpu_stores[dev_idx].release(&ctx.committed_hashes);
                }
                self.remove_context_caches(*ctx_id);
            }
        }

        // Single pass: remove all snapshot entries pointing to this process's contexts.
        self.snapshots.retain(|_, v| !ctx_ids.contains(v));

        let t_destroy = t_start.elapsed();

        if let Some(market) = MARKET.get(self.model_idx) {
            market.balances.remove(&pid);
            market.endowments.remove(&pid);
            market.tokens_remaining.remove(&pid);
        }

        let t_pre_drain = t_start.elapsed();
        // Early-exit: skip drain_queues if both queues are empty.
        if !self.restore_queue.is_empty() || !self.alloc_queue.is_empty() {
            self.drain_queues();
        }
        let t_total = t_start.elapsed();

        self.sched_counters.unreg_queues_us += t_queues.as_micros() as u64;
        self.sched_counters.unreg_destroy_us += (t_destroy - t_queues).as_micros() as u64;
        self.sched_counters.unreg_drain_us += (t_total - t_pre_drain).as_micros() as u64;
    }

    /// Get a mutable reference to a registered process's entry.
    /// Panics if the process is not registered — callers must ensure
    /// `register_process()` was called first.
    pub(crate) fn process_entry(&mut self, pid: ProcessId) -> &mut ProcessEntry {
        self.processes.get_mut(&pid)
            .expect("process_entry: process not registered (missing register_process call)")
    }

    // =========================================================================
    // Best Device — Per-Context Shapley Placement (§4.3)
    // =========================================================================

    /// Evaluate the cheapest device for a single context using Shapley costs.
    ///
    /// Computes `effective_pages(ctx, d) × clearing_price(d)` on every device,
    /// accounting for migration cost. Returns the device index with lowest cost.
    ///
    /// Called from `drain_queues` before each restore.
    pub(crate) fn best_device_for(&self, ctx: &Context) -> usize {
        let num_devices = self.gpu_stores.len();
        let current_dev = ctx.device.unwrap_or(0) as usize;
        if num_devices <= 1 { return current_dev; }

        let hashes = &ctx.committed_hashes;
        if hashes.is_empty() { return current_dev; }

        let working_count = if ctx.is_off_gpu() {
            ctx.suspended_working_count as f64
        } else {
            ctx.working_pages.len() as f64
        };

        let mut best_dev = current_dev;
        let mut best_cost = f64::MAX;

        for d in 0..num_devices {
            let shared = self.gpu_stores[d].prefix_len(hashes);
            let shared_eff = if shared > 0 {
                self.gpu_stores[d].effective_pages(&hashes[..shared])
            } else {
                0.0
            };
            let unique_eff = (hashes.len() - shared) as f64 + working_count;
            let eff = shared_eff + unique_eff;
            let price = self.auction_results.get(d)
                .map(|a| a.clearing_price).unwrap_or(0.0);

            let migration_cost = if d != current_dev {
                hashes.len() as f64
            } else {
                0.0
            };

            let total_cost = eff * price + migration_cost;
            if total_cost < best_cost {
                best_cost = total_cost;
                best_dev = d;
            }
        }

        best_dev
    }

    // =========================================================================
    // Market Tick — Price Discovery + Payments + Dividends
    // =========================================================================

    /// Run one market tick for a single device: price discovery + payments + dividends.
    ///
    /// Called once per batch completion on the device that just finished.
    /// Only runs the auction for `dev_idx`, charges payments to contexts on
    /// that device, and distributes dividends proportional to the single-device
    /// revenue.
    ///
    /// No eviction or page movement — those are handled by `when_allocated`
    /// and `drain_queues`.
    pub(crate) fn tick(&mut self, dev_idx: usize, latency_secs: f64, batch_ctx_ids: &[ContextId]) {
        let t_start = Instant::now();

        // Ensure auction_results vec is large enough.
        if self.auction_results.len() <= dev_idx {
            self.auction_results.resize(dev_idx + 1, AuctionResult::default());
        }

        // Build set of in-batch context IDs for O(1) lookup.
        let batch_set: std::collections::HashSet<ContextId> =
            batch_ctx_ids.iter().copied().collect();

        // All contexts on this device are admitted by construction (the
        // contention/eviction system enforces physical capacity).
        // Clearing price = smallest bid on the device when contended; zero otherwise.
        let mut min_bid = f64::MAX;
        let mut n_active = 0usize;
        let mut n_pinned = 0usize;

        // Pass 1: find min bid across all GPU-resident contexts.
        for ctx in self.contexts.values() {
            if ctx.owner.is_none() { continue; }
            if ctx.device.unwrap_or(0) as usize != dev_idx { continue; }
            if ctx.is_off_gpu() { continue; }
            min_bid = min_bid.min(ctx.bid);
            if ctx.is_pinned() { n_pinned += 1; }
            else if ctx.is_active() { n_active += 1; }
        }

        // Contention gate (SCHED.md §3.2): the clearing price is the critical
        // value — the bid at which an admitted context would be displaced.
        // When no one is waiting and the device has free capacity, no context
        // faces displacement pressure, so the critical value is zero.
        //
        // Contended iff: device is full, or a waiter exists (restore/alloc queue).
        // restore_queue / alloc_queue are manager-global, so any waiter anywhere
        // conservatively marks every device contended this tick.
        let device_full = self.gpu_stores[dev_idx].available() == 0;
        let has_waiters = !self.restore_queue.is_empty() || !self.alloc_queue.is_empty();
        let contended = device_full || has_waiters;
        let clearing_price = if contended && min_bid != f64::MAX { min_bid } else { 0.0 };

        let t_pass1 = t_start.elapsed();

        // Pass 2: compute rent for IN-BATCH contexts only.
        // Only contexts that participated in the just-completed batch pay rent.
        // This ensures 1:1 tick:step correspondence assumed by the bid formula.
        // Contexts that are Pinned but NOT in the batch (stale unpins, new pins)
        // are not charged — they didn't consume compute this tick.
        let mut ctx_payments: Vec<(ContextId, ProcessId, f64)> = Vec::new();
        let mut n_charged = 0usize;

        for (&ctx_id, ctx) in &self.contexts {
            if !batch_set.contains(&ctx_id) { continue; }  // Only charge in-batch
            if ctx.owner.is_none() { continue; }

            let raw_pages = ctx.committed_hashes.len() + ctx.working_pages.len();
            if raw_pages == 0 { continue; }

            let eff = ctx.cached_effective_pages
                + ctx.working_pages.len() as f64;

            let payment = clearing_price * eff;
            n_charged += 1;
            if let Some(pid) = ctx.owner {
                ctx_payments.push((ctx_id, pid, payment));
            }
        }

        let t_pass2 = t_start.elapsed();

        // Pass 3: debit rent, summing actual collected amounts.
        // Revenue is the *actual* credits moved, not the nominal owed amount —
        // clamping at the payer's balance would otherwise create credits
        // (dividends distributed > rent collected → balances grow from nothing).
        let mut device_revenue = 0.0f64;
        for (ctx_id, pid, payment) in &ctx_payments {
            if let Some(proc) = self.processes.get_mut(pid) {
                let defaulted = proc.balance < *payment;
                let actual = payment.min(proc.balance);
                proc.balance -= actual;
                device_revenue += actual;
                if let Some(ctx) = self.contexts.get_mut(ctx_id) {
                    if defaulted && !ctx.defaulted {
                        self.sched_counters.defaults_flagged += 1;
                    }
                    ctx.defaulted = defaulted;
                }
            }
        }

        // Endowment-proportional dividends from this device's *actual* revenue.
        let total_endowment: f64 = self.processes.values().map(|p| p.endowment).sum();
        let dividend_rate = if total_endowment > 0.0 {
            device_revenue / total_endowment
        } else {
            0.0
        };

        self.auction_results[dev_idx] = AuctionResult {
            clearing_price,
            cpu_clearing_price: 0.0,
            total_revenue: device_revenue,
            dividend_per_endowment: dividend_rate,
        };

        // Credit dividends to all processes.
        for proc in self.processes.values_mut() {
            proc.balance += dividend_rate * proc.endowment;
        }

        let t_pass3 = t_start.elapsed();

        // Publish market data + balances to lock-free cache.
        // Only publish every 5 ticks to reduce DashMap insert overhead.
        if let Some(market) = MARKET.get(self.model_idx) {
            // Active/pinned/charged counts: always publish (cheap, 3 inserts).
            market.gpu_active.insert(dev_idx, n_active);
            market.gpu_pinned.insert(dev_idx, n_pinned);
            market.gpu_charged.insert(dev_idx, n_charged);

            let should_publish = self.sched_counters.ticks % 5 == 0;
            if should_publish {
                market.clearing_prices.insert(dev_idx, clearing_price);
                // EWA-smooth the tick latency (α = 0.1).
                let alpha = 0.1;
                let prev = market.tick_latency_ewa.get(&dev_idx).map(|v| *v).unwrap_or(latency_secs);
                market.tick_latency_ewa.insert(dev_idx, alpha * latency_secs + (1.0 - alpha) * prev);
                let rate_sum: f64 = self.auction_results.iter()
                    .map(|a| a.dividend_per_endowment).sum();
                market.set_dividend_rate(rate_sum);
                for (&pid, proc) in &self.processes {
                    market.balances.insert(pid, proc.balance);
                }
            }
        }

        let t_publish = t_start.elapsed();

        // Accumulate sub-timing for periodic dump.
        self.sched_counters.tick_pass1_us += t_pass1.as_micros() as u64;
        self.sched_counters.tick_pass2_us += (t_pass2 - t_pass1).as_micros() as u64;
        self.sched_counters.tick_pass3_us += (t_pass3 - t_pass2).as_micros() as u64;
        self.sched_counters.tick_publish_us += (t_publish - t_pass3).as_micros() as u64;
    }

    /// Set a context's bid (willingness to pay per page per step).
    pub(crate) fn bid(&mut self, id: ContextId, bid: f64) -> anyhow::Result<()> {
        match self.contexts.get_mut(&id) {
            Some(ctx) => {
                ctx.bid = bid.max(0.0);
                Ok(())
            }
            None => Err(anyhow::anyhow!("Context {id} not found")),
        }
    }

    // =========================================================================
    // Token Budget (compute wallet)
    // =========================================================================

    /// Debit the token wallet by `num_tokens` on a successful forward pass.
    /// Processes with an unlimited wallet (`None`) are not affected.
    /// For capped wallets, saturates at zero — never wraps.
    pub(crate) fn debit_tokens(&mut self, pid: ProcessId, num_tokens: usize) {
        if num_tokens == 0 { return; }
        if let Some(proc) = self.processes.get_mut(&pid) {
            if let Some(cap) = proc.tokens_remaining.as_mut() {
                *cap = cap.saturating_sub(num_tokens);
                let new_val = *cap;
                if let Some(market) = MARKET.get(self.model_idx) {
                    market.tokens_remaining.insert(pid, Some(new_val));
                }
            }
        }
    }

    /// Check whether the process owning `ctx_id` has at least `num_tokens`
    /// in its token wallet. Returns true when the wallet is unlimited (None)
    /// or when the remaining cap covers the request. Returns false if the
    /// context/process is unknown or the cap is insufficient.
    pub(crate) fn has_token_budget(&self, ctx_id: ContextId, num_tokens: usize) -> bool {
        if num_tokens == 0 { return true; }
        let pid = match self.contexts.get(&ctx_id).and_then(|c| c.owner) {
            Some(pid) => pid,
            None => return true, // snapshots (no owner) bypass budget check
        };
        match self.processes.get(&pid) {
            Some(proc) => match proc.tokens_remaining {
                None => true,                  // unlimited
                Some(rem) => rem >= num_tokens,
            },
            None => false,
        }
    }

    // =========================================================================
    // GPU Page Contention
    // =========================================================================

    /// Pending-aware operation helper (no pages needed).
    ///
    /// If the owning context is Suspended, defers `on_ready` as a zero-page
    /// `PendingAlloc`. Otherwise calls `on_ready` immediately. Used for
    /// operations like `pin` that don't need page allocation but must respect
    /// context suspension.
    pub(crate) fn when_active(
        &mut self,
        ctx_id: ContextId,
        on_ready: impl FnOnce(&mut ContextManager) + Send + 'static,
    ) {
        self.when_allocated(ctx_id, 0, 0, move |mgr, _pages| on_ready(mgr));
    }

    /// Universal GPU page contention primitive.
    ///
    /// Attempts to allocate `num_pages` GPU pages on `dev_idx` for context `ctx_id`.
    /// Goes through: Suspended check → num_pages==0 fast-path →
    /// priority gate → free pool → eviction loop → self-suspend.
    ///
    /// On success, invokes `on_alloc` with the allocated pages.
    /// On deferral, the operation is stashed on `ctx.deferred_ops`
    /// and will be replayed when pages become available.
    pub(crate) fn when_allocated(
        &mut self,
        ctx_id: ContextId,
        dev_idx: usize,
        num_pages: usize,
        on_alloc: impl FnOnce(&mut ContextManager, Vec<PhysicalPageId>) + Send + 'static,
    ) {
        // SUSPENSION CHECK: If the context is Suspended, store as deferred op.
        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            if ctx.is_off_gpu() {
                let pending = PendingAlloc {
                    device: dev_idx, num_pages,
                    on_alloc: Box::new(on_alloc),
                };
                ctx.deferred_ops.push(pending);
                return;
            }
        } else {
            tracing::error!("when_allocated: context not found: {}", ctx_id);
            return;
        }

        // Short-circuit: no pages needed.
        if num_pages == 0 {
            (on_alloc)(self, Vec::new());
            return;
        }

        // Compute requester's bid from the context.
        let requester_bid = self.contexts[&ctx_id].bid;

        // Step 2: PRIORITY GATE — compare requester bid vs restore_queue head.
        // If a higher-bidding Suspended context is waiting, the requester yields.
        if let Some(top_id) = self.highest_bid_in_restore_queue() {
            let top_bid = self.contexts[&top_id].bid;
            if requester_bid < top_bid {
                self.sched_counters.priority_gate_suspends += 1;
                let pending = PendingAlloc {
                    device: dev_idx, num_pages,
                    on_alloc: Box::new(on_alloc),
                };
                self.suspend(ctx_id);
                self.enqueue_restore(ctx_id);
                self.contexts.get_mut(&ctx_id).unwrap().deferred_ops.push(pending);
                self.drain_queues();
                return;
            }
        }

        // Step 3: TRY ALLOCATE from free pool.
        if let Some(pages) = self.gpu_stores[dev_idx].alloc(num_pages) {
            (on_alloc)(self, pages);
            return;
        }

        // Step 4: EVICTION LOOP — with deferred page tracking.
        let mut deferred_pages: usize = 0;
        let mut has_deferred = false;
        let mut alloc_result: Option<Vec<PhysicalPageId>> = None;

        loop {
            match self.find_eviction_victim(dev_idx, requester_bid, Some(ctx_id)) {
                Some(victim_ctx_id) => {
                    self.sched_counters.eviction_suspends += 1;
                    let victim = self.contexts.get(&victim_ctx_id).unwrap();

                    if victim.is_pinned() {
                        // Pinned victim: set pending_suspend, pages freed later on unpin.
                        let reclaimable = victim.working_pages.len()
                            + self.gpu_stores[dev_idx].count_reclaimable(&victim.committed_hashes);
                        self.contexts.get_mut(&victim_ctx_id).unwrap().pending_suspend = true;
                        deferred_pages += reclaimable;
                        has_deferred = true;
                    } else {
                        // Active victim: suspend immediately.
                        self.suspend(victim_ctx_id);
                        self.enqueue_restore(victim_ctx_id);
                    }

                    // Retry alloc after victim suspension freed pages.
                    if let Some(pages) = self.gpu_stores[dev_idx].alloc(num_pages) {
                        alloc_result = Some(pages);
                        break;
                    }

                    if has_deferred {
                        let free_now = self.gpu_stores[dev_idx].available();
                        if free_now + deferred_pages >= num_pages {
                            break;
                        }
                    }
                }
                None => break,
            }
        }

        // Post-loop: handle eviction results.
        if let Some(pages) = alloc_result {
            (on_alloc)(self, pages);
            self.drain_queues();
        } else {
            // No pages available — defer the operation.
            let pending = PendingAlloc {
                device: dev_idx, num_pages,
                on_alloc: Box::new(on_alloc),
            };
            self.contexts.get_mut(&ctx_id).unwrap().deferred_ops.push(pending);

            if has_deferred {
                // Step 5: Deferred pages from Pinned contexts will cover the gap.
                self.alloc_queue.push_back(ctx_id);
            } else {
                // Step 6: NO VICTIM — requester self-suspends.
                self.sched_counters.no_victim_suspends += 1;
                self.suspend(ctx_id);
                self.enqueue_restore(ctx_id);
                self.drain_queues();
            }
        }
    }

    // =========================================================================
    // Eviction
    // =========================================================================

    /// Find the best eviction victim context on a device.
    ///
    /// Priority: defaulted contexts first (regardless of bid), then by
    /// lowest bid, then by most-recently-spawned process (FCFS tiebreaker).
    ///
    /// Non-defaulted victims must have bid ≤ requester_bid.
    /// Defaulted victims are always eligible (they can't pay rent).
    pub(crate) fn find_eviction_victim(
        &self,
        dev_idx: usize,
        requester_bid: f64,
        requester: Option<ContextId>,
    ) -> Option<ContextId> {
        // (defaulted, bid, spawn_time, ctx_id) — best victim has highest
        // defaulted, lowest bid, latest spawn_time.
        let mut best: Option<(bool, f64, Instant, ContextId)> = None;

        for (&ctx_id, ctx) in &self.contexts {
            if requester == Some(ctx_id) { continue; }
            if ctx.is_off_gpu() { continue; }
            let ctx_dev = ctx.device.unwrap_or(0) as usize;
            if ctx_dev != dev_idx { continue; }
            let pages = ctx.committed_hashes.len() + ctx.working_pages.len();
            if pages == 0 { continue; }
            if ctx.pending_suspend { continue; }

            // Non-defaulted contexts must have bid ≤ requester's bid.
            // Defaulted contexts are always evictable (they owe rent).
            if !ctx.defaulted && ctx.bid > requester_bid { continue; }

            let spawn_time = ctx.owner
                .and_then(|pid| self.processes.get(&pid))
                .map(|p| p.created_at)
                .unwrap_or_else(Instant::now);

            let dominated = if let Some((best_def, best_bid, best_time, _)) = best {
                // Prefer defaulted over non-defaulted
                (ctx.defaulted && !best_def)
                // Same default status: prefer lower bid
                || (ctx.defaulted == best_def && ctx.bid < best_bid)
                // Same default + bid: prefer later spawn (FCFS)
                || (ctx.defaulted == best_def && ctx.bid == best_bid && spawn_time > best_time)
            } else {
                true
            };
            if dominated {
                best = Some((ctx.defaulted, ctx.bid, spawn_time, ctx_id));
            }
        }

        best.map(|(_, _, _, ctx_id)| ctx_id)
    }

    /// Helper: enqueue a context for restoration.
    pub(crate) fn enqueue_restore(&mut self, ctx_id: ContextId) {
        let (bid, defaulted) = self.contexts.get(&ctx_id)
            .map(|c| (c.bid, c.defaulted))
            .unwrap_or((0.0, true));
        self.restore_queue.push(RestoreEntry { ctx_id, bid, defaulted });
    }

    /// Peek at the highest-bid context in the restore queue.
    /// Returns None if the queue is empty.
    /// O(1) with BinaryHeap vs previous O(N) scan.
    pub(crate) fn highest_bid_in_restore_queue(&self) -> Option<ContextId> {
        self.restore_queue.peek().map(|e| e.ctx_id)
    }

    // =========================================================================
    // CPU Eviction — tier-boundary contention (STORAGE.md §4.1b)
    // =========================================================================

    /// Find the best CPU eviction victim on a device.
    ///
    /// Iterates all **Stashed** contexts (CPU-resident pages) on the device.
    /// Returns the context with the lowest bid, using FCFS (latest spawn
    /// time first) as tiebreaker.
    ///
    /// The requester is excluded. Only victims with bid ≤ `requester_bid`
    /// are eligible (a higher-bid context should not be evicted to make
    /// room for a lower-bid one).
    fn find_cpu_eviction_victim(
        &self,
        dev_idx: usize,
        requester_bid: f64,
        requester: Option<ContextId>,
    ) -> Option<ContextId> {
        let mut best: Option<(f64, Instant, ContextId)> = None;

        for (&ctx_id, ctx) in &self.contexts {
            if requester == Some(ctx_id) { continue; }
            if !ctx.is_stashed() { continue; }
            let ctx_dev = ctx.device.unwrap_or(0) as usize;
            if ctx_dev != dev_idx { continue; }

            // Must have CPU-resident pages (working stash or committed stash).
            let has_cpu_working = !ctx.cpu_working_pages.is_empty();
            let has_cpu_committed = !ctx.committed_hashes.is_empty()
                && self.cpu_stores[dev_idx].prefix_len(&ctx.committed_hashes) > 0;
            if !has_cpu_working && !has_cpu_committed { continue; }

            // Only evict contexts with bid ≤ requester's bid.
            if ctx.bid > requester_bid { continue; }

            let spawn_time = ctx.owner
                .and_then(|pid| self.processes.get(&pid))
                .map(|p| p.created_at)
                .unwrap_or_else(Instant::now);

            let dominated = if let Some((best_bid, best_time, _)) = best {
                ctx.bid < best_bid
                || (ctx.bid == best_bid && spawn_time > best_time)
            } else {
                true
            };
            if dominated {
                best = Some((ctx.bid, spawn_time, ctx_id));
            }
        }

        best.map(|(_, _, ctx_id)| ctx_id)
    }

    /// Evict a Stashed context's pages from CPU to recompute.
    ///
    /// Releases committed pages from the CPU FlatPageStore (rc--, free at
    /// rc=0) and frees working page stash from the CPU pool. Transitions
    /// the context from Stashed to Suspended (no CPU cache, full recompute
    /// on restore).
    fn evict_from_cpu(&mut self, ctx_id: ContextId) {
        let (dev_idx, committed_hashes, cpu_working) = match self.contexts.get(&ctx_id) {
            Some(ctx) if ctx.is_stashed() => (
                ctx.device.unwrap_or(0) as usize,
                ctx.committed_hashes.clone(),
                ctx.cpu_working_pages.clone(),
            ),
            _ => return,
        };

        // Release committed pages from CPU store.
        if !committed_hashes.is_empty() {
            self.cpu_stores[dev_idx].release(&committed_hashes);
        }

        // Free working page stash from CPU pool.
        if !cpu_working.is_empty() {
            self.cpu_stores[dev_idx].free(&cpu_working);
            if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
                ctx.cpu_working_pages.clear();
            }
        }

        // Transition Stashed → Suspended (no longer has CPU pages).
        if let Some(ctx) = self.contexts.get_mut(&ctx_id) {
            ctx.state = State::Suspended;
        }
    }

    // =========================================================================
    // Suspension
    // =========================================================================

    /// Suspend a single Active context: stash pages to CPU, then release GPU.
    ///
    /// Uses `would_free()` to identify which committed pages will reach rc=0
    /// on release, and stashes only those to CPU via FlatPageStore. Shared
    /// prefix pages (rc > 1) stay on GPU.
    ///
    /// When the CPU pool is full, runs an eviction loop to free CPU pages
    /// from the lowest-bid suspended context before falling through to
    /// recompute (STORAGE.md §4.1b).
    pub(crate) fn suspend(&mut self, ctx_id: ContextId) {
        let (dev_idx, working, committed_hashes) = match self.contexts.get(&ctx_id) {
            Some(ctx) if ctx.is_active() || ctx.is_pinned() => {
                (ctx.device.unwrap_or(0) as usize, ctx.working_pages.clone(), ctx.committed_hashes.clone())
            }
            _ => return,
        };

        // Compute total CPU pages needed upfront for a single eviction pass.
        let requester_bid = self.contexts.get(&ctx_id).map(|c| c.bid).unwrap_or(0.0);
        let working_cpu_needed = working.len();
        let evictable_hashes = if !committed_hashes.is_empty() {
            self.gpu_stores[dev_idx].would_free(&committed_hashes)
        } else {
            Vec::new()
        };
        let committed_cpu_needed = evictable_hashes.len();
        let total_cpu_needed = working_cpu_needed + committed_cpu_needed;

        // Single eviction pass: free enough CPU pages for both phases.
        if total_cpu_needed > 0 && self.cpu_stores[dev_idx].available() < total_cpu_needed {
            while self.cpu_stores[dev_idx].available() < total_cpu_needed {
                match self.find_cpu_eviction_victim(dev_idx, requester_bid, Some(ctx_id)) {
                    Some(victim_id) => {
                        self.evict_from_cpu(victim_id);
                    }
                    None => break,
                }
            }
        }

        // All-or-nothing: only stash to CPU if enough space for the full
        // request. Partial stashing (working on CPU, committed dropped) would
        // waste CPU capacity on a context that still needs full recompute.
        let cpu_offload = total_cpu_needed > 0
            && self.cpu_stores[dev_idx].available() >= total_cpu_needed;

        // Phase 1: Stash working pages to CPU.
        // Working pages are exclusive — just D2H copy to CPU pool.
        if !working.is_empty() {
            let ctx = self.contexts.get_mut(&ctx_id).unwrap();
            ctx.suspended_working_count = ctx.working_pages.len();
            ctx.working_pages.clear();

            if cpu_offload {
                if let Some(cpu_pages) = self.cpu_stores[dev_idx].alloc(working.len()) {
                    let _ = device::copy_d2h(dev_idx as DeviceId, &working, &cpu_pages);
                    let ctx = self.contexts.get_mut(&ctx_id).unwrap();
                    ctx.cpu_working_pages = cpu_pages;
                }
            }

            self.gpu_stores[dev_idx].free(&working);
        }

        // Phase 2: Stash evictable committed pages to CPU.
        // Only pages with rc=1 (will reach rc=0 on release) need stashing.
        // Shared prefix pages (rc > 1) stay on GPU for other contexts.
        if cpu_offload && !evictable_hashes.is_empty() {
            let gpu_phys = self.gpu_stores[dev_idx].physical_ids(&evictable_hashes);
            if let Some(cpu_pages) = self.cpu_stores[dev_idx].alloc(gpu_phys.len()) {
                let _ = device::copy_d2h(dev_idx as DeviceId, &gpu_phys, &cpu_pages);
                self.cpu_stores[dev_idx].insert(&evictable_hashes, &cpu_pages);
            }
        }

        // Phase 3: Release committed chain refcounts from GPU trie.
        if !committed_hashes.is_empty() {
            if let Some(dev) = self.gpu_stores.get_mut(dev_idx) {
                dev.release(&committed_hashes);
            }
        }

        // Phase 4: Mark stashed or suspended.
        let ctx = self.contexts.get_mut(&ctx_id).unwrap();
        ctx.state = if cpu_offload { State::Stashed } else { State::Suspended };
        ctx.pending_suspend = false;

        // Remove this context from alloc_queue (can't serve while suspended;
        // deferred_ops are already on the context and will replay on restore).
        self.alloc_queue.retain(|&id| id != ctx_id);
    }

    /// Voluntarily suspend a context (program-initiated).
    pub(crate) fn voluntary_suspend(&mut self, id: ContextId) -> anyhow::Result<()> {
        match self.contexts.get(&id) {
            Some(ctx) if ctx.is_active() => {
                self.suspend(id);
                self.enqueue_restore(id);
                self.drain_queues();
                Ok(())
            }
            Some(ctx) if ctx.is_off_gpu() => Ok(()), // already off GPU
            Some(_) => Err(anyhow::anyhow!("Context {id} is pinned, cannot voluntarily suspend")),
            None => Err(anyhow::anyhow!("Context {id} not found")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{Context, RestoreEntry, State};

    /// Build a ContextManager with one device and `num_pages` GPU pages.
    /// Installs one process (`pid`) and one GPU-resident context (`ctx_id`)
    /// holding `held_pages` working pages, with `bid` as its declared bid.
    fn fixture(num_pages: usize, held_pages: usize, bid: f64)
        -> (ContextManager, ProcessId, ContextId)
    {
        let mut mgr = ContextManager::new(0, 16, &[num_pages], &[num_pages], 10, None, 10000.0);
        let pid = ProcessId::new_v4();
        mgr.register_process(pid, Some(160)).unwrap(); // 10 pages at page_size=16

        // Allocate `held_pages` from the GPU pool so `available()` reflects usage.
        let pages = mgr.gpu_stores[0].alloc(held_pages).expect("alloc");

        let ctx_id = 1u64;
        let mut ctx = Context::new(Some(pid));
        ctx.device = Some(0);
        ctx.working_pages = pages;
        ctx.bid = bid;
        ctx.state = State::Active;
        mgr.contexts.insert(ctx_id, ctx);
        mgr.process_entry(pid).context_ids.push(ctx_id);

        (mgr, pid, ctx_id)
    }

    /// On an uncontended device (free capacity, no waiters), clearing price
    /// is zero and no rent is charged even when contexts bid positively.
    #[test]
    fn uncontended_device_charges_no_rent() {
        let (mut mgr, pid, ctx_id) = fixture(100, 5, 2.0);
        let balance_before = mgr.process_entry(pid).balance;
        assert!(mgr.gpu_stores[0].available() > 0, "device should have slack");
        assert!(mgr.restore_queue.is_empty());
        assert!(mgr.alloc_queue.is_empty());

        mgr.tick(0, 0.001, &[ctx_id]);

        assert_eq!(mgr.auction_results[0].clearing_price, 0.0,
            "uncontended device must quote zero clearing price");
        assert_eq!(mgr.auction_results[0].total_revenue, 0.0,
            "no revenue when uncontended");
        assert_eq!(mgr.process_entry(pid).balance, balance_before,
            "balance must not drain when uncontended");
    }

    /// A non-empty restore_queue marks the device as contended even if the
    /// GPU has free pages. Revenue is collected at min(bid) × eff_pages.
    ///
    /// (Balance of a lone process is invariant under rent/dividend because
    /// the rent it pays flows back as its own dividend — zero-sum with one
    /// endowment-holder. So we assert on `total_revenue`, which is the
    /// direct observable of "rent was charged.")
    #[test]
    fn waiter_in_queue_triggers_rent() {
        let (mut mgr, _pid, ctx_id) = fixture(100, 5, 2.0);

        mgr.restore_queue.push(RestoreEntry {
            ctx_id: 9999, bid: 0.1, defaulted: false,
        });

        mgr.tick(0, 0.001, &[ctx_id]);

        assert_eq!(mgr.auction_results[0].clearing_price, 2.0,
            "contended device quotes min(bid) as clearing price");
        assert_eq!(mgr.auction_results[0].total_revenue, 10.0,
            "revenue = clearing_price × eff_pages = 2 × 5");
    }

    /// A fully-packed device with no waiters is still contended — any new
    /// arrival would force eviction, so the critical value is positive.
    #[test]
    fn full_device_charges_rent_even_with_no_waiters() {
        let (mut mgr, _pid, ctx_id) = fixture(5, 5, 1.5);
        assert_eq!(mgr.gpu_stores[0].available(), 0, "device is full");
        assert!(mgr.restore_queue.is_empty());

        mgr.tick(0, 0.001, &[ctx_id]);

        assert_eq!(mgr.auction_results[0].clearing_price, 1.5);
        assert_eq!(mgr.auction_results[0].total_revenue, 7.5,
            "revenue = 1.5 × 5 eff pages");
    }

    /// Wealth transfer under contention: when only one of two equally-
    /// endowed contexts is in-batch, rent flows from the payer to the
    /// non-payer via the endowment-weighted dividend. Conservation holds
    /// when payment isn't capped (both processes afford full rent).
    #[test]
    fn rent_redistributes_between_processes_under_contention() {
        // Both processes get large budgets so payment isn't balance-capped.
        let mut mgr = ContextManager::new(0, 16, &[10], &[10], 10, None, 10000.0);

        let payer_pid = ProcessId::new_v4();
        mgr.register_process(payer_pid, Some(16 * 1000)).unwrap(); // 1000-page budget
        let payer_pages = mgr.gpu_stores[0].alloc(5).expect("alloc");
        let payer_ctx = 1u64;
        let mut c1 = Context::new(Some(payer_pid));
        c1.device = Some(0);
        c1.working_pages = payer_pages;
        c1.bid = 4.0;
        c1.state = State::Active;
        mgr.contexts.insert(payer_ctx, c1);
        mgr.process_entry(payer_pid).context_ids.push(payer_ctx);

        let recv_pid = ProcessId::new_v4();
        mgr.register_process(recv_pid, Some(16 * 1000)).unwrap();
        let recv_pages = mgr.gpu_stores[0].alloc(5).expect("alloc");
        let recv_ctx = 2u64;
        let mut c2 = Context::new(Some(recv_pid));
        c2.device = Some(0);
        c2.working_pages = recv_pages;
        c2.bid = 4.0;
        c2.state = State::Active;
        mgr.contexts.insert(recv_ctx, c2);
        mgr.process_entry(recv_pid).context_ids.push(recv_ctx);

        let payer_before = mgr.process_entry(payer_pid).balance;
        let recv_before = mgr.process_entry(recv_pid).balance;

        // Only payer_ctx in batch — recv_ctx sits out this tick.
        mgr.tick(0, 0.001, &[payer_ctx]);

        let payer_after = mgr.process_entry(payer_pid).balance;
        let recv_after = mgr.process_entry(recv_pid).balance;

        assert!(payer_after < payer_before, "payer should lose balance");
        assert!(recv_after > recv_before, "receiver should gain balance");
        let before = payer_before + recv_before;
        let after = payer_after + recv_after;
        assert!((before - after).abs() < 1e-9,
            "total balance conserved: before={before} after={after}");
    }

    // =============================================================================
    // Phase 2: Two-wallet split (tokens vs credits)
    // =============================================================================

    /// Token wallet is initialized from token_budget, credit wallet from
    /// ⌈budget / page_size⌉ pages. The two are independent quantities.
    #[test]
    fn register_process_sets_both_wallets() {
        let mut mgr = ContextManager::new(0, 16, &[100], &[100], 10, None, 10000.0);
        let pid = ProcessId::new_v4();
        mgr.register_process(pid, Some(1000)).unwrap();

        let entry = mgr.process_entry(pid);
        assert_eq!(entry.tokens_remaining, Some(1000),
            "token wallet = Some(token_budget) when explicit cap requested");
        // 1000 tokens / 16 tokens/page = 63 pages (ceil)
        assert_eq!(entry.endowment, 63.0,
            "endowment = ⌈budget / page_size⌉ pages");
        assert_eq!(entry.balance, 63.0,
            "initial balance = endowment");
    }

    /// Processes launched without an explicit budget inherit the default,
    /// which is unlimited (None) by system policy.
    #[test]
    fn register_process_without_budget_is_unlimited() {
        let mut mgr = ContextManager::new(0, 16, &[100], &[100], 10, None, 10000.0);
        let pid = ProcessId::new_v4();
        mgr.register_process(pid, None).unwrap();
        assert_eq!(mgr.process_entry(pid).tokens_remaining, None,
            "no explicit cap → unlimited wallet");
        assert_eq!(mgr.process_entry(pid).endowment, 10.0,
            "endowment falls back to configured default");
    }

    /// debit_tokens decrements a capped wallet; saturates at zero; leaves
    /// unlimited wallets untouched.
    #[test]
    fn debit_tokens_is_monotone_and_saturates() {
        let mut mgr = ContextManager::new(0, 16, &[100], &[100], 10, None, 10000.0);
        let pid = ProcessId::new_v4();
        mgr.register_process(pid, Some(100)).unwrap();
        assert_eq!(mgr.process_entry(pid).tokens_remaining, Some(100));

        mgr.debit_tokens(pid, 30);
        assert_eq!(mgr.process_entry(pid).tokens_remaining, Some(70));

        mgr.debit_tokens(pid, 500); // underflow attempt
        assert_eq!(mgr.process_entry(pid).tokens_remaining, Some(0),
            "saturates at zero on underflow");

        // Unlimited wallet is unaffected by debit.
        let unlimited_pid = ProcessId::new_v4();
        mgr.register_process(unlimited_pid, None).unwrap();
        mgr.debit_tokens(unlimited_pid, 1_000_000);
        assert_eq!(mgr.process_entry(unlimited_pid).tokens_remaining, None,
            "debit has no effect on an unlimited wallet");
    }

    /// has_token_budget gates forward passes against the compute wallet only.
    /// A context with a full credit balance but zero tokens should be rejected.
    #[test]
    fn has_token_budget_independent_of_credit_balance() {
        let (mut mgr, pid, ctx_id) = fixture(100, 5, 2.0);
        // Install a finite cap, then drain it.
        mgr.process_entry(pid).tokens_remaining = Some(0);

        assert!(mgr.process_entry(pid).balance > 0.0,
            "credit balance still positive");
        assert!(!mgr.has_token_budget(ctx_id, 1),
            "token budget exhausted blocks forward pass");
        assert!(mgr.has_token_budget(ctx_id, 0),
            "zero-token pass always admitted");
    }

    /// Under no-make-cost, allocating pages leaves the credit balance
    /// untouched. (Historical bug: `charge_make_cost` would drain balance
    /// by `pages.len()` on every alloc.)
    #[test]
    fn allocation_does_not_touch_credit_balance() {
        let (mut mgr, pid, _ctx_id) = fixture(100, 0, 1.0);
        let balance_before = mgr.process_entry(pid).balance;

        // Allocate pages through the contention primitive (zero-page no-op
        // path is skipped; allocating 3 pages exercises the free-pool path).
        let ctx_id = 1u64;
        let new_ctx_id = 2u64;
        let mut ctx = Context::new(Some(pid));
        ctx.device = Some(0);
        ctx.state = State::Active;
        mgr.contexts.insert(new_ctx_id, ctx);
        mgr.process_entry(pid).context_ids.push(new_ctx_id);
        mgr.when_allocated(new_ctx_id, 0, 3, |_mgr, _pages| {});

        assert_eq!(mgr.process_entry(pid).balance, balance_before,
            "credit balance unchanged by alloc (no make cost)");
        let _ = ctx_id;
    }

    /// Conservation under default: when a payer can't cover full rent,
    /// dividends must be computed from *actual* collected credits, not
    /// the nominal `payment`. Otherwise the system mints credits.
    #[test]
    fn conservation_holds_under_default() {
        // Small token budget → small endowment → payer goes under.
        let mut mgr = ContextManager::new(0, 16, &[10], &[10], 10, None, 10000.0);

        let payer_pid = ProcessId::new_v4();
        mgr.register_process(payer_pid, Some(16)).unwrap(); // endowment = 1 page
        let p_pages = mgr.gpu_stores[0].alloc(5).expect("alloc");
        let payer_ctx = 1u64;
        let mut c1 = Context::new(Some(payer_pid));
        c1.device = Some(0); c1.state = State::Active;
        c1.working_pages = p_pages; c1.bid = 10.0;
        mgr.contexts.insert(payer_ctx, c1);
        mgr.process_entry(payer_pid).context_ids.push(payer_ctx);

        let recv_pid = ProcessId::new_v4();
        mgr.register_process(recv_pid, Some(16 * 1000)).unwrap();
        let r_pages = mgr.gpu_stores[0].alloc(5).expect("alloc");
        let recv_ctx = 2u64;
        let mut c2 = Context::new(Some(recv_pid));
        c2.device = Some(0); c2.state = State::Active;
        c2.working_pages = r_pages; c2.bid = 10.0;
        mgr.contexts.insert(recv_ctx, c2);
        mgr.process_entry(recv_pid).context_ids.push(recv_ctx);

        let total_before: f64 = mgr.processes.values().map(|p| p.balance).sum();

        // Only payer in batch. Payment owed = 10 × 5 = 50, but payer has 1.
        mgr.tick(0, 0.001, &[payer_ctx]);

        let total_after: f64 = mgr.processes.values().map(|p| p.balance).sum();
        assert!((total_after - total_before).abs() < 1e-9,
            "Σ balance conserved under default: before={total_before} after={total_after}");

        // Payer should be flagged defaulted.
        assert!(mgr.contexts[&payer_ctx].defaulted,
            "payer under-capitalized → defaulted flag set");
    }

    // =============================================================================
    // Phase 3: Admission gate (Σ endowment ≤ capacity × oversubscription_factor)
    // =============================================================================

    /// At factor = 1.0, exactly `capacity` pages worth of endowment fits.
    /// The N+1th page pushes Σ over the cap and is rejected.
    #[test]
    fn admission_gate_at_factor_1_enforces_capacity() {
        // 10 pages total, strict (factor = 1.0).
        let mut mgr = ContextManager::new(0, 16, &[10], &[10], 1, None, 1.0);

        // 10 processes of 1 endowment-page each fit exactly.
        for _ in 0..10 {
            mgr.register_process(ProcessId::new_v4(), Some(16)).unwrap();
        }
        // The 11th process pushes Σ = 11 over cap = 10.
        let eleventh = ProcessId::new_v4();
        let err = mgr.register_process(eleventh, Some(16)).unwrap_err();
        assert!(err.to_string().contains("admission denied"),
            "expected admission-denied error, got: {err}");
        // Rejected process must not have been inserted.
        assert!(!mgr.processes.contains_key(&eleventh));
    }

    /// At factor = 2.0, the cap is 2× physical capacity.
    #[test]
    fn admission_gate_overbook_factor_scales_cap() {
        let mut mgr = ContextManager::new(0, 16, &[10], &[10], 1, None, 2.0);

        // 20 processes at 1 page each = 20 endowment ≤ 20 cap. All admit.
        for _ in 0..20 {
            mgr.register_process(ProcessId::new_v4(), Some(16)).unwrap();
        }
        // 21st rejected.
        assert!(mgr.register_process(ProcessId::new_v4(), Some(16)).is_err());
    }

    /// After a process unregisters, its endowment is released back into the
    /// admission budget and the next admission succeeds.
    #[test]
    fn admission_frees_budget_on_unregister() {
        let mut mgr = ContextManager::new(0, 16, &[10], &[10], 1, None, 1.0);

        let pids: Vec<_> = (0..10).map(|_| {
            let pid = ProcessId::new_v4();
            mgr.register_process(pid, Some(16)).unwrap();
            pid
        }).collect();
        assert!(mgr.register_process(ProcessId::new_v4(), Some(16)).is_err(),
            "saturated");

        mgr.unregister_process(pids[0]);

        // Now a new process can take the freed slot.
        mgr.register_process(ProcessId::new_v4(), Some(16))
            .expect("admission should succeed after unregister");
    }

    /// Capacity is summed across devices — endowment competes against the
    /// total GPU pool, not just one device.
    #[test]
    fn admission_cap_is_sum_across_devices() {
        let mut mgr = ContextManager::new(0, 16, &[5, 5], &[5, 5], 1, None, 1.0);
        // 10 pages total across 2 devices.
        for _ in 0..10 {
            mgr.register_process(ProcessId::new_v4(), Some(16)).unwrap();
        }
        assert!(mgr.register_process(ProcessId::new_v4(), Some(16)).is_err());
    }
}
