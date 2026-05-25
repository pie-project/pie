//! SDK Context — stateful wrapper over the WIT `Context` resource.
//!
//! Owns the native context handle, caches page metadata, buffers instruct
//! tokens, and exposes ergonomic fill / flush / generate methods.

mod constraint;

// Re-export submodule public types.
pub use constraint::*;

use crate::ForwardPassExt;
use crate::Result;
use crate::inference::ForwardPass;
use crate::model::Model;
use crate::sample::Sampler;
use serde_json;

/// The raw WIT context resource, re-exported for power users.
pub use crate::pie::core::context::Context as RawContext;

// Instruct WIT bindings.
use crate::pie::instruct::chat;

/// Budget-exhausting bid: the maximum per-page-per-step rent the process
/// can sustain over `μ` steps of generation without going bankrupt.
///
/// Formula:
///
/// ```text
///     bid = (B/μ + d) / (p + μ(1 + cv²) / (2s))
/// ```
///
/// where:
/// - `B` = credit balance (market wallet, unit: pages)
/// - `μ` = expected remaining steps
/// - `d` = endowment-weighted dividend per step
/// - `p` = pages currently held
/// - `s` = page_size (tokens per page)
/// - `cv²` = squared coefficient of variation of the remaining-steps
///   distribution (0 = deterministic, 1 = geometric/memoryless)
///
/// The numerator is the per-step budget available for rent (balance
/// amortized over horizon, plus incoming dividend). The denominator is
/// the *total* page-steps of rent exposure: current pages `p` held for
/// `μ` steps, plus the triangular accumulation of newly created pages.
///
/// This is the truthful bid under critical-value payments — bidding this
/// value exhausts the wallet exactly at the end of the horizon. No
/// make-cost term: forward-pass compute is billed against the token
/// wallet, not the credit wallet.
pub(crate) fn compute_bid(
    balance: f64,
    pages: f64,
    mu: f64,
    cv2: f64,
    page_size: f64,
    dividend: f64,
) -> f64 {
    let mu = mu.max(1.0);
    let numerator = balance / mu + dividend;
    let denominator = pages + mu * (1.0 + cv2) / (2.0 * page_size);
    if denominator > 0.0 {
        numerator / denominator
    } else {
        numerator
    }
}

// =============================================================================
// Context
// =============================================================================

/// High-level inference context.
///
/// Wraps the native WIT [`RawContext`] resource and provides:
/// - **Buffered instruct fills** (`system`, `user`, `cue`, …) that accumulate
///   tokens locally.
/// - **`flush()`** to drain the buffer through a forward pass and commit pages.
/// - **`generate()`** to create a [`Generator`](crate::generation::Generator) for token-by-token generation.
/// - **Cached page metadata** (`seq_len`, `committed_pages`, `working_tokens`)
///   to avoid redundant WIT host calls.
pub struct Context {
    pub(crate) inner: RawContext,
    pub(crate) model: Model,
    pub(crate) page_size: u32,
    /// SDK-side token buffer filled by instruct operations.
    pub(crate) buffer: Vec<u32>,
    /// Deferred system text, so model templates that fold system into the
    /// first user turn can render the pair correctly.
    pending_system: Option<String>,
    /// Total tokens in committed + working pages (tracked locally).
    pub(crate) seq_len: u32,
    /// Number of committed pages (tracked locally).
    pub(crate) committed_pages: u32,
    /// Number of currently reserved working pages (tracked locally).
    pub(crate) working_pages: u32,
    /// Number of tokens in working (uncommitted) pages (tracked locally).
    pub(crate) working_tokens: u32,
}

impl Context {
    // ── Constructors ────────────────────────────────────────────────

    /// Create a fresh empty context for the given model.
    pub fn new(model: &Model) -> Result<Self> {
        let inner = RawContext::create(model)?;
        Ok(Self::wrap(inner))
    }

    /// Open a saved snapshot (implicit fork — snapshot stays immutable).
    pub fn open(model: &Model, name: &str) -> Result<Self> {
        let inner = RawContext::open(model, name)?;
        Ok(Self::wrap(inner))
    }

    /// Take ownership of a saved snapshot (snapshot is deleted).
    pub fn take(model: &Model, name: &str) -> Result<Self> {
        let inner = RawContext::take(model, name)?;
        Ok(Self::wrap(inner))
    }

    /// Delete a saved snapshot by name (static — no context needed).
    pub fn delete(model: &Model, name: &str) -> Result<()> {
        RawContext::delete(model, name)
    }

    /// Wrap an existing raw context, syncing cached state from the host.
    fn wrap(inner: RawContext) -> Self {
        let page_size = inner.tokens_per_page();
        let committed_pages = inner.committed_page_count();
        let working_pages = inner.working_page_count();
        let working_tokens = inner.working_page_token_count();
        let seq_len = committed_pages * page_size + working_tokens;
        let model = inner.model();
        Self {
            inner,
            model,
            page_size,
            buffer: Vec::new(),
            pending_system: None,
            seq_len,
            committed_pages,
            working_pages,
            working_tokens,
        }
    }

    // ── Lifecycle ───────────────────────────────────────────────────

    /// Fork into a new anonymous context (working pages are copied).
    ///
    /// The forked context inherits a copy of the parent's buffered tokens.
    pub fn fork(&self) -> Result<Self> {
        let forked = self.inner.fork()?;
        let model = forked.model();
        Ok(Self {
            inner: forked,
            model,
            page_size: self.page_size,
            buffer: self.buffer.clone(),
            pending_system: self.pending_system.clone(),
            seq_len: self.seq_len,
            committed_pages: self.committed_pages,
            working_pages: self.working_pages,
            working_tokens: self.working_tokens,
        })
    }

    /// Save the context under a user-chosen name.
    pub fn save(&self, name: &str) -> Result<()> {
        self.inner.save(name)
    }

    /// Anonymous save — returns a runtime-generated snapshot name.
    pub fn snapshot(&self) -> Result<String> {
        self.inner.snapshot()
    }

    /// Force-destroy the context immediately, consuming it.
    pub fn destroy(self) {
        self.inner.destroy()
    }

    // ── Market operations ────────────────────────────────────────────

    /// Suspend this context (release pages, stop rent).
    /// Restoration is system-driven based on bid priority.
    pub fn suspend(&self) -> Result<()> {
        self.inner.suspend()
    }

    /// Override the auto-computed bid (willingness to pay per page per
    /// step). Higher bid = harder to evict, restored first under
    /// contention. Bids are bounded below by zero; the runtime refuses
    /// negative values.
    ///
    /// Most callers should NOT use this — the [`Generator`] auto-bids
    /// every step using a budget-exhausting strategy that drains the
    /// wallet over the horizon. Reach for `set_bid` only when you have a
    /// strategy of your own.
    ///
    /// Stopping forward progress when the compute budget is spent is
    /// handled on the **token wallet**, not here — calling forward
    /// passes after `tokens_remaining == 0` fails with an error,
    /// independent of any bid you set. A low bid only affects admission
    /// under contention.
    ///
    /// [`Generator`]: crate::generation::Generator
    pub fn set_bid(&self, value: f64) {
        self.inner.bid(value);
    }

    /// Mark this context as idle: drop the bid to zero so other
    /// contexts can take its pages under contention. Returns an opaque
    /// RAII guard; the truthful generation bid is restored on drop.
    ///
    /// Use across an external wait (HTTP, tool call, anything off-GPU)
    /// where holding the bid would buy you nothing:
    ///
    /// ```ignore
    /// let _idle = ctx.idle();
    /// let result = http_get(url).await?;
    /// // _idle dropped here → bid restored
    /// ```
    ///
    /// On an uncontended device the runtime charges zero rent anyway —
    /// `idle` is a no-op cost-wise but still safe to call. Under load,
    /// it yields priority to other workloads for the duration of the
    /// wait.
    pub fn idle(&self) -> Idle<'_> {
        // Snapshot the truthful bid to restore on drop. If pages == 0
        // (rare), there's nothing to bid for; default to 0.0.
        let pages = (self.committed_pages + self.working_pages) as f64;
        let saved = if pages > 0.0 {
            let balance = crate::scheduling::balance(&self.model);
            let dividend = crate::scheduling::dividend(&self.model);
            let page_size = self.page_size as f64;
            // Conservative μ = 4096; no per-generation horizon visible here.
            compute_bid(balance, pages, 4096.0, 1.0, page_size, dividend)
        } else {
            0.0
        };
        self.inner.bid(0.0);
        Idle { ctx: self, saved }
    }

    // ── Accessors (no WIT calls) ────────────────────────────────────

    /// The model this context was created with.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Tokens per page.
    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    /// Total sequence length (committed + working tokens, excluding buffer).
    pub fn seq_len(&self) -> u32 {
        self.seq_len
    }

    /// Pending (buffered but not yet flushed) tokens.
    pub fn buffer(&self) -> &[u32] {
        &self.buffer
    }

    /// Access the underlying WIT context resource (escape hatch).
    pub fn inner(&self) -> &RawContext {
        &self.inner
    }

    // ── Instruct Fillers ────────────────────────────────────────────
    //
    // Each filler delegates to the WIT free function (which only needs the
    // model for template lookup / tokenization) and appends the resulting
    // tokens to the local buffer.

    fn flush_pending_system(&mut self) {
        if let Some(system) = self.pending_system.take() {
            let tokens = chat::system(&self.model, &system);
            self.buffer.extend(tokens);
        }
    }

    fn is_first_chat_fill(&self) -> bool {
        self.seq_len == 0 && self.buffer.is_empty()
    }

    /// Fill a system message; tokens are buffered for the next `flush()`.
    pub fn system(&mut self, message: &str) -> &mut Self {
        self.flush_pending_system();
        self.pending_system = Some(message.to_string());
        self
    }

    /// Fill a user message.
    pub fn user(&mut self, message: &str) -> &mut Self {
        let tokens = match self.pending_system.take() {
            Some(system) => chat::system_user(&self.model, &system, message),
            None if self.is_first_chat_fill() => chat::first_user(&self.model, message),
            None => chat::user(&self.model, message),
        };
        self.buffer.extend(tokens);
        self
    }

    /// Fill an assistant message (for history replay).
    pub fn assistant(&mut self, message: &str) -> &mut Self {
        self.flush_pending_system();
        let tokens = chat::assistant(&self.model, message);
        self.buffer.extend(tokens);
        self
    }

    /// Cue the model to generate (fills the generation header).
    pub fn cue(&mut self) -> &mut Self {
        self.flush_pending_system();
        let tokens = chat::cue(&self.model);
        self.buffer.extend(tokens);
        self
    }

    /// Seal the current turn (insert stop token).
    pub fn seal(&mut self) -> &mut Self {
        self.flush_pending_system();
        let tokens = chat::seal(&self.model);
        self.buffer.extend(tokens);
        self
    }

    /// Append raw tokens to the buffer directly.
    pub fn append(&mut self, tokens: &[u32]) -> &mut Self {
        self.flush_pending_system();
        self.buffer.extend_from_slice(tokens);
        self
    }

    /// Register `tools` in the chat template's tool block. Each tool's
    /// metadata is wrapped in the `{name, description, parameters}` envelope
    /// the host expects, then spliced into the buffer via the model's
    /// `equip_prefix`.
    ///
    /// Use the [`#[tool]`](inferlet_macros::tool) macro to derive a `Tool`
    /// impl from a Rust async fn, or implement the trait by hand for
    /// dynamically-loaded tools.
    ///
    /// # Errors
    /// Returns the underlying schema-parse or `equip_prefix` error if a
    /// tool's `schema()` is not valid JSON, or if the model has no tool
    /// template.
    pub fn equip(&mut self, tools: &[&dyn crate::tools::Tool]) -> Result<&mut Self> {
        self.flush_pending_system();
        let envelopes: Vec<String> = tools
            .iter()
            .map(|t| {
                let parsed: serde_json::Value = serde_json::from_str(t.schema())
                    .map_err(|e| format!("tool `{}`: invalid schema: {e}", t.name()))?;
                Ok(serde_json::json!({
                    "name": t.name(),
                    "description": t.description(),
                    "parameters": parsed,
                })
                .to_string())
            })
            .collect::<Result<_>>()?;
        let prefix = crate::tools::equip_prefix(&self.model, &envelopes)?;
        self.buffer.extend_from_slice(&prefix);
        Ok(self)
    }

    /// Drop the trailing `n` tokens from the working pages and re-sync the
    /// cached page/seq counters from the host.
    ///
    /// Use after a forward pass that wrote speculative draft tokens, to
    /// roll back the rejected suffix. `n` counts only working-page tokens —
    /// pages that already committed cannot be truncated through this API.
    pub fn truncate(&mut self, n: u32) {
        if n == 0 {
            return;
        }
        let keep = self.inner.working_page_token_count().saturating_sub(n);
        self.inner.truncate_working_page_tokens(keep);
        // Re-sync from the host: truncation can release pages, and the
        // safe thing is to read the authoritative counts back.
        self.committed_pages = self.inner.committed_page_count();
        self.working_pages = self.inner.working_page_count();
        self.working_tokens = self.inner.working_page_token_count();
        self.seq_len = self.committed_pages * self.page_size + self.working_tokens;
    }

    // ── Flush ───────────────────────────────────────────────────────

    /// Drain the buffered tokens through a forward pass and commit pages.
    ///
    /// After flush, the buffer is empty and `seq_len` reflects all
    /// consumed tokens.
    pub async fn flush(&mut self) -> Result<()> {
        self.flush_pending_system();
        if self.buffer.is_empty() {
            return Ok(());
        }

        let tokens = std::mem::take(&mut self.buffer);
        let num_tokens = tokens.len() as u32;

        // Reserve additional pages if we need more than currently allocated.
        let total_tokens_after = self.working_tokens + num_tokens;
        let pages_needed = (total_tokens_after + self.page_size - 1) / self.page_size;
        let additional = pages_needed.saturating_sub(self.working_pages);
        if additional > 0 {
            self.inner
                .reserve_working_pages(additional)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
            self.working_pages = pages_needed;
        }

        // Build and execute a forward pass.
        let pass = ForwardPass::new(&self.model);
        pass.context(&self.inner);

        let positions: Vec<u32> = (self.seq_len..self.seq_len + num_tokens).collect();
        pass.input_tokens(&tokens, &positions);

        pass.execute_async()
            .await
            .map_err(|e| format!("Forward pass failed: {}", e))?;

        // Commit full pages.
        let new_working = self.working_tokens + num_tokens;
        let pages_to_commit = new_working / self.page_size;
        if pages_to_commit > 0 {
            self.inner
                .commit_working_pages(pages_to_commit)
                .map_err(|e| format!("Failed to commit pages: {}", e))?;
        }

        // Update cached state.
        self.committed_pages += pages_to_commit;
        self.working_pages -= pages_to_commit;
        self.working_tokens = new_working % self.page_size;
        self.seq_len += num_tokens;

        Ok(())
    }

    // ── Pass ────────────────────────────────────────────────────────

    /// Build a single [`Forward`](crate::forward::Forward) — a forward pass with
    /// automatic page reservation, position derivation, and post-execute
    /// commit. Use for prefill, scoring, custom decode loops, and anywhere
    /// the [`generate`](Self::generate) loop is too high-level.
    pub fn forward(&mut self) -> crate::forward::Forward<'_> {
        self.flush_pending_system();
        crate::forward::Forward::new(self)
    }

    // ── Generate ────────────────────────────────────────────────────

    /// Build a [`Generator`](crate::generation::Generator) — the
    /// multi-step token-generation state machine. Any tokens already in
    /// the buffer (from `system / user / cue / …`) are drained on the
    /// first step.
    pub fn generate(&mut self, sampler: Sampler) -> crate::generation::Generator<'_> {
        self.flush_pending_system();
        crate::generation::Generator::new(self, sampler)
    }
}

// =============================================================================
// Idle — RAII guard returned by Context::idle()
// =============================================================================

/// Opaque RAII guard. Created by [`Context::idle`]. Drops the bid to
/// zero for the duration of the guard's lifetime; restores the
/// truthful bid on drop.
pub struct Idle<'a> {
    ctx: &'a Context,
    saved: f64,
}

impl<'a> Drop for Idle<'a> {
    fn drop(&mut self) {
        self.ctx.inner.bid(self.saved);
    }
}
