//! SDK Context — stateful wrapper over the WIT `Context` resource.
//!
//! Owns the native context handle, caches page metadata, buffers instruct
//! tokens, and exposes ergonomic fill / flush / generate methods.

mod constraint;

// Re-export submodule public types.
pub use constraint::*;

use crate::Result;
use crate::inference::ForwardPass;
use crate::sample::Sampler;
use serde_json;

/// The raw WIT context resource, re-exported for power users.
pub use crate::pie::core::context::Context as RawContext;

// Instruct WIT bindings.
use crate::pie::instruct::chat;

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

    /// Create a fresh empty context.
    pub fn new() -> Result<Self> {
        let inner = RawContext::create()?;
        Ok(Self::wrap(inner))
    }

    /// Open a saved snapshot (implicit fork — snapshot stays immutable).
    pub fn open(name: &str) -> Result<Self> {
        let inner = RawContext::open(name)?;
        Ok(Self::wrap(inner))
    }

    /// Take ownership of a saved snapshot (snapshot is deleted).
    pub fn take(name: &str) -> Result<Self> {
        let inner = RawContext::take(name)?;
        Ok(Self::wrap(inner))
    }

    /// Delete a saved snapshot by name (static — no context needed).
    pub fn delete(name: &str) -> Result<()> {
        RawContext::delete(name)
    }

    /// Wrap an existing raw context, syncing cached state from the host.
    fn wrap(inner: RawContext) -> Self {
        let page_size = inner.tokens_per_page();
        let committed_pages = inner.committed_page_count();
        let working_pages = inner.working_page_count();
        let working_tokens = inner.working_page_token_count();
        let seq_len = committed_pages * page_size + working_tokens;
        Self {
            inner,
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
        Ok(Self {
            inner: forked,
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

    // ── Suspend / restore ────────────────────────────────────────────

    /// Suspend this context (release GPU pages, offload to CPU).
    /// Restoration is system-driven under FCFS: suspended contexts are
    /// restored oldest-launched first as memory frees up.
    pub fn suspend(&self) -> Result<()> {
        self.inner.suspend()
    }

    // ── Accessors (no WIT calls) ────────────────────────────────────

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
            let tokens = chat::system(&system);
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
            Some(system) => chat::system_user(&system, message),
            None if self.is_first_chat_fill() => chat::first_user(message),
            None => chat::user(message),
        };
        self.buffer.extend(tokens);
        self
    }

    /// Fill an assistant message (for history replay).
    pub fn assistant(&mut self, message: &str) -> &mut Self {
        self.flush_pending_system();
        let tokens = chat::assistant(message);
        self.buffer.extend(tokens);
        self
    }

    /// Cue the model to generate (fills the generation header).
    pub fn cue(&mut self) -> &mut Self {
        self.flush_pending_system();
        let tokens = chat::cue();
        self.buffer.extend(tokens);
        self
    }

    /// Seal the current turn (insert stop token).
    pub fn seal(&mut self) -> &mut Self {
        self.flush_pending_system();
        let tokens = chat::seal();
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
        let prefix = crate::tools::equip_prefix(&envelopes)?;
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
        // `truncate_working_page_tokens` REMOVES the last N working tokens
        // (see context.wit): its argument is the count to drop, not the
        // count to keep. Pass `n` straight through — clamped to the working
        // length so an over-large request can't underflow into the runtime's
        // range check.
        let removable = self.inner.working_page_token_count().min(n);
        if removable == 0 {
            return;
        }
        self.inner.truncate_working_page_tokens(removable);
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
        let pass = ForwardPass::new();
        pass.context(&self.inner);

        let positions: Vec<u32> = (self.seq_len..self.seq_len + num_tokens).collect();
        pass.input_tokens(&tokens, &positions);

        pass.execute()
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

    /// Splice an encoded image (or video clip) into the context. Runs the
    /// vision encoder driver-side and commits the resulting soft-token KV pages,
    /// exactly like [`flush`] does for text. The image's `token_count()` soft
    /// tokens occupy KV slots; the sequence cursor advances by
    /// `position_span()` (equal for Gemma's 1-D positions). See MULTIMODAL.md.
    ///
    /// Any buffered text is flushed first so ordering (text → image → text) is
    /// preserved in the KV cache.
    pub async fn append_image(&mut self, image: &crate::media::Image) -> Result<()> {
        // The model's own span delimiters (host-provided; empty for models that
        // need none) are applied here so the inferlet stays model-agnostic — it
        // never names `<|vision_start|>` etc.
        let prefix = image.prefix_tokens();
        let suffix = image.suffix_tokens();
        if !prefix.is_empty() {
            self.append(&prefix);
        }
        self.flush().await?; // commit any pending text + the span prefix

        let num_tokens = image.token_count();
        if num_tokens == 0 {
            if !suffix.is_empty() {
                self.append(&suffix);
            }
            return Ok(());
        }
        let span = image.position_span();

        let total_tokens_after = self.working_tokens + num_tokens;
        let pages_needed = (total_tokens_after + self.page_size - 1) / self.page_size;
        let additional = pages_needed.saturating_sub(self.working_pages);
        if additional > 0 {
            self.inner
                .reserve_working_pages(additional)
                .map_err(|e| format!("append_image: reserve pages: {}", e))?;
            self.working_pages = pages_needed;
        }

        let pass = ForwardPass::new();
        pass.context(&self.inner);
        pass.input_image(image, self.seq_len);
        pass.execute()
            .await
            .map_err(|e| format!("append_image: forward pass: {}", e))?;

        let new_working = self.working_tokens + num_tokens;
        let pages_to_commit = new_working / self.page_size;
        if pages_to_commit > 0 {
            self.inner
                .commit_working_pages(pages_to_commit)
                .map_err(|e| format!("append_image: commit pages: {}", e))?;
        }
        self.committed_pages += pages_to_commit;
        self.working_pages -= pages_to_commit;
        self.working_tokens = new_working % self.page_size;
        // The image occupies `num_tokens` physical KV rows whose 1-D
        // bookkeeping positions are `anchor..anchor+num_tokens`. Advance the
        // sequence cursor past them so the next text token's position is
        // strictly greater than every committed image-row position (the KV
        // commit enforces strict monotonicity). M-RoPE attention positions for
        // the image rows themselves are carried on the dedicated 3-axis
        // side-channel (`image_mrope_positions`), so the encoder/attention use
        // the correct (t,h,w) span regardless of this 1-D advance.
        let _ = span;
        self.seq_len += num_tokens;

        // Span suffix delimiter (buffered; flushed with the next fill).
        if !suffix.is_empty() {
            self.append(&suffix);
        }
        Ok(())
    }

    /// Splice an encoded audio clip into the context. Runs the gemma4_audio
    /// encoder driver-side and commits the resulting soft-token KV pages,
    /// exactly like [`append_image`](Self::append_image) does for vision. The
    /// clip's `token_count()` soft tokens occupy KV slots; the sequence cursor
    /// advances past them. See audio_frontend.md.
    ///
    /// Any buffered text is flushed first so ordering (text → audio → text) is
    /// preserved in the KV cache.
    pub async fn append_audio(&mut self, audio: &crate::media::Audio) -> Result<()> {
        // Host-provided span delimiters (e.g. Gemma `<|audio>` / `<audio|>`),
        // applied here so the inferlet never names them.
        let prefix = audio.prefix_tokens();
        let suffix = audio.suffix_tokens();
        if !prefix.is_empty() {
            self.append(&prefix);
        }
        self.flush().await?; // commit any pending text + the span prefix

        let num_tokens = audio.token_count();
        if num_tokens == 0 {
            if !suffix.is_empty() {
                self.append(&suffix);
            }
            return Ok(());
        }

        let total_tokens_after = self.working_tokens + num_tokens;
        let pages_needed = (total_tokens_after + self.page_size - 1) / self.page_size;
        let additional = pages_needed.saturating_sub(self.working_pages);
        if additional > 0 {
            self.inner
                .reserve_working_pages(additional)
                .map_err(|e| format!("append_audio: reserve pages: {}", e))?;
            self.working_pages = pages_needed;
        }

        let pass = ForwardPass::new();
        pass.context(&self.inner);
        pass.input_audio(audio, self.seq_len);
        pass.execute()
            .await
            .map_err(|e| format!("append_audio: forward pass: {}", e))?;

        let new_working = self.working_tokens + num_tokens;
        let pages_to_commit = new_working / self.page_size;
        if pages_to_commit > 0 {
            self.inner
                .commit_working_pages(pages_to_commit)
                .map_err(|e| format!("append_audio: commit pages: {}", e))?;
        }
        self.committed_pages += pages_to_commit;
        self.working_pages -= pages_to_commit;
        self.working_tokens = new_working % self.page_size;
        self.seq_len += num_tokens;

        // Span suffix delimiter (buffered; flushed with the next fill).
        if !suffix.is_empty() {
            self.append(&suffix);
        }
        Ok(())
    }

    /// Splice a decoded video clip ([`crate::media::Video`]) into the context.
    ///
    /// The host already demuxed + uniformly sampled the clip and preprocessed
    /// each frame per the bound model (see [`crate::media::Video::from_bytes`]).
    /// This injects each frame's soft tokens via [`append_image`], preceded by a
    /// short generic `mm:ss` timestamp text marker (plain tokens, encoded by
    /// whatever tokenizer — not model-specific). Each frame becomes committed KV
    /// exactly like an image, so fork / snapshot / prefix-cache apply. The
    /// per-frame soft-token budget and any span delimiters are the host's job, so
    /// this is identical across models. See MULTIMODAL.md §8.
    pub async fn append_video(&mut self, video: &crate::media::Video) -> Result<()> {
        let n = video.frame_count();
        for i in 0..n {
            let secs = video.timestamp(i).max(0.0) as u32;
            let marker = format!(" {:02}:{:02} ", secs / 60, secs % 60);
            let toks = crate::model::encode(&marker);
            self.append(&toks);
            let frame = video
                .frame(i)
                .map_err(|e| format!("append_video: frame {i}: {e}"))?;
            self.append_image(&frame).await?;
        }
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
