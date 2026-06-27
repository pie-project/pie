//! SDK Context — stateful facade over the `kv-working-set` host resource.
//!
//! Post-working-set refactor, the runtime is token-agnostic (W4): it owns only
//! physical KV page slots (a dense ordered array, addressed by relative index).
//! Everything semantic — the token buffer, per-token positions, the sequence
//! length, and the page layout — is **owned here in the SDK**. `Context` wraps a
//! [`KvWorkingSet`] and exposes the same ergonomic fill / flush / generate API
//! as before; forward passes are expressed as explicit `kv-working-set`
//! read+write descriptors instead of an opaque context handle.
//!
//! Single-model runtime: there is exactly one bound model, so the working set
//! binds to it implicitly (`KvWorkingSet::new()` takes no handle) and model
//! metadata is reached through the global `crate::model::*` free functions.

mod constraint;

// Re-export submodule public types.
pub use constraint::*;

use crate::Result;
use crate::inference::ForwardPass;
use crate::pie::core::working_set::KvWorkingSet;
use crate::sample::Sampler;
use serde::{Deserialize, Serialize};
use serde_json;
use std::sync::atomic::{AtomicU64, Ordering};

// Instruct WIT bindings.
use crate::pie::instruct::chat;

// =============================================================================
// Snapshot blob (W14 / §7)
// =============================================================================

const SNAPSHOT_VERSION: u32 = 1;
static SNAPSHOT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Inferlet-owned replayable snapshot manifest, serialized to a CPU-resident
/// blob. v1 stores the token log for replay-based restore; `cas_hashes` is
/// reserved for the future physical-reattach path (brief §7).
#[derive(Serialize, Deserialize)]
struct SnapshotManifest {
    version: u32,
    page_size: u32,
    seq_len: u32,
    tokens: Vec<u32>,
    buffer: Vec<u32>,
    pending_system: Option<String>,
    cas_hashes: Vec<u64>,
}

fn snapshot_path(name: &str) -> String {
    // The runtime preopens the per-instance scratch dir as `/scratch` in the
    // guest (runtime/src/instance.rs); a relative path has no matching preopen,
    // so snapshot blobs must be written there.
    format!("/scratch/{name}.pie-snapshot")
}

fn read_manifest(name: &str) -> Result<SnapshotManifest> {
    let bytes =
        std::fs::read(snapshot_path(name)).map_err(|e| format!("snapshot '{name}': read: {e}"))?;
    let manifest: SnapshotManifest =
        serde_json::from_slice(&bytes).map_err(|e| format!("snapshot '{name}': parse: {e}"))?;
    if manifest.version != SNAPSHOT_VERSION {
        return Err(format!(
            "snapshot '{name}': version {} unsupported (expected {SNAPSHOT_VERSION})",
            manifest.version
        ));
    }
    Ok(manifest)
}

// =============================================================================
// Context
// =============================================================================

/// High-level inference context.
///
/// Wraps a [`KvWorkingSet`] (the runtime's dense KV page-slot array) and owns
/// all semantic metadata the token-agnostic runtime no longer tracks:
/// - **Buffered instruct fills** (`system`, `user`, `cue`, …) that accumulate
///   tokens locally.
/// - **`flush()`** drains the buffer through a forward pass that writes the new
///   KV into freshly `alloc`'d page slots (full pages auto-seal host-side).
/// - **`generate()`** creates a [`Generator`](crate::generation::Generator).
/// - **Sequence cursor** (`seq_len`): the number of materialized KV tokens. The
///   per-page valid-token lengths and the read/write page split for every
///   forward are derived from it.
/// The merged `kv-working-set` read+write descriptor for a tail write,
/// produced by [`Context::prepare_write`] and emitted by
/// [`Context::attach_kv`]. Encodes the **Option B (disjoint)** convention: the
/// read range covers the prior FULL pages only and the write sub-range covers
/// the new-KV pages; the driver attends read ∪ write (the 1a-verified path,
/// zero driver change). Units: `inp_len`/`output_*` are PAGE/slot indices;
/// `valid_read`/`offset` are TOKEN counts.
pub(crate) struct KvWrite {
    /// Read page span length: `first_write_page` (prior full pages `[0, n)`).
    inp_len: u32,
    /// Read valid tokens: `first_write_page · page_size` (all read pages full).
    valid_read: u32,
    /// Write sub-range start page: `first_write_page`.
    output_start: u32,
    /// Write sub-range length: `total_pages − first_write_page`.
    output_len: u32,
    /// In-page token offset of the first NEW token in the first write page:
    /// `seq_len % page_size` (the partial-prior prefix rides the write page).
    offset: u32,
}

pub struct Context {
    /// The runtime KV working set: a dense ordered array of page slots.
    pub(crate) kv: KvWorkingSet,
    pub(crate) page_size: u32,
    /// SDK-side token buffer filled by instruct operations (not yet flushed).
    pub(crate) buffer: Vec<u32>,
    /// Deferred system text, so model templates that fold system into the
    /// first user turn can render the pair correctly.
    pending_system: Option<String>,
    /// Materialized KV tokens (committed to page slots). The next token's
    /// position is `seq_len`; the live page count is `ceil(seq_len/page_size)`.
    pub(crate) seq_len: u32,
    /// Replayable token log: every materialized text token, in order (W4
    /// replay history). Backs the SDK-blob snapshot facade — `open`/`take`
    /// replay this through forward passes to rebuild the KV.
    pub(crate) history: Vec<u32>,
    /// `false` once a non-replayable span (image / audio soft tokens) is
    /// materialized — such a context cannot be snapshotted by token replay in
    /// v1 (the encoder inputs are not in the token log).
    pub(crate) snapshottable: bool,
}

impl Context {
    // ── Constructors ────────────────────────────────────────────────

    /// Create a fresh empty context bound to the single runtime model.
    pub fn new() -> Result<Self> {
        let kv = KvWorkingSet::new();
        let page_size = kv.page_size();
        Ok(Self {
            kv,
            page_size,
            buffer: Vec::new(),
            pending_system: None,
            seq_len: 0,
            history: Vec::new(),
            snapshottable: true,
        })
    }

    // ── Lifecycle ───────────────────────────────────────────────────

    /// Fork into a new anonymous context. The KV page slots are shared by
    /// reference (lazy copy-on-write; the first divergent write copies); the
    /// forked context inherits a copy of the SDK-side metadata (buffer,
    /// pending system, sequence cursor).
    pub fn fork(&self) -> Result<Self> {
        let kv = self.kv.fork().map_err(|e| format!("Context::fork: {e}"))?;
        Ok(Self {
            kv,
            page_size: self.page_size,
            buffer: self.buffer.clone(),
            pending_system: self.pending_system.clone(),
            seq_len: self.seq_len,
            history: self.history.clone(),
            snapshottable: self.snapshottable,
        })
    }

    /// Force-destroy the context immediately, consuming it. Dropping the
    /// `KvWorkingSet` releases (decrefs) its page slots in the arena.
    pub fn destroy(self) {
        drop(self)
    }

    // ── Snapshots (SDK-owned CPU-resident blobs over wasi:filesystem, W14/§7) ──
    //
    // A snapshot is an inferlet-owned, replayable manifest: the materialized
    // token log plus the unflushed buffer / pending-system. It is serialized to
    // a CPU-resident file; `open`/`take` rebuild a fresh `kv-working-set` by
    // replaying the token log through forward passes. (Physical CAS-ref reattach
    // — reuse sealed pages by hash, skip replay — is a future perf path that
    // needs a runtime attach-by-cas op; out of v1.)

    /// Save the context under a user-chosen name (a CPU-resident blob).
    pub fn save(&self, name: &str) -> Result<()> {
        if !self.snapshottable {
            return Err(
                "Context::save: multimodal contexts are not snapshottable in v1 (soft-token \
                 KV cannot be replayed from a token log)"
                    .to_string(),
            );
        }
        let manifest = SnapshotManifest {
            version: SNAPSHOT_VERSION,
            page_size: self.page_size,
            seq_len: self.seq_len,
            tokens: self.history.clone(),
            buffer: self.buffer.clone(),
            pending_system: self.pending_system.clone(),
            cas_hashes: Vec::new(),
        };
        let bytes =
            serde_json::to_vec(&manifest).map_err(|e| format!("Context::save: serialize: {e}"))?;
        std::fs::write(snapshot_path(name), bytes)
            .map_err(|e| format!("Context::save: write '{name}': {e}"))?;
        Ok(())
    }

    /// Anonymous save — returns a freshly-generated snapshot name.
    pub fn snapshot(&self) -> Result<String> {
        let name = format!(
            "anon-{}-{}",
            crate::runtime::instance_id(),
            SNAPSHOT_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        self.save(&name)?;
        Ok(name)
    }

    /// Open a saved snapshot (the saved blob stays on disk — an implicit fork).
    /// Async because it replays the token log through forward passes.
    pub async fn open(name: &str) -> Result<Self> {
        let manifest = read_manifest(name)?;
        Self::from_manifest(manifest).await
    }

    /// Take ownership of a saved snapshot: open it, then delete the blob.
    pub async fn take(name: &str) -> Result<Self> {
        let ctx = Self::open(name).await?;
        let _ = std::fs::remove_file(snapshot_path(name));
        Ok(ctx)
    }

    /// Delete a saved snapshot by name. Missing snapshots are a no-op.
    pub fn delete(name: &str) -> Result<()> {
        let _ = std::fs::remove_file(snapshot_path(name));
        Ok(())
    }

    /// Rebuild a context from a manifest by replaying its token log.
    async fn from_manifest(manifest: SnapshotManifest) -> Result<Self> {
        let mut ctx = Self::new()?;
        if !manifest.tokens.is_empty() {
            // Replay the whole materialized log in one prefill pass; `flush`
            // re-records `history` and advances `seq_len`.
            ctx.buffer = manifest.tokens;
            ctx.flush().await?;
        }
        // Restore the unflushed tail (buffer + pending system) on top.
        ctx.buffer = manifest.buffer;
        ctx.pending_system = manifest.pending_system;
        Ok(ctx)
    }

    // ── Accessors (no host calls) ───────────────────────────────────

    /// Tokens per KV page.
    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    /// Total sequence length (materialized tokens, excluding the buffer).
    pub fn seq_len(&self) -> u32 {
        self.seq_len
    }

    /// Pending (buffered but not yet flushed) tokens.
    pub fn buffer(&self) -> &[u32] {
        &self.buffer
    }

    /// Escape hatch: the underlying KV working set (power users).
    pub fn working_set(&self) -> &KvWorkingSet {
        &self.kv
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

    // ── Sequence / page bookkeeping (SDK-owned, W4) ──────────────────

    /// Drop the trailing `n` materialized tokens from the sequence and free
    /// any page slots that become empty.
    ///
    /// Use after a forward pass that wrote speculative draft tokens, to roll
    /// back the rejected suffix. `n` is clamped to `seq_len`.
    pub fn truncate(&mut self, n: u32) {
        let n = n.min(self.seq_len);
        if n == 0 {
            return;
        }
        self.seq_len -= n;
        let keep = self.history.len().saturating_sub(n as usize);
        self.history.truncate(keep);
        // Free page slots that no longer hold any valid token.
        let live_pages = self.seq_len.div_ceil(self.page_size);
        let have = self.kv.size();
        if have > live_pages {
            let drop_idx: Vec<u32> = (live_pages..have).collect();
            // Best-effort: a structural-mutation failure here is non-fatal; the
            // stale trailing pages are simply overwritten by the next forward.
            let _ = self.kv.free(&drop_idx);
        }
    }

    /// Allocate page slots to cover `seq_len + n` tokens and build the
    /// `(generation, write indices, per-page valid lens, context page count)`
    /// for a forward that writes `n` new tokens at the current cursor.
    ///
    /// The new tokens land in slots `[seq_len/page_size .. ceil((seq_len+n)/
    /// page_size))` (the trailing partial page, if any, is rewritten — CoW
    /// preserves its prior tokens). The fully-complete prior pages
    /// `[0 .. seq_len/page_size)` are the read context.
    pub(crate) fn prepare_write(&mut self, n: u32) -> Result<KvWrite> {
        let p = self.page_size;
        let first_write_page = self.seq_len / p;
        let total_after = self.seq_len + n;
        let total_pages = total_after.div_ceil(p);
        let have = self.kv.size();
        if total_pages > have {
            self.kv
                .alloc(total_pages - have)
                .map_err(|e| format!("Context::prepare_write: alloc: {e}"))?;
        }
        // Option B (the 1a-verified DISJOINT convention; the merged `kv-working-set`
        // is a SYNTACTIC merge of the old kv-context(read) + kv-output(write), not a
        // semantic re-split): read = the prior FULL pages only `[0, first_write_page)`
        // (all full → fully valid); write = `[first_write_page, total_pages)`. The
        // partial-prior prefix (when `seq_len` is mid-page) + the new tokens ride the
        // write pages' per-page valid-lens, reconstructable host-side from `offset`.
        // The driver attends read ∪ write — unchanged.
        Ok(KvWrite {
            inp_len: first_write_page,
            valid_read: first_write_page * p,
            output_start: first_write_page,
            output_len: total_pages - first_write_page,
            offset: self.seq_len % p,
        })
    }

    /// Attach the merged `kv-working-set` read+write descriptor for a tail write
    /// produced by [`prepare_write`](Self::prepare_write) onto `pass`.
    pub(crate) fn attach_kv(&self, pass: &ForwardPass, w: &KvWrite) {
        pass.kv_working_set(
            &self.kv,
            0,
            w.inp_len,
            w.valid_read,
            w.output_start,
            w.output_len,
            w.offset,
        );
    }

    /// Attach a read-only `kv-working-set` spanning every materialized page (a
    /// pure decode / scoring pass that writes no new tail tokens). No write
    /// sub-range (`output-len = 0`); `valid-tokens = seq_len` (the real mid-page
    /// count, since there is no write page to carry the tail).
    pub(crate) fn attach_full_context(&self, pass: &ForwardPass) {
        let total_pages = self.seq_len.div_ceil(self.page_size);
        if total_pages > 0 {
            pass.kv_working_set(&self.kv, 0, total_pages, self.seq_len, 0, 0, 0);
        }
    }

    // ── Flush ───────────────────────────────────────────────────────

    /// Drain the buffered tokens through a forward pass and materialize their
    /// KV into page slots.
    ///
    /// After flush, the buffer is empty and `seq_len` reflects all consumed
    /// tokens.
    pub async fn flush(&mut self) -> Result<()> {
        self.flush_pending_system();
        if self.buffer.is_empty() {
            return Ok(());
        }
        let tokens = std::mem::take(&mut self.buffer);
        let n = tokens.len() as u32;
        let positions: Vec<u32> = (self.seq_len..self.seq_len + n).collect();

        let w = self.prepare_write(n)?;

        let pass = ForwardPass::new();
        self.attach_kv(&pass, &w);
        pass.input_tokens(&tokens, &positions);
        pass.execute();

        self.history.extend_from_slice(&tokens);
        self.seq_len += n;
        Ok(())
    }

    /// Splice an encoded image (or video clip) into the context. Runs the
    /// vision encoder driver-side and materializes the resulting soft-token KV
    /// pages, exactly like [`flush`](Self::flush) does for text. The image's
    /// `token_count()` soft tokens occupy KV slots; the sequence cursor
    /// advances by that many. See MULTIMODAL.md.
    ///
    /// Any buffered text is flushed first so ordering (text → image → text) is
    /// preserved in the KV cache.
    pub async fn append_image(&mut self, image: &crate::media::Image) -> Result<()> {
        // The model's own span delimiters (host-provided; empty for models that
        // need none) are applied here so the inferlet stays model-agnostic.
        let prefix = image.prefix_tokens();
        let suffix = image.suffix_tokens();
        if !prefix.is_empty() {
            self.append(&prefix);
        }
        self.flush().await?; // materialize any pending text + the span prefix

        let num_tokens = image.token_count();
        if num_tokens == 0 {
            if !suffix.is_empty() {
                self.append(&suffix);
            }
            return Ok(());
        }

        let w = self.prepare_write(num_tokens)?;
        let pass = ForwardPass::new();
        self.attach_kv(&pass, &w);
        pass.input_image(image, self.seq_len);
        pass.execute();

        // The image occupies `num_tokens` physical KV rows; advance the 1-D
        // sequence cursor past them (M-RoPE attention positions for the rows
        // themselves ride the dedicated 3-axis side channel).
        self.seq_len += num_tokens;
        // Soft-token KV cannot be reconstructed from the token log.
        self.snapshottable = false;

        if !suffix.is_empty() {
            self.append(&suffix);
        }
        Ok(())
    }

    /// Splice an encoded audio clip into the context. Runs the gemma4_audio
    /// encoder driver-side and materializes the resulting soft-token KV pages,
    /// exactly like [`append_image`](Self::append_image) does for vision.
    ///
    /// Any buffered text is flushed first so ordering (text → audio → text) is
    /// preserved in the KV cache.
    pub async fn append_audio(&mut self, audio: &crate::media::Audio) -> Result<()> {
        let prefix = audio.prefix_tokens();
        let suffix = audio.suffix_tokens();
        if !prefix.is_empty() {
            self.append(&prefix);
        }
        self.flush().await?; // materialize any pending text + the span prefix

        let num_tokens = audio.token_count();
        if num_tokens == 0 {
            if !suffix.is_empty() {
                self.append(&suffix);
            }
            return Ok(());
        }

        let w = self.prepare_write(num_tokens)?;
        let pass = ForwardPass::new();
        self.attach_kv(&pass, &w);
        pass.input_audio(audio, self.seq_len);
        pass.execute();

        self.seq_len += num_tokens;
        // Soft-token KV cannot be reconstructed from the token log.
        self.snapshottable = false;

        if !suffix.is_empty() {
            self.append(&suffix);
        }
        Ok(())
    }

    /// Splice a decoded video clip ([`crate::media::Video`]) into the context.
    ///
    /// Injects each frame's soft tokens via [`append_image`], preceded by a
    /// short generic `mm:ss` timestamp text marker. See MULTIMODAL.md §8.
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
    /// automatic page allocation, position derivation, and post-execute cursor
    /// advance. Use for prefill, scoring, custom decode loops, and anywhere
    /// the [`generate`](Self::generate) loop is too high-level.
    pub fn forward(&mut self) -> crate::forward::Forward<'_> {
        self.flush_pending_system();
        crate::forward::Forward::new(self)
    }

    // ── Generate ────────────────────────────────────────────────────

    /// Build a [`Generator`](crate::generation::Generator) — the multi-step
    /// token-generation state machine. Any tokens already in the buffer (from
    /// `system / user / cue / …`) are drained on the first step.
    pub fn generate(&mut self, sampler: Sampler) -> crate::generation::Generator<'_> {
        self.flush_pending_system();
        crate::generation::Generator::new(self, sampler)
    }
}
