//! `Forward` — a single forward-pass primitive with automatic page management.
//!
//! `Forward` wraps the WIT `forward-pass` resource and folds in the page math
//! that every consumer would otherwise repeat: working-page reservation
//! before submission, position derivation, page commit after execution.
//! Slots are attached via [`Forward::sample`] / [`Forward::probe`] and read back
//! through typed handles on the returned [`Output`].
//!
//! For the rare case where the caller needs full WIT control (custom
//! attention masks, manual page management for speculation rollback, etc.),
//! the underlying `forward-pass` resource is still available via
//! `inferlet::inference::ForwardPass`.

use std::marker::PhantomData;

use crate::ForwardPassExt;
use crate::Result;
use crate::adapter::Adapter;
use crate::context::Context;
use crate::pie::core::inference::Output as RawOutput;
use crate::pie::core::inference::{ForwardPass, SlotOutput};
use crate::sample::{self, Probe, Sampler};

// =============================================================================
// Slot handles
// =============================================================================

/// Opaque handle to a sampler slot, returned by [`Forward::sample`]. Pass to
/// [`Output::token`] / [`Output::tokens`] to read the result.
#[derive(Copy, Clone, Debug)]
pub struct SampleHandle {
    slot: u32,
    /// Number of positions the sampler was attached to. Lets `Output::tokens`
    /// read back the right window without separate bookkeeping.
    arity: u32,
}

impl SampleHandle {
    pub(crate) fn new(slot: u32, arity: u32) -> Self {
        Self { slot, arity }
    }
    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }
    pub(crate) fn arity(&self) -> u32 {
        self.arity
    }
}

/// Phantom-typed handle to a probe slot, returned by [`Forward::probe`]. The
/// type parameter selects which `Output::*` accessor compiles.
#[derive(Debug)]
pub struct ProbeHandle<P> {
    slot: u32,
    _kind: PhantomData<fn() -> P>,
}

// `PhantomData<fn() -> P>` is `Copy + Clone` regardless of `P`, but the
// auto-derive generates `where P: Copy/Clone` bounds which surprise
// callers using probe markers like `Logprobs(Vec<u32>)`. Hand-roll the
// impls so handles are always `Copy`.
impl<P> Copy for ProbeHandle<P> {}
impl<P> Clone for ProbeHandle<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P> ProbeHandle<P> {
    pub(crate) fn new(slot: u32) -> Self {
        Self {
            slot,
            _kind: PhantomData,
        }
    }
    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }
}

// =============================================================================
// Pass
// =============================================================================

/// One slot attach, kept until `execute` so the WIT calls land in the same
/// order the handles were handed out. Probes go through `into_wit()` at
/// construction time and stash the resulting variant here.
enum SlotSpec {
    Sample {
        indices: Vec<u32>,
        sampler: Sampler,
    },
    Probe {
        index: u32,
        wit: crate::pie::core::inference::Sampler,
    },
}

/// Builder for a single forward pass. Construct via
/// [`Context::pass`](crate::Context::pass).
///
/// Builder methods return `&mut Self` so chains compose. `execute(self)` runs
/// the host call, commits pages, and returns an [`Output`].
pub struct Forward<'ctx> {
    ctx: &'ctx mut Context,
    /// Tokens fed at auto-derived positions (appended at the next free slot).
    auto_inputs: Vec<u32>,
    /// Tokens fed at caller-supplied positions. Used for non-contiguous
    /// scoring (e.g. validation) where positions don't extend `seq_len`.
    explicit_inputs: Vec<(Vec<u32>, Vec<u32>)>,
    /// Slot attachments in declaration order.
    slots: Vec<SlotSpec>,
    next_slot: u32,
    mask: Option<Vec<u32>>,
    attn_mask: Option<Vec<Vec<u32>>>,
    adapter: Option<&'ctx Adapter>,
    zo_seed: Option<i64>,
    /// Visual spans spliced at the given anchor positions, in declaration
    /// order. Emitted as `input-image` calls at execute time.
    images: Vec<(&'ctx crate::media::Image, u32)>,
    /// Audio clips spliced at the given anchor positions, in declaration
    /// order. Emitted as `input-audio` calls at execute time.
    audios: Vec<(&'ctx crate::media::Audio, u32)>,
    /// When true, `execute` reserves pages and runs the pass but does NOT
    /// commit the newly-filled working pages. Used by manual speculative
    /// loops that must keep all written tokens in *working* pages so a
    /// follow-up [`Context::truncate`] can roll back the rejected suffix —
    /// committing first (the default) would lock unverified drafts into
    /// committed KV that `truncate` cannot reach. The caller commits the
    /// verified prefix itself once acceptance is known.
    defer_commit: bool,
}

impl<'ctx> Forward<'ctx> {
    /// Position the *first* auto-input token will occupy. Equals the
    /// owning context's `seq_len` at the time `pass()` was called. Useful
    /// for callers that need to derive an attention mask or condition on
    /// the upcoming position before `execute`. The sampler at index `i`
    /// (when `pass.sample(&[i], …)`) lands at `start_position() + i`.
    pub fn start_position(&self) -> u32 {
        self.ctx.seq_len()
    }

    /// Page size of the owning context — for callers that need to size
    /// per-position structures (masks, etc.) without re-querying.
    pub fn page_size(&self) -> u32 {
        self.ctx.page_size()
    }

    /// Construct a `Forward` against a context. Prefer [`Context::pass`].
    pub(crate) fn new(ctx: &'ctx mut Context) -> Self {
        Self {
            ctx,
            auto_inputs: Vec::new(),
            explicit_inputs: Vec::new(),
            slots: Vec::new(),
            next_slot: 0,
            mask: None,
            attn_mask: None,
            adapter: None,
            zo_seed: None,
            images: Vec::new(),
            audios: Vec::new(),
            defer_commit: false,
        }
    }

    /// Skip the post-execute page commit, keeping every written token in
    /// *working* pages. The pass still reserves pages and writes KV, and the
    /// context's `seq_len` / `working_tokens` advance, but no working page is
    /// committed.
    ///
    /// Required for correct manual speculative decoding (draft + verify +
    /// truncate). The default auto-commit promotes full working pages into
    /// committed KV as soon as they fill — but a speculative verify pass
    /// writes anchor + draft tokens whose acceptance is not yet known. A
    /// committed page can no longer be rolled back by [`Context::truncate`],
    /// and committing pages mid-speculation also perturbs the KV the
    /// remaining decode attends to. Keeping the verified prefix in working
    /// pages (and letting `truncate` drop the rejected suffix) is what makes
    /// the generated sequence independent of the draft length, as lossless
    /// speculative decoding requires.
    pub fn defer_commit(&mut self) -> &mut Self {
        self.defer_commit = true;
        self
    }

    // ── Inputs ─────────────────────────────────────────────────────────

    /// Append `tokens` at positions starting at the context's current
    /// sequence length. Multiple calls accumulate. After `execute()`, these
    /// tokens occupy KV slots and the context's `seq_len` advances.
    pub fn input(&mut self, tokens: &[u32]) -> &mut Self {
        self.auto_inputs.extend_from_slice(tokens);
        self
    }

    /// Feed `tokens` at caller-supplied `positions`. Use for scoring at
    /// arbitrary positions (e.g. log-probability evaluation across a
    /// candidate window). These tokens are NOT auto-committed — the caller
    /// is responsible for any page bookkeeping if positions overlap or
    /// extend beyond `seq_len`.
    ///
    /// Panics if `tokens.len() != positions.len()`.
    pub fn input_at(&mut self, tokens: &[u32], positions: &[u32]) -> &mut Self {
        assert_eq!(
            tokens.len(),
            positions.len(),
            "input_at: tokens and positions must be the same length"
        );
        self.explicit_inputs
            .push((tokens.to_vec(), positions.to_vec()));
        self
    }

    // ── Slot attach ────────────────────────────────────────────────────

    /// Attach a sampler at one or more `indices`. Indices are 0-based into
    /// the auto-input window: `0` is the first auto-input token, `n-1` the
    /// last. Returns a handle for reading the sampled token(s) on the
    /// resulting [`Output`].
    pub fn sample(&mut self, indices: &[u32], sampler: Sampler) -> SampleHandle {
        let arity = indices.len() as u32;
        let h = SampleHandle::new(self.next_slot, arity);
        self.slots.push(SlotSpec::Sample {
            indices: indices.to_vec(),
            sampler,
        });
        // Multi-arity samplers produce `arity` Token slots in the output
        // (one per index); advance the slot index by that count so any
        // subsequent `sample` / `probe` call sees the right offset.
        self.next_slot += arity;
        h
    }

    /// Attach a probe at a single `index`. Returns a typed handle whose
    /// type parameter selects which `Output::*` accessor compiles.
    pub fn probe<P: Probe>(&mut self, index: u32, probe: P) -> ProbeHandle<P::Out> {
        let h = ProbeHandle::new(self.next_slot);
        self.slots.push(SlotSpec::Probe {
            index,
            wit: probe.into_wit(),
        });
        self.next_slot += 1;
        h
    }

    // ── Decoration ─────────────────────────────────────────────────────

    /// Set a static logit mask (BRLE) applied at every sampled position.
    pub fn mask(&mut self, brle: &[u32]) -> &mut Self {
        self.mask = Some(brle.to_vec());
        self
    }

    /// Set per-query-position attention masks. Length must match the total
    /// number of query positions across all `input` / `input_at` calls.
    /// If unset, the runtime synthesizes a causal mask.
    pub fn attention_mask(&mut self, masks: &[Vec<u32>]) -> &mut Self {
        self.attn_mask = Some(masks.to_vec());
        self
    }

    /// Apply an adapter (LoRA, etc.) for this forward pass.
    pub fn adapter(&mut self, adapter: &'ctx Adapter) -> &mut Self {
        self.adapter = Some(adapter);
        self
    }

    /// Splice an encoded visual span (image or video clip) at sequence
    /// position `anchor`. The driver runs the vision encoder and scatters the
    /// projected rows into the hidden state. Advance your position cursor by
    /// `image.position_span()` for any tokens that follow. See MULTIMODAL.md.
    pub fn input_image(&mut self, image: &'ctx crate::media::Image, anchor: u32) -> &mut Self {
        self.images.push((image, anchor));
        self
    }

    /// Splice an encoded audio clip at sequence position `anchor`. The driver
    /// runs the gemma4_audio encoder and scatters the projected rows into the
    /// hidden state. Advance your position cursor by `audio.position_span()`
    /// for any tokens that follow. See audio_frontend.md.
    pub fn input_audio(&mut self, audio: &'ctx crate::media::Audio, anchor: u32) -> &mut Self {
        self.audios.push((audio, anchor));
        self
    }

    /// Set a `zo` (Evolution Strategies) seed for this forward pass.
    pub fn zo_seed(&mut self, seed: i64) -> &mut Self {
        self.zo_seed = Some(seed);
        self
    }

    // ── Execute ────────────────────────────────────────────────────────

    /// Run the forward pass. Reserves working pages for any auto-inputs,
    /// submits all attached inputs and slots, awaits the host, commits any
    /// newly-filled pages, and updates the context's cached state.
    pub async fn execute(self) -> Result<Output> {
        let Forward {
            ctx,
            auto_inputs,
            explicit_inputs,
            slots,
            next_slot: _,
            mask,
            attn_mask,
            adapter,
            zo_seed,
            images,
            audios,
            defer_commit,
        } = self;

        let n_auto = auto_inputs.len() as u32;

        // Reserve working pages for auto-inputs (those occupy KV slots and
        // commit on the way out). Explicit inputs are scoring-only — the
        // caller manages their pages.
        if n_auto > 0 {
            let total_after = ctx.working_tokens + n_auto;
            let pages_needed = (total_after + ctx.page_size - 1) / ctx.page_size;
            let additional = pages_needed.saturating_sub(ctx.working_pages);
            if additional > 0 {
                ctx.inner
                    .reserve_working_pages(additional)
                    .map_err(|e| format!("Forward::execute: reserve_working_pages: {e}"))?;
                ctx.working_pages = pages_needed;
            }
        }

        let pass = ForwardPass::new(&ctx.model);
        pass.context(&ctx.inner);

        for (image, anchor) in images {
            pass.input_image(image, anchor);
        }
        for (audio, anchor) in audios {
            pass.input_audio(audio, anchor);
        }

        if let Some(a) = adapter {
            pass.adapter(a);
        }
        if let Some(seed) = zo_seed {
            crate::pie::zo::zo::adapter_seed(&pass, seed);
        }

        if n_auto > 0 {
            let positions: Vec<u32> = (ctx.seq_len..ctx.seq_len + n_auto).collect();
            pass.input_tokens(&auto_inputs, &positions);
        }
        for (toks, pos) in &explicit_inputs {
            pass.input_tokens(toks, pos);
        }

        for spec in slots {
            match spec {
                SlotSpec::Sample { indices, sampler } => {
                    let wit: crate::pie::core::inference::Sampler = sampler.into();
                    pass.sampler(&indices, &wit);
                }
                SlotSpec::Probe { index, wit } => {
                    pass.sampler(&[index], &wit);
                }
            }
        }

        if let Some(m) = mask {
            pass.logit_mask(&m);
        }
        if let Some(m) = attn_mask {
            pass.attention_mask(&m);
        }

        let raw = pass
            .execute_async()
            .await
            .map_err(|e| format!("Forward::execute: forward pass failed: {e}"))?;

        // Account for the newly-written KV. By default we also commit any
        // pages the auto-input tokens fully filled. When `defer_commit` is
        // set (manual speculation), we keep every written token in *working*
        // pages so a later `truncate` can roll back rejected drafts — the
        // caller commits the verified prefix afterwards.
        if n_auto > 0 {
            let new_working = ctx.working_tokens + n_auto;
            let to_commit = if defer_commit {
                0
            } else {
                new_working / ctx.page_size
            };
            if to_commit > 0 {
                ctx.inner
                    .commit_working_pages(to_commit)
                    .map_err(|e| format!("Forward::execute: commit_working_pages: {e}"))?;
            }
            ctx.committed_pages += to_commit;
            ctx.working_pages -= to_commit;
            ctx.working_tokens = new_working - to_commit * ctx.page_size;
            ctx.seq_len += n_auto;
        }

        Ok(Output::new(raw))
    }
}

// =============================================================================
// Output
// =============================================================================

/// Result of one forward-pass execution — produced by both
/// [`Forward::execute`] and [`GenStep::execute`](crate::generation::GenStep::execute).
///
/// **Common path** ([`Generator`](crate::generation::Generator)): read
/// the [`tokens`](Self::tokens) field for the accepted tokens this step
/// (post stop / max-tokens truncation). The auto-attached sampler's
/// [`SampleHandle`] is also exposed via [`auto_sampler`](Self::auto_sampler)
/// for callers that want the pre-truncation verifier output.
///
/// **Raw `Forward`**: read sampler slots via [`token`](Self::token) /
/// [`tokens_at`](Self::tokens_at) using the handles returned at attach
/// time. The [`tokens`](Self::tokens) field is empty.
///
/// **Probes** (both paths): [`distribution`](Self::distribution) /
/// [`logits`](Self::logits) / [`logprobs`](Self::logprobs) /
/// [`entropy`](Self::entropy) take a typed [`ProbeHandle`].
///
/// Mismatched access (reading a sampler slot through a probe handle, or
/// vice versa) returns `None`.
pub struct Output {
    raw: RawOutput,
    /// Generator-accepted tokens this step, post stop / max-tokens
    /// truncation. Empty for raw `Forward::execute` (no Generator state).
    pub tokens: Vec<u32>,
    auto_sampler: Option<SampleHandle>,
}

impl Output {
    pub(crate) fn new(raw: RawOutput) -> Self {
        Self {
            raw,
            tokens: Vec::new(),
            auto_sampler: None,
        }
    }

    pub(crate) fn from_generator(
        raw: RawOutput,
        tokens: Vec<u32>,
        auto_sampler: Option<SampleHandle>,
    ) -> Self {
        Self {
            raw,
            tokens,
            auto_sampler,
        }
    }

    /// Underlying WIT output, for callers who need the raw slot list or the
    /// speculative side channel (`raw.spec_tokens`, `raw.spec_positions`).
    pub fn raw(&self) -> &RawOutput {
        &self.raw
    }

    /// Handle for the generator's auto-attached sampler. `None` for raw
    /// `Forward` results and for steps where
    /// [`GenStep::clear_sampler`](crate::generation::GenStep::clear_sampler)
    /// was called.
    pub fn auto_sampler(&self) -> Option<SampleHandle> {
        self.auto_sampler
    }

    /// First token from a single-index sampler slot.
    pub fn token(&self, h: SampleHandle) -> Option<u32> {
        match self.raw.slots.get(h.slot() as usize)? {
            SlotOutput::Token(t) => Some(*t),
            _ => None,
        }
    }

    /// Tokens at the slot range a multi-index sampler covers. Returns one
    /// token per index the sampler was attached to. In speculative mode
    /// the slice may be shorter than `arity` if the verifier rejected
    /// drafts.
    pub fn tokens_at(&self, h: SampleHandle) -> Vec<u32> {
        let mut out = Vec::with_capacity(h.arity() as usize);
        let start = h.slot() as usize;
        for i in 0..(h.arity() as usize) {
            match self.raw.slots.get(start + i) {
                Some(SlotOutput::Token(t)) => out.push(*t),
                _ => break,
            }
        }
        out
    }

    /// Distribution as `(ids, probs)` for a [`Distribution`](sample::Distribution) probe.
    pub fn distribution(&self, h: ProbeHandle<sample::Distribution>) -> Option<(&[u32], &[f32])> {
        match self.raw.slots.get(h.slot() as usize)? {
            SlotOutput::Distribution((ids, ps)) => Some((ids.as_slice(), ps.as_slice())),
            _ => None,
        }
    }

    /// Raw logits bytes for a [`Logits`](sample::Logits) probe.
    pub fn logits(&self, h: ProbeHandle<sample::Logits>) -> Option<&[u8]> {
        match self.raw.slots.get(h.slot() as usize)? {
            SlotOutput::Logits(b) => Some(b.as_slice()),
            _ => None,
        }
    }

    /// Logprob list for a [`Logprob`](sample::Logprob) or
    /// [`Logprobs`](sample::Logprobs) probe. Length is 1 for a single-token
    /// query, K for a list query.
    pub fn logprobs(&self, h: ProbeHandle<sample::Logprobs>) -> Option<&[f32]> {
        match self.raw.slots.get(h.slot() as usize)? {
            SlotOutput::Logprobs(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Entropy for an [`Entropy`](sample::Entropy) probe.
    pub fn entropy(&self, h: ProbeHandle<sample::Entropy>) -> Option<f32> {
        match self.raw.slots.get(h.slot() as usize)? {
            SlotOutput::Entropy(v) => Some(*v),
            _ => None,
        }
    }
}
