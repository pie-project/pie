//! `Generator` — multi-step token-generation state machine.
//!
//! A [`Generator`] is configured up front (sampler, max-tokens, stop set,
//! constraints, optional speculator/adapter) and then iterated step-by-
//! step. Each call to [`Generator::next`] yields a [`GenStep`] — a
//! short-lived configuration handle for the upcoming forward pass that
//! the user can tweak (add probes, replace the auto-attached sampler).
//! Calling [`GenStep::execute`] runs the host call and does the loop's
//! post-step bookkeeping (commit, constraint advance, speculation
//! reconciliation, max-tokens / stop accounting).
//!
//! Three terminal-style sugars cover the common case:
//!
//! - [`Generator::collect_tokens`] — drains until done, returns all tokens.
//! - [`Generator::collect_text`] — drains, decodes through a chat decoder,
//!   returns the full string.
//! - [`Generator::collect_json`] — adds a `T`-derived JSON-schema
//!   constraint, drains, parses into `T`.
//!
//! For per-step control (watermarking, custom sampling), call `next()` /
//! `execute()` directly and use [`Generator::accept`] when sampling
//! manually.

use crate::ForwardPassExt;
use crate::Result;
use crate::adapter::Adapter;
use crate::context::{Context, compute_bid, brle_and};
use crate::forward::{Output, ProbeHandle, SampleHandle};
use crate::pie::core::inference::{ForwardPass, Sampler as WitSampler, SlotOutput};
use crate::pie::core::inference::Output as RawOutput;
use crate::sample::{Probe, Sampler};
use crate::spec::Speculator;

// Re-export so callers don't have to pull from `context` directly.
pub use crate::context::{Constrain, GrammarConstraint, Schema};

// =============================================================================
// Speculation mode
// =============================================================================

enum SpecMode {
    None,
    System {
        spec_tokens: Vec<u32>,
        spec_positions: Vec<u32>,
    },
    Custom(Box<dyn Speculator>),
}

// =============================================================================
// Generator
// =============================================================================

/// Builder + iterator for token generation. See module docs.
pub struct Generator<'ctx> {
    ctx: &'ctx mut Context,
    sampler: Sampler,
    stop: Vec<u32>,
    max_tokens: Option<usize>,
    horizon: Option<usize>,
    constraints: Vec<Box<dyn Constrain>>,
    /// Tokens accepted last step; advanced into each constraint's `step`
    /// at the start of the next iteration.
    constraint_pending: Vec<u32>,
    speculation: SpecMode,
    adapter: Option<&'ctx Adapter>,
    zo_seed: Option<i64>,
    /// Probes added via [`Generator::probe_each_step`] — re-attached every
    /// step. Stored as `(query_index, wit_variant)` pairs.
    step_probes: Vec<(u32, WitSampler)>,
    tokens_generated: usize,
    done: bool,
}

impl<'ctx> Generator<'ctx> {
    /// Construct a generator over `ctx` with the given sampler. Prefer
    /// [`Context::generate`].
    pub(crate) fn new(ctx: &'ctx mut Context, sampler: Sampler) -> Self {
        // Prime the bid with the budget-exhausting rate (geometric prior,
        // cv²=1) so the scheduler sees a reasonable number from the
        // start. Re-bid each step using the horizon cascade.
        let balance = crate::scheduling::balance(&ctx.model);
        let dividend = crate::scheduling::dividend(&ctx.model);
        let pages = (ctx.committed_pages + ctx.working_pages).max(1) as f64;
        let page_size = ctx.page_size as f64;
        ctx.set_bid(compute_bid(balance, pages, 4096.0, 1.0, page_size, dividend));

        Self {
            ctx,
            sampler,
            stop: Vec::new(),
            max_tokens: None,
            horizon: None,
            constraints: Vec::new(),
            constraint_pending: Vec::new(),
            speculation: SpecMode::None,
            adapter: None,
            zo_seed: None,
            step_probes: Vec::new(),
            tokens_generated: 0,
            done: false,
        }
    }

    // ── Builder ────────────────────────────────────────────────────────

    /// Hard cap on tokens generated across all steps. Once reached, `next`
    /// returns `None`.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Stop tokens. Generation halts when any of these is sampled. Pass
    /// `chat::stop_tokens(&model)` for chat-template defaults.
    pub fn stop(mut self, tokens: &[u32]) -> Self {
        self.stop = tokens.to_vec();
        self
    }

    /// Append `tokens` to the stop set without replacing existing entries.
    pub fn add_stop(mut self, tokens: &[u32]) -> Self {
        self.stop.extend_from_slice(tokens);
        self
    }

    /// Attach a constraint. Multiple constraints compose by AND-ing their
    /// per-step masks.
    pub fn constrain<C: Constrain + 'static>(mut self, constraint: C) -> Self {
        self.constraints.push(Box::new(constraint));
        self
    }

    /// Attach a [`Schema`] — compiled into a [`GrammarConstraint`]
    /// internally. Composes with other constraints just like
    /// [`constrain`](Self::constrain).
    pub fn constrain_with<S: Schema>(mut self, schema: S) -> Result<Self> {
        let c = schema.build_constraint(&self.ctx.model)?;
        self.constraints.push(Box::new(c));
        Ok(self)
    }

    /// Use a custom drafter for speculative decoding.
    pub fn speculator<S: Speculator + 'static>(mut self, s: S) -> Self {
        self.speculation = SpecMode::Custom(Box::new(s));
        self
    }

    /// Enable host-driven speculation: the runtime returns next-iter draft
    /// tokens via the forward-pass output and the Generator stages them
    /// for the next step.
    pub fn system_speculation(mut self) -> Self {
        self.speculation = SpecMode::System {
            spec_tokens: Vec::new(),
            spec_positions: Vec::new(),
        };
        self
    }

    /// Apply an adapter (LoRA etc.) on every forward pass.
    pub fn adapter(mut self, a: &'ctx Adapter) -> Self {
        self.adapter = Some(a);
        self
    }

    /// Set a zo (Evolution Strategies) seed on every forward pass.
    pub fn zo_seed(mut self, seed: i64) -> Self {
        self.zo_seed = Some(seed);
        self
    }

    /// Hint the expected output length for budget planning. Falls back to
    /// `max_tokens` then a Lindy heuristic.
    pub fn horizon(mut self, n: usize) -> Self {
        self.horizon = Some(n);
        self
    }

    /// Attach a probe to every step at `index`. Returns a typed handle
    /// that's reusable across each `Output`.
    pub fn probe_each_step<P: Probe>(&mut self, index: u32, probe: P) -> ProbeHandle<P::Out> {
        // Slot 0 is reserved for the auto-sampler; per-step probes follow.
        let slot = (1 + self.step_probes.len()) as u32;
        self.step_probes.push((index, probe.into_wit()));
        ProbeHandle::new(slot)
    }

    // ── Iteration ──────────────────────────────────────────────────────

    /// Whether generation has terminated (max_tokens or stop hit).
    pub fn is_done(&self) -> bool {
        self.done
            || self
                .max_tokens
                .map_or(false, |m| self.tokens_generated >= m)
    }

    /// Tokens generated so far across all steps.
    pub fn tokens_generated(&self) -> usize {
        self.tokens_generated
    }

    /// Begin the next step. Returns `Ok(None)` when generation is finished.
    /// The returned [`GenStep`] borrows the generator mutably; complete it
    /// with [`GenStep::execute`] (or drop it to skip the iteration).
    pub fn next(&mut self) -> Result<Option<GenStep<'_, 'ctx>>> {
        if self.is_done() {
            return Ok(None);
        }

        // Re-bid using the horizon cascade.
        self.recompute_bid();

        // Drain the context's buffer (filled by `system / user / cue / …`).
        let pending = std::mem::take(&mut self.ctx.buffer);

        // Pull drafts from the speculator.
        let (drafts, draft_positions) = match &mut self.speculation {
            SpecMode::None => (Vec::new(), Vec::new()),
            SpecMode::System {
                spec_tokens,
                spec_positions,
            } => (std::mem::take(spec_tokens), std::mem::take(spec_positions)),
            SpecMode::Custom(s) => s.draft(),
        };

        // Compose constraint masks. Each constraint advances on the
        // tokens accepted last step, then yields its next-position mask.
        let mask = if self.constraints.is_empty() {
            None
        } else {
            let advance = std::mem::take(&mut self.constraint_pending);
            let masks: Vec<Vec<u32>> = self
                .constraints
                .iter_mut()
                .map(|c| c.step(&advance).to_vec())
                .filter(|m| !m.is_empty())
                .collect();
            match masks.len() {
                0 => None,
                1 => Some(masks.into_iter().next().unwrap()),
                _ => {
                    let mut iter = masks.into_iter();
                    let first = iter.next().unwrap();
                    Some(iter.fold(first, |acc, m| brle_and(&acc, &m)))
                }
            }
        };

        Ok(Some(GenStep {
            parent: self,
            pending,
            drafts,
            draft_positions,
            mask,
            extra_probes: Vec::new(),
            user_cleared_sampler: false,
        }))
    }

    /// Register a manually-sampled token (or sequence) with the generator.
    /// Use after [`GenStep::clear_sampler`] when the inferlet sampled by
    /// hand off a probe — the generator updates max-tokens / stop /
    /// constraint counters and seeds the next iteration's input. The
    /// token doesn't enter KV here; the next `next() / execute()` flushes
    /// it through a forward pass like a normal pending input.
    pub fn accept(&mut self, tokens: &[u32]) -> Vec<u32> {
        if tokens.is_empty() {
            return Vec::new();
        }

        let mut accepted = tokens.to_vec();
        if let Some(pos) = accepted.iter().position(|t| self.stop.contains(t)) {
            accepted.truncate(pos);
            self.done = true;
        }
        if let Some(max) = self.max_tokens {
            let remaining = max.saturating_sub(self.tokens_generated);
            if accepted.len() > remaining {
                accepted.truncate(remaining);
                self.done = true;
            }
        }
        if accepted.is_empty() {
            return Vec::new();
        }

        // Stage for next forward pass via the buffer; advance counters.
        self.ctx.buffer.extend_from_slice(&accepted);
        self.constraint_pending.extend_from_slice(&accepted);
        self.tokens_generated += accepted.len();
        if let SpecMode::Custom(s) = &mut self.speculation {
            s.accept(&accepted);
        }
        accepted
    }

    // ── Terminal sugar ─────────────────────────────────────────────────

    /// Run to completion; return the full token stream.
    pub async fn collect_tokens(mut self) -> Result<Vec<u32>> {
        let mut all = Vec::new();
        while let Some(step) = self.next()? {
            let out = step.execute().await?;
            all.extend(out.tokens);
        }
        Ok(all)
    }

    /// Run to completion, decode through a chat decoder, return text.
    pub async fn collect_text(mut self) -> Result<String> {
        use crate::chat;
        let mut decoder = chat::Decoder::new(&self.ctx.model);
        let mut text = String::new();
        while let Some(step) = self.next()? {
            let out = step.execute().await?;
            match decoder.feed(&out.tokens)? {
                chat::Event::Delta(s) => text.push_str(&s),
                chat::Event::Done(s) => return Ok(s),
                chat::Event::Idle | chat::Event::Interrupt(_) => {}
            }
        }
        Ok(text)
    }

    /// Constrain to JSON conforming to `T`'s schema, run to completion,
    /// parse. Composes with any constraints already attached.
    pub async fn collect_json<T>(self) -> Result<T>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema,
    {
        let schema = schemars::schema_for!(T);
        let schema_str = serde_json::to_string(&schema)
            .map_err(|e| format!("collect_json: serialize schema: {e}"))?;
        let constraint =
            GrammarConstraint::from_json_schema(&schema_str, &self.ctx.model)?;
        let text = self.constrain(constraint).collect_text().await?;
        serde_json::from_str(&text).map_err(|e| format!("collect_json: deserialize: {e}"))
    }

    // ── Internal helpers ───────────────────────────────────────────────

    fn recompute_bid(&mut self) {
        let balance = crate::scheduling::balance(&self.ctx.model);
        let dividend = crate::scheduling::dividend(&self.ctx.model);
        let pages = (self.ctx.committed_pages + self.ctx.working_pages) as f64;
        let page_size = self.ctx.page_size as f64;

        // Horizon cascade: explicit → max_tokens → Lindy.
        let (mu, cv2) = if let Some(h) = self.horizon {
            ((h.saturating_sub(self.tokens_generated)).max(1) as f64, 0.0)
        } else if let Some(m) = self.max_tokens {
            ((m.saturating_sub(self.tokens_generated)).max(1) as f64, 1.0)
        } else {
            (self.tokens_generated.max(64) as f64, 1.0)
        };

        self.ctx
            .set_bid(compute_bid(balance, pages, mu, cv2, page_size, dividend));
    }
}

// =============================================================================
// GenStep — short-lived per-iteration handle
// =============================================================================

/// Configuration handle for the upcoming forward pass. Returned by
/// [`Generator::next`]. Pre-populated with the generator's pending fills,
/// configured sampler, constraint mask, and any speculator drafts. Add
/// extra probes or call [`clear_sampler`](Self::clear_sampler) to take
/// over sampling, then [`execute`](Self::execute).
pub struct GenStep<'g, 'ctx> {
    parent: &'g mut Generator<'ctx>,
    pending: Vec<u32>,
    drafts: Vec<u32>,
    draft_positions: Vec<u32>,
    mask: Option<Vec<u32>>,
    /// Extra probes added via [`probe`](Self::probe). Each is `(index,
    /// wit_variant)` and is appended to the slot list after the auto-
    /// sampler and the generator's per-step probes.
    extra_probes: Vec<(u32, WitSampler)>,
    user_cleared_sampler: bool,
}

impl<'g, 'ctx> GenStep<'g, 'ctx> {
    /// Drop the generator's auto-attached sampler. The caller must read
    /// the distribution off a probe and register their own pick via
    /// [`Generator::accept`] after `execute`. Useful for custom sampling
    /// (watermarking, externally-driven samplers, etc.).
    pub fn clear_sampler(&mut self) -> &mut Self {
        self.user_cleared_sampler = true;
        self
    }

    /// Attach an extra probe at `index`. Returns a typed handle.
    pub fn probe<P: Probe>(&mut self, index: u32, probe: P) -> ProbeHandle<P::Out> {
        // Slot count: sampler (if not cleared) + step-probes + earlier
        // extra probes added in this step.
        let base = if self.user_cleared_sampler { 0 } else { 1 };
        let slot = (base + self.parent.step_probes.len() + self.extra_probes.len()) as u32;
        self.extra_probes.push((index, probe.into_wit()));
        ProbeHandle::new(slot)
    }

    /// Run the forward pass and fold the result into the generator's
    /// state.
    pub async fn execute(self) -> Result<Output> {
        let GenStep {
            parent,
            pending,
            drafts,
            draft_positions,
            mask,
            extra_probes,
            user_cleared_sampler,
        } = self;

        let n_pending = pending.len() as u32;
        let n_drafted = drafts.len() as u32;

        if n_pending == 0 && n_drafted == 0 && user_cleared_sampler && extra_probes.is_empty() {
            // Truly nothing to do — no input, no sampler, no probes.
            // Mark done so collect_* sugars terminate cleanly.
            parent.done = true;
            return Ok(Output::new(RawOutput {
                slots: Vec::new(),
                spec_tokens: Vec::new(),
                spec_positions: Vec::new(),
            }));
        }

        // No input + sampler still attached: the SDK once advertised this
        // as a "no-input bootstrap" path that samples from the last cached
        // KV position without growing the working tail (see the comment
        // below). The portable driver's `plan_single_request` rejects
        // zero-token requests, and silent failure here lets `collect_*`
        // spin forever firing empty batches once the previous step's
        // sample came back empty (e.g. grammar mask exhaustion). Fail
        // loudly so the caller sees the real problem instead of an
        // infinite retry loop.
        if n_pending == 0 && n_drafted == 0 && extra_probes.is_empty() {
            parent.done = true;
            return Err(
                "GenStep::execute: no input tokens and no probes — previous \
                 step likely sampled zero tokens (constraint mask exhausted, \
                 or driver returned empty response). Refusing to fire a \
                 zero-input forward pass that the driver would reject."
                    .to_string(),
            );
        }

        // Reserve pages for pending (drafts share the working tail —
        // their commit/truncate happens after we know what was accepted).
        // No-input bootstrap path skips reservation: the host samples from
        // the last cached KV position without growing the working tail.
        let n_total_input = n_pending + n_drafted;
        if n_total_input > 0 {
            let total_after = parent.ctx.working_tokens + n_total_input;
            let pages_needed = (total_after + parent.ctx.page_size - 1) / parent.ctx.page_size;
            let additional = pages_needed.saturating_sub(parent.ctx.working_pages);
            if additional > 0 {
                parent.ctx
                    .inner
                    .reserve_working_pages(additional)
                    .map_err(|e| format!("GenStep::execute reserve: {e}"))?;
                parent.ctx.working_pages = pages_needed;
            }
        }

        // Build forward pass.
        let pass = ForwardPass::new(&parent.ctx.model);
        pass.context(&parent.ctx.inner);
        if let Some(a) = parent.adapter {
            pass.adapter(a);
        }
        if let Some(seed) = parent.zo_seed {
            crate::pie::zo::zo::adapter_seed(&pass, seed);
        }

        // Custom-mode drafts ride alongside pending in `input_tokens` and
        // are verified by the SDK walk below. This keeps verification
        // driver-agnostic: the runtime's `spec_token_ids` channel only
        // carries the host-side `output_speculative_tokens` flow used
        // by System-mode (where the runtime returns next-iter drafts).
        let is_custom = matches!(parent.speculation, SpecMode::Custom(_));
        let do_sdk_verify = is_custom && n_drafted > 0 && n_pending > 0;

        if do_sdk_verify {
            let mut all_tokens = Vec::with_capacity((n_pending + n_drafted) as usize);
            all_tokens.extend_from_slice(&pending);
            all_tokens.extend_from_slice(&drafts);
            let mut all_positions: Vec<u32> =
                (parent.ctx.seq_len..parent.ctx.seq_len + n_pending).collect();
            all_positions.extend_from_slice(&draft_positions);
            pass.input_tokens(&all_tokens, &all_positions);
        } else {
            if n_pending > 0 {
                let positions: Vec<u32> =
                    (parent.ctx.seq_len..parent.ctx.seq_len + n_pending).collect();
                pass.input_tokens(&pending, &positions);
            }
            if !drafts.is_empty() {
                pass.input_speculative_tokens(&drafts, &draft_positions);
            }
        }
        if matches!(parent.speculation, SpecMode::System { .. }) {
            pass.output_speculative_tokens(true);
        }

        // Sampler attach. Custom-with-drafts attaches (1 + n_drafted)
        // consecutive samplers: one for the anchor's free pick plus one
        // per draft position. The walk below compares each pick against
        // the corresponding draft.
        let sample_idx = if n_pending > 0 { n_pending - 1 } else { 0 };
        if !user_cleared_sampler {
            let wit: WitSampler = parent.sampler.clone().into();
            if do_sdk_verify {
                let indices: Vec<u32> = (sample_idx..=sample_idx + n_drafted).collect();
                pass.sampler(&indices, &wit);
            } else {
                pass.sampler(&[sample_idx], &wit);
            }
        }

        // Per-step probes (registered on the Generator).
        for (idx, wit) in &parent.step_probes {
            pass.sampler(&[*idx], wit);
        }
        // Per-call extra probes (registered on this GenStep).
        for (idx, wit) in &extra_probes {
            pass.sampler(&[*idx], wit);
        }

        if let Some(m) = &mask {
            pass.logit_mask(m);
        }

        let raw = pass
            .execute_async()
            .await
            .map_err(|e| format!("GenStep::execute forward: {e}"))?;

        // Read accepted tokens off the auto-sampler slots.
        //
        // System-mode and non-spec passes: the verifier (or single-shot
        // sampler) emits one Token slot per accept, so the leading
        // Token-slot run from index 0 is exactly the accepted chain.
        //
        // Custom-with-drafts (do_sdk_verify): walk the (1 + n_drafted)
        // sampled picks against `drafts`. Slot 0 is the anchor's free
        // pick (always accepted). Each subsequent slot is accepted iff
        // the previous draft matched its picked token. The final slot
        // is either the next verification anchor or, after a full-
        // accept run, the post-draft free pick.
        let accepted_tokens: Vec<u32> = if user_cleared_sampler {
            Vec::new()
        } else if do_sdk_verify {
            let n_picks = (n_drafted + 1) as usize;
            let lm_picks: Vec<u32> = raw
                .slots
                .iter()
                .take(n_picks)
                .filter_map(|s| match s {
                    SlotOutput::Token(t) => Some(*t),
                    _ => None,
                })
                .collect();
            if lm_picks.len() != n_picks {
                return Err(format!(
                    "GenStep::execute verify: expected {} Token slots, got {}",
                    n_picks,
                    lm_picks.len()
                ));
            }
            let mut accepted = Vec::with_capacity(n_picks);
            accepted.push(lm_picks[0]);
            for k in 0..n_drafted as usize {
                if lm_picks[k] != drafts[k] {
                    break;
                }
                accepted.push(lm_picks[k + 1]);
            }
            accepted
        } else {
            raw.slots
                .iter()
                .take_while(|s| matches!(s, SlotOutput::Token(_)))
                .filter_map(|s| match s {
                    SlotOutput::Token(t) => Some(*t),
                    _ => None,
                })
                .collect()
        };

        // Stash next-iter system drafts (and let custom speculators see
        // accepted tokens).
        match &mut parent.speculation {
            SpecMode::None => {}
            SpecMode::System {
                spec_tokens,
                spec_positions,
            } => {
                *spec_tokens = raw.spec_tokens.clone();
                *spec_positions = raw.spec_positions.clone();
            }
            SpecMode::Custom(s) => {
                s.accept(&accepted_tokens);
            }
        }

        // Truncate rejected drafts.
        if n_drafted > 0 {
            let n_verified = (accepted_tokens.len() as u32).saturating_sub(1);
            let n_rejected = n_drafted.saturating_sub(n_verified);
            if n_rejected > 0 {
                parent.ctx.inner.truncate_working_page_tokens(n_rejected);
            }
            // Roll back custom speculator's own state too.
            if let SpecMode::Custom(s) = &mut parent.speculation {
                if n_rejected > 0 {
                    s.rollback(n_rejected);
                }
            }
        }

        // Commit pages: pending tokens always commit (they're real KV);
        // verified drafts also commit (they survived the verifier).
        let n_verified_drafts = if n_drafted > 0 {
            (accepted_tokens.len() as u32).saturating_sub(1)
        } else {
            0
        };
        let n_kv_tokens = n_pending + n_verified_drafts;
        if n_kv_tokens > 0 {
            let new_working = parent.ctx.working_tokens + n_kv_tokens;
            let pages_to_commit = new_working / parent.ctx.page_size;
            if pages_to_commit > 0 {
                parent.ctx
                    .inner
                    .commit_working_pages(pages_to_commit)
                    .map_err(|e| format!("GenStep::execute commit: {e}"))?;
            }
            parent.ctx.committed_pages += pages_to_commit;
            parent.ctx.working_pages -= pages_to_commit;
            parent.ctx.working_tokens = new_working % parent.ctx.page_size;
            parent.ctx.seq_len += n_kv_tokens;
        } else if n_drafted > 0 && accepted_tokens.is_empty() {
            // All drafts rejected with no anchor token — re-sync from host
            // since truncation may have released pages.
            parent.ctx.committed_pages = parent.ctx.inner.committed_page_count();
            parent.ctx.working_pages = parent.ctx.inner.working_page_count();
            parent.ctx.working_tokens = parent.ctx.inner.working_page_token_count();
            parent.ctx.seq_len =
                parent.ctx.committed_pages * parent.ctx.page_size + parent.ctx.working_tokens;
        }

        // Advance constraint state with the accepted tokens (read by the
        // next iteration's mask compute).
        if !parent.constraints.is_empty() {
            parent.constraint_pending.extend_from_slice(&accepted_tokens);
        }

        // Truncate at stop / max_tokens, accumulate counters, seed buffer.
        let mut tokens = accepted_tokens;
        if let Some(pos) = tokens.iter().position(|t| parent.stop.contains(t)) {
            tokens.truncate(pos);
            parent.done = true;
        }
        if let Some(max) = parent.max_tokens {
            let remaining = max.saturating_sub(parent.tokens_generated);
            if tokens.len() > remaining {
                tokens.truncate(remaining);
                parent.done = true;
            }
        }
        parent.tokens_generated += tokens.len();
        if let Some(&last) = tokens.last() {
            parent.ctx.buffer.push(last);
        }

        // Generator owns slot 0 for its auto-attached sampler. The
        // post-truncation tokens land on `Output::tokens` (the common
        // path); callers wanting the raw verifier accepted tokens can
        // walk slots from `output.auto_sampler()`.
        let auto = if user_cleared_sampler {
            None
        } else {
            Some(SampleHandle::new(0, 1))
        };
        Ok(crate::forward::Output::from_generator(raw, tokens, auto))
    }
}
