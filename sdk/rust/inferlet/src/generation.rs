//! `Generator` — multi-step token-generation state machine.
//!
//! A [`Generator`] is configured up front (sampler, max-tokens, stop set,
//! constraints, optional speculator/adapter) and then iterated step-by-
//! step. Each call to [`Generator::next`] yields a [`GenStep`] — a
//! short-lived configuration handle for the upcoming forward pass that
//! the user can tweak — and [`GenStep::execute`] runs the host call and the
//! loop's post-step bookkeeping.
//!
//! Three terminal sugars cover the common case:
//!
//! - [`Generator::collect_tokens`] — drains until done, returns all tokens.
//! - [`Generator::collect_text`] — drains, decodes through a chat decoder.
//! - [`Generator::collect_json`] — adds a `T`-derived JSON-schema constraint.
//!
//! # Stage 1 (WIT reconciliation window)
//!
//! The program-based front door retired the hardwired WIT `sampler` variant,
//! the slot-output decode, the probe slots, and the speculative side-channel.
//! Sampling now flows through a tensor [`Program`](crate::tensor::Program).
//! Lowering the ergonomic [`Sampler`] enum to a program is **foxtrot's guest
//! emit (Stage 2)** — until it lands, [`GenStep::execute`] surfaces a clear
//! error at the sampler-attach seam. The builder/iteration API is intact so
//! dependent modules compile; speculation and probe ergonomics are restored in
//! Stage 2 on the program model. Program-native single-pass authoring is fully
//! available now via [`Forward::sampler`](crate::forward::Forward::sampler).

use crate::Result;
use crate::adapter::Adapter;
use crate::context::{Context, brle_and};
use crate::forward::Output;
use crate::pie::core::inference::{ForwardPass, SlotOutput};
use crate::sample::Sampler;
use crate::spec::Speculator;
use std::collections::VecDeque;

// Re-export so callers don't have to pull from `context` directly.
pub use crate::context::{Constrain, GrammarConstraint, Schema};

// =============================================================================
// Speculation mode
// =============================================================================

/// Speculative-decoding mode. Retained as builder state across the Stage-1
/// window; the per-step draft/verify path is restored in Stage 2 on the
/// program model (the speculative WIT side-channel was retired with the
/// hardwired sampler).
enum SpecMode {
    None,
    System,
    Custom(Box<dyn Speculator>),
}

// =============================================================================
// Generator
// =============================================================================

/// Builder + iterator for token generation. See module docs.
pub struct Generator<'ctx> {
    ctx: &'ctx mut Context,
    sampler: Sampler,
    /// Lazily lowered sampler program (foxtrot's guest emit): the reusable
    /// `tensor::Program`, its binding-free `Binding` template, and the output
    /// count. Compiled once on the first decode step and reused across steps
    /// (the program is binding-free, so only the attach-time bindings change).
    program_cache: Option<crate::forward::LoweredSampler>,
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
    output_buffer: VecDeque<u32>,
    tokens_generated: usize,
    rebid_each_step: bool,
    done: bool,
}

impl<'ctx> Generator<'ctx> {
    /// Construct a generator over `ctx` with the given sampler. Prefer
    /// [`Context::generate`].
    pub(crate) fn new(ctx: &'ctx mut Context, sampler: Sampler) -> Self {
        Self {
            ctx,
            sampler,
            program_cache: None,
            stop: Vec::new(),
            max_tokens: None,
            horizon: None,
            constraints: Vec::new(),
            constraint_pending: Vec::new(),
            speculation: SpecMode::None,
            adapter: None,
            zo_seed: None,
            output_buffer: VecDeque::new(),
            tokens_generated: 0,
            rebid_each_step: true,
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
        let c = schema.build_constraint()?;
        self.constraints.push(Box::new(c));
        Ok(self)
    }

    /// Use a custom drafter for speculative decoding.
    pub fn speculator<S: Speculator + 'static>(mut self, s: S) -> Self {
        self.speculation = SpecMode::Custom(Box::new(s));
        self
    }

    /// Enable host-driven speculation.
    pub fn system_speculation(mut self) -> Self {
        self.speculation = SpecMode::System;
        self
    }

    /// Disable host-driven system speculation, leaving custom speculators
    /// untouched.
    pub fn disable_system_speculation(mut self) -> Self {
        if matches!(self.speculation, SpecMode::System) {
            self.speculation = SpecMode::None;
        }
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

    /// Control whether the generator refreshes its scheduler bid before
    /// every decode step (default `true`).
    pub fn rebid_each_step(mut self, enabled: bool) -> Self {
        self.rebid_each_step = enabled;
        self
    }

    // ── Iteration ──────────────────────────────────────────────────────

    /// Whether generation has terminated (max_tokens or stop hit).
    pub fn is_done(&self) -> bool {
        self.done
            || self
                .max_tokens
                .is_some_and(|m| self.tokens_generated >= m)
    }

    /// Tokens generated so far across all steps.
    pub fn tokens_generated(&self) -> usize {
        self.tokens_generated
    }

    /// Convenience wrapper for callers that only need the next sampled token.
    pub async fn next_token(&mut self) -> Result<Option<u32>> {
        if let Some(token) = self.output_buffer.pop_front() {
            return Ok(Some(token));
        }

        loop {
            let Some(step) = self.next()? else {
                return Ok(None);
            };
            let out = step.execute().await?;
            self.output_buffer.extend(out.tokens);
            if let Some(token) = self.output_buffer.pop_front() {
                return Ok(Some(token));
            }
            if self.is_done() {
                return Ok(None);
            }
        }
    }

    /// Begin the next step. Returns `Ok(None)` when generation is finished.
    /// The returned [`GenStep`] borrows the generator mutably; complete it
    /// with [`GenStep::execute`] (or drop it to skip the iteration).
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<GenStep<'_, 'ctx>>> {
        if self.is_done() {
            return Ok(None);
        }

        // Re-bid using the horizon cascade.
        if self.rebid_each_step {
            self.recompute_bid();
        }

        // Drain the context's buffer (filled by `system / user / cue / …`).
        let pending = std::mem::take(&mut self.ctx.buffer);

        // Compose constraint masks. Each constraint advances on the tokens
        // accepted last step, then yields its next-position mask. (Stage 1:
        // the composed mask is computed but not yet applied — application is
        // program-side in Stage 2.)
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
            mask,
            user_cleared_sampler: false,
        }))
    }

    /// Register a manually-sampled token (or sequence) with the generator.
    /// Updates max-tokens / stop / constraint counters and seeds the next
    /// iteration's input. The token doesn't enter KV here; the next
    /// `next() / execute()` flushes it through a forward pass.
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
        let mut decoder = chat::Decoder::new();
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
        let constraint = GrammarConstraint::from_json_schema(&schema_str)?;
        let text = self.constrain(constraint).collect_text().await?;
        serde_json::from_str(&text).map_err(|e| format!("collect_json: deserialize: {e}"))
    }

    // ── Internal helpers ───────────────────────────────────────────────

    // P3 is single-model FCFS — no scheduler bidding. Retained as a no-op so the
    // per-step call sites stay stable if the bid path ever returns.
    fn recompute_bid(&mut self) {}

    /// Lazily lower the configured [`Sampler`] to a reusable, binding-free
    /// `tensor::Program` (+ its binding template + output count) via foxtrot's
    /// guest emit, caching it for reuse across decode steps.
    fn ensure_program(&mut self) -> Result<()> {
        if self.program_cache.is_none() {
            // Lower over the model's LOGITS vocab (output-vocab-size, e.g.
            // 151936 for qwen3-0.6b), NOT the tokenizer token count — the
            // sampler operates on logits, and the host recognizer + bake key on
            // the logits vocab. (Using tokenizer vocab mis-shapes the program
            // and misses the recognizer → CustomJIT.)
            let vocab = crate::model::output_vocab_size();
            let spec: sampling_edsl::SamplerSpec = self.sampler.clone().into();
            // (B) de-hardwiring: emit the canonical `standard_program` (params
            // as host-submit tensors) so the host recognizer hash-matches;
            // `submit_values` carry the per-fire params (T / top-p / min-p),
            // reused across decode steps.
            let (built, submit_values) = sampling_edsl::lower_sampler_standard(spec, vocab)
                .map_err(|e| format!("Generator: lower sampler: {e:?}"))?;
            let program = crate::emit::emit_program(&built.program)?;
            self.program_cache = Some((
                program,
                built.bindings,
                built.host_inputs,
                built.outputs.len() as u32,
                submit_values,
            ));
        }
        Ok(())
    }

    // P3's working-set model auto-manages KV page lifetime (the working set frees
    // its own pages); the SDK no longer tracks/releases working pages here.
    fn release_empty_working_pages(&mut self) {}
}

impl Drop for Generator<'_> {
    fn drop(&mut self) {
        self.release_empty_working_pages();
    }
}

// =============================================================================
// GenStep — short-lived per-iteration handle
// =============================================================================

/// Configuration handle for the upcoming forward pass. Returned by
/// [`Generator::next`]. Pre-populated with the generator's pending fills,
/// configured sampler, and the composed constraint mask. Complete it with
/// [`execute`](Self::execute).
pub struct GenStep<'g, 'ctx> {
    parent: &'g mut Generator<'ctx>,
    pending: Vec<u32>,
    #[allow(dead_code)]
    mask: Option<Vec<u32>>,
    user_cleared_sampler: bool,
}

impl<'g, 'ctx> GenStep<'g, 'ctx> {
    /// Drop the generator's auto-attached sampler. The caller samples by hand
    /// off a probe/program and registers their pick via
    /// [`Generator::accept`] after `execute`.
    pub fn clear_sampler(&mut self) -> &mut Self {
        self.user_cleared_sampler = true;
        self
    }

    /// Run the forward pass and fold the result into the generator's state.
    pub async fn execute(self) -> Result<Output> {
        let GenStep {
            parent,
            pending,
            mask: _,
            user_cleared_sampler,
        } = self;

        let n_pending = pending.len() as u32;
        if n_pending == 0 {
            // No input — no query position for a sampler to land on. Mark done
            // so the iteration loop terminates cleanly.
            parent.done = true;
            return Ok(Output::new(Vec::new()));
        }

        // Build the forward pass over P3's working-set KV model: the pending
        // fill writes fresh tail slots; prior pages are the read context.
        let pass = ForwardPass::new();
        let (generation, indices, valid_lens, ctx_pages) = parent.ctx.prepare_write(n_pending)?;
        parent
            .ctx
            .attach_kv(&pass, generation, indices, valid_lens, ctx_pages);
        if let Some(a) = parent.adapter {
            pass.adapter(a);
        }
        if let Some(seed) = parent.zo_seed {
            crate::pie::zo::zo::adapter_seed(&pass, seed);
        }

        let positions: Vec<u32> =
            (parent.ctx.seq_len..parent.ctx.seq_len + n_pending).collect();
        pass.input_tokens(&pending, &positions);

        // Sampler attach + execute. The configured `Sampler` is lowered once
        // (lazily, then reused) to a binding-free tensor program; each step
        // binds its `Logits` input to the decode position. The sampler-cleared
        // path runs a pure fill (the caller samples by hand + `accept`s).
        let sampled: Option<u32> = if user_cleared_sampler {
            pass.execute()
                .await
                .map_err(|e| format!("GenStep::execute forward: {e}"))?;
            None
        } else {
            parent.ensure_program()?;
            let decode_pos = parent.ctx.seq_len + n_pending - 1;
            {
                let (program, template, host_inputs, _n_out, submit_values) =
                    parent.program_cache.as_ref().unwrap();
                let bindings = crate::program::resolve_bindings(
                    template,
                    host_inputs,
                    &[decode_pos],
                    submit_values,
                )?;
                pass.sampler(program, bindings);
            }
            let wit_out = pass
                .execute()
                .await
                .map_err(|e| format!("GenStep::execute forward: {e}"))?;
            let token = match wit_out.slots.first() {
                Some(SlotOutput::Token(t)) => *t,
                Some(other) => {
                    return Err(format!(
                        "GenStep::execute: sampler slot is {other:?}, not a token"
                    ));
                }
                None => {
                    return Err("GenStep::execute: sampler produced no output slot".to_string());
                }
            };
            Some(token)
        };

        // Advance the sequence cursor past the materialized fill. Full pages
        // auto-seal host-side on the forward-txn commit — no explicit page
        // commit under the P3 working-set model.
        parent.ctx.history.extend_from_slice(&pending);
        parent.ctx.seq_len += n_pending;

        // Fold the sampled token into generator state (stop / max-tokens /
        // constraints / next-step buffer), mirroring `accept`.
        let mut tokens = Vec::new();
        if let Some(token) = sampled {
            if parent.stop.contains(&token) {
                parent.done = true;
            } else {
                tokens.push(token);
                parent.tokens_generated += 1;
                parent.constraint_pending.push(token);
                parent.ctx.buffer.push(token);
                if let SpecMode::Custom(s) = &mut parent.speculation {
                    s.accept(&tokens);
                }
                if parent
                    .max_tokens
                    .is_some_and(|max| parent.tokens_generated >= max)
                {
                    parent.done = true;
                }
            }
        }

        Ok(Output::from_generator(Vec::new(), tokens))
    }
}
