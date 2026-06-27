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
use crate::pie::core::inference::ForwardPass;
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

    /// One-ahead **run-ahead** decode (Seam A §2.1 carryover): instead of the
    /// guest reading each sampled token back and re-feeding it, the device-side
    /// carrier carries a producer pass's sampled token directly into the next
    /// pass's input. The producer declares the destination via
    /// [`next_inputs(positions)`](crate::forward::Forward::next_inputs) (host
    /// owns the link + src-row); the consumer supplies a placeholder token at
    /// that slot which the carrier overwrites pre-forward.
    ///
    /// **Two-in-flight overlap (1c):** the step-t+1 consumer is eager-submitted
    /// (`execute()`) BEFORE step-t's `output()` is awaited, so t+1's host-prep +
    /// enqueue overlap t's GPU compute. The scheduler fires t+1 in a *later*
    /// batch (its carrier inject reads t's retained sample; the dependency-aware
    /// accumulate refuses co-batch). The submit/await split rides the sync-
    /// `execute()` / async-`output()` surface: `submit_*` eager-submits and
    /// advances the cursor (so an overlapped consumer reserves the following slot
    /// while the producer is still in flight), `await_commit` awaits + records
    /// history.
    ///
    /// **Drop-on-terminate (R9):** no successor is speculated on a terminating
    /// step — at the max-tokens boundary (count-predictable) no consumer is
    /// submitted, so nothing is committed beyond the emitted stream. Stop-token
    /// decode runs non-speculatively (sequential, correct).
    ///
    /// Falls back to [`collect_tokens`](Self::collect_tokens) for constrained /
    /// speculative decode, where step t+1's input depends on token t.
    pub async fn collect_tokens_pipelined(mut self) -> Result<Vec<u32>> {
        if !self.constraints.is_empty() || !matches!(self.speculation, SpecMode::None) {
            return self.collect_tokens().await;
        }
        if self.done {
            return Ok(Vec::new());
        }
        self.ensure_program()?;

        // Prime producer: the pending prompt tail. It carries its sampled token
        // into the first consumer via `next-inputs`.
        let pending = core::mem::take(&mut self.ctx.buffer);
        if pending.is_empty() {
            return Ok(Vec::new());
        }
        let mut producer = self.submit_producer(&pending)?;

        let mut all = Vec::new();
        loop {
            // Speculate the next consumer (eager one-ahead submit) BEFORE awaiting
            // the producer — the overlap — UNLESS this step may terminate: a stop
            // is configured (predictable only post-sample) or the next kept token
            // would hit max-tokens (R9: no successor beyond a terminating step).
            let speculate = self.stop.is_empty()
                && self
                    .max_tokens
                    .is_none_or(|m| self.tokens_generated + 1 < m);
            let consumer = if speculate {
                Some(self.submit_consumer()?)
            } else {
                None
            };

            // Await the producer's sampled token, overlapped with the consumer's
            // in-flight compute (the carrier injects this token into the consumer).
            let token = self.await_commit(producer).await?;

            if self.stop.contains(&token) {
                // No consumer was speculated (stop non-empty) — clean break.
                break;
            }
            all.push(token);
            self.tokens_generated += 1;
            let hit_max = self.max_tokens.is_some_and(|m| self.tokens_generated >= m);

            match consumer {
                Some(mut c) => {
                    // The speculated successor carries this token (the producer's
                    // sample); record it as the successor's history token.
                    c.history_tokens = vec![token];
                    producer = c;
                }
                None => {
                    if hit_max {
                        break;
                    }
                    // Stop configured but not hit → decode the next step
                    // sequentially (no overlap this step).
                    let mut c = self.submit_consumer()?;
                    c.history_tokens = vec![token];
                    producer = c;
                }
            }
        }
        Ok(all)
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

    // ── Run-ahead carryover helpers (Seam A §2.1) ──────────────────────

    /// Eager-submit the prime **producer** pass over `pending` (the prompt tail).
    /// Declares the carrier destination (`next-inputs`) at the next slot, advances
    /// the cursor on submit, and returns the in-flight handle; the sampled token
    /// is read later by [`await_commit`](Self::await_commit).
    fn submit_producer(&mut self, pending: &[u32]) -> Result<InflightPass> {
        let n = pending.len() as u32;
        let pass = ForwardPass::new();
        let w = self.ctx.prepare_write(n)?;
        self.ctx.attach_kv(&pass, &w);
        if let Some(a) = self.adapter {
            pass.adapter(a);
        }
        if let Some(seed) = self.zo_seed {
            crate::pie::zo::zo::adapter_seed(&pass, seed);
        }
        let positions: Vec<u32> = (self.ctx.seq_len..self.ctx.seq_len + n).collect();
        pass.input_tokens(pending, &positions);
        let decode_pos = self.ctx.seq_len + n - 1;
        self.attach_pipelined_sampler(&pass, decode_pos)?;
        // Carry this pass's sample into the next (consumer) pass's input ROW slot.
        // The consumer is a single-token decode (`input_tokens([0], [pos])`), so its
        // carried token lands at dense `pi.tokens` row 0 — the carrier dest is the
        // ROW index into the consumer forward, NOT the sequence position (the driver
        // INJECT does `pi.tokens[dest] = retained[src_row]` over the dense per-row
        // input buffer). Batched multi-seq run-ahead → per-sequence rows [0..B);
        // single-sequence one-ahead → row [0].
        pass.next_inputs(&[0]);
        pass.execute();
        // Advance the cursor on SUBMIT so an overlapped consumer reserves the
        // following slot while this pass is still in flight.
        self.ctx.seq_len += n;
        Ok(InflightPass {
            pass,
            history_tokens: pending.to_vec(),
        })
    }

    /// Eager-submit a **consumer** pass: a placeholder `0` token at the carried
    /// slot (the carrier overwrites it pre-forward with the producer's sample),
    /// the carrier destination for the next step, and advance the cursor.
    /// Returns the in-flight handle; the carried token (== the producer's sample)
    /// is recorded into `history_tokens` by the loop once the producer is read.
    fn submit_consumer(&mut self) -> Result<InflightPass> {
        let pass = ForwardPass::new();
        let w = self.ctx.prepare_write(1)?;
        self.ctx.attach_kv(&pass, &w);
        if let Some(a) = self.adapter {
            pass.adapter(a);
        }
        if let Some(seed) = self.zo_seed {
            crate::pie::zo::zo::adapter_seed(&pass, seed);
        }
        let pos = self.ctx.seq_len;
        pass.input_tokens(&[0u32], &[pos]);
        self.attach_pipelined_sampler(&pass, pos)?;
        // Dest = the next consumer's input ROW slot (row 0 of its single-token
        // decode), not the sequence position. See `submit_producer`.
        pass.next_inputs(&[0]);
        pass.execute();
        self.ctx.seq_len += 1;
        Ok(InflightPass {
            pass,
            history_tokens: Vec::new(),
        })
    }

    /// Await an in-flight pass's single-output `output()` (finalize), record its
    /// history tokens, and return its sampled token. The cursor already advanced
    /// at submit.
    async fn await_commit(&mut self, inflight: InflightPass) -> Result<u32> {
        let InflightPass {
            pass,
            history_tokens,
        } = inflight;
        let t = pass
            .output()
            .await
            .map_err(|e| format!("collect_tokens_pipelined output: {e}"))?;
        self.ctx.history.extend_from_slice(&history_tokens);
        first_token(&t)
    }

    /// Attach the cached sampler program with its `Logits` bound to `decode_pos`
    /// (mirrors [`GenStep::execute`]'s attach so the pipelined path is
    /// numerically identical to the synchronous one).
    fn attach_pipelined_sampler(&self, pass: &ForwardPass, decode_pos: u32) -> Result<()> {
        let (program, template, host_inputs, _n_out, submit_values) = self
            .program_cache
            .as_ref()
            .ok_or_else(|| "collect_tokens_pipelined: sampler program not lowered".to_string())?;
        let bindings =
            crate::program::resolve_bindings(template, host_inputs, &[decode_pos], submit_values)?;
        pass.sampler(program, bindings);
        Ok(())
    }
}

/// Read the single sampled token (first `u32`/`i32` lane) from an output tensor.
fn first_token(t: &crate::tensor::Tensor) -> Result<u32> {
    let bytes = t
        .read()
        .map_err(|e| format!("collect_tokens_pipelined: tensor read: {e}"))?;
    bytes
        .chunks_exact(4)
        .next()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .ok_or_else(|| "collect_tokens_pipelined: empty output tensor".to_string())
}

/// An eager-submitted forward pass still in flight — the run-ahead one-ahead
/// handle. Owns the `ForwardPass` resource (the host's `PendingForward` rides
/// it) plus the deferred history tokens, and holds NO borrow of the context, so
/// the loop can keep a producer and its speculated consumer in flight at once.
struct InflightPass {
    pass: ForwardPass,
    /// Tokens recorded into history when the pass's `output()` is awaited — the
    /// producer's prompt tail, or a consumer's carried token (set by the loop
    /// once the producer is read).
    history_tokens: Vec<u32>,
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
        let w = parent.ctx.prepare_write(n_pending)?;
        parent.ctx.attach_kv(&pass, &w);
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
            // No-sampler fill: eager submit; no output tensors to await.
            pass.execute();
            None
        } else {
            parent.ensure_program()?;
            let decode_pos = parent.ctx.seq_len + n_pending - 1;
            let n_out;
            {
                let (program, template, host_inputs, n, submit_values) =
                    parent.program_cache.as_ref().unwrap();
                n_out = *n;
                let bindings = crate::program::resolve_bindings(
                    template,
                    host_inputs,
                    &[decode_pos],
                    submit_values,
                )?;
                pass.sampler(program, bindings);
            }
            pass.execute();
            // The sampled token is the first declared output: single-output
            // samplers use `output()`; multi-output programs use `outputs()`.
            let t = if n_out <= 1 {
                pass.output()
                    .await
                    .map_err(|e| format!("GenStep::execute output: {e}"))?
            } else {
                pass.outputs()
                    .await
                    .map_err(|e| format!("GenStep::execute outputs: {e}"))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| "GenStep::execute: sampler produced no output".to_string())?
            };
            Some(first_token(&t)?)
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
