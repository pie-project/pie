//! `Forward` — a single forward-pass primitive with automatic page management.
//!
//! `Forward` wraps the WIT `forward-pass` resource and folds in the page math
//! that every consumer would otherwise repeat: working-page reservation
//! before submission, position derivation, page commit after execution.
//! Samplers are attached via [`Forward::sample`] / [`Forward::sampler`] and
//! their output tensors read back through typed handles on the returned
//! [`Output`].
//!
//! For the rare case where the caller needs full WIT control (custom
//! attention masks, manual page management for speculation rollback, etc.),
//! the underlying `forward-pass` resource is still available via
//! `inferlet::inference::ForwardPass`.


use crate::Result;
use crate::adapter::Adapter;
use crate::context::Context;
use crate::pie::core::inference::{ForwardPass, InputBinding};
use crate::program::{HostInputDecl, ProgramHandle};
use crate::sample::Sampler;
use crate::tensor;
use sampling_edsl::ir;

/// A lowered standard sampler ready to attach: the tensor program, its
/// binding-free template (`ir::Binding` per input slot), the host-input decls
/// (shape/dtype per submit slot), the output count, and the per-fire submit
/// values (standard-sampler params — T / top-p / min-p — keyed `T@0, filter@1`).
pub(crate) type LoweredSampler = (
    tensor::Program,
    Vec<ir::Binding>,
    Vec<HostInputDecl>,
    u32,
    Vec<(u32, Vec<u8>)>,
);

// =============================================================================
// Late-bound input references (Seam A)
// =============================================================================

/// A reference to a value produced *during* decoding and bound into a forward
/// pass after submit rather than at build time — the §3 run-ahead carrier
/// vocabulary. The high-level run-ahead loop translates this to a
/// [`Forward::next_inputs`] carrier declaration on the producing pass; the
/// host-late transform channel (grammar mask / mirostat mu) is supplied
/// directly via `tensor.write` on a handle bound at [`Forward::sampler`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenRef {
    /// The token sampled by the *previous* forward pass. The producer pass
    /// declares the carrier destination(s) via [`Forward::next_inputs`]; the
    /// consumer (the next pass) supplies a placeholder `input(0)` at each
    /// destination that the carrier overwrites pre-forward from the producer's
    /// retained sample (device path, zero host round-trip).
    PrevSample,
}

// =============================================================================
// Sampler attach
// =============================================================================

/// The program a sampler attach references: borrowed (program-native reuse —
/// the caller keeps the compiled program across passes) or owned (lowered from
/// a [`Sampler`] enum, lives only for this pass).
enum AttachedProgram<'p> {
    Borrowed(&'p tensor::Program),
    Owned(tensor::Program),
}

impl AttachedProgram<'_> {
    fn get(&self) -> &tensor::Program {
        match self {
            AttachedProgram::Borrowed(p) => p,
            AttachedProgram::Owned(p) => p,
        }
    }
}

/// How the attach's positional `input-binding` list is produced.
enum SamplerBindings {
    /// Fully resolved by the caller (program-native [`Forward::sampler`]).
    Resolved(Vec<InputBinding>),
    /// Sugar ([`Forward::sample`]): the lowered standard program's binding
    /// template + host-input decls + submit values (the per-fire standard-sampler
    /// params — temperature / top-p / min-p — bound as submit tensors). The
    /// `Logits` slot is bound to the decode position at execute.
    SugarTemplate {
        template: Vec<ir::Binding>,
        host_inputs: Vec<HostInputDecl>,
        submit_values: Vec<(u32, Vec<u8>)>,
    },
}

/// One tensor-program attached as the pass's sampler, kept until `execute`.
/// `bindings[i]` supplies the program input at index `i` at attach time
/// (`logits(positions)` or a device `tensor`); `outputs` is the program's
/// declared output-value count (single → `output()`, many → `outputs()`).
struct SamplerAttach<'p> {
    program: AttachedProgram<'p>,
    bindings: SamplerBindings,
    outputs: u32,
}

/// Lower the ergonomic [`Sampler`] enum to a tensor [`Program`](tensor::Program)
/// plus its binding-free [`Binding`](ir::Binding) template, host-input decls,
/// output count, and per-fire submit values. Routes through the **canonical
/// `standard_program`** lowering ([`sampling_edsl::lower_sampler_standard`]):
/// the standard-sampler params (T / top-p / min-p) are **host-submit tensors**
/// (`submit_values`), not baked immediates — so the bytecode is param-INVARIANT
/// and the host recognizer hash-matches it (the #12/#15 de-hardwiring contract).
fn lower_sampler_to_program(sampler: &Sampler, vocab: u32) -> Result<LoweredSampler> {
    let spec: sampling_edsl::SamplerSpec = sampler.clone().into();
    let (built, submit_values) = sampling_edsl::lower_sampler_standard(spec, vocab)
        .map_err(|e| format!("Forward::sample: lower sampler: {e:?}"))?;
    let program = crate::emit::emit_program(&built.program)?;
    Ok((
        program,
        built.bindings,
        built.host_inputs,
        built.outputs.len() as u32,
        submit_values,
    ))
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
    /// Optional tensor-program sampler attached via [`Forward::sampler`].
    /// Emitted as a `sampler(program, bindings)` call at execute time; its
    /// declared output tensors come back through [`Output`].
    sampler: Option<SamplerAttach<'ctx>>,
    attn_mask: Option<Vec<Vec<u32>>>,
    adapter: Option<&'ctx Adapter>,
    zo_seed: Option<i64>,
    /// Run-ahead carrier (§2.1): destination slots in the NEXT forward pass on
    /// this context that carry THIS pass's sampled token(s). Emitted as a single
    /// `next-inputs(positions)` call after the sampler; the source row is
    /// host-computed and the consumer pass is implicit. Set via
    /// [`Forward::next_inputs`].
    next_inputs: Vec<u32>,
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
    /// (when `pass.sampler(...)`) lands at `start_position() + i`.
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
            sampler: None,
            attn_mask: None,
            adapter: None,
            zo_seed: None,
            next_inputs: Vec::new(),
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

    // ── Sampler attach ─────────────────────────────────────────────────

    /// Attach a tensor [`Program`](tensor::Program) as this pass's sampler.
    /// `bindings[i]` supplies the program's input at index `i` at attach time:
    /// [`InputBinding::Logits(positions)`](crate::pie::core::inference::InputBinding)
    /// binds forward-pass output logits at those positions (shape `[vocab]` for
    /// one, `[len, vocab]` for many), or [`InputBinding::Tensor`] binds a
    /// device tensor (a per-fire value, mask, etc.).
    ///
    /// Returns one [`ProgramHandle`] per declared program output, in declared
    /// order; read them off the [`Output`] with `Output::read_*`. The program
    /// is **binding-free and reusable** — the same compiled program attaches
    /// across passes with different bindings.
    ///
    /// Only one sampler per pass (single-sampler `output()`); a second call
    /// replaces the first.
    pub fn sampler(
        &mut self,
        program: &'ctx tensor::Program,
        bindings: Vec<InputBinding>,
        outputs: u32,
    ) -> Vec<ProgramHandle> {
        self.sampler = Some(SamplerAttach {
            program: AttachedProgram::Borrowed(program),
            bindings: SamplerBindings::Resolved(bindings),
            outputs,
        });
        (0..outputs).map(ProgramHandle::new).collect()
    }

    /// Ergonomic sugar: attach the legacy [`Sampler`] enum, lowered to the
    /// canonical `standard_program` over the model's **output (logits) vocab**
    /// (derived internally via `output_vocab_size()` — a caller cannot supply
    /// the wrong vocab). The program's `Logits` input is bound to the pass's
    /// decode position (the last auto-input token), so call after
    /// [`input`](Self::input). Returns one [`ProgramHandle`] per output (a sugar
    /// sampler yields a single `Token`); read it with [`Output::token`].
    pub fn sample(&mut self, sampler: Sampler) -> Result<Vec<ProgramHandle>> {
        let vocab = crate::model::output_vocab_size();
        let (program, template, host_inputs, outputs, submit_values) =
            lower_sampler_to_program(&sampler, vocab)?;
        self.sampler = Some(SamplerAttach {
            program: AttachedProgram::Owned(program),
            bindings: SamplerBindings::SugarTemplate {
                template,
                host_inputs,
                submit_values,
            },
            outputs,
        });
        Ok((0..outputs).map(ProgramHandle::new).collect())
    }

    // ── Decoration ─────────────────────────────────────────────────────

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

    // ── Run-ahead carrier: §2.1 ────────────────────────────────────────

    /// Declare the destination slots in the **NEXT** forward pass on this
    /// context that carry **this** pass's sampled token(s) — `positions` are
    /// destinations in the next pass's coordinate space (the decode slots the
    /// carrier overwrites pre-forward). The guest names only destinations:
    /// the source row in this pass is host-computed (prefill → last row,
    /// single-row decode → row 0), and the consumer is implicit — the next pass
    /// supplies a placeholder `input(0)` at each position that the carrier
    /// overwrites from this pass's retained sample. Emitted after the sampler
    /// (WIT `forward-pass.next-inputs`).
    pub fn next_inputs(&mut self, positions: Vec<u32>) -> &mut Self {
        self.next_inputs = positions;
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
            sampler,
            attn_mask,
            adapter,
            zo_seed,
            next_inputs,
            images,
            audios,
            defer_commit,
        } = self;

        let n_auto = auto_inputs.len() as u32;

        // Soft-token KV rows contributed by image/audio spans occupy tail KV
        // slots just like text tokens.
        let mut soft_tokens = 0u32;
        for (image, _) in &images {
            soft_tokens += image.token_count();
        }
        for (audio, _) in &audios {
            soft_tokens += audio.token_count();
        }
        let n_write = n_auto + soft_tokens;
        // Tokens that yield a decode position: the auto/explicit text inputs. A
        // carried token rides as a placeholder `input(0)` supplied by the
        // run-ahead loop, so it is already counted in `n_auto`.
        let n_decode = n_auto;

        let pass = ForwardPass::new();

        // KV read/write descriptors. New tail tokens (auto inputs + soft
        // image/audio rows) are written into freshly-allocated slots, with the
        // prior full pages as read context. A pass with no new tail tokens
        // (pure decode-from-cache / scoring) reads the whole materialized
        // context read-only.
        if n_write > 0 {
            let w = ctx.prepare_write(n_write)?;
            ctx.attach_kv(&pass, &w);
        } else {
            ctx.attach_full_context(&pass);
        }

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

        if let Some(m) = attn_mask {
            pass.attention_mask(&m);
        }

        // Attach the tensor-program sampler (if any). Binding `i` supplies the
        // program input at index `i`; each program output value comes back as a
        // raw `tensor` via `output()` / `outputs()`, in declared output order. A
        // no-sampler pass (prefill/flush) carries no sampler and yields none.
        let mut sampler_outputs: Option<u32> = None;
        if let Some(SamplerAttach {
            program,
            bindings,
            outputs,
        }) = sampler
        {
            let resolved = match bindings {
                SamplerBindings::Resolved(b) => b,
                SamplerBindings::SugarTemplate {
                    template,
                    host_inputs,
                    submit_values,
                } => {
                    // Sugar: bind the program's `Logits` slot to the decode
                    // position (the last input/carried token). Requires a decode
                    // position. The standard-sampler params ride `submit_values`,
                    // resolved against `host_inputs` into submit tensors.
                    if n_decode == 0 {
                        return Err("Forward::sample: no decode position — a sampler \
                            needs an input token; call `input(...)` before \
                            `sample(...)`."
                            .to_string());
                    }
                    let decode_pos = ctx.seq_len + n_decode - 1;
                    crate::program::resolve_bindings(
                        &template,
                        &host_inputs,
                        &[decode_pos],
                        &submit_values,
                    )?
                }
            };
            pass.sampler(program.get(), resolved);
            sampler_outputs = Some(outputs);
        }

        // Run-ahead carrier (§2.1): declare the destination slots in the NEXT
        // pass that carry this pass's sampled token(s). Emitted after the
        // sampler (the source is its first output). The §2.2 host-late transform
        // channel (grammar mask / mirostat mu) is supplied separately by the
        // inferlet via `tensor.write` on a bound handle.
        if !next_inputs.is_empty() {
            pass.next_inputs(&next_inputs);
        }

        // Eager submit (synchronous fire); the result is awaited via `output()`
        // / `outputs()`.
        pass.execute();

        // Retrieve the sampler's declared output tensor(s): a single-output
        // sampler uses `output()`; a multi-output program (e.g. mirostat's
        // `[token, surprise]`) uses `outputs()`. A no-sampler pass yields none.
        let tensors = match sampler_outputs {
            Some(n) if n <= 1 => vec![
                pass.output()
                    .await
                    .map_err(|e| format!("Forward::execute: output: {e}"))?,
            ],
            Some(_) => pass
                .outputs()
                .await
                .map_err(|e| format!("Forward::execute: outputs: {e}"))?,
            None => Vec::new(),
        };

        // Advance the sequence cursor past the materialized tail tokens. Full
        // pages auto-seal host-side on the forward-txn commit; there is no
        // explicit page commit. `defer_commit` keeps the cursor where it was so
        // a manual speculative loop can `truncate` the rejected suffix (the
        // written KV is simply overwritten by the next pass).
        if n_write > 0 && !defer_commit {
            ctx.seq_len += n_write;
            // The auto inputs are the materialized text tokens; soft image/audio
            // rows make the context non-replayable.
            ctx.history.extend_from_slice(&auto_inputs);
        }
        if soft_tokens > 0 {
            ctx.snapshottable = false;
        }

        Ok(Output::new(tensors))
    }
}

// =============================================================================
// Output
// =============================================================================

/// Result of one forward-pass execution — produced by both
/// [`Forward::execute`] and [`GenStep::execute`](crate::generation::GenStep::execute).
///
/// Holds the attached sampler program's output **tensors**, in the program's
/// declared output order (the WIT `forward-pass.output()` list). Read them with
/// the typed `read_*` accessors using the [`ProgramHandle`]s returned by
/// [`Forward::sampler`]; reads are async because the tensors are device-resident
/// (`tensor.read()`).
///
/// A no-sampler pass (prefill / flush) carries no output tensors.
pub struct Output {
    /// The attached sampler program's output tensors, in declared output order.
    /// Each is a device-resident typed buffer; read via the typed accessors
    /// using the [`ProgramHandle`]s returned by [`Forward::sample`].
    tensors: Vec<tensor::Tensor>,
    /// Generator-accepted tokens this step, post stop / max-tokens truncation.
    /// Empty for a raw `Forward::execute` (no Generator state).
    pub tokens: Vec<u32>,
}

impl Output {
    pub(crate) fn new(tensors: Vec<tensor::Tensor>) -> Self {
        Self {
            tensors,
            tokens: Vec::new(),
        }
    }

    pub(crate) fn from_generator(tensors: Vec<tensor::Tensor>, tokens: Vec<u32>) -> Self {
        Self { tensors, tokens }
    }

    /// Number of sampler output tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// True when the pass produced no output tensors (e.g. a prefill/flush).
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Borrow the output tensor for a handle (declared output order), or
    /// `None` if the index is out of range.
    pub fn tensor(&self, h: ProgramHandle) -> Option<&tensor::Tensor> {
        self.tensors.get(h.index() as usize)
    }

    /// Raw little-endian bytes of an output tensor (downloads it host-side).
    pub async fn read_bytes(&self, h: ProgramHandle) -> Result<Vec<u8>> {
        let t = self
            .tensors
            .get(h.index() as usize)
            .ok_or_else(|| format!("Output::read_bytes: no output tensor at index {}", h.index()))?;
        t.read()
            .map_err(|e| format!("Output::read_bytes: tensor read failed: {e}"))
    }

    /// `u32` lanes of an output tensor — the sampled-token / index case.
    pub async fn read_u32(&self, h: ProgramHandle) -> Result<Vec<u32>> {
        let bytes = self.read_bytes(h).await?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// `f32` lanes of an output tensor — scalar / distribution measurement.
    pub async fn read_f32(&self, h: ProgramHandle) -> Result<Vec<f32>> {
        let bytes = self.read_bytes(h).await?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Sampled token id for a handle — the common single-token case (the first
    /// `u32`/`i32` lane of the output tensor).
    pub async fn token(&self, h: ProgramHandle) -> Result<u32> {
        self.read_u32(h)
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| format!("Output::token: output tensor at index {} is empty", h.index()))
    }

    /// Scalar measurement for a handle (e.g. mirostat surprise `S`) — the first
    /// `f32` lane of the output tensor.
    pub async fn scalar(&self, h: ProgramHandle) -> Result<f32> {
        self.read_f32(h)
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| format!("Output::scalar: output tensor at index {} is empty", h.index()))
    }
}

