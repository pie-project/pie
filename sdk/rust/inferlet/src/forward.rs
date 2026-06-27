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

use crate::ForwardPassExt;
use crate::Result;
use crate::adapter::Adapter;
use crate::context::Context;
use crate::pie::core::inference::{ForwardPass, InputBinding};
use crate::program::ProgramHandle;
use crate::sample::Sampler;
use crate::tensor;
use sampling_edsl::ir;

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
    /// Sugar ([`Forward::sample`]): the lowered program's binding template,
    /// whose `Logits` slot is bound to the decode position at execute.
    SugarTemplate(Vec<ir::Binding>),
}

/// One tensor-program attached as the pass's sampler, kept until `execute`.
/// `bindings[i]` supplies the program input at index `i` at attach time
/// (`logits(positions)` or a device `tensor`).
struct SamplerAttach<'p> {
    program: AttachedProgram<'p>,
    bindings: SamplerBindings,
}

/// Lower the ergonomic [`Sampler`] enum to a tensor [`Program`](tensor::Program)
/// plus its binding-free [`Binding`](ir::Binding) template and output count.
/// Routes through foxtrot's guest emit ([`crate::emit::emit_program`]) over
/// alpha's zero-drift `op-kind` oracle. A sugar sampler binds only its `Logits`
/// input and emits a single `Token` output.
fn lower_sampler_to_program(
    sampler: &Sampler,
    vocab: u32,
) -> Result<(tensor::Program, Vec<ir::Binding>, u32)> {
    let spec: sampling_edsl::SamplerSpec = sampler.clone().into();
    let built = sampling_edsl::build_sampler(spec, vocab)
        .map_err(|e| format!("Forward::sample: lower sampler: {e:?}"))?;
    let program = crate::emit::emit_program(&built.program)?;
    Ok((program, built.bindings, built.outputs.len() as u32))
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
        });
        (0..outputs).map(ProgramHandle::new).collect()
    }

    /// Ergonomic sugar: attach the legacy [`Sampler`] enum, lowered to a tensor
    /// program for `vocab` (foxtrot's guest emit). The program's `Logits` input
    /// is bound to the pass's decode position (the last auto-input token), so
    /// call after [`input`](Self::input). Returns one [`ProgramHandle`] per
    /// output (a sugar sampler yields a single `Token`); read it with
    /// [`Output::token`].
    pub fn sample(&mut self, sampler: Sampler, vocab: u32) -> Result<Vec<ProgramHandle>> {
        let (program, template, outputs) = lower_sampler_to_program(&sampler, vocab)?;
        self.sampler = Some(SamplerAttach {
            program: AttachedProgram::Owned(program),
            bindings: SamplerBindings::SugarTemplate(template),
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
            let pages_needed = total_after.div_ceil(ctx.page_size);
            let additional = pages_needed.saturating_sub(ctx.working_pages);
            if additional > 0 {
                ctx.inner
                    .reserve_working_pages(additional)
                    .map_err(|e| format!("Forward::execute: reserve_working_pages: {e}"))?;
                ctx.working_pages = pages_needed;
            }
        }

        let pass = ForwardPass::new();
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

        if let Some(m) = attn_mask {
            pass.attention_mask(&m);
        }

        // Attach the tensor-program sampler (if any). Binding `i` supplies the
        // program input at index `i`; results follow the program's declared
        // output order. A no-sampler pass (prefill/flush) carries no sampler
        // and `output()` returns an empty tensor list.
        if let Some(SamplerAttach { program, bindings }) = sampler {
            let resolved = match bindings {
                SamplerBindings::Resolved(b) => b,
                SamplerBindings::SugarTemplate(template) => {
                    // Sugar: bind the program's `Logits` slot to the decode
                    // position (the last auto-input token). Requires an input.
                    if n_auto == 0 {
                        return Err("Forward::sample: no input tokens — a sampler \
                            needs a decode position; call `input(...)` before `sample(...)`."
                            .to_string());
                    }
                    let decode_pos = ctx.seq_len + n_auto - 1;
                    crate::program::resolve_bindings(&template, &[], &[decode_pos], &[])?
                }
            };
            pass.sampler(program.get(), resolved);
        }

        let tensors = pass
            .execute_outputs()
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

    /// Number of program output tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// True when the pass produced no output tensors (e.g. a prefill/flush).
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Borrow the output tensor for a handle (declared output order), or `None`
    /// if the index is out of range.
    pub fn tensor(&self, h: ProgramHandle) -> Option<&tensor::Tensor> {
        self.tensors.get(h.index() as usize)
    }

    /// Read a program output as raw little-endian bytes.
    pub async fn read_bytes(&self, h: ProgramHandle) -> Result<Vec<u8>> {
        let t = self
            .tensors
            .get(h.index() as usize)
            .ok_or_else(|| format!("Output: no output tensor at index {}", h.index()))?;
        t.read()
            .map_err(|e| format!("Output::read_bytes: {e:?}"))
    }

    /// Read a `u32` program output (e.g. a sampled-token vector).
    pub async fn read_u32(&self, h: ProgramHandle) -> Result<Vec<u32>> {
        Ok(cast_le_u32(&self.read_bytes(h).await?))
    }

    /// Read an `f32` program output (e.g. entropy, a distribution row).
    pub async fn read_f32(&self, h: ProgramHandle) -> Result<Vec<f32>> {
        Ok(cast_le_f32(&self.read_bytes(h).await?))
    }

    /// First `u32` lane of a program output — the common single-token case.
    pub async fn token(&self, h: ProgramHandle) -> Result<u32> {
        self.read_u32(h)
            .await?
            .first()
            .copied()
            .ok_or_else(|| "Output::token: empty output tensor".to_string())
    }

    /// First `f32` lane of a program output — the common scalar case.
    pub async fn scalar(&self, h: ProgramHandle) -> Result<f32> {
        self.read_f32(h)
            .await?
            .first()
            .copied()
            .ok_or_else(|| "Output::scalar: empty output tensor".to_string())
    }
}

/// Reinterpret a little-endian byte buffer as `u32` lanes (trailing partial
/// lane, if any, is dropped).
fn cast_le_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Reinterpret a little-endian byte buffer as `f32` lanes (trailing partial
/// lane, if any, is dropped).
fn cast_le_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod cast_tests {
    use super::{cast_le_f32, cast_le_u32};

    #[test]
    fn u32_round_trips_little_endian() {
        assert_eq!(cast_le_u32(&[1, 2, 3, 4]), vec![0x04030201]);
        assert_eq!(cast_le_u32(&42u32.to_le_bytes()), vec![42]);
        // Trailing partial lane dropped.
        assert_eq!(cast_le_u32(&[1, 0, 0, 0, 9]), vec![1]);
    }

    #[test]
    fn f32_round_trips_little_endian() {
        let bytes: Vec<u8> = [1.0f32, -2.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(cast_le_f32(&bytes), vec![1.0, -2.5]);
    }
}
