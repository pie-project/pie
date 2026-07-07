//! Standard-sampler lowering — the keep-core primitive that turns a sampler
//! spec into an attachable tensor program plus its per-fire input bindings.
//!
//! This is the **sampler analog of [`geometry`](crate::geometry) /
//! [`carrier`](crate::carrier)**: a genuine primitive, NOT facade sugar. Without
//! it, every low-level inferlet that wants top-p / top-k / min-p / temperature
//! sampling would hand-build the per-kind Sampling-IR in raw EDSL — greedy is
//! four lines, but the parametric kinds are real per-kind programs, so the
//! corpus would carry the same lowering copied dozens of times. The ergonomic
//! `Sampler` enum + the `.generate()` decode-loop facade are the sugar that the
//! SDK-minimization pass deletes; THIS lowering (the param-invariant
//! `standard_program` bridge) is what stays (see `ptir-sdk-minimization-audit`).
//!
//! Kept as thin free functions over `sampling-edsl` (the IR/lowering crate) and
//! the raw WIT `inference` / `tensor` surface — no dependency on the deleted
//! `Context` / `Forward` / `program` facade modules.
//!
//! ## Usage (raw-WIT inferlet)
//! ```ignore
//! let s = sampler::sampler_program(SamplerSpec::TopP { temperature: 0.8, p: 0.9 }, vocab)?;
//! // per decode step, for the row at `decode_pos`:
//! pass.sampler(&s.program, s.bindings(decode_pos)?);
//! ```

use crate::inference::InputBinding;
use crate::tensor;
use crate::Result;
use sampling_edsl::{ir, DType, Graph, HostInputDecl, OutputKind, Readiness};

// Re-export the spec vocabulary so inferlets name it as `sampler::SamplerSpec`
// without a direct `sampling-edsl` dependency.
pub use sampling_edsl::SamplerSpec;

/// A lowered standard sampler: an attachable [`tensor::Program`] plus the data
/// needed to build its per-fire [`InputBinding`] list.
///
/// Build once per `(spec, vocab)` with [`sampler_program`], then call
/// [`LoweredSampler::bindings`] each decode step with the current decode
/// position — the program is param-invariant (temperature / top-p / min-p / k
/// ride as host-submit tensors), so it is reused across steps and the host
/// recognizer hash-matches it.
pub struct LoweredSampler {
    /// The attachable sampler program. Standard samplers declare a single
    /// [`OutputKind::Token`](sampling_edsl::OutputKind) output. Pass to
    /// `ForwardPass::sampler`.
    pub program: tensor::Program,
    /// Declared program output count (`1` for the standard single-`Token`
    /// samplers; a multi-output program — e.g. mirostat `[token, surprise]` —
    /// reports `> 1`, driving `output()` vs `outputs()`).
    pub outputs: u32,
    /// Per-slot binding template (`Op::Input(i) ↔ bindings[i]`).
    bindings: Vec<ir::Binding>,
    /// Host-input decls for the `Tensor` binding slots (param shapes / dtypes).
    host_inputs: Vec<HostInputDecl>,
    /// Per-fire submit params (temperature / top-p / min-p / k), keyed by slot.
    submit_values: Vec<(u32, Vec<u8>)>,
}

impl LoweredSampler {
    /// Resolve the per-fire positional [`InputBinding`] list for a sampler fired
    /// at `decode_pos` — the forward-pass output row whose next-token logits
    /// feed the program.
    ///
    /// Threads the standard-sampler params as submit tensors (NOT baked
    /// immediates): baking them would produce a bytecode-only program the driver
    /// recognizer can't hash-match → `CustomJIT`, and would drop the params (the
    /// #17 de-hardwiring contract). Pass the returned list to
    /// `ForwardPass::sampler(&self.program, bindings)`.
    pub fn bindings(&self, decode_pos: u32) -> Result<Vec<InputBinding>> {
        resolve_bindings(
            &self.bindings,
            &self.host_inputs,
            &[decode_pos],
            &self.submit_values,
        )
    }
}

/// Lower a standard sampler `spec` to an attachable program + bindings for the
/// runtime `vocab`.
///
/// Wraps `sampling_edsl::lower_sampler_standard` (the canonical
/// param-as-submit-tensor lowering) plus the guest program emit
/// ([`emit::emit_program`](crate::emit::emit_program)). Greedy specs
/// (`temperature <= 0`) collapse to argmax and declare no submit params.
pub fn sampler_program(spec: SamplerSpec, vocab: u32) -> Result<LoweredSampler> {
    let (built, submit_values) = sampling_edsl::lower_sampler_standard(spec, vocab)
        .map_err(|e| format!("sampler_program: lower sampler: {e:?}"))?;
    let program = crate::emit::emit_program(&built.program)?;
    Ok(LoweredSampler {
        program,
        outputs: built.outputs.len() as u32,
        bindings: built.bindings,
        host_inputs: built.host_inputs,
        submit_values,
    })
}

/// A lowered **N-position greedy verify sampler**: `argmax` of the target logits
/// at each of `rows` positions → one `[rows]`-Token output. The multi-position
/// analog of the single-position [`sampler_program`]`(Argmax)` — one fire samples
/// the anchor + every draft/guess position so the host can compute the accepted
/// (converged/matched) prefix. For the manual speculative-decode inferlets
/// (jacobi / cacheback / custom-spec) off the `Forward` facade. Pure intrinsic
/// (no submit params); `rows` is trace-known.
pub struct LoweredMatrix {
    /// The attachable `[rows, vocab]`-argmax program (single `[rows]`-Token
    /// output — read via `ForwardPass::output()`).
    pub program: tensor::Program,
    /// Per-slot binding template (`Op::Input(i) ↔ bindings[i]`) — a single
    /// `Logits` slot for the target-logits matrix intrinsic.
    bindings: Vec<ir::Binding>,
    /// Verify-row count (`rows`) — the required length of the `positions` arg.
    pub rows: u32,
}

impl LoweredMatrix {
    /// Positional bindings for the fire: the target-logits matrix intrinsic bound
    /// to the `positions` rows (must be [`rows`](Self::rows)-long) whose
    /// next-token logits are argmaxed. Pass to
    /// `ForwardPass::sampler(&self.program, bindings)`.
    pub fn bindings(&self, positions: &[u32]) -> Result<Vec<InputBinding>> {
        resolve_bindings(&self.bindings, &[], positions, &[])
    }
}

/// Build a `[rows, vocab]`-argmax verify program: one `Token` output of `rows`
/// greedy per-row picks. Zero submit params (pure `Logits` intrinsic).
pub fn argmax_matrix_program(vocab: u32, rows: u32) -> Result<LoweredMatrix> {
    let g = Graph::new(vocab);
    let toks = g.intrinsic_logits_matrix_dyn(rows).argmax(); // [rows] i32
    g.output(&toks, OutputKind::Token);
    let built = g
        .build()
        .map_err(|e| format!("argmax_matrix_program: {e:?}"))?;
    let program = crate::emit::emit_program(&built.program)?;
    Ok(LoweredMatrix {
        program,
        bindings: built.bindings,
        rows,
    })
}

/// Resolve a binding-free program's per-slot [`Binding`](ir::Binding) template
/// into the positional [`InputBinding`] list the forward-pass attach consumes
/// (`bindings[i]` binds program input slot `i`, i.e. `Op::Input(i)`):
///
/// - **`Logits`** → [`InputBinding::Logits`]`(logits_positions)` — the
///   forward-pass output rows whose next-token logits feed the program.
/// - **`MtpLogits`** → [`InputBinding::MtpLogits`] — the speculator's draft-row
///   logits intrinsic (no host data).
/// - **`Tensor { key, .. }`** → [`InputBinding::Tensor`] — a device tensor built
///   from the submit value bound to `key`, shaped/typed per its
///   [`HostInputDecl`].
///
/// Self-contained (mirrors the retired `program::resolve_bindings`) so this
/// primitive survives the facade deletion.
fn resolve_bindings(
    bindings: &[ir::Binding],
    host_inputs: &[HostInputDecl],
    logits_positions: &[u32],
    submit_values: &[(u32, Vec<u8>)],
) -> Result<Vec<InputBinding>> {
    bindings
        .iter()
        .map(|b| match b {
            ir::Binding::Logits => Ok(InputBinding::Logits(logits_positions.to_vec())),
            ir::Binding::MtpLogits => Ok(InputBinding::MtpLogits),
            ir::Binding::MtpDrafts => Ok(InputBinding::MtpDrafts),
            ir::Binding::Tensor { key, .. } => {
                let decl = host_inputs
                    .iter()
                    .find(|d| d.key == *key)
                    .ok_or_else(|| format!("sampler: no host-input decl for key {key}"))?;
                let data = submit_values
                    .iter()
                    .find(|(k, _)| k == key)
                    .map(|(_, v)| v.clone())
                    .ok_or_else(|| format!("sampler: no submit value bound for key {key}"))?;
                let t = tensor::Tensor::from_data(
                    &crate::emit::shape_to_wit(decl.shape),
                    crate::emit::dtype_to_wit(decl.dtype),
                    &data,
                )
                .map_err(|e| format!("sampler: tensor::from_data: {e:?}"))?;
                Ok(InputBinding::Tensor(t))
            }
        })
        .collect()
}

// ── Constrained / grammar sampler (the masked twin of the standard sampler) ──

/// A lowered CONSTRAINED (grammar) sampler: greedy `argmax(mask_apply(logits,
/// mask))` — the packed allowed-token bitmask sets disallowed logits to `−∞` —
/// with the per-fire mask a **submit** tensor (`Token`-only output). The masked
/// analog of [`LoweredSampler`], for the grammar/constrained inferlets
/// (`ptir-grammar-tranche-conversion-spec`).
///
/// **Grammar decode is SEQUENTIAL**, not run-ahead: the mask for step N+1 depends
/// on the token accepted at step N (the host `constraint::Matcher` advances on it),
/// so it is a per-step `submit_pass`-style fire with a fresh mask — NOT the
/// `carrier::submit_pass` run-ahead path. The mask is submit-known (computed
/// host-side from the already-accepted prior token before the fire).
pub struct LoweredGrammar {
    /// The attachable constrained sampler program (`[Token]` output).
    pub program: tensor::Program,
    /// Declared output count (`1` — a single `Token`).
    pub outputs: u32,
    bindings: Vec<ir::Binding>,
    host_inputs: Vec<HostInputDecl>,
    mask_key: ir::TensorKey,
    /// Packed-mask length (`ceil(vocab/32)` u32 words) — the length of the
    /// `packed_mask` slice [`bindings`](LoweredGrammar::bindings) expects.
    pub mask_words: usize,
}

/// Lower a constrained-greedy grammar program: `argmax(mask_apply(logits, mask))`,
/// optionally also emitting the raw (unmasked) logits (`with_logits`) as a second
/// output for the CONFORM verify, and with the mask supplied at `ready`
/// (`Submit` = host-computed before each fire, staged; `Late` = the production
/// device-alias supply channel — M-batch-eligible when Token-only). Zero new
/// driver ops (`mask_apply` + multi-output forward are golden) — program
/// composition only.
fn grammar_lowered(vocab: u32, with_logits: bool, ready: Readiness) -> Result<LoweredGrammar> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    let mask = g.host_vector_dyn(DType::U32, vocab.div_ceil(32), ready);
    let token = logits.mask_apply(&mask).argmax();
    g.output(&token, OutputKind::Token); // 0: constrained token
    if with_logits {
        g.output(&logits, OutputKind::Logits); // 1: raw (unmasked) logits
    }
    let mask_key = mask
        .input_key()
        .ok_or_else(|| "grammar_lowered: mask has no input key".to_string())?;
    let built = g
        .build()
        .map_err(|e| format!("grammar_lowered: build: {e:?}"))?;
    let program = crate::emit::emit_program(&built.program)?;
    Ok(LoweredGrammar {
        program,
        outputs: built.outputs.len() as u32,
        bindings: built.bindings,
        host_inputs: built.host_inputs,
        mask_key,
        mask_words: vocab.div_ceil(32) as usize,
    })
}

/// Constrained-greedy grammar, `[Token]`-only, **submit** mask (host-computed per
/// fire). The production sequential-decode path. Call
/// [`bindings`](LoweredGrammar::bindings) each step with the current packed mask.
pub fn grammar_program(vocab: u32) -> Result<LoweredGrammar> {
    grammar_lowered(vocab, false, Readiness::Submit)
}

/// `[Token]`-only with a **Late** mask (the production device-alias supply channel
/// — M-batch-eligible). Same `bindings` supply (`Tensor::from_data`); the `Late`
/// readiness routes it into the device-alias carrier host-side.
pub fn grammar_program_late(vocab: u32) -> Result<LoweredGrammar> {
    grammar_lowered(vocab, false, Readiness::Late)
}

impl LoweredGrammar {
    /// Per-fire [`InputBinding`] list: `Logits`@`decode_pos` + the packed
    /// allowed-token mask (`[ceil(vocab/32)]` u32, bit `i` set = token `i`
    /// allowed) as the submit mask tensor. An **all-ones** mask = no restriction
    /// (transparent). Pass to `ForwardPass::sampler(&self.program, bindings)`.
    pub fn bindings(&self, decode_pos: u32, packed_mask: &[u32]) -> Result<Vec<InputBinding>> {
        let bytes: Vec<u8> = packed_mask.iter().flat_map(|w| w.to_le_bytes()).collect();
        resolve_bindings(
            &self.bindings,
            &self.host_inputs,
            &[decode_pos],
            &[(self.mask_key, bytes)],
        )
    }
}

/// Like [`grammar_program`] but ALSO emits the **unmasked** logits as a second
/// output (`outputs = [Token, Logits]`, read via `ForwardPass::outputs()`) — the
/// grammar mask-op **verify** path: recompute `mask_apply(raw_logits, mask)`
/// host-side (`mask::apply_mask_argmax`) and assert the device token matches (the
/// CONFORM gate, `ptir-grammar-tranche-conversion-spec §7`). **Submit** mask.
pub fn grammar_program_with_logits(vocab: u32) -> Result<LoweredGrammar> {
    grammar_lowered(vocab, true, Readiness::Submit)
}

/// The **Late**-mask twin of [`grammar_program_with_logits`] — exercises the
/// production device-alias Late supply channel with the raw-logits CONFORM
/// output. (`[Token, Logits]` = rich → not M-batch-eligible; the verify harness's
/// job, not production decode.)
pub fn grammar_program_with_logits_late(vocab: u32) -> Result<LoweredGrammar> {
    grammar_lowered(vocab, true, Readiness::Late)
}

/// Lower a constrained-**sampled** grammar program: `argmax(mask_apply(logits,
/// mask) + gumbel(stream:0))` — the masked twin's diversity-preserving variant
/// (Gumbel-max = temperature-1 categorical UNDER the mask; disallowed tokens are
/// `−∞` so noise can never lift them above an allowed token). Mirrors
/// `sampling_edsl::program::grammar_sampled` but with a **submit** mask so it
/// binds via [`resolve_bindings`], and returns the same [`LoweredGrammar`] handle
/// as [`grammar_program`] (identical mask binding surface).
///
/// Use this for the agentic grammar inferlets that sampled (`Multinomial`/`TopP`)
/// under the grammar to break greedy Argmax loops: the grammar is now ENFORCED
/// (the shipped facade computed-but-dropped the mask) while sampling diversity is
/// preserved via the Gumbel noise. Greedy grammar callers use [`grammar_program`].
pub fn grammar_program_sampled(vocab: u32) -> Result<LoweredGrammar> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    let mask = g.host_vector_dyn(DType::U32, vocab.div_ceil(32), Readiness::Submit);
    let token = logits.mask_apply(&mask).add(&g.rng_gumbel_vec(0, vocab)).argmax();
    g.output(&token, OutputKind::Token);
    let mask_key = mask
        .input_key()
        .ok_or_else(|| "grammar_program_sampled: mask has no input key".to_string())?;
    let built = g
        .build()
        .map_err(|e| format!("grammar_program_sampled: build: {e:?}"))?;
    let program = crate::emit::emit_program(&built.program)?;
    Ok(LoweredGrammar {
        program,
        outputs: built.outputs.len() as u32,
        bindings: built.bindings,
        host_inputs: built.host_inputs,
        mask_key,
        mask_words: vocab.div_ceil(32) as usize,
    })
}

// ── Measurement / probe (the de-hardwired `probe(pos, kind)` wire op) ────────

/// Which single-channel **measurement** to lower over the model's output
/// logits — the keep-core replacement for the facade `forward::Probe` enum (the
/// de-hardwiring of the removed `probe(pos, kind)` wire op). Each kind lowers to
/// a Graph-authored program over `intrinsic_logits`; no new wire op, no submit
/// params (pure `Logits`-bound).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeKind {
    /// Raw pre-softmax logits, `[vocab]` f32 ([`OutputKind::Logits`]) — read via
    /// the pass output's `read_f32` / `read_bytes`.
    Logits,
    /// Shannon entropy of the softmax distribution, a scalar
    /// ([`OutputKind::Scalar`]) — read via the pass output's `scalar`.
    Entropy,
    /// The full softmax distribution, `[vocab]` f32 ([`OutputKind::Distribution`])
    /// — read via the pass output's `distribution` / `read_f32`.
    Distribution,
}

/// A lowered measurement program: a single non-sampling channel over the model's
/// output logits ([`ProbeKind`]). The measurement analog of [`LoweredSampler`] —
/// binding-free except the implicit `Logits` input (no submit params). Attach
/// via `ForwardPass::sampler(&self.program, self.bindings(decode_pos)?)` and read
/// the declared [`OutputKind`] off the pass output.
pub struct LoweredProbe {
    /// The attachable measurement program.
    pub program: tensor::Program,
    /// Declared output count (`1` — a single measurement channel).
    pub outputs: u32,
    bindings: Vec<ir::Binding>,
    host_inputs: Vec<HostInputDecl>,
}

impl LoweredProbe {
    /// Per-fire [`InputBinding`] list: `Logits`@`decode_pos` (the output row the
    /// measurement reads). No submit params. Pass to
    /// `ForwardPass::sampler(&self.program, bindings)`.
    pub fn bindings(&self, decode_pos: u32) -> Result<Vec<InputBinding>> {
        resolve_bindings(&self.bindings, &self.host_inputs, &[decode_pos], &[])
    }
}

/// Lower a [`ProbeKind`] measurement to an attachable program + bindings for the
/// runtime `vocab`. Wraps the canonical `sampling_edsl::program::{logits,
/// entropy, distribution}` measurement lowerings plus the guest program emit
/// ([`emit::emit_program`](crate::emit::emit_program)) — the measurement twin of
/// [`sampler_program`]. Build once per `(kind, vocab)`; call
/// [`bindings`](LoweredProbe::bindings) each fire with the read row.
///
/// The keep-core replacement for the facade `Forward::probe` path (which dies
/// with the SDK-min facade deletion); the measurement inferlets
/// (`raw-logits-demo` / `watermarking` / `output-validation` / …) attach this
/// over a raw `ForwardPass` instead of `context.forward().probe(..)`.
pub fn probe_program(kind: ProbeKind, vocab: u32) -> Result<LoweredProbe> {
    let built = match kind {
        ProbeKind::Logits => sampling_edsl::program::logits(vocab),
        ProbeKind::Entropy => sampling_edsl::program::entropy(vocab),
        ProbeKind::Distribution => sampling_edsl::program::distribution(vocab),
    }
    .map_err(|e| format!("probe_program: lower measurement: {e:?}"))?;
    let program = crate::emit::emit_program(&built.program)?;
    Ok(LoweredProbe {
        program,
        outputs: built.outputs.len() as u32,
        bindings: built.bindings,
        host_inputs: built.host_inputs,
    })
}

/// A lowered GENERIC multi-output measurement program — the direct keep-core
/// analog of the facade `Forward::measure(built)`. Wraps a caller-authored
/// [`sampling_edsl::Built`] (any Graph program over `intrinsic_logits` with N
/// declared outputs) into an attachable `[program]` + per-fire `Logits`-bound
/// bindings, of which [`probe_program`] (single-kind) is the special case.
///
/// Attach via `ForwardPass::sampler(&self.program, self.bindings(decode_pos)?)`
/// and read the N declared outputs off `ForwardPass::outputs().await` (declared
/// order). The keep-core replacement for the facade `measure()` (dies with the
/// SDK-min deletion); the multi-measurement inferlets (`sampler-suite`) author
/// their Built and attach this over a raw `ForwardPass`.
pub struct LoweredMeasure {
    /// The attachable measurement program (N declared outputs).
    pub program: tensor::Program,
    /// Declared output count (read `outputs()[0..outputs]` in declared order).
    pub outputs: u32,
    bindings: Vec<ir::Binding>,
    host_inputs: Vec<HostInputDecl>,
}

impl LoweredMeasure {
    /// Per-fire [`InputBinding`] list: `Logits`@`decode_pos` (the output row the
    /// measurement reads). No submit params. Pass to
    /// `ForwardPass::sampler(&self.program, bindings)`.
    pub fn bindings(&self, decode_pos: u32) -> Result<Vec<InputBinding>> {
        resolve_bindings(&self.bindings, &self.host_inputs, &[decode_pos], &[])
    }
}

/// Lower a caller-authored [`sampling_edsl::Built`] measurement program to an
/// attachable program + bindings. The generic `measure()` analog: emit the guest
/// program ([`emit::emit_program`](crate::emit::emit_program)) and carry the
/// binding template + host-input decls for per-fire resolution. Build once; call
/// [`bindings`](LoweredMeasure::bindings) each fire with the read row.
pub fn measure_program(built: sampling_edsl::Built) -> Result<LoweredMeasure> {
    let program = crate::emit::emit_program(&built.program)?;
    Ok(LoweredMeasure {
        program,
        outputs: built.outputs.len() as u32,
        bindings: built.bindings,
        host_inputs: built.host_inputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // The pure binding-resolution arms (`Logits` / `MtpLogits`) need no host —
    // resolvable natively. The `Tensor` arm + the full `sampler_program`
    // lowering call WIT host fns (`tensor::from_data` / `Program::new`), so
    // they are exercised e2e on the mock by the `samplerprobe` inferlet.
    #[test]
    fn resolve_logits_and_mtp_arms() {
        let bindings = [ir::Binding::Logits, ir::Binding::MtpLogits];
        let out = resolve_bindings(&bindings, &[], &[7], &[]).expect("resolve");
        assert_eq!(out.len(), 2);
        assert!(matches!(&out[0], InputBinding::Logits(p) if p.as_slice() == [7]));
        assert!(matches!(&out[1], InputBinding::MtpLogits));
    }

    // A `Tensor` binding with no matching submit value is a loud error (not a
    // silent param drop — the #17 contract).
    #[test]
    fn tensor_arm_missing_submit_errors() {
        let bindings = [ir::Binding::Tensor {
            key: 3,
            ready: sampling_edsl::Readiness::Submit,
        }];
        let decl = HostInputDecl {
            key: 3,
            shape: ir::Shape::SCALAR,
            dtype: ir::DType::F32,
            ready: sampling_edsl::Readiness::Submit,
        };
        let err = resolve_bindings(&bindings, &[decl], &[0], &[]).unwrap_err();
        assert!(err.contains("no submit value"), "got: {err}");
    }
}
