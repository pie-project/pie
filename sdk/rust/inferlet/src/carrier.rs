//! Run-ahead carrier mechanics — the two correctness-critical primitives for
//! pipelined (run-ahead + EOS-rollback) decoding on the raw WIT.
//!
//! In Gim's SDK-minimize thesis: the decode LOOP stays hand-written and VISIBLE
//! in each inferlet (see `runtime/tests/inferlets/runahead::decode_pipelined`).
//! Only the delicate, correctness-critical carrier mechanics are factored here
//! as a thin primitive so their leak/hang-safety is proven ONCE rather than
//! re-derived per inferlet:
//!   - the device carrier declaration (`next_inputs(&[0])`) that injects a
//!     pass's sample into the next pass's placeholder input row 0,
//!   - the cursor **advance-on-submit** (so an overlapped successor reserves its
//!     slot while this pass is still in flight),
//!   - the #26 dangling-carrier clear (`fresh_generate`) on a generate's first pass,
//!   - the depth-1 EOS-rollback discard (finalize-drain + cursor rollback).
//!
//! This is NOT a decode-loop helper — the inferlet writes the loop and the stop
//! logic; these are the ~2 primitives it calls. Mechanics owned by bravo
//! (`ptir-lowlevel-runahead-mechanics`, `ptir-pipelined-eos-rollback-spec`,
//! extracted from `runahead::decode_pipelined` @ e750ae4b). Composes the KV
//! page-geometry primitive ([`crate::geometry`]) — a keep-core building block,
//! per `ptir-sdk-minimization-audit`.

use crate::geometry;
use crate::inference::ForwardPass;
use crate::sampler::LoweredSampler;
use crate::working_set::KvWorkingSet;
use crate::Result;

/// Submit ONE forward pass (producer OR consumer) over `tokens` at the current
/// cursor `*seq_len`, and advance the cursor **on submit**.
///
/// Sequence: `fresh_generate` (iff `*fresh`, then clears it) → KV write geometry
/// (alloc + `kv_working_set`, via [`crate::geometry`]) → `input_tokens` at
/// `[*seq_len, *seq_len + n)` → attach `sampler`'s program bound to the decode
/// row `*seq_len + n - 1` (with its FULL binding list — see below) → declare the
/// run-ahead carrier `next_inputs(&[0])` **iff `carry`** → `execute()` →
/// `*seq_len += n`.
///
/// `sampler` is a [`LoweredSampler`](crate::sampler::LoweredSampler) carrying the
/// sampler semantics (greedy = `sampler_program(SamplerSpec::Argmax)`, or
/// top-p/top-k/min-p/…); this primitive binds its program to the decode position
/// with the sampler's full per-fire binding list (`Logits` + any param submit
/// tensors `T`/`p`/`k`), so a **parametric** sampler flows through the run-ahead
/// carrier without its params being dropped (the #17 de-hardwiring contract).
/// `carry` must be `false` on a pass with no
/// successor (the terminal pass) to avoid a dangling carrier; when a stop is
/// configured but not yet hit, `carry` should be `true` (the terminal pass is
/// not predictable at submit — a dangling link is cleared by the next generate's
/// `fresh_generate`). The returned pass is **in flight**; finalize it with
/// `output().await` (to read its token) or [`discard_pass`] (to roll it back).
pub fn submit_pass(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    sampler: &LoweredSampler,
    tokens: &[u32],
    carry: bool,
) -> Result<ForwardPass> {
    submit_pass_with(kv, seq_len, fresh, sampler, tokens, carry, |_| {})
}

/// Like [`submit_pass`], but calls `bind(&pass)` **after `input_tokens` and
/// before the sampler / carrier / execute** — the seam where an inferlet
/// attaches a per-pass input-side binding (`attention_mask` / `rs_working_set` /
/// `adapter` / `zo_seed`) **without** re-implementing the leak/hang-critical
/// carrier mechanics (#26 clear, advance-on-submit, carrier decl, and the
/// [`discard_pass`] WAR drain — all stay inside this primitive).
///
/// [`submit_pass`] is the no-op-bind wrapper, so this is the single source of
/// the safe `create → #26 → geometry → input → [bind] → sampler → carrier →
/// execute → advance` sequence.
///
/// # The `bind` contract
/// `bind(&pass)` MUST attach only input-side bindings and MUST NOT: call
/// `execute()` / `output()` (this primitive owns the lifecycle), mutate
/// `*seq_len` / `*fresh` (owned here), or call `sampler` / `next_inputs` (the
/// fixed tail). Positions are computed internally from the **pre-advance**
/// `*seq_len`; a `bind` closure captures whatever geometry it needs (cursor,
/// `n`, mask params) from its own scope before the call — it does not read them
/// back off the pass.
///
/// ```ignore
/// let sl = d.seq_len;                 // pre-advance cursor
/// let n = tokens.len() as u32;
/// let pass = carrier::submit_pass_with(
///     &d.kv, &mut d.seq_len, &mut d.fresh, &sampler, tokens, carry,
///     |pass| pass.attention_mask(&[build_sink_mask(sl + n, sink, window)]),
/// )?;
/// ```
pub fn submit_pass_with(
    kv: &KvWorkingSet,
    seq_len: &mut u32,
    fresh: &mut bool,
    sampler: &LoweredSampler,
    tokens: &[u32],
    carry: bool,
    bind: impl FnOnce(&ForwardPass),
) -> Result<ForwardPass> {
    let n = tokens.len() as u32;
    let pass = ForwardPass::new();
    // #26 dangling-carrier clear: on the FIRST pass of this generate, drop any
    // carrier a prior generate left pending on this context (+ free its retained
    // device buffer) BEFORE this pass's carrier inject.
    if *fresh {
        pass.fresh_generate();
        *fresh = false;
    }
    // KV write geometry for [*seq_len, *seq_len + n) — the keep-core primitive.
    let geom = geometry::ensure_pages(
        kv,
        geometry::kv_write_geometry(*seq_len, n, kv.page_size()),
    )?;
    geometry::attach_kv_write(&pass, kv, &geom);
    let positions: Vec<u32> = (*seq_len..*seq_len + n).collect();
    pass.input_tokens(tokens, &positions);
    // ── the SEAM: input is set + geometry bound, cursor NOT yet advanced ──
    // The inferlet attaches its per-pass input-side binding here (mask covering
    // the just-attached rows / RS window / adapter). See the `bind` contract.
    bind(&pass);
    // ── the fixed carrier tail (identical to the monolithic submit_pass) ──
    let decode_pos = *seq_len + n - 1;
    // Attach the sampler's FULL binding list: `Logits` @ decode_pos PLUS any
    // param submit-tensors (temperature / top-p / top-k). Threading ONLY
    // `Logits` would drop a parametric sampler's params → the driver recognizer
    // can't hash-match the program → `CustomJIT` + wrong sampling (the #17
    // de-hardwiring contract). Greedy is `sampler_program(SamplerSpec::Argmax)`,
    // whose `bindings()` is exactly `[Logits]` — one unified path, no split.
    pass.sampler(&sampler.program, sampler.bindings(decode_pos)?);
    // CARRIER: inject THIS pass's sampled token into the NEXT pass's input row 0
    // device-side (zero host round-trip). Declared only when a successor runs.
    if carry {
        pass.next_inputs(&[0]);
    }
    pass.execute();
    *seq_len += n; // advance on SUBMIT (the overlap)
    Ok(pass)
}

/// Discard an over-shot speculative pass (depth-1 EOS rollback): **finalize** it
/// by awaiting its output — the WAR-guard drain, so its KV write completes
/// before a later generate overwrites-after the tail slot — then roll the cursor
/// back by one. The pass's sampled token is IGNORED (never emitted).
///
/// No `Result`: a discard never fails the decode, and its output is deliberately
/// dropped. But the pass **MUST** be finalized here (awaited) — never dropped
/// un-finalized, or its in-flight device write races a later overwrite.
pub async fn discard_pass(pass: ForwardPass, seq_len: &mut u32) {
    // Finalize (drain) the in-flight pass; ignore its output.
    let _ = pass.output().await;
    *seq_len = seq_len.saturating_sub(1);
}

/// Declare the device-resident **drafts-channel window carrier** on a producer
/// pass: the `[k+1]` `[seed, drafts]` window is retained device-side and
/// injected into the *next* fire's input slots `0..=k` with zero host
/// round-trip — the `[k+1]`-window generalization of [`submit_pass`]'s
/// single-token `next_inputs(&[0])` carrier. The enabler for pipelined MTP
/// spec-decode (and any `[k+1]`-window ① spec-decode drafter).
///
/// `k` = the number of MTP draft proposals; the window is `k + 1` slots —
/// `[seed, d_0 … d_{k-1}]`, with the seed (`gather(picked, n_acc)`) at row 0 and
/// the fresh drafts at rows `1..=k`.
///
/// This is the guest surface of the ratified drafts-channel **carrier-kind**
/// (`ptir-drafts-channel-carrier-kind`): the pass declares the window via the
/// existing multi-slot `next-inputs` (slots `0..=k`, live today), and the driver
/// routes the retain SOURCE off `pipeline_source_kind = PrevDrafts (1)` →
/// charlie's `mtp_drafts` **per-link COPY** buffer (guru's LOCKED condition-2
/// retain: per-link `DeviceBuffer` + `done` event) instead of `pi.sampled`.
/// Additive `u8` tag on `pipeline_source_link`; no layout change; mirrors the
/// shipped `SamplingBinding::MtpLogits`(=2)-vs-`Logits`(=0) manifest-kind
/// precedent.
///
/// # Wiring status (landed with `mtp_drafts` retain)
/// The `[k+1]` window declaration is emitted here via the raw multi-slot
/// `next-inputs`, and the `pipeline_source_kind = 1` routing is applied via the
/// `set-pipeline-source-kind` guest surface — landed as ONE coordinated unit with
/// charlie/bravo's `mtp_drafts` retain-source (`5bf5f8b5`) + its host binding.
/// A producer that does NOT call this stays `PrevSample (0)` (the `pi.sampled`
/// source, byte-identical to the shipped single-token carry).
pub fn next_inputs_drafts(pass: &ForwardPass, k: u32) {
    // The `[k+1]` device window: slots `0..=k` = `[seed, d_0 … d_{k-1}]`.
    let window: Vec<u32> = (0..=k).collect();
    pass.next_inputs(&window);
    // Route THIS producer's retain SOURCE to the MTP-drafts per-link COPY buffer
    // via `pipeline_source_kind = PrevDrafts(1)` (ptir-drafts-channel-carrier-kind
    // §1): the host retains the `[k+1]` `[seed, drafts]` window off charlie's
    // `mtp_drafts` buffer (bravo's `5bf5f8b5` retain) instead of `pi.sampled`.
    pass.set_pipeline_source_kind(1);
}
