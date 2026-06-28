//! Ready-to-use **runtime-vocab** sampler programs (Phase 2). Each returns a
//! [`Built`] program plus a small `*Keys` struct naming the host-input tensor
//! keys, so an inferlet binds by name, not declaration order.
//!
//! RNG is **Model B**: `Op::Rng{stream}` with an ambient per-fire seed (runtime
//! owned) — programs declare no seed input. Multiple draws decorrelate via
//! distinct `stream` ids (`stream:0` ≡ today's hash for parity).

use crate::builder::{Built, BuildError, Graph, OutputKind};
use crate::dynamic::{
    DynValue, dselect, dyn_eq_const, dyn_residual_resample_rows, dyn_softmax, dyn_softmax_rows,
};
use crate::ir::{DType, Readiness, TensorKey};

/// Host-input keys for the [`mirostat`] program.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MirostatKeys {
    /// Submit-bound scalar μ (running target surprise), rebound each fire.
    pub mu: TensorKey,
}

/// **Mirostat v2** (sequential late-bind). Truncates to tokens whose surprise
/// `-log p ≤ μ`, Gumbel-samples among them (`stream:0`), and outputs the sampled
/// `token` plus the observed surprise `S = -log p(token)` (nats) for the CPU
/// update `μ ← μ − lr·(S − τ)`. `outputs = [Token, Scalar]`.
pub fn mirostat(vocab: u32) -> Result<(Built, MirostatKeys), BuildError> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    let mu = g.host_scalar_dyn(DType::F32, Readiness::Submit);

    let probs = dyn_softmax(&logits);
    let surprise = probs.log().neg(); // -log p, [vocab]

    let keep = mu.broadcast_vec(vocab).ge(&surprise); // surprise <= μ
    let perturbed = logits.add(&g.rng_gumbel_vec(0, vocab));
    let neg_inf = g.constant_f32_dyn(f32::NEG_INFINITY).broadcast_vec(vocab);
    let token = dselect(&keep, &perturbed, &neg_inf).argmax();

    // S = surprise[token] via a length-1 gather -> reduce.
    let s = surprise.gather(&token.broadcast_vec(1)).reduce_sum();

    g.output(&token, OutputKind::Token);
    g.output(&s, OutputKind::Scalar);

    let keys = MirostatKeys { mu: mu.input_key().expect("mu host input") };
    Ok((g.build()?, keys))
}

/// **Mirostat v2 with a min-kept-set floor** (#19 structural repetition-attractor
/// fix). Identical to [`mirostat`] but the surprise gate is unioned with the
/// **top-`k_min` by logit**, so the kept set can never collapse below `k_min`
/// tokens regardless of μ:
///
/// ```text
/// keep = (surprise ≤ μ)  OR  (rank_by_logit < k_min)
/// ```
///
/// This preserves mirostat's adaptive truncation *above* the floor (when μ admits
/// more than `k_min` tokens the gate is unchanged) while guaranteeing the gate can
/// never degenerate to greedy — the entry mechanism of the repetition-attractor
/// (a single high-confidence token → repeat). `k_min` is baked as a `Const`
/// value-id (so `RankLe` references it, #25-consistent — not an immediate).
/// `outputs = [Token, Scalar]`; the host μ-update (`μ ← μ − lr·(S − τ)`) is
/// unchanged. A small floor (e.g. `k_min = 8`) is enough to keep diversity.
pub fn mirostat_floor(vocab: u32, k_min: u32) -> Result<(Built, MirostatKeys), BuildError> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    let mu = g.host_scalar_dyn(DType::F32, Readiness::Submit);

    let probs = dyn_softmax(&logits);
    let surprise = probs.log().neg(); // -log p, [vocab]

    let keep_mu = mu.broadcast_vec(vocab).ge(&surprise); // surprise ≤ μ
    // Min-kept-set floor: always keep the top-`k_min` by logit so the gate can
    // never collapse below `k_min` tokens.
    let kfloor = g.constant_i32_dyn(k_min as i32);
    let keep_floor = logits.pivot_rank_le(&kfloor);
    let always = g.constant_bool_dyn(true).broadcast_vec(vocab);
    let keep = dselect(&keep_mu, &always, &keep_floor); // keep_mu OR keep_floor

    let perturbed = logits.add(&g.rng_gumbel_vec(0, vocab));
    let neg_inf = g.constant_f32_dyn(f32::NEG_INFINITY).broadcast_vec(vocab);
    let token = dselect(&keep, &perturbed, &neg_inf).argmax();

    // S = surprise[token] via a length-1 gather -> reduce.
    let s = surprise.gather(&token.broadcast_vec(1)).reduce_sum();

    g.output(&token, OutputKind::Token);
    g.output(&s, OutputKind::Scalar);

    let keys = MirostatKeys { mu: mu.input_key().expect("mu host input") };
    Ok((g.build()?, keys))
}
/// speculator's draft token from the on-device **draft logits**
/// ([`Graph::intrinsic_mtp_logits_dyn`]) — `argmax(mtp_logits)`. The binding
/// ([`crate::ir::Binding::MtpLogits`]) source-selects the draft row of
/// `ws.logits` (M=1 ⇒ row 0), so the bytecode is byte-identical to a plain
/// logits `argmax` — the source is a manifest property, not bytecode.
/// `outputs = [Token]`; no host inputs.
pub fn mtp_argmax(vocab: u32) -> Result<Built, BuildError> {
    let g = Graph::new(vocab);
    let draft = g.intrinsic_mtp_logits_dyn();
    let token = draft.argmax();
    g.output(&token, OutputKind::Token);
    g.build()
}

/// Host-input keys for the [`grammar`] program.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GrammarKeys {
    /// Late-bound packed allowed-token bitmask (`[ceil(vocab/32)]` u32, bit 1 =
    /// allowed / 0 = disallowed). Recomputed each step by the matcher and
    /// written via `tensor.write`; applied in-program by [`Op::MaskApply`].
    pub mask: TensorKey,
}

/// **Constrained / grammar decoding** (greedy). `argmax(mask_apply(logits,
/// mask))` — the packed allowed-token bitmask sets disallowed logits to `−∞`.
/// `outputs = [Token]`.
pub fn grammar(vocab: u32) -> Result<(Built, GrammarKeys), BuildError> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    let mask = g.host_vector_dyn(DType::U32, vocab.div_ceil(32), Readiness::Late);

    let token = logits.mask_apply(&mask).argmax();
    g.output(&token, OutputKind::Token);

    let keys = GrammarKeys { mask: mask.input_key().expect("mask host input") };
    Ok((g.build()?, keys))
}

/// **Constrained / grammar decoding with the raw logits** — same masked
/// `argmax` as [`grammar`], but also emits the **unmasked** logits as a second
/// output (`outputs = [Token, Logits]`). The test/verify path reads the raw
/// logits at the constrained step to recompute `mask_apply` host-side (the
/// CPU reference) and prove the device `−∞` fired. Production decode uses
/// [`grammar`] (`[Token]` only).
pub fn grammar_with_logits(vocab: u32) -> Result<(Built, GrammarKeys), BuildError> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    let mask = g.host_vector_dyn(DType::U32, vocab.div_ceil(32), Readiness::Late);

    let token = logits.mask_apply(&mask).argmax();
    g.output(&token, OutputKind::Token); // 0: constrained token
    g.output(&logits, OutputKind::Logits); // 1: raw (unmasked) logits

    let keys = GrammarKeys { mask: mask.input_key().expect("mask host input") };
    Ok((g.build()?, keys))
}

/// **Constrained / grammar decoding with the raw logits — SUBMIT-mask verify
/// variant.** Identical program shape to [`grammar_with_logits`]
/// (`mask_apply(logits, mask).argmax()`, `outputs = [Token, Logits]`) but the
/// packed mask is `Readiness::Submit` rather than `Late`. The sequential grammar
/// mask is submit-known (computed from the already-accepted prior token, before
/// the fire), so it rides the existing `resolve_bindings` Submit gather — which
/// lets the `0x65` mask-apply OP be verified now, decoupled from the Late-channel
/// supply path (the `0x65` op runs identically regardless of supply path). The
/// production constrained path uses the `Late` mask ([`grammar`] /
/// [`grammar_with_logits`]); this variant is for the op-semantics verify.
pub fn grammar_submit_with_logits(vocab: u32) -> Result<(Built, GrammarKeys), BuildError> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    let mask = g.host_vector_dyn(DType::U32, vocab.div_ceil(32), Readiness::Submit);

    let token = logits.mask_apply(&mask).argmax();
    g.output(&token, OutputKind::Token); // 0: constrained token
    g.output(&logits, OutputKind::Logits); // 1: raw (unmasked) logits

    let keys = GrammarKeys { mask: mask.input_key().expect("mask host input") };
    Ok((g.build()?, keys))
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GrammarSampledKeys {
    pub mask: TensorKey,
}

/// **Constrained decoding, sampled.** `argmax(mask_apply(logits, mask) +
/// gumbel(stream:0))` — disallowed tokens are `−∞` so the Gumbel noise can
/// never lift them above an allowed token.
pub fn grammar_sampled(vocab: u32) -> Result<(Built, GrammarSampledKeys), BuildError> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    let mask = g.host_vector_dyn(DType::U32, vocab.div_ceil(32), Readiness::Late);

    let masked = logits.mask_apply(&mask);
    let token = masked.add(&g.rng_gumbel_vec(0, vocab)).argmax();
    g.output(&token, OutputKind::Token);

    let keys = GrammarSampledKeys { mask: mask.input_key().expect("mask host input") };
    Ok((g.build()?, keys))
}

// ============================================================================
// Speculative-decode verify (v4 matrix)
// ============================================================================

/// Host-input keys for [`spec_verify_greedy`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SpecVerifyKeys {
    /// Submit-bound `[k]` i32 draft token ids.
    pub draft: TensorKey,
}

/// **Greedy speculative-verify** (v4). Verifies a block of `k` draft tokens
/// against the target model's greedy decode in one matrix pass. Target logits =
/// intrinsic `[k, vocab]`; draft tokens = submit-bound `[k]` i32. Output is a
/// sentinel-coded `[k]` Token: the accepted prefix, then `-1` from the first
/// reject. Greedy DAG: `argmax -> eq -> cumprod -> select`.
pub fn spec_verify_greedy(vocab: u32, k: u32) -> Result<(Built, SpecVerifyKeys), BuildError> {
    build_spec_verify_greedy(vocab, k, Readiness::Submit, -1, false)
}

/// **Self-spec greedy-verify** (#31). The MTP head as the draft SOURCE, producing
/// the COMPLETE greedy accept set. The draft is bound
/// [`Readiness::SelfSpecDraftInput`] — the driver source-selects the refed draft
/// tokens at the verify matrix base (`pi.tokens + sample_row + 1`, the drafts start
/// after the anchor at `sample_row`), no host upload, de-hardwiring the hardwired
/// spec loop. Output is a sentinel-coded `[k]` Token `[d0..d_{j-1}, t_j, -1…]`: the
/// `j` accepted drafts PLUS the target's correction `t_j = target_argmax[j]` spliced
/// at the first reject (so a full reject advances by 1 instead of stalling, and the
/// set is lossless-greedy). All-accept emits all `k` drafts (bonus deferred). The
/// loop reads the truncated output and accepts it whole.
pub fn mtp_self_spec_greedy(vocab: u32, k: u32) -> Result<(Built, SpecVerifyKeys), BuildError> {
    build_spec_verify_greedy(vocab, k, Readiness::SelfSpecDraftInput, -1, true)
}

/// **Observable self-spec greedy-verify** (#31 verify harness, NOT production).
/// [`mtp_self_spec_greedy`] but with a `0` reject-sentinel instead of `-1`: since
/// `0 >= 0` the host marshal does NOT truncate (`if (x < 0) break`), so the full
/// `[k]` is emitted and a cross-row reject / draft-clobber is OBSERVABLE rather
/// than masked by the `-1` compaction (delta's reject-MID detector). Test-only:
/// the harness MUST construct `{d0..d_{j-1}, t_j}` all ∈ `[1, vocab)` (non-zero) so
/// a `0` in the output is unambiguously a reject-sentinel — including the boundary
/// correction `t_j`, else a `t_j == 0` collides with the past-boundary sentinels.
pub fn mtp_self_spec_greedy_observable(
    vocab: u32,
    k: u32,
) -> Result<(Built, SpecVerifyKeys), BuildError> {
    build_spec_verify_greedy(vocab, k, Readiness::SelfSpecDraftInput, 0, true)
}

/// Shared greedy-verify graph builder. Core DAG: `argmax(target[k,vocab]) ->
/// eq(draft) -> cumprod -> select`. `draft_ready` picks the draft SOURCE (`Submit`
/// = host-injected #27/#35-A; `SelfSpecDraftInput` = driver-internal MTP #31) — a
/// manifest property, so the core DAG is identical across sources. `sentinel` codes
/// a reject (`-1` production/truncating-to-`[j]`; `0` observable/non-truncating).
/// `emit_correction` (#31): splice the target's greedy token `t_j` at the first
/// reject so the output is the COMPLETE accept set `[d0..d_{j-1}, t_j]` (lossless,
/// always advances ≥1) rather than the bare accept-prefix detector (`false` = the
/// landed #35-A `spec_verify_greedy` shape).
fn build_spec_verify_greedy(
    vocab: u32,
    k: u32,
    draft_ready: Readiness,
    sentinel: i32,
    emit_correction: bool,
) -> Result<(Built, SpecVerifyKeys), BuildError> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_matrix_dyn(k); // [k, vocab]
    let draft = g.host_vector_dyn(DType::I32, k, draft_ready);

    let target = logits.argmax(); // [k] i32 per-row greedy
    let matched = target.eq(&draft); // [k] bool

    // Accept the longest matching prefix: prefix-AND via cumprod over {0,1}.
    let ones = g.constant_f32_dyn(1.0).broadcast_vec(k);
    let zeros = g.constant_f32_dyn(0.0).broadcast_vec(k);
    let acc = dselect(&matched, &ones, &zeros).cumprod();
    let keep = acc.gt(&g.constant_f32_dyn(0.5));

    let sentinel_v = g.constant_i32_dyn(sentinel).broadcast_vec(k);
    let out = if emit_correction {
        // A2: emit the COMPLETE greedy accept set `[d0..d_{j-1}, t_j]` — at the
        // first reject (boundary) splice the target's own greedy token `t_j =
        // target_argmax[j]` (the FREE correct token) instead of the sentinel, so a
        // full reject (j=0) advances by 1 (`[t_0]`) rather than stalling, and the
        // set is lossless-greedy. Boundary = the first reject: `c = cumsum(rejects)`
        // increments, so `c == 1` is true only at the first rejected row (mirrors
        // the lossless-verify boundary). All-accept (j=k) → no boundary → all drafts
        // (the bonus token `t_k` is deferred to the next block — needs a [k+1] row).
        let reject_f = dselect(&keep, &zeros, &ones); // 1.0 where rejected (keep=0)
        let c = reject_f.cumsum();
        let boundary = dyn_eq_const(&g, &c, 1.0); // first reject only
        let inner = dselect(&boundary, &target, &sentinel_v); // t_j at boundary, else sentinel
        dselect(&keep, &draft, &inner) // accepted draft, else (correction | sentinel)
    } else {
        dselect(&keep, &draft, &sentinel_v) // plain accept-prefix detector (#35-A)
    };
    g.output(&out, OutputKind::Token);

    let keys = SpecVerifyKeys { draft: draft.input_key().expect("draft host input") };
    Ok((g.build()?, keys))
}

/// Host-input keys for [`spec_verify_lossless`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SpecLosslessKeys {
    /// Submit-bound `[k, vocab]` draft probabilities `q`.
    pub q: TensorKey,
    /// Submit-bound `[k]` i32 draft token ids (the `GatherRow` index).
    pub draft: TensorKey,
}

/// **Lossless (rejection-sampling) speculative-verify** (v4). Accept draft token
/// `x_i` with prob `min(1, p(x_i)/q(x_i))` (`u < min(1, p/q)`, `stream:0`
/// uniform); at the first rejection resample from the renormalized residual
/// `max(0, p − q)` (`stream:1` gumbel). Output = sentinel-coded `[k]` Token.
/// `p` = target softmax; `q` = host `[k, vocab]`. Per-row `p(x_i)`/`q(x_i)` via
/// `GatherRow` (`out[i]=src[i,draft[i]]`).
pub fn spec_verify_lossless(vocab: u32, k: u32) -> Result<(Built, SpecLosslessKeys), BuildError> {
    let g = Graph::new(vocab);
    let target = g.intrinsic_logits_matrix_dyn(k); // [k, vocab]
    let q = g.host_matrix_dyn(DType::F32, k, vocab, Readiness::Submit);
    let draft = g.host_vector_dyn(DType::I32, k, Readiness::Submit);

    let p = dyn_softmax_rows(&target, vocab);
    let p_at = p.gather_row(&draft); // [k] p(x_i)
    let q_at = q.gather_row(&draft); // [k] q(x_i)

    let one = g.constant_f32_dyn(1.0);
    let ratio = p_at.div(&q_at).min_elem(&one);
    let u = g.rng_uniform_vec(0, k);
    let accept = ratio.gt(&u); // [k] bool

    let resample = dyn_residual_resample_rows(&g, &p, &q, 1, k, vocab); // [k] i32

    // Sentinel combine: c = cumsum(rejects); accepted prefix = c==0; first reject
    // = c==1 ∧ ¬accept; after = -1.
    let reject_f = dselect(&accept, &g.constant_f32_dyn(0.0), &g.constant_f32_dyn(1.0));
    let c = reject_f.cumsum();
    let accepted_mask = dyn_eq_const(&g, &c, 0.0);
    let not_accept = dselect(&accept, &g.constant_bool_dyn(false), &g.constant_bool_dyn(true));
    let boundary = dselect(&dyn_eq_const(&g, &c, 1.0), &not_accept, &g.constant_bool_dyn(false));

    let neg1 = g.constant_i32_dyn(-1).broadcast_vec(k);
    let inner = dselect(&boundary, &resample, &neg1);
    let out = dselect(&accepted_mask, &draft, &inner);
    g.output(&out, OutputKind::Token);

    let keys = SpecLosslessKeys {
        q: q.input_key().expect("q host input"),
        draft: draft.input_key().expect("draft host input"),
    };
    Ok((g.build()?, keys))
}

/// Per-row softmax over the target logits `[k, vocab]` — shared lossless building
/// block (exposed for tests / composition).
pub fn target_softmax_rows(logits: &DynValue, vocab: u32) -> DynValue {
    dyn_softmax_rows(logits, vocab)
}

// ============================================================================
// Measurement probes (Scalar-output graphs) — #15 enum->graph migration
// ============================================================================

/// **Entropy probe** — the Shannon entropy `H = -Σ p·log p` of the softmax
/// distribution, as a `Scalar`. Migrated from the built-in `entropy` sampler-enum
/// case to the program/graph contract (#15): a **Scalar-output measurement**
/// program (not a token producer), routed to the measurement read-back path by
/// its output kind. Its `canonical_kind` is `Custom` (Graph-authored — not a
/// token-sampler dispatch kind). `outputs = [Scalar]`.
///
/// Computed in log-space for numerical safety — `log_p = (x - max) - logΣexp(x-max)`
/// is always finite (no `log(0)`), and `p·log_p` with `p = exp(log_p)` underflowing
/// to 0 stays 0, never `NaN`.
pub fn entropy(vocab: u32) -> Result<Built, BuildError> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    let m = logits.reduce_max(); // scalar
    let shifted = logits.sub(&m); // x - max, [vocab]
    let lse = shifted.exp().reduce_sum().log(); // log Σ exp(x-max), scalar
    let log_p = shifted.sub(&lse); // log_softmax, [vocab]
    let h = log_p.exp().mul(&log_p).reduce_sum().neg(); // -Σ p·log p, scalar
    g.output(&h, OutputKind::Scalar);
    g.build()
}

/// **Raw logits** — the model's pre-softmax logit vector, passed through
/// unchanged. The program's single input (the intrinsic logits, `[vocab]`) is
/// its single output, declared [`OutputKind::Logits`]; `outputs = [Logits]`.
/// `canonical_kind` is `Custom` (Graph-authored — not a token-sampler dispatch
/// kind). Read back as native-endian `f32` bytes via `Output::read_bytes` /
/// `read_f32`.
pub fn logits(vocab: u32) -> Result<Built, BuildError> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    g.output(&logits, OutputKind::Logits);
    g.build()
}

/// **Full distribution** — the softmax over the model's entire output vocab,
/// `[vocab]` f32, declared [`OutputKind::Distribution`]; `outputs =
/// [Distribution]`. The companion token ids are the implicit vocab range
/// `0..vocab` (synthesized guest-side by [`Output::distribution`]), so this is a
/// values-only program — the explicit `(ids, values)` two-tensor form is for the
/// top-k case where the ids are a non-trivial selection. `canonical_kind` is
/// `Custom`. Computed with the standard max-shift for numerical stability.
pub fn distribution(vocab: u32) -> Result<Built, BuildError> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    let m = logits.reduce_max(); // scalar
    let exp = logits.sub(&m).exp(); // exp(x - max), [vocab]
    let probs = exp.div(&exp.reduce_sum()); // softmax, [vocab]
    g.output(&probs, OutputKind::Distribution);
    g.build()
}

/// **Mirostat v2 with an ARGMAX floor** (#19 fallback — proven-ops only). Same as
/// [`mirostat_floor`] but the rank floor is replaced by an **argmax floor** built
/// from `Ge`/`ReduceMax`/`Broadcast` only (all proven in the NVRTC mirostat kernel),
/// avoiding the custom-JIT `RankLe` (`PR_RANKLE`) path:
///
/// ```text
/// keep = (surprise ≤ μ)  OR  (logits ≥ max(logits))   // the max-logit token
/// ```
///
/// `logits ≥ max(logits)` is true only for the argmax, so the kept set is never
/// empty (≥1 token) → no `keep_count==0` → no token-0 → no μ runaway, robust to any
/// surprise floor. It is a `k=1` floor (greedy when μ is below the floor), so
/// [`mirostat_floor`] (k_min≥1 diversity) is preferred when `RankLe` is confirmed on
/// the custom-JIT path; this is the zero-codegen-risk fallback. `outputs = [Token, Scalar]`.
pub fn mirostat_argmax_floor(vocab: u32) -> Result<(Built, MirostatKeys), BuildError> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    let mu = g.host_scalar_dyn(DType::F32, Readiness::Submit);

    let probs = dyn_softmax(&logits);
    let surprise = probs.log().neg(); // -log p, [vocab]

    let keep_mu = mu.broadcast_vec(vocab).ge(&surprise); // surprise ≤ μ
    // Argmax floor: keep the max-logit token (`logits ≥ max ⟺ argmax`).
    let max_logit = logits.reduce_max().broadcast_vec(vocab);
    let keep_argmax = logits.ge(&max_logit);
    let always = g.constant_bool_dyn(true).broadcast_vec(vocab);
    let keep = dselect(&keep_mu, &always, &keep_argmax); // keep_mu OR keep_argmax

    let perturbed = logits.add(&g.rng_gumbel_vec(0, vocab));
    let neg_inf = g.constant_f32_dyn(f32::NEG_INFINITY).broadcast_vec(vocab);
    let token = dselect(&keep, &perturbed, &neg_inf).argmax();

    let s = surprise.gather(&token.broadcast_vec(1)).reduce_sum();

    g.output(&token, OutputKind::Token);
    g.output(&s, OutputKind::Scalar);

    let keys = MirostatKeys { mu: mu.input_key().expect("mu host input") };
    Ok((g.build()?, keys))
}
