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
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_matrix_dyn(k); // [k, vocab]
    let draft = g.host_vector_dyn(DType::I32, k, Readiness::Submit);

    let target = logits.argmax(); // [k] i32 per-row greedy
    let matched = target.eq(&draft); // [k] bool

    // Accept the longest matching prefix: prefix-AND via cumprod over {0,1}.
    let ones = g.constant_f32_dyn(1.0).broadcast_vec(k);
    let zeros = g.constant_f32_dyn(0.0).broadcast_vec(k);
    let acc = dselect(&matched, &ones, &zeros).cumprod();
    let keep = acc.gt(&g.constant_f32_dyn(0.5));

    let neg1 = g.constant_i32_dyn(-1).broadcast_vec(k);
    let out = dselect(&keep, &draft, &neg1);
    g.output(&out, OutputKind::Token);

    let keys = SpecVerifyKeys { draft: draft.input_key().expect("draft host input") };
    Ok((g.build()?, keys))
}

/// Host-input keys for [`mtp_native_verify`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MtpNativeVerifyKeys {
    /// Submit-bound `[k]` i32 — the drafts EMBEDDED in this window (the
    /// previous step\'s MTP proposals; the decode loop knows them — it fed
    /// them to `input-tokens`).
    pub draft: TensorKey,
    /// Submit-bound `[k]` i32 lane indices `[0, 1, .., k-1]` (constant; v4
    /// has no vector-const/iota — feed once per attach).
    pub lanes_k: TensorKey,
    /// Submit-bound `[k+1]` f32 lane indices `[0.0, .., k.0]` (constant).
    pub lanes_k1: TensorKey,
}

/// Stage-2 PTIR-native MTP **match-verify with the bonus token** — the §6.1
/// `[K+1]` tail contract, byte-aligned to echo\'s `mtp_verify_tail` golden
/// (`out = select(i <= n_acc, picked, -1)`: accepted prefix + ONE bonus at
/// lane `n_acc` + `-1` sentinel; a full-reject step still advances by 1;
/// acceptance is a value, never a shape). Both logits are INTRINSICS
/// (`logits [k+1, vocab]` target + `mtp_logits [k, vocab]` proposals) — the
/// only host inputs are the embedded window drafts and two constant lane
/// vectors (v4 has no `iota`; PTIR\'s 0x64 is container-only).
///
/// Outputs: `[0]` = the `[k+1]`-Token committed window (prefix+bonus+`-1`);
/// `[1]` = the `[k]`-Token FRESH drafts (`argmax(mtp_logits)`) — next
/// window\'s proposals (feed back as `draft` next step).
pub fn mtp_native_verify(
    vocab: u32,
    k: u32,
) -> Result<(Built, MtpNativeVerifyKeys), BuildError> {
    let kp1 = k + 1;
    let g = Graph::new(vocab);
    let target = g.intrinsic_logits_matrix_dyn(kp1); // [k+1, vocab]
    let mtp = g.intrinsic_mtp_logits_matrix_dyn(k); // [k, vocab]
    let draft = g.host_vector_dyn(DType::I32, k, Readiness::Submit);
    let lanes_k = g.host_vector_dyn(DType::I32, k, Readiness::Submit);
    let lanes_k1 = g.host_vector_dyn(DType::F32, kp1, Readiness::Submit);

    let picked = target.argmax(); // [k+1] i32 — per-row target picks
    let head = picked.gather(&lanes_k); // picked[0..k]
    let hit = head.eq(&draft); // [k] bool — match-verify

    // n_acc = length of the leading all-hit run (cumprod over {1,0}).
    let ones = g.constant_f32_dyn(1.0).broadcast_vec(k);
    let zeros = g.constant_f32_dyn(0.0).broadcast_vec(k);
    let n_acc = dselect(&hit, &ones, &zeros).cumprod().reduce_sum(); // scalar f32

    // keep[i] = i <= n_acc  (bonus at lane n_acc included; f32 compare is
    // exact for lane indices — no cast op needed in v4).
    let keep = n_acc.broadcast_vec(kp1).ge(&lanes_k1); // [k+1] bool

    let neg1 = g.constant_i32_dyn(-1).broadcast_vec(kp1);
    let commit = dselect(&keep, &picked, &neg1); // [k+1] i32
    g.output(&commit, OutputKind::Token);

    let drf = mtp.argmax(); // [k] i32 — NEXT window\'s proposals
    g.output(&drf, OutputKind::Token);

    let keys = MtpNativeVerifyKeys {
        draft: draft.input_key().expect("draft host input"),
        lanes_k: lanes_k.input_key().expect("lanes_k host input"),
        lanes_k1: lanes_k1.input_key().expect("lanes_k1 host input"),
    };
    Ok((g.build()?, keys))
}

/// **Stage-2 device-resident MTP spec-decode** — [`mtp_native_verify`] plus the
/// on-device SEED extraction, so the `[seed, drafts]` next-window is composed by
/// the driver retain (echo drafts-channel path (a)) with NO host round-trip.
/// Identical verify+draft to [`mtp_native_verify`] (out[0]=`[k+1]` commit tail,
/// out[1]=`[k]` fresh drafts) plus **out[2]=`[1]` seed = `picked[n_acc]`** (the
/// bonus token — the host's `committed.last()` in `mtp-native-verify`'s loop,
/// materialized on-device so it never crosses to host).
///
/// The seed index `n_acc` is the accepted-draft count, computed by
/// `mtp_native_verify` as an f32 reduce_sum — but `Op::Gather` **requires an
/// integer index** (`validate.rs`: `!ti.dtype.is_int() -> dtype_err`). The
/// edge-safe i32 recovery reuses the `lanes_k1` host input: a one-hot
/// `n_acc == lanes_k1` has EXACTLY one set lane (at `n_acc`, including the
/// full-accept case `n_acc == k` → lane `k`), so its `argmax` is the unambiguous
/// i32 `n_acc` — where an `argmax(1 - cumprod)` would mis-fire on full-accept.
/// Zero new ops (broadcast/eq/select/argmax/gather all exist); `picked.gather`
/// is value-correct for the i32 `picked` intrinsic (same idiom as this program's
/// own `head = picked.gather(&lanes_k)`).
///
/// See `ptir-mtp-specdecode-spec §8.1`. The `[k+1]` window is assembled by
/// charlie's driver retain (`seed→row0`, `drafts→rows1..k`), NOT in-program.
pub fn mtp_specdecode(
    vocab: u32,
    k: u32,
) -> Result<(Built, MtpNativeVerifyKeys), BuildError> {
    let kp1 = k + 1;
    let g = Graph::new(vocab);
    let target = g.intrinsic_logits_matrix_dyn(kp1); // [k+1, vocab]
    let mtp = g.intrinsic_mtp_logits_matrix_dyn(k); // [k, vocab]
    let draft = g.host_vector_dyn(DType::I32, k, Readiness::Submit);
    let lanes_k = g.host_vector_dyn(DType::I32, k, Readiness::Submit);
    let lanes_k1 = g.host_vector_dyn(DType::F32, kp1, Readiness::Submit);

    let picked = target.argmax(); // [k+1] i32 — per-row target picks
    let head = picked.gather(&lanes_k); // picked[0..k]
    let hit = head.eq(&draft); // [k] bool — match-verify

    let ones = g.constant_f32_dyn(1.0).broadcast_vec(k);
    let zeros = g.constant_f32_dyn(0.0).broadcast_vec(k);
    let n_acc = dselect(&hit, &ones, &zeros).cumprod().reduce_sum(); // scalar f32

    // out[0] — the [k+1] committed tail (accepted prefix + bonus@n_acc + -1).
    let keep = n_acc.broadcast_vec(kp1).ge(&lanes_k1); // [k+1] bool
    let neg1 = g.constant_i32_dyn(-1).broadcast_vec(kp1);
    let commit = dselect(&keep, &picked, &neg1); // [k+1] i32
    g.output(&commit, OutputKind::Token);

    // out[1] — the [k] fresh drafts (next window's proposals).
    let drf = mtp.argmax(); // [k] i32
    g.output(&drf, OutputKind::Token);

    // out[2] — the [1] seed = picked[n_acc] (the bonus). Edge-safe i32 index via
    // the one-hot argmax (full-accept safe: n_acc==k → lane k).
    let onehot = n_acc.broadcast_vec(kp1).eq(&lanes_k1); // [k+1] bool, single 1 @ n_acc
    let one1 = g.constant_f32_dyn(1.0).broadcast_vec(kp1);
    let zero1 = g.constant_f32_dyn(0.0).broadcast_vec(kp1);
    let n_acc_i = dselect(&onehot, &one1, &zero1).argmax(); // scalar i32 = n_acc
    let seed = picked.gather(&n_acc_i.broadcast_vec(1)); // [1] i32 = picked[n_acc]
    g.output(&seed, OutputKind::Token);

    let keys = MtpNativeVerifyKeys {
        draft: draft.input_key().expect("draft host input"),
        lanes_k: lanes_k.input_key().expect("lanes_k host input"),
        lanes_k1: lanes_k1.input_key().expect("lanes_k1 host input"),
    };
    Ok((g.build()?, keys))
}

/// **Fire-0 BOOTSTRAP** for the device-resident swap ([`mtp_specdecode_device`]).
/// The MTP head + target greedy at the PROMPT's last REAL position → the first
/// `[seed, drafts]` window, so the device-resident loop has real drafts to verify
/// from fire 1. Fixes the zero-cascade charlie's A/B caught: `mtp_specdecode_device`
/// needs a full `[k+1]` window every fire, but a fire-0 window of `[prompt, 0…0]`
/// anchors the driver's MTP head on a **zero placeholder** (the last verify row) →
/// out[1] drafts `[0,0,0,0]` → the retain feeds zeros forever. This bootstrap fires
/// over the PROMPT (last row = the prompt's real last token) with NO verify (no
/// prior drafts exist): out[0]=`[1]` seed (target argmax @ decode pos), out[1]=`[k]`
/// drafts (MTP argmax @ the real anchor), out[2]=`[1]` seed. The carrier retains
/// out[2]→row0 / out[1]→rows1..k → fire 1's device-resident window. Bindings
/// `[Logits (M=1), MtpLogits]`; no host inputs.
pub fn mtp_specdecode_bootstrap(vocab: u32, k: u32) -> Result<Built, BuildError> {
    let g = Graph::new(vocab);
    // CONSISTENT k-row matrix geometry (mirrors `mtp_specdecode_device` /
    // `spec_verify_lossless` / mtp-native-verify — all fire k draft rows). A M=1
    // `intrinsic_logits_dyn()` target here was geometrically INCONSISTENT with the
    // `[k,vocab]` MtpLogits matrix: the matrix maps its K rows onto "the pass's draft
    // rows" (dynamic.rs), but a M=1 fire has only ONE draft row ⇒ both intrinsics
    // collapse onto the single anchor/target row (out[1] read the anchor token, not
    // the MTP draft; out[2]/seed read a stray row → 0). The matrix target + a
    // k-position (prompt + k-1 fillers) fire gives each intrinsic its K real rows.
    let target = g.intrinsic_logits_matrix_dyn(k); // [k, vocab] — one row per draft pos
    let mtp = g.intrinsic_mtp_logits_matrix_dyn(k); // [k, vocab]
    let picked = target.argmax(); // [k] — per-row target argmax
    // seed = picked[0] = row 0 (prompt-last position) argmax = the first real token.
    // Const-index gather (the device program's edge-safe idiom) — no host input.
    let idx0 = g.constant_i32_dyn(0).broadcast_vec(1); // [1] index → row 0
    let seed = picked.gather(&idx0); // [1] i32
    g.output(&seed, OutputKind::Token); // out[0] — commit = [seed]
    let drafts = mtp.argmax(); // [k] i32 — the first real drafts (MTP head, k positions)
    g.output(&drafts, OutputKind::Token); // out[1]
    g.output(&seed, OutputKind::Token); // out[2] — seed → row 0 of fire-1's window
    g.build()
}

/// Host-input keys for [`mtp_specdecode_device`] — ONLY the constant lane
/// vectors; the drafts are a device intrinsic ([`Graph::intrinsic_mtp_drafts_dyn`]),
/// not a host submit.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MtpSpecdecodeDeviceKeys {
    /// Submit-bound `[k]` i32 lane indices `[0..k-1]` (constant; v4 has no iota).
    pub lanes_k: TensorKey,
    /// Submit-bound `[k+1]` f32 lane indices `[0.0..k.0]` (constant).
    pub lanes_k1: TensorKey,
}

/// **Device-resident MTP spec-decode** — the drafts-channel swap of
/// [`mtp_native_verify`]/[`mtp_specdecode`]: the current window's drafts are read
/// from the retained `mtp_drafts` buffer via the `Binding::MtpDrafts` **device
/// intrinsic** ([`Graph::intrinsic_mtp_drafts_dyn`]) instead of a host `draft`
/// submit — ZERO host round-trip on the `[k]` drafts (the (b)-chain / seam-lock
/// (a) resolution; supersedes `ptir-mtp-specdecode-spec §8.1`'s MtpLogits-only
/// framing — `MtpLogits` is the NEXT drafts' logits, not the CURRENT drafts the
/// verify compares against). Outputs identical to [`mtp_specdecode`]:
/// out[0]=`[k+1]` commit tail (accepted + bonus@n_acc + -1), out[1]=`[k]` fresh
/// drafts (next window), out[2]=`[1]` seed = `picked[n_acc]`. Only host inputs =
/// the constant lane vectors. Charlie's driver source-selects the retained `[k]`
/// I32 buffer onto the `MtpDrafts` intrinsic.
pub fn mtp_specdecode_device(
    vocab: u32,
    k: u32,
) -> Result<(Built, MtpSpecdecodeDeviceKeys), BuildError> {
    let kp1 = k + 1;
    let g = Graph::new(vocab);
    let target = g.intrinsic_logits_matrix_dyn(kp1); // [k+1, vocab]
    let mtp = g.intrinsic_mtp_logits_matrix_dyn(k); // [k, vocab]
    let draft = g.intrinsic_mtp_drafts_dyn(k); // [k] i32 — DEVICE (was host submit)
    let lanes_k = g.host_vector_dyn(DType::I32, k, Readiness::Submit);
    let lanes_k1 = g.host_vector_dyn(DType::F32, kp1, Readiness::Submit);

    let picked = target.argmax(); // [k+1] i32
    let head = picked.gather(&lanes_k); // picked[0..k]
    let hit = head.eq(&draft); // [k] bool — match-verify vs device drafts

    let ones = g.constant_f32_dyn(1.0).broadcast_vec(k);
    let zeros = g.constant_f32_dyn(0.0).broadcast_vec(k);
    let n_acc = dselect(&hit, &ones, &zeros).cumprod().reduce_sum(); // scalar f32

    // out[0] — the [k+1] committed tail (accepted prefix + bonus@n_acc + -1).
    let keep = n_acc.broadcast_vec(kp1).ge(&lanes_k1);
    let neg1 = g.constant_i32_dyn(-1).broadcast_vec(kp1);
    let commit = dselect(&keep, &picked, &neg1);
    g.output(&commit, OutputKind::Token);

    // out[1] — the [k] fresh drafts (next window's proposals).
    let drf = mtp.argmax();
    g.output(&drf, OutputKind::Token);

    // out[2] — the [1] seed = picked[n_acc] (edge-safe i32 index via one-hot argmax).
    let onehot = n_acc.broadcast_vec(kp1).eq(&lanes_k1);
    let one1 = g.constant_f32_dyn(1.0).broadcast_vec(kp1);
    let zero1 = g.constant_f32_dyn(0.0).broadcast_vec(kp1);
    let n_acc_i = dselect(&onehot, &one1, &zero1).argmax();
    let seed = picked.gather(&n_acc_i.broadcast_vec(1));
    g.output(&seed, OutputKind::Token);

    let keys = MtpSpecdecodeDeviceKeys {
        lanes_k: lanes_k.input_key().expect("lanes_k host input"),
        lanes_k1: lanes_k1.input_key().expect("lanes_k1 host input"),
    };
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
