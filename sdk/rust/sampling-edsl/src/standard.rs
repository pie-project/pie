//! **Production standard-sampler programs** (Task #4 (B) batched de-hardwiring).
//!
//! The canonical, single-source-of-truth authoring for the driver-baked standard
//! sampler table (`pie_standard_samplers.h` — charlie bakes; echo's batched
//! executor binds). Unlike [`crate::sugar`] — whose `Sampler`→IR forms bake the
//! params as `Const` for the single-instance ergonomic path — these declare the
//! continuous params as **scalar `HostSubmit` inputs**, so one compiled program
//! serves a whole batch group across varying per-row params.
//!
//! ## Geometry: per-sequence vector `[vocab]`, batched at launch
//! A standard sampler is a **single-sequence** computation, authored as a
//! `[vocab]` **vector** program. The batched executor lowers it `batched=true`
//! and runs it over `N` independent sequences (`[N, vocab]` launch) — that is the
//! manager's "`[Ng,V]`" view; it is **not** a matrix-geometry program. (The
//! matrix `[rows, vocab]` geometry is spec-verify-ONLY: one sequence, `rows`
//! draft positions under a single seed.)
//!
//! ## RNG axis (load-bearing — `sample_temp` parity)
//! Selection RNG is the **ambient batch axis**: `Op::Rng{stream:0}` (no seed
//! operand; the per-row ambient seed `S[r]` is launch-delivered, Model B). Under
//! `batched=true` lowering this resolves to `col=j` (per-row `seed[r]`), bit-exact
//! with `sample_temp.cu` — the axis hotel's removal-gate (`c94fd762`, 0 mismatches
//! at B=128) proved. The axis is a **lowering choice, not a program property**
//! (alpha, `6657f23b`): authoring is axis-neutral `Rng{stream:0}` and production's
//! `batched=true` picks `col=j`. `stream:0 ⇒ seed_eff = S ^ 0xA5A5A5A5`.
//!
//! ## Param contract (manager/alpha/echo/golf-ratified, Task #4)
//! - **Per-row HostSubmit (scalar inputs):** temperature, min_p, top_p `p`.
//! - **Group-level immediate (program-identity / cache key):** top-k `k`
//!   (`Predicate::RankLe(u32)` is a baked immediate — grouping buckets by
//!   `(sampler_type, k)`).
//! - **Ambient:** the RNG seed `S` — never a host input.
//!
//! Covers the full standard set: argmax, temperature, min-p, top-k, top-p,
//! top-k-top-p (the (B) production order rides them in incrementally).

use crate::builder::{Built, BuildError, Graph, OutputKind};
use crate::dynamic::{
    dselect, dyn_gumbel_argmax, dyn_mask_to_score, dyn_min_p_mask, dyn_softmax,
    dyn_temperature_scale,
};
use crate::ir::{DType, Readiness, TensorKey};
use crate::kinds::CanonicalKind;
use alloc::vec::Vec;

impl From<StandardSampler> for CanonicalKind {
    fn from(s: StandardSampler) -> Self {
        match s {
            StandardSampler::Argmax => CanonicalKind::Argmax,
            StandardSampler::Temperature => CanonicalKind::Temperature,
            StandardSampler::MinP => CanonicalKind::MinP,
            StandardSampler::TopK { .. } => CanonicalKind::TopK,
            StandardSampler::TopP => CanonicalKind::TopP,
            StandardSampler::TopKTopP { .. } => CanonicalKind::TopKTopP,
        }
    }
}

/// The per-row HostSubmit parameter keys a standard-sampler program declares. A
/// field is `Some(key)` iff the program consumes that param — bind row `r`'s
/// value under the key each fire. The RNG seed is intentionally absent: it is
/// ambient (Model B), launch-delivered per row, never a host input.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct StdParamKeys {
    /// Temperature `T` (f32 scalar submit) — scales logits by `1/T`.
    pub temperature: Option<TensorKey>,
    /// Min-p `p` (f32 scalar submit) — keep `logit ≥ max_logit + log(p)`.
    pub min_p: Option<TensorKey>,
    /// Top-p mass `p` (f32 scalar submit) — nucleus threshold.
    pub top_p: Option<TensorKey>,
}

/// The standard samplers authored as parametric per-sequence programs. The
/// `k`-bearing forms bake `k` as a build-time immediate (`Predicate::RankLe`),
/// keying the program cache by `(sampler_type, k)` so intra-group `k` never
/// varies; the continuous params (temperature / top-p `p` / min_p) are scalar
/// HostSubmit inputs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StandardSampler {
    /// Greedy `argmax(logits)` — no params, no RNG.
    Argmax,
    /// Temperature-scaled Gumbel-max multinomial.
    Temperature,
    /// Min-p truncation (logit space) + temperature, Gumbel-max.
    MinP,
    /// Top-k: keep the top `k` by logit (`k` baked immediate, group key) +
    /// temperature, Gumbel-max.
    TopK {
        /// The (group-uniform) top-k cutoff — baked into `Predicate::RankLe`.
        k: u32,
    },
    /// Top-p (nucleus): keep the smallest prob-mass-`p` set + temperature,
    /// Gumbel-max. `p` and temperature are scalar inputs.
    TopP,
    /// Top-k ∩ top-p (`k` baked immediate; `p`/temperature scalar inputs).
    TopKTopP {
        /// The (group-uniform) top-k cutoff — baked into `Predicate::RankLe`.
        k: u32,
    },
}

/// Build the parametric program for `kind` at runtime `vocab`. `outputs =
/// [Token]` (one sampled token per sequence; `[N]` under the batched launch).
pub fn build_standard(
    kind: StandardSampler,
    vocab: u32,
) -> Result<(Built, StdParamKeys), BuildError> {
    match kind {
        StandardSampler::Argmax => Ok((argmax(vocab)?, StdParamKeys::default())),
        StandardSampler::Temperature => temperature(vocab),
        StandardSampler::MinP => min_p(vocab),
        StandardSampler::TopK { k } => top_k(vocab, k),
        StandardSampler::TopP => top_p(vocab),
        StandardSampler::TopKTopP { k } => top_k_top_p(vocab, k),
    }
}

/// **Greedy** — `argmax(logits)`. No params, no RNG (so it needs no ambient seed;
/// the (B) order lands it first to prove the batched bind + scatter).
pub fn argmax(vocab: u32) -> Result<Built, BuildError> {
    let g = Graph::new(vocab);
    g.set_canonical_kind(CanonicalKind::Argmax);
    let logits = g.intrinsic_logits_dyn(); // [vocab]
    let token = logits.argmax();
    g.output(&token, OutputKind::Token);
    g.build()
}

/// **Temperature multinomial** — `argmax(logits/T + Gumbel(stream:0))`. `T` is a
/// scalar HostSubmit input (per-row at launch); seed is ambient batch axis.
pub fn temperature(vocab: u32) -> Result<(Built, StdParamKeys), BuildError> {
    let g = Graph::new(vocab);
    g.set_canonical_kind(CanonicalKind::Temperature);
    let logits = g.intrinsic_logits_dyn();
    let temp = g.host_scalar_dyn(DType::F32, Readiness::Submit);

    let scaled = dyn_temperature_scale(&logits, &temp);
    let token = dyn_gumbel_argmax(&g, &scaled, 0, vocab);
    g.output(&token, OutputKind::Token);

    let keys = StdParamKeys {
        temperature: Some(temp.input_key().expect("temperature host input")),
        ..Default::default()
    };
    Ok((g.build()?, keys))
}

/// **Min-p** — keep tokens with `logit ≥ max_logit + log(min_p)` (logit space, on
/// the raw logits, matching [`crate::sugar`]'s parity-locked form), then
/// temperature-scaled Gumbel-max over the kept set. `T` and `min_p` are scalar
/// HostSubmit inputs; seed is ambient batch axis (`stream:0`).
pub fn min_p(vocab: u32) -> Result<(Built, StdParamKeys), BuildError> {
    let g = Graph::new(vocab);
    g.set_canonical_kind(CanonicalKind::MinP);
    let logits = g.intrinsic_logits_dyn();
    let temp = g.host_scalar_dyn(DType::F32, Readiness::Submit);
    let minp = g.host_scalar_dyn(DType::F32, Readiness::Submit);

    let scaled = dyn_temperature_scale(&logits, &temp);
    let keep = dyn_min_p_mask(&logits, &minp);
    let score = dyn_mask_to_score(&g, &keep, &scaled);
    let token = dyn_gumbel_argmax(&g, &score, 0, vocab);
    g.output(&token, OutputKind::Token);

    let keys = StdParamKeys {
        temperature: Some(temp.input_key().expect("temperature host input")),
        min_p: Some(minp.input_key().expect("min_p host input")),
        ..Default::default()
    };
    Ok((g.build()?, keys))
}

/// **Top-k** — keep the top `k` tokens by logit, then temperature-scaled
/// Gumbel-max over the kept set. `k` is a **baked immediate** (the group key);
/// temperature is a scalar HostSubmit input; seed is ambient (`stream:0`).
pub fn top_k(vocab: u32, k: u32) -> Result<(Built, StdParamKeys), BuildError> {
    let g = Graph::new(vocab);
    g.set_canonical_kind(CanonicalKind::TopK);
    let logits = g.intrinsic_logits_dyn();
    let temp = g.host_scalar_dyn(DType::F32, Readiness::Submit);

    let scaled = dyn_temperature_scale(&logits, &temp);
    let keep = logits.pivot_rank_le(k); // k baked immediate
    let score = dyn_mask_to_score(&g, &keep, &scaled);
    let token = dyn_gumbel_argmax(&g, &score, 0, vocab);
    g.output(&token, OutputKind::Token);

    let keys = StdParamKeys {
        temperature: Some(temp.input_key().expect("temperature host input")),
        ..Default::default()
    };
    Ok((g.build()?, keys))
}

/// **Top-p** (nucleus) — keep the smallest set of tokens whose cumulative
/// (temperature-scaled) softmax mass reaches `p`, then Gumbel-max over it. `T`
/// and `p` are scalar HostSubmit inputs; seed is ambient (`stream:0`).
pub fn top_p(vocab: u32) -> Result<(Built, StdParamKeys), BuildError> {
    let g = Graph::new(vocab);
    g.set_canonical_kind(CanonicalKind::TopP);
    let logits = g.intrinsic_logits_dyn();
    let temp = g.host_scalar_dyn(DType::F32, Readiness::Submit);
    let pv = g.host_scalar_dyn(DType::F32, Readiness::Submit);

    let scaled = dyn_temperature_scale(&logits, &temp);
    let probs = dyn_softmax(&scaled);
    let keep = probs.pivot_cummass_le(&pv);
    let score = dyn_mask_to_score(&g, &keep, &scaled);
    let token = dyn_gumbel_argmax(&g, &score, 0, vocab);
    g.output(&token, OutputKind::Token);

    let keys = StdParamKeys {
        temperature: Some(temp.input_key().expect("temperature host input")),
        top_p: Some(pv.input_key().expect("top_p host input")),
        ..Default::default()
    };
    Ok((g.build()?, keys))
}

/// **Top-k ∩ top-p** — keep tokens passing *both* the top-`k` (baked immediate)
/// and the top-`p` (scalar input) cuts, then temperature-scaled Gumbel-max. `T`
/// and `p` are scalar HostSubmit inputs; seed is ambient (`stream:0`).
pub fn top_k_top_p(vocab: u32, k: u32) -> Result<(Built, StdParamKeys), BuildError> {
    let g = Graph::new(vocab);
    g.set_canonical_kind(CanonicalKind::TopKTopP);
    let logits = g.intrinsic_logits_dyn();
    let temp = g.host_scalar_dyn(DType::F32, Readiness::Submit);
    let pv = g.host_scalar_dyn(DType::F32, Readiness::Submit);

    let scaled = dyn_temperature_scale(&logits, &temp);
    let probs = dyn_softmax(&scaled);
    let keep_k = logits.pivot_rank_le(k); // k baked immediate
    let keep_p = probs.pivot_cummass_le(&pv);
    let f = g.constant_bool_dyn(false).broadcast_vec(vocab);
    let keep = dselect(&keep_k, &keep_p, &f); // keep_k AND keep_p
    let score = dyn_mask_to_score(&g, &keep, &scaled);
    let token = dyn_gumbel_argmax(&g, &score, 0, vocab);
    g.output(&token, OutputKind::Token);

    let keys = StdParamKeys {
        temperature: Some(temp.input_key().expect("temperature host input")),
        top_p: Some(pv.input_key().expect("top_p host input")),
        ..Default::default()
    };
    Ok((g.build()?, keys))
}

// ============================================================================
// #12 reference set: canonical program -> kind (the hash-table source)
// ============================================================================

/// The canonical bytecode + [`CanonicalKind`] for one standard sampler — the
/// `program → kind` reference the #12 classifier hash-matches against (alpha's
/// table interns `program_hash(bytecode) → kind`). For a **k-bearing** kind the
/// bytecode is k-specific (the baked `RankLe(k)` immediate); the **k-invariant**
/// kinds (their continuous params are host-submit inputs, not baked) have a
/// single canonical bytecode per `vocab`.
pub fn standard_program(
    kind: StandardSampler,
    vocab: u32,
) -> Result<(Vec<u8>, CanonicalKind), BuildError> {
    let (built, _) = build_standard(kind, vocab)?;
    Ok((built.lower().bytecode, kind.into()))
}

/// The **k-invariant** standard programs (Argmax, Temperature, MinP, TopP) for
/// runtime `vocab` — each has a single canonical bytecode per `vocab`, so their
/// hashes are stable reference entries for the #12 `{hash → kind}` table. The
/// **k-bearing** kinds (TopK, TopKTopP) bake `k` into a `RankLe` immediate so
/// their bytecode varies by `k`: obtain those per-`k` via [`standard_program`],
/// or recognize them structurally via the op-shape matcher.
pub fn standard_programs(vocab: u32) -> Result<Vec<(Vec<u8>, CanonicalKind)>, BuildError> {
    [
        StandardSampler::Argmax,
        StandardSampler::Temperature,
        StandardSampler::MinP,
        StandardSampler::TopP,
    ]
    .into_iter()
    .map(|k| standard_program(k, vocab))
    .collect()
}

/// The `{ program_hash → CanonicalKind }` reference entries for the 4 k-invariant
/// standard kinds at `vocab` — the exact-hash half of the #12 recognizer.
///
/// Each hash is [`program_hash`](crate::ir::program_hash) (the single FNV-1a ==
/// the driver `ProgramHandle`) of the canonical bytecode from [`standard_programs`].
/// The driver self-recognizes an attached program by matching its hash against
/// these — which equal the driver's *own* baked-program hashes by canonical
/// encode — and routes to the kind. The k-bearing kinds (TopK/TopKTopP) bake `k`
/// into a `RankLe` immediate so their bytecode/hash varies by `k`; they are
/// recognized structurally (op-shape), not by these fixed entries.
pub fn standard_program_hashes(vocab: u32) -> Result<Vec<(u64, CanonicalKind)>, BuildError> {
    Ok(standard_programs(vocab)?
        .into_iter()
        .map(|(bytecode, kind)| (crate::ir::program_hash(&bytecode), kind))
        .collect())
}

/// A representative non-zero `k` used to build the k-bearing canonical references.
/// The choice is irrelevant: [`canonicalize_op_shape`](crate::ir::canonicalize_op_shape)
/// zeroes the `RankLe(k)` immediate, so any `k` yields the same canonical bytecode.
const CANONICAL_REPRESENTATIVE_K: u32 = 40;

/// The **canonicalized** op-shape references for the 2 k-bearing kinds
/// (`TopK`, `TopKTopP`) at runtime `vocab` — the phase-2 byte-level guard
/// (the `hex` column of the canonical fixture).
///
/// Unlike the 4 k-invariant kinds (whose bytecode is already fixed per `vocab`),
/// the k-bearing kinds bake `k` into a `RankLe` immediate, so their raw bytecode
/// varies by `k`. This builds each at a representative `k` and applies alpha's
/// [`canonicalize_op_shape`](crate::ir::canonicalize_op_shape) (which zeroes the
/// `RankLe(k)` immediate — the *sole* parametric byte, since temperature / top-p
/// `p` are host-submit inputs). The result is **k-invariant**: every `TopK{any k}`
/// canonicalizes to this one bytecode. Precise, not over-broad — a custom program
/// with an extra op canonicalizes differently → no false-match → CustomJIT.
pub fn standard_programs_canonical(vocab: u32) -> Result<Vec<(Vec<u8>, CanonicalKind)>, BuildError> {
    [
        StandardSampler::TopK {
            k: CANONICAL_REPRESENTATIVE_K,
        },
        StandardSampler::TopKTopP {
            k: CANONICAL_REPRESENTATIVE_K,
        },
    ]
    .into_iter()
    .map(|kind| {
        let (built, _) = build_standard(kind, vocab)?;
        let mut program = built.program.clone();
        crate::ir::canonicalize_op_shape(&mut program);
        Ok((crate::ir::encode(&program), kind.into()))
    })
    .collect()
}

/// The `{ canonical program_hash → CanonicalKind }` reference entries for the 2
/// k-bearing standard kinds at `vocab` — the phase-2 half of the #12 recognizer,
/// mirroring [`standard_program_hashes`].
///
/// On a phase-1 exact-hash miss, the recognizer canonicalizes the decoded program
/// ([`canonicalize_op_shape`](crate::ir::canonicalize_op_shape)) and hashes it with
/// the same FNV-1a [`program_hash`](crate::ir::program_hash); a match against these
/// entries recognizes `TopK` / `TopKTopP` for *any* `k` (the kind), and
/// [`extract_top_k`] reads the `k` for the dedicated kernel's `per_slot_top_k`.
pub fn standard_program_hashes_canonical(
    vocab: u32,
) -> Result<Vec<(u64, CanonicalKind)>, BuildError> {
    Ok(standard_programs_canonical(vocab)?
        .into_iter()
        .map(|(bytecode, kind)| (crate::ir::program_hash(&bytecode), kind))
        .collect())
}

/// Extract the top-k cutoff `k` from a recognized `TopK` / `TopKTopP` program —
/// the baked `RankLe(k)` immediate. `None` if there is no `RankLe` op (not a
/// k-bearing standard form). The #12 **extract** path reads this to populate the
/// dedicated kernel's `per_slot_top_k` ("op-shape yields `k`") — the *kind* comes
/// from the canonical hash, the `k` from here.
pub fn extract_top_k(program: &crate::ir::SamplingProgram) -> Option<u32> {
    program.ops.iter().find_map(|op| match op {
        crate::ir::Op::PivotThreshold {
            predicate: crate::ir::Predicate::RankLe(k),
            ..
        } => Some(*k),
        _ => None,
    })
}
