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

/// Host-input keys for the [`grammar`] program.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GrammarKeys {
    /// Submit-bound additive logit-bias mask (`[vocab]` f32): `0` allowed,
    /// large-negative/`-inf` disallowed. Recomputed each step by the matcher.
    pub mask: TensorKey,
}

/// **Constrained / grammar decoding** (greedy). `argmax(logits + mask)`.
/// `outputs = [Token]`.
pub fn grammar(vocab: u32) -> Result<(Built, GrammarKeys), BuildError> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    let mask = g.host_vocab_vector_dyn(DType::F32, Readiness::Submit);

    let token = logits.add(&mask).argmax();
    g.output(&token, OutputKind::Token);

    let keys = GrammarKeys { mask: mask.input_key().expect("mask host input") };
    Ok((g.build()?, keys))
}

/// Host-input keys for [`grammar_sampled`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GrammarSampledKeys {
    pub mask: TensorKey,
}

/// **Constrained decoding, sampled.** `argmax(logits + mask + gumbel(stream:0))`.
pub fn grammar_sampled(vocab: u32) -> Result<(Built, GrammarSampledKeys), BuildError> {
    let g = Graph::new(vocab);
    let logits = g.intrinsic_logits_dyn();
    let mask = g.host_vocab_vector_dyn(DType::F32, Readiness::Submit);

    let biased = logits.add(&mask);
    let token = biased.add(&g.rng_gumbel_vec(0, vocab)).argmax();
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
