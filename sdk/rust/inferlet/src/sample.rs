//! Sampler and probe specifications.
//!
//! The host's `forward-pass.sampler` slot accepts a single WIT variant that
//! folds two unrelated concerns together:
//!
//! - **Sampling**: pick a token from the next-token distribution (top-p,
//!   top-k, multinomial, …). Produces an `Output::Token` slot.
//! - **Probing**: read out shape information without picking a token (raw
//!   logits, distribution, logprobs, entropy). Produces an `Output::Logits` /
//!   `Distribution` / `Logprobs` / `Entropy` slot.
//!
//! The SDK keeps these distinct. [`Sampler`] is for "give me a token";
//! probes are unit-like or small structs (e.g. [`Distribution`], [`Logits`])
//! that double as their own type-level marker so each
//! [`ProbeHandle`](crate::forward::ProbeHandle) statically dispatches to the
//! matching `Output::*` accessor. Both compile down to the same WIT slot,
//! so a single forward pass can mix them freely.

use crate::pie::core::inference::Sampler as WitSampler;

// =============================================================================
// Sampler — picks a token
// =============================================================================

/// A token-producing sampler. Attach to one or more positions via
/// [`Forward::sample`](crate::forward::Forward::sample) to get back a sampled token id
/// per position.
#[derive(Clone, Debug)]
pub enum Sampler {
    /// Greedy: pick the maximum-probability token.
    Argmax,
    /// Top-p (nucleus) sampling. `temperature = 0.0` collapses to argmax.
    TopP { temperature: f32, p: f32 },
    /// Top-k sampling: sample from the top `k` tokens by probability.
    TopK { temperature: f32, k: u32 },
    /// Min-p sampling: keep tokens with probability ≥ `p × max_prob`.
    MinP { temperature: f32, p: f32 },
    /// Combined top-k + top-p: first restrict to top `k`, then apply nucleus `p`.
    TopKTopP { temperature: f32, k: u32, p: f32 },
    /// Plain multinomial: sample from the full temperature-scaled distribution.
    /// `draws` is a per-sample multiplier (typically 1).
    Multinomial { temperature: f32, draws: u32 },
}

impl Sampler {
    /// Top-p (nucleus) sampling. `temperature = 0.0` collapses to argmax.
    pub const fn top_p(temperature: f32, p: f32) -> Self {
        Self::TopP { temperature, p }
    }
    /// Top-k sampling: sample from the top `k` tokens by probability.
    pub const fn top_k(temperature: f32, k: u32) -> Self {
        Self::TopK { temperature, k }
    }
    /// Min-p sampling: keep tokens with probability ≥ `p × max_prob`.
    pub const fn min_p(temperature: f32, p: f32) -> Self {
        Self::MinP { temperature, p }
    }
    /// Combined top-k + top-p: first restrict to top `k`, then nucleus `p`.
    pub const fn top_k_top_p(temperature: f32, k: u32, p: f32) -> Self {
        Self::TopKTopP { temperature, k, p }
    }
    /// Plain multinomial after temperature scaling. `draws` is a per-sample
    /// multiplier (typically 1).
    pub const fn multinomial(temperature: f32, draws: u32) -> Self {
        Self::Multinomial { temperature, draws }
    }
}

impl From<Sampler> for WitSampler {
    fn from(s: Sampler) -> Self {
        match s {
            // The host treats `top-p` with `temperature = 0` as argmax — we
            // reuse that path so callers don't pay for an extra variant.
            Sampler::Argmax => WitSampler::TopP((0.0, 1.0)),
            Sampler::TopP { temperature, p } => WitSampler::TopP((temperature, p)),
            Sampler::TopK { temperature, k } => WitSampler::TopK((temperature, k)),
            Sampler::MinP { temperature, p } => WitSampler::MinP((temperature, p)),
            Sampler::TopKTopP { temperature, k, p } => WitSampler::TopKTopP((temperature, k, p)),
            Sampler::Multinomial { temperature, draws } => {
                WitSampler::Multinomial((temperature, draws))
            }
        }
    }
}

// =============================================================================
// Probes — distribution access
// =============================================================================

/// Probe spec. Each implementation describes a single probe and its output
/// shape; [`Forward::probe`](crate::forward::Forward::probe) consumes a value of this
/// trait and returns a `ProbeHandle<P::Out>` whose phantom type matches the
/// `Output::*` accessor that decodes the result.
///
/// `Self::Out` is the *output marker* — usually `Self`, except for
/// [`Logprob`] which collapses to [`Logprobs`] because both produce the same
/// `SlotOutput::Logprobs` shape and are read by the same accessor.
pub trait Probe: sealed::Sealed {
    /// Output marker selecting the matching `Output::*` accessor.
    type Out;
    /// Lower into the WIT sampler variant.
    fn into_wit(self) -> WitSampler;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Logits {}
    impl Sealed for super::Distribution {}
    impl Sealed for super::Logprob {}
    impl Sealed for super::Logprobs {}
    impl Sealed for super::Entropy {}
}

// ── Markers / probe specs ─────────────────────────────────────────────

/// Pre-softmax, untemperatured logits as packed native-endian f32 bytes
/// (length = `vocab_size * 4`). Decode via `bytemuck::cast_slice` or equiv.
#[derive(Copy, Clone, Debug)]
pub struct Logits;

/// Top-`k` token ids paired with their (post-softmax, temperature-scaled)
/// probabilities. `k = 0` returns the full vocabulary.
#[derive(Copy, Clone, Debug)]
pub struct Distribution {
    pub temperature: f32,
    pub k: u32,
}

/// `log p(token | context)` at this position, without temperature scaling.
/// Returned as a length-1 logprob list — read with [`Output::logprobs`].
///
/// [`Output::logprobs`]: crate::forward::Output::logprobs
#[derive(Copy, Clone, Debug)]
pub struct Logprob(pub u32);

/// `log p(t | context)` for each `t` in the list, without temperature
/// scaling. Returned as a length-K list in the order requested.
#[derive(Clone, Debug)]
pub struct Logprobs(pub Vec<u32>);

/// Shannon entropy `H(p) = -sum(p log p)` of the unscaled distribution.
#[derive(Copy, Clone, Debug)]
pub struct Entropy;

// ── Probe impls ───────────────────────────────────────────────────────

impl Probe for Logits {
    type Out = Logits;
    fn into_wit(self) -> WitSampler {
        WitSampler::RawLogits
    }
}

impl Probe for Distribution {
    type Out = Distribution;
    fn into_wit(self) -> WitSampler {
        WitSampler::Dist((self.temperature, self.k))
    }
}

impl Probe for Logprob {
    /// Both `Logprob` and `Logprobs` collapse to the same output shape — the
    /// host returns a length-K list per slot, with `K = 1` for the singular
    /// case. Sharing the marker lets a single `output.logprobs(h)` accessor
    /// serve both.
    type Out = Logprobs;
    fn into_wit(self) -> WitSampler {
        WitSampler::Logprob(self.0)
    }
}

impl Probe for Logprobs {
    type Out = Logprobs;
    fn into_wit(self) -> WitSampler {
        WitSampler::Logprobs(self.0)
    }
}

impl Probe for Entropy {
    type Out = Entropy;
    fn into_wit(self) -> WitSampler {
        WitSampler::Entropy
    }
}
