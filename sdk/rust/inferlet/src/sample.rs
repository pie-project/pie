//! Sampler specifications.
//!
//! [`Sampler`] is the ergonomic "give me a token" enum (top-p, top-k,
//! multinomial, …). Under the program-based front door it is **sugar that
//! lowers to a tensor program** (WS5 de-hardwiring): [`Sampler::lower`] produces
//! a Sampling-IR program that the host runs through the IR-JIT path, replacing
//! the retired hardwired WIT `sampler` variant.
//!
//! NOTE (Stage 2): the lowering currently produces a `sampling-edsl`
//! `LoweredProgram` (bytecode). Foxtrot's guest emit re-targets it to a
//! [`tensor::Program`](crate::tensor::Program) so [`Forward::sampler`] can
//! attach it directly; until then the `Sampler → program` attach seam in
//! [`crate::forward`] is a tracked stub.

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
    /// True when this sampler is deterministic greedy decoding.
    pub const fn is_argmax(&self) -> bool {
        matches!(self, Self::Argmax)
    }

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

impl From<Sampler> for sampling_edsl::SamplerSpec {
    fn from(s: Sampler) -> Self {
        use sampling_edsl::SamplerSpec as Spec;
        match s {
            Sampler::Argmax => Spec::Argmax,
            Sampler::TopP { temperature, p } => Spec::TopP { temperature, p },
            Sampler::TopK { temperature, k } => Spec::TopK { temperature, k },
            Sampler::MinP { temperature, p } => Spec::MinP { temperature, p },
            Sampler::TopKTopP { temperature, k, p } => Spec::TopKTopP { temperature, k, p },
            // The IR multinomial is a single temperature-scaled draw; the
            // legacy `draws` per-sample multiplier has no IR analog (always 1
            // token per slot at M=1), so it is dropped.
            Sampler::Multinomial { temperature, .. } => Spec::Multinomial { temperature },
        }
    }
}

impl Sampler {
    /// Lower this sampler to a Sampling-IR [`LoweredProgram`](crate::program::LoweredProgram)
    /// for the given runtime `vocab`. The program outputs a single
    /// [`OutputKind::Token`]. Under RNG model B the seed is **ambient** (the
    /// runtime's per-row seed) and the stream id is static, so a lowered
    /// sampler declares no seed host input.
    ///
    /// This is the WS5 de-hardwiring bridge. **Stage 2**: foxtrot's guest emit
    /// re-targets the lowered program to a [`tensor::Program`](crate::tensor::Program)
    /// so it attaches via [`Forward::sampler`](crate::forward::Forward::sampler).
    pub fn lower(&self, vocab: u32) -> crate::Result<crate::program::LoweredProgram> {
        sampling_edsl::sugar::lower_sampler(self.clone().into(), vocab)
            .map_err(|e| format!("Sampler::lower: {e:?}"))
    }
}

#[cfg(test)]
mod ws5_sugar_tests {
    use super::Sampler;
    use sampling_edsl::OutputKind;

    const VOCAB: u32 = 128;

    /// Every sampler variant lowers to a valid single-Token program.
    #[test]
    fn all_variants_lower_to_token_program() {
        let samplers = [
            Sampler::Argmax,
            Sampler::TopP { temperature: 0.8, p: 0.9 },
            Sampler::TopK { temperature: 0.8, k: 40 },
            Sampler::MinP { temperature: 0.8, p: 0.05 },
            Sampler::TopKTopP { temperature: 0.8, k: 40, p: 0.9 },
            Sampler::Multinomial { temperature: 0.8, draws: 1 },
        ];
        for s in samplers {
            let prog = s.lower(VOCAB).expect("lower");
            assert_eq!(prog.outputs, vec![OutputKind::Token], "{s:?}");
            assert!(!prog.bytecode.is_empty(), "{s:?}");
        }
    }
}
