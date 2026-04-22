pub mod primitives;

pub use primitives::{
    EmittedHistory, apply_repetition_penalty, apply_top_k, apply_top_p, weighted_sample,
};

/// Model-authored sampling defaults sourced from the engine's `generation_config.json`.
///
/// Only includes knobs a model publishes in its gen_config: the four filter-style
/// fields (`temperature`, `top_p`, `top_k`, `min_p`) plus `repetition_penalty`.
/// `frequency_penalty` and `presence_penalty` are intentionally absent — HF
/// `generation_config.json` never carries them; they are client-intent only and
/// live exclusively on `SamplingOverrides`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerationDefaults {
    pub temperature:        Option<f32>,
    pub top_p:              Option<f32>,
    pub top_k:              Option<u32>,
    pub min_p:              Option<f32>,
    pub repetition_penalty: Option<f32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingOverrides {
    pub temperature:        Option<f32>,
    pub top_p:              Option<f32>,
    pub top_k:              Option<u32>,
    pub min_p:              Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty:  Option<f32>,  // F3 (client-only, not in gen_config)
    pub presence_penalty:   Option<f32>,  // F3 (client-only, not in gen_config)
}

/// Token-penalty parameters attached to every `Sampler` variant.
///
/// Neutral defaults (`repetition=1.0, frequency=0.0, presence=0.0`) match
/// vLLM's `SamplingParams` defaults — multiplying by 1.0 or adding 0.0 is a
/// mathematical no-op, so code paths that unconditionally apply penalties stay
/// correct when the client hasn't set them.
///
/// `frequency` and `presence` have no source in HF `generation_config.json`
/// (confirmed via survey of 9 models); they are client-only and only `repetition`
/// can be filled from `GenerationDefaults`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Penalties {
    pub repetition: f32,  // 1.0 = neutral
    pub frequency:  f32,  // 0.0 = neutral
    pub presence:   f32,  // 0.0 = neutral
}

impl Default for Penalties {
    fn default() -> Self { Self { repetition: 1.0, frequency: 0.0, presence: 0.0 } }
}

pub enum Sampler {
    Custom {
        temperature: f32,
        sampler: Box<dyn Sample>,
        penalties: Penalties,
    },
    Multinomial {
        temperature: f32,
        penalties: Penalties,
    },
    TopP {
        temperature: f32,
        top_p: f32,
        penalties: Penalties,
    },
    TopK {
        temperature: f32,
        top_k: u32,
        penalties: Penalties,
    },
    MinP {
        temperature: f32,
        min_p: f32,
        penalties: Penalties,
    },
    TopKTopP {
        temperature: f32,
        top_k: u32,
        top_p: f32,
        penalties: Penalties,
    },
}

impl Sampler {
    pub fn greedy() -> Self {
        Sampler::Multinomial { temperature: 0.0, penalties: Penalties::default() }
    }

    pub fn top_p(temperature: f32, top_p: f32) -> Self {
        Sampler::TopP { temperature, top_p, penalties: Penalties::default() }
    }

    pub fn top_k(temperature: f32, top_k: u32) -> Self {
        Sampler::TopK { temperature, top_k, penalties: Penalties::default() }
    }

    pub fn min_p(temperature: f32, min_p: f32) -> Self {
        Sampler::MinP { temperature, min_p, penalties: Penalties::default() }
    }

    pub fn top_k_top_p(temperature: f32, top_k: u32, top_p: f32) -> Self {
        Sampler::TopKTopP {
            temperature,
            top_k,
            top_p,
            penalties: Penalties::default(),
        }
    }

    pub fn reasoning() -> Self {
        Self::top_k_top_p(0.6, 20, 0.95)
    }

    /// Merge client-supplied `SamplingOverrides` with model-authored
    /// `GenerationDefaults` into a ready-to-use `Sampler` variant.
    ///
    /// Precedence: `overrides.field.or(defaults.field).unwrap_or(neutral)`.
    /// Client-set fields win; the model's generation_config fills in; neutrals
    /// (temperature=1.0, top_p=1.0 disabled, top_k=0 disabled, min_p=0.0
    /// disabled) are the last resort.
    ///
    /// Filter selection (temperature, top_p, top_k, min_p) and penalty
    /// parameters (repetition, frequency, presence) are resolved independently.
    /// Penalty fields flow into every variant via the `Penalties` struct;
    /// `frequency_penalty` and `presence_penalty` are client-only (no defaults
    /// source) while `repetition_penalty` can come from `GenerationDefaults`.
    pub fn merge(overrides: SamplingOverrides, defaults: GenerationDefaults) -> Self {
        let temperature = overrides.temperature.or(defaults.temperature).unwrap_or(1.0);
        let top_p = overrides.top_p.or(defaults.top_p);
        let top_k = overrides.top_k.or(defaults.top_k);
        let min_p = overrides.min_p.or(defaults.min_p);
        // top_p == 1.0 or top_k == 0 means "disabled"; treat as None.
        let top_p_active = top_p.filter(|v| *v < 1.0 && *v > 0.0);
        let top_k_active = top_k.filter(|v| *v > 0);
        let min_p_active = min_p.filter(|v| *v > 0.0);
        let penalties = Penalties {
            repetition: overrides.repetition_penalty.or(defaults.repetition_penalty).unwrap_or(1.0),
            frequency:  overrides.frequency_penalty.unwrap_or(0.0),   // no defaults source
            presence:   overrides.presence_penalty.unwrap_or(0.0),    // no defaults source
        };
        match (top_k_active, top_p_active, min_p_active) {
            (Some(k), Some(p), _) => Sampler::TopKTopP { temperature, top_k: k, top_p: p, penalties },
            (Some(k), None, _)    => Sampler::TopK { temperature, top_k: k, penalties },
            (None, Some(p), _)    => Sampler::TopP { temperature, top_p: p, penalties },
            (None, None, Some(m)) => Sampler::MinP { temperature, min_p: m, penalties },
            (None, None, None)    => Sampler::Multinomial { temperature, penalties },
        }
    }
}

pub trait Sample {
    /// Samples a token ID from a given sparse distribution of token IDs and their probabilities.
    ///
    /// # Arguments
    /// * `ids` - A slice of token IDs.
    /// * `probs` - A slice of corresponding probabilities for each token ID.
    fn sample(&self, ids: &[u32], probs: &[f32]) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_empty_is_neutral() {
        let o = SamplingOverrides::default();
        assert!(o.temperature.is_none());
        assert!(o.top_p.is_none());
    }

    #[test]
    fn defaults_empty_is_all_none() {
        let d = GenerationDefaults::default();
        assert!(d.repetition_penalty.is_none());
    }

    fn qwen25_defaults() -> GenerationDefaults {
        GenerationDefaults {
            temperature:        Some(0.7),
            top_p:              Some(0.8),
            top_k:              Some(20),
            min_p:              None,
            repetition_penalty: Some(1.05),
        }
    }

    #[test]
    fn client_overrides_win() {
        let o = SamplingOverrides { temperature: Some(0.5), ..Default::default() };
        let s = Sampler::merge(o, qwen25_defaults());
        // Expect TopKTopP { temperature=0.5, top_p=0.8, top_k=20 }
        assert!(matches!(s, Sampler::TopKTopP { temperature: t, top_p: 0.8, top_k: 20, .. } if (t - 0.5).abs() < 1e-6));
    }

    #[test]
    fn defaults_fill_missing() {
        let o = SamplingOverrides::default();
        let s = Sampler::merge(o, qwen25_defaults());
        assert!(matches!(s, Sampler::TopKTopP { top_p: 0.8, top_k: 20, .. }));
    }

    #[test]
    fn neutral_fallback_when_both_empty() {
        let o = SamplingOverrides::default();
        let s = Sampler::merge(o, GenerationDefaults::default());
        // No filters set → Multinomial at temperature 1.0
        assert!(matches!(s, Sampler::Multinomial { temperature: t, .. } if (t - 1.0).abs() < 1e-6));
    }

    #[test]
    fn empty_gen_config_client_only_params() {
        // Mistral/Gemma/Yi case: model ships empty gen_config.
        let o = SamplingOverrides { temperature: Some(0.9), top_p: Some(0.95), ..Default::default() };
        let s = Sampler::merge(o, GenerationDefaults::default());
        assert!(matches!(s, Sampler::TopP { top_p: p, .. } if (p - 0.95).abs() < 1e-6));
    }

    #[test]
    fn disabled_top_p_equals_one_falls_through() {
        let o = SamplingOverrides { temperature: Some(0.9), top_p: Some(1.0), ..Default::default() };
        let s = Sampler::merge(o, GenerationDefaults::default());
        assert!(matches!(s, Sampler::Multinomial { .. }));
    }

    #[test]
    fn defaults_only_top_k_yields_topk() {
        // Model gen_config sets only top_k; client sends nothing → Sampler::TopK.
        let d = GenerationDefaults { top_k: Some(50), temperature: Some(0.7), ..Default::default() };
        let s = Sampler::merge(SamplingOverrides::default(), d);
        assert!(matches!(s, Sampler::TopK { top_k: 50, temperature: t, .. } if (t - 0.7).abs() < 1e-6));
    }

    #[test]
    fn client_only_min_p_yields_minp() {
        // Empty gen_config; client sets min_p only → Sampler::MinP.
        let o = SamplingOverrides { min_p: Some(0.1), ..Default::default() };
        let s = Sampler::merge(o, GenerationDefaults::default());
        assert!(matches!(s, Sampler::MinP { min_p: m, .. } if (m - 0.1).abs() < 1e-6));
    }

    #[test]
    fn override_disables_default_top_p() {
        // Qwen2.5 defaults have top_p=0.8. Client explicitly sends top_p=1.0 ("disable").
        // Must fall through past TopP/TopKTopP — client wins even when client chooses "disabled".
        let d = GenerationDefaults {
            temperature: Some(0.7), top_p: Some(0.8), top_k: Some(20),
            ..Default::default()
        };
        let o = SamplingOverrides { top_p: Some(1.0), ..Default::default() };
        let s = Sampler::merge(o, d);
        // With top_p disabled by client but top_k=20 still active from defaults → TopK only
        assert!(matches!(s, Sampler::TopK { top_k: 20, .. }));
    }

    #[test]
    fn penalties_neutral_is_no_op() {
        let p = Penalties::default();
        assert_eq!(p.repetition, 1.0);
        assert_eq!(p.frequency, 0.0);
        assert_eq!(p.presence, 0.0);
    }

    #[test]
    fn sampler_carries_penalties_from_merge() {
        let o = SamplingOverrides {
            temperature: Some(0.5),
            repetition_penalty: Some(1.1),
            frequency_penalty: Some(0.2),
            ..Default::default()
        };
        let s = Sampler::merge(o, GenerationDefaults::default());
        // With no filter active and no defaults, result is Multinomial
        if let Sampler::Multinomial { penalties, .. } = s {
            assert_eq!(penalties.repetition, 1.1);
            assert_eq!(penalties.frequency, 0.2);
            assert_eq!(penalties.presence, 0.0);  // neutral fallback
        } else {
            panic!("expected Multinomial variant");
        }
    }
}
