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

pub enum Sampler {
    Custom {
        temperature: f32,
        sampler: Box<dyn Sample>,
    },
    Multinomial {
        temperature: f32,
    },
    TopP {
        temperature: f32,
        top_p: f32,
    },
    TopK {
        temperature: f32,
        top_k: u32,
    },
    MinP {
        temperature: f32,
        min_p: f32,
    },
    TopKTopP {
        temperature: f32,
        top_k: u32,
        top_p: f32,
    },
}

impl Sampler {
    pub fn greedy() -> Self {
        Sampler::Multinomial { temperature: 0.0 }
    }

    pub fn top_p(temperature: f32, top_p: f32) -> Self {
        Sampler::TopP { temperature, top_p }
    }

    pub fn top_k(temperature: f32, top_k: u32) -> Self {
        Sampler::TopK { temperature, top_k }
    }

    pub fn min_p(temperature: f32, min_p: f32) -> Self {
        Sampler::MinP { temperature, min_p }
    }

    pub fn top_k_top_p(temperature: f32, top_k: u32, top_p: f32) -> Self {
        Sampler::TopKTopP {
            temperature,
            top_k,
            top_p,
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
    /// Only the four filter-style fields (temperature, top_p, top_k, min_p) are
    /// considered here. `repetition_penalty` / `frequency_penalty` /
    /// `presence_penalty` will be folded in by Task B1 when `Sampler` variants
    /// grow a `Penalties` field.
    pub fn merge(overrides: SamplingOverrides, defaults: GenerationDefaults) -> Self {
        let temperature = overrides.temperature.or(defaults.temperature).unwrap_or(1.0);
        let top_p = overrides.top_p.or(defaults.top_p);
        let top_k = overrides.top_k.or(defaults.top_k);
        let min_p = overrides.min_p.or(defaults.min_p);
        // top_p == 1.0 or top_k == 0 means "disabled"; treat as None.
        let top_p_active = top_p.filter(|v| *v < 1.0 && *v > 0.0);
        let top_k_active = top_k.filter(|v| *v > 0);
        let min_p_active = min_p.filter(|v| *v > 0.0);
        match (top_k_active, top_p_active, min_p_active) {
            (Some(k), Some(p), _) => Sampler::TopKTopP { temperature, top_k: k, top_p: p },
            (Some(k), None, _)    => Sampler::TopK { temperature, top_k: k },
            (None, Some(p), _)    => Sampler::TopP { temperature, top_p: p },
            (None, None, Some(m)) => Sampler::MinP { temperature, min_p: m },
            (None, None, None)    => Sampler::Multinomial { temperature },
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
        assert!(matches!(s, Sampler::TopKTopP { temperature: t, top_p: 0.8, top_k: 20 } if (t - 0.5).abs() < 1e-6));
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
        assert!(matches!(s, Sampler::Multinomial { temperature: t } if (t - 1.0).abs() < 1e-6));
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
        assert!(matches!(s, Sampler::TopK { top_k: 50, temperature: t } if (t - 0.7).abs() < 1e-6));
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
}
