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
}
