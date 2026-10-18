//! Which model a piece of state belongs to, and how it was computed.

use serde::{Deserialize, Serialize};

use crate::key::{Hash256, cache_domain};

/// The model, independent of how it is served: not its quantization, its
/// tensor parallelism, its page size or its engine. Two workers whose
/// identities are equal compute state with the same meaning.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// The package's model id, such as `qwen-3.5-0.8b`.
    pub model: String,
    /// Which weights. Until artifacts carry a digest of their source weights
    /// this is the package's model id: two workers of one model agree, and
    /// nothing yet tells a fine-tune apart from its base.
    pub weights: String,
    /// A digest of the tokenizer's objects: a different tokenizer gives the
    /// same text different tokens, and so different keys.
    pub tokenizer: String,
}

impl ModelIdentity {
    /// The domain every content key of this model is derived under.
    pub fn domain(&self) -> Hash256 {
        cache_domain(&self.seed())
    }

    fn seed(&self) -> Vec<u8> {
        let mut seed = Vec::new();
        for part in [&self.model, &self.weights, &self.tokenizer] {
            seed.extend_from_slice(&(part.len() as u64).to_le_bytes());
            seed.extend_from_slice(part.as_bytes());
        }
        seed
    }
}

/// How state was computed: by which engine, from which weight variant, into
/// which KV dtype. Same-class state is interchangeable bit for bit; other
/// classes are close, and using them is a policy's grant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NumericsClass {
    /// `cuda`, `metal`, `vulkan`, `wgpu`, `xla`, `cpu`.
    pub engine: String,
    /// The served variant of the weights, such as `bf16` or `q4`.
    pub weights: String,
    /// The KV cache's element type, such as `bf16` or `fp8-e4m3`.
    pub kv: String,
}

impl std::fmt::Display for NumericsClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/kv-{}", self.engine, self.weights, self.kv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(model: &str, weights: &str) -> ModelIdentity {
        ModelIdentity {
            model: model.into(),
            weights: weights.into(),
            tokenizer: "t".into(),
        }
    }

    #[test]
    fn every_part_of_the_identity_moves_the_domain() {
        let base = identity("m", "w").domain();
        assert_ne!(base, identity("m", "w2").domain());
        assert_ne!(base, identity("m2", "w").domain());
        // Length-prefixed: moving bytes between parts is a different seed.
        assert_ne!(identity("ab", "c").domain(), identity("a", "bc").domain());
    }
}
