//! Request and response types for inference.
//!
//! Defines the wire format for forward pass batching.

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::adapter::AdapterId;
use crate::context::ContextId;
use crate::brle::Brle;
use crate::device::DeviceId;

/// Sampler configuration for token generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Sampler {
    Multinomial { temperature: f32, seed: Option<u32> },
    TopK { temperature: f32, k: u32 },
    TopP { temperature: f32, p: f32 },
    MinP { temperature: f32, p: f32 },
    TopKTopP { temperature: f32, k: u32, p: f32 },
    Embedding,
    Dist { temperature: f32, num_tokens: u32 },
}

impl Sampler {
    /// Get the sampler type ID for serialization.
    pub fn type_id(&self) -> u32 {
        match self {
            Sampler::Multinomial { .. } => 1,
            Sampler::TopK { .. } => 2,
            Sampler::TopP { .. } => 3,
            Sampler::MinP { .. } => 4,
            Sampler::TopKTopP { .. } => 5,
            Sampler::Embedding => 6,
            Sampler::Dist { .. } => 0,
        }
    }

    /// Get the temperature value.
    pub fn temperature(&self) -> f32 {
        match self {
            Sampler::Multinomial { temperature, .. } => *temperature,
            Sampler::TopK { temperature, .. } => *temperature,
            Sampler::TopP { temperature, .. } => *temperature,
            Sampler::MinP { temperature, .. } => *temperature,
            Sampler::TopKTopP { temperature, .. } => *temperature,
            Sampler::Embedding => 0.0,
            Sampler::Dist { temperature, .. } => *temperature,
        }
    }

    /// Get top_k value (0 if not applicable).
    pub fn top_k(&self) -> u32 {
        match self {
            Sampler::TopK { k, .. } => *k,
            Sampler::TopKTopP { k, .. } => *k,
            _ => 0,
        }
    }

    /// Get top_p value (1.0 if not applicable).
    pub fn top_p(&self) -> f32 {
        match self {
            Sampler::TopP { p, .. } => *p,
            Sampler::TopKTopP { p, .. } => *p,
            _ => 1.0,
        }
    }

    /// Get min_p value (0.0 if not applicable).
    pub fn min_p(&self) -> f32 {
        match self {
            Sampler::MinP { p, .. } => *p,
            _ => 0.0,
        }
    }

    /// Get seed value (0 if not applicable or unset).
    pub fn seed(&self) -> u32 {
        match self {
            Sampler::Multinomial { seed, .. } => seed.unwrap_or(0),
            _ => 0,
        }
    }
}

/// Forward pass request for a single sequence.
#[derive(Debug, Clone)]
pub struct ForwardPassRequest {
    /// Context ID for KV cache page resolution (resolved by inference service).
    pub context_id: Option<ContextId>,
    /// Input token IDs.
    pub tokens: Vec<u32>,
    /// Token positions.
    pub positions: Vec<u32>,
    /// Speculative token IDs.
    pub speculative_tokens: Vec<u32>,
    /// Speculative token positions.
    pub speculative_positions: Vec<u32>,
    /// Whether to include speculative tokens in the output.
    pub output_speculative_tokens: bool,
    /// Attention masks (BRLE encoded, one per token).
    pub masks: Vec<Brle>,
    /// Logit mask (BRLE encoded, applied to vocabulary).
    pub logit_mask: Option<Brle>,
    /// Indices of tokens to sample from.
    pub sampling_indices: Vec<u32>,
    /// Sampler configurations for each sampling index.
    pub samplers: Vec<Sampler>,
    /// Optional adapter ID.
    pub adapter_id: Option<AdapterId>,
    /// Optional adapter seed (for ZO optimization).
    pub adapter_seed: Option<i64>,
    /// Arrival time for scheduler estimation.
    pub arrival_time: Option<Instant>,
}

/// Output from a forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForwardPassOutput {
    None,
    Tokens(Vec<u32>),
    /// (accepted tokens, next speculative tokens, next speculative positions)
    TokensWithSpeculation(Vec<u32>, Vec<u32>, Vec<u32>),
    Embeddings(Vec<Vec<u8>>),
    Distributions(Vec<(Vec<u32>, Vec<f32>)>),
}

/// Response for a single forward pass request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardPassResponse {
    pub tokens: Vec<u32>,
    pub dists: Vec<(Vec<u32>, Vec<f32>)>,
    /// Next speculative tokens (empty if non-speculative).
    pub spec_tokens: Vec<u32>,
    /// Next speculative positions (empty if non-speculative).
    pub spec_positions: Vec<u32>,
}

// =============================================================================
// Batched Request (for Python RPC)
// =============================================================================

/// Wrapper for Vec<u32> that serializes as raw bytes.
#[derive(Debug, Clone, Default)]
pub struct ByteVec(pub Vec<u32>);

impl Serialize for ByteVec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let bytes: &[u8] = bytemuck::cast_slice(&self.0);
        serializer.serialize_bytes(bytes)
    }
}

impl<'de> Deserialize<'de> for ByteVec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        let values: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(ByteVec(values))
    }
}

/// Wrapper for Vec<f32> that serializes as raw bytes.
#[derive(Debug, Clone, Default)]
pub struct ByteVecF32(pub Vec<f32>);

impl Serialize for ByteVecF32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let bytes: &[u8] = bytemuck::cast_slice(&self.0);
        serializer.serialize_bytes(bytes)
    }
}

impl<'de> Deserialize<'de> for ByteVecF32 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(ByteVecF32(values))
    }
}

/// Batched forward pass request sent to Python.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedForwardPassRequest {
    // === Core token data (concatenated, ByteVec for zero-copy) ===
    pub token_ids: ByteVec,
    pub position_ids: ByteVec,

    // === KV cache layout ===
    pub kv_page_indices: ByteVec,
    pub kv_page_indptr: ByteVec,
    pub kv_last_page_lens: ByteVec,

    // === Query/output indirection ===
    pub qo_indptr: ByteVec,

    // === Attention masks (BRLE, flattened) ===
    pub flattened_masks: ByteVec,
    pub mask_indptr: ByteVec,

    // === Logit masks (BRLE, per request) ===
    pub logit_masks: ByteVec,
    pub logit_mask_indptr: ByteVec,

    // === Sampling indices ===
    pub sampling_indices: ByteVec,
    pub sampling_indptr: ByteVec,

    // === Sampler parameters (SoA, one per sampler) ===
    pub sampler_temperatures: ByteVecF32,
    pub sampler_top_k: ByteVec,
    pub sampler_top_p: ByteVecF32,
    pub sampler_min_p: ByteVecF32,
    pub sampler_types: ByteVec,
    pub sampler_seeds: ByteVec,
    pub request_num_samplers: ByteVec,

    // === Adapter (per request) ===
    pub adapter_indices: Vec<Option<AdapterId>>,
    pub adapter_seeds: Vec<Option<i64>>,

    // === Speculative decoding (concatenated with indptr) ===
    pub spec_token_ids: ByteVec,
    pub spec_position_ids: ByteVec,
    pub spec_indptr: ByteVec,
    pub output_spec_flags: Vec<bool>,

    // === Inference hints ===
    pub single_token_mode: bool,

    // === Routing ===
    pub device_id: DeviceId,
}

impl BatchedForwardPassRequest {
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            token_ids: ByteVec(Vec::new()),
            position_ids: ByteVec(Vec::new()),
            kv_page_indices: ByteVec(Vec::new()),
            kv_page_indptr: ByteVec(vec![0]),
            kv_last_page_lens: ByteVec(Vec::new()),
            qo_indptr: ByteVec(vec![0]),
            flattened_masks: ByteVec(Vec::new()),
            mask_indptr: ByteVec(vec![0]),
            logit_masks: ByteVec(Vec::new()),
            logit_mask_indptr: ByteVec(vec![0]),
            sampling_indices: ByteVec(Vec::new()),
            sampling_indptr: ByteVec(vec![0]),
            sampler_temperatures: ByteVecF32(Vec::new()),
            sampler_top_k: ByteVec(Vec::new()),
            sampler_top_p: ByteVecF32(Vec::new()),
            sampler_min_p: ByteVecF32(Vec::new()),
            sampler_types: ByteVec(Vec::new()),
            sampler_seeds: ByteVec(Vec::new()),
            request_num_samplers: ByteVec(Vec::new()),
            adapter_indices: Vec::new(),
            adapter_seeds: Vec::new(),
            spec_token_ids: ByteVec(Vec::new()),
            spec_position_ids: ByteVec(Vec::new()),
            spec_indptr: ByteVec(vec![0]),
            output_spec_flags: Vec::new(),
            single_token_mode: true,
            device_id,
        }
    }

    /// Add a request to the batch.
    pub fn add_request(&mut self, req: &ForwardPassRequest, physical_page_ids: &[u32], last_page_len: u32) {
        // Tokens and positions
        self.token_ids.0.extend(&req.tokens);
        self.position_ids.0.extend(&req.positions);

        // KV cache layout
        self.kv_page_indices.0.extend(physical_page_ids);
        self.kv_page_indptr.0.push(self.kv_page_indices.0.len() as u32);
        self.kv_last_page_lens.0.push(last_page_len);

        // Query/output indirection
        self.qo_indptr.0.push(self.token_ids.0.len() as u32);

        // Attention masks (flatten BRLE)
        for mask in &req.masks {
            self.flattened_masks.0.extend(&mask.buffer);
            self.mask_indptr.0.push(self.flattened_masks.0.len() as u32);
        }

        // Logit mask (flatten BRLE, per request)
        if let Some(ref mask) = req.logit_mask {
            self.logit_masks.0.extend(&mask.buffer);
        }
        self.logit_mask_indptr.0.push(self.logit_masks.0.len() as u32);

        // Sampling indices
        self.sampling_indices.0.extend(&req.sampling_indices);
        self.sampling_indptr.0.push(self.sampling_indices.0.len() as u32);

        // Samplers (SoA)
        self.request_num_samplers.0.push(req.samplers.len() as u32);
        for sampler in &req.samplers {
            self.sampler_types.0.push(sampler.type_id());
            self.sampler_temperatures.0.push(sampler.temperature());
            self.sampler_top_k.0.push(sampler.top_k());
            self.sampler_top_p.0.push(sampler.top_p());
            self.sampler_min_p.0.push(sampler.min_p());
            self.sampler_seeds.0.push(sampler.seed());
        }

        // Adapter
        self.adapter_indices.push(req.adapter_id);
        self.adapter_seeds.push(req.adapter_seed);

        // Speculative decoding
        self.spec_token_ids.0.extend(&req.speculative_tokens);
        self.spec_position_ids.0.extend(&req.speculative_positions);
        self.spec_indptr.0.push(self.spec_token_ids.0.len() as u32);
        self.output_spec_flags.push(req.output_speculative_tokens);

        // Inference hint
        if req.tokens.len() > 1 {
            self.single_token_mode = false;
        }
    }

    pub fn num_requests(&self) -> usize {
        self.adapter_indices.len()
    }

    pub fn total_tokens(&self) -> usize {
        self.token_ids.0.len()
    }
}

/// Batched forward pass response from Python.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedForwardPassResponse {
    pub results: Vec<ForwardPassResponse>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytevec_u32_serde_roundtrip() {
        let original = ByteVec(vec![0, 1, u32::MAX, 42, 0xDEAD_BEEF]);
        let packed = rmp_serde::to_vec(&original).expect("serialize");
        let decoded: ByteVec = rmp_serde::from_slice(&packed).expect("deserialize");
        assert_eq!(original.0, decoded.0);
    }

    #[test]
    fn bytevec_f32_serde_roundtrip() {
        let original = ByteVecF32(vec![0.0, 1.0, -1.5, f32::INFINITY, f32::NAN]);
        let packed = rmp_serde::to_vec(&original).expect("serialize");
        let decoded: ByteVecF32 = rmp_serde::from_slice(&packed).expect("deserialize");
        // NaN != NaN, so compare element-by-element
        for (a, b) in original.0.iter().zip(decoded.0.iter()) {
            if a.is_nan() {
                assert!(b.is_nan(), "expected NaN, got {b}");
            } else {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn bytevec_empty_roundtrip() {
        let original = ByteVec(vec![]);
        let packed = rmp_serde::to_vec(&original).expect("serialize");
        let decoded: ByteVec = rmp_serde::from_slice(&packed).expect("deserialize");
        assert!(decoded.0.is_empty());
    }
}
