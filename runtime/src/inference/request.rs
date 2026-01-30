//! Request and response types for inference.
//!
//! Defines the wire format for forward pass batching.

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::adapter::AdapterId;
use crate::brle::Brle;
use crate::kvcache::{NodeId, PageId};

/// Sampler configuration for token generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Sampler {
    Multinomial { temperature: f32 },
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
            Sampler::Multinomial { temperature } => *temperature,
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
}

/// Forward pass request for a single sequence.
#[derive(Debug, Clone)]
pub struct ForwardPassRequest {
    /// Logical page IDs for KV cache.
    pub page_ids: Vec<PageId>,
    /// Length of last page (for partial pages).
    pub last_page_len: u32,
    /// Input token IDs.
    pub tokens: Vec<u32>,
    /// Token positions.
    pub positions: Vec<u32>,
    /// Attention masks (BRLE encoded, one per token).
    pub masks: Vec<Brle>,
    /// Indices of tokens to sample from.
    pub sampling_indices: Vec<u32>,
    /// Sampler configurations for each sampling index.
    pub samplers: Vec<Sampler>,
    /// Optional adapter ID.
    pub adapter_id: Option<AdapterId>,
    /// Arrival time for scheduler estimation.
    pub arrival_time: Option<Instant>,
}

/// Output from a forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForwardPassOutput {
    None,
    Tokens(Vec<u32>),
    Embeddings(Vec<Vec<u8>>),
    Distributions(Vec<(Vec<u32>, Vec<f32>)>),
}

/// Response for a single forward pass request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardPassResponse {
    pub tokens: Vec<u32>,
    pub dists: Vec<(Vec<u32>, Vec<f32>)>,
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

/// Batched forward pass request sent to Python.
#[derive(Debug, Clone, Serialize)]
pub struct BatchedForwardPassRequest {
    /// Token IDs (concatenated).
    pub token_ids: ByteVec,
    /// Position IDs (concatenated).
    pub position_ids: ByteVec,
    /// Physical page indices for KV cache (concatenated).
    pub kv_page_indices: ByteVec,
    /// Pointer into kv_page_indices per request.
    pub kv_page_indptr: ByteVec,
    /// Last page length per request.
    pub kv_last_page_lens: ByteVec,
    /// Pointer into tokens per request.
    pub qo_indptr: ByteVec,
    /// Flattened attention masks (BRLE data).
    pub flattened_masks: ByteVec,
    /// Pointer into flattened_masks per token.
    pub mask_indptr: ByteVec,
    /// Sampling indices (flattened).
    pub sampling_indices: ByteVec,
    /// Pointer into sampling_indices per request.
    pub sampling_indptr: ByteVec,
    /// Sampler temperatures.
    pub sampler_temperatures: ByteVecF32,
    /// Sampler top_k values.
    pub sampler_top_k: ByteVec,
    /// Sampler top_p values.
    pub sampler_top_p: ByteVecF32,
    /// Sampler min_p values.
    pub sampler_min_p: ByteVecF32,
    /// Sampler type IDs.
    pub sampler_types: ByteVec,
    /// Number of samplers per request.
    pub request_num_samplers: ByteVec,
    /// Adapter indices (optional per request).
    pub adapter_indices: Vec<Option<AdapterId>>,
    /// Node ID for routing.
    pub node_id: NodeId,
}

impl BatchedForwardPassRequest {
    pub fn new(node_id: NodeId) -> Self {
        Self {
            token_ids: ByteVec(Vec::new()),
            position_ids: ByteVec(Vec::new()),
            kv_page_indices: ByteVec(Vec::new()),
            kv_page_indptr: ByteVec(vec![0]),
            kv_last_page_lens: ByteVec(Vec::new()),
            qo_indptr: ByteVec(vec![0]),
            flattened_masks: ByteVec(Vec::new()),
            mask_indptr: ByteVec(vec![0]),
            sampling_indices: ByteVec(Vec::new()),
            sampling_indptr: ByteVec(vec![0]),
            sampler_temperatures: ByteVecF32(Vec::new()),
            sampler_top_k: ByteVec(Vec::new()),
            sampler_top_p: ByteVecF32(Vec::new()),
            sampler_min_p: ByteVecF32(Vec::new()),
            sampler_types: ByteVec(Vec::new()),
            request_num_samplers: ByteVec(Vec::new()),
            adapter_indices: Vec::new(),
            node_id,
        }
    }

    /// Add a request to the batch.
    pub fn add_request(&mut self, req: &ForwardPassRequest, physical_page_ids: &[u32]) {
        // Tokens and positions
        self.token_ids.0.extend(&req.tokens);
        self.position_ids.0.extend(&req.positions);

        // KV cache layout
        self.kv_page_indices.0.extend(physical_page_ids);
        self.kv_page_indptr.0.push(self.kv_page_indices.0.len() as u32);
        self.kv_last_page_lens.0.push(req.last_page_len);

        // Query/output indirection
        self.qo_indptr.0.push(self.token_ids.0.len() as u32);

        // Attention masks (flatten BRLE)
        for mask in &req.masks {
            self.flattened_masks.0.extend(&mask.buffer);
            self.mask_indptr.0.push(self.flattened_masks.0.len() as u32);
        }

        // Sampling indices
        self.sampling_indices.0.extend(&req.sampling_indices);
        self.sampling_indptr.0.push(self.sampling_indices.0.len() as u32);

        // Samplers
        self.request_num_samplers.0.push(req.samplers.len() as u32);
        for sampler in &req.samplers {
            self.sampler_types.0.push(sampler.type_id());
            self.sampler_temperatures.0.push(sampler.temperature());
            self.sampler_top_k.0.push(sampler.top_k());
            self.sampler_top_p.0.push(sampler.top_p());
            self.sampler_min_p.0.push(sampler.min_p());
        }

        // Adapter
        self.adapter_indices.push(req.adapter_id);
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
