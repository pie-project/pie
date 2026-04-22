use anyhow::{Result, bail};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

pub static HANDSHAKE_ID: u32 = 0;

pub static QUERY_ID: u32 = 2;
pub static FORWARD_PASS_ID: u32 = 3;
pub static EMBED_IMAGE_ID: u32 = 4;
pub static INITIALIZE_ADAPTER_ID: u32 = 5;
pub static UPDATE_ADAPTER_ID: u32 = 6;
pub static UPLOAD_ADAPTER_ID: u32 = 7;
pub static DOWNLOAD_ADAPTER_ID: u32 = 8;
pub static FORMAT_CHAT_ID: u32 = 9;

#[derive(Debug)]
pub enum Request {
    Handshake(HandshakeRequest, oneshot::Sender<HandshakeResponse>),
    Query(QueryRequest, oneshot::Sender<QueryResponse>),
    Synchronize(oneshot::Sender<()>),

    ForwardPass(
        ForwardPassRequest,
        Option<oneshot::Sender<ForwardPassResponse>>,
    ),
    EmbedImage(EmbedImageRequest),
    InitializeAdapter(InitializeAdapterRequest),
    UpdateAdapter(UpdateAdapterRequest),
    UploadAdapter(UploadAdapterRequest),
    DownloadAdapter(DownloadAdapterRequest),
    FormatChat(FormatChatRequest, oneshot::Sender<FormatChatResponse>),
}

impl Request {
    pub fn is_eager(&self) -> bool {
        match self {
            Request::ForwardPass(_, _) => false,
            _ => true,
        }
    }

    pub fn is_sync_req(&self) -> bool {
        match self {
            Request::Synchronize(_) => true,
            _ => false,
        }
    }

    pub fn has_response(&self) -> bool {
        match self {
            Request::Handshake(_, _) => true,
            Request::Query(_, _) => true,
            Request::ForwardPass(_, r) => r.is_some(),
            Request::FormatChat(_, _) => true,
            _ => false,
        }
    }

    pub fn handler_id(&self) -> u32 {
        match self {
            Request::Handshake(_, _) => HANDSHAKE_ID,
            Request::Query(_, _) => QUERY_ID,
            Request::Synchronize(_) => unreachable!("Synchronize request has no handler ID"),

            Request::ForwardPass(_, _) => FORWARD_PASS_ID,
            Request::EmbedImage(_) => EMBED_IMAGE_ID,
            Request::InitializeAdapter(_) => INITIALIZE_ADAPTER_ID,
            Request::UpdateAdapter(_) => UPDATE_ADAPTER_ID,
            Request::UploadAdapter(_) => UPLOAD_ADAPTER_ID,
            Request::DownloadAdapter(_) => DOWNLOAD_ADAPTER_ID,
            Request::FormatChat(_, _) => FORMAT_CHAT_ID,
        }
    }

    pub fn serialize_req(&self) -> Result<Bytes> {
        let b = match self {
            Request::Handshake(req, _) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::Query(req, _) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::Synchronize(_) => bail!("cannot serialize synchronize request"),
            Request::ForwardPass(req, _) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::EmbedImage(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::InitializeAdapter(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::UpdateAdapter(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::UploadAdapter(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::DownloadAdapter(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::FormatChat(req, _) => Bytes::from(rmp_serde::to_vec_named(&req)?),
        };
        Ok(b)
    }

    pub fn deserialize_resp(self, b: Bytes) -> Result<()> {
        match self {
            Request::Handshake(_, resp) => {
                let r: HandshakeResponse = rmp_serde::from_slice(&b)?;
                resp.send(r).ok();
            }
            Request::Query(_, resp) => {
                let r: QueryResponse = rmp_serde::from_slice(&b)?;
                resp.send(r).ok();
            }
            Request::ForwardPass(_, resp) => {
                let r: ForwardPassResponse = rmp_serde::from_slice(&b)?;
                if let Some(tx) = resp {
                    tx.send(r).ok();
                }
            }
            Request::FormatChat(_, resp) => {
                let r: FormatChatResponse = rmp_serde::from_slice(&b)?;
                resp.send(r).ok();
            }
            _ => {
                bail!("cannot deserialize response for request {:?}", self);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeRequest {
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub version: String,
    pub model_name: String,
    pub model_traits: Vec<String>,
    pub model_description: String,
    pub prompt_template: String,
    pub prompt_template_type: String,
    pub prompt_stop_tokens: Vec<String>,
    pub kv_page_size: u32,
    pub max_batch_tokens: usize,
    pub max_batch_size: usize,
    pub resources: HashMap<u32, u32>,
    pub tokenizer_num_vocab: usize,
    pub tokenizer_merge_table: HashMap<u32, Vec<u8>>,
    pub tokenizer_special_tokens: HashMap<String, u32>,
    pub tokenizer_split_regex: String,
    pub tokenizer_escape_non_printable: bool,
    pub tokenizer_sentencepiece_space: bool,
    /// Raw HF chat template string for in-process minijinja rendering.
    #[serde(default)]
    pub chat_template: String,
    /// Whether the chat template is minijinja-compatible (no Python-only Jinja2 features).
    #[serde(default)]
    pub template_minijinja_compatible: bool,
    /// Sampling defaults sourced from HF `generation_config.json` on the
    /// Python side. Optional — absent or empty means the model ships no
    /// sampling defaults and inferlets should fall back to their own.
    /// Keys are a subset of `{temperature, top_p, top_k, min_p,
    /// repetition_penalty}`; values are JSON scalars (int/float).
    /// Unknown keys are ignored by the runtime.
    #[serde(default)]
    pub generation_defaults: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query: String,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResponse {
    pub value: String,
}



#[derive(Debug, Serialize, Deserialize)]
pub struct ForwardPassRequest {
    pub input_tokens: Vec<u32>,
    pub input_token_positions: Vec<u32>,
    pub input_embed_ptrs: Vec<u32>,
    pub input_embed_positions: Vec<u32>,
    pub inst_id: Option<Uuid>,
    pub adapter: Option<u32>,
    pub adapter_seed: Option<i64>,
    pub mask: Vec<Vec<u32>>,
    pub kv_page_ptrs: Vec<u32>,
    pub kv_page_last_len: u32,
    pub output_token_indices: Vec<u32>,
    pub output_token_samplers: Vec<HashMap<String, rmpv::Value>>,
    pub output_embed_ptrs: Vec<u32>,
    pub output_embed_indices: Vec<u32>,
    /// Maximum number of sequential decode steps to run before returning.
    /// Default 1 (current behavior). When >1, the host runs N autoregressive
    /// steps internally, feeding each sampled token back as input for the next.
    /// Requires engine-side sampling (output_tokens variants). Incompatible
    /// with output_distributions (Custom sampler).
    #[serde(default = "default_one")]
    pub max_decode_steps: u32,
    /// When true, capture and return probability distributions alongside
    /// sampled tokens for each step. Only valid with output_tokens variants.
    #[serde(default)]
    pub return_distributions: bool,
    /// Accumulated tokens across multi-step re-enqueue cycles (scheduler-only).
    #[serde(skip)]
    pub multi_step_tokens: Vec<u32>,
    /// KV page size for multi-step KV state tracking (scheduler-only).
    #[serde(skip)]
    pub kv_page_size: u32,
    /// Number of active KV pages (excluding pre-allocated extras).
    /// Only the first `actual_kv_pages` are serialized into kv_page_indptr.
    /// The scheduler extends this on page-boundary crossings.
    #[serde(skip)]
    pub actual_kv_pages: u32,
    /// Arrival time for scheduler estimation (not serialized).
    #[serde(skip)]
    pub arrival_time: Option<Instant>,
    /// Per-token streaming channel (scheduler-only, not serialized).
    /// When set, each multi-step continuation sends its token immediately
    /// instead of only accumulating in multi_step_tokens. Enables the
    /// WASM inferlet to process tokens while the next GPU step runs.
    #[serde(skip)]
    pub token_stream_tx: Option<mpsc::UnboundedSender<u32>>,
    /// Whether this request has been sent to Python before.
    /// Set to true after first fire_batch; used to compute is_new in batched request.
    #[serde(skip)]
    pub has_been_fired: bool,
}

fn default_one() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardPassResponse {
    pub tokens: Vec<u32>,
    pub dists: Vec<(Vec<u32>, Vec<f32>)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedImageRequest {
    pub embed_ptrs: Vec<u32>,
    pub image_blob: Vec<u8>,
    pub position_offset: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeAdapterRequest {
    pub adapter_ptr: u32,
    pub rank: u32,
    pub alpha: f32,
    pub population_size: u32,
    pub mu_fraction: f32,
    pub initial_sigma: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateAdapterRequest {
    pub adapter_ptr: u32,
    pub scores: Vec<f32>,
    pub seeds: Vec<i64>,
    pub max_sigma: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadAdapterRequest {
    pub adapter_ptr: u32,
    pub name: String,
    pub adapter_data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadAdapterRequest {
    pub adapter_ptr: u32,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatChatRequest {
    pub messages_json: String,
    pub tools_json: Option<String>,
    pub add_generation_prompt: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FormatChatResponse {
    pub token_ids: Vec<u32>,
}

/// Wrapper for Vec<u32> that serializes as raw bytes for zero-copy Python deserialization.
/// Python can then use `np.frombuffer(data, dtype=np.uint32)` for O(1) deserialization.
#[derive(Debug, Clone, Default)]
pub struct ByteVec(pub Vec<u32>);

impl ByteVec {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl serde::Serialize for ByteVec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let bytes: &[u8] = bytemuck::cast_slice(&self.0);
        serializer.serialize_bytes(bytes)
    }
}

/// Wrapper for Vec<f32> that serializes as raw bytes for zero-copy Python deserialization.
/// Python can then use `np.frombuffer(data, dtype=np.float32)` for O(1) deserialization.
#[derive(Debug, Clone, Default)]
pub struct ByteVecF32(pub Vec<f32>);

impl serde::Serialize for ByteVecF32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let bytes: &[u8] = bytemuck::cast_slice(&self.0);
        serializer.serialize_bytes(bytes)
    }
}

/// Batched forward pass request sent to Python via pycrust.
/// Rust performs partial batch formation (concatenating arrays),
/// while Python handles attention mask decoding and tensor creation.
///
/// Numerical arrays use ByteVec/ByteVecF32 for zero-copy deserialization in Python.
#[derive(Debug, Clone, Serialize)]
pub struct BatchedForwardPassRequest {
    // Concatenated arrays from all requests in the batch (as raw bytes)
    pub token_ids: ByteVec,
    pub position_ids: ByteVec,

    // KV cache layout (concatenated, as raw bytes)
    pub kv_page_indices: ByteVec,
    pub kv_page_indptr: ByteVec, // [0, n1, n1+n2, ...] indices into kv_page_indices
    pub kv_last_page_lens: ByteVec, // One per request

    // Query/Output indirection (as raw bytes)
    pub qo_indptr: ByteVec, // [0, tokens1, tokens1+tokens2, ...]

    // Attention masks (BRLE encoded, flattened, as raw bytes)
    pub flattened_masks: ByteVec, // Concatenation of all BRLE buffers
    pub mask_indptr: ByteVec,     // Pointers into flattened_masks for each token

    // Adapter info (one per request) - keep as Vec since these are small
    pub adapter_indices: Vec<Option<u32>>,
    pub adapter_seeds: Vec<Option<i64>>,

    // === SoA Sampler Parameters (flattened) ===
    // Each sampler across all requests is flattened into these arrays
    pub sampler_temperatures: ByteVecF32, // f32 array, one per sampler
    pub sampler_top_k: ByteVec,           // u32 array (will cast to i32 in Python)
    pub sampler_top_p: ByteVecF32,        // f32 array
    pub sampler_min_p: ByteVecF32,        // f32 array
    pub sampler_types: ByteVec,           // u32 array (0=dist, 1/2/3=sampler types)
    pub request_num_samplers: ByteVec,    // u32 array, num samplers per request

    // Output token indices (flattened with indptr)
    pub flat_output_token_indices: ByteVec, // Concatenated indices
    pub output_token_indptr: ByteVec,       // [0, n1, n1+n2, ...] per request

    // Embed outputs (keep nested since usually empty/small)
    pub output_embed_ptrs: Vec<Vec<u32>>,
    pub output_embed_indices: Vec<Vec<u32>>,

    // Inference mode hint
    pub single_token_mode: bool,

    // Trace context for cross-language propagation (W3C traceparent)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_context: Option<String>,

    // Target group ID for Data Parallelism routing
    pub group_id: Option<usize>,

    // Maximum sequential decode steps for multi-step decode.
    // When >1, the Python backend loops internally.
    #[serde(default = "default_one")]
    pub max_decode_steps: u32,

    // KV page IDs freed since last batch (explicit finish signal for SequenceTracker).
    // Sent from Rust ResourceManager on deallocate/cleanup.
    #[serde(default, skip_serializing_if = "ByteVec::is_empty")]
    pub freed_block_ids: ByteVec,

    // Per-request identity from Rust (inst_id UUID strings).
    // Python uses these as authoritative sequence keys instead of block IDs.
    #[serde(default)]
    pub request_ids: Vec<String>,

    // Per-request first-fire flag. true = first time this request is sent to Python.
    #[serde(default)]
    pub is_new: Vec<bool>,

    // Per-request active KV page count. kv_page_indices contains ALL pages
    // (active + pre-allocated extras for decode_n headroom). This array tells
    // Python how many of each request's pages have actual KV data. Used by:
    //   - compute_seq_lens (derive sequence length from active pages only)
    //   - multi-step page crossing (extra pages available = can cross without Rust round-trip)
    #[serde(default, skip_serializing_if = "ByteVec::is_empty")]
    pub kv_actual_page_counts: ByteVec,

    // Instance UUIDs that have finished (from cleanup_instance).
    // Sent alongside freed_block_ids but carries the authoritative identity
    // so Python can finish requests directly by key without block-ID reverse map.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub finished_request_ids: Vec<String>,
}

impl BatchedForwardPassRequest {
    /// Create a new empty batched request.
    pub fn new() -> Self {
        Self {
            token_ids: ByteVec(Vec::new()),
            position_ids: ByteVec(Vec::new()),
            kv_page_indices: ByteVec(Vec::new()),
            kv_page_indptr: ByteVec(vec![0]),
            kv_last_page_lens: ByteVec(Vec::new()),
            qo_indptr: ByteVec(vec![0]),
            flattened_masks: ByteVec(Vec::new()),
            mask_indptr: ByteVec(vec![0]),
            adapter_indices: Vec::new(),
            adapter_seeds: Vec::new(),
            // SoA sampler fields
            sampler_temperatures: ByteVecF32(Vec::new()),
            sampler_top_k: ByteVec(Vec::new()),
            sampler_top_p: ByteVecF32(Vec::new()),
            sampler_min_p: ByteVecF32(Vec::new()),
            sampler_types: ByteVec(Vec::new()),
            request_num_samplers: ByteVec(Vec::new()),
            flat_output_token_indices: ByteVec(Vec::new()),
            output_token_indptr: ByteVec(vec![0]),
            output_embed_ptrs: Vec::new(),
            output_embed_indices: Vec::new(),
            single_token_mode: true,
            trace_context: None,
            group_id: None,
            max_decode_steps: 1,
            freed_block_ids: ByteVec(Vec::new()),
            request_ids: Vec::new(),
            is_new: Vec::new(),
            kv_actual_page_counts: ByteVec(Vec::new()),
            finished_request_ids: Vec::new(),
        }
    }

    /// Add a single ForwardPassRequest to the batch.
    pub fn add_request(&mut self, req: &ForwardPassRequest) {
        // Concatenate tokens and positions
        self.token_ids.0.extend(&req.input_tokens);
        self.position_ids.0.extend(&req.input_token_positions);

        // KV cache layout — send ALL pages (active + pre-allocated extras).
        // Python's multi-step loop needs the extras to cross page boundaries
        // without a Rust round-trip. The active count is sent separately in
        // kv_actual_page_counts so Python can compute seq_lens correctly.
        let total_pages = req.kv_page_ptrs.len();
        let active_pages = if req.actual_kv_pages > 0 {
            (req.actual_kv_pages as usize).min(total_pages)
        } else {
            total_pages
        };
        self.kv_page_indices.0.extend(&req.kv_page_ptrs);
        self.kv_page_indptr.0.push(self.kv_page_indices.0.len() as u32);
        self.kv_last_page_lens.0.push(req.kv_page_last_len);
        self.kv_actual_page_counts.0.push(active_pages as u32);

        // Query/output indirection
        let total_tokens = self.token_ids.0.len() as u32;
        self.qo_indptr.0.push(total_tokens);

        // Masks (flatten nested structure)
        for token_mask in &req.mask {
            self.flattened_masks.0.extend(token_mask);
            self.mask_indptr.0.push(self.flattened_masks.0.len() as u32);
        }

        // Adapter info
        self.adapter_indices.push(req.adapter);
        self.adapter_seeds.push(req.adapter_seed);

        // Output token indices (flatten with indptr)
        self.flat_output_token_indices.0.extend(&req.output_token_indices);
        self.output_token_indptr.0.push(self.flat_output_token_indices.0.len() as u32);

        // Extract sampler parameters (SoA flattening)
        let num_samplers = req.output_token_samplers.len() as u32;
        self.request_num_samplers.0.push(num_samplers);

        for sampler_cfg in &req.output_token_samplers {
            // Extract sampler type (default: 1 = standard sampling)
            let sampler_type = sampler_cfg
                .get("sampler")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as u32;
            self.sampler_types.0.push(sampler_type);

            // Extract temperature (default: 1.0)
            let temperature = sampler_cfg
                .get("temperature")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            self.sampler_temperatures.0.push(temperature);

            // Extract top_k (default: 0 = disabled)
            let top_k = sampler_cfg
                .get("top_k")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            self.sampler_top_k.0.push(top_k);

            // Extract top_p (default: 1.0 = disabled)
            let top_p = sampler_cfg
                .get("top_p")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            self.sampler_top_p.0.push(top_p);

            // Extract min_p (default: 0.0 = disabled)
            let min_p = sampler_cfg
                .get("min_p")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            self.sampler_min_p.0.push(min_p);
        }

        // Embed outputs
        self.output_embed_ptrs.push(req.output_embed_ptrs.clone());
        self.output_embed_indices.push(req.output_embed_indices.clone());

        // Update inference mode hint
        if req.input_tokens.len() > 1 {
            self.single_token_mode = false;
        }

        // Propagate max_decode_steps (use max across batch)
        if req.max_decode_steps > self.max_decode_steps {
            self.max_decode_steps = req.max_decode_steps;
        }

        // Request identity for Python SequenceTracker
        self.request_ids.push(
            req.inst_id.map(|id| id.to_string()).unwrap_or_default()
        );
        self.is_new.push(!req.has_been_fired);
    }

    /// Get the number of requests in this batch.
    pub fn num_requests(&self) -> usize {
        self.adapter_indices.len()
    }

    /// Get the total number of tokens in this batch.
    pub fn total_tokens(&self) -> usize {
        self.token_ids.0.len()
    }

    /// Set trace context for cross-language propagation.
    /// The trace_context should be a W3C traceparent string.
    pub fn set_trace_context(&mut self, trace_context: String) {
        self.trace_context = Some(trace_context);
    }
}

impl Default for BatchedForwardPassRequest {
    fn default() -> Self {
        Self::new()
    }
}

/// Batched forward pass response from Python via pycrust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedForwardPassResponse {
    /// Results indexed by request order in the batch.
    pub results: Vec<ForwardPassResponse>,
}
