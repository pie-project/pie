//! Model Service - Model registration and tokenizer management
//!
//! This module provides model metadata and tokenizer management via a global cache.
//! All model and tokenizer operations access the cache directly without message passing.

pub mod tokenizer;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, RwLock};
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::context;
use crate::ffi::RpcBackend;
use crate::inference;
use tokenizer::BytePairEncoder;

/// Counter for generating unique model IDs.
static MODEL_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Global registry mapping model names to model IDs.
static NAME_REGISTRY: LazyLock<RwLock<HashMap<String, ModelId>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Global cache for models (keyed by ModelId).
static MODELS: LazyLock<RwLock<HashMap<ModelId, Model>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Type alias for model identifiers.
pub type ModelId = usize;

/// Looks up a model by name and returns its model ID.
pub fn model_id(model_name: &str) -> Option<ModelId> {
    NAME_REGISTRY.read().ok()?.get(model_name).copied()
}

/// Returns a list of all registered model names.
pub fn registered_models() -> Vec<String> {
    NAME_REGISTRY
        .read()
        .map(|r| r.keys().cloned().collect())
        .unwrap_or_default()
}

/// Gets cached model by model ID.
pub fn get_model(model_id: ModelId) -> Option<Model> {
    MODELS.read().ok()?.get(&model_id).cloned()
}

// =============================================================================
// Handshake Types
// =============================================================================

/// Handshake request sent to Python backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeRequest {
    pub version: String,
}

/// Handshake response from Python backend.
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
}

// =============================================================================
// Public API
// =============================================================================

/// Installs a new model by performing handshake with the backend.
/// Returns the global model ID that can be used to access the model.
pub async fn install_model_with_backend(backend: RpcBackend) -> Result<ModelId> {
    let resp = Model::handshake(&backend).await?;
    let model_name = resp.model_name.clone();
    let model = Model::from_handshake(resp);

    // Generate a unique model ID
    let model_id = MODEL_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

    // Register the model name
    if let Ok(mut registry) = NAME_REGISTRY.write() {
        registry.insert(model_name, model_id);
    }

    // Store in the global cache
    if let Ok(mut cache) = MODELS.write() {
        cache.insert(model_id, model);
    }

    // Store the backend for RPC calls
    if let Ok(mut backends) = BACKENDS.write() {
        backends.insert(model_id, backend);
    }

    // Spawn the associated actors with the same model ID
    // Note: PageStore is now owned by ContextManagerActor, so no separate kvcache actor
    let _ = context::spawn();
    let _ = inference::spawn();

    Ok(model_id)
}

/// Global storage for RPC backends (keyed by ModelId).
static BACKENDS: LazyLock<RwLock<HashMap<ModelId, RpcBackend>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));


// =============================================================================
// Model Info
// =============================================================================

/// Model information.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub traits: Vec<String>,
    pub description: String,
    pub prompt_template: String,
    pub prompt_template_type: String,
    pub stop_tokens: Vec<String>,
    pub stop_token_ids: Vec<u32>,
    pub kv_page_size: u32,
    pub max_batch_tokens: usize,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            name: String::new(),
            traits: Vec::new(),
            description: String::new(),
            prompt_template: String::new(),
            prompt_template_type: String::new(),
            stop_tokens: Vec::new(),
            stop_token_ids: Vec::new(),
            kv_page_size: 0,
            max_batch_tokens: 0,
        }
    }
}

/// Tokenizer information (for external queries).
#[derive(Debug, Clone)]
pub struct TokenizerInfo {
    pub vocabs: (Vec<u32>, Vec<Vec<u8>>),
    pub split_regex: String,
    pub special_tokens: (Vec<u32>, Vec<Vec<u8>>),
}

// =============================================================================
// Model - Business Logic
// =============================================================================

/// The model service handles model metadata and tokenization.
///
/// This is the core business logic, separate from the actor message handling.
#[derive(Debug, Clone)]
pub struct Model {
    pub info: Arc<ModelInfo>,
    pub tokenizer: Arc<BytePairEncoder>,
}

impl Model {
    /// Handshake timeout.
    const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

    /// Perform handshake with Python backend.
    pub async fn handshake(backend: &RpcBackend) -> Result<HandshakeResponse> {
        let req = HandshakeRequest {
            version: "0.1.0".to_string(),
        };
        backend
            .call_with_timeout("handshake", &req, Self::HANDSHAKE_TIMEOUT)
            .await
    }

    /// Create from handshake response.
    pub fn from_handshake(resp: HandshakeResponse) -> Self {
        let tokenizer = Arc::new(BytePairEncoder::new(
            resp.tokenizer_num_vocab,
            resp.tokenizer_merge_table.into_iter().collect(),
            resp.tokenizer_special_tokens,
            &resp.tokenizer_split_regex,
            resp.tokenizer_escape_non_printable,
            resp.tokenizer_sentencepiece_space,
        ));

        // Compute stop_token_ids by tokenizing the stop_tokens
        let stop_token_ids: Vec<u32> = resp.prompt_stop_tokens
            .iter()
            .flat_map(|s| tokenizer.encode_with_special_tokens(s))
            .collect();

        let info = Arc::new(ModelInfo {
            name: resp.model_name,
            traits: resp.model_traits,
            description: resp.model_description,
            prompt_template: resp.prompt_template,
            prompt_template_type: resp.prompt_template_type,
            stop_tokens: resp.prompt_stop_tokens,
            stop_token_ids,
            kv_page_size: resp.kv_page_size,
            max_batch_tokens: resp.max_batch_tokens,
        });

        Model { info, tokenizer }
    }

    /// Gets the model name.
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Gets the prompt template.
    pub fn get_prompt_template(&self) -> String {
        self.info.prompt_template.clone()
    }

    /// Tokenizes text into token IDs.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode_with_special_tokens(text)
    }

    /// Detokenizes token IDs into text.
    pub fn detokenize(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens).unwrap_or_default()
    }

    /// Gets the vocabulary.
    pub fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.tokenizer.get_vocabs()
    }

    /// Gets the split regex.
    pub fn get_split_regex(&self) -> String {
        self.tokenizer.get_split_regex()
    }

    /// Gets the special tokens.
    pub fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.tokenizer.get_special_tokens()
    }

    /// Gets a clone of the tokenizer Arc.
    pub fn tokenizer(&self) -> Arc<BytePairEncoder> {
        Arc::clone(&self.tokenizer)
    }

    /// Gets the stop tokens.
    pub fn get_stop_tokens(&self) -> Vec<u32> {
        self.info.stop_token_ids.clone()
    }

    /// Gets the KV page size.
    pub fn kv_page_size(&self) -> u32 {
        self.info.kv_page_size
    }

    /// Gets the max batch tokens.
    pub fn max_batch_tokens(&self) -> usize {
        self.info.max_batch_tokens
    }
}
