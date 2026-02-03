//! Model Service - Model registration and tokenizer management
//!
//! This module provides a model-specific actor for managing model metadata,
//! tokenization, and coordinating associated actors (context, inference, kvcache).

pub mod tokenizer;

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, RwLock};
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::actor::{Actors, Handle, SendError};
use crate::context;
use crate::inference;

use crate::ffi::AsyncIpcClient;
use tokenizer::BytePairEncoder;

/// Global address table for model actors.
static ACTOR: LazyLock<Actors<Message>> = LazyLock::new(Actors::new);

/// Global registry mapping model names to service IDs.
static NAME_REGISTRY: LazyLock<RwLock<HashMap<String, usize>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Looks up a model by name and returns its service ID.
pub fn model_service_id(model_name: &str) -> Option<usize> {
    NAME_REGISTRY.read().ok()?.get(model_name).copied()
}

/// Returns a list of all registered model names.
pub fn registered_models() -> Vec<String> {
    NAME_REGISTRY
        .read()
        .map(|r| r.keys().cloned().collect())
        .unwrap_or_default()
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
// RPC Backend
// =============================================================================

/// RPC backend for Python IPC communication.
#[derive(Clone)]
pub struct RpcBackend {
    client: AsyncIpcClient,
}

impl RpcBackend {
    /// Create a new RPC backend from an IPC client.
    pub fn new(client: AsyncIpcClient) -> Self {
        Self { client }
    }

    /// Call a Python method asynchronously via IPC.
    pub async fn call<T, R>(&self, method: &str, args: &T) -> Result<R>
    where
        T: Serialize + Send + Sync + Clone + 'static,
        R: serde::de::DeserializeOwned + Send + 'static,
    {
        self.client.call(method, args).await
    }

    /// Call with timeout.
    pub async fn call_with_timeout<T, R>(
        &self,
        method: &str,
        args: &T,
        timeout: Duration,
    ) -> Result<R>
    where
        T: Serialize + Send + Sync + Clone + 'static,
        R: serde::de::DeserializeOwned + Send + 'static,
    {
        self.client.call_with_timeout(method, args, timeout).await
    }
}

impl std::fmt::Debug for RpcBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RpcBackend").finish()
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Installs a new model by performing handshake with the backend.
/// Returns the global model ID that can be used with all actors.
pub async fn install_model_with_backend(backend: RpcBackend) -> Result<usize> {
    let resp = Model::handshake(&backend).await?;
    let model_name = resp.model_name.clone();
    let service = Model::from_handshake(resp);

    let model_id = ACTOR.spawn_with::<ModelActor, _>(|| ModelActor {
        service,
        #[allow(dead_code)]
        backend: Some(backend),
    });

    // Register the model name
    if let Ok(mut registry) = NAME_REGISTRY.write() {
        registry.insert(model_name, model_id);
    }

    // Spawn the associated actors with the same model ID
    // Note: PageStore is now owned by ContextActor, so no separate kvcache actor
    let _ = context::spawn();
    let _ = inference::spawn();

    Ok(model_id)
}

/// Installs a new model from pre-existing info (for testing or manual setup).
/// Returns the global model ID that can be used with all actors.
pub fn install_model(info: ModelInfo) -> usize {
    let model_name = info.name.clone();
    
    // Spawn the model actor and get its ID
    let model_id = ACTOR.spawn_with::<ModelActor, _>(|| ModelActor::with_info(info));

    // Register the model name
    if let Ok(mut registry) = NAME_REGISTRY.write() {
        registry.insert(model_name, model_id);
    }

    // Spawn the associated actors with the same model ID
    // Note: PageStore is now owned by ContextActor, so no separate kvcache actor
    let _ = context::spawn();
    let _ = inference::spawn();

    model_id
}

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
// Messages
// =============================================================================

/// Messages for the model actor.
#[derive(Debug)]
pub enum Message {
    /// Gets the prompt template for this model.
    GetPromptTemplate {
        response: oneshot::Sender<String>,
    },
    /// Tokenizes text.
    Tokenize {
        text: String,
        response: oneshot::Sender<Vec<u32>>,
    },
    /// Detokenizes token IDs.
    Detokenize {
        tokens: Vec<u32>,
        response: oneshot::Sender<String>,
    },
    /// Gets the vocabulary.
    GetVocabs {
        response: oneshot::Sender<(Vec<u32>, Vec<Vec<u8>>)>,
    },
    /// Gets the split regex.
    GetSplitRegex {
        response: oneshot::Sender<String>,
    },
    /// Gets the special tokens.
    GetSpecialTokens {
        response: oneshot::Sender<(Vec<u32>, Vec<Vec<u8>>)>,
    },
    /// Gets the stop tokens (as token IDs).
    GetStopTokens {
        response: oneshot::Sender<Vec<u32>>,
    },
    /// Gets the model info.
    GetInfo {
        response: oneshot::Sender<ModelInfo>,
    },
}

impl Message {
    /// Sends this message to the model actor for the given model.
    pub fn send(self, model_idx: usize) -> Result<(), SendError> {
        ACTOR.send(model_idx, self)
    }
}

// =============================================================================
// Model Actor
// =============================================================================

/// The model actor provides model and tokenizer functionality.
struct ModelActor {
    service: Model,
    #[allow(dead_code)]
    backend: Option<RpcBackend>,
}

impl ModelActor {
    fn with_info(info: ModelInfo) -> Self {
        ModelActor {
            service: Model::new(info),
            backend: None,
        }
    }
}

impl std::fmt::Debug for ModelActor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelActor")
            .field("service", &self.service)
            .finish()
    }
}

impl Handle for ModelActor {
    type Message = Message;

    fn new() -> Self {
        ModelActor {
            service: Model::new(ModelInfo::default()),
            backend: None,
        }
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::GetPromptTemplate { response } => {
                let _ = response.send(self.service.get_prompt_template());
            }
            Message::Tokenize { text, response } => {
                let _ = response.send(self.service.tokenize(&text));
            }
            Message::Detokenize { tokens, response } => {
                let _ = response.send(self.service.detokenize(&tokens));
            }
            Message::GetVocabs { response } => {
                let _ = response.send(self.service.get_vocabs());
            }
            Message::GetSplitRegex { response } => {
                let _ = response.send(self.service.get_split_regex());
            }
            Message::GetSpecialTokens { response } => {
                let _ = response.send(self.service.get_special_tokens());
            }
            Message::GetStopTokens { response } => {
                let _ = response.send(self.service.get_stop_tokens());
            }
            Message::GetInfo { response } => {
                let _ = response.send(self.service.info.clone());
            }
        }
    }
}

// =============================================================================
// Model - Business Logic
// =============================================================================

/// The model service handles model metadata and tokenization.
///
/// This is the core business logic, separate from the actor message handling.
#[derive(Debug)]
pub struct Model {
    info: ModelInfo,
    tokenizer: Option<Arc<BytePairEncoder>>,
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

        let info = ModelInfo {
            name: resp.model_name,
            traits: resp.model_traits,
            description: resp.model_description,
            prompt_template: resp.prompt_template,
            prompt_template_type: resp.prompt_template_type,
            stop_tokens: resp.prompt_stop_tokens,
            stop_token_ids,
            kv_page_size: resp.kv_page_size,
            max_batch_tokens: resp.max_batch_tokens,
        };

        Model {
            info,
            tokenizer: Some(tokenizer),
        }
    }

    /// Creates a new model service with the given info.
    pub fn new(info: ModelInfo) -> Self {
        Model {
            info,
            tokenizer: None,
        }
    }

    /// Sets the tokenizer from raw components.
    pub fn set_tokenizer(
        &mut self,
        num_vocab: usize,
        merge_table: HashMap<u32, Vec<u8>>,
        special_tokens: HashMap<String, u32>,
        split_regex: &str,
        escape_non_printable: bool,
        sentencepiece_space: bool,
    ) {
        self.tokenizer = Some(Arc::new(BytePairEncoder::new(
            num_vocab,
            merge_table,
            special_tokens,
            split_regex,
            escape_non_printable,
            sentencepiece_space,
        )));
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
        self.tokenizer
            .as_ref()
            .map(|t| t.encode_with_special_tokens(text))
            .unwrap_or_default()
    }

    /// Detokenizes token IDs into text.
    pub fn detokenize(&self, tokens: &[u32]) -> String {
        self.tokenizer
            .as_ref()
            .and_then(|t| t.decode(tokens).ok())
            .unwrap_or_default()
    }

    /// Gets the vocabulary.
    pub fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.tokenizer
            .as_ref()
            .map(|t| t.get_vocabs())
            .unwrap_or_default()
    }

    /// Gets the split regex.
    pub fn get_split_regex(&self) -> String {
        self.tokenizer
            .as_ref()
            .map(|t| t.get_split_regex())
            .unwrap_or_default()
    }

    /// Gets the special tokens.
    pub fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.tokenizer
            .as_ref()
            .map(|t| t.get_special_tokens())
            .unwrap_or_default()
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
