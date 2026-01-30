//! Model Service - Model registration and tokenizer management
//!
//! This module provides a model-specific actor for managing model metadata,
//! tokenization, and coordinating associated actors (context, inference, kvcache).

use std::sync::LazyLock;
use tokio::sync::oneshot;

use crate::actor::{Handle, Actors, SendError};
use crate::context;
use crate::inference;

/// Global address table for model actors.
static ACTOR: LazyLock<Actors<Message>> = LazyLock::new(Actors::new);

/// Installs a new model and spawns its associated actors.
/// Returns the global model ID that can be used with all actors.
pub fn install_model(info: ModelInfo) -> usize {
    // Spawn the model actor and get its ID
    let model_id = ACTOR.spawn_with::<ModelActor, _>(|| ModelActor::with_info(info));

    // Spawn the associated actors with the same model ID
    // Note: PageStore is now owned by ContextActor, so no separate kvcache actor
    let _ = context::spawn();
    let _ = inference::spawn();

    model_id
}

/// Model information.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub prompt_template: String,
    pub stop_tokens: Vec<String>,
}

/// Tokenizer information.
#[derive(Debug, Clone)]
pub struct TokenizerInfo {
    pub vocabs: (Vec<u32>, Vec<Vec<u8>>),
    pub split_regex: String,
    pub special_tokens: (Vec<u32>, Vec<Vec<u8>>),
}

/// Messages for the model actor.
#[derive(Debug)]
pub enum Message {
    /// Gets the prompt template for this model.
    GetPromptTemplate {
        response: oneshot::Sender<String>,
    },
    /// Gets the stop tokens for this model.
    GetStopTokens {
        response: oneshot::Sender<Vec<String>>,
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
}

impl Message {
    /// Sends this message to the model actor for the given model.
    pub fn send(self, model_idx: usize) -> Result<(), SendError> {
        ACTOR.send(model_idx, self)
    }
}

/// The model actor provides model and tokenizer functionality.
#[derive(Debug)]
struct ModelActor {
    service: ModelService,
}

impl ModelActor {
    fn with_info(info: ModelInfo) -> Self {
        ModelActor {
            service: ModelService::new(info),
        }
    }
}

impl Handle for ModelActor {
    type Message = Message;

    fn new() -> Self {
        ModelActor {
            service: ModelService::new(ModelInfo {
                name: String::new(),
                prompt_template: String::new(),
                stop_tokens: Vec::new(),
            }),
        }
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::GetPromptTemplate { response } => {
                let _ = response.send(self.service.get_prompt_template());
            }
            Message::GetStopTokens { response } => {
                let _ = response.send(self.service.get_stop_tokens());
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
        }
    }
}

// =============================================================================
// ModelService - Business Logic
// =============================================================================

/// The model service handles model metadata and tokenization.
///
/// This is the core business logic, separate from the actor message handling.
#[derive(Debug)]
pub struct ModelService {
    info: ModelInfo,
    tokenizer: Option<TokenizerInfo>,
}

impl ModelService {
    /// Creates a new model service with the given info.
    pub fn new(info: ModelInfo) -> Self {
        ModelService {
            info,
            tokenizer: None,
        }
    }

    /// Sets the tokenizer info.
    pub fn set_tokenizer(&mut self, tokenizer: TokenizerInfo) {
        self.tokenizer = Some(tokenizer);
    }

    /// Gets the model name.
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Gets the prompt template.
    pub fn get_prompt_template(&self) -> String {
        self.info.prompt_template.clone()
    }

    /// Gets the stop tokens.
    pub fn get_stop_tokens(&self) -> Vec<String> {
        self.info.stop_tokens.clone()
    }

    /// Tokenizes text into token IDs.
    pub fn tokenize(&self, _text: &str) -> Vec<u32> {
        // TODO: Implement actual tokenization via backend
        vec![]
    }

    /// Detokenizes token IDs into text.
    pub fn detokenize(&self, _tokens: &[u32]) -> String {
        // TODO: Implement actual detokenization via backend
        String::new()
    }

    /// Gets the vocabulary.
    pub fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.tokenizer
            .as_ref()
            .map(|t| t.vocabs.clone())
            .unwrap_or_default()
    }

    /// Gets the split regex.
    pub fn get_split_regex(&self) -> String {
        self.tokenizer
            .as_ref()
            .map(|t| t.split_regex.clone())
            .unwrap_or_default()
    }

    /// Gets the special tokens.
    pub fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.tokenizer
            .as_ref()
            .map(|t| t.special_tokens.clone())
            .unwrap_or_default()
    }
}
