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
    info: ModelInfo,
    tokenizer: Option<TokenizerInfo>,
}

impl ModelActor {
    fn with_info(info: ModelInfo) -> Self {
        ModelActor {
            info,
            tokenizer: None,
        }
    }
}

impl Handle for ModelActor {
    type Message = Message;

    fn new() -> Self {
        ModelActor {
            info: ModelInfo {
                name: String::new(),
                prompt_template: String::new(),
                stop_tokens: Vec::new(),
            },
            tokenizer: None,
        }
    }

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::GetPromptTemplate { response } => {
                let _ = response.send(self.info.prompt_template.clone());
            }
            Message::GetStopTokens { response } => {
                let _ = response.send(self.info.stop_tokens.clone());
            }
            Message::Tokenize { text: _, response } => {
                // TODO: Implement actual tokenization via backend
                let _ = response.send(vec![]);
            }
            Message::Detokenize { tokens: _, response } => {
                // TODO: Implement actual detokenization via backend
                let _ = response.send(String::new());
            }
            Message::GetVocabs { response } => {
                let vocabs = self
                    .tokenizer
                    .as_ref()
                    .map(|t| t.vocabs.clone())
                    .unwrap_or_default();
                let _ = response.send(vocabs);
            }
            Message::GetSplitRegex { response } => {
                let regex = self
                    .tokenizer
                    .as_ref()
                    .map(|t| t.split_regex.clone())
                    .unwrap_or_default();
                let _ = response.send(regex);
            }
            Message::GetSpecialTokens { response } => {
                let tokens = self
                    .tokenizer
                    .as_ref()
                    .map(|t| t.special_tokens.clone())
                    .unwrap_or_default();
                let _ = response.send(tokens);
            }
        }
    }
}
