use std::sync::LazyLock;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio::task;

use super::context;
use super::inference;

/// Model-indexed dispatcher for model services.
static MODEL_NEW_DISPATCHER: LazyLock<ModelDispatcher> = LazyLock::new(|| ModelDispatcher {
    services: boxcar::Vec::new(),
});

#[derive(Debug, Error)]
pub enum ModelDispatchError {
    #[error("Invalid model index: {0}")]
    InvalidModelIndex(usize),
}

#[derive(Debug)]
struct ModelDispatcher {
    services: boxcar::Vec<mpsc::UnboundedSender<Command>>,
}

/// Installs a new model and its associated context and inference services.
/// Returns the global model ID that can be used with all three services.
pub fn install_model(info: ModelInfo) -> usize {
    // Install the model service and get the ID from boxcar
    let svc = ModelService::new(info);
    let (tx, mut rx) = mpsc::unbounded_channel();
    let model_id = MODEL_NEW_DISPATCHER.services.push(tx);

    task::spawn(async move {
        let mut svc = svc;
        while let Some(cmd) = rx.recv().await {
            svc.handle(cmd).await;
        }
    });

    // Install the associated context service with the same ID
    context::install_service(model_id);

    // Install the associated inference service with the same ID
    inference::install_service(model_id);

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

/// Defines the set of operations available for the model service.
#[derive(Debug)]
pub enum Command {
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

impl Command {
    /// Dispatches this command to the model service for the given model.
    pub fn dispatch(self, model_id: usize) -> Result<(), ModelDispatchError> {
        let tx = MODEL_NEW_DISPATCHER
            .services
            .get(model_id)
            .ok_or(ModelDispatchError::InvalidModelIndex(model_id))?;
        tx.send(self).unwrap();
        Ok(())
    }
}

/// The model service provides model and tokenizer functionality.
#[derive(Debug)]
struct ModelService {
    info: ModelInfo,
    tokenizer: Option<TokenizerInfo>,
}

impl ModelService {
    /// Creates a new `ModelService`.
    fn new(info: ModelInfo) -> Self {
        ModelService {
            info,
            tokenizer: None,
        }
    }

    async fn handle(&mut self, cmd: Command) {
        match cmd {
            Command::GetPromptTemplate { response } => {
                let _ = response.send(self.info.prompt_template.clone());
            }
            Command::GetStopTokens { response } => {
                let _ = response.send(self.info.stop_tokens.clone());
            }
            Command::Tokenize { text: _, response } => {
                // TODO: Implement actual tokenization via backend
                let _ = response.send(vec![]);
            }
            Command::Detokenize { tokens: _, response } => {
                // TODO: Implement actual detokenization via backend
                let _ = response.send(String::new());
            }
            Command::GetVocabs { response } => {
                let vocabs = self
                    .tokenizer
                    .as_ref()
                    .map(|t| t.vocabs.clone())
                    .unwrap_or_default();
                let _ = response.send(vocabs);
            }
            Command::GetSplitRegex { response } => {
                let regex = self
                    .tokenizer
                    .as_ref()
                    .map(|t| t.split_regex.clone())
                    .unwrap_or_default();
                let _ = response.send(regex);
            }
            Command::GetSpecialTokens { response } => {
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
