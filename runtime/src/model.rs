//! Model Service - Model registration and tokenizer management
//!
//! This module provides model metadata and tokenizer management via a global cache.
//! All model and tokenizer operations access the cache directly without message passing.

use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use anyhow::Result;

pub mod chat_templates;
pub mod tokenizer;

use chat_templates::ChatTemplate;
use tokenizer::Tokenizer;

/// Global cache for models (keyed by ModelId).
static MODELS: LazyLock<boxcar::Vec<Arc<Model>>> =
    LazyLock::new(|| boxcar::Vec::new());

/// Type alias for model identifiers.
pub type ModelId = usize;

/// Looks up a model by name and returns its model ID.
pub fn get_model_id(model_name: &str) -> Option<ModelId> {
    for (model_id, model) in MODELS.iter() {
        if model.name() == model_name {
            return Some(model_id);
        }
    }
    None
}

pub fn register(
    name: String,
    arch_name: &str,
    kv_page_size: u32,
    tokenizer_path: PathBuf,
) -> Result<()> {
    let tokenizer = Arc::new(Tokenizer::from_file(&tokenizer_path)?);

    let chat_template = chat_templates::lookup(arch_name)
        .unwrap_or_else(|| chat_templates::lookup("dummy").unwrap());

    // Convert stop token strings to IDs using the tokenizer's vocabulary.
    let stop_tokens: Vec<u32> = chat_template
        .stop_tokens
        .iter()
        .filter_map(|s| tokenizer.token_to_id(s))
        .collect();

    let model = Arc::new(Model {
        name,
        chat_template,
        stop_tokens,
        kv_page_size,
        tokenizer,
    });
    MODELS.push(model);
    Ok(())
}

/// Returns a list of all registered model names.
pub fn models() -> Vec<String> {
    MODELS.iter().map(|(_, model)| model.name().to_string()).collect()
}

/// Gets cached model by model ID.
pub fn get_model(model_id: ModelId) -> Option<&'static Arc<Model>> {
    MODELS.get(model_id)
}

// =============================================================================
// Model
// =============================================================================

pub struct Model {
    name: String,
    chat_template: &'static ChatTemplate,
    stop_tokens: Vec<u32>,
    kv_page_size: u32,
    tokenizer: Arc<Tokenizer>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("name", &self.name)
            .finish()
    }
}

impl Model {
    /// Gets the model name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the chat template.
    pub fn chat_template(&self) -> &'static ChatTemplate {
        self.chat_template
    }

    /// Gets the tokenizer.
    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    /// Tokenizes text into token IDs.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// Detokenizes token IDs into text.
    pub fn detokenize(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens, false)
    }

    /// Gets the vocabulary as parallel vectors of (token IDs, token bytes).
    pub fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        let size = self.tokenizer.vocab_size();
        let mut ids = Vec::with_capacity(size);
        let mut bytes = Vec::with_capacity(size);
        for id in 0..size as u32 {
            if let Some(tok_bytes) = self.tokenizer.id_to_token(id) {
                ids.push(id);
                bytes.push(tok_bytes);
            }
        }
        (ids, bytes)
    }

    /// Gets the split regex pattern.
    pub fn get_split_regex(&self) -> String {
        self.tokenizer.get_split_regex()
    }

    /// Gets the special tokens.
    pub fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.tokenizer.get_special_tokens()
    }

    /// Gets the stop tokens.
    pub fn stop_tokens(&self) -> &[u32] {
        &self.stop_tokens
    }

    /// Gets the KV page size.
    pub fn kv_page_size(&self) -> u32 {
        self.kv_page_size
    }
}
