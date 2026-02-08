//! Model Service - Model registration and tokenizer management
//!
//! This module provides model metadata and tokenizer management via a global cache.
//! All model and tokenizer operations access the cache directly without message passing.

pub mod tokenizer;

use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use anyhow::Result;
use tokenizer::BytePairEncoder;

/// Global cache for models (keyed by ModelId).
static MODELS: LazyLock<boxcar::Vec<Model>> =
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
    chat_template: String,
    stop_tokens: Vec<u32>,
    kv_page_size: u32,
    tokenizer_path: PathBuf,
) -> Result<()> {
    let tokenizer = BytePairEncoder::load(&tokenizer_path)?;
    let model = Model {
        inner: Arc::new(Inner {
            name,
            chat_template,
            stop_tokens,
            kv_page_size,
            tokenizer,
        }),
    };
    MODELS.push(model);
    Ok(())
}

/// Returns a list of all registered model names.
pub fn models() -> Vec<String> {
    MODELS.iter().map(|(_, model)| model.name().to_string()).collect()
}

/// Gets cached model by model ID.
pub fn get_model(model_id: ModelId) -> Option<&'static Model> {
    MODELS.get(model_id)
}

// =============================================================================
// Model
// =============================================================================

#[derive(Clone)]
pub struct Model {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("name", &self.inner.name)
            .finish()
    }
}

struct Inner {
    name: String,
    chat_template: String,
    stop_tokens: Vec<u32>,
    kv_page_size: u32,
    tokenizer: BytePairEncoder,
}

impl Model {
    /// Gets the model name.
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Gets the chat template.
    pub fn chat_template(&self) -> &str {
        &self.inner.chat_template
    }

    /// Tokenizes text into token IDs.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.inner.tokenizer.encode_with_special_tokens(text)
    }

    /// Detokenizes token IDs into text.
    pub fn detokenize(&self, tokens: &[u32]) -> String {
        self.inner.tokenizer.decode(tokens).unwrap_or_default()
    }

    /// Gets the vocabulary.
    pub fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.tokenizer.get_vocabs()
    }

    /// Gets the split regex.
    pub fn get_split_regex(&self) -> String {
        self.inner.tokenizer.get_split_regex()
    }

    /// Gets the special tokens.
    pub fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.inner.tokenizer.get_special_tokens()
    }

    /// Gets the stop tokens.
    pub fn stop_tokens(&self) -> &[u32] {
        &self.inner.stop_tokens
    }

    /// Gets the KV page size.
    pub fn kv_page_size(&self) -> u32 {
        self.inner.kv_page_size
    }
}
