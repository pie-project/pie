//! Model Service - Model registration and tokenizer management
//!
//! This module provides model metadata and tokenizer management via a global cache.
//! All model and tokenizer operations access the cache directly without message passing.

use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use anyhow::Result;

pub mod instruct;
pub mod tokenizer;

use instruct::Instruct;
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
    kv_capacity_tokens: u32,
    default_token_limit: Option<u32>,
    tokenizer_path: PathBuf,
) -> Result<()> {
    let tokenizer = Arc::new(Tokenizer::from_file(&tokenizer_path)?);
    let instruct = instruct::create(arch_name, tokenizer.clone());

    let model = Arc::new(Model {
        name,
        instruct,
        kv_page_size,
        kv_capacity_tokens,
        default_token_limit,
        tokenizer,
    });
    MODELS.push(model);
    Ok(())
}

/// Returns a list of all registered model names.
pub fn models() -> Vec<String> {
    MODELS.iter().map(|(_, model)| model.name().to_string()).collect()
}

/// Minimum per-request output-token ceiling across registered models;
/// 0 if none. Per model: the configured `default_token_limit` (the
/// scheduler's total-token compute cap) when set, else the raw KV
/// capacity.
pub fn min_output_token_ceiling() -> u32 {
    MODELS
        .iter()
        .map(|(_, model)| {
            output_token_ceiling_for_model(model.default_token_limit, model.kv_capacity_tokens())
        })
        .min()
        .unwrap_or(0)
}

fn output_token_ceiling_for_model(
    default_token_limit: Option<u32>,
    kv_capacity_tokens: u32,
) -> u32 {
    default_token_limit
        .unwrap_or(kv_capacity_tokens)
        .min(kv_capacity_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_default_token_limit_is_clamped_to_physical_kv_capacity() {
        assert_eq!(output_token_ceiling_for_model(Some(4096), 1024), 1024);
    }
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
    instruct: Arc<dyn Instruct>,
    kv_page_size: u32,
    kv_capacity_tokens: u32,
    default_token_limit: Option<u32>,
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

    /// Gets the instruct implementation for this model.
    pub fn instruct(&self) -> &dyn Instruct {
        &*self.instruct
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

    /// Gets the KV page size.
    pub fn kv_page_size(&self) -> u32 {
        self.kv_page_size
    }

    /// Gets the engine KV-cache capacity for this model in tokens.
    pub fn kv_capacity_tokens(&self) -> u32 {
        self.kv_capacity_tokens
    }
}
