//! Model Service - Model registration and tokenizer management
//!
//! This module provides model metadata and tokenizer management via a global cache.
//! All model and tokenizer operations access the cache directly without message passing.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow};

pub mod instruct;
pub mod tokenizer;

use instruct::Instruct;
use tokenizer::Tokenizer;

/// The single model this engine serves. Set once at bootstrap.
static MODEL: OnceLock<Arc<Model>> = OnceLock::new();

pub fn register(
    name: String,
    arch_name: &str,
    kv_page_size: u32,
    tokenizer_path: PathBuf,
    system_speculation_supported: bool,
    enable_system_speculation: bool,
) -> Result<()> {
    let tokenizer = Arc::new(Tokenizer::from_file(&tokenizer_path)?);
    let instruct = instruct::create(arch_name, tokenizer.clone());

    let model = Arc::new(Model {
        name,
        arch_name: arch_name.to_string(),
        instruct,
        kv_page_size,
        tokenizer,
        system_speculation_supported,
        enable_system_speculation,
    });
    MODEL
        .set(model)
        .map_err(|_| anyhow!("a model is already registered; the engine serves exactly one model"))?;
    Ok(())
}

/// Returns the single registered model. Panics if called before bootstrap
/// registers the model.
pub fn model() -> &'static Arc<Model> {
    MODEL
        .get()
        .expect("model accessed before registration")
}

// =============================================================================
// Model
// =============================================================================

pub struct Model {
    name: String,
    /// Architecture identifier supplied at registration (e.g. "gemma4",
    /// "qwen3_6"). Used to select the multimodal processor / vision front-end.
    arch_name: String,
    instruct: Arc<dyn Instruct>,
    kv_page_size: u32,
    tokenizer: Arc<Tokenizer>,
    system_speculation_supported: bool,
    enable_system_speculation: bool,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").field("name", &self.name).finish()
    }
}

impl Model {
    /// Gets the model name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the architecture identifier (e.g. "gemma4", "qwen3_6").
    pub fn arch_name(&self) -> &str {
        &self.arch_name
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

    /// Whether the driver wired a system drafter for this model (capability).
    /// Required to verify manual drafts; auto-drafting additionally requires
    /// [`Self::enable_system_speculation`].
    pub fn system_speculation_supported(&self) -> bool {
        self.system_speculation_supported
    }

    /// Operator opt-in for system speculation (deployment config, default
    /// false). The runtime drives system drafts only when this is true; manual
    /// (user-supplied) drafts are honored regardless of this flag.
    pub fn enable_system_speculation(&self) -> bool {
        self.enable_system_speculation
    }
}
