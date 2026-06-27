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
static MODELS: LazyLock<boxcar::Vec<Arc<Model>>> = LazyLock::new(|| boxcar::Vec::new());

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

/// Read `vocab_size` (the logits/output dim) from the model snapshot's
/// `config.json`, located beside the tokenizer. `None` if absent (e.g. mock
/// fixtures) — callers fall back to the tokenizer vocab. Mirrors the driver's
/// `config.json` discovery so host + driver agree on the logits dim.
fn read_snapshot_vocab_size(tokenizer_path: &std::path::Path) -> Option<u32> {
    let cfg = tokenizer_path.parent()?.join("config.json");
    let text = std::fs::read_to_string(cfg).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("vocab_size")?.as_u64().map(|n| n as u32)
}

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

    // Logits dim = the model's hf_config.vocab_size (read from the snapshot's
    // config.json beside the tokenizer). Falls back to the tokenizer vocab for
    // mock/fixture setups without a config.json. This is the dim the sampler
    // operates on + the driver's recognizer table is keyed by — NOT the
    // tokenizer token count, which may be smaller (qwen3: 151669 vs 151936).
    let vocab_size = read_snapshot_vocab_size(&tokenizer_path)
        .unwrap_or_else(|| tokenizer.vocab_size() as u32);

    let model = Arc::new(Model {
        name,
        arch_name: arch_name.to_string(),
        instruct,
        kv_page_size,
        tokenizer,
        vocab_size,
        system_speculation_supported,
        enable_system_speculation,
    });
    MODELS.push(model);
    Ok(())
}

/// Returns a list of all registered model names.
pub fn models() -> Vec<String> {
    MODELS
        .iter()
        .map(|(_, model)| model.name().to_string())
        .collect()
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
    /// Architecture identifier supplied at registration (e.g. "gemma4",
    /// "qwen3_6"). Used to select the multimodal processor / vision front-end.
    arch_name: String,
    instruct: Arc<dyn Instruct>,
    kv_page_size: u32,
    tokenizer: Arc<Tokenizer>,
    /// Logits/output vocab dimension (= hf_config.vocab_size from the model's
    /// config.json). May EXCEED tokenizer.vocab_size() due to padding — use
    /// THIS for sampler lowering / logits-shaped ops, NOT the tokenizer vocab.
    vocab_size: u32,
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

    /// Logits/output vocab dimension (= hf_config.vocab_size). The dim the
    /// sampler operates on and the driver's recognizer table is keyed by. May
    /// EXCEED the tokenizer's vocab (qwen3: 151936 logits vs 151669 tokens) —
    /// use this for sampler lowering / logits-shaped ops, NOT tokenizer vocab.
    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
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
