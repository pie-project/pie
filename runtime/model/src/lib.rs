//! Model Service - Model registration and tokenizer management
//!
//! This module provides model metadata and tokenizer management via a global cache.
//! All model and tokenizer operations access the cache directly without message passing.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow};

pub mod instruct;
pub mod multimodal;

use instruct::Instruct;
use pie_tokenizer::Tokenizer;

/// The single model this engine serves. Set once at bootstrap.
static MODEL: OnceLock<Arc<Model>> = OnceLock::new();

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
    rs: RsCaps,
    ptir: PtirCaps,
    tokenizer_path: PathBuf,
) -> Result<()> {
    let tokenizer = Arc::new(Tokenizer::from_file(&tokenizer_path)?);
    let instruct = instruct::create(arch_name, tokenizer.clone());

    // Logits dim = the model's hf_config.vocab_size (read from the snapshot's
    // config.json beside the tokenizer). Falls back to the tokenizer vocab for
    // mock/fixture setups without a config.json. This is the dim the sampler
    // operates on + the driver's recognizer table is keyed by — NOT the
    // tokenizer token count, which may be smaller (qwen3: 151669 vs 151936).
    let vocab_size =
        read_snapshot_vocab_size(&tokenizer_path).unwrap_or_else(|| tokenizer.vocab_size() as u32);

    let model = Arc::new(Model {
        name,
        arch_name: arch_name.to_string(),
        instruct,
        kv_page_size,
        rs_caps: rs,
        ptir_caps: ptir,
        tokenizer,
        vocab_size,
    });
    MODEL.set(model).map_err(|_| {
        anyhow!("a model is already registered; the engine serves exactly one model")
    })?;
    Ok(())
}

/// Returns the single registered model. Panics if called before bootstrap
/// registers the model.
pub fn model() -> &'static Arc<Model> {
    MODEL.get().expect("model accessed before registration")
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
    /// Recurrent-state (working-set) capabilities surfaced via model.wit
    /// (`rs-state-size`/`rs-buffer-page-size`/`rs-fold-granularity`). All
    /// 0/0/1 for pure-attention models.
    rs_caps: RsCaps,
    ptir_caps: PtirCaps,
    tokenizer: Arc<Tokenizer>,
    /// Logits/output vocab dimension (= hf_config.vocab_size from the model's
    /// config.json). May EXCEED tokenizer.vocab_size() due to padding — use
    /// THIS for sampler lowering / logits-shaped ops, NOT the tokenizer vocab.
    vocab_size: u32,
}

/// RS (recurrent-state) working-set capabilities surfaced to inferlets via
/// `model.wit`. Sourced from the driver handshake `DriverCapabilities` at
/// registration (`rs_cache_slot_bytes` etc.). All 0/0/1 for pure-attention
/// models (no folded recurrent state).
#[derive(Debug, Clone, Copy)]
pub struct RsCaps {
    /// Bytes of one folded recurrent-state object (`rs-state-size`).
    pub state_size: u64,
    /// Tokens per buffered RS page (`rs-buffer-page-size`; v1 = kv_page_size).
    pub buffer_page_size: u32,
    /// Fold granularity in tokens (`rs-fold-granularity`; 1 = token-causal).
    pub fold_granularity: u32,
}

/// Model-gated values that a loaded backend can bind into PTIR programs.
#[derive(Debug, Clone, Copy, Default)]
pub struct PtirCaps {
    pub has_mtp_logits: bool,
    pub has_mtp_drafts: bool,
    pub has_value_head: bool,
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

    /// RS working-set capabilities (`rs-state-size`/`rs-buffer-page-size`/
    /// `rs-fold-granularity`). 0/0/1 for pure-attention models.
    pub fn rs_caps(&self) -> RsCaps {
        self.rs_caps
    }

    pub fn ptir_caps(&self) -> PtirCaps {
        self.ptir_caps
    }
}
