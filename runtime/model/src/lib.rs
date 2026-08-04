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
///
/// A multimodal release nests the text decoder's facts under `text_config`,
/// which `read_snapshot_num_layers` below already accounts for. Without the
/// same fallback here the read misses, the caller silently substitutes the
/// *tokenizer* vocab, and host and driver disagree by exactly the LM head's
/// padding — which the emitted logits gather then rejects as a device fault.
fn read_snapshot_vocab_size(tokenizer_path: &std::path::Path) -> Option<u32> {
    let cfg = tokenizer_path.parent()?.join("config.json");
    let text = std::fs::read_to_string(cfg).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("vocab_size")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| {
            v.get("text_config")
                .and_then(|text| text.get("vocab_size"))
                .and_then(serde_json::Value::as_u64)
        })
        .map(|n| n as u32)
}

fn read_snapshot_num_layers(tokenizer_path: &std::path::Path) -> Option<u32> {
    let cfg = tokenizer_path.parent()?.join("config.json");
    let text = std::fs::read_to_string(cfg).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    ["num_hidden_layers", "num_layers", "n_layer"]
        .into_iter()
        .find_map(|key| value.get(key).and_then(serde_json::Value::as_u64))
        .or_else(|| {
            value
                .get("text_config")
                .and_then(|text| text.get("num_hidden_layers"))
                .and_then(serde_json::Value::as_u64)
        })
        .and_then(|layers| u32::try_from(layers).ok())
}

/// The compiled metadata a `.zt` artifact carries, lifted out by the worker.
///
/// Present, the runtime builds its tokenizer and reads its model facts from
/// here. Absent, it falls back to parsing the files beside a snapshot — the
/// path this format exists to retire, kept until serving from an artifact has
/// been exercised on real hardware.
#[derive(Clone, Debug)]
pub struct ArtifactMetadata {
    /// The `pie.tokenizer/1` objects, by their name under `__meta__/`.
    pub tokenizer: Vec<(String, Vec<u8>)>,
    /// The `pie.model/1` descriptor, as JSON.
    pub descriptor: Vec<u8>,
}

impl ArtifactMetadata {
    /// Rebuilds the tokenizer without touching the filesystem.
    fn tokenizer(&self) -> Result<Tokenizer> {
        let canonical = pie_tokenizer::canonical::CanonicalTokenizer::from_objects(|name| {
            self.tokenizer
                .iter()
                .find(|(have, _)| have == name)
                .map(|(_, bytes)| bytes.clone())
        })?;
        Tokenizer::from_canonical(&canonical)
    }

    /// The descriptor, parsed once.
    ///
    /// Reading it per field would re-parse the whole document each time — two
    /// fields, two parses — for a document that only grows as architectures
    /// land.
    fn descriptor(&self) -> Result<serde_json::Value> {
        serde_json::from_slice(&self.descriptor)
            .map_err(|err| anyhow!("the artifact's model descriptor is not JSON: {err}"))
    }
}

/// One `u32` field of a parsed descriptor. Every field is present by
/// construction — the writer resolves them all — so a missing one is a broken
/// artifact rather than a case to default around.
fn descriptor_field(descriptor: &serde_json::Value, key: &str) -> Result<u32> {
    descriptor
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| anyhow!("the artifact's model descriptor has no {key}"))
}

pub fn register(
    name: String,
    arch_name: &str,
    kv_page_size: u32,
    rs: RsCaps,
    ptir: PtirCaps,
    tokenizer_path: PathBuf,
    artifact: Option<ArtifactMetadata>,
) -> Result<()> {
    // Logits dim = the model's `vocab_size`. This is the dim the sampler
    // operates on and the driver's recognizer table is keyed by — NOT the
    // tokenizer token count, which may be smaller (qwen3: 151669 vs 151936).
    // Getting that wrong is the vocab-padding device fault the note above
    // describes, which is exactly the skew an artifact removes: there,
    // tokenizer and config are one object under one digest.
    let (tokenizer, vocab_size, num_layers) = match &artifact {
        Some(artifact) => {
            let descriptor = artifact.descriptor()?;
            (
                artifact.tokenizer()?,
                descriptor_field(&descriptor, "vocab_size")?,
                descriptor_field(&descriptor, "num_hidden_layers")?,
            )
        }
        None => {
            let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
            let vocab_size = read_snapshot_vocab_size(&tokenizer_path)
                .unwrap_or_else(|| tokenizer.vocab_size() as u32);
            let num_layers = read_snapshot_num_layers(&tokenizer_path).ok_or_else(|| {
                anyhow!(
                    "model config beside {} does not declare num_hidden_layers, num_layers, \
                     n_layer, or text_config.num_hidden_layers",
                    tokenizer_path.display()
                )
            })?;
            (tokenizer, vocab_size, num_layers)
        }
    };
    let tokenizer = Arc::new(tokenizer);
    let instruct = instruct::create(arch_name, tokenizer.clone());

    let model = Arc::new(Model {
        name,
        arch_name: arch_name.to_string(),
        instruct,
        kv_page_size,
        rs_caps: rs,
        ptir_caps: ptir,
        tokenizer,
        vocab_size,
        num_layers,
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
    /// Transformer layer count from the model snapshot's config.json.
    num_layers: u32,
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
    /// Backend can execute the `envelope_dot` second-party kernel (Quest).
    pub has_kv_envelopes: bool,
    /// Backend can observe per-position softmax attention weights at an
    /// `OnAttn` tap (`IntrinsicId::AttnScore`) -- H2O/TOVA.
    pub has_attn_score: bool,
    /// Backend honours the `attn_page_mask` sink (page-granular eviction).
    pub has_attn_page_mask: bool,
    /// Backend honours the `lora` sink (pass-wide low-rank adapter delta).
    pub has_lora: bool,
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

    pub fn num_layers(&self) -> u32 {
        self.num_layers
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
