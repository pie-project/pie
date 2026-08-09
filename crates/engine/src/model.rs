//! The served model: the runtime's global model/tokenizer cache.
//!
//! Set once at bootstrap and read from everywhere after. It lived in
//! `model` while that crate was the only place a `Tokenizer` and an
//! `Instruct` could be assembled; it is here now because `model` defines
//! itself as *what each model family is, backend-blind*, and a process-global
//! `OnceLock` holding whatever this engine happens to have booted is a fact
//! about the process, not about any model.
//!
//! What stayed behind is [`::model::ModelMetadata`]: the shape an
//! artifact's compiled metadata arrives in, which the worker reads without
//! linking the runtime.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow};

use ::model::ModelMetadata;
use ::model::catalog::{self, Deployed, Variant};
use ::model::instruct::Instruct;
use tokenizer::Tokenizer;

/// The single model this engine serves. Set once at bootstrap.
static MODEL: OnceLock<Arc<Model>> = OnceLock::new();

/// Rebuild the compiled tokenizer an artifact carried, without touching the
/// filesystem. `None` when the model is a snapshot, whose tokenizer is a file.
///
/// Here rather than on [`ModelMetadata`] because it is *consumption*: the type
/// says what an artifact carries, and this says what this runtime does with
/// it. `model` should not need a `Tokenizer` to describe a shape.
fn compiled_tokenizer(metadata: &ModelMetadata) -> Option<Result<Tokenizer>> {
    let objects = metadata.tokenizer.as_ref()?;
    Some((|| {
        let canonical = tokenizer::canonical::CanonicalTokenizer::from_objects(|name| {
            objects
                .iter()
                .find(|(have, _)| have == name)
                .map(|(_, bytes)| bytes.clone())
        })?;
        Tokenizer::from_canonical(&canonical)
    })())
}

/// The row the driver loaded, or a refusal naming what is close.
///
/// Two numbers used to come from a `pie.model/1` descriptor parsed
/// here — `vocab_size` and `num_hidden_layers` — and a third place
/// parsed the same document with a different failure policy. The
/// descriptor was one document with two readers, and this was the
/// second one.
///
/// It is the row now, and the row is the same `const` table the driver
/// linked to answer with this id. Both sides knowing one fact
/// differently is not a bug that got fixed; it is a sentence that can
/// no longer be written.
fn loaded_row(model_id: &str) -> Result<&'static dyn Variant> {
    catalog::find(model_id).ok_or_else(|| {
        anyhow!(
            "the driver loaded {model_id:?}, which this build's model catalog \
             does not contain; nearest ids: {:?}",
            catalog::nearest_ids(model_id, 3)
        )
    })
}

#[allow(
    clippy::too_many_arguments,
    reason = "one model row, stated once: its name and architecture, the catalog id \
              it resolves to, the KV page size, its recurrent-state and PTIR \
              capability sets, its tokenizer, and the checkpoint metadata. These \
              come from different sources at the call site, so a struct would need \
              assembling field-by-field first"
)]
pub fn register(
    name: String,
    arch_name: &str,
    model_id: &str,
    kv_page_size: u32,
    rs: RsCaps,
    ptir: PtirCaps,
    tokenizer_path: PathBuf,
    metadata: &ModelMetadata,
) -> Result<()> {
    // Logits dim = the model's `vocab_size`. This is the dim the sampler
    // operates on and the driver's recognizer table is keyed by — NOT the
    // tokenizer token count, which may be smaller (qwen3: 151669 vs 151936).
    // Getting that wrong is the vocab-padding device fault the note above
    // describes.
    //
    // Both facts come off the ROW, which is the same table the driver
    // linked. They used to come from a descriptor parsed here; before
    // that, a snapshot got them from two probes right here that walked
    // `text_config` nesting and key alternatives by hand and had to
    // agree with the driver's parser by coincidence. `pie.model/1`
    // removed the second parser and left one document with two readers;
    // this removes the document.
    let row = loaded_row(model_id)?;
    let num_layers = row.load_shape().layers;
    // `vocab` is the LOGICAL width and does not shard, so the single-rank
    // projection answers it at any tensor-parallel width.
    let vocab_size = row
        .deployment(Deployed::single())
        .map_err(|refusal| {
            anyhow!("the driver loaded {model_id:?} but this build refuses it: {refusal}")
        })?
        .shape
        .vocab;
    // The tokenizer is the half that genuinely differs: compiled objects from
    // an artifact, a file beside a snapshot.
    let tokenizer = match compiled_tokenizer(metadata) {
        Some(compiled) => compiled?,
        None => Tokenizer::from_file(&tokenizer_path)?,
    };
    let tokenizer = Arc::new(tokenizer);
    // THE CHAT TEMPLATE, CHOSEN BY THE ROW THE DRIVER LOADED.
    //
    // `create` used to take `arch_name` — a `model_type` string off the
    // checkpoint's `config.json` — and match it against a thirty-arm
    // table ending in `_ => QwenInstruct`. A model the table did not
    // know got ChatML, silently: no error, no warning, just a model
    // generating fluent turns of a conversation it was never trained to
    // hold, terminated with an `<|im_end|>` its tokenizer does not
    // contain. The bug was invisible precisely because the output read
    // well.
    //
    // The id names a catalog row, every row answers `chat()`, and there
    // is no arm left to fall through. An unknown id was already an error
    // above, at `loaded_row`, with the nearest known ids named — before
    // the first token rather than after a thousand plausible ones.
    //
    // The SAME row that answered the two numbers answers this, which is
    // the property worth having: a build cannot size its sampler from
    // one model and format its prompts for another.
    let instruct = row.chat(tokenizer.clone());

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
