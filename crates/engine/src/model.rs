//! The served model: the runtime's global model/tokenizer cache, and the
//! serving table it is built from.
//!
//! Set once at bootstrap and read from everywhere after. It lived in
//! `model` while that crate was the only place a `Tokenizer` and an
//! `Instruct` could be assembled; it is here now because `model` defines
//! itself as *what each model family is, backend-blind*, and a process-global
//! `OnceLock` holding whatever this engine happens to have booted is a fact
//! about the process, not about any model.
//!
//! # What M18 sent over the wall
//!
//! `model::serve` was that crate's answer to four serving questions, and this
//! engine plus `worker` were its only readers. M18 deleted it whole. Two of
//! the four are still `model`'s to answer and are asked through its own doors
//! — [`::model::template::template_of`] writes a turn, and the decoders that
//! read the tokens back are `model::template` re-exports. The other two are
//! HERE, because a serving fabric is the party that asks them:
//!
//! * [`ROWS`], the `(layers, vocab, arch)` a sampler, the PTIR lowering and
//!   the media front-ends are sized and dispatched from;
//! * [`ModelMetadata`], the shape an artifact's compiled metadata arrives in.
//!   `worker` lifts it off the checkpoint and hands it here in the boot
//!   bundle; it is stated in this crate because this crate is what reads it.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow};

use ::model::template::Instruct;
use tokenizer::Tokenizer;

/// The single model this engine serves. Set once at bootstrap.
static MODEL: OnceLock<Arc<Model>> = OnceLock::new();

// =============================================================================
// What an artifact carried
// =============================================================================

/// The compiled metadata a `.zt` artifact carries, lifted once by the worker.
///
/// Two objects and no schema: the checkpoint's own `config.json` bytes,
/// verbatim, and — for an artifact, absent for a snapshot — the compiled
/// tokenizer as the named objects that make up a `pie.tokenizer/1`.
///
/// It was `model::serve::ModelMetadata`, six lines with no method on it, and
/// it names nothing `model` knows: not a family, not a plan, not a load
/// contract. It is the shape one process hands another across the boot
/// bundle, so it is stated where that bundle is defined.
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    /// The compiled tokenizer's objects, `(name, bytes)`. `None` for a
    /// snapshot, whose tokenizer is a file beside the weights, and `None`
    /// too when only *some* of the objects were found — see
    /// `worker::weights`, which treats a partial one as absent.
    pub tokenizer: Option<Vec<(String, Vec<u8>)>>,
    /// The checkpoint's `config.json`, verbatim.
    pub config: Vec<u8>,
}

// =============================================================================
// The serving table
// =============================================================================

/// One shipping SKU, as a *serving* runtime sees it.
///
/// Three columns, and each is here because a serving process needs it before
/// it holds a plan:
///
/// * `layers` and `vocab` are computation facts — the tower's depth and the
///   `embed` table's leading extent — that the sampler and the PTIR lowering
///   are sized from at boot, when nothing in this process has traced
///   anything;
/// * `arch` is a deployment fact and no plan states one: a trace says what a
///   layer computes, not what a fleet calls the model. The vision and speech
///   front-ends dispatch on it.
///
/// # The pin that came off, named rather than hidden
///
/// `model/tests/rows_are_the_traces.rs` held `layers` and `vocab` equal to
/// what the row's own trace says, which is what made them a measurement
/// rather than a claim. That test died with `model::serve` and with
/// `model::deployment`, and this crate cannot restate it: reading a plan
/// means linking `model-dsl` and tracing at test time, which is a dependency
/// this crate does not have and a port this milestone is not. So the numbers
/// below are the ones the pin last held, and re-establishing the pin — here,
/// or back in `model` against a table it can see — is open work.
///
/// `max_model_len` STOOD as a fourth column and does not survive the move.
/// Nothing in this crate ever read it; the context ceiling a worker admits
/// comes off the driver's own capabilities (`DriverCapabilities::max_model_len`),
/// which is a different number from a different party. A column with no
/// reader is a fact this table cannot be held to.
pub struct Row {
    /// The SKU — a `::model::catalog()` row name, and the id every part of
    /// the tree spells. The driver identifies a checkpoint against
    /// `::model::imports()` and reports the SKU it matched; this is that
    /// string.
    pub id: &'static str,
    /// Transformer layers in the tower.
    pub layers: u32,
    /// The LOGITS width — the leading extent of the `embed` table, which is
    /// the dim the sampler operates on and the driver's recognizer table is
    /// keyed by.
    ///
    /// It may EXCEED the tokenizer's token count (qwen3: 151 936 logits vs
    /// 151 669 tokens). Sizing a sampler from the tokenizer instead is the
    /// vocab-padding device fault.
    pub vocab: u32,
    /// The architecture label a driver advertises and a control plane files
    /// this model under, and the string the vision and speech front-ends
    /// dispatch on.
    pub arch: &'static str,
}

/// Every SKU this build can serve, in `::model::catalog()` order.
///
/// One row per catalog entry, keyed by the SKU — which is the only id space
/// in the tree. The chat template is NOT a column: it is
/// [`::model::template::template_of`], keyed by the same string, so the two
/// tables cannot disagree about which model a build is formatting turns for.
pub const ROWS: &[Row] = &[
    Row {
        id: "dsv4-base-bf16-kv-bf16",
        layers: 6,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "dsv4-base-bf16-kv-bf16-tp2",
        layers: 6,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "gemma4-e4b-bf16-kv-bf16",
        layers: 42,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-31b-bf16-kv-bf16",
        layers: 60,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-31b-bf16-kv-bf16-tp2",
        layers: 60,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "glm5-a12b-bf16-bf16-kv-bf16",
        layers: 46,
        vocab: 151_552,
        arch: "glm_moe_dsa",
    },
    Row {
        id: "glm5-a12b-bf16-bf16-kv-bf16-tp2",
        layers: 46,
        vocab: 151_552,
        arch: "glm_moe_dsa",
    },
    Row {
        id: "gptoss-20b-bf16-mxfp4-kv-bf16",
        layers: 24,
        vocab: 201_088,
        arch: "gptoss",
    },
    Row {
        id: "gptoss-120b-bf16-mxfp4-kv-bf16",
        layers: 36,
        vocab: 201_088,
        arch: "gptoss",
    },
    Row {
        id: "gptoss-120b-bf16-mxfp4-kv-bf16-tp2",
        layers: 36,
        vocab: 201_088,
        arch: "gptoss",
    },
    Row {
        id: "kimik3-bf16-mxfp4-kv-bf16",
        layers: 8,
        vocab: 163_840,
        arch: "kimi_k3",
    },
    Row {
        id: "kimik3-bf16-mxfp4-kv-bf16-tp2",
        layers: 8,
        vocab: 163_840,
        arch: "kimi_k3",
    },
    // The one shipping SKU whose checkpoint publishes a draft head (palo C3).
    // FIRST among the qwen rows, as it is first in `::model::catalog()` — this
    // table is that one in order, and `the_catalog_and_the_serving_table_are_the_same_ids`
    // is what says so. `arch` is `qwen3_5` because that is what the SKU IS: a
    // qwen3_5 tower with fifteen `mtp.*` tensors on the end, not one trunk op
    // changed.
    Row {
        id: "qwen36-27b-bf16-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-a3b-bf16-kv-bf16",
        layers: 40,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-a3b-bf16-kv-bf16-tp2",
        layers: 40,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d3b-bf16-kv-bf16",
        layers: 24,
        vocab: 151_936,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d0.8b-bf16-kv-bf16",
        layers: 24,
        vocab: 248_320,
        arch: "qwen3_5",
    },
];

/// The row with this id, or `None` if this build ships no such model.
#[must_use]
pub fn row(id: &str) -> Option<&'static Row> {
    ROWS.iter().find(|row| row.id == id)
}

/// Every shipping id, in table order.
#[must_use]
pub fn ids() -> Vec<&'static str> {
    ROWS.iter().map(|row| row.id).collect()
}

/// The `take` ids closest to `id` by edit distance — what a refusal names so
/// a typo reads as a typo.
#[must_use]
pub fn nearest_ids(id: &str, take: usize) -> Vec<&'static str> {
    let mut scored: Vec<(usize, &'static str)> = ids()
        .into_iter()
        .map(|k| (edit_distance(id, k), k))
        .collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored.into_iter().take(take).map(|(_, k)| k).collect()
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Rebuild the compiled tokenizer an artifact carried, without touching the
/// filesystem. `None` when the model is a snapshot, whose tokenizer is a file.
///
/// Here rather than on [`ModelMetadata`] because it is *consumption*: the type
/// says what an artifact carries, and this says what this runtime does with
/// it.
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
/// It is the row now: [`ROWS`], the serving face of the same catalog the
/// driver identified the checkpoint against. Both sides knowing one fact
/// differently is not a bug that got fixed; it is a sentence that can no
/// longer be written.
fn loaded_row(model_id: &str) -> Result<&'static Row> {
    row(model_id).ok_or_else(|| {
        anyhow!(
            "the driver loaded {model_id:?}, which this build's model catalog \
             does not contain; nearest ids: {:?}",
            nearest_ids(model_id, 3)
        )
    })
}

pub fn register(
    name: String,
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
    let num_layers = row.layers;
    // `vocab` is the LOGICAL width and does not shard, so one number answers it
    // at any tensor-parallel width.
    //
    // IT USED TO BE A SECOND REFUSAL: the width came out of
    // `row.deployment(Deployed::single())`, which could fail, and this line
    // turned that failure into "the driver loaded X but this build refuses it".
    // Three of the fourteen rows do fail it — the MLA ones — and the refusal
    // they raise is about the PAGER ("this build provisions no MLA latent
    // store"), not about the model. Asking it HERE was asking the wrong party
    // twice: by the time `register` runs, the driver has already loaded the
    // model, so a second opinion on whether it can be loaded is either
    // redundant or wrong. The width is a fact about the row and is stated as
    // one.
    let vocab_size = row.vocab;
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
    // The id names a serving row, every SKU states a template, and there
    // is no arm left to fall through. An unknown id was already an error
    // above, at `loaded_row`, with the nearest known ids named — before
    // the first token rather than after a thousand plausible ones.
    //
    // The template is `model`'s to state and this crate's table does not
    // restate it — `::model::template::templates()` is keyed by the SAME
    // string that answered the two numbers, which is the property worth
    // having: a build cannot size its sampler from one model and format its
    // prompts for another. A SKU with no template row is a coverage hole in
    // `model`'s own `every_sku_ships_whole`, and it is named here rather than
    // answered with ChatML.
    let instruct = match ::model::template::template_of(row.id) {
        Some(make) => make(tokenizer.clone()),
        None => {
            return Err(anyhow!(
                "this build serves {:?} but ships no chat template for it; \
                 `model::template::templates()` has no row under that SKU",
                row.id
            ));
        }
    };

    // THE CLASSIFY COLUMN, off the same row. It is not optional and it has no
    // fallback: word 0 is the all-false class, so an engine that could not
    // find its SKU's classifier would not fail — it would fire every decode
    // lane through the prefill arm and return plausible garbage.
    let classify = ::model::classify_of(row.id).ok_or_else(|| {
        anyhow!(
            "this build serves {:?} but its model catalog states no classifier \
             for it; a lane's fact word cannot be computed",
            row.id
        )
    })?;

    let model = Arc::new(Model {
        name,
        arch_name: row.arch,
        instruct,
        classify,
        kv_page_size,
        rs_caps: rs,
        ptir_caps: ptir,
        tokenizer,
        vocab: OnceLock::new(),
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
    /// The family label the vision front-end and the speech front-end dispatch
    /// on (e.g. "gemma4", "qwen3_5"). The ROW's, like the chat template and
    /// the two numbers: the driver advertises the same string off the same
    /// row, so reading its copy only opened the way for a build to select a
    /// processor for one model and a template for another.
    arch_name: &'static str,
    instruct: Arc<dyn Instruct>,
    /// How this SKU sorts a request into the fact word its lanes carry — the
    /// catalog's fourth column, taken once at registration.
    ///
    /// IT IS A COLUMN OF THE ROW, like the template and the two numbers, and
    /// for the same reason: the fire path holds this handle and nothing else,
    /// and a build that classified its lanes for one model while tracing
    /// another would compose every fire out of windows the plan does not
    /// have. See [`Model::word`].
    classify: ::model::ClassifyFn,
    kv_page_size: u32,
    /// Recurrent-state (working-set) capabilities surfaced via model.wit
    /// (`rs-state-size`/`rs-buffer-page-size`/`rs-fold-granularity`). All
    /// 0/0/1 for pure-attention models.
    rs_caps: RsCaps,
    ptir_caps: PtirCaps,
    tokenizer: Arc<Tokenizer>,
    /// **THE TOKEN TABLE, BUILT ONCE PER ENGINE.**
    ///
    /// [`Model::get_vocabs`] walks `id_to_token` over the whole vocabulary and
    /// mints one `Vec<u8>` per token — 248 070 of them on qwen3.6, ~16 ms —
    /// and it used to do that on every guest call, which is once per LAUNCH
    /// for any inferlet that asks. An engine serves exactly one model and a
    /// model's tokenizer never changes, so the table is a constant and is
    /// treated as one; it is also what
    /// [`tokens_with_prefix`](Model::tokens_with_prefix) scans instead of
    /// shipping.
    ///
    /// Built lazily rather than at registration: a deployment whose guests
    /// never ask should not pay 16 ms and ~20 MiB for a table nobody reads.
    vocab: OnceLock<(Vec<u32>, Vec<Vec<u8>>)>,
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

    /// Gets the architecture identifier (e.g. "gemma4", "qwen3_5").
    pub fn arch_name(&self) -> &'static str {
        self.arch_name
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

    /// The whole vocabulary as parallel vectors of (token IDs, token bytes) —
    /// a COPY of the table built once by [`Model::vocab`].
    ///
    /// The copy is unavoidable: the WIT surface hands the guest an owned list.
    /// What is avoidable, and now avoided, is rebuilding the table from
    /// `id_to_token` on every call. Callers that want a few tokens' bytes or a
    /// prefix match should take [`token_bytes`](Model::token_bytes) or
    /// [`tokens_with_prefix`](Model::tokens_with_prefix) instead and copy
    /// nothing.
    pub fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.vocab().clone()
    }

    /// The token table, built on first ask and kept.
    fn vocab(&self) -> &(Vec<u32>, Vec<Vec<u8>>) {
        self.vocab.get_or_init(|| {
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
        })
    }

    /// The raw bytes each of `tokens` stands for, in order; an empty vector
    /// for an id the vocabulary does not hold.
    ///
    /// The same answer indexing a table built from [`get_vocabs`] gives, at
    /// the size of the QUESTION rather than the size of the vocabulary —
    /// which is the whole point (see [`tokens_with_prefix`]).
    ///
    /// [`get_vocabs`]: Model::get_vocabs
    /// [`tokens_with_prefix`]: Model::tokens_with_prefix
    pub fn token_bytes(&self, tokens: &[u32]) -> Vec<Vec<u8>> {
        tokens
            .iter()
            .map(|id| self.tokenizer.id_to_token(*id).unwrap_or_default())
            .collect()
    }

    /// Every token id whose bytes begin with `prefix`, ascending.
    ///
    /// **THIS IS THE QUERY THE GUESTS WERE SHIPPING A VOCABULARY TO ASK.**
    /// Token healing rolls a prompt back by a token or two and re-expands it
    /// under the mask of every token that reproduces the rolled-back BYTES as
    /// a prefix. It was building that mask guest-side, which meant lowering
    /// 248 070 records across the component boundary and looping over them
    /// twice — ~115 ms of a 158 ms per-launch constant (palo build log 23) to
    /// compute a set the host can name from a table it already holds.
    ///
    /// The set is exactly the one that loop produced: the ids the token table
    /// holds whose bytes start with `prefix`, and no others. An empty prefix
    /// matches the whole vocabulary, because every byte string starts with
    /// nothing — the caller that cares (a rollback that carried no bytes) has
    /// to say so itself, as it did before.
    /// It is the tokenizer's own table that is scanned, borrowed rather than
    /// copied — so this asks nothing of [`Model::vocab`]'s cache and a
    /// deployment whose guests only ever heal never builds one.
    pub fn tokens_with_prefix(&self, prefix: &[u8]) -> Vec<u32> {
        self.tokenizer.ids_with_prefix(prefix)
    }

    /// Gets the split regex pattern.
    pub fn get_split_regex(&self) -> String {
        self.tokenizer.get_split_regex()
    }

    /// Gets the special tokens.
    pub fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.tokenizer.get_special_tokens()
    }

    /// The fact word a lane carries: `query_len` rows, whether the lane states
    /// a custom attention mask of its own, whether it routes to an adapter
    /// bank, whether it wants the model's draft head run over its rows, and
    /// whether it wants its attention's mass kept.
    ///
    /// **THIS IS WHAT A FIRE IS COMPOSED FROM** (palo design §0). The driver
    /// turns a lane's word into a class, and the class into the row WINDOW
    /// every guarded node runs over — so `qo_one` decides whether a lane's
    /// rows go through the decode attention or the prefill one, and `masked`
    /// (gemma) whether they go through the mask-aware arm. Which bit is which
    /// is the model's own business and stays there: this calls the family's
    /// `Classify::of(..).word()` through the catalog pointer and never reads a
    /// bit.
    ///
    /// The engine states all five because it knows all five: a lane's rows are
    /// the tokens it submits, a custom mask is a lane's only once the fire's
    /// mask has lowered, and the adapter, the draft ask and the capture ask
    /// are what whoever built the lane put on it
    /// (`pipeline::fire::stamp_lane_words`).
    ///
    /// **THE LIST GROWS ONE ARGUMENT PER AXIS, AND SO DOES THE REFUSAL SET**
    /// (palo C3b/C4b). `drafts` and `captures_scores` join on `adapter`'s
    /// terms exactly: the shell asks per lane whether the class the word
    /// resolved to runs the export's arm, and refuses BOTH directions by name
    /// — `driver_cuda::Fault::DraftWord` and `Fault::ScoreWord`. What is
    /// different is that these two carry no runtime input, so the wrong answer
    /// they prevent is not "staged and never read" but "computed and nobody
    /// told": a drafting class runs a whole transformer block into a column
    /// no reader collects, and a capturing class writes a mass column the
    /// readout skips. Stamping all five from ONE reading of ONE lane at one
    /// instant is what makes the refusals unreachable from this path.
    ///
    /// **`adapter` HAS TO AGREE WITH `Lane::adapter`, AND THE SHELL CHECKS
    /// THAT IT DOES.** The word puts the lane's rows inside the correction's
    /// window or outside it (palo design §8); the id is what the arm routes
    /// with. A word that said `has_adapter` with no id behind it would send
    /// the arm at a routes vector nobody staged, and an id with no word would
    /// be staged and never read — `driver_cuda::Fault::AdapterWord` refuses
    /// both, by name, before anything launches. So the two are stamped from
    /// ONE reading of the lane, at one instant, which is what
    /// `stamp_lane_words` is.
    #[must_use]
    pub fn word(
        &self,
        query_len: u32,
        custom_mask: bool,
        adapter: bool,
        drafts: bool,
        captures_scores: bool,
    ) -> u64 {
        (self.classify)(
            &::model::Request::new(query_len, custom_mask)
                .adapted(adapter)
                .drafting(drafts)
                .capturing_scores(captures_scores),
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_is_found_by_its_own_id() {
        for r in ROWS {
            assert_eq!(row(r.id).map(|found| found.id), Some(r.id));
        }
    }

    #[test]
    fn ids_are_unique() {
        let mut seen = ids();
        let before = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), before, "two rows share an id");
    }

    /// A typo is answered with the row it is a typo OF — the property the
    /// engine's "the driver loaded X, nearest ids: .." refusal rests on.
    #[test]
    fn an_unknown_id_is_none_and_names_its_near_misses() {
        assert!(row("qwen35-d0.8b-bf16-kv-bf1").is_none());
        assert_eq!(
            nearest_ids("qwen35-d0.8b-bf16-kv-bf1", 1),
            vec!["qwen35-d0.8b-bf16-kv-bf16"]
        );
        assert!(row("gptoss-21b-bf16-mxfp4-kv-bf16").is_none());
        assert_eq!(
            nearest_ids("gptoss-21b-bf16-mxfp4-kv-bf16", 1),
            vec!["gptoss-20b-bf16-mxfp4-kv-bf16"]
        );
    }

    /// This table and `model`'s are ONE id space.
    ///
    /// It is what is left of `model/tests/rows_are_the_traces.rs` after M18:
    /// the numbers can no longer be held against a trace from here, but the
    /// KEYS can be held against the catalog the driver identifies checkpoints
    /// against. A catalog SKU with no serving row is a model a driver can load
    /// and this engine cannot name; a serving row with no catalog SKU is two
    /// numbers for a model nothing traces.
    #[test]
    fn the_catalog_and_the_serving_table_are_the_same_ids() {
        let mut catalog: Vec<&str> = ::model::catalog().into_iter().map(|(id, ..)| id).collect();
        let mut serving = ids();
        catalog.sort_unstable();
        serving.sort_unstable();
        assert_eq!(catalog, serving);
    }

    /// Every serving row has the template `register` will ask for.
    ///
    /// `register` refuses a SKU with no template rather than falling through
    /// to ChatML, and this is what keeps that refusal unreachable for a model
    /// this build actually ships.
    #[test]
    fn every_serving_row_ships_a_chat_template() {
        for r in ROWS {
            assert!(
                ::model::template::template_of(r.id).is_some(),
                "`{}` is a serving row with no chat template",
                r.id
            );
        }
    }
}
