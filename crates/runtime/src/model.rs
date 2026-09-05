//! The served model: the runtime's global model/tokenizer cache, and the
//! serving table ([`ROWS`], [`ModelMetadata`]) it is built from. Set once at
//! bootstrap and read from everywhere after.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow};

use models::template::Instruct;
use tokenizer::Tokenizer;

/// The single model this runtime serves. Set once at bootstrap.
static MODEL: OnceLock<Arc<Model>> = OnceLock::new();

/// The compiled metadata a `.zt` artifact carries, lifted once by the worker.
///
/// The checkpoint's `config.json` bytes, verbatim, and — for an artifact,
/// absent for a snapshot — the compiled tokenizer objects for `pie.tokenizer/1`.
#[derive(Clone, Debug)]
pub struct ModelMetadata {
    /// `(name, bytes)` per compiled tokenizer object. `None` for a snapshot
    /// (tokenizer is a file beside the weights) or when only some objects
    /// were found.
    pub tokenizer: Option<Vec<(String, Vec<u8>)>>,
    /// The checkpoint's `config.json`, verbatim.
    pub config: Vec<u8>,
}

/// One shipping SKU, as a *serving* runtime sees it.
///
/// `layers`/`vocab` size the sampler and ETA lowering at boot, before a plan
/// is traced; `arch` is the label the vision/speech front-ends dispatch on.
pub struct Row {
    /// The SKU — a `models::skus()` row name and the id every part of the
    /// tree spells.
    pub id: &'static str,
    /// Transformer layers in the tower.
    pub layers: u32,
    /// Logits width: the leading extent of the `embed` table. May exceed the
    /// tokenizer's token count (qwen3: 151 936 logits vs 151 669 tokens) —
    /// size a sampler from this, not the tokenizer vocab.
    pub vocab: u32,
    /// Architecture label the vision/speech front-ends dispatch on.
    pub arch: &'static str,
}

/// Every SKU this build can serve, in `models::skus()` order.
///
/// The chat template is not a column here: it is
/// [`models::template::template_of`], keyed by the same SKU string.
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
        id: "dsv4-flash-bf16-kv-bf16",
        layers: 43,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    // Mini snapshot: 5 layers (`num_hidden_layers: 5`, renumbered from 0,1,2,3,42).
    Row {
        id: "dsv4-flash-u4g64-u2g64-kv-bf16",
        layers: 5,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    // Full checkpoint the row above is a mini carve of, at its own 43 layers.
    Row {
        id: "dsv4-flash-mtp-u4g64-u2g64-mxfp4-kv-bf16",
        layers: 5,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "dsv4-flash-full-mtp-u4g64-u2g64-mxfp4-kv-bf16",
        layers: 43,
        vocab: 129_280,
        arch: "deepseek_v4",
    },
    Row {
        id: "dsv4-flash-full-u4g64-u2g64-kv-bf16",
        layers: 43,
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
        id: "gemma4-e4b-eagle-bf16-kv-bf16",
        layers: 42,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-e4b-vision-bf16-kv-bf16",
        layers: 42,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-26b-a4b-u4g64-kv-bf16",
        layers: 30,
        vocab: 262_144,
        arch: "gemma4",
    },
    // The same mixture with Google's assistant drafter overlaid.
    Row {
        id: "gemma4-26b-a4b-mtp-u4g64-kv-bf16",
        layers: 30,
        vocab: 262_144,
        arch: "gemma4",
    },
    // Same trunk as its already-listed twin; neither quant nor the vision
    // tower moves layers, vocab or arch.
    Row {
        id: "gemma4-26b-a4b-vision-u4g64-kv-bf16",
        layers: 30,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-31b-bf16-kv-bf16",
        layers: 60,
        vocab: 262_144,
        arch: "gemma4",
    },
    // mlx 4-bit: same trunk geometry as its bf16 sibling.
    Row {
        id: "gemma4-31b-mtp-u4g64-kv-bf16",
        layers: 60,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-31b-u4g64-kv-bf16",
        layers: 60,
        vocab: 262_144,
        arch: "gemma4",
    },
    Row {
        id: "gemma4-31b-vision-u4g64-kv-bf16",
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
        id: "glm5-a12b-bf16-kv-bf16",
        layers: 46,
        vocab: 151_552,
        arch: "glm_moe_dsa",
    },
    Row {
        id: "glm5-a12b-bf16-kv-bf16-tp2",
        layers: 46,
        vocab: 151_552,
        arch: "glm_moe_dsa",
    },
    // GLM-5.3-Flash: 45 layers, the KDA/DSA cadence under mHC, text only
    // (`model.visual.*` and the one `mtp` block unread).
    Row {
        id: "glm53-flash-mtp-u8g64-u2g64-u4g64-kv-bf16",
        layers: 45,
        vocab: 154_880,
        arch: "glm5_next",
    },
    Row {
        id: "glm53-flash-mtp-vision-u8g64-u2g64-u4g64-kv-bf16",
        layers: 45,
        vocab: 154_880,
        arch: "glm5_next",
    },
    Row {
        id: "glm53-flash-vision-u8g64-u2g64-kv-bf16",
        layers: 45,
        vocab: 154_880,
        arch: "glm5_next",
    },
    Row {
        id: "glm53-flash-u8g64-u2g64-kv-bf16",
        layers: 45,
        vocab: 154_880,
        arch: "glm5_next",
    },
    Row {
        id: "glm53-flash-mtp-vision-u4g64-u2g64-u4g64-kv-bf16",
        layers: 45,
        vocab: 154_880,
        arch: "glm5_next",
    },
    Row {
        id: "gptoss-20b-bf16-mxfp4-kv-bf16",
        layers: 24,
        vocab: 201_088,
        arch: "gptoss",
    },
    // mlx 4-bit: same trunk geometry as its bf16 sibling.
    Row {
        id: "gptoss-20b-u4g64-mxfp4-kv-bf16",
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
    // Ships a draft head (fifteen `mtp.*` tensors); arch stays qwen3_5 since
    // the trunk is unchanged.
    Row {
        id: "qwen36-27b-bf16-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    // 3.8 checkpoint is 3.6's artifact tensor-for-tensor; only the chat
    // template and seven reserved specials differ, not these numbers.
    Row {
        id: "qwen38-27b-bf16-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    // mlx 4-bit rows: same trunk geometry as their bf16 siblings.
    Row {
        id: "qwen36-27b-mtp-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    // The same trunk with a DFlash BLOCK drafter overlaid by `--aux`; the
    // drafter's own five layers are not the trunk's and are not counted.
    Row {
        id: "qwen36-27b-dflash-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-27b-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    // The same trunk with the DFlash2 block drafter overlaid by `--aux`.
    Row {
        id: "qwen38-27b-dflash2-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen38-27b-mtp-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen38-27b-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    // From the 4-bit artifact's own text_config (num_hidden_layers: 40,
    // vocab_size: 248320) — the qwen35-a3b geometry.
    Row {
        id: "qwen36-35b-a3b-mtp-u4g64-kv-bf16",
        layers: 40,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-35b-a3b-u4g64-kv-bf16",
        layers: 40,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    // `mini-l5-e16-k8` carve of that artifact: 5 of its 40 layers, vocab whole.
    Row {
        id: "qwen36-35b-a3b-mini-u4g64-kv-bf16",
        layers: 5,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    // `mini-l5-e64-k8` twin: differs only in routed-bank width, which this
    // table doesn't carry, so it reads identically to the row above.
    Row {
        id: "qwen36-35b-a3b-mini64-u4g64-kv-bf16",
        layers: 5,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d0.8b-u4g64-kv-bf16",
        layers: 24,
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
    // qwen4 hybrid: geometry off `Model::flash`'s own Dims; arch is the
    // checkpoint's `model_type: qwen4_exp`.
    Row {
        id: "qwen38-flash-next-u4g64-kv-bf16",
        layers: 48,
        vocab: 248_320,
        arch: "qwen4_exp",
    },
    Row {
        id: "qwen38-flash-next-bf16-kv-bf16",
        layers: 48,
        vocab: 248_320,
        arch: "qwen4_exp",
    },
    // Mini 2-bit snapshot: 4 layers (`num_hidden_layers: 4`,
    // layer_types linear/linear/linear/full).
    Row {
        id: "qwen38-flash-next-u4g64-u2g128-kv-bf16",
        layers: 4,
        vocab: 248_320,
        arch: "qwen4_exp",
    },
    // Shipped 2-bit artifact: the mini's parent, 48 layers again.
    Row {
        id: "qwen38-flash-next-full-u4g64-u2g128-kv-bf16",
        layers: 48,
        vocab: 248_320,
        arch: "qwen4_exp",
    },
    Row {
        id: "qwen38-flash-next-full-mtp-u4g64-u2g128-kv-bf16",
        layers: 48,
        vocab: 248_320,
        arch: "qwen4_exp",
    },
    Row {
        id: "qwen38-flash-next-full-mtp-vision-u4g64-u2g128-kv-bf16",
        layers: 48,
        vocab: 248_320,
        arch: "qwen4_exp",
    },
    Row {
        id: "qwen38-flash-next-full-vision-u4g64-u2g128-kv-bf16",
        layers: 48,
        vocab: 248_320,
        arch: "qwen4_exp",
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
    // Overlaid draft head: a second readout of the same 24 trunk layers, not
    // a second model, so the numbers are the trunk's.
    Row {
        id: "qwen35-d0.8b-vision-eagle-bf16-kv-bf16",
        layers: 24,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d0.8b-vision-bf16-kv-bf16",
        layers: 24,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d0.8b-vision-u4g64-kv-bf16",
        layers: 24,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-27b-vision-bf16-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen36-27b-vision-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen38-27b-vision-bf16-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen38-27b-vision-u4g64-kv-bf16",
        layers: 64,
        vocab: 248_320,
        arch: "qwen3_5",
    },
    Row {
        id: "qwen35-d0.8b-eagle-bf16-kv-bf16",
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

/// The row the engine loaded, or a refusal naming what is close.
fn loaded_row(model_id: &str) -> Result<&'static Row> {
    row(model_id).ok_or_else(|| {
        anyhow!(
            "the engine loaded {model_id:?}, which this build's model catalog \
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
    eta: EtaCaps,
    tokenizer_path: PathBuf,
    metadata: &ModelMetadata,
) -> Result<()> {
    // Logits dim = the model's `vocab_size`, keyed by the engine's recognizer
    // table — not the tokenizer token count, which may be smaller (qwen3:
    // 151669 vs 151936).
    let row = loaded_row(model_id)?;
    let num_layers = row.layers;
    // `vocab` is the logical width and does not shard, so one number answers
    // it at any tensor-parallel width.
    let vocab_size = row.vocab;
    // The tokenizer is the half that genuinely differs: compiled objects from
    // an artifact, a file beside a snapshot.
    let tokenizer = match compiled_tokenizer(metadata) {
        Some(compiled) => compiled?,
        None => Tokenizer::from_file(&tokenizer_path)?,
    };
    let tokenizer = Arc::new(tokenizer);
    // Verify the row's tokenizer contract (stop markers, media delimiters,
    // pinned specials) against the artifact's tokenizer before the template
    // resolves a marker, so a mismatched artifact refuses at boot.
    match models::tokenizer::contract_of(row.id) {
        Some(contract) => contract.verify(&tokenizer).map_err(|fault| {
            anyhow!("`{}` refuses this artifact's tokenizer: {fault}", row.id)
        })?,
        None => {
            return Err(anyhow!(
                "this build serves {:?} but ships no tokenizer contract for \
                 it; `models::tokenizer::contracts()` has no row under that \
                 SKU",
                row.id
            ));
        }
    }
    // Chat template chosen by the row the engine loaded. No fallback to
    // ChatML: an unrecognized SKU is refused above at `loaded_row`.
    let instruct = match models::template::template_of(row.id) {
        Some(make) => make(tokenizer.clone()),
        None => {
            return Err(anyhow!(
                "this build serves {:?} but ships no chat template for it; \
                 `models::template::templates()` has no row under that SKU",
                row.id
            ));
        }
    };

    // Classify column, off the same row: no fallback, since word 0 is the
    // all-false class and would silently fire every decode lane as prefill.
    let classify = models::sku(row.id).map(|sku| sku.classify).ok_or_else(|| {
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
        eta_caps: eta,
        tokenizer,
        vocab: OnceLock::new(),
        vocab_size,
        num_layers,
    });
    MODEL.set(model).map_err(|_| {
        anyhow!("a model is already registered; the runtime serves exactly one model")
    })?;
    Ok(())
}

/// Returns the single registered model. Panics if called before bootstrap
/// registers the model.
pub fn model() -> &'static Arc<Model> {
    MODEL.get().expect("model accessed before registration")
}

/// The reserved token id this model spells a media run's placeholder with,
/// or `None` for a model with no media front-end. Cached once (one process
/// serves one model).
pub fn media_pad() -> Option<u32> {
    static PAD: OnceLock<Option<u32>> = OnceLock::new();
    *PAD.get_or_init(|| {
        use crate::inferlet::host::media::multimodal;
        let m = model();
        let arch = m.arch_name();
        // The pad spelling is the front-end's own; read from it directly
        // rather than a second table that can drift out of sync.
        let spelling = models::media::vision_front_end(arch)
            .map(|fe| fe.delimiters().placeholder)
            .or_else(|| multimodal::audio_arch_supported(arch).then(multimodal::audio_placeholder))?;
        match m.tokenize(spelling)[..] {
            [id] => Some(id),
            // Tokenizer can't spell the arch's pad as one token: nothing to
            // look for.
            _ => None,
        }
    })
}

pub struct Model {
    name: String,
    /// Family label the vision/speech front-ends dispatch on (e.g. "gemma4",
    /// "qwen3_5"). Taken from the row, matching what the engine advertises.
    arch_name: &'static str,
    instruct: Arc<dyn Instruct>,
    /// How this SKU sorts a request into the fact word its lanes carry.
    /// Taken from the row once at registration. See [`Model::word`].
    classify: models::ClassifyFn,
    kv_page_size: u32,
    /// Recurrent-state (working-set) capabilities surfaced via model.wit
    /// (`rs-state-size`/`rs-buffer-page-size`/`rs-fold-granularity`). All
    /// 0/0/1 for pure-attention models.
    rs_caps: RsCaps,
    eta_caps: EtaCaps,
    tokenizer: Arc<Tokenizer>,
    /// Token table, built lazily on first ask and kept: a model's tokenizer
    /// never changes, and a deployment whose guests never ask should not pay
    /// to build one.
    vocab: OnceLock<(Vec<u32>, Vec<Vec<u8>>)>,
    /// Logits/output vocab dimension (= hf_config.vocab_size from the model's
    /// config.json). May EXCEED tokenizer.vocab_size() due to padding — use
    /// THIS for sampler lowering / logits-shaped ops, NOT the tokenizer vocab.
    vocab_size: u32,
    /// Transformer layer count from the model snapshot's config.json.
    num_layers: u32,
}

/// RS (recurrent-state) working-set capabilities surfaced to inferlets via
/// `model.wit`. Sourced from the engine handshake `EngineCapabilities` at
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

/// Model-gated values that a loaded backend can bind into ETA programs.
#[derive(Debug, Clone, Copy, Default)]
pub struct EtaCaps {
    pub has_mtp_logits: bool,
    /// The draft head's chain depth; zero without one (`mtp-depth`).
    pub mtp_depth: u32,
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

    /// Logits/output vocab dimension (= hf_config.vocab_size). May exceed the
    /// tokenizer's vocab (qwen3: 151936 logits vs 151669 tokens) — use this
    /// for sampler lowering / logits-shaped ops, not tokenizer vocab.
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
    /// a copy of the table built once by [`Model::vocab`]. Callers that want
    /// a few tokens' bytes or a prefix match should use
    /// [`token_bytes`](Model::token_bytes) or
    /// [`tokens_with_prefix`](Model::tokens_with_prefix) instead.
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
    pub fn token_bytes(&self, tokens: &[u32]) -> Vec<Vec<u8>> {
        tokens
            .iter()
            .map(|id| self.tokenizer.id_to_token(*id).unwrap_or_default())
            .collect()
    }

    /// Every token id whose bytes begin with `prefix`, ascending. Used for
    /// token healing (mask of tokens that reproduce rolled-back bytes as a
    /// prefix). An empty prefix matches the whole vocabulary. Scans the
    /// tokenizer's own table directly, so this does not build
    /// [`Model::vocab`]'s cache.
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

    /// The fact word a lane carries: `query_len` rows, custom mask,
    /// adapter routing, draft head, attention mass capture, media spans, and
    /// whether the rows are a block drafter's proposal rather than the
    /// sequence's own.
    /// The engine turns this word into a class, and the class into the row
    /// window every guarded node runs over. This calls the family's
    /// `Classify::of(..).word()` through the catalog pointer and never
    /// reads a bit itself; all seven facts are stamped from one reading of
    /// the lane at one instant.
    #[must_use]
    pub fn word(
        &self,
        query_len: u32,
        custom_mask: bool,
        adapter: bool,
        drafts: bool,
        captures_scores: bool,
        media: bool,
        block_draft: bool,
    ) -> u64 {
        (self.classify)(
            &models::Request::new(query_len, custom_mask)
                .adapted(adapter)
                .drafting(drafts)
                .capturing_scores(captures_scores)
                .with_media(media)
                .drafting_a_block(block_draft),
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

    pub fn eta_caps(&self) -> EtaCaps {
        self.eta_caps
    }
}

