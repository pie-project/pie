//! The serving half: what a runtime needs to know about the model it serves
//! that is **not** its forward pass.
//!
//! The rest of this crate answers "what computation is this model" — a text
//! traces a [`Plan`](model_dsl::Plan) and an import table says how the
//! checkpoint's bytes become its weights. None of that is what an `engine`
//! process asks. It asks four questions, and they are all about *serving*:
//!
//! * how do I format a turn of conversation for this model, and how do I read
//!   its tokens back as text ([`instruct`], and the templates behind
//!   [`Row::chat`]);
//! * how wide is the logits row the sampler operates on, and how many layers
//!   deep is the tower ([`Row::vocab`], [`Row::layers`]);
//! * how do I turn an image or a clip into the tokens and the geometry the
//!   model was trained on (`multimodal`, behind the `serve` feature);
//! * what did the artifact this process was handed actually carry
//!   ([`ModelMetadata`], [`encoding`]).
//!
//! # Why it is here and not in `model-legacy`
//!
//! It was there, under a `chat` feature, and `engine` was the only consumer
//! that ever enabled it. Not one line of it touched the DSL, the catalog facts
//! or the load contract: the templates read a `Tokenizer` and return token ids,
//! `multimodal` reads `image` and returns a grid. It rode along in that crate
//! because that crate was, once, the only place a `Tokenizer` and a chat
//! template could meet — and it was the single reason `engine` and `worker`
//! linked the legacy declarations at all.
//!
//! So the R1 cutover moved it whole rather than porting it: same code, same
//! behaviour, one namespace over. What changed is the shape of the thing it
//! reads.
//!
//! # `ROWS` is the catalog's serving face, and there is no other catalog
//!
//! `model-legacy`'s `catalog::Variant` answered seven questions: an id, a
//! manifest, a load shape, a deployment, an authoring pass, a trace, a chat
//! template. R3 deleted that crate. What is left is this table and
//! [`crate::catalog`]: the table states the four things a serving process
//! asks that no computation states — the SKU, the architecture label, the
//! context ceiling, and the chat template — and everything else is read off
//! the SKU's own traced plan ([`crate::deployment::Deployment::of`]).
//!
//! The two numbers that ARE also computation facts, `layers` and `vocab`,
//! are stated here because `engine` and `worker` size a sampler and a
//! recognizer without ever holding a plan. They are not a second opinion:
//! `tests/rows_are_the_traces.rs` asserts each one equal to what the row's
//! own trace says, which is the pin `model-legacy/tests/serve_rows.rs` used
//! to hold against the OTHER catalog and now holds against this crate's.
//!
//! THE IDS ARE THE SKU. They used to be the legacy catalog's spelling
//! (`"qwen3.5-35b-a3b"`), because that is what a driver reported having
//! loaded, and `driver-cuda`'s `baker::sku_for` BRIDGE existed to translate.
//! R3 collapsed the two id spaces into one: a driver identifies a checkpoint
//! against [`crate::imports`] and reports the SKU it matched, the engine
//! looks that SKU up here, and the bridge is deleted.

use std::sync::Arc;

use tokenizer::Tokenizer;

pub mod encoding;
pub mod instruct;
/// Host-side image and clip decode. The one part of this module that needs a
/// codec, and so the one part behind the `serve` feature: a driver links this
/// crate for its catalog and must not link twenty image crates to get one.
#[cfg(feature = "serve")]
pub mod multimodal;

mod chatml;
mod decoders;
mod deepseek;
mod gemma;
mod gpt_oss;
mod kimi;
mod metadata;

pub use instruct::Instruct;
pub use metadata::ModelMetadata;

/// One shipping SKU, as a *serving* runtime sees it.
pub struct Row {
    /// The SKU — a `crate::catalog()` row name, and the id every part of the
    /// tree now spells (see the module doc on whose spelling this is).
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
    /// this model under. A DEPLOYMENT fact: no trace states it, because a
    /// trace says what a layer computes and not what a fleet calls it.
    pub arch: &'static str,
    /// The context ceiling this deployment admits, or `0` for a row whose
    /// legacy spec stated none. Also a deployment fact, and also stated
    /// nowhere in a plan.
    pub max_model_len: u32,
    /// How this model's turns are written and read back.
    ///
    /// A `fn` and not a `dyn` because a template needs the process's
    /// tokenizer, which the table cannot hold: rows are `const`, tokenizers
    /// are loaded.
    template: fn(Arc<Tokenizer>) -> Arc<dyn Instruct>,
}

impl Row {
    /// The chat template for this row, bound to the served tokenizer.
    #[must_use]
    pub fn chat(&self, tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
        (self.template)(tokenizer)
    }
}

fn qwen_chatml(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(chatml::QwenInstruct::new(tokenizer, chatml::QWEN_CHATML))
}

fn glm_chatml(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(chatml::QwenInstruct::new(tokenizer, chatml::GLM_CHATML))
}

fn gemma4(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(gemma::Gemma4Instruct::new(tokenizer))
}

fn gpt_oss(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(gpt_oss::GptOssInstruct::new(tokenizer))
}

fn kimi(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(kimi::KimiInstruct::new(tokenizer))
}

fn deepseek_r1(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(deepseek::R1Instruct::new(tokenizer))
}

/// Every SKU this build can serve, in catalog order.
///
/// One row per `crate::catalog()` entry that has a chat template, keyed by
/// the SKU — which is now the ONLY id space in the tree. Every number is
/// asserted equal to the SKU's own traced [`Plan`](model_ir::plan::Plan) by
/// `tests/rows_are_the_traces.rs`.
pub const ROWS: &[Row] = &[
    Row {
        id: "dsv4-base-bf16-kv-bf16",
        layers: 6,
        vocab: 129_280,
        arch: "deepseek_v4",
        max_model_len: 0,
        template: deepseek_r1,
    },
    Row {
        id: "dsv4-base-bf16-kv-bf16-tp2",
        layers: 6,
        vocab: 129_280,
        arch: "deepseek_v4",
        max_model_len: 0,
        template: deepseek_r1,
    },
    Row {
        id: "gemma4-e4b-bf16-kv-bf16",
        layers: 42,
        vocab: 262_144,
        arch: "gemma4",
        max_model_len: 131_072,
        template: gemma4,
    },
    Row {
        id: "gemma4-31b-bf16-kv-bf16",
        layers: 60,
        vocab: 262_144,
        arch: "gemma4",
        max_model_len: 262_144,
        template: gemma4,
    },
    Row {
        id: "gemma4-31b-bf16-kv-bf16-tp2",
        layers: 60,
        vocab: 262_144,
        arch: "gemma4",
        max_model_len: 262_144,
        template: gemma4,
    },
    Row {
        id: "glm5-a12b-bf16-bf16-kv-bf16",
        layers: 46,
        vocab: 151_552,
        arch: "glm_moe_dsa",
        max_model_len: 0,
        template: glm_chatml,
    },
    Row {
        id: "glm5-a12b-bf16-bf16-kv-bf16-tp2",
        layers: 46,
        vocab: 151_552,
        arch: "glm_moe_dsa",
        max_model_len: 0,
        template: glm_chatml,
    },
    Row {
        id: "gptoss-20b-bf16-mxfp4-kv-bf16",
        layers: 24,
        vocab: 201_088,
        arch: "gptoss",
        max_model_len: 131_072,
        template: gpt_oss,
    },
    Row {
        id: "gptoss-120b-bf16-mxfp4-kv-bf16",
        layers: 36,
        vocab: 201_088,
        arch: "gptoss",
        max_model_len: 131_072,
        template: gpt_oss,
    },
    Row {
        id: "gptoss-120b-bf16-mxfp4-kv-bf16-tp2",
        layers: 36,
        vocab: 201_088,
        arch: "gptoss",
        max_model_len: 131_072,
        template: gpt_oss,
    },
    Row {
        id: "kimik3-bf16-mxfp4-kv-bf16",
        layers: 8,
        vocab: 163_840,
        arch: "kimi_k3",
        max_model_len: 0,
        template: kimi,
    },
    Row {
        id: "kimik3-bf16-mxfp4-kv-bf16-tp2",
        layers: 8,
        vocab: 163_840,
        arch: "kimi_k3",
        max_model_len: 0,
        template: kimi,
    },
    Row {
        id: "qwen35-a3b-bf16-kv-bf16",
        layers: 40,
        vocab: 248_320,
        arch: "qwen3_5",
        max_model_len: 262_144,
        template: qwen_chatml,
    },
    Row {
        id: "qwen35-a3b-bf16-kv-bf16-tp2",
        layers: 40,
        vocab: 248_320,
        arch: "qwen3_5",
        max_model_len: 262_144,
        template: qwen_chatml,
    },
    Row {
        id: "qwen35-d3b-bf16-kv-bf16",
        layers: 24,
        vocab: 151_936,
        arch: "qwen3_5",
        max_model_len: 262_144,
        template: qwen_chatml,
    },
    Row {
        id: "qwen35-d0.8b-bf16-kv-bf16",
        layers: 24,
        vocab: 248_320,
        arch: "qwen3_5",
        max_model_len: 262_144,
        template: qwen_chatml,
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
}
