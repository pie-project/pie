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
//!   model was trained on ([`multimodal`]);
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
//! # `ROWS` is a projection, not a second catalog
//!
//! `model-legacy`'s `catalog::Variant` answered seven questions: an id, a
//! manifest, a load shape, a deployment, an authoring pass, a trace, a chat
//! template. `engine` asked three of them — layers, logits width, template —
//! and the other four are precisely what the baker path replaces. [`Row`]
//! states those three and nothing else, which is why it is a flat `const` table
//! rather than a trait: three columns per shipping id, no deployment algebra to
//! run and nothing to refuse.
//!
//! The numbers are MEASURED, not transcribed: every column is asserted equal to
//! the legacy catalog's own answer by `model-legacy`'s `serve_rows_agree` test,
//! which links both crates and dies with the legacy one.
//!
//! THE IDS ARE THE DRIVER'S SPELLING. `engine::model::register` is handed the
//! id the driver reported loading, and today every driver reports the legacy
//! catalog's spelling (`"qwen3.5-35b-a3b"`), not the baker SKU's
//! (`"qwen35-a3b-bf16-kv-bf16"`) — `driver-cuda`'s `baker::sku_for` BRIDGE
//! exists exactly because the two spellings differ. This table is therefore
//! keyed the way the wire is keyed. When R2/R3 collapses the two id spaces,
//! this table's keys move with the wire and its columns come off
//! [`crate::catalog`], which states the same numbers already.

use std::sync::Arc;

use tokenizer::Tokenizer;

pub mod encoding;
pub mod instruct;
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

/// One shipping model, as a *serving* runtime sees it.
pub struct Row {
    /// The id the driver reports having loaded (see the module doc on
    /// whose spelling this is).
    pub id: &'static str,
    /// Transformer layers in the tower.
    pub layers: u32,
    /// The LOGITS width — `config.json`'s `vocab_size`, which is the dim the
    /// sampler operates on and the driver's recognizer table is keyed by.
    ///
    /// It may EXCEED the tokenizer's token count (qwen3: 151 936 logits vs
    /// 151 669 tokens). Sizing a sampler from the tokenizer instead is the
    /// vocab-padding device fault.
    pub vocab: u32,
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

/// Every id this build can serve, in catalog order.
pub const ROWS: &[Row] = &[
    Row {
        id: "qwen3.5-0.8b-base",
        layers: 24,
        vocab: 248_320,
        template: qwen_chatml,
    },
    Row {
        id: "qwen3.5-4b",
        layers: 32,
        vocab: 248_320,
        template: qwen_chatml,
    },
    Row {
        id: "qwen3.5-9b",
        layers: 32,
        vocab: 248_320,
        template: qwen_chatml,
    },
    Row {
        id: "qwen3.5-35b-a3b",
        layers: 40,
        vocab: 248_320,
        template: qwen_chatml,
    },
    Row {
        id: "qwen3.6-27b",
        layers: 64,
        vocab: 248_320,
        template: qwen_chatml,
    },
    Row {
        id: "gemma-4-e2b",
        layers: 35,
        vocab: 262_144,
        template: gemma4,
    },
    Row {
        id: "gemma-4-e4b",
        layers: 42,
        vocab: 262_144,
        template: gemma4,
    },
    Row {
        id: "gemma-4-31b",
        layers: 60,
        vocab: 262_144,
        template: gemma4,
    },
    Row {
        id: "gemma-4-26b-a4b",
        layers: 30,
        vocab: 262_144,
        template: gemma4,
    },
    Row {
        id: "glm-5-106b-a12b",
        layers: 46,
        vocab: 151_552,
        template: glm_chatml,
    },
    Row {
        id: "gpt-oss-20b",
        layers: 24,
        vocab: 201_088,
        template: gpt_oss,
    },
    Row {
        id: "gpt-oss-120b",
        layers: 36,
        vocab: 201_088,
        template: gpt_oss,
    },
    Row {
        id: "kimi-k3",
        layers: 8,
        vocab: 163_840,
        template: kimi,
    },
    Row {
        id: "deepseek-v4",
        layers: 6,
        vocab: 129_280,
        template: deepseek_r1,
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
        assert!(row("qwen3.5-0.8b-bas").is_none());
        assert_eq!(
            nearest_ids("qwen3.5-0.8b-bas", 1),
            vec!["qwen3.5-0.8b-base"]
        );
        assert!(row("gpt-oss-21b").is_none());
        assert_eq!(nearest_ids("gpt-oss-21b", 1), vec!["gpt-oss-20b"]);
    }
}
