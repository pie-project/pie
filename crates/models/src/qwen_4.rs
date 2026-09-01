//! The qwen4 family — `Qwen3.8-Flash-Next`, `model_type: qwen4_exp`: the
//! qwen_3 hybrid under a gated residual, with the hashed n-gram PLE beside
//! it. `model.rs` says what one is; this file says which ones ship.
//!
//! **THE TEMPLATE AND THE TOKENIZER ARE THE 3.8 GENERATION'S OWN** — the
//! artifact's `tokenizer_config.json` publishes the audio/tts specials
//! (248070–248076) at the ids `qwen_3::tokenizer::CONTRACT_38` pins, and the
//! same interleaved-thinking chat template the qwen38 twins serve — so both
//! registries point across the family line rather than restating either.

pub mod forward;
pub mod import;
pub mod model;

use model::{Mix, Model};
use model_dsl::Dtype;

use crate::qwen_3::{template, tokenizer};

pub const CATALOG: &[crate::Row] = model_dsl::catalog![
    // The mlxu4 row leads for `qwen_3`'s reason: its every projection is a
    // triplet, so it misses on a bf16 artifact where the bf16 row would
    // claim a 4-bit one and fail four stages later.
    (
        "qwen38-flash-mlxu4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::flash(Dtype::U4g64, Dtype::Bf16, 1)
    ),
    (
        "qwen38-flash-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::flash(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    // The mini 2-bit snapshot's row: the same flash organs at the artifact's
    // own four layers, sixteen experts and eight n-gram shards, its trunk
    // 4-bit at group 64, its embedding and head plain bf16, and its routed
    // expert banks the two-bit mix (`Mix::MIXED_2BIT` says the whole of it).
    (
        "qwen38-flash-mlxu2-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::flash_mini(Mix::MIXED_2BIT, Dtype::Bf16, 1)
    ),
];

pub const IMPORTS: &[crate::ImportRow] = &[
    // **THE MIXED-4/8 ROW STILL LEADS, AND THE 2-BIT ROW GOES SECOND.** The
    // ladder's rule is that a row must miss on a file that is not its own, and
    // these two miss on each other from opposite directions: the 4/8 row
    // declares the embedding at eight bits and the 2-bit file ships it as a
    // bare bf16 plane with no `.scales` beside it, while the 2-bit row
    // declares every projection a four-bit triplet the bf16 file does not
    // hold. Order between them is therefore not load-bearing — but the 4/8
    // artifact is the shipped one and is what `identify` is gated on
    // (`the_checkpoints_state_what_the_texts_read`), so it keeps its turn
    // first and the miniature's row asks after it.
    ("qwen38-flash-mlxu4-kv-bf16", 1, |src, tp| {
        Model::flash(Dtype::U4g64, Dtype::Bf16, tp).import(src)
    }),
    ("qwen38-flash-mlxu2-kv-bf16", 1, |src, tp| {
        Model::flash_mini(Mix::MIXED_2BIT, Dtype::Bf16, tp).import(src)
    }),
    ("qwen38-flash-bf16-kv-bf16", 1, |src, tp| {
        Model::flash(Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
];

pub const TEMPLATES: &[crate::template::TemplateRow] = &[
    ("qwen38-flash-mlxu4-kv-bf16", template::chatml_interleaved),
    ("qwen38-flash-bf16-kv-bf16", template::chatml_interleaved),
    ("qwen38-flash-mlxu2-kv-bf16", template::chatml_interleaved),
];

pub const TOKENIZERS: &[crate::tokenizer::ContractRow] = &[
    ("qwen38-flash-mlxu4-kv-bf16", &tokenizer::CONTRACT_38),
    ("qwen38-flash-bf16-kv-bf16", &tokenizer::CONTRACT_38),
    ("qwen38-flash-mlxu2-kv-bf16", &tokenizer::CONTRACT_38),
];
