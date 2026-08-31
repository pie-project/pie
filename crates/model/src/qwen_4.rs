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

use model::Model;
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
        Model::flash(Dtype::MlxU4, Dtype::Bf16, 1)
    ),
    (
        "qwen38-flash-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::flash(Dtype::Bf16, Dtype::Bf16, 1)
    ),
];

pub const IMPORTS: &[crate::ImportRow] = &[
    ("qwen38-flash-mlxu4-kv-bf16", |src| {
        Model::flash(Dtype::MlxU4, Dtype::Bf16, 1).import(src)
    }),
    ("qwen38-flash-bf16-kv-bf16", |src| {
        Model::flash(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
];

pub const TEMPLATES: &[crate::template::TemplateRow] = &[
    ("qwen38-flash-mlxu4-kv-bf16", template::chatml_interleaved),
    ("qwen38-flash-bf16-kv-bf16", template::chatml_interleaved),
];

pub const TOKENIZERS: &[crate::tokenizer::ContractRow] = &[
    ("qwen38-flash-mlxu4-kv-bf16", &tokenizer::CONTRACT_38),
    ("qwen38-flash-bf16-kv-bf16", &tokenizer::CONTRACT_38),
];
