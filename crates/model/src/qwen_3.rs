pub mod forward;
pub mod import;
pub mod model;
pub mod template;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[crate::Row] = model_dsl::catalog![
    (
        "qwen36-27b-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d27b(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-a3b-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::a3b(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-d3b-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d3b(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-d0.8b-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::d0_8b(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-a3b-bf16-kv-bf16-tp2",
        2,
        model_dsl::trace_hybrid,
        Model::a3b(Dtype::Bf16, Dtype::Bf16, 2)
    ),
];

/// **`qwen36-27b` IS FIRST, AND THE ORDER IS LOAD-BEARING.** `model::identify`
/// walks these rows and returns the first whose contract the checkpoint can
/// satisfy, and a contract is satisfied by NAMES: `qwen35-d3b` asks for
/// twenty-four `model.language_model.layers.*` and a dense mlp, all of which a
/// Qwen3.6-27B file also holds, so a d3b row reached first would claim a 27B
/// artifact and land the first twenty-four of its sixty-four layers. The
/// reverse cannot happen — `qwen36-27b` asks for layers up to sixty-three and
/// for fifteen `mtp.*` planes, and a 3B file holds neither — so the strictly
/// more demanding row goes first and the ambiguity is closed by construction.
pub const IMPORTS: &[crate::ImportRow] = &[
    ("qwen36-27b-bf16-kv-bf16", |src| {
        Model::d27b(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("qwen35-a3b-bf16-kv-bf16", |src| {
        Model::a3b(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("qwen35-d3b-bf16-kv-bf16", |src| {
        Model::d3b(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("qwen35-d0.8b-bf16-kv-bf16", |src| {
        Model::d0_8b(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("qwen35-a3b-bf16-kv-bf16-tp2", |src| {
        Model::a3b(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
];

pub const TEMPLATES: &[crate::template::TemplateRow] = &[
    ("qwen36-27b-bf16-kv-bf16", template::chatml),
    ("qwen35-a3b-bf16-kv-bf16", template::chatml),
    ("qwen35-d3b-bf16-kv-bf16", template::chatml),
    ("qwen35-d0.8b-bf16-kv-bf16", template::chatml),
    ("qwen35-a3b-bf16-kv-bf16-tp2", template::chatml),
];
