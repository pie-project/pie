pub mod forward;
pub mod import;
pub mod model;
pub mod template;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[crate::Row] = model_dsl::catalog![
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

pub const IMPORTS: &[crate::ImportRow] = &[
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
    ("qwen35-a3b-bf16-kv-bf16", template::chatml),
    ("qwen35-d3b-bf16-kv-bf16", template::chatml),
    ("qwen35-d0.8b-bf16-kv-bf16", template::chatml),
    ("qwen35-a3b-bf16-kv-bf16-tp2", template::chatml),
];
