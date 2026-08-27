pub mod forward;
pub mod import;
pub mod model;
pub mod template;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[(&str, model_dsl::TraceFn)] = model_dsl::catalog![
    (
        "gemma4-e4b-bf16-kv-bf16",
        model_dsl::trace_hybrid,
        Model::e4b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "gemma4-31b-bf16-kv-bf16",
        model_dsl::trace_hybrid,
        Model::b31(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "gemma4-31b-bf16-kv-bf16-tp2",
        model_dsl::trace_hybrid,
        Model::b31(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 2)
    ),
];

pub const IMPORTS: &[crate::ImportRow] = &[
    ("gemma4-e4b-bf16-kv-bf16", |src| {
        Model::e4b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("gemma4-31b-bf16-kv-bf16", |src| {
        Model::b31(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("gemma4-31b-bf16-kv-bf16-tp2", |src| {
        Model::b31(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
];

pub const TEMPLATES: &[crate::template::TemplateRow] = &[
    ("gemma4-e4b-bf16-kv-bf16", template::gemma4),
    ("gemma4-31b-bf16-kv-bf16", template::gemma4),
    ("gemma4-31b-bf16-kv-bf16-tp2", template::gemma4),
];
