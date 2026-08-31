pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[crate::Row] = model_dsl::catalog![
    (
        "glm5-a12b-bf16-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::a12b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "glm5-a12b-bf16-bf16-kv-bf16-tp2",
        2,
        model_dsl::trace_hybrid,
        Model::a12b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 2)
    ),
];

pub const IMPORTS: &[crate::ImportRow] = &[
    ("glm5-a12b-bf16-bf16-kv-bf16", |src| {
        Model::a12b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("glm5-a12b-bf16-bf16-kv-bf16-tp2", |src| {
        Model::a12b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
];

pub const TEMPLATES: &[crate::template::TemplateRow] = &[
    ("glm5-a12b-bf16-bf16-kv-bf16", template::instruct),
    ("glm5-a12b-bf16-bf16-kv-bf16-tp2", template::instruct),
];

pub const TOKENIZERS: &[crate::tokenizer::ContractRow] = &[
    ("glm5-a12b-bf16-bf16-kv-bf16", &tokenizer::CONTRACT),
    ("glm5-a12b-bf16-bf16-kv-bf16-tp2", &tokenizer::CONTRACT),
];
