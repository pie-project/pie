pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[crate::Row] = model_dsl::catalog![
    (
        "kimik3-bf16-mxfp4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::k3(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, 1)
    ),
    (
        "kimik3-bf16-mxfp4-kv-bf16-tp2",
        2,
        model_dsl::trace_hybrid,
        Model::k3(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, 2)
    ),
];

pub const IMPORTS: &[crate::ImportRow] = &[
    ("kimik3-bf16-mxfp4-kv-bf16", 1, |src, tp| {
        Model::k3(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, tp).import(src)
    }),
    ("kimik3-bf16-mxfp4-kv-bf16-tp2", 2, |src, tp| {
        Model::k3(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, tp).import(src)
    }),
];

pub const TEMPLATES: &[crate::template::TemplateRow] = &[
    ("kimik3-bf16-mxfp4-kv-bf16", template::instruct),
    ("kimik3-bf16-mxfp4-kv-bf16-tp2", template::instruct),
];

pub const TOKENIZERS: &[crate::tokenizer::ContractRow] = &[
    ("kimik3-bf16-mxfp4-kv-bf16", &tokenizer::CONTRACT),
    ("kimik3-bf16-mxfp4-kv-bf16-tp2", &tokenizer::CONTRACT),
];
