pub mod forward;
pub mod import;
pub mod model;
pub mod template;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[crate::Row] = model_dsl::catalog![
    (
        "gptoss-20b-bf16-mxfp4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::b20(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, 1)
    ),
    (
        "gptoss-120b-bf16-mxfp4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::b120(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, 1)
    ),
    (
        "gptoss-120b-bf16-mxfp4-kv-bf16-tp2",
        2,
        model_dsl::trace_hybrid,
        Model::b120(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, 2)
    ),
];

pub const IMPORTS: &[crate::ImportRow] = &[
    ("gptoss-20b-bf16-mxfp4-kv-bf16", |src| {
        Model::b20(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, 1).import(src)
    }),
    ("gptoss-120b-bf16-mxfp4-kv-bf16", |src| {
        Model::b120(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, 1).import(src)
    }),
    ("gptoss-120b-bf16-mxfp4-kv-bf16-tp2", |src| {
        Model::b120(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, 1).import(src)
    }),
];

pub const TEMPLATES: &[crate::template::TemplateRow] = &[
    ("gptoss-20b-bf16-mxfp4-kv-bf16", template::gpt_oss),
    ("gptoss-120b-bf16-mxfp4-kv-bf16", template::gpt_oss),
    ("gptoss-120b-bf16-mxfp4-kv-bf16-tp2", template::gpt_oss),
];
