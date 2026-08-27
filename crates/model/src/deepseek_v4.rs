pub mod forward;
pub mod import;
pub mod model;
pub mod template;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[(&str, model_dsl::TraceFn)] = model_dsl::catalog![
    (
        "dsv4-base-bf16-kv-bf16",
        model_dsl::trace_hybrid,
        Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "dsv4-base-bf16-kv-bf16-tp2",
        model_dsl::trace_hybrid,
        Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 2)
    ),
];

pub const IMPORTS: &[crate::ImportRow] = &[
    ("dsv4-base-bf16-kv-bf16", |src| {
        Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("dsv4-base-bf16-kv-bf16-tp2", |src| {
        Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
];

pub const TEMPLATES: &[crate::template::TemplateRow] = &[
    ("dsv4-base-bf16-kv-bf16", template::r1),
    ("dsv4-base-bf16-kv-bf16-tp2", template::r1),
];
