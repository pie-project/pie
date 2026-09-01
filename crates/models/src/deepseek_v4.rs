pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::{Model, Routed};
use model_dsl::Dtype;

/// **THE 2-BIT DQ ROW'S OWN MODEL**, said once and read by all four registries
/// below.
///
/// Four dtypes and not one: the trunk (attention, the shared expert, the
/// embedding and the head) is MLX's own 4-bit at group 64, and the ROUTED
/// experts are the per-tensor mix [`Routed::DQ_2BIT`] states. The rest of the
/// text — norms, the compressor's position embedding, the router gate — reads
/// what those banks compute in, which is bf16.
pub fn flash_mlxu2(tp: u32) -> Model {
    Model::flash_mini(Dtype::U4g64, Routed::DQ_2BIT, Dtype::Bf16, Dtype::Bf16, tp)
}

pub const CATALOG: &[crate::Row] = model_dsl::catalog![
    (
        "dsv4-base-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "dsv4-base-bf16-kv-bf16-tp2",
        2,
        model_dsl::trace_hybrid,
        Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 2)
    ),
    (
        "dsv4-flash-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::flash(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    // The mini DQ snapshot's row: the same flash organs at the artifact's own
    // five layers and sixteen experts, its trunk 4-bit and its routed experts
    // the 2-bit mix.
    (
        "dsv4-flash-mlxu2-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        flash_mlxu2(1)
    ),
];

pub const IMPORTS: &[crate::ImportRow] = &[
    // **THE 2-BIT DQ ROW LEADS**, for `qwen_3::IMPORTS`'s reason: every
    // projection it names is a `.weight`/`.scales`/`.biases` triplet, so it
    // MISSES on a bf16 artifact and the next row gets its turn — where a bf16
    // row listed first would claim a quantized checkpoint and fail four stages
    // later.
    ("dsv4-flash-mlxu2-kv-bf16", 1, |src, tp| flash_mlxu2(tp).import(src)),
    ("dsv4-base-bf16-kv-bf16", 1, |src, tp| {
        Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
    ("dsv4-base-bf16-kv-bf16-tp2", 2, |src, tp| {
        Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
    ("dsv4-flash-bf16-kv-bf16", 1, |src, tp| {
        Model::flash(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, tp).import(src)
    }),
];

pub const TEMPLATES: &[crate::template::TemplateRow] = &[
    ("dsv4-base-bf16-kv-bf16", template::r1),
    ("dsv4-base-bf16-kv-bf16-tp2", template::r1),
    ("dsv4-flash-bf16-kv-bf16", template::r1),
    ("dsv4-flash-mlxu2-kv-bf16", template::r1),
];

pub const TOKENIZERS: &[crate::tokenizer::ContractRow] = &[
    ("dsv4-base-bf16-kv-bf16", &tokenizer::CONTRACT),
    ("dsv4-base-bf16-kv-bf16-tp2", &tokenizer::CONTRACT),
    ("dsv4-flash-bf16-kv-bf16", &tokenizer::CONTRACT),
    ("dsv4-flash-mlxu2-kv-bf16", &tokenizer::CONTRACT),
];
