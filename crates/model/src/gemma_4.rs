pub mod forward;
pub mod import;
pub mod model;
pub mod template;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[crate::Row] = model_dsl::catalog![
    (
        "gemma4-e4b-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::e4b(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "gemma4-31b-bf16-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::b31(Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "gemma4-31b-mlxu4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::b31(Dtype::MlxU4, Dtype::Bf16, 1)
    ),
    (
        "gemma4-31b-bf16-kv-bf16-tp2",
        2,
        model_dsl::trace_hybrid,
        Model::b31(Dtype::Bf16, Dtype::Bf16, 2)
    ),
];

/// **THE `mlxu4` ROW COMES FIRST, AND THE ORDER IS LOAD-BEARING.**
/// `model::identify` returns the first row whose contract the file can
/// satisfy. The 4-bit row is the strictly more demanding one — every
/// projection it reads is a `.weight` / `.scales` / `.biases` triplet, and a
/// bf16 checkpoint holds only the first — so it misses on a bf16 file and the
/// bf16 row below gets its turn. The reverse does not hold: a bf16 row asked
/// about an MLX file finds every name it asks for, at a width it does not
/// check here, and would claim it.
///
/// `gemma4-e4b` has no 4-bit row. Its per-layer-embedding table is a column
/// band of one stored `embed_tokens_per_layer` bank, and a band of packed
/// codes is cut in words while a band of its scales is cut in groups — two
/// arithmetics this import does not state. `gemma4-31b` declares no PLE at
/// all, so nothing about it is sliced and the triplet reading is whole.
pub const IMPORTS: &[crate::ImportRow] = &[
    ("gemma4-31b-mlxu4-kv-bf16", |src| {
        Model::b31(Dtype::MlxU4, Dtype::Bf16, 1).import(src)
    }),
    ("gemma4-e4b-bf16-kv-bf16", |src| {
        Model::e4b(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("gemma4-31b-bf16-kv-bf16", |src| {
        Model::b31(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
    ("gemma4-31b-bf16-kv-bf16-tp2", |src| {
        Model::b31(Dtype::Bf16, Dtype::Bf16, 1).import(src)
    }),
];

pub const TEMPLATES: &[crate::template::TemplateRow] = &[
    ("gemma4-e4b-bf16-kv-bf16", template::gemma4),
    ("gemma4-31b-bf16-kv-bf16", template::gemma4),
    ("gemma4-31b-mlxu4-kv-bf16", template::gemma4),
    ("gemma4-31b-bf16-kv-bf16-tp2", template::gemma4),
];
