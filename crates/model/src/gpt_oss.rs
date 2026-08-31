pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[crate::Row] = model_dsl::catalog![
    (
        "gptoss-20b-mlxu4-mxfp4-kv-bf16",
        1,
        model_dsl::trace_hybrid,
        Model::b20(Dtype::MlxU4, Dtype::Mxfp4, Dtype::Bf16, 1)
    ),
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

/// **THE `mlxu4` ROW COMES FIRST, AND THE ORDER IS LOAD-BEARING** —
/// `qwen_3::IMPORTS`' rule in this family's spelling. `model::identify` returns
/// the first row whose contract the file can satisfy, and the 4-bit row is the
/// strictly more demanding one: every dense projection it reads is a
/// `.weight` / `.scales` / `.biases` triplet and its router gate is a fourth,
/// none of which a bf16 checkpoint holds, so it misses on one and the row
/// below gets its turn. The reverse does not hold — a bf16 row asked about an
/// MLX file finds every `.weight` it names, at a container and a width it never
/// checks here, and would claim it and then fail four stages later against a
/// shape nobody wrote.
///
/// **THE EXPERT SPELLING DOES NOT DISCRIMINATE AND MUST NOT BE ASKED TO**:
/// both rows read both, through `import::Layout`, because a checkpoint's MoE
/// packing and its dense weights' representation are independent facts.
pub const IMPORTS: &[crate::ImportRow] = &[
    ("gptoss-20b-mlxu4-mxfp4-kv-bf16", |src| {
        Model::b20(Dtype::MlxU4, Dtype::Mxfp4, Dtype::Bf16, 1).import(src)
    }),
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
    ("gptoss-20b-mlxu4-mxfp4-kv-bf16", template::gpt_oss),
    ("gptoss-20b-bf16-mxfp4-kv-bf16", template::gpt_oss),
    ("gptoss-120b-bf16-mxfp4-kv-bf16", template::gpt_oss),
    ("gptoss-120b-bf16-mxfp4-kv-bf16-tp2", template::gpt_oss),
];

pub const TOKENIZERS: &[crate::tokenizer::ContractRow] = &[
    ("gptoss-20b-mlxu4-mxfp4-kv-bf16", &tokenizer::CONTRACT),
    ("gptoss-20b-bf16-mxfp4-kv-bf16", &tokenizer::CONTRACT),
    ("gptoss-120b-bf16-mxfp4-kv-bf16", &tokenizer::CONTRACT),
    ("gptoss-120b-bf16-mxfp4-kv-bf16-tp2", &tokenizer::CONTRACT),
];
