pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        (
            "kimik3-mini",
            1,
            [Dtype::Bf16, Dtype::Mxfp4],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct3,
            &tokenizer::CONTRACT3,
            || Model::k3_mini(8, 32, 4, Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16),
        ),
        (
            "kimik3-mini",
            2,
            [Dtype::Bf16, Dtype::Mxfp4],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct3,
            &tokenizer::CONTRACT3,
            || Model::k3_mini(8, 32, 4, Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16),
        ),
        (
            "kimik3",
            1,
            [Dtype::Bf16, Dtype::Mxfp4],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            || Model::k3(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16),
        ),
        (
            "kimik3",
            2,
            [Dtype::Bf16, Dtype::Mxfp4],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            || Model::k3(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16),
        ),
    ]
}
