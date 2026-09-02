pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

/// Identification order: the first row whose import fits the checkpoint wins.
pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        (
            "glm5-a12b",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a12b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "glm5-a12b",
            2,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a12b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, tp),
        ),
    ]
}
