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
            "inkling",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::inkling,
            &tokenizer::CONTRACT,
            |tp: u32| Model::full(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "inkling-mini-l7-e8",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::inkling,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini(7, 8, Dtype::Bf16, Dtype::Bf16, tp),
        ),
    ]
}
