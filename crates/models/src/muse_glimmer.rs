pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        (
            "muse-glimmer-30b",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::muse_glimmer,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b30(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "muse-glimmer-30b",
            2,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::muse_glimmer,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b30(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "muse-glimmer-30b",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::muse_glimmer,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b30(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "muse-glimmer-30b",
            2,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::muse_glimmer,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b30(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "muse-glimmer-30b-mini-l8",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::muse_glimmer,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b30_mini(8, Dtype::Bf16, Dtype::Bf16, tp),
        ),
    ]
}
