pub mod forward;
pub mod import;
pub mod model;

use model::Model;
use poem_dsl::Dtype;

use crate::gemma_4::{template, tokenizer};

pub const ARCH: &str = "diffusion_gemma";

pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "diffusiongemma-26b-a4b",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            || Model::a4b(Dtype::U4g64, Dtype::Bf16),
        ),
        (
            "diffusiongemma-26b-a4b",
            1,
            [Dtype::U8g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            || Model::a4b(Dtype::U8g64, Dtype::Bf16),
        ),
        (
            "diffusiongemma-26b-a4b",
            1,
            [Dtype::U8g64, Dtype::U4g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            || Model::a4b_experts(Dtype::U8g64, Dtype::U4g64, Dtype::Bf16),
        ),
        (
            "diffusiongemma-26b-a4b",
            1,
            [Dtype::U4g64, Dtype::U8g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            || Model::a4b_experts(Dtype::U4g64, Dtype::U8g64, Dtype::Bf16),
        ),
        (
            "diffusiongemma-26b-a4b",
            1,
            [Dtype::U8g64, Dtype::U4g64, Dtype::U4g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            || Model::a4b_experts_self_cond(Dtype::U8g64, Dtype::U4g64, Dtype::U4g64, Dtype::Bf16,),
        ),
        (
            "diffusiongemma-26b-a4b",
            1,
            [Dtype::Bf16, Dtype::U4g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            || Model::a4b_experts(Dtype::Bf16, Dtype::U4g64, Dtype::Bf16),
        ),
    ];
    for row in &mut rows {
        row.diffusion = Some(crate::Diffusion {
            canvas: model::CANVAS,
            hidden: model::HIDDEN,
            self_cond_taps: crate::gemma_4::model::SELF_COND_TAPS,
        });
    }
    rows
}
