pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub const ARCH: &str = "hunyuan_image_3_moe";

pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "hunyuanimage3-80b-a13b",
            1,
            [Dtype::Bf16, Dtype::U8g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::flagship(Dtype::Bf16, Dtype::U8g64, Dtype::Bf16, tp),
        ),
        (
            "hunyuanimage3-80b-a13b",
            4,
            [Dtype::Bf16, Dtype::U8g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::flagship(Dtype::Bf16, Dtype::U8g64, Dtype::Bf16, tp),
        ),
        (
            "hunyuanimage3-80b-a13b",
            4,
            [Dtype::Bf16, Dtype::U4g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::flagship(Dtype::Bf16, Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "hunyuanimage3-mini",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini(Dtype::Bf16, Dtype::Bf16, tp),
        ),
    ];
    for row in &mut rows {
        let model = match row.recipe.text {
            "hunyuanimage3-80b-a13b" => {
                Model::flagship(Dtype::Bf16, row.recipe.weights[1], Dtype::Bf16, 1)
            }
            "hunyuanimage3-mini" => Model::mini(Dtype::Bf16, Dtype::Bf16, 1),
            other => unreachable!("no hunyuan_image_3 row is called `{other}`"),
        };
        row.diffusion = Some(crate::Diffusion {
            canvas: canvas_rows(&model),
            hidden: model.dims.hidden,
            self_cond_taps: 0,
        });
        row.generative = Some(model.generative());
    }
    rows
}

fn canvas_rows(model: &Model) -> u32 {
    let side = if model.dims.layers > 4 { 64 } else { 8 };
    side * side
}
