pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;
pub mod vae;

use model::Model;
use poem_dsl::Dtype;

pub const ARCH: &str = "z_image";

pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "z-image-turbo",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::turbo(Dtype::Bf16, tp),
        ),
        (
            "z-image-turbo",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::turbo(Dtype::U4g64, tp),
        ),
        (
            "z-image-mini",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini(Dtype::Bf16, tp),
        ),
    ];
    for row in &mut rows {
        let model = match row.recipe.text {
            "z-image-turbo" => Model::turbo(Dtype::Bf16, 1),
            "z-image-mini" => Model::mini(Dtype::Bf16, 1),
            other => unreachable!("no z-image row is called `{other}`"),
        };
        row.generative = Some(model.generative());
    }
    rows
}
