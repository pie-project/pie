pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub const ARCH: &str = "minimax_h3";

pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "minimax-h3-fl2va",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::fl2va(Dtype::Bf16, tp),
        ),
        (
            "minimax-h3-fl2va",
            2,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::fl2va(Dtype::Bf16, tp),
        ),
        (
            "minimax-h3-fl2va",
            4,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::fl2va(Dtype::Bf16, tp),
        ),
        (
            "minimax-h3-mini",
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
            "minimax-h3-fl2va" => Model::fl2va(Dtype::Bf16, row.recipe.tp),
            "minimax-h3-mini" => Model::mini(Dtype::Bf16, row.recipe.tp),
            other => unreachable!("no minimax_h3 row is called `{other}`"),
        };
        row.generative = Some(model.generative());
    }
    rows
}
