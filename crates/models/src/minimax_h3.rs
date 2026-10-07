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
            || Model::fl2va(Dtype::Bf16),
        ),
        (
            "minimax-h3-fl2va",
            2,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            || Model::fl2va(Dtype::Bf16),
        ),
        (
            "minimax-h3-fl2va",
            4,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            || Model::fl2va(Dtype::Bf16),
        ),
        (
            "minimax-h3-mini",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            || Model::mini(Dtype::Bf16),
        ),
    ];
    for row in &mut rows {
        let model = match row.recipe.text {
            "minimax-h3-fl2va" => Model::fl2va(Dtype::Bf16),
            "minimax-h3-mini" => Model::mini(Dtype::Bf16),
            other => unreachable!("no minimax_h3 row is called `{other}`"),
        };
        row.generative = Some(model.generative());
    }
    rows
}
