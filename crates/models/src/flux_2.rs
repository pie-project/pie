pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;
pub mod vae;

use model::Model;
use poem_dsl::Dtype;

pub const ARCH: &str = "flux_2";

pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "flux2-klein-4b",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            || Model::klein_4b(Dtype::Bf16),
        ),
        (
            "flux2-klein-4b",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            || Model::klein_4b(Dtype::U4g64),
        ),
        (
            "flux2-mini",
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
            "flux2-klein-4b" => Model::klein_4b(Dtype::Bf16),
            "flux2-mini" => Model::mini(Dtype::Bf16),
            other => unreachable!("no flux_2 row is called `{other}`"),
        };
        row.generative = Some(model.generative());
    }
    rows
}
