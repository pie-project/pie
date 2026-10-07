pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub const ARCH: &str = "ltx_2";

pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "ltx25",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::ltx_2_5(Dtype::Bf16, tp),
        ),
        (
            "ltx25",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::ltx_2_5(Dtype::U4g64, tp),
        ),
        (
            "ltx25-mini",
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
            "ltx25" => Model::ltx_2_5(Dtype::Bf16, 1),
            "ltx25-mini" => Model::mini(Dtype::Bf16, 1),
            other => unreachable!("no ltx_2 row is called `{other}`"),
        };
        row.generative = Some(model.generative());
    }
    rows
}
