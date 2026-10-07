pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub const ARCH: &str = "wan_2";

pub fn skus() -> Vec<crate::Sku> {
    let mut rows = crate::skus![
        (
            "wan22-ti2v-5b",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::ti2v_5b(Dtype::Bf16, tp),
        ),
        (
            "wan22-ti2v-5b",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::ti2v_5b(Dtype::U4g64, tp),
        ),
        (
            "wan22-mini-d128",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini_d128(Dtype::Bf16, tp),
        ),
        (
            "wan22-mini-nano",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            |tp: u32| Model::mini_nano(Dtype::Bf16, tp),
        ),
    ];
    for row in &mut rows {
        let model = match row.recipe.text {
            "wan22-ti2v-5b" => Model::ti2v_5b(Dtype::Bf16, 1),
            "wan22-mini-d128" => Model::mini_d128(Dtype::Bf16, 1),
            "wan22-mini-nano" => Model::mini_nano(Dtype::Bf16, 1),
            other => unreachable!("no wan_2 row is called `{other}`"),
        };
        row.generative = Some(model.generative());
    }
    rows
}
