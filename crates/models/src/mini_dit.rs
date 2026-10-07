pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub const ARCH: &str = "mini_dit";

pub fn skus() -> Vec<crate::Sku> {
    let tap = forward::Tap::from_env();
    let mut rows = crate::skus![
        (
            "mini-dit",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            || Model::mini(Dtype::Bf16).tapped(forward::Tap::from_env()),
        ),
        (
            "mini-dit",
            2,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            || Model::mini(Dtype::Bf16).tapped(forward::Tap::from_env()),
        ),
        (
            "mini-dit",
            4,
            [Dtype::Bf16],
            Dtype::Bf16,
            poem_dsl::trace_hybrid,
            template::instruct,
            &tokenizer::CONTRACT,
            || Model::mini(Dtype::Bf16).tapped(forward::Tap::from_env()),
        ),
    ];
    for row in &mut rows {
        row.generative = Some(forward::generative(tap.as_deref()));
    }
    rows
}
