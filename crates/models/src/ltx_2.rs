pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub const ARCH: &str = "ltx_2";

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::{Bf16, U4g64};
    vec![
        crate::entry! {
            id: "ltx25",
            mini: false,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::ltx_2_5(d.dtype()?)) },
            rows: [(0, [Bf16], Bf16, [], None), (1, [U4g64], Bf16, [], None)],
        },
        crate::entry! {
            id: "ltx25-mini",
            mini: true,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::mini(d.dtype()?)) },
            rows: [(2, [Bf16], Bf16, [], None)],
        },
    ]
}
