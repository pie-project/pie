pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;
pub mod vae;

use model::Model;
use poem_dsl::Dtype;

pub const ARCH: &str = "flux_2";

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::{Bf16, U4g64};
    vec![
        crate::entry! {
            id: "flux2-klein-4b",
            fixture: false,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::klein_4b(d.dtype()?)) },
            rows: [(0, 1, [Bf16], Bf16, [], None), (1, 1, [U4g64], Bf16, [], None)],
        },
        crate::entry! {
            id: "flux2-mini",
            fixture: true,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::mini(d.dtype()?)) },
            rows: [(2, 1, [Bf16], Bf16, [], None)],
        },
    ]
}
