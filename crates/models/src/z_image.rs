pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;
pub mod vae;

use model::Model;
use poem_dsl::Dtype;

pub const ARCH: &str = "z_image";

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::{Bf16, U4g64};
    vec![
        crate::entry! {
            id: "z-image-turbo",
            fixture: false,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::turbo(d.dtype()?)) },
            rows: [(0, "z-image-turbo", 1, [Bf16], Bf16, [], None), (1, "z-image-turbo", 1, [U4g64], Bf16, [], None)],
        },
        crate::entry! {
            id: "z-image-mini",
            fixture: true,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::mini(d.dtype()?)) },
            rows: [(2, "z-image-mini", 1, [Bf16], Bf16, [], None)],
        },
    ]
}
