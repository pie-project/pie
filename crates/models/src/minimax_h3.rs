pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub const ARCH: &str = "minimax_h3";

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::Bf16;
    vec![
        crate::entry! {
            id: "minimax-h3-fl2va",
            fixture: false,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::fl2va(d.dtype()?)) },
            rows: [
                (0, "minimax-h3-fl2va", 1, [Bf16], Bf16, [], None),
                (1, "minimax-h3-fl2va", 2, [Bf16], Bf16, [], None),
                (2, "minimax-h3-fl2va", 4, [Bf16], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "minimax-h3-mini",
            fixture: true,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::mini(d.dtype()?)) },
            rows: [(3, "minimax-h3-mini", 1, [Bf16], Bf16, [], None)],
        },
    ]
}
