pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem::Dtype;

pub const ARCH: &str = "minimax_h3";

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::Bf16;
    vec![
        crate::entry! {
            id: "minimax-h3-fl2va",
            mini: false,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::fl2va(d.dtype()?)) },
            rows: [
                (0, [Bf16], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "minimax-h3-mini",
            mini: true,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::mini(d.dtype()?)) },
            rows: [(3, [Bf16], Bf16, [], None)],
        },
    ]
}
