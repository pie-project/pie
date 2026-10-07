pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::Bf16;
    vec![
        crate::entry! {
            id: "inkling",
            fixture: false,
            parts: [],
            drafters: [],
            template: template::inkling,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model { Ok(Model::full(d.dtype()?, d.kv)) },
            rows: [(0, "inkling", 1, [Bf16], Bf16, [], None)],
        },
        crate::entry! {
            id: "inkling-mini-l7-e8",
            fixture: true,
            parts: [],
            drafters: [],
            template: template::inkling,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model { Ok(Model::mini(7, 8, d.dtype()?, d.kv)) },
            rows: [(1, "inkling-mini-l7-e8", 1, [Bf16], Bf16, [], None)],
        },
    ]
}
