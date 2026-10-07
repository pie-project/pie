pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::Bf16;
    vec![crate::entry! {
        id: "glm5-a12b",
        fixture: false,
        parts: [],
        drafters: [],
        template: template::instruct,
        tokenizer: &tokenizer::CONTRACT,
        diffusion: None,
        generative: None,
        build: |d| -> Model { let w = d.dtype()?; Ok(Model::a12b(w, w, d.kv)) },
        rows: [(0, 1, [Bf16], Bf16, [], None), (1, 2, [Bf16], Bf16, [], None)],
    }]
}
