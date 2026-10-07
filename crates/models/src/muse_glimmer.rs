pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::{Bf16, U4g64};
    vec![
        crate::entry! {
            id: "muse-glimmer-30b",
            fixture: false,
            parts: [],
            drafters: [],
            template: template::muse_glimmer,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model { Ok(Model::b30(d.dtype()?, d.kv)) },
            rows: [(0, "muse-glimmer-30b", 1, [Bf16], Bf16, [], None), (1, "muse-glimmer-30b", 2, [Bf16], Bf16, [], None), (2, "muse-glimmer-30b", 1, [U4g64], Bf16, [], None), (3, "muse-glimmer-30b", 2, [U4g64], Bf16, [], None)],
        },
        crate::entry! {
            id: "muse-glimmer-30b-mini-l8",
            fixture: true,
            parts: [],
            drafters: [],
            template: template::muse_glimmer,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model { Ok(Model::b30_mini(8, d.dtype()?, d.kv)) },
            rows: [(4, "muse-glimmer-30b-mini-l8", 1, [Bf16], Bf16, [], None)],
        },
    ]
}
