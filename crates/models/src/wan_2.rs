pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub const ARCH: &str = "wan_2";

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::{Bf16, U4g64};
    vec![
        crate::entry! {
            id: "wan22-ti2v-5b",
            mini: false,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::ti2v_5b(d.dtype()?)) },
            rows: [(0, [Bf16], Bf16, [], None), (1, [U4g64], Bf16, [], None)],
        },
        crate::entry! {
            id: "wan22-mini-d128",
            mini: true,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::mini_d128(d.dtype()?)) },
            rows: [(2, [Bf16], Bf16, [], None)],
        },
        crate::entry! {
            id: "wan22-mini-nano",
            mini: true,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::mini_nano(d.dtype()?)) },
            rows: [(3, [Bf16], Bf16, [], None)],
        },
    ]
}
