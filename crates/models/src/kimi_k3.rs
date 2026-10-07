pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::{Bf16, Mxfp4};
    vec![
        crate::entry! {
            id: "kimik3-mini",
            fixture: true,
            parts: [],
            drafters: [],
            template: template::instruct3,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model { match d.weights[..] { [Bf16, experts] => Ok(Model::k3_mini(8, 32, 4, Bf16, experts, d.kv)), _ => Err(crate::catalog::Refused::unsupported("kimik3-mini", d)) } },
            rows: [(0, 1, [Bf16, Mxfp4], Bf16, [], None), (1, 2, [Bf16, Mxfp4], Bf16, [], None)],
        },
        crate::entry! {
            id: "kimik3",
            fixture: false,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model { match d.weights[..] { [Bf16, experts] => Ok(Model::k3(Bf16, experts, d.kv)), _ => Err(crate::catalog::Refused::unsupported("kimik3", d)) } },
            rows: [(2, 1, [Bf16, Mxfp4], Bf16, [], None), (3, 2, [Bf16, Mxfp4], Bf16, [], None)],
        },
    ]
}
