pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::{Bf16, Mxfp4, U4g64};
    vec![
        crate::entry! {
            id: "gptoss-20b",
            mini: false,
            parts: [],
            drafters: [DFlash],
            template: template::gpt_oss,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model { match (&d.weights[..], d.drafter) { ([w, experts], None) => Ok(Model::b20(*w, *experts, d.kv)), ([w, experts], Some(crate::catalog::Drafter::DFlash)) => Ok(Model::b20_dflash(*w, *experts, d.kv)), _ => Err(crate::catalog::Refused::unsupported("gptoss-20b", d)) } },
            rows: [(0, [U4g64, Mxfp4], Bf16, [], Some(crate::catalog::Drafter::DFlash)), (1, [U4g64, Mxfp4], Bf16, [], None), (3, [Bf16, Mxfp4], Bf16, [], None)],
        },
        crate::entry! {
            id: "gptoss-20b-mini",
            mini: true,
            parts: [],
            drafters: [],
            template: template::gpt_oss,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model { match (&d.weights[..], d.drafter) { ([w, experts], None) => Ok(Model::b20_mini(*w, *experts, d.kv)), _ => Err(crate::catalog::Refused::unsupported("gptoss-20b-mini", d)) } },
            rows: [(5, [Bf16, Mxfp4], Bf16, [], None)],
        },
        crate::entry! {
            id: "gptoss-120b",
            mini: false,
            parts: [],
            drafters: [],
            template: template::gpt_oss,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model { match (&d.weights[..], d.drafter) { ([w, experts], None) => Ok(Model::b120(*w, *experts, d.kv)), _ => Err(crate::catalog::Refused::unsupported("gptoss-120b", d)) } },
            rows: [(7, [Bf16, Mxfp4], Bf16, [], None)],
        },
    ]
}
