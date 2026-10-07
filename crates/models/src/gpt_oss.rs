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
            fixture: false,
            parts: [],
            drafters: [DFlash],
            template: template::gpt_oss,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model { match (&d.weights[..], d.drafter) { ([w, experts], None) => Ok(Model::b20(*w, *experts, d.kv)), ([w, experts], Some(crate::catalog::Drafter::DFlash)) => Ok(Model::b20_dflash(*w, *experts, d.kv)), _ => Err(crate::catalog::Refused::unsupported("gptoss-20b", d)) } },
            rows: [(0, "gptoss-20b-dflash", 1, [U4g64, Mxfp4], Bf16, [], Some(crate::catalog::Drafter::DFlash)), (1, "gptoss-20b", 1, [U4g64, Mxfp4], Bf16, [], None), (2, "gptoss-20b", 2, [U4g64, Mxfp4], Bf16, [], None), (3, "gptoss-20b", 1, [Bf16, Mxfp4], Bf16, [], None), (4, "gptoss-20b", 2, [Bf16, Mxfp4], Bf16, [], None)],
        },
        crate::entry! {
            id: "gptoss-20b-mini",
            fixture: true,
            parts: [],
            drafters: [],
            template: template::gpt_oss,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model { match (&d.weights[..], d.drafter) { ([w, experts], None) => Ok(Model::b20_mini(*w, *experts, d.kv)), _ => Err(crate::catalog::Refused::unsupported("gptoss-20b-mini", d)) } },
            rows: [(5, "gptoss-20b-mini", 1, [Bf16, Mxfp4], Bf16, [], None), (6, "gptoss-20b-mini", 2, [Bf16, Mxfp4], Bf16, [], None)],
        },
        crate::entry! {
            id: "gptoss-120b",
            fixture: false,
            parts: [],
            drafters: [],
            template: template::gpt_oss,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model { match (&d.weights[..], d.drafter) { ([w, experts], None) => Ok(Model::b120(*w, *experts, d.kv)), _ => Err(crate::catalog::Refused::unsupported("gptoss-120b", d)) } },
            rows: [(7, "gptoss-120b", 1, [Bf16, Mxfp4], Bf16, [], None), (8, "gptoss-120b", 2, [Bf16, Mxfp4], Bf16, [], None)],
        },
    ]
}
