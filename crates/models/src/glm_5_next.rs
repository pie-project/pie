pub mod forward;
pub mod import;
pub mod media;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub fn entries() -> Vec<crate::catalog::Entry> {
    use crate::catalog::{Drafter, Refused};
    use Dtype::{Bf16, U2g64, U4g64, U8g64};
    vec![
        crate::entry! {
            id: "glm53-flash",
            fixture: false,
            parts: [Vision],
            drafters: [Mtp],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                let vision = d.has(crate::catalog::Part::Vision);
                match (&d.weights[..], d.drafter, vision) {
                    ([w, experts], None, false) => Ok(Model::flash(*w, *experts, d.kv)),
                    ([w, experts], None, true) => Ok(Model::flash_vision(*w, *experts, d.kv)),
                    ([w, experts, mtp], Some(Drafter::Mtp), false) => {
                        Ok(Model::flash_mtp(*w, *experts, *mtp, d.kv))
                    }
                    ([w, experts, mtp], Some(Drafter::Mtp), true) => {
                        Ok(Model::flash_mtp_vision(*w, *experts, *mtp, d.kv))
                    }
                    _ => Err(Refused::unsupported("glm53-flash", d)),
                }
            },
            rows: [
                (0, 1, [U8g64, U2g64, U4g64], Bf16, [], Some(Drafter::Mtp)),
                (3, 1, [U8g64, U2g64], Bf16, [], None),
                (4, 1, [U8g64, U2g64, U4g64], Bf16, [Vision], Some(Drafter::Mtp)),
                (5, 1, [U8g64, U2g64], Bf16, [Vision], None),
                (6, 1, [U4g64, U2g64, U4g64], Bf16, [Vision], Some(Drafter::Mtp)),
            ],
        },
        crate::entry! {
            id: "glm53-flash-mini",
            fixture: true,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                match d.weights[..] {
                    [w, experts] => Ok(Model::flash_mini(8, 32, w, experts, d.kv)),
                    _ => Err(Refused::unsupported("glm53-flash-mini", d)),
                }
            },
            rows: [
                (1, 1, [U4g64, U4g64], Bf16, [], None),
                (2, 2, [U4g64, U4g64], Bf16, [], None),
            ],
        },
    ]
}
