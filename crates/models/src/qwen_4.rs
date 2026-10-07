pub mod forward;
pub mod import;
pub mod model;

use model::{Mix, Model};
use poem_dsl::Dtype;

use crate::qwen_3::{template, tokenizer};

pub const ARCH: &str = "qwen4_exp";

pub fn entries() -> Vec<crate::catalog::Entry> {
    use crate::catalog::{Drafter, Part, Refused};
    use Dtype::{Bf16, U2g128, U4g64};
    vec![
        crate::entry! {
            id: "qwen38-flash-next",
            mini: false,
            parts: [Vision],
            drafters: [Mtp],
            template: template::chatml_interleaved,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                let vision = d.has(Part::Vision);
                let mtp = d.drafts_with(Drafter::Mtp);
                match (&d.weights[..], vision, mtp, d.drafter) {
                    ([w], false, false, None) => Ok(Model::flash(*w, d.kv)),
                    ([U4g64, U2g128], false, false, None) => Ok(Model::flash_mix(Mix::MIXED_2BIT, d.kv)),
                    ([U4g64, U2g128], false, true, _) => Ok(Model::flash_mix_mtp(Mix::MIXED_2BIT, d.kv)),
                    ([U4g64, U2g128], true, false, None) => {
                        Ok(Model::flash_mix_vision(Mix::MIXED_2BIT, d.kv))
                    }
                    ([U4g64, U2g128], true, true, _) => {
                        Ok(Model::flash_mix_mtp_vision(Mix::MIXED_2BIT, d.kv))
                    }
                    _ => Err(Refused::unsupported("qwen38-flash-next", d)),
                }
            },
            rows: [
                (0, [U4g64], Bf16, [], None),
                (1, [U4g64, U2g128], Bf16, [], Some(Drafter::Mtp)),
                (2, [U4g64, U2g128], Bf16, [], None),
                (4, [Bf16], Bf16, [], None),
                (5, [U4g64, U2g128], Bf16, [Vision], Some(Drafter::Mtp)),
                (6, [U4g64, U2g128], Bf16, [Vision], None),
            ],
        },
        crate::entry! {
            id: "qwen38-flash-next-mini",
            mini: true,
            parts: [],
            drafters: [],
            template: template::chatml_interleaved,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                match d.weights[..] {
                    [U4g64, U2g128] => Ok(Model::flash_mini(Mix::MIXED_2BIT, d.kv)),
                    _ => Err(Refused::unsupported("qwen38-flash-next-mini", d)),
                }
            },
            rows: [(3, [U4g64, U2g128], Bf16, [], None)],
        },
    ]
}
