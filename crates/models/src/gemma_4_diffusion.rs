pub mod forward;
pub mod import;
pub mod model;

use model::Model;
use poem_dsl::Dtype;

use crate::gemma_4::{template, tokenizer};

pub const ARCH: &str = "diffusion_gemma";

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::{Bf16, U4g64, U8g64};
    fn diffusion(_: &Model) -> crate::Diffusion {
        crate::Diffusion {
            canvas: model::CANVAS,
            hidden: model::HIDDEN,
            self_cond_taps: crate::gemma_4::model::SELF_COND_TAPS,
        }
    }
    vec![crate::entry! {
        id: "diffusiongemma-26b-a4b",
        fixture: false,
        parts: [SelfCond],
        drafters: [],
        template: template::gemma4,
        tokenizer: &tokenizer::CONTRACT,
        diffusion: Some(diffusion),
        generative: None,
        build: |d| -> Model {
            let self_cond = d.has(crate::catalog::Part::SelfCond);
            match (&d.weights[..], self_cond) {
                ([w], false) => Ok(Model::a4b(*w, d.kv)),
                ([w, experts], false) => Ok(Model::a4b_experts(*w, *experts, d.kv)),
                ([w, experts, sw], true) => {
                    Ok(Model::a4b_experts_self_cond(*w, *experts, *sw, d.kv))
                }
                _ => Err(crate::catalog::Refused::unsupported("diffusiongemma-26b-a4b", d)),
            }
        },
        rows: [
            (0, 1, [U4g64], Bf16, [], None),
            (1, 1, [U8g64], Bf16, [], None),
            (2, 1, [U8g64, U4g64], Bf16, [], None),
            (3, 1, [U4g64, U8g64], Bf16, [], None),
            (4, 1, [U8g64, U4g64, U4g64], Bf16, [SelfCond], None),
            (5, 1, [Bf16, U4g64], Bf16, [], None),
        ],
    }]
}
