pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;

use crate::catalog::Refused;
use poem_dsl::Dtype;

pub const ARCH: &str = "hunyuan_image_3_moe";

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::{Bf16, U4g64, U8g64};
    fn diffusion(model: &Model) -> crate::Diffusion {
        crate::Diffusion {
            canvas: canvas_rows(model),
            hidden: model.dims.hidden,
            self_cond_taps: 0,
        }
    }
    vec![
        crate::entry! {
            id: "hunyuanimage3-80b-a13b",
            fixture: false,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: Some(diffusion),
            generative: Some(Model::generative),
            build: |d| -> Model {
                match d.weights[..] {
                    [Bf16, experts] => Ok(Model::flagship(Bf16, experts, d.kv)),
                    _ => Err(Refused::unsupported("hunyuanimage3-80b-a13b", d)),
                }
            },
            rows: [
                (0, "hunyuanimage3-80b-a13b", 1, [Bf16, U8g64], Bf16, [], None),
                (1, "hunyuanimage3-80b-a13b", 4, [Bf16, U8g64], Bf16, [], None),
                (2, "hunyuanimage3-80b-a13b", 4, [Bf16, U4g64], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "hunyuanimage3-mini",
            fixture: true,
            parts: [],
            drafters: [],
            template: template::instruct,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: Some(diffusion),
            generative: Some(Model::generative),
            build: |d| -> Model { Ok(Model::mini(d.dtype()?, d.kv)) },
            rows: [(3, "hunyuanimage3-mini", 1, [Bf16], Bf16, [], None)],
        },
    ]
}

fn canvas_rows(model: &Model) -> u32 {
    let side = if model.dims.layers > 4 { 64 } else { 8 };
    side * side
}
