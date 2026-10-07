pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub const ARCH: &str = "mini_dit";

pub fn entries() -> Vec<crate::catalog::Entry> {
    use Dtype::Bf16;
    fn generative(_: &Model) -> crate::Generative {
        forward::generative(forward::Tap::from_env().as_deref())
    }
    vec![crate::entry! {
        id: "mini-dit",
        fixture: true,
        parts: [],
        drafters: [],
        template: template::instruct,
        tokenizer: &tokenizer::CONTRACT,
        diffusion: None,
        generative: Some(generative),
        build: |d| -> Model { Ok(Model::mini(d.dtype()?).tapped(forward::Tap::from_env())) },
        rows: [
            (0, 1, [Bf16], Bf16, [], None),
            (1, 2, [Bf16], Bf16, [], None),
            (2, 4, [Bf16], Bf16, [], None),
        ],
    }]
}
