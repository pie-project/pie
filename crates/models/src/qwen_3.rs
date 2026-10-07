pub mod forward;
pub mod import;
pub mod media;
pub mod model;
pub mod rotation;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem_dsl::Dtype;

pub fn entries() -> Vec<crate::catalog::Entry> {
    use crate::catalog::{Drafter, Part, Refused};
    use Dtype::{Bf16, U4g64};
    vec![
        crate::entry! {
            id: "qwen36-27b",
            mini: false,
            parts: [Vision],
            drafters: [Mtp, DFlash],
            template: template::chatml,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                let w = d.dtype()?;
                match (d.has(Part::Vision), d.drafter) {
                    (false, None) => Ok(Model::d27b_undrafted(w, d.kv)),
                    (false, Some(Drafter::Mtp)) => Ok(Model::d27b(w, d.kv)),
                    (true, None) => Ok(Model::d27b_vision_undrafted(w, d.kv)),
                    (true, Some(Drafter::Mtp)) => Ok(Model::d27b_vision(w, d.kv)),
                    (false, Some(Drafter::DFlash)) => Ok(Model::d27b_dflash(w, d.kv)),
                    _ => Err(Refused::unsupported("qwen36-27b", d)),
                }
            },
            rows: [
                (0, [U4g64], Bf16, [], Some(Drafter::Mtp)),
                (1, [U4g64], Bf16, [], Some(Drafter::DFlash)),
                (2, [U4g64], Bf16, [], None),
                (21, [Bf16], Bf16, [], Some(Drafter::Mtp)),
                (38, [U4g64], Bf16, [Vision], None),
                (39, [Bf16], Bf16, [Vision], Some(Drafter::Mtp)),
            ],
        },
        crate::entry! {
            id: "qwen38-27b",
            mini: false,
            parts: [Vision],
            drafters: [Mtp, DFlash2, DSpark],
            template: template::chatml_interleaved,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                let w = d.dtype()?;
                match (d.has(Part::Vision), d.drafter) {
                    (false, None) => Ok(Model::d27b_undrafted(w, d.kv)),
                    (false, Some(Drafter::Mtp)) => Ok(Model::d27b(w, d.kv)),
                    (true, None) => Ok(Model::d27b_vision_undrafted(w, d.kv)),
                    (true, Some(Drafter::Mtp)) => Ok(Model::d27b_vision(w, d.kv)),
                    (false, Some(Drafter::DFlash2)) => Ok(Model::d27b_dflash2(w, d.kv)),
                    (false, Some(Drafter::DSpark)) => Ok(Model::d27b_dspark(w, d.kv)),
                    _ => Err(Refused::unsupported("qwen38-27b", d)),
                }
            },
            rows: [
                (23, [Bf16], Bf16, [], Some(Drafter::Mtp)),
                (25, [U4g64], Bf16, [], Some(Drafter::DFlash2)),
                (26, [U4g64], Bf16, [], Some(Drafter::DSpark)),
                (27, [U4g64], Bf16, [], Some(Drafter::Mtp)),
                (28, [U4g64], Bf16, [], None),
                (40, [U4g64], Bf16, [Vision], None),
                (41, [Bf16], Bf16, [Vision], Some(Drafter::Mtp)),
            ],
        },
        crate::entry! {
            id: "qwen35-d0.8b",
            mini: false,
            parts: [Vision],
            drafters: [Eagle],
            template: template::chatml,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                let w = d.dtype()?;
                match (d.has(Part::Vision), d.drafter) {
                    (false, None) => Ok(Model::d0_8b(w, d.kv)),
                    (false, Some(Drafter::Eagle)) => Ok(Model::d0_8b_eagle(w, d.kv)),
                    (true, None) => Ok(Model::d0_8b_vision(w, d.kv)),
                    (true, Some(Drafter::Eagle)) => Ok(Model::d0_8b_vision_eagle(w, d.kv)),
                    _ => Err(Refused::unsupported("qwen35-d0.8b", d)),
                }
            },
            rows: [
                (5, [U4g64], Bf16, [], None),
                (33, [Bf16], Bf16, [], Some(Drafter::Eagle)),
                (34, [Bf16], Bf16, [], None),
                (37, [Bf16], Bf16, [Vision], Some(Drafter::Eagle)),
                (42, [U4g64], Bf16, [Vision], None),
                (43, [Bf16], Bf16, [Vision], None),
            ],
        },
        crate::entry! {
            id: "qwen35-d2b",
            mini: false,
            parts: [],
            drafters: [],
            template: template::chatml,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                Ok(Model::d2b(d.dtype()?, d.kv))
            },
            rows: [
                (7, [U4g64], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen35-d3b",
            mini: false,
            parts: [],
            drafters: [],
            template: template::chatml,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                Ok(Model::d3b(d.dtype()?, d.kv))
            },
            rows: [
                (31, [Bf16], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen35-d4b",
            mini: false,
            parts: [],
            drafters: [],
            template: template::chatml,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                Ok(Model::d4b(d.dtype()?, d.kv))
            },
            rows: [
                (9, [U4g64], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen35-a3b",
            mini: false,
            parts: [],
            drafters: [],
            template: template::chatml,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                Ok(Model::a3b(d.dtype()?, d.kv))
            },
            rows: [
                (30, [Bf16], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen35-d9b",
            mini: false,
            parts: [],
            drafters: [DFlash],
            template: template::chatml,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                let w = d.dtype()?;
                match (d.has(Part::Vision), d.drafter) {
                    (false, None) => Ok(Model::d9b(w, d.kv)),
                    (false, Some(Drafter::DFlash)) => Ok(Model::d9b_dflash(w, d.kv)),
                    _ => Err(Refused::unsupported("qwen35-d9b", d)),
                }
            },
            rows: [
                (11, [U4g64], Bf16, [], Some(Drafter::DFlash)),
                (12, [U4g64], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen36-35b-a3b",
            mini: false,
            parts: [],
            drafters: [Mtp, DFlash],
            template: template::chatml,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                let w = d.dtype()?;
                match (d.has(Part::Vision), d.drafter) {
                    (false, None) => Ok(Model::a3b(w, d.kv)),
                    (false, Some(Drafter::Mtp)) => Ok(Model::a3b_mtp(w, d.kv)),
                    (false, Some(Drafter::DFlash)) => Ok(Model::a3b_dflash(w, d.kv)),
                    _ => Err(Refused::unsupported("qwen36-35b-a3b", d)),
                }
            },
            rows: [
                (14, [U4g64], Bf16, [], Some(Drafter::DFlash)),
                (15, [U4g64], Bf16, [], Some(Drafter::Mtp)),
                (16, [U4g64], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen35-tiny",
            mini: true,
            parts: [],
            drafters: [],
            template: template::chatml,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                Ok(Model::tiny(d.dtype()?, d.kv))
            },
            rows: [
                (4, [U4g64], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen36-35b-a3b-mini",
            mini: true,
            parts: [],
            drafters: [],
            template: template::chatml,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                Ok(Model::a3b_mini(d.dtype()?, d.kv))
            },
            rows: [
                (18, [U4g64], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen36-35b-a3b-mini64",
            mini: true,
            parts: [],
            drafters: [],
            template: template::chatml,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                Ok(Model::a3b_mini64(d.dtype()?, d.kv))
            },
            rows: [
                (20, [U4g64], Bf16, [], None),
            ],
        },
    ]
}
