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
            fixture: false,
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
                (0, 1, [U4g64], Bf16, [], Some(Drafter::Mtp)),
                (1, 1, [U4g64], Bf16, [], Some(Drafter::DFlash)),
                (2, 1, [U4g64], Bf16, [], None),
                (3, 2, [U4g64], Bf16, [], None),
                (21, 1, [Bf16], Bf16, [], Some(Drafter::Mtp)),
                (22, 2, [Bf16], Bf16, [], Some(Drafter::Mtp)),
                (38, 1, [U4g64], Bf16, [Vision], None),
                (39, 1, [Bf16], Bf16, [Vision], Some(Drafter::Mtp)),
            ],
        },
        crate::entry! {
            id: "qwen38-27b",
            fixture: false,
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
                (23, 1, [Bf16], Bf16, [], Some(Drafter::Mtp)),
                (24, 2, [Bf16], Bf16, [], Some(Drafter::Mtp)),
                (25, 1, [U4g64], Bf16, [], Some(Drafter::DFlash2)),
                (26, 1, [U4g64], Bf16, [], Some(Drafter::DSpark)),
                (27, 1, [U4g64], Bf16, [], Some(Drafter::Mtp)),
                (28, 1, [U4g64], Bf16, [], None),
                (29, 2, [U4g64], Bf16, [], None),
                (40, 1, [U4g64], Bf16, [Vision], None),
                (41, 1, [Bf16], Bf16, [Vision], Some(Drafter::Mtp)),
            ],
        },
        crate::entry! {
            id: "qwen35-d0.8b",
            fixture: false,
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
                (5, 1, [U4g64], Bf16, [], None),
                (6, 2, [U4g64], Bf16, [], None),
                (33, 1, [Bf16], Bf16, [], Some(Drafter::Eagle)),
                (34, 1, [Bf16], Bf16, [], None),
                (35, 2, [Bf16], Bf16, [], None),
                (37, 1, [Bf16], Bf16, [Vision], Some(Drafter::Eagle)),
                (42, 1, [U4g64], Bf16, [Vision], None),
                (43, 1, [Bf16], Bf16, [Vision], None),
            ],
        },
        crate::entry! {
            id: "qwen35-d2b",
            fixture: false,
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
                (7, 1, [U4g64], Bf16, [], None),
                (8, 2, [U4g64], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen35-d3b",
            fixture: false,
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
                (31, 1, [Bf16], Bf16, [], None),
                (32, 2, [Bf16], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen35-d4b",
            fixture: false,
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
                (9, 1, [U4g64], Bf16, [], None),
                (10, 2, [U4g64], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen35-a3b",
            fixture: false,
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
                (30, 1, [Bf16], Bf16, [], None),
                (36, 2, [Bf16], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen35-d9b",
            fixture: false,
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
                (11, 1, [U4g64], Bf16, [], Some(Drafter::DFlash)),
                (12, 1, [U4g64], Bf16, [], None),
                (13, 2, [U4g64], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen36-35b-a3b",
            fixture: false,
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
                (14, 1, [U4g64], Bf16, [], Some(Drafter::DFlash)),
                (15, 1, [U4g64], Bf16, [], Some(Drafter::Mtp)),
                (16, 1, [U4g64], Bf16, [], None),
                (17, 2, [U4g64], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen35-tiny",
            fixture: true,
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
                (4, 1, [U4g64], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen36-35b-a3b-mini",
            fixture: true,
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
                (18, 1, [U4g64], Bf16, [], None),
                (19, 2, [U4g64], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "qwen36-35b-a3b-mini64",
            fixture: true,
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
                (20, 1, [U4g64], Bf16, [], None),
            ],
        },
    ]
}
