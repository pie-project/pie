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
                (0, "qwen36-27b-mtp", 1, [U4g64], Bf16, [], Some(Drafter::Mtp)),
                (1, "qwen36-27b-dflash", 1, [U4g64], Bf16, [], Some(Drafter::DFlash)),
                (2, "qwen36-27b", 1, [U4g64], Bf16, [], None),
                (3, "qwen36-27b", 2, [U4g64], Bf16, [], None),
                (21, "qwen36-27b", 1, [Bf16], Bf16, [], Some(Drafter::Mtp)),
                (22, "qwen36-27b", 2, [Bf16], Bf16, [], Some(Drafter::Mtp)),
                (38, "qwen36-27b-vision", 1, [U4g64], Bf16, [Vision], None),
                (39, "qwen36-27b-vision", 1, [Bf16], Bf16, [Vision], Some(Drafter::Mtp)),
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
                (23, "qwen38-27b", 1, [Bf16], Bf16, [], Some(Drafter::Mtp)),
                (24, "qwen38-27b", 2, [Bf16], Bf16, [], Some(Drafter::Mtp)),
                (25, "qwen38-27b-dflash2", 1, [U4g64], Bf16, [], Some(Drafter::DFlash2)),
                (26, "qwen38-27b-dspark", 1, [U4g64], Bf16, [], Some(Drafter::DSpark)),
                (27, "qwen38-27b-mtp", 1, [U4g64], Bf16, [], Some(Drafter::Mtp)),
                (28, "qwen38-27b", 1, [U4g64], Bf16, [], None),
                (29, "qwen38-27b", 2, [U4g64], Bf16, [], None),
                (40, "qwen38-27b-vision", 1, [U4g64], Bf16, [Vision], None),
                (41, "qwen38-27b-vision", 1, [Bf16], Bf16, [Vision], Some(Drafter::Mtp)),
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
                (5, "qwen35-d0.8b", 1, [U4g64], Bf16, [], None),
                (6, "qwen35-d0.8b", 2, [U4g64], Bf16, [], None),
                (33, "qwen35-d0.8b-eagle", 1, [Bf16], Bf16, [], Some(Drafter::Eagle)),
                (34, "qwen35-d0.8b", 1, [Bf16], Bf16, [], None),
                (35, "qwen35-d0.8b", 2, [Bf16], Bf16, [], None),
                (37, "qwen35-d0.8b-vision-eagle", 1, [Bf16], Bf16, [Vision], Some(Drafter::Eagle)),
                (42, "qwen35-d0.8b-vision", 1, [U4g64], Bf16, [Vision], None),
                (43, "qwen35-d0.8b-vision", 1, [Bf16], Bf16, [Vision], None),
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
                (7, "qwen35-d2b", 1, [U4g64], Bf16, [], None),
                (8, "qwen35-d2b", 2, [U4g64], Bf16, [], None),
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
                (31, "qwen35-d3b", 1, [Bf16], Bf16, [], None),
                (32, "qwen35-d3b", 2, [Bf16], Bf16, [], None),
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
                (9, "qwen35-d4b", 1, [U4g64], Bf16, [], None),
                (10, "qwen35-d4b", 2, [U4g64], Bf16, [], None),
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
                (30, "qwen35-a3b", 1, [Bf16], Bf16, [], None),
                (36, "qwen35-a3b", 2, [Bf16], Bf16, [], None),
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
                (11, "qwen35-d9b-dflash", 1, [U4g64], Bf16, [], Some(Drafter::DFlash)),
                (12, "qwen35-d9b", 1, [U4g64], Bf16, [], None),
                (13, "qwen35-d9b", 2, [U4g64], Bf16, [], None),
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
                (14, "qwen36-35b-a3b-dflash", 1, [U4g64], Bf16, [], Some(Drafter::DFlash)),
                (15, "qwen36-35b-a3b-mtp", 1, [U4g64], Bf16, [], Some(Drafter::Mtp)),
                (16, "qwen36-35b-a3b", 1, [U4g64], Bf16, [], None),
                (17, "qwen36-35b-a3b", 2, [U4g64], Bf16, [], None),
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
                (4, "qwen35-tiny", 1, [U4g64], Bf16, [], None),
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
                (18, "qwen36-35b-a3b-mini", 1, [U4g64], Bf16, [], None),
                (19, "qwen36-35b-a3b-mini", 2, [U4g64], Bf16, [], None),
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
                (20, "qwen36-35b-a3b-mini64", 1, [U4g64], Bf16, [], None),
            ],
        },
    ]
}
