pub mod forward;
pub mod import;
pub mod media;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use poem::Dtype;

pub fn entries() -> Vec<crate::catalog::Entry> {
    use crate::catalog::{Drafter, Part, Refused};
    use Dtype::{Bf16, U4g64};
    vec![
        crate::entry! {
            id: "gemma4-26b-a4b",
            mini: false,
            parts: [Vision],
            drafters: [Mtp, DFlash],
            template: template::gemma4,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                let w = d.dtype()?;
                match (d.has(Part::Vision), d.drafter) {
                    (false, None) => Ok(Model::a4b(w, d.kv)),
                    (false, Some(Drafter::Mtp)) => Ok(Model::a4b_mtp(w, d.kv)),
                    (false, Some(Drafter::DFlash)) => Ok(Model::a4b_dflash(w, d.kv)),
                    (true, None) => Ok(Model::a4b_vision(w, d.kv)),
                    _ => Err(Refused::unsupported("gemma4-26b-a4b", d)),
                }
            },
            rows: [
                (0, [U4g64], Bf16, [], Some(Drafter::DFlash)),
                (1, [U4g64], Bf16, [], Some(Drafter::Mtp)),
                (2, [U4g64], Bf16, [], None),
                (17, [U4g64], Bf16, [Vision], None),
            ],
        },
        crate::entry! {
            id: "gemma4-31b",
            mini: false,
            parts: [Vision],
            drafters: [Mtp],
            template: template::gemma4,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                let w = d.dtype()?;
                match (d.has(Part::Vision), d.drafter) {
                    (false, None) => Ok(Model::b31(w, d.kv)),
                    (false, Some(Drafter::Mtp)) => Ok(Model::b31_mtp(w, d.kv)),
                    (true, None) => Ok(Model::b31_vision(w, d.kv)),
                    _ => Err(Refused::unsupported("gemma4-31b", d)),
                }
            },
            rows: [
                (3, [U4g64], Bf16, [], Some(Drafter::Mtp)),
                (4, [U4g64], Bf16, [], None),
                (9, [Bf16], Bf16, [], None),
                (18, [U4g64], Bf16, [Vision], None),
            ],
        },
        crate::entry! {
            id: "gemma4-e4b",
            mini: false,
            parts: [Vision],
            drafters: [Eagle],
            template: template::gemma4,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                let w = d.dtype()?;
                match (d.has(Part::Vision), d.drafter) {
                    (false, None) => Ok(Model::e4b(w, d.kv)),
                    (false, Some(Drafter::Eagle)) => Ok(Model::e4b_eagle(w, d.kv)),
                    (true, None) => Ok(Model::e4b_vision(w, d.kv)),
                    _ => Err(Refused::unsupported("gemma4-e4b", d)),
                }
            },
            rows: [
                (6, [Bf16], Bf16, [], Some(Drafter::Eagle)),
                (7, [Bf16], Bf16, [], None),
                (16, [Bf16], Bf16, [Vision], None),
            ],
        },
        crate::entry! {
            id: "gemma4-e4b-mini-l1",
            mini: true,
            parts: [],
            drafters: [],
            template: template::gemma4,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                Ok(Model::e4b_mini(1, d.dtype()?, d.kv))
            },
            rows: [
                (11, [Bf16], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "gemma4-e4b-mini-l6",
            mini: true,
            parts: [],
            drafters: [],
            template: template::gemma4,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                Ok(Model::e4b_mini(6, d.dtype()?, d.kv))
            },
            rows: [
                (12, [Bf16], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "gemma4-e4b-mini-l24",
            mini: true,
            parts: [],
            drafters: [],
            template: template::gemma4,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                Ok(Model::e4b_mini(24, d.dtype()?, d.kv))
            },
            rows: [
                (13, [Bf16], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "gemma4-e4b-mini-l30",
            mini: true,
            parts: [],
            drafters: [],
            template: template::gemma4,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                Ok(Model::e4b_mini(30, d.dtype()?, d.kv))
            },
            rows: [
                (14, [Bf16], Bf16, [], None),
            ],
        },
        crate::entry! {
            id: "gemma4-e4b-mini-l36",
            mini: true,
            parts: [],
            drafters: [],
            template: template::gemma4,
            tokenizer: &tokenizer::CONTRACT,
            diffusion: None,
            generative: None,
            build: |d| -> Model {
                Ok(Model::e4b_mini(36, d.dtype()?, d.kv))
            },
            rows: [
                (15, [Bf16], Bf16, [], None),
            ],
        },
    ]
}

#[cfg(test)]
mod tests {
    use poem_ir::{Def, Dtype, Linear, Operation, Platform};

    // A row-major u4 plane decodes through the scalar `matmul_affine` arm, whose
    // rate follows SM clock rather than bandwidth (an A100 decodes 31b at half an L40S).
    #[test]
    fn a_u4_trunk_projection_decodes_on_the_tiled_arm() {
        let sku = crate::deployment("gemma4-31b-u4g64-kv-bf16").expect("the 31b u4 row ships");
        let trace = sku.trace(Platform::Cuda);
        let mut row_major = Vec::new();
        for node in &trace.nodes {
            let Operation::Linear(Linear::Matmul { w, .. }) = &node.op else {
                continue;
            };
            let Def::Weight(p) = trace.values[w.0 as usize].def else {
                continue;
            };
            let param = &trace.params[p as usize];
            if param.name.starts_with("layer.") && param.dtype != Dtype::U4g64tiled {
                row_major.push(format!("{} {:?}", param.name, param.dtype));
            }
        }
        assert!(row_major.is_empty(), "{row_major:#?}");
    }
}
