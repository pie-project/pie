pub mod forward;
pub mod import;
pub mod media;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        (
            "gemma4-26b-a4b-dflash",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a4b_dflash(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "gemma4-26b-a4b-mtp",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a4b_mtp(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "gemma4-26b-a4b",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::a4b(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "gemma4-31b-mtp",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b31_mtp(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "gemma4-31b",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b31(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "gemma4-31b",
            2,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b31(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b-eagle",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b_eagle(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b",
            2,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-31b",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b31(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-31b",
            2,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b31(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b-mini-l1",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b_mini(1, Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b-mini-l6",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b_mini(6, Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b-mini-l24",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b_mini(24, Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b-mini-l30",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b_mini(30, Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b-mini-l36",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT,
            |tp: u32| Model::e4b_mini(36, Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-e4b-vision",
            1,
            [Dtype::Bf16],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT_VISION,
            |tp: u32| Model::e4b_vision(Dtype::Bf16, Dtype::Bf16, tp),
        ),
        (
            "gemma4-26b-a4b-vision",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT_VISION,
            |tp: u32| Model::a4b_vision(Dtype::U4g64, Dtype::Bf16, tp),
        ),
        (
            "gemma4-31b-vision",
            1,
            [Dtype::U4g64],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gemma4,
            &tokenizer::CONTRACT_VISION,
            |tp: u32| Model::b31_vision(Dtype::U4g64, Dtype::Bf16, tp),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use model_ir::{Def, Dtype, Linear, Operation, Platform};

    // A row-major u4 plane decodes through the scalar `matmul_affine` arm, whose
    // rate follows SM clock rather than bandwidth (an A100 decodes 31b at half an L40S).
    #[test]
    fn a_u4_trunk_projection_decodes_on_the_tiled_arm() {
        let sku = crate::sku("gemma4-31b-u4g64-kv-bf16").expect("the 31b u4 row ships");
        let trace = (sku.trace)(Platform::Cuda);
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
