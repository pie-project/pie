pub mod forward;
pub mod import;
pub mod model;
pub mod template;
pub mod tokenizer;

use model::Model;
use model_dsl::Dtype;

/// Identification order: the first row whose import fits the checkpoint wins.
pub fn skus() -> Vec<crate::Sku> {
    crate::skus![
        // Before the plain rows: the head is extra tensors a plain row would
        // ignore, so the row that needs them is asked first.
        (
            "gptoss-20b-dflash",
            1,
            [Dtype::U4g64, Dtype::Mxfp4],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gpt_oss,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b20_dflash(Dtype::U4g64, Dtype::Mxfp4, Dtype::Bf16, tp),
        ),
        (
            "gptoss-20b",
            1,
            [Dtype::U4g64, Dtype::Mxfp4],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gpt_oss,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b20(Dtype::U4g64, Dtype::Mxfp4, Dtype::Bf16, tp),
        ),
        (
            "gptoss-20b",
            1,
            [Dtype::Bf16, Dtype::Mxfp4],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gpt_oss,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b20(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, tp),
        ),
        (
            "gptoss-120b",
            1,
            [Dtype::Bf16, Dtype::Mxfp4],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gpt_oss,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b120(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, tp),
        ),
        (
            "gptoss-120b",
            2,
            [Dtype::Bf16, Dtype::Mxfp4],
            Dtype::Bf16,
            model_dsl::trace_hybrid,
            template::gpt_oss,
            &tokenizer::CONTRACT,
            |tp: u32| Model::b120(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, tp),
        ),
    ]
}
