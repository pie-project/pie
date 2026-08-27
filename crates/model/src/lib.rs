pub mod contract;
pub mod deepseek_v4;
pub mod gemma_4;
pub mod glm_5;
pub mod gpt_oss;
pub mod kimi_k3;
pub mod qwen_3;
pub mod template;

use model_dsl::Dtype;
use model_loader::contract::ModelContract;
use model_loader::types::{DType, Encoding, QuantScheme, QuantSpec};

pub type Row = (&'static str, model_dsl::TraceFn);

pub type ImportRow = (
    &'static str,
    fn(&ztensor::Source) -> Result<ModelContract, contract::ModelError>,
);

#[must_use]
pub fn catalog() -> Vec<Row> {
    [
        deepseek_v4::CATALOG,
        gemma_4::CATALOG,
        glm_5::CATALOG,
        gpt_oss::CATALOG,
        kimi_k3::CATALOG,
        qwen_3::CATALOG,
    ]
    .concat()
}

#[must_use]
pub fn imports() -> Vec<ImportRow> {
    [
        deepseek_v4::IMPORTS,
        gemma_4::IMPORTS,
        glm_5::IMPORTS,
        gpt_oss::IMPORTS,
        kimi_k3::IMPORTS,
        qwen_3::IMPORTS,
    ]
    .concat()
}

#[must_use]
pub fn trace_of(sku: &str) -> Option<model_dsl::TraceFn> {
    catalog()
        .into_iter()
        .find(|(n, _)| *n == sku)
        .map(|(_, f)| f)
}

#[must_use]
pub fn import_of(
    sku: &str,
) -> Option<fn(&ztensor::Source) -> Result<ModelContract, contract::ModelError>> {
    imports()
        .into_iter()
        .find(|(n, _)| *n == sku)
        .map(|(_, make)| make)
}

pub(crate) fn encoding(dtype: Dtype) -> Encoding {
    match dtype {
        Dtype::Bf16 => Encoding::Raw(DType::BF16),
        Dtype::F16 => Encoding::Raw(DType::F16),
        Dtype::F32 => Encoding::Raw(DType::F32),
        Dtype::I32 => Encoding::Raw(DType::I32),
        Dtype::U32 => Encoding::Raw(DType::U32),
        Dtype::U8 => Encoding::Raw(DType::U8),
        Dtype::I8 => Encoding::Raw(DType::I8),
        Dtype::Fp8E4m3 => Encoding::Raw(DType::F8E4M3),
        Dtype::E8m0 => Encoding::Raw(DType::E8M0),
        Dtype::Mxfp4 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: None,
        }),
        Dtype::Fp4 => panic!(
            "`Dtype::Fp4` names a kv-page quantization scheme, not a stored \
             weight plane; no load contract declares one"
        ),
    }
}

pub fn identify(src: &ztensor::Source) -> Result<&'static str, Unmatched> {
    let mut misses: Vec<(&'static str, String)> = Vec::new();

    for (sku, import) in imports() {
        if is_one_rank_of_a_world(sku) {
            continue;
        }
        match import(src) {
            Ok(_) => return Ok(sku),
            Err(why) => misses.push((sku, why.to_string())),
        }
    }
    Err(Unmatched { misses })
}

fn is_one_rank_of_a_world(sku: &str) -> bool {
    use model_dsl::{Collective, Operation};

    trace_of(sku).is_some_and(|trace| {
        trace(model_dsl::Plane::Cuda)
            .nodes
            .iter()
            .any(|node| match &node.op {
                Operation::Collective(Collective::AllReduce { .. }) => true,
                Operation::Collective(
                    Collective::AllGather { .. } | Collective::ReduceScatter { .. },
                ) => false,
                Operation::Attention(_)
                | Operation::Linear(_)
                | Operation::Elementwise(_)
                | Operation::Layout(_)
                | Operation::CustomCuda(_) => false,
            })
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unmatched {
    pub misses: Vec<(&'static str, String)>,
}

impl std::fmt::Display for Unmatched {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "this checkpoint matches no SKU this build ships")?;
        for (sku, why) in &self.misses {
            write!(f, "\n  {sku}: {why}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Unmatched {}
