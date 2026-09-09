use crate::contract::{Expr, ModelContract, TensorContract};
use crate::error::Result;
use crate::file::Metadata;
use crate::types::{DType, Encoding};

pub struct Materialization {
    pub contract: ModelContract,
    pub decoded: Vec<String>,
    pub passthrough: Vec<String>,
    pub meta: Vec<String>,
}

pub fn materialize_contract(metadata: &Metadata) -> Result<Materialization> {
    let mut decoded = Vec::new();
    let mut passthrough = Vec::new();
    let mut tensors = Vec::new();
    let meta = metadata
        .meta_objects()
        .map(|tensor| tensor.name.clone())
        .collect();
    for tensor in metadata.weights() {
        match &tensor.encoding {
            Encoding::Quant(spec) if spec.scheme.is_self_contained() => {
                passthrough.push(tensor.name.clone());
            }
            Encoding::Raw(DType::F16) | Encoding::Raw(DType::F32) => {
                decoded.push(tensor.name.clone());
                tensors.push(TensorContract::new(
                    &tensor.name,
                    Expr::src(&tensor.name).cast(Encoding::Raw(DType::Bf16)),
                    tensor.shape.clone(),
                    Encoding::Raw(DType::Bf16),
                ));
            }
            Encoding::Raw(_) | Encoding::Quant(_) => {
                passthrough.push(tensor.name.clone());
            }
        }
    }
    Ok(Materialization {
        contract: ModelContract {
            alignment: 1,
            tensors,
            groups: Vec::new(),
        },
        decoded,
        passthrough,
        meta,
    })
}
