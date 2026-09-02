//! Splits a checkpoint's tensors into rewritten (narrowed to BF16) and
//! passthrough (kept in their source encoding) for `pie model import`.
//! Derived purely from the checkpoint's own metadata, no model family.

use crate::file::Metadata;
use crate::contract::{Expr, ModelContract, TensorContract};
use crate::error::Result;
use crate::types::{DType, Encoding};

/// What materializing one checkpoint means, stated before it is done — the
/// shape a `--dry-run` reports. The three sets partition the source's
/// objects exactly.
pub struct Materialization {
    /// The rewritten set's contract; empty `tensors` when nothing is
    /// rewritten.
    pub contract: ModelContract,
    /// Tensors rewritten on the way in: F16 or F32 narrowed to BF16, as this
    /// function leaves it. `pie model import` may later move tensors here
    /// from `passthrough` when a family transform (e.g. llama's Q/K
    /// permutation) can't be a byte copy.
    pub decoded: Vec<String>,
    /// Tensors that pass through byte for byte, encoding and all.
    pub passthrough: Vec<String>,
    /// pie's own metadata objects, when the source is already a pie
    /// artifact. Kept separate from `passthrough` since whether to drop or
    /// carry them over is caller policy, not this loader's decision.
    pub meta: Vec<String>,
}

/// Splits `metadata`'s objects into rewrite, passthrough and metadata, and
/// writes the contract for the first set.
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
            // A self-contained block (scales interleaved with codes) passes
            // through packed; it's decoded at the point that unpacks it, not
            // here.
            Encoding::Quant(spec) if spec.scheme.is_self_contained() => {
                passthrough.push(tensor.name.clone());
            }
            // Every device kernel reads BF16, so F16/F32 is cast on the way
            // in; this can't be a reinterpretation, since F16 and BF16 place
            // the exponent differently. Narrows away exactly the mantissa
            // bits a cold load would also drop.
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

