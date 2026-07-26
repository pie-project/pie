//! Contracts to HIGH IR.
//!
//! Every contract is one expression in the algebra of `contract.rs`. The affine
//! part of that expression — slicing, sharding, concatenation, stacking,
//! reshaping — is compiled here, once, into a list of strided copies, and the
//! rest of the compiler never has to reason about it again. What survives above
//! the gather is only what genuinely changes the *bytes*: re-encoding and
//! layout repacking.

use std::collections::HashMap;

use crate::checkpoint::{CheckpointMetadata, RawTensor};
use crate::contract::compile::{Leaf, compile};
use crate::contract::{Expr, ModelContract, Resolver, TensorContract, TensorType};
use crate::error::CompileError;
use crate::ir::{GatherDim, GatherPiece, LayoutExpr, LayoutPlan};
use crate::load_plan::StorageTarget;
use crate::types::{DType, Encoding, ExprId, QuantScheme, TensorDecl, TensorId, encoding_nbytes};

/// How many contiguous stretches one expression may break into before the
/// compiler refuses to keep going.
///
/// This is an allocation guard, not the cost model: a row shard of the largest
/// embedding table in circulation is ~150K stretches and folds to a single
/// copy. An expression that exceeds this is not merely strided, it is a gather,
/// and it should say so rather than materialize a million-entry list.
const MAX_RUNS: usize = 1 << 20;

pub fn plan_from_contracts(
    metadata: &CheckpointMetadata,
    contract: &ModelContract,
    target: &StorageTarget,
) -> Result<LayoutPlan, CompileError> {
    let mut frontend = Frontend {
        metadata,
        target,
        plan: LayoutPlan::new(),
        resolver: Resolver::new(metadata),
        sources: HashMap::new(),
        values: HashMap::new(),
        alignment: contract.alignment.max(1),
        next_generated_tensor: contract.tensors.len() as u32,
    };
    for (index, tensor) in contract.tensors.iter().enumerate() {
        // Rejected here rather than left to collide downstream: two entries
        // under one name would each publish a value and each be finalized, and
        // the plan would carry a name that means two things.
        if frontend.values.contains_key(&tensor.name) {
            return Err(CompileError::InvalidInput(format!(
                "the contract declares '{}' twice",
                tensor.name
            )));
        }
        frontend.lower_contract(tensor, TensorId(index as u32))?;
    }
    Ok(frontend.plan)
}

struct Frontend<'a> {
    metadata: &'a CheckpointMetadata,
    target: &'a StorageTarget,
    plan: LayoutPlan,
    resolver: Resolver<'a>,
    /// Interned `Source` nodes, by checkpoint tensor.
    sources: HashMap<TensorId, ExprId>,
    /// The value each finished contract produced, by name.
    values: HashMap<String, ExprId>,
    /// Byte alignment every materialized buffer must satisfy. A target
    /// property, stated once on the contract rather than on every tensor.
    alignment: u32,
    next_generated_tensor: u32,
}

impl Frontend<'_> {
    fn lower_contract(
        &mut self,
        contract: &TensorContract,
        output_id: TensorId,
    ) -> Result<(), CompileError> {
        let (mut current, mut current_decl) = self
            .lower_source(contract, output_id)
            .map_err(|err| CompileError::InvalidInput(format!("'{}': {err}", contract.name)))?;

        current = self.lower_encoding_change(current, &mut current_decl, contract, output_id)?;

        // The layout and alignment a contract asks for are properties of the
        // declaration, not of the bytes: nothing moves because of them. So they
        // are simply stated on the realized declaration.
        let realized = TensorDecl {
            id: output_id,
            name: contract.name.clone(),
            shape: current_decl.shape.clone(),
            encoding: contract.encoding.clone(),
            alignment: self.alignment,
        };

        self.values.insert(contract.name.clone(), current);
        self.resolver.publish(
            &contract.name,
            TensorType {
                shape: realized.shape.clone(),
                encoding: realized.encoding.clone(),
            },
        );
        let node = self.plan.push(LayoutExpr::Realize {
            input: current,
            runtime_name: contract.name.clone(),
            decl: realized,
        });
        self.plan.outputs.push(node);
        Ok(())
    }

    /// Lower everything below the encoding change: the affine expression, or
    /// the one escape hatch that is not affine.
    fn lower_source(
        &mut self,
        contract: &TensorContract,
        output_id: TensorId,
    ) -> Result<(ExprId, TensorDecl), CompileError> {
        match &contract.expr {
            // A repack is opaque: the layout transform declares its own result
            // and the compiler does not model the permutation.
            Expr::Repack { src, spec, out } => {
                let Expr::Src(name) = src.as_ref() else {
                    return Err(CompileError::InvalidInput(
                        "Repack must read a checkpoint tensor directly".to_string(),
                    ));
                };
                let raw = self.raw_by_name(name)?;
                let input = self.source_node(raw)?;
                let decl = TensorDecl {
                    id: output_id,
                    name: contract.name.clone(),
                    shape: out.shape.clone(),
                    encoding: out.encoding.clone(),
                    alignment: self.alignment,
                };
                let node = self.plan.push(LayoutExpr::Repack {
                    input,
                    spec: *spec,
                    decl: decl.clone(),
                });
                Ok((node, decl))
            }
            _ => self.lower_affine(contract, output_id),
        }
    }

    fn lower_affine(
        &mut self,
        contract: &TensorContract,
        output_id: TensorId,
    ) -> Result<(ExprId, TensorDecl), CompileError> {
        // Sharding is resolved here and nowhere else: below this line the
        // expression means the same thing on every rank.
        let expr = self.resolver.specialize(
            contract.expr.clone(),
            self.target.tp_rank,
            self.target.tp_size,
            &contract.name,
        )?;
        let ty = self.resolver.infer(&expr)?;
        if ty.shape != contract.shape {
            return Err(CompileError::InvalidInput(format!(
                "declared shape {:?} does not match source shape {:?}",
                contract.shape, ty.shape
            )));
        }

        let lowering = compile(&expr, self.resolver.checked(), MAX_RUNS)?;
        if lowering.needs_zero_fill() {
            return Err(CompileError::InvalidInput(
                "padding needs a zero-fill instruction, which the low IR does not have yet"
                    .to_string(),
            ));
        }
        let byte_pieces = lowering.byte_pieces(&ty.encoding)?;

        let mut inputs = Vec::with_capacity(lowering.leaves.len());
        for leaf in &lowering.leaves {
            inputs.push(match leaf {
                Leaf::Checkpoint(name) => {
                    let raw = self.raw_by_name(name)?;
                    self.source_node(raw)?
                }
                Leaf::Contract(name) => *self.values.get(name).ok_or_else(|| {
                    CompileError::InvalidInput(format!("reads '{name}' before it exists"))
                })?,
            });
        }

        let decl = TensorDecl {
            id: output_id,
            name: contract.name.clone(),
            shape: ty.shape.clone(),
            encoding: ty.encoding.clone(),
            alignment: self.alignment,
        };
        let node = self.plan.push(LayoutExpr::Gather {
            inputs,
            pieces: byte_pieces
                .into_iter()
                .map(|piece| GatherPiece {
                    input: piece.leaf as u32,
                    src_offset: piece.src_offset,
                    dst_offset: piece.dst_offset,
                    dims: piece
                        .dims
                        .into_iter()
                        .map(|dim| GatherDim {
                            count: dim.count,
                            src_stride: dim.src_stride,
                            dst_stride: dim.dst_stride,
                        })
                        .collect(),
                })
                .collect(),
            zero_fill: false,
            decl: decl.clone(),
        });
        Ok((node, decl))
    }

    fn lower_encoding_change(
        &mut self,
        input: ExprId,
        current_decl: &mut TensorDecl,
        contract: &TensorContract,
        output_id: TensorId,
    ) -> Result<ExprId, CompileError> {
        if current_decl.encoding == contract.encoding {
            return Ok(input);
        }
        let node = match (current_decl.encoding.clone(), contract.encoding.clone()) {
            (Encoding::Raw(_), Encoding::Raw(dtype)) => {
                let mut decl = current_decl.clone();
                decl.encoding = Encoding::Raw(dtype);
                *current_decl = decl.clone();
                self.plan.push(LayoutExpr::Cast { input, dtype, decl })
            }
            (Encoding::Quant(source), Encoding::Raw(dtype)) => {
                if source.logical_dtype != dtype {
                    return Err(CompileError::InvalidInput(format!(
                        "runtime tensor '{}' requests raw {:?} from quantized {:?}",
                        contract.name, dtype, source.logical_dtype
                    )));
                }
                let mut decl = current_decl.clone();
                decl.encoding = Encoding::Raw(dtype);
                *current_decl = decl.clone();
                self.plan.push(LayoutExpr::Decode {
                    metadata: Vec::new(),
                    scheme: source.scheme,
                    data: input,
                    decl,
                })
            }
            (Encoding::Raw(_), Encoding::Quant(target)) => {
                let mut decl = current_decl.clone();
                decl.id = output_id;
                decl.name = contract.name.clone();
                decl.encoding = Encoding::Quant(target.clone());
                *current_decl = decl.clone();
                let metadata_outputs = self.quant_metadata_outputs(contract, &decl);
                self.plan.push(LayoutExpr::Encode {
                    scheme: target.scheme,
                    input,
                    metadata_outputs,
                    decl,
                })
            }
            (Encoding::Quant(source), Encoding::Quant(target)) => {
                let mut decl = current_decl.clone();
                decl.encoding = Encoding::Quant(target.clone());
                *current_decl = decl.clone();
                self.plan.push(LayoutExpr::Transcode {
                    metadata: Vec::new(),
                    from: source.scheme,
                    to: target.scheme,
                    data: input,
                    metadata_outputs: Vec::new(),
                    decl,
                })
            }
        };
        Ok(node)
    }

    fn quant_metadata_outputs(
        &mut self,
        contract: &TensorContract,
        quant_decl: &TensorDecl,
    ) -> Vec<TensorDecl> {
        let Encoding::Quant(spec) = &quant_decl.encoding else {
            return Vec::new();
        };
        if quant_decl.shape.len() != 2 {
            return Vec::new();
        }
        let (name, shape, encoding) = match spec.scheme {
            QuantScheme::Fp8E4M3 | QuantScheme::Int8Symmetric => (
                format!("{}_scale_inv", contract.name),
                vec![quant_decl.shape[0]],
                Encoding::Raw(DType::F32),
            ),
            QuantScheme::Mxfp4E2M1E8M0 => {
                // E8M0 block scale: one uint8 per 32-element block along K. The
                // encode-tile kernel writes a row-major `[rows, cols/32]` byte
                // tensor. The name matches GPT-OSS so downstream lookups of
                // `*.weight_scale` find it.
                let cols = quant_decl.shape[1];
                if cols % 32 != 0 {
                    return Vec::new();
                }
                (
                    format!("{}_scale", contract.name),
                    vec![quant_decl.shape[0], cols / 32],
                    Encoding::Raw(DType::U8),
                )
            }
            _ => return Vec::new(),
        };
        let id = TensorId(self.next_generated_tensor);
        self.next_generated_tensor = self.next_generated_tensor.saturating_add(1);
        vec![TensorDecl {
            id,
            name,
            shape,
            encoding,
            alignment: self.alignment,
        }]
    }

    /// The interned `Source` node for a checkpoint tensor.
    fn source_node(&mut self, id: TensorId) -> Result<ExprId, CompileError> {
        if let Some(node) = self.sources.get(&id) {
            return Ok(*node);
        }
        let raw = self.raw_by_id(id)?;
        let decl = source_decl(raw);
        let node = self.plan.push(LayoutExpr::Source { tensor: id, decl });
        self.sources.insert(id, node);
        Ok(node)
    }

    fn raw_by_id(&self, id: TensorId) -> Result<&RawTensor, CompileError> {
        self.metadata.tensor(id).ok_or_else(|| {
            CompileError::InvalidInput(format!("references missing source tensor {}", id.0))
        })
    }

    fn raw_by_name(&self, name: &str) -> Result<TensorId, CompileError> {
        self.metadata
            .tensor_by_name(name)
            .map(|raw| raw.id)
            .ok_or_else(|| {
                CompileError::InvalidInput(format!("checkpoint has no tensor named '{name}'"))
            })
    }
}

fn source_decl(raw: &RawTensor) -> TensorDecl {
    TensorDecl {
        id: raw.id,
        name: raw.name.clone(),
        shape: raw.shape.clone(),
        encoding: crate::types::normalize_encoding(&raw.encoding),
        alignment: 1,
    }
}

pub fn runtime_bytes(shape: &[i64], encoding: &Encoding) -> Result<u64, CompileError> {
    encoding_nbytes(shape, encoding)
        .ok_or_else(|| CompileError::InvalidInput("runtime tensor byte size overflow".to_string()))
}
