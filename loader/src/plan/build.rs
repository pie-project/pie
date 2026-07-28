//! Contracts to a load plan, in one pass.
//!
//! This is where the middle IR used to be. A contract became a `LayoutPlan` of
//! `LayoutExpr` nodes, the optimizer rewrote it, a second type checker
//! re-derived the shapes `contract::infer` had already proved, and only then
//! did a `StorageCompiler` walk the nodes and emit instructions.
//!
//! The IR earned none of that. Every contract lowered to the *same linear
//! chain* — an affine gather (or the one escape hatch), at most one encoding
//! change, and a name — so there was never a graph to rewrite. The goldens
//! agreed: all fourteen recorded `rewrites: 0` and an expression count the
//! optimizer left exactly as it found it.
//!
//! So the chain is emitted directly:
//!
//! ```text
//! contract::Expr --specialize--> Expr --compile--> Lowering --> instructions
//!                                                      |
//!                                              cost() picks the lowering
//! ```
//!
//! Two shapes must stay zero-copy or the plan regresses badly, and both are
//! decided here rather than recovered by a later pass:
//!
//! * a single copy off a checkpoint tensor stays a lazy [`SourceView`], so a
//!   shard is read straight out of the file at an offset;
//! * a single contiguous copy out of an already-materialized buffer becomes a
//!   `CreateView`, so republishing a slice of a fused tensor costs nothing.

use std::collections::{HashMap, HashSet};

use crate::checkpoint::{CheckpointMetadata, RawTensor, Sources};
use crate::contract::compile::{Leaf, Lowering, compile};
use crate::contract::infer::Resolver;
use crate::contract::{Expr, ModelContract, TensorContract, TensorType};
use crate::error::{Error, OrOverflow, Result};
use crate::extent::Extent;
use crate::plan::geometry::{
    full_dest_extent, narrow_repack_source, repack_stage_bytes, storage_extent_for_shape,
    strided_physical_source_bytes,
};
use crate::plan::{
    BufferDecl, CheckpointFileDecl, DestExtent, LoadPlan, SourceExtent, SourceTensorDecl,
    StorageInstr, StorageTarget, TileMapKind, TileSpec, TransformSpec,
};
use crate::types::{
    BufferId, DType, Encoding, InstrId, QuantScheme, RepackSpec, TensorDecl, TensorId,
    encoding_dense_element_bytes, encoding_nbytes,
};

/// How many contiguous stretches one expression may break into before the
/// compiler refuses to keep going.
///
/// An allocation guard, not the cost model: a row shard of the largest
/// embedding table in circulation is ~150K stretches and folds to a single
/// copy. An expression that exceeds this is not merely strided, it is a gather,
/// and it should say so rather than materialize a million-entry list.
const MAX_RUNS: usize = 1 << 20;

/// Turn a checked contract into the plan that satisfies it.
pub fn build(
    metadata: &CheckpointMetadata,
    contract: &ModelContract,
    target: StorageTarget,
) -> Result<LoadPlan> {
    let sources = Sources::new(metadata);
    let mut builder = Builder {
        sources: &sources,
        program: LoadPlan::empty(target),
        resolver: Resolver::new(&sources),
        values: HashMap::new(),
        finalized: HashSet::new(),
        alignment: contract.alignment.max(1),
        next_buffer: 0,
        next_instr: 0,
        next_generated_tensor: contract.tensors.len() as u32,
        tensor_at: HashMap::new(),
    };
    builder.program.files = metadata
        .files
        .iter()
        .map(|file| CheckpointFileDecl {
            id: file.id,
            path: file.path.clone(),
            size_bytes: file.size_bytes,
            format: file.format,
        })
        .collect();
    builder.program.sources = metadata
        .tensors
        .iter()
        .map(|tensor| SourceTensorDecl {
            id: tensor.id,
            name: tensor.name.clone(),
            file_id: tensor.file_id,
            file_offset: tensor.file_offset,
            span_bytes: tensor.span_bytes,
            shape: tensor.shape.clone(),
            encoding: tensor.encoding.clone(),
        })
        .collect();

    for (index, tensor) in contract.tensors.iter().enumerate() {
        // Rejected here rather than left to collide downstream: two entries
        // under one name would each publish a value and each be finalized, and
        // the plan would carry a name that means two things.
        if builder.values.contains_key(&tensor.name) {
            return Err(Error::Contract(format!(
                "the contract declares '{}' twice",
                tensor.name
            )));
        }
        builder.tensor(tensor, TensorId(index as u32))?;
    }
    Ok(builder.program)
}

/// Where a value lives once its contract has been built.
#[derive(Clone, Debug)]
pub(crate) enum Value {
    /// Still on disk, as a window on a checkpoint tensor. Nothing has been
    /// copied yet, and nothing will be unless something forces it.
    Source(SourceView),
    /// Materialized in a device buffer.
    Buffer(BufferId),
}

/// A lazy window on a checkpoint tensor.
#[derive(Clone, Debug)]
pub(crate) struct SourceView {
    pub tensor_id: TensorId,
    pub shape: Vec<i64>,
    pub encoding: Encoding,
    pub offset_bytes: u64,
    pub stride: Extent,
}

struct Builder<'a> {
    sources: &'a Sources<'a>,
    program: LoadPlan,
    resolver: Resolver<'a>,
    /// What each finished contract produced, by name. Also the scope
    /// `Leaf::Contract` resolves against.
    values: HashMap<String, Value>,
    finalized: HashSet<String>,
    /// Byte alignment every materialized buffer must satisfy. A target
    /// property, stated once on the contract rather than on every tensor.
    alignment: u32,
    next_buffer: u32,
    next_instr: u32,
    next_generated_tensor: u32,
    /// Position of each declaration in `program.tensors`. The table is filled
    /// in as contracts are built and looked up on every promotion, so a scan
    /// here is quadratic in the contract's size.
    tensor_at: HashMap<TensorId, usize>,
}

impl Builder<'_> {
    /// One contract: its expression, its encoding, and its name.
    fn tensor(&mut self, contract: &TensorContract, id: TensorId) -> Result<()> {
        let (value, decl) = match &contract.expr {
            // A repack is opaque: the layout transform declares its own result
            // and the compiler does not model the permutation.
            Expr::Repack { src, spec, out } => {
                let Expr::Src(name) = src.as_ref() else {
                    return Err(Error::Contract(
                        "Repack must read a checkpoint tensor directly".to_string(),
                    ));
                };
                let decl = TensorDecl {
                    id,
                    name: contract.name.clone(),
                    shape: out.shape.clone(),
                    encoding: out.encoding.clone(),
                    alignment: self.alignment,
                };
                let source = self.source_view(name)?;
                let value = self.repack(Value::Source(source), *spec, &decl)?;
                (value, decl)
            }
            _ => self
                .affine(contract, id)
                .map_err(|err| annotate(err, &contract.name))?,
        };

        let (value, shape) = self.encoding_change(value, decl, contract, id)?;

        // The layout and alignment a contract asks for are properties of the
        // declaration, not of the bytes: nothing moves because of them. So they
        // are simply stated on the realized declaration.
        let realized = TensorDecl {
            id,
            name: contract.name.clone(),
            shape,
            encoding: contract.encoding.clone(),
            alignment: self.alignment,
        };
        self.resolver.publish(
            &contract.name,
            TensorType {
                shape: realized.shape.clone(),
                encoding: realized.encoding.clone(),
            },
        );
        let value = self.realize(value, &realized)?;
        self.values.insert(contract.name.clone(), value);
        Ok(())
    }

    /// Solve the affine expression and emit the copies that satisfy it.
    fn affine(&mut self, contract: &TensorContract, id: TensorId) -> Result<(Value, TensorDecl)> {
        // Sharding is resolved here and nowhere else: below this line the
        // expression means the same thing on every rank.
        let expr = self.resolver.specialize(
            contract.expr.clone(),
            self.program.target.tp_rank,
            self.program.target.tp_size,
            &contract.name,
        )?;
        let ty = self.resolver.infer(&expr)?;
        // A declaration that declined to predict the shape has nothing to check
        // here; one that predicted is held to it.
        if let Some(declared) = &contract.shape
            && ty.shape != *declared
        {
            return Err(Error::Contract(format!(
                "declares shape {declared:?} but its expression yields {:?}",
                ty.shape
            )));
        }

        let lowering = compile(&expr, self.resolver.checked(), MAX_RUNS)?;
        let decl = TensorDecl {
            id,
            name: contract.name.clone(),
            shape: ty.shape.clone(),
            encoding: ty.encoding.clone(),
            alignment: self.alignment,
        };
        let value = self.copies(&lowering, &decl)?;
        Ok((value, decl))
    }

    /// Emit the rectangular copies of a solved expression.
    fn copies(&mut self, lowering: &Lowering, decl: &TensorDecl) -> Result<Value> {
        let output_bytes = encoding_nbytes(&decl.shape, &decl.encoding)
            .or_overflow(format!("'{}' size overflow", decl.name))?;
        let rects = lowering.byte_pieces(&decl.encoding)?;

        // `spec.md` §3.3's cost model, deciding the lowering. A cost of 1 is a
        // single rectangle covering the whole destination, and the cheapest way
        // to execute one copy is not to: the tensor can alias the checkpoint
        // bytes, or view a buffer that already holds them. Everything above 1
        // has to move something.
        //
        // A hole is priced into `cost`, so `cost == 1` already excludes a
        // padded destination — aliasing one would hand back bytes the
        // expression says are zero. The check is spelled out anyway, because a
        // silent wrong answer here is indistinguishable from a correct one.
        if lowering.cost() == 1
            && let [rect] = rects.as_slice()
            && !lowering.needs_zero_fill()
            && rect.dst_offset == 0
            && rect.bytes() == output_bytes
        {
            match self.leaf(lowering, rect.leaf)? {
                Value::Source(source) if source_is_dense(&source)? => {
                    let (stride, _) = rect.split()?;
                    return Ok(Value::Source(SourceView {
                        tensor_id: source.tensor_id,
                        shape: decl.shape.clone(),
                        encoding: decl.encoding.clone(),
                        offset_bytes: source.offset_bytes + rect.src_offset,
                        stride,
                    }));
                }
                Value::Buffer(buffer) if rect.is_byte_run() => {
                    let out = self.declare_view_buffer(decl);
                    let instr = self.next_instr();
                    self.program.instrs.push(StorageInstr::CreateView {
                        id: instr,
                        input: buffer,
                        output: out,
                        view: DestExtent {
                            buffer: out,
                            offset: rect.src_offset,
                            stride: storage_extent_for_shape(&decl.shape, &decl.encoding)?,
                        },
                    });
                    self.program.schedule.push(instr);
                    return Ok(Value::Buffer(out));
                }
                _ => {}
            }
        }

        let out = self.allocate(decl, true)?;
        if lowering.needs_zero_fill() {
            let instr = self.next_instr();
            self.program.instrs.push(StorageInstr::Fill {
                id: instr,
                buffer: out,
            });
            self.program.schedule.push(instr);
        }
        for rect in &rects {
            match self.leaf(lowering, rect.leaf)? {
                Value::Source(source) => {
                    let raw = self.raw(source.tensor_id)?;
                    let (file_id, tensor_id, base) = (raw.file_id, raw.id, raw.file_offset);
                    let (src_stride, dst_stride) = rect.split()?;
                    let instr = self.next_instr();
                    self.program.instrs.push(StorageInstr::ExtentWrite {
                        id: instr,
                        source: SourceExtent {
                            file_id,
                            tensor_id,
                            file_offset: base + source.offset_bytes + rect.src_offset,
                            span_bytes: rect.bytes(),
                            stride: src_stride,
                            dtype: source.encoding.dtype(),
                        },
                        dest: DestExtent {
                            buffer: out,
                            offset: rect.dst_offset,
                            stride: dst_stride,
                        },
                    });
                    self.program.schedule.push(instr);
                }
                Value::Buffer(buffer) => {
                    let (shape, encoding) = self.leaf_type(lowering, rect.leaf)?;
                    let input_bytes = encoding_nbytes(&shape, &encoding);
                    if !rect.is_byte_run()
                        || rect.src_offset != 0
                        || input_bytes != Some(rect.bytes())
                    {
                        return Err(Error::Unsupported(format!(
                            "'{}' takes a strided or partial slice of a computed buffer; \
                             only whole-buffer moves are lowered",
                            decl.name
                        )));
                    }
                    self.tile_map(
                        TileMapKind::Reblock,
                        None,
                        Some(DestExtent {
                            buffer: out,
                            offset: rect.dst_offset,
                            stride: storage_extent_for_shape(&shape, &encoding)?,
                        }),
                        vec![buffer],
                        vec![out],
                        TransformSpec::default(),
                    );
                }
            }
        }
        Ok(Value::Buffer(out))
    }

    /// Re-encode a value to the encoding its contract declares.
    ///
    /// The encoding change is **not an operator in the algebra** — it is the
    /// difference between the type the expression yields and the type the
    /// record declares, so which kernel runs is derived from that pair rather
    /// than written down by an author. See `spec.md` §3.
    fn encoding_change(
        &mut self,
        value: Value,
        decl: TensorDecl,
        contract: &TensorContract,
        id: TensorId,
    ) -> Result<(Value, Vec<i64>)> {
        if decl.encoding == contract.encoding {
            return Ok((value, decl.shape));
        }
        let shape = decl.shape.clone();
        let value = match (decl.encoding.clone(), contract.encoding.clone()) {
            (Encoding::Raw(_), Encoding::Raw(dtype)) => {
                let out = TensorDecl {
                    encoding: Encoding::Raw(dtype),
                    ..decl
                };
                self.transform(
                    value,
                    &out,
                    TileMapKind::Cast,
                    TransformSpec {
                        to: Some(dtype_to_quant_marker(dtype)),
                        ..TransformSpec::default()
                    },
                )?
            }
            (Encoding::Quant(source), Encoding::Raw(dtype)) => {
                if source.logical_dtype != dtype {
                    return Err(Error::Contract(format!(
                        "runtime tensor '{}' requests raw {dtype:?} from quantized {:?}",
                        contract.name, source.logical_dtype
                    )));
                }
                let out = TensorDecl {
                    encoding: Encoding::Raw(dtype),
                    ..decl
                };
                self.transform(
                    value,
                    &out,
                    TileMapKind::Decode,
                    TransformSpec {
                        from: Some(source.scheme),
                        ..TransformSpec::default()
                    },
                )?
            }
            (Encoding::Raw(_), Encoding::Quant(target)) => {
                let out = TensorDecl {
                    id,
                    name: contract.name.clone(),
                    encoding: Encoding::Quant(target.clone()),
                    ..decl
                };
                let metadata = self.quant_metadata_outputs(contract, &out);
                self.encode(value, &out, target.scheme, &metadata)?
            }
            (Encoding::Quant(source), Encoding::Quant(target)) => {
                let out = TensorDecl {
                    encoding: Encoding::Quant(target.clone()),
                    ..decl
                };
                self.transform(
                    value,
                    &out,
                    TileMapKind::Transcode,
                    TransformSpec {
                        from: Some(source.scheme),
                        to: Some(target.scheme),
                        ..TransformSpec::default()
                    },
                )?
            }
        };
        Ok((value, shape))
    }

    /// A one-in, one-out transform kernel over a whole tensor.
    fn transform(
        &mut self,
        value: Value,
        decl: &TensorDecl,
        kind: TileMapKind,
        transform: TransformSpec,
    ) -> Result<Value> {
        let out = self.allocate(decl, true)?;
        let mut inputs = Vec::new();
        let source = match value {
            Value::Source(source) => Some(self.source_extent(&source)?),
            Value::Buffer(buffer) => {
                inputs.push(buffer);
                None
            }
        };
        let transform = self.with_block_scale_source(transform, source.as_ref());
        self.tile_map(
            kind,
            source,
            Some(full_dest_extent(out, decl)?),
            inputs,
            vec![out],
            transform,
        );
        Ok(Value::Buffer(out))
    }

    /// Quantize, which unlike every other transform may publish extra tensors:
    /// the block scales its output cannot be read without.
    fn encode(
        &mut self,
        value: Value,
        decl: &TensorDecl,
        scheme: QuantScheme,
        metadata: &[TensorDecl],
    ) -> Result<Value> {
        if metadata.is_empty() {
            return self.transform(
                value,
                decl,
                TileMapKind::Encode,
                TransformSpec {
                    to: Some(scheme),
                    ..TransformSpec::default()
                },
            );
        }
        let out = self.allocate(decl, false)?;
        let mut inputs = Vec::new();
        let source = match value {
            Value::Source(source) => Some(self.source_extent(&source)?),
            Value::Buffer(buffer) => {
                inputs.push(buffer);
                None
            }
        };
        let transform = self.with_block_scale_source(
            TransformSpec {
                to: Some(scheme),
                ..TransformSpec::default()
            },
            source.as_ref(),
        );
        let mut outputs = Vec::with_capacity(metadata.len() + 1);
        outputs.push(out);
        for decl in metadata {
            outputs.push(self.allocate(decl, false)?);
        }
        self.tile_map(
            TileMapKind::Encode,
            source,
            None,
            inputs,
            outputs.clone(),
            transform,
        );
        for (decl, buffer) in metadata.iter().zip(outputs.iter().skip(1)) {
            self.finalize(*buffer, &decl.name)?;
        }
        Ok(Value::Buffer(out))
    }

    /// The one transform the algebra cannot denote (`spec.md` §3.5).
    fn repack(&mut self, value: Value, spec: RepackSpec, decl: &TensorDecl) -> Result<Value> {
        let out = self.allocate(decl, true)?;
        let mut inputs = Vec::new();
        let mut repack = spec;
        let (source, input_bytes) = match value {
            Value::Source(source) => {
                let (source, narrowed) = narrow_repack_source(source, spec)?;
                repack = narrowed;
                let extent = self.source_extent(&source)?;
                let bytes = extent.span_bytes;
                (Some(extent), bytes)
            }
            Value::Buffer(buffer) => {
                let bytes = self.buffer_bytes(buffer)?;
                inputs.push(buffer);
                (None, bytes)
            }
        };
        let stage_bytes = repack_stage_bytes(repack)?;
        self.tile_map(
            TileMapKind::Repack,
            source,
            Some(full_dest_extent(out, decl)?),
            inputs,
            vec![out],
            TransformSpec {
                repack,
                scratch_bytes: input_bytes
                    .checked_add(stage_bytes)
                    .or_overflow("Repack scratch bytes")?,
                ..TransformSpec::default()
            },
        );
        Ok(Value::Buffer(out))
    }

    /// Give a value the name the driver will bind it under.
    fn realize(&mut self, value: Value, decl: &TensorDecl) -> Result<Value> {
        let buffer = match value {
            Value::Buffer(buffer) => {
                self.promote_buffer(buffer, decl)?;
                buffer
            }
            Value::Source(source) => {
                let buffer = self.allocate(decl, false)?;
                self.extent_write(source, buffer, &decl.shape)?;
                buffer
            }
        };
        self.finalize(buffer, &decl.name)?;
        Ok(Value::Buffer(buffer))
    }

    fn finalize(&mut self, buffer: BufferId, name: &str) -> Result<()> {
        if !self.finalized.insert(name.to_string()) {
            return Err(Error::Contract(format!(
                "duplicate runtime tensor '{name}'"
            )));
        }
        let instr = self.next_instr();
        self.program.instrs.push(StorageInstr::Finalize {
            id: instr,
            tensor: buffer,
            name: name.to_string(),
        });
        self.program.schedule.push(instr);
        Ok(())
    }

    /// The value one of a lowering's leaves refers to.
    fn leaf(&mut self, lowering: &Lowering, index: usize) -> Result<Value> {
        let leaf = lowering
            .leaves
            .get(index)
            .ok_or_else(|| Error::Internal(format!("lowering has no leaf {index}")))?;
        match leaf {
            Leaf::Checkpoint(name) => Ok(Value::Source(self.source_view(name)?)),
            Leaf::Contract(name) => self
                .values
                .get(name)
                .cloned()
                .ok_or_else(|| Error::Contract(format!("reads '{name}' before it exists"))),
        }
    }

    /// The shape and encoding of one of a lowering's leaves.
    fn leaf_type(&self, lowering: &Lowering, index: usize) -> Result<(Vec<i64>, Encoding)> {
        let leaf = lowering
            .leaves
            .get(index)
            .ok_or_else(|| Error::Internal(format!("lowering has no leaf {index}")))?;
        let ty =
            self.resolver.checked().type_of(leaf).ok_or_else(|| {
                Error::Contract(format!("'{}' has no resolved type", leaf.name()))
            })?;
        Ok((ty.shape.clone(), ty.encoding.clone()))
    }

    /// A lazy whole-tensor window on a checkpoint tensor.
    ///
    /// The two checks below used to sit on the middle IR's `Source` node, which
    /// is why they read like assertions about a node rather than about a file.
    fn source_view(&mut self, name: &str) -> Result<SourceView> {
        let raw = self
            .sources
            .by_name(name)
            .ok_or_else(|| Error::Checkpoint(format!("has no tensor named '{name}'")))?;
        let encoding = crate::types::normalize_encoding(&raw.encoding);
        if encoding_dense_element_bytes(&encoding).is_none()
            && encoding_nbytes(&raw.shape, &encoding) != Some(raw.span_bytes)
        {
            return Err(Error::Checkpoint(format!(
                "packed tensor '{name}' has non-affine physical size {}",
                raw.span_bytes
            )));
        }
        Ok(SourceView {
            tensor_id: raw.id,
            shape: raw.shape.clone(),
            encoding: encoding.clone(),
            offset_bytes: 0,
            stride: storage_extent_for_shape(&raw.shape, &encoding)?,
        })
    }

    /// The tensors a quantizing encode must publish alongside its output.
    fn quant_metadata_outputs(
        &mut self,
        contract: &TensorContract,
        decl: &TensorDecl,
    ) -> Vec<TensorDecl> {
        let Encoding::Quant(spec) = &decl.encoding else {
            return Vec::new();
        };
        if decl.shape.len() != 2 {
            return Vec::new();
        }
        let (name, shape, encoding) = match spec.scheme {
            QuantScheme::Fp8E4M3 | QuantScheme::Int8Symmetric => (
                format!("{}_scale_inv", contract.name),
                vec![decl.shape[0]],
                Encoding::Raw(DType::F32),
            ),
            QuantScheme::Mxfp4E2M1E8M0 => {
                // E8M0 block scale: one uint8 per 32-element block along K. The
                // encode-tile kernel writes a row-major `[rows, cols/32]` byte
                // tensor. The name matches GPT-OSS so downstream lookups of
                // `*.weight_scale` find it.
                let cols = decl.shape[1];
                if cols % 32 != 0 {
                    return Vec::new();
                }
                (
                    format!("{}_scale", contract.name),
                    vec![decl.shape[0], cols / 32],
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

    fn tile_map(
        &mut self,
        kind: TileMapKind,
        source: Option<SourceExtent>,
        dest: Option<DestExtent>,
        inputs: Vec<BufferId>,
        outputs: Vec<BufferId>,
        transform: TransformSpec,
    ) {
        let instr = self.next_instr();
        self.program.instrs.push(StorageInstr::TileMap {
            id: instr,
            kind,
            source,
            dest,
            inputs,
            outputs,
            tile: TileSpec {
                max_tile_bytes: self.program.target.max_tile_bytes,
                // A budget, not yet a decision: the tile pass turns it into a
                // row count once the whole plan exists.
                rows_per_tile: 0,
            },
            transform,
        });
        self.program.schedule.push(instr);
    }

    fn extent_write(&mut self, source: SourceView, dest: BufferId, shape: &[i64]) -> Result<()> {
        let source_extent = self.source_extent(&source)?;
        let bytes =
            encoding_nbytes(shape, &source.encoding).or_overflow("extent write byte size")?;
        let instr = self.next_instr();
        self.program.instrs.push(StorageInstr::ExtentWrite {
            id: instr,
            source: SourceExtent {
                span_bytes: bytes,
                ..source_extent
            },
            dest: DestExtent {
                buffer: dest,
                offset: 0,
                stride: storage_extent_for_shape(shape, &source.encoding)?,
            },
        });
        self.program.schedule.push(instr);
        Ok(())
    }

    /// The checkpoint tensor holding `source`'s block scales, if it has any.
    ///
    /// Block-scaled FP8 (DeepSeek-V3 and its descendants) ships the factors in a
    /// sibling tensor whose name is the payload's plus `_scale_inv`. The naming
    /// convention is the checkpoint's, so it is read off the tensor table here
    /// rather than reconstructed by whoever consumes the instruction.
    fn with_block_scale_source(
        &self,
        mut transform: TransformSpec,
        source: Option<&SourceExtent>,
    ) -> TransformSpec {
        transform.metadata_source = source.and_then(|source| {
            let raw = self.sources.tensor(source.tensor_id)?;
            if !matches!(raw.encoding, Encoding::Raw(DType::F8E4M3)) {
                return None;
            }
            self.sources
                .by_name(&format!("{}_scale_inv", raw.name))
                .map(|scale| scale.id)
        });
        transform
    }

    fn source_extent(&self, source: &SourceView) -> Result<SourceExtent> {
        let raw = self.raw(source.tensor_id)?;
        let span_bytes = encoding_nbytes(&source.shape, &source.encoding)
            .or_overflow("source extent byte size")?;
        let physical_bytes = strided_physical_source_bytes(&source.stride)?;
        if source.offset_bytes + physical_bytes > raw.span_bytes {
            return Err(Error::Contract(format!(
                "source extent for '{}' exceeds tensor span",
                raw.name
            )));
        }
        Ok(SourceExtent {
            file_id: raw.file_id,
            tensor_id: raw.id,
            file_offset: raw.file_offset + source.offset_bytes,
            span_bytes,
            stride: source.stride.clone(),
            dtype: source.encoding.dtype(),
        })
    }

    fn allocate(&mut self, decl: &TensorDecl, temporary: bool) -> Result<BufferId> {
        let buffer = BufferId(self.next_buffer);
        self.next_buffer += 1;
        let bytes = encoding_nbytes(&decl.shape, &decl.encoding)
            .or_overflow(format!("'{}' byte size", decl.name))?;
        self.program.buffers.push(BufferDecl {
            id: buffer,
            tensor: if temporary { None } else { Some(decl.id) },
            bytes,
            alignment: decl.alignment,
            temporary,
            persistent_offset: None,
        });
        if !temporary {
            self.declare(decl.clone());
        }
        let instr = self.next_instr();
        self.program
            .instrs
            .push(StorageInstr::Allocate { id: instr, buffer });
        self.program.schedule.push(instr);
        Ok(buffer)
    }

    /// A buffer that owns no bytes of its own: a window on another one.
    fn declare_view_buffer(&mut self, decl: &TensorDecl) -> BufferId {
        let buffer = BufferId(self.next_buffer);
        self.next_buffer += 1;
        self.program.buffers.push(BufferDecl {
            id: buffer,
            tensor: Some(decl.id),
            bytes: 0,
            alignment: decl.alignment,
            temporary: false,
            persistent_offset: None,
        });
        self.declare(decl.clone());
        buffer
    }

    fn promote_buffer(&mut self, buffer: BufferId, decl: &TensorDecl) -> Result<()> {
        let existing = self
            .program
            .buffers
            .get_mut(buffer.0 as usize)
            .filter(|candidate| candidate.id == buffer)
            .ok_or_else(|| Error::Internal(format!("buffer {} does not exist", buffer.0)))?;
        if let Some(existing_id) = existing.tensor
            && existing_id != decl.id
        {
            return Err(Error::Contract(format!(
                "buffer {} already belongs to tensor {}, cannot promote to {}",
                buffer.0, existing_id.0, decl.id.0
            )));
        }
        existing.tensor = Some(decl.id);
        existing.temporary = false;
        self.declare(decl.clone());
        Ok(())
    }

    /// Publish a tensor declaration, replacing an earlier one under the same id.
    fn declare(&mut self, decl: TensorDecl) {
        match self.tensor_at.get(&decl.id) {
            Some(at) => self.program.tensors[*at] = decl,
            None => {
                self.tensor_at.insert(decl.id, self.program.tensors.len());
                self.program.tensors.push(decl);
            }
        }
    }

    fn buffer_bytes(&self, buffer: BufferId) -> Result<u64> {
        self.program
            .buffers
            .get(buffer.0 as usize)
            .filter(|decl| decl.id == buffer)
            .map(|decl| decl.bytes)
            .ok_or_else(|| Error::Internal(format!("buffer {} does not exist", buffer.0)))
    }

    fn raw(&self, id: TensorId) -> Result<&RawTensor> {
        self.sources
            .tensor(id)
            .ok_or_else(|| Error::Checkpoint(format!("missing source tensor {}", id.0)))
    }

    fn next_instr(&mut self) -> InstrId {
        let id = InstrId(self.next_instr);
        self.next_instr += 1;
        id
    }
}

/// Name the contract an error came from, once, at the boundary.
fn annotate(err: Error, name: &str) -> Error {
    match err {
        Error::Contract(msg) => Error::Contract(format!("'{name}': {msg}")),
        Error::Shard(msg) => Error::Shard(format!("'{name}': {msg}")),
        other => other,
    }
}

fn dtype_to_quant_marker(dtype: DType) -> QuantScheme {
    match dtype {
        DType::F8E4M3 => QuantScheme::Fp8E4M3,
        DType::F8E5M2 => QuantScheme::Fp8E5M2,
        DType::I8 | DType::U8 => QuantScheme::Int8Symmetric,
        _ => QuantScheme::None,
    }
}

/// Whether a lazy source view is a plain dense window on its checkpoint tensor.
///
/// Gather piece offsets and strides are expressed in the input's own dense
/// layout, so they can be rebased onto a view that is itself dense — a shard of
/// a shard stays lazy — but not onto one that already skips. A strided view has
/// to be materialized before it can be gathered from again.
fn source_is_dense(source: &SourceView) -> Result<bool> {
    Ok(source.stride == storage_extent_for_shape(&source.shape, &source.encoding)?)
}
