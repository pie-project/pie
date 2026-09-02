//! Contracts to a load plan, in one pass (specialize, compile, lower to
//! instructions). A single copy off a checkpoint tensor stays a lazy
//! `SourceView`; a single contiguous copy out of an already-materialized
//! buffer becomes a `CreateView` — both zero-copy, decided here.

use std::collections::{HashMap, HashSet};

use crate::file::{Metadata, RawTensor, Sources};
use crate::contract::compile::{CopyList, GatherList, Leaf, Lowering, compile};
use crate::contract::infer::{Resolver, repack_spec};
use crate::contract::{
    BiasBy, Expr, ModelContract, Partition, ScaleFactor, TensorContract, TensorType, Visibility,
};
use crate::error::{Error, OrOverflow, Result};
use crate::extent::Extent;
use crate::plan::geometry::{
    full_dest_extent, repack_stage_bytes, storage_extent_for_shape, strided_physical_source_bytes,
};
use crate::plan::{
    BufferDecl, CheckpointFileDecl, DestExtent, GatherSpec, LoadPlan, QuantAttachment,
    SourceExtent, SourceTensorDecl, StorageInstr, StorageTarget, TileMapKind, TileSpec,
    TransformSpec,
};
use crate::types::{
    BufferId, DType, Encoding, InstrId, QuantGranularity, QuantScheme, RepackSpec, ScaleForm,
    TensorDecl, TensorId, encoding_dense_element_bytes, encoding_nbytes,
};

/// How many contiguous stretches one expression may break into before the
/// compiler refuses to keep going. An allocation guard, not the cost model.
const MAX_RUNS: usize = 1 << 20;

/// Turn a checked contract into the plan that satisfies it.
pub fn build(
    metadata: &Metadata,
    contract: &ModelContract,
    target: StorageTarget,
) -> Result<LoadPlan> {
    build_instance(metadata, contract, target, None)
}

/// [`build`], resolved as one instance of a group. `instance` is what the
/// group's index nodes stand for; `None` is the resident contract, where an
/// index node is a contract error rather than a number.
pub fn build_instance(
    metadata: &Metadata,
    contract: &ModelContract,
    target: StorageTarget,
    instance: Option<u32>,
) -> Result<LoadPlan> {
    let sources = Sources::new(metadata);
    let partition = Partition::new(target.tp_rank, target.tp_size);
    let resolver = Resolver::new(&sources, partition);
    let resolver = match instance {
        Some(index) => resolver.for_instance(index),
        None => resolver,
    };
    let mut builder = Builder {
        sources: &sources,
        program: LoadPlan::empty(target),
        resolver,
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

    let mut declared_at: HashMap<&str, TensorId> = HashMap::new();
    for (index, tensor) in contract.tensors.iter().enumerate() {
        if builder.values.contains_key(&tensor.name) {
            return Err(Error::Contract(format!(
                "the contract declares '{}' twice",
                tensor.name
            )));
        }
        builder.tensor(tensor, TensorId(index as u32))?;
        declared_at.insert(tensor.name.as_str(), TensorId(index as u32));
    }

    // Resolved after every declaration, not against the ones before it:
    // `Scales` pairs two published tensors and carries no ordering requirement.
    for (index, tensor) in contract.tensors.iter().enumerate() {
        let Some(scales) = &tensor.scales else {
            continue;
        };
        let Some(&of) = declared_at.get(scales.of.as_str()) else {
            return Err(Error::Contract(format!(
                "'{}' holds the scales for '{}', which the contract does not declare",
                tensor.name, scales.of
            )));
        };
        if builder.program.attachments.iter().any(|a| a.tensor == of) {
            return Err(Error::Contract(format!(
                "'{}' holds the scales for '{}', which already has scales; a weight the \
                 loader quantizes gets the scales the loader writes, so the contract must \
                 not name a second set",
                tensor.name, scales.of
            )));
        }
        builder.program.attachments.push(QuantAttachment {
            tensor: of,
            scale_tensor: TensorId(index as u32),
            // Set only when the loader's own encode generates zero points; a
            // checkpoint-shipped triplet declares its own zero-point tensor.
            zero_point_tensor: None,
            granularity: scales.granularity,
            group_size: scales.group_size,
            channel_axis: scales.channel_axis,
            scale_form: scales.form,
        });
    }
    // Zero points, after scales: an affine attachment is created by the
    // scales entry and completed here.
    for tensor in &contract.tensors {
        let Some(of) = &tensor.zero_points else {
            continue;
        };
        let Some(&of_id) = declared_at.get(of.as_str()) else {
            return Err(Error::Contract(format!(
                "'{}' holds the zero points for '{of}', which the contract does not declare",
                tensor.name
            )));
        };
        let index = contract
            .tensors
            .iter()
            .position(|entry| entry.name == tensor.name)
            .expect("an entry of the list being walked");
        let Some(attachment) = builder
            .program
            .attachments
            .iter_mut()
            .find(|a| a.tensor == of_id)
        else {
            return Err(Error::Contract(format!(
                "'{}' holds the zero points for '{of}', which declares no scales; an \
                 affine weight is `code * scale + zero` and a contract that names one \
                 companion without the other describes a tensor no kernel reads",
                tensor.name
            )));
        };
        if attachment.zero_point_tensor.is_some() {
            return Err(Error::Contract(format!(
                "'{}' holds the zero points for '{of}', which already has them",
                tensor.name
            )));
        }
        attachment.zero_point_tensor = Some(TensorId(index as u32));
    }
    Ok(builder.program)
}

/// Whether a lowered value *is* the declaration it was lowered for, or an
/// anonymous intermediate on the way to one.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    /// The value the contract publishes.
    Declared,
    /// A value a kernel node reads.
    Operand,
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
    ///
    /// Errors are named here and only here, so that no path through the
    /// compiler can report an anonymous one.
    fn tensor(&mut self, contract: &TensorContract, id: TensorId) -> Result<()> {
        self.build_tensor(contract, id)
            .map_err(|err| annotate(err, &contract.name))
    }

    /// Build one contract, leaving it to the caller to name any error.
    ///
    /// The preamble — check the declared shape, specialize, infer, declare —
    /// is shared by every node, kernel or not; only the lowering is per-arm.
    fn build_tensor(&mut self, contract: &TensorContract, id: TensorId) -> Result<()> {
        self.check_declared_shape(contract)?;
        // Sharding resolved before lowering: below this line the expression
        // means the same thing on every rank.
        let expr = self
            .resolver
            .specialize(contract.expr.clone(), &contract.name)?;
        // Type-checked before lowering, for every node.
        let ty = self.resolver.infer(&expr, &contract.name)?;
        let decl = TensorDecl {
            id,
            name: contract.name.clone(),
            shape: ty.shape.clone(),
            encoding: ty.encoding.clone(),
            alignment: self.alignment,
            visibility: Visibility::Public,
        };

        let (value, decl) = match &expr {
            // A repack is opaque: the compiler does not model the permutation.
            // What it reads is not, so the operand lowers normally and only
            // the swizzle is a kernel.
            Expr::Repack { src, layout, to } => {
                let operand = self.resolver.infer(src, &contract.name)?;
                let spec = repack_spec(&operand, *layout, to)?;
                let (payload, _) = self.operand_bytes(src, &decl)?;
                let value = self.repack(payload, spec, &decl)?;
                (value, decl)
            }
            // A scale needs a kernel; its operand still lowers normally.
            Expr::Scale { src, factor } => {
                let (payload, elements) = self.operand_bytes(src, &decl)?;
                let (spec, extra) = match factor {
                    ScaleFactor::Uniform(bits) => (
                        TransformSpec {
                            scale_factor_bits: *bits,
                            ..TransformSpec::default()
                        },
                        Vec::new(),
                    ),
                    ScaleFactor::PerBlock { by } => {
                        // `infer` has already required equal rank and an exact
                        // division on every axis.
                        let operand = self.resolver.infer(src, &contract.name)?;
                        let factor_ty = self.resolver.infer(by, &contract.name)?;
                        let blocks = block_sizes(&operand.shape, &factor_ty.shape)?;
                        let factors = self.scale_factors(by, &contract.name)?;
                        (
                            TransformSpec {
                                scale_blocks: blocks,
                                from: source_scheme(&elements),
                                ..TransformSpec::default()
                            },
                            vec![factors],
                        )
                    }
                };
                let value = self.transform_with(payload, &decl, TileMapKind::Scale, extra, spec)?;
                (value, decl)
            }
            // A bias mirrors a scale, with one asymmetry: the operand reaches
            // a bias as numbers already, so there is no `from` scheme to state.
            Expr::Bias { src, by } => {
                let (payload, _) = self.operand_bytes(src, &decl)?;
                let (spec, extra) = match by {
                    BiasBy::Uniform(bits) => (
                        TransformSpec {
                            bias_bits: *bits,
                            ..TransformSpec::default()
                        },
                        Vec::new(),
                    ),
                    BiasBy::PerBlock { by } => {
                        let operand = self.resolver.infer(src, &contract.name)?;
                        let addend_ty = self.resolver.infer(by, &contract.name)?;
                        let blocks = block_sizes(&operand.shape, &addend_ty.shape)?;
                        let addends = self.scale_factors(by, &contract.name)?;
                        (
                            TransformSpec {
                                scale_blocks: blocks,
                                ..TransformSpec::default()
                            },
                            vec![addends],
                        )
                    }
                };
                let value = self.transform_with(payload, &decl, TileMapKind::Bias, extra, spec)?;
                (value, decl)
            }
            // A cast needs a kernel; the staging declaration wears the
            // operand's encoding, since `convert` returns the cast's own.
            Expr::Cast { src, to } => {
                let operand = TensorDecl {
                    encoding: self.resolver.infer(src, &contract.name)?.encoding,
                    ..decl
                };
                let (value, _) = self.operand_bytes(src, &operand)?;
                self.convert(value, operand, to)?
            }
            // The affine fragment: everything `compile` answers with byte
            // spans. Spelled out rather than left to a wildcard, so a new node
            // must say which side of the cost ladder it falls on.
            Expr::Src(_)
            | Expr::Out(_)
            | Expr::Fill { .. }
            | Expr::Slice { .. }
            | Expr::Stride { .. }
            | Expr::Gather { .. }
            | Expr::Concat { .. }
            | Expr::Transmute { .. }
            // Unreachable: `specialize` has already rewritten every shard into
            // this rank's slice.
            | Expr::Shard { .. }
            // Unreachable: `specialize` resolves an instance's name and band
            // before typing, so by lowering time these are a `Src`/`Slice`.
            | Expr::SrcIndexed(_)
            | Expr::Select { .. } => (self.affine(&expr, &decl)?, decl),
        };

        check_declared_encoding(contract, &decl)?;

        // Layout and alignment are properties of the declaration, not the
        // bytes, so they're simply stated on the realized declaration.
        let realized = TensorDecl {
            alignment: self.alignment,
            visibility: Visibility::Public,
            ..decl
        };
        self.resolver.publish(
            &contract.name,
            TensorType {
                shape: realized.shape.clone(),
                encoding: realized.encoding.clone(),
            },
        );
        let value = self.realize(value, &realized, contract.visibility)?;
        self.values.insert(contract.name.clone(), value);
        Ok(())
    }

    /// The declared shape is a claim about the *whole* tensor (`tp = 1`), so
    /// it's checked against
    /// [`Resolver::infer_whole`](crate::contract::infer::Resolver::infer_whole)
    /// — the unspecialized expression typed with every shard read at
    /// [`Partition::WHOLE`] — not against this rank's specialized shape.
    /// `TensorContract::shape` is optional; a no-op when absent.
    fn check_declared_shape(&mut self, contract: &TensorContract) -> Result<()> {
        let Some(declared) = &contract.shape else {
            return Ok(());
        };
        let claimed = self
            .resolver
            .infer_whole(&contract.expr, &contract.name)?
            .shape;
        if claimed != *declared {
            return Err(Error::Contract(format!(
                "declares shape {declared:?} but its expression yields {claimed:?}"
            )));
        }
        Ok(())
    }

    /// Solve the affine expression and emit the copies that satisfy it.
    fn affine(&mut self, expr: &Expr, decl: &TensorDecl) -> Result<Value> {
        let lowering = compile(expr, self.resolver.checked(), MAX_RUNS)?;
        self.emit(&lowering, decl, Role::Declared)
    }

    /// Emit whichever lowering the expression got. The choice was made in
    /// `contract::compile` and is not revisited here.
    fn emit(&mut self, lowering: &Lowering, decl: &TensorDecl, role: Role) -> Result<Value> {
        match lowering {
            Lowering::Copy(copies) => self.copies(copies, decl, role),
            Lowering::Gather(gather) => self.gather(gather, decl),
        }
    }

    /// Emit the one instruction a gather lowering is. A gather reads a whole
    /// checkpoint tensor and writes a whole destination — nothing to alias or
    /// fill, since a permutation is exactly where source and destination
    /// bytes are not in the same order. A gather off a computed buffer is
    /// refused rather than staged: no contract currently produces one.
    fn gather(&mut self, gather: &GatherList, decl: &TensorDecl) -> Result<Value> {
        let geometry = gather.byte_geometry(&decl.encoding)?;
        let output_bytes = encoding_nbytes(&decl.shape, &decl.encoding)
            .or_overflow(format!("'{}' size overflow", decl.name))?;
        let Value::Source(source) = self.leaf(gather.leaves.as_slice(), gather.leaf)? else {
            return Err(Error::Unsupported(format!(
                "'{}' gathers from a computed buffer; the gather lowering reads \
                 the checkpoint",
                decl.name
            )));
        };
        let raw = self.raw(source.tensor_id)?;
        let (file_id, tensor_id, base) = (raw.file_id, raw.id, raw.file_offset);
        let file_offset = base
            .checked_add(source.offset_bytes)
            .or_overflow("gather source file offset")?;
        let source_bytes = geometry.source_bytes();
        let dtype = source.encoding.dtype();

        let out = self.allocate(decl, true)?;
        let instr = self.next_instr();
        self.program.instrs.push(StorageInstr::GatherWrite {
            id: instr,
            source: SourceExtent {
                file_id,
                tensor_id,
                file_offset,
                span_bytes: source_bytes,
                stride: Extent::byte_run(source_bytes),
                dtype,
            },
            dest: DestExtent {
                buffer: out,
                offset: 0,
                stride: Extent::byte_run(output_bytes),
            },
            gather: GatherSpec {
                indices: gather.indices.clone(),
                block_bytes: geometry.block_bytes,
                rows: geometry.rows,
                src_row_bytes: geometry.src_row_bytes,
            },
        });
        self.program.schedule.push(instr);
        Ok(Value::Buffer(out))
    }

    /// Emit the rectangular copies of a solved expression.
    fn copies(&mut self, lowering: &CopyList, decl: &TensorDecl, role: Role) -> Result<Value> {
        let output_bytes = encoding_nbytes(&decl.shape, &decl.encoding)
            .or_overflow(format!("'{}' size overflow", decl.name))?;
        let rects = lowering.byte_pieces(&decl.encoding)?;

        // cost == 1: a single rectangle covering the whole destination, so the
        // tensor can alias the checkpoint bytes or view a buffer that already
        // holds them instead of copying. `cost` already prices in a hole, so
        // this excludes a padded destination, but the check is spelled out
        // explicitly anyway.
        if lowering.cost() == 1
            && let [rect] = rects.as_slice()
            && !lowering.needs_zero_fill()
            && rect.dst_offset == 0
            && rect.bytes() == output_bytes
        {
            match self.leaf(lowering.leaves.as_slice(), rect.leaf)? {
                Value::Source(source) if source_is_dense(&source)? => {
                    let (stride, _) = rect.split()?;
                    return Ok(Value::Source(SourceView {
                        tensor_id: source.tensor_id,
                        shape: decl.shape.clone(),
                        encoding: decl.encoding.clone(),
                        offset_bytes: source
                            .offset_bytes
                            .checked_add(rect.src_offset)
                            .or_overflow("source view offset")?,
                        stride,
                    }));
                }
                Value::Buffer(buffer) if rect.is_byte_run() => {
                    let out = self.view_buffer(decl, role == Role::Declared);
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
            match self.leaf(lowering.leaves.as_slice(), rect.leaf)? {
                Value::Source(source) => {
                    let raw = self.raw(source.tensor_id)?;
                    let (file_id, tensor_id, base) = (raw.file_id, raw.id, raw.file_offset);
                    let (src_stride, dst_stride) = rect.split()?;
                    let file_offset = base
                        .checked_add(source.offset_bytes)
                        .and_then(|at| at.checked_add(rect.src_offset))
                        .or_overflow("source extent file offset")?;
                    let instr = self.next_instr();
                    self.program.instrs.push(StorageInstr::ExtentWrite {
                        id: instr,
                        source: SourceExtent {
                            file_id,
                            tensor_id,
                            file_offset,
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
                    let (shape, encoding) = self.leaf_type(lowering.leaves.as_slice(), rect.leaf)?;
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

    /// Lower an [`Expr::Cast`]: pick the kernel that puts `from`'s values into
    /// `to`'s representation. Direction is read off the pair of encodings.
    fn convert(
        &mut self,
        value: Value,
        decl: TensorDecl,
        to: &Encoding,
    ) -> Result<(Value, TensorDecl)> {
        let out = TensorDecl {
            encoding: to.clone(),
            ..decl.clone()
        };
        let value = match (decl.encoding.clone(), to.clone()) {
            (Encoding::Raw(_), Encoding::Raw(dtype)) => self.transform(
                value,
                &out,
                TileMapKind::Cast,
                TransformSpec {
                    to: Some(dtype_to_quant_marker(dtype)),
                    ..TransformSpec::default()
                },
            )?,
            (Encoding::Quant(source), Encoding::Raw(dtype)) => {
                if source.logical_dtype != dtype {
                    return Err(Error::Contract(format!(
                        "Cast to raw {dtype:?} from a scheme whose logical type is {:?}",
                        source.logical_dtype
                    )));
                }
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
                let metadata = self.quant_metadata_outputs(&out)?;
                self.encode(value, &out, target.scheme, &metadata)?
            }
            // `infer_cast` refuses this pair; reaching it means it was
            // lowered without being typed.
            (Encoding::Quant(_), Encoding::Quant(_)) => {
                return Err(Error::Internal(
                    "Cast between quantized schemes should have been rejected by infer".to_string(),
                ));
            }
        };
        Ok((value, out))
    }

    /// The bytes an escape hatch reads, and the encoding they are in.
    /// Anything the affine fragment can say is allowed here. Encoding
    /// returned is the operand's, not the output's — a [`Transmute`] under
    /// the scale is how a contract says checkpoint `U8` bytes are packed
    /// quantization elements. A kernel directly under a kernel is refused by
    /// `compile`; sequencing two requires naming the intermediate.
    ///
    /// [`Transmute`]: crate::contract::Expr::Transmute
    fn operand_bytes(&mut self, src: &Expr, decl: &TensorDecl) -> Result<(Value, Encoding)> {
        let ty = self
            .resolver
            .infer(src, &decl.name)
            .map_err(|err| annotate(err, &decl.name))?;
        if let Expr::Src(name) = src {
            return Ok((Value::Source(self.source_view(name)?), ty.encoding));
        }
        let lowering = compile(src, self.resolver.checked(), MAX_RUNS)?;
        let operand = TensorDecl {
            shape: ty.shape,
            encoding: ty.encoding.clone(),
            ..decl.clone()
        };
        Ok((
            self.emit(&lowering, &operand, Role::Operand)?,
            ty.encoding,
        ))
    }

    /// The buffer holding a per-group `Scale`'s factors. Must be a tensor an
    /// earlier contract declared, so the plan keeps one set of scale bytes
    /// rather than two.
    fn scale_factors(&mut self, by: &Expr, what: &str) -> Result<BufferId> {
        let Expr::Out(name) = by else {
            return Err(Error::Internal(format!(
                "'{what}' scale factors should have been rejected by infer"
            )));
        };
        match self.values.get(name).cloned() {
            Some(Value::Buffer(buffer)) => Ok(buffer),
            Some(Value::Source(source)) => {
                // Still an alias of checkpoint bytes; the kernel needs them
                // materialized in memory.
                let decl = TensorDecl {
                    id: source.tensor_id,
                    name: name.clone(),
                    shape: source.shape.clone(),
                    encoding: source.encoding.clone(),
                    alignment: self.alignment,
                    visibility: Visibility::Public,
                };
                let dtype = source.encoding.dtype();
                let staged = self.transform(
                    Value::Source(source),
                    &decl,
                    TileMapKind::Cast,
                    TransformSpec {
                        to: Some(dtype_to_quant_marker(dtype)),
                        ..TransformSpec::default()
                    },
                )?;
                match staged {
                    Value::Buffer(buffer) => Ok(buffer),
                    Value::Source(_) => Err(Error::Internal(
                        "staged scale factors did not become a buffer".to_string(),
                    )),
                }
            }
            None => Err(Error::Contract(format!(
                "'{what}' scales by '{name}', which no earlier contract declares"
            ))),
        }
    }

    /// A one-in, one-out transform kernel over a whole tensor.
    fn transform(
        &mut self,
        value: Value,
        decl: &TensorDecl,
        kind: TileMapKind,
        transform: TransformSpec,
    ) -> Result<Value> {
        self.transform_with(value, decl, kind, Vec::new(), transform)
    }

    /// [`Builder::transform`], plus operands the kernel reads beside its
    /// input. Extras come after the input, so `inputs[0]` is always the
    /// payload and extras are found from the end.
    fn transform_with(
        &mut self,
        value: Value,
        decl: &TensorDecl,
        kind: TileMapKind,
        extra: Vec<BufferId>,
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
        inputs.extend(extra);
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

    /// The one transform the algebra cannot denote. `spec` arrives derived
    /// from the operand's type and the declaration, so no narrowing is left
    /// to do here.
    fn repack(&mut self, value: Value, spec: RepackSpec, decl: &TensorDecl) -> Result<Value> {
        let out = self.allocate(decl, true)?;
        let mut inputs = Vec::new();
        let repack = spec;
        let (source, input_bytes) = match value {
            Value::Source(source) => {
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
                repack: Some(repack),
                scratch_bytes: input_bytes
                    .checked_add(stage_bytes)
                    .or_overflow("Repack scratch bytes")?,
                ..TransformSpec::default()
            },
        );
        Ok(Value::Buffer(out))
    }

    /// Give a value the name it is declared under. A [`Visibility::Public`]
    /// entry is promoted into the persistent arena and `Finalize`d (the
    /// engine learns its name); an [`Visibility::Internal`] entry gets
    /// neither and stays a temporary the memory planner may reuse, though it
    /// is still declared so later entries can resolve it via [`Expr::Out`].
    fn realize(
        &mut self,
        value: Value,
        decl: &TensorDecl,
        visibility: Visibility,
    ) -> Result<Value> {
        let decl = &TensorDecl {
            visibility,
            ..decl.clone()
        };
        if visibility == Visibility::Internal {
            let buffer = match value {
                Value::Buffer(buffer) => {
                    self.attach_type(buffer, decl)?;
                    buffer
                }
                // Still a copy: a kernel needs a buffer to read factors from.
                Value::Source(source) => {
                    let buffer = self.allocate_as(decl, true, true)?;
                    self.extent_write(source, buffer, &decl.shape)?;
                    buffer
                }
            };
            return Ok(Value::Buffer(buffer));
        }
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
    fn leaf(&mut self, leaves: &[Leaf], index: usize) -> Result<Value> {
        let leaf = leaves
            .get(index)
            .ok_or_else(|| Error::Internal(format!("lowering has no leaf {index}")))?
            .clone();
        match leaf {
            Leaf::Checkpoint(name) => Ok(Value::Source(self.source_view(&name)?)),
            Leaf::Contract(name) => self
                .values
                .get(&name)
                .cloned()
                .ok_or_else(|| Error::Contract(format!("reads '{name}' before it exists"))),
        }
    }

    /// The shape and encoding of one of a lowering's leaves.
    fn leaf_type(&self, leaves: &[Leaf], index: usize) -> Result<(Vec<i64>, Encoding)> {
        let leaf = leaves
            .get(index)
            .ok_or_else(|| Error::Internal(format!("lowering has no leaf {index}")))?;
        let ty =
            self.resolver.checked().type_of(leaf).ok_or_else(|| {
                Error::Contract(format!("'{}' has no resolved type", leaf.name()))
            })?;
        Ok((ty.shape.clone(), ty.encoding.clone()))
    }

    /// A lazy whole-tensor window on a checkpoint tensor.
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

    /// The tensors a quantizing encode must publish alongside its output. A
    /// quantized weight without its scales is unreadable, so every failure
    /// path here is a named error rather than an empty result.
    fn quant_metadata_outputs(&mut self, decl: &TensorDecl) -> Result<Vec<TensorDecl>> {
        let Encoding::Quant(spec) = &decl.encoding else {
            return Err(Error::Internal(
                "quant_metadata_outputs on a non-quantized declaration".to_string(),
            ));
        };
        let layout = ScaleLayout::for_encode(spec.scheme, &decl.shape)
            .map_err(|err| annotate(err, &decl.name))?;
        // Both metadata tensors are allocated before the attachment is pushed:
        // the attachment must name the zero point.
        let scales = self.generated_metadata_decl(decl, &layout, layout.suffix)?;
        let zero_point = match layout.zero_point_suffix {
            Some(suffix) => Some(self.generated_metadata_decl(decl, &layout, suffix)?),
            None => None,
        };
        // Granularity is the encode kernel's own, describing the layout it
        // writes, not anything readable off `spec`.
        self.program.attachments.push(QuantAttachment {
            tensor: decl.id,
            scale_tensor: scales.id,
            zero_point_tensor: zero_point.as_ref().map(|decl| decl.id),
            granularity: layout.granularity,
            group_size: layout.group_size,
            channel_axis: layout.channel_axis,
            scale_form: layout.scale_form,
        });
        // Order is the encode instruction's output order, after the weight: a
        // kernel writing three tensors is told which is which by position.
        let mut out = vec![scales];
        out.extend(zero_point);
        Ok(out)
    }

    /// One of the tensors an encode publishes beside its output, named and
    /// numbered. Split out of `quant_metadata_outputs` only because an affine
    /// scheme needs two of them and a borrow cannot be held across both.
    fn generated_metadata_decl(
        &mut self,
        weight: &TensorDecl,
        layout: &ScaleLayout,
        suffix: &str,
    ) -> Result<TensorDecl> {
        let name = layout
            .naming
            .apply(&weight.name, suffix)
            .map_err(|err| annotate(err, &weight.name))?;
        let id = TensorId(self.next_generated_tensor);
        self.next_generated_tensor = self.next_generated_tensor.saturating_add(1);
        Ok(TensorDecl {
            id,
            name,
            shape: layout.shape.clone(),
            encoding: layout.encoding.clone(),
            alignment: self.alignment,
            visibility: Visibility::Public,
        })
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

    /// The checkpoint tensor holding `source`'s block scales, if any.
    /// Block-scaled FP8 ships the factors in a sibling tensor named the
    /// payload's plus `_scale_inv`.
    fn with_block_scale_source(
        &self,
        mut transform: TransformSpec,
        source: Option<&SourceExtent>,
    ) -> TransformSpec {
        transform.metadata_source = source.and_then(|source| {
            let raw = self.sources.tensor(source.tensor_id)?;
            if !matches!(raw.encoding, Encoding::Raw(DType::E4m3)) {
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
        let source_end = source
            .offset_bytes
            .checked_add(physical_bytes)
            .or_overflow("source extent end")?;
        if source_end > raw.span_bytes {
            return Err(Error::Contract(format!(
                "source extent for '{}' exceeds tensor span",
                raw.name
            )));
        }
        Ok(SourceExtent {
            file_id: raw.file_id,
            tensor_id: raw.id,
            file_offset: raw
                .file_offset
                .checked_add(source.offset_bytes)
                .or_overflow("source extent file offset")?,
            span_bytes,
            stride: source.stride.clone(),
            dtype: source.encoding.dtype(),
        })
    }

    fn allocate(&mut self, decl: &TensorDecl, temporary: bool) -> Result<BufferId> {
        self.allocate_as(decl, temporary, !temporary)
    }

    /// Allocate a buffer, choosing independently whether it is temporary and
    /// whether it carries `decl`'s type. The two coincide for everything
    /// except an [`Visibility::Internal`] declaration, which is both:
    /// temporary since nothing binds it, and typed since a kernel that reads
    /// it still needs to know its elements.
    fn allocate_as(
        &mut self,
        decl: &TensorDecl,
        temporary: bool,
        declared: bool,
    ) -> Result<BufferId> {
        let buffer = BufferId(self.next_buffer);
        self.next_buffer += 1;
        let bytes = encoding_nbytes(&decl.shape, &decl.encoding)
            .or_overflow(format!("'{}' byte size", decl.name))?;
        self.program.buffers.push(BufferDecl {
            id: buffer,
            tensor: declared.then_some(decl.id),
            ty: crate::contract::TensorType::new(decl.shape.clone(), decl.encoding.clone()),
            bytes,
            alignment: decl.alignment,
            temporary,
            persistent_offset: None,
            scratch_offset: None,
        });
        if declared {
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
    fn view_buffer(&mut self, decl: &TensorDecl, declared: bool) -> BufferId {
        let buffer = BufferId(self.next_buffer);
        self.next_buffer += 1;
        self.program.buffers.push(BufferDecl {
            id: buffer,
            tensor: declared.then_some(decl.id),
            ty: crate::contract::TensorType::new(decl.shape.clone(), decl.encoding.clone()),
            bytes: 0,
            alignment: decl.alignment,
            temporary: false,
            persistent_offset: None,
            scratch_offset: None,
        });
        if declared {
            self.declare(decl.clone());
        }
        buffer
    }

    /// Give `buffer` `decl`'s type without promoting it out of the temporary
    /// pool: what an [`Visibility::Internal`] declaration needs.
    fn attach_type(&mut self, buffer: BufferId, decl: &TensorDecl) -> Result<()> {
        self.claim(buffer, decl)?;
        self.declare(decl.clone());
        Ok(())
    }

    fn promote_buffer(&mut self, buffer: BufferId, decl: &TensorDecl) -> Result<()> {
        self.claim(buffer, decl)?.temporary = false;
        self.declare(decl.clone());
        Ok(())
    }

    /// Record that `buffer` IS `decl`, and restate its type from the
    /// declaration. The byte check guards against renaming bytes of a
    /// different shape.
    fn claim(&mut self, buffer: BufferId, decl: &TensorDecl) -> Result<&mut BufferDecl> {
        let bytes = encoding_nbytes(&decl.shape, &decl.encoding)
            .or_overflow(format!("'{}' byte size", decl.name))?;
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
                "buffer {} already belongs to tensor {}, cannot name it {}",
                buffer.0, existing_id.0, decl.id.0
            )));
        }
        // A view owns no bytes of its own, so its zero is not a disagreement.
        if existing.bytes != 0 && existing.bytes != bytes {
            return Err(Error::Contract(format!(
                "buffer {} holds {} bytes, which is not the {bytes} '{}' declares",
                buffer.0, existing.bytes, decl.name
            )));
        }
        existing.tensor = Some(decl.id);
        existing.ty = crate::contract::TensorType::new(decl.shape.clone(), decl.encoding.clone());
        Ok(existing)
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

/// Elements of the operand per factor, on each axis. `infer_scale_per_block`
/// has already checked equal rank and an exact division, so a failure here
/// is a compiler fault rather than a bad contract.
fn block_sizes(operand: &[i64], factors: &[i64]) -> Result<Vec<i64>> {
    if operand.len() != factors.len() {
        return Err(Error::Internal(
            "scale factors of a different rank should have been rejected by infer".to_string(),
        ));
    }
    operand
        .iter()
        .zip(factors)
        .map(|(&extent, &count)| {
            if count <= 0 || extent % count != 0 {
                return Err(Error::Internal(
                    "scale factors that do not block the operand should have been \
                     rejected by infer"
                        .to_string(),
                ));
            }
            Ok(extent / count)
        })
        .collect()
}

fn source_scheme(encoding: &Encoding) -> Option<QuantScheme> {
    match encoding {
        Encoding::Quant(spec) => Some(spec.scheme),
        Encoding::Raw(_) => None,
    }
}

/// The scale tensor an encode kernel writes beside its output. Lives here
/// rather than on [`QuantSpec`] because none of it is readable off the spec:
/// granularity, layout and form are the kernel's own, describing what it
/// writes.
struct ScaleLayout {
    /// Appended to the weight's declared name. Two conventions, inherited
    /// from what the engines already look for.
    suffix: &'static str,
    /// The zero point's suffix, for an affine scheme, or `None` for a
    /// symmetric one. Shape and encoding are the scales', so only the name
    /// differs.
    zero_point_suffix: Option<&'static str>,
    /// Whether the suffixes extend the weight's name or replace its last
    /// component. Both conventions are in the wild and neither is derivable.
    naming: MetaNaming,
    shape: Vec<i64>,
    encoding: Encoding,
    granularity: QuantGranularity,
    group_size: u32,
    channel_axis: u32,
    scale_form: ScaleForm,
}

/// An axis index as the `u32` a [`QuantAttachment`] states it in. Saturating
/// rather than fallible: the index comes from a shape this compiler already
/// walked, and no shape in this tree has 4 billion axes.
fn axis(at: usize) -> u32 {
    u32::try_from(at).unwrap_or(u32::MAX)
}

/// Where a generated metadata tensor's name is rooted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetaNaming {
    /// `w` publishes `w<suffix>`.
    Extend,
    /// `w.weight` publishes `w<suffix>`; a declaration not ending in
    /// `.weight` (a text's own name, e.g. `lm_head`) extends instead —
    /// the companions are found structurally (`LoadPlan::attachments`),
    /// never by suffix, so the name only has to be unique.
    ReplaceWeight,
}

impl MetaNaming {
    fn apply(self, name: &str, suffix: &str) -> Result<String> {
        match self {
            Self::Extend => Ok(format!("{name}{suffix}")),
            Self::ReplaceWeight => match name.strip_suffix(".weight") {
                Some(stem) => Ok(format!("{stem}{suffix}")),
                None => Ok(format!("{name}{suffix}")),
            },
        }
    }
}

impl ScaleLayout {
    /// What encoding `shape` into `scheme` publishes, or why it cannot. Rank
    /// 3 (e.g. `[experts, rows, cols]`) is the same row-major rectangle the
    /// kernels walk at rank 2 — see [`crate::types::rectangle`]. Scales keep
    /// the payload's leading axes and replace its last one: `[experts, rows,
    /// cols]` publishes `[experts, rows, cols / 32]`.
    fn for_encode(scheme: QuantScheme, shape: &[i64]) -> Result<Self> {
        let Some((_rows, cols)) = crate::types::rectangle(shape) else {
            return Err(Error::Contract(format!(
                "encoding to {scheme:?} scales a [rows, cols] rectangle, so it \
                 cannot produce scales for the rank-{} shape {shape:?}",
                shape.len()
            )));
        };
        let cols = &cols;
        // The payload's axes minus its contracted one: what a scales shape is
        // built on, in the declaration's own rank.
        let lead = &shape[..shape.len() - 1];
        match scheme {
            QuantScheme::Fp8E4M3 | QuantScheme::Int8Symmetric => Ok(Self {
                suffix: "_scale_inv",
                zero_point_suffix: None,
                naming: MetaNaming::Extend,
                // One F32 per output channel: the payload's leading axes,
                // whole -- `[rows]` at rank 2, `[experts, rows]` at rank 3.
                shape: lead.to_vec(),
                encoding: Encoding::Raw(DType::F32),
                granularity: QuantGranularity::PerChannel,
                group_size: 0,
                // The last axis a channel is counted along, which is the one
                // just inside the contracted axis. `0` at rank 2, as before.
                channel_axis: axis(lead.len() - 1),
                scale_form: ScaleForm::F32Factors,
            }),
            // E8M0 block scale: one uint8 per 32-element block along K. The
            // encode-tile kernel writes a row-major `[.., cols/32]` byte
            // tensor. Suffix is `.scales` — the spelling `model_dsl`'s
            // `Weight::planes` interns and the engine's residency sink looks
            // up by name, so a different spelling here would be unfindable.
            // `MetaNaming::Extend`: this tree's canonical names
            // (`layer.7.experts_down`) don't end in `.weight`.
            QuantScheme::Mxfp4E2M1E8M0 => {
                if cols % 32 != 0 {
                    return Err(Error::Contract(format!(
                        "encoding to MXFP4 blocks 32 columns under one scale, \
                         but the output has {cols}"
                    )));
                }
                Ok(Self {
                    suffix: ".scales",
                    zero_point_suffix: None,
                    naming: MetaNaming::Extend,
                    shape: crate::types::grouped_shape(lead, cols / 32),
                    encoding: Encoding::Raw(DType::U8),
                    granularity: QuantGranularity::PerGroup,
                    group_size: 32,
                    channel_axis: axis(lead.len()),
                    scale_form: ScaleForm::RawE8M0,
                })
            }
            // MLX's affine U4: 64 columns under one BF16 scale and one BF16
            // zero point (element = `code * scale + zero`). `.scales` and
            // `.biases` are MLX's own names, so an encoded weight binds like
            // a shipped one.
            QuantScheme::MlxAffineU4 => {
                if cols % 64 != 0 {
                    return Err(Error::Contract(format!(
                        "encoding to MLX affine U4 groups 64 columns under one \
                         scale, but the output has {cols}"
                    )));
                }
                Ok(Self {
                    suffix: ".scales",
                    zero_point_suffix: Some(".biases"),
                    naming: MetaNaming::ReplaceWeight,
                    shape: crate::types::grouped_shape(lead, cols / 64),
                    encoding: Encoding::Raw(DType::Bf16),
                    granularity: QuantGranularity::PerGroup,
                    group_size: 64,
                    channel_axis: axis(lead.len()),
                    scale_form: ScaleForm::Bf16AffineFactors,
                })
            }
            other => Err(Error::Contract(format!(
                "no encode kernel writes {other:?}, so a declaration cannot ask \
                 to be quantized into it"
            ))),
        }
    }
}

/// The declared encoding is a claim about the expression, checked here (not
/// against the whole tensor, unlike [`Builder::check_declared_shape`] — an
/// encoding is the same however the tensor is cut).
fn check_declared_encoding(contract: &TensorContract, decl: &TensorDecl) -> Result<()> {
    if contract.encoding == decl.encoding {
        return Ok(());
    }
    Err(Error::Contract(format!(
        "declares {:?} but its expression yields {:?}; a change of \
         representation is an explicit cast",
        contract.encoding, decl.encoding
    )))
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
        DType::E4m3 => QuantScheme::Fp8E4M3,
        DType::E5m2 => QuantScheme::Fp8E5M2,
        DType::I8 | DType::U8 => QuantScheme::Int8Symmetric,
        _ => QuantScheme::None,
    }
}

/// Whether a lazy source view is a plain dense window on its checkpoint
/// tensor. Gather offsets/strides are expressed in the input's dense layout,
/// so only a dense view can be rebased onto again lazily; a strided view must
/// be materialized first.
fn source_is_dense(source: &SourceView) -> Result<bool> {
    Ok(source.stride == storage_extent_for_shape(&source.shape, &source.encoding)?)
}
