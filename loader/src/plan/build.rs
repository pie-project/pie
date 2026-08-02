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
//! agreed: every one of them recorded `rewrites: 0` and an expression count the
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
//! * a single copy off a checkpoint tensor stays a lazy `SourceView`, so a
//!   shard is read straight out of the file at an offset;
//! * a single contiguous copy out of an already-materialized buffer becomes a
//!   `CreateView`, so republishing a slice of a fused tensor costs nothing.

use std::collections::{HashMap, HashSet};

use crate::checkpoint::{CheckpointMetadata, RawTensor, Sources};
use crate::contract::compile::{Leaf, Lowering, compile};
use crate::contract::infer::{Resolver, repack_spec};
use crate::contract::{
    Expr, ModelContract, Partition, ScaleFactor, TensorContract, TensorType, Visibility,
};
use crate::error::{Error, OrOverflow, Result};
use crate::extent::Extent;
use crate::plan::geometry::{
    full_dest_extent, repack_stage_bytes, storage_extent_for_shape, strided_physical_source_bytes,
};
use crate::plan::{
    BufferDecl, CheckpointFileDecl, DestExtent, LoadPlan, QuantAttachment, SourceExtent,
    SourceTensorDecl, StorageInstr, StorageTarget, TileMapKind, TileSpec, TransformSpec,
};
use crate::types::{
    BufferId, DType, Encoding, InstrId, QuantGranularity, QuantScheme, RepackSpec, ScaleForm,
    TensorDecl, TensorId, encoding_dense_element_bytes, encoding_nbytes,
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
    build_instance(metadata, contract, target, None)
}

/// [`build`], resolved as one instance of a group.
///
/// `instance` is what the group's index nodes stand for. `None` is the resident
/// contract, where an index node is a contract error rather than a number --
/// which is why this takes an `Option` instead of defaulting to zero.
pub fn build_instance(
    metadata: &CheckpointMetadata,
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
        declared_at.insert(tensor.name.as_str(), TensorId(index as u32));
    }

    // Resolved after every declaration, not against the ones before it:
    // `Scales` pairs two published tensors rather than feeding one into the
    // other, so it carries none of the ordering `Expr::Out` needs — and the one
    // authoring site in the tree, `dsv4_block_scales_to_fp32`, runs before the
    // pass that publishes the weights it names.
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
            // A checkpoint that ships an affine triplet ships its zero points as
            // a tensor of their own, which the contract declares in its own
            // right; only an encode the loader performs generates one, and only
            // that path has an id to record here.
            zero_point_tensor: None,
            granularity: scales.granularity,
            group_size: scales.group_size,
            channel_axis: scales.channel_axis,
            scale_form: scales.form,
        });
    }
    Ok(builder.program)
}

/// Whether a lowered value *is* the declaration it was lowered for, or an
/// anonymous intermediate on the way to one.
///
/// They differ in exactly one place. A lowering that costs 1 is executed by
/// aliasing rather than copying, and an alias has to be declared under some
/// tensor id. The declaration's own is right for [`Role::Declared`] and wrong
/// for [`Role::Operand`], because the kernel's result is about to claim it. So
/// an operand's view is anonymous — which is what the staging buffer
/// [`Builder::allocate`] hands the general case has always been.
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
    /// The preamble — specialize, infer, check the declared shape, declare —
    /// is shared by every node, kernel or not. It used to be written out once
    /// per arm, and four copies of four steps drift: the `Repack` arm came to
    /// skip [`check_declared_shape`] entirely, and the kernel arms came to
    /// report an unnamed error where the affine path reported a named one. A
    /// node that needs a kernel differs from an affine one in how it is
    /// *lowered*, not in how it is checked, so only the lowering is per-arm.
    fn build_tensor(&mut self, contract: &TensorContract, id: TensorId) -> Result<()> {
        // Sharding is resolved before lowering, for every node: below this line
        // the expression means the same thing on every rank, the factors a
        // scale reads line up with the payload, and neither `compile` nor any
        // kernel arm needs a case for `Expr::Shard`.
        let expr = self
            .resolver
            .specialize(contract.expr.clone(), &contract.name)?;
        // Type-checked before lowering, for every node. `infer` is where the
        // algebra's rules live — the operand rule, `infer_scale`'s finite
        // non-zero factor, `repack_spec`'s element width — and an arm that
        // skips it demotes those rules to comments that fire only for
        // expressions nested deep enough to reach an arm that does check.
        let ty = self.resolver.infer(&expr, &contract.name)?;
        check_declared_shape(contract, &ty)?;
        let decl = TensorDecl {
            id,
            name: contract.name.clone(),
            shape: ty.shape.clone(),
            encoding: ty.encoding.clone(),
            alignment: self.alignment,
            visibility: Visibility::Public,
        };

        let (value, decl) = match &expr {
            // A repack is opaque: the layout transform declares its own result
            // and the compiler does not model the permutation. What it *reads*
            // is not opaque, so the operand is lowered the way any other
            // expression is and only the swizzle is a kernel.
            //
            // `infer` yields `to` for a repack, so the shared declaration is
            // already the transform's own and there is nothing to override.
            Expr::Repack { src, layout, to } => {
                let operand = self.resolver.infer(src, &contract.name)?;
                let spec = repack_spec(&operand, *layout, to)?;
                let (payload, _) = self.operand_bytes(src, &decl)?;
                let value = self.repack(payload, spec, &decl)?;
                (value, decl)
            }
            // A scale needs a kernel, so unlike the affine fragment it cannot
            // be answered with byte spans. What it reads still can be: the
            // operand is lowered the way any other expression is, and only the
            // multiply is a kernel.
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
                        // Written down here because here is where both shapes
                        // are known and neither executor should have to trust
                        // that it reconstructed the same ratio the type checker
                        // did. `infer` has already required equal rank and an
                        // exact division on every axis.
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
            // A cast needs a kernel for the same reason a scale does, and its
            // operand is lowered the same way: the conversion is the only part
            // that costs anything.
            //
            // The staging declaration is the shared one wearing the operand's
            // encoding, because the operand's bytes are what gets written; it
            // is `convert` that returns the declaration the cast yields.
            Expr::Cast { src, to } => {
                let operand = TensorDecl {
                    encoding: self.resolver.infer(src, &contract.name)?.encoding,
                    ..decl
                };
                let (value, _) = self.operand_bytes(src, &operand)?;
                self.convert(value, operand, to)?
            }
            // The affine fragment: everything `compile` answers with byte
            // spans. Spelled out rather than left to a wildcard, so that a node
            // added to the algebra has to say which side of the cost ladder it
            // falls on instead of silently arriving here and being rejected as
            // an expression that "needs a kernel".
            Expr::Src(_)
            | Expr::Out(_)
            | Expr::Fill { .. }
            | Expr::Slice { .. }
            | Expr::Stride { .. }
            | Expr::Gather { .. }
            | Expr::Concat { .. }
            | Expr::Transmute { .. }
            // `specialize` has already rewritten every shard into this rank's
            // slice, so this case is unreachable. It is listed with the
            // fragment it belongs to rather than made an exception.
            | Expr::Shard { .. }
            // The two group nodes are unreachable here for the same reason and
            // one step earlier: `specialize` resolves an instance's name and
            // its band before typing, so by the time a group plan is lowered
            // they are a `Src` and a `Slice`.
            | Expr::SrcIndexed(_)
            | Expr::Select { .. } => (self.affine(&expr, &decl)?, decl),
        };

        check_declared_encoding(contract, &decl)?;

        // The layout and alignment a contract asks for are properties of the
        // declaration, not of the bytes: nothing moves because of them. So they
        // are simply stated on the realized declaration.
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

    /// Solve the affine expression and emit the copies that satisfy it.
    fn affine(&mut self, expr: &Expr, decl: &TensorDecl) -> Result<Value> {
        let lowering = compile(expr, self.resolver.checked(), MAX_RUNS)?;
        self.copies(&lowering, decl, Role::Declared)
    }

    /// Emit the rectangular copies of a solved expression.
    fn copies(&mut self, lowering: &Lowering, decl: &TensorDecl, role: Role) -> Result<Value> {
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
            match self.leaf(lowering, rect.leaf)? {
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

    /// Lower an [`Expr::Cast`]: pick the kernel that puts `from`'s values into
    /// `to`'s representation.
    ///
    /// The direction is read off the pair of encodings because the pair *is*
    /// the question — an author who names a destination has said everything
    /// there is to say. What changed when `Cast` became a node is where the
    /// pair comes from: it used to be the accident of an expression's inferred
    /// type differing from its declaration's, so a converter ran because two
    /// lines disagreed rather than because anybody asked. See `spec.md` §3.
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
            // `infer_cast` refuses this pair, so reaching it means the node
            // was lowered without being typed.
            (Encoding::Quant(_), Encoding::Quant(_)) => {
                return Err(Error::Internal(
                    "Cast between quantized schemes should have been rejected by infer".to_string(),
                ));
            }
        };
        Ok((value, out))
    }

    /// The bytes an escape hatch reads, and the encoding they are in.
    ///
    /// Anything the affine fragment can say is allowed here, which is what lets
    /// a tensor-parallel rank scale its own shard: the shard is byte spans, the
    /// multiply is the kernel, and only the second one costs anything. A span
    /// the checkpoint can be read through directly stays a source; one that has
    /// to be gathered becomes a temporary buffer the memory planner is free to
    /// reuse once the kernel has read it.
    ///
    /// The encoding returned is the operand's, not the output's — a [`Transmute`]
    /// under the scale is how a contract says the bytes a checkpoint stores as
    /// `U8` are packed elements of a quantization scheme, and that statement is
    /// what tells the kernel to unpack them.
    ///
    /// The leaves may be earlier contracts as well as checkpoint tensors, so a
    /// kernel node is closed under the whole affine fragment and under
    /// [`Expr::Out`]. That is what makes the two-step re-encode
    /// [`Expr::Cast`]'s doc promises actually spellable: name the decoded
    /// tensor with a [`Visibility::Internal`] declaration and cast that. The
    /// only thing still refused is a kernel directly under a kernel, which
    /// `compile` rejects because it lowers to byte runs and a kernel is not
    /// one — naming the intermediate is how a contract sequences two.
    ///
    /// [`Transmute`]: crate::contract::Expr::Transmute
    /// [`Expr::Out`]: crate::contract::Expr::Out
    /// [`Expr::Cast`]: crate::contract::Expr::Cast
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
            self.copies(&lowering, &operand, Role::Operand)?,
            ty.encoding,
        ))
    }

    /// The buffer holding a per-group `Scale`'s factors.
    ///
    /// The factors must be a tensor an earlier contract declared, and the
    /// restriction is deliberate. Scales are published anyway — a driver reads
    /// them for the GEMM that consumes the weight — so referring to that
    /// declaration is what keeps one set of bytes rather than two, and it makes
    /// the pairing a name the contract already checks instead of one this
    /// lowering would have to invent.
    fn scale_factors(&mut self, by: &Expr, what: &str) -> Result<BufferId> {
        let Expr::Out(name) = by else {
            return Err(Error::Internal(format!(
                "'{what}' scale factors should have been rejected by infer"
            )));
        };
        match self.values.get(name).cloned() {
            Some(Value::Buffer(buffer)) => Ok(buffer),
            Some(Value::Source(source)) => {
                // Declared, but still an alias of the checkpoint bytes: the
                // kernel reads factors from memory, so they have to be there.
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

    /// [`Builder::transform`], plus operands the kernel reads beside its input.
    ///
    /// The extras come *after* the input, so `inputs[0]` is still the payload
    /// for every reader that predates them and the extras are still found from
    /// the end. Neither position moves when the payload arrives as a source
    /// extent instead of a buffer.
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

    /// The one transform the algebra cannot denote (`spec.md` §3.5).
    ///
    /// `spec` arrives derived from the operand's type and the declaration, so
    /// by the time this runs the operand *is* the block the kernel reads: no
    /// narrowing is left to do here, and a strided operand -- an interleaved
    /// half, say -- is already a strided extent the executor gathers.
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

    /// Give a value the name it is declared under.
    ///
    /// A [`Visibility::Public`] entry is promoted out of the temporary pool and
    /// finalized, which is the pair of steps that make it a runtime weight: the
    /// promotion puts it in the persistent arena, and the `Finalize` is how the
    /// driver learns its name. An [`Visibility::Internal`] entry gets neither,
    /// so it stays a temporary the memory planner may reuse and the driver never
    /// hears about — while still being declared, because later entries resolve
    /// it through [`Expr::Out`].
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
                // Still a copy: an internal name that aliased the checkpoint
                // would have no buffer for a kernel to read its factors from.
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
    ///
    /// Total, and that is the point. A quantized weight without its scales is
    /// not a smaller weight, it is an unreadable one, so every way this could
    /// decline to produce them is a way to publish a tensor no kernel can use.
    /// Each used to be a bare `return Vec::new()`; each is now an error naming
    /// the declaration and what about it the encode path cannot express.
    fn quant_metadata_outputs(&mut self, decl: &TensorDecl) -> Result<Vec<TensorDecl>> {
        let Encoding::Quant(spec) = &decl.encoding else {
            return Err(Error::Internal(
                "quant_metadata_outputs on a non-quantized declaration".to_string(),
            ));
        };
        let layout = ScaleLayout::for_encode(spec.scheme, &decl.shape)
            .map_err(|err| annotate(err, &decl.name))?;
        // Both metadata tensors are allocated before the attachment is pushed,
        // because the attachment has to name the zero point and an affine weight
        // whose zero point is unnamed is not a weight with a defaulted offset --
        // it is one no kernel can dequantize.
        let scales = self.generated_metadata_decl(decl, &layout, layout.suffix)?;
        let zero_point = match layout.zero_point_suffix {
            Some(suffix) => Some(self.generated_metadata_decl(decl, &layout, suffix)?),
            None => None,
        };
        // Stated here because here is where both halves are known. The
        // granularity is the encode kernel's, describing the layout it writes,
        // not anything readable off `spec`.
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
    /// whether it carries `decl`'s type.
    ///
    /// The two coincide for everything except an [`Visibility::Internal`]
    /// declaration, which is both: temporary, because nothing binds it and the
    /// memory planner should be free to reuse it, and typed, because a kernel
    /// that reads it -- the multiply in a `Scale`, say -- needs to know what
    /// its elements are.
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
            bytes,
            alignment: decl.alignment,
            temporary,
            persistent_offset: None,
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
            bytes: 0,
            alignment: decl.alignment,
            temporary: false,
            persistent_offset: None,
        });
        if declared {
            self.declare(decl.clone());
        }
        buffer
    }

    /// Give `buffer` `decl`'s type without promoting it out of the temporary
    /// pool.
    ///
    /// What an [`Visibility::Internal`] declaration needs and
    /// [`Builder::promote_buffer`] cannot give it: the type, so a kernel can
    /// read the buffer, but not the persistence, because nothing binds it.
    fn attach_type(&mut self, buffer: BufferId, decl: &TensorDecl) -> Result<()> {
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
        existing.tensor = Some(decl.id);
        self.declare(decl.clone());
        Ok(())
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

/// Hold a declaration to the shape it predicted.
///
/// A declaration that declined to predict has nothing to check; one that
/// predicted is held to it. Shared by `affine` and the escape hatches so a node
/// added to the latter cannot quietly opt out of the check.
/// The quantization scheme a per-group `Scale` is unpacking, if any.
///
/// `None` says the elements are already numbers and only the multiply is
/// wanted; `Some` says they have to be unpacked first, and names how.
/// Elements of the operand per factor, on each axis.
///
/// `infer_scale_per_block` has already checked equal rank and an exact
/// division, so a failure here is a compiler fault rather than a bad contract.
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

/// The scale tensor an encode kernel writes beside its output.
///
/// The one place the pairing between a scheme and the bytes its encoder emits
/// for scales is written down. It lives here rather than on [`QuantSpec`]
/// because none of it is readable off the spec: the granularity, the layout and
/// the form are the *kernel's*, describing what it writes, and a checkpoint that
/// ships the same scheme may well store its scales some other way.
struct ScaleLayout {
    /// Appended to the weight's declared name. Two conventions, inherited from
    /// what the drivers already look for.
    suffix: &'static str,
    /// The zero point's suffix, for an affine scheme, or `None` for a symmetric
    /// one. Its shape and encoding are the scales' -- an affine scheme offsets
    /// exactly the groups it scales -- so only the name differs and only the
    /// name is stated.
    zero_point_suffix: Option<&'static str>,
    /// Whether the suffixes extend the weight's name or replace its last
    /// component.
    ///
    /// Both conventions are in the wild and neither is derivable: a scheme whose
    /// encoder was written here names `w` and `w_scale`, while MLX names the
    /// triplet `w.weight`, `w.scales`, `w.biases` -- siblings, not descendants.
    /// A driver binding a converted checkpoint looks the shipped names up, so an
    /// encode that produced descendants would publish tensors correct in every
    /// respect except findable.
    naming: MetaNaming,
    shape: Vec<i64>,
    encoding: Encoding,
    granularity: QuantGranularity,
    group_size: u32,
    channel_axis: u32,
    scale_form: ScaleForm,
}

/// Where a generated metadata tensor's name is rooted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MetaNaming {
    /// `w` publishes `w<suffix>`.
    Extend,
    /// `w.weight` publishes `w<suffix>`. Refused, rather than silently treated
    /// as [`Self::Extend`], for a declaration not ending in `.weight`: the
    /// scheme's whole naming convention rests on that component being there.
    ReplaceWeight,
}

impl MetaNaming {
    fn apply(self, name: &str, suffix: &str) -> Result<String> {
        match self {
            Self::Extend => Ok(format!("{name}{suffix}")),
            Self::ReplaceWeight => match name.strip_suffix(".weight") {
                Some(stem) => Ok(format!("{stem}{suffix}")),
                None => Err(Error::Contract(format!(
                    "'{name}' is encoded into a scheme whose scales are named \
                     beside a '.weight', but its declared name does not end in one"
                ))),
            },
        }
    }
}

impl ScaleLayout {
    /// What encoding `shape` into `scheme` publishes, or why it cannot.
    ///
    /// Every encode kernel in the tree is two-dimensional -- it walks `[rows,
    /// cols]` tiles -- so the rank restriction is not a simplification, it is
    /// the kernel's actual domain. A rank-3 declaration is refused here instead
    /// of silently producing a scale-less weight, which is what a contract that
    /// stacks experts and then asks for runtime quantization used to get.
    fn for_encode(scheme: QuantScheme, shape: &[i64]) -> Result<Self> {
        let [rows, cols] = shape else {
            return Err(Error::Contract(format!(
                "encoding to {scheme:?} writes [rows, cols] tiles, so it cannot \
                 produce scales for the rank-{} shape {shape:?}",
                shape.len()
            )));
        };
        match scheme {
            QuantScheme::Fp8E4M3 | QuantScheme::Int8Symmetric => Ok(Self {
                suffix: "_scale_inv",
                zero_point_suffix: None,
                naming: MetaNaming::Extend,
                // `[rows]` of F32, one per output channel.
                shape: vec![*rows],
                encoding: Encoding::Raw(DType::F32),
                granularity: QuantGranularity::PerChannel,
                group_size: 0,
                channel_axis: 0,
                scale_form: ScaleForm::F32Factors,
            }),
            // E8M0 block scale: one uint8 per 32-element block along K. The
            // encode-tile kernel writes a row-major `[rows, cols/32]` byte
            // tensor, and the MXFP4 kernels want those bytes as they are.
            QuantScheme::Mxfp4E2M1E8M0 => {
                if cols % 32 != 0 {
                    return Err(Error::Contract(format!(
                        "encoding to MXFP4 blocks 32 columns under one scale, \
                         but the output has {cols}"
                    )));
                }
                Ok(Self {
                    suffix: "_scale",
                    zero_point_suffix: None,
                    naming: MetaNaming::Extend,
                    shape: vec![*rows, cols / 32],
                    encoding: Encoding::Raw(DType::U8),
                    granularity: QuantGranularity::PerGroup,
                    group_size: 32,
                    channel_axis: 1,
                    scale_form: ScaleForm::RawE8M0,
                })
            }
            // MLX's affine U4: 64 columns under one BF16 scale AND one BF16
            // zero point, so an element is `code * scale + zero`. The two are
            // shaped alike and written by one kernel pass -- the pass that found
            // each group's min and max had to see both to produce either.
            //
            // `.scales` and `.biases` are MLX's own names, and the drivers that
            // read a converted checkpoint already look for exactly those, so an
            // encoded weight is bindable by the code that binds a shipped one.
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
                    shape: vec![*rows, cols / 64],
                    encoding: Encoding::Raw(DType::BF16),
                    granularity: QuantGranularity::PerGroup,
                    group_size: 64,
                    channel_axis: 1,
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

/// The declared shape is a claim about the expression, checked against it.
///
/// `TensorContract::shape` is optional, so this is a no-op for a contract that
/// declares none. The caller names every error it returns, once.
fn check_declared_shape(contract: &TensorContract, ty: &TensorType) -> Result<()> {
    if let Some(declared) = &contract.shape
        && ty.shape != *declared
    {
        return Err(Error::Contract(format!(
            "declares shape {declared:?} but its expression yields {:?}",
            ty.shape
        )));
    }
    Ok(())
}

/// The declared encoding is a claim about the expression, not an instruction.
///
/// It was an instruction until [`Expr::Cast`] existed: a declaration whose
/// encoding differed from what its expression yielded used to *cause* a
/// conversion, so the same two lines meant "check this" or "run a kernel"
/// depending on whether they happened to agree. Now the kernel has a name and
/// this is only the check.
///
/// Like [`check_declared_shape`], this leaves the contract unnamed: the caller
/// names every error it returns, once.
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
