use std::collections::{HashMap, HashSet};

use crate::checkpoint::{CheckpointMetadata, RawTensor};
use crate::contract::ModelContract;
use crate::error::CompileError;
use crate::frontend::{plan_from_contracts, runtime_bytes};
use crate::ir::{GatherPiece, LayoutExpr, LayoutPlan};
use crate::load_plan::{
    BufferDecl, DestExtent, DimSpec, LoadPlan, SlabPlacement, SourceExtent, StorageInstr,
    StorageTarget, StridedExtent, TileMapKind, TileSpec, TransformSpec,
};
use crate::optimizer::{OptimizerPassStats, optimize_with_report};
use crate::typecheck::typecheck;
use crate::types::{
    Axis, BufferId, DType, Encoding, ExprId, InstrId, QuantScheme, RepackLayout, RepackSpec,
    TensorDecl, TensorId, encoding_dense_element_bytes, encoding_nbytes, tensor_nbytes,
};

mod arena;
mod extents;
mod memory;
mod passes;
mod rewrite;

use arena::{assign_persistent_offsets, validate_persistent_layout};
use extents::{
    buffer_bytes, byte_extent, dtype_to_quant_marker, extent_storage_bytes, full_dest_extent,
    gather_extents, instr_by_id, narrow_repack_source, repack_stage_bytes, source_is_dense,
    storage_extent_for_shape, strided_physical_source_bytes,
};
use memory::recompute_memory_plan;
use passes::{
    build_slab_scatter_writes, coalesce_persistent_arena_writes, hoist_bulk_extent_writes,
    merge_adjacent_extent_writes, validate_target_support,
};

pub fn compile_load_plan(
    metadata: &CheckpointMetadata,
    contract: &ModelContract,
    target: StorageTarget,
) -> Result<LoadPlan, CompileError> {
    let contract = rewrite::coalesce_direct_row_shards(contract, metadata, &target)?;
    let plan = plan_from_contracts(metadata, &contract, &target)?;
    let optimized = optimize_with_report(plan)?;
    let mut program = lower_layout_plan(metadata, &optimized.plan, target)?;
    program.optimizer = optimized.report;
    Ok(program)
}

pub fn lower_layout_plan(
    metadata: &CheckpointMetadata,
    plan: &LayoutPlan,
    target: StorageTarget,
) -> Result<LoadPlan, CompileError> {
    typecheck(plan)?;
    let mut compiler = StorageCompiler {
        metadata,
        plan,
        program: LoadPlan::empty(target),
        values: HashMap::new(),
        finalized_names: HashSet::new(),
        next_buffer: 0,
        next_instr: 0,
    };
    compiler.program.files = metadata
        .files
        .iter()
        .map(|file| crate::load_plan::CheckpointFileDecl {
            id: file.id,
            path: file.path.clone(),
            size_bytes: file.size_bytes,
            format: file.format,
        })
        .collect();
    compiler.program.sources = metadata
        .tensors
        .iter()
        .map(|tensor| crate::load_plan::SourceTensorDecl {
            id: tensor.id,
            name: tensor.name.clone(),
            file_id: tensor.file_id,
            file_offset: tensor.file_offset,
            span_bytes: tensor.span_bytes,
            shape: tensor.shape.clone(),
            encoding: tensor.encoding.clone(),
        })
        .collect();
    compiler.lower()?;
    // The backend pass runs here, not in `compile_load_plan`, so a plan is never
    // observable in a state where its tiling and fusion fields are still
    // placeholders.
    let mut program = compiler.program;
    crate::backend::lower(&mut program);
    program.attachments = crate::load_plan::derive_quant_attachments(&program.tensors);
    Ok(program)
}

#[derive(Clone, Debug)]
enum ValueLoc {
    Source(SourceView),
    Buffer(BufferId),
}

#[derive(Clone, Debug)]
struct SourceView {
    tensor_id: TensorId,
    shape: Vec<i64>,
    encoding: Encoding,
    offset_bytes: u64,
    stride: StridedExtent,
}

struct StorageCompiler<'a> {
    metadata: &'a CheckpointMetadata,
    plan: &'a LayoutPlan,
    program: LoadPlan,
    values: HashMap<ExprId, ValueLoc>,
    finalized_names: HashSet<String>,
    next_buffer: u32,
    next_instr: u32,
}

impl StorageCompiler<'_> {
    fn lower(&mut self) -> Result<(), CompileError> {
        let mut reachable = HashSet::new();
        for output in &self.plan.outputs {
            mark_reachable(self.plan, *output, &mut reachable)?;
        }
        let mut ids = reachable.into_iter().collect::<Vec<_>>();
        ids.sort_by_key(|id| id.0);
        for expr_id in ids {
            let value = self.lower_expr(expr_id, &self.plan.exprs[expr_id.0 as usize])?;
            self.values.insert(expr_id, value);
        }
        assign_persistent_offsets(&mut self.program)?;
        // `BulkExtentWrite` / `SlabScatter` address the single contiguous device
        // arena every backend loads into. There was a guard here for a backend
        // that had no arena and needed plain per-buffer `ExtentWrite`s; no such
        // backend was ever written, and the guard admitted all three.
        coalesce_persistent_arena_writes(&mut self.program)?;
        hoist_bulk_extent_writes(&mut self.program)?;
        build_slab_scatter_writes(&mut self.program)?;
        merge_adjacent_extent_writes(&mut self.program)?;
        recompute_memory_plan(&mut self.program)?;
        validate_target_support(&self.program)?;
        validate_persistent_layout(&self.program)?;
        Ok(())
    }

    fn lower_expr(&mut self, id: ExprId, expr: &LayoutExpr) -> Result<ValueLoc, CompileError> {
        match expr {
            LayoutExpr::Source { tensor, decl } => {
                let raw = self.raw(*tensor)?;
                Ok(ValueLoc::Source(SourceView {
                    tensor_id: *tensor,
                    shape: decl.shape.clone(),
                    encoding: decl.encoding.clone(),
                    offset_bytes: 0,
                    stride: storage_extent_for_shape(&decl.shape, &decl.encoding)?,
                }))
                .and_then(|value| {
                    if raw.shape != decl.shape {
                        Err(CompileError::InvalidInput(format!(
                            "source expr {} shape {:?} does not match raw '{}' shape {:?}",
                            id.0, decl.shape, raw.name, raw.shape
                        )))
                    } else if encoding_dense_element_bytes(&decl.encoding).is_none()
                        && encoding_nbytes(&decl.shape, &decl.encoding) != Some(raw.span_bytes)
                    {
                        Err(CompileError::InvalidInput(format!(
                            "source expr {} packed tensor '{}' has non-affine physical size {}; use ByteSpans or explicit quant block metadata",
                            id.0, raw.name, raw.span_bytes
                        )))
                    } else {
                        Ok(value)
                    }
                })
            }
            LayoutExpr::Gather {
                inputs,
                pieces,
                zero_fill,
                decl,
            } => self.lower_gather(id, inputs, pieces, *zero_fill, decl),
            LayoutExpr::Cast { input, dtype, .. } => self.lower_tiled_unary(
                id,
                *input,
                TileMapKind::Cast,
                TransformSpec {
                    from: None,
                    to: Some(dtype_to_quant_marker(*dtype)),
                    ..TransformSpec::default()
                },
            ),
            LayoutExpr::Decode {
                scheme,
                data,
                metadata,
                ..
            } => self.lower_tiled_with_metadata(
                id,
                *data,
                metadata,
                TileMapKind::Decode,
                TransformSpec {
                    from: Some(*scheme),
                    to: None,
                    ..TransformSpec::default()
                },
            ),
            LayoutExpr::Encode {
                scheme,
                input,
                metadata_outputs,
                ..
            } => self.lower_encode(id, *input, *scheme, metadata_outputs),
            LayoutExpr::Transcode {
                from,
                to,
                data,
                metadata,
                ..
            } => self.lower_tiled_with_metadata(
                id,
                *data,
                metadata,
                TileMapKind::Transcode,
                TransformSpec {
                    from: Some(*from),
                    to: Some(*to),
                    ..TransformSpec::default()
                },
            ),
            LayoutExpr::Repack { input, spec, .. } => self.lower_repack(id, *input, *spec),
            LayoutExpr::Realize {
                input,
                runtime_name,
                decl,
            } => self.lower_realize(id, *input, runtime_name, decl),
        }
    }

    fn lower_realize(
        &mut self,
        id: ExprId,
        input: ExprId,
        runtime_name: &str,
        decl: &TensorDecl,
    ) -> Result<ValueLoc, CompileError> {
        if !self.finalized_names.insert(runtime_name.to_string()) {
            return Err(CompileError::InvalidInput(format!(
                "duplicate runtime tensor '{}'",
                runtime_name
            )));
        }
        match self.value(input)? {
            ValueLoc::Buffer(buffer) => {
                self.promote_buffer(buffer, decl)?;
                let instr = self.next_instr();
                self.program.instrs.push(StorageInstr::Finalize {
                    id: instr,
                    tensor: buffer,
                    name: runtime_name.to_string(),
                });
                self.program.schedule.push(instr);
                Ok(ValueLoc::Buffer(buffer))
            }
            ValueLoc::Source(source) => {
                let buffer = self.allocate_decl(decl, false)?;
                self.emit_extent_write(source, buffer, 0, &decl.shape)?;
                let finalize = self.next_instr();
                self.program.instrs.push(StorageInstr::Finalize {
                    id: finalize,
                    tensor: buffer,
                    name: runtime_name.to_string(),
                });
                self.program.schedule.push(finalize);
                self.values.insert(id, ValueLoc::Buffer(buffer));
                Ok(ValueLoc::Buffer(buffer))
            }
        }
    }

    /// Lower a compiled affine expression.
    ///
    /// `contract::compile` has already resolved slicing, sharding,
    /// concatenation, stacking and reshaping into rectangular copies, so there
    /// is no algebra left here — only the choice between aliasing and copying.
    /// Two cases must stay zero-copy or the plan regresses badly:
    ///
    /// * a single piece over a checkpoint tensor stays a lazy [`SourceView`],
    ///   so a shard is read straight out of the file at an offset rather than
    ///   staged through a temporary;
    /// * a single contiguous piece over an already-materialized buffer becomes
    ///   a `CreateView`, so republishing a slice of a fused tensor under
    ///   another name costs nothing.
    fn lower_gather(
        &mut self,
        id: ExprId,
        inputs: &[ExprId],
        pieces: &[GatherPiece],
        zero_fill: bool,
        decl: &TensorDecl,
    ) -> Result<ValueLoc, CompileError> {
        if zero_fill {
            return Err(CompileError::InvalidInput(format!(
                "Gather expr {} needs a zero fill, which the storage program \
                 has no instruction for",
                id.0
            )));
        }
        let output_bytes = encoding_nbytes(&decl.shape, &decl.encoding).ok_or_else(|| {
            CompileError::InvalidInput(format!("Gather expr {} size overflow", id.0))
        })?;
        let input_of = |piece: &GatherPiece| -> Result<ExprId, CompileError> {
            inputs.get(piece.input as usize).copied().ok_or_else(|| {
                CompileError::InvalidInput(format!(
                    "Gather expr {} names input {}",
                    id.0, piece.input
                ))
            })
        };

        if let [piece] = pieces
            && piece.dst_offset == 0
            && piece.bytes() == output_bytes
        {
            let input = input_of(piece)?;
            match self.value(input)? {
                ValueLoc::Source(source) if source_is_dense(&source)? => {
                    let (stride, _) = gather_extents(piece)?;
                    return Ok(ValueLoc::Source(SourceView {
                        tensor_id: source.tensor_id,
                        shape: decl.shape.clone(),
                        encoding: decl.encoding.clone(),
                        offset_bytes: source.offset_bytes + piece.src_offset,
                        stride,
                    }));
                }
                ValueLoc::Buffer(buffer) if piece.is_contiguous() => {
                    let out = self.declare_view_buffer(decl);
                    let instr = self.next_instr();
                    self.program.instrs.push(StorageInstr::CreateView {
                        id: instr,
                        input: buffer,
                        output: out,
                        view: DestExtent {
                            buffer: out,
                            offset: piece.src_offset,
                            stride: storage_extent_for_shape(&decl.shape, &decl.encoding)?,
                        },
                    });
                    self.program.schedule.push(instr);
                    return Ok(ValueLoc::Buffer(out));
                }
                _ => {}
            }
        }

        let out = self.allocate_decl(decl, true)?;
        for piece in pieces {
            let input = input_of(piece)?;
            match self.value(input)? {
                ValueLoc::Source(source) => {
                    let raw = self.raw(source.tensor_id)?;
                    let (file_id, tensor_id, base) = (raw.file_id, raw.id, raw.file_offset);
                    let (src_stride, dst_stride) = gather_extents(piece)?;
                    let file_offset = base + source.offset_bytes + piece.src_offset;
                    let instr = self.next_instr();
                    self.program.instrs.push(StorageInstr::ExtentWrite {
                        id: instr,
                        source: SourceExtent {
                            file_id,
                            tensor_id,
                            file_offset,
                            span_bytes: piece.bytes(),
                            stride: src_stride,
                        },
                        dest: DestExtent {
                            buffer: out,
                            offset: piece.dst_offset,
                            stride: dst_stride,
                        },
                    });
                    self.program.schedule.push(instr);
                }
                ValueLoc::Buffer(buffer) => {
                    let input_decl = self.plan.decl(input).ok_or_else(|| {
                        CompileError::InvalidInput(format!("Gather input {} has no decl", input.0))
                    })?;
                    let input_bytes = encoding_nbytes(&input_decl.shape, &input_decl.encoding);
                    if !piece.is_contiguous()
                        || piece.src_offset != 0
                        || input_bytes != Some(piece.bytes())
                    {
                        return Err(CompileError::InvalidInput(format!(
                            "Gather expr {} takes a strided or partial slice of a \
                             computed buffer; only whole-buffer moves are lowered",
                            id.0
                        )));
                    }
                    self.emit_view_or_tile(
                        TileMapKind::Reblock,
                        None,
                        Some(DestExtent {
                            buffer: out,
                            offset: piece.dst_offset,
                            stride: storage_extent_for_shape(
                                &input_decl.shape,
                                &input_decl.encoding,
                            )?,
                        }),
                        vec![buffer],
                        vec![out],
                        TransformSpec::default(),
                    );
                }
            }
        }
        Ok(ValueLoc::Buffer(out))
    }

    fn lower_tiled_unary(
        &mut self,
        id: ExprId,
        input: ExprId,
        kind: TileMapKind,
        transform: TransformSpec,
    ) -> Result<ValueLoc, CompileError> {
        self.lower_tiled_with_metadata(id, input, &[], kind, transform)
    }

    fn lower_encode(
        &mut self,
        id: ExprId,
        input: ExprId,
        scheme: QuantScheme,
        metadata_outputs: &[TensorDecl],
    ) -> Result<ValueLoc, CompileError> {
        if metadata_outputs.is_empty() {
            return self.lower_tiled_unary(
                id,
                input,
                TileMapKind::Encode,
                TransformSpec {
                    from: None,
                    to: Some(scheme),
                    ..TransformSpec::default()
                },
            );
        }

        let out_decl = self.plan.decl(id).ok_or_else(|| {
            CompileError::InvalidInput(format!("expr {} has no tensor decl", id.0))
        })?;
        let out = self.allocate_decl(out_decl, false)?;
        let mut inputs = Vec::new();
        let source = match self.value(input)? {
            ValueLoc::Source(source) => Some(self.source_extent(&source)?),
            ValueLoc::Buffer(buffer) => {
                inputs.push(buffer);
                None
            }
        };
        let transform = self.with_block_scale_source(
            TransformSpec {
                from: None,
                to: Some(scheme),
                ..TransformSpec::default()
            },
            source.as_ref(),
        );
        let mut outputs = Vec::with_capacity(metadata_outputs.len() + 1);
        outputs.push(out);
        for metadata in metadata_outputs {
            outputs.push(self.allocate_decl(metadata, false)?);
        }
        self.emit_view_or_tile(
            TileMapKind::Encode,
            source,
            None,
            inputs,
            outputs.clone(),
            transform,
        );
        for (decl, buffer) in metadata_outputs.iter().zip(outputs.iter().skip(1)) {
            if !self.finalized_names.insert(decl.name.clone()) {
                return Err(CompileError::InvalidInput(format!(
                    "duplicate runtime tensor '{}'",
                    decl.name
                )));
            }
            let instr = self.next_instr();
            self.program.instrs.push(StorageInstr::Finalize {
                id: instr,
                tensor: *buffer,
                name: decl.name.clone(),
            });
            self.program.schedule.push(instr);
        }
        Ok(ValueLoc::Buffer(out))
    }

    fn lower_tiled_with_metadata(
        &mut self,
        id: ExprId,
        input: ExprId,
        metadata: &[ExprId],
        kind: TileMapKind,
        transform: TransformSpec,
    ) -> Result<ValueLoc, CompileError> {
        let out = self.allocate_expr(id, true)?;
        let mut inputs = Vec::with_capacity(metadata.len() + 1);
        let source = match self.value(input)? {
            ValueLoc::Source(source) => Some(self.source_extent(&source)?),
            ValueLoc::Buffer(buffer) => {
                inputs.push(buffer);
                None
            }
        };
        let transform = self.with_block_scale_source(transform, source.as_ref());
        for meta in metadata {
            inputs.push(self.ensure_buffer(*meta)?);
        }
        let decl = self.plan.decl(id).ok_or_else(|| {
            CompileError::InvalidInput(format!("expr {} has no tensor decl", id.0))
        })?;
        self.emit_view_or_tile(
            kind,
            source,
            Some(full_dest_extent(out, decl)?),
            inputs,
            vec![out],
            transform,
        );
        Ok(ValueLoc::Buffer(out))
    }

    fn lower_repack(
        &mut self,
        id: ExprId,
        input: ExprId,
        spec: RepackSpec,
    ) -> Result<ValueLoc, CompileError> {
        let out = self.allocate_expr(id, true)?;
        let mut inputs = Vec::new();
        let mut repack = spec;
        let (source, input_bytes) = match self.value(input)? {
            ValueLoc::Source(source) => {
                let (source, narrowed) = narrow_repack_source(source, spec)?;
                repack = narrowed;
                let extent = self.source_extent(&source)?;
                let bytes = extent.span_bytes;
                (Some(extent), bytes)
            }
            ValueLoc::Buffer(buffer) => {
                let bytes = buffer_bytes(&self.program, buffer)?;
                inputs.push(buffer);
                (None, bytes)
            }
        };
        let decl = self.plan.decl(id).ok_or_else(|| {
            CompileError::InvalidInput(format!("expr {} has no tensor decl", id.0))
        })?;
        let stage_bytes = repack_stage_bytes(repack)?;
        self.emit_view_or_tile(
            TileMapKind::Repack,
            source,
            Some(full_dest_extent(out, decl)?),
            inputs,
            vec![out],
            TransformSpec {
                repack,
                scratch_bytes: input_bytes.checked_add(stage_bytes).ok_or_else(|| {
                    CompileError::InvalidInput("Repack scratch byte overflow".to_string())
                })?,
                ..TransformSpec::default()
            },
        );
        Ok(ValueLoc::Buffer(out))
    }

    fn ensure_buffer(&mut self, expr: ExprId) -> Result<BufferId, CompileError> {
        match self.value(expr)? {
            ValueLoc::Buffer(buffer) => Ok(buffer),
            ValueLoc::Source(source) => {
                let decl = self.plan.decl(expr).ok_or_else(|| {
                    CompileError::InvalidInput(format!("expr {} has no tensor decl", expr.0))
                })?;
                let buffer = self.allocate_decl(decl, true)?;
                self.emit_extent_write(source, buffer, 0, &decl.shape)?;
                Ok(buffer)
            }
        }
    }

    fn emit_view_or_tile(
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
                // A budget, not yet a decision: `backend::lower` turns it into a
                // row count once the whole plan exists.
                rows_per_tile: 0,
            },
            transform,
        });
        self.program.schedule.push(instr);
    }

    fn emit_extent_write(
        &mut self,
        source: SourceView,
        dest: BufferId,
        dest_offset: u64,
        shape: &[i64],
    ) -> Result<(), CompileError> {
        let source_extent = self.source_extent(&source)?;
        let bytes = encoding_nbytes(shape, &source.encoding).ok_or_else(|| {
            CompileError::InvalidInput("extent write byte size overflow".to_string())
        })?;
        let instr = self.next_instr();
        self.program.instrs.push(StorageInstr::ExtentWrite {
            id: instr,
            source: SourceExtent {
                span_bytes: bytes,
                ..source_extent
            },
            dest: DestExtent {
                buffer: dest,
                offset: dest_offset,
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
    fn block_scale_source(&self, source: &SourceExtent) -> Option<TensorId> {
        let raw = self.metadata.tensor(source.tensor_id)?;
        if !matches!(raw.encoding, Encoding::Raw(DType::F8E4M3)) {
            return None;
        }
        self.metadata
            .tensor_by_name(&format!("{}_scale_inv", raw.name))
            .map(|scale| scale.id)
    }

    fn with_block_scale_source(
        &self,
        mut transform: TransformSpec,
        source: Option<&SourceExtent>,
    ) -> TransformSpec {
        transform.metadata_source = source.and_then(|source| self.block_scale_source(source));
        transform
    }

    fn source_extent(&self, source: &SourceView) -> Result<SourceExtent, CompileError> {
        let raw = self.raw(source.tensor_id)?;
        let span_bytes = encoding_nbytes(&source.shape, &source.encoding).ok_or_else(|| {
            CompileError::InvalidInput("source extent byte size overflow".to_string())
        })?;
        let physical_bytes = strided_physical_source_bytes(&source.stride)?;
        if source.offset_bytes + physical_bytes > raw.span_bytes {
            return Err(CompileError::InvalidInput(format!(
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
        })
    }

    fn allocate_expr(&mut self, expr: ExprId, temporary: bool) -> Result<BufferId, CompileError> {
        let decl = self.plan.decl(expr).ok_or_else(|| {
            CompileError::InvalidInput(format!("expr {} has no tensor decl", expr.0))
        })?;
        self.allocate_decl(decl, temporary)
    }

    fn allocate_decl(
        &mut self,
        decl: &TensorDecl,
        temporary: bool,
    ) -> Result<BufferId, CompileError> {
        let buffer = BufferId(self.next_buffer);
        self.next_buffer += 1;
        let bytes = runtime_bytes(&decl.shape, &decl.encoding)?;
        self.program.buffers.push(BufferDecl {
            id: buffer,
            tensor: if temporary { None } else { Some(decl.id) },
            bytes,
            alignment: decl.alignment,
            temporary,
            persistent_offset: None,
        });
        if !temporary {
            self.program.tensors.push(decl.clone());
        }
        let instr = self.next_instr();
        self.program
            .instrs
            .push(StorageInstr::Allocate { id: instr, buffer });
        self.program.schedule.push(instr);
        Ok(buffer)
    }

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
        if !self
            .program
            .tensors
            .iter()
            .any(|tensor| tensor.id == decl.id)
        {
            self.program.tensors.push(decl.clone());
        }
        buffer
    }

    fn promote_buffer(&mut self, buffer: BufferId, decl: &TensorDecl) -> Result<(), CompileError> {
        let existing = self
            .program
            .buffers
            .iter_mut()
            .find(|candidate| candidate.id == buffer)
            .ok_or_else(|| {
                CompileError::InvalidInput(format!("buffer {} does not exist", buffer.0))
            })?;
        if let Some(existing_id) = existing.tensor
            && existing_id != decl.id
        {
            return Err(CompileError::InvalidInput(format!(
                "buffer {} already belongs to tensor {}, cannot promote to {}",
                buffer.0, existing_id.0, decl.id.0
            )));
        }
        existing.tensor = Some(decl.id);
        if existing.temporary {
            existing.temporary = false;
        }
        if let Some(tensor) = self
            .program
            .tensors
            .iter_mut()
            .find(|tensor| tensor.id == decl.id)
        {
            *tensor = decl.clone();
        } else {
            self.program.tensors.push(decl.clone());
        }
        Ok(())
    }

    fn value(&self, id: ExprId) -> Result<ValueLoc, CompileError> {
        self.values.get(&id).cloned().ok_or_else(|| {
            CompileError::InvalidInput(format!("expr {} has not been lowered", id.0))
        })
    }

    fn raw(&self, id: TensorId) -> Result<&RawTensor, CompileError> {
        self.metadata
            .tensor(id)
            .ok_or_else(|| CompileError::InvalidInput(format!("missing source tensor {}", id.0)))
    }

    fn next_instr(&mut self) -> InstrId {
        let id = InstrId(self.next_instr);
        self.next_instr += 1;
        id
    }
}

fn mark_reachable(
    plan: &LayoutPlan,
    id: ExprId,
    reachable: &mut HashSet<ExprId>,
) -> Result<(), CompileError> {
    if id.0 as usize >= plan.exprs.len() {
        return Err(CompileError::InvalidInput(format!(
            "layout output expr {} is out of range",
            id.0
        )));
    }
    if !reachable.insert(id) {
        return Ok(());
    }
    for input in plan.exprs[id.0 as usize].inputs() {
        mark_reachable(plan, input, reachable)?;
    }
    Ok(())
}

#[cfg(test)]
mod persistent_layout_tests {
    use super::passes::try_merge_bulk_extent_write;
    use super::*;
    use crate::types::FileId;

    fn operand(id: u32, bytes: u64, alignment: u32, offset: Option<u64>) -> BufferDecl {
        BufferDecl {
            id: BufferId(id),
            tensor: Some(TensorId(id)),
            bytes,
            alignment,
            temporary: false,
            persistent_offset: offset,
        }
    }

    fn program_with(buffers: Vec<BufferDecl>) -> LoadPlan {
        let mut p = LoadPlan::empty(StorageTarget {
            preferred_alignment: 256,
            ..StorageTarget::default()
        });
        p.buffers = buffers;
        p
    }

    #[test]
    fn accepts_aligned_disjoint_operands() {
        let p = program_with(vec![
            operand(0, 256, 1, Some(0)),
            operand(1, 256, 1, Some(256)),
        ]);
        assert!(validate_persistent_layout(&p).is_ok());
    }

    #[test]
    fn rejects_misaligned_operand_base() {
        // 128 is not a multiple of the fixture target's 256-byte alignment.
        let p = program_with(vec![operand(0, 64, 1, Some(128))]);
        assert!(validate_persistent_layout(&p).is_err());
    }

    #[test]
    fn rejects_overlapping_operands() {
        // [0,512) and [256,512) overlap; both bases are 256-aligned.
        let p = program_with(vec![
            operand(0, 512, 1, Some(0)),
            operand(1, 256, 1, Some(256)),
        ]);
        assert!(validate_persistent_layout(&p).is_err());
    }

    #[test]
    fn rejects_view_escaping_backing() {
        let mut p = program_with(vec![operand(0, 64, 256, Some(0))]);
        p.instrs.push(StorageInstr::CreateView {
            id: InstrId(0),
            input: BufferId(0),
            output: BufferId(1),
            view: DestExtent {
                buffer: BufferId(1),
                offset: 32,
                stride: StridedExtent {
                    base_offset: 0,
                    element_bytes: 1,
                    dims: vec![DimSpec {
                        count: 64,
                        src_stride: 1,
                        dst_stride: 1,
                    }],
                },
            },
        });
        // window [32, 96) escapes the 64-byte backing buffer.
        assert!(validate_persistent_layout(&p).is_err());
    }

    #[test]
    fn bulk_merge_respects_target_tile_bound() {
        let make = |id, file_offset, dest_offset| StorageInstr::BulkExtentWrite {
            id: InstrId(id),
            source: SourceExtent {
                file_id: FileId(0),
                tensor_id: TensorId(id),
                file_offset,
                span_bytes: 8,
                stride: byte_extent(8),
            },
            dest_offset,
        };
        let mut first = make(0, 0, 0);
        let second = make(1, 8, 8);
        assert!(!try_merge_bulk_extent_write(&mut first, &second, 8).unwrap());
        assert!(try_merge_bulk_extent_write(&mut first, &second, 16).unwrap());
    }

    #[test]
    fn target_transform_matrix_matches_host_and_metal_executors() {
        let tile = |kind| StorageInstr::TileMap {
            id: InstrId(0),
            kind,
            source: None,
            dest: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            tile: TileSpec {
                max_tile_bytes: 1,
                rows_per_tile: 0,
            },
            transform: TransformSpec::default(),
        };

        let mut host = LoadPlan::empty(StorageTarget::default());
        host.instrs.push(tile(TileMapKind::Cast));
        assert!(validate_target_support(&host).is_ok());
        host.instrs[0] = tile(TileMapKind::Reorder);
        assert!(validate_target_support(&host).is_err());

        let mut metal = LoadPlan::empty(StorageTarget {
            backend: crate::types::BackendKind::Metal,
            tile_map_mask: crate::load_plan::METAL_TILE_MAP_MASK,
            ..StorageTarget::default()
        });
        metal.instrs.push(tile(TileMapKind::Cast));
        assert!(validate_target_support(&metal).is_err());
    }
}
