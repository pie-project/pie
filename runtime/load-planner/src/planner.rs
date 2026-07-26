use std::collections::{HashMap, HashSet};

use crate::abi::{RuntimeAbi, RuntimeTensorSource};
use crate::error::CompileError;
use crate::frontend::{plan_from_semantics, runtime_bytes};
use crate::ir::{LayoutExpr, LayoutPlan};
use crate::load_plan::{
    BufferDecl, DestExtent, DimSpec, LoadPlan, MetadataSpec, SlabPlacement, SourceExtent,
    StorageInstr, StorageTarget, StridedExtent, TileMapKind, TileSpec, TransformSpec,
};
use crate::optimizer::{OptimizerPassStats, optimize_with_report};
use crate::schema::build_semantic_graph;
use crate::source::{CheckpointMetadata, RawTensor};
use crate::typecheck::typecheck;
use crate::types::{
    Axis, BufferId, DType, Encoding, ExprId, InstrId, QuantScheme, RepackLayout, RepackSpec,
    TensorDecl, TensorId, encoding_dense_element_bytes, encoding_nbytes, tensor_nbytes,
};

mod arena;
mod extents;
mod memory;
mod passes;

use arena::{assign_persistent_offsets, validate_persistent_layout};
use extents::{
    buffer_bytes, byte_extent, dense_axis_offset_bytes, dtype_to_quant_marker,
    extent_storage_bytes, full_dest_extent, instr_by_id, narrow_repack_source, narrow_source_axis,
    repack_stage_bytes, storage_extent_for_shape, strided_physical_source_bytes,
};
use memory::recompute_memory_plan;
use passes::{
    build_slab_scatter_writes, coalesce_persistent_arena_writes, hoist_bulk_extent_writes,
    merge_adjacent_extent_writes, validate_target_support,
};

pub fn compile_load_plan(
    metadata: &CheckpointMetadata,
    cfg: &crate::config::ModelConfig,
    abi: &RuntimeAbi,
    target: StorageTarget,
) -> Result<LoadPlan, CompileError> {
    let abi = abi.coalesce_direct_row_shards(metadata, &target)?;
    let needs_semantic_graph = abi
        .tensors
        .iter()
        .any(|contract| matches!(contract.source, RuntimeTensorSource::Semantic { .. }));
    let graph = if needs_semantic_graph {
        build_semantic_graph(metadata, cfg)?
    } else {
        crate::semantic::SemanticGraph::empty()
    };
    let plan = plan_from_semantics(metadata, &graph, &abi, &target)?;
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
    Ok(compiler.program)
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
        // `BulkExtentWrite` / `SlabScatter` address a single contiguous device
        // arena. Backends without one must keep plain per-buffer
        // `ExtentWrite`s — they have no arena base to resolve an
        // arena-relative destination against.
        if self.program.target.backend.uses_persistent_arena() {
            coalesce_persistent_arena_writes(&mut self.program)?;
            hoist_bulk_extent_writes(&mut self.program)?;
            build_slab_scatter_writes(&mut self.program)?;
        }
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
            LayoutExpr::ByteSpans { spans, decl } => self.lower_byte_spans(spans, decl),
            LayoutExpr::Select {
                input,
                axis,
                start,
                length,
                ..
            } => self.lower_select(id, *input, *axis, *start, *length),
            LayoutExpr::Partition {
                input,
                axis,
                parts,
                index,
                ..
            } => {
                let source = self.value(*input)?;
                let shape = self.plan.decl(*input).ok_or_else(|| {
                    CompileError::InvalidInput(format!("Partition expr {} has tuple input", id.0))
                })?;
                let axis_index = axis.0 as usize;
                let length = shape.shape[axis_index] / i64::from(*parts);
                let start = i64::from(*index) * length;
                match source {
                    ValueLoc::Source(_) => self.lower_select(id, *input, *axis, start, length),
                    ValueLoc::Buffer(buffer) => {
                        let out = self.allocate_expr(id, true)?;
                        let out_decl = self.plan.decl(id).ok_or_else(|| {
                            CompileError::InvalidInput(format!(
                                "Partition expr {} has no decl",
                                id.0
                            ))
                        })?;
                        self.emit_view_or_tile(
                            TileMapKind::Reblock,
                            None,
                            Some(full_dest_extent(out, out_decl)?),
                            vec![buffer],
                            vec![out],
                            TransformSpec::default(),
                        );
                        Ok(ValueLoc::Buffer(out))
                    }
                }
            }
            LayoutExpr::Join { inputs, axis, decl } => self.lower_join(id, inputs, *axis, decl),
            LayoutExpr::Stack { inputs, axis, decl } => self.lower_stack(id, inputs, *axis, decl),
            LayoutExpr::Unzip { .. } => Err(CompileError::InvalidInput(
                "Unzip lowering must be consumed by explicit output selections".to_string(),
            )),
            LayoutExpr::Reorder { input, .. } => {
                self.lower_tiled_unary(id, *input, TileMapKind::Reorder, TransformSpec::default())
            }
            LayoutExpr::View {
                input,
                layout,
                axis,
                start,
                length,
                decl,
            } => {
                if let Some(axis) = axis {
                    return self.lower_select(id, *input, *axis, *start, *length);
                }
                let input_value = self.value(*input)?;
                match input_value {
                    ValueLoc::Buffer(input_buffer) => {
                        let input_decl = self.plan.decl(*input).ok_or_else(|| {
                            CompileError::InvalidInput(format!(
                                "View input {} has no decl",
                                input.0
                            ))
                        })?;
                        if input_decl.shape != decl.shape || input_decl.encoding != decl.encoding {
                            return Err(CompileError::InvalidInput(format!(
                                "metadata-only View {} cannot change shape/encoding",
                                id.0
                            )));
                        }
                        let _ = layout;
                        Ok(ValueLoc::Buffer(input_buffer))
                    }
                    ValueLoc::Source(source) => Ok(ValueLoc::Source(source)),
                }
            }
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
            LayoutExpr::Attach { data, metadata, .. } => {
                let value = self.value(*data)?;
                if metadata.is_empty() {
                    return Ok(value);
                }
                let buffer = self.ensure_buffer(*data)?;
                let mut metadata_buffers = Vec::with_capacity(metadata.len());
                for meta in metadata {
                    metadata_buffers.push(self.ensure_buffer(*meta)?);
                }
                let instr = self.next_instr();
                self.program.instrs.push(StorageInstr::Attach {
                    id: instr,
                    tensor: buffer,
                    metadata: metadata_buffers,
                    spec: MetadataSpec {
                        kind: "quant".to_string(),
                    },
                });
                self.program.schedule.push(instr);
                Ok(ValueLoc::Buffer(buffer))
            }
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

    fn lower_byte_spans(
        &mut self,
        spans: &[crate::ir::ByteSpan],
        decl: &TensorDecl,
    ) -> Result<ValueLoc, CompileError> {
        let out = self.allocate_decl(decl, true)?;
        for span in spans {
            let raw = self.raw(span.tensor)?;
            let source_end = span
                .source_offset_bytes
                .checked_add(span.span_bytes)
                .ok_or_else(|| {
                    CompileError::InvalidInput(format!(
                        "ByteSpans source offset overflow for '{}'",
                        raw.name
                    ))
                })?;
            if source_end > raw.span_bytes {
                return Err(CompileError::InvalidInput(format!(
                    "ByteSpans source range exceeds '{}'",
                    raw.name
                )));
            }
            let file_id = raw.file_id;
            let tensor_id = raw.id;
            let file_offset = raw
                .file_offset
                .checked_add(span.source_offset_bytes)
                .ok_or_else(|| {
                    CompileError::InvalidInput(format!(
                        "ByteSpans file offset overflow for '{}'",
                        raw.name
                    ))
                })?;
            let instr = self.next_instr();
            self.program.instrs.push(StorageInstr::ExtentWrite {
                id: instr,
                source: SourceExtent {
                    file_id,
                    tensor_id,
                    file_offset,
                    span_bytes: span.span_bytes,
                    stride: byte_extent(span.span_bytes),
                },
                dest: DestExtent {
                    buffer: out,
                    offset: span.dest_offset_bytes,
                    stride: byte_extent(span.span_bytes),
                },
            });
            self.program.schedule.push(instr);
        }
        Ok(ValueLoc::Buffer(out))
    }

    fn lower_join(
        &mut self,
        id: ExprId,
        inputs: &[ExprId],
        axis: Axis,
        decl: &TensorDecl,
    ) -> Result<ValueLoc, CompileError> {
        let out = self.allocate_decl(decl, true)?;
        let mut axis_offset = 0i64;
        for input in inputs {
            let input_decl = self.plan.decl(*input).ok_or_else(|| {
                CompileError::InvalidInput(format!("Join expr {} has tuple input", id.0))
            })?;
            let dest_offset =
                dense_axis_offset_bytes(&decl.shape, axis, axis_offset, &decl.encoding)?;
            match self.value(*input)? {
                ValueLoc::Source(source) => {
                    self.emit_extent_write(source, out, dest_offset, &input_decl.shape)?;
                }
                ValueLoc::Buffer(buffer) => {
                    self.emit_view_or_tile(
                        TileMapKind::Reblock,
                        None,
                        Some(DestExtent {
                            buffer: out,
                            offset: dest_offset,
                            stride: storage_extent_for_shape(&input_decl.shape, &decl.encoding)?,
                        }),
                        vec![buffer],
                        vec![out],
                        TransformSpec::default(),
                    );
                }
            }
            axis_offset += input_decl.shape[axis.0 as usize];
        }
        Ok(ValueLoc::Buffer(out))
    }

    fn lower_stack(
        &mut self,
        id: ExprId,
        inputs: &[ExprId],
        axis: Axis,
        decl: &TensorDecl,
    ) -> Result<ValueLoc, CompileError> {
        let out = self.allocate_decl(decl, true)?;
        for (stack_index, input) in inputs.iter().enumerate() {
            let input_decl = self.plan.decl(*input).ok_or_else(|| {
                CompileError::InvalidInput(format!("Stack expr {} has tuple input", id.0))
            })?;
            let dest_offset =
                dense_axis_offset_bytes(&decl.shape, axis, stack_index as i64, &decl.encoding)?;
            match self.value(*input)? {
                ValueLoc::Source(source) => {
                    self.emit_extent_write(source, out, dest_offset, &input_decl.shape)?;
                }
                ValueLoc::Buffer(buffer) => {
                    self.emit_view_or_tile(
                        TileMapKind::Reblock,
                        None,
                        Some(DestExtent {
                            buffer: out,
                            offset: dest_offset,
                            stride: storage_extent_for_shape(&input_decl.shape, &decl.encoding)?,
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

    fn lower_select(
        &mut self,
        id: ExprId,
        input: ExprId,
        axis: Axis,
        start: i64,
        length: i64,
    ) -> Result<ValueLoc, CompileError> {
        match self.value(input)? {
            ValueLoc::Source(source) => Ok(ValueLoc::Source(narrow_source_axis(
                source, axis, start, length,
            )?)),
            ValueLoc::Buffer(input_buffer) => {
                let input_decl = self.plan.decl(input).ok_or_else(|| {
                    CompileError::InvalidInput(format!("Select input {} has no decl", input.0))
                })?;
                let out_decl = self.plan.decl(id).ok_or_else(|| {
                    CompileError::InvalidInput(format!("Select expr {} has no decl", id.0))
                })?;
                let offset =
                    dense_axis_offset_bytes(&input_decl.shape, axis, start, &input_decl.encoding)?;
                let out = self.declare_view_buffer(out_decl);
                let instr = self.next_instr();
                self.program.instrs.push(StorageInstr::CreateView {
                    id: instr,
                    input: input_buffer,
                    output: out,
                    view: DestExtent {
                        buffer: out,
                        offset,
                        stride: storage_extent_for_shape(&out_decl.shape, &out_decl.encoding)?,
                    },
                    layout: out_decl.layout.clone(),
                });
                self.program.schedule.push(instr);
                Ok(ValueLoc::Buffer(out))
            }
        }
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
            TransformSpec {
                from: None,
                to: Some(scheme),
                ..TransformSpec::default()
            },
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
        if let Some(existing_id) = existing.tensor {
            if existing_id != decl.id {
                return Err(CompileError::InvalidInput(format!(
                    "buffer {} already belongs to tensor {}, cannot promote to {}",
                    buffer.0, existing_id.0, decl.id.0
                )));
            }
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
            layout: crate::types::Layout::dense(1),
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
            tile: TileSpec { max_tile_bytes: 1 },
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
