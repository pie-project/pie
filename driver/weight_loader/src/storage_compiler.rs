use std::collections::{HashMap, HashSet};

use crate::abi::{RuntimeAbi, RuntimeTensorSource};
use crate::error::CompileError;
use crate::frontend::{plan_from_semantics, runtime_bytes};
use crate::ir::{LayoutExpr, LayoutPlan};
use crate::optimizer::{OptimizerPassStats, optimize_with_report};
use crate::schema::build_semantic_graph;
use crate::source::{CheckpointMetadata, RawTensor};
use crate::storage::{
    BufferDecl, DestExtent, DimSpec, MetadataSpec, SlabPlacement, SourceExtent, StorageInstr,
    StorageProgram, StorageTarget, StridedExtent, TileMapKind, TileSpec, TransformSpec,
};
use crate::typecheck::typecheck;
use crate::types::{
    Axis, BackendKind, BufferId, DType, Encoding, ExprId, InstrId, QuantScheme, RepackLayout,
    RepackSpec, TensorDecl, TensorId, encoding_dense_element_bytes, encoding_nbytes, tensor_nbytes,
};

pub fn compile_storage_program(
    metadata: &CheckpointMetadata,
    cfg: &crate::config::ModelConfig,
    abi: &RuntimeAbi,
    target: StorageTarget,
) -> Result<StorageProgram, CompileError> {
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
) -> Result<StorageProgram, CompileError> {
    typecheck(plan)?;
    let mut compiler = StorageCompiler {
        metadata,
        plan,
        program: StorageProgram::empty(target),
        values: HashMap::new(),
        finalized_names: HashSet::new(),
        next_buffer: 0,
        next_instr: 0,
    };
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
    program: StorageProgram,
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

fn recompute_memory_plan(program: &mut StorageProgram) -> Result<(), CompileError> {
    let mut persistent_bytes = 0u64;
    let mut live_bytes = 0u64;
    let mut live_peak = 0u64;
    let mut checkpoint_read_bytes = 0u64;
    let mut device_write_bytes = 0u64;
    let mut transform_scratch_peak_bytes = 0u64;
    let mut live = HashSet::new();

    for buffer in &program.buffers {
        if let Some(offset) = buffer.persistent_offset {
            persistent_bytes = persistent_bytes.max(
                offset
                    .checked_add(buffer.bytes)
                    .ok_or_else(|| CompileError::InvalidInput("persistent byte overflow".to_string()))?,
            );
        } else if !buffer.temporary && buffer.tensor.is_some() {
            persistent_bytes = persistent_bytes.checked_add(buffer.bytes).ok_or_else(|| {
                CompileError::InvalidInput("persistent byte overflow".to_string())
            })?;
        }
    }

    for instr_id in &program.schedule {
        let instr = instr_by_id(&program.instrs, *instr_id)?;
        match instr {
            StorageInstr::Allocate { buffer, .. } => {
                let bytes = buffer_bytes(program, *buffer)?;
                if live.insert(*buffer) {
                    live_bytes = live_bytes.checked_add(bytes).ok_or_else(|| {
                        CompileError::InvalidInput("live byte overflow".to_string())
                    })?;
                    live_peak = live_peak.max(live_bytes);
                }
            }
            StorageInstr::Release { buffer, .. } => {
                if live.remove(buffer) {
                    live_bytes = live_bytes
                        .checked_sub(buffer_bytes(program, *buffer)?)
                        .ok_or_else(|| {
                            CompileError::InvalidInput("live byte underflow".to_string())
                        })?;
                }
            }
            StorageInstr::ExtentWrite { source, .. } => {
                checkpoint_read_bytes = checkpoint_read_bytes
                    .checked_add(source.span_bytes)
                    .ok_or_else(|| CompileError::InvalidInput("read byte overflow".to_string()))?;
                device_write_bytes = device_write_bytes
                    .checked_add(source.span_bytes)
                    .ok_or_else(|| CompileError::InvalidInput("write byte overflow".to_string()))?;
            }
            StorageInstr::BulkExtentWrite { source, .. } => {
                checkpoint_read_bytes = checkpoint_read_bytes
                    .checked_add(source.span_bytes)
                    .ok_or_else(|| CompileError::InvalidInput("read byte overflow".to_string()))?;
                device_write_bytes = device_write_bytes
                    .checked_add(source.span_bytes)
                    .ok_or_else(|| CompileError::InvalidInput("write byte overflow".to_string()))?;
            }
            StorageInstr::SlabScatter {
                span_bytes,
                placements,
                ..
            } => {
                checkpoint_read_bytes = checkpoint_read_bytes
                    .checked_add(*span_bytes)
                    .ok_or_else(|| CompileError::InvalidInput("read byte overflow".to_string()))?;
                let mut payload_bytes = 0u64;
                for placement in placements {
                    payload_bytes = payload_bytes
                        .checked_add(placement.bytes)
                        .ok_or_else(|| CompileError::InvalidInput("write byte overflow".to_string()))?;
                }
                device_write_bytes = device_write_bytes
                    .checked_add(payload_bytes)
                    .ok_or_else(|| CompileError::InvalidInput("write byte overflow".to_string()))?;
                transform_scratch_peak_bytes = transform_scratch_peak_bytes.max(*span_bytes);
            }
            StorageInstr::TileMap {
                source,
                dest,
                outputs,
                transform,
                ..
            } => {
                if let Some(source) = source {
                    checkpoint_read_bytes = checkpoint_read_bytes
                        .checked_add(source.span_bytes)
                        .ok_or_else(|| {
                            CompileError::InvalidInput("read byte overflow".to_string())
                        })?;
                }
                let write_bytes = if let Some(dest) = dest {
                    extent_storage_bytes(&dest.stride)?
                } else {
                    let mut total = 0u64;
                    for output in outputs {
                        total = total
                            .checked_add(buffer_bytes(program, *output)?)
                            .ok_or_else(|| {
                                CompileError::InvalidInput("write byte overflow".to_string())
                            })?;
                    }
                    total
                };
                device_write_bytes = device_write_bytes
                    .checked_add(write_bytes)
                    .ok_or_else(|| CompileError::InvalidInput("write byte overflow".to_string()))?;
                transform_scratch_peak_bytes =
                    transform_scratch_peak_bytes.max(write_bytes.max(transform.scratch_bytes));
            }
            StorageInstr::CreateView { .. }
            | StorageInstr::Attach { .. }
            | StorageInstr::Finalize { .. } => {}
        }
    }

    program.memory.persistent_bytes = persistent_bytes;
    program.memory.temporary_peak_bytes = live_peak.saturating_sub(persistent_bytes);
    program.memory.transform_scratch_peak_bytes = transform_scratch_peak_bytes;
    program.memory.checkpoint_read_bytes = checkpoint_read_bytes;
    program.memory.device_write_bytes = device_write_bytes;
    Ok(())
}

/// Minimum alignment (bytes) for every persistent operand buffer in the
/// arena. 256 matches the cudaMalloc arena-base granularity and guarantees
/// the ≥16-byte alignment cuBLASLt needs to select its fast `align8` sm_80
/// bf16 tensor kernels (vs. slow `align1` sm_75 fallbacks).
const PERSISTENT_OPERAND_ALIGNMENT: u32 = 256;

fn assign_persistent_offsets(program: &mut StorageProgram) -> Result<(), CompileError> {
    let mut next = 0u64;
    let source_order = persistent_source_order(program)?;
    let mut order = (0..program.buffers.len()).collect::<Vec<_>>();
    order.sort_by_key(|&idx| {
        source_order
            .get(&program.buffers[idx].id)
            .copied()
            .unwrap_or((u32::MAX, u64::MAX, program.buffers[idx].id.0))
    });
    for idx in order {
        let buffer = &mut program.buffers[idx];
        if buffer.temporary || buffer.tensor.is_none() || buffer.bytes == 0 {
            buffer.persistent_offset = None;
            continue;
        }
        // Alignment is a property of the allocation *unit* — the persistent
        // buffer a runtime kernel receives as an operand (a standalone
        // weight, or a packed backing buffer that experts/shards are written
        // into and exposed from via single-backing `CreateView`s). Packed
        // members live as internal offsets *within* one backing buffer, so
        // aligning the buffer moves the whole unit together and never breaks
        // a packed view. We therefore align every persistent operand buffer
        // to `PERSISTENT_OPERAND_ALIGNMENT`: a <16-byte base pointer forces
        // cuBLASLt off its fast `align8` sm_80 tensor kernels onto slow
        // `align1` sm_75 ones (~6% dense-bf16 decode regression on
        // gemma-4-E4B). Honor any larger declared `buffer.alignment` too.
        // Cost: minor per-unit arena padding + reduced *cross-unit* bulk-copy
        // coalescing (intra-unit coalescing is preserved) — inference
        // throughput dominates one-time load. Previously hardcoded to 1
        // purely to maximize coalescing.
        let alignment = u64::from(buffer.alignment.max(PERSISTENT_OPERAND_ALIGNMENT));
        let offset = align_up_u64(next, alignment)?;
        next = offset
            .checked_add(buffer.bytes)
            .ok_or_else(|| CompileError::InvalidInput("persistent arena overflow".to_string()))?;
        buffer.persistent_offset = Some(offset);
    }
    Ok(())
}

fn persistent_source_order(
    program: &StorageProgram,
) -> Result<HashMap<BufferId, (u32, u64, u32)>, CompileError> {
    let mut order = HashMap::new();
    for instr in &program.instrs {
        let StorageInstr::ExtentWrite { source, dest, .. } = instr else {
            continue;
        };
        let Some(buffer) = program.buffers.iter().find(|buffer| buffer.id == dest.buffer) else {
            continue;
        };
        if buffer.temporary || buffer.tensor.is_none() || buffer.bytes == 0 {
            continue;
        }
        if !compact_extent_for_copy(&source.stride)
            || !compact_extent_for_copy(&dest.stride)
            || source.span_bytes != extent_storage_bytes(&dest.stride)?
        {
            continue;
        }
        let source_start = source
            .file_offset
            .checked_add(source.stride.base_offset)
            .ok_or_else(|| CompileError::InvalidInput("source offset overflow".to_string()))?;
        order
            .entry(dest.buffer)
            .or_insert((source.file_id.0, source_start, dest.buffer.0));
    }
    Ok(order)
}

fn align_up_u64(value: u64, alignment: u64) -> Result<u64, CompileError> {
    if alignment <= 1 {
        return Ok(value);
    }
    let rem = value % alignment;
    if rem == 0 {
        return Ok(value);
    }
    value
        .checked_add(alignment - rem)
        .ok_or_else(|| CompileError::InvalidInput("alignment overflow".to_string()))
}

/// Operand-unit invariants the optimizer/ABI must preserve and the C++ executor
/// relies on. Checked explicitly on the final program so a future rewrite fails
/// fast instead of silently regressing — these were previously only an implicit
/// assumption in `assign_persistent_offsets`:
///   1. every persistent operand buffer base is aligned to its contract
///      (`>= PERSISTENT_OPERAND_ALIGNMENT`, honoring a larger declared
///      `alignment`). A sub-16-byte base drops cuBLASLt off its fast `align8`
///      sm_80 kernels (the ~6% gemma-4-E4B dense regression).
///   2. persistent operand buffers occupy disjoint arena ranges.
///   3. every `CreateView` reads a single backing buffer that exists, and the
///      view window lies within it — i.e. packed members stay *internal* to one
///      backing buffer, which is what makes (1) safe for packed weights.
fn validate_persistent_layout(program: &StorageProgram) -> Result<(), CompileError> {
    let mut spans: Vec<(u64, u64, u32)> = Vec::new();
    for buffer in &program.buffers {
        let Some(offset) = buffer.persistent_offset else {
            continue;
        };
        let alignment = u64::from(buffer.alignment.max(PERSISTENT_OPERAND_ALIGNMENT));
        if offset % alignment != 0 {
            return Err(CompileError::InvalidInput(format!(
                "persistent buffer {} base offset {} violates operand alignment {}",
                buffer.id.0, offset, alignment
            )));
        }
        let end = offset.checked_add(buffer.bytes).ok_or_else(|| {
            CompileError::InvalidInput("persistent arena offset overflow".to_string())
        })?;
        spans.push((offset, end, buffer.id.0));
    }
    spans.sort_by_key(|span| span.0);
    for pair in spans.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(CompileError::InvalidInput(format!(
                "persistent buffers {} and {} overlap in the arena: [{}, {}) vs [{}, {})",
                pair[0].2, pair[1].2, pair[0].0, pair[0].1, pair[1].0, pair[1].1
            )));
        }
    }
    for instr in &program.instrs {
        let StorageInstr::CreateView { input, view, .. } = instr else {
            continue;
        };
        let Some(backing) = program.buffers.iter().find(|buffer| buffer.id == *input) else {
            return Err(CompileError::InvalidInput(format!(
                "CreateView references missing backing buffer {}",
                input.0
            )));
        };
        let extent = extent_storage_bytes(&view.stride)?;
        let end = view.offset.checked_add(extent).ok_or_else(|| {
            CompileError::InvalidInput("CreateView window overflow".to_string())
        })?;
        if end > backing.bytes {
            return Err(CompileError::InvalidInput(format!(
                "CreateView window [{}, {}) escapes backing buffer {} ({} bytes)",
                view.offset, end, backing.id.0, backing.bytes
            )));
        }
    }
    Ok(())
}

fn coalesce_persistent_arena_writes(program: &mut StorageProgram) -> Result<(), CompileError> {
    // The persistent-arena bulk-write form (`BulkExtentWrite`) is consumed
    // only by executors that own a flat device arena addressed by absolute
    // offset — i.e. the CUDA storage executor. The portable direct/cast
    // executor models each tensor as its own ggml buffer and has no arena
    // base to write into, so it never services an arena-offset write.
    // Coalescing for that target emits instructions its executor cannot run,
    // silently leaving the affected weight tensors uninitialized. Keep the
    // per-buffer `ExtentWrite`s intact for the portable backend.
    if program.target.backend == BackendKind::Portable {
        return Ok(());
    }
    if program.schedule.is_empty() {
        return Ok(());
    }
    let old_instrs = program.instrs.clone();
    let blocked_buffers = non_bulk_compatible_persistent_write_buffers(program)?;
    let mut merged: Vec<StorageInstr> = Vec::with_capacity(old_instrs.len());
    let mut rewrites = 0_u64;

    for instr_id in &program.schedule {
        let instr = instr_by_id(&old_instrs, *instr_id)?;

        if let Some(bulk) = extent_write_as_bulk(program, instr, &blocked_buffers)? {
            if let Some(previous) = merged.last_mut()
                && try_merge_bulk_extent_write(previous, &bulk)?
            {
                rewrites += 1;
                continue;
            }
            merged.push(bulk);
            continue;
        }
        merged.push(instr.clone());
    }

    rewrite_program_instrs(program, merged)?;
    if rewrites > 0 {
        program.optimizer.passes.push(OptimizerPassStats {
            name: "coalesce-persistent-arena-writes".to_string(),
            exprs_before: old_instrs.len(),
            exprs_after: program.instrs.len(),
            rewrites: usize::try_from(rewrites).unwrap_or(usize::MAX),
        });
    }
    Ok(())
}

fn non_bulk_compatible_persistent_write_buffers(
    program: &StorageProgram,
) -> Result<HashSet<BufferId>, CompileError> {
    let mut blocked = HashSet::new();
    for instr in &program.instrs {
        let StorageInstr::ExtentWrite { source, dest, .. } = instr else {
            continue;
        };
        let Some(buffer) = program.buffers.iter().find(|buffer| buffer.id == dest.buffer) else {
            continue;
        };
        if buffer.persistent_offset.is_none() {
            continue;
        }
        if !compact_extent_for_copy(&source.stride)
            || !compact_extent_for_copy(&dest.stride)
            || source.span_bytes != extent_storage_bytes(&dest.stride)?
        {
            blocked.insert(dest.buffer);
        }
    }
    Ok(blocked)
}

fn hoist_bulk_extent_writes(program: &mut StorageProgram) -> Result<(), CompileError> {
    if program.schedule.len() < 2 {
        return Ok(());
    }
    let old_instrs = program.instrs.clone();
    let mut pending_bulk: Vec<StorageInstr> = Vec::new();
    let mut allocations: Vec<StorageInstr> = Vec::new();
    let mut rest: Vec<StorageInstr> = Vec::with_capacity(old_instrs.len());
    let mut result: Vec<StorageInstr> = Vec::with_capacity(old_instrs.len());
    let mut rewrites = 0_u64;

    for instr_id in &program.schedule {
        let instr = instr_by_id(&old_instrs, *instr_id)?;
        if matches!(instr, StorageInstr::BulkExtentWrite { .. }) {
            pending_bulk.push(instr.clone());
        } else if matches!(instr, StorageInstr::Allocate { .. }) {
            allocations.push(instr.clone());
        } else {
            rest.push(instr.clone());
        }
    }
    result.append(&mut allocations);
    flush_pending_bulk(&mut result, &mut pending_bulk, &mut rewrites)?;
    result.append(&mut rest);

    rewrite_program_instrs(program, result)?;
    if rewrites > 0 {
        program.optimizer.passes.push(OptimizerPassStats {
            name: "hoist-bulk-arena-writes".to_string(),
            exprs_before: old_instrs.len(),
            exprs_after: program.instrs.len(),
            rewrites: usize::try_from(rewrites).unwrap_or(usize::MAX),
        });
    }
    Ok(())
}

fn flush_pending_bulk(
    result: &mut Vec<StorageInstr>,
    pending_bulk: &mut Vec<StorageInstr>,
    rewrites: &mut u64,
) -> Result<(), CompileError> {
    if pending_bulk.is_empty() {
        return Ok(());
    }
    pending_bulk.sort_by_key(|instr| match instr {
        StorageInstr::BulkExtentWrite {
            source,
            dest_offset,
            ..
        } => (
            source.file_id.0,
            source.file_offset + source.stride.base_offset,
            *dest_offset,
        ),
        StorageInstr::SlabScatter {
            file_id,
            file_offset,
            ..
        } => (file_id.0, *file_offset, 0),
        _ => (u32::MAX, u64::MAX, u64::MAX),
    });
    for instr in pending_bulk.drain(..) {
        if let Some(previous) = result.last_mut()
            && try_merge_bulk_extent_write(previous, &instr)?
        {
            *rewrites += 1;
            continue;
        }
        result.push(instr);
    }
    Ok(())
}

/// Coalescing thresholds for the slab-scatter pass, bundled so the knobs are
/// passed by name — a transposed pair of same-typed positional args would
/// silently change coalescing behavior.
#[derive(Clone, Copy)]
struct SlabConfig {
    max_slab_bytes: u64,
    max_gap_bytes: u64,
    max_placements: usize,
    min_placements: usize,
    min_payload_bytes: u64,
    max_overread_num: u64,
    max_overread_den: u64,
}

fn build_slab_scatter_writes(program: &mut StorageProgram) -> Result<(), CompileError> {
    // `SlabScatter` is the same persistent-arena class as `BulkExtentWrite`
    // (scatter a coalesced source read into packed arena slots) and is only
    // serviceable by the CUDA arena executor. The portable backend writes
    // stacked-expert members through per-buffer copy contracts instead, so
    // skip slab coalescing for it. See `coalesce_persistent_arena_writes`.
    if program.target.backend == BackendKind::Portable {
        return Ok(());
    }
    if program.schedule.len() < 2 {
        return Ok(());
    }
    const DEFAULT_MAX_SLAB_BYTES: u64 = 256 * 1024 * 1024;
    let cfg = SlabConfig {
        // Cap one coalesced slab read at 256 MiB, or the target tile budget if larger.
        max_slab_bytes: DEFAULT_MAX_SLAB_BYTES.max(program.target.max_tile_bytes),
        max_gap_bytes: 64 * 1024 * 1024, // tolerate up to 64 MiB holes between members
        max_placements: 4096,            // max members coalesced into one slab
        min_placements: 2,               // fewer than 2 members isn't worth a slab
        min_payload_bytes: 1024 * 1024,  // skip slabs with <1 MiB of useful payload
        max_overread_num: 5,             // reject if span:payload exceeds 5:4 (>25% wasted)
        max_overread_den: 4,
    };
    let old_instrs = program.instrs.clone();
    let mut result = Vec::with_capacity(old_instrs.len());
    let mut pending = Vec::new();
    let mut rewrites = 0u64;

    for instr_id in &program.schedule {
        let instr = instr_by_id(&old_instrs, *instr_id)?;
        if matches!(instr, StorageInstr::BulkExtentWrite { .. }) {
            pending.push(instr.clone());
        } else {
            flush_pending_slab_scatter(&mut result, &mut pending, &mut rewrites, cfg)?;
            result.push(instr.clone());
        }
    }
    flush_pending_slab_scatter(&mut result, &mut pending, &mut rewrites, cfg)?;

    rewrite_program_instrs(program, result)?;

    if crate::wl_debug_enabled() {
        let bulk_count = old_instrs
            .iter()
            .filter(|i| matches!(i, StorageInstr::BulkExtentWrite { .. }))
            .count();
        let slab_count = program
            .instrs
            .iter()
            .filter(|i| matches!(i, StorageInstr::SlabScatter { .. }))
            .count();
        let remaining_bulk = program
            .instrs
            .iter()
            .filter(|i| matches!(i, StorageInstr::BulkExtentWrite { .. }))
            .count();
        eprintln!(
            "[pie-weight-loader] slab-scatter pass: input_bulk={bulk_count} → output_slab={slab_count} remaining_bulk={remaining_bulk} rewrites={rewrites}"
        );

        if slab_count == 0 && bulk_count > 0 {
            let mut file_groups: std::collections::HashMap<u32, Vec<(u64, u64)>> =
                std::collections::HashMap::new();
            for instr in &old_instrs {
                if let StorageInstr::BulkExtentWrite { source, .. } = instr {
                    file_groups
                        .entry(source.file_id.0)
                        .or_default()
                        .push((
                            source.file_offset + source.stride.base_offset,
                            source.span_bytes,
                        ));
                }
            }
            for (fid, mut entries) in file_groups {
                entries.sort();
                let count = entries.len();
                let total_bytes: u64 = entries.iter().map(|(_, b)| b).sum();
                let mut max_gap = 0u64;
                for w in entries.windows(2) {
                    let end_prev = w[0].0 + w[0].1;
                    if w[1].0 > end_prev {
                        max_gap = max_gap.max(w[1].0 - end_prev);
                    }
                }
                let span = if let (Some(first), Some(last)) = (entries.first(), entries.last()) {
                    last.0 + last.1 - first.0
                } else {
                    0
                };
                eprintln!(
                    "[pie-weight-loader]   file={fid} entries={count} total={:.1}MiB span={:.1}MiB max_gap={:.1}MiB overread_ratio={:.2}",
                    total_bytes as f64 / (1024.0 * 1024.0),
                    span as f64 / (1024.0 * 1024.0),
                    max_gap as f64 / (1024.0 * 1024.0),
                    if total_bytes > 0 { span as f64 / total_bytes as f64 } else { 0.0 }
                );
            }
        }
    }

    if rewrites > 0 {
        program.optimizer.passes.push(OptimizerPassStats {
            name: "slab-scatter-arena-writes".to_string(),
            exprs_before: old_instrs.len(),
            exprs_after: program.instrs.len(),
            rewrites: usize::try_from(rewrites).unwrap_or(usize::MAX),
        });
    }
    Ok(())
}

fn flush_pending_slab_scatter(
    result: &mut Vec<StorageInstr>,
    pending: &mut Vec<StorageInstr>,
    rewrites: &mut u64,
    cfg: SlabConfig,
) -> Result<(), CompileError> {
    if pending.is_empty() {
        return Ok(());
    }
    pending.sort_by_key(|instr| match instr {
        StorageInstr::BulkExtentWrite { source, .. } => (
            source.file_id.0,
            source.file_offset + source.stride.base_offset,
        ),
        _ => (u32::MAX, u64::MAX),
    });

    let mut current = Vec::new();
    for instr in pending.drain(..) {
        if current.is_empty() {
            current.push(instr);
            continue;
        }
        if slab_can_accept(&current, &instr, cfg)? {
            current.push(instr);
        } else {
            emit_slab_or_bulk(result, &mut current, rewrites, cfg)?;
            current.push(instr);
        }
    }
    emit_slab_or_bulk(result, &mut current, rewrites, cfg)?;
    Ok(())
}

fn slab_can_accept(
    current: &[StorageInstr],
    next: &StorageInstr,
    cfg: SlabConfig,
) -> Result<bool, CompileError> {
    if current.len() >= cfg.max_placements {
        return Ok(false);
    }
    let Some((file_id, first_start, _, last_end)) = slab_bounds(current)? else {
        return Ok(false);
    };
    let StorageInstr::BulkExtentWrite { source, .. } = next else {
        return Ok(false);
    };
    if source.file_id != file_id || !is_byte_extent(&source.stride) {
        return Ok(false);
    }
    let start = source
        .file_offset
        .checked_add(source.stride.base_offset)
        .ok_or_else(|| CompileError::InvalidInput("source offset overflow".to_string()))?;
    let end = start
        .checked_add(source.span_bytes)
        .ok_or_else(|| CompileError::InvalidInput("source span overflow".to_string()))?;
    if start < last_end || start - last_end > cfg.max_gap_bytes {
        return Ok(false);
    }
    Ok(end - first_start <= cfg.max_slab_bytes)
}

fn slab_bounds(
    instrs: &[StorageInstr],
) -> Result<Option<(crate::types::FileId, u64, u64, u64)>, CompileError> {
    let Some(first) = instrs.first() else {
        return Ok(None);
    };
    let StorageInstr::BulkExtentWrite { source, .. } = first else {
        return Ok(None);
    };
    let file_id = source.file_id;
    let first_start = source
        .file_offset
        .checked_add(source.stride.base_offset)
        .ok_or_else(|| CompileError::InvalidInput("source offset overflow".to_string()))?;
    let mut payload = 0u64;
    let mut last_end = first_start;
    for instr in instrs {
        let StorageInstr::BulkExtentWrite { source, .. } = instr else {
            return Ok(None);
        };
        let start = source
            .file_offset
            .checked_add(source.stride.base_offset)
            .ok_or_else(|| CompileError::InvalidInput("source offset overflow".to_string()))?;
        let end = start
            .checked_add(source.span_bytes)
            .ok_or_else(|| CompileError::InvalidInput("source span overflow".to_string()))?;
        payload = payload
            .checked_add(source.span_bytes)
            .ok_or_else(|| CompileError::InvalidInput("slab payload overflow".to_string()))?;
        last_end = last_end.max(end);
    }
    Ok(Some((file_id, first_start, payload, last_end)))
}

fn emit_slab_or_bulk(
    result: &mut Vec<StorageInstr>,
    current: &mut Vec<StorageInstr>,
    rewrites: &mut u64,
    cfg: SlabConfig,
) -> Result<(), CompileError> {
    if current.is_empty() {
        return Ok(());
    }
    if current.len() < cfg.min_placements {
        result.append(current);
        return Ok(());
    }
    let Some((file_id, file_offset, payload, last_end)) = slab_bounds(current)? else {
        result.append(current);
        return Ok(());
    };
    let span_bytes = last_end - file_offset;
    if span_bytes <= payload || payload < cfg.min_payload_bytes {
        result.append(current);
        return Ok(());
    }
    if span_bytes
        .checked_mul(cfg.max_overread_den)
        .ok_or_else(|| CompileError::InvalidInput("slab overread overflow".to_string()))?
        > payload
            .checked_mul(cfg.max_overread_num)
            .ok_or_else(|| CompileError::InvalidInput("slab overread overflow".to_string()))?
    {
        result.append(current);
        return Ok(());
    }
    let mut placements = Vec::with_capacity(current.len());
    for instr in current.drain(..) {
        let StorageInstr::BulkExtentWrite {
            source,
            dest_offset,
            ..
        } = instr
        else {
            continue;
        };
        let source_start = source
            .file_offset
            .checked_add(source.stride.base_offset)
            .ok_or_else(|| CompileError::InvalidInput("source offset overflow".to_string()))?;
        placements.push(SlabPlacement {
            src_offset: source_start - file_offset,
            dest_offset,
            bytes: source.span_bytes,
        });
    }
    *rewrites = rewrites
        .checked_add(placements.len().saturating_sub(1) as u64)
        .ok_or_else(|| CompileError::InvalidInput("slab rewrite overflow".to_string()))?;
    result.push(StorageInstr::SlabScatter {
        id: InstrId(0),
        file_id,
        file_offset,
        span_bytes,
        placements,
    });
    Ok(())
}

fn extent_write_as_bulk(
    program: &StorageProgram,
    instr: &StorageInstr,
    blocked_buffers: &HashSet<BufferId>,
) -> Result<Option<StorageInstr>, CompileError> {
    let StorageInstr::ExtentWrite { id, source, dest } = instr else {
        return Ok(None);
    };
    if !compact_extent_for_copy(&source.stride)
        || !compact_extent_for_copy(&dest.stride)
        || source.span_bytes != extent_storage_bytes(&dest.stride)?
    {
        return Ok(None);
    }
    let Some(buffer) = program.buffers.iter().find(|buffer| buffer.id == dest.buffer) else {
        return Err(CompileError::InvalidInput(format!(
            "destination buffer {} is missing",
            dest.buffer.0
        )));
    };
    let Some(base) = buffer.persistent_offset else {
        return Ok(None);
    };
    if blocked_buffers.contains(&dest.buffer) {
        return Ok(None);
    }
    let dest_offset = base
        .checked_add(dest.offset)
        .and_then(|v| v.checked_add(dest.stride.base_offset))
        .ok_or_else(|| CompileError::InvalidInput("bulk destination offset overflow".to_string()))?;
    Ok(Some(StorageInstr::BulkExtentWrite {
        id: *id,
        source: SourceExtent {
            file_id: source.file_id,
            tensor_id: source.tensor_id,
            file_offset: source
                .file_offset
                .checked_add(source.stride.base_offset)
                .ok_or_else(|| CompileError::InvalidInput("source offset overflow".to_string()))?,
            span_bytes: source.span_bytes,
            stride: byte_extent(source.span_bytes),
        },
        dest_offset,
    }))
}

fn try_merge_bulk_extent_write(
    previous: &mut StorageInstr,
    current: &StorageInstr,
) -> Result<bool, CompileError> {
    let (
        StorageInstr::BulkExtentWrite {
            source: prev_source,
            dest_offset: prev_dest_offset,
            ..
        },
        StorageInstr::BulkExtentWrite {
            source: cur_source,
            dest_offset: cur_dest_offset,
            ..
        },
    ) = (previous, current)
    else {
        return Ok(false);
    };

    if prev_source.file_id != cur_source.file_id
        || !is_byte_extent(&prev_source.stride)
        || !is_byte_extent(&cur_source.stride)
    {
        return Ok(false);
    }
    let prev_source_start = prev_source.file_offset + prev_source.stride.base_offset;
    let cur_source_start = cur_source.file_offset + cur_source.stride.base_offset;
    if prev_source_start
        .checked_add(prev_source.span_bytes)
        .ok_or_else(|| CompileError::InvalidInput("source span overflow".to_string()))?
        != cur_source_start
        || prev_dest_offset
            .checked_add(prev_source.span_bytes)
            .ok_or_else(|| CompileError::InvalidInput("destination span overflow".to_string()))?
            != *cur_dest_offset
    {
        return Ok(false);
    }
    let span_bytes = prev_source
        .span_bytes
        .checked_add(cur_source.span_bytes)
        .ok_or_else(|| CompileError::InvalidInput("merged bulk extent overflow".to_string()))?;
    prev_source.file_offset = prev_source_start;
    prev_source.span_bytes = span_bytes;
    prev_source.stride = byte_extent(span_bytes);
    Ok(true)
}

fn compact_extent_for_copy(extent: &StridedExtent) -> bool {
    if extent.dims.iter().any(|dim| dim.count < 0) {
        return false;
    }
    let mut stride = i64::from(extent.element_bytes);
    for dim in extent.dims.iter().rev() {
        if dim.src_stride != stride || dim.dst_stride != stride {
            return false;
        }
        stride = match stride.checked_mul(dim.count) {
            Some(value) => value,
            None => return false,
        };
    }
    true
}

fn validate_target_support(program: &StorageProgram) -> Result<(), CompileError> {
    for instr in &program.instrs {
        let StorageInstr::TileMap {
            kind, transform, ..
        } = instr
        else {
            continue;
        };
        let supported = match program.target.backend {
            BackendKind::Unknown => true,
            BackendKind::Cuda => {
                matches!(
                    kind,
                    TileMapKind::Cast | TileMapKind::Reblock | TileMapKind::Reorder
                ) || (*kind == TileMapKind::Encode
                    && matches!(
                        transform.to,
                        Some(QuantScheme::Fp8E4M3
                             | QuantScheme::Int8Symmetric
                             | QuantScheme::Mxfp4E2M1E8M0)
                    ))
                    || (*kind == TileMapKind::Repack
                        && (matches!(transform.repack.layout, RepackLayout::DenseRowGather)
                            || (program.target.native_mxfp4_moe
                                && matches!(
                                    transform.repack.layout,
                                    RepackLayout::MarlinMxfp4Weight
                                        | RepackLayout::MarlinMxfp4Scale
                                ))))
            }
            BackendKind::Portable => match kind {
                TileMapKind::Cast | TileMapKind::Reblock | TileMapKind::Reorder => true,
                TileMapKind::Decode => transform.from == Some(QuantScheme::Fp8E4M3),
                TileMapKind::Encode | TileMapKind::Transcode | TileMapKind::Repack => false,
            },
        };
        if !supported {
            return Err(CompileError::InvalidInput(format!(
                "{:?} target does not support {:?} TileMap ({:?}->{:?})",
                program.target.backend, kind, transform.from, transform.to
            )));
        }
    }
    Ok(())
}

fn merge_adjacent_extent_writes(program: &mut StorageProgram) -> Result<(), CompileError> {
    if program.schedule.len() < 2 {
        return Ok(());
    }

    let old_instrs = program.instrs.clone();
    let mut merged: Vec<StorageInstr> = Vec::with_capacity(old_instrs.len());
    let mut rewrites = 0_u64;

    for instr_id in &program.schedule {
        let instr = instr_by_id(&old_instrs, *instr_id)?;

        if let Some(previous) = merged.last_mut()
            && try_merge_extent_write(previous, instr)?
        {
            rewrites += 1;
            continue;
        }
        merged.push(instr.clone());
    }

    rewrite_program_instrs(program, merged)?;
    if rewrites > 0 {
        program.optimizer.passes.push(OptimizerPassStats {
            name: "merge-adjacent-extent-writes".to_string(),
            exprs_before: old_instrs.len(),
            exprs_after: program.instrs.len(),
            rewrites: usize::try_from(rewrites).unwrap_or(usize::MAX),
        });
    }
    Ok(())
}

fn rewrite_program_instrs(
    program: &mut StorageProgram,
    merged: Vec<StorageInstr>,
) -> Result<(), CompileError> {
    program.instrs.clear();
    program.schedule.clear();
    program.instrs.reserve(merged.len());
    program.schedule.reserve(merged.len());
    for mut instr in merged {
        let id = InstrId(
            u32::try_from(program.instrs.len())
                .map_err(|_| CompileError::InvalidInput("too many instructions".to_string()))?,
        );
        set_instr_id(&mut instr, id);
        program.schedule.push(id);
        program.instrs.push(instr);
    }
    Ok(())
}

fn try_merge_extent_write(
    previous: &mut StorageInstr,
    current: &StorageInstr,
) -> Result<bool, CompileError> {
    let (
        StorageInstr::ExtentWrite {
            source: prev_source,
            dest: prev_dest,
            ..
        },
        StorageInstr::ExtentWrite {
            source: cur_source,
            dest: cur_dest,
            ..
        },
    ) = (previous, current)
    else {
        return Ok(false);
    };

    if prev_source.file_id != cur_source.file_id
        || prev_dest.buffer != cur_dest.buffer
        || !is_byte_extent(&prev_source.stride)
        || !is_byte_extent(&cur_source.stride)
        || !is_byte_extent(&prev_dest.stride)
        || !is_byte_extent(&cur_dest.stride)
    {
        return Ok(false);
    }

    let prev_source_start = prev_source
        .file_offset
        .checked_add(prev_source.stride.base_offset)
        .ok_or_else(|| CompileError::InvalidInput("source offset overflow".to_string()))?;
    let cur_source_start = cur_source
        .file_offset
        .checked_add(cur_source.stride.base_offset)
        .ok_or_else(|| CompileError::InvalidInput("source offset overflow".to_string()))?;
    let prev_dest_start = prev_dest
        .offset
        .checked_add(prev_dest.stride.base_offset)
        .ok_or_else(|| CompileError::InvalidInput("destination offset overflow".to_string()))?;
    let cur_dest_start = cur_dest
        .offset
        .checked_add(cur_dest.stride.base_offset)
        .ok_or_else(|| CompileError::InvalidInput("destination offset overflow".to_string()))?;

    if prev_source_start
        .checked_add(prev_source.span_bytes)
        .ok_or_else(|| CompileError::InvalidInput("source span overflow".to_string()))?
        != cur_source_start
        || prev_dest_start
            .checked_add(prev_source.span_bytes)
            .ok_or_else(|| CompileError::InvalidInput("destination span overflow".to_string()))?
            != cur_dest_start
    {
        return Ok(false);
    }

    let span_bytes = prev_source
        .span_bytes
        .checked_add(cur_source.span_bytes)
        .ok_or_else(|| CompileError::InvalidInput("merged extent overflow".to_string()))?;
    prev_source.file_offset = prev_source_start;
    prev_source.span_bytes = span_bytes;
    prev_source.stride = byte_extent(span_bytes);
    prev_dest.offset = prev_dest_start;
    prev_dest.stride = byte_extent(span_bytes);
    Ok(true)
}

fn is_byte_extent(extent: &StridedExtent) -> bool {
    extent.base_offset == 0
        && extent.element_bytes == 1
        && extent.dims.len() == 1
        && extent.dims[0].src_stride == 1
        && extent.dims[0].dst_stride == 1
        && extent.dims[0].count >= 0
}

fn instr_id_of(instr: &StorageInstr) -> InstrId {
    match instr {
        StorageInstr::Allocate { id, .. }
        | StorageInstr::ExtentWrite { id, .. }
        | StorageInstr::BulkExtentWrite { id, .. }
        | StorageInstr::SlabScatter { id, .. }
        | StorageInstr::TileMap { id, .. }
        | StorageInstr::CreateView { id, .. }
        | StorageInstr::Attach { id, .. }
        | StorageInstr::Release { id, .. }
        | StorageInstr::Finalize { id, .. } => *id,
    }
}

fn set_instr_id(instr: &mut StorageInstr, new_id: InstrId) {
    match instr {
        StorageInstr::Allocate { id, .. }
        | StorageInstr::ExtentWrite { id, .. }
        | StorageInstr::BulkExtentWrite { id, .. }
        | StorageInstr::SlabScatter { id, .. }
        | StorageInstr::TileMap { id, .. }
        | StorageInstr::CreateView { id, .. }
        | StorageInstr::Attach { id, .. }
        | StorageInstr::Release { id, .. }
        | StorageInstr::Finalize { id, .. } => *id = new_id,
    }
}

fn narrow_repack_source(
    mut source: SourceView,
    spec: RepackSpec,
) -> Result<(SourceView, RepackSpec), CompileError> {
    let mut narrowed = spec;
    let valid_rows = if narrowed.valid_rows == 0 {
        narrowed.target_rows
    } else {
        narrowed.valid_rows
    };
    narrowed.valid_rows = valid_rows;

    if source.shape.len() < 2 {
        return Err(CompileError::InvalidInput(
            "Repack source must have batch and row axes".to_string(),
        ));
    }
    if source.shape[0] != i64::from(narrowed.batch)
        || source.shape[1] != i64::from(narrowed.source_rows)
    {
        return Err(CompileError::InvalidInput(format!(
            "Repack source shape {:?} does not match batch/source_rows {:?}/{}",
            source.shape, narrowed.batch, narrowed.source_rows
        )));
    }

    let (row_start, row_count) = match narrowed.row_map {
        crate::types::RowMap::Identity => (narrowed.source_row_offset, valid_rows),
        crate::types::RowMap::Even | crate::types::RowMap::Odd => {
            let start = narrowed.source_row_offset.checked_mul(2).ok_or_else(|| {
                CompileError::InvalidInput("Repack row offset overflow".to_string())
            })?;
            let rows = valid_rows.checked_mul(2).ok_or_else(|| {
                CompileError::InvalidInput("Repack row count overflow".to_string())
            })?;
            (start, rows)
        }
    };
    if row_start != 0 || row_count != narrowed.source_rows {
        source = narrow_source_axis(source, Axis(1), i64::from(row_start), i64::from(row_count))?;
        narrowed.source_rows = row_count;
        narrowed.source_row_offset = 0;
    }

    match narrowed.layout {
        RepackLayout::MarlinMxfp4Weight => {
            if source.shape.len() != 4 || source.shape[3] != 16 {
                return Err(CompileError::InvalidInput(format!(
                    "MarlinMxfp4Weight Repack source must be [B, R, K/32, 16], got {:?}",
                    source.shape
                )));
            }
            if narrowed.source_col_offset % 32 != 0
                || narrowed.source_cols % 32 != 0
                || narrowed.source_stride_cols % 32 != 0
            {
                return Err(CompileError::InvalidInput(
                    "MarlinMxfp4Weight source narrowing requires 32-wide MXFP4 group alignment"
                        .to_string(),
                ));
            }
            let group_start = narrowed.source_col_offset / 32;
            let group_count = narrowed.source_cols / 32;
            if source.shape[2] != i64::from(narrowed.source_stride_cols / 32) {
                return Err(CompileError::InvalidInput(format!(
                    "MarlinMxfp4Weight source group axis {:?} does not match stride cols {}",
                    source.shape, narrowed.source_stride_cols
                )));
            }
            if group_start != 0 || narrowed.source_cols != narrowed.source_stride_cols {
                source = narrow_source_axis(
                    source,
                    Axis(2),
                    i64::from(group_start),
                    i64::from(group_count),
                )?;
                narrowed.source_stride_cols = narrowed.source_cols;
                narrowed.source_col_offset = 0;
            }
        }
        RepackLayout::MarlinMxfp4Scale => {
            if source.shape.len() != 3 {
                return Err(CompileError::InvalidInput(format!(
                    "MarlinMxfp4Scale Repack source must be [B, R, groups], got {:?}",
                    source.shape
                )));
            }
            if source.shape[2] != i64::from(narrowed.source_stride_cols) {
                return Err(CompileError::InvalidInput(format!(
                    "MarlinMxfp4Scale source group axis {:?} does not match stride cols {}",
                    source.shape, narrowed.source_stride_cols
                )));
            }
            if narrowed.source_col_offset != 0
                || narrowed.source_cols != narrowed.source_stride_cols
            {
                source = narrow_source_axis(
                    source,
                    Axis(2),
                    i64::from(narrowed.source_col_offset),
                    i64::from(narrowed.source_cols),
                )?;
                narrowed.source_stride_cols = narrowed.source_cols;
                narrowed.source_col_offset = 0;
            }
        }
        RepackLayout::DenseRowGather => {
            if source.shape.len() != 2 {
                return Err(CompileError::InvalidInput(format!(
                    "DenseRowGather Repack source must be [B, R], got {:?}",
                    source.shape
                )));
            }
        }
        RepackLayout::None => {}
    }

    Ok((source, narrowed))
}

fn buffer_bytes(program: &StorageProgram, id: BufferId) -> Result<u64, CompileError> {
    program
        .buffers
        .get(id.0 as usize)
        .filter(|buffer| buffer.id == id)
        .or_else(|| program.buffers.iter().find(|buffer| buffer.id == id))
        .map(|buffer| buffer.bytes)
        .ok_or_else(|| CompileError::InvalidInput(format!("buffer {} is missing", id.0)))
}

/// Resolve a scheduled instruction by id: index directly when ids are dense (the
/// common case), else fall back to a linear scan. Used by every pass that walks
/// `program.schedule` against an instruction slice.
fn instr_by_id(instrs: &[StorageInstr], id: InstrId) -> Result<&StorageInstr, CompileError> {
    instrs
        .get(id.0 as usize)
        .filter(|instr| instr_id_of(instr) == id)
        .or_else(|| instrs.iter().find(|instr| instr_id_of(instr) == id))
        .ok_or_else(|| CompileError::InvalidInput(format!("scheduled instr {} is missing", id.0)))
}

fn repack_stage_bytes(spec: RepackSpec) -> Result<u64, CompileError> {
    match spec.layout {
        RepackLayout::MarlinMxfp4Weight => {
            let elems = u64::from(spec.target_rows)
                .checked_mul(u64::from(spec.target_cols))
                .ok_or_else(|| {
                    CompileError::InvalidInput("MXFP4 repack stage size overflow".to_string())
                })?;
            Ok(elems.div_ceil(2))
        }
        RepackLayout::MarlinMxfp4Scale | RepackLayout::DenseRowGather | RepackLayout::None => Ok(0),
    }
}

fn extent_storage_bytes(extent: &StridedExtent) -> Result<u64, CompileError> {
    tensor_nbytes(
        &extent.dims.iter().map(|dim| dim.count).collect::<Vec<_>>(),
        u64::from(extent.element_bytes),
    )
    .ok_or_else(|| CompileError::InvalidInput("extent byte size overflow".to_string()))
}

fn strided_physical_source_bytes(extent: &StridedExtent) -> Result<u64, CompileError> {
    let mut max_offset = extent.base_offset;
    for dim in &extent.dims {
        if dim.count < 0 || dim.src_stride < 0 {
            return Err(CompileError::InvalidInput(
                "negative source extent dimension or stride".to_string(),
            ));
        }
        if dim.count == 0 {
            return Ok(0);
        }
        let count = u64::try_from(dim.count - 1)
            .map_err(|_| CompileError::InvalidInput("source extent count overflow".to_string()))?;
        let stride = u64::try_from(dim.src_stride)
            .map_err(|_| CompileError::InvalidInput("source extent stride overflow".to_string()))?;
        max_offset = max_offset
            .checked_add(count.checked_mul(stride).ok_or_else(|| {
                CompileError::InvalidInput("source extent byte overflow".to_string())
            })?)
            .ok_or_else(|| CompileError::InvalidInput("source extent byte overflow".to_string()))?;
    }
    max_offset
        .checked_add(u64::from(extent.element_bytes))
        .ok_or_else(|| CompileError::InvalidInput("source extent byte overflow".to_string()))
}

fn compact_extent(shape: &[i64], element_bytes: u64) -> StridedExtent {
    let mut stride = i64::try_from(element_bytes).unwrap_or(i64::MAX);
    let mut dims = Vec::with_capacity(shape.len());
    for dim in shape.iter().rev() {
        dims.push(DimSpec {
            count: *dim,
            src_stride: stride,
            dst_stride: stride,
        });
        stride = stride.saturating_mul(*dim);
    }
    dims.reverse();
    StridedExtent {
        base_offset: 0,
        element_bytes: u32::try_from(element_bytes).unwrap_or(u32::MAX),
        dims,
    }
}

fn byte_extent(bytes: u64) -> StridedExtent {
    StridedExtent {
        base_offset: 0,
        element_bytes: 1,
        dims: vec![DimSpec {
            count: i64::try_from(bytes).unwrap_or(i64::MAX),
            src_stride: 1,
            dst_stride: 1,
        }],
    }
}

fn full_dest_extent(buffer: BufferId, decl: &TensorDecl) -> Result<DestExtent, CompileError> {
    Ok(DestExtent {
        buffer,
        offset: 0,
        stride: storage_extent_for_shape(&decl.shape, &decl.encoding)?,
    })
}

fn storage_extent_for_shape(
    shape: &[i64],
    encoding: &Encoding,
) -> Result<StridedExtent, CompileError> {
    if let Some(element_bytes) = encoding_dense_element_bytes(encoding) {
        return Ok(compact_extent(shape, element_bytes));
    }
    Ok(byte_extent(encoding_nbytes(shape, encoding).ok_or_else(
        || CompileError::InvalidInput("packed extent byte size overflow".to_string()),
    )?))
}

fn selected_source_extent(
    source: &StridedExtent,
    shape: &[i64],
    encoding: &Encoding,
) -> Result<StridedExtent, CompileError> {
    let Some(element_bytes) = encoding_dense_element_bytes(encoding) else {
        return storage_extent_for_shape(shape, encoding);
    };
    if source.dims.len() != shape.len() {
        return Err(CompileError::InvalidInput(format!(
            "source stride rank {} does not match selected shape rank {}",
            source.dims.len(),
            shape.len()
        )));
    }
    let dest = compact_extent(shape, element_bytes);
    let dims = source
        .dims
        .iter()
        .zip(shape.iter())
        .zip(dest.dims.iter())
        .map(|((dim, count), dest_dim)| DimSpec {
            count: *count,
            src_stride: dim.src_stride,
            dst_stride: dest_dim.dst_stride,
        })
        .collect();
    Ok(StridedExtent {
        base_offset: source.base_offset,
        element_bytes: u32::try_from(element_bytes).unwrap_or(u32::MAX),
        dims,
    })
}

fn narrow_source_axis(
    mut source: SourceView,
    axis: Axis,
    start: i64,
    length: i64,
) -> Result<SourceView, CompileError> {
    let axis_index = axis.0 as usize;
    if axis_index >= source.shape.len() {
        return Err(CompileError::InvalidInput(format!(
            "source slice axis {} out of range for shape {:?}",
            axis.0, source.shape
        )));
    }
    if start < 0 || length < 0 || start + length > source.shape[axis_index] {
        return Err(CompileError::InvalidInput(format!(
            "source slice [{start}, {}) on axis {} exceeds shape {:?}",
            start + length,
            axis.0,
            source.shape
        )));
    }
    let old_stride = source.stride.clone();
    let can_preserve_strides = encoding_dense_element_bytes(&source.encoding).is_some()
        && old_stride.dims.len() == source.shape.len();
    let axis_stride_bytes = if can_preserve_strides {
        u64::try_from(old_stride.dims[axis_index].src_stride).map_err(|_| {
            CompileError::InvalidInput("negative source stride in slice lowering".to_string())
        })?
    } else {
        dense_axis_stride_bytes(&source.shape, axis, &source.encoding)?
    };
    source.offset_bytes = source
        .offset_bytes
        .checked_add(
            u64::try_from(start)
                .ok()
                .and_then(|start| start.checked_mul(axis_stride_bytes))
                .ok_or_else(|| {
                    CompileError::InvalidInput("source slice offset overflow".to_string())
                })?,
        )
        .ok_or_else(|| CompileError::InvalidInput("source slice offset overflow".to_string()))?;
    source.shape[axis_index] = length;
    source.stride = if can_preserve_strides {
        selected_source_extent(&old_stride, &source.shape, &source.encoding)?
    } else {
        storage_extent_for_shape(&source.shape, &source.encoding)?
    };
    Ok(source)
}

fn dense_axis_stride_bytes(
    shape: &[i64],
    axis: Axis,
    encoding: &Encoding,
) -> Result<u64, CompileError> {
    let axis = axis.0 as usize;
    if axis >= shape.len() {
        return Err(CompileError::InvalidInput(format!(
            "axis {} out of range for shape {:?}",
            axis, shape
        )));
    }
    let suffix_elements = shape[axis + 1..].iter().try_fold(1u64, |acc, dim| {
        let dim = u64::try_from(*dim).ok()?;
        acc.checked_mul(dim)
    });
    match encoding {
        Encoding::Raw(dtype) => suffix_elements
            .and_then(|elements| elements.checked_mul(dtype.bytes()))
            .ok_or_else(|| CompileError::InvalidInput("dense stride overflow".to_string())),
        Encoding::Quant(spec) => {
            let spec = spec.clone().normalized();
            let suffix = suffix_elements
                .ok_or_else(|| CompileError::InvalidInput("dense stride overflow".to_string()))?;
            let bits = suffix
                .checked_mul(u64::from(spec.bits_per_element))
                .ok_or_else(|| CompileError::InvalidInput("packed stride overflow".to_string()))?;
            if bits % 8 != 0 {
                return Err(CompileError::InvalidInput(format!(
                    "packed {:?} select on axis {} is not byte-aligned",
                    spec.scheme, axis
                )));
            }
            Ok(bits / 8)
        }
    }
}

fn dense_axis_offset_bytes(
    shape: &[i64],
    axis: Axis,
    index: i64,
    encoding: &Encoding,
) -> Result<u64, CompileError> {
    let index = u64::try_from(index)
        .map_err(|_| CompileError::InvalidInput("negative dense axis offset".to_string()))?;
    let stride = dense_axis_stride_bytes(shape, axis, encoding)?;
    index
        .checked_mul(stride)
        .ok_or_else(|| CompileError::InvalidInput("dense axis offset overflow".to_string()))
}

fn dtype_to_quant_marker(dtype: DType) -> QuantScheme {
    match dtype {
        DType::F8E4M3 => QuantScheme::Fp8E4M3,
        DType::F8E5M2 => QuantScheme::Fp8E5M2,
        DType::I8 | DType::U8 => QuantScheme::Int8Symmetric,
        _ => QuantScheme::None,
    }
}

#[cfg(test)]
mod persistent_layout_tests {
    use super::*;

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

    fn program_with(buffers: Vec<BufferDecl>) -> StorageProgram {
        let mut p = StorageProgram::empty(StorageTarget::default());
        p.buffers = buffers;
        p
    }

    #[test]
    fn accepts_aligned_disjoint_operands() {
        let p = program_with(vec![operand(0, 256, 1, Some(0)), operand(1, 256, 1, Some(256))]);
        assert!(validate_persistent_layout(&p).is_ok());
    }

    #[test]
    fn rejects_misaligned_operand_base() {
        // 128 is not a multiple of PERSISTENT_OPERAND_ALIGNMENT (256).
        let p = program_with(vec![operand(0, 64, 1, Some(128))]);
        assert!(validate_persistent_layout(&p).is_err());
    }

    #[test]
    fn rejects_overlapping_operands() {
        // [0,512) and [256,512) overlap; both bases are 256-aligned.
        let p = program_with(vec![operand(0, 512, 1, Some(0)), operand(1, 256, 1, Some(256))]);
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
                    dims: vec![DimSpec { count: 64, src_stride: 1, dst_stride: 1 }],
                },
            },
            layout: crate::types::Layout::dense(1),
        });
        // window [32, 96) escapes the 64-byte backing buffer.
        assert!(validate_persistent_layout(&p).is_err());
    }
}
