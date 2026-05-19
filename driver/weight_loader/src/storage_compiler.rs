use std::collections::{HashMap, HashSet};

use crate::abi::RuntimeAbi;
use crate::error::CompileError;
use crate::frontend::{plan_from_semantics, runtime_bytes};
use crate::ir::{LayoutExpr, LayoutPlan};
use crate::optimizer::optimize_with_report;
use crate::schema::build_semantic_graph;
use crate::source::{CheckpointMetadata, RawTensor};
use crate::storage::{
    BufferDecl, DestExtent, DimSpec, MetadataSpec, SourceExtent, StorageInstr, StorageProgram,
    StorageTarget, StridedExtent, TileMapKind, TileSpec, TransformSpec,
};
use crate::typecheck::typecheck;
use crate::types::{
    Axis, BackendKind, BufferId, DType, Encoding, ExprId, InstrId, QuantScheme, TensorDecl,
    TensorId, encoding_dense_element_bytes, encoding_nbytes, tensor_nbytes,
};

pub fn compile_storage_program(
    metadata: &CheckpointMetadata,
    cfg: &crate::config::ModelConfig,
    abi: &RuntimeAbi,
    target: StorageTarget,
) -> Result<StorageProgram, CompileError> {
    let graph = build_semantic_graph(metadata, cfg)?;
    let plan = plan_from_semantics(metadata, &graph, abi, &target)?;
    let optimized = optimize_with_report(plan)?;
    let mut program = lower_layout_plan(metadata, &optimized.plan, target)?;
    program.optimizer = optimized.report;
    Ok(program)
}

pub fn lower_dense_copies(
    metadata: &CheckpointMetadata,
    abi: &RuntimeAbi,
    target: StorageTarget,
) -> Result<StorageProgram, CompileError> {
    let cfg = crate::config::ModelConfig {
        model_type: String::new(),
        quant_method: String::new(),
        num_hidden_layers: 0,
        num_experts: 0,
        num_experts_per_tok: 0,
    };
    compile_storage_program(metadata, &cfg, abi, target)
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
        recompute_memory_plan(&mut self.program)?;
        validate_target_support(&self.program)?;
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
                length: _,
                decl,
            } => {
                let input_value = self.value(*input)?;
                match input_value {
                    ValueLoc::Buffer(input_buffer) => {
                        let input_decl = self.plan.decl(*input).ok_or_else(|| {
                            CompileError::InvalidInput(format!(
                                "View input {} has no decl",
                                input.0
                            ))
                        })?;
                        let offset = if let Some(axis) = axis {
                            dense_axis_offset_bytes(
                                &input_decl.shape,
                                *axis,
                                *start,
                                &input_decl.encoding,
                            )?
                        } else {
                            0
                        };
                        let out = self.declare_view_buffer(decl);
                        let instr = self.next_instr();
                        self.program.instrs.push(StorageInstr::CreateView {
                            id: instr,
                            input: input_buffer,
                            output: out,
                            view: DestExtent {
                                buffer: out,
                                offset,
                                stride: storage_extent_for_shape(&decl.shape, &decl.encoding)?,
                            },
                            layout: layout.clone(),
                        });
                        self.program.schedule.push(instr);
                        Ok(ValueLoc::Buffer(out))
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
                },
            ),
            LayoutExpr::Encode { scheme, input, .. } => self.lower_tiled_unary(
                id,
                *input,
                TileMapKind::Encode,
                TransformSpec {
                    from: None,
                    to: Some(*scheme),
                },
            ),
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
                },
            ),
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
            ValueLoc::Source(mut source) => {
                let axis_index = axis.0 as usize;
                let mut shape = source.shape.clone();
                if axis_index >= shape.len() {
                    return Err(CompileError::InvalidInput(format!(
                        "Select axis {} out of range for source shape {:?}",
                        axis.0, shape
                    )));
                }
                let old_stride = source.stride.clone();
                let can_preserve_strides = encoding_dense_element_bytes(&source.encoding).is_some()
                    && old_stride.dims.len() == shape.len();
                let row_bytes = if can_preserve_strides {
                    u64::try_from(old_stride.dims[axis_index].src_stride).map_err(|_| {
                        CompileError::InvalidInput(
                            "negative source stride in Select lowering".to_string(),
                        )
                    })?
                } else {
                    dense_axis_stride_bytes(&shape, axis, &source.encoding)?
                };
                source.offset_bytes += u64::try_from(start)
                    .ok()
                    .and_then(|start| start.checked_mul(row_bytes))
                    .ok_or_else(|| {
                        CompileError::InvalidInput("slice offset overflow".to_string())
                    })?;
                shape[axis_index] = length;
                source.shape = shape.clone();
                source.stride = if can_preserve_strides {
                    selected_source_extent(&old_stride, &shape, &source.encoding)?
                } else {
                    storage_extent_for_shape(&shape, &source.encoding)?
                };
                Ok(ValueLoc::Source(source))
            }
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
        if existing.tensor.is_none() {
            existing.tensor = Some(decl.id);
        }
        if existing.temporary {
            existing.temporary = false;
        }
        if !self
            .program
            .tensors
            .iter()
            .any(|tensor| tensor.id == decl.id)
        {
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
        if !buffer.temporary && buffer.tensor.is_some() {
            persistent_bytes = persistent_bytes.checked_add(buffer.bytes).ok_or_else(|| {
                CompileError::InvalidInput("persistent byte overflow".to_string())
            })?;
        }
    }

    for instr_id in &program.schedule {
        let instr = program
            .instrs
            .iter()
            .find(|instr| instr_id_of(instr) == *instr_id)
            .ok_or_else(|| {
                CompileError::InvalidInput(format!("scheduled instr {} is missing", instr_id.0))
            })?;
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
            StorageInstr::TileMap {
                source,
                dest,
                outputs,
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
                transform_scratch_peak_bytes = transform_scratch_peak_bytes.max(write_bytes);
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
            BackendKind::Cuda => matches!(
                kind,
                TileMapKind::Cast | TileMapKind::Reblock | TileMapKind::Reorder
            ),
            BackendKind::Portable => match kind {
                TileMapKind::Cast | TileMapKind::Reblock | TileMapKind::Reorder => true,
                TileMapKind::Decode => transform.from == Some(QuantScheme::Fp8E4M3),
                TileMapKind::Encode | TileMapKind::Transcode => false,
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

fn instr_id_of(instr: &StorageInstr) -> InstrId {
    match instr {
        StorageInstr::Allocate { id, .. }
        | StorageInstr::ExtentWrite { id, .. }
        | StorageInstr::TileMap { id, .. }
        | StorageInstr::CreateView { id, .. }
        | StorageInstr::Attach { id, .. }
        | StorageInstr::Release { id, .. }
        | StorageInstr::Finalize { id, .. } => *id,
    }
}

fn buffer_bytes(program: &StorageProgram, id: BufferId) -> Result<u64, CompileError> {
    program
        .buffers
        .iter()
        .find(|buffer| buffer.id == id)
        .map(|buffer| buffer.bytes)
        .ok_or_else(|| CompileError::InvalidInput(format!("buffer {} is missing", id.0)))
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
