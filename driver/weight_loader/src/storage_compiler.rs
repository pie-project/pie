use std::collections::{HashMap, HashSet};

use crate::abi::RuntimeAbi;
use crate::error::CompileError;
use crate::frontend::{plan_from_semantics, runtime_bytes};
use crate::ir::{LayoutExpr, LayoutPlan};
use crate::optimizer::optimize;
use crate::schema::build_semantic_graph;
use crate::source::{CheckpointMetadata, RawTensor};
use crate::storage::{
    BufferDecl, DestExtent, DimSpec, MetadataSpec, SourceExtent, StorageInstr, StorageProgram,
    StorageTarget, StridedExtent, TileMapKind, TileSpec, TransformSpec,
};
use crate::typecheck::typecheck;
use crate::types::{
    Axis, BufferId, DType, ExprId, InstrId, QuantScheme, TensorDecl, TensorId,
    encoding_storage_bytes, tensor_nbytes,
};

pub fn compile_storage_program(
    metadata: &CheckpointMetadata,
    cfg: &crate::config::ModelConfig,
    abi: &RuntimeAbi,
    target: StorageTarget,
) -> Result<StorageProgram, CompileError> {
    let graph = build_semantic_graph(metadata, cfg)?;
    let plan = plan_from_semantics(metadata, &graph, abi, &target)?;
    let plan = optimize(plan)?;
    lower_layout_plan(metadata, &plan, target)
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
    dtype: DType,
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
        for id in 0..self.plan.exprs.len() {
            let expr_id = ExprId(id as u32);
            let value = self.lower_expr(expr_id, &self.plan.exprs[id])?;
            self.values.insert(expr_id, value);
        }
        Ok(())
    }

    fn lower_expr(&mut self, id: ExprId, expr: &LayoutExpr) -> Result<ValueLoc, CompileError> {
        match expr {
            LayoutExpr::Source { tensor, decl } => {
                let raw = self.raw(*tensor)?;
                Ok(ValueLoc::Source(SourceView {
                    tensor_id: *tensor,
                    shape: decl.shape.clone(),
                    dtype: decl.dtype(),
                    offset_bytes: 0,
                    stride: compact_extent(&decl.shape, encoding_storage_bytes(&decl.encoding)),
                }))
                .and_then(|value| {
                    if raw.shape != decl.shape {
                        Err(CompileError::InvalidInput(format!(
                            "source expr {} shape {:?} does not match raw '{}' shape {:?}",
                            id.0, decl.shape, raw.name, raw.shape
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
                            Some(full_dest_extent(out, out_decl)),
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
                                input_decl.dtype(),
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
                                stride: compact_extent(&decl.shape, decl.dtype().bytes()),
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
            self.program.memory.checkpoint_read_bytes += span.span_bytes;
            self.program.memory.device_write_bytes += span.span_bytes;
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
                dense_axis_offset_bytes(&decl.shape, axis, axis_offset, decl.dtype())?;
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
                            stride: compact_extent(&input_decl.shape, decl.dtype().bytes()),
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
                dense_axis_offset_bytes(&decl.shape, axis, stack_index as i64, decl.dtype())?;
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
                            stride: compact_extent(&input_decl.shape, decl.dtype().bytes()),
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
                let row_bytes = dense_axis_stride_bytes(&shape, axis, source.dtype)?;
                source.offset_bytes += u64::try_from(start)
                    .ok()
                    .and_then(|start| start.checked_mul(row_bytes))
                    .ok_or_else(|| {
                        CompileError::InvalidInput("slice offset overflow".to_string())
                    })?;
                shape[axis_index] = length;
                source.shape = shape.clone();
                source.stride = compact_extent(&shape, u64::from(source.stride.element_bytes));
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
                    dense_axis_offset_bytes(&input_decl.shape, axis, start, input_decl.dtype())?;
                let out = self.declare_view_buffer(out_decl);
                let instr = self.next_instr();
                self.program.instrs.push(StorageInstr::CreateView {
                    id: instr,
                    input: input_buffer,
                    output: out,
                    view: DestExtent {
                        buffer: out,
                        offset,
                        stride: compact_extent(&out_decl.shape, out_decl.dtype().bytes()),
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
            Some(full_dest_extent(out, decl)),
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
        if let Some(source) = &source {
            self.program.memory.checkpoint_read_bytes += source.span_bytes;
        }
        for output in &outputs {
            if let Some(buffer) = self
                .program
                .buffers
                .iter()
                .find(|buffer| buffer.id == *output)
            {
                self.program.memory.device_write_bytes += buffer.bytes;
                self.program.memory.transform_scratch_peak_bytes = self
                    .program
                    .memory
                    .transform_scratch_peak_bytes
                    .max(buffer.bytes);
            }
        }
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
        let elem = u64::from(source_extent.stride.element_bytes);
        let bytes = tensor_nbytes(shape, elem).ok_or_else(|| {
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
                stride: compact_extent(shape, elem),
            },
        });
        self.program.schedule.push(instr);
        self.program.memory.checkpoint_read_bytes += bytes;
        self.program.memory.device_write_bytes += bytes;
        Ok(())
    }

    fn source_extent(&self, source: &SourceView) -> Result<SourceExtent, CompileError> {
        let raw = self.raw(source.tensor_id)?;
        let elem = u64::from(source.stride.element_bytes);
        let span_bytes = tensor_nbytes(&source.shape, elem).ok_or_else(|| {
            CompileError::InvalidInput("source extent byte size overflow".to_string())
        })?;
        if source.offset_bytes + span_bytes > raw.span_bytes {
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
        if temporary {
            self.program.memory.temporary_peak_bytes =
                self.program.memory.temporary_peak_bytes.max(bytes);
        } else {
            self.program.memory.persistent_bytes += bytes;
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
            self.program.memory.persistent_bytes += existing.bytes;
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

fn full_dest_extent(buffer: BufferId, decl: &TensorDecl) -> DestExtent {
    DestExtent {
        buffer,
        offset: 0,
        stride: compact_extent(&decl.shape, encoding_storage_bytes(&decl.encoding)),
    }
}

fn dense_axis_stride_bytes(shape: &[i64], axis: Axis, dtype: DType) -> Result<u64, CompileError> {
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
    suffix_elements
        .and_then(|elements| elements.checked_mul(dtype.bytes()))
        .ok_or_else(|| CompileError::InvalidInput("dense stride overflow".to_string()))
}

fn dense_axis_offset_bytes(
    shape: &[i64],
    axis: Axis,
    index: i64,
    dtype: DType,
) -> Result<u64, CompileError> {
    let index = u64::try_from(index)
        .map_err(|_| CompileError::InvalidInput("negative dense axis offset".to_string()))?;
    let stride = dense_axis_stride_bytes(shape, axis, dtype)?;
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
