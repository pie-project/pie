use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use half::{bf16, f16};

use crate::error::CompileError;
use crate::inproc::{deserialize_load_plan, parse_checkpoint_metadata};
use crate::load_plan::{
    DestExtent, HOST_TILE_MAP_MASK, LoadPlan, SourceExtent, StorageInstr, StridedExtent,
    TileMapKind,
};
use crate::types::{BufferId, DType, Encoding};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostTensor {
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostStorage {
    pub arena: Vec<u8>,
    pub tensors: HashMap<String, HostTensor>,
    pub max_tile_write_bytes: usize,
}

#[derive(Debug, Clone)]
enum BufferLoc {
    Arena {
        offset: usize,
        len: usize,
    },
    Owned(Vec<u8>),
    View {
        input: BufferId,
        offset: usize,
        len: usize,
    },
}

#[derive(Debug, Clone, Copy)]
enum Root {
    Arena,
    Owned(BufferId),
}

pub fn execute_serialized_plan(
    bytes: &[u8],
    snapshot_dir: &Path,
) -> Result<HostStorage, CompileError> {
    let plan = deserialize_load_plan(bytes)?;
    execute_plan(&plan, snapshot_dir)
}

pub fn execute_plan(plan: &LoadPlan, snapshot_dir: &Path) -> Result<HostStorage, CompileError> {
    if plan.target.tile_map_mask & !HOST_TILE_MAP_MASK != 0 {
        return Err(invalid(
            "host executor received a plan advertising unsupported TileMap transforms",
        ));
    }
    let metadata = parse_checkpoint_metadata(snapshot_dir)?;
    let files = metadata
        .files
        .into_iter()
        .map(|file| (file.id.0, PathBuf::from(file.path)))
        .collect::<HashMap<_, _>>();
    let arena_len = usize::try_from(plan.memory.persistent_bytes)
        .map_err(|_| invalid("persistent arena does not fit host address space"))?;
    let mut executor = HostExecutor {
        plan,
        files,
        arena: vec![0; arena_len],
        buffers: HashMap::new(),
        tensors: HashMap::new(),
        max_tile_write_bytes: 0,
    };
    executor.execute()?;
    Ok(HostStorage {
        arena: executor.arena,
        tensors: executor.tensors,
        max_tile_write_bytes: executor.max_tile_write_bytes,
    })
}

struct HostExecutor<'a> {
    plan: &'a LoadPlan,
    files: HashMap<u32, PathBuf>,
    arena: Vec<u8>,
    buffers: HashMap<BufferId, BufferLoc>,
    tensors: HashMap<String, HostTensor>,
    max_tile_write_bytes: usize,
}

impl HostExecutor<'_> {
    fn execute(&mut self) -> Result<(), CompileError> {
        for id in &self.plan.schedule {
            let instr = self
                .plan
                .instrs
                .iter()
                .find(|instr| instr_id(instr) == *id)
                .ok_or_else(|| invalid(format!("scheduled instruction {} is missing", id.0)))?
                .clone();
            match instr {
                StorageInstr::Allocate { buffer, .. } => self.allocate(buffer)?,
                StorageInstr::ExtentWrite { source, dest, .. } => {
                    let bytes = self.read_extent(&source)?;
                    self.write_extent(&dest, &bytes, &source.stride)?;
                }
                StorageInstr::BulkExtentWrite {
                    source,
                    dest_offset,
                    ..
                } => {
                    let bytes = self.read_extent(&source)?;
                    self.write_arena(dest_offset, &bytes)?;
                }
                StorageInstr::SlabScatter {
                    file_id,
                    file_offset,
                    span_bytes,
                    placements,
                    ..
                } => {
                    let slab = self.read_file(
                        file_id.0,
                        file_offset,
                        span_bytes,
                        self.plan.target.max_tile_bytes,
                    )?;
                    for placement in placements {
                        let start = usize::try_from(placement.src_offset)
                            .map_err(|_| invalid("slab source offset overflow"))?;
                        let len = usize::try_from(placement.bytes)
                            .map_err(|_| invalid("slab placement size overflow"))?;
                        let end = start
                            .checked_add(len)
                            .ok_or_else(|| invalid("slab source range overflow"))?;
                        let bytes = slab
                            .get(start..end)
                            .ok_or_else(|| invalid("slab source placement is out of bounds"))?;
                        self.write_arena(placement.dest_offset, bytes)?;
                    }
                }
                StorageInstr::TileMap {
                    kind,
                    source,
                    dest,
                    inputs,
                    outputs,
                    tile,
                    ..
                } => self.tile_map(
                    kind,
                    source.as_ref(),
                    dest.as_ref(),
                    &inputs,
                    &outputs,
                    tile.max_tile_bytes,
                )?,
                StorageInstr::CreateView {
                    input,
                    output,
                    view,
                    ..
                } => {
                    let len = extent_bytes(&view.stride)?;
                    let offset = checked_usize(view.offset)?
                        .checked_add(checked_usize(view.stride.base_offset)?)
                        .ok_or_else(|| invalid("view offset overflow"))?;
                    self.resolve(input, offset, len)?;
                    self.buffers
                        .insert(output, BufferLoc::View { input, offset, len });
                }
                StorageInstr::Attach { .. } => {}
                StorageInstr::Release { buffer, .. } => {
                    self.buffers.remove(&buffer);
                }
                StorageInstr::Finalize { tensor, name, .. } => {
                    let bytes = self.buffer_bytes(tensor)?.to_vec();
                    if self
                        .tensors
                        .insert(name.clone(), HostTensor { bytes })
                        .is_some()
                    {
                        return Err(invalid(format!("tensor '{name}' was finalized twice")));
                    }
                }
            }
        }
        Ok(())
    }

    fn allocate(&mut self, id: BufferId) -> Result<(), CompileError> {
        let decl = self
            .plan
            .buffers
            .iter()
            .find(|buffer| buffer.id == id)
            .ok_or_else(|| invalid(format!("buffer {} is missing", id.0)))?;
        let len = checked_usize(decl.bytes)?;
        let loc = if let Some(offset) = decl.persistent_offset {
            let offset = checked_usize(offset)?;
            let end = offset
                .checked_add(len)
                .ok_or_else(|| invalid("persistent buffer range overflow"))?;
            if end > self.arena.len() {
                return Err(invalid(format!("persistent buffer {} exceeds arena", id.0)));
            }
            BufferLoc::Arena { offset, len }
        } else {
            BufferLoc::Owned(vec![0; len])
        };
        if self.buffers.insert(id, loc).is_some() {
            return Err(invalid(format!("buffer {} was allocated twice", id.0)));
        }
        Ok(())
    }

    fn read_extent(&self, source: &SourceExtent) -> Result<Vec<u8>, CompileError> {
        let mut normalized = source.stride.clone();
        let base_offset = normalized.base_offset;
        normalized.base_offset = 0;
        let physical = physical_source_bytes(&normalized)?;
        let raw = self.read_file(
            source.file_id.0,
            source
                .file_offset
                .checked_add(base_offset)
                .ok_or_else(|| invalid("source base offset overflow"))?,
            physical,
            self.plan.target.max_tile_bytes,
        )?;
        gather_strided(&raw, &normalized)
    }

    fn read_file(
        &self,
        file_id: u32,
        offset: u64,
        len: u64,
        tile_bound: u64,
    ) -> Result<Vec<u8>, CompileError> {
        let path = self
            .files
            .get(&file_id)
            .ok_or_else(|| invalid(format!("plan references unknown file id {file_id}")))?;
        let len = checked_usize(len)?;
        let mut out = vec![0u8; len];
        let mut file =
            File::open(path).map_err(|err| invalid(format!("open {}: {err}", path.display())))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|err| invalid(format!("seek {}: {err}", path.display())))?;
        let tile = if tile_bound == 0 {
            len.max(1)
        } else {
            checked_usize(tile_bound)?.max(1)
        };
        for chunk in out.chunks_mut(tile) {
            file.read_exact(chunk)
                .map_err(|err| invalid(format!("read {}: {err}", path.display())))?;
        }
        Ok(out)
    }

    fn write_extent(
        &mut self,
        dest: &DestExtent,
        compact: &[u8],
        source_stride: &StridedExtent,
    ) -> Result<(), CompileError> {
        if source_stride
            .dims
            .iter()
            .map(|dim| dim.count)
            .collect::<Vec<_>>()
            != dest
                .stride
                .dims
                .iter()
                .map(|dim| dim.count)
                .collect::<Vec<_>>()
            || source_stride.element_bytes != dest.stride.element_bytes
        {
            return Err(invalid("source and destination extent shapes differ"));
        }
        if !compact_extent(&dest.stride) {
            return Err(invalid(
                "non-compact ExtentWrite destinations are unsupported",
            ));
        }
        let base = checked_usize(dest.offset)?
            .checked_add(checked_usize(dest.stride.base_offset)?)
            .ok_or_else(|| invalid("destination offset overflow"))?;
        self.write_buffer(dest.buffer, base, compact)
    }

    fn write_arena(&mut self, offset: u64, bytes: &[u8]) -> Result<(), CompileError> {
        let offset = checked_usize(offset)?;
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| invalid("arena write range overflow"))?;
        let dest = self
            .arena
            .get_mut(offset..end)
            .ok_or_else(|| invalid("arena write is out of bounds"))?;
        dest.copy_from_slice(bytes);
        Ok(())
    }

    fn tile_map(
        &mut self,
        kind: TileMapKind,
        source: Option<&SourceExtent>,
        dest: Option<&DestExtent>,
        inputs: &[BufferId],
        outputs: &[BufferId],
        max_tile_bytes: u64,
    ) -> Result<(), CompileError> {
        let input = if let Some(source) = source {
            self.read_extent(source)?
        } else {
            let input = inputs
                .first()
                .ok_or_else(|| invalid("TileMap has no source or input buffer"))?;
            self.buffer_bytes(*input)?.to_vec()
        };
        let output = match kind {
            TileMapKind::Reblock => input,
            TileMapKind::Cast => self.cast_bytes(
                source,
                inputs.first().copied(),
                outputs.first().copied(),
                &input,
            )?,
            other => {
                return Err(invalid(format!(
                    "host storage executor does not implement {other:?} transforms"
                )));
            }
        };
        let tile = if max_tile_bytes == 0 {
            output.len().max(1)
        } else {
            checked_usize(max_tile_bytes)?.max(1)
        };
        if let Some(dest) = dest {
            let source_stride = source.map(|source| &source.stride).unwrap_or(&dest.stride);
            if source_stride.dims.iter().map(|dim| dim.count).ne(dest
                .stride
                .dims
                .iter()
                .map(|dim| dim.count))
                || source_stride.element_bytes != dest.stride.element_bytes
            {
                return Err(invalid("source and destination extent shapes differ"));
            }
            if !compact_extent(&dest.stride) {
                return Err(invalid("non-compact TileMap destinations are unsupported"));
            }
            let base = checked_usize(dest.offset)?
                .checked_add(checked_usize(dest.stride.base_offset)?)
                .ok_or_else(|| invalid("destination offset overflow"))?;
            for (offset, chunk) in output.chunks(tile).enumerate() {
                self.max_tile_write_bytes = self.max_tile_write_bytes.max(chunk.len());
                self.write_buffer(dest.buffer, base + offset * tile, chunk)?;
            }
            return Ok(());
        }
        let output_id = outputs
            .first()
            .copied()
            .ok_or_else(|| invalid("TileMap has no output buffer"))?;
        if output.len() != self.buffer_bytes(output_id)?.len() {
            return Err(invalid(format!(
                "TileMap produced {} bytes for {}-byte output buffer",
                output.len(),
                self.buffer_bytes(output_id)?.len()
            )));
        }
        for (offset, chunk) in output.chunks(tile).enumerate() {
            self.max_tile_write_bytes = self.max_tile_write_bytes.max(chunk.len());
            self.write_buffer(output_id, offset * tile, chunk)?;
        }
        Ok(())
    }

    fn cast_bytes(
        &self,
        source: Option<&SourceExtent>,
        input: Option<BufferId>,
        output: Option<BufferId>,
        bytes: &[u8],
    ) -> Result<Vec<u8>, CompileError> {
        let output = output.ok_or_else(|| invalid("host Cast requires an output buffer"))?;
        let from = if let Some(input) = input {
            self.buffer_dtype(input)?
        } else if let Some(source) = source {
            self.source_dtype(source.tensor_id)?
        } else {
            return Err(invalid("host Cast requires a source or input buffer"));
        };
        let to = self.buffer_dtype(output)?;
        if from == to {
            return Ok(bytes.to_vec());
        }
        let values = decode_values(bytes, from)?;
        encode_values(&values, to)
    }

    fn buffer_dtype(&self, id: BufferId) -> Result<DType, CompileError> {
        let buffer = self
            .plan
            .buffers
            .iter()
            .find(|buffer| buffer.id == id)
            .ok_or_else(|| invalid(format!("buffer {} is missing", id.0)))?;
        let tensor_id = buffer
            .tensor
            .ok_or_else(|| invalid(format!("buffer {} has no tensor type", id.0)))?;
        let tensor = self
            .plan
            .tensors
            .iter()
            .find(|tensor| tensor.id == tensor_id)
            .ok_or_else(|| invalid(format!("tensor {} is missing", tensor_id.0)))?;
        match tensor.encoding {
            Encoding::Raw(dtype) => Ok(dtype),
            Encoding::Quant(_) => Err(invalid("host Cast does not accept quantized buffers")),
        }
    }

    fn source_dtype(&self, id: crate::types::TensorId) -> Result<DType, CompileError> {
        let source = self
            .plan
            .sources
            .iter()
            .find(|source| source.id == id)
            .ok_or_else(|| invalid(format!("source tensor {} is missing", id.0)))?;
        match source.encoding {
            Encoding::Raw(dtype) => Ok(dtype),
            Encoding::Quant(_) => Err(invalid("host Cast does not accept quantized sources")),
        }
    }

    fn buffer_bytes(&self, id: BufferId) -> Result<&[u8], CompileError> {
        let (root, offset, len) = self.resolve(id, 0, usize::MAX)?;
        match root {
            Root::Arena => self
                .arena
                .get(offset..offset + len)
                .ok_or_else(|| invalid("arena buffer range is out of bounds")),
            Root::Owned(root) => match self.buffers.get(&root) {
                Some(BufferLoc::Owned(bytes)) => bytes
                    .get(offset..offset + len)
                    .ok_or_else(|| invalid("owned buffer range is out of bounds")),
                _ => Err(invalid("resolved owned buffer is missing")),
            },
        }
    }

    fn write_buffer(
        &mut self,
        id: BufferId,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), CompileError> {
        let (root, base, _) = self.resolve(id, offset, bytes.len())?;
        let end = base
            .checked_add(bytes.len())
            .ok_or_else(|| invalid("buffer write range overflow"))?;
        match root {
            Root::Arena => self
                .arena
                .get_mut(base..end)
                .ok_or_else(|| invalid("arena buffer write is out of bounds"))?
                .copy_from_slice(bytes),
            Root::Owned(root) => match self.buffers.get_mut(&root) {
                Some(BufferLoc::Owned(dest)) => dest
                    .get_mut(base..end)
                    .ok_or_else(|| invalid("owned buffer write is out of bounds"))?
                    .copy_from_slice(bytes),
                _ => return Err(invalid("resolved owned buffer is missing")),
            },
        }
        Ok(())
    }

    fn resolve(
        &self,
        id: BufferId,
        extra_offset: usize,
        requested_len: usize,
    ) -> Result<(Root, usize, usize), CompileError> {
        let loc = self
            .buffers
            .get(&id)
            .ok_or_else(|| invalid(format!("buffer {} is not allocated", id.0)))?;
        match loc {
            BufferLoc::Arena { offset, len } => {
                resolve_range(Root::Arena, *offset, *len, extra_offset, requested_len)
            }
            BufferLoc::Owned(bytes) => {
                resolve_range(Root::Owned(id), 0, bytes.len(), extra_offset, requested_len)
            }
            BufferLoc::View { input, offset, len } => {
                let requested = if requested_len == usize::MAX {
                    len.saturating_sub(extra_offset)
                } else {
                    requested_len
                };
                if extra_offset > *len || requested > len.saturating_sub(extra_offset) {
                    return Err(invalid("view range is out of bounds"));
                }
                self.resolve(*input, offset + extra_offset, requested)
            }
        }
    }
}

fn resolve_range(
    root: Root,
    base: usize,
    available: usize,
    extra_offset: usize,
    requested_len: usize,
) -> Result<(Root, usize, usize), CompileError> {
    if extra_offset > available {
        return Err(invalid("buffer offset is out of bounds"));
    }
    let len = if requested_len == usize::MAX {
        available - extra_offset
    } else {
        requested_len
    };
    if len > available - extra_offset {
        return Err(invalid("buffer range is out of bounds"));
    }
    Ok((root, base + extra_offset, len))
}

fn gather_strided(raw: &[u8], extent: &StridedExtent) -> Result<Vec<u8>, CompileError> {
    let shape = extent
        .dims
        .iter()
        .map(|dim| checked_usize_i64(dim.count))
        .collect::<Result<Vec<_>, _>>()?;
    let elements = shape.iter().try_fold(1usize, |n, dim| {
        n.checked_mul(*dim)
            .ok_or_else(|| invalid("extent element count overflow"))
    })?;
    let elem = extent.element_bytes as usize;
    let mut out = vec![0u8; elements.saturating_mul(elem)];
    for linear in 0..elements {
        let index = unravel(linear, &shape);
        let src = extent_offset(&index, &extent.dims, true)?;
        let dst = linear * elem;
        out.get_mut(dst..dst + elem)
            .ok_or_else(|| invalid("compact extent range overflow"))?
            .copy_from_slice(
                raw.get(src..src + elem)
                    .ok_or_else(|| invalid("source extent is out of bounds"))?,
            );
    }
    Ok(out)
}

fn compact_extent(extent: &StridedExtent) -> bool {
    let mut stride = i64::from(extent.element_bytes);
    for dim in extent.dims.iter().rev() {
        if dim.count < 0 || dim.dst_stride != stride {
            return false;
        }
        let Some(next) = stride.checked_mul(dim.count) else {
            return false;
        };
        stride = next;
    }
    true
}

fn unravel(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        index[axis] = linear % shape[axis].max(1);
        linear /= shape[axis].max(1);
    }
    index
}

fn extent_offset(
    index: &[usize],
    dims: &[crate::load_plan::DimSpec],
    source: bool,
) -> Result<usize, CompileError> {
    index
        .iter()
        .zip(dims)
        .try_fold(0usize, |offset, (index, dim)| {
            let stride = if source {
                checked_usize_i64(dim.src_stride)?
            } else {
                checked_usize_i64(dim.dst_stride)?
            };
            offset
                .checked_add(
                    index
                        .checked_mul(stride)
                        .ok_or_else(|| invalid("extent index overflow"))?,
                )
                .ok_or_else(|| invalid("extent offset overflow"))
        })
}

fn extent_bytes(extent: &StridedExtent) -> Result<usize, CompileError> {
    let elements = extent.dims.iter().try_fold(1usize, |n, dim| {
        n.checked_mul(checked_usize_i64(dim.count)?)
            .ok_or_else(|| invalid("extent byte count overflow"))
    })?;
    elements
        .checked_mul(extent.element_bytes as usize)
        .ok_or_else(|| invalid("extent byte count overflow"))
}

fn physical_source_bytes(extent: &StridedExtent) -> Result<u64, CompileError> {
    physical_bytes(extent, true)
}

fn physical_bytes(extent: &StridedExtent, source: bool) -> Result<u64, CompileError> {
    let mut end = 0u64;
    for dim in &extent.dims {
        if dim.count == 0 {
            return Ok(0);
        }
        let count =
            u64::try_from(dim.count - 1).map_err(|_| invalid("negative extent dimension"))?;
        let stride = u64::try_from(if source {
            dim.src_stride
        } else {
            dim.dst_stride
        })
        .map_err(|_| invalid("negative extent stride"))?;
        end = end
            .checked_add(
                count
                    .checked_mul(stride)
                    .ok_or_else(|| invalid("extent range overflow"))?,
            )
            .ok_or_else(|| invalid("extent range overflow"))?;
    }
    end.checked_add(u64::from(extent.element_bytes))
        .ok_or_else(|| invalid("extent range overflow"))
}

fn decode_values(bytes: &[u8], dtype: DType) -> Result<Vec<f64>, CompileError> {
    let width = dtype.bytes() as usize;
    if bytes.len() % width != 0 {
        return Err(invalid("cast input byte count is not element-aligned"));
    }
    bytes
        .chunks_exact(width)
        .map(|chunk| {
            Ok(match dtype {
                DType::F32 => f32::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::F16 => {
                    f16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32() as f64
                }
                DType::BF16 => {
                    bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32() as f64
                }
                DType::I32 => i32::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::I16 => i16::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::I8 => i8::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::U32 => u32::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::U16 => u16::from_le_bytes(chunk.try_into().unwrap()) as f64,
                DType::U8 | DType::Bool => chunk[0] as f64,
                DType::F8E4M3 | DType::F8E5M2 => {
                    return Err(invalid("host Cast does not implement FP8"));
                }
            })
        })
        .collect()
}

fn encode_values(values: &[f64], dtype: DType) -> Result<Vec<u8>, CompileError> {
    let mut out = Vec::with_capacity(values.len() * dtype.bytes() as usize);
    for &value in values {
        match dtype {
            DType::F32 => out.extend_from_slice(&(value as f32).to_le_bytes()),
            DType::F16 => {
                out.extend_from_slice(&f16::from_f32(value as f32).to_bits().to_le_bytes())
            }
            DType::BF16 => {
                out.extend_from_slice(&bf16::from_f32(value as f32).to_bits().to_le_bytes())
            }
            DType::I32 => out.extend_from_slice(&(value as i32).to_le_bytes()),
            DType::I16 => out.extend_from_slice(&(value as i16).to_le_bytes()),
            DType::I8 => out.push(value as i8 as u8),
            DType::U32 => out.extend_from_slice(&(value as u32).to_le_bytes()),
            DType::U16 => out.extend_from_slice(&(value as u16).to_le_bytes()),
            DType::U8 => out.push(value as u8),
            DType::Bool => out.push(u8::from(value != 0.0)),
            DType::F8E4M3 | DType::F8E5M2 => {
                return Err(invalid("host Cast does not implement FP8"));
            }
        }
    }
    Ok(out)
}

fn instr_id(instr: &StorageInstr) -> crate::types::InstrId {
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

fn checked_usize(value: u64) -> Result<usize, CompileError> {
    usize::try_from(value).map_err(|_| invalid("value does not fit usize"))
}

fn checked_usize_i64(value: i64) -> Result<usize, CompileError> {
    usize::try_from(value).map_err(|_| invalid("negative or oversized extent value"))
}

fn invalid(message: impl Into<String>) -> CompileError {
    CompileError::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inproc::serialize_load_plan;
    use crate::load_plan::{
        BufferDecl, DestExtent, DimSpec, MemoryPlan, SourceTensorDecl, StorageTarget, TileSpec,
        TransformSpec,
    };
    use crate::types::{FileId, InstrId, Layout, TensorDecl, TensorId};

    fn extent(base_offset: u64, element_bytes: u32, dims: &[(i64, i64, i64)]) -> StridedExtent {
        StridedExtent {
            base_offset,
            element_bytes,
            dims: dims
                .iter()
                .map(|&(count, src_stride, dst_stride)| DimSpec {
                    count,
                    src_stride,
                    dst_stride,
                })
                .collect(),
        }
    }

    fn fixture() -> (PathBuf, LoadPlan) {
        let dir = std::env::temp_dir().join(format!(
            "pie_host_storage_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let header = r#"{"raw":{"dtype":"U8","shape":[6],"data_offsets":[0,6]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&[99, 1, 2, 99, 3, 4]);
        std::fs::write(dir.join("model.safetensors"), file).unwrap();

        let mut program = LoadPlan::empty(StorageTarget {
            max_tile_bytes: 2,
            preferred_alignment: 8,
            ..StorageTarget::default()
        });
        program.sources.push(SourceTensorDecl {
            id: TensorId(0),
            name: "raw".to_string(),
            file_id: FileId(0),
            file_offset: data_offset,
            span_bytes: 6,
            shape: vec![2, 3],
            encoding: Encoding::Raw(DType::U8),
        });
        program.tensors = vec![
            TensorDecl {
                id: TensorId(0),
                name: "selected".to_string(),
                shape: vec![2, 2],
                encoding: Encoding::Raw(DType::U8),
                layout: Layout::dense(8),
                sharding: crate::types::Sharding::replicated(),
                alignment: 8,
            },
            TensorDecl {
                id: TensorId(1),
                name: "cast".to_string(),
                shape: vec![2, 2],
                encoding: Encoding::Raw(DType::U16),
                layout: Layout::dense(8),
                sharding: crate::types::Sharding::replicated(),
                alignment: 8,
            },
        ];
        program.buffers = vec![
            BufferDecl {
                id: BufferId(0),
                tensor: Some(TensorId(0)),
                bytes: 4,
                alignment: 8,
                temporary: false,
                persistent_offset: Some(0),
            },
            BufferDecl {
                id: BufferId(1),
                tensor: Some(TensorId(1)),
                bytes: 8,
                alignment: 8,
                temporary: false,
                persistent_offset: Some(8),
            },
        ];
        program.instrs = vec![
            StorageInstr::Allocate {
                id: InstrId(0),
                buffer: BufferId(0),
            },
            StorageInstr::Allocate {
                id: InstrId(1),
                buffer: BufferId(1),
            },
            StorageInstr::ExtentWrite {
                id: InstrId(2),
                source: SourceExtent {
                    file_id: FileId(0),
                    tensor_id: TensorId(0),
                    file_offset: data_offset,
                    span_bytes: 4,
                    stride: extent(1, 1, &[(2, 3, 2), (2, 1, 1)]),
                },
                dest: DestExtent {
                    buffer: BufferId(0),
                    offset: 0,
                    stride: extent(0, 1, &[(2, 2, 2), (2, 1, 1)]),
                },
            },
            StorageInstr::TileMap {
                id: InstrId(3),
                kind: TileMapKind::Cast,
                source: None,
                dest: None,
                inputs: vec![BufferId(0)],
                outputs: vec![BufferId(1)],
                tile: TileSpec { max_tile_bytes: 2 },
                transform: TransformSpec::default(),
            },
            StorageInstr::Finalize {
                id: InstrId(4),
                tensor: BufferId(1),
                name: "cast".to_string(),
            },
        ];
        program.schedule = (0..5).map(InstrId).collect();
        program.memory = MemoryPlan {
            persistent_bytes: 16,
            checkpoint_read_bytes: 4,
            device_write_bytes: 12,
            ..MemoryPlan::default()
        };
        (dir, program)
    }

    #[test]
    fn executes_place_strided_cast_and_tiled_writes() {
        let (dir, program) = fixture();
        let bytes = serialize_load_plan(&program).unwrap();
        let storage = execute_serialized_plan(&bytes, &dir).unwrap();
        let values = storage.tensors["cast"]
            .bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1, 2, 3, 4]);
        assert_eq!(&storage.arena[..4], &[1, 2, 3, 4]);
        assert_eq!(storage.max_tile_write_bytes, 2);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rejects_noncompact_destination_extents() {
        let (dir, mut program) = fixture();
        let StorageInstr::ExtentWrite { dest, .. } = &mut program.instrs[2] else {
            panic!("fixture instruction changed");
        };
        dest.stride = extent(0, 1, &[(2, 2, 3), (2, 1, 1)]);
        let bytes = serialize_load_plan(&program).unwrap();
        let error = execute_serialized_plan(&bytes, &dir)
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-compact ExtentWrite destination"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn casts_direct_checkpoint_extents() {
        let (dir, mut plan) = fixture();
        let file_offset = plan.sources[0].file_offset;
        let StorageInstr::TileMap { source, inputs, .. } = &mut plan.instrs[3] else {
            panic!("fixture instruction changed");
        };
        *source = Some(SourceExtent {
            file_id: FileId(0),
            tensor_id: TensorId(0),
            file_offset,
            span_bytes: 4,
            stride: extent(1, 1, &[(2, 3, 2), (2, 1, 1)]),
        });
        inputs.clear();
        let bytes = serialize_load_plan(&plan).unwrap();
        let storage = execute_serialized_plan(&bytes, &dir).unwrap();
        let values = storage.tensors["cast"]
            .bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1, 2, 3, 4]);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn half_casts_round_and_overflow_to_infinity() {
        let f16_bytes = encode_values(&[100_000.0], DType::F16).unwrap();
        let f16_value = f16::from_bits(u16::from_le_bytes(f16_bytes.try_into().unwrap()));
        assert!(f16_value.is_infinite() && !f16_value.is_nan());

        let input = f32::from_bits(0x3f80_8001);
        let bf16_bytes = encode_values(&[f64::from(input)], DType::BF16).unwrap();
        let actual = u16::from_le_bytes(bf16_bytes.try_into().unwrap());
        assert_eq!(actual, bf16::from_f32(input).to_bits());
    }

    #[test]
    fn rejects_plan_and_compiler_version_mismatches() {
        let (dir, mut program) = fixture();
        program.version += 1;
        let bytes = serde_json::to_vec(&program).unwrap();
        assert!(execute_serialized_plan(&bytes, &dir).is_err());
        program.version = crate::load_plan::LOAD_PLAN_VERSION;
        program.compiler_version ^= 1;
        let bytes = serde_json::to_vec(&program).unwrap();
        assert!(execute_serialized_plan(&bytes, &dir).is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rejects_unsupported_advertised_transforms() {
        let (dir, mut plan) = fixture();
        plan.target.tile_map_mask |= crate::load_plan::TILE_MAP_REORDER;
        let error = execute_plan(&plan, &dir).unwrap_err().to_string();
        assert!(error.contains("unsupported TileMap transforms"));
        std::fs::remove_dir_all(dir).ok();
    }
}
