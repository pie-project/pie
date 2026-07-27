//! Running a finished plan on the CPU, against the real checkpoint bytes.
//!
//! Not a production path (§10.3). It exists so a plan can be executed without
//! a GPU and its output compared against `crate::reference`, which is the only
//! offline check that can fail because the plan moved the *wrong* bytes rather
//! than an ill-formed number of them.
//!
//! It is the one module below `lib.rs` that opens a file, which is why it is
//! named for what it is rather than sharing a name with the backend whose
//! plans it accepts: `crate::plan::passes::tile::host` decides how a plan is lowered, and
//! this executes the result. The compiler is on the other side of that line —
//! `tests/standalone.rs` pins it.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use half::{bf16, f16};

use crate::error::Error;
use crate::plan::{
    DestExtent, Extent, HOST_TILE_MAP_MASK, LoadPlan, SourceExtent, StorageInstr, TileMapKind,
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

/// Execute a plan against the checkpoint it names.
///
/// `snapshot_dir` is only a base for relative paths. The files themselves come
/// from `plan.files`, which is the rule step 6 established for every executor:
/// an executor that rediscovered the checkpoint by scanning a directory could
/// disagree with the plan about which file id means which file, and every
/// offset in the plan is expressed against that table.
pub fn execute_plan(plan: &LoadPlan, snapshot_dir: &Path) -> Result<HostStorage, Error> {
    if plan.target.tile_map_mask & !HOST_TILE_MAP_MASK != 0 {
        return Err(invalid(
            "host executor received a plan advertising unsupported TileMap transforms",
        ));
    }
    let files = plan
        .files
        .iter()
        .map(|file| {
            let path = PathBuf::from(&file.path);
            let path = if path.is_absolute() {
                path
            } else {
                snapshot_dir.join(path)
            };
            (file.id.0, path)
        })
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
    fn execute(&mut self) -> Result<(), Error> {
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

    fn allocate(&mut self, id: BufferId) -> Result<(), Error> {
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

    fn read_extent(&self, source: &SourceExtent) -> Result<Vec<u8>, Error> {
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
    ) -> Result<Vec<u8>, Error> {
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
        source_stride: &Extent,
    ) -> Result<(), Error> {
        require_same_byte_count(source_stride, &dest.stride)?;
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

    fn write_arena(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
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
    ) -> Result<(), Error> {
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
            require_same_byte_count(source_stride, &dest.stride)?;
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
    ) -> Result<Vec<u8>, Error> {
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

    fn buffer_dtype(&self, id: BufferId) -> Result<DType, Error> {
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

    fn source_dtype(&self, id: crate::types::TensorId) -> Result<DType, Error> {
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

    fn buffer_bytes(&self, id: BufferId) -> Result<&[u8], Error> {
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

    fn write_buffer(&mut self, id: BufferId, offset: usize, bytes: &[u8]) -> Result<(), Error> {
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
    ) -> Result<(Root, usize, usize), Error> {
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
) -> Result<(Root, usize, usize), Error> {
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

fn gather_strided(raw: &[u8], extent: &Extent) -> Result<Vec<u8>, Error> {
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

/// A copy is well-formed when the two sides span the same number of bytes.
///
/// Not the same *dimensions*: the two extents describe different things. The
/// source's dims say how to walk the checkpoint, which the compiler states as
/// coarsely as it can — a contiguous row shard becomes one dim of wide
/// elements. The destination's dims describe a compact block, which is checked
/// separately. Requiring the two decompositions to match rejected every sharded
/// plan, because `[2048] x 6144 bytes` and `[2048, 3072] x 2 bytes` are the same
/// 12 MiB written the same way. Neither the CUDA nor the Metal executor makes
/// that comparison; this one should not either.
fn require_same_byte_count(source: &Extent, dest: &Extent) -> Result<(), Error> {
    let bytes = |extent: &Extent| -> Option<i64> {
        extent
            .dims
            .iter()
            .try_fold(i64::from(extent.element_bytes), |acc, dim| {
                acc.checked_mul(dim.count)
            })
    };
    let (source_bytes, dest_bytes) = (bytes(source), bytes(dest));
    if source_bytes.is_none() || source_bytes != dest_bytes {
        return Err(invalid(format!(
            "source extent spans {source_bytes:?} bytes but the destination spans \
             {dest_bytes:?}"
        )));
    }
    Ok(())
}

fn compact_extent(extent: &Extent) -> bool {
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

fn extent_offset(index: &[usize], dims: &[crate::plan::Dim], source: bool) -> Result<usize, Error> {
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

fn extent_bytes(extent: &Extent) -> Result<usize, Error> {
    let elements = extent.dims.iter().try_fold(1usize, |n, dim| {
        n.checked_mul(checked_usize_i64(dim.count)?)
            .ok_or_else(|| invalid("extent byte count overflow"))
    })?;
    elements
        .checked_mul(extent.element_bytes as usize)
        .ok_or_else(|| invalid("extent byte count overflow"))
}

fn physical_source_bytes(extent: &Extent) -> Result<u64, Error> {
    physical_bytes(extent, true)
}

fn physical_bytes(extent: &Extent, source: bool) -> Result<u64, Error> {
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

fn decode_values(bytes: &[u8], dtype: DType) -> Result<Vec<f64>, Error> {
    let width = dtype.bytes() as usize;
    if !bytes.len().is_multiple_of(width) {
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

fn encode_values(values: &[f64], dtype: DType) -> Result<Vec<u8>, Error> {
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
        | StorageInstr::Release { id, .. }
        | StorageInstr::Finalize { id, .. } => *id,
    }
}

fn checked_usize(value: u64) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| invalid("value does not fit usize"))
}

fn checked_usize_i64(value: i64) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| invalid("negative or oversized extent value"))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{
        BufferDecl, DestExtent, Dim, MemoryPlan, SourceTensorDecl, StorageTarget, TileSpec,
        TransformSpec,
    };
    use crate::types::{FileId, InstrId, TensorDecl, TensorId};

    fn extent(base_offset: u64, element_bytes: u32, dims: &[(i64, i64, i64)]) -> Extent {
        Extent {
            base_offset,
            element_bytes,
            dims: dims
                .iter()
                .map(|&(count, src_stride, dst_stride)| Dim {
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
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), file).unwrap();

        let mut program = LoadPlan::empty(StorageTarget {
            max_tile_bytes: 2,
            preferred_alignment: 8,
            ..StorageTarget::default()
        });
        // The plan names its own files; the executor does not go looking.
        // Relative, so `snapshot_dir` still has a job.
        program.files.push(crate::plan::CheckpointFileDecl {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes,
            format: crate::types::CheckpointFormat::Safetensors,
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
                alignment: 8,
            },
            TensorDecl {
                id: TensorId(1),
                name: "cast".to_string(),
                shape: vec![2, 2],
                encoding: Encoding::Raw(DType::U16),
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
                tile: TileSpec {
                    max_tile_bytes: 2,
                    rows_per_tile: 0,
                },
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
        let storage = execute_plan(&program, &dir).unwrap();
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
        let error = execute_plan(&program, &dir).unwrap_err().to_string();
        assert!(error.contains("non-compact ExtentWrite destination"));
        std::fs::remove_dir_all(dir).ok();
    }

    /// The compiler states the source as coarsely as it can, so a contiguous
    /// read arrives as one dim of wide elements while the destination it lands
    /// in is described element by element. Requiring the two decompositions to
    /// agree rejected every sharded plan — `pie-loader replay` on a real
    /// checkpoint at tp 1/2 failed outright — even though both sides named the
    /// same bytes.
    #[test]
    fn accepts_a_source_factored_differently_from_its_destination() {
        let (dir, mut program) = fixture();
        let StorageInstr::ExtentWrite { source, dest, .. } = &mut program.instrs[2] else {
            panic!("fixture instruction changed");
        };
        // Same four bytes, described as one 4-byte element instead of four.
        source.stride = extent(0, 4, &[(1, 1, 1)]);
        assert_eq!(dest.stride.element_bytes, 1);
        assert_eq!(dest.stride.dims.iter().map(|d| d.count).sum::<i64>(), 4);
        execute_plan(&program, &dir).expect("equal byte counts should be accepted");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rejects_a_source_and_destination_of_different_byte_counts() {
        let (dir, mut program) = fixture();
        let StorageInstr::ExtentWrite { source, .. } = &mut program.instrs[2] else {
            panic!("fixture instruction changed");
        };
        source.stride = extent(0, 1, &[(3, 1, 1)]);
        let error = execute_plan(&program, &dir).unwrap_err().to_string();
        assert!(
            error.contains("bytes but the destination spans"),
            "unexpected error: {error}"
        );
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
        let storage = execute_plan(&plan, &dir).unwrap();
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
    fn rejects_unsupported_advertised_transforms() {
        let (dir, mut plan) = fixture();
        plan.target.tile_map_mask |= crate::plan::TILE_MAP_REORDER;
        let error = execute_plan(&plan, &dir).unwrap_err().to_string();
        assert!(error.contains("unsupported TileMap transforms"));
        std::fs::remove_dir_all(dir).ok();
    }
}
