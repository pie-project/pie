//! Running a finished plan on the CPU, against the real checkpoint bytes.
//!
//! Not a production path (`architecture.md` §10.3). It exists so a plan can be
//! executed without a GPU and its output compared against
//! `crate::testkit::reference`, which is the only offline check that can fail
//! because the plan moved the *wrong* bytes rather than an ill-formed number of
//! them.
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
use crate::plan::index::{PlanIndex, instr_by_id};
use crate::plan::{
    DestExtent, Extent, HOST_TILE_MAP_MASK, LoadPlan, SourceExtent, StorageInstr, TileMapKind,
    TransformSpec,
};
use crate::types::{BufferId, DType, Encoding, QuantScheme};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostStorage {
    pub arena: Vec<u8>,
    pub tensors: HashMap<String, Vec<u8>>,
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
/// from `plan.files`, which is the rule for every executor: an executor that
/// rediscovered the checkpoint by scanning a directory could disagree with the
/// plan about which file id means which file, and every offset in the plan is
/// expressed against that table.
/// What a freshly allocated buffer holds before anything writes to it.
///
/// Not zero, and deliberately: `cudaMalloc` does not zero, so an executor that
/// handed out zeroed memory would silently satisfy any tensor with a region no
/// source covers -- which is exactly the region [`StorageInstr::Fill`] exists
/// to cover. With zeroed allocation a missing fill and a working one produce
/// identical bytes, so no test could tell them apart; this makes the
/// difference visible.
const POISON: u8 = 0xAB;

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
        index: PlanIndex::new(plan),
        files,
        arena: vec![POISON; arena_len],
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
    /// The sparse half of plan lookup. Buffers and instructions are dense, so
    /// they go through [`LoadPlan::buffer`] and [`instr_by_id`] directly; tensor
    /// ids interleave two allocators and need the map.
    index: PlanIndex,
    files: HashMap<u32, PathBuf>,
    arena: Vec<u8>,
    buffers: HashMap<BufferId, BufferLoc>,
    tensors: HashMap<String, Vec<u8>>,
    max_tile_write_bytes: usize,
}

impl HostExecutor<'_> {
    fn execute(&mut self) -> Result<(), Error> {
        for id in &self.plan.schedule {
            let instr = instr_by_id(&self.plan.instrs, *id)?.clone();
            match instr {
                StorageInstr::Allocate { buffer, .. } => self.allocate(buffer)?,
                StorageInstr::Fill { buffer, .. } => self.fill(buffer)?,
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
                StorageInstr::TileMap { .. } => self.tile_map(&instr)?,
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
                StorageInstr::Finalize { tensor, name, .. } => {
                    let bytes = self.buffer_bytes(tensor)?.to_vec();
                    if self.tensors.insert(name.clone(), bytes).is_some() {
                        return Err(invalid(format!("tensor '{name}' was finalized twice")));
                    }
                }
            }
        }
        Ok(())
    }

    fn allocate(&mut self, id: BufferId) -> Result<(), Error> {
        let decl = self.plan.buffer(id)?;
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
            BufferLoc::Owned(vec![POISON; len])
        };
        if self.buffers.insert(id, loc).is_some() {
            return Err(invalid(format!("buffer {} was allocated twice", id.0)));
        }
        Ok(())
    }

    /// Zero a buffer. An `Owned` buffer is already zero at allocation, but a
    /// persistent one is a window into an arena that may hold anything.
    fn fill(&mut self, id: BufferId) -> Result<(), Error> {
        let (root, offset, len) = self.resolve(id, 0, usize::MAX)?;
        match root {
            Root::Arena => self.arena[offset..offset + len].fill(0),
            Root::Owned(owner) => match self.buffers.get_mut(&owner) {
                Some(BufferLoc::Owned(bytes)) => bytes[offset..offset + len].fill(0),
                _ => return Err(invalid(format!("buffer {} is not writable", id.0))),
            },
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
        if !dest.stride.has_dense_destination() {
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

    fn tile_map(&mut self, instr: &StorageInstr) -> Result<(), Error> {
        let StorageInstr::TileMap {
            kind,
            source,
            dest,
            inputs,
            outputs,
            tile,
            transform,
            ..
        } = instr
        else {
            return Err(invalid("tile_map was handed something else"));
        };
        let (kind, max_tile_bytes) = (*kind, tile.max_tile_bytes);
        let (source, dest) = (source.as_ref(), dest.as_ref());
        let transform = transform.clone();
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
            TileMapKind::Scale => {
                self.scale_bytes(source, inputs, outputs.first().copied(), &input, &transform)?
            }
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
            // A per-group `Scale` is the one transform whose output is a
            // different width from its input by design: unpacking four-bit
            // codes into `BF16` quadruples the bytes. Every other kind moves
            // the same bytes it read, and the mismatch is a bug worth catching.
            if transform.scale_blocks.is_empty() {
                require_same_byte_count(source_stride, &dest.stride)?;
            }
            if !dest.stride.has_dense_destination() {
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

    /// `Scale` multiplies, and what it multiplies by is the only thing that
    /// varies: a constant every element shares, or one factor per group of
    /// elements read from a second operand.
    ///
    /// The uniform form keeps the type it was handed — `infer` gives it back
    /// unchanged — so there is no output dtype to look up. The per-group form
    /// is the one that also *decodes*, because a quantized tensor's elements
    /// are only numbers once their factors are applied; its output dtype is
    /// therefore the one the output buffer declares.
    ///
    /// The multiply happens in `f32`, not in the `f64` `decode_values` yields,
    /// because that is what the CUDA kernel does and the two executors are
    /// compared bit for bit. Every input dtype either form accepts is exactly
    /// representable in `f32`, so the narrowing before the multiply cannot
    /// lose anything the device would have kept.
    fn scale_bytes(
        &self,
        source: Option<&SourceExtent>,
        inputs: &[BufferId],
        output: Option<BufferId>,
        bytes: &[u8],
        transform: &TransformSpec,
    ) -> Result<Vec<u8>, Error> {
        let payload = || -> Result<DType, Error> {
            if let Some(source) = source {
                self.source_dtype(source.tensor_id)
            } else if let Some(input) = inputs.first() {
                self.buffer_dtype(*input)
            } else {
                Err(invalid("host Scale requires a source or input buffer"))
            }
        };
        if !transform.scale_blocks.is_empty() {
            let elements = match transform.from {
                None => decode_values(bytes, payload()?)?,
                Some(QuantScheme::Mxfp4E2M1E8M0) => decode_mxfp4_elements(bytes),
                Some(QuantScheme::Int4B8) => decode_int4b8_elements(bytes),
                Some(QuantScheme::Fp8E4M3) => decode_fp8_e4m3_elements(bytes),
                Some(other) => {
                    return Err(invalid(format!(
                        "host Scale does not implement {other:?} elements"
                    )));
                }
            };
            let output =
                output.ok_or_else(|| invalid("per-block Scale requires an output buffer"))?;
            return self.scale_per_block(elements, inputs, output, transform);
        }
        let dtype = payload()?;
        let factor = f32::from_bits(transform.scale_factor_bits);
        let mut values = decode_values(bytes, dtype)?;
        for value in &mut values {
            *value = f64::from(*value as f32 * factor);
        }
        encode_values(&values, dtype)
    }

    /// One factor per block, read from the last input buffer.
    ///
    /// The blocking is `transform.scale_blocks`, one entry per axis, so this
    /// walks the destination's logical shape rather than flattening. Flattening
    /// was exact while groups were confined to the last axis — there, the runs
    /// of the row-major layout and the runs of the logical shape are the same —
    /// but a block that spans rows has its factor at a stride, and chunking
    /// cannot see it.
    ///
    /// The factors' own extents are derived from the two, not read: the shape
    /// ratio is what defines the blocking, and reading a third statement of it
    /// would be a third thing to disagree.
    fn scale_per_block(
        &self,
        mut values: Vec<f64>,
        inputs: &[BufferId],
        output: BufferId,
        transform: &TransformSpec,
    ) -> Result<Vec<u8>, Error> {
        let factors = *inputs
            .last()
            .ok_or_else(|| invalid("per-block Scale has no factor operand"))?;
        let factors = decode_values(self.buffer_bytes(factors)?, self.buffer_dtype(factors)?)?;
        let shape = self
            .index
            .buffer_tensor(self.plan, output)
            .ok_or_else(|| invalid("per-block Scale output has no tensor type"))?
            .shape
            .clone();
        let blocks = &transform.scale_blocks;
        if shape.len() != blocks.len() {
            return Err(invalid(format!(
                "per-block Scale blocks {blocks:?} do not match output shape {shape:?}"
            )));
        }

        // Extents of the factor tensor, and the row-major strides to index it
        // by. Both are folded in one reverse pass so the two can never be
        // computed from different shapes.
        let mut counts = vec![0i64; shape.len()];
        let mut strides = vec![0i64; shape.len()];
        let mut running = 1i64;
        for axis in (0..shape.len()).rev() {
            let block = blocks[axis];
            if block <= 0 || shape[axis] % block != 0 {
                return Err(invalid(format!(
                    "per-block Scale block {block} does not divide axis {axis} of {shape:?}"
                )));
            }
            counts[axis] = shape[axis] / block;
            strides[axis] = running;
            running *= counts[axis];
        }
        let total: i64 = shape.iter().product();
        if values.len() as i64 != total {
            return Err(invalid(format!(
                "per-block Scale has {} elements but shape {shape:?} needs {total}",
                values.len()
            )));
        }
        if factors.len() as i64 != running {
            return Err(invalid(format!(
                "per-block Scale has {} factors but blocking {blocks:?} of {shape:?} \
                 needs {running}",
                factors.len()
            )));
        }

        // One odometer over the logical shape. `index` is the factor's flat
        // position, carried alongside so the division happens once per axis
        // step instead of once per element.
        let mut coord = vec![0i64; shape.len()];
        for value in &mut values {
            let mut index = 0i64;
            for axis in 0..shape.len() {
                index += (coord[axis] / blocks[axis]) * strides[axis];
            }
            let factor = factors[index as usize] as f32;
            *value = f64::from(*value as f32 * factor);
            for axis in (0..shape.len()).rev() {
                coord[axis] += 1;
                if coord[axis] < shape[axis] {
                    break;
                }
                coord[axis] = 0;
            }
        }
        encode_values(&values, self.buffer_dtype(output)?)
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
        let tensor = self
            .index
            .buffer_tensor(self.plan, id)
            .ok_or_else(|| invalid(format!("buffer {} has no tensor type", id.0)))?;
        match tensor.encoding {
            Encoding::Raw(dtype) => Ok(dtype),
            Encoding::Quant(_) => Err(invalid("host Cast does not accept quantized buffers")),
        }
    }

    fn source_dtype(&self, id: crate::types::TensorId) -> Result<DType, Error> {
        let source = self
            .index
            .source(self.plan, id)
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

/// One `f64` per E2M1 code, low nibble first.
///
/// The nibble order and the codepoint table are the OCP MX FP4 spec's, and
/// have to stay the CUDA kernel's: `kFp4Lut` in `kernels/dequant_fp4.cu` is
/// the same sixteen values in the same order, and the two executors are
/// compared element for element.
/// Unpack `QuantScheme::Int4B8` nibbles, low nibble first.
///
/// The nibbles are stored eight to a 32-bit word, but a little-endian word's
/// nibbles run low-to-high across its bytes in exactly that order, so reading
/// bytes is reading words. An element is `nibble - 8`.
fn decode_int4b8_elements(bytes: &[u8]) -> Vec<f64> {
    let mut values = Vec::with_capacity(bytes.len() * 2);
    for byte in bytes {
        values.push(f64::from((byte & 0xF) as i8 - 8));
        values.push(f64::from((byte >> 4) as i8 - 8));
    }
    values
}

/// One `f64` per `Fp8E4M3` byte: sign, four exponent bits, three mantissa
/// bits, bias 7.
///
/// This is the OCP `E4M3` the CUDA side reaches through
/// `__nv_cvt_fp8_to_halfraw(.., __NV_E4M3)`, which has no infinity: the
/// all-ones exponent carries ordinary values up to 448 and only `S.1111.111`
/// is NaN. A subnormal is `mantissa/8 * 2^-6`, which is what makes the two
/// branches differ by more than the implicit bit.
fn decode_fp8_e4m3_elements(bytes: &[u8]) -> Vec<f64> {
    bytes
        .iter()
        .map(|&byte| {
            let sign = if byte & 0x80 != 0 { -1.0f64 } else { 1.0 };
            let exponent = i32::from((byte >> 3) & 0x0F);
            let mantissa = f64::from(byte & 0x07);
            if exponent == 0x0F && mantissa == 7.0 {
                return f64::NAN;
            }
            let magnitude = if exponent == 0 {
                mantissa / 8.0 * (-6.0f64).exp2()
            } else {
                (1.0 + mantissa / 8.0) * f64::from(exponent - 7).exp2()
            };
            sign * magnitude
        })
        .collect()
}

fn decode_mxfp4_elements(bytes: &[u8]) -> Vec<f64> {
    const LUT: [f64; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let mut values = Vec::with_capacity(bytes.len() * 2);
    for byte in bytes {
        values.push(LUT[(byte & 0xF) as usize]);
        values.push(LUT[(byte >> 4) as usize]);
    }
    values
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
                DType::E8M0 => (chunk[0] as f64 - 127.0).exp2(),
                DType::F8E4M3 | DType::F8E5M2 => {
                    return Err(invalid("host Cast does not implement FP8"));
                }
                // A 64-bit integer does not survive the f64 pivot this cast
                // is written around, and nothing asks it to: `I64`/`U64`
                // tensors are index tables that move byte-for-byte.
                DType::I64 | DType::U64 => {
                    return Err(invalid("host Cast does not implement 64-bit integers"));
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
            DType::E8M0 => {
                return Err(invalid("host Cast does not encode to E8M0"));
            }
            DType::F8E4M3 | DType::F8E5M2 => {
                return Err(invalid("host Cast does not implement FP8"));
            }
            DType::I64 | DType::U64 => {
                return Err(invalid("host Cast does not implement 64-bit integers"));
            }
        }
    }
    Ok(out)
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

    /// A padded tensor, compiled and then actually run.
    ///
    /// Every other test here hands the executor a plan built by hand, which
    /// cannot catch a compiler that emits the fill in the wrong place. This one
    /// goes contract -> `compile` -> `execute_plan` and reads the bytes back,
    /// so the padded columns being zero is a fact about the whole path.
    #[test]
    fn a_padded_tensor_comes_back_with_zeros_where_no_source_reaches() {
        use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract, TensorType};

        let dir = std::env::temp_dir().join(format!("pie_host_pad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let header = r#"{"raw":{"dtype":"U8","shape":[2,3],"data_offsets":[0,6]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = CheckpointMetadata {
            files: vec![CheckpointFile {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "raw".to_string(),
                file_id: FileId(0),
                file_offset: data_offset,
                span_bytes: 6,
                shape: vec![2, 3],
                encoding: Encoding::Raw(DType::U8),
            }],
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "padded",
                Expr::concat(
                    1,
                    vec![
                        Expr::fill(0.0, TensorType::raw(vec![2, 1], DType::U8)),
                        Expr::src("raw"),
                        Expr::fill(0.0, TensorType::raw(vec![2, 1], DType::U8)),
                    ],
                ),
                vec![2, 5],
                Encoding::Raw(DType::U8),
            )],
            groups: Vec::new(),
        };

        let plan = crate::plan::compile(&metadata, &contract, StorageTarget::default()).unwrap();
        let storage = execute_plan(&plan, &dir).unwrap();
        assert_eq!(
            storage.tensors["padded"],
            vec![0, 1, 2, 3, 0, 0, 4, 5, 6, 0]
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// The one node that changes a value, compiled and then actually run.
    #[test]
    fn a_scaled_tensor_comes_back_multiplied() {
        use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};

        let dir = std::env::temp_dir().join(format!("pie_host_scale_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let values: [f32; 4] = [1.0, 2.0, 3.0, 5.0];
        let header = r#"{"raw":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        for value in values {
            file.extend_from_slice(&value.to_le_bytes());
        }
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = CheckpointMetadata {
            files: vec![CheckpointFile {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "raw".to_string(),
                file_id: FileId(0),
                file_offset: data_offset,
                span_bytes: 16,
                shape: vec![4],
                encoding: Encoding::Raw(DType::F32),
            }],
        };
        // Not a round number: a factor the executor merely copied instead of
        // multiplying by would still pass with 1.0, and one it truncated to
        // `f64` would still pass with 0.25.
        let factor = 1.0f32 / 3.0;
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "scaled",
                Expr::src("raw").scale(factor),
                vec![4],
                Encoding::Raw(DType::F32),
            )],
            groups: Vec::new(),
        };

        let plan = crate::plan::compile(&metadata, &contract, StorageTarget::default()).unwrap();
        assert!(
            plan.instrs.iter().any(|instr| matches!(
                instr,
                StorageInstr::TileMap {
                    kind: TileMapKind::Scale,
                    transform,
                    ..
                } if transform.scale_factor_bits == factor.to_bits()
            )),
            "the factor did not survive lowering: {:?}",
            plan.instrs
        );

        let storage = execute_plan(&plan, &dir).unwrap();
        let expected: Vec<u8> = values
            .iter()
            .flat_map(|value| (value * factor).to_le_bytes())
            .collect();
        assert_eq!(storage.tensors["scaled"], expected);
        std::fs::remove_dir_all(dir).ok();
    }

    /// `Int4B8` is a different element format under the same `Scale` node.
    ///
    /// Nothing about the node changes -- one operand, one factor tensor, the
    /// same grouping rule -- so what this pins is that the *scheme* is what
    /// says how a code becomes a number: `nibble - 8` here against a lookup
    /// table for MXFP4. Reading one as the other is silent, because both are
    /// four bits packed low nibble first.
    #[test]
    fn an_int4b8_source_is_dequantized_by_its_factors() {
        use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract, TensorType};
        use crate::types::{Axis, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_int4b8_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Two rows of 32 codes. Every byte is 0x9E, so the codes alternate
        // 14 (+6) and 9 (+1) -- an asymmetric pair straddling the bias, so
        // neither a swapped nibble order nor a missing `- 8` still passes.
        let packed = [0x9Eu8; 32];
        // bf16 2.0 and -0.5, one per row: different enough that a factor
        // applied to the wrong row cannot go unnoticed, and signed so that a
        // factor read as unsigned cannot either.
        let mut factors = Vec::new();
        for value in [2.0f32, -0.5] {
            factors.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }

        let header = r#"{"w":{"dtype":"U8","shape":[2,16],"data_offsets":[0,32]},"#.to_string()
            + r#""s":{"dtype":"BF16","shape":[2,1],"data_offsets":[32,36]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&packed);
        file.extend_from_slice(&factors);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = CheckpointMetadata {
            files: vec![CheckpointFile {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![
                RawTensor {
                    id: TensorId(0),
                    name: "w".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset,
                    span_bytes: 32,
                    shape: vec![2, 16],
                    encoding: Encoding::Raw(DType::U8),
                },
                RawTensor {
                    id: TensorId(1),
                    name: "s".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset + 32,
                    span_bytes: 4,
                    shape: vec![2, 1],
                    encoding: Encoding::Raw(DType::BF16),
                },
            ],
        };

        let int4b8 = QuantSpec {
            scheme: QuantScheme::Int4B8,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![
                TensorContract::new(
                    "scales",
                    Expr::src("s"),
                    vec![2, 1],
                    Encoding::Raw(DType::BF16),
                ),
                TensorContract::new(
                    "w",
                    Expr::src("w")
                        .transmute(TensorType {
                            shape: vec![2, 32],
                            encoding: Encoding::Quant(int4b8),
                        })
                        .scale_per_block(Expr::out("scales")),
                    vec![2, 32],
                    Encoding::Raw(DType::BF16),
                ),
            ],
            groups: Vec::new(),
        };

        let plan = crate::plan::compile(&metadata, &contract, StorageTarget::default()).unwrap();
        let storage = execute_plan(&plan, &dir).unwrap();

        let mut expected = Vec::new();
        for row_scale in [2.0f32, -0.5] {
            for _ in 0..16 {
                expected
                    .extend_from_slice(&bf16::from_f32(6.0 * row_scale).to_bits().to_le_bytes());
                expected
                    .extend_from_slice(&bf16::from_f32(1.0 * row_scale).to_bits().to_le_bytes());
            }
        }
        assert_eq!(storage.tensors["w"], expected);
        std::fs::remove_dir_all(dir).ok();
    }

    /// A block that spans rows *and* columns -- the shape GLM-5.1's FP8
    /// `kv_b_proj` ships and the one thing the old `group`/`axis` pair could
    /// not name.
    ///
    /// A single group size along a single axis can only ever describe a
    /// one-dimensional blocking. Deriving it from the ratio of the two shapes
    /// makes the two-dimensional case fall out with no new field, so this is
    /// the test that the generalization is real rather than a rename: the
    /// factors are `[2, 2]` over a `[4, 4]` payload, i.e. 2x2 tiles, and each
    /// of the four is distinct so a wrong index cannot land on the right
    /// number.
    ///
    /// FP8 elements on purpose. It is the format the 2-D blocking arrives in,
    /// and the values (1, 2, 3, ... 16) are exactly representable in E4M3, so
    /// a decode that is off by an exponent shows up as a factor of two rather
    /// than a rounding difference.
    #[test]
    fn a_two_dimensional_block_indexes_its_factors_by_row_and_column() {
        use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract, TensorType};
        use crate::types::{Axis, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_block2d_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // E4M3 for 1..=16: exponent = floor(log2(v)) + 7, mantissa = the three
        // bits below the leading one.
        let encode = |value: f64| -> u8 {
            let exponent = value.log2().floor() as i32;
            let mantissa = (value / f64::from(exponent).exp2() - 1.0) * 8.0;
            (((exponent + 7) as u8) << 3) | (mantissa.round() as u8)
        };
        let payload: Vec<u8> = (1..=16).map(|v| encode(f64::from(v))).collect();
        let factors: Vec<f32> = vec![1.0, 10.0, 100.0, 1000.0];
        let mut factor_bytes = Vec::new();
        for factor in &factors {
            factor_bytes.extend_from_slice(&factor.to_le_bytes());
        }

        let header = r#"{"w":{"dtype":"F8_E4M3","shape":[4,4],"data_offsets":[0,16]},"#.to_string()
            + r#""s":{"dtype":"F32","shape":[2,2],"data_offsets":[16,32]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&payload);
        file.extend_from_slice(&factor_bytes);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = CheckpointMetadata {
            files: vec![CheckpointFile {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![
                RawTensor {
                    id: TensorId(0),
                    name: "w".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset,
                    span_bytes: 16,
                    shape: vec![4, 4],
                    encoding: Encoding::Raw(DType::F8E4M3),
                },
                RawTensor {
                    id: TensorId(1),
                    name: "s".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset + 16,
                    span_bytes: 16,
                    shape: vec![2, 2],
                    encoding: Encoding::Raw(DType::F32),
                },
            ],
        };

        let fp8 = QuantSpec {
            scheme: QuantScheme::Fp8E4M3,
            logical_dtype: DType::BF16,
            bits_per_element: 8,
            group_size: 2,
            channel_axis: Some(Axis(1)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![
                TensorContract::new(
                    "scales",
                    Expr::src("s"),
                    vec![2, 2],
                    Encoding::Raw(DType::F32),
                )
                .internal(),
                TensorContract::new(
                    "w",
                    Expr::src("w")
                        .transmute(TensorType {
                            shape: vec![4, 4],
                            encoding: Encoding::Quant(fp8),
                        })
                        .scale_per_block(Expr::out("scales")),
                    vec![4, 4],
                    Encoding::Raw(DType::BF16),
                ),
            ],
            groups: Vec::new(),
        };

        let plan = crate::plan::compile(&metadata, &contract, StorageTarget::default()).unwrap();
        assert!(
            plan.instrs.iter().any(|instr| matches!(
                instr,
                StorageInstr::TileMap { transform, .. } if transform.scale_blocks == vec![2, 2]
            )),
            "the blocking is derived at plan time: {:?}",
            plan.instrs
        );
        let storage = execute_plan(&plan, &dir).unwrap();

        let mut expected = Vec::new();
        for row in 0..4i64 {
            for col in 0..4i64 {
                let value = f64::from((row * 4 + col + 1) as i32);
                let factor = factors[((row / 2) * 2 + col / 2) as usize];
                expected.extend_from_slice(
                    &bf16::from_f32(value as f32 * factor)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
        }
        assert_eq!(storage.tensors["w"], expected);
        std::fs::remove_dir_all(dir).ok();
    }

    /// Factors a contract needed but the driver does not: declared, used,
    /// never bound.
    ///
    /// The algebra has no `let`, so scaling by a tensor means publishing that
    /// tensor. Published as a runtime weight, a slab of dequantization factors
    /// lands in the persistent arena and stays there for the life of the
    /// process -- an arena view reclaims nothing when erased -- and the driver
    /// gets a name in its bind table that no kernel will ever ask for.
    ///
    /// `Visibility::Internal` is that name without either consequence. What
    /// this pins is that both go away together and the arithmetic does not
    /// change: the same bytes come out of `w`, `scales` is absent from the
    /// bind table, and the plan's persistent footprint drops by exactly the
    /// factors it no longer keeps.
    #[test]
    fn internal_factors_are_used_but_never_bound() {
        use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract, TensorType};
        use crate::types::{Axis, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_internal_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let packed = [0x9Eu8; 32];
        let mut factors = Vec::new();
        for value in [2.0f32, -0.5] {
            factors.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        let header = r#"{"w":{"dtype":"U8","shape":[2,16],"data_offsets":[0,32]},"#.to_string()
            + r#""s":{"dtype":"BF16","shape":[2,1],"data_offsets":[32,36]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&packed);
        file.extend_from_slice(&factors);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = CheckpointMetadata {
            files: vec![CheckpointFile {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![
                RawTensor {
                    id: TensorId(0),
                    name: "w".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset,
                    span_bytes: 32,
                    shape: vec![2, 16],
                    encoding: Encoding::Raw(DType::U8),
                },
                RawTensor {
                    id: TensorId(1),
                    name: "s".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset + 32,
                    span_bytes: 4,
                    shape: vec![2, 1],
                    encoding: Encoding::Raw(DType::BF16),
                },
            ],
        };

        let int4b8 = QuantSpec {
            scheme: QuantScheme::Int4B8,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        };
        let contract = |scales: TensorContract| ModelContract {
            alignment: 1,
            tensors: vec![
                scales,
                TensorContract::new(
                    "w",
                    Expr::src("w")
                        .transmute(TensorType {
                            shape: vec![2, 32],
                            encoding: Encoding::Quant(int4b8.clone()),
                        })
                        .scale_per_block(Expr::out("scales")),
                    vec![2, 32],
                    Encoding::Raw(DType::BF16),
                ),
            ],
            groups: Vec::new(),
        };
        let declare_scales = || {
            TensorContract::new(
                "scales",
                Expr::src("s"),
                vec![2, 1],
                Encoding::Raw(DType::BF16),
            )
        };

        let public = crate::plan::compile(
            &metadata,
            &contract(declare_scales()),
            StorageTarget::default(),
        )
        .unwrap();
        let internal = crate::plan::compile(
            &metadata,
            &contract(declare_scales().internal()),
            StorageTarget::default(),
        )
        .unwrap();

        let storage = execute_plan(&internal, &dir).unwrap();
        let mut expected = Vec::new();
        for row_scale in [2.0f32, -0.5] {
            for _ in 0..16 {
                expected
                    .extend_from_slice(&bf16::from_f32(6.0 * row_scale).to_bits().to_le_bytes());
                expected
                    .extend_from_slice(&bf16::from_f32(1.0 * row_scale).to_bits().to_le_bytes());
            }
        }
        assert_eq!(storage.tensors["w"], expected);
        assert!(
            !storage.tensors.contains_key("scales"),
            "an internal declaration must not reach the driver's bind table"
        );
        // 2 rows x 1 bf16 factor: the whole of what the public declaration was
        // keeping. Stated exactly, because "smaller" would also pass if the
        // arena had merely stopped aligning it.
        assert_eq!(
            public.memory.persistent_bytes - internal.memory.persistent_bytes,
            4
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// A block-scaled MXFP4 tensor and its factors, dequantized by the plan.
    ///
    /// This is the shape the DeepSeek families' expert weights arrive in, and
    /// the reason `Scale` takes a tensor factor at all: before it did, a driver
    /// loaded both halves, ran its own kernel, and left the packed originals
    /// resident because a view into the arena cannot be freed.
    #[test]
    fn a_block_scaled_source_is_dequantized_by_its_factors() {
        use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract, TensorType};
        use crate::types::{Axis, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_mxfp4_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Two rows of 32 codes. Every byte is 0x21, so the codes alternate
        // 1 (0.5) and 2 (1.0) — an asymmetric pair, so a nibble order that
        // swapped low for high would not still pass.
        let packed = [0x21u8; 32];
        // 2^(128-127) = 2 on row 0, 2^(126-127) = 0.5 on row 1: different
        // enough that a factor applied to the wrong row cannot go unnoticed.
        let exponents = [128u8, 126];

        let header = r#"{"w":{"dtype":"U8","shape":[2,16],"data_offsets":[0,32]},"#.to_string()
            + r#""s":{"dtype":"U8","shape":[2,1],"data_offsets":[32,34]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&packed);
        file.extend_from_slice(&exponents);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = CheckpointMetadata {
            files: vec![CheckpointFile {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![
                RawTensor {
                    id: TensorId(0),
                    name: "w".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset,
                    span_bytes: 32,
                    shape: vec![2, 16],
                    encoding: Encoding::Raw(DType::U8),
                },
                RawTensor {
                    id: TensorId(1),
                    name: "s".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset + 32,
                    span_bytes: 2,
                    shape: vec![2, 1],
                    encoding: Encoding::Raw(DType::U8),
                },
            ],
        };

        let mxfp4 = QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![
                TensorContract::new(
                    "scales",
                    Expr::src("s").transmute(TensorType {
                        shape: vec![2, 1],
                        encoding: Encoding::Raw(DType::E8M0),
                    }),
                    vec![2, 1],
                    Encoding::Raw(DType::E8M0),
                ),
                TensorContract::new(
                    "w",
                    Expr::src("w")
                        .transmute(TensorType {
                            shape: vec![2, 32],
                            encoding: Encoding::Quant(mxfp4),
                        })
                        .scale_per_block(Expr::out("scales")),
                    vec![2, 32],
                    Encoding::Raw(DType::BF16),
                ),
            ],
            groups: Vec::new(),
        };

        let plan = crate::plan::compile(&metadata, &contract, StorageTarget::default()).unwrap();
        let storage = execute_plan(&plan, &dir).unwrap();

        let mut expected = Vec::new();
        for row_scale in [2.0f32, 0.5] {
            for _ in 0..16 {
                expected
                    .extend_from_slice(&bf16::from_f32(0.5 * row_scale).to_bits().to_le_bytes());
                expected
                    .extend_from_slice(&bf16::from_f32(1.0 * row_scale).to_bits().to_le_bytes());
            }
        }
        assert_eq!(storage.tensors["w"], expected);
        std::fs::remove_dir_all(dir).ok();
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
                visibility: crate::types::Visibility::Public,
            },
            TensorDecl {
                id: TensorId(1),
                name: "cast".to_string(),
                shape: vec![2, 2],
                encoding: Encoding::Raw(DType::U16),
                alignment: 8,
                visibility: crate::types::Visibility::Public,
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
                    dtype: DType::U16,
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
            dtype: DType::U16,
        });
        inputs.clear();
        let storage = execute_plan(&plan, &dir).unwrap();
        let values = storage.tensors["cast"]
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
        plan.target.tile_map_mask |= crate::plan::TILE_MAP_TRANSCODE;
        let error = execute_plan(&plan, &dir).unwrap_err().to_string();
        assert!(error.contains("unsupported TileMap transforms"));
        std::fs::remove_dir_all(dir).ok();
    }
}
