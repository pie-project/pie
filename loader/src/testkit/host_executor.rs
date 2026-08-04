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
    CONVERT_TILE_MAP_MASK, DestExtent, Extent, LoadPlan, SourceExtent, StorageInstr, TileMapKind,
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

/// One retired instruction of an executing plan, for a caller rendering
/// progress.
///
/// The smooth axis is bytes, not instructions: one tensor can be half the
/// model, so an instruction count jerks where the checkpoint bytes consumed
/// so far advance evenly. The total is the plan's own statement
/// ([`crate::plan::MemoryPlan::checkpoint_read_bytes`]), known before the
/// first instruction runs — which is what makes a percentage possible at all.
pub struct Progress<'a> {
    /// Checkpoint bytes consumed so far.
    pub read_bytes: u64,
    /// What `read_bytes` counts toward.
    pub total_read_bytes: u64,
    /// The runtime tensor this instruction published, when it published one.
    pub finalized: Option<&'a str>,
}

pub fn execute_plan(plan: &LoadPlan, snapshot_dir: &Path) -> Result<HostStorage, Error> {
    execute_plan_with_progress(plan, snapshot_dir, &mut |_| {})
}

/// [`execute_plan`], reporting a [`Progress`] after every retired instruction.
///
/// The callback is for rendering, so it can do no harm: it sees each state
/// once, after the work is done, and returns nothing.
pub fn execute_plan_with_progress(
    plan: &LoadPlan,
    snapshot_dir: &Path,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<HostStorage, Error> {
    // The gate is what this executor *implements* — the convert mask — not
    // `HOST_TILE_MAP_MASK`, which is what a host-lowered plan may *advertise*.
    // A CUDA plan that carries an `Encode` is executable here too; what stays
    // refused is anything carrying a transform with no host implementation,
    // `Repack` being the standing example.
    if plan.target.tile_map_mask & !CONVERT_TILE_MAP_MASK != 0 {
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
        progress,
        read_bytes: 0,
    };
    executor.execute()?;
    Ok(HostStorage {
        arena: executor.arena,
        tensors: executor.tensors,
        max_tile_write_bytes: executor.max_tile_write_bytes,
    })
}

struct HostExecutor<'a, 'p> {
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
    progress: &'p mut dyn FnMut(Progress<'_>),
    read_bytes: u64,
}

/// Decode one GGUF `Q4_0` block: one F16 scale, then sixteen packed bytes whose
/// low nibbles are elements 0..16 and high nibbles 16..32, each
/// `(nibble − 8) × scale`.
///
/// The only block decoder the loader carries, and it is here rather than beside
/// a reader because decoding is not reading: the runtime materialization is the
/// driver's job, and this exists so the offline executor can check the driver's
/// answer against an independent one.
fn decode_gguf_q4_0_block_into(block: &[u8; 18], values: &mut [f32; 32]) {
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    for i in 0..16 {
        let packed = block[2 + i];
        let lo = (packed & 0x0f) as i32 - 8;
        let hi = ((packed >> 4) & 0x0f) as i32 - 8;
        values[i] = scale * lo as f32;
        values[i + 16] = scale * hi as f32;
    }
}

impl HostExecutor<'_, '_> {
    fn execute(&mut self) -> Result<(), Error> {
        for id in &self.plan.schedule {
            let instr = instr_by_id(&self.plan.instrs, *id)?.clone();
            // Accounted before the match consumes the instruction, reported
            // after its work is done.
            let consumed = match &instr {
                StorageInstr::ExtentWrite { source, .. }
                | StorageInstr::BulkExtentWrite { source, .. } => source.span_bytes,
                StorageInstr::TileMap { source, .. } => {
                    source.as_ref().map_or(0, |source| source.span_bytes)
                }
                _ => 0,
            };
            let finalized = match &instr {
                StorageInstr::Finalize { name, .. } => Some(name.clone()),
                _ => None,
            };
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
            self.read_bytes += consumed;
            (self.progress)(Progress {
                read_bytes: self.read_bytes,
                total_read_bytes: self.plan.memory.checkpoint_read_bytes,
                finalized: finalized.as_deref(),
            });
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
        gather_strided(raw, &normalized)
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

        // A large read is split across threads on positioned reads — the file
        // handle is shared, only the offsets differ, and byte `i` lands at
        // `out[i]` either way, so the result is the serial read's. What it
        // buys is page-cache bandwidth: one thread memcpying out of the cache
        // was the second-largest serial cost of a conversion after the encode
        // itself.
        #[cfg(unix)]
        {
            const PARALLEL_READ_MIN: usize = 64 << 20;
            let workers = std::thread::available_parallelism()
                .map_or(1, std::num::NonZero::get)
                .min(16);
            if len >= PARALLEL_READ_MIN && workers > 1 {
                use std::os::unix::fs::FileExt;
                let chunk = len.div_ceil(workers);
                let failures: Vec<std::io::Result<()>> = std::thread::scope(|scope| {
                    out.chunks_mut(chunk)
                        .enumerate()
                        .map(|(at, buf)| {
                            let file = &file;
                            scope.spawn(move || {
                                file.read_exact_at(buf, offset + (at * chunk) as u64)
                            })
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .map(|worker| worker.join().expect("a read worker does not panic"))
                        .collect()
                });
                for failure in failures {
                    failure.map_err(|err| invalid(format!("read {}: {err}", path.display())))?;
                }
                return Ok(out);
            }
        }

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
            TileMapKind::Decode => {
                self.decode_bytes(outputs.first().copied(), &input, &transform)?
            }
            // Encode is the one transform with more than one output — the
            // payload and the scale (and zero-point) tensors it cannot be
            // read without — so both arms write their own buffers and return
            // rather than flowing into the single-output plumbing below. The
            // MLX affine scheme keeps its own arm: it publishes three tensors
            // where every other scheme publishes two.
            TileMapKind::Encode if transform.to == Some(QuantScheme::MlxAffineU4) => {
                let written = self.encode_mlx_affine_u4(&input, outputs, &transform)?;
                for (buffer, bytes) in written {
                    self.max_tile_write_bytes = self.max_tile_write_bytes.max(bytes.len());
                    self.write_buffer(buffer, 0, &bytes)?;
                }
                return Ok(());
            }
            TileMapKind::Encode => {
                return self.encode_bytes(
                    source,
                    inputs,
                    outputs,
                    &input,
                    &transform,
                    max_tile_bytes,
                );
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
            // A per-group `Scale` and a `Decode` are the transforms whose
            // output is a different width from their input by design:
            // unpacking four-bit codes into `BF16` quadruples the bytes, and a
            // GGUF block trades 18 bytes for 64. Every other kind moves the
            // same bytes it read, and the mismatch is a bug worth catching.
            if transform.scale_blocks.is_empty() && kind != TileMapKind::Decode {
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
    /// The MLX affine-U4 encode: the one scheme that publishes *three*
    /// tensors — a quantized weight is unreadable without the metadata the
    /// same pass computes, and an affine scheme's metadata is scales *and*
    /// zero points. Every two-output scheme lives in [`Self::encode_bytes`].
    ///
    /// `outputs` is positional and the loader fixed the order: the weight, then
    /// the metadata in the order `quant_metadata_outputs` declared it.
    fn encode_mlx_affine_u4(
        &self,
        bytes: &[u8],
        outputs: &[BufferId],
        transform: &TransformSpec,
    ) -> Result<Vec<(BufferId, Vec<u8>)>, Error> {
        let Some(scheme) = transform.to else {
            return Err(invalid("Encode carries no destination scheme"));
        };
        if scheme != QuantScheme::MlxAffineU4 {
            return Err(invalid(format!(
                "host Encode does not implement {scheme:?}"
            )));
        }
        let [weight, scales, biases] = outputs else {
            return Err(invalid(format!(
                "encoding to {scheme:?} writes a weight, its scales and its zero \
                 points, but the instruction has {} outputs",
                outputs.len()
            )));
        };
        let shape = self
            .index
            .buffer_tensor(self.plan, *weight)
            .ok_or_else(|| invalid("Encode output has no tensor type"))?
            .shape
            .clone();
        let [rows, cols] = shape[..] else {
            return Err(invalid(format!(
                "encoding to {scheme:?} walks [rows, cols] tiles, not {shape:?}"
            )));
        };
        let group = i64::from(scheme.default_group_size());
        if group <= 0 || cols % group != 0 {
            return Err(invalid(format!(
                "encoding to {scheme:?} groups {group} columns, which does not \
                 divide the {cols} of {shape:?}"
            )));
        }
        // The operand is whatever the chain decoded to; `decode_values` gives
        // f64 and the quantizer works in f32, which is what MLX's own encoder
        // does and therefore what the codes have to be rounded from.
        let values = decode_values(bytes, self.buffer_dtype_of_input(*weight, bytes.len())?)?;
        if values.len() as i64 != rows * cols {
            return Err(invalid(format!(
                "Encode was handed {} elements for the {shape:?} it must quantize",
                values.len()
            )));
        }

        let n_groups = (rows * cols / group) as usize;
        let mut packed = Vec::with_capacity(values.len() / 8 * 4);
        let mut scale_values = Vec::with_capacity(n_groups);
        let mut bias_values = Vec::with_capacity(n_groups);
        for chunk in values.chunks(group as usize) {
            let (scale, bias) = mlx_affine_group_params(chunk);
            for word in chunk.chunks(8) {
                let mut out: u32 = 0;
                for (k, &value) in word.iter().enumerate() {
                    let code = (((value as f32) - bias) / scale)
                        .round()
                        .clamp(0.0, 15.0) as u32;
                    out |= code << (k * 4);
                }
                packed.extend_from_slice(&out.to_le_bytes());
            }
            scale_values.push(f64::from(scale));
            bias_values.push(f64::from(bias));
        }
        Ok(vec![
            (*weight, packed),
            (*scales, encode_values(&scale_values, DType::BF16)?),
            (*biases, encode_values(&bias_values, DType::BF16)?),
        ])
    }

    /// The dtype an Encode's operand bytes are read as.
    ///
    /// The operand is an intermediate the chain produced, so its width is what
    /// the byte count and the destination's element count agree on: the
    /// destination buffer is quantized, and asking it for a `DType` would get
    /// the scheme's logical type rather than the operand's storage type.
    fn buffer_dtype_of_input(&self, weight: BufferId, bytes: usize) -> Result<DType, Error> {
        let shape = &self
            .index
            .buffer_tensor(self.plan, weight)
            .ok_or_else(|| invalid("Encode output has no tensor type"))?
            .shape;
        let elements: i64 = shape.iter().product();
        match bytes as i64 / elements.max(1) {
            2 => Ok(DType::BF16),
            4 => Ok(DType::F32),
            other => Err(invalid(format!(
                "Encode operand is {other} bytes per element, which is neither \
                 the BF16 nor the F32 a quantizer reads"
            ))),
        }
    }

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

    /// Decode a self-contained blocked payload to its logical dtype: the
    /// `Cast(Quant → Raw)` direction, which the builder lowers to `Decode`.
    ///
    /// Only schemes whose scales live inside the block reach here — the
    /// validate pass admits exactly those — so the payload bytes are the whole
    /// story and there is no factor operand to fetch. GGUF Q4_0 decodes through
    /// [`decode_gguf_q4_0_block_into`] below, and the `f32` values it yields
    /// narrow to the scheme's logical `BF16` by round to nearest even.
    fn decode_bytes(
        &self,
        output: Option<BufferId>,
        bytes: &[u8],
        transform: &TransformSpec,
    ) -> Result<Vec<u8>, Error> {
        if transform.from != Some(QuantScheme::GgufQ4_0) {
            return Err(invalid(format!(
                "host Decode implements GGUF Q4_0 only, not {:?}",
                transform.from
            )));
        }
        let output = output.ok_or_else(|| invalid("host Decode requires an output buffer"))?;
        let dtype = self.buffer_dtype(output)?;
        if dtype != DType::BF16 {
            return Err(invalid(format!(
                "GGUF Q4_0 decodes to its logical BF16, not {dtype:?}"
            )));
        }
        if !bytes.len().is_multiple_of(18) {
            return Err(invalid(format!(
                "host Decode read {} bytes, not whole 18-byte Q4_0 blocks",
                bytes.len()
            )));
        }
        let blocks = bytes.len() / 18;
        let mut out = vec![0u8; blocks * 64];
        // Blocks are self-contained, so they decode in parallel the same way
        // encode rows do: disjoint output slices, any worker count, the same
        // bytes.
        let workers = if blocks < (1 << 15) {
            1
        } else {
            std::thread::available_parallelism()
                .map_or(1, std::num::NonZero::get)
                .min(blocks)
        };
        let per_worker = blocks.div_ceil(workers);
        std::thread::scope(|scope| {
            let mut out_rest = &mut out[..];
            let mut start = 0usize;
            while start < blocks {
                let count = per_worker.min(blocks - start);
                let (chunk, rest) = std::mem::take(&mut out_rest).split_at_mut(count * 64);
                out_rest = rest;
                let source = &bytes[start * 18..(start + count) * 18];
                scope.spawn(move || {
                    let mut values = [0.0f32; 32];
                    for (block, out) in source.chunks_exact(18).zip(chunk.chunks_exact_mut(64)) {
                        decode_gguf_q4_0_block_into(
                            block.try_into().expect("chunks are 18 bytes"),
                            &mut values,
                        );
                        for (value, le) in values.iter().zip(out.chunks_exact_mut(2)) {
                            le.copy_from_slice(&bf16::from_f32(*value).to_bits().to_le_bytes());
                        }
                    }
                });
                start += count;
            }
        });
        Ok(out)
    }

    /// Quantize on the host: the ports of the CUDA encode kernels, arithmetic
    /// first, one arm per scheme `validate-target-support` admits.
    ///
    /// * `Mxfp4E2M1E8M0` — `quant_bf16_to_mxfp4.cu`: per 32-element group
    ///   along the last axis, absmax → one E8M0 byte scale (the smallest
    ///   power of two whose ×6 covers the absmax), each element divided by it
    ///   and rounded through the kernel's midpoint table.
    /// * `Fp8E4M3` — `quant_bf16_to_fp8.cu::quant_per_channel_kernel`: per
    ///   row, absmax → one `F32` factor `absmax/448` (`1.0` for a dead row),
    ///   elements saturate-cast to E4M3, round to nearest even.
    /// * `Int8Symmetric` — the same shape with `127` for `448` and
    ///   `rintf`'s round-half-even for the cast.
    ///
    /// The operand reaches every scheme as `BF16` rows because that is what
    /// the device encodes from (`materialize_encode_input_bf16_rows`): a
    /// wider weight is narrowed first — encoding straight from `f32` would
    /// keep mantissa bits the device never saw — and an FP8 block-scaled
    /// source, the one the instruction names by `metadata_source`, is
    /// dequantized with its block factors the way
    /// `transcode_engine.hpp::fp8_tile_scale` resolves them.
    ///
    /// Every row's output depends on that row alone, in all three schemes, so
    /// rows are encoded in parallel; the worker count changes wall-clock and
    /// nothing else.
    fn encode_bytes(
        &mut self,
        source: Option<&SourceExtent>,
        inputs: &[BufferId],
        outputs: &[BufferId],
        bytes: &[u8],
        transform: &TransformSpec,
        max_tile_bytes: u64,
    ) -> Result<(), Error> {
        let Some(scheme) = transform.to else {
            return Err(invalid("host Encode names no target scheme"));
        };
        let &[payload, scales] = outputs else {
            return Err(invalid("host Encode expects weight and scale outputs"));
        };
        let shape = self
            .index
            .buffer_tensor(self.plan, payload)
            .ok_or_else(|| invalid("host Encode payload has no tensor type"))?
            .shape
            .clone();
        let &[rows, cols] = shape.as_slice() else {
            return Err(invalid(format!(
                "host Encode expects a 2-D weight, got shape {shape:?}"
            )));
        };
        let (rows, cols) = (checked_usize_i64(rows)?, checked_usize_i64(cols)?);
        let scale_shape = self
            .index
            .buffer_tensor(self.plan, scales)
            .ok_or_else(|| invalid("host Encode scale has no tensor type"))?
            .shape
            .clone();

        let dtype = if let Some(source) = source {
            self.source_dtype(source.tensor_id)?
        } else if let Some(input) = inputs.first() {
            self.buffer_dtype(*input)?
        } else {
            return Err(invalid("host Encode requires a source or input buffer"));
        };
        if bytes.len() != rows * cols * dtype.bytes() as usize {
            return Err(invalid(format!(
                "host Encode read {} bytes for a [{rows}, {cols}] {dtype:?} weight",
                bytes.len()
            )));
        }
        let operand = if dtype == DType::F8E4M3 {
            let source = source.ok_or_else(|| {
                invalid("host Encode of an FP8 operand requires a checkpoint source")
            })?;
            self.fp8_block_operand(source, transform, bytes, rows, cols)?
        } else {
            EncodeOperand::Widened { bytes, dtype }
        };
        if let EncodeOperand::Widened { dtype, .. } = &operand
            && !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        {
            return Err(invalid(format!(
                "host Encode reads BF16 (or F16/F32 narrowed to it, or \
                 block-scaled F8E4M3), not {dtype:?}"
            )));
        }

        let (packed, scale_out) = match scheme {
            QuantScheme::Mxfp4E2M1E8M0 => {
                if cols == 0 || !cols.is_multiple_of(32) {
                    return Err(invalid(format!(
                        "host MXFP4 Encode cols must be a multiple of 32, got {cols}"
                    )));
                }
                let groups = cols / 32;
                if scale_shape.as_slice() != [rows as i64, groups as i64] {
                    return Err(invalid(format!(
                        "host MXFP4 Encode scale must be [{rows}, {groups}], got {scale_shape:?}"
                    )));
                }
                let mut packed = vec![0u8; rows * cols / 2];
                let mut scale_out = vec![0u8; rows * groups];
                let job = |row: usize, buf: &mut [f32], out: &mut [u8], scale: &mut [u8]| {
                    operand.row_bf16(row, cols, buf);
                    for g in 0..groups {
                        scale[g] = encode_mxfp4_group(
                            &buf[g * 32..g * 32 + 32],
                            &mut out[g * 16..g * 16 + 16],
                        );
                    }
                };
                encode_rows(
                    rows,
                    cols,
                    cols / 2,
                    groups,
                    &mut packed,
                    &mut scale_out,
                    &job,
                );
                (packed, scale_out)
            }
            QuantScheme::Fp8E4M3 | QuantScheme::Int8Symmetric => {
                if scale_shape.as_slice() != [rows as i64] {
                    return Err(invalid(format!(
                        "host per-channel Encode scale must be [{rows}], got {scale_shape:?}"
                    )));
                }
                let int8 = scheme == QuantScheme::Int8Symmetric;
                let code_max = if int8 { 127.0f32 } else { 448.0 };
                let mut packed = vec![0u8; rows * cols];
                let mut scale_out = vec![0u8; rows * 4];
                let job = |row: usize, buf: &mut [f32], out: &mut [u8], scale: &mut [u8]| {
                    operand.row_bf16(row, cols, buf);
                    let mut absmax = 0.0f32;
                    for &v in buf.iter() {
                        let a = v.abs();
                        if a > absmax {
                            absmax = a;
                        }
                    }
                    // A dead row quantizes by 1.0 to all zeros, exactly as
                    // the kernel's degenerate arm says.
                    let (recip, factor) = if absmax > 0.0 {
                        (code_max / absmax, absmax / code_max)
                    } else {
                        (1.0, 1.0)
                    };
                    scale.copy_from_slice(&factor.to_le_bytes());
                    if int8 {
                        for (value, out) in buf.iter().zip(out.iter_mut()) {
                            let q = (value * recip).round_ties_even() as i32;
                            *out = q.clamp(-128, 127) as i8 as u8;
                        }
                    } else {
                        for (value, out) in buf.iter().zip(out.iter_mut()) {
                            *out = f32_to_fp8_e4m3(value * recip);
                        }
                    }
                };
                encode_rows(rows, cols, cols, 4, &mut packed, &mut scale_out, &job);
                (packed, scale_out)
            }
            other => {
                return Err(invalid(format!(
                    "no encode kernel writes {other:?}, so the host executor \
                     does not either"
                )));
            }
        };

        if self.buffer_bytes(payload)?.len() != packed.len() {
            return Err(invalid("host Encode payload buffer size mismatch"));
        }
        if self.buffer_bytes(scales)?.len() != scale_out.len() {
            return Err(invalid("host Encode scale buffer size mismatch"));
        }
        let tile = if max_tile_bytes == 0 {
            packed.len().max(1)
        } else {
            checked_usize(max_tile_bytes)?.max(1)
        };
        for (offset, chunk) in packed.chunks(tile).enumerate() {
            self.max_tile_write_bytes = self.max_tile_write_bytes.max(chunk.len());
            self.write_buffer(payload, offset * tile, chunk)?;
        }
        self.write_buffer(scales, 0, &scale_out)?;
        Ok(())
    }

    /// Resolve an FP8 block-scaled Encode operand the way
    /// `transcode_engine.hpp::fp8_tile_scale` does.
    ///
    /// The factor tensor is the one the instruction names — the loader read
    /// the tensor table, so the loader answered — the group size is the ratio
    /// of the weight's *on-disk* shape to the factor tensor's (square, or the
    /// checkpoint is not one either kernel indexes), and a TP shard's offset
    /// within the full weight is decoded from the extent's base offset, which
    /// is in elements because FP8 is one byte each.
    fn fp8_block_operand<'b>(
        &self,
        source: &SourceExtent,
        transform: &TransformSpec,
        bytes: &'b [u8],
        rows: usize,
        cols: usize,
    ) -> Result<EncodeOperand<'b>, Error> {
        let Some(metadata_source) = transform.metadata_source else {
            return Err(invalid(
                "host Encode of an FP8 source needs the block-scale tensor its \
                 instruction names, and this instruction names none",
            ));
        };
        let weight = self
            .index
            .source(self.plan, source.tensor_id)
            .ok_or_else(|| invalid("FP8 Encode source is not in the plan"))?;
        let scale = self
            .index
            .source(self.plan, metadata_source)
            .ok_or_else(|| invalid("FP8 Encode scale tensor is not in the plan"))?;
        let (&[true_rows, true_cols], &[scale_rows, scale_cols]) =
            (weight.shape.as_slice(), scale.shape.as_slice())
        else {
            return Err(invalid(format!(
                "FP8 Encode weight and scale must be 2-D, got {:?} and {:?}",
                weight.shape, scale.shape
            )));
        };
        if !matches!(scale.encoding, Encoding::Raw(DType::F32)) {
            return Err(invalid(format!(
                "FP8 Encode scale '{}' must be F32",
                scale.name
            )));
        }
        let (true_rows, true_cols) = (checked_usize_i64(true_rows)?, checked_usize_i64(true_cols)?);
        let (scale_rows, scale_cols) = (
            checked_usize_i64(scale_rows)?,
            checked_usize_i64(scale_cols)?,
        );
        let group_rows = true_rows.checked_div(scale_rows).unwrap_or(0);
        let group_cols = true_cols.checked_div(scale_cols).unwrap_or(0);
        if group_rows == 0 || group_rows != group_cols {
            return Err(invalid(format!(
                "FP8 Encode source '{}' has unsupported scale shape \
                 [{scale_rows}, {scale_cols}] for weight [{true_rows}, {true_cols}]",
                weight.name
            )));
        }
        let group = group_rows;

        // The rank's corner within the full weight, then within the factors.
        let base = checked_usize(source.stride.base_offset)?;
        let scale_row_offset = (base / true_cols) / group;
        let scale_col_offset = (base % true_cols) / group;
        if scale_row_offset + rows.div_ceil(group) > scale_rows + 1
            || scale_col_offset + cols.div_ceil(group) > scale_cols + 1
        {
            return Err(invalid(format!(
                "FP8 Encode shard [{rows}, {cols}] at offset {base} reaches past \
                 the [{scale_rows}, {scale_cols}] factors of '{}'",
                scale.name
            )));
        }

        let factor_bytes = self.read_extent(&SourceExtent {
            file_id: scale.file_id,
            tensor_id: scale.id,
            file_offset: scale.file_offset,
            span_bytes: scale.span_bytes,
            stride: Extent {
                base_offset: 0,
                element_bytes: 4,
                dims: vec![crate::extent::Dim {
                    count: (scale_rows * scale_cols) as i64,
                    src_stride: 4,
                    dst_stride: 4,
                }],
            },
            dtype: DType::F32,
        })?;
        let factors: Vec<f32> = factor_bytes
            .chunks_exact(4)
            .map(|le| f32::from_le_bytes(le.try_into().unwrap()))
            .collect();
        // The last block may be short, so the strict check is on the first
        // element each row reads; the loop below indexes the same way the
        // blocked dequant kernel does.
        let last_index = (scale_row_offset + (rows.saturating_sub(1)) / group) * scale_cols
            + scale_col_offset
            + (cols.saturating_sub(1)) / group;
        if rows > 0 && cols > 0 && last_index >= factors.len() {
            return Err(invalid(format!(
                "FP8 Encode factors for '{}' end at {} but the shard reads {}",
                scale.name,
                factors.len(),
                last_index
            )));
        }
        Ok(EncodeOperand::BlockScaledFp8 {
            bytes,
            factors,
            scale_cols,
            group,
            scale_row_offset,
            scale_col_offset,
        })
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

fn gather_strided(raw: Vec<u8>, extent: &Extent) -> Result<Vec<u8>, Error> {
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
    let total = elements.saturating_mul(elem);

    // Fold every trailing source-dense dimension into one contiguous run, so
    // the loop below moves a run per iteration instead of an element. The
    // common extents collapse entirely — a whole tensor, or a row shard whose
    // rows are contiguous — and what is left iterates only the axes that
    // actually stride. Walking element-wise here was measured at ~2.8 s of a
    // 3 s conversion, on bounds checks and a `Vec` per element.
    let mut run = elem;
    let mut outer = extent.dims.len();
    while outer > 0 {
        let dim = &extent.dims[outer - 1];
        if dim.src_stride != run as i64 {
            break;
        }
        match run.checked_mul(checked_usize_i64(dim.count)?) {
            Some(next) => run = next,
            None => return Err(invalid("extent run length overflow")),
        }
        outer -= 1;
    }
    if outer == 0 {
        // One dense run: the physical bytes *are* the compact bytes, and the
        // read already owns them — a copy here would double the largest
        // allocation of a conversion to move nothing.
        if raw.len() < total {
            return Err(invalid("source extent is out of bounds"));
        }
        let mut raw = raw;
        raw.truncate(total);
        return Ok(raw);
    }

    let mut out = vec![0u8; total];
    let mut coord = vec![0usize; outer];
    let mut dst = 0usize;
    while dst < total {
        let src = extent_offset(&coord, &extent.dims[..outer], true)?;
        out.get_mut(dst..dst + run)
            .ok_or_else(|| invalid("compact extent range overflow"))?
            .copy_from_slice(
                raw.get(src..src + run)
                    .ok_or_else(|| invalid("source extent is out of bounds"))?,
            );
        dst += run;
        for axis in (0..outer).rev() {
            coord[axis] += 1;
            if coord[axis] < shape[axis] {
                break;
            }
            coord[axis] = 0;
        }
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

/// How an Encode reads its operand as `BF16` rows.
///
/// Resolved once, before the row loop, so the per-row work is indexing and
/// arithmetic only — which is also what lets the rows run on any thread.
enum EncodeOperand<'a> {
    /// A raw operand, narrowed to `BF16` element by element the way the
    /// device's cast does.
    Widened { bytes: &'a [u8], dtype: DType },
    /// An FP8 payload and the `F32` block factors that make it numbers,
    /// multiplied out per element: `bf16(f32(fp8) · factor)`, the blocked
    /// dequant kernel's expression.
    BlockScaledFp8 {
        bytes: &'a [u8],
        factors: Vec<f32>,
        scale_cols: usize,
        group: usize,
        scale_row_offset: usize,
        scale_col_offset: usize,
    },
}

impl EncodeOperand<'_> {
    /// Fill `buf` with row `row` as the `f32` widening of its `BF16` reading.
    fn row_bf16(&self, row: usize, cols: usize, buf: &mut [f32]) {
        match self {
            EncodeOperand::Widened { bytes, dtype } => {
                let width = dtype.bytes() as usize;
                let row_bytes = &bytes[row * cols * width..(row + 1) * cols * width];
                match dtype {
                    DType::BF16 => {
                        #[cfg(target_arch = "x86_64")]
                        if std::arch::is_x86_feature_detected!("avx2") {
                            // Sound: the feature was just detected, and the
                            // slices agree on length by the arm's slicing.
                            unsafe { avx2::decode_bf16_row(row_bytes, buf) };
                            return;
                        }
                        for (le, out) in row_bytes.chunks_exact(2).zip(buf.iter_mut()) {
                            // BF16 widens by a shift; spelled directly rather
                            // than through `half` so the decode vectorizes.
                            let bits = u16::from_le_bytes(le.try_into().unwrap());
                            *out = f32::from_bits(u32::from(bits) << 16);
                        }
                    }
                    DType::F16 => {
                        for (le, out) in row_bytes.chunks_exact(2).zip(buf.iter_mut()) {
                            let wide =
                                f16::from_bits(u16::from_le_bytes(le.try_into().unwrap())).to_f32();
                            *out = bf16::from_f32(wide).to_f32();
                        }
                    }
                    DType::F32 => {
                        for (le, out) in row_bytes.chunks_exact(4).zip(buf.iter_mut()) {
                            *out =
                                bf16::from_f32(f32::from_le_bytes(le.try_into().unwrap())).to_f32();
                        }
                    }
                    // `encode_bytes` admitted only the three above.
                    _ => unreachable!("EncodeOperand::Widened holds a vetted dtype"),
                }
            }
            EncodeOperand::BlockScaledFp8 {
                bytes,
                factors,
                scale_cols,
                group,
                scale_row_offset,
                scale_col_offset,
            } => {
                let row_bytes = &bytes[row * cols..(row + 1) * cols];
                let scale_row = (scale_row_offset + row / group) * scale_cols;
                for (col, (&code, out)) in row_bytes.iter().zip(buf.iter_mut()).enumerate() {
                    let factor = factors[scale_row + scale_col_offset + col / group];
                    *out = bf16::from_f32(fp8_e4m3_to_f32(code) * factor).to_f32();
                }
            }
        }
    }
}

/// One row of an encode: `(row, f32 scratch, payload-row out, scale-row out)`.
type EncodeRowJob<'a> = dyn Fn(usize, &mut [f32], &mut [u8], &mut [u8]) + Sync + 'a;

/// Run `job` over every row of an encode, in parallel when the tensor pays
/// for it.
///
/// The outputs are handed to each worker as disjoint `split_at_mut` slices of
/// the two flat buffers, so the parallelism needs no synchronisation — and
/// because every row's bytes depend on that row alone, the worker count
/// changes wall-clock and nothing else. Each worker keeps one `f32` scratch
/// row rather than one per call.
fn encode_rows(
    rows: usize,
    cols: usize,
    out_row_bytes: usize,
    scale_row_bytes: usize,
    out: &mut [u8],
    scales: &mut [u8],
    job: &EncodeRowJob<'_>,
) {
    // Below about a megabyte of input the threads cost more than they carry.
    let workers = if rows * cols < (1 << 20) {
        1
    } else {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(rows.max(1))
    };
    if workers <= 1 {
        let mut buf = vec![0.0f32; cols];
        for row in 0..rows {
            let out = &mut out[row * out_row_bytes..(row + 1) * out_row_bytes];
            let scale = &mut scales[row * scale_row_bytes..(row + 1) * scale_row_bytes];
            job(row, &mut buf, out, scale);
        }
        return;
    }
    let rows_per = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut out_rest = out;
        let mut scale_rest = scales;
        let mut start = 0usize;
        while start < rows {
            let count = rows_per.min(rows - start);
            let (out_chunk, next) =
                std::mem::take(&mut out_rest).split_at_mut(count * out_row_bytes);
            out_rest = next;
            let (scale_chunk, next) =
                std::mem::take(&mut scale_rest).split_at_mut(count * scale_row_bytes);
            scale_rest = next;
            let first = start;
            scope.spawn(move || {
                let mut buf = vec![0.0f32; cols];
                for i in 0..count {
                    let out = &mut out_chunk[i * out_row_bytes..(i + 1) * out_row_bytes];
                    let scale = &mut scale_chunk[i * scale_row_bytes..(i + 1) * scale_row_bytes];
                    job(first + i, &mut buf, out, scale);
                }
            });
            start += count;
        }
    });
}

/// One 32-element MXFP4 group: absmax → E8M0 byte, elements → 16 packed
/// nibble pairs in `out`. Returns the scale byte.
///
/// Dispatches to the AVX2 restatement when the CPU has it; the scalar body
/// below is the reference the vector path is checked against
/// (`avx2_group_encode_matches_the_scalar_reference`), and both are the
/// kernel's arithmetic.
fn encode_mxfp4_group(group: &[f32], out: &mut [u8]) -> u8 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // Sound: the feature was just detected, and the slices are checked by
        // the callee's debug asserts and the caller's slicing.
        return unsafe { avx2::encode_mxfp4_group(group, out) };
    }
    encode_mxfp4_group_scalar(group, out)
}

fn encode_mxfp4_group_scalar(group: &[f32], out: &mut [u8]) -> u8 {
    // `f32::max` ignores a NaN operand, which is the kernel's
    // `if (a > absmax)` by another spelling.
    let mut absmax = 0.0f32;
    for &v in group {
        absmax = absmax.max(v.abs());
    }
    let sb = encode_e8m0(absmax);
    // An exact power of two, so the reciprocal is exact too and multiplying
    // by it is the kernel's divide.
    let inv_s = 1.0 / exp2_e8m0(sb);
    let mut codes = [0u8; 32];
    for (v, code) in group.iter().zip(codes.iter_mut()) {
        *code = encode_fp4_e2m1(v * inv_s);
    }
    for (k, pair) in codes.chunks_exact(2).enumerate() {
        out[k] = (pair[1] << 4) | pair[0];
    }
    sb
}

/// The encode hot loops as AVX2 lanes, lane-for-lane the scalar arithmetic.
///
/// Nothing here is a different algorithm — every intrinsic was chosen to
/// reproduce one scalar expression bit for bit, NaN cases included:
///
/// * `_CMP_NLT_UQ` is `!(a < t)` — true on NaN — which is the E2M1 ladder's
///   test and how a NaN element lands on magnitude 7;
/// * `_mm256_max_ps(x, acc)` returns `acc` when `x` is NaN, which is
///   `acc.max(x)`'s NaN-ignoring absmax;
/// * `_CMP_LT_OQ` against zero is `x < 0.0` — false for NaN and `-0.0` — so
///   the sign bit lands exactly where the scalar puts it.
///
/// The multiply is `vmulps`, IEEE round-to-nearest like the scalar's, and the
/// BF16 widening is the same `<< 16`. Everything is `unsafe` only for the
/// `target_feature` contract; callers dispatch behind runtime detection.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Widen one BF16 row to `f32`, eight lanes per step.
    ///
    /// # Safety
    /// The caller detected `avx2`. `row` holds `out.len()` little-endian
    /// BF16 values.
    #[target_feature(enable = "avx2")]
    pub unsafe fn decode_bf16_row(row: &[u8], out: &mut [f32]) {
        debug_assert!(row.len() == out.len() * 2);
        let n = out.len();
        let mut at = 0;
        unsafe {
            while at + 8 <= n {
                let half = _mm_loadu_si128(row.as_ptr().add(at * 2).cast());
                let wide = _mm256_cvtepu16_epi32(half);
                let bits = _mm256_slli_epi32::<16>(wide);
                _mm256_storeu_ps(out.as_mut_ptr().add(at), _mm256_castsi256_ps(bits));
                at += 8;
            }
        }
        for k in at..n {
            let bits = u16::from_le_bytes([row[2 * k], row[2 * k + 1]]);
            out[k] = f32::from_bits(u32::from(bits) << 16);
        }
    }

    /// One 32-element MXFP4 group; see [`super::encode_mxfp4_group_scalar`]
    /// for the reference this restates.
    ///
    /// # Safety
    /// The caller detected `avx2`. `group` has 32 elements, `out` has 16.
    #[target_feature(enable = "avx2")]
    pub unsafe fn encode_mxfp4_group(group: &[f32], out: &mut [u8]) -> u8 {
        debug_assert!(group.len() == 32 && out.len() == 16);
        unsafe {
            let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));

            let mut lanes = [_mm256_setzero_ps(); 4];
            let mut acc = _mm256_setzero_ps();
            for (i, chunk) in lanes.iter_mut().enumerate() {
                let v = _mm256_loadu_ps(group.as_ptr().add(i * 8));
                *chunk = v;
                // NaN in the first operand yields the second: `acc.max(|v|)`.
                acc = _mm256_max_ps(_mm256_and_ps(v, abs_mask), acc);
            }
            // The accumulator lanes are NaN-free, so the horizontal order is
            // free to be anything.
            let quad = _mm_max_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
            let pair = _mm_max_ps(quad, _mm_movehl_ps(quad, quad));
            let one = _mm_max_ss(pair, _mm_shuffle_ps::<1>(pair, pair));
            let absmax = _mm_cvtss_f32(one);

            let sb = super::encode_e8m0(absmax);
            let inv_s = _mm256_set1_ps(1.0 / super::exp2_e8m0(sb));

            let thresholds = [0.25f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
            let zero_ps = _mm256_setzero_ps();
            let zero_si = _mm256_setzero_si256();
            let low_bit = _mm256_set1_epi32(1);
            let mut codes = [0i32; 32];
            for (i, &chunk) in lanes.iter().enumerate() {
                let x = _mm256_mul_ps(chunk, inv_s);
                let a = _mm256_and_ps(x, abs_mask);
                let mut mag = zero_si;
                for t in thresholds {
                    // A true lane is -1; subtracting counts the midpoints
                    // `a` is not below, NaN counting all seven.
                    let not_below = _mm256_cmp_ps::<_CMP_NLT_UQ>(a, _mm256_set1_ps(t));
                    mag = _mm256_sub_epi32(mag, _mm256_castps_si256(not_below));
                }
                let negative = _mm256_cmp_ps::<_CMP_LT_OQ>(x, zero_ps);
                let sign = _mm256_slli_epi32::<3>(_mm256_and_si256(
                    _mm256_castps_si256(negative),
                    low_bit,
                ));
                let nonzero = _mm256_cmpgt_epi32(mag, zero_si);
                let code = _mm256_and_si256(_mm256_or_si256(mag, sign), nonzero);
                _mm256_storeu_si256(codes.as_mut_ptr().add(i * 8).cast(), code);
            }
            for (k, pair) in codes.chunks_exact(2).enumerate() {
                out[k] = ((pair[1] as u8) << 4) | pair[0] as u8;
            }
            sb
        }
    }
}

/// `2^e` for a normal-range exponent, built rather than computed.
fn exp2i(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// One E4M3 byte as the `f32` it denotes, `__nv_cvt_fp8_to_halfraw` widened:
/// 1-4-3, bias 7, no infinities, `S.1111.111` the one NaN.
fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = (byte >> 3) & 0xF;
    let mant = (byte & 0x7) as f32;
    if exp == 0xF && byte & 0x7 == 0x7 {
        return f32::NAN;
    }
    let value = if exp == 0 {
        // Subnormal: units of 2^-9.
        mant * exp2i(-9)
    } else {
        (1.0 + mant / 8.0) * exp2i(i32::from(exp) - 7)
    };
    sign * value
}

/// `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)`: round to nearest
/// even, saturate finite overflow — and infinity — to ±448, NaN to the
/// scheme's NaN byte with the sign kept.
fn f32_to_fp8_e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    if x.is_nan() {
        return sign | 0x7F;
    }
    let a = x.abs();
    if a >= 448.0 {
        // Everything past the last code rounds or saturates onto it: the
        // next magnitude up the grid is the NaN slot, which SATFINITE never
        // produces.
        return sign | 0x7E;
    }
    if a < 0.015625 {
        // Subnormal: quantize in units of 2^-9. The multiply is by a power
        // of two, so it is exact and the tie is the true tie; 8 rolls over
        // into exactly the first normal code.
        let q = (a * 512.0).round_ties_even() as u32;
        return sign | q as u8;
    }
    let bits = a.to_bits();
    let mut e = ((bits >> 23) as i32) - 127;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    let mut q = (m * 8.0).round_ties_even() as u32;
    if q == 16 {
        e += 1;
        q = 8;
    }
    sign | (((e + 7) as u8) << 3) | (q as u8 - 8)
}

/// One FP4 E2M1 codepoint: `quant_bf16_to_mxfp4.cu::encode_fp4_e2m1`, its
/// midpoint table verbatim.
///
/// Every comparison is false for a NaN operand, which falls through to the
/// largest magnitude with a positive sign — not a choice here, the port of
/// the kernel's. A signed zero rounds to `+0`.
// The negated comparisons are the port: `!(a < t)` is true for NaN where
// `a >= t` is not, and NaN-rounds-to-the-top-magnitude is the kernel's
// behaviour.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn encode_fp4_e2m1(x: f32) -> u8 {
    let a = x.abs();
    // Magnitudes 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0; boundaries are the
    // midpoints between neighbours, so this is round-to-nearest with ties up.
    //
    // The kernel's ladder, summed instead of chained: the magnitude is how
    // many midpoints `a` is *not below*, which is the same arithmetic — a NaN
    // fails every comparison and lands on 7, exactly as it falls through the
    // ladder — but branch-free, which is what lets the per-element encode
    // loop vectorize.
    let mag = u8::from(!(a < 0.25))
        + u8::from(!(a < 0.75))
        + u8::from(!(a < 1.25))
        + u8::from(!(a < 1.75))
        + u8::from(!(a < 2.5))
        + u8::from(!(a < 3.5))
        + u8::from(!(a < 5.0));
    // Signed zero keeps rounding to +0: the sign lands only on a non-zero
    // magnitude.
    let sign = u8::from(x < 0.0) << 3;
    (mag | sign) * u8::from(mag != 0)
}

/// The E8M0 byte for one group: the smallest `b` with `6 · 2^(b-127) ≥ absmax`
/// (`quant_bf16_to_mxfp4.cu::encode_e8m0`), and `0` for a group of zeros — or
/// of NaNs, which is what the kernel's `!(absmax > 0)` spelling covers.
///
/// The `log2` is the one place this can drift from the device: CUDA's `log2f`
/// is within 1 ulp, the host's is correctly rounded, and a disagreement flips
/// the `ceil` only when `absmax / 6` sits within an ulp of a power of two.
/// The driver-side parity test is the check that it does not happen on real
/// weights.
// The negated comparison is the port: `!(absmax > 0.0)` is false for NaN
// where `absmax <= 0.0` is not, and NaN-means-scale-zero is the kernel's
// behaviour.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn encode_e8m0(absmax: f32) -> u8 {
    if !(absmax > 0.0) {
        return 0;
    }
    // The `+ 127.0` stays in `f32` so an infinite absmax saturates through the
    // int conversion instead of overflowing after it.
    let b = ((absmax / 6.0).log2().ceil() + 127.0) as i32;
    b.clamp(0, 254) as u8
}

/// `2^(sb - 127)` as the kernel's `ldexpf` builds it: an exact power of two,
/// subnormal at `sb == 0`.
fn exp2_e8m0(sb: u8) -> f32 {
    if sb == 0 {
        f32::from_bits(1 << 22)
    } else {
        f32::from_bits(u32::from(sb) << 23)
    }
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

/// One group's affine scale and zero point, by MLX's rule.
///
/// Transcribed from `mlx/backend/cpu/quantized.cpp::quantize`, because the
/// scheme is MLX's and a checkpoint this produces is meant to be
/// interchangeable with one `mlx_lm convert` produced. Three details are worth
/// naming, since none is what an independently written affine quantizer would
/// do:
///
///  * **The scale is usually negative.** It is negated unless the group's
///    minimum is the larger in magnitude, which puts code 0 on whichever end
///    dominates. Dequantization is `code * scale + zero` either way, so a
///    consumer never sees the difference -- but a producer that "fixed" the sign
///    would place the codes on the other end of the group's range.
///  * **The endpoint is snapped, not the scale.** `scale` is recomputed as
///    `edge / round(edge / scale)` so that the dominant endpoint lands exactly on
///    a code, which is what keeps the largest magnitude in the group exact.
///  * **`w_max` starts at zero**, not at negative infinity, so a group whose
///    values are all negative is quantized over the range up to zero rather
///    than up to its own largest element. Nothing about the arithmetic suggests
///    this; it is simply what MLX does.
///  * **The rounding is half AWAY FROM ZERO**, not half to even. That one
///    choice was the whole of an 8.2% disagreement with `mx.quantize` on an
///    MXFP4-derived expert bank, whose values sit on half-integers by
///    construction, and it is why every mismatch was by exactly one.
///  * **`eps` floors the scale** so a constant group -- every expert bias row
///    that is all zeros, for one -- divides by `1e-7` instead of by zero.
fn mlx_affine_group_params(values: &[f64]) -> (f32, f32) {
    const N_BINS: f32 = 15.0;
    const EPS: f32 = 1e-7;
    let mut w_min = f32::INFINITY;
    let mut w_max = 0.0f32;
    for &value in values {
        let value = value as f32;
        w_min = w_min.min(value);
        w_max = w_max.max(value);
    }
    let mask = w_min.abs() > w_max.abs();
    let mut scale = ((w_max - w_min) / N_BINS).max(EPS);
    if !mask {
        scale = -scale;
    }
    let edge = if mask { w_min } else { w_max };
    let q0 = (edge / scale).round();
    let mut bias = 0.0f32;
    if q0 != 0.0 {
        scale = edge / q0;
        bias = edge;
    }
    (scale, bias)
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

    /// Runtime quantization, compiled and then actually run.
    ///
    /// The first row visits every E2M1 codepoint from both sides of each
    /// midpoint boundary; the second forces a different E8M0 scale. An
    /// executor that mis-rounds a boundary, swaps a nibble pair, drops a sign,
    /// or applies one row's scale to the other cannot pass.
    #[test]
    fn a_bf16_tensor_is_encoded_to_mxfp4_with_its_scales() {
        use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};
        use crate::plan::CONVERT_TILE_MAP_MASK;
        use crate::types::{Axis, QuantScheme, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_encode_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Scale 2^0: the row's absmax is exactly 6.0, so every value divides
        // by one and the codepoint is read straight off the midpoint table.
        #[rustfmt::skip]
        let row0: [f32; 32] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            0.2, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
            0.4, 0.6, 1.1, 1.6, 2.2, 3.2, 4.5, 5.5,
        ];
        let mut data = Vec::new();
        for value in row0 {
            data.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        // Scale 2^1: absmax 12.0, and 12/2 = 6 encodes as the top magnitude.
        for _ in 0..32 {
            data.extend_from_slice(&bf16::from_f32(12.0).to_bits().to_le_bytes());
        }
        let header = r#"{"raw":{"dtype":"BF16","shape":[2,32],"data_offsets":[0,128]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&data);
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
                span_bytes: 128,
                shape: vec![2, 32],
                encoding: Encoding::Raw(DType::BF16),
            }],
        };
        let spec = QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src("raw").cast(Encoding::Quant(spec.clone())),
                vec![2, 32],
                Encoding::Quant(spec),
            )],
            groups: Vec::new(),
        };
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };

        let plan = crate::plan::compile(&metadata, &contract, target).unwrap();
        let storage = execute_plan(&plan, &dir).unwrap();

        #[rustfmt::skip]
        let expected_row0: [u8; 16] = [
            0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE,
            0x10, 0x32, 0x54, 0x76, 0x11, 0x32, 0x54, 0x76,
        ];
        let mut expected = expected_row0.to_vec();
        expected.extend(std::iter::repeat_n(0x77u8, 16));
        assert_eq!(storage.tensors["w"], expected);
        assert_eq!(storage.tensors["w_scale"], vec![127, 128]);
        std::fs::remove_dir_all(dir).ok();
    }

    /// The two encode primitives at their edges, against values computed by
    /// hand from the kernel's comments.
    #[test]
    fn mxfp4_encode_primitives_match_the_kernel_at_the_edges() {
        // Signed zero rounds to +0; NaN falls through every comparison to the
        // top magnitude, positive.
        assert_eq!(encode_fp4_e2m1(0.0), 0);
        assert_eq!(encode_fp4_e2m1(-0.0), 0);
        assert_eq!(encode_fp4_e2m1(f32::NAN), 0x7);
        assert_eq!(encode_fp4_e2m1(-5.0), 0xF);
        assert_eq!(encode_fp4_e2m1(f32::INFINITY), 0x7);
        assert_eq!(encode_fp4_e2m1(f32::NEG_INFINITY), 0xF);
        // Each boundary is closed on its upper side.
        assert_eq!(encode_fp4_e2m1(0.25), 1);
        assert_eq!(encode_fp4_e2m1(-1.75), 0xC);

        // b is the smallest byte with 6·2^(b-127) ≥ absmax.
        assert_eq!(encode_e8m0(0.0), 0);
        assert_eq!(encode_e8m0(f32::NAN), 0);
        assert_eq!(encode_e8m0(6.0), 127);
        assert_eq!(encode_e8m0(6.1), 128);
        assert_eq!(encode_e8m0(3.0), 126);
        assert_eq!(encode_e8m0(12.0), 128);
        assert_eq!(encode_e8m0(f32::INFINITY), 254);
        assert_eq!(encode_e8m0(f32::MIN_POSITIVE), 0);

        assert_eq!(exp2_e8m0(127), 1.0);
        assert_eq!(exp2_e8m0(128), 2.0);
        assert_eq!(exp2_e8m0(0), f32::from_bits(1 << 22));
        assert_eq!(exp2_e8m0(254), f32::from_bits(254 << 23));
    }

    /// Progress reports bytes monotonically against the plan's own total and
    /// names each tensor as it is published.
    #[test]
    fn progress_counts_bytes_and_names_finalized_tensors() {
        use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};

        let dir = std::env::temp_dir().join(format!("pie_host_progress_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let header = r#"{"raw":{"dtype":"U8","shape":[16],"data_offsets":[0,16]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&[7u8; 16]);
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
                shape: vec![16],
                encoding: Encoding::Raw(DType::U8),
            }],
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src("raw"),
                vec![16],
                Encoding::Raw(DType::U8),
            )],
            groups: Vec::new(),
        };
        let plan = crate::plan::compile(&metadata, &contract, StorageTarget::default()).unwrap();

        let mut seen = Vec::new();
        let storage = execute_plan_with_progress(&plan, &dir, &mut |progress| {
            seen.push((
                progress.read_bytes,
                progress.total_read_bytes,
                progress.finalized.map(str::to_string),
            ));
        })
        .unwrap();
        assert_eq!(storage.tensors["w"], vec![7u8; 16]);

        assert_eq!(seen.len(), plan.schedule.len());
        assert!(seen.windows(2).all(|pair| pair[0].0 <= pair[1].0));
        let last = seen.last().unwrap();
        assert_eq!(last.0, plan.memory.checkpoint_read_bytes);
        assert_eq!(last.1, plan.memory.checkpoint_read_bytes);
        assert!(
            seen.iter().any(|(.., name)| name.as_deref() == Some("w")),
            "no event named the published tensor: {seen:?}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// The parallel row driver against a plain sequential loop of the same
    /// job, at a size that engages the worker split.
    ///
    /// Determinism is the claim `encode_rows` makes — the worker count changes
    /// wall-clock and nothing else — and this is the check that the row/slice
    /// bookkeeping honours it: an off-by-one in the `split_at_mut` arithmetic
    /// shifts every subsequent row and cannot pass.
    #[test]
    fn parallel_row_encoding_is_bit_identical_to_sequential() {
        let (rows, cols) = (512usize, 2048usize); // 1M elements: the parallel path
        let job = |row: usize, buf: &mut [f32], out: &mut [u8], scale: &mut [u8]| {
            for (c, v) in buf.iter_mut().enumerate() {
                *v = ((row * 31 + c) % 97) as f32;
            }
            for (c, o) in out.iter_mut().enumerate() {
                *o = (buf[c] as u32 % 251) as u8;
            }
            scale[0] = (row % 255) as u8;
        };
        let mut out_parallel = vec![0u8; rows * cols];
        let mut scale_parallel = vec![0u8; rows];
        encode_rows(
            rows,
            cols,
            cols,
            1,
            &mut out_parallel,
            &mut scale_parallel,
            &job,
        );

        let mut out_sequential = vec![0u8; rows * cols];
        let mut scale_sequential = vec![0u8; rows];
        let mut buf = vec![0.0f32; cols];
        for row in 0..rows {
            job(
                row,
                &mut buf,
                &mut out_sequential[row * cols..(row + 1) * cols],
                &mut scale_sequential[row..row + 1],
            );
        }
        assert_eq!(out_parallel, out_sequential);
        assert_eq!(scale_parallel, scale_sequential);
    }

    /// The AVX2 group encode against the scalar reference, over inputs built
    /// to visit every lane behaviour: all magnitudes both sides of each
    /// midpoint, both signs, signed zeros, NaN, infinity, subnormals, and a
    /// spread of group absmaxes — on any machine without AVX2 this reduces to
    /// scalar-vs-scalar and passes vacuously.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_group_encode_matches_the_scalar_reference() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        // A deterministic mix: an LCG over interesting bf16-representable
        // values plus injected specials.
        let specials = [
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
        ];
        let mut state = 0x243F_6A88u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for round in 0..256 {
            let mut group = [0.0f32; 32];
            for (at, value) in group.iter_mut().enumerate() {
                let roll = next();
                *value = if roll % 11 == 0 {
                    specials[(roll / 11) as usize % specials.len()]
                } else {
                    // A finite bf16 value with a bounded exponent, signed.
                    let mantissa = (roll >> 8) & 0xFF;
                    let exponent = 118 + (roll % 21); // 2^-9 .. 2^11
                    let sign = (roll & 1) << 31;
                    f32::from_bits(sign | (exponent << 23) | (mantissa << 15))
                };
                if round == 0 && at == 0 {
                    // One all-zero-leading group exercises the dead-group arm.
                    *value = 0.0;
                }
            }
            let mut scalar_out = [0u8; 16];
            let mut avx2_out = [0u8; 16];
            let scalar_sb = encode_mxfp4_group_scalar(&group, &mut scalar_out);
            let avx2_sb = unsafe { avx2::encode_mxfp4_group(&group, &mut avx2_out) };
            assert_eq!(scalar_sb, avx2_sb, "scale byte diverged on {group:?}");
            assert_eq!(scalar_out, avx2_out, "codes diverged on {group:?}");
        }

        // The decode too, odd tail included.
        let mut bytes = Vec::new();
        for _ in 0..77 {
            bytes.extend_from_slice(&(next() as u16).to_le_bytes());
        }
        let mut scalar = vec![0.0f32; 77];
        for (le, out) in bytes.chunks_exact(2).zip(scalar.iter_mut()) {
            *out = f32::from_bits(u32::from(u16::from_le_bytes(le.try_into().unwrap())) << 16);
        }
        let mut vector = vec![0.0f32; 77];
        unsafe { avx2::decode_bf16_row(&bytes, &mut vector) };
        assert_eq!(
            scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// A GGUF Q4_0 tensor decoded to BF16 through a `Cast` contract: the
    /// `Decode` path, compiled and actually run — and refused for any target
    /// that is not the conversion one.
    ///
    /// The two blocks carry different scales and visit both nibble halves
    /// (Q4_0 splits a block low-nibbles-first, not interleaved), so a decoder
    /// that swapped halves, dropped a scale, or forgot the −8 bias cannot
    /// pass.
    #[test]
    fn a_gguf_q4_0_tensor_is_decoded_to_bf16() {
        use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};
        use crate::plan::CONVERT_TILE_MAP_MASK;
        use crate::types::{Axis, QuantScheme, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_gguf_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Block 0: scale 0.5, every byte 0x9E → low nibbles 14 (+6), high 9
        // (+1): elements 0..16 are 3.0, 16..32 are 0.5. Block 1: scale 2.0,
        // every byte 0x08 → low 8 (0), high 0 (−8): 0.0 then −16.0.
        let mut blocks = Vec::new();
        blocks.extend_from_slice(&[0x00, 0x38]); // f16 0.5
        blocks.extend_from_slice(&[0x9E; 16]);
        blocks.extend_from_slice(&[0x00, 0x40]); // f16 2.0
        blocks.extend_from_slice(&[0x08; 16]);
        std::fs::write(dir.join("model.gguf"), &blocks).unwrap();

        let spec = QuantSpec {
            scheme: QuantScheme::GgufQ4_0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(0)),
        };
        let metadata = CheckpointMetadata {
            files: vec![CheckpointFile {
                id: FileId(0),
                path: "model.gguf".to_string(),
                size_bytes: 36,
                format: crate::types::CheckpointFormat::Gguf,
            }],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "raw".to_string(),
                file_id: FileId(0),
                file_offset: 0,
                span_bytes: 36,
                shape: vec![64],
                encoding: Encoding::Quant(spec),
            }],
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src("raw").cast(Encoding::Raw(DType::BF16)),
                vec![64],
                Encoding::Raw(DType::BF16),
            )],
            groups: Vec::new(),
        };

        // Every non-conversion target refuses the plan at validation — a
        // device has no kernel for a self-scaled block.
        let refused = crate::plan::compile(&metadata, &contract, StorageTarget::default());
        assert!(refused.is_err(), "the host target accepted a Decode");

        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };
        let plan = crate::plan::compile(&metadata, &contract, target).unwrap();
        let storage = execute_plan(&plan, &dir).unwrap();

        let mut expected = Vec::new();
        for value in std::iter::repeat_n(3.0f32, 16)
            .chain(std::iter::repeat_n(0.5, 16))
            .chain(std::iter::repeat_n(0.0, 16))
            .chain(std::iter::repeat_n(-16.0, 16))
        {
            expected.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        assert_eq!(storage.tensors["w"], expected);
        std::fs::remove_dir_all(dir).ok();
    }

    /// The E4M3 conversion pair against the hardware's conversion, at every
    /// code and at the edges the SATFINITE mode defines.
    #[test]
    fn e4m3_primitives_match_the_hardware_conversion() {
        // Every non-NaN byte decodes to a value that encodes back to itself —
        // signed zeros and subnormals included.
        for byte in 0..=255u8 {
            if byte & 0x7F == 0x7F {
                continue;
            }
            assert_eq!(
                f32_to_fp8_e4m3(fp8_e4m3_to_f32(byte)),
                byte,
                "byte {byte:#04x}"
            );
        }
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0);
        assert_eq!(fp8_e4m3_to_f32(0x01), 0.001953125); // 2^-9, the smallest subnormal
        assert!(fp8_e4m3_to_f32(0x7F).is_nan());
        // SATFINITE: finite overflow and infinity clamp to ±448; NaN stays NaN.
        assert_eq!(f32_to_fp8_e4m3(1000.0), 0x7E);
        assert_eq!(f32_to_fp8_e4m3(f32::INFINITY), 0x7E);
        assert_eq!(f32_to_fp8_e4m3(f32::NEG_INFINITY), 0xFE);
        assert_eq!(f32_to_fp8_e4m3(f32::NAN), 0x7F);
        assert_eq!(f32_to_fp8_e4m3(-0.0), 0x80);
        // Round to nearest, ties to the even mantissa.
        assert_eq!(f32_to_fp8_e4m3(1.0625), 0x38); // tie 1.0 / 1.125 → 1.0
        assert_eq!(f32_to_fp8_e4m3(1.1875), 0x3A); // tie 1.125 / 1.25 → 1.25
    }

    /// Per-channel FP8: `quant_per_channel_kernel`'s row scale and cast,
    /// compiled and actually run.
    #[test]
    fn a_bf16_tensor_is_encoded_to_fp8_per_channel() {
        use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};
        use crate::plan::CONVERT_TILE_MAP_MASK;
        use crate::types::{Axis, QuantScheme, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_fp8_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Row 0's absmax is exactly the E4M3 maximum, so its factor is 1.0
        // and the codes read straight off the format; row 1 halves it, so a
        // factor applied to the wrong row cannot pass.
        let values: [[f32; 4]; 2] = [[448.0, -224.0, 1.0, 0.0], [224.0, 112.0, -56.0, 28.0]];
        let mut data = Vec::new();
        for row in values {
            for value in row {
                data.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
            }
        }
        let header = r#"{"raw":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&data);
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
                shape: vec![2, 4],
                encoding: Encoding::Raw(DType::BF16),
            }],
        };
        let spec = QuantSpec {
            scheme: QuantScheme::Fp8E4M3,
            logical_dtype: DType::BF16,
            bits_per_element: 8,
            group_size: 0,
            channel_axis: Some(Axis(0)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src("raw").cast(Encoding::Quant(spec.clone())),
                vec![2, 4],
                Encoding::Quant(spec),
            )],
            groups: Vec::new(),
        };
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };

        let plan = crate::plan::compile(&metadata, &contract, target).unwrap();
        let storage = execute_plan(&plan, &dir).unwrap();
        assert_eq!(
            storage.tensors["w"],
            vec![0x7E, 0xF6, 0x38, 0x00, 0x7E, 0x76, 0xEE, 0x66]
        );
        let mut scales = Vec::new();
        scales.extend_from_slice(&1.0f32.to_le_bytes());
        scales.extend_from_slice(&0.5f32.to_le_bytes());
        assert_eq!(storage.tensors["w_scale_inv"], scales);
        std::fs::remove_dir_all(dir).ok();
    }

    /// Per-channel INT8: same shape as FP8 with `rintf`'s half-to-even.
    #[test]
    fn a_bf16_tensor_is_encoded_to_int8_per_channel() {
        use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};
        use crate::plan::CONVERT_TILE_MAP_MASK;
        use crate::types::{Axis, QuantScheme, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_int8_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Absmax exactly 127 makes the factor 1.0, so the two half-way values
        // pin the rounding rule: 2.5 → 2 and 3.5 → 4 is ties-to-even, and
        // either a truncation or a half-up rounding fails one of them.
        let values: [f32; 4] = [127.0, 2.5, 3.5, -4.0];
        let mut data = Vec::new();
        for value in values {
            data.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        let header = r#"{"raw":{"dtype":"BF16","shape":[1,4],"data_offsets":[0,8]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&data);
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
                span_bytes: 8,
                shape: vec![1, 4],
                encoding: Encoding::Raw(DType::BF16),
            }],
        };
        let spec = QuantSpec {
            scheme: QuantScheme::Int8Symmetric,
            logical_dtype: DType::BF16,
            bits_per_element: 8,
            group_size: 0,
            channel_axis: Some(Axis(0)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src("raw").cast(Encoding::Quant(spec.clone())),
                vec![1, 4],
                Encoding::Quant(spec),
            )],
            groups: Vec::new(),
        };
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };

        let plan = crate::plan::compile(&metadata, &contract, target).unwrap();
        let storage = execute_plan(&plan, &dir).unwrap();
        assert_eq!(storage.tensors["w"], vec![127, 2, 4, 0xFC]);
        assert_eq!(storage.tensors["w_scale_inv"], 1.0f32.to_le_bytes());
        std::fs::remove_dir_all(dir).ok();
    }

    /// An FP8 block-scaled checkpoint encoded to MXFP4 in one `Cast`: the
    /// instruction names its `_scale_inv` sibling and the executor dequantizes
    /// with it before encoding, the way the device's fused path does.
    #[test]
    fn an_fp8_block_scaled_source_is_dequantized_then_encoded() {
        use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};
        use crate::plan::CONVERT_TILE_MAP_MASK;
        use crate::types::{Axis, QuantScheme, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_fp8mx_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // A [64, 64] FP8 weight of ones under a single [1, 1] block factor of
        // 2.0: every dequantized element is 2.0, every MXFP4 group absmax is
        // 2.0 → scale byte 126 (2^-1), and 2.0 / 2^-1 = 4.0 encodes as
        // magnitude 6 in both nibbles. A factor that was dropped — or applied
        // as its reciprocal — produces different bytes everywhere.
        let weight = vec![0x38u8; 64 * 64];
        let factor = 2.0f32.to_le_bytes();
        let header = r#"{"w":{"dtype":"F8_E4M3","shape":[64,64],"data_offsets":[0,4096]},"#
            .to_string()
            + r#""w_scale_inv":{"dtype":"F32","shape":[1,1],"data_offsets":[4096,4100]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&weight);
        file.extend_from_slice(&factor);
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
                    span_bytes: 4096,
                    shape: vec![64, 64],
                    encoding: Encoding::Raw(DType::F8E4M3),
                },
                RawTensor {
                    id: TensorId(1),
                    name: "w_scale_inv".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset + 4096,
                    span_bytes: 4,
                    shape: vec![1, 1],
                    encoding: Encoding::Raw(DType::F32),
                },
            ],
        };
        let spec = QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "q",
                Expr::src("w").cast(Encoding::Quant(spec.clone())),
                vec![64, 64],
                Encoding::Quant(spec),
            )],
            groups: Vec::new(),
        };
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };

        let plan = crate::plan::compile(&metadata, &contract, target).unwrap();
        assert!(
            plan.instrs.iter().any(|instr| matches!(
                instr,
                StorageInstr::TileMap {
                    kind: TileMapKind::Encode,
                    transform,
                    ..
                } if transform.metadata_source == Some(TensorId(1))
            )),
            "the block-scale tensor did not land on the instruction: {:?}",
            plan.instrs
        );
        let storage = execute_plan(&plan, &dir).unwrap();
        assert_eq!(storage.tensors["q"], vec![0x66u8; 64 * 32]);
        assert_eq!(storage.tensors["q_scale"], vec![126u8; 64 * 2]);
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
