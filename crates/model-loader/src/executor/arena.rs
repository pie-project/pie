//! Where a resident plan's persistent arena actually LIVES.
//!
//! [`execute_plan_into_arena`](super::host::execute_plan_into_arena) took a
//! `&mut [u8]`, which is the right shape for a backend whose device memory is
//! addressable by the host — Metal's unified buffers are exactly that, and the
//! executor writes the laid-out weights straight into their final home.
//!
//! CUDA's is not. A discrete GPU's arena is reachable only through a copy, and
//! the two ways to bridge that without this trait are both bad: give the
//! executor a host arena and copy the whole thing across at the end, which
//! holds the model TWICE and is what a 39 GB checkpoint cannot afford; or
//! duplicate the executor inside the driver, which is the C++
//! `load_plan_executor.hpp` this crate exists to replace.
//!
//! So the arena becomes a BACKING the caller supplies. The executor keeps
//! every decision — what to read, what to transform, where each tensor lands —
//! and the backing keeps only the two verbs those decisions bottom out in.
//!
//! The executor is written to write far more than it reads: a `read` is a
//! staging copy for a device backing, so an implementation that makes it
//! expensive is not thereby making loading expensive.
//!
//! # Transforms, and why they are verbs here too
//!
//! Memory got this treatment and transforms did not, which left a backing
//! that can only receive bytes. So a `TileMap` whose input is a tensor
//! already in the arena costs a full round trip — [`ArenaBacking::read`]
//! stages it back to the host (and synchronizes), the host multiplies or
//! casts it, and [`ArenaBacking::write`] sends it across again — to compute
//! something the device it just came from has a kernel for.
//!
//! [`ArenaBacking::tile_map_caps`] and [`ArenaBacking::run_tile_map`] are the
//! same bargain as the memory verbs, for the same reason. **The executor
//! still keeps every decision**: which extents to read, which transform to
//! run, what its operands are and where the result lands. The backing is
//! told to run one, on operands the executor has already resolved to arena
//! offsets, and may decline the whole category by leaving `tile_map_caps` at
//! its default.
//!
//! Default zero is what keeps [`crate::plan::CONVERT_TILE_MAP_MASK`] honest
//! and `pie model convert` working on a machine with no GPU: a backing that
//! says nothing runs nothing, and every transform falls back to the host
//! path that has always run it. **Host mode is not a flag the executor
//! branches on — it is the backing you hand it**, and `&mut [u8]` is one.
//!
//! This is also what makes a device transform checkable: run one plan into a
//! host arena and the same plan into a device arena, and compare the bytes.
//! Nothing about the decisions differs between the two.

use std::borrow::Cow;

use crate::error::Error;
use crate::plan::{TileMapKind, TransformSpec};
use crate::types::Encoding;

/// The bytes an executed plan's persistent arena is made of.
///
/// Offsets are from the start of the arena and are the plan's own
/// (`BufferDecl::persistent_offset`, `BulkExtentWrite::dest_offset`), so an
/// implementation never has to know what it is storing.
pub trait ArenaBacking {
    /// The arena's capacity. The executor refuses a plan needing more.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read `len` bytes at `offset`.
    ///
    /// `Cow` because a host backing lends its own bytes and a device backing
    /// has to stage them; the executor copies either way at every call site,
    /// so borrowing is a saving and not a requirement.
    ///
    /// # Errors
    /// The range is out of bounds, or the backing could not produce the bytes.
    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>, Error>;

    /// # Errors
    /// The range is out of bounds, or the backing could not take the bytes.
    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error>;

    /// Set `len` bytes at `offset` to `byte`.
    ///
    /// Its own verb rather than a `write` of a synthesized buffer because the
    /// executor poisons the WHOLE arena before it starts, and materializing
    /// 39 GB of poison to hand across would defeat the point of the trait.
    ///
    /// # Errors
    /// The range is out of bounds, or the backing could not fill it.
    fn fill(&mut self, offset: usize, len: usize, byte: u8) -> Result<(), Error>;

    /// Which [`TileMapKind`]s this backing runs itself.
    ///
    /// Zero — the default — means none, and is the whole of "host mode":
    /// every transform stays on the executor's own path. A backing widens
    /// this only for kinds its [`run_tile_map`](Self::run_tile_map) actually
    /// dispatches, because the executor stops staging the operands to the
    /// host for anything named here.
    fn tile_map_caps(&self) -> u32 {
        0
    }

    /// Run one transform whose operands are already in this arena.
    ///
    /// Only called for a `kind` inside [`tile_map_caps`](Self::tile_map_caps)
    /// and only when every operand resolved to an arena span — an input the
    /// executor is holding on the host is written across first, or the host
    /// path runs it. So an implementation never has to ask where a span is.
    ///
    /// Three answers, and the middle one is why this is not a `Result<()>`:
    ///
    /// * `Ok(true)` — ran. The executor moves on.
    /// * `Ok(false)` — **declined**. The operands are a shape this backing
    ///   has no kernel for, which a per-kind capability bit cannot express: a
    ///   backing may implement `Cast` and still meet a dtype pair it has no
    ///   kernel for, or `Scale` and meet a uniform factor where its kernel
    ///   wants a per-group operand. The executor runs it on the host, exactly
    ///   as if the kind had never been claimed.
    /// * `Err` — **failed**. The kernel was dispatched and did not work.
    ///   Falling back would hide a broken kernel behind a slower answer, and
    ///   the two are not guaranteed to agree bit for bit.
    ///
    /// # Errors
    ///
    /// The transform was attempted and failed.
    fn run_tile_map(&mut self, op: &TileMapOp<'_>) -> Result<bool, Error> {
        let _ = op;
        Ok(false)
    }
}

/// A contiguous run of the arena: where one operand of a [`TileMapOp`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaSpan {
    /// Byte offset from the start of the arena.
    pub offset: usize,
    /// Length in bytes.
    pub len: usize,
}

/// One transform, addressed entirely in arena offsets.
///
/// Deliberately not the [`StorageInstr`](crate::plan::StorageInstr): the plan
/// walking that turns a `TileMap`'s extents into spans happens once, in the
/// executor, so a backing receives the operands and not the job of finding
/// them. A backing that needed a fact not carried here is a signal the fact
/// belongs in this struct, not that the backing should be handed the plan.
#[derive(Clone, Debug)]
pub struct TileMapOp<'a> {
    pub kind: TileMapKind,
    /// The operand read.
    pub src: ArenaSpan,
    /// Where the result lands. May equal [`src`](Self::src) for a transform
    /// that rewrites in place, which `Scale` does.
    pub dst: ArenaSpan,
    /// `Encode`'s SECOND output: the scales its payload cannot be read
    /// without. `None` for every kind that publishes one tensor.
    ///
    /// Its own field rather than a list, because the two are not
    /// interchangeable — a backing writing them the wrong way round produces
    /// a tensor that loads and is wrong — and because naming them is what
    /// lets a reader see that `Encode` is the kind with two.
    pub dst_scales: Option<ArenaSpan>,
    /// The per-group factors a blocked [`TileMapKind::Scale`] READS, when
    /// [`TransformSpec::scale_blocks`] is non-empty. `None` for the uniform
    /// constant in [`TransformSpec::scale_factor_bits`].
    ///
    /// Distinct from [`dst_scales`](Self::dst_scales) in direction: this is
    /// an input the transform multiplies by, that is an output it computes.
    pub factors: Option<ArenaSpan>,
    /// The primary output's declared shape as `(rows, cols)`, when the plan
    /// states a 2-D one.
    ///
    /// From the TENSOR's declaration, which is the same place the host path
    /// reads it (`host::encode_bytes`). Deriving it from the destination
    /// extent instead would be a second source of truth for one number, and
    /// the two can disagree on a strided or offset destination.
    pub shape: Option<(u32, u32)>,
    /// What the operands ARE, not merely how wide they are.
    ///
    /// [`Encoding`] rather than [`DType`](crate::types::DType) because the
    /// difference is the whole question for the kinds that have two: an
    /// `Encode` destination is quantized, and reporting the byte width of
    /// its storage would be a value every backing has to know to disbelieve.
    pub src_encoding: Encoding,
    pub dst_encoding: Encoding,
    /// Everything the plan states about the transform, unabridged — the
    /// scale factor's bits, its blocking, the scheme on each side.
    pub transform: &'a TransformSpec,
}

fn out_of_bounds(what: &str) -> Error {
    Error::Contract(format!("arena {what} is out of bounds"))
}

impl ArenaBacking for &mut [u8] {
    fn len(&self) -> usize {
        <[u8]>::len(self)
    }

    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>, Error> {
        let end = offset.checked_add(len).ok_or_else(|| out_of_bounds("read"))?;
        self.get(offset..end)
            .map(Cow::Borrowed)
            .ok_or_else(|| out_of_bounds("read"))
    }

    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| out_of_bounds("write"))?;
        self.get_mut(offset..end)
            .ok_or_else(|| out_of_bounds("write"))?
            .copy_from_slice(bytes);
        Ok(())
    }

    fn fill(&mut self, offset: usize, len: usize, byte: u8) -> Result<(), Error> {
        let end = offset.checked_add(len).ok_or_else(|| out_of_bounds("fill"))?;
        self.get_mut(offset..end)
            .ok_or_else(|| out_of_bounds("fill"))?
            .fill(byte);
        Ok(())
    }
}
