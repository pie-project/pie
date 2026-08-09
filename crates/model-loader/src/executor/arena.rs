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
//! # When the arena is finished
//!
//! The verbs above say what an arena is MADE OF. They do not say when it is
//! done, and for a backing that overlaps its transfers those are different
//! moments: `CudaArena::write` returns while its copy is still crossing, so
//! the arena holds a partly written model after the last instruction retires.
//!
//! That was a `pub fn finish` the caller invoked, which is the one thing this
//! trait exists to abolish — three backings, and only the caller of one of
//! them may skip it, so a caller had to know which it was holding. It is
//! [`ArenaBacking::finish`] now, defaulted to `Ok(())`, and the executor
//! calls it. The default being right for every backing except the one that
//! needs it is the argument that it belongs here: an obligation only one
//! implementation carries is exactly the obligation a caller gets wrong.
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
//! [`ArenaBacking::runs_named_kernels`] and [`ArenaBacking::run_tile_map`]
//! are the same bargain as the memory verbs, for the same reason. **The
//! executor still keeps every decision**: which extents to read, which
//! transform to run, what its operands are and where the result lands. And
//! the plan keeps the one decision above those — `cuda_kernel` named the row,
//! per instruction, with the tensor's name in hand — so the backing is handed
//! a symbol and a set of arena offsets, and looks the symbol up.
//!
//! Declining `false` is what keeps [`crate::plan::CONVERT_TILE_MAP_MASK`]
//! honest and `pie model convert` working on a machine with no GPU: a backing
//! that says nothing runs nothing, and every transform falls back to the host
//! path that has always run it. **Host mode is not a flag the executor
//! branches on — it is the backing you hand it**, and `&mut [u8]` is one.
//!
//! This is also what makes a device transform checkable: run one plan into a
//! host arena and the same plan into a device arena, and compare the bytes.
//! Nothing about the decisions differs between the two.

use std::borrow::Cow;

use crate::error::Error;

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

    /// Wait until everything this backing has queued has happened.
    ///
    /// The executor calls this once, after the last instruction, so a caller
    /// never has to. That is the point: without it, whether a load is over
    /// when `run` returns depends on which backing you are holding — no for
    /// `CudaArena`, whose writes are deliberately left in flight, yes for
    /// `&mut [u8]` and for a unified-memory one, which have nothing to wait
    /// for. A caller cannot write the same lines for all three unless the
    /// obligation is on this side of the seam.
    ///
    /// The default is correct for every backing whose `write` has already
    /// happened by the time it returns, which is every host-addressable one.
    /// A backing that overlaps its transfers overrides it, and that is
    /// exactly the backing that must not be trusted to be finished.
    ///
    /// # Errors
    /// The queued work faulted. Reported here rather than at the write that
    /// enqueued it, which had not run yet.
    fn finish(&mut self) -> Result<(), Error> {
        Ok(())
    }

    /// Whether this backing launches the kernels the plan names.
    ///
    /// `false` — the default — is the whole of "host mode": every transform
    /// stays on the executor's own path, and [`run_tile_map`](Self::run_tile_map)
    /// is never called. It is what keeps [`crate::plan::CONVERT_TILE_MAP_MASK`]
    /// honest and `pie model convert` working on a machine with no GPU.
    ///
    /// One bit rather than the per-[`TileMapKind`] mask this was. The mask
    /// answered "which KINDS does this device do", which is a question the
    /// plan already answers per INSTRUCTION and answers better: `cuda_kernel`
    /// names a row for the instructions a device can run and names nothing
    /// for the rest, so a kind-shaped claim could only ever be wider than the
    /// truth — `CudaArena` returned `CAST | SCALE | ENCODE` unconditionally,
    /// having no idea which rows the plan had named. What is left is the one
    /// thing the plan cannot know: whether the caller handed over a device at
    /// all.
    ///
    /// Read ONCE, before the first instruction. A backing whose answer
    /// changed mid-plan would leave half a load on each path, and this is a
    /// property of the backing rather than of a moment.
    fn runs_named_kernels(&self) -> bool {
        false
    }

    /// Launch the kernel the plan named for one transform whose operands are
    /// already in this arena.
    ///
    /// Called only when [`runs_named_kernels`](Self::runs_named_kernels) is
    /// `true`, the plan named a kernel for this instruction, and every operand
    /// resolved to an arena span — an input the executor is holding on the
    /// host is written across first, or the host path runs it. So an
    /// implementation never has to ask where a span is, or whether there is a
    /// row to launch.
    ///
    /// **There is no decline.** This used to answer `Ok(false)` for operands a
    /// backing had no kernel for, and the answer was correct and
    /// unobservable: the load finished, the bytes were right, and the
    /// transform had quietly run on the host at a fraction of the speed. Those
    /// rules are in the compiler now, where the tensor's name is still in
    /// hand, so an op that arrives here is one the plan already decided this
    /// backing can run. Failing to run it is a compiler that named the wrong
    /// row, and hiding that behind a slower answer is how it would stay
    /// hidden.
    ///
    /// # Errors
    ///
    /// The kernel is one this build has no launcher for, the operands
    /// contradict the row the plan named, or the dispatch failed.
    fn run_tile_map(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        Err(Error::Contract(format!(
            "this arena backing was offered the kernel `{}` and has no \
             launcher for anything: `runs_named_kernels` said true",
            op.kernel
        )))
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
///
/// **And the converse, which is the one that matters now that the plan names
/// the kernel: a fact no backing READS does not belong here.** This carried a
/// [`TileMapKind`](crate::plan::TileMapKind), a `&TransformSpec` and the
/// [`Encoding`](crate::types::Encoding) of each operand. Nothing read the
/// encodings — the executor cloned one per instruction to fill them — and the
/// kind reached one error message. They are the operands' dtypes sitting on
/// the far side of a seam whose whole purpose is that no one over there
/// decides anything from them, and a struct carrying dtypes is a standing
/// invitation to match on them. The compiler does that match; that is what
/// `kernel` is the result of.
#[derive(Clone, Debug)]
pub struct TileMapOp<'a> {
    /// The `kernels-cuda` table symbol the PLAN named for this instruction —
    /// `plan::passes::tile::cuda_kernel` chose it, with the tensor's name and
    /// its dtypes still in hand.
    ///
    /// A backing looks it up and launches it. It is not asked to agree, and
    /// it is not given what it would need to disagree: see the type's doc.
    pub kernel: &'a str,
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
    /// the plan states a blocking. `None` for a uniform constant, which the
    /// kernel the plan named carries rather than reads.
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
}

fn out_of_bounds(what: &str) -> Error {
    Error::Contract(format!("arena {what} is out of bounds"))
}

impl ArenaBacking for &mut [u8] {
    fn len(&self) -> usize {
        <[u8]>::len(self)
    }

    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>, Error> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| out_of_bounds("read"))?;
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
        let end = offset
            .checked_add(len)
            .ok_or_else(|| out_of_bounds("fill"))?;
        self.get_mut(offset..end)
            .ok_or_else(|| out_of_bounds("fill"))?
            .fill(byte);
        Ok(())
    }
}
