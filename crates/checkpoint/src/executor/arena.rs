//! Where a resident plan's persistent arena actually LIVES.
//!
//! [`execute_plan_into_arena`](super::host::execute_plan_into_arena) took
//! a `&mut [u8]`, which fits a host-addressable backend but not a
//! discrete GPU, reachable only through a copy. So the arena becomes a
//! BACKING the caller supplies: the executor keeps every decision, and
//! the backing keeps only the verbs those decisions bottom out in.

use std::borrow::Cow;

use crate::error::Error;

/// The bytes an executed plan's persistent arena is made of. Offsets
/// are the plan's own (`BufferDecl::persistent_offset`,
/// `BulkExtentWrite::dest_offset`).
pub trait ArenaBacking {
    /// The arena's capacity. The executor refuses a plan needing more.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read `len` bytes at `offset` (`Cow`: a host backing lends its own bytes, a device backing must stage them).
    ///
    /// # Errors
    /// The range is out of bounds, or the backing could not produce the bytes.
    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>, Error>;

    /// # Errors
    /// The range is out of bounds, or the backing could not take the bytes.
    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error>;

    /// Set `len` bytes at `offset` to `byte`. Its own verb rather than a `write` of a synthesized buffer: materializing 39 GB of poison to hand across would defeat the point of the trait.
    ///
    /// # Errors
    /// The range is out of bounds, or the backing could not fill it.
    fn fill(&mut self, offset: usize, len: usize, byte: u8) -> Result<(), Error>;

    /// Wait until everything this backing has queued has happened. The executor calls this once, after the last instruction; the default (`Ok(())`) is correct for every host-addressable backing, and only one that overlaps its transfers (like CUDA's) overrides it.
    ///
    /// # Errors
    /// The queued work faulted.
    fn finish(&mut self) -> Result<(), Error> {
        Ok(())
    }

    /// Whether this backing launches the kernels the plan names. `false` -- the default -- is the whole of "host mode", keeping `pie model import` working on a machine with no GPU. One bit, read ONCE before the first instruction: the plan already names a kernel row per instruction, so this only states whether a device was handed over at all.
    fn runs_named_kernels(&self) -> bool {
        false
    }

    /// Launch the kernel the plan named for one transform whose operands are already in this arena. Called only when [`runs_named_kernels`](Self::runs_named_kernels) is `true` and every operand resolved to an arena span.
    ///
    /// **There is no decline.** The compiler already decided this backing can run the named row; failing to run it is a compiler that named the wrong one, not a case to fall back to the host for.
    ///
    /// # Errors
    /// The kernel has no launcher, the operands contradict the named row, or the dispatch failed.
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

/// One transform, addressed entirely in arena offsets. Deliberately not the [`StorageInstr`](crate::plan::StorageInstr): the executor resolves extents to spans once, so a backing receives operands, not the job of finding them.
///
/// **A fact no backing READS does not belong here** -- a struct carrying operand dtypes invites a backing to match on them, which is the compiler's job now.
#[derive(Clone, Debug)]
pub struct TileMapOp<'a> {
    /// The `kernels-cuda` table symbol the PLAN named for this instruction. A backing looks it up and launches it; it is not given what it would need to disagree.
    pub kernel: &'a str,
    /// The operand read.
    pub src: ArenaSpan,
    /// Where the result lands. May equal [`src`](Self::src) for a transform
    /// that rewrites in place, which `Scale` does.
    pub dst: ArenaSpan,
    /// `Encode`'s SECOND output: the scales its payload cannot be read without. `None` for every kind that publishes one tensor. Its own field, not a list, since writing them swapped would load a wrong tensor.
    pub dst_scales: Option<ArenaSpan>,
    /// The per-group factors a blocked [`TileMapKind::Scale`] READS (`None` for a uniform constant carried by the kernel itself). Distinct from [`dst_scales`](Self::dst_scales): this is an input, that an output.
    pub factors: Option<ArenaSpan>,
    /// The primary output's declared shape as `(rows, cols)`, from the
    /// TENSOR's declaration -- not derived from the destination extent,
    /// which can disagree on a strided destination.
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
