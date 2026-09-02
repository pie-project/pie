//! Where a resident plan's persistent arena lives: a backing the caller supplies. The executor keeps every decision; the backing keeps only the verbs those decisions bottom out in.

use std::borrow::Cow;

use crate::error::Error;

/// The bytes an executed plan's persistent arena is made of. Offsets are the plan's own (`BufferDecl::persistent_offset`, `BulkExtentWrite::dest_offset`).
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

    /// Set `len` bytes at `offset` to `byte`. Its own verb rather than a `write` of a synthesized buffer, to avoid materializing gigabytes of poison just to hand it across.
    ///
    /// # Errors
    /// The range is out of bounds, or the backing could not fill it.
    fn fill(&mut self, offset: usize, len: usize, byte: u8) -> Result<(), Error>;

    /// Wait until everything this backing has queued has happened. Called once, after the last instruction; the default is correct for any host-addressable backing.
    ///
    /// # Errors
    /// The queued work faulted.
    fn finish(&mut self) -> Result<(), Error> {
        Ok(())
    }

    /// Whether this backing launches the kernels the plan names. `false` (the default) is "host mode". Read once before the first instruction.
    fn runs_named_kernels(&self) -> bool {
        false
    }

    /// Launch the kernel the plan named for one transform whose operands are already in this arena. Called only when [`runs_named_kernels`](Self::runs_named_kernels) is `true`.
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

/// One transform, addressed entirely in arena offsets: the executor resolves extents to spans once, so a backing receives operands, not the job of finding them. Carries no fact a backing does not read (e.g. no dtypes).
#[derive(Clone, Debug)]
pub struct TileMapOp<'a> {
    /// The `kernels-cuda` table symbol the plan named for this instruction.
    pub kernel: &'a str,
    /// The operand read.
    pub src: ArenaSpan,
    /// Where the result lands. May equal [`src`](Self::src) for a transform that rewrites in place, which `Scale` does.
    pub dst: ArenaSpan,
    /// `Encode`'s second output: the scales its payload cannot be read without. `None` for every kind that publishes one tensor.
    pub dst_scales: Option<ArenaSpan>,
    /// The per-group factors a blocked [`TileMapKind::Scale`] reads (`None` for a uniform constant carried by the kernel itself).
    pub factors: Option<ArenaSpan>,
    /// The primary output's declared shape as `(rows, cols)`, from the tensor's declaration, not derived from the destination extent (which can disagree on a strided destination).
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
