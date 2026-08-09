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
