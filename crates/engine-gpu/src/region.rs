//! A run of the activation span's segment that a fire-class reservation
//! takes in place of a buffer of its own (ONE-POOL step 2): the span's
//! buffer, the byte its run starts at, and its bytes. The span lays its
//! regions at its top, above every rung's carve, so no rung lends them to
//! state and none moves once laid ([`crate::arena::Span::with_regions`]).
//!
//! Every place that held such a reservation whole takes an offset already
//! (a binding, a copy, a host write); a region's is its base plus that.

use crate::backend::{Backend, Buffer as _};
use crate::books::Owner;
use crate::error::{Fault, Result};
use crate::handles::Handles;

pub struct Region<B: Backend> {
    buffer: B::Buffer,
    base: u64,
    len: u64,
    owner: Owner,
}

impl<B: Backend> Clone for Region<B> {
    fn clone(&self) -> Self {
        Region {
            buffer: self.buffer.clone(),
            base: self.base,
            len: self.len,
            owner: self.owner,
        }
    }
}

impl<B: Backend> std::fmt::Debug for Region<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Region")
            .field("owner", &self.owner.name)
            .field("base", &self.base)
            .field("len", &self.len)
            .finish()
    }
}

impl<B: Backend> Region<B> {
    /// `len` bytes of `buffer` from `base`, `owner`'s; refused past the
    /// buffer's end.
    pub fn new(buffer: B::Buffer, base: u64, len: u64, owner: Owner) -> Result<Region<B>> {
        buffer.span(base, len)?;
        Ok(Region {
            buffer,
            base,
            len,
            owner,
        })
    }

    /// The span's buffer the region lies in.
    #[must_use]
    pub fn buffer(&self) -> &B::Buffer {
        &self.buffer
    }

    /// The byte of [`Region::buffer`] the region starts at.
    #[must_use]
    pub fn base(&self) -> u64 {
        self.base
    }

    /// The region's bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.len
    }

    #[must_use]
    pub fn owner(&self) -> Owner {
        self.owner
    }

    /// The byte of the span's buffer the region's byte `offset` is, for
    /// `len` bytes from it; refused past the region's end, never into
    /// what lies beside it.
    pub fn at(&self, offset: u64, len: u64) -> Result<u64> {
        match offset.checked_add(len) {
            Some(end) if end <= self.len => Ok(self.base + offset),
            _ => Err(Fault::Ceiling {
                what: "bytes of a region of the activation span",
                need: offset.saturating_add(len),
                have: self.len,
            }),
        }
    }

    /// A binding of `len` bytes of the region from `offset`.
    pub fn bind(&self, handles: &Handles<B>, offset: u64, len: u64) -> Result<u32> {
        handles.bind(&self.buffer, self.at(offset, len)?, len)
    }

    /// Writes `bytes` at the region's `offset` through the queue.
    pub fn write_shared(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        let at = self.at(offset, bytes.len() as u64)?;
        self.buffer.write_shared(at, bytes)
    }

    /// `into.len()` bytes from the region's `offset`, read on the host.
    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        let at = self.at(offset, into.len() as u64)?;
        self.buffer.read(at, into)
    }
}
