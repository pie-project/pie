//! The API half a backend crate supplies: its device, its buffers, its
//! frames. Everything else in this crate is generic over [`Backend`] and is
//! monomorphized per backend, so a call through these traits is static.

use crate::error::{Fault, Result};

/// Where a reservation lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Memory {
    Device,

    Host,

    Staging,

    /// Host memory the device writes and the host maps to read (WebGPU's
    /// `MAP_READ`); a backend with no such kind serves it as [`Memory::Host`].
    Readback,
}

/// A GPU API: the types a backend crate binds the shared engine to.
pub trait Backend: Sized + 'static {
    /// The device one load binds.
    type Context;

    /// One device reservation; clones share it.
    type Buffer: Buffer<Self>;
}

/// A device reservation as the shared engine moves bytes through it.
pub trait Buffer<B: Backend>: Clone + std::fmt::Debug {
    fn with(device: &B::Context, bytes: u64, kind: Memory) -> Result<Self>;

    fn zeroed(device: &B::Context, bytes: u64) -> Result<Self> {
        Self::with(device, bytes, Memory::Device)
    }

    fn host(device: &B::Context, bytes: u64) -> Result<Self> {
        Self::with(device, bytes, Memory::Host)
    }

    /// The bytes asked for.
    fn bytes(&self) -> u64;

    /// The bytes the reservation spans (at least [`Buffer::bytes`]).
    fn slab_size(&self) -> u64;

    /// A name of the reservation, equal across clones.
    fn slab_id(&self) -> u64;

    /// Whether the host maps the reservation.
    fn is_host(&self) -> bool;

    fn span(&self, offset: u64, len: u64) -> Result<()> {
        match offset.checked_add(len) {
            Some(end) if end <= self.bytes() => Ok(()),
            _ => Err(Fault::Ceiling {
                what: "bytes of a device reservation",
                need: offset.saturating_add(len),
                have: self.bytes(),
            }),
        }
    }

    fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()>;

    /// [`Buffer::write`] through a shared reference.
    fn write_shared(&self, offset: u64, bytes: &[u8]) -> Result<()>;

    fn zero_span(&mut self, offset: u64, len: u64) -> Result<()>;

    fn read(&self, offset: u64, into: &mut [u8]) -> Result<()>;

    /// Reads `(into, from, len)` jobs of `file` into the reservation.
    fn write_from_file(
        &mut self,
        file: &std::fs::File,
        jobs: &[(u64, u64, u64)],
        threads: usize,
    ) -> Result<()>;
}
