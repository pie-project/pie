//! Device bytes: one `MTLBuffer` per reservation, and the four things the
//! shell does to one.
//!
//! **STORAGE IS SHARED, AND ON THIS PLATFORM THAT IS NOT A COMPROMISE.**
//! Apple silicon has one physical memory pool behind the CPU and the GPU
//! (`hasUnifiedMemory`, asserted at bind), so a `StorageModeShared` buffer
//! IS the device buffer and IS the host pointer — `write` is a `memcpy`,
//! `read` is a `memcpy`, `zero_span` is a `memset`, and none of them is a
//! transfer. The CUDA sibling's split between `write` (synchronous) and
//! `stage` (asynchronous, on the fire's stream) has no counterpart here and
//! is deliberately not spelled: there is no copy to be asynchronous about.
//! What the fire path must still respect is ORDER — a host write must
//! happen before the command buffer that reads it is committed — and that
//! is the call order in `serve`, not a flag on a buffer.
//!
//! The one cost of the choice is that a Shared buffer is not in the GPU's
//! private, compressed heap; on a discrete-GPU Mac the right answer would be
//! `StorageModePrivate` plus a blit staging path. This shell asserts unified
//! memory rather than carrying a second write path for a machine it has not
//! measured.

use crate::error::{Fault, Result};

#[cfg(target_vendor = "apple")]
use objc2::rc::Retained;
#[cfg(target_vendor = "apple")]
use objc2::runtime::ProtocolObject;
#[cfg(target_vendor = "apple")]
use objc2_metal::MTLBuffer;

/// The retained `MTLBuffer` on Apple, and nothing at all anywhere else.
#[cfg(target_vendor = "apple")]
pub(crate) type Slab = Retained<ProtocolObject<dyn MTLBuffer>>;
/// The retained `MTLBuffer` on Apple, and nothing at all anywhere else.
#[cfg(not(target_vendor = "apple"))]
pub(crate) type Slab = ();

/// The identity of one reservation, as a number two recordings can compare.
///
/// A [`Recording`](crate::record::Recording) has to say whether two walks
/// bound the SAME reservation at two offsets or two different reservations,
/// and the only thing that answers that off the retained object is its
/// address. It is never dereferenced and never used as an offset — equality
/// is the whole contract, and the retain a `Slab` carries is what makes the
/// address stable for the life of the load.
#[cfg(target_vendor = "apple")]
pub(crate) fn slab_id(slab: &Slab) -> u64 {
    let object: &objc2::runtime::ProtocolObject<dyn MTLBuffer> = slab;
    std::ptr::from_ref(object).cast::<u8>() as u64
}

/// The identity of one reservation — off Apple there are no reservations, so
/// every one of them is the same one.
#[cfg(not(target_vendor = "apple"))]
pub(crate) fn slab_id(slab: &Slab) -> u64 {
    let _ = slab;
    0
}

/// One reservation's address in the GPU's own address space.
///
/// **THE ONE NUMBER AN ARGUMENT BUFFER OF POINTERS IS MADE OF.** A tier-2
/// argument buffer encodes a `device T*` member as exactly this 64-bit value,
/// so a shell filling one writes the address rather than calling an encoder —
/// which is what lets `icb::rebind` hold 45 reservations through one binding
/// instead of 45.
#[cfg(target_vendor = "apple")]
pub(crate) fn slab_address(slab: &Slab) -> u64 {
    slab.gpuAddress()
}

/// One device reservation: a buffer and its length.
///
/// Cloning is a RETAIN, not a copy — the same bytes under a second owner,
/// which is what lets [`Handles`](super::Handles) hold a row per carved view
/// without the shell threading a lifetime through every table.
#[derive(Clone)]
pub struct Buffer {
    slab: Slab,
    bytes: u64,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer").field("bytes", &self.bytes).finish()
    }
}

// SAFETY: `MTLBuffer` is documented thread-safe for the operations this
// shell performs on one — retain/release, `contents`, and binding into an
// encoder. The shell itself is single-threaded per load (one lane thread
// owns a `Shell` for the life of the process, the same rule the CUDA
// sibling's `bind_thread` states); what `Send` buys is the MOVE from the
// thread that booted the engine onto the thread that will fire it.
unsafe impl Send for Buffer {}

impl Buffer {
    /// Reserve `bytes` and zero them.
    ///
    /// A zero-length request allocates nothing and answers a buffer no
    /// handle can be minted from — Metal refuses a zero-length buffer, and a
    /// plan with an empty pool row is a legal plan.
    ///
    /// # Errors
    ///
    /// [`Fault::Deviceless`] off Apple, [`Fault::Device`] when Metal
    /// declined the length.
    pub fn zeroed(device: &super::Context, bytes: u64) -> Result<Buffer> {
        #[cfg(target_vendor = "apple")]
        {
            if bytes == 0 {
                return Ok(Buffer {
                    slab: device.empty(),
                    bytes: 0,
                });
            }
            let slab = device.reserve(bytes)?;
            let mut buffer = Buffer { slab, bytes };
            buffer.zero_span(0, bytes)?;
            Ok(buffer)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, bytes);
            Err(Fault::Deviceless)
        }
    }

    /// How many bytes this reservation holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The retained buffer, for the encoder and for handle minting.
    pub(crate) fn slab(&self) -> &Slab {
        &self.slab
    }

    /// The `MTLBuffer` itself, for a caller that binds it directly rather
    /// than through a handle.
    ///
    /// The guest-program plane is the one such caller: an emitted ETA
    /// kernel names its buffers by `[[buffer(n)]]` index rather than through
    /// a `kernels_metal::Tensor`, so there is no handle to resolve and the
    /// binding is this reservation at an offset the plane computed.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn raw(&self) -> &objc2::runtime::ProtocolObject<dyn MTLBuffer> {
        &self.slab
    }

    /// Copy `bytes` in at `offset`.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] when the span leaves the reservation,
    /// [`Fault::Deviceless`] off Apple.
    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.span(offset, bytes.len() as u64)?;
        #[cfg(target_vendor = "apple")]
        {
            // SAFETY: `contents` on a Shared buffer is a live host mapping
            // of the whole reservation, and `span` has just proved
            // `[offset, offset + len)` lies inside it. The source is a live
            // slice and the two cannot overlap — one is host memory the
            // caller owns, the other is this buffer's mapping.
            unsafe {
                let base = self.slab.contents().as_ptr().cast::<u8>();
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    base.add(usize::try_from(offset).expect("an offset inside a live mapping")),
                    bytes.len(),
                );
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = offset;
            Err(Fault::Deviceless)
        }
    }

    /// Zero `len` bytes at `offset`.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] when the span leaves the reservation,
    /// [`Fault::Deviceless`] off Apple.
    pub fn zero_span(&mut self, offset: u64, len: u64) -> Result<()> {
        self.span(offset, len)?;
        #[cfg(target_vendor = "apple")]
        {
            if len == 0 {
                return Ok(());
            }
            // SAFETY: as `write` — a live mapping and a span proved inside it.
            unsafe {
                let base = self.slab.contents().as_ptr().cast::<u8>();
                std::ptr::write_bytes(
                    base.add(usize::try_from(offset).expect("an offset inside a live mapping")),
                    0,
                    usize::try_from(len).expect("a length inside a live mapping"),
                );
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = offset;
            Err(Fault::Deviceless)
        }
    }

    /// Copy `into.len()` bytes out from `offset`.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] when the span leaves the reservation,
    /// [`Fault::Deviceless`] off Apple.
    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.span(offset, into.len() as u64)?;
        #[cfg(target_vendor = "apple")]
        {
            // SAFETY: as `write`, with the direction reversed.
            unsafe {
                let base = self.slab.contents().as_ptr().cast::<u8>();
                std::ptr::copy_nonoverlapping(
                    base.add(usize::try_from(offset).expect("an offset inside a live mapping")),
                    into.as_mut_ptr(),
                    into.len(),
                );
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = offset;
            Err(Fault::Deviceless)
        }
    }

    /// The one bounds check every method above routes through.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] naming the span and the reservation.
    pub fn span(&self, offset: u64, len: u64) -> Result<()> {
        let end = offset.checked_add(len).ok_or(Fault::Ceiling {
            what: "bytes of a device reservation",
            need: u64::MAX,
            have: self.bytes,
        })?;
        if end > self.bytes {
            return Err(Fault::Ceiling {
                what: "bytes of a device reservation",
                need: end,
                have: self.bytes,
            });
        }
        Ok(())
    }
}
