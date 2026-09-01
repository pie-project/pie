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
    /// **The host mapping this reservation is a WINDOW ONTO**, when it is
    /// one — `None` for every buffer Metal allocated itself.
    ///
    /// A zero-copy reservation ([`Buffer::mapped`]) is minted over pages
    /// Metal does not own, so the mapping must outlive the `MTLBuffer` or
    /// the next kernel to read it faults on the device. The `Arc` is that
    /// guarantee, and the FIELD ORDER is the other half of it: Rust drops
    /// fields in declaration order, so `slab` — the retained buffer — is
    /// released here before `keep` unmaps, on the same statement. A clone
    /// extends both, which is the same sentence the type's doc already makes
    /// about the retain.
    ///
    /// It is also the read-only flag. A mapped reservation is `PROT_READ`,
    /// so `write` and `zero_span` on one would fault the process rather than
    /// return; `Some` is what makes them refuse by name instead.
    keep: Option<std::sync::Arc<crate::mapping::Mapping>>,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("bytes", &self.bytes)
            .field("mapped", &self.keep.is_some())
            .finish()
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
                    keep: None,
                });
            }
            let slab = device.reserve(bytes)?;
            let mut buffer = Buffer {
                slab,
                bytes,
                keep: None,
            };
            buffer.zero_span(0, bytes)?;
            Ok(buffer)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, bytes);
            Err(Fault::Deviceless)
        }
    }

    /// **Serve an artifact off its own mapped pages** — the warm-read
    /// primitive (M-1), and the one reservation in this shell that costs no
    /// copy at all.
    ///
    /// The reservation's bytes ARE the mapping's bytes: nothing is read,
    /// nothing is allocated, and the file's pages fault straight into the
    /// frames the kernel will read. What that removes is the DUPLICATE a
    /// full-residency load otherwise pays — the transient host buffer and
    /// the `memcpy` out of it, and with them a boot whose peak is twice the
    /// model.
    ///
    /// **READ [`crate::mapping`]'S HEADER BEFORE REACHING FOR THIS.** A
    /// `StorageModeShared` page WIRES the moment the GPU touches it, and the
    /// measurement says so in numbers: +4.03 GiB of global `Pages wired
    /// down` and free memory down to 0.066 GiB across one kernel's read of a
    /// mapped 4 GiB span, with the pager evicting nothing
    /// (`.wiki/alto/streaming.md`, M1 Max, 2026-08-31). So this bounds NO
    /// memory. It is for a model that already fits, where wired-equals-the-
    /// model is the price of serving it; it must never carry the streamed or
    /// oversized path, whose source is [`crate::host_source`] and is read by
    /// the CPU only, for exactly this reason.
    ///
    /// The reservation reports the artifact's TRUE length, not the
    /// page-rounded span Metal was told — see [`Mapping::span`] — so every
    /// bounds check, every handle minted over it and every kernel that reads
    /// through it are held to the file's own size.
    ///
    /// The `Arc` is held for the reservation's whole life and every clone of
    /// it, so a buffer whose mapping has been unmapped cannot be spelled.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] past `maxBufferLength`, [`Fault::Mapped`] when the
    /// device declined the wrap, [`Fault::Deviceless`] off Apple.
    ///
    /// [`Mapping::span`]: crate::mapping::Mapping::span
    pub fn mapped(
        device: &super::Context,
        map: std::sync::Arc<crate::mapping::Mapping>,
    ) -> Result<Buffer> {
        #[cfg(target_vendor = "apple")]
        {
            let bytes = map.len();
            // SAFETY: `Mapping` is a live `mmap` of exactly `span()` bytes
            // from a page-aligned base — the two alignment rules
            // `newBufferWithBytesNoCopy` states and does not check — and the
            // `Arc` moved into `keep` below is what keeps it live for longer
            // than the buffer this mints, since `slab` is declared first and
            // so is released first.
            let slab = unsafe { device.no_copy(map.base(), map.span()) }.map_err(|why| {
                Fault::Mapped {
                    step: "bind",
                    what: map.path().display().to_string(),
                    why: why.to_string(),
                }
            })?;
            Ok(Buffer {
                slab,
                bytes,
                keep: Some(map),
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (device, map);
            Err(Fault::Deviceless)
        }
    }

    /// Whether this reservation is a window onto a host mapping rather than
    /// bytes Metal allocated — which is also whether it is read-only.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        self.keep.is_some()
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

    /// Where `offset` lands in the GPU's own address space, as a number a
    /// kernel can dereference.
    ///
    /// **THE ONE THING A GROUPED LAUNCH NEEDS THAT A SINGLE-LANE ONE DOES
    /// NOT.** The M2 fused kernel takes every rectangle and every channel
    /// cell as `setBuffer:offset:atIndex:`, so a cell IS a binding and this
    /// number never has to exist; the M3 grouped kernel takes ONE lane table
    /// and dereferences the `ulong`s inside it (`lane.commit_slot`,
    /// `lane.logits_base`, `slot.committed_cell`), which is what lifts the
    /// twelve-channel argument-slot ceiling. So the address is the ABI, and
    /// it is published here rather than recomputed by the caller — the same
    /// argument [`slab_address`] makes for the argument-buffer path.
    ///
    /// **THE CALLER OWES RESIDENCY.** An address handed to a kernel is not a
    /// binding, so Metal does not learn the reservation is used: every buffer
    /// an address escapes to must be declared on the encoder with
    /// `useResource:usage:` or the page it names may not be resident when the
    /// kernel reads it. `program::launch` declares them; `icb.rs` and
    /// `rebind.rs` are the older precedent for the same bookkeeping.
    ///
    /// Answers `None` for an offset outside the reservation, because an
    /// address past the end is the one thing a bounds check can still catch
    /// before the device has it.
    #[must_use]
    pub fn address_at(&self, offset: u64) -> Option<u64> {
        self.span(offset, 0).ok()?;
        #[cfg(target_vendor = "apple")]
        {
            slab_address(&self.slab).checked_add(offset)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            None
        }
    }

    /// The one refusal a mapped reservation owes, spelled once for the two
    /// methods that would otherwise write through `PROT_READ` pages and take
    /// the process down with a fault instead of an answer.
    ///
    /// # Errors
    ///
    /// [`Fault::Mapped`] naming the method, for a reservation minted by
    /// [`Buffer::mapped`].
    fn writable(&self, step: &'static str) -> Result<()> {
        match &self.keep {
            None => Ok(()),
            Some(map) => Err(Fault::Mapped {
                step,
                what: map.path().display().to_string(),
                why: "a reservation served from an artifact's own mapped pages is read-only"
                    .into(),
            }),
        }
    }

    /// Copy `bytes` in at `offset`.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] when the span leaves the reservation,
    /// [`Fault::Mapped`] on a zero-copy reservation, which is read-only,
    /// [`Fault::Deviceless`] off Apple.
    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.writable("write")?;
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
    /// [`Fault::Mapped`] on a zero-copy reservation, which is read-only,
    /// [`Fault::Deviceless`] off Apple.
    pub fn zero_span(&mut self, offset: u64, len: u64) -> Result<()> {
        self.writable("zero_span")?;
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
