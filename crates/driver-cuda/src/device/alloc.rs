//! One owned device allocation, and the two transfers a shell needs.
//!
//! **NOT AN ALLOCATOR.** A shell makes a handful of allocations in its whole
//! life — the arena, the weight store, one pool per cache space, the resident
//! fire inputs, the plan workspace — and every one of them is sized once from
//! a budget and lives until the model is unloaded. What varies per fire is
//! which OFFSETS a kernel is pointed at, and that arithmetic belongs to the
//! compiler's carve (`ArenaMap`), not to a heap. So this is `cudaMalloc`,
//! `cudaFree`, and the memcpys, with a length on the front of each.
//!
//! **Bounds are checked here and nowhere else.** A device pointer is a `u64`
//! by the time it reaches [`Tensor`](kernels_cuda::Tensor), and past that
//! point an off-by-one does not fault — it writes into a neighbour's
//! rectangle and the model produces slightly wrong numbers forever. Every
//! offset therefore meets its length at this door.

use crate::error::{Fault, Result};

/// A device allocation of a stated size.
#[derive(Debug)]
pub struct Buffer {
    ptr: u64,
    bytes: usize,
}

impl Buffer {
    /// Allocate `bytes` and zero them.
    ///
    /// ZEROED, NOT FRESH. `cudaMalloc` hands back whatever the last tenant
    /// left, and the places this crate allocates are exactly the places where
    /// unwritten bytes are read as numbers: a recurrent state slab at slot
    /// open, an arena rectangle a kernel writes only part of, a page table
    /// past this fire's lanes. The loader makes the same call for the same
    /// reason and says so.
    ///
    /// A zero-byte request is a real answer — a plan with no recurrent
    /// caches wants no recurrent pool — and comes back as a null handle no
    /// launch is pointed at.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn zeroed(bytes: usize) -> Result<Buffer> {
        if bytes == 0 {
            return Ok(Buffer { ptr: 0, bytes: 0 });
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            let mut base: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `base` is a live local; the allocation is this
            // buffer's, freed exactly once in `Drop`.
            unsafe {
                crate::device::ctx::check("cudaMalloc", rt::cudaMalloc(&raw mut base, bytes))?;
                crate::device::ctx::check("cudaMemset", rt::cudaMemset(base, 0, bytes))?;
            }
            Ok(Buffer {
                ptr: base as u64,
                bytes,
            })
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// The base address.
    #[must_use]
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// How many bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The address `offset` bytes in, checked against the length.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for an offset past the end.
    pub fn at(&self, offset: u64) -> Result<u64> {
        if offset > self.bytes as u64 {
            return Err(Fault::Ceiling {
                what: "bytes into a device buffer",
                need: offset,
                have: self.bytes as u64,
            });
        }
        Ok(self.ptr + offset)
    }

    /// Copy `bytes` in at `offset`, and wait.
    ///
    /// The load path's transfer: 260 weight planes, once, in front of a fire
    /// that has not started. Synchronous on purpose — there is nothing to
    /// overlap it with, and a load that returns before its bytes have landed
    /// is a first fire that reads poison.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a write past the end, [`Fault::Device`] for the
    /// copy.
    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.span(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span was just checked against this allocation, and
            // `bytes` is a live host slice for the duration of a synchronous
            // copy.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemcpy",
                    rt::cudaMemcpy(
                        (self.ptr + offset) as *mut core::ffi::c_void,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                        rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    ),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// Zero `len` bytes at `offset`, and wait.
    ///
    /// The same `cudaMemset` [`Buffer::zeroed`] makes at allocation, aimed at
    /// one span rather than the whole buffer. It exists because a recurrent
    /// slot has to be re-zeroed every time a fresh sequence takes it
    /// (`Pools::clear`), and staging ten megabytes of host zeros across PCIe
    /// to say "nothing" is the wrong shape for something on the fire path.
    ///
    /// Synchronous, like [`Buffer::write`] and for the same reason: the
    /// caller is about to launch kernels that read these bytes.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a span past the end, [`Fault::Device`] for the
    /// fill.
    pub fn zero_span(&mut self, offset: u64, len: usize) -> Result<()> {
        self.span(offset, len)?;
        if len == 0 {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span was just checked against this allocation.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemset",
                    rt::cudaMemset((self.ptr + offset) as *mut core::ffi::c_void, 0, len),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// Copy `bytes` in at `offset` on `stream`, ordered with the fire.
    ///
    /// The fire path's transfer: the descriptor's derived vectors — tokens,
    /// positions, the indptr, the page tables — written in front of launches
    /// that read them. It must be ON THE STREAM, not synchronous: a
    /// synchronous copy is ordered against every stream in the process, which
    /// is both slower and a lie about what this fire depends on.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] or [`Fault::Device`].
    pub fn stage(&mut self, stream: *mut core::ffi::c_void, offset: u64, bytes: &[u8]) -> Result<()> {
        self.span(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span is checked; `bytes` outlives the enqueue, and
            // the caller synchronizes the stream before it is dropped —
            // every caller in this crate stages inside one `fire`.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemcpyAsync",
                    rt::cudaMemcpyAsync(
                        (self.ptr + offset) as *mut core::ffi::c_void,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                        rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                        stream.cast(),
                    ),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// Zero the whole allocation on `stream`, ordered with the fire.
    ///
    /// THE FIRE-PATH COUNTERPART OF [`Buffer::zeroed`], and the guest-program
    /// plane is what wants it: a PTIR fire's scratch, its per-channel pending
    /// flags and its commit slot are re-used allocation-for-allocation across
    /// fires (nothing allocates on the fire path), so each fire has to start
    /// from the state the emitted kernels assume — zeros — rather than from
    /// what the previous fire left. Synchronous zeroing would be ordered
    /// against every stream in the process, which is both slower and a lie
    /// about what this fire depends on.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn clear(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        if self.bytes == 0 {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span is this allocation's own, and the caller
            // synchronizes the stream before the buffer is dropped.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemsetAsync",
                    rt::cudaMemsetAsync(
                        self.ptr as *mut core::ffi::c_void,
                        0,
                        self.bytes,
                        stream.cast(),
                    ),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// Copy `into.len()` bytes out from `offset`, and wait.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] or [`Fault::Device`].
    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.span(offset, into.len())?;
        if into.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span is checked and `into` is a live host slice.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemcpy",
                    rt::cudaMemcpy(
                        into.as_mut_ptr().cast(),
                        (self.ptr + offset) as *const core::ffi::c_void,
                        into.len(),
                        rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    ),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    fn span(&self, offset: u64, len: usize) -> Result<()> {
        let end = offset.saturating_add(len as u64);
        if end > self.bytes as u64 {
            return Err(Fault::Ceiling {
                what: "bytes of a device buffer",
                need: end,
                have: self.bytes as u64,
            });
        }
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        #[cfg(feature = "_cuda")]
        if self.ptr != 0 {
            // SAFETY: the pointer came from this buffer's own `cudaMalloc`
            // and is freed exactly once.
            unsafe {
                let _ = cudarc::runtime::sys::cudaFree(self.ptr as *mut core::ffi::c_void);
            }
        }
    }
}
