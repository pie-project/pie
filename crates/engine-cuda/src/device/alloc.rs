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

/// A failed allocation, re-said with numbers when it was a shortfall.
///
/// **THE ONE STATUS WORTH TRANSLATING, AND ONLY IT.** `cudaErrorMemoryAllocation`
/// is a fact about a size and a device, and both numbers are askable; every
/// other status is a fact about the runtime and is left as the call and the
/// code. [`Fault::Ceiling`] is deliberately not what comes back: that one is a
/// fire wanting more than the load reserved, and this is a load wanting more
/// than the card has. The free figure is what `cudaMemGetInfo` says AFTER the failure,
/// which is the honest one — it is what was actually available to the ask —
/// and a query that itself fails leaves the original fault alone rather than
/// replacing a true sentence with a guess.
#[cfg(feature = "_cuda")]
fn out_of_room(fault: Fault, bytes: usize) -> Fault {
    use cudarc::runtime::sys as rt;

    let shortfall = matches!(
        fault,
        Fault::Device { code, .. } if code == rt::cudaError::cudaErrorMemoryAllocation as i32
    );
    if !shortfall {
        return fault;
    }
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: two live locals; the call only writes them.
    let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
    if asked != rt::cudaError::cudaSuccess {
        return fault;
    }
    Fault::OutOfMemory {
        need: bytes as u64,
        have: free as u64,
    }
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
    /// **AND A REFUSAL THAT DOES NOT FIT CARRIES BOTH NUMBERS** (palo C3b).
    /// Every allocation this crate makes is sized once from a budget, so the
    /// interesting failure is always the same one — this model does not fit
    /// this device — and `cudaMalloc answered 2` is the least useful sentence
    /// there is about it. `cudaErrorMemoryAllocation` is therefore turned into
    /// a [`Fault::Ceiling`] with the ask and the free, which is the shape a
    /// ceiling refusal already has everywhere else in this shell and which a
    /// caller can act on: a 27B checkpoint against a 46 GiB card says so in
    /// gibibytes instead of in an errno. Every other CUDA status stays what it
    /// was.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a request the device has no room for,
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
            let allocated =
                unsafe { crate::device::ctx::check("cudaMalloc", rt::cudaMalloc(&raw mut base, bytes)) };
            if let Err(fault) = allocated {
                return Err(out_of_room(fault, bytes));
            }
            // SAFETY: `base` is the allocation just made, of `bytes` bytes.
            unsafe {
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

    /// Copy `len` bytes from a HOST address at `offset` on `stream`.
    ///
    /// [`Buffer::stage`]'s twin, and the only difference is where the source
    /// lives — which is the whole of what the staging ring changed. `stage`
    /// takes a `&[u8]` that is almost always a `Vec` the fire built: pageable
    /// memory, so the driver copies it into its own staging buffer before
    /// `cudaMemcpyAsync` returns, which makes the call synchronous in the
    /// source and is why a single-buffered `Inputs` was survivable at depth 1.
    /// This one's source is a claimed ring slot's PINNED bytes: the copy is
    /// genuinely asynchronous, the host must not touch those bytes again until
    /// the GPU has passed it, and the slot's lifetime — claimed at prepare,
    /// released by the step's settlement callback — is exactly that bound.
    ///
    /// # Safety
    ///
    /// `src .. src + len` must be a live, page-locked host allocation that
    /// stays unwritten until the copy completes on `stream`.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a span past this allocation, [`Fault::Device`]
    /// for the copy.
    pub unsafe fn stage_from(
        &mut self,
        stream: *mut core::ffi::c_void,
        offset: u64,
        src: *const u8,
        len: usize,
    ) -> Result<()> {
        self.span(offset, len)?;
        if len == 0 {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the destination span is checked; the source is the
            // caller's promise above.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemcpyAsync",
                    rt::cudaMemcpyAsync(
                        (self.ptr + offset) as *mut core::ffi::c_void,
                        src.cast(),
                        len,
                        rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                        stream.cast(),
                    ),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = (stream, src);
            Err(Fault::Runtimeless)
        }
    }

    /// Zero a span on `stream`, ordered with the fire.
    ///
    /// [`Buffer::zero_span`]'s asynchronous twin, and the fire path wants this
    /// one for the reason F2b made sharp: `cudaMemset` is SYNCHRONOUS, so the
    /// fresh-slot clear that begins a sequence drained everything airborne —
    /// a host wait between two waves, which is exactly what article 2
    /// forbids. On the stream it is what it always meant to be: a zeroing
    /// ordered in front of the launches that read the bank, costing the
    /// pipeline nothing.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a span past this allocation, [`Fault::Device`]
    /// for the memset.
    pub fn zero_span_on(
        &mut self,
        stream: *mut core::ffi::c_void,
        offset: u64,
        len: usize,
    ) -> Result<()> {
        self.span(offset, len)?;
        if len == 0 {
            return Ok(());
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span was just checked against this allocation, and
            // the caller keeps it alive past the enqueue.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemsetAsync",
                    rt::cudaMemsetAsync(
                        (self.ptr + offset) as *mut core::ffi::c_void,
                        0,
                        len,
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

/// **ONE MAPPED PINNED ALLOCATION: THE SAME BYTES ON BOTH SIDES.**
///
/// `cudaHostAlloc(.., cudaHostAllocMapped)` gives a host pointer the guest's
/// own thread stores through and a device address a kernel dereferences —
/// under UVA the same number, which `cudaHostGetDevicePointer` is asked rather
/// than assumed. That equality is the whole of alto's channel crossing
/// (survey §7, invariant I5): a guest's cell reaches the device because
/// `channel::pull_validate` READS it where the guest wrote it, and a pass's
/// cell reaches the guest because `channel::scatter_publish` WRITES it where
/// the guest will read it. Neither direction is a `cudaMemcpy`, and a full
/// guest round trip makes no CUDA call at all on the guest's thread.
///
/// **NOT A [`Buffer`], AND THE DIFFERENCE IS THE POINT.** A `Buffer`'s bytes
/// are device-only and reach the host through a copy; these are one
/// allocation two agents address concurrently. So there is no `write`/`read`
/// pair here: a caller takes the slice and stores into it.
#[derive(Debug)]
pub struct Pinned {
    host: *mut u8,
    device: u64,
    bytes: usize,
}

// SAFETY: a `Pinned` is an allocation and a length. What makes concurrent
// access sound is the SPSC discipline of the channel plane above it — the
// guest owns one control word and the engine the other, and neither writes
// the other's — not anything this type could enforce; moving the allocation
// between threads is no less sound than moving a `Box<[u8]>`.
unsafe impl Send for Pinned {}
// SAFETY: as `Send`; `&Pinned` hands out only the base addresses and the
// length, and the byte access below is `unsafe` at the caller's door.
unsafe impl Sync for Pinned {}

impl Pinned {
    /// Allocate `bytes` of zeroed mapped pinned memory.
    ///
    /// A zero-byte request answers a null allocation, which is what a
    /// channel with no host end wants: nothing to address and nothing to
    /// free.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] without a CUDA runtime, [`Fault::Device`] for
    /// whatever `cudaHostAlloc` said.
    pub fn mapped(bytes: usize) -> Result<Pinned> {
        if bytes == 0 {
            return Ok(Pinned {
                host: core::ptr::null_mut(),
                device: 0,
                bytes: 0,
            });
        }
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;

            let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `host` is a live local; the allocation is this
            // structure's, freed exactly once in `Drop`.
            unsafe {
                crate::device::ctx::check(
                    "cudaHostAlloc",
                    rt::cudaHostAlloc(&raw mut host, bytes, rt::cudaHostAllocMapped),
                )?;
                core::ptr::write_bytes(host.cast::<u8>(), 0, bytes);
            }
            let mut device: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `host` is the allocation just made and mapped.
            unsafe {
                crate::device::ctx::check(
                    "cudaHostGetDevicePointer",
                    rt::cudaHostGetDevicePointer(&raw mut device, host, 0),
                )?;
            }
            Ok(Pinned {
                host: host.cast(),
                device: device as u64,
                bytes,
            })
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = bytes;
            Err(Fault::Runtimeless)
        }
    }

    /// The address the host stores through.
    #[must_use]
    pub fn host(&self) -> *mut u8 {
        self.host
    }

    /// The address a kernel dereferences. Equal to [`Pinned::host`] under
    /// UVA, asked rather than assumed.
    #[must_use]
    pub fn device(&self) -> u64 {
        self.device
    }

    /// How many bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The bytes, as the host sees them right now.
    ///
    /// **A SNAPSHOT OF SOMETHING ANOTHER AGENT MAY BE WRITING.** Sound only
    /// where the channel plane's SPSC discipline says this span is the
    /// caller's: a cell below the reader's tail, or a cell above the writer's
    /// head.
    #[must_use]
    pub fn read(&self, offset: usize, len: usize) -> Vec<u8> {
        if len == 0 || self.host.is_null() || offset + len > self.bytes {
            return vec![0u8; len];
        }
        // SAFETY: the span is inside this allocation, which outlives the copy.
        unsafe { core::slice::from_raw_parts(self.host.add(offset).cast_const(), len).to_vec() }
    }

    /// Store `bytes` at `offset`, answering `false` when they would not fit.
    ///
    /// As [`Pinned::read`]: sound where the discipline says the span is the
    /// caller's.
    pub fn write(&self, offset: usize, bytes: &[u8]) -> bool {
        if self.host.is_null() || offset + bytes.len() > self.bytes {
            return false;
        }
        // SAFETY: the span is inside this allocation, which outlives the copy.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), self.host.add(offset), bytes.len()) }
        true
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        #[cfg(feature = "_cuda")]
        if !self.host.is_null() {
            // SAFETY: the pointer came from this structure's own
            // `cudaHostAlloc` and is freed exactly once.
            unsafe {
                let _ = cudarc::runtime::sys::cudaFreeHost(self.host.cast());
            }
        }
    }
}

/// **Copy `bytes` to a device address the caller resolved, and wait.**
///
/// [`Buffer::write`]'s free-function twin, for the one owner that cannot hand
/// out a `&mut Buffer`: a channel's SHARED device ring lives behind an
/// `Arc<Endpoint>` (`program::endpoint`), so every holder has `&self` and the
/// seeds a bind plants have to reach it anyway. The bounds check is the
/// caller's for the same reason — the address is already resolved when it
/// arrives here.
///
/// Synchronous, like [`Buffer::write`], and for its reason: the only caller is
/// bind-time seeding, which is control plane and orders against nothing.
///
/// # Errors
///
/// [`Fault::Device`] for the copy, [`Fault::Runtimeless`] with no runtime
/// selected.
pub fn write_raw(at: u64, bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: `at` is an address the caller resolved against an allocation
        // it owns, and `bytes` is a live host slice for the duration of a
        // synchronous copy.
        unsafe {
            crate::device::ctx::check(
                "cudaMemcpy",
                rt::cudaMemcpy(
                    at as *mut core::ffi::c_void,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                ),
            )
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = at;
        Err(Fault::Runtimeless)
    }
}

/// **One device-to-device copy, on `stream`** (alto F3).
///
/// A free function rather than a [`Buffer`] method because the two callers do
/// not hold one: the buffered-activation scatter moves bytes between an ARENA
/// rectangle and a POOL slab, and `copy_state` moves them between two slots of
/// the same slab. Both hold resolved device addresses, which is what this
/// takes.
///
/// On the stream, never synchronous, for the reason
/// [`Buffer::zero_span_on`] gives: a synchronous copy is ordered against every
/// stream in the process, which is a host wait between two waves (article 2)
/// and a lie about what this fire depends on.
///
/// # Errors
///
/// [`Fault::Device`] for the copy, [`Fault::Runtimeless`] with no runtime
/// selected.
pub fn copy_d2d(
    stream: *mut core::ffi::c_void,
    dst: u64,
    src: u64,
    bytes: usize,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: both addresses are resolutions the caller made against
        // allocations this shell owns for the load's lifetime, and the copy is
        // enqueued on the same stream as the launches around it.
        unsafe {
            crate::device::ctx::check(
                "cudaMemcpyAsync",
                rt::cudaMemcpyAsync(
                    dst as *mut core::ffi::c_void,
                    src as *const core::ffi::c_void,
                    bytes,
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                    stream.cast(),
                ),
            )
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (stream, dst, src);
        Err(Fault::Runtimeless)
    }
}

/// **Copy `bytes` between two addresses of ANY kind, on `stream`** — the
/// transfer whose ends the caller resolved and whose direction the driver
/// works out.
///
/// [`copy_d2d`]'s twin for the one pair that is not device-to-device: a
/// routed expert's promotion reads MAPPED PINNED HOST memory (alto design
/// §7's T1) and writes the device slab, and both ends are plain `u64`
/// addresses under UVA. `cudaMemcpyDefault` is the kind that reads the
/// direction off the addresses themselves, which is the only honest spelling
/// when one end is a `cudaHostAlloc`ed page and the other is a `cudaMalloc`ed
/// one — naming `DeviceToDevice` for that pair would be a claim about the
/// allocations that this module cannot make.
///
/// Asynchronous, like everything the promotion path enqueues: the source must
/// stay unwritten until the copy completes, which for the tier means the
/// pinned mirror it owns for the load's whole life.
///
/// # Errors
///
/// [`Fault::Device`] for the copy, [`Fault::Runtimeless`] with no runtime
/// selected.
pub fn copy_any(
    stream: *mut core::ffi::c_void,
    dst: u64,
    src: u64,
    bytes: usize,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: both addresses are resolutions the caller made against
        // allocations this shell owns for the load's lifetime, and the copy is
        // enqueued on a stream the caller keeps ordered around it.
        unsafe {
            crate::device::ctx::check(
                "cudaMemcpyAsync",
                rt::cudaMemcpyAsync(
                    dst as *mut core::ffi::c_void,
                    src as *const core::ffi::c_void,
                    bytes,
                    rt::cudaMemcpyKind::cudaMemcpyDefault,
                    stream.cast(),
                ),
            )
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (stream, dst, src);
        Err(Fault::Runtimeless)
    }
}

/// **Zero `len` bytes at a device address, and wait.**
///
/// [`Buffer::zero_span`]'s free-function twin, for the bytes whose bounds
/// were checked somewhere else: an [`Arena`](crate::device::Arena) span
/// checks against what is COMMITTED rather than against a fixed length, and
/// that check cannot live in a method on a `Buffer` that no longer owns the
/// bytes.
///
/// # Errors
///
/// [`Fault::Device`] for the fill, [`Fault::Runtimeless`] with no runtime.
pub fn zero_span(at: u64, len: usize) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: the address is a span the caller resolved against an
        // allocation this shell owns.
        unsafe {
            crate::device::ctx::check(
                "cudaMemset",
                rt::cudaMemset(at as *mut core::ffi::c_void, 0, len),
            )
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = at;
        Err(Fault::Runtimeless)
    }
}

/// **Zero `len` bytes at a device address on `stream`**, ordered with the
/// fire — [`Buffer::zero_span_on`]'s free-function twin.
///
/// # Errors
///
/// As [`zero_span`].
pub fn zero_span_on(stream: *mut core::ffi::c_void, at: u64, len: usize) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: as `zero_span`, and the caller keeps the span alive past
        // the enqueue.
        unsafe {
            crate::device::ctx::check(
                "cudaMemsetAsync",
                rt::cudaMemsetAsync(at as *mut core::ffi::c_void, 0, len, stream.cast()),
            )
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = (stream, at);
        Err(Fault::Runtimeless)
    }
}

/// **Copy `into.len()` bytes back from a device address, and wait** —
/// [`Buffer::read`]'s free-function twin.
///
/// # Errors
///
/// [`Fault::Device`] for the copy, [`Fault::Runtimeless`] with no runtime.
pub fn copy_d2h(at: u64, into: &mut [u8]) -> Result<()> {
    if into.is_empty() {
        return Ok(());
    }
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: the address is a checked span and `into` is a live host
        // slice of exactly the length copied.
        unsafe {
            crate::device::ctx::check(
                "cudaMemcpy",
                rt::cudaMemcpy(
                    into.as_mut_ptr().cast(),
                    at as *const core::ffi::c_void,
                    into.len(),
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                ),
            )
        }
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = at;
        Err(Fault::Runtimeless)
    }
}

/// **What the card says is free, in bytes**, or `None` with no runtime.
///
/// The one observation that can settle whether an elastic commit actually put
/// physical pages on the device rather than only moving a counter: a gate
/// reads it either side of a fire and diffs. Not on any serving path — a
/// `cudaMemGetInfo` is a driver round trip, and the only production caller is
/// [`PhysicalPool`](crate::device::PhysicalPool)'s budget, which asks once at
/// load and again only when a commit has to grow.
#[must_use]
pub fn free_bytes() -> Option<u64> {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: two live locals; the call only writes them.
        let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
        (asked == rt::cudaError::cudaSuccess).then_some(free as u64)
    }
    #[cfg(not(feature = "_cuda"))]
    {
        None
    }
}
