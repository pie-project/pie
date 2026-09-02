//! One owned device allocation, plus the host/device transfers a shell needs. Every offset is bounds-checked here before it becomes a raw device pointer.

use crate::error::{Fault, Result};

/// A device allocation of a stated size.
#[derive(Debug)]
pub struct Buffer {
    ptr: u64,
    bytes: usize,
}

/// Translates `cudaErrorMemoryAllocation` into `Fault::OutOfMemory` with the bytes asked and `cudaMemGetInfo`'s free bytes measured after the failure; every other status passes through unchanged.
#[cfg(feature = "cuda")]
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
    /// Allocate `bytes`, zeroed — `cudaMalloc` hands back whatever the last tenant left, so unwritten bytes would read as data. A zero-byte request returns a null handle rather than an error. # Errors: [`Fault::Ceiling`] for a request the device has no room for, [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn zeroed(bytes: usize) -> Result<Buffer> {
        if bytes == 0 {
            return Ok(Buffer { ptr: 0, bytes: 0 });
        }
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            let mut base: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `base` is a live local; the allocation is this buffer's, freed exactly once in `Drop`.
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
        #[cfg(not(feature = "cuda"))]
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

    /// The address `offset` bytes in, checked against the length. # Errors: [`Fault::Ceiling`] for an offset past the end.
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

    /// Copy `bytes` in at `offset`, and wait. Synchronous: a caller must see the bytes landed before it launches anything that reads them. # Errors: [`Fault::Ceiling`] for a write past the end, [`Fault::Device`] for the copy.
    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.span(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span was just checked against this allocation, and `bytes` is a live host slice for the duration of a synchronous copy.
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
        #[cfg(not(feature = "cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// Zero `len` bytes at `offset`, and wait. Synchronous, like [`Buffer::write`]. # Errors: [`Fault::Ceiling`] for a span past the end, [`Fault::Device`] for the fill.
    pub fn zero_span(&mut self, offset: u64, len: usize) -> Result<()> {
        self.span(offset, len)?;
        if len == 0 {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
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
        #[cfg(not(feature = "cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// Copy `bytes` in at `offset` on `stream`, ordered with the fire. Async, not synchronous, since a synchronous copy would order against every stream in the process. # Errors: [`Fault::Ceiling`] or [`Fault::Device`].
    pub fn stage(&mut self, stream: *mut core::ffi::c_void, offset: u64, bytes: &[u8]) -> Result<()> {
        self.span(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span is checked; `bytes` outlives the enqueue, and the caller synchronizes the stream before it is dropped — every caller in this crate stages inside one `fire`.
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
        #[cfg(not(feature = "cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// Copy `len` bytes from a host address on `stream` — [`Buffer::stage`]'s
    /// twin, but genuinely asynchronous since the source is pinned memory.
    ///
    /// # Safety
    ///
    /// `src .. src + len` must be a live, page-locked host allocation that
    /// stays unwritten until the copy completes on `stream`.
    /// # Errors: [`Fault::Ceiling`] for a span past this allocation, [`Fault::Device`] for the copy.
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
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the destination span is checked; the source is the caller's promise above.
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
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (stream, src);
            Err(Fault::Runtimeless)
        }
    }

    /// Zero a span on `stream`, ordered with the fire — [`Buffer::zero_span`]'s asynchronous twin; `cudaMemset` itself is synchronous and would drain everything in flight. # Errors: [`Fault::Ceiling`] for a span past this allocation, [`Fault::Device`] for the memset.
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
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span was just checked against this allocation, and the caller keeps it alive past the enqueue.
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
        #[cfg(not(feature = "cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// Zero the whole allocation on `stream` — the fire-path counterpart of [`Buffer::zeroed`], for allocations reused across fires. # Errors: [`Fault::Runtimeless`] or [`Fault::Device`].
    pub fn clear(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        if self.bytes == 0 {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            // SAFETY: the span is this allocation's own, and the caller synchronizes the stream before the buffer is dropped.
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
        #[cfg(not(feature = "cuda"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// Copy `into.len()` bytes out from `offset`, and wait. # Errors: [`Fault::Ceiling`] or [`Fault::Device`].
    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.span(offset, into.len())?;
        if into.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
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
        #[cfg(not(feature = "cuda"))]
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
        #[cfg(feature = "cuda")]
        if self.ptr != 0 {
            // SAFETY: the pointer came from this buffer's own `cudaMalloc` and is freed exactly once.
            unsafe {
                let _ = cudarc::runtime::sys::cudaFree(self.ptr as *mut core::ffi::c_void);
            }
        }
    }
}

/// One mapped pinned allocation: the same address on host and device under UVA. Unlike [`Buffer`], two agents address it concurrently, so there is no write/read pair.
#[derive(Debug)]
pub struct Pinned {
    host: *mut u8,
    device: u64,
    bytes: usize,
    /// Which allocator made these bytes, so `Drop` frees them the matching way: `cudaFreeHost` on a self-registered mapping faults, and `munmap` on a `cudaHostAlloc` leaks the driver's bookkeeping.
    origin: Origin,
}

/// How a [`Pinned`] came to be page-locked, for `Drop` to free it correctly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origin {
    /// `cudaHostAlloc` made the allocation and page-locked it in one call.
    Allocated,
    /// This process mapped the pages and `cudaHostRegister` locked them ([`Pinning`]). Unreachable without a runtime.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Registered,
}

// SAFETY: concurrent-access soundness comes from the channel's SPSC discipline above this type (guest and engine each own one control word), not from anything here; moving the allocation is as sound as a `Box<[u8]>`.
unsafe impl Send for Pinned {}
// SAFETY: as `Send`; `&Pinned` only hands out addresses/length, and byte access is `unsafe` at the caller's door.
unsafe impl Sync for Pinned {}

impl Pinned {
    /// Allocate `bytes` of zeroed mapped pinned memory. A zero-byte request returns a null allocation. # Errors: [`Fault::Runtimeless`] without a CUDA runtime, [`Fault::Device`] for whatever `cudaHostAlloc` said.
    pub fn mapped(bytes: usize) -> Result<Pinned> {
        Pinned::alloc(bytes, true)
    }

    /// Same as [`Pinned::mapped`] without the zeroing memset; caller must overwrite the whole length (padding included) before any read. # Errors: as [`Pinned::mapped`].
    pub fn mapped_uninit(bytes: usize) -> Result<Pinned> {
        Pinned::alloc(bytes, false)
    }

    /// The memset [`Pinned::mapped_uninit`] skipped, done late (e.g. to restore zeroed state before a fallback writes only part of it). Sound where [`Pinned::write`] is sound.
    pub fn zero(&self) {
        if self.host.is_null() || self.bytes == 0 {
            return;
        }
        // SAFETY: the span is the allocation, which outlives the write; what makes it writable AT ALL is the discipline the doc above states.
        unsafe { core::ptr::write_bytes(self.host, 0, self.bytes) }
    }

    fn alloc(bytes: usize, zeroed: bool) -> Result<Pinned> {
        if bytes == 0 {
            return Ok(Pinned {
                host: core::ptr::null_mut(),
                device: 0,
                bytes: 0,
                origin: Origin::Allocated,
            });
        }
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `host` is a live local; the allocation is this structure's, freed exactly once in `Drop`.
            unsafe {
                crate::device::ctx::check(
                    "cudaHostAlloc",
                    rt::cudaHostAlloc(&raw mut host, bytes, rt::cudaHostAllocMapped),
                )?;
                if zeroed {
                    core::ptr::write_bytes(host.cast::<u8>(), 0, bytes);
                }
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
                origin: Origin::Allocated,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (bytes, zeroed);
            Err(Fault::Runtimeless)
        }
    }

    /// The address the host stores through.
    #[must_use]
    pub fn host(&self) -> *mut u8 {
        self.host
    }

    /// The address a kernel dereferences. Equal to [`Pinned::host`] under UVA, asked rather than assumed.
    #[must_use]
    pub fn device(&self) -> u64 {
        self.device
    }

    /// How many bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The bytes, as the host sees them right now — a snapshot of something another thread may be writing. Sound only where the channel's SPSC discipline says this span is the caller's.
    #[must_use]
    pub fn read(&self, offset: usize, len: usize) -> Vec<u8> {
        if len == 0 || self.host.is_null() || offset + len > self.bytes {
            return vec![0u8; len];
        }
        // SAFETY: the span is inside this allocation, which outlives the copy.
        unsafe { core::slice::from_raw_parts(self.host.add(offset).cast_const(), len).to_vec() }
    }

    /// Borrow a span without copying, or `None` for one this allocation does not hold. Same SPSC soundness rule as [`Pinned::read`], held for the life of the borrow.
    #[must_use]
    pub fn view(&self, at: u64, len: u64) -> Option<&[u8]> {
        if len == 0 {
            return Some(&[]);
        }
        let at = usize::try_from(at).ok()?;
        let len = usize::try_from(len).ok()?;
        if self.host.is_null() || at.checked_add(len)? > self.bytes {
            return None;
        }
        // SAFETY: the span is inside this allocation, which outlives the borrow; what makes the bytes readable AT ALL is the discipline the doc above states, exactly as for `Pinned::read`.
        Some(unsafe { core::slice::from_raw_parts(self.host.add(at).cast_const(), len) })
    }

    /// Store `bytes` at `offset`, answering `false` when they would not fit. As [`Pinned::read`]: sound where the discipline says the span is the caller's.
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
        if self.host.is_null() {
            return;
        }
        match self.origin {
            #[cfg(feature = "cuda")]
            Origin::Allocated => {
                // SAFETY: the pointer came from this structure's own `cudaHostAlloc` and is freed exactly once.
                unsafe {
                    let _ = cudarc::runtime::sys::cudaFreeHost(self.host.cast());
                }
            }
            // Unregister before unmap: the page-lock claims pages this process owns, and dropping them first leaves that claim dangling.
            Origin::Registered => {
                #[cfg(feature = "cuda")]
                // SAFETY: the span is the one this structure registered.
                unsafe {
                    let _ = cudarc::runtime::sys::cudaHostUnregister(self.host.cast());
                }
                // SAFETY: unmapping the mapping `Pinning::uninit` made, once.
                unsafe {
                    libc::munmap(self.host.cast(), self.bytes.max(1));
                }
            }
            #[cfg(not(feature = "cuda"))]
            Origin::Allocated => {}
        }
    }
}

/// A pinned allocation filled first and page-locked last: mmap'd and advised into huge pages upfront, then locked with one `cudaHostRegister` at the end, avoiding `cudaHostAlloc`'s process-wide lock for the whole allocation.
#[derive(Debug)]
pub struct Pinning {
    host: *mut u8,
    bytes: usize,
}

// SAFETY: as `Pinned`'s — an address and a length, whose sole writer is the thread that holds it. `Pinning` hands out no aliases at all: `host` is the one door, and `lock` consumes the value.
unsafe impl Send for Pinning {}

impl Pinning {
    /// Map `bytes` of pages nothing has written yet, advised into huge pages for a cheap page-lock later. Uninitialized: caller must overwrite every byte first. # Errors: [`Fault::Device`] naming `mmap`, for a mapping the kernel refused.
    pub fn uninit(bytes: usize) -> Result<Pinning> {
        // SAFETY: a fresh private anonymous mapping of a stated length; no fd and no offset are involved, and the pages belong to nobody else.
        let at = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                bytes.max(1),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if at == libc::MAP_FAILED {
            return Err(Fault::Device {
                call: "mmap",
                code: -1,
            });
        }
        // Advisory: a kernel that refuses huge pages falls back to per-page locking silently, not a failure. SAFETY: the range is the mapping this call just made.
        unsafe { libc::madvise(at, bytes.max(1), libc::MADV_HUGEPAGE) };
        Ok(Pinning {
            host: at.cast(),
            bytes,
        })
    }

    /// The base of the mapping, for the reader that fills it.
    #[must_use]
    pub fn host(&self) -> *mut u8 {
        self.host
    }

    /// Page-lock the finished image: one `cudaHostRegister` over the whole mapping, and the device address asked for rather than assumed. # Errors: [`Fault::Runtimeless`] without a CUDA runtime, [`Fault::Device`] for `cudaHostRegister`/`cudaHostGetDevicePointer`; the mapping returns to the kernel on any failure.
    pub fn lock(self) -> Result<Pinned> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            if self.bytes == 0 {
                return Pinned::mapped(0);
            }
            // SAFETY: the span is this structure's own mapping, live until the `forget` below hands it to the `Pinned`.
            let locked = unsafe {
                crate::device::ctx::check(
                    "cudaHostRegister",
                    rt::cudaHostRegister(self.host.cast(), self.bytes, rt::cudaHostRegisterMapped),
                )
            };
            locked?;
            let mut device: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `host` is the mapping just registered.
            let asked = unsafe {
                crate::device::ctx::check(
                    "cudaHostGetDevicePointer",
                    rt::cudaHostGetDevicePointer(&raw mut device, self.host.cast(), 0),
                )
            };
            if asked.is_err() {
                // SAFETY: undoing the registration this call just made.
                unsafe { let _ = rt::cudaHostUnregister(self.host.cast()); }
                asked?;
            }
            let host = self.host;
            let bytes = self.bytes;
            // Ownership passes to the returned `Pinned`; its `Drop` unregisters and unmaps.
            core::mem::forget(self);
            Ok(Pinned {
                host,
                device: device as u64,
                bytes,
                origin: Origin::Registered,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }
}

impl Drop for Pinning {
    fn drop(&mut self) {
        // SAFETY: unmapping the mapping `Pinning::uninit` made, exactly once — `lock` forgets the value rather than reaching here.
        unsafe {
            libc::munmap(self.host.cast(), self.bytes.max(1));
        }
    }
}

/// Copy `bytes` to a resolved, bounds-checked device address, and wait. [`Buffer::write`]'s free-function twin, for an owner with only `&self`. # Errors: [`Fault::Device`] for the copy, [`Fault::Runtimeless`] with no runtime selected.
pub fn write_raw(at: u64, bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: `at` is an address the caller resolved against an allocation it owns, and `bytes` is a live host slice for the duration of a synchronous copy.
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
    #[cfg(not(feature = "cuda"))]
    {
        let _ = at;
        Err(Fault::Runtimeless)
    }
}

/// One device-to-device copy, on `stream`. A free function since callers hold only resolved device addresses, not an owned [`Buffer`]; always async since synchronous would order against every stream in the process. # Errors: [`Fault::Device`] for the copy, [`Fault::Runtimeless`] with no runtime selected.
pub fn copy_d2d(
    stream: *mut core::ffi::c_void,
    dst: u64,
    src: u64,
    bytes: usize,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: both addresses are resolutions the caller made against allocations this shell owns for the load's lifetime, and the copy is enqueued on the same stream as the launches around it.
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
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (stream, dst, src);
        Err(Fault::Runtimeless)
    }
}

/// Copy `bytes` between addresses of any kind, on `stream`, via `cudaMemcpyDefault` since direction may not be device-to-device — [`copy_d2d`]'s twin for that pair. # Errors: [`Fault::Device`] for the copy, [`Fault::Runtimeless`] with no runtime selected.
pub fn copy_any(
    stream: *mut core::ffi::c_void,
    dst: u64,
    src: u64,
    bytes: usize,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: both addresses are resolutions the caller made against allocations this shell owns for the load's lifetime, and the copy is enqueued on a stream the caller keeps ordered around it.
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
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (stream, dst, src);
        Err(Fault::Runtimeless)
    }
}

/// Zero `len` bytes at a device address, and wait. [`Buffer::zero_span`]'s free-function twin; bounds are checked by the caller (e.g. against an [`Arena`](crate::device::Arena)'s committed length, not a fixed size). # Errors: [`Fault::Device`] for the fill, [`Fault::Runtimeless`] with no runtime.
pub fn zero_span(at: u64, len: usize) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: the address is a span the caller resolved against an allocation this shell owns.
        unsafe {
            crate::device::ctx::check(
                "cudaMemset",
                rt::cudaMemset(at as *mut core::ffi::c_void, 0, len),
            )
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = at;
        Err(Fault::Runtimeless)
    }
}

/// Zero `len` bytes at a device address on `stream`, ordered with the fire — [`Buffer::zero_span_on`]'s free-function twin. # Errors: as [`zero_span`].
pub fn zero_span_on(stream: *mut core::ffi::c_void, at: u64, len: usize) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: as `zero_span`, and the caller keeps the span alive past the enqueue.
        unsafe {
            crate::device::ctx::check(
                "cudaMemsetAsync",
                rt::cudaMemsetAsync(at as *mut core::ffi::c_void, 0, len, stream.cast()),
            )
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (stream, at);
        Err(Fault::Runtimeless)
    }
}

/// Copy `into.len()` bytes back from a device address, and wait — [`Buffer::read`]'s free-function twin. # Errors: [`Fault::Device`] for the copy, [`Fault::Runtimeless`] with no runtime.
pub fn copy_d2h(at: u64, into: &mut [u8]) -> Result<()> {
    if into.is_empty() {
        return Ok(());
    }
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: the address is a checked span and `into` is a live host slice of exactly the length copied.
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
    #[cfg(not(feature = "cuda"))]
    {
        let _ = at;
        Err(Fault::Runtimeless)
    }
}

/// What the card says is free, in bytes, or `None` with no runtime. Not on any serving path — the only production caller is [`PhysicalPool`](crate::device::PhysicalPool)'s budget check, at load and when a commit grows.
#[must_use]
pub fn free_bytes() -> Option<u64> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: two live locals; the call only writes them.
        let asked = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
        (asked == rt::cudaError::cudaSuccess).then_some(free as u64)
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}
