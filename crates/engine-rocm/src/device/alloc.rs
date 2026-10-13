use crate::error::{Fault, Result};

#[derive(Debug)]
pub struct Buffer {
    ptr: u64,
    bytes: usize,
}

#[cfg(feature = "rocm")]
fn out_of_room(fault: Fault, bytes: usize) -> Fault {
    use kernels_rocm::hip::rt;

    let shortfall = matches!(
        fault,
        Fault::Device { code, .. } if code == rt::hipErrorOutOfMemory
    );
    if !shortfall {
        return fault;
    }
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: two live locals; the call only writes them.
    let asked = unsafe { rt::hipMemGetInfo(&raw mut free, &raw mut total) };
    if asked != rt::hipSuccess {
        return fault;
    }
    Fault::OutOfMemory {
        need: bytes as u64,
        have: free as u64,
    }
}

impl Buffer {
    pub fn zeroed(bytes: usize) -> Result<Buffer> {
        if bytes == 0 {
            return Ok(Buffer { ptr: 0, bytes: 0 });
        }
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            let mut base: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `base` is a live local; the allocation is this buffer's, freed exactly once in `Drop`.
            let allocated = unsafe {
                crate::device::ctx::check("hipMalloc", rt::hipMalloc(&raw mut base, bytes))
            };
            if let Err(fault) = allocated {
                return Err(out_of_room(fault, bytes));
            }
            // SAFETY: `base` is the allocation just made, of `bytes` bytes.
            unsafe {
                crate::device::ctx::check("hipMemset", rt::hipMemset(base, 0, bytes))?;
            }
            Ok(Buffer {
                ptr: base as u64,
                bytes,
            })
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    pub fn zeroed_on(stream: *mut core::ffi::c_void, bytes: usize) -> Result<Buffer> {
        if bytes == 0 {
            return Ok(Buffer { ptr: 0, bytes: 0 });
        }
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            let mut base: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `base` is a live local; the allocation is this buffer's, freed exactly once in `Drop`.
            let allocated = unsafe {
                crate::device::ctx::check("hipMalloc", rt::hipMalloc(&raw mut base, bytes))
            };
            if let Err(fault) = allocated {
                return Err(out_of_room(fault, bytes));
            }
            // SAFETY: `base` is the allocation just made, of `bytes` bytes, and
            // `stream` is the one every later read of it is ordered on.
            unsafe {
                crate::device::ctx::check(
                    "hipMemsetAsync",
                    rt::hipMemsetAsync(base, 0, bytes, stream.cast()),
                )?;
            }
            Ok(Buffer {
                ptr: base as u64,
                bytes,
            })
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    #[must_use]
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

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

    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.span(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            // SAFETY: the span was just checked against this allocation, and `bytes` is a live host slice for the duration of a synchronous copy.
            unsafe {
                crate::device::ctx::check(
                    "hipMemcpy",
                    rt::hipMemcpy(
                        (self.ptr + offset) as *mut core::ffi::c_void,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                        rt::hipMemcpyHostToDevice,
                    ),
                )
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    pub fn zero_span(&mut self, offset: u64, len: usize) -> Result<()> {
        self.span(offset, len)?;
        if len == 0 {
            return Ok(());
        }
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            // SAFETY: the span was just checked against this allocation.
            unsafe {
                crate::device::ctx::check(
                    "hipMemset",
                    rt::hipMemset((self.ptr + offset) as *mut core::ffi::c_void, 0, len),
                )
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    pub fn stage(
        &mut self,
        stream: *mut core::ffi::c_void,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()> {
        self.span(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            if staged_through_ring(self.ptr + offset, bytes, stream)? {
                return Ok(());
            }
            // SAFETY: the span is checked; `bytes` outlives the enqueue, and the caller synchronizes the stream before it is dropped — every caller in this crate stages inside one `fire`.
            unsafe {
                crate::device::ctx::check(
                    "hipMemcpyAsync",
                    rt::hipMemcpyAsync(
                        (self.ptr + offset) as *mut core::ffi::c_void,
                        bytes.as_ptr().cast(),
                        bytes.len(),
                        rt::hipMemcpyHostToDevice,
                        stream.cast(),
                    ),
                )
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// # Safety
    ///
    /// Each span's source must stay readable for its length until the copy
    /// on `stream` completes.
    pub unsafe fn stage_batch_from(
        &mut self,
        stream: *mut core::ffi::c_void,
        spans: &[(u64, *const u8, usize)],
    ) -> Result<()> {
        for &(offset, _, len) in spans {
            self.span(offset, len)?;
        }
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;
            use std::sync::atomic::{AtomicBool, Ordering};

            static REFUSED: AtomicBool = AtomicBool::new(false);

            let live: Vec<(u64, *const u8, usize)> = spans
                .iter()
                .copied()
                .filter(|&(_, _, len)| len > 0)
                .collect();
            if live.is_empty() {
                return Ok(());
            }
            if live.len() > 1 && !REFUSED.load(Ordering::Relaxed) {
                let mut dsts: Vec<*mut core::ffi::c_void> = live
                    .iter()
                    .map(|&(offset, _, _)| (self.ptr + offset) as *mut core::ffi::c_void)
                    .collect();
                let mut srcs: Vec<*mut core::ffi::c_void> = live
                    .iter()
                    .map(|&(_, src, _)| src.cast_mut().cast())
                    .collect();
                let mut sizes: Vec<usize> = live.iter().map(|&(_, _, len)| len).collect();
                let mut failed = 0usize;
                let mut device: i32 = 0;
                // SAFETY: plain query.
                let _ = unsafe { rt::hipGetDevice(&raw mut device) };
                let mut attrs = [rt::MemcpyAttributes {
                    srcAccessOrder: rt::hipMemcpySrcAccessOrderStream,
                    srcLocHint: rt::MemLocation {
                        type_: rt::hipMemLocationTypeHost,
                        id: 0,
                    },
                    dstLocHint: rt::MemLocation {
                        type_: rt::hipMemLocationTypeDevice,
                        id: device,
                    },
                    flags: 0,
                }];
                let mut attrs_at = [0usize];
                // SAFETY: every destination span is checked above; the sources are the caller's promise.
                let status = unsafe {
                    rt::hipMemcpyBatchAsync(
                        dsts.as_mut_ptr(),
                        srcs.as_mut_ptr(),
                        sizes.as_mut_ptr(),
                        live.len(),
                        attrs.as_mut_ptr(),
                        attrs_at.as_mut_ptr(),
                        1,
                        &raw mut failed,
                        stream.cast(),
                    )
                };
                if status == rt::hipSuccess {
                    return Ok(());
                }
                let _ = unsafe { rt::hipGetLastError() };
                REFUSED.store(true, Ordering::Relaxed);
            }
            for (offset, src, len) in live {
                // SAFETY: as above, one span at a time.
                unsafe { self.stage_from(stream, offset, src, len)? };
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    /// # Safety
    ///
    /// `src` must stay readable for `len` bytes until the copy on `stream`
    /// completes.
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
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            // SAFETY: the destination span is checked; the source is the caller's promise above.
            unsafe {
                crate::device::ctx::check(
                    "hipMemcpyAsync",
                    rt::hipMemcpyAsync(
                        (self.ptr + offset) as *mut core::ffi::c_void,
                        src.cast(),
                        len,
                        rt::hipMemcpyHostToDevice,
                        stream.cast(),
                    ),
                )
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (stream, src);
            Err(Fault::Runtimeless)
        }
    }

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
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            // SAFETY: the span was just checked against this allocation, and the caller keeps it alive past the enqueue.
            unsafe {
                crate::device::ctx::check(
                    "hipMemsetAsync",
                    rt::hipMemsetAsync(
                        (self.ptr + offset) as *mut core::ffi::c_void,
                        0,
                        len,
                        stream.cast(),
                    ),
                )
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    pub fn clear(&mut self, stream: *mut core::ffi::c_void) -> Result<()> {
        if self.bytes == 0 {
            return Ok(());
        }
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            // SAFETY: the span is this allocation's own, and the caller synchronizes the stream before the buffer is dropped.
            unsafe {
                crate::device::ctx::check(
                    "hipMemsetAsync",
                    rt::hipMemsetAsync(
                        self.ptr as *mut core::ffi::c_void,
                        0,
                        self.bytes,
                        stream.cast(),
                    ),
                )
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = stream;
            Err(Fault::Runtimeless)
        }
    }

    pub fn read(&self, offset: u64, into: &mut [u8]) -> Result<()> {
        self.span(offset, into.len())?;
        if into.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            // SAFETY: the span is checked and `into` is a live host slice.
            unsafe {
                crate::device::ctx::check(
                    "hipMemcpy",
                    rt::hipMemcpy(
                        into.as_mut_ptr().cast(),
                        (self.ptr + offset) as *const core::ffi::c_void,
                        into.len(),
                        rt::hipMemcpyDeviceToHost,
                    ),
                )
            }
        }
        #[cfg(not(feature = "rocm"))]
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
        #[cfg(feature = "rocm")]
        if self.ptr != 0 {
            // SAFETY: the pointer came from this buffer's own `cudaMalloc` and is freed exactly once.
            unsafe {
                let _ = kernels_rocm::hip::rt::hipFree(self.ptr as *mut core::ffi::c_void);
            }
        }
    }
}

#[derive(Debug)]
pub struct Pinned {
    host: *mut u8,
    device: u64,
    bytes: usize,
    origin: Origin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Origin {
    Allocated,
    #[cfg_attr(not(feature = "rocm"), allow(dead_code))]
    Registered,
}

// SAFETY: concurrent-access soundness comes from the channel's SPSC discipline above this type (guest and engine each own one control word), not from anything here; moving the allocation is as sound as a `Box<[u8]>`.
unsafe impl Send for Pinned {}
// SAFETY: as `Send`; `&Pinned` only hands out addresses/length, and byte access is `unsafe` at the caller's door.
unsafe impl Sync for Pinned {}

impl Pinned {
    /// An empty span, holding nothing to free.
    #[must_use]
    pub const fn empty() -> Pinned {
        Pinned {
            host: core::ptr::null_mut(),
            device: 0,
            bytes: 0,
            origin: Origin::Allocated,
        }
    }

    pub fn mapped(bytes: usize) -> Result<Pinned> {
        Pinned::alloc(bytes, true)
    }

    pub fn mapped_uninit(bytes: usize) -> Result<Pinned> {
        Pinned::alloc(bytes, false)
    }

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
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `host` is a live local; the allocation is this structure's, freed exactly once in `Drop`.
            unsafe {
                crate::device::ctx::check(
                    "hipHostAlloc",
                    rt::hipHostMalloc(
                        &raw mut host,
                        bytes,
                        rt::hipHostMallocMapped | rt::hipHostMallocPortable,
                    ),
                )?;
                if zeroed {
                    core::ptr::write_bytes(host.cast::<u8>(), 0, bytes);
                }
            }
            let mut device: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `host` is the allocation just made and mapped.
            unsafe {
                crate::device::ctx::check(
                    "hipHostGetDevicePointer",
                    rt::hipHostGetDevicePointer(&raw mut device, host, 0),
                )?;
            }
            Ok(Pinned {
                host: host.cast(),
                device: device as u64,
                bytes,
                origin: Origin::Allocated,
            })
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (bytes, zeroed);
            Err(Fault::Runtimeless)
        }
    }

    #[must_use]
    pub fn host(&self) -> *mut u8 {
        self.host
    }

    #[must_use]
    pub fn device(&self) -> u64 {
        self.device
    }

    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    #[must_use]
    pub fn read(&self, offset: usize, len: usize) -> Vec<u8> {
        if len == 0 || self.host.is_null() || offset + len > self.bytes {
            return vec![0u8; len];
        }
        // SAFETY: the span is inside this allocation, which outlives the copy.
        unsafe { core::slice::from_raw_parts(self.host.add(offset).cast_const(), len).to_vec() }
    }

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

    pub fn write(&self, offset: usize, bytes: &[u8]) -> bool {
        if self.host.is_null() || offset + bytes.len() > self.bytes {
            return false;
        }
        // SAFETY: the span is inside this allocation, which outlives the copy.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), self.host.add(offset), bytes.len())
        }
        true
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        if self.host.is_null() {
            return;
        }
        match self.origin {
            #[cfg(feature = "rocm")]
            Origin::Allocated => {
                // SAFETY: the pointer came from this structure's own `cudaHostAlloc` and is freed exactly once.
                unsafe {
                    let _ = kernels_rocm::hip::rt::hipHostFree(self.host.cast());
                }
            }
            Origin::Registered => {
                #[cfg(feature = "rocm")]
                // SAFETY: the span is the one this structure registered.
                unsafe {
                    let _ = kernels_rocm::hip::rt::hipHostUnregister(self.host.cast());
                }
                unmap_anon(self.host, self.bytes);
            }
            #[cfg(not(feature = "rocm"))]
            Origin::Allocated => {}
        }
    }
}

#[cfg(unix)]
fn map_anon(bytes: usize) -> Option<*mut u8> {
    // SAFETY: a fresh private anonymous mapping of a stated length; no fd and
    // no offset are involved, and the pages belong to nobody else.
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
        None
    } else {
        Some(at.cast())
    }
}

#[cfg(unix)]
fn unmap_anon(at: *mut u8, bytes: usize) {
    // SAFETY: unmapping a mapping `map_anon` made, exactly once.
    unsafe {
        libc::munmap(at.cast(), bytes.max(1));
    }
}

#[cfg(windows)]
fn map_anon(bytes: usize) -> Option<*mut u8> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
    };
    // SAFETY: a fresh committed reservation of a stated length, owned by nobody else.
    let at = unsafe {
        VirtualAlloc(
            core::ptr::null_mut(),
            bytes.max(1),
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        )
    };
    if at.is_null() { None } else { Some(at.cast()) }
}

#[cfg(windows)]
fn unmap_anon(at: *mut u8, _bytes: usize) {
    use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
    // SAFETY: releasing a reservation `map_anon` made, exactly once.
    unsafe {
        VirtualFree(at.cast(), 0, MEM_RELEASE);
    }
}

#[derive(Debug)]
pub struct Pinning {
    host: *mut u8,
    bytes: usize,
}

// SAFETY: as `Pinned`'s — an address and a length, whose sole writer is the thread that holds it. `Pinning` hands out no aliases at all: `host` is the one door, and `lock` consumes the value.
unsafe impl Send for Pinning {}

impl Pinning {
    pub fn uninit(bytes: usize) -> Result<Pinning> {
        let Some(at) = map_anon(bytes) else {
            return Err(Fault::Device {
                call: "mmap",
                code: -1,
            });
        };
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(at.cast(), bytes.max(1), libc::MADV_HUGEPAGE)
        };
        Ok(Pinning { host: at, bytes })
    }

    #[must_use]
    pub fn host(&self) -> *mut u8 {
        self.host
    }

    pub fn lock(self) -> Result<Pinned> {
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            if self.bytes == 0 {
                return Pinned::mapped(0);
            }
            // SAFETY: the span is this structure's own mapping, live until the `forget` below hands it to the `Pinned`.
            let locked = unsafe {
                crate::device::ctx::check(
                    "hipHostRegister",
                    rt::hipHostRegister(self.host.cast(), self.bytes, rt::hipHostRegisterMapped),
                )
            };
            locked?;
            let mut device: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `host` is the mapping just registered.
            let asked = unsafe {
                crate::device::ctx::check(
                    "hipHostGetDevicePointer",
                    rt::hipHostGetDevicePointer(&raw mut device, self.host.cast(), 0),
                )
            };
            if asked.is_err() {
                // SAFETY: undoing the registration this call just made.
                unsafe {
                    let _ = rt::hipHostUnregister(self.host.cast());
                }
                asked?;
            }
            let host = self.host;
            let bytes = self.bytes;
            core::mem::forget(self);
            Ok(Pinned {
                host,
                device: device as u64,
                bytes,
                origin: Origin::Registered,
            })
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(Fault::Runtimeless)
        }
    }
}

impl Drop for Pinning {
    fn drop(&mut self) {
        unmap_anon(self.host, self.bytes);
    }
}

pub fn write_raw(at: u64, bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;

        // SAFETY: `at` is an address the caller resolved against an allocation it owns, and `bytes` is a live host slice for the duration of a synchronous copy.
        unsafe {
            crate::device::ctx::check(
                "hipMemcpy",
                rt::hipMemcpy(
                    at as *mut core::ffi::c_void,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    rt::hipMemcpyHostToDevice,
                ),
            )
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = at;
        Err(Fault::Runtimeless)
    }
}

pub fn stage_raw(stream: *mut core::ffi::c_void, dst: u64, bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;

        // SAFETY: `bytes` is a live host slice for the call; the destination span is the caller's own scratch.
        unsafe {
            crate::device::ctx::check(
                "hipMemcpyAsync",
                rt::hipMemcpyAsync(
                    dst as *mut core::ffi::c_void,
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    rt::hipMemcpyHostToDevice,
                    stream.cast(),
                ),
            )
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (stream, dst);
        Err(Fault::Runtimeless)
    }
}

pub fn copy_d2d(stream: *mut core::ffi::c_void, dst: u64, src: u64, bytes: usize) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;

        // SAFETY: both addresses are resolutions the caller made against allocations this shell owns for the load's lifetime, and the copy is enqueued on the same stream as the launches around it.
        unsafe {
            crate::device::ctx::check(
                "hipMemcpyAsync",
                rt::hipMemcpyAsync(
                    dst as *mut core::ffi::c_void,
                    src as *const core::ffi::c_void,
                    bytes,
                    rt::hipMemcpyDeviceToDevice,
                    stream.cast(),
                ),
            )
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (stream, dst, src);
        Err(Fault::Runtimeless)
    }
}

pub fn copy_any(stream: *mut core::ffi::c_void, dst: u64, src: u64, bytes: usize) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;

        // SAFETY: both addresses are resolutions the caller made against allocations this shell owns for the load's lifetime, and the copy is enqueued on a stream the caller keeps ordered around it.
        unsafe {
            crate::device::ctx::check(
                "hipMemcpyAsync",
                rt::hipMemcpyAsync(
                    dst as *mut core::ffi::c_void,
                    src as *const core::ffi::c_void,
                    bytes,
                    rt::hipMemcpyDefault,
                    stream.cast(),
                ),
            )
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (stream, dst, src);
        Err(Fault::Runtimeless)
    }
}

pub fn zero_span(at: u64, len: usize) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;

        // SAFETY: the address is a span the caller resolved against an allocation this shell owns.
        unsafe {
            crate::device::ctx::check(
                "hipMemset",
                rt::hipMemset(at as *mut core::ffi::c_void, 0, len),
            )
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = at;
        Err(Fault::Runtimeless)
    }
}

pub fn zero_span_on(stream: *mut core::ffi::c_void, at: u64, len: usize) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;

        // SAFETY: as `zero_span`, and the caller keeps the span alive past the enqueue.
        unsafe {
            crate::device::ctx::check(
                "hipMemsetAsync",
                rt::hipMemsetAsync(at as *mut core::ffi::c_void, 0, len, stream.cast()),
            )
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (stream, at);
        Err(Fault::Runtimeless)
    }
}

pub fn copy_d2h(at: u64, into: &mut [u8]) -> Result<()> {
    if into.is_empty() {
        return Ok(());
    }
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;

        // SAFETY: the address is a checked span and `into` is a live host slice of exactly the length copied.
        unsafe {
            crate::device::ctx::check(
                "hipMemcpy",
                rt::hipMemcpy(
                    into.as_mut_ptr().cast(),
                    at as *const core::ffi::c_void,
                    into.len(),
                    rt::hipMemcpyDeviceToHost,
                ),
            )
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = at;
        Err(Fault::Runtimeless)
    }
}

#[must_use]
pub fn free_bytes() -> Option<u64> {
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;

        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: two live locals; the call only writes them.
        let asked = unsafe { rt::hipMemGetInfo(&raw mut free, &raw mut total) };
        (asked == rt::hipSuccess).then_some(free as u64)
    }
    #[cfg(not(feature = "rocm"))]
    {
        None
    }
}

#[must_use]
pub fn is_host_pointer(at: u64) -> bool {
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;
        // SAFETY: a zeroed attribute record the call fills; the query reads nothing else.
        let mut attrs: rt::PointerAttributes = unsafe { core::mem::zeroed() };
        let asked =
            unsafe { rt::hipPointerGetAttributes(&raw mut attrs, at as *const core::ffi::c_void) };
        asked == rt::hipSuccess && attrs.type_ == rt::hipMemoryTypeHost
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = at;
        false
    }
}

#[must_use]
pub fn is_capturing(stream: *mut core::ffi::c_void) -> bool {
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;
        let mut status = rt::hipStreamCaptureStatusNone;
        // SAFETY: a live stream handle and a local the query writes.
        let asked = unsafe { rt::hipStreamIsCapturing(stream.cast(), &raw mut status) };
        asked == rt::hipSuccess && status != rt::hipStreamCaptureStatusNone
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = stream;
        false
    }
}

#[must_use]
pub fn raw_alloc(bytes: usize) -> Option<u64> {
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;
        let mut fresh: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: a local the call fills.
        let code = unsafe { rt::hipMalloc(&raw mut fresh, bytes) };
        (code == rt::hipSuccess && !fresh.is_null()).then_some(fresh as u64)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = bytes;
        None
    }
}

pub fn raw_free(at: u64) {
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;
        // SAFETY: an address `raw_alloc` returned and nobody else freed.
        let _ = unsafe { rt::hipFree(at as *mut core::ffi::c_void) };
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = at;
    }
}

/// The largest copy [`Buffer::stage`] bounces through the pinned ring.
#[cfg(feature = "rocm")]
const RING_SLOT_BYTES: usize = 1 << 20;

/// Slots in a device's ring; a slot is written again only once the copy
/// that last read it has landed, which with this many is long past.
#[cfg(feature = "rocm")]
const RING_SLOTS: usize = 32;

#[cfg(feature = "rocm")]
struct StageRing {
    slots: Vec<(Pinned, kernels_rocm::hip::rt::Event, bool)>,
    next: usize,
}

// SAFETY: the ring is only touched under its mutex; the events and pinned
// spans it owns are process-wide HIP objects.
#[cfg(feature = "rocm")]
unsafe impl Send for StageRing {}

/// A host-to-device copy out of pageable memory past HIP's own small-copy
/// staging waits for the stream to drain before it returns, so a fire that
/// stages a large table would wait out the frame still in flight. Copied
/// into a pinned slot first, the copy is asynchronous for real. Returns
/// `false` (and copies nothing) for a copy the ring does not take: one too
/// large for a slot, or one into a stream being captured, whose recorded
/// copy must not read a slot the ring will reuse.
#[cfg(feature = "rocm")]
fn staged_through_ring(dst: u64, bytes: &[u8], stream: *mut core::ffi::c_void) -> Result<bool> {
    use kernels_rocm::hip::rt;
    use std::collections::HashMap;
    use std::sync::Mutex;

    if bytes.len() > RING_SLOT_BYTES {
        return Ok(false);
    }
    let mut capturing: core::ffi::c_int = 0;
    // SAFETY: writes one int for a live stream.
    unsafe {
        crate::device::ctx::check(
            "hipStreamIsCapturing",
            rt::hipStreamIsCapturing(stream.cast(), &raw mut capturing),
        )?
    };
    if capturing != 0 {
        return Ok(false);
    }
    let mut device: core::ffi::c_int = 0;
    // SAFETY: writes one int.
    unsafe { crate::device::ctx::check("hipGetDevice", rt::hipGetDevice(&raw mut device))? };

    static RINGS: Mutex<Option<HashMap<core::ffi::c_int, StageRing>>> = Mutex::new(None);
    let mut rings = RINGS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rings = rings.get_or_insert_with(HashMap::new);
    let ring = match rings.entry(device) {
        std::collections::hash_map::Entry::Occupied(held) => held.into_mut(),
        std::collections::hash_map::Entry::Vacant(free) => {
            let mut slots = Vec::with_capacity(RING_SLOTS);
            for _ in 0..RING_SLOTS {
                let pinned = Pinned::mapped_uninit(RING_SLOT_BYTES)?;
                let mut event: rt::Event = core::ptr::null_mut();
                // SAFETY: writes the event.
                unsafe {
                    crate::device::ctx::check(
                        "hipEventCreateWithFlags",
                        rt::hipEventCreateWithFlags(&raw mut event, rt::hipEventDisableTiming),
                    )?;
                }
                slots.push((pinned, event, false));
            }
            free.insert(StageRing { slots, next: 0 })
        }
    };
    let at = ring.next;
    ring.next = (ring.next + 1) % RING_SLOTS;
    let (pinned, event, used) = &mut ring.slots[at];
    // SAFETY: the event is this slot's; waiting on it is what frees the slot.
    if *used {
        unsafe {
            crate::device::ctx::check("hipEventSynchronize", rt::hipEventSynchronize(*event))?
        };
    }
    // SAFETY: the slot holds RING_SLOT_BYTES ≥ bytes.len(), and the copy that
    // last read it has landed; the new copy reads it until the event records.
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), pinned.host(), bytes.len());
        crate::device::ctx::check(
            "hipMemcpyAsync",
            rt::hipMemcpyAsync(
                dst as *mut core::ffi::c_void,
                pinned.host().cast_const().cast(),
                bytes.len(),
                rt::hipMemcpyHostToDevice,
                stream.cast(),
            ),
        )?;
        crate::device::ctx::check("hipEventRecord", rt::hipEventRecord(*event, stream.cast()))?;
    }
    *used = true;
    Ok(true)
}
