//! The arena a tower walk allocates from, and the raw copies it needs.
//!
//! `driver-cuda`'s `device::Allocator` backed this before the walks moved
//! here; a tower cannot reach it now, and does not need to — a walk allocates
//! at the start, frees at the end, and never shares a block with the fire
//! path. The whole surface is `cudaMalloc`/`cudaFree` plus three async copies.

use core::ffi::c_void;

use super::{Refused, Result, Stream};

/// A device block owned by one walk.
struct Block(*mut c_void);

impl Drop for Block {
    fn drop(&mut self) {
        free(self.0);
    }
}

/// Release one block.
///
/// A failed free on a walk that is already unwinding has nowhere to report
/// to, so the status is dropped — `gemm::quant`'s rule.
fn free(ptr: *mut c_void) {
    if !ptr.is_null() {
        // SAFETY: the pointer came from `cudaMalloc` in `alloc` and is freed
        // once, here, because `Block` is neither `Clone` nor `Copy`.
        unsafe {
            let _ = cudarc::runtime::sys::cudaFree(ptr);
        }
    }
}

/// The tower's scratch arena: allocations handed out as raw pointers, valid
/// until this value drops — every walk drops it after synchronising the stream.
pub struct Scratch {
    /// Every block handed out, held so it is not freed early.
    live: Vec<Block>,
}

impl Scratch {
    /// An empty arena.
    #[must_use]
    pub const fn new() -> Self {
        Self { live: Vec::new() }
    }

    /// `count` elements of `width` bytes, uninitialised.
    fn raw(&mut self, count: usize, width: usize, what: &'static str) -> Result<*mut c_void> {
        let bytes = count
            .checked_mul(width)
            .ok_or_else(|| Refused::new(what, "allocation size overflowed"))?;
        let pointer = alloc(bytes, what)?;
        self.live.push(Block(pointer));
        Ok(pointer)
    }

    /// `count` bf16 elements.
    pub fn bf16(&mut self, count: usize) -> Result<*mut c_void> {
        self.raw(count, 2, "tower scratch (bf16)")
    }

    /// `count` fp32 elements.
    pub fn f32s(&mut self, count: usize) -> Result<*mut c_void> {
        self.raw(count, 4, "tower scratch (f32)")
    }

    /// `count` fp32 elements, zeroed on the stream.
    pub fn zeroed_f32s(&mut self, count: usize, stream: Stream<'_>) -> Result<*mut c_void> {
        let what = "tower scratch (f32)";
        let bytes = count
            .checked_mul(4)
            .ok_or_else(|| Refused::new(what, "allocation size overflowed"))?;
        let pointer = self.raw(count, 4, what)?;
        // SAFETY: `pointer` names `bytes` writable device bytes, just
        // allocated, and `stream` outlives the fill by the walk's contract.
        unsafe { fill_raw_span(pointer, 0, bytes, stream) }?;
        Ok(pointer)
    }

    /// A host `f32` run uploaded on the stream.
    pub fn upload_f32s(&mut self, src: &[f32], stream: Stream<'_>) -> Result<*mut c_void> {
        // SAFETY: `f32` is plain data with no padding; its bytes read as `u8`.
        let bytes = unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<u8>(), src.len() * 4) };
        self.upload(bytes, stream, "tower scratch (f32 upload)")
    }

    /// A host `i32` run uploaded on the stream.
    pub fn upload_i32s(&mut self, src: &[i32], stream: Stream<'_>) -> Result<*mut c_void> {
        // SAFETY: as `upload_f32s` — `i32` is plain data with no padding.
        let bytes = unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<u8>(), src.len() * 4) };
        self.upload(bytes, stream, "tower scratch (i32 upload)")
    }

    /// A host byte run uploaded on the stream, kept as bytes: the pixel plane
    /// arrives cut by a byte indptr, so there is no offset division here.
    pub fn upload_bytes(&mut self, src: &[u8], stream: Stream<'_>) -> Result<*mut c_void> {
        self.upload(src, stream, "tower scratch (byte upload)")
    }

    /// The shared body of the three uploads.
    fn upload(
        &mut self,
        bytes: &[u8],
        stream: Stream<'_>,
        what: &'static str,
    ) -> Result<*mut c_void> {
        let pointer = alloc(bytes.len(), what)?;
        self.live.push(Block(pointer));
        // SAFETY: `pointer` names `bytes.len()` writable device bytes, just
        // allocated, and `stream` outlives the copy by the walk's contract.
        unsafe { write_raw_span(pointer, bytes, stream) }?;
        Ok(pointer)
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Debug for Scratch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Scratch").field("blocks", &self.live.len()).finish()
    }
}

/// One device block, or the refusal that names the call and the byte count.
fn alloc(bytes: usize, what: &'static str) -> Result<*mut c_void> {
    use cudarc::runtime::sys::{cudaError, cudaMalloc};
    let mut raw: *mut c_void = core::ptr::null_mut();
    // SAFETY: `raw` is a live out-parameter for the call's duration.
    // `max(1)`: a zero-byte request is legal here and a null result is not,
    // so the block is rounded up rather than refused.
    let code = unsafe { cudaMalloc(&raw mut raw, bytes.max(1)) };
    if code != cudaError::cudaSuccess || raw.is_null() {
        return Err(Refused::new(what, format!("cudaMalloc({bytes}) failed with {code:?}")));
    }
    Ok(raw)
}

/// The status of one runtime call, as a refusal that names it.
fn check(code: cudarc::runtime::sys::cudaError, what: &'static str) -> Result<()> {
    if code == cudarc::runtime::sys::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Refused::new(what, format!("{code:?}")))
    }
}

/// A device-to-host read of a span the caller knows names live memory.
///
/// # Errors
///
/// The copy faulted.
///
/// # Safety
///
/// `src` must name at least `dst.len()` readable device bytes for the
/// duration of the copy, and `stream` must outlive it.
pub unsafe fn read_raw_span(src: *const c_void, dst: &mut [u8], stream: Stream<'_>) -> Result<()> {
    if dst.is_empty() {
        return Ok(());
    }
    {
        use cudarc::runtime::sys::{cudaMemcpyAsync, cudaMemcpyKind};
        check(
            // SAFETY: the caller's contract, forwarded.
            unsafe {
                cudaMemcpyAsync(
                    dst.as_mut_ptr().cast(),
                    src,
                    dst.len(),
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    stream.as_raw().cast(),
                )
            },
            "cudaMemcpyAsync (read_raw_span)",
        )
    }
}

/// A host-to-device write into a span the caller knows names live memory.
///
/// # Errors
///
/// The copy faulted.
///
/// # Safety
///
/// `dst` must name at least `src.len()` writable device bytes for the
/// duration of the copy, and `stream` must outlive it.
pub unsafe fn write_raw_span(dst: *mut c_void, src: &[u8], stream: Stream<'_>) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }
    {
        use cudarc::runtime::sys::{cudaMemcpyAsync, cudaMemcpyKind};
        check(
            // SAFETY: the caller's contract, forwarded.
            unsafe {
                cudaMemcpyAsync(
                    dst,
                    src.as_ptr().cast(),
                    src.len(),
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    stream.as_raw().cast(),
                )
            },
            "cudaMemcpyAsync (write_raw_span)",
        )
    }
}

/// A byte fill over a span the caller knows names live memory.
///
/// # Errors
///
/// The fill faulted.
///
/// # Safety
///
/// `dst` must name at least `bytes` writable device bytes for the duration
/// of the fill, and `stream` must outlive it.
pub unsafe fn fill_raw_span(
    dst: *mut c_void,
    value: u8,
    bytes: usize,
    stream: Stream<'_>,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    {
        check(
            // SAFETY: the caller's contract, forwarded.
            unsafe { cudarc::runtime::sys::cudaMemsetAsync(dst, i32::from(value), bytes, stream.as_raw().cast()) },
            "cudaMemsetAsync (fill_raw_span)",
        )
    }
}
