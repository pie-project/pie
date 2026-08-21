
use core::ffi::c_void;

use super::{Refused, Result, Stream};

struct Block(*mut c_void);

impl Drop for Block {
    fn drop(&mut self) {
        free(self.0);
    }
}

fn free(ptr: *mut c_void) {
    if !ptr.is_null() {

        unsafe {
            let _ = cudarc::runtime::sys::cudaFree(ptr);
        }
    }
}

pub struct Scratch {

    live: Vec<Block>,
}

impl Scratch {

    #[must_use]
    pub const fn new() -> Self {
        Self { live: Vec::new() }
    }

    fn raw(&mut self, count: usize, width: usize, what: &'static str) -> Result<*mut c_void> {
        let bytes = count
            .checked_mul(width)
            .ok_or_else(|| Refused::new(what, "allocation size overflowed"))?;
        let pointer = alloc(bytes, what)?;
        self.live.push(Block(pointer));
        Ok(pointer)
    }

    pub fn bf16(&mut self, count: usize) -> Result<*mut c_void> {
        self.raw(count, 2, "tower scratch (bf16)")
    }

    pub fn f32s(&mut self, count: usize) -> Result<*mut c_void> {
        self.raw(count, 4, "tower scratch (f32)")
    }

    pub fn zeroed_f32s(&mut self, count: usize, stream: Stream<'_>) -> Result<*mut c_void> {
        let what = "tower scratch (f32)";
        let bytes = count
            .checked_mul(4)
            .ok_or_else(|| Refused::new(what, "allocation size overflowed"))?;
        let pointer = self.raw(count, 4, what)?;

        unsafe { fill_raw_span(pointer, 0, bytes, stream) }?;
        Ok(pointer)
    }

    pub fn upload_f32s(&mut self, src: &[f32], stream: Stream<'_>) -> Result<*mut c_void> {

        let bytes = unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<u8>(), src.len() * 4) };
        self.upload(bytes, stream, "tower scratch (f32 upload)")
    }

    pub fn upload_i32s(&mut self, src: &[i32], stream: Stream<'_>) -> Result<*mut c_void> {

        let bytes = unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<u8>(), src.len() * 4) };
        self.upload(bytes, stream, "tower scratch (i32 upload)")
    }

    pub fn upload_bytes(&mut self, src: &[u8], stream: Stream<'_>) -> Result<*mut c_void> {
        self.upload(src, stream, "tower scratch (byte upload)")
    }

    fn upload(
        &mut self,
        bytes: &[u8],
        stream: Stream<'_>,
        what: &'static str,
    ) -> Result<*mut c_void> {
        let pointer = alloc(bytes.len(), what)?;
        self.live.push(Block(pointer));

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

fn alloc(bytes: usize, what: &'static str) -> Result<*mut c_void> {
    use cudarc::runtime::sys::{cudaError, cudaMalloc};
    let mut raw: *mut c_void = core::ptr::null_mut();

    let code = unsafe { cudaMalloc(&raw mut raw, bytes.max(1)) };
    if code != cudaError::cudaSuccess || raw.is_null() {
        return Err(Refused::new(what, format!("cudaMalloc({bytes}) failed with {code:?}")));
    }
    Ok(raw)
}

fn check(code: cudarc::runtime::sys::cudaError, what: &'static str) -> Result<()> {
    if code == cudarc::runtime::sys::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Refused::new(what, format!("{code:?}")))
    }
}

pub unsafe fn read_raw_span(src: *const c_void, dst: &mut [u8], stream: Stream<'_>) -> Result<()> {
    if dst.is_empty() {
        return Ok(());
    }
    {
        use cudarc::runtime::sys::{cudaMemcpyAsync, cudaMemcpyKind};
        check(
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

pub unsafe fn write_raw_span(dst: *mut c_void, src: &[u8], stream: Stream<'_>) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }
    {
        use cudarc::runtime::sys::{cudaMemcpyAsync, cudaMemcpyKind};
        check(
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
            unsafe { cudarc::runtime::sys::cudaMemsetAsync(dst, i32::from(value), bytes, stream.as_raw().cast()) },
            "cudaMemsetAsync (fill_raw_span)",
        )
    }
}
