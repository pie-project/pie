//! The runtime calls this smoke makes directly, and nothing more.
//!
//! `driver-cuda` owns a whole device layer -- a capture-safe allocator, an
//! elastic VMM arena, stream priorities, graph nodes. None of that is what
//! this binary is proving. What it needs is: bind a device, take a stream,
//! `cudaMalloc` a slab, move bytes both ways, and hand `kernels-cuda` a
//! cuBLAS handle. Eight calls, spelled once, so the executor above reads as
//! the executor and not as CUDA.
//!
//! The cuBLAS trio mirrors `driver-cuda/src/device/cublas.rs:33-71`
//! exactly -- create, bind the stream, `CUBLAS_TENSOR_OP_MATH` -- because
//! `gemm::matmul`'s bf16 arm is a `cublasGemmEx` that reads the math mode.

use core::ffi::c_void;

use cudarc::runtime::sys as rt;

/// A device allocation, freed when the run ends.
///
/// NOT AN ARENA AND NOT POOLED. Every slab here lives for the whole fire --
/// the weights, the arena, the caches, the plan workspaces -- so `Drop` is
/// the only lifetime anything needs. The reuse `model_compiler::program`
/// does is INSIDE the activation slab and none of this type's business: the
/// walk hands out offsets into one block, and one block is what this
/// allocates.
#[derive(Debug)]
pub struct Slab {
    ptr: *mut c_void,
    bytes: usize,
}

impl Slab {
    /// `cudaMalloc`, zeroed. Zero bytes answers a null slab rather than
    /// failing: a cache row a lane never touches is a legitimate nothing.
    pub fn zeroed(bytes: usize, stream: *mut c_void) -> Result<Slab, String> {
        let slab = Slab::raw(bytes)?;
        if bytes > 0 {
            let code = unsafe { rt::cudaMemsetAsync(slab.ptr, 0, bytes, stream.cast()) };
            check(code, "cudaMemsetAsync")?;
        }
        Ok(slab)
    }

    /// `cudaMalloc`, contents undefined -- for a slab whose every byte the
    /// next call overwrites.
    pub fn raw(bytes: usize) -> Result<Slab, String> {
        if bytes == 0 {
            return Ok(Slab {
                ptr: core::ptr::null_mut(),
                bytes: 0,
            });
        }
        let mut ptr: *mut c_void = core::ptr::null_mut();
        let code = unsafe { rt::cudaMalloc(&raw mut ptr, bytes) };
        check(code, "cudaMalloc")?;
        if ptr.is_null() {
            return Err(format!("cudaMalloc({bytes}) answered null"));
        }
        Ok(Slab { ptr, bytes })
    }

    /// A slab holding `src`, uploaded on `stream`.
    pub fn of(src: &[u8], stream: *mut c_void) -> Result<Slab, String> {
        let slab = Slab::raw(src.len())?;
        upload(slab.ptr, src, stream)?;
        Ok(slab)
    }

    #[must_use]
    pub const fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { rt::cudaFree(self.ptr) };
        }
    }
}

/// Bind device `ordinal` and force its primary context up.
///
/// `driver-cuda/src/device/device.rs:51-58` says why the `cudaFree(null)` is
/// here: `cudaSetDevice` only records a thread-local ordinal, and the
/// driver-API calls the JIT makes need the primary context to exist.
pub fn bind(ordinal: i32) -> Result<(), String> {
    check(unsafe { rt::cudaSetDevice(ordinal) }, "cudaSetDevice")?;
    check(
        unsafe { rt::cudaFree(core::ptr::null_mut()) },
        "cudaFree(null)",
    )
}

/// A non-blocking stream at the default priority.
pub fn stream() -> Result<*mut c_void, String> {
    let mut raw: rt::cudaStream_t = core::ptr::null_mut();
    check(
        unsafe { rt::cudaStreamCreateWithPriority(&raw mut raw, rt::cudaStreamNonBlocking, 0) },
        "cudaStreamCreateWithPriority",
    )?;
    Ok(raw.cast())
}

pub fn sync(stream: *mut c_void) -> Result<(), String> {
    check(
        unsafe { rt::cudaStreamSynchronize(stream.cast()) },
        "cudaStreamSynchronize",
    )
}

pub fn upload(dst: *mut c_void, src: &[u8], stream: *mut c_void) -> Result<(), String> {
    if src.is_empty() {
        return Ok(());
    }
    let code = unsafe {
        rt::cudaMemcpyAsync(
            dst,
            src.as_ptr().cast(),
            src.len(),
            rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
            stream.cast(),
        )
    };
    check(code, "cudaMemcpyAsync H2D")
}

pub fn download(dst: &mut [u8], src: *const c_void, stream: *mut c_void) -> Result<(), String> {
    if dst.is_empty() {
        return Ok(());
    }
    let code = unsafe {
        rt::cudaMemcpyAsync(
            dst.as_mut_ptr().cast(),
            src,
            dst.len(),
            rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            stream.cast(),
        )
    };
    check(code, "cudaMemcpyAsync D2H")?;
    sync(stream)
}

/// The device-to-device copy an `InOut` point forces.
///
/// `model_compiler::program` mints a FRESH rectangle for every result,
/// including the results of the points whose declaration marks an operand
/// `InOut` (`norm.residual_add`, `rope.partial`, `gate.sigmoid_mul`). The
/// kernel writes through the pointer it is handed, so the executor has to
/// put the operand's bytes in the result's rectangle before it fires --
/// otherwise the launch mutates the operand and the result's column stays
/// whatever the arena held. Aliasing the two instead would be a liveness
/// claim this executor has no analysis to make.
pub fn copy(dst: *mut c_void, src: *const c_void, bytes: usize, stream: *mut c_void) -> Result<(), String> {
    if bytes == 0 {
        return Ok(());
    }
    let code = unsafe {
        rt::cudaMemcpyAsync(
            dst,
            src,
            bytes,
            rt::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            stream.cast(),
        )
    };
    check(code, "cudaMemcpyAsync D2D")
}

/// One cuBLAS handle bound to `stream`, in tensor-op math mode.
///
/// The three calls and their order are `driver-cuda/src/device/cublas.rs`'s
/// `CublasHandle::create`: create, `cublasSetStream_v2`, then
/// `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)`.
pub fn cublas(stream: *mut c_void) -> Result<*mut c_void, String> {
    use cudarc::cublas::sys::{
        cublasCreate_v2, cublasHandle_t, cublasMath_t, cublasSetMathMode, cublasSetStream_v2,
        cublasStatus_t,
    };
    let mut h: cublasHandle_t = core::ptr::null_mut();
    let status = unsafe { cublasCreate_v2(&raw mut h) };
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(format!("cublasCreate_v2 -> {status:?}"));
    }
    let status = unsafe { cublasSetStream_v2(h, stream.cast()) };
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(format!("cublasSetStream_v2 -> {status:?}"));
    }
    let status = unsafe { cublasSetMathMode(h, cublasMath_t::CUBLAS_TENSOR_OP_MATH) };
    if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(format!("cublasSetMathMode -> {status:?}"));
    }
    Ok(h.cast())
}

fn check(code: rt::cudaError, call: &str) -> Result<(), String> {
    if code == rt::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(format!("{call} -> {code:?}"))
    }
}
