//! The rocBLAS calls the linear and attention entries make: bf16/f16
//! operands, f32 accumulation, column-major as BLAS reads it.

use core::ffi::{CStr, c_void};

use super::sys::{self, Blas, BlasOp, BlasStatus, BlasType};

/// A failed rocBLAS call: the entry it was, and what rocBLAS answered.
#[derive(Clone, Debug)]
pub struct Refused {
    pub call: &'static str,
    pub status: BlasStatus,
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let text = sys::rocblas()
            .ok()
            // SAFETY: rocBLAS returns a static, NUL-terminated string.
            .map(|blas| unsafe { CStr::from_ptr((blas.rocblas_status_to_string)(self.status)) })
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        write!(f, "`{}` answered {} ({text})", self.call, self.status)
    }
}

fn blas(call: &'static str) -> Result<&'static sys::Rocblas, Refused> {
    sys::rocblas().map_err(|_| Refused { call, status: -1 })
}

fn answered(call: &'static str, status: BlasStatus) -> Result<(), Refused> {
    if status == sys::BLAS_SUCCESS {
        Ok(())
    } else {
        Err(Refused { call, status })
    }
}

/// One side of a gemm: a device address, its element type, its leading
/// dimension, and (for a strided batch) the distance between batches.
#[derive(Clone, Copy, Debug)]
pub struct Operand {
    pub ptr: u64,
    pub dtype: BlasType,
    pub ld: i32,
    pub stride: i64,
}

impl Operand {
    #[must_use]
    pub const fn new(ptr: u64, dtype: BlasType, ld: i32) -> Self {
        Self {
            ptr,
            dtype,
            ld,
            stride: 0,
        }
    }

    #[must_use]
    pub const fn strided(mut self, stride: i64) -> Self {
        self.stride = stride;
        self
    }

    const fn at(self) -> *const c_void {
        self.ptr as usize as *const c_void
    }
}

const ALPHA: f32 = 1.0;
const BETA: f32 = 0.0;

/// Binds `handle` to `stream`; every call below enqueues on the stream the
/// handle was last bound to.
///
/// # Safety
/// `handle` must be a live rocBLAS handle and `stream` a live HIP stream.
pub unsafe fn bind(handle: Blas, stream: sys::Stream) -> Result<(), Refused> {
    let blas = blas("rocblas_set_stream")?;
    answered("rocblas_set_stream", unsafe {
        (blas.rocblas_set_stream)(handle, stream)
    })
}

/// `D = op(A) · op(B)` with `C = D` and beta zero, f32 accumulation.
///
/// # Safety
/// Every operand must be a live device range for its shape, and `handle`
/// live and bound to the stream the caller launches on.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm(
    handle: Blas,
    op_a: BlasOp,
    op_b: BlasOp,
    m: i32,
    n: i32,
    k: i32,
    a: Operand,
    b: Operand,
    d: Operand,
) -> Result<(), Refused> {
    const CALL: &str = "rocblas_gemm_ex";
    let blas = blas(CALL)?;
    answered(CALL, unsafe {
        (blas.rocblas_gemm_ex)(
            handle,
            op_a,
            op_b,
            m,
            n,
            k,
            core::ptr::from_ref(&ALPHA).cast(),
            a.at(),
            a.dtype,
            a.ld,
            b.at(),
            b.dtype,
            b.ld,
            core::ptr::from_ref(&BETA).cast(),
            d.at(),
            d.dtype,
            d.ld,
            d.ptr as usize as *mut c_void,
            d.dtype,
            d.ld,
            sys::BLAS_F32,
            sys::BLAS_ALGO_STANDARD,
            0,
            0,
        )
    })
}

/// [`gemm`] over `batch` problems, each operand stepping by its `stride`.
///
/// # Safety
/// As [`gemm`], for every one of the `batch` problems.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_strided(
    handle: Blas,
    op_a: BlasOp,
    op_b: BlasOp,
    m: i32,
    n: i32,
    k: i32,
    a: Operand,
    b: Operand,
    d: Operand,
    batch: i32,
) -> Result<(), Refused> {
    const CALL: &str = "rocblas_gemm_strided_batched_ex";
    let blas = blas(CALL)?;
    answered(CALL, unsafe {
        (blas.rocblas_gemm_strided_batched_ex)(
            handle,
            op_a,
            op_b,
            m,
            n,
            k,
            core::ptr::from_ref(&ALPHA).cast(),
            a.at(),
            a.dtype,
            a.ld,
            a.stride,
            b.at(),
            b.dtype,
            b.ld,
            b.stride,
            core::ptr::from_ref(&BETA).cast(),
            d.at(),
            d.dtype,
            d.ld,
            d.stride,
            d.ptr as usize as *mut c_void,
            d.dtype,
            d.ld,
            d.stride,
            batch,
            sys::BLAS_F32,
            sys::BLAS_ALGO_STANDARD,
            0,
            0,
        )
    })
}

/// [`gemm`] over `batch` problems whose operands are device arrays of
/// pointers (each `Operand::ptr` is the array's address).
///
/// # Safety
/// As [`gemm`]; each pointer array must hold `batch` live device addresses.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_batched(
    handle: Blas,
    op_a: BlasOp,
    op_b: BlasOp,
    m: i32,
    n: i32,
    k: i32,
    a: Operand,
    b: Operand,
    d: Operand,
    batch: i32,
) -> Result<(), Refused> {
    const CALL: &str = "rocblas_gemm_batched_ex";
    let blas = blas(CALL)?;
    answered(CALL, unsafe {
        (blas.rocblas_gemm_batched_ex)(
            handle,
            op_a,
            op_b,
            m,
            n,
            k,
            core::ptr::from_ref(&ALPHA).cast(),
            a.at(),
            a.dtype,
            a.ld,
            b.at(),
            b.dtype,
            b.ld,
            core::ptr::from_ref(&BETA).cast(),
            d.at(),
            d.dtype,
            d.ld,
            d.ptr as usize as *mut c_void,
            d.dtype,
            d.ld,
            batch,
            sys::BLAS_F32,
            sys::BLAS_ALGO_STANDARD,
            0,
            0,
        )
    })
}

/// A rocBLAS handle owned by its creator, destroyed on drop.
pub struct Handle(Blas);

impl Handle {
    pub fn new() -> Result<Self, Refused> {
        const CALL: &str = "rocblas_create_handle";
        let blas = blas(CALL)?;
        let mut handle: Blas = core::ptr::null_mut();
        answered(CALL, unsafe {
            (blas.rocblas_create_handle)(&raw mut handle)
        })?;
        Ok(Self(handle))
    }

    #[must_use]
    pub fn raw(&self) -> Blas {
        self.0
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Ok(blas) = sys::rocblas() {
            // SAFETY: the handle came from `rocblas_create_handle`, destroyed once.
            unsafe { (blas.rocblas_destroy_handle)(self.0) };
        }
    }
}

/// A rocBLAS handle whose lifetime the caller manages: [`destroy_raw`] it
/// exactly once.
pub fn create_raw() -> Result<Blas, Refused> {
    const CALL: &str = "rocblas_create_handle";
    let blas = blas(CALL)?;
    let mut handle: Blas = core::ptr::null_mut();
    answered(CALL, unsafe {
        (blas.rocblas_create_handle)(&raw mut handle)
    })?;
    Ok(handle)
}

/// # Safety
/// `handle` came from [`create_raw`] and is not used again.
pub unsafe fn destroy_raw(handle: Blas) {
    if let Ok(blas) = sys::rocblas() {
        // SAFETY: the caller hands over a live handle, once.
        unsafe { (blas.rocblas_destroy_handle)(handle) };
    }
}
