//! The rocBLAS calls the linear and attention entries make: bf16/f16
//! operands, f32 accumulation, column-major as BLAS reads it.
#![allow(unsafe_code)]

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


/// Where a rocBLAS call's workspace lies: given the bytes the call asks,
/// the slab laid for it (`Ctx::blas_workspace`).
pub type Workspace<'a> = dyn FnMut(usize) -> Result<u64, crate::Error> + 'a;

/// The least workspace a handle is given, so it is always the caller's and
/// rocBLAS never allocates one of its own.
const WORKSPACE_LEAST: usize = 256;

/// Creates a rocBLAS handle holding no device memory of its own: rocBLAS
/// allocates a default workspace with a handle (128 MiB on gfx942) that no
/// call this crate makes asks for; it is given back at once, and each call
/// runs on the slab laid for it ([`with_workspace`]).
///
/// # Safety
/// `handle` must be writable.
pub unsafe fn create(handle: *mut Blas) -> BlasStatus {
    let Ok(blas) = sys::rocblas() else {
        return -1;
    };
    // SAFETY: the caller's `handle` contract.
    let created = unsafe { (blas.rocblas_create_handle)(handle) };
    if created != sys::BLAS_SUCCESS {
        return created;
    }
    // SAFETY: the handle just created; null and zero give its workspace back.
    unsafe { (blas.rocblas_set_workspace)(*handle, core::ptr::null_mut(), 0) }
}

/// The workspace `run` asks of `handle`, from rocBLAS's device-memory size
/// query: the call launches nothing.
///
/// # Safety
/// `handle` must be live; `run` makes one rocBLAS call on it.
unsafe fn asks(
    blas: &sys::Rocblas,
    handle: Blas,
    call: &'static str,
    run: &dyn Fn() -> BlasStatus,
) -> Result<usize, Refused> {
    const START: &str = "rocblas_start_device_memory_size_query";
    const STOP: &str = "rocblas_stop_device_memory_size_query";
    // SAFETY: the caller's handle contract.
    answered(START, unsafe { (blas.rocblas_start_device_memory_size_query)(handle) })?;
    let status = run();
    let mut bytes = 0usize;
    // SAFETY: as above; `bytes` is a live local.
    let stopped = unsafe { (blas.rocblas_stop_device_memory_size_query)(handle, &raw mut bytes) };
    if !matches!(
        status,
        sys::BLAS_SUCCESS | sys::BLAS_SIZE_INCREASED | sys::BLAS_SIZE_UNCHANGED
    ) {
        return Err(Refused { call, status });
    }
    answered(STOP, stopped)?;
    Ok(bytes)
}

/// Runs `run` on `handle` over the workspace it asks for, taken through
/// `workspace`: a size-only walk (`crate::sizing`) asks and runs nothing,
/// so the slab is laid at what the call takes and rocBLAS never allocates.
///
/// # Safety
/// `handle` must be live and bound to the stream `workspace`'s slab is
/// taken for; `run` makes one rocBLAS call on it.
unsafe fn with_workspace(
    blas: &sys::Rocblas,
    handle: Blas,
    call: &'static str,
    workspace: &mut Workspace<'_>,
    run: &dyn Fn() -> BlasStatus,
) -> Result<(), crate::Error> {
    // SAFETY: the caller's contract.
    let bytes = unsafe { asks(blas, handle, call, run) }?.max(WORKSPACE_LEAST);
    let at = workspace(bytes)?;
    if crate::sizing::active() {
        return Ok(());
    }
    // SAFETY: `at` is a laid slab of at least `bytes`, live while the
    // context that took it fires.
    answered("rocblas_set_workspace", unsafe {
        (blas.rocblas_set_workspace)(handle, at as usize as *mut c_void, bytes)
    })?;
    answered(call, run()).map_err(Into::into)
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
    // SAFETY: the caller guarantees `handle` and `stream` are live.
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
    workspace: &mut Workspace<'_>,
) -> Result<(), crate::Error> {
    const CALL: &str = "rocblas_gemm_ex";
    let blas = blas(CALL)?;
    // SAFETY: `ALPHA` and `BETA` are statics rocBLAS reads as host f32 scalars;
    // the caller guarantees `handle` is live and each operand covers its shape.
    // `C` is `D` with the same `ld` and type, the in-place form rocBLAS allows.
    let run = || unsafe {
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
    };
    // SAFETY: the caller guarantees `handle` is live and bound.
    unsafe { with_workspace(blas, handle, CALL, workspace, &run) }
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
    workspace: &mut Workspace<'_>,
) -> Result<(), crate::Error> {
    const CALL: &str = "rocblas_gemm_strided_batched_ex";
    let blas = blas(CALL)?;
    // SAFETY: as in [`gemm`], for each of the `batch` problems the caller
    // guarantees lie at the given strides.
    let run = || unsafe {
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
    };
    // SAFETY: the caller guarantees `handle` is live and bound.
    unsafe { with_workspace(blas, handle, CALL, workspace, &run) }
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
    workspace: &mut Workspace<'_>,
) -> Result<(), crate::Error> {
    const CALL: &str = "rocblas_gemm_batched_ex";
    let blas = blas(CALL)?;
    // SAFETY: as in [`gemm`]; the caller guarantees each operand's array holds
    // `batch` live device addresses.
    let run = || unsafe {
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
    };
    // SAFETY: the caller guarantees `handle` is live and bound.
    unsafe { with_workspace(blas, handle, CALL, workspace, &run) }
}

/// A rocBLAS handle owned by its creator, destroyed on drop.
pub struct Handle(Blas);

impl Handle {
    pub fn new() -> Result<Self, Refused> {
        const CALL: &str = "rocblas_create_handle";
        let mut handle: Blas = core::ptr::null_mut();
        // SAFETY: `handle` is the live local rocBLAS writes the new handle to.
        answered(CALL, unsafe { create(&raw mut handle) })?;
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
