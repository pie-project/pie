//! The driver's cuBLAS handle — gate-cublas.
//!
//! Ports `kernels-cuda`'s `CublasHandle` (`gemm/gemm.hpp`), which by the
//! campaign's own rule is driver C++ living in the kernels crate — the
//! launchers take a raw `cublasHandle_t`, only the driver ever constructs
//! the wrapper. The generated bodies call `.handle()` 3,590 times and
//! pass it into the gemm launchers; construction is once per engine.
//!
//! # A reproduced leak
//!
//! The C++ constructor runs `cublasCreate`, then `cublasSetStream` (only
//! when a stream was given), then `cublasSetMathMode(TENSOR_OP)` — and a
//! failure in the LAST step throws out of a half-built object, so the
//! destructor never runs and the created handle leaks. The port
//! reproduces that exactly: on a post-create failure the token is dropped
//! WITHOUT a destroy call. Recorded as a finding rather than fixed here
//! (the non-goals rule); in practice construction is at boot and the
//! failure is fatal, so the leak is one handle on a path that ends the
//! process.

use std::ffi::c_void;

/// What the wrapper asks of cuBLAS. The real implementation calls the
/// library; the parity test's recorder answers with tokens.
pub trait CublasOps {
    /// The opaque `cublasHandle_t`.
    type Handle;

    /// `cublasCreate`, or the failing status.
    fn create(&mut self) -> Result<Self::Handle, i32>;
    /// `cublasDestroy`.
    fn destroy(&mut self, handle: Self::Handle);
    /// `cublasSetStream`, or the failing status.
    fn set_stream(&mut self, handle: &Self::Handle, stream: *mut c_void) -> Result<(), i32>;
    /// `cublasGetStream`.
    fn get_stream(&mut self, handle: &Self::Handle) -> *mut c_void;
    /// `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)` — bf16 multiplies with
    /// fp32 accumulation — or the failing status.
    fn set_math_mode_tensor_op(&mut self, handle: &Self::Handle) -> Result<(), i32>;
}

/// The live [`CublasOps`] (retirement plan phase B): the five library calls
/// the wrapper names, through cudarc's dynamically-loaded cuBLAS — nothing
/// links, so the toolkit-free build survives, and the first call is what
/// resolves the library.
///
/// The statuses cross as the raw `cublasStatus_t` numbers, which is what
/// keeps [`CublasError`]'s Display identical to the C++ `check`'s message.
#[derive(Debug, Default, Clone, Copy)]
pub struct LiveCublas;

impl CublasOps for LiveCublas {
    type Handle = cudarc::cublas::sys::cublasHandle_t;

    fn create(&mut self) -> Result<Self::Handle, i32> {
        use cudarc::cublas::sys::{cublasCreate_v2, cublasStatus_t};
        let mut h: Self::Handle = std::ptr::null_mut();
        let status = unsafe { cublasCreate_v2(&mut h) };
        if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(h)
        } else {
            Err(status as i32)
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)] // seam method; recorders share it
    fn destroy(&mut self, handle: Self::Handle) {
        // The C++ destructor ignores the status — a failed destroy on
        // teardown has nowhere to report to.
        let _ = unsafe { cudarc::cublas::sys::cublasDestroy_v2(handle) };
    }

    // The seam's method is safe by design — the recorders that share the
    // trait never touch the pointer, and the live caller hands in a stream
    // it owns. Marking one impl `unsafe` would change the trait every
    // oracle drives.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn set_stream(&mut self, handle: &Self::Handle, stream: *mut c_void) -> Result<(), i32> {
        use cudarc::cublas::sys::{cublasSetStream_v2, cublasStatus_t};
        let status = unsafe { cublasSetStream_v2(*handle, stream.cast()) };
        if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(status as i32)
        }
    }

    fn get_stream(&mut self, handle: &Self::Handle) -> *mut c_void {
        let mut stream: cudarc::cublas::sys::cudaStream_t = std::ptr::null_mut();
        let _ = unsafe { cudarc::cublas::sys::cublasGetStream_v2(*handle, &mut stream) };
        stream.cast()
    }

    fn set_math_mode_tensor_op(&mut self, handle: &Self::Handle) -> Result<(), i32> {
        use cudarc::cublas::sys::{cublasMath_t, cublasSetMathMode, cublasStatus_t};
        let status = unsafe { cublasSetMathMode(*handle, cublasMath_t::CUBLAS_TENSOR_OP_MATH) };
        if status == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(status as i32)
        }
    }
}

/// Why construction or a rebind refused — the C++ `check`'s
/// `runtime_error`, message format included.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CublasError {
    /// The failing `cublasStatus_t`.
    pub status: i32,
    /// The call that failed, as `check` names it.
    pub expr: &'static str,
}

impl std::fmt::Display for CublasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cuBLAS error ({}): {}", self.status, self.expr)
    }
}

impl std::error::Error for CublasError {}

/// The RAII handle. See the module docs; `release` is the destructor, as
/// everywhere else in this crate.
#[derive(Debug)]
pub struct CublasHandle<H> {
    handle: Option<H>,
}

impl<H> CublasHandle<H> {
    /// The C++ constructor: create, bind the stream only when one was
    /// given, then tensor-op math mode. A failure after create drops the
    /// token without destroying it — the reproduced leak.
    pub fn create<O: CublasOps<Handle = H>>(
        ops: &mut O,
        stream: *mut c_void,
    ) -> Result<Self, CublasError> {
        let handle = ops.create().map_err(|status| CublasError {
            status,
            expr: "cublasCreate",
        })?;
        if !stream.is_null()
            && let Err(status) = ops.set_stream(&handle, stream)
        {
            return Err(CublasError {
                status,
                expr: "cublasSetStream",
            });
        }
        if let Err(status) = ops.set_math_mode_tensor_op(&handle) {
            return Err(CublasError {
                status,
                expr: "cublasSetMathMode",
            });
        }
        Ok(Self {
            handle: Some(handle),
        })
    }

    /// The raw handle the gemm launchers take.
    #[must_use]
    pub const fn handle(&self) -> Option<&H> {
        self.handle.as_ref()
    }

    /// Rebind the stream. Ports `set_stream`, throw included.
    pub fn set_stream<O: CublasOps<Handle = H>>(
        &self,
        ops: &mut O,
        stream: *mut c_void,
    ) -> Result<(), CublasError> {
        let h = self.handle.as_ref().expect("a created handle");
        ops.set_stream(h, stream).map_err(|status| CublasError {
            status,
            expr: "cublasSetStream",
        })
    }

    /// The stream currently bound — what keeps a body's loose kernel
    /// launches on the same stream as cuBLAS for graph capture.
    pub fn stream<O: CublasOps<Handle = H>>(&self, ops: &mut O) -> *mut c_void {
        let h = self.handle.as_ref().expect("a created handle");
        ops.get_stream(h)
    }

    /// The C++ destructor: destroy-if-nonnull.
    pub fn release<O: CublasOps<Handle = H>>(&mut self, ops: &mut O) {
        if let Some(h) = self.handle.take() {
            ops.destroy(h);
        }
    }
}

impl<H> Drop for CublasHandle<H> {
    fn drop(&mut self) {
        debug_assert!(
            self.handle.is_none(),
            "CublasHandle dropped without release()"
        );
    }
}
