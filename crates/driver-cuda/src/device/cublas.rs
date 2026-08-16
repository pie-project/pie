//! The driver's cuBLAS handle — gate-cublas. Wraps a raw `cublasHandle_t`:
//! create, set-stream (only if given), then tensor-op math mode. A failure
//! after create drops the token without destroying it — reproduced C++
//! behaviour, benign since create is fatal.

use std::ffi::c_void;

/// What the wrapper asks of cuBLAS — the real impl calls it, tests answer with tokens.
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
    /// `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)`: bf16 x fp32, or the failing status.
    fn set_math_mode_tensor_op(&mut self, handle: &Self::Handle) -> Result<(), i32>;
}

/// The live [`CublasOps`] via cudarc's dynamic cuBLAS, so the toolkit-free build links.
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
        // Status ignored — a failed destroy on teardown has nowhere to report.
        let _ = unsafe { cudarc::cublas::sys::cublasDestroy_v2(handle) };
    }

    // Safe by design: recorders never touch the pointer; the live caller
    // owns the stream it hands in.
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

/// Why construction or a rebind refused — the C++ `check`'s `runtime_error`.
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

/// The RAII handle; `release` is the destructor, as everywhere in this crate.
#[derive(Debug)]
pub struct CublasHandle<H> {
    handle: Option<H>,
}

impl<H> CublasHandle<H> {
    /// Create, bind the stream if one was given, then tensor-op math mode.
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

    /// Rebind the stream.
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

    /// The stream currently bound, kept so capture shares cuBLAS's loose launches.
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
    /// Leak check, skipped while panicking — a nested panic in `Drop` buries the assertion.
    fn drop(&mut self) {
        debug_assert!(
            self.handle.is_none() || std::thread::panicking(),
            "CublasHandle dropped without release()"
        );
    }
}
