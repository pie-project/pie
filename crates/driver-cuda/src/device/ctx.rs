//! The bound device: an ordinal, a stream, a cuBLAS handle, and the
//! [`Ctx`] every kernel entry fires on.
//!
//! ONE CONTEXT PER SHELL, BOUND ON THE THREAD THAT WILL FIRE. `cudaSetDevice`
//! is per-thread state, so binding somewhere other than where the fires
//! happen strands every later call on device 0 — the rewrite's shell learned
//! that and said so in a comment; this states it in the constructor instead.
//!
//! The cuBLAS handle is not optional decoration: `linear.matmul` reaches for
//! it through [`Ctx::cublas`] on every projection in the model, and a context
//! without one answers a typed refusal that would read as a missing kernel.

use core::ffi::c_void;

use kernels_cuda::Ctx;
use kernels_cuda::attn::plan::{Device, Toggles};

use crate::error::{Fault, Result};

/// Is there a CUDA device on this machine?
///
/// **A BUILD THAT NAMES CUDA IS NOT A MACHINE THAT HAS IT**, and the probe
/// has to survive both halves of that: no runtime library at all (cudarc is
/// built `fallback-dynamic-loading`, so a missing `libcudart` PANICS from
/// inside the shim rather than returning a code) and a library with no device
/// behind it. This is the door a GPU test knocks on before it asks for
/// anything; the idiom is `model-loader`'s `device_or_skip`, kept whole
/// because its comment is the reason it is written this way.
#[must_use]
pub fn present() -> bool {
    #[cfg(feature = "_cuda")]
    {
        // Only the FIRST runtime call is wrapped: past it the library is
        // known loaded, and catching panics any wider would turn a real
        // failure into a skip.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let probe = std::panic::catch_unwind(|| {
            // The count lives INSIDE the closure: a `&mut i32` captured
            // across a catch is not `UnwindSafe`, and it does not need to be.
            let mut count: i32 = 0;
            // SAFETY: `count` is a live local, and this is the process's
            // first cudarc call.
            let status = unsafe { cudarc::runtime::sys::cudaGetDeviceCount(&raw mut count) };
            (status, count)
        });
        std::panic::set_hook(hook);
        matches!(
            probe,
            Ok((cudarc::runtime::sys::cudaError::cudaSuccess, count)) if count > 0
        )
    }
    #[cfg(not(feature = "_cuda"))]
    {
        false
    }
}

/// One bound device and the stream this shell's whole life is enqueued on.
pub struct Context {
    ordinal: i32,
    stream: *mut c_void,
    // Read only by `Drop`, which has nothing to destroy in a build with no
    // runtime — the handle is null there and was never created.
    #[cfg_attr(not(feature = "_cuda"), allow(dead_code))]
    cublas: *mut c_void,
    ctx: Ctx,
    device: Device,
    toggles: Toggles,
    capability: (i32, i32),
}

impl Context {
    /// Bind `ordinal`, open a stream and a cuBLAS handle on it, and probe the
    /// facts every plan builder takes as an argument.
    ///
    /// The probes happen ONCE, here: `Device::probe` and `Toggles::from_env`
    /// are the two reads `kernels-cuda` deliberately refuses to do inside a
    /// builder, because purity is what makes a schedule reproducible. Their
    /// answers ride into every fire on [`FireBindings`](crate::FireBindings).
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] for a build with no runtime, [`Fault::Device`]
    /// for a runtime that refused.
    pub fn bind(ordinal: i32) -> Result<Context> {
        #[cfg(feature = "_cuda")]
        {
            use cudarc::cublas::sys as blas;
            use cudarc::runtime::sys as rt;

            // SAFETY: every call below takes a live local out-parameter and
            // the handles the ones before it produced. `cudaSetDevice` binds
            // THIS thread, which is the thread that will fire.
            unsafe {
                check("cudaSetDevice", rt::cudaSetDevice(ordinal))?;
                let mut stream: rt::cudaStream_t = core::ptr::null_mut();
                check("cudaStreamCreate", rt::cudaStreamCreate(&raw mut stream))?;

                let mut handle: blas::cublasHandle_t = core::ptr::null_mut();
                let status = blas::cublasCreate_v2(&raw mut handle);
                if status != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    rt::cudaStreamDestroy(stream);
                    return Err(Fault::Device {
                        call: "cublasCreate_v2",
                        code: status as i32,
                    });
                }
                let status = blas::cublasSetStream_v2(handle, stream.cast());
                if status != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    blas::cublasDestroy_v2(handle);
                    rt::cudaStreamDestroy(stream);
                    return Err(Fault::Device {
                        call: "cublasSetStream_v2",
                        code: status as i32,
                    });
                }

                let stream: *mut c_void = stream.cast();
                let cublas: *mut c_void = handle.cast();
                let ctx = Ctx::on(stream).with_cublas(cublas);
                // A probe that fails is not a load that fails: the facts have
                // a stated fallback, and the builders take them as data.
                let device = Device::probe(&ctx).unwrap_or(Device::L40S);
                Ok(Context {
                    ordinal,
                    stream,
                    cublas,
                    ctx,
                    device,
                    toggles: Toggles::from_env(),
                    capability: capability(ordinal),
                })
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = ordinal;
            Err(Fault::Runtimeless)
        }
    }

    /// **BIND THIS THREAD TO THIS CONTEXT'S DEVICE.** The other half of
    /// [`Context::bind`]'s own doctrine — one context per shell, bound on the
    /// thread that will fire — for the case where those are not the same
    /// thread.
    ///
    /// The engine loads a driver on the thread that boots the worker and then
    /// hands it to a lane thread that owns it for the rest of the process.
    /// `cudaSetDevice` is per-thread and did not travel with it. The RUNTIME
    /// api forgave that on device 0, where an unbound thread's default is the
    /// right one; the DRIVER api did not — `cuModuleLoadData` on a thread with
    /// no current context answers `CUDA_ERROR_INVALID_CONTEXT` (201), which is
    /// what every guest-program registration met.
    ///
    /// Since CUDA 12 `cudaSetDevice` initializes the device's primary context
    /// and makes it current, so this one call is the whole binding; the
    /// `cudaFree(0)` that used to be needed to force it is not, and is not
    /// written here as a cargo-culted second call.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] when the runtime refuses the ordinal.
    pub fn bind_thread(&self) -> Result<()> {
        #[cfg(feature = "_cuda")]
        {
            use cudarc::runtime::sys as rt;
            // SAFETY: an ordinal this context was already bound with.
            unsafe { check("cudaSetDevice", rt::cudaSetDevice(self.ordinal)) }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// The context every kernel entry fires on.
    #[must_use]
    pub fn ctx(&self) -> &Ctx {
        &self.ctx
    }

    /// The stream, for the memcpys this crate issues itself.
    #[must_use]
    pub fn stream(&self) -> *mut c_void {
        self.stream
    }

    /// Which device this is.
    #[must_use]
    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    /// The facts the plan builders take, probed once at bind.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    /// The operator toggles `plan_decode` takes, read once at bind.
    #[must_use]
    pub fn toggles(&self) -> Toggles {
        self.toggles
    }

    /// This device's compute capability, `(major, minor)`, probed once at bind.
    ///
    /// NVRTC's `--gpu-architecture` and the guest-program cubin cache's key
    /// are the only readers ([`program`](crate::program)): a cubin built for
    /// `sm_89` neither loads nor answers on an `sm_90` part, so the pair is
    /// probed at bind and carried rather than asked for per compile.
    #[must_use]
    pub fn capability(&self) -> (i32, i32) {
        self.capability
    }

    /// Wait for everything enqueued so far.
    ///
    /// The shell's, not the kernels': every entry in `kernels-cuda` is
    /// enqueue-only and `Ctx` has no sync at all (decision #15), so the one
    /// place a fire becomes observable is a caller that asks for it — reading
    /// logits back, or timing a decode.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for whatever the stream had queued.
    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "_cuda")]
        {
            // SAFETY: `self.stream` is live for this context's lifetime.
            unsafe {
                check(
                    "cudaStreamSynchronize",
                    cudarc::runtime::sys::cudaStreamSynchronize(self.stream.cast()),
                )
            }
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        #[cfg(feature = "_cuda")]
        {
            // SAFETY: the shell is being torn down, so nothing else holds
            // either handle; both were produced by this context's `bind`.
            unsafe {
                if !self.cublas.is_null() {
                    let _ = cudarc::cublas::sys::cublasDestroy_v2(self.cublas.cast());
                }
                if !self.stream.is_null() {
                    let _ = cudarc::runtime::sys::cudaStreamDestroy(self.stream.cast());
                }
            }
        }
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("ordinal", &self.ordinal)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// This device's compute capability, or `(0, 0)` when it cannot be read.
///
/// `(0, 0)` rather than a guess: it becomes `sm_00`, which NVRTC refuses by
/// name, and a refusal naming the architecture is recoverable in a way that a
/// cubin built for the wrong part is not.
fn capability(ordinal: i32) -> (i32, i32) {
    #[cfg(feature = "_cuda")]
    {
        use cudarc::runtime::sys as rt;

        let attribute = |which: rt::cudaDeviceAttr| -> i32 {
            let mut value = 0i32;
            // SAFETY: `value` is a live out-parameter and `ordinal` was
            // accepted by `cudaSetDevice` immediately above the caller.
            let status = unsafe { rt::cudaDeviceGetAttribute(&raw mut value, which, ordinal) };
            if status == rt::cudaError::cudaSuccess {
                value
            } else {
                0
            }
        };
        (
            attribute(rt::cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor),
            attribute(rt::cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor),
        )
    }
    #[cfg(not(feature = "_cuda"))]
    {
        let _ = ordinal;
        (0, 0)
    }
}

/// One runtime status, as a shell fault.
#[cfg(feature = "_cuda")]
pub(crate) fn check(
    call: &'static str,
    status: cudarc::runtime::sys::cudaError,
) -> Result<()> {
    if status == cudarc::runtime::sys::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Fault::Device {
            call,
            code: status as i32,
        })
    }
}
