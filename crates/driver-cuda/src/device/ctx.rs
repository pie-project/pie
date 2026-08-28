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

use kernels_cuda::attn::plan::{Device, Toggles};
use kernels_cuda::{Ctx, Slabs};

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

/// One side stream: a second place to enqueue, and the companions an entry
/// firing on it needs.
///
/// **A cuBLAS HANDLE PER STREAM, AND IT IS NOT OPTIONAL.** A handle carries
/// the stream its GEMMs go on (`cublasSetStream_v2`), so one handle shared
/// between two streams would mean either re-pointing it per launch — a host
/// call inside a capture, ordering nothing — or every projection landing on
/// the main stream whatever the region said. Gemma forks a region containing
/// `linear.matmul` on the very first layer, so this is the ordinary case
/// rather than a corner.
///
/// **NO COMMUNICATOR, DELIBERATELY.** P6 never puts a collective on a side
/// stream (decision #5: NCCL matches by call order), so a context here that
/// carried one would be a capability nothing may use. A collective fired on a
/// side stream answers `Ctx::comm`'s typed refusal, which is the diagnostic
/// worth having if the pass ever regresses.
#[cfg_attr(not(feature = "_cuda"), allow(dead_code))]
struct Side {
    stream: *mut c_void,
    cublas: *mut c_void,
    ctx: Ctx,
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
    /// P6's side streams, in stream order: `side[0]` is stream 1. Empty until
    /// [`Context::open_lanes`] is told how many the artifact asked for, and
    /// empty forever for an artifact that forked nothing.
    side: Vec<Side>,
    /// One `cudaEvent_t` per `model_compiler::EventId`, created once at load.
    events: Vec<crate::device::graph::Event>,
    /// **THIS SHELL'S SCRATCH SLABS, AND NOBODY ELSE'S.**
    ///
    /// `kernels_cuda::Ctx::scratch` used to hand back a slab keyed by a
    /// static name alone, which made every workspace in the process one
    /// buffer: two shells staged into each other and both computed (build log
    /// 18), and two arms of a P6 fork group did the same, which is the whole
    /// reason [`crate::EXCLUSIVE`] existed (build log 24). An arena is one
    /// context's, every stream in it gets its own slab, and [`Drop`] gives
    /// the bytes back.
    slabs: Slabs,
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
                let slabs = Slabs::open();
                // The main stream is attached before anything fires on it,
                // which is what `Slabs::attach` asks for: growth broadcasts
                // across an arena's attached streams, and a stream that
                // arrives after a name has grown misses that name's slab.
                slabs.attach(stream);
                let ctx = Ctx::on(stream).with_cublas(cublas).with_slabs(slabs);
                // A probe that fails is not a load that fails: the facts have
                // a stated fallback, and the builders take them as data.
                let device = Device::probe(&ctx).unwrap_or(Device::L40S);
                Ok(Context {
                    ordinal,
                    stream,
                    cublas,
                    ctx,
                    slabs,
                    side: Vec::new(),
                    events: Vec::new(),
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

    /// **OPEN P6'S SIDE STREAMS AND ITS EVENTS**, once, at load, for the
    /// counts the artifact asked for (`CompiledModel::streams`).
    ///
    /// Called after `compile`, because how many streams a plan wants is
    /// something the plan decides — and called at LOAD rather than per fire,
    /// because a `cudaStreamCreate` inside a capture is exactly the host work
    /// `Graph::capture`'s thread-local mode exists to refuse.
    ///
    /// Idempotent in the direction that matters: asking for what is already
    /// open does nothing, and asking for fewer keeps what is there. An
    /// artifact that forked nothing asks for `(0, 0)` and this is a no-op with
    /// no handle created, which is the "pays nothing" half of the off arm.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] for a build with no runtime, [`Fault::Device`]
    /// for a stream, a handle or an event the runtime refused. A failure part
    /// way through leaves the streams already opened in place; they are
    /// destroyed with the context.
    pub fn open_lanes(&mut self, side: u32, events: u32) -> Result<()> {
        #[cfg(feature = "_cuda")]
        {
            use cudarc::cublas::sys as blas;
            use cudarc::runtime::sys as rt;

            while self.side.len() < side as usize {
                // SAFETY: every call takes a live local out-parameter, and the
                // handles below are the ones the calls before them produced.
                // This thread bound the device.
                let opened = unsafe {
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
                    // Attached at LOAD, before the first fire, because the
                    // warm pass runs eagerly on the main stream and its
                    // growth has to reach this one — see `Slabs::attach`.
                    self.slabs.attach(stream);
                    Side {
                        stream,
                        cublas,
                        ctx: Ctx::on(stream).with_cublas(cublas).with_slabs(self.slabs),
                    }
                };
                self.side.push(opened);
            }
            while self.events.len() < events as usize {
                self.events.push(crate::device::graph::Event::new()?);
            }
            Ok(())
        }
        #[cfg(not(feature = "_cuda"))]
        {
            if side == 0 && events == 0 {
                return Ok(());
            }
            let _ = (side, events);
            Err(Fault::Runtimeless)
        }
    }

    /// The side streams' handles, in stream order — `streams()[0]` is stream
    /// 1. Empty until [`open_lanes`](Context::open_lanes).
    #[must_use]
    pub fn side_streams(&self) -> Vec<*mut c_void> {
        self.side.iter().map(|side| side.stream).collect()
    }

    /// The kernel contexts for the side streams, in the same order.
    ///
    /// A `Run` takes this slice beside [`ctx`](Context::ctx) and picks by
    /// `Region::stream`; index 0 here is stream 1 there.
    #[must_use]
    pub fn side_ctx(&self) -> Vec<&Ctx> {
        self.side.iter().map(|side| &side.ctx).collect()
    }

    /// The events, indexed by `model_compiler::EventId`.
    #[must_use]
    pub fn events(&self) -> &[crate::device::graph::Event] {
        &self.events
    }

    /// How many side streams are open.
    #[must_use]
    pub fn lanes(&self) -> usize {
        self.side.len()
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
                // The events first: they name points on the streams below.
                self.events.clear();
                // Then the slabs, before the streams they were sized for:
                // freeing is what makes a SECOND shell in this process cost
                // what the first one did.
                self.slabs.release();
                for side in self.side.drain(..) {
                    if !side.cublas.is_null() {
                        let _ = cudarc::cublas::sys::cublasDestroy_v2(side.cublas.cast());
                    }
                    if !side.stream.is_null() {
                        let _ = cudarc::runtime::sys::cudaStreamDestroy(side.stream.cast());
                    }
                }
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
