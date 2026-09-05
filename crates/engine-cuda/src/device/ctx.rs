//! The bound device: an ordinal, a stream, a cuBLAS handle, and the
//! [`Ctx`] every kernel entry fires on.

use core::ffi::c_void;

use kernels_cuda::attn::plan::{Device, Toggles};
use kernels_cuda::{Ctx, Slabs};

use crate::error::{Fault, Result};

/// Whether a CUDA device is present. Must survive both a missing runtime
/// library (cudarc's fallback-dynamic-loading panics on a missing
/// `libcudart` rather than returning a code) and a library with no device.
#[must_use]
pub fn present() -> bool {
    #[cfg(feature = "cuda")]
    {
        // only the first runtime call is wrapped; later ones are known-loaded.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let probe = std::panic::catch_unwind(|| {
            // count lives inside the closure: a captured &mut i32 is not UnwindSafe.
            let mut count: i32 = 0;
            // SAFETY: count is a live local, and this is the process's first cudarc call.
            let status = unsafe { cudarc::runtime::sys::cudaGetDeviceCount(&raw mut count) };
            (status, count)
        });
        std::panic::set_hook(hook);
        matches!(
            probe,
            Ok((cudarc::runtime::sys::cudaError::cudaSuccess, count)) if count > 0
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// A second stream to enqueue on, with its own cuBLAS handle (a handle is
/// bound to one stream via `cublasSetStream_v2`) and no communicator — a
/// collective fired on a side stream answers `Ctx::comm`'s typed refusal.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Side {
    stream: *mut c_void,
    cublas: *mut c_void,
    ctx: Ctx,
}

/// One bound device and the stream this shell's whole life is enqueued on.
pub struct Context {
    ordinal: i32,
    stream: *mut c_void,
    /// The notify stream: carries only callbacks, never work.
    /// `cudaLaunchHostFunc` holds its stream, so a callback enqueued on the
    /// compute stream would block the next wave behind it; non-blocking, so
    /// it never orders against the default stream.
    notify: *mut c_void,
    // Read only by `Drop`, which has nothing to destroy in a build with no
    // runtime — the handle is null there and was never created.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    cublas: *mut c_void,
    ctx: Ctx,
    /// Side streams, in order (`side[0]` is stream 1). Empty until
    /// [`Context::open_lanes`].
    side: Vec<Side>,
    /// The stream a conditional body is captured on, opened by
    /// [`Context::open_conditional`]. Not a side stream: no region names it,
    /// and nothing is enqueued on it outside a `cuStreamBeginCaptureToGraph`.
    conditional: Option<Side>,
    /// One `cudaEvent_t` per `model_compiler::EventId`, created once at load.
    events: Vec<crate::device::graph::Event>,
    /// This context's own scratch slabs, not shared with any other context.
    slabs: Slabs,
    device: Device,
    toggles: Toggles,
    capability: (i32, i32),
}

impl Context {
    /// Binds `ordinal`, opens a stream and cuBLAS handle on it, and probes
    /// device facts once (`Device::probe`, `Toggles::from_env`) that ride
    /// into every fire via `FireBindings`.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] for a build with no runtime, [`Fault::Device`]
    /// for a runtime that refused.
    pub fn bind(ordinal: i32, comm: *mut c_void) -> Result<Context> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::cublas::sys as blas;
            use cudarc::runtime::sys as rt;

            // SAFETY: each call takes a live out-parameter and the handles
            // the prior calls produced. cudaSetDevice binds this thread.
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

                // notify stream, opened with the compute stream, never used for work.
                let mut notify: rt::cudaStream_t = core::ptr::null_mut();
                let status = rt::cudaStreamCreateWithFlags(
                    &raw mut notify,
                    1, // cudaStreamNonBlocking
                );
                if status != rt::cudaError::cudaSuccess {
                    blas::cublasDestroy_v2(handle);
                    rt::cudaStreamDestroy(stream);
                    return Err(Fault::Device {
                        call: "cudaStreamCreateWithFlags",
                        code: status as i32,
                    });
                }

                let stream: *mut c_void = stream.cast();
                let notify: *mut c_void = notify.cast();
                let cublas: *mut c_void = handle.cast();
                let slabs = Slabs::open();
                // main stream attached before anything fires on it; slabs
                // are keyed by (name, region), not stream, so this registers
                // nothing extra.
                slabs.attach(stream);
                let ctx = Ctx::on(stream).with_cublas(cublas).with_slabs(slabs);
                // SAFETY: `comm` is the rank's live communicator (or null),
                // owned by the boot for as long as this shell fires on it.
                let ctx = if comm.is_null() { ctx } else { ctx.with_comm(comm) };
                // a failed probe isn't a failed load: builders take the fallback as data.
                let device = Device::probe(&ctx).unwrap_or(Device::L40S);
                Ok(Context {
                    ordinal,
                    stream,
                    notify,
                    cublas,
                    ctx,
                    slabs,
                    side: Vec::new(),
                    conditional: None,
                    events: Vec::new(),
                    device,
                    toggles: Toggles::from_env(),
                    capability: capability(ordinal),
                })
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (ordinal, comm);
            Err(Fault::Runtimeless)
        }
    }

    /// Binds this thread to this context's device — for the case where
    /// `bind` and the fire don't happen on the same thread. `cudaSetDevice`
    /// is per-thread and doesn't travel across a handoff.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] when the runtime refuses the ordinal.
    pub fn bind_thread(&self) -> Result<()> {
        bind_thread(self.ordinal)
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

    /// The settlement stream — see [`Context::notify`]'s own doc.
    #[must_use]
    pub fn notify_stream(&self) -> *mut c_void {
        self.notify
    }

    /// Runs `work` on the host once everything enqueued on the notify stream
    /// so far has completed.
    ///
    /// Fixed to the notify stream: a caller orders the callback by making
    /// the notify stream wait on a compute-stream event, never by enqueuing
    /// the callback on the compute stream itself.
    ///
    /// `work` runs on a driver-owned thread: it must make no CUDA call, must
    /// not block for long, and must not panic across the FFI boundary (the
    /// trampoline below catches it).
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`], or [`Fault::Device`] for a launch the runtime
    /// refused — in which case `work` is dropped without running.
    pub fn host_fn(&self, work: Box<dyn FnOnce() + Send + 'static>) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            // double box is the ABI: Box<dyn FnOnce> is a fat pointer, void* is thin.
            let carried: *mut Box<dyn FnOnce() + Send + 'static> = Box::into_raw(Box::new(work));
            // SAFETY: carried is a live leaked allocation, reclaimed exactly
            // once — here on the failure path, or by the trampoline.
            let code = unsafe {
                rt::cudaLaunchHostFunc(
                    self.notify.cast(),
                    Some(host_fn_trampoline),
                    carried.cast(),
                )
            };
            if code != rt::cudaError::cudaSuccess {
                // SAFETY: the launch failed, so nothing else will ever see
                // this pointer; reclaiming it here avoids a leak.
                drop(unsafe { Box::from_raw(carried) });
                return Err(Fault::Device {
                    call: "cudaLaunchHostFunc",
                    code: code as i32,
                });
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = work;
            Err(Fault::Runtimeless)
        }
    }

    /// Opens the side streams and events, once, at load, for the counts the
    /// artifact asked for (`CompiledModel::streams`).
    ///
    /// Called after `compile` (the plan decides the counts) and at load
    /// rather than per fire, since a `cudaStreamCreate` inside a capture is
    /// host work the capture mode refuses.
    ///
    /// Idempotent: opening more grows what's there; asking for fewer is a
    /// no-op.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] for a build with no runtime, [`Fault::Device`]
    /// for a stream, handle or event the runtime refused. A partial failure
    /// leaves what was already opened in place; it is destroyed with the
    /// context.
    pub fn open_lanes(&mut self, side: u32, events: u32) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            while self.side.len() < side as usize {
                let opened = self.open_side()?;
                self.side.push(opened);
            }
            while self.events.len() < events as usize {
                self.events.push(crate::device::graph::Event::new()?);
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            if side == 0 && events == 0 {
                return Ok(());
            }
            let _ = (side, events);
            Err(Fault::Runtimeless)
        }
    }

    /// One more stream, its cuBLAS handle, and its seat in the scratch arena
    /// — the shape every companion stream this context opens has.
    #[cfg(feature = "cuda")]
    fn open_side(&mut self) -> Result<Side> {
        use cudarc::cublas::sys as blas;
        use cudarc::runtime::sys as rt;

        // SAFETY: each call takes a live out-parameter, using handles the
        // prior calls produced; this thread bound the device.
        unsafe {
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
            // attached at load, before the first fire; slabs are keyed by
            // region, so the warm pass and the capture share one block
            // whichever stream either fires on.
            self.slabs.attach(stream);
            Ok(Side {
                stream,
                cublas,
                ctx: Ctx::on(stream).with_cublas(cublas).with_slabs(self.slabs),
            })
        }
    }

    /// Opens the stream a conditional body is recorded on, or does nothing
    /// if one is already open. Asked once at load, only for artifacts with
    /// an `If`/`Switch` lowering; costs what a side stream does.
    ///
    /// # Errors
    ///
    /// [`Fault::Runtimeless`] for a build with no runtime, [`Fault::Device`]
    /// for a stream or handle the runtime refused.
    pub fn open_conditional(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            if self.conditional.is_none() {
                self.conditional = Some(self.open_side()?);
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    /// The conditional-body stream, or a null handle for a load that opened
    /// none — which is a load whose artifact holds no conditional region.
    #[must_use]
    pub fn conditional_stream(&self) -> *mut c_void {
        self.conditional
            .as_ref()
            .map_or(core::ptr::null_mut(), |side| side.stream)
    }

    /// The kernel context on that stream, for the launches a body holds.
    #[must_use]
    pub fn conditional_ctx(&self) -> Option<&Ctx> {
        self.conditional.as_ref().map(|side| &side.ctx)
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

    /// This device's compute capability, `(major, minor)`, probed once at
    /// bind. A cubin built for one `sm_XX` won't load or run on another, so
    /// this is carried rather than re-queried per compile.
    #[must_use]
    pub fn capability(&self) -> (i32, i32) {
        self.capability
    }

    /// Waits for everything enqueued so far, on both streams. Compute stream
    /// first (the work), then notify (what its completion triggers).
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for whatever either stream had queued.
    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            // SAFETY: both handles are live for this context's lifetime.
            unsafe {
                check(
                    "cudaStreamSynchronize",
                    cudarc::runtime::sys::cudaStreamSynchronize(self.stream.cast()),
                )?;
                check(
                    "cudaStreamSynchronize",
                    cudarc::runtime::sys::cudaStreamSynchronize(self.notify.cast()),
                )
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(Fault::Runtimeless)
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            // SAFETY: the shell is being torn down, so nothing else holds
            // either handle; both were produced by this context's `bind`.
            unsafe {
                // events first: they name points on the streams below.
                self.events.clear();
                // slabs freed before the streams they were sized for.
                self.slabs.release();
                for side in self.side.drain(..).chain(self.conditional.take()) {
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
                // notify stream destroyed last: a queued settlement callback
                // may still hold it.
                if !self.notify.is_null() {
                    let _ = cudarc::runtime::sys::cudaStreamDestroy(self.notify.cast());
                }
            }
        }
    }
}

/// The C entry point [`Context::host_fn`] hands the driver.
///
/// Reclaims the payload, runs it, and catches a panic rather than unwinding
/// into the CUDA driver, where unwinding is undefined behaviour.
#[cfg(feature = "cuda")]
extern "C" fn host_fn_trampoline(user: *mut c_void) {
    if user.is_null() {
        return;
    }
    // SAFETY: user is the pointer host_fn leaked; the driver calls this
    // exactly once per successful launch, so this reclaims it exactly once.
    let work = unsafe { Box::from_raw(user.cast::<Box<dyn FnOnce() + Send + 'static>>()) };
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || work()));
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("ordinal", &self.ordinal)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// This device's compute capability, or `(0, 0)` if it cannot be read —
/// which becomes `sm_00` and is refused by NVRTC by name, rather than
/// silently building for the wrong part.
fn capability(ordinal: i32) -> (i32, i32) {
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        let attribute = |which: rt::cudaDeviceAttr| -> i32 {
            let mut value = 0i32;
            // SAFETY: value is a live out-parameter; ordinal was just
            // accepted by cudaSetDevice above.
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
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinal;
        (0, 0)
    }
}

/// Which device this thread is on (`cudaGetDevice`). Reads the runtime
/// rather than a `Context` field because some callers hold neither.
///
/// # Errors
///
/// [`Fault::Runtimeless`] for a build with no runtime, [`Fault::Device`] when
/// the runtime has no current device to name.
pub fn current() -> Result<i32> {
    #[cfg(feature = "cuda")]
    {
        let mut ordinal = 0i32;
        // SAFETY: ordinal is a live out-parameter.
        unsafe {
            check(
                "cudaGetDevice",
                cudarc::runtime::sys::cudaGetDevice(&raw mut ordinal),
            )?;
        }
        Ok(ordinal)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(Fault::Runtimeless)
    }
}

/// Binds the calling thread to `ordinal` — [`Context::bind_thread`]'s body,
/// for a thread that carries a plain ordinal instead of a context.
///
/// # Errors
///
/// [`Fault::Runtimeless`] for a build with no runtime, [`Fault::Device`] when
/// the runtime refuses the ordinal.
pub fn bind_thread(ordinal: i32) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        // SAFETY: an ordinal the runtime itself answered with.
        unsafe {
            check(
                "cudaSetDevice",
                cudarc::runtime::sys::cudaSetDevice(ordinal),
            )
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinal;
        Err(Fault::Runtimeless)
    }
}

/// Waits for one stream, for callers holding a raw handle rather than a
/// [`Context`] — `record`'s last-resort seat stall.
///
/// # Errors
///
/// [`Fault::Runtimeless`] or [`Fault::Device`].
pub fn sync(stream: *mut c_void) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        // SAFETY: the handle is the caller's, live for the call.
        unsafe {
            check(
                "cudaStreamSynchronize",
                cudarc::runtime::sys::cudaStreamSynchronize(stream.cast()),
            )
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = stream;
        Err(Fault::Runtimeless)
    }
}

/// One runtime status, as a shell fault.
#[cfg(feature = "cuda")]
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
