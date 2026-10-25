#![allow(unsafe_code)]
use core::ffi::c_void;

use kernels_cuda::attn::plan::{Device, Toggles};
use kernels_cuda::{Ctx, Slabs};

use crate::error::{Fault, Result};

#[must_use]
pub fn count() -> usize {
    #[cfg(feature = "device")]
    {
        if !present() {
            return 0;
        }
        let mut count: i32 = 0;
        // SAFETY: a live local; the runtime is loaded (`present` said so).
        let status = unsafe { crate::device::rt::cudaGetDeviceCount(&raw mut count) };
        if status == crate::device::rt::cudaError::cudaSuccess {
            usize::try_from(count).unwrap_or(0)
        } else {
            0
        }
    }
    #[cfg(not(feature = "device"))]
    {
        0
    }
}

/// Whether a CUDA device answers. Probed once a process: the probe quiets the
/// panic hook while the runtime loads (a missing library panics), and two
/// threads swapping the process's one hook at once could leave it quiet for
/// good, every later panic's message lost; so the first caller probes and
/// the rest read its answer.
#[must_use]
pub fn present() -> bool {
    #[cfg(feature = "device")]
    {
        static PRESENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *PRESENT.get_or_init(|| {
            let hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let probe = std::panic::catch_unwind(|| {
                let mut count: i32 = 0;
                // SAFETY: count is a live local, and this is the process's
                // first cudarc call.
                let status = unsafe { crate::device::rt::cudaGetDeviceCount(&raw mut count) };
                (status, count)
            });
            std::panic::set_hook(hook);
            matches!(
                probe,
                Ok((crate::device::rt::cudaError::cudaSuccess, count)) if count > 0
            )
        })
    }
    #[cfg(not(feature = "device"))]
    {
        false
    }
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
struct Side {
    stream: *mut c_void,
    cublas: *mut c_void,
    ctx: Ctx,
}

pub struct Context {
    ordinal: i32,
    stream: *mut c_void,
    notify: *mut c_void,
    /// Where taps read pages out, behind the layers that wrote them.
    tap: *mut c_void,
    /// Where kv pages move between the device and the host tier, beside
    /// the fires: a move waits on the fires that wrote its pages, and a fire
    /// on the moves that hold its own.
    swap: *mut c_void,
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    cublas: *mut c_void,
    ctx: Ctx,
    side: Vec<Side>,
    conditional: Option<Side>,
    events: Vec<crate::device::graph::Event>,
    #[cfg_attr(not(feature = "device"), allow(dead_code))]
    slabs: Slabs,
    device: Device,
    capability: (i32, i32),
    name: String,
}

impl Context {
    pub fn bind(ordinal: i32, comm: Option<&crate::comm::Comm>) -> Result<Context> {
        #[cfg(feature = "device")]
        {
            use crate::device::blas;
            use crate::device::rt;

            // SAFETY: each call takes a live out-parameter and the handles
            // the prior calls produced. cudaSetDevice binds this thread.
            unsafe {
                check("cudaSetDevice", rt::cudaSetDevice(ordinal))?;
                // What the process keeps idle (an earlier shell's dropped
                // buffers) goes back first: the load measures the card from
                // here, and fits as it would in a fresh process.
                crate::device::held::release_idle();
                let mut stream: rt::cudaStream_t = core::ptr::null_mut();
                check("cudaStreamCreate", rt::cudaStreamCreate(&raw mut stream))?;
                crate::device::held::adopt_stream(ordinal, stream.cast());
                // The heaps' first chunks, while the load may wait.
                let reserved =
                    crate::device::held::heap::reserve(crate::device::held::heap::Kind::Pinned)
                        .and_then(|()| {
                            crate::device::held::heap::reserve(
                                crate::device::held::heap::Kind::Device,
                            )
                        });
                if let Err(fault) = reserved {
                    destroy_stream(ordinal, stream.cast());
                    return Err(fault);
                }

                let handle: blas::cublasHandle_t = match blas_on(ordinal, stream.cast()) {
                    Ok(handle) => handle.cast(),
                    Err(fault) => {
                        destroy_stream(ordinal, stream.cast());
                        return Err(fault);
                    }
                };

                let mut notify: rt::cudaStream_t = core::ptr::null_mut();
                let status = rt::cudaStreamCreateWithFlags(&raw mut notify, 1);
                if status != rt::cudaError::cudaSuccess {
                    give_blas(ordinal, handle.cast());
                    destroy_stream(ordinal, stream.cast());
                    return Err(Fault::Device {
                        call: "cudaStreamCreateWithFlags",
                        code: status as i32,
                    });
                }

                let mut tap: rt::cudaStream_t = core::ptr::null_mut();
                let status = rt::cudaStreamCreateWithFlags(&raw mut tap, 1);
                if status != rt::cudaError::cudaSuccess {
                    let _ = rt::cudaStreamDestroy(notify);
                    give_blas(ordinal, handle.cast());
                    destroy_stream(ordinal, stream.cast());
                    return Err(Fault::Device {
                        call: "cudaStreamCreateWithFlags",
                        code: status as i32,
                    });
                }

                let mut swap: rt::cudaStream_t = core::ptr::null_mut();
                let status = rt::cudaStreamCreateWithFlags(&raw mut swap, 1);
                if status != rt::cudaError::cudaSuccess {
                    let _ = rt::cudaStreamDestroy(tap);
                    let _ = rt::cudaStreamDestroy(notify);
                    give_blas(ordinal, handle.cast());
                    destroy_stream(ordinal, stream.cast());
                    return Err(Fault::Device {
                        call: "cudaStreamCreateWithFlags",
                        code: status as i32,
                    });
                }

                let stream: *mut c_void = stream.cast();
                let notify: *mut c_void = notify.cast();
                let tap: *mut c_void = tap.cast();
                let swap: *mut c_void = swap.cast();
                let cublas: *mut c_void = handle.cast();
                let slabs = Slabs::open();
                slabs.attach(stream);
                #[cfg(not(feature = "rocm"))]
                let ctx = Ctx::on(stream).with_cublas(cublas).with_slabs(slabs);
                #[cfg(feature = "rocm")]
                let ctx = Ctx::on(stream).with_blas(cublas).with_slabs(slabs);
                // SAFETY: `comm` is the rank's live communicator, and its peer
                // stages live as long, owned by the boot for as long as this
                // shell fires on it.
                let ctx = match comm {
                    None => ctx,
                    Some(comm) => match comm.peers() {
                        None => ctx.with_comm(comm.raw()),
                        Some(peers) => ctx.with_comm(comm.raw()).with_peers(peers),
                    }
                    .with_rank(Some(comm.rank())),
                };
                let Some(device) = Device::probe(&ctx) else {
                    return Err(Fault::Unsupported(format!(
                        "device {ordinal} would not say its multiprocessor count, compute \
                         capability or shared memory"
                    )));
                };
                #[cfg(not(feature = "rocm"))]
                if device.cc_major < Device::FLOOR_MAJOR {
                    return Err(Fault::Unsupported(format!(
                        "device {ordinal} is compute capability {}.x; the CUDA engine serves \
                         sm_80 (Ampere) and newer",
                        device.cc_major
                    )));
                }
                // The kernels are wave64, and the prefill attention's bf16
                // MFMA arrived with CDNA 2.
                #[cfg(feature = "rocm")]
                if device.cc_major != 9 {
                    return Err(Fault::Unsupported(format!(
                        "device {ordinal} is gfx{}; the ROCm engine serves CDNA (gfx9: MI200 \
                         and MI300 series)",
                        device.cc_major
                    )));
                }
                Ok(Context {
                    ordinal,
                    stream,
                    notify,
                    tap,
                    swap,
                    cublas,
                    ctx,
                    slabs,
                    side: Vec::new(),
                    conditional: None,
                    events: Vec::new(),
                    device,
                    capability: capability(ordinal),
                    name: crate::chip::probe(ordinal)
                        .map(|(name, _)| name)
                        .unwrap_or_default(),
                })
            }
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = (ordinal, comm);
            Err(Fault::Runtimeless)
        }
    }

    pub fn bind_thread(&self) -> Result<()> {
        bind_thread(self.ordinal)
    }

    #[must_use]
    pub fn ctx(&self) -> &Ctx {
        &self.ctx
    }

    #[must_use]
    pub fn stream(&self) -> *mut c_void {
        self.stream
    }

    #[must_use]
    pub fn notify_stream(&self) -> *mut c_void {
        self.notify
    }

    #[must_use]
    pub fn tap_stream(&self) -> *mut c_void {
        self.tap
    }

    #[must_use]
    pub fn swap_stream(&self) -> *mut c_void {
        self.swap
    }

    pub fn host_fn(&self, work: Box<dyn FnOnce() + Send + 'static>) -> Result<()> {
        host_fn_on(self.notify, work)
    }

    pub fn open_lanes(&mut self, side: u32, events: u32) -> Result<()> {
        #[cfg(feature = "device")]
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
        #[cfg(not(feature = "device"))]
        {
            if side == 0 && events == 0 {
                return Ok(());
            }
            let _ = (side, events);
            Err(Fault::Runtimeless)
        }
    }

    #[cfg(feature = "device")]
    fn open_side(&mut self) -> Result<Side> {
        use crate::device::rt;

        // SAFETY: each call takes a live out-parameter, using handles the
        // prior calls produced; this thread bound the device.
        unsafe {
            let mut stream: rt::cudaStream_t = core::ptr::null_mut();
            check("cudaStreamCreate", rt::cudaStreamCreate(&raw mut stream))?;
            let cublas = match blas_on(self.ordinal, stream.cast()) {
                Ok(handle) => handle,
                Err(fault) => {
                    rt::cudaStreamDestroy(stream);
                    return Err(fault);
                }
            };
            let stream: *mut c_void = stream.cast();
            self.slabs.attach(stream);
            Ok(Side {
                stream,
                cublas,
                #[cfg(not(feature = "rocm"))]
                ctx: Ctx::on(stream)
                    .with_cublas(cublas)
                    .with_slabs(self.slabs)
                    .with_tuning(self.ctx.tuning())
                    .with_rank(self.ctx.rank()),
                #[cfg(feature = "rocm")]
                ctx: Ctx::on(stream)
                    .with_blas(cublas)
                    .with_slabs(self.slabs)
                    .with_tuning(self.ctx.tuning())
                    .with_rank(self.ctx.rank()),
            })
        }
    }

    pub fn open_conditional(&mut self) -> Result<()> {
        #[cfg(feature = "device")]
        {
            if self.conditional.is_none() {
                let mut side = self.open_side()?;
                side.ctx = side.ctx.in_order_of(self.stream);
                self.conditional = Some(side);
            }
            Ok(())
        }
        #[cfg(not(feature = "device"))]
        {
            Err(Fault::Runtimeless)
        }
    }

    #[must_use]
    pub fn conditional_stream(&self) -> *mut c_void {
        self.conditional
            .as_ref()
            .map_or(core::ptr::null_mut(), |side| side.stream)
    }

    #[must_use]
    pub fn conditional_ctx(&self) -> Option<&Ctx> {
        self.conditional.as_ref().map(|side| &side.ctx)
    }

    #[must_use]
    pub fn side_streams(&self) -> Vec<*mut c_void> {
        self.side.iter().map(|side| side.stream).collect()
    }

    #[must_use]
    pub fn side_ctx(&self) -> Vec<&Ctx> {
        self.side.iter().map(|side| &side.ctx).collect()
    }

    #[must_use]
    pub fn events(&self) -> &[crate::device::graph::Event] {
        &self.events
    }

    #[must_use]
    pub fn lanes(&self) -> usize {
        self.side.len()
    }

    #[must_use]
    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    #[must_use]
    pub fn toggles(&self) -> Toggles {
        Toggles::of(self.ctx.tuning())
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Lay a load's kernel decisions on this context, before it opens side
    /// streams.
    pub fn set_tuning(&mut self, tuning: kernels_cuda::tuning::Tuning) {
        self.ctx.set_tuning(tuning);
    }

    #[must_use]
    pub fn capability(&self) -> (i32, i32) {
        self.capability
    }

    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "device")]
        {
            // SAFETY: the handles are live for this context's lifetime.
            unsafe {
                check(
                    "cudaStreamSynchronize",
                    crate::device::rt::cudaStreamSynchronize(self.stream.cast()),
                )?;
                check(
                    "cudaStreamSynchronize",
                    crate::device::rt::cudaStreamSynchronize(self.swap.cast()),
                )?;
                check(
                    "cudaStreamSynchronize",
                    crate::device::rt::cudaStreamSynchronize(self.tap.cast()),
                )?;
                check(
                    "cudaStreamSynchronize",
                    crate::device::rt::cudaStreamSynchronize(self.notify.cast()),
                )
            }
        }
        #[cfg(not(feature = "device"))]
        {
            Err(Fault::Runtimeless)
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        #[cfg(feature = "device")]
        {
            // SAFETY: the shell is being torn down, so nothing else holds
            // either handle; both were produced by this context's `bind`.
            // The slabs are this context's own: every `Ctx` and graph that
            // carried them belongs to the shell going down with it.
            unsafe {
                self.events.clear();
                self.slabs.release();
                for side in self.side.drain(..).chain(self.conditional.take()) {
                    if !side.cublas.is_null() {
                        give_blas(self.ordinal, side.cublas);
                    }
                    if !side.stream.is_null() {
                        let _ = crate::device::rt::cudaStreamDestroy(side.stream.cast());
                    }
                }
                if !self.cublas.is_null() {
                    give_blas(self.ordinal, self.cublas);
                }
                if !self.stream.is_null() {
                    destroy_stream(self.ordinal, self.stream);
                }
                if !self.notify.is_null() {
                    let _ = crate::device::rt::cudaStreamDestroy(self.notify.cast());
                }
                if !self.tap.is_null() {
                    let _ = crate::device::rt::cudaStreamDestroy(self.tap.cast());
                }
                if !self.swap.is_null() {
                    let _ = crate::device::rt::cudaStreamDestroy(self.swap.cast());
                }
            }
        }
    }
}

/// `work` on the host once what `stream` has before it is done.
pub fn host_fn_on(stream: *mut c_void, work: Box<dyn FnOnce() + Send + 'static>) -> Result<()> {
    #[cfg(feature = "device")]
    {
        use crate::device::rt;

        let carried: *mut Box<dyn FnOnce() + Send + 'static> = Box::into_raw(Box::new(work));
        // SAFETY: carried is a live leaked allocation, reclaimed exactly
        // once — here on the failure path, or by the trampoline.
        let code = unsafe {
            rt::cudaLaunchHostFunc(stream.cast(), Some(host_fn_trampoline), carried.cast())
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
    #[cfg(not(feature = "device"))]
    {
        let _ = (stream, work);
        Err(Fault::Runtimeless)
    }
}

#[cfg(feature = "device")]
extern "C" fn host_fn_trampoline(user: *mut c_void) {
    if user.is_null() {
        return;
    }
    // SAFETY: user is the pointer host_fn leaked; the driver calls this
    // exactly once per successful launch, so this reclaims it exactly once.
    let work = unsafe { Box::from_raw(user.cast::<Box<dyn FnOnce() + Send + 'static>>()) };
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("ordinal", &self.ordinal)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
fn capability(ordinal: i32) -> (i32, i32) {
    #[cfg(feature = "device")]
    {
        use crate::device::rt;

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
    #[cfg(not(feature = "device"))]
    {
        let _ = ordinal;
        (0, 0)
    }
}

pub fn current() -> Result<i32> {
    #[cfg(feature = "device")]
    {
        let mut ordinal = 0i32;
        // SAFETY: ordinal is a live out-parameter.
        unsafe {
            check(
                "cudaGetDevice",
                crate::device::rt::cudaGetDevice(&raw mut ordinal),
            )?;
        }
        Ok(ordinal)
    }
    #[cfg(not(feature = "device"))]
    {
        Err(Fault::Runtimeless)
    }
}

pub fn bind_thread(ordinal: i32) -> Result<()> {
    #[cfg(feature = "device")]
    {
        // SAFETY: an ordinal the runtime itself answered with.
        unsafe { check("cudaSetDevice", crate::device::rt::cudaSetDevice(ordinal))? };
        // The kernels' own per-thread binding here, not at a kernel's first
        // resolution mid-fire, where its `cudaFree` waits on the device.
        #[cfg(all(feature = "cuda", not(feature = "rocm")))]
        kernels_cuda::jit::bind_context().map_err(crate::error::kernel)?;
        Ok(())
    }
    #[cfg(not(feature = "device"))]
    {
        let _ = ordinal;
        Err(Fault::Runtimeless)
    }
}

pub fn sync(stream: *mut c_void) -> Result<()> {
    #[cfg(feature = "device")]
    {
        // SAFETY: the handle is the caller's, live for the call.
        unsafe {
            check(
                "cudaStreamSynchronize",
                crate::device::rt::cudaStreamSynchronize(stream.cast()),
            )
        }
    }
    #[cfg(not(feature = "device"))]
    {
        let _ = stream;
        Err(Fault::Runtimeless)
    }
}

/// Destroys a context's compute stream, out of the reach of the memory held
/// behind it first.
///
/// # Safety
///
/// `stream` is a live stream nothing launches on again.
#[cfg(feature = "device")]
unsafe fn destroy_stream(ordinal: i32, stream: *mut c_void) {
    crate::device::held::retire_stream(ordinal, stream, || {
        // SAFETY: as the caller promises.
        let _ = unsafe { crate::device::rt::cudaStreamDestroy(stream.cast()) };
    });
}

/// cuBLAS handles dropped contexts gave back, by device, for the next bind
/// or side stream to take. `cublasDestroy` synchronizes the whole device,
/// and a stream another thread is capturing into is invalidated by it
/// (`cudaErrorStreamCaptureInvalidated`): a shell dropped while another
/// model's load captures its graphs broke that capture. A handle is set to
/// its new stream when taken, and the math mode is laid on each use
/// (`Ctx::cublas`), so a kept one serves as a fresh one would; the process
/// holds at most as many as it ever had live at once.
#[cfg(feature = "device")]
static BLAS: std::sync::Mutex<Vec<(i32, usize)>> = std::sync::Mutex::new(Vec::new());

/// A cuBLAS handle of `ordinal` on `stream`: a kept one, else a fresh one.
///
/// # Safety
///
/// This thread is bound to `ordinal`, and `stream` is a live stream of it.
#[cfg(feature = "device")]
unsafe fn blas_on(ordinal: i32, stream: *mut c_void) -> Result<*mut c_void> {
    use crate::device::blas;

    let kept = {
        let mut kept = BLAS.lock().expect("blas lock poisoned");
        kept.iter()
            .position(|&(device, _)| device == ordinal)
            .map(|at| kept.swap_remove(at).1)
    };
    let handle: blas::cublasHandle_t = match kept {
        Some(handle) => handle as blas::cublasHandle_t,
        None => {
            let mut handle: blas::cublasHandle_t = core::ptr::null_mut();
            // SAFETY: `handle` is a live local the call fills.
            let status = unsafe { blas::cublasCreate_v2(&raw mut handle) };
            if status != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(Fault::Device {
                    call: "cublasCreate_v2",
                    code: status as i32,
                });
            }
            handle
        }
    };
    // SAFETY: the handle is live and no one else's; `stream` is the caller's.
    let status = unsafe { blas::cublasSetStream_v2(handle, stream.cast()) };
    if status != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        give_blas(ordinal, handle.cast());
        return Err(Fault::Device {
            call: "cublasSetStream_v2",
            code: status as i32,
        });
    }
    Ok(handle.cast())
}

/// Keeps `handle`, of `ordinal`, for the next [`blas_on`] rather than
/// destroying it. Its stream may be destroyed after: a kept handle launches
/// nothing until it is taken and set to another.
#[cfg(feature = "device")]
fn give_blas(ordinal: i32, handle: *mut c_void) {
    BLAS.lock()
        .expect("blas lock poisoned")
        .push((ordinal, handle as usize));
}

#[cfg(feature = "device")]
pub(crate) fn check(call: &'static str, status: crate::device::rt::cudaError) -> Result<()> {
    if status == crate::device::rt::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Fault::Device {
            call,
            code: status as i32,
        })
    }
}
