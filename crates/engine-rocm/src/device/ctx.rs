use core::ffi::c_void;

use kernels_rocm::attn::plan::{Device, Toggles};
use kernels_rocm::{Ctx, Slabs};

use crate::error::{Fault, Result};

#[must_use]
pub fn count() -> usize {
    #[cfg(feature = "rocm")]
    {
        if !present() {
            return 0;
        }
        let mut count: i32 = 0;
        // SAFETY: a live local; the runtime is loaded (`present` said so).
        let status = unsafe { kernels_rocm::hip::rt::hipGetDeviceCount(&raw mut count) };
        if status == kernels_rocm::hip::rt::hipSuccess {
            usize::try_from(count).unwrap_or(0)
        } else {
            0
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        0
    }
}

#[must_use]
pub fn present() -> bool {
    #[cfg(feature = "rocm")]
    {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let probe = std::panic::catch_unwind(|| {
            let mut count: i32 = 0;
            // SAFETY: count is a live local, and this is the process's first cudarc call.
            let status = unsafe { kernels_rocm::hip::rt::hipGetDeviceCount(&raw mut count) };
            (status, count)
        });
        std::panic::set_hook(hook);
        matches!(
            probe,
            Ok((kernels_rocm::hip::rt::hipSuccess, count)) if count > 0
        )
    }
    #[cfg(not(feature = "rocm"))]
    {
        false
    }
}

#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
struct Side {
    stream: *mut c_void,
    blas: *mut c_void,
    ctx: Ctx,
}

pub struct Context {
    ordinal: i32,
    stream: *mut c_void,
    notify: *mut c_void,
    #[cfg_attr(not(feature = "rocm"), allow(dead_code))]
    blas: *mut c_void,
    ctx: Ctx,
    side: Vec<Side>,
    conditional: Option<Side>,
    events: Vec<crate::device::graph::Event>,
    #[cfg_attr(not(feature = "rocm"), allow(dead_code))]
    slabs: Slabs,
    device: Device,
    capability: (i32, i32),
    name: String,
}

impl Context {
    pub fn bind(ordinal: i32, comm: Option<&crate::comm::Comm>) -> Result<Context> {
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            // SAFETY: each call takes a live out-parameter and the handles
            // the prior calls produced. cudaSetDevice binds this thread.
            unsafe {
                check("hipSetDevice", rt::hipSetDevice(ordinal))?;
                let mut stream: rt::Stream = core::ptr::null_mut();
                check("hipStreamCreate", rt::hipStreamCreate(&raw mut stream))?;
                crate::program::endpoint::pool::adopt_stream(ordinal, stream.cast());

                // The kernels bind the handle to whichever stream they fire on.
                let handle = match kernels_rocm::hip::blas::create_raw() {
                    Ok(handle) => handle,
                    Err(why) => {
                        rt::hipStreamDestroy(stream);
                        return Err(Fault::Device {
                            call: why.call,
                            code: why.status,
                        });
                    }
                };

                let mut notify: rt::Stream = core::ptr::null_mut();
                let status = rt::hipStreamCreateWithFlags(&raw mut notify, 1);
                if status != rt::hipSuccess {
                    kernels_rocm::hip::blas::destroy_raw(handle);
                    rt::hipStreamDestroy(stream);
                    return Err(Fault::Device {
                        call: "hipStreamCreateWithFlags",
                        code: status as i32,
                    });
                }

                let stream: *mut c_void = stream.cast();
                let notify: *mut c_void = notify.cast();
                let blas: *mut c_void = handle.cast();
                let slabs = Slabs::open();
                slabs.attach(stream);
                let ctx = Ctx::on(stream).with_blas(blas).with_slabs(slabs);
                // SAFETY: `comm` is the rank's live communicator, and its peer
                // stages live as long, owned by the boot for as long as this
                // shell fires on it.
                let ctx = match comm {
                    None => ctx,
                    Some(comm) => match comm.peers() {
                        None => ctx.with_comm(comm.raw()),
                        Some(peers) => ctx.with_comm(comm.raw()).with_peers(peers),
                    },
                };
                let Some(device) = Device::probe(&ctx) else {
                    return Err(Fault::Unsupported(format!(
                        "device {ordinal} would not say its multiprocessor count, compute \
                         capability or shared memory"
                    )));
                };
                // The kernels are wave64 and the prefill attention MFMA's
                // bf16 form arrived with CDNA 2: gfx9.
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
                    blas,
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
        #[cfg(not(feature = "rocm"))]
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

    pub fn host_fn(&self, work: Box<dyn FnOnce() + Send + 'static>) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            use kernels_rocm::hip::rt;

            let carried: *mut Box<dyn FnOnce() + Send + 'static> = Box::into_raw(Box::new(work));
            // SAFETY: carried is a live leaked allocation, reclaimed exactly
            // once — here on the failure path, or by the trampoline.
            let code = unsafe {
                rt::hipLaunchHostFunc(self.notify.cast(), host_fn_trampoline, carried.cast())
            };
            if code != rt::hipSuccess {
                // SAFETY: the launch failed, so nothing else will ever see
                // this pointer; reclaiming it here avoids a leak.
                drop(unsafe { Box::from_raw(carried) });
                return Err(Fault::Device {
                    call: "hipLaunchHostFunc",
                    code: code as i32,
                });
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = work;
            Err(Fault::Runtimeless)
        }
    }

    pub fn open_lanes(&mut self, side: u32, events: u32) -> Result<()> {
        #[cfg(feature = "rocm")]
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
        #[cfg(not(feature = "rocm"))]
        {
            if side == 0 && events == 0 {
                return Ok(());
            }
            let _ = (side, events);
            Err(Fault::Runtimeless)
        }
    }

    #[cfg(feature = "rocm")]
    fn open_side(&mut self) -> Result<Side> {
        use kernels_rocm::hip::rt;

        // SAFETY: each call takes a live out-parameter, using handles the
        // prior calls produced; this thread bound the device.
        unsafe {
            let mut stream: rt::Stream = core::ptr::null_mut();
            check("hipStreamCreate", rt::hipStreamCreate(&raw mut stream))?;
            let handle = match kernels_rocm::hip::blas::create_raw() {
                Ok(handle) => handle,
                Err(why) => {
                    rt::hipStreamDestroy(stream);
                    return Err(Fault::Device {
                        call: why.call,
                        code: why.status,
                    });
                }
            };
            let stream: *mut c_void = stream.cast();
            let blas: *mut c_void = handle.cast();
            self.slabs.attach(stream);
            Ok(Side {
                stream,
                blas,
                ctx: Ctx::on(stream)
                    .with_blas(blas)
                    .with_slabs(self.slabs)
                    .with_tuning(self.ctx.tuning()),
            })
        }
    }

    pub fn open_conditional(&mut self) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            if self.conditional.is_none() {
                let mut side = self.open_side()?;
                side.ctx = side.ctx.in_order_of(self.stream);
                self.conditional = Some(side);
            }
            Ok(())
        }
        #[cfg(not(feature = "rocm"))]
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
    pub fn set_tuning(&mut self, tuning: kernels_rocm::tuning::Tuning) {
        self.ctx.set_tuning(tuning);
    }

    #[must_use]
    pub fn capability(&self) -> (i32, i32) {
        self.capability
    }

    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "rocm")]
        {
            // SAFETY: both handles are live for this context's lifetime.
            unsafe {
                check(
                    "hipStreamSynchronize",
                    kernels_rocm::hip::rt::hipStreamSynchronize(self.stream.cast()),
                )?;
                check(
                    "hipStreamSynchronize",
                    kernels_rocm::hip::rt::hipStreamSynchronize(self.notify.cast()),
                )
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err(Fault::Runtimeless)
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        #[cfg(feature = "rocm")]
        {
            // SAFETY: the shell is being torn down, so nothing else holds
            // either handle; both were produced by this context's `bind`.
            unsafe {
                self.events.clear();
                self.slabs.release();
                for side in self.side.drain(..).chain(self.conditional.take()) {
                    if !side.blas.is_null() {
                        kernels_rocm::hip::blas::destroy_raw(side.blas.cast());
                    }
                    if !side.stream.is_null() {
                        let _ = kernels_rocm::hip::rt::hipStreamDestroy(side.stream.cast());
                    }
                }
                if !self.blas.is_null() {
                    kernels_rocm::hip::blas::destroy_raw(self.blas.cast());
                }
                if !self.stream.is_null() {
                    let _ = kernels_rocm::hip::rt::hipStreamDestroy(self.stream.cast());
                }
                if !self.notify.is_null() {
                    let _ = kernels_rocm::hip::rt::hipStreamDestroy(self.notify.cast());
                }
            }
        }
    }
}

#[cfg(feature = "rocm")]
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

#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
fn capability(ordinal: i32) -> (i32, i32) {
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;

        let attribute = |which: core::ffi::c_int| -> i32 {
            let mut value = 0i32;
            // SAFETY: value is a live out-parameter; ordinal was just
            // accepted by cudaSetDevice above.
            let status = unsafe { rt::hipDeviceGetAttribute(&raw mut value, which, ordinal) };
            if status == rt::hipSuccess { value } else { 0 }
        };
        (
            attribute(rt::hipDeviceAttributeComputeCapabilityMajor),
            attribute(rt::hipDeviceAttributeComputeCapabilityMinor),
        )
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = ordinal;
        (0, 0)
    }
}

pub fn current() -> Result<i32> {
    #[cfg(feature = "rocm")]
    {
        let mut ordinal = 0i32;
        // SAFETY: ordinal is a live out-parameter.
        unsafe {
            check(
                "hipGetDevice",
                kernels_rocm::hip::rt::hipGetDevice(&raw mut ordinal),
            )?;
        }
        Ok(ordinal)
    }
    #[cfg(not(feature = "rocm"))]
    {
        Err(Fault::Runtimeless)
    }
}

pub fn bind_thread(ordinal: i32) -> Result<()> {
    #[cfg(feature = "rocm")]
    {
        // SAFETY: an ordinal the runtime itself answered with.
        unsafe { check("hipSetDevice", kernels_rocm::hip::rt::hipSetDevice(ordinal)) }
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = ordinal;
        Err(Fault::Runtimeless)
    }
}

pub fn sync(stream: *mut c_void) -> Result<()> {
    #[cfg(feature = "rocm")]
    {
        // SAFETY: the handle is the caller's, live for the call.
        unsafe {
            check(
                "hipStreamSynchronize",
                kernels_rocm::hip::rt::hipStreamSynchronize(stream.cast()),
            )
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = stream;
        Err(Fault::Runtimeless)
    }
}

#[cfg(feature = "rocm")]
pub(crate) fn check(call: &'static str, status: kernels_rocm::hip::rt::HipError) -> Result<()> {
    if status == kernels_rocm::hip::rt::hipSuccess {
        Ok(())
    } else {
        Err(Fault::Device { call, code: status })
    }
}
