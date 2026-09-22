use core::ffi::c_void;

use kernels_cuda::attn::plan::{Device, Toggles};
use kernels_cuda::{Ctx, Slabs};

use crate::error::{Fault, Result};

#[must_use]
pub fn count() -> usize {
    #[cfg(feature = "cuda")]
    {
        if !present() {
            return 0;
        }
        let mut count: i32 = 0;
        // SAFETY: a live local; the runtime is loaded (`present` said so).
        let status = unsafe { cudarc::runtime::sys::cudaGetDeviceCount(&raw mut count) };
        if status == cudarc::runtime::sys::cudaError::cudaSuccess {
            usize::try_from(count).unwrap_or(0)
        } else {
            0
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

#[must_use]
pub fn present() -> bool {
    #[cfg(feature = "cuda")]
    {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let probe = std::panic::catch_unwind(|| {
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

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Side {
    stream: *mut c_void,
    cublas: *mut c_void,
    ctx: Ctx,
}

pub struct Context {
    ordinal: i32,
    stream: *mut c_void,
    notify: *mut c_void,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    cublas: *mut c_void,
    ctx: Ctx,
    side: Vec<Side>,
    conditional: Option<Side>,
    events: Vec<crate::device::graph::Event>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    slabs: Slabs,
    device: Device,
    toggles: Toggles,
    capability: (i32, i32),
}

impl Context {
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

                let mut notify: rt::cudaStream_t = core::ptr::null_mut();
                let status = rt::cudaStreamCreateWithFlags(&raw mut notify, 1);
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
                slabs.attach(stream);
                let ctx = Ctx::on(stream).with_cublas(cublas).with_slabs(slabs);
                // SAFETY: `comm` is the rank's live communicator (or null),
                // owned by the boot for as long as this shell fires on it.
                let ctx = if comm.is_null() {
                    ctx
                } else {
                    ctx.with_comm(comm)
                };
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
        #[cfg(feature = "cuda")]
        {
            use cudarc::runtime::sys as rt;

            let carried: *mut Box<dyn FnOnce() + Send + 'static> = Box::into_raw(Box::new(work));
            // SAFETY: carried is a live leaked allocation, reclaimed exactly
            // once — here on the failure path, or by the trampoline.
            let code = unsafe {
                rt::cudaLaunchHostFunc(self.notify.cast(), Some(host_fn_trampoline), carried.cast())
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
            self.slabs.attach(stream);
            Ok(Side {
                stream,
                cublas,
                ctx: Ctx::on(stream).with_cublas(cublas).with_slabs(self.slabs),
            })
        }
    }

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
        self.toggles
    }

    #[must_use]
    pub fn capability(&self) -> (i32, i32) {
        self.capability
    }

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
                self.events.clear();
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
                if !self.notify.is_null() {
                    let _ = cudarc::runtime::sys::cudaStreamDestroy(self.notify.cast());
                }
            }
        }
    }
}

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

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
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

#[cfg(feature = "cuda")]
pub(crate) fn check(call: &'static str, status: cudarc::runtime::sys::cudaError) -> Result<()> {
    if status == cudarc::runtime::sys::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(Fault::Device {
            call,
            code: status as i32,
        })
    }
}
