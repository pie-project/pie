//! The host-sync guard: what a fire calls that waits for the device. A
//! fire only enqueues; a call that blocks the host on the device (a copy or
//! set without a stream, an allocation or free, a stream or device sync) or
//! that reaches the legacy stream (which waits for every blocking stream)
//! stalls the step, and on an expert-parallel group sharing a device
//! deadlocks it: the wait holds back this rank's launch while a peer's
//! kernel spins for it.
//!
//! [`arm`] subscribes to CUPTI's runtime and driver callbacks, process-wide
//! and once; [`watched`] notes such calls on this thread while its closure
//! runs. Off, nothing subscribes and a fire pays nothing.
#![allow(unsafe_code)]

use crate::error::{Fault, Result};

/// One call a watched fire made that waits for the device.
#[derive(Clone, Debug)]
pub struct Wait {
    /// The runtime or driver call, and how it waits.
    pub call: String,
    /// Where it was called from: on this thread's first sighting only.
    pub from: Option<String>,
}

impl core::fmt::Display for Wait {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.call)?;
        if let Some(from) = &self.from {
            write!(f, ", called from:\n{from}")?;
        }
        Ok(())
    }
}

/// Subscribes the guard, once a process.
pub fn arm() -> Result<()> {
    static ARMED: std::sync::OnceLock<std::result::Result<(), String>> = std::sync::OnceLock::new();
    ARMED
        .get_or_init(cupti::subscribe)
        .clone()
        .map_err(|why| Fault::program("device::guard", why))
}

/// Prints each call a watched fire makes as the thread first makes it,
/// from inside the call: a fire that never returns (a group deadlocked on
/// it) has still said what it waited on, and from where.
pub fn speak() {
    cupti::SPOKEN.store(true, core::sync::atomic::Ordering::Relaxed);
}

/// Runs `fire`, noting each call on this thread that waits for the device.
pub fn watched<T>(fire: impl FnOnce() -> T) -> (T, Vec<Wait>) {
    /// Ends the watch however the fire ends: a rank's thread outlives a
    /// fire that panics.
    struct Watch;
    impl Drop for Watch {
        fn drop(&mut self) {
            cupti::FIRING.set(false);
        }
    }
    cupti::NOTED.take();
    cupti::FIRING.set(true);
    let watch = Watch;
    let fired = fire();
    drop(watch);
    (fired, cupti::NOTED.take())
}

/// A runtime or driver call by its CUPTI name, the `_v2` revisions and
/// `_ptsz` per-thread forms read as their base call.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn base(name: &str) -> &str {
    let mut name = name;
    loop {
        let Some((head, tail)) = name.rsplit_once('_') else {
            return name;
        };
        let revision = tail.len() > 1
            && tail.starts_with('v')
            && tail[1..].bytes().all(|b| b.is_ascii_digit());
        if !(revision || tail == "ptsz" || tail == "ptds") {
            return name;
        }
        name = head;
    }
}

/// Calls that block the host until the device (or a stream of it) is done.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const BLOCKING: &[&str] = &[
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaMemcpy",
    "cudaMemcpy2D",
    "cudaMemcpyToSymbol",
    "cudaMemcpyFromSymbol",
    "cudaMemset",
    "cudaMalloc",
    "cudaMallocHost",
    "cudaHostAlloc",
    "cudaFree",
    "cudaFreeHost",
    "cudaHostRegister",
    "cudaHostUnregister",
    "cuCtxSynchronize",
    "cuStreamSynchronize",
    "cuEventSynchronize",
    "cuMemcpy",
    "cuMemcpyHtoD",
    "cuMemcpyDtoH",
    "cuMemcpyDtoD",
    "cuMemsetD8",
    "cuMemsetD32",
    "cuMemAlloc",
    "cuMemAllocHost",
    "cuMemHostAlloc",
    "cuMemFree",
    "cuMemFreeHost",
    "cuMemHostRegister",
    "cuMemHostUnregister",
];

/// Calls that take the context's lock against everything it runs: a module
/// load or unload, a function attribute, a graph's instantiation or
/// teardown, a context limit, a virtual-memory unmap. Each waits for the
/// device to reach a point where the context can change under it, so beside
/// a peer rank's spinning collective on the same device it never returns.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const CONTEXT: &[&str] = &[
    "cuModuleLoad",
    "cuModuleLoadData",
    "cuModuleLoadDataEx",
    "cuModuleLoadFatBinary",
    "cuModuleUnload",
    "cuLibraryLoadData",
    "cuLibraryLoadFromFile",
    "cuLibraryUnload",
    "cuFuncSetAttribute",
    "cuFuncSetCacheConfig",
    "cuFuncSetSharedMemConfig",
    "cuKernelSetAttribute",
    "cuKernelSetCacheConfig",
    "cudaFuncSetAttribute",
    "cudaFuncSetCacheConfig",
    "cudaFuncSetSharedMemConfig",
    "cuGraphInstantiate",
    "cuGraphInstantiateWithFlags",
    "cuGraphInstantiateWithParams",
    "cuGraphExecDestroy",
    "cudaGraphInstantiate",
    "cudaGraphInstantiateWithFlags",
    "cudaGraphInstantiateWithParams",
    "cudaGraphExecDestroy",
    "cuCtxSetLimit",
    "cuCtxSetCacheConfig",
    "cudaDeviceSetLimit",
    "cudaDeviceSetCacheConfig",
    "cuMemUnmap",
    "cuMemRelease",
    "cuMemAddressFree",
    "cuMemPoolTrimTo",
    "cudaMemPoolTrimTo",
];

#[cfg(feature = "cuda")]
mod cupti {
    use core::ffi::{CStr, c_void};
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeSet;
    use std::sync::atomic::AtomicBool;

    use cudarc::cupti::sys as p;
    use cudarc::cupti::sys::{
        CUpti_ApiCallbackSite, CUpti_CallbackData, CUpti_CallbackDomain, CUpti_CallbackId,
        CUpti_SubscriberHandle, CUptiResult, cuptiEnableDomain, cuptiSubscribe,
    };

    use super::{BLOCKING, CONTEXT, Wait, base};

    /// Whether a first sighting is printed as it is made ([`super::speak`]).
    pub(super) static SPOKEN: AtomicBool = AtomicBool::new(false);

    thread_local! {
        pub(super) static FIRING: Cell<bool> = const { Cell::new(false) };
        pub(super) static NOTED: RefCell<Vec<Wait>> = const { RefCell::new(Vec::new()) };
        /// The calls this thread first noted, by how they wait.
        static SEEN: RefCell<BTreeSet<String>> = const { RefCell::new(BTreeSet::new()) };
        /// Runtime calls this thread is inside of.
        static RUNTIME: Cell<u32> = const { Cell::new(0) };
    }

    pub(super) fn subscribe() -> std::result::Result<(), String> {
        let ok = |what: &str, status: CUptiResult| match status {
            CUptiResult::CUPTI_SUCCESS => Ok(()),
            other => Err(format!("{what}: {other:?}")),
        };
        // A missing libcupti panics in cudarc's loader.
        std::panic::catch_unwind(|| {
            let mut subscriber: CUpti_SubscriberHandle = core::ptr::null_mut();
            // SAFETY: `called` lives for the process, and reads its data
            // only as the domain it is subscribed to hands it.
            unsafe {
                ok(
                    "cuptiSubscribe",
                    cuptiSubscribe(&raw mut subscriber, Some(called), core::ptr::null_mut()),
                )?;
                for domain in [
                    CUpti_CallbackDomain::CUPTI_CB_DOMAIN_RUNTIME_API,
                    CUpti_CallbackDomain::CUPTI_CB_DOMAIN_DRIVER_API,
                ] {
                    ok(
                        "cuptiEnableDomain",
                        cuptiEnableDomain(1, subscriber, domain),
                    )?;
                }
            }
            Ok(())
        })
        .unwrap_or_else(|_| Err("the sync guard needs CUPTI (libcupti), which is not found".into()))
    }

    /// CUPTI's callback, on the calling thread: notes a call that waits
    /// while the thread fires. It must not panic, across C.
    unsafe extern "C" fn called(
        _: *mut c_void,
        domain: CUpti_CallbackDomain,
        _: CUpti_CallbackId,
        data: *const c_void,
    ) {
        if !FIRING.get() || data.is_null() {
            return;
        }
        // SAFETY: the API domains hand a `CUpti_CallbackData`.
        let data = unsafe { &*data.cast::<CUpti_CallbackData>() };
        let enter = data.callbackSite == CUpti_ApiCallbackSite::CUPTI_API_ENTER;
        // The driver calls a runtime call makes are its own: noted as it.
        if domain == CUpti_CallbackDomain::CUPTI_CB_DOMAIN_RUNTIME_API {
            let inside = RUNTIME.get();
            RUNTIME.set(if enter {
                inside + 1
            } else {
                inside.saturating_sub(1)
            });
        } else if RUNTIME.get() > 0 {
            return;
        }
        if !enter || data.functionName.is_null() {
            return;
        }
        // SAFETY: CUPTI's name is a static C string.
        let Ok(name) = unsafe { CStr::from_ptr(data.functionName) }.to_str() else {
            return;
        };
        let call = if BLOCKING.contains(&base(name)) {
            format!("{name} (blocks on the device)")
        } else if CONTEXT.contains(&base(name)) {
            format!("{name} (takes the context against the device)")
        } else if legacy(domain, name, data.functionParams) {
            format!("{name} on the legacy stream")
        } else {
            return;
        };
        let first = SEEN
            .try_with(|seen| {
                seen.try_borrow_mut()
                    .is_ok_and(|mut s| s.insert(call.clone()))
            })
            .unwrap_or(false);
        let from = first.then(|| std::backtrace::Backtrace::force_capture().to_string());
        if let Some(from) = &from
            && SPOKEN.load(core::sync::atomic::Ordering::Relaxed)
        {
            let thread = std::thread::current();
            let on = thread
                .name()
                .map_or_else(|| format!("{:?}", thread.id()), ToString::to_string);
            eprintln!("engine-cuda: [sync-guard] a fire on `{on}` calls {call}, from:\n{from}");
        }
        let _ = NOTED.try_with(|noted| {
            if let Ok(mut noted) = noted.try_borrow_mut() {
                noted.push(Wait { call, from });
            }
        });
    }

    /// Whether the stream-taking call `name` takes the legacy stream (null,
    /// or `CU_STREAM_LEGACY`), its parameters as CUPTI lays them. The
    /// per-thread stream (`CU_STREAM_PER_THREAD`, where a copy that names no
    /// stream of its caller's runs) is a stream of its own, and not it.
    fn legacy(domain: CUpti_CallbackDomain, name: &str, params: *const c_void) -> bool {
        if params.is_null() || name.contains("_ptsz") || name.contains("_ptds") {
            return false;
        }
        // SAFETY: each arm reads the parameter block CUPTI names `name`'s.
        let stream = unsafe {
            let at = |offset| {
                params
                    .cast::<u8>()
                    .add(offset)
                    .cast::<*mut c_void>()
                    .read_unaligned()
            };
            macro_rules! field {
                ($ty:ty, $field:ident) => {
                    at(core::mem::offset_of!($ty, $field))
                };
            }
            match (domain, name) {
                (CUpti_CallbackDomain::CUPTI_CB_DOMAIN_RUNTIME_API, _) => match base(name) {
                    "cudaLaunchKernel" => field!(p::cudaLaunchKernel_v7000_params_st, stream),
                    "cudaMemcpyAsync" => field!(p::cudaMemcpyAsync_v3020_params_st, stream),
                    "cudaMemsetAsync" => field!(p::cudaMemsetAsync_v3020_params_st, stream),
                    "cudaGraphLaunch" => field!(p::cudaGraphLaunch_v10000_params_st, stream),
                    "cudaEventRecord" => field!(p::cudaEventRecord_v3020_params_st, stream),
                    "cudaStreamWaitEvent" => field!(p::cudaStreamWaitEvent_v3020_params_st, stream),
                    "cudaEventRecordWithFlags" => {
                        field!(p::cudaEventRecordWithFlags_v11010_params_st, stream)
                    }
                    "cudaMemcpy2DAsync" => field!(p::cudaMemcpy2DAsync_v3020_params_st, stream),
                    "cudaMallocAsync" => field!(p::cudaMallocAsync_v11020_params_st, hStream),
                    "cudaFreeAsync" => field!(p::cudaFreeAsync_v11020_params_st, hStream),
                    "cudaGraphUpload" => field!(p::cudaGraphUpload_v10000_params_st, stream),
                    "cudaLaunchHostFunc" => field!(p::cudaLaunchHostFunc_v10000_params_st, stream),
                    _ => return false,
                },
                (CUpti_CallbackDomain::CUPTI_CB_DOMAIN_DRIVER_API, _) => match name {
                    "cuLaunchKernel" => field!(p::cuLaunchKernel_params_st, hStream),
                    "cuLaunchKernelEx" => {
                        let config: *const cudarc::driver::sys::CUlaunchConfig =
                            field!(p::cuLaunchKernelEx_params_st, config).cast();
                        if config.is_null() {
                            return false;
                        }
                        (*config).hStream.cast()
                    }
                    "cuGraphLaunch" => field!(p::cuGraphLaunch_params_st, hStream),
                    "cuMemcpyAsync" => field!(p::cuMemcpyAsync_params_st, hStream),
                    "cuMemcpyHtoDAsync_v2" => field!(p::cuMemcpyHtoDAsync_v2_params_st, hStream),
                    "cuMemcpyDtoHAsync_v2" => field!(p::cuMemcpyDtoHAsync_v2_params_st, hStream),
                    "cuMemcpyDtoDAsync_v2" => field!(p::cuMemcpyDtoDAsync_v2_params_st, hStream),
                    "cuMemsetD8Async" => field!(p::cuMemsetD8Async_params_st, hStream),
                    "cuMemsetD32Async" => field!(p::cuMemsetD32Async_params_st, hStream),
                    "cuEventRecord" => field!(p::cuEventRecord_params_st, hStream),
                    "cuStreamWaitEvent" => field!(p::cuStreamWaitEvent_params_st, hStream),
                    "cuEventRecordWithFlags" => {
                        field!(p::cuEventRecordWithFlags_params_st, hStream)
                    }
                    "cuMemcpy2DAsync_v2" => field!(p::cuMemcpy2DAsync_v2_params_st, hStream),
                    "cuMemAllocAsync" => field!(p::cuMemAllocAsync_params_st, hStream),
                    "cuMemFreeAsync" => field!(p::cuMemFreeAsync_params_st, hStream),
                    "cuGraphUpload" => field!(p::cuGraphUpload_params_st, hStream),
                    "cuLaunchHostFunc" => field!(p::cuLaunchHostFunc_params_st, hStream),
                    _ => return false,
                },
                _ => return false,
            }
        };
        stream.is_null() || stream as usize == 1
    }
}

#[cfg(not(feature = "cuda"))]
mod cupti {
    use std::cell::{Cell, RefCell};

    use super::Wait;

    pub(super) static SPOKEN: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    thread_local! {
        pub(super) static FIRING: Cell<bool> = const { Cell::new(false) };
        pub(super) static NOTED: RefCell<Vec<Wait>> = const { RefCell::new(Vec::new()) };
    }

    pub(super) fn subscribe() -> std::result::Result<(), String> {
        Err("the sync guard reads CUPTI, which only a CUDA build has".into())
    }
}

#[cfg(test)]
mod tests {
    use super::base;

    /// A read's wait for its copy and a set on the legacy stream are noted,
    /// the first sighting with where it was called; a set on a stream of
    /// its own is not.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_watched_fire_notes_what_waits_for_the_device() {
        use super::{arm, watched};
        use crate::device::{Buffer, rt};

        if !crate::device::present() {
            eprintln!("skipping: no CUDA device on this machine");
            return;
        }
        crate::device::ctx::bind_thread(0).expect("device 0 binds");
        arm().expect("CUPTI subscribes");
        let mut held = Buffer::zeroed(256).expect("a buffer");
        let mut stream: rt::cudaStream_t = core::ptr::null_mut();
        // SAFETY: `stream` is a live local the runtime writes.
        let made = unsafe { rt::cudaStreamCreateWithFlags(&raw mut stream, 1) };
        assert_eq!(made, rt::cudaError::cudaSuccess);
        let stream = stream.cast::<core::ffi::c_void>();
        let mut host = [0u8; 256];

        // A read waits for its own copy, on this thread's own stream, and
        // never reaches the legacy stream.
        let ((), waits) = watched(|| held.read(0, &mut host).expect("the read"));
        assert!(
            waits
                .iter()
                .any(|w| w.call.starts_with("cudaStreamSynchronize ")),
            "{waits:?}"
        );
        assert!(waits.iter().any(|w| w.from.is_some()), "{waits:?}");
        assert!(
            !waits.iter().any(|w| w.call.contains("legacy stream")),
            "{waits:?}"
        );

        // SAFETY: the span is the buffer's own.
        let ((), waits) = watched(|| unsafe {
            let at = held.ptr() as *mut core::ffi::c_void;
            assert_eq!(
                rt::cudaMemsetAsync(at, 0, 256, core::ptr::null_mut()),
                rt::cudaError::cudaSuccess
            );
        });
        assert!(
            waits.iter().any(|w| w.call.contains("legacy stream")),
            "{waits:?}"
        );

        let ((), waits) = watched(|| {
            held.zero_span_on(stream, 0, 256).expect("the set");
        });
        assert!(waits.is_empty(), "a set on its own stream: {waits:?}");
        crate::device::ctx::sync(stream).expect("the stream drains");
        // SAFETY: nothing launches on the stream again.
        unsafe { rt::cudaStreamDestroy(stream.cast()) };
    }

    #[test]
    fn a_revision_or_a_per_thread_form_reads_as_its_base_call() {
        assert_eq!(base("cuMemAlloc_v2"), "cuMemAlloc");
        assert_eq!(base("cuMemcpyHtoDAsync_v2_ptsz"), "cuMemcpyHtoDAsync");
        assert_eq!(base("cudaLaunchKernel_v7000"), "cudaLaunchKernel");
        assert_eq!(base("cudaMemcpy"), "cudaMemcpy");
        assert_eq!(base("cuMemsetD8_v2"), "cuMemsetD8");
        assert_eq!(base("cuStreamWaitEvent"), "cuStreamWaitEvent");
    }
}
