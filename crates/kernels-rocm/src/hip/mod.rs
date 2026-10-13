//! The thin layer over HIP this crate launches through: hiprtc compiles a
//! unit to a code object, a [`Module`] loads it, and [`Module::launch`] fires
//! one of its kernels on a [`Stream`].

pub mod blas;
pub mod lt;
pub mod rt;
pub mod sys;

use core::ffi::{c_char, c_int, c_void};
use std::ffi::{CStr, CString};

use crate::error::Error;

fn hip_fault(op: &'static str, call: &'static str, code: sys::HipError) -> Error {
    let text = sys::api()
        .ok()
        // SAFETY: `hipGetErrorString` returns a static, NUL-terminated string.
        .map(|api| unsafe { CStr::from_ptr((api.hip.hipGetErrorString)(code)) })
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    Error::Backend {
        op,
        detail: format!("`{call}` answered {code} ({text})"),
    }
}

fn loaded(op: &'static str) -> Result<&'static sys::Api, Error> {
    sys::api().map_err(|detail| Error::Backend { op, detail })
}

macro_rules! check {
    ($op:expr, $call:ident ( $($arg:expr),* $(,)? )) => {{
        let api = loaded($op)?;
        // SAFETY: the caller of this macro upholds the HIP contract for
        // `$call`; every pointer it passes is live for the call.
        let code = unsafe { (api.hip.$call)($($arg),*) };
        if code != sys::HIP_SUCCESS {
            return Err(hip_fault($op, stringify!($call), code));
        }
    }};
}

/// How many HIP devices this process can see.
pub fn device_count() -> Result<u32, Error> {
    let mut count: c_int = 0;
    check!("hip.device_count", hipGetDeviceCount(&raw mut count));
    Ok(u32::try_from(count).unwrap_or(0))
}

pub fn set_device(ordinal: u32) -> Result<(), Error> {
    let ordinal = c_int::try_from(ordinal).map_err(|_| Error::Backend {
        op: "hip.set_device",
        detail: format!("ordinal {ordinal} does not fit an int"),
    })?;
    check!("hip.set_device", hipSetDevice(ordinal));
    Ok(())
}

/// Threads per wavefront: 64 on CDNA (MI-series), 32 on RDNA.
pub fn warp_size(ordinal: u32) -> Result<u32, Error> {
    let mut value: c_int = 0;
    check!(
        "hip.warp_size",
        hipDeviceGetAttribute(&raw mut value, sys::ATTR_WARP_SIZE, ordinal as c_int)
    );
    Ok(value as u32)
}

pub fn multiprocessors(ordinal: u32) -> Result<u32, Error> {
    let mut value: c_int = 0;
    check!(
        "hip.multiprocessors",
        hipDeviceGetAttribute(
            &raw mut value,
            sys::ATTR_MULTIPROCESSOR_COUNT,
            ordinal as c_int
        )
    );
    Ok(value as u32)
}

/// A device allocation, freed on drop.
pub struct Buffer {
    ptr: *mut c_void,
    bytes: usize,
}

impl Buffer {
    pub fn alloc(bytes: usize) -> Result<Self, Error> {
        let mut ptr: *mut c_void = core::ptr::null_mut();
        check!("hip.alloc", hipMalloc(&raw mut ptr, bytes.max(1)));
        Ok(Self { ptr, bytes })
    }

    pub fn from_slice<T: Copy>(host: &[T]) -> Result<Self, Error> {
        let buffer = Self::alloc(size_of_val(host))?;
        check!(
            "hip.upload",
            hipMemcpy(
                buffer.ptr,
                host.as_ptr().cast(),
                buffer.bytes,
                sys::MEMCPY_HOST_TO_DEVICE
            )
        );
        Ok(buffer)
    }

    pub fn read<T: Copy + Default>(&self) -> Result<Vec<T>, Error> {
        let mut host = vec![T::default(); self.bytes / size_of::<T>()];
        check!(
            "hip.download",
            hipMemcpy(
                host.as_mut_ptr().cast(),
                self.ptr,
                size_of_val(host.as_slice()),
                sys::MEMCPY_DEVICE_TO_HOST
            )
        );
        Ok(host)
    }

    #[must_use]
    pub fn device_ptr(&self) -> u64 {
        self.ptr as u64
    }

    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Ok(api) = sys::api() {
            // SAFETY: `ptr` came from `hipMalloc` and is freed exactly once.
            unsafe { (api.hip.hipFree)(self.ptr) };
        }
    }
}

pub struct Stream(sys::Stream);

impl Stream {
    pub fn new() -> Result<Self, Error> {
        let mut stream: sys::Stream = core::ptr::null_mut();
        check!("hip.stream", hipStreamCreate(&raw mut stream));
        Ok(Self(stream))
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        check!("hip.synchronize", hipStreamSynchronize(self.0));
        Ok(())
    }

    #[must_use]
    pub fn raw(&self) -> sys::Stream {
        self.0
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Ok(api) = sys::api() {
            // SAFETY: the stream came from `hipStreamCreate` and is destroyed once.
            unsafe { (api.hip.hipStreamDestroy)(self.0) };
        }
    }
}

/// A point on a stream's timeline, for timing the work between two.
pub struct Event(rt::Event);

impl Event {
    pub fn new() -> Result<Self, Error> {
        let mut event: rt::Event = core::ptr::null_mut();
        check!("hip.event", hipEventCreate(&raw mut event));
        Ok(Self(event))
    }

    pub fn record(&self, stream: &Stream) -> Result<(), Error> {
        check!("hip.event_record", hipEventRecord(self.0, stream.0));
        Ok(())
    }

    /// Milliseconds from `start` to `self`, waiting for `self` to land.
    pub fn since(&self, start: &Event) -> Result<f32, Error> {
        check!("hip.event_sync", hipEventSynchronize(self.0));
        let mut ms = 0f32;
        check!(
            "hip.event_elapsed",
            hipEventElapsedTime(&raw mut ms, start.0, self.0)
        );
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Ok(api) = sys::api() {
            // SAFETY: the event came from `hipEventCreate` and is destroyed once.
            unsafe { (api.hip.hipEventDestroy)(self.0) };
        }
    }
}

/// A compiled code object plus the mangled names hiprtc lowered each
/// requested expression to.
pub struct Built {
    pub code: Vec<u8>,
    pub lowered: Vec<String>,
    pub log: String,
}

/// Why [`compile`] produced no code object.
#[derive(Clone, Debug)]
pub enum CompileError {
    /// The source does not compile: the same source always fails.
    Rejected(String),
    /// hiprtc itself failed (not loaded, out of memory, ...): a retry may not.
    Failed(String),
}

impl core::fmt::Display for CompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Rejected(why) | Self::Failed(why) => f.write_str(why),
        }
    }
}

impl From<CompileError> for Error {
    fn from(why: CompileError) -> Self {
        Error::Backend {
            op: "hiprtc.compile",
            detail: why.to_string(),
        }
    }
}

/// `HIPRTC_ERROR_COMPILATION`.
const RTC_COMPILATION: sys::HiprtcResult = 6;

fn rtc_fault(unit: &str, call: &'static str, code: sys::HiprtcResult, log: &str) -> CompileError {
    let text = sys::api()
        .ok()
        // SAFETY: `hiprtcGetErrorString` returns a static, NUL-terminated string.
        .map(|api| unsafe { CStr::from_ptr((api.rtc.hiprtcGetErrorString)(code)) })
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let why = format!("`{unit}`: `{call}` answered {code} ({text}){log}");
    if code == RTC_COMPILATION {
        CompileError::Rejected(why)
    } else {
        CompileError::Failed(why)
    }
}

/// Compiles `source` for `arch` (e.g. `gfx942`) and lowers each of `wanted`
/// (a kernel name or template instantiation) to the symbol to load it by;
/// `wanted` may be empty for `extern "C"` entries, loaded by their own name.
pub fn compile(
    unit: &str,
    source: &str,
    arch: &str,
    options: &[&str],
    wanted: &[&str],
) -> Result<Built, CompileError> {
    let api = sys::api().map_err(CompileError::Failed)?;
    let nul = |what: &str| CompileError::Rejected(format!("`{unit}`: {what} contains a NUL"));
    let c_source = CString::new(source).map_err(|_| nul("the source"))?;
    let c_name = CString::new(unit).map_err(|_| nul("the unit name"))?;
    let mut owned: Vec<CString> =
        vec![CString::new(format!("--offload-arch={arch}")).map_err(|_| nul("the arch"))?];
    for option in options {
        owned.push(CString::new(*option).map_err(|_| nul("an option"))?);
    }
    let option_ptrs: Vec<*const c_char> = owned.iter().map(|o| o.as_ptr()).collect();
    let c_wanted: Vec<CString> = wanted
        .iter()
        .map(|w| CString::new(*w).map_err(|_| nul("a wanted name")))
        .collect::<Result<_, _>>()?;

    let mut program: sys::Program = core::ptr::null_mut();
    // SAFETY: every pointer is a live NUL-terminated string; no headers.
    let code = unsafe {
        (api.rtc.hiprtcCreateProgram)(
            &raw mut program,
            c_source.as_ptr(),
            c_name.as_ptr(),
            0,
            core::ptr::null(),
            core::ptr::null(),
        )
    };
    if code != sys::HIPRTC_SUCCESS {
        return Err(rtc_fault(unit, "hiprtcCreateProgram", code, ""));
    }
    struct Owned(sys::Program, &'static sys::Api);
    impl Drop for Owned {
        fn drop(&mut self) {
            // SAFETY: the program came from `hiprtcCreateProgram` and is destroyed once.
            unsafe { (self.1.rtc.hiprtcDestroyProgram)(&raw mut self.0) };
        }
    }
    let program = Owned(program, api);

    for name in &c_wanted {
        // SAFETY: `program` is live and `name` is NUL-terminated.
        let code = unsafe { (api.rtc.hiprtcAddNameExpression)(program.0, name.as_ptr()) };
        if code != sys::HIPRTC_SUCCESS {
            return Err(rtc_fault(unit, "hiprtcAddNameExpression", code, ""));
        }
    }

    // SAFETY: `option_ptrs` points into `owned`, which outlives the call.
    let compiled = unsafe {
        (api.rtc.hiprtcCompileProgram)(program.0, option_ptrs.len() as c_int, option_ptrs.as_ptr())
    };
    let log = {
        let mut bytes = 0usize;
        // SAFETY: `program` is live; the buffer is sized by the first call.
        unsafe { (api.rtc.hiprtcGetProgramLogSize)(program.0, &raw mut bytes) };
        let mut text = vec![0u8; bytes.max(1)];
        unsafe { (api.rtc.hiprtcGetProgramLog)(program.0, text.as_mut_ptr().cast()) };
        String::from_utf8_lossy(&text)
            .trim_end_matches('\0')
            .trim()
            .to_owned()
    };
    if compiled != sys::HIPRTC_SUCCESS {
        return Err(rtc_fault(
            unit,
            "hiprtcCompileProgram",
            compiled,
            &format!(":\n{log}"),
        ));
    }

    let mut lowered = Vec::with_capacity(c_wanted.len());
    for name in &c_wanted {
        let mut out: *const c_char = core::ptr::null();
        // SAFETY: the lowered string is owned by `program`; it is copied out
        // before the program is destroyed.
        let code =
            unsafe { (api.rtc.hiprtcGetLoweredName)(program.0, name.as_ptr(), &raw mut out) };
        if code != sys::HIPRTC_SUCCESS || out.is_null() {
            return Err(rtc_fault(unit, "hiprtcGetLoweredName", code, ""));
        }
        lowered.push(
            unsafe { CStr::from_ptr(out) }
                .to_string_lossy()
                .into_owned(),
        );
    }

    let mut size = 0usize;
    // SAFETY: `program` compiled; the buffer is sized by the first call.
    let code = unsafe { (api.rtc.hiprtcGetCodeSize)(program.0, &raw mut size) };
    if code != sys::HIPRTC_SUCCESS {
        return Err(rtc_fault(unit, "hiprtcGetCodeSize", code, ""));
    }
    let mut image = vec![0u8; size];
    let code = unsafe { (api.rtc.hiprtcGetCode)(program.0, image.as_mut_ptr().cast()) };
    if code != sys::HIPRTC_SUCCESS {
        return Err(rtc_fault(unit, "hiprtcGetCode", code, ""));
    }
    Ok(Built {
        code: image,
        lowered,
        log,
    })
}

/// A loaded code object, unloaded on drop.
pub struct Module(sys::Module);

impl Module {
    pub fn load(code: &[u8]) -> Result<Self, Error> {
        let mut module: sys::Module = core::ptr::null_mut();
        check!(
            "hip.module_load",
            hipModuleLoadData(&raw mut module, code.as_ptr().cast())
        );
        Ok(Self(module))
    }

    pub fn function(&self, symbol: &str) -> Result<Function, Error> {
        let name = CString::new(symbol).map_err(|_| Error::Backend {
            op: "hip.function",
            detail: format!("`{symbol}` contains a NUL"),
        })?;
        let mut function: sys::Function = core::ptr::null_mut();
        check!(
            "hip.function",
            hipModuleGetFunction(&raw mut function, self.0, name.as_ptr())
        );
        Ok(Function(function))
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if let Ok(api) = sys::api() {
            // SAFETY: the module came from `hipModuleLoadData` and is unloaded once.
            unsafe { (api.hip.hipModuleUnload)(self.0) };
        }
    }
}

/// A kernel inside a [`Module`]; valid only while that module is loaded.
#[derive(Clone, Copy)]
pub struct Function(sys::Function);

impl Function {
    /// Enqueues the kernel. `params` holds one pointer per kernel argument,
    /// each pointing at that argument's value, as `hipModuleLaunchKernel`
    /// reads them.
    ///
    /// # Safety
    /// The argument count, order and types in `params` must match the
    /// kernel's signature, and every device pointer among them must stay
    /// live until the stream reaches the launch.
    pub unsafe fn launch(
        self,
        stream: &Stream,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), Error> {
        check!(
            "hip.launch",
            hipModuleLaunchKernel(
                self.0,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_bytes,
                stream.raw(),
                params.as_mut_ptr(),
                core::ptr::null_mut(),
            )
        );
        Ok(())
    }
}

/// Which way a [`memcpy_async`] moves bytes.
#[derive(Clone, Copy, Debug)]
pub enum Direction {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

/// Enqueues `bytes` from `src` to `dst` on `stream`; `Err` names the HIP
/// code it answered.
///
/// # Safety
/// Both ranges must be live for `bytes` until the stream reaches the copy,
/// and on the side `kind` says.
pub unsafe fn memcpy_async(
    dst: *mut c_void,
    src: *const c_void,
    bytes: usize,
    kind: Direction,
    stream: sys::Stream,
) -> Result<(), sys::HipError> {
    let api = sys::api().map_err(|_| -1)?;
    let kind = match kind {
        Direction::HostToDevice => sys::MEMCPY_HOST_TO_DEVICE,
        Direction::DeviceToHost => sys::MEMCPY_DEVICE_TO_HOST,
        Direction::DeviceToDevice => sys::MEMCPY_DEVICE_TO_DEVICE,
    };
    // SAFETY: the caller vouches for both ranges and the stream.
    let code = unsafe { (api.hip.hipMemcpyAsync)(dst, src, bytes, kind, stream) };
    if code == sys::HIP_SUCCESS {
        Ok(())
    } else {
        Err(code)
    }
}

/// The marketing name of device `ordinal` ("AMD Instinct MI300X"), from
/// `hipDeviceProp_tR0600.name`, its first field.
#[must_use]
pub fn device_name(ordinal: u32) -> Option<String> {
    let api = sys::api().ok()?;
    let mut prop = vec![0u8; sys::DEVICE_PROP_BYTES];
    // SAFETY: `prop` is a zeroed buffer the size of `hipDeviceProp_tR0600`.
    let code = unsafe {
        (api.hip.hipGetDevicePropertiesR0600)(prop.as_mut_ptr().cast(), ordinal as c_int)
    };
    if code != sys::HIP_SUCCESS {
        return None;
    }
    let name = CStr::from_bytes_until_nul(&prop[..256]).ok()?;
    Some(name.to_string_lossy().into_owned())
}
