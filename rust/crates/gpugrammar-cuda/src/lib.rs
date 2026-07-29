//! The device runtime: the fatbin, and enough of the CUDA driver API to launch
//! what is in it.
//!
//! Three properties this exists to have, none of which the Triton runtime had:
//!
//! - **No JIT.** The kernels are SASS by the time the wheel is built. Triton
//!   compiles on first use, which vLLM reports as a latency spike during
//!   inference, and which no amount of warming fully removes because a new
//!   specialisation can appear at any time.
//! - **No toolkit at install.** The fatbin is bytes inside the shared object.
//! - **Launched on the caller's stream**, so a launch made during PyTorch's
//!   graph capture is recorded rather than run. That is the whole architecture:
//!   the parser has to be *inside* a serving engine's decode graph.
//!
//! The driver API rather than the runtime API because loading a fatbin from
//! memory is `cuModuleLoadData`, and because the runtime API would drag in
//! `libcudart` and its own context management alongside PyTorch's.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

/// The kernels, as SASS for every architecture `build.rs` names.
///
/// Empty when the crate was built without a CUDA toolkit, which is allowed so
/// that the Rust front end stays buildable anywhere; `available()` reports it.
static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gpugrammar.fatbin"));

pub type CUresult = c_int;
pub const CUDA_SUCCESS: CUresult = 0;

type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;

// The driver is opened at run time rather than linked at build time. Three
// reasons, in order of how much they cost when ignored:
//
// - `libcuda` is a *driver* library, not a redistributable one. Linking it
//   makes the wheel declare a dependency manylinux does not allow, which is
//   why maturin then wants patchelf to rewrite the result.
// - The build then needs the toolkit's stub, so a machine that can compile the
//   kernels but has no stub cannot link.
// - A machine with no driver at all should report "no CUDA backend", not fail
//   to load the extension - and `dlopen` returning null says exactly that.
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

type FnInit = unsafe extern "C" fn(c_uint) -> CUresult;
type FnLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type FnGetFunction = unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
#[allow(clippy::type_complexity)]
type FnLaunch = unsafe extern "C" fn(
    CUfunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
type FnErrorString = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;

struct Driver {
    init: FnInit,
    load_data: FnLoadData,
    get_function: FnGetFunction,
    launch: FnLaunch,
    error_string: Option<FnErrorString>,
}

// Safety: these are function pointers into a library that stays loaded for the
// life of the process, and the driver API is thread-safe.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();

fn driver() -> Result<&'static Driver, String> {
    DRIVER
        .get_or_init(|| {
            // `.so.1` first: that is the versioned name a driver installs.
            // Plain `.so` is the toolkit's stub and is only a fallback, since
            // loading a stub would find symbols that then fail at every call.
            let mut handle = std::ptr::null_mut();
            for name in ["libcuda.so.1\0", "libcuda.so\0"] {
                // Safety: a NUL-terminated name, and flags the loader defines.
                handle = unsafe { dlopen(name.as_ptr().cast(), RTLD_NOW | RTLD_GLOBAL) };
                if !handle.is_null() {
                    break;
                }
            }
            if handle.is_null() {
                return Err("no NVIDIA driver: libcuda.so.1 could not be opened".to_string());
            }
            // Safety: each symbol is looked up by its documented name and cast
            // to the signature the driver API documents for it.
            unsafe {
                let find = |name: &str| -> Result<*mut c_void, String> {
                    let symbol = CString::new(name).map_err(|_| "bad symbol".to_string())?;
                    let found = dlsym(handle, symbol.as_ptr());
                    if found.is_null() {
                        return Err(format!("libcuda has no `{name}`"));
                    }
                    Ok(found)
                };
                Ok(Driver {
                    init: std::mem::transmute::<*mut c_void, FnInit>(find("cuInit")?),
                    load_data: std::mem::transmute::<*mut c_void, FnLoadData>(find(
                        "cuModuleLoadData",
                    )?),
                    get_function: std::mem::transmute::<*mut c_void, FnGetFunction>(find(
                        "cuModuleGetFunction",
                    )?),
                    launch: std::mem::transmute::<*mut c_void, FnLaunch>(find("cuLaunchKernel")?),
                    error_string: find("cuGetErrorString")
                        .ok()
                        .map(|f| std::mem::transmute::<*mut c_void, FnErrorString>(f)),
                })
            }
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Is there a CUDA backend in this build at all?
pub fn available() -> bool {
    !FATBIN.is_empty()
}

/// How large the embedded fatbin is. Reported so a caller can tell a build
/// with kernels from one without, and so the wheel's size is attributable.
pub fn fatbin_bytes() -> usize {
    FATBIN.len()
}

pub fn describe(result: CUresult) -> String {
    if result == CUDA_SUCCESS {
        return "success".to_string();
    }
    if let Ok(driver) = driver()
        && let Some(error_string) = driver.error_string
    {
        let mut message: *const c_char = std::ptr::null();
        // Safety: the driver writes a pointer to a static string, or leaves it
        // null for a code it does not know.
        unsafe {
            if error_string(result, &mut message) == CUDA_SUCCESS && !message.is_null() {
                return CStr::from_ptr(message).to_string_lossy().into_owned();
            }
        }
    }
    format!("CUDA error {result}")
}

/// The loaded module. One per process.
///
/// A module is bound to a context, and PyTorch keeps one primary context per
/// device for the life of the process, so loading once against whatever
/// context is current when we are first called is correct for the single-device
/// case this engine already assumes elsewhere (`DeviceGrammar.device`).
struct Module(CUmodule);

// Safety: a `CUmodule` is a handle the driver owns; it is not mutated here
// after loading, and `cuModuleGetFunction` is documented as thread-safe.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

static MODULE: OnceLock<Result<Module, String>> = OnceLock::new();

fn module() -> Result<CUmodule, String> {
    let loaded = MODULE.get_or_init(|| {
        if FATBIN.is_empty() {
            return Err(
                "this build has no CUDA kernels: nvcc was not found when it was compiled"
                    .to_string(),
            );
        }
        let driver = driver()?;
        // Safety: `cuInit` is idempotent and required before any other driver
        // call. PyTorch will normally have done it; doing it again is defined.
        let started = unsafe { (driver.init)(0) };
        if started != CUDA_SUCCESS {
            return Err(format!("cuInit: {}", describe(started)));
        }
        let mut handle: CUmodule = std::ptr::null_mut();
        // Safety: the image is a fatbin produced by nvcc at build time and
        // lives for the life of the process, being a `'static` byte slice.
        let result = unsafe { (driver.load_data)(&mut handle, FATBIN.as_ptr().cast()) };
        if result != CUDA_SUCCESS {
            return Err(format!("cuModuleLoadData: {}", describe(result)));
        }
        Ok(Module(handle))
    });
    loaded.as_ref().map(|m| m.0).map_err(Clone::clone)
}

/// A kernel, looked up once and launched many times.
#[derive(Clone, Copy)]
pub struct Kernel(CUfunction);

// Safety: as `Module` - a driver-owned handle, read only after lookup.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    pub fn named(name: &str) -> Result<Self, String> {
        let module = module()?;
        let driver = driver()?;
        let symbol = CString::new(name).map_err(|_| "kernel name has a NUL".to_string())?;
        let mut function: CUfunction = std::ptr::null_mut();
        // Safety: `module` is loaded and `symbol` is NUL-terminated.
        let result = unsafe { (driver.get_function)(&mut function, module, symbol.as_ptr()) };
        if result != CUDA_SUCCESS {
            return Err(format!("no kernel `{name}`: {}", describe(result)));
        }
        Ok(Self(function))
    }

    /// Launch on `stream`, which is the caller's - PyTorch's current stream in
    /// practice. Passing it rather than using the default is what makes a
    /// launch during graph capture get *recorded* instead of run.
    ///
    /// # Safety
    ///
    /// `arguments` must be pointers to values of exactly the types and in
    /// exactly the order the kernel declares, and any device pointer among
    /// them must be valid for the launch. Nothing here can check that: the
    /// driver takes `void**` and the kernel's signature is not visible from
    /// Rust. This is the one unsafe seam in the backend, and every caller of
    /// it lives in `gpugrammar-py` beside the Python that supplies the
    /// tensors.
    pub unsafe fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        stream: u64,
        arguments: &mut [*mut c_void],
    ) -> Result<(), String> {
        let driver = driver()?;
        // Safety: the caller's contract, plus a stream handle that PyTorch
        // owns and keeps alive for the duration of the call.
        let result = unsafe {
            (driver.launch)(
                self.0,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_bytes,
                stream as *mut c_void,
                arguments.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if result != CUDA_SUCCESS {
            return Err(format!("launch: {}", describe(result)));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deliberately does not need a GPU. What it checks is that the *build*
    // produced kernels, which is the thing that silently stops being true.
    #[test]
    fn the_build_embedded_a_fatbin() {
        if std::env::var_os("GPUGRAMMAR_SKIP_CUDA").is_some() {
            assert!(!available());
            return;
        }
        assert!(available(), "no fatbin was embedded; was nvcc found?");
        assert!(fatbin_bytes() > 1024, "fatbin is {} bytes", fatbin_bytes());
    }

    #[test]
    fn an_unknown_error_still_describes_itself() {
        assert_eq!(describe(CUDA_SUCCESS), "success");
        assert!(!describe(999_999).is_empty());
    }
}
