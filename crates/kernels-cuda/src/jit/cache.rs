#![allow(unsafe_code)]
use std::path::PathBuf;
use std::sync::OnceLock;

use cudarc::driver::sys as dr;
use cudarc::runtime::sys as rt;

use crate::jit::{Fault, Root, nvrtc};

pub struct Resolved {
    #[allow(dead_code)]
    module: dr::CUmodule,
    pub function: dr::CUfunction,
    /// Each parameter's size in bytes, as the driver reads the image.
    pub params: Box<[usize]>,
}

// SAFETY: both fields are opaque driver handles that are written once, in
// `load`, and only read after. The driver API is thread-safe and takes these
// handles from any thread; nothing here is tied to the creating thread.
unsafe impl Send for Resolved {}

// SAFETY: as `Send`: `&Resolved` only hands out copies of the handles, and
// the driver serialises whatever it does with them internally.
unsafe impl Sync for Resolved {}

/// The ordinal of the device bound to this thread.
pub(crate) fn current_device() -> Result<usize, Fault> {
    let mut ordinal: i32 = 0;
    // SAFETY: `ordinal` is a local out-pointer.
    let code = unsafe { rt::cudaGetDevice(&raw mut ordinal) };
    if code != rt::cudaError::cudaSuccess {
        return Err(Fault::Device {
            call: "cudaGetDevice",
            code: code as i32,
        });
    }
    usize::try_from(ordinal).map_err(|_| Fault::Device {
        call: "cudaGetDevice",
        code: ordinal,
    })
}

/// `instantiation` of `root` loaded on the current device: from the disk
/// cache, or compiled and written there.
pub(crate) fn load(root: &Root, instantiation: &str) -> Result<Resolved, Fault> {
    let plain = arch().ok_or(Fault::Device {
        call: "cuDeviceGet",
        code: -1,
    })?;
    static TCGEN05: OnceLock<bool> = OnceLock::new();
    let tcgen05 = *TCGEN05.get_or_init(|| features().tcgen05);
    let target = root.target(plain, tcgen05).ok_or_else(|| Fault::Compile {
        unit: root.name,
        log: format!(
            "`{}` compiles for an arch-specific target and {plain} has none: \
             the unit is for another chip",
            root.name
        ),
    })?;
    let arch = target.as_str();
    let key = &root.key(instantiation, arch);
    let started = std::time::Instant::now();
    let (cubin, mangled, compiled) = match read_disk(key) {
        Some(hit) => (hit.0, hit.1, false),
        None => {
            let built = nvrtc::compile_text(&nvrtc::Job {
                name: root.name,
                source: root.text.to_owned(),
                arch,
                options: root.options,
                headers: root.header_set(),
                floor: root.floor,
                wanted: std::slice::from_ref(&instantiation.to_owned()),
            })?;
            if !built.log.trim().is_empty() {
                tracing::warn!(
                    root = root.name,
                    instantiation,
                    arch,
                    log = %built.log,
                    "a device instantiation compiled with something to say"
                );
            }
            let mangled = built
                .lowered
                .into_iter()
                .next()
                .ok_or_else(|| Fault::Compile {
                    unit: root.name,
                    log: format!("`{instantiation}` compiled and NVRTC named nothing for it"),
                })?;
            write_disk(key, &built.cubin, &mangled);
            (built.cubin, mangled, true)
        }
    };

    let module = load_image(root.name, &cubin)?;
    let function = entry_by_name(root.name, module, instantiation, &mangled).inspect_err(|_| {
        // SAFETY: `module` was just loaded by `load_image`, nothing has
        // resolved a function out of it, and it is unloaded once here.
        unsafe { dr::cuModuleUnload(module) };
    })?;
    tracing::info!(
        root = root.name,
        instantiation,
        arch,
        compiled,
        ms = started.elapsed().as_secs_f64() * 1e3,
        "resolved a device instantiation"
    );
    // Loaded now, not at its first launch (lazy loading's), which may come
    // inside a capture.
    // SAFETY: `function` was just resolved from the module loaded above.
    let code = unsafe { dr::cuFuncLoad(function) };
    if code != dr::CUresult::CUDA_SUCCESS {
        // SAFETY: as below: `module` is loaded, and unloaded once here.
        unsafe { dr::cuModuleUnload(module) };
        return Err(Fault::Device {
            call: "cuFuncLoad",
            code: code as i32,
        });
    }
    let params = param_sizes(function).inspect_err(|_| {
        // SAFETY: as above: `module` is loaded, and unloaded once here.
        unsafe { dr::cuModuleUnload(module) };
    })?;
    Ok(Resolved {
        module,
        function,
        params,
    })
}

/// `function`'s parameter sizes, asked of the driver once at load so a fire
/// can hold its arguments to them.
fn param_sizes(function: dr::CUfunction) -> Result<Box<[usize]>, Fault> {
    let mut sizes = Vec::new();
    loop {
        let (mut offset, mut size) = (0usize, 0usize);
        // SAFETY: `function` was just resolved from a loaded module, and
        // both out-pointers are locals.
        let code = unsafe {
            dr::cuFuncGetParamInfo(function, sizes.len(), &raw mut offset, &raw mut size)
        };
        match code {
            dr::CUresult::CUDA_SUCCESS => sizes.push(size),
            // past the last parameter
            dr::CUresult::CUDA_ERROR_INVALID_VALUE => return Ok(sizes.into_boxed_slice()),
            other => {
                return Err(Fault::Device {
                    call: "cuFuncGetParamInfo",
                    code: other as i32,
                });
            }
        }
    }
}

fn load_image(root: &'static str, image: &[u8]) -> Result<dr::CUmodule, Fault> {
    if image.is_empty() {
        return Err(Fault::Compile {
            unit: root,
            log: "the compile produced an empty image, so there is nothing to load".into(),
        });
    }
    let mut module: dr::CUmodule = std::ptr::null_mut();

    // SAFETY: `module` is a local out-pointer and `image` is a non-empty
    // byte slice that outlives the call; the driver copies the image. The
    // driver reads its length from the cubin's own ELF headers, so the
    // bytes must be a whole image: NVRTC's output, or a cache file whose
    // trailing hash `decode` checked against what `write_disk` stored.
    let code = unsafe { dr::cuModuleLoadData(&raw mut module, image.as_ptr().cast()) };
    if code == dr::CUresult::CUDA_SUCCESS {
        Ok(module)
    } else {
        Err(Fault::Device {
            call: "cuModuleLoadData",
            code: code as i32,
        })
    }
}

fn entry_by_name(
    root: &'static str,
    module: dr::CUmodule,
    instantiation: &str,
    mangled: &str,
) -> Result<dr::CUfunction, Fault> {
    let Ok(c_name) = std::ffi::CString::new(mangled) else {
        return Err(Fault::Compile {
            unit: root,
            log: format!("the lowered name for `{instantiation}` contains a NUL"),
        });
    };
    let mut function: dr::CUfunction = std::ptr::null_mut();

    // SAFETY: `module` is the live module `load_image` returned, `c_name` is
    // a CString that outlives the call, and `function` is a local
    // out-pointer.
    let code = unsafe { dr::cuModuleGetFunction(&raw mut function, module, c_name.as_ptr()) };
    match code {
        dr::CUresult::CUDA_SUCCESS => Ok(function),
        dr::CUresult::CUDA_ERROR_NOT_FOUND => Err(Fault::Compile {
            unit: root,
            log: format!("`{instantiation}` compiled and is not in the image"),
        }),
        other => Err(Fault::Device {
            call: "cuModuleGetFunction",
            code: other as i32,
        }),
    }
}

fn disk_path(key: &str) -> Option<PathBuf> {
    let digest = crate::source::fnv1a64(key.as_bytes());
    Some(crate::disk::dir(crate::disk::CUBINS)?.join(format!("{digest:016x}.cubin")))
}

fn read_disk(key: &str) -> Option<(Vec<u8>, String)> {
    decode(key, &std::fs::read(disk_path(key)?).ok()?)
}

/// A cache file's entry, if it is whole and under `key`: a file whose
/// trailing hash disagrees with its body (a torn write, a bad disk) is a
/// miss, and the recompile overwrites it.
fn decode(key: &str, bytes: &[u8]) -> Option<(Vec<u8>, String)> {
    let (body, sum) = bytes.split_at_checked(bytes.len().checked_sub(8)?)?;
    if u64::from_le_bytes(sum.try_into().ok()?) != crate::source::fnv1a64(body) {
        return None;
    }
    let (stored_key, rest) = take_str(body)?;
    if stored_key != key {
        return None;
    }
    let (mangled, cubin) = take_str(rest)?;
    Some((cubin.to_vec(), mangled.to_owned()))
}

fn take_str(bytes: &[u8]) -> Option<(&str, &[u8])> {
    let (len, rest) = bytes.split_at_checked(4)?;
    let len = u32::from_le_bytes(len.try_into().ok()?) as usize;
    let (text, rest) = rest.split_at_checked(len)?;
    Some((std::str::from_utf8(text).ok()?, rest))
}

fn write_disk(key: &str, cubin: &[u8], mangled: &str) {
    let Some(path) = disk_path(key) else { return };
    let Some(parent) = path.parent() else { return };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let out = encode(key, cubin, mangled);
    let staging = path.with_extension(format!("cubin.{}", std::process::id()));
    if std::fs::write(&staging, &out).is_ok() && std::fs::rename(&staging, &path).is_err() {
        let _ = std::fs::remove_file(&staging);
    }
}

/// `key`, `mangled` and `cubin`, then the FNV-1a hash of all three.
fn encode(key: &str, cubin: &[u8], mangled: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(cubin.len() + mangled.len() + key.len() + 16);
    put_str(&mut out, key);
    put_str(&mut out, mangled);
    out.extend_from_slice(cubin);
    let sum = crate::source::fnv1a64(&out);
    out.extend_from_slice(&sum.to_le_bytes());
    out
}

fn put_str(out: &mut Vec<u8>, text: &str) {
    let len = u32::try_from(text.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&text.as_bytes()[..len as usize]);
}

pub(crate) fn bind_context() -> Result<(), Fault> {
    use std::cell::Cell;

    thread_local! {
        static BOUND: Cell<bool> = const { Cell::new(false) };
    }

    if BOUND.with(Cell::get) {
        return Ok(());
    }

    // A context already current (the runtime's, bound by any call that made
    // it, a capture's on the thread that opened one) is the one: a query,
    // which a capture open on this thread allows. A synchronizing call to
    // bind it (`cudaFree(nullptr)`) does not: on a thread's first kernel
    // resolution inside a capture it answered `cudaErrorStreamCaptureUnsupported`
    // and invalidated the capture.
    let driver = |call: &'static str, code: dr::CUresult| {
        if code == dr::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(Fault::Device {
                call,
                code: code as i32,
            })
        }
    };
    // The driver first: in a process no runtime call has touched yet, the
    // query answers `CUDA_ERROR_NOT_INITIALIZED`. Once initialized, `cuInit`
    // returns at once and waits on nothing.
    // SAFETY: takes only its flags, which must be 0.
    driver("cuInit", unsafe { dr::cuInit(0) })?;
    let mut current: dr::CUcontext = std::ptr::null_mut();
    // SAFETY: `current` is a local out-pointer.
    driver("cuCtxGetCurrent", unsafe {
        dr::cuCtxGetCurrent(&raw mut current)
    })?;
    if current.is_null() {
        // None yet, so no capture is open on this thread: the device's
        // primary context, retained and made current, as the runtime binds
        // it; the retain is the thread's for the process.
        let ordinal = i32::try_from(current_device()?).unwrap_or(0);
        let mut device: dr::CUdevice = 0;
        let mut primary: dr::CUcontext = std::ptr::null_mut();
        // SAFETY: each out-pointer is a local, `ordinal` the runtime's own
        // device for this thread, and `primary` the context just retained.
        unsafe {
            driver("cuDeviceGet", dr::cuDeviceGet(&raw mut device, ordinal))?;
            driver(
                "cuDevicePrimaryCtxRetain",
                dr::cuDevicePrimaryCtxRetain(&raw mut primary, device),
            )?;
            driver("cuCtxSetCurrent", dr::cuCtxSetCurrent(primary))?;
        }
    }
    BOUND.with(|bound| bound.set(true));
    Ok(())
}

#[must_use]
pub fn arch() -> Option<&'static str> {
    use dr::CUdevice_attribute as Attr;

    static ARCH: OnceLock<Option<String>> = OnceLock::new();
    ARCH.get_or_init(|| {
        cudarc::driver::result::init().ok()?;
        let mut ordinal: i32 = 0;

        // SAFETY: `ordinal` is a local out-pointer.
        let code = unsafe { rt::cudaGetDevice(&raw mut ordinal) };
        if code != rt::cudaError::cudaSuccess {
            return None;
        }
        let mut device: dr::CUdevice = 0;

        // SAFETY: the driver was initialised just above and `device` is a
        // local out-pointer.
        let code = unsafe { dr::cuDeviceGet(&raw mut device, ordinal) };
        if code != dr::CUresult::CUDA_SUCCESS {
            return None;
        }
        let major = attribute(device, Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = attribute(device, Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Some(format!("sm_{major}{minor}"))
    })
    .as_deref()
}

/// What the current device's arch-specific target admits under the NVRTC
/// this process loaded: all false off a device, or when NVRTC is absent.
#[must_use]
pub fn features() -> crate::jit::Features {
    let nvrtc = nvrtc::version().unwrap_or(crate::jit::Toolchain::ANY);
    crate::jit::Features::of(arch().unwrap_or(""), nvrtc)
}

fn attribute(device: dr::CUdevice, which: dr::CUdevice_attribute) -> Option<i32> {
    let mut value: i32 = 0;

    // SAFETY: `value` is a local out-pointer; a bad `device` or attribute
    // comes back as an error code, not a write.
    let code = unsafe { dr::cuDeviceGetAttribute(&raw mut value, which, device) };
    (code == dr::CUresult::CUDA_SUCCESS).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::{decode, encode};

    #[test]
    fn a_damaged_cache_file_is_a_miss() {
        let file = encode("key", b"\x7fELF image", "_Z6kernel");
        assert_eq!(
            decode("key", &file),
            Some((b"\x7fELF image".to_vec(), "_Z6kernel".to_owned()))
        );
        assert_eq!(decode("other", &file), None);
        for at in 0..file.len() {
            let mut torn = file.clone();
            torn[at] ^= 0x01;
            assert_eq!(decode("key", &torn), None, "a flip at byte {at} was loaded");
        }
        assert_eq!(decode("key", &file[..file.len() - 1]), None);
    }
}
