use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use cudarc::runtime::sys as rt;

use cudarc::driver::sys as dr;

use crate::jit::Root;
use crate::runtime::{Error, nvrtc};

/// One instantiation, compiled and resolved to an entry point.
pub struct Resolved {
    /// The loaded image. Kept so the module outlives the function handle.
    #[allow(dead_code)]
    module: dr::CUmodule,
    /// The entry point.
    pub function: dr::CUfunction,
}

// SAFETY: `CUmodule` and `CUfunction` are context-scoped handles, and this
// process binds every thread to the same primary context. Nothing here
// mutates either after `resolve` returns.
unsafe impl Send for Resolved {}
// SAFETY: as above -- every read is of an immutable handle.
unsafe impl Sync for Resolved {}

/// One instantiation's slot: compiled at most once, in this process.
type Slot = OnceLock<Result<Resolved, Error>>;

/// The slot for a key, created empty if this is the first ask.
///
/// The map lock is held only across the lookup, never across a compile: two
/// threads wanting two different symbols must not serialise on each other, and
/// two wanting the SAME one meet in the slot's `OnceLock` instead.
fn slot(key: &str) -> &'static Slot {
    static SLOTS: OnceLock<Mutex<HashMap<String, &'static Slot>>> = OnceLock::new();
    let mut map = SLOTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(found) = map.get(key) {
        return found;
    }
    // Leaked on purpose. Modules are never unloaded (see `resolve`), so a slot
    // is live for the rest of the process either way, and leaking is what lets
    // the borrow escape the lock.
    let fresh: &'static Slot = Box::leak(Box::new(OnceLock::new()));
    map.insert(key.to_owned(), fresh);
    fresh
}

/// The entry point for one instantiation of `root`, compiling it on first ask.
///
/// **Modules are never unloaded.** The smem-cap memo in `runtime::module` is
/// keyed by function ADDRESS, and the driver reuses addresses after an unload;
/// evicting a module would make that memo answer for a different kernel. If
/// eviction is ever wanted, that memo has to be re-keyed first.
///
/// # Errors
///
/// If there is no device, or NVRTC will not compile the root, or the
/// instantiation is not in the image.
pub fn resolve(root: &'static Root, instantiation: &str) -> Result<&'static Resolved, Error> {
    bind_context()?;
    let arch = arch().ok_or(Error::NoDevice)?;
    let key = root.key(instantiation, arch);
    slot(&key).get_or_init(|| load(root, instantiation, &key, arch)).as_ref().map_err(Clone::clone)
}

/// Compile (or read back) one instantiation and load it.
fn load(
    root: &'static Root,
    instantiation: &str,
    key: &str,
    arch: &str,
) -> Result<Resolved, Error> {
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
                device_link: root.needs_device_runtime(),
            })
            .map_err(|why| Error::Compile { unit: root.name, why: why.to_string() })?;
            if !built.log.trim().is_empty() {
                tracing::warn!(
                    root = root.name,
                    instantiation,
                    arch,
                    log = %built.log,
                    "a device instantiation compiled with something to say"
                );
            }
            let mangled = built.lowered.into_iter().next().ok_or_else(|| Error::Compile {
                unit: root.name,
                why: format!("`{instantiation}` compiled and NVRTC named nothing for it"),
            })?;
            write_disk(key, &built.cubin, &mangled);
            (built.cubin, mangled, true)
        }
    };

    let module = load_image(root.name, &cubin)?;
    let function = entry_by_name(root.name, module, instantiation, &mangled).inspect_err(|_| {
        // SAFETY: `module` loaded, nothing else holds it, and no launch has
        // been issued against it.
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
    Ok(Resolved { module, function })
}

/// `cuModuleLoadData`, with the empty image refused rather than handed over.
fn load_image(root: &'static str, image: &[u8]) -> Result<dr::CUmodule, Error> {
    if image.is_empty() {
        return Err(Error::Compile {
            unit: root,
            why: "the compile produced an empty image, so there is nothing to load".into(),
        });
    }
    let mut module: dr::CUmodule = std::ptr::null_mut();
    // SAFETY: `image` is a live byte image and `module` a live out-parameter.
    let code = unsafe { dr::cuModuleLoadData(&raw mut module, image.as_ptr().cast()) };
    if code == dr::CUresult::CUDA_SUCCESS {
        Ok(module)
    } else {
        Err(Error::Driver { what: "cuModuleLoadData", code: code as i32, why: format!("{code:?}") })
    }
}

/// The entry point, by the mangled name the compile gave it.
fn entry_by_name(
    root: &'static str,
    module: dr::CUmodule,
    instantiation: &str,
    mangled: &str,
) -> Result<dr::CUfunction, Error> {
    let Ok(c_name) = std::ffi::CString::new(mangled) else {
        return Err(Error::Compile {
            unit: root,
            why: format!("the lowered name for `{instantiation}` contains a NUL"),
        });
    };
    let mut function: dr::CUfunction = std::ptr::null_mut();
    // SAFETY: `module` is loaded and `c_name` outlives the call.
    let code = unsafe { dr::cuModuleGetFunction(&raw mut function, module, c_name.as_ptr()) };
    match code {
        dr::CUresult::CUDA_SUCCESS => Ok(function),
        dr::CUresult::CUDA_ERROR_NOT_FOUND => Err(Error::Compile {
            unit: root,
            why: format!("`{instantiation}` compiled and is not in the image"),
        }),
        other => Err(Error::Driver {
            what: "cuModuleGetFunction",
            code: other as i32,
            why: format!("{other:?}"),
        }),
    }
}

/// Where a compiled instantiation is kept between processes.
///
/// The same convention the cuBLAS tactic cache uses. `None` means no cache:
/// every miss then recompiles, which is slow and correct.
fn disk_path(key: &str) -> Option<PathBuf> {
    let base =
        std::env::var("XDG_CACHE_HOME").ok().filter(|s| !s.is_empty()).map(PathBuf::from).or_else(
            || {
                std::env::var("HOME")
                    .ok()
                    .filter(|s| !s.is_empty())
                    .map(|home| PathBuf::from(home).join(".cache"))
            },
        )?;
    // The key is a path-shaped sentence with slashes in it; the file name is
    // its digest, so one key is one file and no key can escape the directory.
    let digest = crate::source::fnv1a64(key.as_bytes());
    Some(base.join("pie").join("kernels").join(format!("{digest:016x}.cubin")))
}

/// A cached image and its mangled name, if this key has one.
///
/// The file holds the key it was written under, so a digest collision reads
/// back as a miss rather than as the wrong kernel.
fn read_disk(key: &str) -> Option<(Vec<u8>, String)> {
    let bytes = std::fs::read(disk_path(key)?).ok()?;
    let (stored_key, rest) = take_str(&bytes)?;
    if stored_key != key {
        return None;
    }
    let (mangled, cubin) = take_str(rest)?;
    Some((cubin.to_vec(), mangled.to_owned()))
}

/// A length-prefixed string off the front of `bytes`, and what follows it.
fn take_str(bytes: &[u8]) -> Option<(&str, &[u8])> {
    let (len, rest) = bytes.split_at_checked(4)?;
    let len = u32::from_le_bytes(len.try_into().ok()?) as usize;
    let (text, rest) = rest.split_at_checked(len)?;
    Some((std::str::from_utf8(text).ok()?, rest))
}

/// Keep this image for the next process. A failure to write is not a failure.
fn write_disk(key: &str, cubin: &[u8], mangled: &str) {
    let Some(path) = disk_path(key) else { return };
    let Some(parent) = path.parent() else { return };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let mut out = Vec::with_capacity(cubin.len() + mangled.len() + key.len() + 8);
    put_str(&mut out, key);
    put_str(&mut out, mangled);
    out.extend_from_slice(cubin);
    // Written beside the target and renamed, so a reader never sees a partial
    // file: two processes may compile the same symbol at once.
    let staging = path.with_extension(format!("cubin.{}", std::process::id()));
    if std::fs::write(&staging, &out).is_ok() && std::fs::rename(&staging, &path).is_err() {
        let _ = std::fs::remove_file(&staging);
    }
}

/// A length-prefixed string.
fn put_str(out: &mut Vec<u8>, text: &str) {
    let len = u32::try_from(text.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&text.as_bytes()[..len as usize]);
}

#[cfg(test)]
mod tests {
    use super::{put_str, take_str};

    /// What `write_disk` lays down is what `read_disk` takes back up.
    #[test]
    fn a_record_round_trips() {
        let mut out = Vec::new();
        put_str(&mut out, "jit/rope/sm_90/…");
        put_str(&mut out, "_ZN4rope6rotateE");
        out.extend_from_slice(&[1, 2, 3, 4]);

        let (key, rest) = take_str(&out).expect("a key");
        assert_eq!(key, "jit/rope/sm_90/…");
        let (mangled, cubin) = take_str(rest).expect("a name");
        assert_eq!(mangled, "_ZN4rope6rotateE");
        assert_eq!(cubin, &[1, 2, 3, 4]);
    }

    /// A truncated file reads as a miss, not as a panic.
    #[test]
    fn a_truncated_record_is_a_miss() {
        let mut out = Vec::new();
        put_str(&mut out, "a key");
        assert!(take_str(&out[..3]).is_none(), "not even the length");
        assert!(take_str(&out[..6]).is_none(), "a length longer than what follows");
    }
}

/// Make this thread's current context the device's primary one.
pub fn bind_context() -> Result<(), Error> {
    use std::cell::Cell;

    thread_local! {
        static BOUND: Cell<bool> = const { Cell::new(false) };
    }

    if BOUND.with(Cell::get) {
        return Ok(());
    }
    // SAFETY: a null pointer is what `cudaFree` documents as the no-op that
    let code = unsafe { rt::cudaFree(std::ptr::null_mut()) };
    if code != rt::cudaError::cudaSuccess {
        return Err(Error::NoDevice);
    }
    BOUND.with(|bound| bound.set(true));
    Ok(())
}

/// The architecture of the device this process is bound to, as `sm_XY`.
#[must_use]
pub fn arch() -> Option<&'static str> {
    use dr::CUdevice_attribute as Attr;

    static ARCH: OnceLock<Option<String>> = OnceLock::new();
    ARCH.get_or_init(|| {
        cudarc::driver::result::init().ok()?;
        let mut ordinal: i32 = 0;
        // SAFETY: `ordinal` is a live out-parameter for the call's duration.
        let code = unsafe { rt::cudaGetDevice(&raw mut ordinal) };
        if code != rt::cudaError::cudaSuccess {
            return None;
        }
        let mut device: dr::CUdevice = 0;
        // SAFETY: `device` is a valid, writable handle slot, and the driver is
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

/// One device attribute, or `None` if the driver would not say.
fn attribute(device: dr::CUdevice, which: dr::CUdevice_attribute) -> Option<i32> {
    let mut value: i32 = 0;
    // SAFETY: `value` is valid and writable; `device` came from `cuDeviceGet`.
    let code = unsafe { dr::cuDeviceGetAttribute(&raw mut value, which, device) };
    (code == dr::CUresult::CUDA_SUCCESS).then_some(value)
}
