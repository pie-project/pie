//! The module cache: each `(root, instantiation, arch)` compiles at most
//! once per process, lands on disk keyed by everything that can change the
//! cubin, and resolves to a loaded `CUfunction` for every fire after the
//! first.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use cudarc::driver::sys as dr;
use cudarc::runtime::sys as rt;

use crate::jit::{Fault, Root, nvrtc};

pub struct Resolved {
    #[allow(dead_code)]
    module: dr::CUmodule,
    pub function: dr::CUfunction,
}

unsafe impl Send for Resolved {}

unsafe impl Sync for Resolved {}

type Slot = OnceLock<Result<Resolved, Fault>>;

fn slot(key: &str) -> &'static Slot {
    static SLOTS: OnceLock<Mutex<HashMap<String, &'static Slot>>> = OnceLock::new();
    let mut map = SLOTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(found) = map.get(key) {
        return found;
    }

    let fresh: &'static Slot = Box::leak(Box::new(OnceLock::new()));
    map.insert(key.to_owned(), fresh);
    fresh
}

pub(crate) fn resolve(root: &Root, instantiation: &str) -> Result<&'static Resolved, Fault> {
    bind_context()?;
    let arch = arch().ok_or(Fault::Device {
        call: "cuDeviceGet",
        code: -1,
    })?;
    let key = root.key(instantiation, arch);

    slot(&key)
        .get_or_init(|| load(root, instantiation, &key, arch))
        .as_ref()
        .map_err(Clone::clone)
}

fn load(root: &Root, instantiation: &str, key: &str, arch: &str) -> Result<Resolved, Fault> {
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

fn load_image(root: &'static str, image: &[u8]) -> Result<dr::CUmodule, Fault> {
    if image.is_empty() {
        return Err(Fault::Compile {
            unit: root,
            log: "the compile produced an empty image, so there is nothing to load".into(),
        });
    }
    let mut module: dr::CUmodule = std::ptr::null_mut();

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

/// This instantiation's cubin, under the deployment's stated cache root. The
/// root arrives through [`crate::disk`]; `None` is a process that stated
/// none, and every instantiation then compiles.
fn disk_path(key: &str) -> Option<PathBuf> {
    let digest = crate::source::fnv1a64(key.as_bytes());
    Some(crate::disk::dir(crate::disk::CUBINS)?.join(format!("{digest:016x}.cubin")))
}

fn read_disk(key: &str) -> Option<(Vec<u8>, String)> {
    let bytes = std::fs::read(disk_path(key)?).ok()?;
    let (stored_key, rest) = take_str(&bytes)?;
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
    let mut out = Vec::with_capacity(cubin.len() + mangled.len() + key.len() + 8);
    put_str(&mut out, key);
    put_str(&mut out, mangled);
    out.extend_from_slice(cubin);

    let staging = path.with_extension(format!("cubin.{}", std::process::id()));
    if std::fs::write(&staging, &out).is_ok() && std::fs::rename(&staging, &path).is_err() {
        let _ = std::fs::remove_file(&staging);
    }
}

fn put_str(out: &mut Vec<u8>, text: &str) {
    let len = u32::try_from(text.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&text.as_bytes()[..len as usize]);
}

/// Ensure this thread holds the primary context before any driver-API call.
pub(crate) fn bind_context() -> Result<(), Fault> {
    use std::cell::Cell;

    thread_local! {
        static BOUND: Cell<bool> = const { Cell::new(false) };
    }

    if BOUND.with(Cell::get) {
        return Ok(());
    }

    let code = unsafe { rt::cudaFree(std::ptr::null_mut()) };
    if code != rt::cudaError::cudaSuccess {
        return Err(Fault::Device {
            call: "cudaFree",
            code: code as i32,
        });
    }
    BOUND.with(|bound| bound.set(true));
    Ok(())
}

/// The current device's `sm_XY`, probed once.
#[must_use]
pub fn arch() -> Option<&'static str> {
    use dr::CUdevice_attribute as Attr;

    static ARCH: OnceLock<Option<String>> = OnceLock::new();
    ARCH.get_or_init(|| {
        cudarc::driver::result::init().ok()?;
        let mut ordinal: i32 = 0;

        let code = unsafe { rt::cudaGetDevice(&raw mut ordinal) };
        if code != rt::cudaError::cudaSuccess {
            return None;
        }
        let mut device: dr::CUdevice = 0;

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

fn attribute(device: dr::CUdevice, which: dr::CUdevice_attribute) -> Option<i32> {
    let mut value: i32 = 0;

    let code = unsafe { dr::cuDeviceGetAttribute(&raw mut value, which, device) };
    (code == dr::CUresult::CUDA_SUCCESS).then_some(value)
}
