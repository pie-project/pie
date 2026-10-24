#![allow(unsafe_code)]
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::hip::sys;
use crate::jit::{Fault, Root, hiprtc};

pub struct Resolved {
    #[allow(dead_code)]
    module: sys::Module,
    pub function: sys::Function,
}

// SAFETY: `module` and `function` are process-wide HIP handles any thread may
// use, and a `Resolved` is never dropped (it lives in a leaked slot), so
// neither handle is ever unloaded under a user.
unsafe impl Send for Resolved {}

// SAFETY: shared access only copies the two handles out; HIP launches one
// function from several threads at once.
unsafe impl Sync for Resolved {}

type Slot = OnceLock<Result<Resolved, Fault>>;

fn slot(key: &str) -> &'static Slot {
    static SLOTS: OnceLock<Mutex<HashMap<String, &'static Slot>>> = OnceLock::new();
    let mut map = SLOTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("lock poisoned");
    if let Some(found) = map.get(key) {
        return found;
    }

    let fresh: &'static Slot = Box::leak(Box::new(OnceLock::new()));
    map.insert(key.to_owned(), fresh);
    fresh
}

pub(crate) fn resolve(root: &Root, instantiation: &str) -> Result<&'static Resolved, Fault> {
    let api = api("hipGetDevice")?;
    let mut ordinal: i32 = 0;
    // SAFETY: `ordinal` is a live local out-parameter.
    let code = unsafe { (api.hip.hipGetDevice)(&raw mut ordinal) };
    if code != sys::HIP_SUCCESS {
        return Err(Fault::Device {
            call: "hipGetDevice",
            code,
        });
    }
    let arch = arch(ordinal).ok_or(Fault::Device {
        call: "hipGetDeviceProperties",
        code: -1,
    })?;
    let key = format!(
        "{}/rtc{}",
        root.key(instantiation, arch),
        hiprtc::version()?
    );
    slot(&format!("hip:{ordinal}|{key}"))
        .get_or_init(|| load(root, instantiation, &key, arch))
        .as_ref()
        .map_err(Clone::clone)
}

fn load(root: &Root, instantiation: &str, key: &str, arch: &str) -> Result<Resolved, Fault> {
    let started = std::time::Instant::now();
    let (image, mangled, compiled) = match read_disk(key) {
        Some(hit) => (hit.0, hit.1, false),
        None => {
            let built = hiprtc::compile_text(&hiprtc::Job {
                name: root.name,
                source: root.text.to_owned(),
                arch,
                headers: crate::source::LIBRARY,
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
                    log: format!("`{instantiation}` compiled and hiprtc named nothing for it"),
                })?;
            write_disk(key, &built.code, &mangled);
            (built.code, mangled, true)
        }
    };

    let module = load_image(root.name, &image)?;
    let function = entry_by_name(root.name, module, instantiation, &mangled).inspect_err(|_| {
        if let Ok(api) = sys::api() {
            // SAFETY: `module` was loaded just above and this error drops its
            // only handle, so it is unloaded once and never used again.
            unsafe { (api.hip.hipModuleUnload)(module) };
        }
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

fn api(call: &'static str) -> Result<&'static sys::Api, Fault> {
    sys::api().map_err(|_| Fault::Device { call, code: -1 })
}

fn load_image(root: &'static str, image: &[u8]) -> Result<sys::Module, Fault> {
    if image.is_empty() {
        return Err(Fault::Compile {
            unit: root,
            log: "the compile produced an empty image, so there is nothing to load".into(),
        });
    }
    let api = api("hipModuleLoadData")?;
    let mut module: sys::Module = std::ptr::null_mut();

    // SAFETY: `image` is a non-empty code object (hiprtc's output, or the bytes
    // the disk cache stored under this exact key, their hash checked by
    // `decode`) alive across the call; `module` is a live out-parameter.
    let code = unsafe { (api.hip.hipModuleLoadData)(&raw mut module, image.as_ptr().cast()) };
    if code == sys::HIP_SUCCESS {
        Ok(module)
    } else {
        Err(Fault::Device {
            call: "hipModuleLoadData",
            code,
        })
    }
}

/// `hipErrorNotFound`.
const NOT_FOUND: sys::HipError = 500;

fn entry_by_name(
    root: &'static str,
    module: sys::Module,
    instantiation: &str,
    mangled: &str,
) -> Result<sys::Function, Fault> {
    let Ok(c_name) = std::ffi::CString::new(mangled) else {
        return Err(Fault::Compile {
            unit: root,
            log: format!("the lowered name for `{instantiation}` contains a NUL"),
        });
    };
    let api = api("hipModuleGetFunction")?;
    let mut function: sys::Function = std::ptr::null_mut();

    // SAFETY: `module` is loaded, `c_name` NUL-terminated and alive, and
    // `function` a live out-parameter.
    let code =
        unsafe { (api.hip.hipModuleGetFunction)(&raw mut function, module, c_name.as_ptr()) };
    match code {
        sys::HIP_SUCCESS => spill_free(root, instantiation, function),
        NOT_FOUND => Err(Fault::Compile {
            unit: root,
            log: format!("`{instantiation}` compiled and is not in the image"),
        }),
        other => Err(Fault::Device {
            call: "hipModuleGetFunction",
            code: other,
        }),
    }
}

/// The roots whose kernels count their own reads (hand-issued loads the
/// compiler does not see): a spill there could store a register before its
/// read lands, or wait on the wrong count, so such a kernel that spills is
/// refused rather than run.
const HAND_COUNTED: &[&str] = &["linear/fp8"];

/// `hipFuncAttributeLocalSizeBytes`: a lane's scratch.
const LOCAL_SIZE_BYTES: i32 = 3;

fn spill_free(
    root: &'static str,
    instantiation: &str,
    function: sys::Function,
) -> Result<sys::Function, Fault> {
    if !HAND_COUNTED.contains(&root) {
        return Ok(function);
    }
    let mut local: i32 = 0;
    // SAFETY: a live out-parameter and the function this module just named.
    let code = unsafe {
        crate::hip::rt::hipFuncGetAttribute(&raw mut local, LOCAL_SIZE_BYTES, function.cast())
    };
    if code == sys::HIP_SUCCESS && local > 0 {
        return Err(Fault::Compile {
            unit: root,
            log: format!(
                "`{instantiation}` spills {local} bytes a lane; its hand-counted reads cannot share registers \
                 with spills"
            ),
        });
    }
    Ok(function)
}

fn disk_path(key: &str) -> Option<PathBuf> {
    let digest = crate::source::fnv1a64(key.as_bytes());
    Some(crate::disk::dir(crate::disk::CODE_OBJECTS)?.join(format!("{digest:016x}.hsaco")))
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
    let (mangled, image) = take_str(rest)?;
    Some((image.to_vec(), mangled.to_owned()))
}

fn take_str(bytes: &[u8]) -> Option<(&str, &[u8])> {
    let (len, rest) = bytes.split_at_checked(4)?;
    let len = u32::from_le_bytes(len.try_into().ok()?) as usize;
    let (text, rest) = rest.split_at_checked(len)?;
    Some((std::str::from_utf8(text).ok()?, rest))
}

fn write_disk(key: &str, image: &[u8], mangled: &str) {
    let Some(path) = disk_path(key) else { return };
    let Some(parent) = path.parent() else { return };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let out = encode(key, image, mangled);
    let staging = path.with_extension(format!("hsaco.{}", std::process::id()));
    if std::fs::write(&staging, &out).is_ok() && std::fs::rename(&staging, &path).is_err() {
        let _ = std::fs::remove_file(&staging);
    }
}

/// `key`, `mangled` and `image`, then the FNV-1a hash of all three.
fn encode(key: &str, image: &[u8], mangled: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(image.len() + mangled.len() + key.len() + 16);
    put_str(&mut out, key);
    put_str(&mut out, mangled);
    out.extend_from_slice(image);
    let sum = crate::source::fnv1a64(&out);
    out.extend_from_slice(&sum.to_le_bytes());
    out
}

fn put_str(out: &mut Vec<u8>, text: &str) {
    let len = u32::try_from(text.len()).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&text.as_bytes()[..len as usize]);
}

/// The device's architecture, read once per ordinal: `gfx942` on MI300.
#[must_use]
pub fn arch(ordinal: i32) -> Option<&'static str> {
    static ARCHES: OnceLock<Mutex<HashMap<i32, Option<&'static str>>>> = OnceLock::new();
    let mut map = ARCHES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("lock poisoned");
    *map.entry(ordinal).or_insert_with(|| {
        let name = crate::jit::device::arch_name(ordinal)?;
        Some(Box::leak(name.into_boxed_str()))
    })
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
