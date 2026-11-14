#![allow(unsafe_code)]
use core::ffi::c_void;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::hip::sys;
use crate::jit::Fault;

struct Slab {
    ptr: *mut c_void,
    bytes: usize,
}

/// One arena's slabs, each laid in memory its caller holds ([`place`]):
/// nothing here allocates or frees device memory.
#[derive(Default)]
struct Arena {
    placed: HashMap<(&'static str, u32, usize), Slab>,
}

use super::ctx::{PLACED_ALIGN, SlabKey};

/// [`super::Slabs::place`].
pub(crate) fn place(base: *mut c_void, slabs: &[(SlabKey, usize)]) {
    let mut arenas = locked();
    let mut at = 0usize;
    for (key, bytes) in slabs {
        let bytes = bytes.next_multiple_of(PLACED_ALIGN);
        arenas.entry(key.arena).or_default().placed.insert(
            (key.name, key.region, key.scope),
            Slab {
                // SAFETY: inside the span the caller holds, by `placed_bytes`.
                ptr: unsafe { base.byte_add(at) },
                bytes,
            },
        );
        at += bytes;
    }
}

/// [`super::Slabs::unplaced_bytes`].
pub(crate) fn unplaced_bytes() -> usize {
    UNPLACED.load(core::sync::atomic::Ordering::Relaxed)
}

static UNPLACED: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

// SAFETY: the pointers here are spans of memory the placing caller holds;
// the map is behind a mutex and never dereferences them.
unsafe impl Send for Arena {}

fn arenas() -> &'static Mutex<HashMap<u32, Arena>> {
    static ARENAS: OnceLock<Mutex<HashMap<u32, Arena>>> = OnceLock::new();
    ARENAS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn locked() -> std::sync::MutexGuard<'static, HashMap<u32, Arena>> {
    arenas().lock().expect("lock poisoned")
}

/// A size-only walk's slabs (`crate::sizing`): the most each key asked.
#[derive(Default)]
struct Sized {
    asked: HashMap<(u32, &'static str, u32, usize), usize>,
}

thread_local! {
    static SIZED: core::cell::RefCell<Sized> = core::cell::RefCell::new(Sized::default());
}

pub(crate) fn sizing_reset() {
    SIZED.with(|sized| *sized.borrow_mut() = Sized::default());
}

pub(crate) fn sizing_take() -> crate::sizing::Footprint {
    SIZED.with(|sized| {
        let sized = core::mem::take(&mut *sized.borrow_mut());
        crate::sizing::Footprint::of(
            sized
                .asked
                .into_iter()
                .map(|((arena, name, region, scope), asked)| crate::sizing::Sized {
                    key: SlabKey {
                        arena,
                        name,
                        region,
                        scope,
                    },
                    asked,
                })
                .collect(),
        )
    })
}

/// [`take`] in a size-only walk: the ask counted, and a stand-in address.
fn take_sized(
    arena: u32,
    owner: *mut c_void,
    name: &'static str,
    region: u32,
    bytes: usize,
) -> *mut c_void {
    let key = (arena, name, region, scope_of(owner, region));
    SIZED.with(|sized| {
        let mut sized = sized.borrow_mut();
        let asked = sized.asked.entry(key).or_insert(0);
        *asked = (*asked).max(bytes);
    });
    crate::sizing::STAND_IN as *mut c_void
}

pub(crate) fn attach(arena: u32, stream: *mut c_void) {
    let _ = (arena, stream);
}

/// Forgets the arena's slabs; their memory is the placing caller's.
pub(crate) fn release(arena: u32) {
    locked().remove(&arena);
}

fn scope_of(stream: *mut c_void, region: u32) -> usize {
    if region == super::ctx::NO_REGION {
        stream.addr()
    } else {
        0
    }
}

/// The slab laid for `(name, region)` in `arena`, at least `bytes` long.
/// `owner` is the stream a stream-scoped slab belongs to: the launching
/// stream, or the one whose order it launches in. A slab is laid zeroed
/// ([`super::Slabs::place`]), so counters a launch puts back to zero
/// (tickets) read zero at every take. A take past what is laid is a fire
/// no walk sized, and faults.
pub(crate) fn take(
    arena: u32,
    owner: *mut c_void,
    name: &'static str,
    region: u32,
    bytes: usize,
) -> Result<*mut c_void, Fault> {
    if bytes == 0 {
        return Ok(core::ptr::null_mut());
    }
    if crate::sizing::active() {
        return Ok(take_sized(arena, owner, name, region, bytes));
    }
    let arenas = locked();
    let have = arenas
        .get(&arena)
        .and_then(|held| held.placed.get(&(name, region, scope_of(owner, region))));
    match have {
        Some(slab) if slab.bytes >= bytes => Ok(slab.ptr),
        _ => {
            UNPLACED.fetch_add(bytes, core::sync::atomic::Ordering::Relaxed);
            Err(Fault::Unplaced {
                name,
                have: have.map_or(0, |slab| slab.bytes),
                need: bytes,
            })
        }
    }
}

fn current() -> Option<i32> {
    let api = sys::api().ok()?;
    let mut ordinal: i32 = 0;
    // SAFETY: `ordinal` is a live local out-parameter.
    (unsafe { (api.hip.hipGetDevice)(&raw mut ordinal) } == sys::HIP_SUCCESS).then_some(ordinal)
}

/// A device fact read once per device: a process may drive several GPUs,
/// and the first one asked does not answer for the others.
fn per_device(
    cache: &'static OnceLock<Mutex<HashMap<i32, Option<u32>>>>,
    read: impl FnOnce(i32) -> Option<u32>,
) -> Option<u32> {
    let ordinal = current()?;
    let mut map = cache
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("lock poisoned");
    *map.entry(ordinal).or_insert_with(|| read(ordinal))
}

#[must_use]
pub(crate) fn multiprocessors() -> Option<u32> {
    static COUNT: OnceLock<Mutex<HashMap<i32, Option<u32>>>> = OnceLock::new();
    per_device(&COUNT, |o| attribute(sys::ATTR_MULTIPROCESSOR_COUNT, o))
}

/// The gfx major: 9 on CDNA (gfx90a, gfx942, gfx950), 11/12 on RDNA.
#[must_use]
pub(crate) fn compute_capability_major() -> Option<u32> {
    static MAJOR: OnceLock<Mutex<HashMap<i32, Option<u32>>>> = OnceLock::new();
    per_device(&MAJOR, |o| attribute(sys::ATTR_COMPUTE_CAPABILITY_MAJOR, o))
}

#[must_use]
pub(crate) fn max_shared_memory_per_sm() -> Option<u32> {
    static BYTES: OnceLock<Mutex<HashMap<i32, Option<u32>>>> = OnceLock::new();
    per_device(&BYTES, |o| {
        attribute(sys::ATTR_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, o)
    })
}

#[must_use]
pub(crate) fn max_shared_memory_per_block_optin() -> Option<u32> {
    static BYTES: OnceLock<Mutex<HashMap<i32, Option<u32>>>> = OnceLock::new();
    per_device(&BYTES, |o| {
        attribute(sys::ATTR_SHARED_MEM_PER_BLOCK_OPTIN, o)
    })
}

/// The device's `gcnArchName` with its target features stripped: `gfx942`
/// for `gfx942:sramecc+:xnack-`.
pub(crate) fn arch_name(ordinal: i32) -> Option<String> {
    let api = sys::api().ok()?;
    let mut prop = vec![0u8; sys::DEVICE_PROP_BYTES];
    // SAFETY: `prop` is a zeroed buffer the size of `hipDeviceProp_tR0600`.
    let code = unsafe { (api.hip.hipGetDevicePropertiesR0600)(prop.as_mut_ptr().cast(), ordinal) };
    if code != sys::HIP_SUCCESS {
        return None;
    }
    let field = &prop[sys::DEVICE_PROP_GCN_ARCH_NAME..sys::DEVICE_PROP_GCN_ARCH_NAME + 256];
    let full = std::ffi::CStr::from_bytes_until_nul(field)
        .ok()?
        .to_str()
        .ok()?;
    let base = full.split(':').next()?;
    base.starts_with("gfx").then(|| base.to_owned())
}

fn attribute(which: i32, ordinal: i32) -> Option<u32> {
    let api = sys::api().ok()?;
    let mut value: i32 = 0;

    // SAFETY: `value` is a live local out-parameter; HIP range-checks `which`
    // and `ordinal`.
    let code = unsafe { (api.hip.hipDeviceGetAttribute)(&raw mut value, which, ordinal) };
    (code == sys::HIP_SUCCESS && value > 0).then(|| value.unsigned_abs())
}
