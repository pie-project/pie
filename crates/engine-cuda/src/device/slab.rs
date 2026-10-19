//! Small device and mapped-host spans carved out of slabs, for the memory a
//! request's session opens (its channels' rings and endpoints, its registry
//! and commit slot).
//!
//! A `cudaMalloc` or `cudaHostAlloc` costs a few microseconds and a session
//! opens a couple of dozen, all on the thread that fires: a burst of 256
//! requests held the card idle ~50 ms before its prefill. A span here is a
//! power-of-two piece of a 2 MiB slab, handed out again once an event
//! recorded on each of the device's compute streams at its return has
//! landed (only fires on those streams read session memory), and zeroed on the way out as
//! fresh memory would be. Slabs are never returned to the runtime.

use std::collections::HashMap;
use std::sync::Mutex;

use super::recycle::{Fence, zero_before_fires};
use crate::device::rt;
use crate::error::Result;

/// The largest span a slab hands out.
pub const LARGEST: usize = 64 << 10;

const FLOOR: usize = 256;

const SLAB: usize = 2 << 20;

/// The span a request of `bytes` takes, if a slab serves it.
#[must_use]
pub fn class(bytes: usize) -> Option<usize> {
    (bytes > 0 && bytes <= LARGEST).then(|| bytes.max(FLOOR).next_power_of_two())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kind {
    Device,
    Host,
}

struct Free {
    /// The span: device memory, or the host side of mapped memory.
    at: u64,
    /// A mapped span's device address.
    mapped: u64,
    /// Raised at its return; open for a span never handed out.
    fence: Fence,
}

type Shelf = Option<HashMap<(Kind, i32, usize), Vec<Free>>>;

static SHELF: Mutex<Shelf> = Mutex::new(None);

fn shelf() -> std::sync::MutexGuard<'static, Shelf> {
    SHELF
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn current_device() -> Option<i32> {
    let mut device = 0i32;
    // SAFETY: writes one int.
    let code = unsafe { rt::cudaGetDevice(&raw mut device) };
    (code == rt::cudaError::cudaSuccess).then_some(device)
}

/// Whether the current device has a compute stream a returned span's event
/// can be recorded on.
#[must_use]
pub fn serves() -> bool {
    current_device().is_some_and(super::recycle::adopted)
}

fn take(kind: Kind, class: usize) -> Result<(u64, u64)> {
    let device = current_device().unwrap_or(0);
    let key = (kind, device, class);
    {
        let mut shelf = shelf();
        if let Some(list) = shelf.as_mut().and_then(|map| map.get_mut(&key))
            && let Some(at) = list.iter().position(|free| free.fence.landed())
        {
            let free = list.swap_remove(at);
            drop(shelf);
            return Ok((free.at, free.mapped));
        }
    }
    let (base, mapped) = match kind {
        Kind::Device => {
            let mut base: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: `base` is a live local the call fills.
            unsafe {
                crate::device::ctx::check("cudaMalloc", rt::cudaMalloc(&raw mut base, SLAB))
                    .map_err(|fault| super::alloc::out_of_room(fault, SLAB))?;
            }
            (base as u64, 0)
        }
        Kind::Host => {
            let mut host: *mut core::ffi::c_void = core::ptr::null_mut();
            let mut device: *mut core::ffi::c_void = core::ptr::null_mut();
            // SAFETY: live locals the calls fill; the slab is never freed.
            unsafe {
                crate::device::ctx::check(
                    "cudaHostAlloc",
                    rt::cudaHostAlloc(
                        &raw mut host,
                        SLAB,
                        rt::cudaHostAllocMapped | rt::cudaHostAllocPortable,
                    ),
                )?;
                crate::device::ctx::check(
                    "cudaHostGetDevicePointer",
                    rt::cudaHostGetDevicePointer(&raw mut device, host, 0),
                )?;
            }
            (host as u64, device as u64)
        }
    };
    let pieces = (SLAB / class) as u64;
    let step = class as u64;
    let mut shelf = shelf();
    let list = shelf.get_or_insert_with(HashMap::new).entry(key).or_default();
    list.extend((1..pieces).map(|at| Free {
        at: base + at * step,
        mapped: if mapped == 0 { 0 } else { mapped + at * step },
        fence: Fence::open(),
    }));
    Ok((base, mapped))
}

fn give(kind: Kind, class: usize, at: u64, mapped: u64) {
    let device = current_device().unwrap_or(0);
    let fence = Fence::raise(device).unwrap_or_else(|| {
        // With no stream to order on, the whole device drains first.
        // SAFETY: no arguments.
        unsafe {
            let _ = rt::cudaDeviceSynchronize();
        }
        Fence::open()
    });
    shelf()
        .get_or_insert_with(HashMap::new)
        .entry((kind, device, class))
        .or_default()
        .push(Free { at, mapped, fence });
}

/// A zeroed device span of `class` bytes, its memset ordered on the compute
/// stream ahead of every later fire.
pub fn device(class: usize) -> Result<u64> {
    let (at, _) = take(Kind::Device, class)?;
    let device = current_device().unwrap_or(0);
    if !zero_before_fires(device, at, class) {
        // SAFETY: the span is this call's and nothing airborne reads it.
        unsafe {
            crate::device::ctx::check(
                "cudaMemset",
                rt::cudaMemset(at as *mut core::ffi::c_void, 0, class),
            )?;
        }
    }
    Ok(at)
}

pub fn release_device(class: usize, at: u64) {
    give(Kind::Device, class, at, 0);
}

/// A zeroed (when `zeroed`) mapped host span of `class` bytes: its host
/// and device addresses.
pub fn host(class: usize, zeroed: bool) -> Result<(*mut u8, u64)> {
    let (at, mapped) = take(Kind::Host, class)?;
    if zeroed {
        // SAFETY: the span is this call's own and no fire reads it.
        unsafe { core::ptr::write_bytes(at as *mut u8, 0, class) };
    }
    Ok((at as *mut u8, mapped))
}

pub fn release_host(class: usize, at: *mut u8, mapped: u64) {
    give(Kind::Host, class, at as u64, mapped);
}
