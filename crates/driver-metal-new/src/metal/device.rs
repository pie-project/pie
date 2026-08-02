//! What device this is: the Metal half of the tuning query.
//!
//! Two facts, from two places, because neither publishes the other:
//! `MTLDevice` knows the GPU family and IOKit knows the core count. The
//! decision made from them is in [`crate::tuning`], which is portable; this
//! module only asks.

use std::ffi::CStr;
use std::sync::OnceLock;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::{CFDictionary, CFNumber, CFRetained, CFString, kCFAllocatorDefault};
use objc2_io_kit::{
    IOIteratorNext, IOObjectRelease, IORegistryEntryCreateCFProperty, IOServiceGetMatchingServices,
    IOServiceMatching, io_iterator_t, kIOMainPortDefault,
};
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily};

use crate::tuning::Device;

/// The device facts tuning is selected from.
///
/// Both fields are 0 when unknown, and every consumer must tolerate that
/// rather than branch on it -- 0 selects the shipped defaults, which is what
/// every device got before the tuning layer existed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceInfo {
    /// `MTLGPUFamilyApple<N>`, resolved newest-first. 0 if no device answered.
    pub apple_family: u32,
    /// IOKit's `gpu-core-count`. 0 when the driver does not publish it.
    pub gpu_core_count: u32,
}

impl From<DeviceInfo> for Device {
    fn from(i: DeviceInfo) -> Self {
        Self {
            apple_family: i.apple_family,
            gpu_core_count: i.gpu_core_count,
        }
    }
}

impl DeviceInfo {
    /// Ask the machine, once per process.
    ///
    /// Cached because neither answer can change while the process runs and
    /// the IOKit walk is not free. The C++ shell caches it in a function-local
    /// `static`, which is the same decision.
    #[must_use]
    pub fn get() -> Self {
        static INFO: OnceLock<DeviceInfo> = OnceLock::new();
        *INFO.get_or_init(|| Self {
            apple_family: query_apple_family(),
            gpu_core_count: query_gpu_core_count(),
        })
    }
}

/// The system default Metal device, or `None` if the process cannot reach one.
pub(crate) fn default_device() -> Option<Retained<ProtocolObject<dyn MTLDevice>>> {
    MTLCreateSystemDefaultDevice()
}

/// The families this shell knows, NEWEST FIRST.
///
/// The order is the whole of the correctness argument: the Apple families are
/// cumulative, so an M4 answers yes to Apple7 as well as to Apple9. An
/// ascending probe would report every Apple silicon GPU ever made as an
/// Apple7 and hand all of them the M1 constants -- a bug that looks exactly
/// like the tuning layer not existing.
///
/// A new family goes at the FRONT, and the test below fails if it does not.
const FAMILIES: [(MTLGPUFamily, u32); 5] = [
    (MTLGPUFamily::Apple9, 9),
    (MTLGPUFamily::Apple8, 8),
    (MTLGPUFamily::Apple7, 7),
    (MTLGPUFamily::Apple6, 6),
    (MTLGPUFamily::Apple5, 5),
];

/// The newest `MTLGPUFamilyApple<N>` this device answers to, or 0.
fn query_apple_family() -> u32 {
    let Some(device) = default_device() else {
        return 0;
    };
    FAMILIES
        .into_iter()
        .find(|&(family, _)| device.supportsFamily(family))
        .map_or(0, |(_, n)| n)
}

/// The IOKit property carrying the count.
const GPU_CORE_COUNT_KEY: &str = "gpu-core-count";
/// The service publishing it. NUL-terminated: it is handed to a C API.
const ACCELERATOR_SERVICE: &CStr = c"AGXAccelerator";

/// IOKit's `gpu-core-count` from the `AGXAccelerator` service, or 0.
///
/// `MTLDevice` does not expose the core count at all, and this is the only
/// place it is published. Absent on a machine whose driver does not publish
/// it, which is why the caller must tolerate 0 rather than branch on a count
/// it assumes it has.
fn query_gpu_core_count() -> u32 {
    // SAFETY: `ACCELERATOR_SERVICE` is a valid NUL-terminated C string with
    // static lifetime; `IOServiceMatching` only reads it, and returns a fresh
    // dictionary owning a +1 reference.
    let Some(matching) = (unsafe { IOServiceMatching(ACCELERATOR_SERVICE.as_ptr()) }) else {
        return 0;
    };
    // The lookup CONSUMES a reference, which is why the binding takes the
    // dictionary by value rather than by reference. `IOServiceMatching` hands
    // back a mutable dictionary and the lookup wants an immutable one.
    //
    // SAFETY: this is the CF class hierarchy, not a reinterpretation --
    // `CFMutableDictionary` declares `CFDictionary` as its superclass, so
    // every mutable dictionary IS a dictionary and the pointer is unchanged.
    // There is no safe `Into` for the widening because CF's superclass
    // relation is expressed through `Deref` on the borrowed form only.
    let matching: CFRetained<CFDictionary> = unsafe { CFRetained::cast_unchecked(matching) };

    let mut iter: io_iterator_t = 0;
    // SAFETY: `iter` is a live, correctly typed out-parameter, and `matching`
    // holds exactly the reference this call consumes.
    let kr =
        unsafe { IOServiceGetMatchingServices(kIOMainPortDefault, Some(matching), &raw mut iter) };
    if kr != 0 || iter == 0 {
        return 0;
    }

    let key = CFString::from_str(GPU_CORE_COUNT_KEY);
    let mut cores = 0u32;
    loop {
        let service = IOIteratorNext(iter);
        if service == 0 {
            break;
        }
        // SAFETY: `service` is a live registry entry handed over by the
        // iterator and `key` outlives the call. The property comes back +1;
        // `CFRetained` releases it when `property` drops at the end of the
        // iteration.
        let property =
            unsafe { IORegistryEntryCreateCFProperty(service, Some(&key), kCFAllocatorDefault, 0) };
        if let Some(property) = property
            && let Some(number) = property.downcast_ref::<CFNumber>()
            && let Some(n) = number.as_i32()
            && n > 0
        {
            cores = u32::try_from(n).unwrap_or(0);
        }
        IOObjectRelease(service);
        if cores != 0 {
            break;
        }
    }
    IOObjectRelease(iter);
    cores
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts the ORDER of the family table rather than the result of a
    /// query, so it runs on a machine with no GPU and still catches the one
    /// mistake this table can make.
    #[test]
    fn the_family_table_descends() {
        let ns: Vec<u32> = FAMILIES.iter().map(|&(_, n)| n).collect();
        assert!(
            ns.windows(2).all(|w| w[0] > w[1]),
            "cumulative families must be probed newest-first, got {ns:?}"
        );
    }

    #[test]
    fn unknown_device_info_maps_to_the_default_tuning_device() {
        assert_eq!(Device::from(DeviceInfo::default()), Device::default());
    }

    /// The query runs, and answers with something a real machine could have.
    ///
    /// Asserts a RANGE rather than this box's numbers, because the crate is
    /// built on more than one Mac and pinning `(7, 24)` would fail on the
    /// next one. What it is really guarding is that the two calls return at
    /// all: both cross into C with a hand-written ownership contract, and the
    /// IOKit walk in particular consumes a reference the compiler cannot
    /// check. A double release there is a crash, not a wrong number, so a
    /// test that merely REACHES the assert has already proven the thing worth
    /// proving.
    ///
    /// Skipped when no device answers -- a headless CI runner has no GPU, and
    /// 0 is the documented answer there rather than a failure.
    #[test]
    fn the_device_query_answers() {
        let info = DeviceInfo::get();
        if info.apple_family == 0 {
            return;
        }
        assert!(
            (5..=99).contains(&info.apple_family),
            "implausible family {}",
            info.apple_family
        );
        assert!(
            info.gpu_core_count <= 1024,
            "implausible core count {}",
            info.gpu_core_count
        );
        // Cached: the second call must be the first one's answer, not a
        // second walk of the registry.
        assert_eq!(DeviceInfo::get(), info);
    }
}
