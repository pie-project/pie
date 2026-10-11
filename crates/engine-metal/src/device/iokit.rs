//! The GPU core count, read from the IORegistry. Metal does not publish it;
//! the AGX accelerator's `gpu-core-count` property does, on every Apple
//! Silicon GPU.

use std::ffi::{c_char, c_void};

type CfTypeRef = *const c_void;
type IoObject = u32;

const UTF8: u32 = 0x0800_0100;
const SINT32: isize = 3;

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOServiceMatching(name: *const c_char) -> *mut c_void;
    fn IOServiceGetMatchingServices(
        main: u32,
        matching: *mut c_void,
        existing: *mut IoObject,
    ) -> i32;
    fn IOIteratorNext(iterator: IoObject) -> IoObject;
    fn IORegistryEntryCreateCFProperty(
        entry: IoObject,
        key: CfTypeRef,
        allocator: CfTypeRef,
        options: u32,
    ) -> CfTypeRef;
    fn IOObjectRelease(object: IoObject) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(
        allocator: CfTypeRef,
        s: *const c_char,
        encoding: u32,
    ) -> CfTypeRef;
    fn CFNumberGetTypeID() -> usize;
    fn CFGetTypeID(cf: CfTypeRef) -> usize;
    fn CFNumberGetValue(number: CfTypeRef, kind: isize, out: *mut c_void) -> bool;
    fn CFRelease(cf: CfTypeRef);
}

/// The GPU core count of the first AGX accelerator, or `None` when the
/// registry does not say.
pub(crate) fn gpu_cores() -> Option<u32> {
    // SAFETY: each IOKit and CoreFoundation call gets the argument types its
    // header declares; every object created here is released here, and
    // `IOServiceGetMatchingServices` consumes the matching dictionary.
    unsafe {
        let matching = IOServiceMatching(c"AGXAccelerator".as_ptr());
        if matching.is_null() {
            return None;
        }
        let mut iterator: IoObject = 0;
        if IOServiceGetMatchingServices(0, matching, &raw mut iterator) != 0 {
            return None;
        }
        let key = CFStringCreateWithCString(std::ptr::null(), c"gpu-core-count".as_ptr(), UTF8);
        let mut found = None;
        loop {
            let service = IOIteratorNext(iterator);
            if service == 0 {
                break;
            }
            let value = IORegistryEntryCreateCFProperty(service, key, std::ptr::null(), 0);
            IOObjectRelease(service);
            if value.is_null() {
                continue;
            }
            let mut cores: i32 = 0;
            let read = CFGetTypeID(value) == CFNumberGetTypeID()
                && CFNumberGetValue(value, SINT32, (&raw mut cores).cast());
            CFRelease(value);
            if read && cores > 0 {
                found = u32::try_from(cores).ok();
                break;
            }
        }
        if !key.is_null() {
            CFRelease(key);
        }
        IOObjectRelease(iterator);
        found
    }
}
