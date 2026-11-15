//! What macOS says about its memory.
//!
//! The kernel states one of three pressure levels, and a dispatch source of
//! `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` fires on every change of it, either
//! way: the reading follows it with no poll. Each level reads as its rank
//! among the kernel's three: normal 0, warn 0.5, critical 1.
#![allow(unsafe_code)]

use std::ffi::c_void;
use std::mem::size_of;

use super::Reading;

const NORMAL: usize = 0x1;
const WARN: usize = 0x2;
const CRITICAL: usize = 0x4;

/// The pressure a kernel level reads as.
fn pressure(level: usize) -> f64 {
    if level & CRITICAL != 0 {
        1.0
    } else if level & WARN != 0 {
        0.5
    } else {
        0.0
    }
}

/// `kern.memorystatus_vm_pressure_level`: the level now, what the source
/// reports on its next change.
fn pressure_level() -> usize {
    let mut level: u32 = NORMAL as u32;
    let mut size = size_of::<u32>();
    // SAFETY: the name is a nul-terminated literal and `level` holds `size`
    // bytes; the kernel writes at most that.
    let status = unsafe {
        libc::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&raw mut level).cast(),
            &raw mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if status == 0 { level as usize } else { NORMAL }
}

#[repr(C)]
struct SourceType {
    _private: [u8; 0],
}

unsafe extern "C" {
    static _dispatch_source_type_memorypressure: SourceType;
    fn dispatch_get_global_queue(identifier: isize, flags: usize) -> *mut c_void;
    fn dispatch_source_create(
        kind: *const SourceType,
        handle: usize,
        mask: usize,
        queue: *mut c_void,
    ) -> *mut c_void;
    fn dispatch_set_context(object: *mut c_void, context: *mut c_void);
    fn dispatch_source_set_event_handler_f(
        source: *mut c_void,
        handler: extern "C" fn(*mut c_void),
    );
    fn dispatch_source_get_data(source: *mut c_void) -> usize;
    fn dispatch_resume(object: *mut c_void);
}

/// `QOS_CLASS_UTILITY`: the queue the source's handler runs on.
const UTILITY: isize = 0x11;

/// What the handler reaches: the source it reads and where it tells.
struct Context {
    source: *mut c_void,
    tell: tokio::sync::watch::Sender<f64>,
}

extern "C" fn changed(context: *mut c_void) {
    // SAFETY: the context is the `Context` [`watch`] leaked for the
    // source, which lives as long as the process.
    let context = unsafe { &*context.cast::<Context>() };
    // SAFETY: the source is the one this handler is set on.
    let level = unsafe { dispatch_source_get_data(context.source) };
    let now = pressure(level);
    context.tell.send_if_modified(|last| {
        let changed = *last != now;
        *last = now;
        changed
    });
}

/// Starts following the kernel's pressure level; `None` where the source
/// cannot be made.
pub fn watch() -> Option<Reading> {
    let (tell, reading) = tokio::sync::watch::channel(pressure(pressure_level()));
    // SAFETY: the global queue is the system's; the source type is the
    // kernel's static; the context is leaked, never freed, as the source
    // is never cancelled.
    unsafe {
        let queue = dispatch_get_global_queue(UTILITY, 0);
        let source = dispatch_source_create(
            &raw const _dispatch_source_type_memorypressure,
            0,
            NORMAL | WARN | CRITICAL,
            queue,
        );
        if source.is_null() {
            return None;
        }
        let context = Box::into_raw(Box::new(Context { source, tell }));
        dispatch_set_context(source, context.cast());
        dispatch_source_set_event_handler_f(source, changed);
        dispatch_resume(source);
    }
    Some(reading)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_level_reads_as_its_rank() {
        assert_eq!(pressure(NORMAL), 0.0);
        assert_eq!(pressure(WARN), 0.5);
        assert_eq!(pressure(CRITICAL), 1.0);
    }

    #[test]
    fn the_level_is_one_the_kernel_states() {
        assert!(matches!(pressure_level(), NORMAL | WARN | CRITICAL));
    }
}
