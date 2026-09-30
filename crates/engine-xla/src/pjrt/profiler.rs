//! The plugin's profiler (`PJRT_Profiler_Extension`, the PLUGIN_Profiler C
//! API of `xla/backends/profiler/plugin/profiler_c_api.h`): a device trace
//! of this process's executions, collected as a serialized `XSpace` (what
//! `jax.profiler` writes as `*.xplane.pb`). The structs are declared here by
//! hand with the upstream layouts; the vendored header does not carry them.

use std::os::raw::c_char;
use std::sync::Arc;

use super::{Api, Error, sys};

#[repr(C)]
struct PluginError {
    _p: [u8; 0],
}

#[repr(C)]
struct PluginProfiler {
    _p: [u8; 0],
}

#[repr(C)]
struct ErrorDestroyArgs {
    struct_size: usize,
    priv_: *mut std::ffi::c_void,
    error: *mut PluginError,
}

#[repr(C)]
struct ErrorMessageArgs {
    struct_size: usize,
    priv_: *mut std::ffi::c_void,
    error: *const PluginError,
    message: *const c_char,
    message_size: usize,
}

#[repr(C)]
struct CreateArgs {
    struct_size: usize,
    options: *const c_char,
    options_size: usize,
    profiler: *mut PluginProfiler,
}

/// Destroy, Start and Stop share this shape.
#[repr(C)]
struct ProfilerArgs {
    struct_size: usize,
    profiler: *mut PluginProfiler,
}

#[repr(C)]
struct CollectArgs {
    struct_size: usize,
    profiler: *mut PluginProfiler,
    buffer: *mut u8,
    buffer_size_in_bytes: usize,
}

type Call<A> = unsafe extern "C" fn(*mut A) -> *mut PluginError;

#[repr(C)]
struct ProfilerApi {
    struct_size: usize,
    priv_: *mut std::ffi::c_void,
    error_destroy: unsafe extern "C" fn(*mut ErrorDestroyArgs),
    error_message: unsafe extern "C" fn(*mut ErrorMessageArgs),
    error_get_code: *const std::ffi::c_void,
    create: Call<CreateArgs>,
    destroy: Call<ProfilerArgs>,
    start: Call<ProfilerArgs>,
    stop: Call<ProfilerArgs>,
    collect_data: Call<CollectArgs>,
}

#[repr(C)]
struct ProfilerExtension {
    base: sys::PJRT_Extension_Base,
    profiler_api: *const ProfilerApi,
}

/// One running device trace.
pub struct Profiler {
    api: Arc<Api>,
    table: *const ProfilerApi,
    raw: *mut PluginProfiler,
}

// SAFETY: the plugin's profiler is driven from one thread at a time here.
unsafe impl Send for Profiler {}

impl Profiler {
    /// Starts a trace with host and device tracing on (tensorflow
    /// `ProfileOptions`: host level 2, device level 1, version 1, HLO protos).
    pub fn start(api: &Arc<Api>) -> Result<Profiler, Error> {
        let table =
            find(api).ok_or_else(|| Error::local("the plugin has no profiler extension"))?;
        const OPTIONS: [u8; 8] = [0x10, 0x02, 0x18, 0x01, 0x28, 0x01, 0x38, 0x01];
        let mut create = CreateArgs {
            struct_size: std::mem::size_of::<CreateArgs>(),
            options: OPTIONS.as_ptr().cast(),
            options_size: OPTIONS.len(),
            profiler: std::ptr::null_mut(),
        };
        // SAFETY: `table` is the plugin's profiler vtable; args are well formed.
        unsafe { check(table, ((*table).create)(&mut create))? };
        let profiler = Profiler {
            api: Arc::clone(api),
            table,
            raw: create.profiler,
        };
        let mut args = profiler.args();
        // SAFETY: as above.
        unsafe { check(table, ((*table).start)(&mut args))? };
        Ok(profiler)
    }

    /// Stops the trace and returns it as a serialized `XSpace`.
    pub fn finish(self) -> Result<Vec<u8>, Error> {
        let mut args = self.args();
        // SAFETY: the profiler was started by `start`.
        unsafe { check(self.table, ((*self.table).stop)(&mut args))? };
        let mut collect = CollectArgs {
            struct_size: std::mem::size_of::<CollectArgs>(),
            profiler: self.raw,
            buffer: std::ptr::null_mut(),
            buffer_size_in_bytes: 0,
        };
        // SAFETY: a null buffer asks the plugin for its own copy and size.
        unsafe { check(self.table, ((*self.table).collect_data)(&mut collect))? };
        let mut bytes = vec![0u8; collect.buffer_size_in_bytes];
        if collect.buffer.is_null() {
            collect.buffer = bytes.as_mut_ptr();
            // SAFETY: a buffer of the size the plugin asked for.
            unsafe { check(self.table, ((*self.table).collect_data)(&mut collect))? };
        } else {
            // SAFETY: the plugin holds `buffer_size_in_bytes` bytes there.
            bytes.copy_from_slice(unsafe {
                std::slice::from_raw_parts(collect.buffer, collect.buffer_size_in_bytes)
            });
        }
        // The plugin counts a terminating NUL in the size; a serialized
        // proto3 message never ends in a lone zero byte.
        if bytes.last() == Some(&0) {
            bytes.pop();
        }
        Ok(bytes)
    }

    fn args(&self) -> ProfilerArgs {
        ProfilerArgs {
            struct_size: std::mem::size_of::<ProfilerArgs>(),
            profiler: self.raw,
        }
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        let mut args = self.args();
        // SAFETY: destroys the profiler `start` created, once.
        let _ = unsafe { check(self.table, ((*self.table).destroy)(&mut args)) };
        let _ = &self.api;
    }
}

fn find(api: &Api) -> Option<*const ProfilerApi> {
    // SAFETY: the extension list is the plugin's, immutable for its life.
    let mut at = unsafe { (*api.raw).extension_start };
    while !at.is_null() {
        let base = unsafe { &*at };
        if base.type_ == sys::PJRT_Extension_Type_Profiler {
            let ext = at.cast::<ProfilerExtension>();
            let table = unsafe { (*ext).profiler_api };
            return (!table.is_null()).then_some(table);
        }
        at = base.next;
    }
    None
}

unsafe fn check(table: *const ProfilerApi, error: *mut PluginError) -> Result<(), Error> {
    if error.is_null() {
        return Ok(());
    }
    let mut message = ErrorMessageArgs {
        struct_size: std::mem::size_of::<ErrorMessageArgs>(),
        priv_: std::ptr::null_mut(),
        error,
        message: std::ptr::null(),
        message_size: 0,
    };
    let text = unsafe {
        ((*table).error_message)(&mut message);
        super::text(message.message, message.message_size)
    };
    let mut destroy = ErrorDestroyArgs {
        struct_size: std::mem::size_of::<ErrorDestroyArgs>(),
        priv_: std::ptr::null_mut(),
        error,
    };
    unsafe { ((*table).error_destroy)(&mut destroy) };
    Err(Error::local(format!("profiler: {text}")))
}
