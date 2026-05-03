//! Driver-flavor FFI: feature-gated extern declarations.
//!
//! The build.rs picks ONE of the two static libs (portable or cuda)
//! based on the active Cargo feature; the matching `inner` module
//! below declares only the symbols that lib actually exports. The
//! rest of the codebase consumes [`run`], [`request_stop`], [`FLAVOR`],
//! and [`ARGV0`] without knowing which flavor is linked.

#[cfg(feature = "driver-portable")]
mod inner {
    use std::os::raw::{c_char, c_int, c_void};

    pub type ReadyCb = unsafe extern "C" fn(caps_json: *const c_char, ctx: *mut c_void);

    unsafe extern "C" {
        pub fn pie_driver_portable_run(
            argc: c_int,
            argv: *mut *mut c_char,
            install_signal_handlers: c_int,
            ready_cb: ReadyCb,
            ready_ctx: *mut c_void,
        ) -> c_int;
        pub fn pie_driver_portable_request_stop();
    }

    pub const FLAVOR: &str = "portable";
    pub const ARGV0: &str = "pie_driver_portable";

    pub use pie_driver_portable_request_stop as request_stop;
    pub use pie_driver_portable_run as run;
}

#[cfg(feature = "driver-cuda")]
mod inner {
    use std::os::raw::{c_char, c_int, c_void};

    pub type ReadyCb = unsafe extern "C" fn(caps_json: *const c_char, ctx: *mut c_void);

    unsafe extern "C" {
        pub fn pie_driver_cuda_run(
            argc: c_int,
            argv: *mut *mut c_char,
            install_signal_handlers: c_int,
            ready_cb: ReadyCb,
            ready_ctx: *mut c_void,
        ) -> c_int;
        pub fn pie_driver_cuda_request_stop();
    }

    pub const FLAVOR: &str = "cuda";
    pub const ARGV0: &str = "pie_driver_cuda";

    pub use pie_driver_cuda_request_stop as request_stop;
    pub use pie_driver_cuda_run as run;
}

#[cfg(feature = "driver-dummy")]
mod inner {
    // The dummy driver is a Rust crate (rlib) in this workspace. We
    // call its `extern "C"` entries directly — no `unsafe extern { ... }`
    // declaration needed, which sidesteps the link-time symbol GC that
    // strips unreferenced `#[no_mangle]` exports out of an rlib.
    pub use pie_driver_dummy_lib::pie_driver_dummy_request_stop as request_stop;
    pub use pie_driver_dummy_lib::pie_driver_dummy_run as run;

    pub const FLAVOR: &str = "dummy";
    pub const ARGV0: &str = "pie_driver_dummy";
}

pub use inner::{ARGV0, FLAVOR, request_stop, run};
