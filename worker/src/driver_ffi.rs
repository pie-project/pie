//! Driver-flavor FFI: feature-gated extern declarations and a
//! runtime [`Flavor`] dispatcher.
//!
//! Each driver crate / static lib exports a uniquely-named C entry
//! pair (`pie_driver_{portable,cuda,dummy}_run` / `_request_stop`).
//! Portable and cuda are selected by Cargo features; dummy is always
//! linked. The [`Flavor`] enum picks which one to invoke at runtime
//! based on the model's TOML `driver.type`.

use std::os::raw::{c_char, c_int, c_void};

use crate::config::DriverKind;

pub type ReadyCb = unsafe extern "C" fn(caps_json: *const c_char, ctx: *mut c_void);

#[cfg(feature = "driver-portable")]
unsafe extern "C" {
    /// In-process entry: hands the C++ driver a vtable of FFI callbacks
    /// for receiving requests and posting responses. There is no
    /// shmem variant — portable is embedded-only.
    fn pie_driver_portable_run_inproc(
        argc: c_int,
        argv: *mut *mut c_char,
        install_signal_handlers: c_int,
        ready_cb: ReadyCb,
        ready_ctx: *mut c_void,
        vtable: pie::driver::InProcVTable,
    ) -> c_int;
    fn pie_driver_portable_request_stop();
}

#[cfg(feature = "driver-cuda")]
unsafe extern "C" {
    /// In-process entry: hands the C++ driver a vtable of FFI callbacks
    /// for receiving requests and posting responses. There is no
    /// shmem variant — cuda_native is embedded-only.
    fn pie_driver_cuda_run_inproc(
        argc: c_int,
        argv: *mut *mut c_char,
        install_signal_handlers: c_int,
        ready_cb: ReadyCb,
        ready_ctx: *mut c_void,
        vtable: pie::driver::InProcVTable,
    ) -> c_int;
    fn pie_driver_cuda_request_stop();
}

#[cfg(feature = "driver-metal")]
unsafe extern "C" {
    /// In-process entry: hands the C++ driver a vtable of FFI callbacks
    /// for receiving requests and posting responses. There is no
    /// shmem variant — metal is embedded-only.
    fn pie_driver_metal_run_inproc(
        argc: c_int,
        argv: *mut *mut c_char,
        install_signal_handlers: c_int,
        ready_cb: ReadyCb,
        ready_ctx: *mut c_void,
        vtable: pie::driver::InProcVTable,
    ) -> c_int;
    fn pie_driver_metal_request_stop();
}

// The dummy driver is a Rust crate (rlib) in this workspace. We call
// its `extern "C"` entries directly — no `unsafe extern { ... }`
// declaration needed, which sidesteps the link-time symbol GC that
// strips unreferenced `#[no_mangle]` exports out of an rlib.
use pie_driver_dummy_lib::{pie_driver_dummy_request_stop, pie_driver_dummy_run};

/// Which driver flavor to dispatch to at runtime. Variants are
/// gated by Cargo features for portable/cuda; dummy is always present.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Flavor {
    #[cfg(feature = "driver-portable")]
    Portable,
    #[cfg(feature = "driver-cuda")]
    Cuda,
    #[cfg(feature = "driver-metal")]
    Metal,
    Dummy,
}

impl Flavor {
    /// Lowercase string used in error messages, RPC `query`, and the
    /// driver thread's argv\[0\].
    pub fn as_str(self) -> &'static str {
        match self {
            #[cfg(feature = "driver-portable")]
            Flavor::Portable => "portable",
            #[cfg(feature = "driver-cuda")]
            Flavor::Cuda => "cuda",
            #[cfg(feature = "driver-metal")]
            Flavor::Metal => "metal",
            Flavor::Dummy => "dummy",
        }
    }

    /// argv\[0\] passed to the driver entry. Each driver expects its
    /// own program name for usage messages.
    pub fn argv0(self) -> &'static str {
        match self {
            #[cfg(feature = "driver-portable")]
            Flavor::Portable => "pie_driver_portable",
            #[cfg(feature = "driver-cuda")]
            Flavor::Cuda => "pie_driver_cuda",
            #[cfg(feature = "driver-metal")]
            Flavor::Metal => "pie_driver_metal",
            Flavor::Dummy => "pie_driver_dummy",
        }
    }

    /// Map a TOML `driver.type` to the flavor that should host it,
    /// erroring with a clear message when the requested flavor was
    /// not compiled into this binary.
    pub fn from_kind(kind: DriverKind) -> Result<Self, String> {
        match kind {
            DriverKind::Portable => {
                #[cfg(feature = "driver-portable")]
                {
                    Ok(Flavor::Portable)
                }
                #[cfg(not(feature = "driver-portable"))]
                {
                    Err(missing_feature_msg("portable", "driver-portable"))
                }
            }
            DriverKind::CudaNative => {
                #[cfg(feature = "driver-cuda")]
                {
                    Ok(Flavor::Cuda)
                }
                #[cfg(not(feature = "driver-cuda"))]
                {
                    Err(missing_feature_msg("cuda_native", "driver-cuda"))
                }
            }
            DriverKind::Metal => {
                #[cfg(feature = "driver-metal")]
                {
                    Ok(Flavor::Metal)
                }
                #[cfg(not(feature = "driver-metal"))]
                {
                    Err(missing_feature_msg("metal", "driver-metal"))
                }
            }
            DriverKind::Dummy => Ok(Flavor::Dummy),
        }
    }
}

#[cfg(any(
    not(feature = "driver-portable"),
    not(feature = "driver-cuda"),
    not(feature = "driver-metal"),
))]
fn missing_feature_msg(toml_type: &str, feature: &str) -> String {
    format!(
        "driver type {toml_type:?} is not built into this binary. \
         Rebuild `pie-worker` with `--features {feature}` (or include \
         it alongside other `driver-*` features). Compiled flavors: {compiled}.",
        compiled = compiled_summary(),
    )
}

/// Comma-separated list of flavors compiled into this binary, in
/// build-priority order. Used by error messages and `pie doctor`.
pub fn compiled_summary() -> String {
    let mut out = Vec::new();
    #[cfg(feature = "driver-portable")]
    out.push("portable");
    #[cfg(feature = "driver-cuda")]
    out.push("cuda");
    #[cfg(feature = "driver-metal")]
    out.push("metal");
    out.push("dummy");
    out.join(", ")
}

/// Per-flavor compiled-in status, in TOML-discriminator form
/// (`portable` / `cuda_native` / `metal` / `dummy`). Used by both
/// `pie driver list` and `pie doctor` to render the embedded-driver section.
pub fn compiled_embedded() -> [(&'static str, bool); 4] {
    [
        ("portable", cfg!(feature = "driver-portable")),
        ("cuda_native", cfg!(feature = "driver-cuda")),
        ("metal", cfg!(feature = "driver-metal")),
        ("dummy", true),
    ]
}

/// Pick a sensible default flavor for commands that don't specify one
/// (e.g. `pie smoke` without `--flavor`, `pie config init`'s template).
/// Order: cuda → metal → portable → dummy.
pub fn default_flavor() -> Option<Flavor> {
    #[cfg(feature = "driver-cuda")]
    {
        return Some(Flavor::Cuda);
    }
    #[cfg(all(not(feature = "driver-cuda"), feature = "driver-metal"))]
    {
        return Some(Flavor::Metal);
    }
    #[cfg(all(
        not(feature = "driver-cuda"),
        not(feature = "driver-metal"),
        feature = "driver-portable"
    ))]
    {
        return Some(Flavor::Portable);
    }
    #[cfg(all(
        not(feature = "driver-cuda"),
        not(feature = "driver-metal"),
        not(feature = "driver-portable")
    ))]
    {
        return Some(Flavor::Dummy);
    }
    #[allow(unreachable_code)]
    None
}

/// Invoke the driver entry for the given flavor. Mirrors the C
/// signature: blocks until the driver's serve loop exits, returns
/// the driver's rc.
pub unsafe fn run(
    flavor: Flavor,
    argc: c_int,
    argv: *mut *mut c_char,
    install_signal_handlers: c_int,
    ready_cb: ReadyCb,
    ready_ctx: *mut c_void,
) -> c_int {
    match flavor {
        #[cfg(feature = "driver-portable")]
        Flavor::Portable => panic!("portable is embedded-only; use run_inproc"),
        #[cfg(feature = "driver-cuda")]
        Flavor::Cuda => panic!("cuda_native is embedded-only; use run_inproc"),
        #[cfg(feature = "driver-metal")]
        Flavor::Metal => panic!("metal is embedded-only; use run_inproc"),
        Flavor::Dummy => unsafe {
            pie_driver_dummy_run(argc, argv, install_signal_handlers, ready_cb, ready_ctx)
        },
    }
}

/// In-process variant of [`run`]. The runtime hands the driver a
/// `vtable` of FFI callbacks; both `cuda_native` and `portable` support
/// this and use it exclusively (no shmem fallback). `dummy` doesn't
/// have a C++ driver and isn't accepted here.
pub unsafe fn run_inproc(
    flavor: Flavor,
    argc: c_int,
    argv: *mut *mut c_char,
    install_signal_handlers: c_int,
    ready_cb: ReadyCb,
    ready_ctx: *mut c_void,
    vtable: pie::driver::InProcVTable,
) -> Result<c_int, &'static str> {
    match flavor {
        #[cfg(feature = "driver-cuda")]
        Flavor::Cuda => Ok(unsafe {
            pie_driver_cuda_run_inproc(
                argc,
                argv,
                install_signal_handlers,
                ready_cb,
                ready_ctx,
                vtable,
            )
        }),
        #[cfg(feature = "driver-portable")]
        Flavor::Portable => Ok(unsafe {
            pie_driver_portable_run_inproc(
                argc,
                argv,
                install_signal_handlers,
                ready_cb,
                ready_ctx,
                vtable,
            )
        }),
        #[cfg(feature = "driver-metal")]
        Flavor::Metal => Ok(unsafe {
            pie_driver_metal_run_inproc(
                argc,
                argv,
                install_signal_handlers,
                ready_cb,
                ready_ctx,
                vtable,
            )
        }),
        _ => Err("in-process driver entry not supported for this flavor"),
    }
}

/// Signal the given flavor's driver(s) to exit their serve loop.
/// Each driver's stop signal is process-global to that flavor — if
/// multiple driver instances of the same flavor are running (DP > 1),
/// they all exit on the next request boundary.
pub fn request_stop(flavor: Flavor) {
    match flavor {
        #[cfg(feature = "driver-portable")]
        Flavor::Portable => unsafe { pie_driver_portable_request_stop() },
        #[cfg(feature = "driver-cuda")]
        Flavor::Cuda => unsafe { pie_driver_cuda_request_stop() },
        #[cfg(feature = "driver-metal")]
        Flavor::Metal => unsafe { pie_driver_metal_request_stop() },
        Flavor::Dummy => unsafe { pie_driver_dummy_request_stop() },
    }
}
