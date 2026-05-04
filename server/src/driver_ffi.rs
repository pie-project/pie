//! Driver-flavor FFI: feature-gated extern declarations and a
//! runtime [`Flavor`] dispatcher.
//!
//! Each driver crate / static lib exports a uniquely-named C entry
//! pair (`pie_driver_{portable,cuda,dummy}_run` / `_request_stop`),
//! so any subset of `driver-portable` / `driver-cuda` / `driver-dummy`
//! can be linked into the same binary. The [`Flavor`] enum picks
//! which one to invoke at runtime based on the model's TOML
//! `driver.type`.

use std::os::raw::{c_char, c_int, c_void};

use crate::config::DriverKind;

pub type ReadyCb = unsafe extern "C" fn(caps_json: *const c_char, ctx: *mut c_void);

#[cfg(feature = "driver-portable")]
unsafe extern "C" {
    fn pie_driver_portable_run(
        argc: c_int,
        argv: *mut *mut c_char,
        install_signal_handlers: c_int,
        ready_cb: ReadyCb,
        ready_ctx: *mut c_void,
    ) -> c_int;
    fn pie_driver_portable_request_stop();
}

#[cfg(feature = "driver-cuda")]
unsafe extern "C" {
    fn pie_driver_cuda_run(
        argc: c_int,
        argv: *mut *mut c_char,
        install_signal_handlers: c_int,
        ready_cb: ReadyCb,
        ready_ctx: *mut c_void,
    ) -> c_int;
    fn pie_driver_cuda_request_stop();
}

// The dummy driver is a Rust crate (rlib) in this workspace. We call
// its `extern "C"` entries directly — no `unsafe extern { ... }`
// declaration needed, which sidesteps the link-time symbol GC that
// strips unreferenced `#[no_mangle]` exports out of an rlib.
#[cfg(feature = "driver-dummy")]
use pie_driver_dummy_lib::{pie_driver_dummy_request_stop, pie_driver_dummy_run};

/// Which driver flavor to dispatch to at runtime. Variants are
/// gated by Cargo features — only the flavors compiled into this
/// binary are constructible.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Flavor {
    #[cfg(feature = "driver-portable")]
    Portable,
    #[cfg(feature = "driver-cuda")]
    Cuda,
    #[cfg(feature = "driver-dummy")]
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
            #[cfg(feature = "driver-dummy")]
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
            #[cfg(feature = "driver-dummy")]
            Flavor::Dummy => "pie_driver_dummy",
        }
    }

    /// Map a TOML `driver.type` to the flavor that should host it,
    /// erroring with a clear message when the requested flavor was
    /// not compiled into this binary or is torch-hosted.
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
            DriverKind::Dummy => {
                #[cfg(feature = "driver-dummy")]
                {
                    Ok(Flavor::Dummy)
                }
                #[cfg(not(feature = "driver-dummy"))]
                {
                    Err(missing_feature_msg("dummy", "driver-dummy"))
                }
            }
            DriverKind::Dev | DriverKind::Vllm | DriverKind::Sglang => Err(format!(
                "driver type {kind:?} is hosted by Python (`server/torch`) \
                 and is not available in `server`."
            )),
        }
    }
}

#[cfg(any(
    not(feature = "driver-portable"),
    not(feature = "driver-cuda"),
    not(feature = "driver-dummy"),
))]
fn missing_feature_msg(toml_type: &str, feature: &str) -> String {
    format!(
        "driver type {toml_type:?} is not built into this binary. \
         Rebuild `pie-standalone` with `--features {feature}` (or include \
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
    #[cfg(feature = "driver-dummy")]
    out.push("dummy");
    out.join(", ")
}

/// Per-flavor compiled-in status, in TOML-discriminator form
/// (`portable` / `cuda_native` / `dummy`). Used by both `pie driver list`
/// and `pie doctor` to render the embedded-driver section.
pub fn compiled_embedded() -> [(&'static str, bool); 3] {
    [
        ("portable", cfg!(feature = "driver-portable")),
        ("cuda_native", cfg!(feature = "driver-cuda")),
        ("dummy", cfg!(feature = "driver-dummy")),
    ]
}

/// Pick a sensible default flavor for commands that don't specify one
/// (e.g. `pie smoke` without `--flavor`, `pie config init`'s template).
/// Order: cuda → portable → dummy. Returns `None` only if no `driver-*`
/// feature is enabled, which `build.rs` already disallows.
pub fn default_flavor() -> Option<Flavor> {
    #[cfg(feature = "driver-cuda")]
    {
        return Some(Flavor::Cuda);
    }
    #[cfg(all(not(feature = "driver-cuda"), feature = "driver-portable"))]
    {
        return Some(Flavor::Portable);
    }
    #[cfg(all(
        not(feature = "driver-cuda"),
        not(feature = "driver-portable"),
        feature = "driver-dummy",
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
        Flavor::Portable => unsafe {
            pie_driver_portable_run(
                argc,
                argv,
                install_signal_handlers,
                ready_cb,
                ready_ctx,
            )
        },
        #[cfg(feature = "driver-cuda")]
        Flavor::Cuda => unsafe {
            pie_driver_cuda_run(argc, argv, install_signal_handlers, ready_cb, ready_ctx)
        },
        #[cfg(feature = "driver-dummy")]
        Flavor::Dummy => unsafe {
            pie_driver_dummy_run(argc, argv, install_signal_handlers, ready_cb, ready_ctx)
        },
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
        #[cfg(feature = "driver-dummy")]
        Flavor::Dummy => unsafe { pie_driver_dummy_request_stop() },
    }
}
