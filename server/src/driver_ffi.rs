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

// Fatal callback: invoked at most once with the failure reason
// (NUL-terminated) just before the driver entry returns a nonzero code.
// Non-nullable — pass a no-op callback to keep the legacy behavior where
// the reason only reaches stderr. Lets the embedded host capture the
// reason on the structured `anyhow::Error` instead of just pointing at
// stderr (#356).
pub type FatalCb = unsafe extern "C" fn(reason: *const c_char, ctx: *mut c_void);

#[cfg(feature = "driver-portable")]
unsafe extern "C" {
    fn pie_driver_portable_run(
        argc: c_int,
        argv: *mut *mut c_char,
        install_signal_handlers: c_int,
        ready_cb: ReadyCb,
        ready_ctx: *mut c_void,
        fatal_cb: FatalCb,
        fatal_ctx: *mut c_void,
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
        fatal_cb: FatalCb,
        fatal_ctx: *mut c_void,
    ) -> c_int;
    fn pie_driver_cuda_request_stop();
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
            Flavor::Dummy => "pie_driver_dummy",
        }
    }

    /// Map a TOML `driver.type` to the flavor that should host it,
    /// erroring with a clear message when the requested flavor was
    /// not compiled into this binary or is hosted by a Python subprocess.
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
            DriverKind::Dummy => Ok(Flavor::Dummy),
            DriverKind::Dev | DriverKind::Vllm | DriverKind::Sglang => Err(format!(
                "driver type {kind:?} is hosted by a Python subprocess, \
                 not an embedded FFI flavor."
            )),
        }
    }
}

#[cfg(any(not(feature = "driver-portable"), not(feature = "driver-cuda"),))]
fn missing_feature_msg(toml_type: &str, feature: &str) -> String {
    format!(
        "driver type {toml_type:?} is not built into this binary. \
         Rebuild `pie-server` with `--features {feature}` (or include \
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
        ("dummy", true),
    ]
}

/// Whether the compiled portable driver includes ggml-metal support.
pub fn portable_metal_enabled() -> bool {
    cfg!(feature = "driver-portable") && option_env!("PIE_COMPILED_PORTABLE_METAL") == Some("1")
}

/// Pick a sensible default flavor for commands that don't specify one
/// (e.g. `pie smoke` without `--flavor`, `pie config init`'s template).
/// Order: cuda → portable → dummy.
pub fn default_flavor() -> Option<Flavor> {
    #[cfg(feature = "driver-cuda")]
    {
        return Some(Flavor::Cuda);
    }
    #[cfg(all(not(feature = "driver-cuda"), feature = "driver-portable"))]
    {
        return Some(Flavor::Portable);
    }
    #[cfg(all(not(feature = "driver-cuda"), not(feature = "driver-portable")))]
    {
        return Some(Flavor::Dummy);
    }
    #[allow(unreachable_code)]
    None
}

/// Invoke the driver entry for the given flavor. Mirrors the C
/// signature: blocks until the driver's serve loop exits, returns
/// the driver's rc.
///
/// `fatal_cb` (with `fatal_ctx`) is invoked at most once with the failure
/// reason just before the driver returns a nonzero rc; pass a no-op callback
/// to keep the legacy stderr-only behavior.
pub unsafe fn run(
    flavor: Flavor,
    argc: c_int,
    argv: *mut *mut c_char,
    install_signal_handlers: c_int,
    ready_cb: ReadyCb,
    ready_ctx: *mut c_void,
    fatal_cb: FatalCb,
    fatal_ctx: *mut c_void,
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
                fatal_cb,
                fatal_ctx,
            )
        },
        #[cfg(feature = "driver-cuda")]
        Flavor::Cuda => unsafe {
            pie_driver_cuda_run(
                argc,
                argv,
                install_signal_handlers,
                ready_cb,
                ready_ctx,
                fatal_cb,
                fatal_ctx,
            )
        },
        Flavor::Dummy => unsafe {
            pie_driver_dummy_run(
                argc,
                argv,
                install_signal_handlers,
                ready_cb,
                ready_ctx,
                fatal_cb,
                fatal_ctx,
            )
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
        Flavor::Dummy => unsafe { pie_driver_dummy_request_stop() },
    }
}
