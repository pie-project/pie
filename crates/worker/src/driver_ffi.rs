//! Driver-flavor selection helpers for worker bootstrap.

use crate::config::DriverKind;

/// Which driver flavor to dispatch to at runtime.
///
/// EVERY variant is feature-gated, and there is no longer an ungated one.
/// `Dummy` used to be that: an always-present interpreter flavor that a
/// build with neither device feature fell back to. It went with the crate,
/// so a binary built without `driver-cuda` or `driver-metal` now has no
/// flavor at all — which is the truth it was papering over, since that
/// build could never reach a device.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Flavor {
    #[cfg(feature = "driver-cuda")]
    Cuda,
    #[cfg(feature = "driver-metal")]
    Metal,
}

impl Flavor {
    /// Lowercase string used in error messages and configuration plumbing.
    pub fn as_str(self) -> &'static str {
        match self {
            #[cfg(feature = "driver-cuda")]
            Flavor::Cuda => "cuda",
            #[cfg(feature = "driver-metal")]
            Flavor::Metal => "metal",
        }
    }

    /// Map a TOML `driver.type` to the flavor that should host it,
    /// erroring with a clear message when the requested flavor was
    /// not compiled into this binary.
    pub fn from_kind(kind: DriverKind) -> Result<Self, String> {
        match kind {
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
        }
    }
}

#[cfg(any(not(feature = "driver-cuda"), not(feature = "driver-metal"),))]
fn missing_feature_msg(toml_type: &str, feature: &str) -> String {
    format!(
        "driver type {toml_type:?} is not built into this binary. \
         Rebuild `worker` with `--features {feature}` (or include \
         it alongside other `driver-*` features). Compiled flavors: {compiled}.",
        compiled = compiled_summary(),
    )
}

/// Comma-separated list of flavors compiled into this binary, in
/// build-priority order. Used by error messages and `pie doctor`.
#[allow(
    clippy::vec_init_then_push,
    reason = "the pushes are `#[cfg]`-gated, and an attribute cannot be \
              attached to an element inside `vec![]`"
)]
pub fn compiled_summary() -> String {
    #[cfg_attr(
        not(any(feature = "driver-cuda", feature = "driver-metal")),
        allow(unused_mut, reason = "every push below is feature-gated")
    )]
    let mut out: Vec<&'static str> = Vec::new();
    #[cfg(feature = "driver-cuda")]
    out.push("cuda");
    #[cfg(feature = "driver-metal")]
    out.push("metal");
    out.join(", ")
}

/// Per-flavor compiled-in status, in TOML-discriminator form
/// (`cuda_native` / `metal`). Used by both `pie driver list` and
/// `pie doctor` to render the embedded-driver section.
pub fn compiled_embedded() -> [(&'static str, bool); 2] {
    [
        ("cuda_native", cfg!(feature = "driver-cuda")),
        ("metal", cfg!(feature = "driver-metal")),
    ]
}

/// Pick a sensible default flavor for commands that don't specify one
/// (e.g. `pie smoke` without `--flavor`, `pie config init`'s template).
/// Order: cuda → metal, and `None` when neither was compiled in.
pub fn default_flavor() -> Option<Flavor> {
    #[cfg(feature = "driver-cuda")]
    {
        return Some(Flavor::Cuda);
    }
    #[cfg(all(not(feature = "driver-cuda"), feature = "driver-metal"))]
    {
        return Some(Flavor::Metal);
    }
    #[allow(unreachable_code)]
    None
}
