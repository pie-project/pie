//! Driver-flavor selection helpers for worker bootstrap.

use crate::config::DriverKind;

/// Which driver flavor to dispatch to at runtime.
///
/// EVERY variant is feature-gated, and there is no longer an ungated one.
/// `Dummy` used to be that: an always-present interpreter flavor that a
/// build with no device feature fell back to. It went with the crate,
/// so a binary built without one of the `driver-*` features now has no
/// flavor at all — which is the truth it was papering over, since that
/// build could never reach a device.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Flavor {
    #[cfg(feature = "_driver-cuda")]
    Cuda,
    #[cfg(feature = "driver-metal")]
    Metal,
    #[cfg(feature = "driver-vulkan")]
    Vulkan,
    #[cfg(feature = "driver-wgpu")]
    Wgpu,
}

impl Flavor {
    /// Lowercase string used in error messages and configuration plumbing.
    pub fn as_str(self) -> &'static str {
        match self {
            #[cfg(feature = "_driver-cuda")]
            Flavor::Cuda => "cuda",
            #[cfg(feature = "driver-metal")]
            Flavor::Metal => "metal",
            #[cfg(feature = "driver-vulkan")]
            Flavor::Vulkan => "vulkan",
            #[cfg(feature = "driver-wgpu")]
            Flavor::Wgpu => "wgpu",
        }
    }

    /// Map a TOML `driver.type` to the flavor that should host it,
    /// erroring with a clear message when the requested flavor was
    /// not compiled into this binary.
    pub fn from_kind(kind: DriverKind) -> Result<Self, String> {
        match kind {
            DriverKind::CudaNative => {
                #[cfg(feature = "_driver-cuda")]
                {
                    Ok(Flavor::Cuda)
                }
                #[cfg(not(feature = "_driver-cuda"))]
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
            DriverKind::Vulkan => {
                #[cfg(feature = "driver-vulkan")]
                {
                    Ok(Flavor::Vulkan)
                }
                #[cfg(not(feature = "driver-vulkan"))]
                {
                    Err(missing_feature_msg("vulkan", "driver-vulkan"))
                }
            }
            DriverKind::Wgpu => {
                #[cfg(feature = "driver-wgpu")]
                {
                    Ok(Flavor::Wgpu)
                }
                #[cfg(not(feature = "driver-wgpu"))]
                {
                    Err(missing_feature_msg("wgpu", "driver-wgpu"))
                }
            }
        }
    }
}

#[cfg(any(
    not(feature = "_driver-cuda"),
    not(feature = "driver-metal"),
    not(feature = "driver-vulkan"),
    not(feature = "driver-wgpu"),
))]
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
        not(any(
            feature = "_driver-cuda",
            feature = "driver-metal",
            feature = "driver-vulkan",
            feature = "driver-wgpu"
        )),
        allow(unused_mut, reason = "every push below is feature-gated")
    )]
    let mut out: Vec<&'static str> = Vec::new();
    #[cfg(feature = "_driver-cuda")]
    out.push("cuda");
    #[cfg(feature = "driver-metal")]
    out.push("metal");
    #[cfg(feature = "driver-vulkan")]
    out.push("vulkan");
    #[cfg(feature = "driver-wgpu")]
    out.push("wgpu");
    out.join(", ")
}

/// Per-flavor compiled-in status, in TOML-discriminator form (`cuda_native` /
/// `metal` / `vulkan` / `wgpu`). Used by both `pie driver list` and
/// `pie doctor` to render the embedded-driver section.
///
/// EVERY flavor is listed whether or not it was compiled, which is what makes
/// this the answer to "why can this binary not serve my config": a table of
/// only the compiled ones cannot say that the one you asked for is missing.
pub fn compiled_embedded() -> [(&'static str, bool); 4] {
    [
        ("cuda_native", cfg!(feature = "_driver-cuda")),
        ("metal", cfg!(feature = "driver-metal")),
        ("vulkan", cfg!(feature = "driver-vulkan")),
        ("wgpu", cfg!(feature = "driver-wgpu")),
    ]
}

/// Pick a sensible default flavor for commands that don't specify one
/// (e.g. `pie smoke` without `--flavor`, `pie config init`'s template).
/// Order: cuda → metal → vulkan → wgpu, and `None` when none was compiled in.
///
/// The two portable shells come last on purpose, and the reason covers both:
/// a machine with a CUDA build has an NVIDIA card, and either of them would
/// serve that card through a portable path rather than the vendor one. Between
/// themselves, Vulkan precedes wgpu because a build that asked for both asked
/// for the more specific one too — wgpu may well end up running over Vulkan,
/// and if that is what a deployment wants it can say so in one word.
pub fn default_flavor() -> Option<Flavor> {
    #[cfg(feature = "_driver-cuda")]
    {
        return Some(Flavor::Cuda);
    }
    #[cfg(all(not(feature = "_driver-cuda"), feature = "driver-metal"))]
    {
        return Some(Flavor::Metal);
    }
    #[cfg(all(
        not(feature = "_driver-cuda"),
        not(feature = "driver-metal"),
        feature = "driver-vulkan"
    ))]
    {
        return Some(Flavor::Vulkan);
    }
    #[cfg(all(
        not(feature = "_driver-cuda"),
        not(feature = "driver-metal"),
        not(feature = "driver-vulkan"),
        feature = "driver-wgpu"
    ))]
    {
        return Some(Flavor::Wgpu);
    }
    #[allow(unreachable_code)]
    None
}
