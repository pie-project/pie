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
    // `Metal`, `Vulkan` and `Wgpu` STOOD HERE. Their drivers left the
    // workspace at R3 — the last two were the last consumers of
    // `model-legacy` and `model_compiler::lower`, and none of the three can
    // be brought forward without a baker executor of its own (P5).
    // `DriverKind` still NAMES all three, so a deployment that asks for one
    // is told what happened instead of being told its config is malformed.
}

impl Flavor {
    /// Lowercase string used in error messages and configuration plumbing.
    pub fn as_str(self) -> &'static str {
        match self {
            #[cfg(feature = "_driver-cuda")]
            Flavor::Cuda => "cuda",
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
            // METAL IS BACK IN THE WORKSPACE and still hosted by no build,
            // which are two different facts and this arm now states the
            // second one only. See `unhosted_msg`.
            DriverKind::Metal => Err(unhosted_msg()),
            DriverKind::Vulkan => Err(retired_msg("vulkan")),
            DriverKind::Wgpu => Err(retired_msg("wgpu")),
        }
    }
}

/// A shader flavor that no build can host, whatever it was built with.
///
/// Distinct from [`missing_feature_msg`] on purpose: "rebuild with a feature"
/// is advice that would not work, because the crate the feature would name is
/// not in the workspace.
///
/// TWO OF THE THREE, since P5. `driver-metal` is a member again and has the
/// baker executor R3 named as the condition of its return, so the sentence
/// below stopped being true of it; [`unhosted_msg`] is the half that still
/// is.
fn retired_msg(toml_type: &str) -> String {
    format!(
        "driver type {toml_type:?} is not hosted by any build of pie right \
         now. `driver-{toml_type}` left the workspace with the legacy \
         declarations it was the last consumer of, and returns when its \
         baker executor lands (P5). Compiled flavors: {compiled}.",
        compiled = compiled_summary(),
    )
}

/// Metal: IN the workspace, and hosted by nothing.
///
/// Worth a second function rather than a reworded first. `driver-metal` came
/// back at P5 with the executor R3 named as the condition, and its portable
/// half — the walk, the bound statements, the layout arithmetic — builds and
/// tests on every host in the tree. What no build hosts is the SERVING half,
/// which is behind that crate's `metal-4` feature and `compile_error!`s off
/// an Apple target. So "rebuild with a feature" is advice that would work, on
/// a Mac, once `worker` grows the feature to name it with — and saying "it
/// left the workspace" would send a reader looking for a crate that is right
/// there.
fn unhosted_msg() -> String {
    format!(
        "driver type \"metal\" is not hosted by any build of pie right now. \
         `driver-metal` IS in the workspace and has its baker executor (P5); \
         what is missing is the wiring — its serving half is behind that \
         crate\'s `metal-4` feature, which needs an Apple target, and `worker` \
         states no `driver-metal` feature to select it with. Compiled \
         flavors: {compiled}.",
        compiled = compiled_summary(),
    )
}

#[cfg(not(feature = "_driver-cuda"))]
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
        not(feature = "_driver-cuda"),
        allow(unused_mut, reason = "the push below is feature-gated")
    )]
    let mut out: Vec<&'static str> = Vec::new();
    #[cfg(feature = "_driver-cuda")]
    out.push("cuda");
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
        // FALSE, and not because a feature is off: these three drivers are
        // out of the workspace until P5. See `retired_msg`.
        ("metal", false),
        ("vulkan", false),
        ("wgpu", false),
    ]
}

/// Pick a sensible default flavor for commands that don't specify one
/// (e.g. `pie smoke` without `--flavor`, `pie config init`'s template).
/// CUDA, or `None` when it was not compiled in. It was a preference order
/// while there were four flavors; the three shader shells are out of the
/// workspace until P5, so one candidate is the whole list.
pub fn default_flavor() -> Option<Flavor> {
    #[cfg(feature = "_driver-cuda")]
    {
        return Some(Flavor::Cuda);
    }
    #[allow(unreachable_code)]
    None
}
