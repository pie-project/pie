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
    // BACK AT P5, and only this one. `driver-metal` walks a
    // `model_compiler::CompiledModel` through a baker executor of its own,
    // which was the stated precondition for all three.
    #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
    Metal,
    // `Vulkan` and `Wgpu` STOOD HERE and are still out: they were the last
    // consumers of `model-legacy` and `model_compiler::lower`, and neither has
    // its executor yet. `DriverKind` still NAMES both, so a deployment that
    // asks for one is told what happened instead of being told its config is
    // malformed.
}

impl Flavor {
    /// Lowercase string used in error messages and configuration plumbing.
    pub fn as_str(self) -> &'static str {
        match self {
            #[cfg(feature = "_driver-cuda")]
            Flavor::Cuda => "cuda",
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Flavor::Metal => "metal",
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
            // HOSTED NOW. The wiring `unhosted_msg` stood here to describe —
            // "`worker` states no `driver-metal` feature to select it with" —
            // is this feature, so the arm is the same cfg pair CUDA's is. The
            // message a non-Apple or featureless build gets is the ordinary
            // "rebuild with a feature", which is advice that works.
            DriverKind::Metal => {
                // THREE ANSWERS, because there are three states and only one
                // of them is "rebuild with a feature". Metal's device half is
                // Apple-only at the crate level, so a Linux build with the
                // feature ON still hosts nothing — and telling that operator
                // to enable a flag they already enabled is advice that cannot
                // work, which is the failure `retired_msg` exists to avoid one
                // case over.
                #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
                {
                    Ok(Flavor::Metal)
                }
                #[cfg(all(feature = "driver-metal", not(target_vendor = "apple")))]
                {
                    Err(non_apple_msg())
                }
                #[cfg(not(feature = "driver-metal"))]
                {
                    Err(missing_feature_msg("metal", "driver-metal"))
                }
            }
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
/// TWO OF THE THREE. `driver-metal` is a member again, has the baker executor
/// R3 named as the condition of its return, and is now wired through to a
/// binary — so the sentence below is false of it and it takes the ordinary
/// `missing_feature_msg` instead. `vulkan` and `wgpu` are what is left.
fn retired_msg(toml_type: &str) -> String {
    format!(
        "driver type {toml_type:?} is not hosted by any build of pie right \
         now. `driver-{toml_type}` left the workspace with the legacy \
         declarations it was the last consumer of, and returns when its \
         baker executor lands (P5). Compiled flavors: {compiled}.",
        compiled = compiled_summary(),
    )
}

/// The feature is on and the target cannot host it.
///
/// Not [`missing_feature_msg`], whose advice is "rebuild with a feature" — it
/// is already on. Not [`retired_msg`] either: the crate is right there and
/// builds, it is the DEVICE half that has no implementation off Apple, which
/// `driver-metal`'s own `compile_error!` says in as many words.
#[cfg(all(feature = "driver-metal", not(target_vendor = "apple")))]
fn non_apple_msg() -> String {
    format!(
        "driver type \"metal\" needs an Apple target and this binary was \
         built for another. The `driver-metal` feature IS on — what it selects \
         off Apple is the driver's portable half, which answers questions no \
         GPU changes and serves nothing. Compiled flavors: {compiled}.",
        compiled = compiled_summary(),
    )
}

#[cfg(not(all(feature = "_driver-cuda", feature = "driver-metal")))]
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
            all(feature = "driver-metal", target_vendor = "apple")
        )),
        allow(unused_mut, reason = "the pushes below are feature-gated")
    )]
    let mut out: Vec<&'static str> = Vec::new();
    #[cfg(feature = "_driver-cuda")]
    out.push("cuda");
    #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
    out.push("metal");
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
        // A FEATURE NOW, like cuda's. Metal is wired through to a binary; the
        // two below are still false for the reason `retired_msg` gives, which
        // is not a feature being off.
        (
            "metal",
            cfg!(all(feature = "driver-metal", target_vendor = "apple")),
        ),
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
