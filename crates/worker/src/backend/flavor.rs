//! Engine-flavor selection helpers for worker bootstrap.

use crate::config::EngineKind;

/// Which engine flavor to dispatch to at runtime.
///
/// EVERY variant is feature-gated, and there is no ungated one: a binary
/// built without a backend feature has no flavor at all, which is the truth
/// of it — such a build can never reach a device.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Flavor {
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    Metal,
    // NO TARGET PAIR on this one, unlike `Metal` above: the Vulkan shell binds
    // a loader rather than a platform, so the feature alone decides.
    #[cfg(feature = "vulkan")]
    Vulkan,
    // No target pair here either: wgpu picks its own backend (Vulkan, Metal,
    // DX12) at run time, so there is no platform this shell cannot be built
    // for and the feature alone decides.
    #[cfg(feature = "wgpu")]
    Wgpu,
}

impl Flavor {
    /// Lowercase string used in error messages and configuration plumbing.
    pub fn as_str(self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Flavor::Cuda => "cuda",
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Flavor::Metal => "metal",
            #[cfg(feature = "vulkan")]
            Flavor::Vulkan => "vulkan",
            #[cfg(feature = "wgpu")]
            Flavor::Wgpu => "wgpu",
        }
    }

    /// Map a TOML `engine.type` to the flavor that should host it,
    /// erroring with a clear message when the requested flavor was
    /// not compiled into this binary.
    pub fn from_kind(kind: EngineKind) -> Result<Self, String> {
        match kind {
            EngineKind::CudaNative => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Flavor::Cuda)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(missing_feature_msg("cuda_native", "cuda"))
                }
            }
            EngineKind::Metal => {
                // THREE ANSWERS, because there are three states and only one
                // of them is "rebuild with a feature". Metal's device half is
                // Apple-only at the crate level, so a Linux build with the
                // feature ON still hosts nothing — and telling that operator
                // to enable a flag they already enabled is advice that cannot
                // work. It is the one kind with a third answer: every other
                // shell is one feature flag away on every target.
                #[cfg(all(feature = "metal", target_vendor = "apple"))]
                {
                    Ok(Flavor::Metal)
                }
                #[cfg(all(feature = "metal", not(target_vendor = "apple")))]
                {
                    Err(non_apple_msg())
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(missing_feature_msg("metal", "metal"))
                }
            }
            EngineKind::Vulkan => {
                // TWO ANSWERS, not Metal's three: there is no target this
                // shell cannot be built for, so the only way to lack it is to
                // have left the feature off.
                #[cfg(feature = "vulkan")]
                {
                    Ok(Flavor::Vulkan)
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    Err(missing_feature_msg("vulkan", "vulkan"))
                }
            }
            EngineKind::Wgpu => {
                // Two answers, like Vulkan's: this shell has no target half
                // either, so leaving the feature off is the only way to lack
                // it. NO `retired_msg` ARM IS LEFT ANYWHERE — every kind
                // `EngineKind` names is a `--features` flag away, so the
                // "no build hosts this" answer has no kind to be about and
                // the function it lived in is gone.
                #[cfg(feature = "wgpu")]
                {
                    Ok(Flavor::Wgpu)
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    Err(missing_feature_msg("wgpu", "wgpu"))
                }
            }
        }
    }
}

/// The feature is on and the target cannot host it.
///
/// Not [`missing_feature_msg`], whose advice is "rebuild with a feature" — it
/// is already on. The crate is right there and builds; it is the DEVICE half
/// that has no implementation off Apple, which `engine-metal`'s own
/// `compile_error!` says in as many words.
#[cfg(all(feature = "metal", not(target_vendor = "apple")))]
fn non_apple_msg() -> String {
    format!(
        "engine type \"metal\" needs an Apple target and this binary was \
         built for another. The `metal` feature IS on — what it selects \
         off Apple is the engine's portable half, which answers questions no \
         GPU changes and serves nothing. Compiled flavors: {compiled}.",
        compiled = compiled_summary(),
    )
}

#[cfg(not(all(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "wgpu"
)))]
fn missing_feature_msg(toml_type: &str, feature: &str) -> String {
    format!(
        "engine type {toml_type:?} is not built into this binary. \
         Rebuild `worker` with `--features {feature}` (or include \
         it alongside the other backend features). Compiled flavors: {compiled}.",
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
            feature = "cuda",
            feature = "vulkan",
            feature = "wgpu",
            all(feature = "metal", target_vendor = "apple")
        )),
        allow(unused_mut, reason = "the pushes below are feature-gated")
    )]
    let mut out: Vec<&'static str> = Vec::new();
    #[cfg(feature = "cuda")]
    out.push("cuda");
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    out.push("metal");
    #[cfg(feature = "vulkan")]
    out.push("vulkan");
    #[cfg(feature = "wgpu")]
    out.push("wgpu");
    out.join(", ")
}

/// Per-flavor compiled-in status, in TOML-discriminator form (`cuda_native` /
/// `metal` / `vulkan` / `wgpu`). Used by both `pie engine list` and
/// `pie doctor` to render the embedded-engine section.
///
/// EVERY flavor is listed whether or not it was compiled, which is what makes
/// this the answer to "why can this binary not serve my config": a table of
/// only the compiled ones cannot say that the one you asked for is missing.
pub fn compiled_embedded() -> [(&'static str, bool); 4] {
    [
        ("cuda_native", cfg!(feature = "cuda")),
        (
            "metal",
            cfg!(all(feature = "metal", target_vendor = "apple")),
        ),
        ("vulkan", cfg!(feature = "vulkan")),
        ("wgpu", cfg!(feature = "wgpu")),
    ]
}

/// The flavor this binary carries, for commands that don't name one
/// (`pie config init`'s template, `pie smoke` without `--flavor`).
///
/// **ONE QUESTION, ONE ANSWER.** The linked platform is
/// `runtime::engine::load::this_box`'s to answer — the converter reads the
/// same fact — so this maps that answer onto a flavor rather than deciding it
/// a second time.
pub fn default_flavor() -> Option<Flavor> {
    match runtime::engine::load::this_box()? {
        #[cfg(feature = "cuda")]
        runtime::engine::load::Platform::Cuda => Some(Flavor::Cuda),
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        runtime::engine::load::Platform::Metal => Some(Flavor::Metal),
        #[cfg(feature = "vulkan")]
        runtime::engine::load::Platform::Vulkan => Some(Flavor::Vulkan),
        #[cfg(feature = "wgpu")]
        runtime::engine::load::Platform::Wgpu => Some(Flavor::Wgpu),
        // A platform this build did not link a flavor for. `this_box` answers
        // for the CONVERTER and this for the HOST, and the two sets are the
        // same today — the arm exists so that the day they are not, the
        // mismatch is a `None` and not a wrong flavor.
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

use anyhow::{Result, anyhow};

/// Resolve the `[model].engine.type` to the [`Flavor`] that hosts it, naming
/// the model in the refusal when this binary hosts none.
///
/// Every engine is a static lib, so "which of the ways of hosting one" has a
/// single answer and the flavor is the whole result.
pub fn resolve(kind: EngineKind, model_name: &str) -> Result<Flavor> {
    Flavor::from_kind(kind).map_err(|msg| anyhow!("model {model_name:?}: {msg}"))
}
