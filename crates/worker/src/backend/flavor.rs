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
    // No `Vulkan` or `Wgpu`: no build hosts those engines. `EngineKind` still
    // NAMES both, so a deployment that asks for one is refused by name instead
    // of being told its config is malformed.
}

impl Flavor {
    /// Lowercase string used in error messages and configuration plumbing.
    pub fn as_str(self) -> &'static str {
        match self {
            #[cfg(feature = "cuda")]
            Flavor::Cuda => "cuda",
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Flavor::Metal => "metal",
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
                // work, which is the failure `retired_msg` exists to avoid one
                // case over.
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
            EngineKind::Vulkan => Err(retired_msg("vulkan")),
            EngineKind::Wgpu => Err(retired_msg("wgpu")),
        }
    }
}

/// A shader flavor that no build can host, whatever it was built with.
///
/// Distinct from [`missing_feature_msg`] on purpose: "rebuild with a feature"
/// is advice that would not work, because the crate the feature would name is
/// not in the workspace.
///
/// `vulkan` and `wgpu` are the two kinds this covers; `metal` is hosted and
/// takes the ordinary [`missing_feature_msg`] instead.
fn retired_msg(toml_type: &str) -> String {
    format!(
        "engine type {toml_type:?} is not hosted by any build of pie. \
         Compiled flavors: {compiled}.",
        compiled = compiled_summary(),
    )
}

/// The feature is on and the target cannot host it.
///
/// Not [`missing_feature_msg`], whose advice is "rebuild with a feature" — it
/// is already on. Not [`retired_msg`] either: the crate is right there and
/// builds, it is the DEVICE half that has no implementation off Apple, which
/// `engine-metal`'s own `compile_error!` says in as many words.
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

#[cfg(not(all(feature = "cuda", feature = "metal")))]
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
            all(feature = "metal", target_vendor = "apple")
        )),
        allow(unused_mut, reason = "the pushes below are feature-gated")
    )]
    let mut out: Vec<&'static str> = Vec::new();
    #[cfg(feature = "cuda")]
    out.push("cuda");
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    out.push("metal");
    out.join(", ")
}

/// Per-flavor compiled-in status, in TOML-discriminator form (`cuda_native` /
/// `metal` / `vulkan` / `wgpu`). Used by both `pie engine list` and
/// `pie doctor` to render the embedded-engine section.
///
/// EVERY flavor is listed whether or not it was compiled, which is what makes
/// this the answer to "why can this binary not serve my config": a table of
/// only the compiled ones cannot say that the one you asked for is missing.
pub fn compiled_embedded() -> [(&'static str, bool); 2] {
    [
        ("cuda_native", cfg!(feature = "cuda")),
        (
            "metal",
            cfg!(all(feature = "metal", target_vendor = "apple")),
        ),
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
