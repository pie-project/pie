//! The shader tree, embedded — what a driver reads instead of a filesystem.
//!
//! ## Why the sources are in the rlib
//!
//! `kernels-metal` publishes `DEP_PIE_KERNELS_METAL_KERNELS_DIR` and a Metal
//! shell reads `.metal` files off disk at model load. That works, and it makes
//! a deployment carry a directory beside its binary: a `pie` moved without its
//! `crates/` tree compiles no kernels, and the failure arrives at the first
//! fire rather than at the build.
//!
//! `kernels-vulkan` has the same shape with a worse constant, because its
//! directory is `OUT_DIR/spv` — a path inside a `target/` tree that is not
//! meant to outlive a build.
//!
//! There is no reason for either here. A WGSL source is text, `build.rs` can
//! see it, and `include_str!` puts it in the binary. So [`SOURCES`] is the
//! whole tree, keyed by its path relative to `kernels/`, and a driver needs no
//! path, no environment variable and no files. The `links` variables are still
//! published — a tool that wants the tree on disk should not have to guess —
//! but nothing in the serving path reads them.
//!
//! ## What a caller does with it
//!
//! [`entrypoint_source`] is the whole interface: give it one of the 480 names
//! and a tier, get back the WGSL that entrypoint compiles from, with its
//! includes spliced, its `//#if` arms resolved and its defines declared as
//! `const`. That is exactly what `wgpu::Device::create_shader_module` wants.
//!
//! The lookup is by scan rather than by map, and deliberately: 480 entrypoints
//! over 38 files is a scan of a few thousand string comparisons, once per
//! distinct kernel in a model, at model load. A driver caches the pipeline, not
//! this.

use std::collections::BTreeMap;

use crate::Capability;
use crate::preproc::{Malformed, Variant};

include!(concat!(env!("OUT_DIR"), "/sources.rs"));

/// Why an entrypoint has no source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Missing {
    /// No `// pie:instantiate` line in the tree names it at this tier.
    ///
    /// For a tier other than [`Capability::Baseline`] this is ORDINARY: it is
    /// how a driver learns to fall back. For Baseline it is a defect the build
    /// should already have caught.
    NoVariant {
        /// The entrypoint asked for.
        entrypoint: String,
        /// The tier asked for.
        tier: Capability,
    },
    /// The variant exists and its source does not expand.
    Unexpandable {
        /// The file the directive sits in.
        file: String,
        /// What was wrong with it.
        why: Malformed,
    },
}

impl std::fmt::Display for Missing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoVariant { entrypoint, tier } => {
                write!(f, "no `{}` variant of `{entrypoint}`", tier.tag())
            }
            Self::Unexpandable { file, why } => write!(f, "{file}: {why}"),
        }
    }
}

impl std::error::Error for Missing {}

/// One shader source, by its path relative to `kernels/`.
///
/// `None` for a path the tree does not have — which is what an `//#include`
/// typo looks like, and the expander turns it into a
/// [`Malformed::Unincluded`] naming the line.
#[must_use]
pub fn source(path: &str) -> Option<&'static str> {
    SOURCES
        .iter()
        .find(|(name, _)| *name == path)
        .map(|(_, text)| *text)
}

/// Every variant the tree declares, with the file each was declared in.
///
/// The census `entrypoints.generated.txt` is written from, and what
/// `tests/entrypoints.rs` compares against the table's own product. A tree and
/// a table that disagree is a test failure here rather than a "no such
/// pipeline" at the first fire.
///
/// # Panics
///
/// Never in a shipped build: `build.rs` parses the same directives with the
/// same function and fails the build on a malformed one, so a source that
/// reaches this point has already been read once. The `expect` is what makes
/// that ordering an assertion rather than a hope.
#[must_use]
pub fn declared() -> Vec<(&'static str, Variant)> {
    let mut out = Vec::new();
    for (path, text) in SOURCES {
        let variants = crate::preproc::instantiations(text)
            .unwrap_or_else(|why| panic!("`{path}` was parsed at build time: {why}"));
        out.extend(variants.into_iter().map(|v| (*path, v)));
    }
    out
}

/// The WGSL one entrypoint compiles from, at one tier.
///
/// # Errors
///
/// [`Missing::NoVariant`] when the tree declares no such variant — for a tier
/// above Baseline that is the ordinary answer and a driver should fall back —
/// and [`Missing::Unexpandable`] when it does and the source is wrong.
pub fn entrypoint_source(entrypoint: &str, tier: Capability) -> Result<String, Missing> {
    let found = declared()
        .into_iter()
        .find(|(_, v)| v.entrypoint == entrypoint && v.tier == tier);

    let Some((file, variant)) = found else {
        return Err(Missing::NoVariant {
            entrypoint: entrypoint.to_owned(),
            tier,
        });
    };

    // The tier's own defines go UNDER the variant's, so a directive that states
    // `PIE_FP16=0` explicitly wins over the tier that would have set it. No
    // line in the tree does that today; the ordering is stated so that the one
    // that eventually does means what it reads as.
    let mut defines: BTreeMap<String, String> = tier
        .defines()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    defines.extend(variant.defines.clone());

    crate::preproc::expand(
        source(file).expect("the path came from SOURCES"),
        &defines,
        &|path| source(path).map(ToOwned::to_owned),
    )
    .map_err(|why| Missing::Unexpandable {
        file: file.to_owned(),
        why,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tree_is_embedded_and_not_empty() {
        assert!(
            !SOURCES.is_empty(),
            "`build.rs` found no `.wgsl` under `kernels/`",
        );
        assert!(
            source("common/bf16.inc.wgsl").is_some(),
            "the bf16 fragment every tensor body includes is missing",
        );
        assert!(source("nope/nothing.wgsl").is_none());
    }

    #[test]
    fn every_declared_variant_expands() {
        // The strongest thing that can be said without an adapter: the tree's
        // 480-odd variants all preprocess. `tests/gpu.rs` is what proves they
        // then COMPILE, and only where a device exists.
        for (path, variant) in declared() {
            let got = entrypoint_source(&variant.entrypoint, variant.tier);
            assert!(
                got.is_ok(),
                "`{}` (`{path}`, line {}) does not expand: {}",
                variant.entrypoint,
                variant.line,
                got.unwrap_err(),
            );
        }
    }

    #[test]
    fn a_tier_with_no_variant_says_so_rather_than_falling_back_silently() {
        // A driver's fallback is the DRIVER's decision. If this returned the
        // baseline source for a tier that has none, a driver could not tell
        // "the tree has a fast path" from "it does not", and would report a
        // tier it is not running.
        let baseline = declared()
            .into_iter()
            .find(|(_, v)| v.tier == Capability::Baseline)
            .expect("the tree has baseline variants");

        let got = entrypoint_source(&baseline.1.entrypoint, Capability::Fp16);
        if let Err(Missing::NoVariant { entrypoint, tier }) = got {
            assert_eq!(entrypoint, baseline.1.entrypoint);
            assert_eq!(tier, Capability::Fp16);
        } else {
            // Not a failure: it means this entrypoint HAS an fp16 variant, and
            // the check is then vacuous rather than wrong. Say so.
            assert!(
                got.is_ok(),
                "an fp16 variant that neither exists nor expands",
            );
        }
    }
}
