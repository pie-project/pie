//! The compiled SPIR-V, embedded — what a shell reads instead of a directory.
//!
//! ## Why the words are in the rlib
//!
//! `build.rs` writes 666 `.spv` files into `OUT_DIR/spv` and publishes the path
//! as `DEP_PIE_KERNELS_VULKAN_SPV_DIR`. That directory is still written and
//! still published, because a tool that wants to disassemble a module should
//! not have to extract one from an archive. What it stopped being is how a
//! SERVER finds its kernels.
//!
//! `OUT_DIR` names a build, not a deployment. A `pie` that resolved its
//! kernels there worked exactly as long as the machine that ran it was the
//! machine that built it, and when it was not, the failure arrived at the
//! first fire with a message about a missing kernel — a long way from the
//! configuration that caused it. Shipping the files beside the binary instead
//! is the same deployment with more moving parts: a release becomes an
//! archive, `pie init` writes a path into a config, and each of those is a way
//! for the words to go missing after the build has already proved they were
//! there.
//!
//! `kernels-wgpu` reached this conclusion first and named this crate's
//! directory handoff as the thing worth not copying. The cost here is smaller
//! than that argument needs: the whole compiled tree is 5.5 MB.
//!
//! ## What is in it, and what is not
//!
//! Without the `native` feature the table is EMPTY, and empty rather than
//! absent. `native` is what costs `slangc`, and the portable half of this
//! crate exists so that `model-ir` can read the signature table without owning
//! a shader toolchain — so a build without it must still compile, and must say
//! it has no modules rather than fail to mention them.
//!
//! The key is the FILE STEM — `add_bias_bfloat16`, or
//! `affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32.coopmat` — because that is the
//! vocabulary `driver-vulkan`'s `Modules` seam already looks a module up
//! under, and the tier walk that turns a symbol plus a [`Capability`] into one
//! of these names belongs to the driver that knows what its device supports.
//! Embedding changes where the words come from and nothing about what they are
//! called.

use crate::Capability;

include!(concat!(env!("OUT_DIR"), "/modules.rs"));

/// The module compiled under this exact stem.
///
/// The stem is [`Capability::module`] minus its `.spv`. A caller that has an
/// entrypoint and a tier should use [`code`], which spells the name from the
/// same rule the build stamped it with.
#[must_use]
pub fn stem(stem: &str) -> Option<&'static [u8]> {
    // Binary search and not a scan, unlike `kernels-wgpu`'s: that crate has 38
    // sources and resolves an entrypoint by expanding one of them, so its
    // lookup is dwarfed by the work it precedes. Here there are 666 keys, the
    // table is sorted by `build.rs`, and the answer IS the work.
    MODULES
        .binary_search_by_key(&stem, |&(name, _)| name)
        .ok()
        .map(|i| MODULES[i].1)
}

/// The module for `entrypoint` at exactly `tier`, or `None`.
///
/// Exactly, with no fallback: a tier is an ADDITIONAL module for an entrypoint
/// that already has a baseline, so "does this tier have one" is a question with
/// a real answer and walking down on the caller's behalf would hide it. The
/// walk down [`Capability::PREFERENCE`] is `driver-vulkan`'s `Modules`, which
/// is where the device's tier is known.
#[must_use]
pub fn code(entrypoint: &str, tier: Capability) -> Option<&'static [u8]> {
    let name = tier.module(entrypoint);
    stem(
        name.strip_suffix(".spv")
            .expect("`Capability::module` names a `.spv`"),
    )
}

/// Whether this build compiled any modules at all.
///
/// The `native`/portable split as a value rather than a `cfg`, so a caller can
/// say "this build has no kernels" in a message instead of failing to compile.
#[must_use]
pub fn embedded() -> bool {
    !MODULES.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table is sorted, which [`stem`] binary-searches on.
    ///
    /// Checked rather than trusted because `build.rs` sorts by stem while its
    /// source map is ordered by `(entrypoint, tier)`; the two agree for every
    /// name Slang accepts, and a test is what keeps that from being a fact
    /// nothing states.
    #[test]
    fn the_table_is_sorted_by_stem() {
        assert!(
            MODULES.windows(2).all(|w| w[0].0 < w[1].0),
            "the embedded module table is not sorted, so `stem` cannot find its keys"
        );
    }

    /// Every module has words in it.
    ///
    /// An empty `.spv` is what a `slangc` that failed without a failing exit
    /// status leaves behind, and it reaches a device as
    /// `vkCreateShaderModule` with a zero-length code array.
    #[test]
    fn no_module_is_empty() {
        for (name, code) in MODULES {
            assert!(!code.is_empty(), "`{name}` embedded as zero bytes");
            assert!(
                code.len() % 4 == 0,
                "`{name}` is {} bytes, which is not a whole number of SPIR-V words",
                code.len()
            );
            assert_eq!(
                &code[..4],
                // The SPIR-V magic number, little-endian. A module that does
                // not start with it is not one, whatever its extension said.
                &0x0723_0203_u32.to_le_bytes(),
                "`{name}` does not begin with the SPIR-V magic number"
            );
        }
    }

    /// Every tiered module has a baseline beside it.
    ///
    /// `capability.rs` states why: a tier with no baseline is an entrypoint
    /// that resolves on the author's GPU and on no other. `build.rs` asserts
    /// it over the DIRECTIVES; this asserts it over what was actually
    /// embedded, which is the set a driver will look in.
    #[test]
    fn no_tier_is_orphaned() {
        for (name, _) in MODULES {
            for tier in Capability::PREFERENCE {
                if tier == Capability::Baseline {
                    continue;
                }
                if let Some(entrypoint) = name.strip_suffix(&format!(".{}", tier.tag())) {
                    assert!(
                        stem(entrypoint).is_some(),
                        "`{entrypoint}` is embedded at tier `{}` with no baseline",
                        tier.tag()
                    );
                }
            }
        }
    }

    /// [`code`] finds what [`stem`] holds, under the name the build stamped.
    #[test]
    fn code_spells_the_name_the_build_stamped() {
        let Some((baseline, _)) = MODULES.iter().find(|(n, _)| !n.contains('.')) else {
            // No `native`, no modules. The other three tests are vacuous here
            // too; this one says so rather than pretending to have checked.
            assert!(!embedded(), "a non-empty table with no baseline module");
            return;
        };
        assert_eq!(
            code(baseline, Capability::Baseline),
            stem(baseline),
            "`code` and `stem` disagree about `{baseline}`"
        );
    }
}
