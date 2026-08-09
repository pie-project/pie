//! Capability tiers: what to do when the fast path is optional.
//!
//! ## Why a tier and not a refusal
//!
//! The three siblings each answer "the hardware may not have this" differently,
//! and none of the three answers transfers.
//!
//! `kernels-cuda` asks CMake `pie_cuda_has_arch(90)` and compiles either the
//! real Hopper source or a stub, because a CUDA archive is built FOR an
//! architecture. `driver-metal` asks `supportsFamily(MTLGPUFamily::Metal4)` and
//! **refuses the device**, because "a fallback path would be a second driver"
//! and Metal is one vendor, so refusing costs a known and small set of Macs.
//! `kernels-vulkan` compiles the same entrypoint once per tier and chooses at
//! pipeline-creation time, because one SPIR-V tree ships to five vendors.
//!
//! WebGPU is the extreme of Vulkan's case. The same module runs on Vulkan,
//! Metal, D3D12 and a browser, and the two things worth having —
//! [`Capability::Fp16`] and [`Capability::Subgroup`] — are `wgpu::Features`
//! bits an adapter may simply not report. Refusing would refuse most of the
//! market. Building per-device is not available, because there is no build.
//!
//! So this crate takes Vulkan's answer, one layer cheaper: a tier is a set of
//! defines, the module is expanded and compiled at pipeline-creation time
//! anyway, and choosing a tier costs nothing but choosing a different
//! `BTreeMap`.
//!
//! ## The invariant that makes it safe
//!
//! **Every entrypoint has a Baseline variant.** A tier is an ADDITIONAL variant
//! for an entrypoint that already exists — never a new entrypoint, never a
//! replacement. Adding a tier does not move the 480.
//!
//! Two things enforce it, and the redundancy is deliberate because the failure
//! it prevents is invisible until a specific device runs a specific model — an
//! entrypoint that resolves on the author's GPU and on no other:
//!
//! * `build.rs` fails the build on an orphan tier;
//! * `tests/entrypoints.rs::every_tier_has_a_baseline_beneath_it` runs
//!   everywhere, including where no adapter exists.
//!
//! A driver that implements no tier at all is still **correct**, only slower.
//! It reads [`Capability::PREFERENCE`] (best first), takes the first tier the
//! adapter's features allow and the tree has a variant for, and falls back to
//! [`Capability::Baseline`].
//!
//! ## Why the tier is not in the signature table
//!
//! `model-compiler` reads the table, and the compiler must not learn which
//! device it is compiling for. A plan that named a tier would stop being
//! portable between two machines running the same build. The tier lives one
//! layer down, where the module is selected — the same seam `Prepare` and
//! `Source` already draw between what a contract STATES and what a backend DOES
//! about it.
//!
//! This module therefore depends on nothing, so `build.rs` can pull it in with
//! `#[path]` — the trick `kernels-cuda`'s build script uses for its tables. The
//! build STAMPS the variant names and the library TELLS a driver what to look
//! for; two copies of that vocabulary would drift.

/// A tier of optional device features a variant may require.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    /// Core WebGPU. Always present, and the fallback for everything.
    Baseline,
    /// `wgpu::Features::SHADER_F16` — WGSL's `enable f16;`.
    ///
    /// Buys true `f16` arithmetic instead of rounding through fp16 storage
    /// with f32 math. Note this is about the MATH: WGSL has no 16-bit type in
    /// a storage buffer whatever the feature says, so bf16 tensors stay
    /// `array<u32>` and get unpacked either way.
    Fp16,
    /// `wgpu::Features::SUBGROUP` — `subgroupAdd`, `subgroupMax` and friends.
    ///
    /// Buys a reduction that costs one instruction where the baseline costs a
    /// workgroup barrier and a shared-memory tree. Every reduction in this tree
    /// has a baseline body for the reason the module docs give, and the
    /// subgroup body is the same recurrence with the inner level replaced.
    Subgroup,
}

impl Capability {
    /// Best first. A driver walks this and takes what the adapter allows.
    pub const PREFERENCE: [Self; 3] = [Self::Subgroup, Self::Fp16, Self::Baseline];

    /// Every tier, for a test that must not silently skip one.
    pub const ALL: [Self; 3] = [Self::Baseline, Self::Fp16, Self::Subgroup];

    /// The `@tag` a `// pie:instantiate` line spells this tier with.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Fp16 => "fp16",
            Self::Subgroup => "subgroup",
        }
    }

    /// The tier a `@tag` names, or `None`.
    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.tag() == tag)
    }

    /// The key a variant of `entrypoint` at this tier is stored under.
    ///
    /// Baseline is **unsuffixed on purpose**, so a driver that has never heard
    /// of tiers finds the right variant knowing only the entrypoint name.
    #[must_use]
    pub fn variant(self, entrypoint: &str) -> String {
        match self {
            Self::Baseline => entrypoint.to_owned(),
            other => format!("{entrypoint}.{}", other.tag()),
        }
    }

    /// The `wgpu::Features` names an adapter must report for this tier.
    ///
    /// Named as strings rather than as `wgpu::Features` because this crate does
    /// not depend on `wgpu` — a table and a shader tree must not, for the
    /// reason `Cargo.toml` gives. A driver maps them; the mapping is one match.
    ///
    /// A driver must check EVERY name, not the first. `kernels-vulkan` records
    /// the session this rule cost: its matrix tier named its matrix extension
    /// and not the `shaderFloat16` its operands actually needed, the driver
    /// built the pipeline anyway, and the answer was `-9.5` where it should
    /// have been `-0.0618`.
    #[must_use]
    pub const fn requires(self) -> &'static [&'static str] {
        match self {
            Self::Baseline => &[],
            Self::Fp16 => &["SHADER_F16"],
            // SUBGROUP alone is what the reductions need. A body that used
            // `subgroupMatrix*` would need `SUBGROUP_MATRIX` as well, and would
            // be a fourth tier rather than this one widened -- a tier is a
            // promise about a body, and two bodies must not share one.
            Self::Subgroup => &["SUBGROUP"],
        }
    }

    /// The defines a variant at this tier is expanded with, on top of its own.
    ///
    /// A tier is a set of defines and nothing else, which is what makes adding
    /// one cheap: the body says `//#if defined(PIE_FP16)` in the one place the
    /// arithmetic differs, and the same source serves both.
    #[must_use]
    pub const fn defines(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Baseline => &[],
            Self::Fp16 => &[("PIE_FP16", "1")],
            Self::Subgroup => &[("PIE_SUBGROUP", "1")],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tag_round_trips() {
        for tier in Capability::ALL {
            assert_eq!(Capability::from_tag(tier.tag()), Some(tier));
        }
        assert_eq!(Capability::from_tag("coopmat"), None);
    }

    #[test]
    fn baseline_is_unsuffixed_and_a_tier_is_not() {
        assert_eq!(
            Capability::Baseline.variant("rms_single_row_bfloat16"),
            "rms_single_row_bfloat16"
        );
        assert_eq!(
            Capability::Fp16.variant("rms_single_row_bfloat16"),
            "rms_single_row_bfloat16.fp16"
        );
    }

    #[test]
    fn preference_is_best_first_and_ends_at_baseline() {
        assert_eq!(Capability::PREFERENCE.len(), Capability::ALL.len());
        assert_eq!(
            *Capability::PREFERENCE.last().expect("three tiers"),
            Capability::Baseline,
            "the walk has to end somewhere every adapter can reach",
        );
        for tier in Capability::ALL {
            assert!(
                Capability::PREFERENCE.contains(&tier),
                "`{}` is a tier no driver would ever choose",
                tier.tag(),
            );
        }
    }

    #[test]
    fn only_baseline_requires_nothing() {
        for tier in Capability::ALL {
            assert_eq!(
                tier.requires().is_empty(),
                tier == Capability::Baseline,
                "`{}` is a tier that either asks for nothing or is not optional",
                tier.tag(),
            );
        }
    }
}
