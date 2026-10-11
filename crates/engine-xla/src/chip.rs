//! The TPU's profile: the XLA registry, the checked-in profiles, and
//! resolution for one load, by the device kind PJRT reports.
//!
//! XLA fuses on its own and prices nothing at compile time, so the registry
//! holds only the kernels' decisions. See `.wiki/device-profiles.md`.

use std::path::Path;
use std::sync::LazyLock;

use profile::{Facts, Profile, Registry, Report, Shelf};

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("xla", &[], &["rows"]);
    kernels_xla::tuning::register(&mut r);
    poem_compiler::fuse::flags::register(&mut r, &crate::FUSED);
    r
});

/// Every decision the XLA engine exposes.
#[must_use]
pub fn registry() -> &'static Registry {
    &REGISTRY
}

macro_rules! seeded {
    ($($file:literal),* $(,)?) => {
        /// The profiles checked in with pie, as `(file name, text)`.
        pub const SEEDED: &[(&str, &str)] = &[
            $(($file, include_str!(concat!("../profiles/", $file)))),*
        ];
    };
}

seeded!("tpu-v6-lite.toml");

/// Where XLA's profiles live.
pub static SHELF: Shelf = Shelf {
    backend: "xla",
    seeded: SEEDED,
};

/// Resolve the profile one load on a device of `kind` runs under: the
/// user's file, then the checked-in one, then the defaults (which are the
/// v6e's).
#[must_use]
pub fn resolve(kind: &str, profiles: Option<&Path>) -> (Profile, Report) {
    SHELF.resolve(registry(), kind, &Facts::of(registry()), profiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_holds_together() {
        assert_eq!(registry().check(), Vec::<String>::new());
    }

    #[test]
    fn every_checked_in_profile_reads_strictly_and_is_filed_under_its_device() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("profiles");
        SHELF
            .check(registry(), &dir)
            .unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn a_v6e_runs_its_file_and_the_file_is_what_it_ran_before() {
        let (profile, report) = resolve("TPU v6 lite", None);
        assert_eq!(
            report.used,
            profile::Source::Seeded("tpu-v6-lite.toml".into())
        );
        assert_eq!(
            kernels_xla::tuning::Tuning::of(&profile),
            kernels_xla::tuning::Tuning::DEFAULT
        );
        let (_, report) = resolve("TPU v5 lite", None);
        assert_eq!(report.used, profile::Source::Defaults);
    }
}
