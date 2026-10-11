//! The chip's profile: the Metal registry, the facts this engine measures,
//! the checked-in profiles, and resolution for one load.
//!
//! A profile is chosen by the device's name and applies whole or not at all;
//! see `.wiki/device-profiles.md`. Kernel decisions are declared in
//! `kernels-metal`, fusions in `poem-compiler`; the decisions this crate
//! owns are below.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use profile::{Decision, Domain, Facts, Profile, Registry, Report, Shelf, Var};

/// What this engine measures about its device, for `target.assumes` and
/// what an alternative requires.
pub const FACTS: &[Var] = &[
    Var::fixed("family", Domain::COUNT),
    Var::fixed("cores", Domain::COUNT),
    Var::fixed("ane", Domain::Bool),
];

/// The recording axes a dynamic variable may name.
pub const AXES: &[&str] = &["rows"];

/// How the dense MLPs share work with the Neural Engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ane.mlp", default = "gpu")]
pub enum AneMlp {
    /// The GPU runs every unit.
    Gpu,
    /// The Neural Engine takes `percent` of the 512-wide units.
    #[requires("ane")]
    Share {
        #[constant(range(1..=99), default = 47)]
        percent: u32,
    },
    /// The Neural Engine takes a fixed count of units: what keeps it level
    /// with a GPU that reads prompts through MPP.
    #[requires("ane")]
    #[consistent_with("kernel.qmm.path = mpp")]
    Units {
        #[constant(range(1..=64), default = 8)]
        units: u32,
    },
}

/// How wide quantised projections share work with the Neural Engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ane.projections", default = "gpu")]
pub enum AneProjections {
    Gpu,
    /// The Neural Engine takes `percent` of each projection's columns.
    #[requires("ane")]
    Share {
        #[constant(range(1..=99), default = 47)]
        percent: u32,
    },
}

/// How dense weights are laid out when a load makes them resident.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "layout.weights", default = "plain")]
pub enum WeightsLayout {
    /// As the checkpoint stores them.
    Plain,
    /// Repacked once for the MPP matmul, which then reads the copy.
    #[consistent_with("kernel.qmm.path = mpp")]
    MppPacked,
}

/// How many rows a streamed expert cut serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.stream.cut", default = "fan_out")]
pub enum StreamCut {
    /// The resident slots over the router's fan-out.
    FanOut,
    /// A fixed row count.
    Fixed {
        #[constant(range(1..=65_536), default = 64)]
        rows: u32,
    },
}

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("metal", FACTS, AXES);
    kernels_metal::tuning::register(&mut r);
    r.add::<AneMlp>()
        .add::<AneProjections>()
        .add::<WeightsLayout>()
        .add::<StreamCut>();
    poem_compiler::fuse::flags::register(&mut r, &crate::FUSED);
    // Metal forks no side streams: one queue serves every region.
    poem_compiler::costs::register(&mut r, 0);
    r
});

/// Every decision the Metal engine exposes.
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

seeded!(
    "apple-m1.toml",
    "apple-m1-pro.toml",
    "apple-m1-max.toml",
    "apple-m1-ultra.toml",
    "apple-m2.toml",
    "apple-m2-pro.toml",
    "apple-m2-max.toml",
    "apple-m2-ultra.toml",
    "apple-m3.toml",
    "apple-m3-pro.toml",
    "apple-m3-max.toml",
    "apple-m3-ultra.toml",
    "apple-m4.toml",
    "apple-m4-pro.toml",
    "apple-m4-max.toml",
    "apple-m5.toml",
    "apple-m5-pro.toml",
    "apple-m5-max.toml",
);

/// Where Metal's profiles live.
pub static SHELF: Shelf = Shelf {
    backend: "metal",
    seeded: SEEDED,
};

/// The facts of a device with this family and core count, and whether the
/// Neural Engine bridge loads.
#[must_use]
pub fn facts(family: u32, cores: u32, ane: bool) -> Facts {
    Facts::of(registry())
        .int("family", i64::from(family))
        .int("cores", i64::from(cores))
        .flag("ane", ane)
}

/// Whether the Neural Engine bridge loads on this machine: the `ane` fact.
#[must_use]
pub fn ane_loads() -> bool {
    #[cfg(target_vendor = "apple")]
    {
        kernels_metal::ane::available().is_ok()
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        false
    }
}

/// The name and facts of this machine's Metal device: for `pie profile`
/// and doctor. `None` without a Metal 4 device.
#[must_use]
pub fn probe() -> Option<(String, Facts)> {
    let device = crate::device::Context::bind().ok()?;
    let facts = facts(device.family(), device.cores(), ane_loads());
    Some((device.name().to_string(), facts))
}

/// Resolve the profile one load runs under: the user's file for this
/// device, then the checked-in one, then the registry defaults.
#[must_use]
pub fn resolve(device: &str, facts: &Facts, profiles: Option<&Path>) -> (Profile, Report) {
    SHELF.resolve(registry(), device, facts, profiles)
}

/// Where a user's profile for `device` lives under `profiles`.
#[must_use]
pub fn user_path(profiles: &Path, device: &str) -> PathBuf {
    SHELF.user_path(profiles, device)
}

/// Say once per device that it runs without a profile measured on it.
pub fn tell_untuned(device: &str, report: &Report) {
    SHELF.tell_untuned(device, report);
}

/// The registry defaults with `overrides` (profile TOML after `api = 1`)
/// laid over them: for a test that exercises one decision on any chip.
/// Panics on an override the registry refuses.
#[must_use]
pub fn tuning(overrides: &str, wide_range: bool) -> kernels_metal::Tuning {
    let text = format!("api = 1\n{overrides}");
    let profile = registry()
        .load_user(&text, "test overrides")
        .unwrap_or_else(|e| panic!("{e}"));
    kernels_metal::Tuning::new(profile, wide_range)
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
    fn every_checked_in_profile_dumps_to_a_file_that_reads_back_strictly() {
        for (name, text) in SEEDED {
            let p = registry()
                .load(text, name)
                .unwrap_or_else(|e| panic!("{e}"));
            let dumped = profile::render::dump(&p);
            let again = registry()
                .parse(&dumped, "dump", profile::Strict::CheckedIn)
                .unwrap_or_else(|e| panic!("{name}: {e}\n---\n{dumped}"));
            for (spec, rungs, _) in p.entries() {
                assert_eq!(
                    again.entries.get(spec.id).map(Vec::as_slice),
                    Some(rungs),
                    "{name}: {}",
                    spec.id
                );
            }
            assert_eq!(again.target.assumes, p.target().assumes, "{name}");
        }
    }

    #[test]
    fn a_device_without_a_file_runs_the_defaults_and_is_told_so() {
        let (profile, report) = resolve("Apple A17 Pro GPU", &facts(9, 6, true), None);
        assert_eq!(report.used, profile::Source::Defaults);
        assert!(report.untuned());
        assert_eq!(
            *profile.get::<AneMlp>(),
            AneMlp::Gpu,
            "a default may require nothing"
        );
    }

    #[test]
    fn a_user_file_wins_and_a_broken_one_falls_back_whole() {
        let dir = std::env::temp_dir().join(format!("pie-chip-{}", std::process::id()));
        let path = user_path(&dir, "Apple M4 Max");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "api = 1\n[kernel]\nqmm.crossover_tg = 120\n").unwrap();
        let (profile, report) = resolve("Apple M4 Max", &facts(9, 40, true), Some(&dir));
        assert!(
            matches!(report.used, profile::Source::User(_)),
            "{report:?}"
        );
        assert_eq!(
            profile.get::<kernels_metal::tuning::QmmCrossoverTg>().0,
            120
        );

        std::fs::write(&path, "api = 1\n[kernel]\nqmm.crossover_tg = 0\n").unwrap();
        let (profile, report) = resolve("Apple M4 Max", &facts(9, 40, true), Some(&dir));
        assert_eq!(
            report.used,
            profile::Source::Seeded("apple-m4-max.toml".into())
        );
        assert_eq!(report.rejected.len(), 1, "{report:?}");
        assert_eq!(profile.get::<kernels_metal::tuning::QmmCrossoverTg>().0, 96);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
