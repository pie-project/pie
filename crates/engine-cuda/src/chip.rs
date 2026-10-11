//! The GPU's profile: the CUDA registry, the facts this engine measures, the
//! checked-in profiles, and resolution for one load.
//!
//! A profile is chosen by the GPU's name and applies whole or not at all;
//! see `.wiki/device-profiles.md`. Kernel decisions are declared in
//! `kernels-cuda`, fusions and compiler prices in `poem-compiler`; the
//! engine's own are below.

use std::path::Path;
use std::sync::LazyLock;

use profile::{Anywhere, Decision, Domain, Facts, Profile, Registry, Report, Shelf, Var};

use crate::{Graphs, Recording};

/// What this engine measures about its GPU.
pub const FACTS: &[Var] = &[
    // Compute capability as major * 10 + minor: 89 is sm_89.
    Var::fixed("cc", Domain::COUNT),
    Var::fixed("sms", Domain::COUNT),
];

/// The recording axes a dynamic variable may name.
pub const AXES: &[&str] = &["rows"];

/// Whether fires are captured as CUDA graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.graphs", default = "on")]
pub enum GraphMode {
    /// Every fire launches eagerly: for diagnostics, not serving.
    Off,
    /// Fires are shaped for capture but launch eagerly.
    Shaped,
    /// Fires are captured and replayed.
    On,
}

/// How captured fires are kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "engine.recording",
    default = r#"{ use = "bodies", mem_mb = 4096 }"#
)]
pub enum RecordingMode {
    Off,
    Shaped,
    /// Recorded bodies are kept, up to `mem_mb` of device memory.
    Bodies {
        #[constant(range(0..=1_048_576), default = 4096)]
        mem_mb: u32,
    },
}

/// Whether a window that cannot be viewed in place is copied rather than
/// split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.copies", flag, default = true)]
pub struct Copies;

/// Whether the ops this engine can launch as one grouped kernel are grouped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "compiler.grouped", flag, default = true)]
pub struct Grouped;

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("cuda", FACTS, AXES);
    kernels_cuda::tuning::register(&mut r);
    r.add::<GraphMode>()
        .add::<RecordingMode>()
        .add::<Copies>()
        .add::<Grouped>();
    poem_compiler::fuse::flags::register(&mut r, &crate::FUSED);
    poem_compiler::costs::register(&mut r, 2);
    r
});

/// Every decision the CUDA engine exposes.
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

seeded!("nvidia-l40s.toml");

/// Where CUDA's profiles live.
pub static SHELF: Shelf = Shelf {
    backend: "cuda",
    seeded: SEEDED,
};

/// The facts of a GPU at compute capability `major.minor` with `sms`
/// multiprocessors.
#[must_use]
pub fn facts(major: u32, minor: u32, sms: u32) -> Facts {
    Facts::of(registry())
        .int("cc", i64::from(major * 10 + minor))
        .int("sms", i64::from(sms))
}

/// The settings one load on this GPU runs under, read from its profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Engine {
    pub graphs: Graphs,
    pub recording: Recording,
    pub copies: bool,
    pub grouped: bool,
}

impl Engine {
    #[must_use]
    pub fn of(profile: &Profile) -> Engine {
        Engine {
            graphs: match profile.get::<GraphMode>() {
                GraphMode::Off => Graphs::Off,
                GraphMode::Shaped => Graphs::Shaped,
                GraphMode::On => Graphs::On,
            },
            recording: match *profile.get::<RecordingMode>() {
                RecordingMode::Off => Recording::Off,
                RecordingMode::Shaped => Recording::Shaped,
                RecordingMode::Bodies { mem_mb } => Recording::Bodies {
                    mem_megabytes: mem_mb,
                },
            },
            copies: profile.on::<Copies>(&Anywhere),
            grouped: profile.on::<Grouped>(&Anywhere),
        }
    }
}

/// The name and facts of GPU `ordinal`, read without opening a context:
/// for `pie profile` and doctor. `None` without a CUDA runtime or device.
#[must_use]
pub fn probe(ordinal: i32) -> Option<(String, Facts)> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;
        // SAFETY: `prop` is a live out-parameter the call fills; an all-zero
        // `cudaDeviceProp` is a valid value of the plain C struct.
        let mut prop: rt::cudaDeviceProp = unsafe { core::mem::zeroed() };
        // SAFETY: as above.
        let status = unsafe { rt::cudaGetDeviceProperties(&raw mut prop, ordinal) };
        if status != rt::cudaError::cudaSuccess {
            return None;
        }
        let bytes: Vec<u8> = prop
            .name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        let name = String::from_utf8_lossy(&bytes).into_owned();
        let facts = facts(
            u32::try_from(prop.major).ok()?,
            u32::try_from(prop.minor).ok()?,
            u32::try_from(prop.multiProcessorCount).ok()?,
        );
        Some((name, facts))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinal;
        None
    }
}

/// The registry defaults with `overrides` (profile TOML after `api = 1`)
/// laid over them: for a test that exercises one decision on any GPU.
/// Panics on an override the registry refuses.
#[must_use]
pub fn profile(overrides: &str) -> Profile {
    registry()
        .load_user(&format!("api = 1\n{overrides}"), "test overrides")
        .unwrap_or_else(|e| panic!("{e}"))
}

/// Resolve the profile one load runs under: the user's file for this GPU,
/// then the checked-in one, then the registry defaults.
#[must_use]
pub fn resolve(device: &str, facts: &Facts, profiles: Option<&Path>) -> (Profile, Report) {
    SHELF.resolve(registry(), device, facts, profiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_holds_together() {
        assert_eq!(registry().check(), Vec::<String>::new());
    }

    #[test]
    fn every_checked_in_profile_reads_strictly_and_is_filed_under_its_gpu() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("profiles");
        SHELF
            .check(registry(), &dir)
            .unwrap_or_else(|e| panic!("{e}"));
        for (name, text) in SEEDED {
            let file = registry()
                .parse(text, name, profile::Strict::CheckedIn)
                .expect("checked above");
            let cc = file
                .target
                .assumes
                .iter()
                .find(|(i, _)| FACTS[*i].name == "cc")
                .map(|(_, cc)| *cc)
                .expect("a GPU profile assumes its compute capability");
            assert_eq!(
                file.target.tier.as_deref(),
                Some(format!("sm_{cc}").as_str()),
                "{name}: the tier is the capability it assumes"
            );
        }
    }

    use crate::DEFAULT_BODIES_MEGABYTES;

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
    fn the_defaults_are_what_cuda_ran_before_profiles() {
        let engine = Engine::of(&registry().defaults());
        assert_eq!(
            engine,
            Engine {
                graphs: Graphs::default(),
                recording: Recording::Bodies {
                    mem_megabytes: DEFAULT_BODIES_MEGABYTES
                },
                copies: true,
                grouped: true,
            }
        );
        assert_eq!(
            poem_compiler::DeviceProfile::of(&registry().defaults(), 132, &[], &[]),
            poem_compiler::DeviceProfile::default()
        );
        let mut fused = poem_compiler::fuse::flags::kernels(&registry().defaults());
        let mut shipped = crate::FUSED.to_vec();
        fused.sort_unstable();
        shipped.sort_unstable();
        assert_eq!(fused, shipped);
        assert_eq!(
            kernels_cuda::tuning::Tuning::of(&registry().defaults()),
            kernels_cuda::tuning::Tuning::default()
        );
    }

    #[test]
    fn the_l40s_runs_its_file_and_another_gpu_the_defaults() {
        let (_, report) = resolve("NVIDIA L40S", &facts(8, 9, 142), None);
        assert_eq!(
            report.used,
            profile::Source::Seeded("nvidia-l40s.toml".into())
        );
        let (_, report) = resolve("NVIDIA GeForce RTX 4090", &facts(8, 9, 128), None);
        assert_eq!(report.used, profile::Source::Defaults);
        assert!(report.untuned());
    }
}

#[cfg(all(test, feature = "cuda"))]
mod device {
    #[test]
    fn the_probe_reads_the_gpu_it_names() {
        let Some((name, facts)) = super::probe(0) else {
            eprintln!("not asked: no CUDA device answers");
            return;
        };
        assert!(!name.is_empty(), "the driver names the GPU");
        let cc = facts.get("cc").expect("probed");
        assert!(
            cc >= 80,
            "{name} is sm_{cc}: the CUDA engine serves sm_80 and newer"
        );
        assert!(facts.get("sms").is_some_and(|n| n > 0));
    }
}
