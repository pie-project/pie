//! The GPU's profile: the CUDA registry, the facts this engine measures, the
//! checked-in profiles, and resolution for one load.

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

pub const AXES: &[&str] = &["rows"];

/// Whether fires are captured as CUDA graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.graphs", default = "on", tune = engine)]
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
    default = r#"{ use = "bodies", mem_mb = 4096 }"#,
    tune = engine
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
#[decision(id = "engine.copies", flag, default = true, tune = engine)]
pub struct Copies;

/// Whether the ops this engine can launch as one grouped kernel are grouped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "compiler.grouped", flag, default = true, tune = engine)]
pub struct Grouped;

/// The share of the card past the weights a fire's working set (the
/// activation arena, attention workspaces, decoded tiles) may take, in
/// percent; the rest is owed to the cache rows, bodies and guests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.working_set_percent", constant(range(10..=90)), default = 50, tune = engine)]
pub struct WorkingSetPercent(pub u32);

/// Row buckets per doubling of the default lattice: 1 is powers of two, 2
/// adds a bucket between each, which pads less and records more bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.lattice_steps", constant(set(1, 2)), default = 1, tune = engine)]
pub struct LatticeSteps(pub u32);

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("cuda", FACTS, AXES);
    kernels_cuda::tuning::register(&mut r);
    r.add::<GraphMode>()
        .add::<RecordingMode>()
        .add::<Copies>()
        .add::<Grouped>()
        .add::<WorkingSetPercent>()
        .add::<LatticeSteps>();
    poem_compiler::fuse::flags::register(&mut r, &crate::FUSED);
    poem_compiler::costs::register(&mut r, 2);
    r
});

#[must_use]
pub fn registry() -> &'static Registry {
    &REGISTRY
}

macro_rules! seeded {
    ($($file:literal),* $(,)?) => {
        pub const SEEDED: &[(&str, &str)] = &[
            $(($file, include_str!(concat!("../profiles/", $file)))),*
        ];
    };
}

seeded!(
    "sm80.toml",
    "sm86.toml",
    "sm87.toml",
    "sm89.toml",
    "sm89/nvidia-l40s.toml",
    "sm90.toml",
    "sm100.toml",
    "sm103.toml",
    "sm110.toml",
    "sm120.toml",
    "sm121.toml",
);

pub static SHELF: Shelf = Shelf {
    backend: "cuda",
    seeded: SEEDED,
    derived: None,
};

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
    pub working_set_percent: u32,
    pub lattice_steps: u32,
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
            working_set_percent: profile.get::<WorkingSetPercent>().0,
            lattice_steps: profile.get::<LatticeSteps>().0,
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
        // A box with no CUDA runtime has no GPU to name: cudarc panics on the
        // first call that loads the library, which `present` catches.
        if !crate::device::present() {
            return None;
        }
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
    registry().with_overrides(overrides)
}

/// The registry defaults with the graph mode, and the recording mode where
/// given, pinned: for a test or a diagnostic that runs eagerly.
#[must_use]
pub fn pinned(graphs: Graphs, recording: Option<Recording>) -> Profile {
    let mut text = format!(
        "[engine]\ngraphs = \"{}\"\n",
        match graphs {
            Graphs::Off => "off",
            Graphs::Shaped => "shaped",
            Graphs::On => "on",
        }
    );
    match recording {
        None => {}
        Some(Recording::Off) => text.push_str("recording = \"off\"\n"),
        Some(Recording::Shaped) => text.push_str("recording = \"shaped\"\n"),
        Some(Recording::Bodies { mem_megabytes }) => text.push_str(&format!(
            "recording = {{ use = \"bodies\", mem_mb = {mem_megabytes} }}\n"
        )),
    }
    profile(&text)
}

/// Resolve the profile one load runs under: its compute capability's
/// family file, the file for this GPU, then the user's.
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
    fn the_checked_in_profiles_hold_together_and_each_family_names_its_tier() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("profiles");
        SHELF
            .check(registry(), &dir)
            .unwrap_or_else(|e| panic!("{e}"));
        for (name, text) in SEEDED.iter().filter(|(n, _)| !n.contains('/')) {
            let file = registry()
                .parse(text, name, profile::Strict::CheckedIn)
                .expect("checked above");
            let cc = file
                .target
                .assumes
                .iter()
                .find(|(i, _)| FACTS[*i].name == "cc")
                .map(|(_, cc)| *cc)
                .expect("a family file assumes its compute capability");
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
    fn each_capability_takes_its_family_and_from_10_0_the_blackwell_kernels() {
        let defaults = kernels_cuda::tuning::Tuning::of(&registry().defaults());
        for (major, minor, gpu) in [
            (8, 0, "NVIDIA A100-SXM4-80GB"),
            (8, 6, "NVIDIA A10"),
            (8, 7, "Orin"),
            (8, 9, "NVIDIA L4"),
            (9, 0, "NVIDIA H100 80GB HBM3"),
            (10, 0, "NVIDIA B200"),
            (10, 3, "NVIDIA B300"),
            (11, 0, "NVIDIA Thor"),
            (12, 0, "NVIDIA RTX PRO 6000"),
            (12, 1, "NVIDIA GB10"),
        ] {
            let (profile, report) = resolve(gpu, &facts(major, minor, 100), None);
            assert_eq!(
                report.layers,
                [profile::Source::Seeded(format!("sm{major}{minor}.toml"))],
                "{gpu}"
            );
            assert!(report.untuned(), "{gpu}");
            assert_eq!(Engine::of(&profile), Engine::of(&registry().defaults()));
            let tuning = kernels_cuda::tuning::Tuning::of(&profile);
            if major >= 10 {
                assert!(tuning.gemv_narrow && tuning.mla_naive, "{gpu}");
                assert_eq!(
                    kernels_cuda::tuning::Tuning {
                        gemv_narrow: false,
                        mla_naive: false,
                        ..tuning
                    },
                    defaults,
                    "{gpu}: only the Blackwell kernels differ"
                );
            } else {
                assert_eq!(tuning, defaults, "{gpu}");
            }
        }
    }

    #[test]
    fn a_capability_with_no_family_runs_the_defaults_untuned() {
        let (profile, report) = resolve("NVIDIA Next", &facts(13, 0, 200), None);
        assert_eq!(report.used, profile::Source::Defaults);
        assert!(report.layers.is_empty());
        assert!(report.untuned());
        assert_eq!(
            kernels_cuda::tuning::Tuning::of(&profile),
            kernels_cuda::tuning::Tuning::of(&registry().defaults())
        );
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
                working_set_percent: 50,
                lattice_steps: 1,
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
    }

    #[test]
    fn the_l40s_runs_its_family_measured_and_another_ada_the_same_unmeasured() {
        let (l40s, report) = resolve("NVIDIA L40S", &facts(8, 9, 142), None);
        assert_eq!(
            report.layers,
            [
                profile::Source::Seeded("sm89.toml".into()),
                profile::Source::Seeded("sm89/nvidia-l40s.toml".into())
            ]
        );
        assert!(!report.untuned());
        let (rtx, report) = resolve("NVIDIA GeForce RTX 4090", &facts(8, 9, 128), None);
        assert_eq!(report.used, profile::Source::Seeded("sm89.toml".into()));
        assert!(report.untuned());
        for p in [&l40s, &rtx] {
            assert_eq!(Engine::of(p), Engine::of(&registry().defaults()));
            assert_eq!(
                kernels_cuda::tuning::Tuning::of(p),
                kernels_cuda::tuning::Tuning::of(&registry().defaults()),
                "what an Ada GPU ran before its family file"
            );
            for (spec, rungs, _) in p.entries() {
                assert_eq!(
                    rungs,
                    registry()
                        .defaults()
                        .entries()
                        .find(|(s, ..)| s.id == spec.id)
                        .unwrap()
                        .1,
                    "{}",
                    spec.id
                );
            }
        }
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
