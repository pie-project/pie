//! The GPU's profile: the ROCm registry, the facts this engine measures, the
//! checked-in profiles, and resolution for one load.

use std::path::Path;
use std::sync::LazyLock;

use profile::{Anywhere, Decision, Domain, Facts, Profile, Registry, Report, Shelf, Var};

use crate::{Graphs, Recording};

/// What this engine measures about its GPU.
pub const FACTS: &[Var] = &[
    // The gfx version as major * 10 + minor: 94 is gfx94x (CDNA 3).
    Var::fixed("cc", Domain::COUNT),
    Var::fixed("sms", Domain::COUNT),
];

pub const AXES: &[&str] = &["rows"];

/// Whether fires are captured as HIP graphs.
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
#[decision(id = "engine.lattice_steps", constant(set(1, 2, 4)), default = 1, tune = engine)]
pub struct LatticeSteps(pub u32);

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("rocm", FACTS, AXES);
    kernels_rocm::tuning::register(&mut r);
    r.add::<GraphMode>()
        .add::<RecordingMode>()
        .add::<Copies>()
        .add::<Grouped>()
        .add::<WorkingSetPercent>()
        .add::<LatticeSteps>();
    poem_compiler::fuse::flags::register(&mut r, &crate::FUSED);
    poem_compiler::costs::register_for(&mut r, 2, poem_compiler::costs::Conditionals::Absent);
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

seeded!("gfx94.toml");

pub static SHELF: Shelf = Shelf {
    backend: "rocm",
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
/// for `pie profile` and doctor. `None` without a ROCm runtime or device.
#[must_use]
pub fn probe(ordinal: i32) -> Option<(String, Facts)> {
    #[cfg(feature = "rocm")]
    {
        use kernels_rocm::hip::rt;
        // A box with no ROCm runtime has no GPU to name.
        if !crate::device::present() {
            return None;
        }
        let attribute = |which| {
            let mut value = 0i32;
            // SAFETY: a live out-parameter.
            let status = unsafe { rt::hipDeviceGetAttribute(&raw mut value, which, ordinal) };
            (status == rt::hipSuccess)
                .then_some(value)
                .and_then(|v| u32::try_from(v).ok())
        };
        let name = kernels_rocm::hip::device_name(u32::try_from(ordinal).ok()?)?;
        let facts = facts(
            attribute(rt::hipDeviceAttributeComputeCapabilityMajor)?,
            attribute(rt::hipDeviceAttributeComputeCapabilityMinor)?,
            attribute(rt::hipDeviceAttributeMultiprocessorCount)?,
        );
        Some((name, facts))
    }
    #[cfg(not(feature = "rocm"))]
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
                .expect("a family file assumes its gfx version");
            assert_eq!(
                file.target.tier.as_deref(),
                Some(format!("gfx{cc}").as_str()),
                "{name}: the tier is the gfx version it assumes"
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
    fn a_cdna3_gpu_takes_its_family() {
        for gpu in ["AMD Instinct MI300X", "AMD Instinct MI325X"] {
            let (profile, report) = resolve(gpu, &facts(9, 4, 304), None);
            assert_eq!(
                report.layers,
                [profile::Source::Seeded("gfx94.toml".into())],
                "{gpu}"
            );
            assert!(report.untuned(), "{gpu}");
            assert_eq!(Engine::of(&profile), Engine::of(&registry().defaults()));
            // The defaults, but for the routed MLP's paths the family measured.
            let mut want = kernels_rocm::tuning::Tuning::of(&registry().defaults());
            want.moe_lt_grouped = true;
            want.moe_lt_mlp_from = 128;
            want.moe_lt_select_from = 0;
            want.moe_lt_tile_deep = 256_192_064;
            want.moe_lt_tile_shallow = 128_288_032;
            want.moe_lt_deep_k = 2048;
            want.moe_ordered_from = 256;
            assert_eq!(kernels_rocm::tuning::Tuning::of(&profile), want, "{gpu}");
        }
    }

    #[test]
    fn a_capability_with_no_family_runs_the_defaults_untuned() {
        let (profile, report) = resolve("AMD Instinct Next", &facts(9, 6, 256), None);
        assert_eq!(report.used, profile::Source::Defaults);
        assert!(report.layers.is_empty());
        assert!(report.untuned());
        assert_eq!(
            kernels_rocm::tuning::Tuning::of(&profile),
            kernels_rocm::tuning::Tuning::of(&registry().defaults())
        );
    }

    #[test]
    fn the_defaults_are_the_cuda_engines_but_for_conditionals() {
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
        // HIP graphs have no conditional node: one is priced past any launch
        // it could skip, so the compiler lowers every region to AlwaysLaunch.
        assert_eq!(
            poem_compiler::DeviceProfile::of(&registry().defaults(), 132, &[], &[]),
            poem_compiler::DeviceProfile {
                cond_fixed_us: 1_000_000.0,
                ..poem_compiler::DeviceProfile::default()
            }
        );
        let mut fused = poem_compiler::fuse::flags::kernels(&registry().defaults());
        let mut shipped = crate::FUSED.to_vec();
        fused.sort_unstable();
        shipped.sort_unstable();
        assert_eq!(fused, shipped);
    }
}

#[cfg(all(test, feature = "rocm"))]
mod device {
    #[test]
    fn the_probe_reads_the_gpu_it_names() {
        let Some((name, facts)) = super::probe(0) else {
            eprintln!("not asked: no ROCm device answers");
            return;
        };
        assert!(!name.is_empty(), "the driver names the GPU");
        let cc = facts.get("cc").expect("probed");
        assert!(
            (90..100).contains(&cc),
            "{name} is gfx{cc}: the ROCm engine serves CDNA (gfx9)"
        );
        assert!(facts.get("sms").is_some_and(|n| n > 0));
    }
}
