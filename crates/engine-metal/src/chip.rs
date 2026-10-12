//! The Metal profile registry, the checked-in profiles, and resolution for
//! one load. Kernel decisions are declared in `kernels-metal`.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use profile::{Decision, Domain, Facts, Profile, Registry, Report, Shelf, Var};

const FACTS: &[Var] = &[
    Var::fixed("family", Domain::COUNT),
    Var::fixed("cores", Domain::COUNT),
    Var::fixed("cores_from", Domain::Enum(&["registry", "model"])),
    Var::fixed("perf_cores", Domain::COUNT),
    Var::fixed("ane", Domain::Bool),
];

/// The recording axes a dynamic variable may name.
const AXES: &[&str] = &["rows"];

/// How the dense MLPs share work with the Neural Engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ane.mlp", default = "gpu", tune = engine)]
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
    #[consistent_with("kernel.qmm = mpp")]
    Units {
        #[constant(range(1..=64), default = 8)]
        units: u32,
    },
}

/// How wide quantised projections share work with the Neural Engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ane.projections", default = "gpu", tune = engine)]
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
#[decision(id = "layout.weights", default = "plain", tune = engine)]
pub enum WeightsLayout {
    /// As the checkpoint stores them.
    Plain,
    /// Repacked once for the MPP matmul, which then reads the copy.
    #[consistent_with("kernel.qmm = mpp")]
    MppPacked,
}

/// How many rows a streamed expert cut serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.stream.cut", default = "fan_out", tune = engine)]
pub enum StreamCut {
    /// The resident slots over the router's fan-out.
    FanOut,
    /// A fixed row count.
    Fixed {
        #[constant(range(1..=65_536), default = 64)]
        rows: u32,
    },
}

/// The fewest prompt rows the Neural Engine takes: its programs hold a
/// procedure from 256 rows in steps of 128, so a raise only narrows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.ane.min_rows", constant(set(256, 384, 512, 768, 1024)), default = 256, tune = engine)]
pub struct AneMinRows(pub u32);

/// The fewest columns a projection must land to be split with the Neural
/// Engine, in its 512-column units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.ane.min_columns",
    constant(set(2048, 3072, 4096, 6144, 8192)),
    default = 4096,
    tune = engine
)]
pub struct AneMinColumns(pub u32);

/// How many predicted experts a streamed mixture-of-experts load reads
/// ahead from storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.experts.prefetch", constant(range(1..=16)), default = 4, tune = engine)]
pub struct ExpertsPrefetch(pub u32);

/// The copy threads a streamed mixture-of-experts load seats experts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "engine.experts.threads",
    default = r#"{ use = "perf_cores", most = 8 }"#,
    tune = engine
)]
pub enum ExpertsThreads {
    /// One a CPU performance core, at most `most`.
    PerfCores {
        #[constant(range(1..=32), default = 8)]
        most: u32,
    },
    Fixed {
        #[constant(range(1..=32), default = 8)]
        threads: u32,
    },
}

impl ExpertsThreads {
    #[must_use]
    pub fn count(self, perf_cores: u32) -> u32 {
        match self {
            ExpertsThreads::PerfCores { most } => perf_cores.clamp(1, most),
            ExpertsThreads::Fixed { threads } => threads,
        }
    }
}

/// The GPU cores the compiler prices a device at that does not say its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "compiler.unreported_cores", constant(range(1..=1024)), default = 32, tune = price)]
pub struct UnreportedCores(pub u32);

/// The GPU cores the compiler prices at: the device's, or where it does not
/// say, the profile's stand-in.
#[must_use]
pub fn pricing_cores(profile: &Profile, cores: Option<u32>) -> u32 {
    cores.unwrap_or_else(|| profile.get::<UnreportedCores>().0)
}

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("metal", FACTS, AXES);
    kernels_metal::tuning::register(&mut r);
    r.add::<AneMlp>()
        .add::<AneProjections>()
        .add::<WeightsLayout>()
        .add::<StreamCut>()
        .add::<AneMinRows>()
        .add::<AneMinColumns>()
        .add::<ExpertsPrefetch>()
        .add::<ExpertsThreads>()
        .add::<UnreportedCores>();
    poem_compiler::fuse::flags::register(&mut r, &crate::FUSED);
    // Metal forks no side streams: one queue serves every region.
    poem_compiler::costs::register(&mut r, 0);
    r
});

#[must_use]
pub fn registry() -> &'static Registry {
    &REGISTRY
}

macro_rules! seeded {
    ($($file:literal),* $(,)?) => {
        /// `(file name, text)`.
        pub const SEEDED: &[(&str, &str)] = &[
            $(($file, include_str!(concat!("../profiles/", $file)))),*
        ];
    };
}

seeded!(
    "apple7.toml",
    "apple8.toml",
    "apple9.toml",
    "apple9/apple-m4-pro.toml",
    "apple10.toml",
);

pub static SHELF: Shelf = Shelf {
    backend: "metal",
    seeded: SEEDED,
    derived: None,
};

#[must_use]
pub fn facts(family: u32, cores: Option<u32>, perf_cores: u32, ane: bool) -> Facts {
    let facts = Facts::of(registry())
        .int("family", i64::from(family))
        .int("perf_cores", i64::from(perf_cores))
        .flag("ane", ane);
    match cores {
        Some(cores) => facts.int("cores", i64::from(cores)),
        None => facts,
    }
}

/// The CPU's performance cores, or every physical core on a CPU without
/// performance levels; 0 where the system does not say.
#[must_use]
pub fn perf_cores() -> u32 {
    #[cfg(target_vendor = "apple")]
    {
        let read = |name: &std::ffi::CStr| {
            let mut value: i32 = 0;
            let mut len = std::mem::size_of::<i32>();
            // SAFETY: each name is an int sysctl; `len` states the out
            // buffer's size and the call writes at most that many bytes.
            let rc = unsafe {
                libc::sysctlbyname(
                    name.as_ptr(),
                    (&raw mut value).cast(),
                    &raw mut len,
                    std::ptr::null_mut(),
                    0,
                )
            };
            (rc == 0).then(|| u32::try_from(value).unwrap_or(0))
        };
        match read(c"hw.perflevel0.physicalcpu") {
            Some(cores) => cores,
            None if read(c"hw.nperflevels").is_none() => read(c"hw.physicalcpu").unwrap_or(0),
            None => 0,
        }
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        0
    }
}

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

/// The facts of the device `device` opened, with where its GPU core count
/// came from.
#[must_use]
pub fn measured(device: &crate::device::Context, perf_cores: u32) -> Facts {
    let measured = facts(device.family(), device.cores(), perf_cores, ane_loads());
    match device.cores_from() {
        Some(from) => measured.word("cores_from", from.word()),
        None => measured,
    }
}

/// `None` without a Metal 4 device.
#[must_use]
pub fn probe() -> Option<(String, Facts)> {
    let device = crate::device::Context::bind().ok()?;
    Some((device.name().to_string(), measured(&device, perf_cores())))
}

#[must_use]
pub fn resolve(device: &str, facts: &Facts, profiles: Option<&Path>) -> (Profile, Report) {
    SHELF.resolve(registry(), device, facts, profiles)
}

#[must_use]
pub fn user_path(profiles: &Path, device: &str) -> PathBuf {
    SHELF.user_path(profiles, device)
}

pub fn tell_untuned(device: &str, report: &Report) {
    SHELF.tell_untuned(device, report);
}

/// The registry defaults with `overrides` (profile TOML after `api = 1`)
/// laid over them, on a mid-size chip: for tests. Panics on an override the
/// registry refuses.
#[must_use]
pub fn tuning(overrides: &str, wide_range: bool) -> kernels_metal::Tuning {
    kernels_metal::Tuning::new(registry().with_overrides(overrides), wide_range, Some(16))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_holds_together() {
        assert_eq!(registry().check(), Vec::<String>::new());
    }

    #[test]
    fn the_checked_in_profiles_hold_together() {
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
    fn the_copy_threads_follow_the_performance_cores_up_to_their_cap() {
        let threads = *registry().defaults().get::<ExpertsThreads>();
        assert_eq!(
            [2, 4, 8, 10, 16].map(|p| threads.count(p)),
            [2, 4, 8, 8, 8],
            "an M4 Pro, with 8 or 10, keeps the 8 it ran before"
        );
        assert_eq!(threads.count(0), 1);
        assert_eq!(ExpertsThreads::Fixed { threads: 12 }.count(4), 12);
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn an_apple_cpu_says_how_many_performance_cores_it_has() {
        let p = perf_cores();
        assert!((1..=64).contains(&p), "{p}");
    }

    #[test]
    fn a_checked_in_file_holds_only_what_differs_from_the_layer_below() {
        let defaults = registry().defaults();
        let parse = |name: &str| {
            let text = SEEDED.iter().find(|(n, _)| *n == name).unwrap().1;
            registry()
                .parse(text, name, profile::Strict::CheckedIn)
                .unwrap_or_else(|e| panic!("{e}"))
        };
        for (name, _) in SEEDED {
            let family = name
                .split_once('/')
                .map(|(dir, _)| parse(&format!("{dir}.toml")));
            for (id, rungs) in &parse(name).entries {
                let below = match &family {
                    Some(family) => family.entries.get(id).cloned(),
                    None => None,
                }
                .or_else(|| {
                    defaults
                        .entries()
                        .find(|(spec, ..)| spec.id == *id)
                        .map(|(_, rungs, _)| rungs.to_vec())
                });
                assert_ne!(
                    below.as_ref(),
                    Some(rungs),
                    "{name}: {id} restates the layer below"
                );
            }
        }
    }

    #[test]
    fn a_device_no_file_names_runs_its_family_and_is_told_so() {
        let (profile, report) = resolve("Apple A17 Pro GPU", &facts(9, None, 10, true), None);
        assert_eq!(report.used, profile::Source::Seeded("apple9.toml".into()));
        assert!(report.untuned());
        assert_eq!(*profile.get::<AneMlp>(), AneMlp::Share { percent: 47 });
        let (profile, report) = resolve("Apple A17 Pro GPU", &facts(9, None, 10, false), None);
        assert!(report.rejected.is_empty(), "{report:?}");
        assert_eq!(
            *profile.get::<AneMlp>(),
            AneMlp::Gpu,
            "without the Neural Engine, its rung goes"
        );
    }

    #[test]
    fn the_m4_pro_runs_its_family_and_what_it_measured() {
        let (profile, report) = resolve("Apple M4 Pro", &facts(9, Some(20), 10, true), None);
        assert_eq!(
            report.layers,
            [
                profile::Source::Seeded("apple9.toml".into()),
                profile::Source::Seeded("apple9/apple-m4-pro.toml".into())
            ]
        );
        assert_eq!(report.measured, ["kernel.ane.mlp", "kernel.qmm_splitk"]);
        assert!(!report.untuned());
        assert_eq!(*profile.get::<AneMlp>(), AneMlp::Share { percent: 71 });
    }

    #[test]
    fn a_user_file_wins_and_a_broken_one_falls_back_whole() {
        let dir = std::env::temp_dir().join(format!("pie-chip-{}", std::process::id()));
        let path = user_path(&dir, "Apple M4 Max");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            "api = 1\n[kernel]\nqmm = [{ use = \"tiled\", crossover_tg = 120 }, \"qmv\"]\n",
        )
        .unwrap();
        let (profile, report) = resolve("Apple M4 Max", &facts(9, Some(40), 10, true), Some(&dir));
        assert!(
            matches!(report.used, profile::Source::User(_)),
            "{report:?}"
        );
        assert_eq!(
            profile
                .ladder::<kernels_metal::tuning::Qmm>()
                .values()
                .next(),
            Some(&kernels_metal::tuning::Qmm::Tiled { crossover_tg: 120 })
        );

        std::fs::write(
            &path,
            "api = 1\n[kernel]\nqmm = [{ use = \"tiled\", crossover_tg = 0 }, \"qmv\"]\n",
        )
        .unwrap();
        let (profile, report) = resolve("Apple M4 Max", &facts(9, Some(40), 10, true), Some(&dir));
        assert_eq!(report.used, profile::Source::Seeded("apple9.toml".into()));
        assert_eq!(report.rejected.len(), 1, "{report:?}");
        assert_eq!(
            profile
                .ladder::<kernels_metal::tuning::Qmm>()
                .values()
                .next(),
            Some(&kernels_metal::tuning::Qmm::Tiled { crossover_tg: 96 }),
            "the Apple9 family file again"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
