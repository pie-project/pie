//! The device's profile: the Vulkan registry, the facts this engine
//! measures, and resolution for one load.
//!
//! Vulkan serves every GPU the other engines do not, so most devices have no
//! file: their profile is derived from the facts, as `derived` writes it,
//! after the user's file and any checked-in file whose `target.match` names
//! the device. See `.wiki/device-profiles.md`.

use std::path::Path;
use std::sync::LazyLock;

use kernels_vulkan::tuning::{AXES, FACTS};
use profile::{Candidate, Facts, Profile, Registry, Report, Shelf, Source};

/// Rows from which a step is submitted a layer at a time rather than in one
/// submit: in one, a long prefill on a slow GPU outlasts the driver's
/// watchdog (about 2 s on Adreno, which a 64-row prefill reaches) and loses
/// the device. A split step is not recorded, so a desktop GPU with no such
/// watchdog may say 0, never.
#[derive(Clone, Copy, Debug, PartialEq, Eq, profile::Decision)]
#[decision(id = "engine.submit.split_rows", constant(range(0..=1_048_576)), default = 32, tune = engine)]
pub struct SubmitSplitRows(pub u32);

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("vulkan", FACTS, AXES);
    kernels_vulkan::tuning::register(&mut r);
    r.add::<SubmitSplitRows>();
    poem_compiler::fuse::flags::register(&mut r, &crate::FUSED);
    // One queue serves every region.
    poem_compiler::costs::register(&mut r, 0);
    r
});

/// Every decision the Vulkan engine exposes.
#[must_use]
pub fn registry() -> &'static Registry {
    &REGISTRY
}

/// The profiles checked in with pie, as `(file name, text)`, matched by
/// device name or by `target.match`. A device none names runs the profile
/// its facts derive.
pub const SEEDED: &[(&str, &str)] = &[
    (
        "qualcomm-adreno-64.toml",
        include_str!("../profiles/qualcomm-adreno-64.toml"),
    ),
    (
        "qualcomm-adreno-128.toml",
        include_str!("../profiles/qualcomm-adreno-128.toml"),
    ),
    (
        "qualcomm-adreno.toml",
        include_str!("../profiles/qualcomm-adreno.toml"),
    ),
    (
        "imagination-powervr-128.toml",
        include_str!("../profiles/imagination-powervr-128.toml"),
    ),
];

/// Where Vulkan's profiles live.
pub static SHELF: Shelf = Shelf {
    backend: "vulkan",
    seeded: SEEDED,
};

/// The cores the compiler prices a device at when no extension reports a
/// count: 16, what this engine assumed for every such device before.
pub const UNREPORTED_CORES: u32 = 16;

/// What this engine measured about a device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    pub name: String,
    pub vendor: &'static str,
    pub subgroup: u32,
    /// Multiprocessors or compute units, where an extension says.
    pub cores: Option<u32>,
    pub coopmat: bool,
}

impl Device {
    #[must_use]
    pub fn facts(&self) -> Facts {
        let facts = Facts::of(registry())
            .word("vendor", self.vendor)
            .int("subgroup", i64::from(self.subgroup))
            .flag("coopmat", self.coopmat);
        match self.cores {
            Some(cores) => facts.int("cores", i64::from(cores)),
            None => facts,
        }
    }

    /// The cores the compiler prices this device at.
    #[must_use]
    pub fn pricing_cores(&self) -> u32 {
        self.cores.unwrap_or(UNREPORTED_CORES)
    }
}

/// The profile a device's facts derive, as a user file would spell it:
/// what every device without a measured or checked-in file runs.
///
/// - The cooperative-matrix kernels where the device has them.
/// - A decode row's keys split across one workgroup per core the device
///   reports, else across 16.
///
/// A vendor's own habits (Adreno's slow workgroup memory) live in a
/// checked-in file matched by vendor, not here.
#[must_use]
pub fn derived(device: &Device) -> String {
    let mut assumes = vec![
        format!("vendor = \"{}\"", device.vendor),
        format!("subgroup = {}", device.subgroup),
        format!("coopmat = {}", device.coopmat),
    ];
    if let Some(cores) = device.cores {
        assumes.push(format!("cores = {cores}"));
    }
    let mut kernel = Vec::new();
    if device.coopmat {
        kernel.push(
            "qmm = [\n  { when = \"rows >= 12\", use = \"coopmat\" },\n  \
             { when = \"rows >= 12\", use = \"tiled\", crossover_tg = 160 },\n  \
             { use = \"qmv\", vecs = 4 },\n]"
                .to_string(),
        );
        kernel.push("gemm.path = \"coopmat\"".to_string());
        kernel.push("moe.path = \"coopmat\"".to_string());
    }
    if device.cores.is_some() {
        kernel.push("sdpa.decode.split = { use = \"per_core\", per = 1 }".to_string());
    } else {
        kernel.push(format!(
            "sdpa.decode.split = {{ use = \"fixed\", groups = {UNREPORTED_CORES} }}"
        ));
    }
    format!(
        "api = 1\n\n[target]\ndevice = \"{}\"\nassumes = {{ {} }}\n\n[kernel]\n{}\n",
        device.name.replace('"', "'"),
        assumes.join(", "),
        kernel.join("\n")
    )
}

/// The checked-in files whose `target` names this device: by its exact
/// name, or by `match` on its vendor and, where stated, a substring of its
/// name.
fn matching(device: &Device) -> Vec<Candidate> {
    SEEDED
        .iter()
        .filter(|(name, text)| {
            let Ok(file) = registry().parse(text, name, profile::Strict::CheckedIn) else {
                return true; // read again by resolution, which says why it is refused
            };
            let t = &file.target;
            t.device.as_deref() == Some(device.name.as_str())
                || (t.vendor.as_deref() == Some(device.vendor)
                    && (t.name_contains.is_empty()
                        || t.name_contains
                            .iter()
                            .any(|s| device.name.contains(s.as_str()))))
        })
        .map(|(name, text)| Candidate {
            source: Source::Seeded((*name).to_string()),
            text: (*text).to_string(),
        })
        .collect()
}

/// Resolve the profile one load runs under: the user's file for this
/// device, then a checked-in file that names it, then the profile its facts
/// derive, then the registry defaults.
#[must_use]
pub fn resolve(device: &Device, profiles: Option<&Path>) -> (Profile, Report) {
    let mut candidates = Vec::new();
    if let Some(dir) = profiles {
        let path = SHELF.user_path(dir, &device.name);
        if let Ok(text) = std::fs::read_to_string(&path) {
            candidates.push(Candidate {
                source: Source::User(path.display().to_string()),
                text,
            });
        }
    }
    candidates.extend(matching(device));
    candidates.push(Candidate {
        source: Source::Derived,
        text: derived(device),
    });
    registry().resolve(&candidates, &device.facts())
}

/// The name and facts of the device a load would open: for `pie profile`
/// and doctor. `None` without a Vulkan loader or device.
#[must_use]
pub fn probe() -> Option<(String, Facts)> {
    let context = crate::device::Context::bind(&crate::api::DeviceBoot::default()).ok()?;
    let device = context.chip();
    Some((device.name.clone(), device.facts()))
}

/// Resolve by name and facts, as `probe` reports them.
#[must_use]
pub fn resolve_facts(name: &str, facts: &Facts, profiles: Option<&Path>) -> (Profile, Report) {
    let vendor = facts
        .get("vendor")
        .and_then(|i| usize::try_from(i).ok())
        .and_then(|i| match FACTS[0].domain {
            profile::Domain::Enum(names) => names.get(i).copied(),
            _ => None,
        })
        .unwrap_or("other");
    let device = Device {
        name: name.to_string(),
        vendor,
        subgroup: facts
            .get("subgroup")
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0),
        cores: facts.get("cores").and_then(|v| u32::try_from(v).ok()),
        coopmat: facts.get("coopmat") == Some(1),
    };
    resolve(&device, profiles)
}

/// The registry defaults with `overrides` (profile TOML after `api = 1`)
/// laid over them, for a device with `cores`: for a test that exercises one
/// decision. Panics on an override the registry refuses.
#[must_use]
pub fn tuning(overrides: &str, cores: u32) -> kernels_vulkan::Tuning {
    kernels_vulkan::Tuning::new(
        registry().with_overrides(overrides),
        kernels_vulkan::tuning::Limits::with_cores(cores),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_holds_together() {
        assert_eq!(registry().check(), Vec::<String>::new());
    }

    fn device(vendor: &'static str, subgroup: u32, cores: Option<u32>, coopmat: bool) -> Device {
        Device {
            name: format!("a {vendor} GPU"),
            vendor,
            subgroup,
            cores,
            coopmat,
        }
    }

    #[test]
    fn a_device_without_a_file_runs_what_its_facts_derive() {
        for d in [
            device("nvidia", 32, Some(142), true),
            device("amd", 64, Some(60), false),
            device("intel", 16, None, false),
        ] {
            let (_, report) = resolve(&d, None);
            assert_eq!(report.used, Source::Derived, "{d:?}: {report:?}");
            assert!(report.untuned(), "a derived profile is worth tuning");
        }
    }

    #[test]
    fn a_vendor_runs_its_checked_in_file() {
        for (d, file) in [
            (device("qualcomm", 64, None, false), "qualcomm-adreno-64.toml"),
            (device("qualcomm", 128, Some(6), false), "qualcomm-adreno-128.toml"),
            (device("qualcomm", 16, None, false), "qualcomm-adreno.toml"),
            (device("imagination", 128, None, false), "imagination-powervr-128.toml"),
        ] {
            let (_, report) = resolve(&d, None);
            assert_eq!(report.used, Source::Seeded(file.to_string()), "{d:?}: {report:?}");
        }
    }

    /// The kernel the cascade before the `kernel.qmm` ladder ran a dense
    /// matmul on, for a device: coopmat, then register tiles (Adreno), then
    /// tiled, from 12 rows and a contraction of whole 32-code tiles; the
    /// one-row kernel for one row on an Adreno 32 lanes wide or wider; else
    /// four vectors a workgroup.
    fn old_qmm(d: &Device, rows: u32, n: u32, k: u32) -> kernels_vulkan::tuning::Qmm {
        use kernels_vulkan::tuning::Qmm;
        let qualcomm = d.vendor == "qualcomm";
        let powervr = d.vendor == "imagination";
        let tiles = rows >= 12 && k % 32 == 0;
        if d.coopmat && tiles {
            Qmm::Coopmat
        } else if qualcomm && tiles {
            Qmm::RegisterTiled { bm: if rows > 32 { 64 } else { 32 } }
        } else if tiles && n % 16 == 0 {
            Qmm::Tiled { crossover_tg: 160 }
        } else if rows == 1 && qualcomm && d.subgroup >= 32 && (k * 4) % 128 == 0 && k % 8 == 0 {
            Qmm::OneRow { lanes: 32, rows: 1 }
        } else if rows == 1 && powervr && d.subgroup == 128 && (k * 4) % 128 == 0 && k % 8 == 0 {
            Qmm::OneRow { lanes: 8, rows: 2 }
        } else {
            Qmm::Qmv { vecs: 4 }
        }
    }

    #[test]
    fn every_device_runs_what_it_ran_before_the_ladder() {
        for d in [
            device("nvidia", 32, Some(142), true),
            device("nvidia", 32, Some(24), false),
            device("amd", 64, Some(60), true),
            device("amd", 32, None, false),
            device("intel", 16, None, false),
            device("intel", 32, Some(96), true),
            device("apple", 32, None, false),
            device("qualcomm", 64, None, false),
            device("qualcomm", 128, Some(6), false),
            device("qualcomm", 16, None, false),
            device("imagination", 128, None, false),
            device("imagination", 16, None, false),
            device("other", 32, None, false),
        ] {
            let (profile, report) = resolve(&d, None);
            let t = kernels_vulkan::Tuning::new(
                profile,
                kernels_vulkan::tuning::Limits::with_cores(d.pricing_cores()),
            );
            for rows in [1, 2, 4, 11, 12, 16, 32, 33, 64, 512] {
                for (n, k) in [(4096, 4096), (4096, 4100), (1000, 4096)] {
                    let at = t.qmm_at(rows, n, k, 4, 64);
                    assert_eq!(
                        t.qmm(&at)[0],
                        old_qmm(&d, rows, n, k),
                        "{d:?} from {}: {rows} x {n} x {k}",
                        report.used
                    );
                }
            }
            let qualcomm = d.vendor == "qualcomm";
            assert_eq!(
                t.sdpa_split_groups(),
                if qualcomm { 128 } else { d.cores.unwrap_or(0).max(16) },
                "{d:?}"
            );
            assert_eq!(t.sdpa_split_min_keys(), if qualcomm { 128 } else { 256 });
            assert_eq!(t.gemm_coopmat() && t.moe_coopmat(), d.coopmat, "{d:?}");
            assert_eq!(t.sdpa_tile_rows_per_request(), 32);
            for (pairs, experts) in [(0, 8), (15, 8), (16, 8), (255, 8), (256, 8), (4096, 64)] {
                let per = pairs / experts;
                let old_tile = if pairs < experts * 2 {
                    1
                } else if per >= 32 {
                    32
                } else {
                    16
                };
                assert_eq!(t.moe_tile_rows(pairs, experts), old_tile, "{pairs} over {experts}");
            }
        }
    }

    #[test]
    fn a_device_reporting_fewer_than_16_cores_is_split_by_what_it_reported() {
        let d = device("nvidia", 32, Some(10), false);
        let (profile, _) = resolve(&d, None);
        let t = kernels_vulkan::Tuning::new(
            profile,
            kernels_vulkan::tuning::Limits::with_cores(d.pricing_cores()),
        );
        assert_eq!(t.sdpa_split_groups(), 10, "the old floor of 16 was a guess");
    }

    #[test]
    fn a_user_file_that_picks_a_kernel_the_device_lacks_is_refused_whole() {
        let dir = std::env::temp_dir().join(format!("pie-vk-chip-{}", std::process::id()));
        let d = device("intel", 16, None, false);
        let path = SHELF.user_path(&dir, &d.name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            "api = 1\n[target]\nassumes = { subgroup = 32 }\n[kernel]\nqmm = [\"one_row\", \"qmv\"]\n",
        )
        .unwrap();
        let (profile, report) = resolve(&d, Some(&dir));
        assert_eq!(report.used, Source::Derived);
        assert!(
            report.rejected[0]
                .1
                .contains("it assumes subgroup = 32, and this device has 16"),
            "{:?}",
            report.rejected
        );
        let t = kernels_vulkan::Tuning::new(profile, kernels_vulkan::tuning::Limits::ASSUMED);
        assert!(!matches!(
            t.qmm(&t.qmm_at(1, 4096, 4096, 4, 64))[0],
            kernels_vulkan::tuning::Qmm::OneRow { .. }
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod device_probe {
    #[test]
    fn the_probe_reads_the_device_it_names() {
        let bound = crate::device::Context::bind(&crate::api::DeviceBoot::default());
        let Ok(context) = bound else {
            eprintln!("not asked: {:?}", bound.err());
            return;
        };
        let device = context.chip();
        eprintln!("{device:?}");
        assert!(!device.name.is_empty());
        assert!(device.subgroup > 0);
    }
}
