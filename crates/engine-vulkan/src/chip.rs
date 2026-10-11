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

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("vulkan", FACTS, AXES);
    kernels_vulkan::tuning::register(&mut r);
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

/// The profiles checked in with pie, as `(file name, text)`. None yet: every
/// device runs the profile its facts derive until someone measures one.
pub const SEEDED: &[(&str, &str)] = &[];

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

/// The vendor name a PCI vendor id stands for, as the `vendor` fact spells it.
#[must_use]
pub fn vendor_of(id: u32) -> &'static str {
    match id {
        0x10DE => "nvidia",
        0x1002 | 0x1022 => "amd",
        0x8086 => "intel",
        0x106B => "apple",
        0x5143 => "qualcomm",
        _ => "other",
    }
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
/// what every device without a measured file runs.
///
/// - The cooperative-matrix kernels where the device has them.
/// - A decode row's keys split across one workgroup per core the device
///   reports, else across 16.
/// - Adreno's workgroup memory is slow next to its ALUs: the plain tiled
///   point runs a 64x1024x7168 projection at ~27 GFLOPS there, so it takes
///   the register-tiled point, the one-row vector kernel where subgroups
///   are 32 wide or wider, and a decode split across 128 workgroups of 128
///   keys or more (3k keys of Qwen3.5-0.8B attend in 1.9 ms a token over six
///   layers, against 11.7 split two ways).
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
        kernel.push("qmm.path = \"coopmat\"".to_string());
        kernel.push("gemm.path = \"coopmat\"".to_string());
        kernel.push("moe.path = \"coopmat\"".to_string());
    }
    if device.vendor == "qualcomm" {
        kernel.push("qmm.register_tiles = true".to_string());
        if device.subgroup >= 32 {
            kernel.push("qmv.path = \"one_row\"".to_string());
        }
        kernel.push("sdpa.decode.split = { use = \"fixed\", groups = 128 }".to_string());
        kernel.push("sdpa.decode.split_min_keys = 128".to_string());
    } else if device.cores.is_some() {
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
    let profile = registry()
        .load_user(&format!("api = 1\n{overrides}"), "test overrides")
        .unwrap_or_else(|e| panic!("{e}"));
    kernels_vulkan::Tuning::new(profile, cores)
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
            device("qualcomm", 64, None, false),
            device("qualcomm", 16, None, false),
        ] {
            let (_, report) = resolve(&d, None);
            assert_eq!(report.used, Source::Derived, "{d:?}: {report:?}");
            assert!(report.untuned(), "a derived profile is worth tuning");
        }
    }

    /// What `DeviceTuning::of(info)` gave a device before profiles, frozen:
    /// `(min rows, crossover, register tiles, one-row, coopmat paths, split
    /// groups, split min keys)`. `cores` is what the device reported to it,
    /// floored at 16 then.
    fn old(d: &Device) -> (u32, u32, bool, bool, bool, u32, u32) {
        let qualcomm = d.vendor == "qualcomm";
        (
            12,
            160,
            qualcomm,
            qualcomm && d.subgroup >= 32,
            d.coopmat,
            if qualcomm {
                128
            } else {
                d.cores.unwrap_or(0).max(16)
            },
            if qualcomm { 128 } else { 256 },
        )
    }

    #[test]
    fn the_derived_profile_is_the_old_tuning_where_the_device_reported_honestly() {
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
            device("other", 32, None, false),
        ] {
            let (profile, report) = resolve(&d, None);
            assert_eq!(report.used, Source::Derived, "{d:?}: {report:?}");
            let cores = d.pricing_cores();
            let t = kernels_vulkan::Tuning::new(profile, cores);
            let new = (
                t.qmm_min_rows(),
                t.qmm_crossover_tg(),
                t.qmm_register_tiles(),
                t.qmv_one_row(),
                t.qmm_coopmat() && t.gemm_coopmat() && t.moe_coopmat(),
                t.sdpa_split_groups(),
                t.sdpa_split_min_keys(),
            );
            assert_eq!(new, old(&d), "{d:?}");
            assert_eq!(
                t.qmm_coopmat() || t.gemm_coopmat() || t.moe_coopmat(),
                d.coopmat,
                "{d:?}: every coopmat path follows the device"
            );
            assert!(!t.sdpa_mma());
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
                assert_eq!(
                    t.moe_tile_rows(pairs, experts),
                    old_tile,
                    "{pairs} over {experts}"
                );
            }
        }
    }

    #[test]
    fn a_device_reporting_fewer_than_16_cores_is_split_by_what_it_reported() {
        let d = device("nvidia", 32, Some(10), false);
        let (profile, _) = resolve(&d, None);
        let t = kernels_vulkan::Tuning::new(profile, d.pricing_cores());
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
            "api = 1\n[target]\nassumes = { subgroup = 32 }\n[kernel]\nqmv.path = \"one_row\"\n",
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
        assert!(!kernels_vulkan::Tuning::new(profile, 16).qmv_one_row());
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
