//! The Vulkan registry and the profile one load resolves.

use std::path::Path;
use std::sync::LazyLock;

use kernels_vulkan::tuning::{AXES, FACTS};
use profile::{Facts, Profile, Registry, Report, Shelf};

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

seeded!("qualcomm.toml", "imagination.toml");

/// What a device no family file applies to runs.
pub const DERIVED: &str = include_str!("derived.toml");

pub static SHELF: Shelf = Shelf {
    backend: "vulkan",
    seeded: SEEDED,
    derived: Some(|_, _| DERIVED.to_string()),
};

/// The cores a device is sized by when no extension reports a count.
pub const UNREPORTED_CORES: u32 = 16;

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

    #[must_use]
    pub fn pricing_cores(&self) -> u32 {
        self.cores.unwrap_or(UNREPORTED_CORES)
    }
}

/// The vendor's family file, or the derived profile, then the user's file.
#[must_use]
pub fn resolve(device: &Device, profiles: Option<&Path>) -> (Profile, Report) {
    SHELF.resolve(registry(), &device.name, &device.facts(), profiles)
}

#[must_use]
pub fn probe() -> Option<(String, Facts)> {
    let context = crate::device::Context::bind(
        &crate::api::DeviceBoot::default(),
        engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
    )
    .ok()?;
    let device = context.chip();
    Some((device.name.clone(), device.facts()))
}

#[must_use]
pub fn resolve_facts(name: &str, facts: &Facts, profiles: Option<&Path>) -> (Profile, Report) {
    SHELF.resolve(registry(), name, facts, profiles)
}

#[cfg(test)]
mod tests {
    use kernels_vulkan::Tuning;
    use kernels_vulkan::tuning::{Limits, Qmm};

    use profile::Source;

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
        registry()
            .parse(DERIVED, "derived.toml", profile::Strict::CheckedIn)
            .unwrap_or_else(|e| panic!("{e}"));
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

    fn tuning(d: &Device) -> Tuning {
        let cores = d.pricing_cores();
        Tuning::new(
            resolve(d, None).0,
            Limits {
                cores,
                ..Limits::ASSUMED
            },
        )
    }

    #[test]
    fn a_device_runs_its_vendors_files_or_what_its_facts_derive() {
        let qualcomm = "qualcomm.toml";
        let imagination = "imagination.toml";
        for (d, files) in [
            (device("nvidia", 32, Some(142), true), &[][..]),
            (device("amd", 64, Some(60), false), &[]),
            (device("intel", 16, None, false), &[]),
            (device("qualcomm", 32, None, true), &[]),
            (device("qualcomm", 64, None, false), &[qualcomm]),
            (device("qualcomm", 128, Some(6), false), &[qualcomm]),
            (device("qualcomm", 16, None, false), &[qualcomm]),
            (device("qualcomm", 32, None, false), &[qualcomm]),
            (device("imagination", 32, None, false), &[imagination]),
            (device("imagination", 64, None, false), &[imagination]),
            (device("imagination", 128, None, false), &[imagination]),
            (device("imagination", 16, Some(24), false), &[imagination]),
        ] {
            let (_, report) = resolve(&d, None);
            let want: Vec<Source> = if files.is_empty() {
                vec![Source::Derived]
            } else {
                files
                    .iter()
                    .map(|f| Source::Seeded((*f).to_string()))
                    .collect()
            };
            assert_eq!(report.layers, want, "{d:?}");
            assert!(report.rejected.is_empty(), "{d:?}: {report:?}");
            assert!(report.untuned());
        }
    }

    /// The dense matmul the per-vendor cascade before `kernel.qmm` picked:
    /// coopmat, register tiles on Adreno and PowerVR, then tiled, from 12 rows over whole
    /// 32-code tiles; for one row, the one-row kernel on Adreno (32 lanes a
    /// row) and on PowerVR (8 lanes a row, two rows a lane); else
    /// four vectors a workgroup.
    fn old_qmm(d: &Device, rows: u32, n: u32, k: u32) -> Qmm {
        let qualcomm = d.vendor == "qualcomm";
        let tiles = rows >= 12 && k.is_multiple_of(32);
        let one_row = rows == 1 && k.is_multiple_of(32);
        if d.coopmat && tiles {
            Qmm::Coopmat
        } else if (qualcomm || d.vendor == "imagination") && tiles {
            Qmm::RegisterTiled {
                bm: if rows > 32 { 64 } else { 32 },
                k16: true,
            }
        } else if tiles && n.is_multiple_of(16) {
            Qmm::Tiled { crossover_tg: 160 }
        } else if one_row && qualcomm && d.subgroup >= 32 {
            Qmm::OneRow { lanes: 32, rows: 1 }
        } else if one_row && d.vendor == "imagination" && d.subgroup >= 32 {
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
            device("qualcomm", 8, None, false),
            device("qualcomm", 32, None, false),
            device("imagination", 8, None, false),
            device("imagination", 32, None, false),
            device("imagination", 64, None, false),
            device("imagination", 128, None, false),
            device("imagination", 16, None, false),
            device("imagination", 16, Some(24), false),
            device("other", 32, None, false),
        ] {
            let t = tuning(&d);
            for rows in [1, 2, 4, 11, 12, 16, 32, 33, 64, 512] {
                for (n, k) in [(4096, 4096), (4096, 4100), (1000, 4096)] {
                    let at = t.qmm_at(rows, n, k, 4, 64);
                    assert_eq!(
                        t.qmm(&at)[0],
                        old_qmm(&d, rows, n, k),
                        "{d:?}: {rows} x {n} x {k}"
                    );
                }
            }
            let qualcomm = d.vendor == "qualcomm";
            assert_eq!(
                t.sdpa_split_groups(),
                if qualcomm {
                    128
                } else {
                    d.cores.unwrap_or(0).max(16)
                },
                "{d:?}"
            );
            assert_eq!(t.sdpa_split_min_keys(), if qualcomm { 128 } else { 256 });
            assert_eq!(t.gemm_coopmat() && t.moe_coopmat(), d.coopmat, "{d:?}");
            assert_eq!(t.sdpa_tile_rows_per_request(), 32);
            for (pairs, experts, tile) in [
                (0, 8, 1),
                (15, 8, 1),
                (16, 8, 16),
                (255, 8, 16),
                (256, 8, 32),
                (4096, 64, 32),
            ] {
                assert_eq!(
                    t.moe_tile_rows(pairs, experts),
                    tile,
                    "{pairs} over {experts}"
                );
            }
        }
    }

    #[test]
    fn a_device_reporting_fewer_than_16_cores_is_split_by_what_it_reported() {
        assert_eq!(
            tuning(&device("nvidia", 32, Some(10), false)).sdpa_split_groups(),
            10
        );
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
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(report.used, Source::Derived);
        assert!(
            report.rejected[0]
                .1
                .contains("it assumes subgroup = 32, and this device has 16"),
            "{:?}",
            report.rejected
        );
        let t = Tuning::new(profile, Limits::ASSUMED);
        assert!(!matches!(
            t.qmm(&t.qmm_at(1, 4096, 4096, 4, 64))[0],
            Qmm::OneRow { .. }
        ));
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod device_probe {
    #[test]
    fn the_probe_reads_the_device_it_names() {
        let bound = crate::device::Context::bind(
            &crate::api::DeviceBoot::default(),
            engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        );
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
