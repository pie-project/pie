//! The wgpu registry and the profile one load resolves. A browser masks the
//! adapter, so there are no checked-in profiles.

use std::path::Path;
use std::sync::LazyLock;

use kernels_wgpu::tuning::{AXES, FACTS};
use profile::{Facts, Profile, Registry, Report, Shelf};

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("wgpu", FACTS, AXES);
    kernels_wgpu::tuning::register(&mut r);
    poem_compiler::fuse::flags::register(&mut r, &crate::FUSED);
    poem_compiler::costs::register(&mut r, 0);
    r
});

#[must_use]
pub fn registry() -> &'static Registry {
    &REGISTRY
}

pub static SHELF: Shelf = Shelf {
    backend: "wgpu",
    seeded: &[],
    derived: None,
};

/// The cores an adapter is sized by when it reports no count.
pub const UNREPORTED_CORES: u32 = 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    pub name: String,
    pub vendor: &'static str,
    /// The narrowest subgroup a dispatch may run at; `None` without the
    /// subgroup feature.
    pub subgroup_min: Option<u32>,
    pub cores: Option<u32>,
    pub f16: bool,
}

impl Device {
    #[must_use]
    pub fn facts(&self) -> Facts {
        let mut facts = Facts::of(registry())
            .word("vendor", self.vendor)
            .flag("f16", self.f16);
        if let Some(n) = self.subgroup_min {
            facts = facts.int("subgroup_min", i64::from(n));
        }
        if let Some(n) = self.cores {
            facts = facts.int("cores", i64::from(n));
        }
        facts
    }

    /// Whether the subgroup kernels, which fold with 32-lane shuffles, are
    /// safe: no dispatch may run narrower than 32.
    #[must_use]
    pub fn subgroups_32(&self) -> bool {
        self.subgroup_min.is_some_and(|n| n >= 32)
    }
}

/// The user's file for the adapter, over the registry defaults.
#[must_use]
pub fn resolve(device: &Device, profiles: Option<&Path>) -> (Profile, Report) {
    SHELF.resolve(registry(), &device.name, &device.facts(), profiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_holds_together() {
        assert_eq!(registry().check(), Vec::<String>::new());
    }

    #[test]
    fn an_adapter_runs_the_defaults_wgpu_ran_before() {
        let d = Device {
            name: "an adapter".to_string(),
            vendor: "intel",
            subgroup_min: Some(8),
            cores: None,
            f16: true,
        };
        let (profile, report) = resolve(&d, None);
        assert_eq!(report.used, profile::Source::Defaults);
        let t = kernels_wgpu::Tuning::new(profile, UNREPORTED_CORES, 0);
        assert_eq!(
            (
                t.sdpa_tile_rows_per_request(),
                t.sdpa_split_groups(),
                t.sdpa_split_min_keys()
            ),
            (32, 16, 256)
        );
        use kernels_wgpu::tuning::Qmm;
        for (rows, k, want) in [
            (1, 4096, Qmm::Qmv),
            (11, 4096, Qmm::Qmv),
            (12, 4096, Qmm::Tiled { crossover_tg: 160 }),
            (512, 4096, Qmm::Tiled { crossover_tg: 160 }),
            (512, 4100, Qmm::Qmv),
        ] {
            assert_eq!(
                t.qmm(&t.qmm_at(rows, 4096, k, 4, 64))[0],
                want,
                "{rows} x {k}"
            );
        }
        for (pairs, experts, tile) in [(15, 8, 1), (16, 8, 16), (256, 8, 32), (4096, 64, 32)] {
            assert_eq!(t.moe_tile_rows(pairs, experts), tile);
        }
    }

    #[test]
    fn only_an_adapter_no_narrower_than_32_lanes_runs_the_subgroup_kernels() {
        let at = |subgroup_min| Device {
            name: String::new(),
            vendor: "other",
            subgroup_min,
            cores: None,
            f16: false,
        };
        for (width, safe) in [
            (Some(32), true),
            (Some(64), true),
            (Some(16), false),
            (Some(8), false),
            (None, false),
        ] {
            assert_eq!(at(width).subgroups_32(), safe, "{width:?}");
        }
    }
}
