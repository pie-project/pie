//! The adapter's profile: the wgpu registry, the facts this engine reads off
//! its adapter, and resolution for one load.
//!
//! wgpu is the browser's engine, where the adapter is masked: there are no
//! checked-in profiles, so a load runs the registry defaults under the facts
//! the adapter reports, after a user's file where pie runs natively. See
//! `.wiki/device-profiles.md`.

use std::path::Path;
use std::sync::LazyLock;

use kernels_wgpu::tuning::{AXES, FACTS};
use profile::{Candidate, Facts, Profile, Registry, Report, Shelf, Source};

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("wgpu", FACTS, AXES);
    kernels_wgpu::tuning::register(&mut r);
    poem_compiler::fuse::flags::register(&mut r, &crate::FUSED);
    poem_compiler::costs::register(&mut r, 0);
    r
});

/// Every decision the wgpu engine exposes.
#[must_use]
pub fn registry() -> &'static Registry {
    &REGISTRY
}

/// Where wgpu's profiles live: a native user's only.
pub static SHELF: Shelf = Shelf {
    backend: "wgpu",
    seeded: &[],
};

/// The cores a decode split and the compiler size work by when the adapter
/// reports no count, as a browser never does: 16, what this engine assumed
/// for every such adapter before.
pub const UNREPORTED_CORES: u32 = 16;

/// What this engine read off an adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Device {
    pub name: String,
    pub vendor: &'static str,
    /// The narrowest subgroup a dispatch may run at; `None` without the
    /// subgroup feature.
    pub subgroup_min: Option<u32>,
    /// Multiprocessors or compute units, where the native driver says.
    pub cores: Option<u32>,
    pub f16: bool,
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

    /// The cores a decode split and the compiler size work by.
    #[must_use]
    pub fn sizing_cores(&self) -> u32 {
        self.cores.unwrap_or(UNREPORTED_CORES)
    }

    /// Whether the subgroup kernels, which fold with 32-lane shuffles, are
    /// safe here: only where no dispatch can run narrower than 32.
    #[must_use]
    pub fn subgroups_32(&self) -> bool {
        self.subgroup_min.is_some_and(|n| n >= 32)
    }
}

/// Resolve the profile one load runs under: a native user's file for this
/// adapter, else the registry defaults under the adapter's facts.
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
    candidates.push(Candidate {
        source: Source::Derived,
        text: "api = 1\n".to_string(),
    });
    registry().resolve(&candidates, &device.facts())
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
        assert_eq!(report.used, Source::Derived);
        let t = kernels_wgpu::Tuning::new(profile, d.sizing_cores(), 0);
        assert_eq!(
            (
                t.qmm_min_rows(),
                t.qmm_crossover_tg(),
                t.sdpa_tile_rows_per_request(),
                t.sdpa_mma(),
                t.sdpa_split_max(),
                t.sdpa_split_min_keys(),
                t.cores(),
            ),
            (12, 160, 32, false, 32, 256, 16)
        );
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
        assert!(at(Some(32)).subgroups_32());
        assert!(at(Some(64)).subgroups_32());
        assert!(
            !at(Some(16)).subgroups_32(),
            "an Intel SIMD16 dispatch would fold wrong"
        );
        assert!(!at(Some(8)).subgroups_32());
        assert!(!at(None).subgroups_32(), "an unreported width is not 32");
    }
}
