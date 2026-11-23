//! The wgpu registry and the profile one load resolves. A browser masks the
//! adapter, so there are no checked-in decision profiles; a native
//! adapter's committed measured profile is listed in [`SEEDED`].

use std::path::Path;
use std::sync::LazyLock;

use crate::tuning::{AXES, FACTS};
use profile::{Facts, Profile, Registry, Report, Shelf};

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("wgpu", FACTS, AXES);
    crate::tuning::register(&mut r);
    poem_compiler::fuse::flags::register(&mut r, &crate::FUSIONS);
    poem_compiler::costs::register(&mut r, 0);
    r
});

#[must_use]
pub fn registry() -> &'static Registry {
    &REGISTRY
}

/// The committed measured profiles of adapters native wgpu serves
/// (`profiles/<device>.measured.toml`, `pie tune export --engine wgpu`),
/// each listed here as `"<device>.measured.toml"`: a browser's masked
/// adapter matches none. No decision profile is checked in.
macro_rules! seeded {
    ($($file:literal),* $(,)?) => {
        pub const SEEDED: &[(&str, &str)] = &[
            $(($file, include_str!(concat!("../profiles/", $file)))),*
        ];
    };
}

seeded!("nvidia-rtx-spark-n1x-6144-core-blackwell-rtx-gpu.measured.toml");

/// The host covers committed for the adapters native wgpu serves
/// (`profiles/<device>.covers.json`, `pie tune export --engine wgpu`), each
/// listed here: the runtime takes a cover whose key matches where the
/// home's own cache holds none (`engine::Engine::committed_covers`).
pub const COVERS: &[&str] = &[include_str!(
    "../profiles/nvidia-rtx-spark-n1x-6144-core-blackwell-rtx-gpu.covers.json"
)];

pub static SHELF: Shelf = Shelf {
    backend: "wgpu",
    seeded: SEEDED,
    derived: None,
};

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

/// [`resolve`] for a load on the device `on` names (its vendor, device id
/// and driver): the measurements the user's file stores kept where their
/// keys match this build on this device, the rest dropped to be measured
/// again ([`profile::fire::Rates::sift`]), each said in the load log.
#[must_use]
pub fn resolve_stored(device: &Device, on: &str, profiles: Option<&Path>) -> (Profile, Report) {
    let keys = engine_gpu::stored::Keys::new(
        on.to_string(),
        <crate::device::Api as engine_gpu::serve::Serve>::TARGET,
    );
    let (profile, report) =
        SHELF.resolve_keyed(registry(), &device.name, &device.facts(), profiles, &keys);
    for line in &report.stored {
        tracing::info!("{line}");
    }
    (profile, report)
}

/// The adapter this machine would serve, by name, with its facts.
#[must_use]
pub fn probe() -> Option<(String, Facts)> {
    probe_at(0)
}

/// What this build keys device `index`'s stored measurements by
/// (`profile::fire::Now`: its device and driver, each kernel's code): what a
/// committed entry is checked against before an export carries it.
#[must_use]
pub fn now_at(index: u32) -> Option<Box<dyn profile::fire::Now>> {
    measured_on_at(index).map(|on| {
        Box::new(engine_gpu::stored::Keys::new(on, kernels_slang::Target::Wgsl))
            as Box<dyn profile::fire::Now>
    })
}

/// The device and driver device `index`'s measurements are keyed as taken
/// on (`Context::measured_on`): what a home's host covers of it name.
#[must_use]
pub fn measured_on_at(index: u32) -> Option<String> {
    let boot = crate::api::DeviceBoot {
        adapter_index: index,
        ..crate::api::DeviceBoot::default()
    };
    let context = crate::device::Context::bind(&boot, engine::Allowance::WHOLE).ok()?;
    Some(context.measured_on())
}

/// [`probe`] of the device `index` names, in the order the engine numbers
/// them (`wgpu:<n>`).
#[must_use]
pub fn probe_at(index: u32) -> Option<(String, Facts)> {
    let boot = crate::api::DeviceBoot {
        adapter_index: index,
        ..crate::api::DeviceBoot::default()
    };
    let context = crate::device::Context::bind(&boot, engine::Allowance::WHOLE).ok()?;
    let device = context.chip();
    Some((device.name.clone(), device.facts()))
}

/// [`resolve`] by the adapter's name and facts.
#[must_use]
pub fn resolve_facts(name: &str, facts: &Facts, profiles: Option<&Path>) -> (Profile, Report) {
    SHELF.resolve(registry(), name, facts, profiles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_gpu::backend::Tactics as _;

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
        let t = crate::tuning::Tuning::new(profile.clone());
        // Before the rates step has measured the workgroups the device holds,
        // the device is sized as one (that step's own tuning); a decode then
        // splits to the workgroups its profile states it holds.
        assert_eq!(
            (
                t.sdpa_tile_rows_per_request(),
                t.sdpa_split_groups(),
                t.sdpa_split_min_keys()
            ),
            (32, 1, 256)
        );
        let measured = profile.with_tuned(profile::fire::Tuned {
            rates: profile::fire::Rates {
                resident_workgroups: Some(47.6),
                ..profile.rates()
            },
            ..profile.tuned().clone()
        });
        let sized = crate::tuning::Tuning::new(measured);
        assert_eq!(
            (sized.resident(), sized.sdpa_split_groups()),
            (Some(48), 48)
        );
        use crate::tuning::Qmm;
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
