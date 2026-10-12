//! The XLA profile registry, what a TPU of each device kind is, and
//! resolution of one load's profile by the device kind PJRT reports.

use std::path::Path;
use std::sync::LazyLock;

use profile::{Decision, Domain, Facts, Profile, Registry, Report, Shelf, Var};

/// Read from [`Tpu::of`] by device kind, since PJRT reports no sizes.
pub const FACTS: &[Var] = &[
    Var::fixed("generation", Domain::COUNT),
    Var::fixed("hbm_gib", Domain::COUNT),
    Var::fixed("vmem_mib", Domain::COUNT),
    Var::fixed("mxu", Domain::COUNT),
];

/// One TPU generation's sizes, per PJRT device: a chip, or on v7 one of
/// its two chiplets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tpu {
    /// 0 for a kind pie does not know, which no family file assumes.
    pub generation: u32,
    pub hbm_gib: u32,
    pub vmem_mib: u32,
    pub mxu: u32,
}

impl Tpu {
    /// What the defaults were measured on.
    pub const V6E: Tpu = Tpu {
        generation: 6,
        hbm_gib: 32,
        vmem_mib: 128,
        mxu: 256,
    };

    /// Ironwood ("TPU7x" to PJRT), one device per chiplet: half the chip's
    /// 192 GiB, and JAX's `tpu_info` VMEM and MXU figures.
    pub const V7: Tpu = Tpu {
        generation: 7,
        hbm_gib: 96,
        vmem_mib: 64,
        mxu: 256,
    };

    /// A kind pie does not know: no family applies, and the sizes are a
    /// v6e's.
    pub const UNKNOWN: Tpu = Tpu {
        generation: 0,
        ..Tpu::V6E
    };

    /// The TPU PJRT names `kind`. The v4 and v5p VMEM sizes are unmeasured
    /// public figures, taken low: a VMEM figure too large fails to compile,
    /// one too small only cuts blocks finer.
    #[must_use]
    pub fn of(kind: &str) -> Tpu {
        let k = kind.to_ascii_lowercase();
        let k = k.trim();
        if k.starts_with("tpu v6") {
            Tpu::V6E
        } else if k == "tpu7x" || k == "tpu7" {
            Tpu::V7
        } else if k.starts_with("tpu v5 lite") || k.starts_with("tpu v5e") {
            Tpu {
                generation: 5,
                hbm_gib: 16,
                vmem_mib: 128,
                mxu: 128,
            }
        } else if k.starts_with("tpu v5") {
            Tpu {
                generation: 5,
                hbm_gib: 95,
                vmem_mib: 64,
                mxu: 128,
            }
        } else if k.starts_with("tpu v4") {
            Tpu {
                generation: 4,
                hbm_gib: 32,
                vmem_mib: 16,
                mxu: 128,
            }
        } else {
            Tpu::UNKNOWN
        }
    }

    #[must_use]
    pub fn known(&self) -> bool {
        self.generation > 0
    }

    #[must_use]
    pub fn facts(&self) -> Facts {
        Facts::of(registry())
            .int("generation", i64::from(self.generation))
            .int("hbm_gib", i64::from(self.hbm_gib))
            .int("vmem_mib", i64::from(self.vmem_mib))
            .int("mxu", i64::from(self.mxu))
    }

    #[must_use]
    pub fn hbm_bytes(&self) -> u64 {
        u64::from(self.hbm_gib) << 30
    }

    #[must_use]
    pub fn vmem_bytes(&self) -> i64 {
        i64::from(self.vmem_mib) << 20
    }

    #[must_use]
    pub fn fp8_native(&self) -> bool {
        self.generation >= 7
    }

    /// Device memory a load leaves free past its weights and pool.
    #[must_use]
    pub fn headroom(&self) -> u64 {
        self.hbm_bytes() / 16
    }
}

/// Whether an mxfp4 bank lands pre-scaled to e5m2 (twice the bytes, an
/// in-dot decode saved: 0.45 against 0.8 ms on v6e) or as its e2m1 codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "layout.mxfp4_prescale", default = "fits", tune = search)]
pub enum Mxfp4Prescale {
    /// Pre-scaled where the weights still fit the device beside the pool.
    Fits,
    /// Always as codes: the bytes go to kv pages instead.
    Never,
}

/// Whether gemm-only weights land transposed `[K, N]`, which streams at the
/// HBM roofline on v6e where `[N, K]` reaches 1.2-1.4 TB/s
/// (`crate::weights::gemm_only`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "layout.gemm_transpose", flag, default = true, tune = search)]
pub struct GemmTranspose;

/// Whether gather-only sub-byte tables land folded, so a gather does not
/// relayout the whole table (`crate::weights::gather_only`; gpt-oss's
/// embedding: 1.75 ms a step as codes, 0.78 ms folded).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "layout.embed_fold", flag, default = true, tune = search)]
pub struct EmbedFold;

/// How far apart the canonical fire shapes sit, in percent: each row,
/// lane and page rung is this much of the one below (rounded up). 200 is
/// powers of two; fewer programs against less padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.pad.growth", constant(set(150, 200)), default = 200, tune = search)]
pub struct PadGrowth(pub u32);

/// Rows past the fire limit a prefill fire may pad into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.pad.room", constant(range(0..=1024)), default = 128, tune = search)]
pub struct PadRoom(pub u32);

/// The least lane rung of the prefill class in a fire that decodes too, so
/// such fires vary by their decode rung and row rung alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.pad.mixed_host_rung", constant(set(8, 16, 32)), default = 16, tune = search)]
pub struct MixedHostRung(pub u32);

/// The least page bound a fire's tables are laid out to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.pad.page_floor", constant(set(4, 8, 16, 32, 64)), default = 16, tune = search)]
pub struct PageFloor(pub u32);

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("xla", FACTS, &["rows"]);
    kernels_xla::tuning::register(&mut r);
    r.add::<Mxfp4Prescale>()
        .add::<GemmTranspose>()
        .add::<EmbedFold>()
        .add::<PadGrowth>()
        .add::<PadRoom>()
        .add::<MixedHostRung>()
        .add::<PageFloor>();
    poem_compiler::fuse::flags::register(&mut r, &crate::FUSED);
    poem_compiler::costs::register(&mut r, 2);
    r
});

#[must_use]
pub fn registry() -> &'static Registry {
    &REGISTRY
}

/// XLA has no multiprocessor count: the compiler's default 132 stands in.
#[must_use]
pub fn compiler_profile(profile: &Profile) -> poem_compiler::DeviceProfile {
    poem_compiler::DeviceProfile::of(profile, 132, &[], &[])
}

/// The engine's settings one load runs under, read from its profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Engine {
    pub prescale: Mxfp4Prescale,
    pub gemm_transpose: bool,
    pub embed_fold: bool,
    pub pad: Pad,
}

/// How fires are padded to canonical shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pad {
    /// Percent each rung is of the one below.
    pub growth: u32,
    pub room: u32,
    pub mixed_host_rung: u32,
    pub page_floor: u32,
}

impl Pad {
    /// The rung after `r`: `growth` percent of it, rounded up, and one more
    /// at least.
    #[must_use]
    pub fn next(&self, r: u32) -> u32 {
        let up = (u64::from(r) * u64::from(self.growth)).div_ceil(100);
        u32::try_from(up)
            .unwrap_or(u32::MAX)
            .max(r.saturating_add(1))
    }

    /// The least rung at or above `n` (1 for 0): at growth 200, the power
    /// of two at or above it.
    #[must_use]
    pub fn rung(&self, n: u32) -> u32 {
        let mut r = 1;
        while r < n {
            r = self.next(r);
        }
        r
    }

    /// Every rung from 1 below `top`, then `top`.
    #[must_use]
    pub fn ladder(&self, top: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let mut r = 1;
        while r < top {
            out.push(r);
            r = self.next(r);
        }
        out.push(top);
        out
    }
}

impl Engine {
    #[must_use]
    pub fn of(profile: &Profile) -> Engine {
        let any = &profile::Anywhere;
        Engine {
            prescale: *profile.get::<Mxfp4Prescale>(),
            gemm_transpose: profile.on::<GemmTranspose>(any),
            embed_fold: profile.on::<EmbedFold>(any),
            pad: Pad {
                growth: profile.get::<PadGrowth>().0,
                room: profile.get::<PadRoom>().0,
                mixed_host_rung: profile.get::<MixedHostRung>().0,
                page_floor: profile.get::<PageFloor>().0,
            },
        }
    }
}

macro_rules! seeded {
    ($($file:literal),* $(,)?) => {
        /// The profiles checked in with pie, as `(file name, text)`.
        pub const SEEDED: &[(&str, &str)] = &[
            $(($file, include_str!(concat!("../profiles/", $file)))),*
        ];
    };
}

seeded!("v6e.toml", "v7.toml");

pub static SHELF: Shelf = Shelf {
    backend: "xla",
    seeded: SEEDED,
    derived: None,
};

/// The profile a load on a device of `kind` runs under: its generation's
/// family file, then the user's file, over the defaults.
#[must_use]
pub fn resolve(kind: &str, profiles: Option<&Path>) -> (Profile, Report) {
    SHELF.resolve(registry(), kind, &Tpu::of(kind).facts(), profiles)
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
    fn a_v6e_runs_its_file_and_the_file_is_what_it_ran_before() {
        let (profile, report) = resolve("TPU v6 lite", None);
        assert_eq!(report.used, profile::Source::Seeded("v6e.toml".into()));
        assert!(!report.untuned());
        assert_eq!(
            kernels_xla::tuning::Tuning::on(&profile, Tpu::of("TPU v6 lite").vmem_bytes()),
            kernels_xla::tuning::Tuning::DEFAULT
        );
        assert_eq!(Engine::of(&profile), Engine::of(&registry().defaults()));
        assert_eq!(
            Engine::of(&profile).pad,
            Pad {
                growth: 200,
                room: 128,
                mixed_host_rung: 16,
                page_floor: 16,
            }
        );
        assert_eq!(
            compiler_profile(&profile),
            poem_compiler::DeviceProfile::default()
        );
    }

    #[test]
    fn a_kind_names_its_family_and_its_sizes() {
        for (kind, tpu, family) in [
            ("TPU v6 lite", Tpu::V6E, Some("v6e.toml")),
            ("TPU v6e", Tpu::V6E, Some("v6e.toml")),
            ("TPU7x", Tpu::V7, Some("v7.toml")),
            ("TPU v5 lite", Tpu::of("TPU v5e"), None),
            ("TPU v5p", Tpu::of("TPU v5"), None),
            ("TPU v9 ultra", Tpu::UNKNOWN, None),
            ("", Tpu::UNKNOWN, None),
        ] {
            assert_eq!(Tpu::of(kind), tpu, "{kind}");
            let (_, report) = resolve(kind, None);
            let used = family.map_or(profile::Source::Defaults, |f| {
                profile::Source::Seeded(f.into())
            });
            assert_eq!(report.used, used, "{kind}");
            assert_eq!(report.untuned(), kind != "TPU v6 lite" && kind != "TPU v6e");
        }
        assert_eq!(
            Tpu::of("TPU7x"),
            Tpu {
                generation: 7,
                hbm_gib: 96,
                vmem_mib: 64,
                mxu: 256,
            }
        );
        assert!(!Tpu::of("TPU v9 ultra").known());
        assert_eq!(Tpu::of("TPU v9 ultra").vmem_bytes(), Tpu::V6E.vmem_bytes());
    }

    #[test]
    fn sizes_scale_with_the_chip() {
        assert_eq!(Tpu::V6E.headroom(), 2 << 30);
        assert_eq!(Tpu::of("TPU v5 lite").headroom(), 1 << 30);
        assert!(!Tpu::V6E.fp8_native());
        let v4 = kernels_xla::tuning::Tuning {
            vmem_bytes: Tpu::of("TPU v4").vmem_bytes(),
            ..kernels_xla::tuning::Tuning::DEFAULT
        };
        assert!(v4.vmem_cap() < 16 << 20, "a v4 never asks past its VMEM");

        let v7 = Tpu::V7;
        assert!(v7.fp8_native());
        assert_eq!(v7.headroom(), 6 << 30);
        assert_eq!(v7.hbm_bytes() / 8 * 5, 60 << 30, "the dry prescale bound");
        let (profile, _) = resolve("TPU7x", None);
        let tuning = kernels_xla::tuning::Tuning::on(&profile, v7.vmem_bytes());
        assert_eq!(
            tuning,
            kernels_xla::tuning::Tuning {
                vmem_bytes: 64 << 20,
                ..kernels_xla::tuning::Tuning::DEFAULT
            }
        );
        assert_eq!(tuning.vmem_cap(), 55 << 20);
        assert_eq!(tuning.decode_round_bytes(), 16 << 20);
    }

    #[test]
    fn the_default_padding_is_powers_of_two() {
        let pad = Engine::of(&registry().defaults()).pad;
        for n in 0..5000u32 {
            assert_eq!(pad.rung(n), n.max(1).next_power_of_two(), "{n}");
        }
        assert_eq!(
            pad.ladder(2048),
            vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        );
        assert_eq!(pad.ladder(100), vec![1, 2, 4, 8, 16, 32, 64, 100]);
        let finer = Pad { growth: 150, ..pad };
        assert_eq!(finer.ladder(40), vec![1, 2, 3, 5, 8, 12, 18, 27, 40]);
    }
}
