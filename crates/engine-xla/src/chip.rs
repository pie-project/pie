//! The TPU's profile: the XLA registry, what a TPU of each device kind is
//! (its facts), the checked-in profiles, and resolution for one load, by
//! the device kind PJRT reports.
//!
//! XLA ships no fused kernels (`crate::FUSED` is empty), so the registry
//! holds no `fuse.*` flag. See `.wiki/device-profiles.md`.

use std::path::Path;
use std::sync::LazyLock;

use profile::{Decision, Domain, Facts, Profile, Registry, Report, Shelf, Var};

/// What the XLA engine knows about its device, for `target.assumes`: read
/// from [`Tpu::of`] by device kind, since PJRT reports no sizes.
pub const FACTS: &[Var] = &[
    Var::fixed("generation", Domain::COUNT),
    Var::fixed("hbm_gib", Domain::COUNT),
    Var::fixed("vmem_mib", Domain::COUNT),
    Var::fixed("mxu", Domain::COUNT),
];

/// One TPU generation's sizes, per chip (one PJRT device).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tpu {
    pub generation: u32,
    pub hbm_gib: u32,
    pub vmem_mib: u32,
    /// The MXU's side.
    pub mxu: u32,
}

impl Tpu {
    /// A v6e: what every default was measured on, and what a kind this
    /// table does not know is taken for (as every load was before).
    pub const V6E: Tpu = Tpu {
        generation: 6,
        hbm_gib: 32,
        vmem_mib: 128,
        mxu: 256,
    };

    /// The TPU PJRT names `kind` (`TPU v6 lite`, `TPU v5 lite`, `TPU v5`,
    /// `TPU v4`). The v6e and v5e rows are the ones pie has run on; the v4
    /// and v5p VMEM sizes are public figures not measured here, taken low
    /// where sources disagree, since a VMEM figure too large fails to compile
    /// and one too small only cuts blocks finer.
    #[must_use]
    pub fn of(kind: &str) -> Tpu {
        let k = kind.to_ascii_lowercase();
        let k = k.trim();
        if k.starts_with("tpu v6") {
            Tpu::V6E
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
            Tpu::V6E
        }
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

    /// Whether the MXU multiplies fp8 natively (from v7).
    #[must_use]
    pub fn fp8_native(&self) -> bool {
        self.generation >= 7
    }

    /// Device memory a load leaves free past its weights and pool: 2 GiB of
    /// a v6e's 32, in proportion elsewhere.
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
    // The compiler's own defaults, two side streams included: what it
    // priced XLA's regions by before the profile held them.
    poem_compiler::costs::register(&mut r, 2);
    r
});

/// Every decision the XLA engine exposes.
#[must_use]
pub fn registry() -> &'static Registry {
    &REGISTRY
}

/// The compiler's prices for a load under `profile`. XLA has no
/// multiprocessor count; the compiler's default 132 keeps the plan every
/// load compiled before.
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
    /// Today's shapes: powers of two, 128 rows of room, mixed fires' host
    /// at 16 lanes, tables at 16 pages at least.
    pub const DEFAULT: Pad = Pad {
        growth: 200,
        room: 128,
        mixed_host_rung: 16,
        page_floor: 16,
    };

    /// The rung after `r`: `growth` percent of it, rounded up, and one more
    /// at least.
    #[must_use]
    pub fn next(&self, r: u32) -> u32 {
        let up = (u64::from(r) * u64::from(self.growth)).div_ceil(100);
        u32::try_from(up).unwrap_or(u32::MAX).max(r.saturating_add(1))
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

seeded!("tpu-v6-lite.toml");

/// Where XLA's profiles live.
pub static SHELF: Shelf = Shelf {
    backend: "xla",
    seeded: SEEDED,
};

/// Resolve the profile one load on a device of `kind` runs under: the
/// user's file, then the checked-in one, then the defaults (which are the
/// v6e's), against the facts of the TPU that kind names.
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
    fn every_checked_in_profile_reads_strictly_and_is_filed_under_its_device() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("profiles");
        SHELF
            .check(registry(), &dir)
            .unwrap_or_else(|e| panic!("{e}"));
    }

    #[test]
    fn a_v6e_runs_its_file_and_the_file_is_what_it_ran_before() {
        let (profile, report) = resolve("TPU v6 lite", None);
        assert_eq!(
            report.used,
            profile::Source::Seeded("tpu-v6-lite.toml".into())
        );
        assert_eq!(
            kernels_xla::tuning::Tuning::on(&profile, Tpu::of("TPU v6 lite").vmem_bytes()),
            kernels_xla::tuning::Tuning::DEFAULT
        );
        assert_eq!(Engine::of(&profile), Engine::of(&registry().defaults()));
        assert_eq!(Engine::of(&profile).pad, Pad::DEFAULT);
        assert_eq!(compiler_profile(&profile), poem_compiler::DeviceProfile::default());
        let (_, report) = resolve("TPU v5 lite", None);
        assert_eq!(report.used, profile::Source::Defaults);
    }

    #[test]
    fn an_unknown_kind_is_taken_for_a_v6e_and_sizes_scale_with_the_chip() {
        assert_eq!(Tpu::of("TPU v9 ultra"), Tpu::V6E);
        assert_eq!(Tpu::of(""), Tpu::V6E);
        assert_eq!(Tpu::V6E.headroom(), 2 << 30);
        assert_eq!(Tpu::of("TPU v5 lite").headroom(), 1 << 30);
        assert!(!Tpu::V6E.fp8_native());
        let v4 = kernels_xla::tuning::Tuning {
            vmem_bytes: Tpu::of("TPU v4").vmem_bytes(),
            ..kernels_xla::tuning::Tuning::DEFAULT
        };
        assert!(v4.vmem_cap() < 16 << 20, "a v4 never asks past its VMEM");
    }

    #[test]
    fn the_default_padding_is_powers_of_two() {
        let pad = Pad::DEFAULT;
        for n in 0..5000u32 {
            assert_eq!(pad.rung(n), n.max(1).next_power_of_two(), "{n}");
        }
        assert_eq!(pad.ladder(2048), vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]);
        assert_eq!(pad.ladder(100), vec![1, 2, 4, 8, 16, 32, 64, 100]);
        let finer = Pad { growth: 150, ..pad };
        assert_eq!(finer.ladder(40), vec![1, 2, 3, 5, 8, 12, 18, 27, 40]);
    }
}

