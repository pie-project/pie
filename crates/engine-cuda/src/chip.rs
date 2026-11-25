//! The GPU's profile: the CUDA registry, the facts this engine measures, the
//! checked-in profiles, and resolution for one load. A ROCm build reads the
//! same registry against AMD's chips, keyed by their gfx version.
#![allow(unsafe_code)]

use std::path::Path;
use std::sync::LazyLock;

use profile::{Anywhere, Decision, Domain, Facts, Profile, Registry, Report, Shelf, Var};

/// The fact a family file assumes: CUDA's compute capability as major * 10
/// + minor (89 is sm_89), or AMD's gfx version the same way (94 is gfx94x).
#[cfg(not(feature = "rocm"))]
pub const ARCH: &str = "cc";
#[cfg(feature = "rocm")]
pub const ARCH: &str = "gfx";

/// What this engine measures about its GPU.
pub const FACTS: &[Var] = &[
    Var::fixed(ARCH, Domain::COUNT),
    Var::fixed("sms", Domain::COUNT),
];

pub const AXES: &[&str] = &["rows"];

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
#[decision(id = "engine.lattice_steps", constant(set(1, 2)), default = 1, tune = engine)]
pub struct LatticeSteps(pub u32);

/// Whether a windowed pool of short lanes takes a page a lane past its kv
/// pages (`Paging::spare`), for a card whose ring, not its kv pages, runs
/// dry first. Off cuts it 1:1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.window_spare", flag, default = false, tune = engine)]
pub struct WindowSpare;

/// Prompt rows a lane brings to a fire over chunked gated delta layers of `H`
/// value heads when `max_forward_tokens` is unset: a fire takes at most
/// `floor(SMs / H)` such lanes, so the walk runs in one wave. 0: the fire top.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.walk_fire_rows", constant(range(0..=65536)), default = 0, tune = engine)]
pub struct WalkFireRows(pub u32);
/// The most rows a fire takes when `max_forward_tokens` is unset, below the
/// fixed 8192 even over sparse experts: a batch's prompts then end over two
/// fires, and those that finish in the first are answered, and replaced,
/// while the second runs. 0: the fire width above.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.fire_rows", constant(range(0..=65536)), default = 0, tune = engine)]
pub struct FireRows(pub u32);
/// The most lanes a fire of prompts alone (no decoding lane) carries and pads
/// its graph to, its lanes' state walk and readout rows with them, so a load
/// seating many decoding lanes does not pad its prompt fires to all of them;
/// a prompt fire of more lanes walks eagerly. 0: every lane the load seats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.prefill_lanes", constant(range(0..=65536)), default = 0, tune = engine)]
pub struct PrefillLanes(pub u32);

/// The most lanes one fire carries, whatever the load seats: the runtime
/// fires a wave of more as fires of this many, each a recorded body. Wider
/// decode bodies armed beside the rest slowed every graph's launches (an
/// L40S's Qwen3-30B-A3B-NVFP4 at c1: some 290 ns a node, 3.6% of its
/// decode, with decode bodies past 256 lanes armed). 0: every lane the load
/// seats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.fire_lanes", constant(range(0..=65536)), default = 256, tune = engine)]
pub struct FireLanes(pub u32);
/// Whether a plane declared `host` (read only by row gathers) lives in pinned,
/// device-mapped host memory, leaving device bytes to KV pages and state slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.host_planes", flag, default = false, tune = engine)]
pub struct HostPlanes;

/// Rows from which the default bucket lattice takes
/// `engine.prefill_lattice_steps` buckets a doubling whatever
/// `engine.lattice_steps` says (a prefill's GEMMs run its bucket's rows); 0
/// never. Past a GEMM's ridge, where its rows cost compute rather than the
/// weights' read, every padded row costs as much as a real one; the ridge
/// (peak flops over memory bandwidth) is some 150 to 450 rows on the cards
/// CUDA serves, so the default starts there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.prefill_lattice_from", constant(range(0..=65536)), default = 256, tune = engine)]
pub struct PrefillLatticeFrom(pub u32);

/// Buckets a doubling the default lattice takes past `engine.prefill_lattice_from`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.prefill_lattice_steps", constant(set(4, 8)), default = 4, tune = engine)]
pub struct PrefillLatticeSteps(pub u32);

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("cuda", FACTS, AXES);
    kernels_cuda::tuning::register(&mut r);
    r.add::<Copies>()
        .add::<Grouped>()
        .add::<WorkingSetPercent>()
        .add::<LatticeSteps>()
        .add::<WindowSpare>()
        .add::<WalkFireRows>()
        .add::<FireRows>()
        .add::<PrefillLanes>()
        .add::<HostPlanes>()
        .add::<PrefillLatticeFrom>()
        .add::<PrefillLatticeSteps>()
        .add::<FireLanes>();
    poem_compiler::fuse::flags::register(&mut r, &crate::FUSIONS);
    // The residual norms run as generated row kernels where they are on
    // (`residual_norms` then forms no hand residual norm).
    #[cfg(not(feature = "rocm"))]
    poem_compiler::costs::register_regions(&mut r, 2);
    // HIP graphs have no conditional node: one is priced past any launch it
    // could skip, so the compiler lowers every region to `AlwaysLaunch`. The
    // residual norms run as generated row kernels there too; the products'
    // epilogues run on their hand rules (`FUSIONS`), ROCm holding no
    // contraction skeletons.
    #[cfg(feature = "rocm")]
    poem_compiler::costs::register_regions_for(
        &mut r,
        2,
        poem_compiler::costs::Conditionals::Absent,
        false,
    );
    r
});

#[must_use]
pub fn registry() -> &'static Registry {
    &REGISTRY
}

/// The checked-in profiles, CUDA's and AMD's side by side in `profiles/`:
/// a build seeds its own, and the other's are the files beside them.
macro_rules! shelved {
    (cuda: [$($cuda:literal),* $(,)?], rocm: [$($rocm:literal),* $(,)?] $(,)?) => {
        #[cfg(not(feature = "rocm"))]
        pub const SEEDED: &[(&str, &str)] =
            &[$(($cuda, include_str!(concat!("../profiles/", $cuda)))),*];
        #[cfg(all(test, not(feature = "rocm")))]
        const BESIDE: &[&str] = &[$($rocm),*];
        #[cfg(feature = "rocm")]
        pub const SEEDED: &[(&str, &str)] =
            &[$(($rocm, include_str!(concat!("../profiles/", $rocm)))),*];
        #[cfg(all(test, feature = "rocm"))]
        const BESIDE: &[&str] = &[$($cuda),*];
    };
}

shelved!(
    cuda: [
        "sm80.toml",
        "sm80/nvidia-a100.toml",
        "sm86.toml",
        "sm86/nvidia-a40.toml",
        "sm87.toml",
        "sm89.toml",
        "sm89/nvidia-geforce-rtx-4090.toml",
        "sm89/nvidia-l4.toml",
        "sm89/nvidia-l40s.toml",
        "sm90.toml",
        "sm90/nvidia-h100-80gb-hbm3.toml",
        "sm90/nvidia-h100-pcie.toml",
        "sm90/nvidia-h200.toml",
        "sm100.toml",
        "sm100/nvidia-b200.toml",
        "sm103.toml",
        "sm110.toml",
        "sm120.toml",
        "sm120/nvidia-geforce-rtx-5090.toml",
        "sm120/nvidia-rtx-pro-4500-blackwell.toml",
        "sm120/nvidia-rtx-pro-6000-blackwell-server-edition.toml",
        "sm121.toml",
        // Committed measured profiles (`pie tune export --engine cuda`).
        "nvidia-l40s.measured.toml",
    ],
    rocm: [
        "gfx94.toml",
        "gfx94/amd-instinct-mi300x.toml",
        // Committed measured profiles (`pie tune export --engine cuda`).
        "amd-instinct-mi300x-vf.measured.toml",
    ],
);

/// The host covers committed for this backend's devices
/// (`profiles/<device>.covers.json`, `pie tune export --engine cuda`), each
/// listed here: the runtime takes a cover whose key matches where the
/// home's own cache holds none (`engine::Engine::committed_covers`).
pub const COVERS: &[&str] = &[include_str!("../profiles/nvidia-l40s.covers.json")];

pub static SHELF: Shelf = Shelf {
    backend: "cuda",
    seeded: SEEDED,
    derived: None,
};

#[must_use]
pub fn facts(major: u32, minor: u32, sms: u32) -> Facts {
    Facts::of(registry())
        .int(ARCH, i64::from(major * 10 + minor))
        .int("sms", i64::from(sms))
}

/// The settings one load on this GPU runs under, read from its profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Engine {
    pub copies: bool,
    pub grouped: bool,
    pub working_set_percent: u32,
    pub lattice_steps: u32,
    pub window_spare: bool,
    pub walk_fire_rows: u32,
    pub fire_rows: u32,
    pub prefill_lanes: u32,
    pub host_planes: bool,
    pub prefill_lattice_from: u32,
    pub prefill_lattice_steps: u32,
    pub fire_lanes: u32,
}

impl Engine {
    #[must_use]
    pub fn of(profile: &Profile) -> Engine {
        Engine {
            copies: profile.on::<Copies>(&Anywhere),
            grouped: profile.on::<Grouped>(&Anywhere),
            working_set_percent: profile.get::<WorkingSetPercent>().0,
            lattice_steps: profile.get::<LatticeSteps>().0,
            window_spare: profile.on::<WindowSpare>(&Anywhere),
            walk_fire_rows: profile.get::<WalkFireRows>().0,
            fire_rows: profile.get::<FireRows>().0,
            prefill_lanes: profile.get::<PrefillLanes>().0,
            host_planes: profile.on::<HostPlanes>(&Anywhere),
            prefill_lattice_from: profile.get::<PrefillLatticeFrom>().0,
            prefill_lattice_steps: profile.get::<PrefillLatticeSteps>().0,
            fire_lanes: profile.get::<FireLanes>().0,
        }
    }

    /// The lanes one fire of a load seating `max_lanes` carries
    /// (`Shell::fire_lanes`): every one, or `engine.fire_lanes` where fewer.
    #[must_use]
    pub fn fire_lanes_of(&self, max_lanes: u32) -> u32 {
        match self.fire_lanes {
            0 => max_lanes,
            cap => max_lanes.min(cap),
        }
    }
}

/// The name and facts of GPU `ordinal`, read without opening a context:
/// for `pie profile` and doctor. `None` without a CUDA runtime or device.
#[must_use]
pub fn probe(ordinal: i32) -> Option<(String, Facts)> {
    #[cfg(feature = "rocm")]
    {
        use crate::device::rt;
        if !crate::device::present() {
            return None;
        }
        let attribute = |which| {
            let mut value = 0i32;
            // SAFETY: a live out-parameter.
            let status = unsafe { rt::cudaDeviceGetAttribute(&raw mut value, which, ordinal) };
            (status == rt::cudaError::cudaSuccess)
                .then_some(value)
                .and_then(|v| u32::try_from(v).ok())
        };
        let name = kernels_cuda::hip::device_name(u32::try_from(ordinal).ok()?)?;
        let facts = facts(
            attribute(rt::cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor)?,
            attribute(rt::cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor)?,
            attribute(rt::cudaDeviceAttr::cudaDevAttrMultiProcessorCount)?,
        );
        Some((name, facts))
    }
    #[cfg(feature = "cuda")]
    {
        use crate::device::rt;
        // A box with no CUDA runtime has no GPU to name: cudarc panics on the
        // first call that loads the library, which `present` catches.
        if !crate::device::present() {
            return None;
        }
        // SAFETY: `prop` is a live out-parameter the call fills; an all-zero
        // `cudaDeviceProp` is a valid value of the plain C struct.
        let mut prop: rt::cudaDeviceProp = unsafe { core::mem::zeroed() };
        // SAFETY: as above.
        let status = unsafe { rt::cudaGetDeviceProperties(&raw mut prop, ordinal) };
        if status != rt::cudaError::cudaSuccess {
            return None;
        }
        let bytes: Vec<u8> = prop
            .name
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        let name = String::from_utf8_lossy(&bytes).into_owned();
        let facts = facts(
            u32::try_from(prop.major).ok()?,
            u32::try_from(prop.minor).ok()?,
            u32::try_from(prop.multiProcessorCount).ok()?,
        );
        Some((name, facts))
    }
    #[cfg(not(feature = "device"))]
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

/// Resolve the profile one load runs under: its compute capability's
/// family file, the file for this GPU, then the user's, each as it states
/// itself (no stored measurement checked against a build): what `pie tune`
/// and `pie profile` read.
#[must_use]
pub fn resolve(device: &str, facts: &Facts, profiles: Option<&Path>) -> (Profile, Report) {
    SHELF.resolve(registry(), device, facts, profiles)
}

/// [`resolve`] for a load: also the device's committed measured profile
/// (`<device>.measured.toml`), and each stored measurement, the committed
/// profile's and the user's, taken where its key matches this build on this
/// device (`keys`) and dropped to be measured again where it does not, each
/// drop said in the load log.
#[must_use]
pub fn resolve_keyed(
    device: &str,
    facts: &Facts,
    profiles: Option<&Path>,
    keys: &crate::stored::Keys,
) -> (Profile, Report) {
    let (profile, report) = SHELF.resolve_keyed(registry(), device, facts, profiles, keys);
    for line in &report.stored {
        tracing::info!("{line}");
    }
    (profile, report)
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
            .check_beside(registry(), &dir, BESIDE)
            .unwrap_or_else(|e| panic!("{e}"));
        let family = |n: &str| !n.contains('/') && !n.ends_with(".measured.toml");
        for (name, text) in SEEDED.iter().filter(|(n, _)| family(n)) {
            let file = registry()
                .parse(text, name, profile::Strict::CheckedIn)
                .expect("checked above");
            let cc = file
                .target
                .assumes
                .iter()
                .find(|(i, _)| FACTS[*i].name == ARCH)
                .map(|(_, cc)| *cc)
                .expect("a family file assumes its compute capability");
            let tier = if cfg!(feature = "rocm") {
                format!("gfx{cc}")
            } else {
                format!("sm_{cc}")
            };
            assert_eq!(
                file.target.tier.as_deref(),
                Some(tier.as_str()),
                "{name}: the tier is the capability it assumes"
            );
        }
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn every_checked_in_profile_dumps_to_a_file_that_reads_back_strictly() {
        SHELF
            .check_dumps(registry())
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

#[cfg(all(test, feature = "rocm"))]
mod hip {
    use super::*;

    #[test]
    fn a_cdna3_gpu_takes_its_family() {
        // The MI300X (the card and a cloud VF of it) reads its rates from its
        // device file; the MI325X has none yet. The VF also carries its
        // committed measured profile, under the device files.
        for (gpu, device, committed) in [
            ("AMD Instinct MI300X", true, false),
            ("AMD Instinct MI300X VF", true, true),
            ("AMD Instinct MI325X", false, false),
        ] {
            let (profile, report) = resolve(gpu, &facts(9, 4, 304), None);
            let mut layers = vec![profile::Source::Seeded("gfx94.toml".into())];
            if device {
                layers.push(profile::Source::Seeded(
                    "gfx94/amd-instinct-mi300x.toml".into(),
                ));
            }
            if committed {
                layers.push(profile::Source::Committed(
                    "amd-instinct-mi300x-vf.measured.toml".into(),
                ));
            }
            assert_eq!(report.layers, layers, "{gpu}");
            assert!(report.untuned(), "{gpu}");
            assert_eq!(
                Engine::of(&profile),
                Engine {
                    prefill_lattice_from: 2048,
                    prefill_lattice_steps: 8,
                    ..Engine::of(&registry().defaults())
                }
            );
            // The defaults, but for the paths the family measured.
            let mut want = kernels_cuda::tuning::Tuning::of(&registry().defaults());
            want.mla_absorb_q_until = 2048;
            want.mla_absorb_out_from = 33;
            want.mla_absorb_out_until = 1536;
            want.fp8_decode_until = 32;
            want.fp8_narrow_until = 128;
            want.moe_lt_grouped = true;
            want.moe_lt_mlp_from = 128;
            want.moe_lt_tile_deep = 256_192_064;
            want.moe_lt_tile_shallow = 128_288_032;
            want.moe_lt_deep_k = 2048;
            want.moe_ordered_from = 128;
            want.gemm_live_from = 129;
            want.gemm_turn_from = 3;
            want.gemm_turn_wide_until = 32;
            want.ssm_gdn_wide_from = 512;
            // The MI300X's own file measured one prefill block a CU; the
            // MI325X, with none, keeps the default.
            if device {
                want.prefill_ctas_per_sm = 1;
            }
            // The rates are the device's own file's (the norm fold's pricing
            // reads them), not the family's: not what this checks.
            let got = kernels_cuda::tuning::Tuning::of(&profile);
            want.launch_ns = got.launch_ns;
            want.hbm_gbps = got.hbm_gbps;
            want.cached_gbps = got.cached_gbps;
            assert_eq!(got, want, "{gpu}");
        }
    }

    #[test]
    fn a_gfx_version_with_no_family_runs_the_defaults_untuned() {
        let (profile, report) = resolve("AMD Instinct Next", &facts(9, 6, 256), None);
        assert_eq!(report.used, profile::Source::Defaults);
        assert!(report.layers.is_empty());
        assert!(report.untuned());
        assert_eq!(
            kernels_cuda::tuning::Tuning::of(&profile),
            kernels_cuda::tuning::Tuning::of(&registry().defaults())
        );
    }

    #[test]
    fn the_defaults_price_conditionals_past_any_launch() {
        // HIP graphs have no conditional node: one is priced past any launch
        // it could skip, so the compiler lowers every region to AlwaysLaunch.
        // Chains fold into generated row kernels; no contraction skeleton yet.
        assert_eq!(
            poem_compiler::DeviceProfile::of(&registry().defaults(), 132, &[], &[]),
            poem_compiler::DeviceProfile {
                cond_fixed_us: 1_000_000.0,
                synth_regions: true,
                ..poem_compiler::DeviceProfile::default()
            }
        );
        let on = poem_compiler::fuse::flags::on(&registry().defaults(), &crate::FUSIONS);
        assert_eq!(
            on.len(),
            crate::FUSIONS.len(),
            "every fusion is on by default"
        );
    }
}

#[cfg(all(test, feature = "device"))]
mod device {
    #[test]
    fn the_probe_reads_the_gpu_it_names() {
        let Some((name, facts)) = super::probe(0) else {
            eprintln!("not asked: no device answers");
            return;
        };
        assert!(!name.is_empty(), "the driver names the GPU");
        let cc = facts.get(super::ARCH).expect("probed");
        #[cfg(not(feature = "rocm"))]
        assert!(
            cc >= 80,
            "{name} is sm_{cc}: the CUDA engine serves sm_80 and newer"
        );
        #[cfg(feature = "rocm")]
        assert!(
            (90..100).contains(&cc),
            "{name} is gfx{cc}: the ROCm engine serves CDNA (gfx9)"
        );
        assert!(facts.get("sms").is_some_and(|n| n > 0));
    }
}
