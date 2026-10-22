//! The GPU's profile: the CUDA registry, the facts this engine measures, the
//! checked-in profiles, and resolution for one load. A ROCm build reads the
//! same registry against AMD's chips, keyed by their gfx version.

use std::path::Path;
use std::sync::LazyLock;

use profile::{Anywhere, Decision, Domain, Facts, Profile, Registry, Report, Shelf, Var};

use crate::{Graphs, Recording};

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

/// Whether fires are captured as CUDA graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.graphs", default = "on", tune = engine)]
pub enum GraphMode {
    /// Every fire launches eagerly: for diagnostics, not serving.
    Off,
    /// Fires are shaped for capture but launch eagerly.
    Shaped,
    /// Fires are captured and replayed.
    On,
}

/// How captured fires are kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "engine.recording",
    default = r#"{ use = "bodies", mem_mb = 4096 }"#,
    tune = engine
)]
pub enum RecordingMode {
    Off,
    Shaped,
    /// Recorded bodies are kept, up to `mem_mb` of device memory.
    Bodies {
        #[constant(range(0..=1_048_576), default = 4096)]
        mem_mb: u32,
    },
}

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

/// The lanes of one fire that may capture attention scores: the score slab
/// seats this many lanes' planes, each the whole model's heads by the kv
/// ceiling, and the scheduler holds a fire's capturing lanes to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.scoring_lanes", constant(range(1..=256)), default = 8, tune = engine)]
pub struct ScoringLanes(pub u32);

/// Row buckets per doubling of the default lattice: 1 is powers of two, 2
/// adds a bucket between each, which pads less and records more bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.lattice_steps", constant(set(1, 2)), default = 1, tune = engine)]
pub struct LatticeSteps(pub u32);

/// Recurrent-state slots a lane gets when `[engine] max_state_slots` is unset,
/// so a hybrid trunk seats every lane it batches; 0 keeps the fixed 256.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.state_slots_per_lane", constant(range(0..=8)), default = 0, tune = engine)]
pub struct StateSlotsPerLane(pub u32);

/// Whether a hybrid trunk whose state slots cannot seat every lane at the
/// configured dispatch depth (two frames a lane) asks the runtime to post one
/// frame a lane instead, seating up to twice the lanes. Off keeps the depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.one_frame_when_short", flag, default = false, tune = engine)]
pub struct OneFrameWhenShort;

/// Whether a decoding lane's next pages come before seating a waiting prompt,
/// and a prompt waits rather than suspending a lane
/// (`PoolFacts::seated_first`). Off keeps arrival order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.seated_first", flag, default = false, tune = engine)]
pub struct SeatedFirst;

/// The rows one fire takes when `[engine] max_forward_tokens` is not
/// stated; 0 keeps the fixed 8192. Wider fires batch more prompts' chunks
/// into one pass of the weights, and their activation arena grows with them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.forward_tokens", constant(range(0..=65536)), default = 0, tune = engine)]
pub struct ForwardTokens(pub u32);

/// Prompt rows a lane brings to a fire over chunked gated delta layers of `H`
/// value heads when `max_forward_tokens` is unset: a fire takes at most
/// `floor(SMs / H)` such lanes, so the walk runs in one wave. 0: `forward_tokens`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.walk_fire_rows", constant(range(0..=65536)), default = 0, tune = engine)]
pub struct WalkFireRows(pub u32);
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
    r.add::<GraphMode>()
        .add::<RecordingMode>()
        .add::<Copies>()
        .add::<Grouped>()
        .add::<WorkingSetPercent>()
        .add::<ScoringLanes>()
        .add::<LatticeSteps>()
        .add::<StateSlotsPerLane>()
        .add::<OneFrameWhenShort>()
        .add::<SeatedFirst>()
        .add::<ForwardTokens>()
        .add::<WalkFireRows>()
        .add::<HostPlanes>()
        .add::<PrefillLatticeFrom>()
        .add::<PrefillLatticeSteps>();
    poem_compiler::fuse::flags::register(&mut r, &crate::FUSED);
    #[cfg(not(feature = "rocm"))]
    poem_compiler::costs::register(&mut r, 2);
    // HIP graphs have no conditional node: one is priced past any launch it
    // could skip, so the compiler lowers every region to `AlwaysLaunch`.
    #[cfg(feature = "rocm")]
    poem_compiler::costs::register_for(&mut r, 2, poem_compiler::costs::Conditionals::Absent);
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
        "sm87.toml",
        "sm89.toml",
        "sm89/nvidia-l40s.toml",
        "sm90.toml",
        "sm100.toml",
        "sm100/nvidia-b200.toml",
        "sm103.toml",
        "sm110.toml",
        "sm120.toml",
        "sm121.toml",
    ],
    rocm: ["gfx94.toml"],
);

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
    pub graphs: Graphs,
    pub recording: Recording,
    pub copies: bool,
    pub grouped: bool,
    pub working_set_percent: u32,
    pub scoring_lanes: u32,
    pub lattice_steps: u32,
    pub state_slots_per_lane: u32,
    pub one_frame_when_short: bool,
    pub seated_first: bool,
    pub forward_tokens: u32,
    pub walk_fire_rows: u32,
    pub host_planes: bool,
    pub prefill_lattice_from: u32,
    pub prefill_lattice_steps: u32,
}

impl Engine {
    #[must_use]
    pub fn of(profile: &Profile) -> Engine {
        Engine {
            graphs: match profile.get::<GraphMode>() {
                GraphMode::Off => Graphs::Off,
                GraphMode::Shaped => Graphs::Shaped,
                GraphMode::On => Graphs::On,
            },
            recording: match *profile.get::<RecordingMode>() {
                RecordingMode::Off => Recording::Off,
                RecordingMode::Shaped => Recording::Shaped,
                RecordingMode::Bodies { mem_mb } => Recording::Bodies {
                    mem_megabytes: mem_mb,
                },
            },
            copies: profile.on::<Copies>(&Anywhere),
            grouped: profile.on::<Grouped>(&Anywhere),
            working_set_percent: profile.get::<WorkingSetPercent>().0,
            scoring_lanes: profile.get::<ScoringLanes>().0,
            lattice_steps: profile.get::<LatticeSteps>().0,
            state_slots_per_lane: profile.get::<StateSlotsPerLane>().0,
            one_frame_when_short: profile.on::<OneFrameWhenShort>(&Anywhere),
            seated_first: profile.on::<SeatedFirst>(&Anywhere),
            forward_tokens: profile.get::<ForwardTokens>().0,
            walk_fire_rows: profile.get::<WalkFireRows>().0,
            host_planes: profile.on::<HostPlanes>(&Anywhere),
            prefill_lattice_from: profile.get::<PrefillLatticeFrom>().0,
            prefill_lattice_steps: profile.get::<PrefillLatticeSteps>().0,
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

/// The registry defaults with the graph mode, and the recording mode where
/// given, pinned: for a test or a diagnostic that runs eagerly.
#[must_use]
pub fn pinned(graphs: Graphs, recording: Option<Recording>) -> Profile {
    let mut text = format!(
        "[engine]\ngraphs = \"{}\"\n",
        match graphs {
            Graphs::Off => "off",
            Graphs::Shaped => "shaped",
            Graphs::On => "on",
        }
    );
    match recording {
        None => {}
        Some(Recording::Off) => text.push_str("recording = \"off\"\n"),
        Some(Recording::Shaped) => text.push_str("recording = \"shaped\"\n"),
        Some(Recording::Bodies { mem_megabytes }) => text.push_str(&format!(
            "recording = {{ use = \"bodies\", mem_mb = {mem_megabytes} }}\n"
        )),
    }
    profile(&text)
}

/// Resolve the profile one load runs under: its compute capability's
/// family file, the file for this GPU, then the user's.
#[must_use]
pub fn resolve(device: &str, facts: &Facts, profiles: Option<&Path>) -> (Profile, Report) {
    SHELF.resolve(registry(), device, facts, profiles)
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
        for (name, text) in SEEDED.iter().filter(|(n, _)| !n.contains('/')) {
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
    use crate::DEFAULT_BODIES_MEGABYTES;

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

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn each_capability_takes_its_family_and_from_10_0_the_blackwell_kernels() {
        let defaults = kernels_cuda::tuning::Tuning::of(&registry().defaults());
        for (major, minor, gpu) in [
            (8, 0, "NVIDIA A100-SXM4-80GB"),
            (8, 6, "NVIDIA A10"),
            (8, 7, "Orin"),
            (8, 9, "NVIDIA L4"),
            (9, 0, "NVIDIA H100 80GB HBM3"),
            (10, 0, "NVIDIA GB200"),
            (10, 3, "NVIDIA B300"),
            (11, 0, "NVIDIA Thor"),
            (12, 0, "NVIDIA RTX PRO 6000"),
            (12, 1, "NVIDIA GB10"),
        ] {
            let (profile, report) = resolve(gpu, &facts(major, minor, 100), None);
            assert_eq!(
                report.layers,
                [profile::Source::Seeded(format!("sm{major}{minor}.toml"))],
                "{gpu}"
            );
            assert!(report.untuned(), "{gpu}");
            // Ampere through Blackwell's consumer parts bucket a prefill
            // eight a doubling from 256 rows; the rest keep the defaults.
            let fine = matches!((major, minor), (8, 0 | 6 | 9) | (9, 0) | (12, 0));
            assert_eq!(
                Engine::of(&profile),
                Engine {
                    prefill_lattice_from: if fine {
                        256
                    } else {
                        Engine::of(&registry().defaults()).prefill_lattice_from
                    },
                    prefill_lattice_steps: if fine { 8 } else { 4 },
                    ..Engine::of(&registry().defaults())
                },
                "{gpu}"
            );
            let tuning = kernels_cuda::tuning::Tuning::of(&profile);
            // sm_100 alone runs the tcgen05 prefill and reads a wide
            // group's single-token rows through the prefill kernel; every
            // other chip keeps FA2 and the decode kernel.
            assert_eq!(
                (
                    tuning.prefill_tc5,
                    tuning.decode_grouped_prefill_from,
                    tuning.decode_grouped_prefill_lse,
                ),
                if (major, minor) == (10, 0) {
                    (true, 4, true)
                } else {
                    (false, 0, false)
                },
                "{gpu}"
            );
            if major >= 10 {
                // sm_100 alone reads dense latent attention on the FA2 arms.
                assert!(tuning.gemv_narrow, "{gpu}");
                assert_eq!(tuning.mla_naive, (major, minor) != (10, 0), "{gpu}");
                // B200 alone also runs the tcgen05 selects and GLU epilogues.
                let b200 = (major, minor) == (10, 0);
                assert_eq!(tuning.moe_tc5, b200, "{gpu}");
                assert_eq!(
                    tuning.ssm_gdn_fused_scan,
                    (major, minor) == (10, 0) || (major, minor) == (12, 0),
                    "{gpu}"
                );
                assert_eq!(tuning.ssm_gdn_tc5, b200, "{gpu}");
                assert_eq!(tuning.rows_vec, b200, "{gpu}");
                assert_eq!(tuning.hc_tile, b200, "{gpu}");
                assert_eq!(tuning.ssm_conv_vec4, b200, "{gpu}");
                assert_eq!(tuning.ssm_gdn_out_bf16, b200, "{gpu}");
                assert_eq!(tuning.qg_prologue_until, if b200 { 64 } else { 0 }, "{gpu}");
                assert_eq!(tuning.merge_warp, b200, "{gpu}");
                assert_eq!(tuning.gemm_bias_epilogue, b200, "{gpu}");
                assert_eq!(tuning.decode_xqa, b200, "{gpu}");
                assert_eq!(tuning.pool_mma, major == 10 && minor == 0, "{gpu}");
                assert_eq!(tuning.index_radix, major == 10 && minor == 0, "{gpu}");
                assert_eq!(tuning.mla_sparse, major == 10 && minor == 0, "{gpu}");
                assert_eq!(
                    tuning.dsv4_prefill_from,
                    if major == 10 && minor == 0 { 512 } else { 0 },
                    "{gpu}"
                );
                assert_eq!(
                    tuning.dsv4_decode_until,
                    if major == 10 && minor == 0 { 64 } else { 0 },
                    "{gpu}"
                );
                assert_eq!(tuning.mxfp4_cvt, major == 10 && minor == 0, "{gpu}");
                assert_eq!(
                    tuning.mxfp4_a8_until,
                    if major == 10 && minor == 0 { 48 } else { 0 },
                    "{gpu}"
                );
                assert_eq!(
                    tuning.mxfp4_a8_from,
                    u32::from(major == 10 && minor == 0),
                    "{gpu}"
                );
                assert_eq!(tuning.fp8_o_down, major == 10 && minor == 0, "{gpu}");
                assert_eq!(
                    tuning.gemm_fp8_mma_gemv_until,
                    if major == 10 && minor == 0 { 8 } else { 0 },
                    "{gpu}"
                );
                assert_eq!(
                    tuning.gemm_fp8_mma_k_until,
                    if major == 10 && minor == 0 { 5120 } else { 0 },
                    "{gpu}"
                );
                assert_eq!(
                    tuning.mxfp4_cvt_ksplit_until,
                    if major == 10 && minor == 0 { 6 } else { 0 },
                    "{gpu}"
                );
                assert_eq!(
                    tuning.decode_xqa_from_64,
                    if b200 { 16 } else { 0 },
                    "{gpu}"
                );
                assert_eq!(
                    tuning.decode_xqa_whole_window,
                    if b200 { 128 } else { 0 },
                    "{gpu}"
                );
                assert_eq!(tuning.decode_xqa_cluster, if b200 { 8 } else { 0 }, "{gpu}");
                assert_eq!(tuning.decode_xqa_deep_64, b200, "{gpu}");
                // B200 alone runs Gemma 4's q norm a warp a head and its
                // norm-and-fold 128 threads a row, skipping unread rows.
                assert_eq!(tuning.q_norm_rope_warp, b200, "{gpu}");
                assert_eq!(tuning.skip_unread, b200, "{gpu}");
                assert_eq!(tuning.norm_fold_block, if b200 { 128 } else { 0 }, "{gpu}");
                assert_eq!(
                    kernels_cuda::tuning::Tuning {
                        gemv_narrow: false,
                        mla_naive: false,
                        moe_tc5: defaults.moe_tc5,
                        ssm_gdn_fused_scan: false,
                        ssm_gdn_tc5: false,
                        rows_vec: false,
                        hc_tile: false,
                        ssm_conv_vec4: false,
                        ssm_gdn_out_bf16: false,
                        qg_prologue_until: 0,
                        dsv4_decode_until: 0,
                        merge_warp: false,
                        mxfp4_mma_until: defaults.mxfp4_mma_until,
                        mxfp4_mma_from: defaults.mxfp4_mma_from,
                        prefill_tc5: false,
                        prefill_wide_sliding: false,
                        decode_grouped_prefill_from: 0,
                        gemm_bias_epilogue: false,
                        moe_w4a4_fused_from: 65_536,
                        moe_w4a4_fused_layout: defaults.moe_w4a4_fused_layout,
                        moe_even_waves_from: defaults.moe_even_waves_from,
                        moe_routes_sized: defaults.moe_routes_sized,
                        moe_gather_ahead_until: defaults.moe_gather_ahead_until,
                        moe_w4a8_fused_from: 65_536,
                        mla_expanded_from: 65_536,
                        gemm_fp8_gemv_until: 0,
                        attn_res_fused_norm: false,
                        ssm_kda_chunked: false,
                        ssm_kda_step_fused: false,
                        gemm_split_launch_ns: 0,
                        gemm_split_launch_from: 0,
                        gemm_tc5_raster: 8,
                        pdl: false,
                        gemm_rows_m: 0,
                        gemm_tc5_split_tickets: 0,
                        nvfp4_mma_from: defaults.nvfp4_mma_from,
                        nvfp4_mma_until: defaults.nvfp4_mma_until,
                        decode_grouped_prefill_lse: false,
                        decode_xqa: false,
                        decode_xqa_from_64: 0,
                        decode_xqa_whole_window: 0,
                        decode_xqa_cluster: 0,
                        decode_xqa_deep_64: false,
                        pool_mma: false,
                        dsv4_prefill_from: 0,
                        index_radix: false,
                        mla_sparse: false,
                        mla_dense_short: false,
                        dsv4_replay: defaults.dsv4_replay,
                        dsv4_fp4_pass_from: defaults.dsv4_fp4_pass_from,
                        gemm_fp8_grouped_tc5_until: defaults.gemm_fp8_grouped_tc5_until,
                        mxfp4_cvt: false,
                        mxfp4_a8_until: 0,
                        mxfp4_a8_from: 0,
                        fp8_o_down: false,
                        gemm_fp8_mma_gemv_until: 0,
                        gemm_fp8_mma_k_until: 0,
                        mxfp4_cvt_ksplit_until: 0,
                        hc_mma: false,
                        index_tc5: false,
                        norm_emit: false,
                        qkv_rows_from: defaults.qkv_rows_from,
                        q_norm_rope_warp: false,
                        skip_unread: false,
                        norm_fold_block: 0,
                        ..tuning
                    },
                    defaults,
                    "{gpu}: only the Blackwell kernels differ"
                );
            } else if major == 9 {
                assert_eq!(
                    kernels_cuda::tuning::Tuning {
                        pdl: false,
                        ssm_gdn_fused_scan: false,
                        ssm_kda_chunked: false,
                        ssm_gdn_out_bf16: false,
                        ssm_conv_vec4: false,
                        rows_vec: false,
                        hc_tile: false,
                        hc_mma: false,
                        qg_prologue_until: 0,
                        merge_warp: false,
                        pool_mma: false,
                        dsv4_decode_until: 0,
                        gemm_bias_epilogue: false,
                        gemm_rows_m: 0,
                        attn_res_fused_norm: false,
                        mxfp4_mma_from: defaults.mxfp4_mma_from,
                        mxfp4_mma_until: defaults.mxfp4_mma_until,
                        nvfp4_mma_from: defaults.nvfp4_mma_from,
                        nvfp4_mma_until: defaults.nvfp4_mma_until,
                        bf16_mma_from: defaults.bf16_mma_from,
                        bf16_mma_until: defaults.bf16_mma_until,
                        selected_mma: defaults.selected_mma,
                        gemm_fp8_gemv_until: defaults.gemm_fp8_gemv_until,
                        mla_dense_short: defaults.mla_dense_short,
                        gemm_skinny_rows: defaults.gemm_skinny_rows,
                        ..tuning
                    },
                    defaults,
                    "{gpu}: Hopper takes the arch-generic arms alone"
                );
                assert!(
                    tuning.ssm_gdn_fused_scan && tuning.pool_mma && tuning.pdl,
                    "{gpu}"
                );
                assert_eq!(
                    (tuning.mxfp4_mma_from, tuning.mxfp4_mma_until),
                    (28, 16_777_216)
                );
                assert_eq!((tuning.bf16_mma_from, tuning.bf16_mma_until), (16, 65536));
            } else {
                // The 99 KiB and A100 families take the two-launch gated
                // delta scan and a prefill's MXFP4 selects on mma.sync
                // alone; Orin keeps the three wmma launches and the GEMV.
                assert_eq!(
                    kernels_cuda::tuning::Tuning {
                        ssm_gdn_fused_scan: false,
                        mxfp4_mma_from: defaults.mxfp4_mma_from,
                        mxfp4_mma_until: defaults.mxfp4_mma_until,
                        mxfp4_mma_per_from: defaults.mxfp4_mma_per_from,
                        ..tuning
                    },
                    defaults,
                    "{gpu}"
                );
                assert_eq!(tuning.ssm_gdn_fused_scan, minor != 7, "{gpu}");
                assert_eq!(
                    tuning.mxfp4_mma_until,
                    if minor == 7 { 0 } else { 16_777_216 },
                    "{gpu}"
                );
            }
        }
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn a_capability_with_no_family_runs_the_defaults_untuned() {
        let (profile, report) = resolve("NVIDIA Next", &facts(13, 0, 200), None);
        assert_eq!(report.used, profile::Source::Defaults);
        assert!(report.layers.is_empty());
        assert!(report.untuned());
        assert_eq!(
            kernels_cuda::tuning::Tuning::of(&profile),
            kernels_cuda::tuning::Tuning::of(&registry().defaults())
        );
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn the_defaults_are_what_cuda_ran_before_profiles() {
        let engine = Engine::of(&registry().defaults());
        assert_eq!(
            engine,
            Engine {
                graphs: Graphs::default(),
                recording: Recording::Bodies {
                    mem_megabytes: DEFAULT_BODIES_MEGABYTES
                },
                copies: true,
                grouped: true,
                working_set_percent: 50,
                scoring_lanes: 8,
                lattice_steps: 1,
                state_slots_per_lane: 0,
                one_frame_when_short: false,
                seated_first: false,
                forward_tokens: 0,
                walk_fire_rows: 0,
                host_planes: false,
                prefill_lattice_from: 256,
                prefill_lattice_steps: 4,
            }
        );
        assert_eq!(
            poem_compiler::DeviceProfile::of(&registry().defaults(), 132, &[], &[]),
            poem_compiler::DeviceProfile::default()
        );
        let mut fused = poem_compiler::fuse::flags::kernels(&registry().defaults());
        let mut shipped = crate::FUSED.to_vec();
        fused.sort_unstable();
        shipped.sort_unstable();
        assert_eq!(fused, shipped);
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn the_b200_runs_its_family_measured_with_the_skinny_gemm_off() {
        let (b200, report) = resolve("NVIDIA B200", &facts(10, 0, 148), None);
        assert_eq!(
            report.layers,
            [
                profile::Source::Seeded("sm100.toml".into()),
                profile::Source::Seeded("sm100/nvidia-b200.toml".into())
            ]
        );
        assert!(!report.untuned());
        let tuning = kernels_cuda::tuning::Tuning::of(&b200);
        assert_eq!(
            tuning.gemm_skinny_rows, 0,
            "cuBLAS beats the skinny GEMM on a B200"
        );
        let (family, _) = resolve("NVIDIA GB200", &facts(10, 0, 152), None);
        assert_eq!(
            kernels_cuda::tuning::Tuning {
                gemm_skinny_rows: kernels_cuda::tuning::Tuning::of(&family).gemm_skinny_rows,
                selected_mma: false,
                index_mma: false,
                gemm_narrow_n: 0,
                ..tuning
            },
            kernels_cuda::tuning::Tuning::of(&family),
            "only the skinny GEMM's reach and the two sparse-attention walks differ from \
             the family"
        );
        assert!(
            tuning.selected_mma,
            "a B200 walks a selection on the tensor cores"
        );
        assert_eq!(Engine::of(&b200).state_slots_per_lane, 2);
        assert!(Engine::of(&b200).one_frame_when_short);
        assert!(Engine::of(&b200).seated_first);
        assert_eq!(Engine::of(&b200).forward_tokens, 16384);
        assert_eq!(Engine::of(&b200).walk_fire_rows, 4096);
        assert!(Engine::of(&b200).host_planes);
        assert_eq!(Engine::of(&b200).lattice_steps, 2);
        assert_eq!(Engine::of(&b200).prefill_lattice_from, 4096);
        assert_eq!(
            Engine {
                lattice_steps: 1,
                prefill_lattice_from: Engine::of(&family).prefill_lattice_from,
                state_slots_per_lane: 0,
                one_frame_when_short: false,
                seated_first: false,
                forward_tokens: 0,
                walk_fire_rows: 0,
                host_planes: false,
                ..Engine::of(&b200)
            },
            Engine::of(&family),
            "only the row lattice, the state seating, the planner's order, the fire width and \
             the host planes differ from the family's engine"
        );
        assert_eq!(Engine::of(&family).state_slots_per_lane, 0);
        assert!(!Engine::of(&family).one_frame_when_short);
        assert!(!Engine::of(&family).seated_first);
        assert!(!Engine::of(&family).host_planes);
        assert_eq!(Engine::of(&family).lattice_steps, 1);
    }

    /// The two-launch gated delta scan is a decision of the A100, the 99 KiB
    /// cards (a quarter head a block there), Hopper and the sm_100 family;
    /// every other family keeps the three wmma launches it ran before.
    #[cfg(not(feature = "rocm"))]
    #[test]
    fn the_gated_delta_scan_runs_where_its_family_measured_it() {
        for (name, major, minor, sms) in [
            ("NVIDIA B200", 10, 0, 148),
            ("NVIDIA GB200", 10, 0, 152),
            ("NVIDIA H100 80GB HBM3", 9, 0, 132),
            ("NVIDIA A100-SXM4-80GB", 8, 0, 108),
            ("NVIDIA GeForce RTX 3090", 8, 6, 82),
            ("NVIDIA L40S", 8, 9, 142),
            ("NVIDIA GeForce RTX 5090", 12, 0, 170),
        ] {
            let (p, _) = resolve(name, &facts(major, minor, sms), None);
            assert!(
                kernels_cuda::tuning::Tuning::of(&p).ssm_gdn_fused_scan,
                "{name}"
            );
        }
        for (name, major, minor, sms) in [
            ("NVIDIA Jetson AGX Orin", 8, 7, 16),
            ("NVIDIA B300", 10, 3, 148),
            ("NVIDIA Thor", 11, 0, 20),
            ("NVIDIA GB10", 12, 1, 48),
        ] {
            let (p, _) = resolve(name, &facts(major, minor, sms), None);
            assert!(
                !kernels_cuda::tuning::Tuning::of(&p).ssm_gdn_fused_scan,
                "{name}"
            );
        }
    }

    /// `kernels-cuda`'s tests run the routed selects under `common::SM100`,
    /// a mirror of these: a change here changes that mirror too.
    #[cfg(not(feature = "rocm"))]
    #[test]
    fn the_b200_moe_decisions_are_what_the_kernel_tests_run() {
        let b200 =
            kernels_cuda::tuning::Tuning::of(&resolve("NVIDIA B200", &facts(10, 0, 148), None).0);
        assert_eq!((b200.moe_tc5, b200.moe_tc5_from), (true, 16));
        assert_eq!(
            (b200.mxfp4_mma_from, b200.mxfp4_mma_until, b200.mxfp4_cvt),
            (28, 1024, true)
        );
        assert_eq!((b200.nvfp4_mma_from, b200.nvfp4_mma_until), (32, 8192));
        assert_eq!(b200.moe_w4a4_fused_from, 1);
        assert_eq!((b200.moe_even_waves_from, b200.moe_routes_sized), (3, true));
        assert_eq!(b200.moe_gather_ahead_until, 128);
        assert_eq!(b200.moe_w4a8_fused_from, 512);
        assert_eq!(b200.mla_expanded_from, 256);
        assert_eq!(b200.gemm_fp8_gemv_until, 4);
        assert_eq!((b200.mxfp4_a8_from, b200.mxfp4_a8_until), (1, 48));
        assert_eq!(b200.mxfp4_cvt_ksplit_until, 6);
        assert!(b200.attn_res_fused_norm);
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn only_a_b200_runs_the_tcgen05_selects() {
        let tuning = |major, minor, sms| {
            kernels_cuda::tuning::Tuning::of(
                &resolve("NVIDIA GPU", &facts(major, minor, sms), None).0,
            )
        };
        let b200 = tuning(10, 0, 148);
        assert!(b200.moe_tc5);
        assert_eq!((b200.mxfp4_mma_from, b200.mxfp4_mma_until), (28, 1024));
        assert!(b200.merge_warp);
        assert!(b200.pdl);
        assert_eq!(b200.gemm_tc5_split_tickets, 8);
        assert_eq!(b200.gemm_rows_m, 8);
        // Hopper takes the register-dequant selects and none of tcgen05's.
        let h100 = tuning(9, 0, 132);
        assert!(!h100.moe_tc5 && !h100.mxfp4_cvt && h100.gemm_tc5_split_tickets == 0);
        assert_eq!(
            (h100.mxfp4_mma_from, h100.mxfp4_mma_until),
            (28, 16_777_216)
        );
        assert_eq!((h100.nvfp4_mma_from, h100.nvfp4_mma_until), (32, 8192));
        for (major, minor) in [
            (8, 0),
            (8, 6),
            (8, 7),
            (8, 9),
            (10, 3),
            (11, 0),
            (12, 0),
            (12, 1),
        ] {
            let other = tuning(major, minor, 132);
            assert!(!other.moe_tc5, "sm_{major}{minor} keeps the old selects");
            assert!(
                !other.merge_warp,
                "sm_{major}{minor} keeps the staged merge"
            );
            // Ampere, Ada and the RTX Blackwells take a prefill's MXFP4
            // selects on mma.sync; the rest keep the GEMV.
            assert_eq!(
                other.mxfp4_mma_until,
                if matches!((major, minor), (8, 0 | 6 | 9) | (12, 0)) {
                    16_777_216
                } else {
                    0
                },
                "sm_{major}{minor} MXFP4 selects"
            );
            assert_eq!(
                other.gemm_split_launch_ns, 0,
                "sm_{major}{minor} charges a split's launch nothing"
            );
            assert_eq!(
                other.nvfp4_mma_until, 0,
                "sm_{major}{minor} keeps the NVFP4 W4A16 select off the register-dequant kernel"
            );
            assert_eq!(
                other.gemm_tc5_raster, 8,
                "sm_{major}{minor} rasters its dense tcgen05 GEMM 8 tiles a band"
            );
            assert_eq!(
                other.gemm_rows_m, 0,
                "sm_{major}{minor} keeps a few-row dense product on cuBLAS"
            );
            assert!(
                other.gemm_tc5_split_tickets == 0,
                "sm_{major}{minor} sums a tcgen05 GEMM's splits in a finalize launch"
            );
            assert!(
                !other.pdl,
                "sm_{major}{minor} launches no kernel as a programmatic dependent"
            );
            assert_eq!(
                other.moe_w4a4_fused_from, 65_536,
                "sm_{major}{minor} keeps the W4A4 routed MLP apart"
            );
            assert_eq!(
                other.moe_w4a8_fused_from, 65_536,
                "sm_{major}{minor} keeps the W4A8 routed MLP apart"
            );
            assert_eq!(
                other.mla_expanded_from, 65_536,
                "sm_{major}{minor} scores a latent prefill absorbed"
            );
            assert_eq!(
                other.gemm_fp8_gemv_until, 0,
                "sm_{major}{minor} runs no FP8 projection as a GEMV"
            );
            assert!(
                !other.attn_res_fused_norm,
                "sm_{major}{minor} blends and norms apart"
            );
        }
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn every_a100_runs_the_a100_file_and_another_ampere_its_family_alone() {
        for gpu in [
            "NVIDIA A100 80GB PCIe",
            "NVIDIA A100-SXM4-80GB",
            "NVIDIA A100-PCIE-40GB",
        ] {
            let (_, report) = resolve(gpu, &facts(8, 0, 108), None);
            assert_eq!(
                report.layers,
                [
                    profile::Source::Seeded("sm80.toml".into()),
                    profile::Source::Seeded("sm80/nvidia-a100.toml".into())
                ],
                "{gpu}"
            );
        }
        let (_, report) = resolve("NVIDIA A30", &facts(8, 0, 56), None);
        assert_eq!(report.layers, [profile::Source::Seeded("sm80.toml".into())]);
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn the_l40s_runs_its_family_measured_and_another_ada_the_same_unmeasured() {
        let (l40s, report) = resolve("NVIDIA L40S", &facts(8, 9, 142), None);
        assert_eq!(
            report.layers,
            [
                profile::Source::Seeded("sm89.toml".into()),
                profile::Source::Seeded("sm89/nvidia-l40s.toml".into())
            ]
        );
        assert!(!report.untuned());
        let (rtx, report) = resolve("NVIDIA GeForce RTX 4090", &facts(8, 9, 128), None);
        assert_eq!(report.used, profile::Source::Seeded("sm89.toml".into()));
        assert!(report.untuned());
        for p in [&l40s, &rtx] {
            assert_eq!(
                Engine::of(p),
                Engine {
                    prefill_lattice_from: 256,
                    prefill_lattice_steps: 8,
                    ..Engine::of(&registry().defaults())
                }
            );
            assert!(kernels_cuda::tuning::Tuning::of(p).ssm_gdn_fused_scan);
            let defaults = kernels_cuda::tuning::Tuning::of(&registry().defaults());
            assert_eq!(
                kernels_cuda::tuning::Tuning {
                    ssm_gdn_fused_scan: false,
                    mxfp4_mma_from: defaults.mxfp4_mma_from,
                    mxfp4_mma_until: defaults.mxfp4_mma_until,
                    mxfp4_mma_per_from: defaults.mxfp4_mma_per_from,
                    ..kernels_cuda::tuning::Tuning::of(p)
                },
                defaults,
                "what an Ada GPU ran before its family file, past the gated delta scan and the MXFP4 selects"
            );
            for (spec, rungs, _) in p.entries() {
                if spec.id == "kernel.ssm.gdn_fused_scan"
                    || spec.id.starts_with("kernel.moe.mxfp4_mma_")
                    || spec.id.starts_with("engine.prefill_lattice_")
                {
                    continue;
                }
                assert_eq!(
                    rungs,
                    registry()
                        .defaults()
                        .entries()
                        .find(|(s, ..)| s.id == spec.id)
                        .unwrap()
                        .1,
                    "{}",
                    spec.id
                );
            }
        }
    }
}

#[cfg(all(test, feature = "rocm"))]
mod hip {
    use super::*;

    #[test]
    fn a_cdna3_gpu_takes_its_family() {
        for gpu in ["AMD Instinct MI300X", "AMD Instinct MI325X"] {
            let (profile, report) = resolve(gpu, &facts(9, 4, 304), None);
            assert_eq!(
                report.layers,
                [profile::Source::Seeded("gfx94.toml".into())],
                "{gpu}"
            );
            assert!(report.untuned(), "{gpu}");
            assert_eq!(
                Engine::of(&profile),
                Engine {
                    forward_tokens: 10240,
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
            assert_eq!(kernels_cuda::tuning::Tuning::of(&profile), want, "{gpu}");
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
        assert_eq!(
            poem_compiler::DeviceProfile::of(&registry().defaults(), 132, &[], &[]),
            poem_compiler::DeviceProfile {
                cond_fixed_us: 1_000_000.0,
                ..poem_compiler::DeviceProfile::default()
            }
        );
        let mut fused = poem_compiler::fuse::flags::kernels(&registry().defaults());
        let mut shipped = crate::FUSED.to_vec();
        fused.sort_unstable();
        shipped.sort_unstable();
        assert_eq!(fused, shipped);
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
