//! Every chip of a family resolves to what `DeviceTuning::of(family)` gave
//! it before profiles existed, but what was measured on it since; the
//! functions below are that table, frozen.

use engine_metal::chip::{self, AneMlp, AneProjections, ExpertsThreads, StreamCut, WeightsLayout};
use kernels_metal::Tuning;
use kernels_metal::tuning::{GdnScan, Qmm, QmmAt, fill};
use profile::Source;

/// `(fp16 tiled from rows, crossover_tg, qmm_mpp, sdpa_mpp, gdn scan)`.
fn old(family: u32) -> (i64, u32, bool, bool, GdnScan) {
    let simd = GdnScan::Simd { rows: 4 };
    match family {
        10 => (5, 160, true, true, GdnScan::Staged { min_rows: 32 }),
        9 => (5, 96, false, false, simd),
        8 => (8, 160, false, true, simd),
        _ => (5, 160, false, false, simd),
    }
}

/// The old routed tile choice.
fn old_tile(pairs: u32, experts: u32) -> u32 {
    if experts == 0 || pairs < experts * 2 {
        return 1;
    }
    let per = pairs / experts;
    if per >= 32 {
        32
    } else if per >= 1 {
        16
    } else {
        8
    }
}

/// The kernel the old cascade in `act_x_wt_over` ran a dense matmul on.
fn old_qmm(family: u32, at: &QmmAt) -> Qmm {
    let (fp16_from, crossover_tg, mpp, ..) = old(family);
    let bn = if at.rows <= 8 { 64 } else { 128 };
    if mpp
        && at.bits == 4
        && at.group == 64
        && at.rows >= 8
        && at.n % bn == 0
        && (at.rows > 16 || at.n >= 1024)
    {
        return Qmm::Mpp;
    }
    let from = match (at.bits, at.format) {
        (2, _) => 5,
        (_, "fp16") => fp16_from,
        _ => 12,
    };
    if at.rows >= from && at.k % 32 == 0 {
        return Qmm::Tiled { crossover_tg };
    }
    Qmm::Qmv {
        rows_max: 8,
        packs: 1,
        crossover_tg,
    }
}

/// Chips of each family by the newest `MTLGPUFamily::AppleN` they support,
/// every iPhone and iPad chip that runs Metal 4 among them, and the GPU
/// cores each says (iOS does not): none is named by a file but the M4 Pro.
const CHIPS: &[(u32, &str, Option<u32>)] = &[
    (7, "Apple M1", Some(8)),
    (7, "Apple M1 Ultra", Some(64)),
    (7, "Apple A14 GPU", None),
    (8, "Apple M2 Max", Some(38)),
    (8, "Apple A15 GPU", None),
    (8, "Apple A16 GPU", None),
    (9, "Apple M3", Some(10)),
    (9, "Apple M4 Max", Some(40)),
    (9, "Apple A17 Pro GPU", None),
    (9, "Apple A18 GPU", None),
    (9, "Apple A18 Pro GPU", None),
    (10, "Apple M5", Some(10)),
    (10, "Apple M5 Pro", Some(20)),
    (10, "Apple A19 GPU", None),
    (10, "Apple A19 Pro GPU", None),
];

/// What `DeviceTuning::of(family)` gave a chip with the Neural Engine, but
/// the split-K fill and the Neural Engine's MLP share, which `fill_tg` and
/// `mlp` stand in for.
fn assert_old_tuning(
    device: &str,
    family: u32,
    cores: Option<u32>,
    fill_tg: u32,
    mlp: Option<AneMlp>,
) {
    let (profile, report) = chip::resolve(device, &chip::facts(family, cores, 10, true), None);
    assert_eq!(
        report.layers.first(),
        Some(&Source::Seeded(format!("apple{family}.toml"))),
        "{device}"
    );
    assert!(report.rejected.is_empty(), "{device}: {report:?}");
    let t = Tuning::new(profile.clone(), false, cores);
    let (_, _, mpp, sdpa_mpp, gdn_scan) = old(family);
    for rows in [1, 2, 4, 5, 7, 8, 9, 11, 12, 16, 17, 32, 512] {
        for (n, k) in [
            (512, 4096),
            (1000, 4096),
            (1024, 4096),
            (4096, 1000),
            (8192, 8192),
        ] {
            for (bits, group, format) in [
                (4, 64, "fp16"),
                (4, 64, "emulated"),
                (4, 32, "emulated"),
                (2, 64, "emulated"),
                (8, 64, "emulated"),
            ] {
                // No checked-in ladder reads `fill`.
                let at = QmmAt {
                    rows,
                    fill: cores.map(|cores| fill(rows as u32, n as u32, cores)),
                    n,
                    k,
                    bits,
                    group,
                    format,
                };
                assert_eq!(
                    t.qmm(&at).first(),
                    Some(&old_qmm(family, &at)),
                    "{device}: {at:?}"
                );
            }
        }
    }
    for experts in [1, 8, 64, 128, 256] {
        for pairs in [0, 1, 7, 8, 64, 255, 256, 2048, 4096, 8191, 8192, 65_536] {
            assert_eq!(
                t.moe_tile_rows(pairs, experts),
                old_tile(pairs, experts),
                "{device}: {pairs} pairs over {experts} experts"
            );
        }
    }
    assert!(
        t.fp16_gemm_format(4, 64) && !t.fp16_gemm_format(4, 32),
        "{device}"
    );
    assert_eq!(
        (t.qmm_mpp(), t.moe_mpp(), t.sdpa_mpp()),
        (mpp, mpp, sdpa_mpp),
        "{device}"
    );
    assert_eq!(t.qmm_splitk(), Some((fill_tg, 8)), "{device}");
    assert_eq!(t.sdpa_tile_rows_per_request(), 32, "{device}");
    assert!(t.sdpa_mma() && !t.kv_q8(), "{device}");
    assert_eq!(t.gdn_scan(), gdn_scan, "{device}");
    let mlp = mlp.unwrap_or(if mpp {
        AneMlp::Units { units: 8 }
    } else {
        AneMlp::Share { percent: 47 }
    });
    assert_eq!(*profile.get::<AneMlp>(), mlp, "{device}");
    assert_eq!(
        *profile.get::<AneProjections>(),
        AneProjections::Share { percent: 47 }
    );
    assert_eq!(
        *profile.get::<WeightsLayout>(),
        WeightsLayout::Plain,
        "{device}"
    );
    assert_eq!(*profile.get::<StreamCut>(), StreamCut::FanOut);
    assert_eq!(
        profile.get::<ExpertsThreads>().count(10),
        8,
        "{device}: on 10 performance cores"
    );
    assert_eq!(
        poem_compiler::DeviceProfile::of(&profile, chip::pricing_cores(&profile, cores), &[], &[]),
        poem_compiler::DeviceProfile {
            sms: cores.unwrap_or(32),
            side_streams: 0,
            ..poem_compiler::DeviceProfile::default()
        },
        "{device}"
    );
    assert_eq!(
        poem_compiler::fuse::flags::kernels(&profile),
        ["elementwise.residual_add_rmsnorm", "linear.mlp_swiglu"],
        "{device}"
    );
}

#[test]
fn each_checked_in_profile_runs_its_familys_old_tuning() {
    for &(family, device, cores) in CHIPS {
        assert_old_tuning(device, family, cores, 512, None);
    }
}

#[test]
fn the_m4_pro_runs_its_family_and_the_two_values_it_measured() {
    assert_old_tuning(
        "Apple M4 Pro",
        9,
        Some(20),
        896,
        Some(AneMlp::Share { percent: 71 }),
    );
    let (_, report) = chip::resolve("Apple M4 Pro", &chip::facts(9, Some(20), 10, true), None);
    assert_eq!(report.measured, ["kernel.ane.mlp", "kernel.qmm_splitk"]);
}

#[test]
fn a_wide_range_model_leaves_the_fp16_format_on_any_chip() {
    let (profile, _) = chip::resolve("Apple M4 Max", &chip::facts(9, Some(40), 10, true), None);
    let wide = Tuning::new(profile, true, Some(40));
    assert!(!wide.fp16_gemm_format(4, 64));
    let at = |rows| QmmAt {
        rows,
        fill: Some(fill(rows as u32, 4096, 40)),
        n: 4096,
        k: 4096,
        bits: 4,
        group: 64,
        format: "emulated",
    };
    assert!(matches!(wide.qmm(&at(11))[0], Qmm::Qmv { .. }));
    assert!(matches!(wide.qmm(&at(12))[0], Qmm::Tiled { .. }));
}

#[test]
fn a_mac_whose_bridge_does_not_load_runs_its_family_on_the_gpu() {
    let (profile, report) =
        chip::resolve("Apple M5 Max", &chip::facts(10, Some(40), 10, false), None);
    assert_eq!(report.used, Source::Seeded("apple10.toml".into()));
    assert!(report.rejected.is_empty(), "{report:?}");
    let t = Tuning::new(profile.clone(), false, Some(40));
    assert!(
        t.qmm_mpp() && t.sdpa_mpp(),
        "the family's GPU tuning stands"
    );
    assert_eq!(*profile.get::<AneMlp>(), AneMlp::Gpu);
    assert_eq!(*profile.get::<AneProjections>(), AneProjections::Gpu);
}

#[test]
fn a_device_file_applies_only_within_its_family() {
    let (profile, report) =
        chip::resolve("Apple M4 Pro", &chip::facts(10, Some(40), 10, true), None);
    assert_eq!(report.layers, [Source::Seeded("apple10.toml".into())]);
    assert_eq!(*profile.get::<AneMlp>(), AneMlp::Units { units: 8 });
    let (_, report) = chip::resolve("Apple M4 Pro", &chip::facts(11, Some(40), 10, true), None);
    assert_eq!(report.used, Source::Defaults, "no family applies");
}
