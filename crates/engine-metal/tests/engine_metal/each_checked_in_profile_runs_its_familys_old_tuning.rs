//! Every checked-in profile not measured on its chip resolves to what
//! `DeviceTuning::of(family)` gave that chip before profiles existed; the
//! functions below are that table, frozen.

use engine_metal::chip::{self, AneMlp, AneProjections, StreamCut, WeightsLayout};
use kernels_metal::Tuning;
use kernels_metal::tuning::{GdnScan, Qmm, QmmAt};
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

fn family_of(device: &str) -> u32 {
    match device.split_whitespace().nth(1) {
        Some(m) if m.starts_with("M1") => 7,
        Some(m) if m.starts_with("M2") => 8,
        Some(m) if m.starts_with("M3") || m.starts_with("M4") => 9,
        Some(m) if m.starts_with("M5") => 10,
        other => panic!("no family for {other:?}"),
    }
}

#[test]
fn each_checked_in_profile_runs_its_familys_old_tuning() {
    assert_eq!(chip::SEEDED.len(), 18);
    for (name, text) in chip::SEEDED {
        let file = chip::registry()
            .parse(text, name, profile::Strict::CheckedIn)
            .unwrap_or_else(|e| panic!("{e}"));
        if file.measured() {
            continue;
        }
        let device = file.target.device.expect("named");
        let family = family_of(&device);
        let (profile, report) = chip::resolve(&device, &chip::facts(family, 10, true), None);
        assert_eq!(report.used, Source::Seeded((*name).to_string()), "{device}");
        let t = Tuning::new(profile.clone(), false, 10);
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
                        fill: 0,
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
        assert_eq!(t.sdpa_tile_rows_per_request(), 32, "{device}");
        assert!(t.sdpa_mma() && !t.kv_q8(), "{device}");
        assert_eq!(t.gdn_scan(), gdn_scan, "{device}");
        let mlp = if mpp {
            AneMlp::Units { units: 8 }
        } else {
            AneMlp::Share { percent: 47 }
        };
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
            poem_compiler::DeviceProfile::of(&profile, 10, &[], &[]),
            poem_compiler::DeviceProfile {
                sms: 10,
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
}

#[test]
fn a_wide_range_model_leaves_the_fp16_format_on_any_chip() {
    let (profile, _) = chip::resolve("Apple M4 Max", &chip::facts(9, 40, true), None);
    let wide = Tuning::new(profile, true, 10);
    assert!(!wide.fp16_gemm_format(4, 64));
    let at = |rows| QmmAt {
        rows,
        fill: 0,
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
fn a_mac_whose_bridge_does_not_load_runs_the_defaults_whole() {
    let (profile, report) = chip::resolve("Apple M5 Max", &chip::facts(10, 40, false), None);
    assert_eq!(report.used, Source::Defaults);
    assert!(
        report.rejected[0].1.contains("ane"),
        "{:?}",
        report.rejected
    );
    let t = Tuning::new(profile.clone(), false, 10);
    assert!(!t.qmm_mpp() && !t.sdpa_mpp());
    assert_eq!(*profile.get::<AneMlp>(), AneMlp::Gpu);
}

#[test]
fn a_profile_for_another_family_is_refused() {
    let (_, report) = chip::resolve("Apple M4 Max", &chip::facts(10, 40, true), None);
    assert_eq!(report.used, Source::Defaults);
    assert!(
        report.rejected[0]
            .1
            .contains("it assumes family = 9, and this device has 10"),
        "{:?}",
        report.rejected
    );
}
