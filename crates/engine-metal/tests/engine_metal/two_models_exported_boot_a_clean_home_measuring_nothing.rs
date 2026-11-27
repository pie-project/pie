#![cfg(target_vendor = "apple")]

//! Two models measured in one home and exported: a clean home holding only
//! the export (as a user's file reads it; a committed profile is read by the
//! same keys) boots each measuring nothing (`Shell::measured_at_load` empty,
//! the load's "measured 0"), as engine-gpu's suite checks the same.

use engine_metal::Shell;

use super::a_scattered_pool_serves_the_same_logits::{Load, Fixture, fixture_of};

fn load(fixture: &Fixture, profiles: &std::path::Path) -> Shell {
    Load {
        profiles: Some(profiles.to_path_buf()),
        ..Load::default()
    }
    .boot(fixture)
    .unwrap_or_else(|why| panic!("the shell loads: {why}"))
}

/// The user's profile under `dir`: the one file the loads stored there.
fn stored_file(dir: &std::path::Path) -> std::path::PathBuf {
    let mut walk = vec![dir.to_path_buf()];
    while let Some(at) = walk.pop() {
        for entry in std::fs::read_dir(&at).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk.push(path);
            } else if path.extension().is_some_and(|e| e == "toml") {
                return path;
            }
        }
    }
    panic!("the loads stored no profile under {}", dir.display());
}

#[test]
fn two_models_exported_boot_a_clean_home_measuring_nothing() {
    let _device = crate::device_alone();
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let home = |what: &str| {
        std::env::temp_dir().join(format!("pie-metal-measured-none-{what}-{}", std::process::id()))
    };
    let (measured, clean) = (home("measured"), home("clean"));
    let _ = std::fs::remove_dir_all(&measured);
    let _ = std::fs::remove_dir_all(&clean);
    let models = [fixture_of("qwen3-micro-text", None), fixture_of("qwen35-tiny", None)];
    let first: Vec<Vec<String>> = models
        .iter()
        .map(|fixture| load(fixture, &measured).measured_at_load().to_vec())
        .collect();
    assert!(
        first.iter().all(|timed| timed.iter().any(|m| m.starts_with("dense curves"))),
        "each model's first load times its kernel.dense curves: {first:?}"
    );
    let stored = stored_file(&measured);
    let text = std::fs::read_to_string(&stored).expect("the stored profile reads");
    let (exported, report) =
        profile::export_reporting(engine_metal::chip::registry(), "the device", &text, "user.toml")
            .expect("the home exports");
    assert!(report.iter().all(|line| !line.starts_with("WARN")), "{report:?}");
    let into = clean.join(stored.strip_prefix(&measured).expect("under the home"));
    std::fs::create_dir_all(into.parent().expect("a directory")).expect("the clean home");
    std::fs::write(&into, exported).expect("the export lands");
    for (fixture, name) in models.iter().zip(["qwen3-micro-text", "qwen35-tiny"]) {
        let timed = load(fixture, &clean).measured_at_load().to_vec();
        assert!(timed.is_empty(), "{name}'s load from the export measured {timed:?}");
    }
    let _ = std::fs::remove_dir_all(&measured);
    let _ = std::fs::remove_dir_all(&clean);
}
