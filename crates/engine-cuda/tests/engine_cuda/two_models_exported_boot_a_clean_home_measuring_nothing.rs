#![cfg(feature = "device")]

//! Two models measured in one home and exported: a clean home holding only
//! the export (the profile as a user's file reads it, the bodies records as
//! its cache holds them; a committed profile and committed records are read
//! by the same keys) loads each measuring nothing (`Shell::measured_at_load`
//! empty, the load's "measured 0"), as engine-gpu's suite checks the same.

use std::path::{Path, PathBuf};

use engine_cuda::Shell;

use super::golden_logits::{Fixture, cache_dir, fixture, knobs, load_under};

/// A load of `fixture` whose profiles and bodies records are `home`'s.
fn load(fixture: &Fixture, home: &Path) -> Shell {
    let mut knobs = knobs();
    knobs.profiles = Some(home.join("profiles"));
    load_under(fixture, &home.join("cache"), knobs, engine::Allowance::WHOLE)
        .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model))
}

/// The user's profile under `dir`: the one file the loads stored there.
fn stored_file(dir: &Path) -> PathBuf {
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
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let home = |what: &str| {
        std::env::temp_dir().join(format!("pie-cuda-measured-none-{what}-{}", std::process::id()))
    };
    let (measured, clean) = (home("measured"), home("clean"));
    let _ = std::fs::remove_dir_all(&measured);
    let _ = std::fs::remove_dir_all(&clean);
    let fixtures = cache_dir();
    let models = [
        fixture("qwen3-micro-text", &fixtures),
        fixture("qwen35-tiny", &fixtures),
    ];
    for fixture in &models {
        let timed = load(fixture, &measured).measured_at_load().to_vec();
        assert!(
            timed.iter().any(|m| m == "bodies"),
            "{}'s first load measures its bodies: {timed:?}",
            fixture.model
        );
    }
    let profiles = measured.join("profiles");
    let stored = stored_file(&profiles);
    let text = std::fs::read_to_string(&stored).expect("the stored profile reads");
    let (exported, report) =
        profile::export_reporting(engine_cuda::chip::registry(), "the device", &text, "user.toml")
            .expect("the home exports");
    assert!(report.iter().all(|line| !line.starts_with("WARN")), "{report:?}");
    let into = clean
        .join("profiles")
        .join(stored.strip_prefix(&profiles).expect("under the home"));
    std::fs::create_dir_all(into.parent().expect("a directory")).expect("the clean home");
    std::fs::write(&into, exported).expect("the export lands");
    let bodies = engine_cuda::serve::export_bodies(&measured.join("cache"), &clean.join("cache"))
        .expect("the bodies records export");
    assert_eq!(bodies.len(), models.len(), "a record a model: {bodies:?}");
    for fixture in &models {
        let timed = load(fixture, &clean).measured_at_load().to_vec();
        assert!(timed.is_empty(), "{}'s load from the export measured {timed:?}", fixture.model);
    }
    let _ = std::fs::remove_dir_all(&measured);
    let _ = std::fs::remove_dir_all(&clean);
}
