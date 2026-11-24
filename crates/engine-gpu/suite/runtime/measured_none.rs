//! Two models measured in one home and exported: a clean home holding only
//! the export (as a user's file reads it; a committed profile is read by the
//! same keys) boots each measuring nothing (`Shell::measured_at_load` empty,
//! the load's "measured 0").

use super::outputs::{Outputs, loaded_boot};
use crate::backend;

/// The user's profile under `dir`: the one file a load stored there.
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
    panic!("the boots stored no profile under {}", dir.display());
}

#[test]
fn two_models_exported_boot_a_clean_home_measuring_nothing() {
    let home = |what: &str| {
        std::env::temp_dir().join(format!(
            "pie-{}-measured-none-{what}-{}",
            backend::NAME,
            std::process::id()
        ))
    };
    let (measured, clean) = (home("measured"), home("clean"));
    let _ = std::fs::remove_dir_all(&measured);
    let _ = std::fs::remove_dir_all(&clean);
    let boot = |profiles: &std::path::Path| backend::DeviceBoot {
        profiles: Some(profiles.to_path_buf()),
        ..crate::gpu::device_boot()
    };
    // Two loads of different keys (lanes key the settle): two "models".
    let models = [4u32, 2];
    let mut first = Vec::new();
    for lanes in models {
        let Some(mut loaded) = loaded_boot("measured-none", &Outputs, lanes, boot(&measured)) else {
            return;
        };
        first.push(super::outputs::measure_behind(&mut loaded.engine));
    }
    assert!(first[0] > 0, "the first model settles behind serving: {first:?}");
    let stored = stored_file(&measured);
    let text = std::fs::read_to_string(&stored).expect("the stored profile reads");
    let (exported, report) =
        profile::export_reporting(backend::registry(), "the device", &text, "user.toml")
            .expect("the home exports");
    assert_eq!(report.len(), models.len(), "a settle a model: {report:?}");
    assert!(report.iter().all(|line| !line.starts_with("WARN")), "{report:?}");
    let into = clean.join(stored.strip_prefix(&measured).expect("under the home"));
    std::fs::create_dir_all(into.parent().expect("a directory")).expect("the clean home");
    std::fs::write(&into, exported).expect("the export lands");
    for lanes in models {
        let mut loaded = loaded_boot("measured-none", &Outputs, lanes, boot(&clean))
            .expect("the device opened once");
        let timed = loaded.engine.loaded_mut().expect("loaded").measured_at_load();
        assert!(timed.is_empty(), "the {lanes}-lane load measured {timed:?}");
    }
    let _ = std::fs::remove_dir_all(&measured);
    let _ = std::fs::remove_dir_all(&clean);
}
