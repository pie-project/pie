//! A model whose classes read a guest input settles its dense ranges
//! (`Shell::settle_dense`) as any other: each settle fire feeds zeros of
//! every read input's declared shape, so the load weighs its ranges and
//! stores them in the profile, and a second boot under the same profile
//! reads them back and measures nothing.

use super::outputs::{Outputs, loaded_boot};
use crate::backend;

/// Every file under `dir`, by path, with its bytes.
fn files(dir: &std::path::Path) -> Vec<(std::path::PathBuf, Vec<u8>)> {
    let mut all = Vec::new();
    let mut walk = vec![dir.to_path_buf()];
    while let Some(at) = walk.pop() {
        let Ok(entries) = std::fs::read_dir(&at) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk.push(path);
            } else {
                let bytes = std::fs::read(&path).expect("a profile file reads");
                all.push((path, bytes));
            }
        }
    }
    all.sort();
    all
}

#[test]
fn a_guest_input_model_settles_and_stores_and_a_second_boot_measures_nothing() {
    let profiles = std::env::temp_dir().join(format!(
        "pie-{}-guest-settle-profiles-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&profiles);
    let boot = || backend::DeviceBoot {
        profiles: Some(profiles.clone()),
        ..crate::gpu::device_boot()
    };

    let Some(mut first) = loaded_boot("guest-settle", &Outputs, 4, boot()) else {
        return;
    };
    let shell = first.engine.loaded_mut().expect("loaded");
    let settled = shell.settled().to_vec();
    assert!(
        settled.iter().any(|range| range.checked),
        "the first boot weighed no range: {settled:?}"
    );
    drop(first);
    let stored = files(&profiles);
    assert!(
        stored
            .iter()
            .any(|(_, bytes)| String::from_utf8_lossy(bytes).contains("dense_settled")),
        "the first boot stored no settle"
    );

    let Some(mut second) = loaded_boot("guest-settle", &Outputs, 4, boot()) else {
        return;
    };
    let shell = second.engine.loaded_mut().expect("loaded");
    let settled = shell.settled().to_vec();
    assert!(
        settled.iter().all(|range| !range.checked),
        "the second boot weighed a range again: {settled:?}"
    );
    drop(second);
    let after = files(&profiles);
    assert!(
        stored == after,
        "the second boot stored what it measured anew: {:?}",
        after
            .iter()
            .filter(|file| !stored.contains(file))
            .map(|(path, _)| path)
            .collect::<Vec<_>>()
    );
    let _ = std::fs::remove_dir_all(&profiles);
}
