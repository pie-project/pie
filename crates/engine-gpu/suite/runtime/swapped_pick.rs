//! A dense pick changed in the profile a load reads (a stored settle's
//! range given another kernel; a committed entry replacing a stored one is
//! read the same way, entry by entry): the next load serves the new pick,
//! builds what it fires before ready (no list of an earlier load's is
//! reused), and keys its host cover anew by the picks as resolved, never by
//! the file that stated them.

use ::engine::Engine as _;

use super::outputs::{Outputs, loaded_boot};
use crate::backend;

/// The user's profile under `dir` holding a stored settle.
fn settled_file(dir: &std::path::Path) -> std::path::PathBuf {
    let mut walk = vec![dir.to_path_buf()];
    while let Some(at) = walk.pop() {
        for entry in std::fs::read_dir(&at).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk.push(path);
            } else if std::fs::read_to_string(&path)
                .is_ok_and(|text| text.contains("dense_settled_keyed"))
            {
                return path;
            }
        }
    }
    panic!("the first boot stored no settle under {}", dir.display());
}

#[test]
fn a_swapped_pick_is_built_before_ready_and_keys_the_cover_anew() {
    let profiles = std::env::temp_dir().join(format!(
        "pie-{}-swapped-pick-profiles-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&profiles);
    let boot = || backend::DeviceBoot {
        profiles: Some(profiles.clone()),
        ..crate::gpu::device_boot()
    };

    let Some(mut first) = loaded_boot("swapped-pick", &Outputs, 4, boot()) else {
        return;
    };
    let cover = first
        .engine
        .cover_fingerprint()
        .expect("the load keys its host cover");
    let shell = first.engine.loaded_mut().expect("loaded");
    let before = shell.dense_tactics_at(1)[0];
    super::outputs::measure_behind(&mut first.engine);
    drop(first);

    // The one-row range given a vector kernel the first load did not serve
    // there.
    let swapped = kernels_slang::linear::gemm::Tactic {
        vector: if before.vector == 8 { 4 } else { 8 },
        wide: false,
        ..before
    };
    let path = settled_file(&profiles);
    let text = std::fs::read_to_string(&path).expect("the profile reads");
    let at = text.find("[[1, 1, ").expect("the settle decided the one-row range") + 1;
    let end = at + text[at..].find(']').expect("a range closes") + 1;
    let row = backend::Shell::settled_row(1, 1, swapped);
    let swapped_text = format!(
        "{}[{}]{}",
        &text[..at],
        row.map(|v| v.to_string()).join(", "),
        &text[end..]
    );
    std::fs::write(&path, swapped_text).expect("the profile writes");

    let Some(mut second) = loaded_boot("swapped-pick", &Outputs, 4, boot()) else {
        return;
    };
    let swapped_cover = second
        .engine
        .cover_fingerprint()
        .expect("the load keys its host cover");
    let shell = second.engine.loaded_mut().expect("loaded");
    assert!(
        shell.settled().iter().all(|range| !range.checked),
        "the stored settle's key still matches: nothing is weighed again"
    );
    assert!(
        shell.dense_tactics_at(1).iter().all(|&t| t == swapped),
        "the one-row range serves the swapped pick: {:?}",
        shell.dense_tactics_at(1)
    );
    let built = shell.built_kernels();
    for entry in shell.dense_entries_at(1).expect("the pick has its kernel") {
        assert!(
            built.iter().any(|kernel| kernel.starts_with(entry)),
            "`{entry}`, which the swapped pick fires, was not built before ready: {built:?}"
        );
    }
    drop(second);
    assert_ne!(
        cover, swapped_cover,
        "the host cover is keyed by the picks as resolved"
    );
    let _ = std::fs::remove_dir_all(&profiles);
}
