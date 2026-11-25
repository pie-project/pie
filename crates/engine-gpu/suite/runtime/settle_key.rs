//! A model's first boot, which times its curves and leaves its settle to
//! weigh behind serving, and its next, which reads them back, name the same
//! kernels and look the settle up by the same key: the second weighs
//! nothing. (A load that times its curves once named a decode's split walk
//! out on wgpu: the admitted decodes were dropped with the tuning the
//! curves rebuilt, and the key moved.) The first boot is ready without
//! settling and serves its fallback picks while the settle runs behind it;
//! a settle point another lane's work overlaps is never stored.

use ::engine::Engine as _;

use super::outputs::{Outputs, loaded_boot, measure_behind};
use crate::backend;

fn profiles_at(tag: &str) -> std::path::PathBuf {
    let profiles = std::env::temp_dir().join(format!(
        "pie-{}-{tag}-profiles-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&profiles);
    profiles
}

#[test]
fn a_measured_boot_and_the_next_name_the_same_kernels_under_one_settle_key() {
    let profiles = profiles_at("settle-key");
    let boot = || backend::DeviceBoot {
        profiles: Some(profiles.clone()),
        ..crate::gpu::device_boot()
    };
    let read = |loaded: &mut super::outputs::Loaded| {
        let shell = loaded.engine.loaded_mut().expect("loaded");
        (
            shell.settle_keyed().expect("the settle's key was read"),
            shell.asked_kernels(),
            shell.settled().iter().any(|range| range.checked),
        )
    };
    let Some(mut first) = loaded_boot("settle-key", &Outputs, 4, boot()) else {
        return;
    };
    let (key, kernels, weighed_at_load) = read(&mut first);
    assert!(!weighed_at_load, "a boot weighs nothing before ready");
    assert!(first.engine.measures_behind(), "the settle is left behind serving");
    let shell = first.engine.loaded_mut().expect("loaded");
    let rows: Vec<u32> = shell.settled().iter().map(|r| r.top).chain([1, 2, 4]).collect();
    let served: Vec<_> = rows.iter().map(|&r| shell.dense_tactics_at(r)).collect();
    assert!(measure_behind(&mut first.engine) > 0, "the settle ran behind serving");
    let shell = first.engine.loaded_mut().expect("loaded");
    assert!(
        shell.settled().iter().any(|range| range.checked),
        "the settle weighed its ranges behind serving"
    );
    let after: Vec<_> = rows.iter().map(|&r| shell.dense_tactics_at(r)).collect();
    assert_eq!(after, served, "the settle behind serving changes nothing this load serves");
    drop(first);
    let mut second = loaded_boot("settle-key", &Outputs, 4, boot()).expect("the device opened once");
    let (again, kernels_again, weighed_again) = read(&mut second);
    let pending = second.engine.measures_behind();
    drop(second);
    let _ = std::fs::remove_dir_all(&profiles);
    assert_eq!(
        kernels_again, kernels,
        "the boot that read its curves names what the boot that timed them named"
    );
    assert_eq!(again, key, "one settle key");
    assert!(!weighed_again, "the stored settle is taken: nothing weighed again");
    assert!(!pending, "the next boot reads the stored settle: nothing left behind serving");
}

/// A boot whose 1-row range an earlier one decided times one decode step of
/// its own before any wider point: no boot fires a point it knows no limit
/// for.
#[test]
fn a_boot_with_the_one_row_range_decided_times_a_decode_step_first() {
    let profiles = profiles_at("settle-limit");
    let boot = || backend::DeviceBoot {
        profiles: Some(profiles.clone()),
        ..crate::gpu::device_boot()
    };
    let behind = ::engine::BehindServing {
        others: &|| 0,
        arrived: &|| false,
        seats: None,
        step: None,
    };
    let Some(mut first) = loaded_boot("settle-limit", &Outputs, 4, boot()) else {
        return;
    };
    let stored = |loaded: &mut super::outputs::Loaded| {
        let shell = loaded.engine.loaded_mut().expect("loaded");
        shell.behind_status().map_or(1, |(_, stored, _)| stored)
    };
    while stored(&mut first) == 0 {
        assert!(first.engine.measure_behind(&behind).expect("measuring behind"));
    }
    drop(first);
    let mut second = loaded_boot("settle-limit", &Outputs, 4, boot()).expect("the device opened once");
    assert!(second.engine.measures_behind(), "wider ranges are left");
    assert!(second.engine.measure_behind(&behind).expect("measuring behind"));
    let need = second.engine.behind_need().expect("a point is next");
    drop(second);
    let _ = std::fs::remove_dir_all(&profiles);
    assert_eq!((need.lanes, need.tokens), (1, 1), "one decode step, before any wider point");
}

/// Every timed settle point overlapped by another lane's submission (the
/// runtime's count moves while it runs) is discarded: none is stored, and
/// the settle stays pending.
#[test]
fn a_settle_point_overlapped_by_a_fire_is_not_stored() {
    let profiles = profiles_at("settle-overlap");
    let boot = || backend::DeviceBoot {
        profiles: Some(profiles.clone()),
        ..crate::gpu::device_boot()
    };
    let Some(mut loaded) = loaded_boot("settle-overlap", &Outputs, 4, boot()) else {
        return;
    };
    let submitted = std::cell::Cell::new(0u64);
    // Another lane submits at every read: each point sees the count move.
    let busy = || {
        submitted.set(submitted.get() + 1);
        submitted.get()
    };
    let behind = ::engine::BehindServing {
        others: &busy,
        arrived: &|| false,
        seats: None,
        step: None,
    };
    let mut slices = 0;
    while slices < 64 && loaded.engine.measure_behind(&behind).expect("measuring behind") {
        slices += 1;
    }
    let shell = loaded.engine.loaded_mut().expect("loaded");
    assert!(
        shell.settled().iter().all(|range| !range.checked),
        "no range was weighed from overlapped points: {:?}",
        shell.settled()
    );
    let (_, _, discarded) = shell.behind_status().expect("the settle is still pending");
    assert!(discarded > 0, "the overlapped points were discarded");
    drop(loaded);
    let stored = std::fs::read_dir(&profiles).is_ok()
        && walk(&profiles)
            .iter()
            .any(|text| text.contains("dense_settled_keyed"));
    let _ = std::fs::remove_dir_all(&profiles);
    assert!(!stored, "no settle was stored from overlapped points");
}

fn walk(dir: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut at = vec![dir.to_path_buf()];
    while let Some(d) = at.pop() {
        for entry in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                at.push(path);
            } else if let Ok(text) = std::fs::read_to_string(&path) {
                out.push(text);
            }
        }
    }
    out
}

/// A request arriving before every fire of a settle slice: each slice
/// gives the device back before it fires, nothing is weighed or stored,
/// and the settle stays pending for a later idle turn.
#[test]
fn a_settle_slice_yields_to_an_arrival_before_it_fires() {
    let profiles = profiles_at("settle-yield");
    let boot = || backend::DeviceBoot {
        profiles: Some(profiles.clone()),
        ..crate::gpu::device_boot()
    };
    let Some(mut loaded) = loaded_boot("settle-yield", &Outputs, 4, boot()) else {
        return;
    };
    let behind = ::engine::BehindServing {
        others: &|| 0,
        arrived: &|| true,
        seats: None,
        step: None,
    };
    let mut slices = 0;
    while slices < 32 && loaded.engine.measure_behind(&behind).expect("measuring behind") {
        slices += 1;
    }
    let shell = loaded.engine.loaded_mut().expect("loaded");
    assert!(
        shell.settled().iter().all(|range| !range.checked),
        "no range was weighed: {:?}",
        shell.settled()
    );
    assert!(shell.behind_status().is_some(), "the settle is still pending");
    drop(loaded);
    let stored = walk(&profiles)
        .iter()
        .any(|text| text.contains("dense_settled_keyed"));
    let _ = std::fs::remove_dir_all(&profiles);
    assert!(!stored, "nothing was stored");
}
