//! A model's first boot, which times its curves and settles, and its next,
//! which reads them back, name the same kernels and look the settle up by
//! the same key: the second weighs nothing. (A load that times its curves
//! once named a decode's split walk out on wgpu: the admitted decodes were
//! dropped with the tuning the curves rebuilt, and the key moved.)

use super::outputs::{Outputs, loaded_boot};
use crate::backend;

#[test]
fn a_measured_boot_and_the_next_name_the_same_kernels_under_one_settle_key() {
    let profiles = std::env::temp_dir().join(format!(
        "pie-{}-settle-key-profiles-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&profiles);
    let boot = || backend::DeviceBoot {
        profiles: Some(profiles.clone()),
        ..crate::gpu::device_boot()
    };
    let read = |loaded: &mut super::outputs::Loaded| {
        let shell = loaded.engine.loaded_mut().expect("loaded");
        (
            shell.settle_keyed().expect("the settle ran"),
            shell.asked_kernels(),
            shell.settled().iter().any(|range| range.checked),
        )
    };
    let Some(mut first) = loaded_boot("settle-key", &Outputs, 4, boot()) else {
        return;
    };
    let (key, kernels, weighed) = read(&mut first);
    drop(first);
    assert!(weighed, "the first boot settles");
    let mut second = loaded_boot("settle-key", &Outputs, 4, boot()).expect("the device opened once");
    let (again, kernels_again, weighed_again) = read(&mut second);
    drop(second);
    let _ = std::fs::remove_dir_all(&profiles);
    assert_eq!(
        kernels_again, kernels,
        "the boot that read its curves names what the boot that timed them named"
    );
    assert_eq!(again, key, "one settle key");
    assert!(!weighed_again, "the stored settle is taken: nothing weighed again");
}
