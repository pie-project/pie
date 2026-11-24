#![cfg(target_vendor = "apple")]
//! The resident-groups probe against the cores a device states: a device
//! that states none is priced at what the probe counts, so on one that does,
//! the probe's count is its cores, `RESIDENT_PER_CORE` full groups a core.

use engine_metal::device::{Context, RESIDENT_PER_CORE};

#[test]
fn the_resident_probe_counts_the_cores() {
    let Ok(device) = Context::bind() else {
        eprintln!("no Metal device; skipped");
        return;
    };
    let groups = device.resident_groups().expect("the probe runs");
    assert!(groups >= RESIDENT_PER_CORE, "the probe counts at least a core's groups");
    if let Some(cores) = device.cores() {
        assert_eq!(
            groups,
            cores * RESIDENT_PER_CORE,
            "{} keeps {groups} full groups resident on {cores} cores: not {RESIDENT_PER_CORE} a \
             core, so a device that states no cores would be priced wrong",
            device.name()
        );
    }
}
