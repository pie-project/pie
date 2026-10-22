//! `compiler.dag_streams` schedules DeepSeek-V4-Flash's regions (an op each)
//! onto the main and side streams by their dependencies: the side streams
//! carry work, every wait names an event a region before it closes, every
//! event closed is waited on (so a graph's fork groups settle), and a fire
//! ends on the main stream.

use std::collections::BTreeMap;

use poem_compiler::DeviceProfile;

use crate::common::bake_with;

#[test]
fn a_dag_schedule_orders_every_wait_after_its_close() {
    let row = poem_compiler::catalog::deployment("dsv4-flash-fp8b128a8-mxfp4a8-kv-bf16")
        .expect("the row is in the catalog");
    let fused = poem_compiler::fuse::fuse(
        row.trace(poem::Platform::Cuda),
        &["fused.pool_attention", "fused.lse_sink"],
    );
    let profile = DeviceProfile {
        dag_streams: true,
        side_streams: 2,
        ..DeviceProfile::default()
    };
    let compiled = bake_with(&fused, &profile).expect("the trace bakes with the schedule");
    let regions = &compiled.regions;
    assert!(compiled.streams.streams > 1, "the side streams carry work");
    let side = regions.iter().filter(|r| r.stream > 0).count();
    assert!(
        side > regions.len() / 10,
        "{side} of {} regions on side streams",
        regions.len()
    );

    let mut closed: BTreeMap<u32, usize> = BTreeMap::new();
    let mut waited: BTreeMap<u32, usize> = BTreeMap::new();
    for (at, region) in regions.iter().enumerate() {
        for event in &region.wait {
            let Some(&by) = closed.get(&event.0) else {
                panic!(
                    "region {at} waits on event {} no region before it closes",
                    event.0
                );
            };
            assert_ne!(
                regions[by].stream, region.stream,
                "region {at} waits on its own stream's region {by}"
            );
            waited.entry(event.0).or_insert(at);
        }
        if let Some(event) = region.close {
            assert!(
                closed.insert(event.0, at).is_none(),
                "event {} closed twice",
                event.0
            );
        }
    }
    for (event, by) in &closed {
        assert!(
            waited.get(event).is_some_and(|w| w > by),
            "event {event} (closed by region {by}) is never waited on"
        );
    }
    assert_eq!(
        regions.last().map(|r| r.stream),
        Some(0),
        "a fire ends on the main stream"
    );
}
