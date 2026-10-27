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
        &poem_compiler::fuse::rules::forming(&["fused.pool_attention", "fused.lse_sink"]),
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

    let vlm = poem_compiler::catalog::deployment("qwen35-d0.8b-vision-bf16-kv-bf16")
        .expect("the vision row is in the catalog");
    let vlm_compiled =
        bake_with(&vlm.trace(poem::Platform::Cuda), &profile).expect("the vision trace bakes");
    assert_eq!(vlm_compiled.units.len(), 2, "tower and trunk units");
    let mut vlm_closed: BTreeMap<u32, usize> = BTreeMap::new();
    for (at, region) in vlm_compiled.regions.iter().enumerate() {
        if region.phase != poem_compiler::Phase::Capture {
            assert_eq!(region.stream, 0, "prepare region {at} stays on stream 0");
            assert!(
                region.wait.is_empty() && region.open.is_none() && region.close.is_none(),
                "prepare region {at} carries no stream events"
            );
            continue;
        }
        for event in &region.wait {
            let Some(&by) = vlm_closed.get(&event.0) else {
                panic!("region {at} waits on unclosed event {}", event.0);
            };
            assert_eq!(
                vlm_compiled.unit_of(by),
                vlm_compiled.unit_of(at),
                "event {} crosses from unit {} (region {by}) to unit {} (region {at})",
                event.0,
                vlm_compiled.unit_of(by),
                vlm_compiled.unit_of(at)
            );
        }
        if let Some(event) = region.close {
            vlm_closed.insert(event.0, at);
        }
    }
    for unit in 0..vlm_compiled.units.len() as u32 {
        let span = vlm_compiled.unit_script(unit).expect("unit has capture regions");
        assert_eq!(
            vlm_compiled.regions[span.start as usize].stream, 0,
            "unit {unit} starts on stream 0"
        );
        assert_eq!(
            vlm_compiled.regions[span.end as usize - 1].stream, 0,
            "unit {unit} ends on stream 0"
        );
    }
}
