//! A load that verifies its fills (`DeviceBoot::verify_fills`) holds every
//! fill a lay-only pass lays (the arm walk's, a capturing fire's among them,
//! and the unwalked lengths' at serving) and every fill an arm derives from
//! the first's to the fire's own walk, byte for byte.

use crate::backend::DeviceBoot;
use crate::model_logits::{CONTEXT, PAGE, fixture, load_on};
use crate::runtime::laid_lengths::serve;

#[test]
fn every_laid_or_derived_fill_is_the_fires_own_walks() {
    let fixture = fixture("qwen3-micro-text", None);
    let device = DeviceBoot {
        verify_fills: true,
        ..crate::gpu::device_boot()
    };
    // Two arms: the second derives the first's fills.
    let runahead = engine::runahead::Runahead {
        frames_in_flight: 2,
    };
    let residency = engine::load::Residency::default();
    let Some(mut shell) = load_on(&fixture, PAGE, CONTEXT, &device, residency, runahead) else {
        return;
    };
    shell.record_bodies(None).expect("the load arms its bodies");
    let armed = shell.shelf.verified();
    eprintln!("the arm walk verified {armed:?}");
    assert!(armed.laid > 0, "the arm walk held no laid fill to a walk");
    assert!(armed.derived > 0, "the arm walk held no derived fill to a walk");
    shell.serving = true;
    serve(&mut shell, &fixture.trace);
    let served = shell.shelf.verified();
    eprintln!("and serving, {served:?}");
    assert!(
        served.laid > armed.laid,
        "no unwalked length's laid fill was held to a walk"
    );
    assert_eq!(served.mismatched, 0, "a laid or derived fill is not its walk's");
}
