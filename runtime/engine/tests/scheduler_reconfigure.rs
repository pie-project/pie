//! The batching knobs move after bootstrap, and only while nothing is running.
//!
//! Its own test binary on purpose. These knobs are process-global, and
//! `scheduler::frame`'s unit tests assert the untouched defaults in the
//! lib-test binary — writing them from a test that shares that process would
//! make one or the other fail depending on thread order.

use pie_engine::scheduler::{
    ReconfigureRefused, configured_frame_size, configured_submit_depth, reconfigure,
};

#[test]
fn the_knobs_are_no_longer_write_once() {
    // The regression this guards is the reason `pie config tune` exists in
    // the shape it does. These were `OnceLock`, first-writer-wins, so a sweep
    // could measure exactly one candidate per process — which put a model-size
    // ceiling on the whole plan, because a candidate then cost whatever the
    // weights cost to load.
    assert_eq!(configured_frame_size(), 2, "default k");
    assert_eq!(configured_submit_depth(), 3, "default window");

    reconfigure(4, 5, 3).expect("idle engine accepts new knobs");
    assert_eq!(configured_frame_size(), 4);
    assert_eq!(configured_submit_depth(), 5);

    // Twice, because once could be a lazily-initialised `OnceLock` taking its
    // first write and looking like a successful change.
    reconfigure(1, 2, 6).expect("second round");
    assert_eq!(configured_frame_size(), 1);
    assert_eq!(configured_submit_depth(), 2);
}

#[test]
fn a_refusal_names_what_is_still_running() {
    // The message is the whole interface here: a sweep that gets refused has to
    // be able to say why it could not measure the round it planned.
    let refused = ReconfigureRefused::Busy(7);
    let text = refused.to_string();
    assert!(text.contains('7'), "got: {text}");
    assert!(text.contains("idle"), "got: {text}");
}
