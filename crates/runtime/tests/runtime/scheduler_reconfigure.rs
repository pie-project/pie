use runtime::scheduler::{
    ReconfigureRefused, channel_capacity, configured_dispatch_depth, configured_frame_size,
    configured_submit_depth, reconfigure,
};

#[test]
fn scheduler_reconfigure_every_case() {
    the_knobs_are_no_longer_write_once();
    a_refusal_names_what_is_still_running();
}

fn the_knobs_are_no_longer_write_once() {
    assert_eq!(configured_frame_size(), 2, "default k");
    assert_eq!(configured_dispatch_depth(), 2, "default horizon");
    assert_eq!(configured_submit_depth(), 3, "default window");
    assert_eq!(channel_capacity(), 7, "3 frames x k=2, plus the margin");

    reconfigure(4, 3).expect("idle runtime accepts new knobs");
    assert_eq!(configured_frame_size(), 4);
    assert_eq!(configured_dispatch_depth(), 3);
    assert_eq!(configured_submit_depth(), 4);
    assert_eq!(channel_capacity(), 17);

    reconfigure(1, 6).expect("second round");
    assert_eq!(configured_frame_size(), 1);
    assert_eq!(configured_dispatch_depth(), 6);
    assert_eq!(configured_submit_depth(), 7);
}

fn a_refusal_names_what_is_still_running() {
    let refused = ReconfigureRefused::Busy(7);
    let text = refused.to_string();
    assert!(text.contains('7'), "got: {text}");
    assert!(text.contains("idle"), "got: {text}");
}
