use eta_dsl::{Channel, Dtype};

#[test]
fn a_released_channel_is_gone_and_a_second_release_does_nothing() {
    let kept = Channel::new([4], Dtype::F32);
    let gone = Channel::from([1u32, 2, 3]);
    assert!(Channel::release(gone.gid()));
    assert!(Channel::by_gid(gone.gid()).is_none());
    assert!(!Channel::release(gone.gid()));
    assert!(Channel::by_gid(kept.gid()).is_some());
}
