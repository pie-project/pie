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

#[test]
fn a_scopes_channels_are_released_with_it() {
    let kept = Channel::from([1.0f32]);
    let inner = Channel::released(|| Channel::from([2.0f32]).gid());
    assert!(Channel::by_gid(inner).is_none());
    let after = Channel::from([3.0f32]);
    assert!(Channel::by_gid(kept.gid()).is_some() && Channel::by_gid(after.gid()).is_some());
}
