#![cfg(target_vendor = "apple")]

use kernels_metal::ane::{Element, Surface, available};

#[test]
fn the_neural_engine_bridge_resolves() {
    if let Err(why) = available() {
        eprintln!("skipping: {why}");
        return;
    }
    // Rows pad to 64 bytes, the allocation to 16 KB, and both engines see
    // the same bytes: what the host writes is what is at `base`.
    let surface = Surface::new(3, 100, Element::Int8).unwrap();
    assert_eq!(surface.element(), Element::Int8);
    assert_eq!(surface.stride(), 128);
    assert_eq!(surface.bytes(), 16384);
    assert_eq!(surface.strides(3), "[384, 384, 128, 1]");
    unsafe { *surface.base().add(2 * 128 + 99) = 0x5a };
    assert_eq!(unsafe { *surface.base().add(2 * 128 + 99) }, 0x5a);

    let halves = Surface::new(1, 2048, Element::Fp16).unwrap();
    assert_eq!(halves.stride(), 4096);
    assert_eq!(
        halves.buffer_type(1, 2048),
        "tensor_buffer<fp16, shape=[1, 1, 1, 2048], strides=[2048, 2048, 2048, 1], interleave_factors=[1, 1, 1, 1]>"
    );
}
