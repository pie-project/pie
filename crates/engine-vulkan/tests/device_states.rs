//! What the Vulkan device the engine picks states of sparse binding and of
//! indexing an array of storage buffers, one line each under the device's
//! name. It loads no model and builds no kernel:
//!
//! `cargo test -p engine-vulkan --features vulkan --test device_states -- --nocapture`

#![cfg(feature = "vulkan")]

#[test]
fn the_device_states_sparse_binding_and_buffer_array_indexing() {
    let device = engine_vulkan::device::Context::bind(
        &engine_vulkan::api::DeviceBoot::default(),
        engine::Allowance::WHOLE,
    )
    .expect("the first Vulkan device opens");
    let stated = device.stated();
    assert!(!stated.is_empty());
    for (what, value) in stated {
        println!("STATED device=\"{}\" {what}={value}", device.name());
    }
}
