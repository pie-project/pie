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

/// An allocation past the device's largest buffer is refused by name before
/// the driver is asked, not as a device memory ceiling; the boot's
/// `max_buffer` lowers that largest as a smaller device's limit does.
#[test]
fn an_allocation_past_the_largest_buffer_is_refused_by_name() {
    let device = engine_vulkan::device::Context::bind(
        &engine_vulkan::api::DeviceBoot {
            max_buffer: Some(1 << 20),
            ..engine_vulkan::api::DeviceBoot::default()
        },
        engine::Allowance::WHOLE,
    )
    .expect("the first Vulkan device opens");
    assert_eq!(device.max_buffer(), 1 << 20);
    engine_vulkan::device::Buffer::zeroed(&device, 1 << 20, engine_gpu::books::owners::TUNING)
        .expect("a buffer at the limit is made");
    let refused =
        engine_vulkan::device::Buffer::zeroed(&device, (1 << 20) + 4, engine_gpu::books::owners::TUNING)
            .expect_err("a buffer past the limit is refused");
    let why = refused.to_string();
    assert!(
        why.contains("an allocation of 1048580 bytes exceeds this device's"),
        "{why}"
    );
    println!(
        "STATED device=\"{}\" unified_memory={} budget_source={:?}",
        device.name(),
        device.unified_memory(),
        device.budget_source()
    );
}
