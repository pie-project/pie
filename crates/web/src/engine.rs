//! The WebGPU engine on the page's device.
//!
//! Requesting an adapter and a device are promises in a browser, so the
//! engine cannot do it inside its synchronous `open`. The host awaits the
//! engine's own request routine (one feature and limit policy for both
//! hosts) and hands the result to `open_with_device`.

use anyhow::{Result, anyhow};

use crate::boot::BootConfig;

/// An adapter and device the page requested, with its queue.
pub struct Device {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// The `[wgpu]` boot document the engine parses.
fn boot_doc(config: &BootConfig) -> String {
    let mut doc = format!(
        "[wgpu]\nadapter_index = 0\ngpu_mem_utilization = {:?}\npower_preference = {}\n",
        config.gpu_mem_utilization,
        toml::Value::String(config.power_preference.clone()),
    );
    if let Some(mib) = config.device_memory_mb {
        doc.push_str(&format!("device_memory = {}\n", mib << 20));
    }
    doc
}

/// Request the browser's adapter and a device with every limit the adapter
/// offers: several kernels bind more storage buffers than WebGPU's floor
/// guarantees, and a device created at the default limits would refuse them.
pub async fn request(config: &BootConfig) -> Result<Device> {
    let (adapter, device, queue) = engine_wgpu::request_device(boot_doc(config).as_bytes())
        .await
        .map_err(|e| anyhow!("{e}"))?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    tracing::info!(
        name = %info.name,
        backend = ?info.backend,
        max_buffer = limits.max_buffer_size,
        max_binding = limits.max_storage_buffer_binding_size,
        storage_buffers = limits.max_storage_buffers_per_shader_stage,
        features = ?device.features(),
        "WebGPU device"
    );
    Ok(Device {
        adapter,
        device,
        queue,
    })
}

/// The engine itself, on the device.
pub fn open(config: &BootConfig, device: Device) -> Result<runtime::engine::EngineBox> {
    runtime::engine::backend::open::wgpu_on_device(
        boot_doc(config).as_bytes(),
        device.adapter,
        device.device,
        device.queue,
    )
}
