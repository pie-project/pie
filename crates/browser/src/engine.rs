use anyhow::{Result, anyhow};

use crate::boot::BootConfig;

pub struct Device {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

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

pub fn open(config: &BootConfig, device: Device) -> Result<runtime::engine::EngineBox> {
    runtime::engine::backend::open::wgpu_on_device(
        boot_doc(config).as_bytes(),
        device.adapter,
        device.device,
        device.queue,
    )
}
