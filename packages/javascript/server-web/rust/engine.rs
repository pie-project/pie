use anyhow::{Context, Result, anyhow};

use worker::embedded::Settings;

/// A WebGPU device with the boot document the engine opens on it.
pub struct Device {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    doc: String,
}

/// The page's boot config: the host-neutral `Settings` plus the two keys
/// only the WebGPU engine reads.
pub fn parse_config(text: &str) -> Result<(Settings, String)> {
    let mut document: serde_json::Value = if text.trim_start().starts_with('{') {
        serde_json::from_str(text)?
    } else {
        toml::from_str(text)?
    };
    let table = document
        .as_object_mut()
        .ok_or_else(|| anyhow!("the boot config is not a table"))?;
    let device_memory_mb: Option<u64> = table
        .remove("device_memory_mb")
        .map(serde_json::from_value)
        .transpose()?
        .flatten();
    let power_preference: String = table
        .remove("power_preference")
        .map(serde_json::from_value)
        .transpose()?
        .unwrap_or_else(|| "high-performance".into());
    let config: Settings = serde_json::from_value(document).context("parse the boot config")?;
    let mut doc = format!(
        "[wgpu]\nadapter_index = 0\ngpu_mem_utilization = {:?}\npower_preference = {}\n",
        config.gpu_mem_utilization,
        toml::Value::String(power_preference),
    );
    if let Some(mib) = device_memory_mb {
        doc.push_str(&format!("device_memory = {}\n", mib << 20));
    }
    Ok((config, doc))
}

pub async fn request(doc: String) -> Result<Device> {
    let (adapter, device, queue) = engine_wgpu::request_device(doc.as_bytes())
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
        doc,
    })
}

pub fn open(device: Device) -> Result<runtime::engine::EngineBox> {
    runtime::engine::backend::open::wgpu_on_device(
        device.doc.as_bytes(),
        device.adapter,
        device.device,
        device.queue,
    )
}
