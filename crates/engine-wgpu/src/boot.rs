use crate::api::{ContractFor, DeviceBoot, Wgpu};

pub const DEFAULT_POWER_PREFERENCE: &str = "high-performance";

pub const DEFAULT_DEVICE_MEMORY: u64 = 8 << 30;

pub fn open(config_bytes: &[u8], contract_for: ContractFor) -> Result<Wgpu, String> {
    let doc = parse(config_bytes)?;
    Ok(Wgpu::new(device_boot(&doc)?, contract_for))
}

#[cfg(feature = "wgpu")]
pub fn open_with_device(
    config_bytes: &[u8],
    contract_for: ContractFor,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
) -> Result<Wgpu, String> {
    let doc = parse(config_bytes)?;
    Ok(Wgpu::with_device(
        device_boot(&doc)?,
        contract_for,
        crate::device::Handed {
            adapter,
            device,
            queue,
        },
    ))
}

#[cfg(feature = "wgpu")]
pub async fn request_device(
    config_bytes: &[u8],
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue), String> {
    let doc = parse(config_bytes)?;
    let handed = crate::device::request(&device_boot(&doc)?)
        .await
        .map_err(|fault| fault.to_string())?;
    Ok((handed.adapter, handed.device, handed.queue))
}

fn parse(config_bytes: &[u8]) -> Result<toml::Table, String> {
    std::str::from_utf8(config_bytes)
        .map_err(|error| format!("the wgpu boot config is not utf-8: {error}"))?
        .parse()
        .map_err(|error| format!("the wgpu boot config is not TOML: {error}"))
}

fn device_boot(doc: &toml::Table) -> Result<DeviceBoot, String> {
    Ok(DeviceBoot {
        adapter_index: adapter_index(doc),
        backends: backends(doc),
        power_preference: power_preference(doc),
        pipeline_cache: pipeline_cache(doc),
        device_memory: device_memory(doc),
        profiles: profiles(doc)?,
    })
}

fn table(doc: &toml::Table) -> Option<&toml::Table> {
    doc.get("wgpu").and_then(toml::Value::as_table)
}

fn adapter_index(doc: &toml::Table) -> u32 {
    table(doc)
        .and_then(|t| t.get("adapter_index"))
        .and_then(toml::Value::as_integer)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(0)
}

fn backends(doc: &toml::Table) -> Option<String> {
    table(doc)
        .and_then(|t| t.get("backends"))
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|list| !list.is_empty())
        .map(str::to_string)
}

fn power_preference(doc: &toml::Table) -> String {
    table(doc)
        .and_then(|t| t.get("power_preference"))
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or(DEFAULT_POWER_PREFERENCE)
        .to_string()
}

fn pipeline_cache(doc: &toml::Table) -> Option<std::path::PathBuf> {
    table(doc)
        .and_then(|t| t.get("pipeline_cache"))
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(std::path::PathBuf::from)
}

fn device_memory(doc: &toml::Table) -> Option<u64> {
    table(doc)
        .and_then(|t| t.get("device_memory"))
        .and_then(toml::Value::as_integer)
        .and_then(|v| u64::try_from(v).ok())
        .filter(|&v| v > 0)
}

fn profiles(doc: &toml::Table) -> Result<Option<std::path::PathBuf>, String> {
    let Some(wgpu) = table(doc) else {
        return Ok(None);
    };
    match wgpu.get("profiles") {
        None => Ok(None),
        Some(toml::Value::String(dir)) if dir.trim().is_empty() => Ok(None),
        Some(toml::Value::String(dir)) => Ok(Some(std::path::PathBuf::from(dir))),
        Some(other) => Err(format!("[wgpu] profiles is a directory path, not {other}")),
    }
}
