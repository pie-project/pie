use crate::api::{ContractFor, DeviceBoot, Vulkan};

pub fn open(config_bytes: &[u8], contract_for: ContractFor) -> Result<Vulkan, String> {
    let doc: toml::Table = std::str::from_utf8(config_bytes)
        .map_err(|error| format!("the vulkan boot config is not utf-8: {error}"))?
        .parse()
        .map_err(|error| format!("the vulkan boot config is not TOML: {error}"))?;
    let profiles = profiles(&doc)?;
    Ok(Vulkan::new(
        DeviceBoot {
            device_index: device_index(&doc),
            validation: validation(&doc),
            pipeline_cache: pipeline_cache(&doc),
            profiles,
        },
        contract_for,
    ))
}

fn table(doc: &toml::Table) -> Option<&toml::Table> {
    doc.get("vulkan").and_then(toml::Value::as_table)
}

fn device_index(doc: &toml::Table) -> u32 {
    table(doc)
        .and_then(|t| t.get("device_index"))
        .and_then(toml::Value::as_integer)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(0)
}

fn validation(doc: &toml::Table) -> bool {
    table(doc)
        .and_then(|t| t.get("validation"))
        .and_then(toml::Value::as_bool)
        .unwrap_or(false)
}

fn pipeline_cache(doc: &toml::Table) -> Option<std::path::PathBuf> {
    table(doc)
        .and_then(|t| t.get("pipeline_cache"))
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(std::path::PathBuf::from)
}

fn profiles(doc: &toml::Table) -> Result<Option<std::path::PathBuf>, String> {
    let Some(vulkan) = table(doc) else {
        return Ok(None);
    };
    match vulkan.get("profiles") {
        None => Ok(None),
        Some(toml::Value::String(dir)) if dir.trim().is_empty() => Ok(None),
        Some(toml::Value::String(dir)) => Ok(Some(std::path::PathBuf::from(dir))),
        Some(other) => Err(format!(
            "[vulkan] profiles is a directory path, not {other}"
        )),
    }
}
