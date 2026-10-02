//! The `[xla]` boot table.

use crate::api::{ContractFor, DeviceBoot, Xla};

pub const DEFAULT_MEM_UTILIZATION: f64 = 0.9;

pub fn open(config_bytes: &[u8], contract_for: ContractFor) -> Result<Xla, String> {
    let doc = parse(config_bytes)?;
    Ok(Xla::new(device_boot(&doc), contract_for))
}

fn parse(config_bytes: &[u8]) -> Result<toml::Table, String> {
    std::str::from_utf8(config_bytes)
        .map_err(|error| format!("the xla boot config is not utf-8: {error}"))?
        .parse()
        .map_err(|error| format!("the xla boot config is not TOML: {error}"))
}

fn table(doc: &toml::Table) -> Option<&toml::Table> {
    doc.get("xla").and_then(toml::Value::as_table)
}

fn device_boot(doc: &toml::Table) -> DeviceBoot {
    let t = table(doc);
    DeviceBoot {
        plugin: t
            .and_then(|t| t.get("plugin"))
            .and_then(toml::Value::as_str)
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(std::path::PathBuf::from),
        ordinal: t
            .and_then(|t| t.get("device"))
            .and_then(toml::Value::as_integer)
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0),
        mem_utilization: t
            .and_then(|t| t.get("mem_utilization"))
            .and_then(toml::Value::as_float)
            .filter(|f| f.is_finite() && *f > 0.0 && *f <= 1.0)
            .unwrap_or(DEFAULT_MEM_UTILIZATION),
    }
}
