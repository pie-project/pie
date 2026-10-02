//! The `[cerebras]` boot table.

use crate::api::{Cerebras, ContractFor, DeviceBoot};

pub fn open(config_bytes: &[u8], contract_for: ContractFor) -> Result<Cerebras, String> {
    let doc = parse(config_bytes)?;
    Ok(Cerebras::new(device_boot(&doc)?, contract_for))
}

fn parse(config_bytes: &[u8]) -> Result<toml::Table, String> {
    std::str::from_utf8(config_bytes)
        .map_err(|error| format!("the cerebras boot config is not utf-8: {error}"))?
        .parse()
        .map_err(|error| format!("the cerebras boot config is not TOML: {error}"))
}

fn table(doc: &toml::Table) -> Option<&toml::Table> {
    doc.get("cerebras").and_then(toml::Value::as_table)
}

fn device_boot(doc: &toml::Table) -> Result<DeviceBoot, String> {
    let t = table(doc);
    let target = match t
        .and_then(|t| t.get("target"))
        .and_then(toml::Value::as_str)
    {
        None | Some("wse3") => crate::sdk::Target::Wse3,
        Some("wse2") => crate::sdk::Target::Wse2,
        Some(other) => {
            return Err(format!(
                "the cerebras boot config names target {other:?}; wse2 or wse3"
            ));
        }
    };
    Ok(DeviceBoot {
        target,
        cmaddr: t
            .and_then(|t| t.get("cmaddr"))
            .and_then(toml::Value::as_str)
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(str::to_string),
        num_threads: t
            .and_then(|t| t.get("num_threads"))
            .and_then(toml::Value::as_integer)
            .and_then(|v| u32::try_from(v).ok())
            .filter(|v| *v > 0)
            .unwrap_or(16),
        ordinal: t
            .and_then(|t| t.get("device"))
            .and_then(toml::Value::as_integer)
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0),
    })
}
