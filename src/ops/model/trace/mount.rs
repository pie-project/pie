//! `mount.json`: what the trace was built from. It names the models
//! directory, the stanzas the banks belong to in the order they first
//! appear, and the sha256 of every file the package read and of each
//! stanza's own file (`//lib/stanzas/{stanza}.poem`).

use std::path::Path;

use anyhow::{Context, Result};
use poem::Param;
use poem::star::Package;
use serde_json::{Map, Value, json};

/// `mount.json` for `package`, read from `dir`, whose trace registers `banks`.
pub fn of(package: &Package, dir: &Path, banks: &[&Param]) -> Result<Value> {
    let mut stanzas: Vec<&str> = Vec::new();
    for (stanza, _) in banks
        .iter()
        .filter_map(|bank| super::banks::stanza_of(&bank.name))
    {
        if !stanzas.contains(&stanza) {
            stanzas.push(stanza);
        }
    }
    let files: Map<String, Value> = package
        .files()
        .map(|(file, text)| (file.to_string(), json!(sha256(text))))
        .collect();
    let own: Map<String, Value> = stanzas
        .iter()
        .filter_map(|stanza| {
            let file = format!("//lib/stanzas/{stanza}.poem");
            let text = package.files().find(|(f, _)| *f == file)?.1;
            Some((stanza.to_string(), json!(sha256(text))))
        })
        .collect();
    let dir = std::fs::canonicalize(dir).with_context(|| format!("resolve {}", dir.display()))?;
    Ok(json!({
        "models_dir": dir,
        "stanzas": stanzas,
        "files": files,
        "stanza_sha256": own,
    }))
}

fn sha256(text: &str) -> String {
    ztensor::format::DigestAlgorithm::Sha256
        .digest(text.as_bytes())
        .hex()
}
