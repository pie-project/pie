//! Maps the operator-facing config file onto the structs pie reads; the two
//! are deliberately not the same shape (six file sections: server, model,
//! engine, runtime, sandbox, cluster). The mapping is data, not code, so
//! [`reshape`] and `config::schema` cannot disagree about it.

use anyhow::Result;

/// `(file path, internal path)` for every key whose two spellings differ.
///
/// Section moves are listed as whole sections where the whole section moves;
/// individual keys appear only where one key left its neighbours.
const MOVES: &[(&str, &str)] = &[
    ("engine", "model.engine"),
    ("server.telemetry", "telemetry.enabled"),
    ("server.otlp_endpoint", "telemetry.endpoint"),
    ("server.service_name", "telemetry.service_name"),
    ("cluster.max_clients", "executor.max_clients"),
    ("cluster.offload", "offload.enabled"),
    ("cluster.transfer", "offload.transfer"),
    (
        "cluster.prefill_min_suffix_tokens",
        "offload.prefill_min_suffix_tokens",
    ),
    (
        "cluster.max_outstanding_per_partner",
        "offload.max_outstanding_per_partner",
    ),
];

/// Keys `[engine]` carries that are common to every engine; everything else
/// in that section goes to the engine-specific options struct.
const ENGINE_COMMON: &[&str] = &["type", "device", "tensor_parallel_size", "activation_dtype"];

/// The file path a key is written as, given where it lives internally.
///
/// Used by `config::schema` so `pie config list` prints the paths `pie config
/// set` accepts.
pub fn to_file_path(internal: &str) -> String {
    // Longest internal prefix first, so a key that left its section on its own
    // is not rewritten by a whole-section entry that does not apply to it.
    let mut best: Option<(&str, &str)> = None;
    for (file, inner) in MOVES {
        if (internal == *inner || internal.starts_with(&format!("{inner}.")))
            && best.is_none_or(|(_, b)| inner.len() > b.len())
        {
            best = Some((file, inner));
        }
    }
    let moved = match best {
        Some((file, inner)) => {
            let rest = internal[inner.len()..].trim_start_matches('.');
            if rest.is_empty() {
                file.to_string()
            } else {
                format!("{file}.{rest}")
            }
        }
        None => internal.to_string(),
    };
    // The options table is flattened into its section rather than moved, so
    // the level disappears after the move rather than instead of it.
    moved.replace(".options.", ".")
}

/// Turn the file's tables into the shape `Config` deserializes from. One pass
/// over the leaves, mapping each full path once, so a moved key can't be
/// swept into a second place by a section rule applied afterward.
pub fn reshape(file: toml::Table) -> Result<toml::Table> {
    let mut leaves = Vec::new();
    collect_leaves(&toml::Value::Table(file), "", &mut leaves);

    let mut out = toml::Table::new();
    for (path, value) in leaves {
        insert_at(&mut out, &to_internal_path(&path), value)?;
    }

    // `[engine]` arrives whole; split it by name into the common fields and the
    // engine-specific bag that `type` decides the struct for.
    if let Some(engine) = out
        .get_mut("model")
        .and_then(|m| m.get_mut("engine"))
        .and_then(|e| e.as_table_mut())
    {
        let mut options = toml::Table::new();
        let specific: Vec<String> = engine
            .keys()
            .filter(|k| !ENGINE_COMMON.contains(&k.as_str()))
            .cloned()
            .collect();
        for key in specific {
            if let Some(value) = engine.remove(&key) {
                options.insert(key, value);
            }
        }
        engine.insert("options".to_string(), toml::Value::Table(options));
    }
    Ok(out)
}

/// The internal path for one full file path, longest file prefix first.
fn to_internal_path(file_path: &str) -> String {
    let mut best: Option<(&str, &str)> = None;
    for (file, inner) in MOVES {
        if (file_path == *file || file_path.starts_with(&format!("{file}.")))
            && best.is_none_or(|(b, _)| file.len() > b.len())
        {
            best = Some((file, inner));
        }
    }
    match best {
        Some((file, inner)) => {
            let rest = file_path[file.len()..].trim_start_matches('.');
            if rest.is_empty() {
                inner.to_string()
            } else {
                format!("{inner}.{rest}")
            }
        }
        None => file_path.to_string(),
    }
}

/// Insert `value` at a dotted path, merging tables rather than replacing them.
fn insert_at(root: &mut toml::Table, path: &str, value: toml::Value) -> Result<()> {
    let mut parts: Vec<&str> = path.split('.').collect();
    let last = parts.pop().expect("non-empty path");
    let mut cursor = root;
    for part in parts {
        let entry = cursor
            .entry(part.to_string())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()));
        cursor = entry
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("{path}: {part} is not a table"))?;
    }
    // A section that moves wholesale can land where per-key moves already put
    // something -- `[server]`'s telemetry keys and `[server]` itself both
    // target parts of the same tree.
    match (cursor.get_mut(last), value) {
        (Some(toml::Value::Table(existing)), toml::Value::Table(incoming)) => {
            for (k, v) in incoming {
                existing.insert(k, v);
            }
        }
        (_, value) => {
            cursor.insert(last.to_string(), value);
        }
    }
    Ok(())
}

/// Flatten a document to `(dotted path, scalar)` pairs.
fn collect_leaves(value: &toml::Value, prefix: &str, out: &mut Vec<(String, toml::Value)>) {
    match value {
        toml::Value::Table(table) if !table.is_empty() => {
            for (key, child) in table {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                collect_leaves(child, &path, out);
            }
        }
        // An empty table is a section header with nothing under it -- exactly
        // what `[controller]` and `[gateway]` always were.
        toml::Value::Table(_) => out.push((prefix.to_string(), value.clone())),
        _ => out.push((prefix.to_string(), value.clone())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_reads_back_as_the_path_the_file_spells_it() {
        // What `pie config list` prints has to be what `pie config set` takes.
        assert_eq!(
            to_file_path("runtime.request_timeout"),
            "runtime.request_timeout"
        );
        assert_eq!(
            to_file_path("sandbox.allow_network"),
            "sandbox.allow_network"
        );
        assert_eq!(to_file_path("model.engine.type"), "engine.type");
        assert_eq!(to_file_path("model.engine.device"), "engine.device");
        assert_eq!(to_file_path("model.weight_dtype"), "model.weight_dtype");
        assert_eq!(
            to_file_path("model.engine.options.gpu_mem_utilization"),
            "engine.gpu_mem_utilization"
        );
        assert_eq!(to_file_path("telemetry.enabled"), "server.telemetry");
        assert_eq!(to_file_path("offload.enabled"), "cluster.offload");
        // Unmoved keys are themselves.
        assert_eq!(to_file_path("model.name"), "model.name");
        assert_eq!(to_file_path("server.host"), "server.host");
    }
}
