//! The file the operator writes, and how it maps onto the structs pie reads.
//!
//! These are deliberately not the same shape. The old file was the struct
//! layout exposed verbatim, which is how `[controller]` and `[gateway]` -- two
//! sections empty in every single-node config, existing only because three role
//! libs each parse their own -- ended up as the first thing anyone saw. And how
//! `worker.` came to prefix all 44 keys, and how `[model.driver.options]` got to
//! be four levels deep.
//!
//! So the file is a user interface and the structs are an implementation, and
//! this module is the one place they meet. Six sections:
//!
//! ```toml
//! [server]    where it listens, what it fetches from, what it reports
//! [model]     which weights
//! [driver]    which hardware, and every knob that hardware has
//! [runtime]   batching and timeouts
//! [sandbox]   what an inferlet may do, and how big it may get
//! [cluster]   distributed only; absent for single-node
//! ```
//!
//! The mapping is data rather than code so that [`reshape`] and
//! `config_schema` cannot disagree about it -- one saying where a key goes and
//! the other saying what keys exist would put `pie config set` and `pie config
//! list` at odds.

use anyhow::{Result, bail};

/// `(file path, internal path)` for every key whose two spellings differ.
///
/// Section moves are listed as whole sections where the whole section moves;
/// individual keys appear only where one key left its neighbours.
const MOVES: &[(&str, &str)] = &[
    // `[runtime]` is the batching policy now. Reusing a name for different
    // contents is normally the worst kind of rename -- a reader assumes the two
    // are the same thing. It is safe here only because the old file spelled
    // this `[worker.runtime]`, and a bare `[worker]` is rejected outright: no
    // file can carry the old meaning into the new name.
    ("runtime", "model.scheduler"),
    // The box an inferlet runs in: its walls and its size, which used to be
    // split across `[runtime]` and nothing.
    ("sandbox", "runtime"),
    // The `wasm_` prefix drops off inside `[sandbox]`: the section already says
    // which box these size, and a prefix on every key of a section is the same
    // dead weight `worker.` was on every key of the file.
    ("sandbox.max_instances", "runtime.wasm_max_instances"),
    ("sandbox.max_memory", "runtime.wasm_max_memory"),
    ("sandbox.warm_memory", "runtime.wasm_warm_memory"),
    ("sandbox.warm_slots", "runtime.wasm_warm_slots"),
    // Hardware, lifted out of the model it happens to serve.
    ("driver", "model.driver"),
    // Observability is three keys, and they belong with the process that
    // emits them rather than in a section of their own.
    ("server.telemetry", "telemetry.enabled"),
    ("server.otlp_endpoint", "telemetry.endpoint"),
    ("server.service_name", "telemetry.service_name"),
    // Tuning that was in `[runtime]` because `[runtime]` was where tuning went.
    ("server.worker_threads", "runtime.worker_threads"),
    ("server.max_upload", "runtime.max_upload"),
    // Admission is scheduling.
    ("runtime.max_concurrent_processes", "server.max_concurrent_processes"),
    // Distributed serving was three sections for one deployment shape.
    ("cluster.max_clients", "executor.max_clients"),
    ("cluster.offload", "offload.enabled"),
    ("cluster.transfer", "offload.transfer"),
    ("cluster.prefill_min_suffix_tokens", "offload.prefill_min_suffix_tokens"),
    ("cluster.max_outstanding_per_partner", "offload.max_outstanding_per_partner"),
];

/// Keys `[driver]` carries that are common to every driver. Everything else in
/// that section is handed to the driver-specific options struct.
///
/// This list is why `[model.driver.options]` could go: the section is split by
/// name rather than by nesting, and the leftovers still land in a struct with
/// `deny_unknown_fields`, so a typo is still a loud failure -- now one that
/// names the driver it was rejected by.
const DRIVER_COMMON: &[&str] = &[
    "type",
    "device",
    "tensor_parallel_size",
    "activation_dtype",
    "random_seed",
];

/// The file path a key is written as, given where it lives internally.
///
/// Used by `config_schema` so `pie config list` prints the paths `pie config
/// set` accepts.
pub fn to_file_path(internal: &str) -> String {
    // Longest internal prefix first, so `runtime.worker_threads` is not
    // rewritten by the `runtime` -> ... entry that does not apply to it.
    let mut best: Option<(&str, &str)> = None;
    for (file, inner) in MOVES {
        if internal == *inner || internal.starts_with(&format!("{inner}.")) {
            if best.is_none_or(|(_, b)| inner.len() > b.len()) {
                best = Some((file, inner));
            }
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

/// Turn the file's tables into the shape `Config` deserializes from.
///
/// One pass over the leaves rather than a section pass and then a key pass:
/// with two passes the second one re-reads paths the first one had already
/// rewritten, so `[server] worker_threads` -- moved to `runtime.worker_threads`
/// by the key pass -- was then swept into `model.scheduler` by the section
/// rule for `runtime`. Mapping each full path once removes the ordering
/// question entirely.
pub fn reshape(file: toml::Table) -> Result<toml::Table> {
    for retired in ["controller", "gateway"] {
        if file.contains_key(retired) {
            bail!(
                "[{retired}] is no longer part of the config: single-node pie uses \
                 the role defaults, and a distributed deployment states what it \
                 needs in [cluster]."
            );
        }
    }
    if file.contains_key("worker") {
        bail!(
            "[worker] is no longer a section: its contents are the file."
        );
    }
    if file
        .get("model")
        .and_then(|m| m.as_table())
        .is_some_and(|t| t.contains_key("driver") || t.contains_key("scheduler"))
    {
        bail!(
            "[model.driver] / [model.scheduler] are now [driver] and [runtime]."
        );
    }

    let mut leaves = Vec::new();
    collect_leaves(&toml::Value::Table(file), "", &mut leaves);

    let mut out = toml::Table::new();
    for (path, value) in leaves {
        insert_at(&mut out, &to_internal_path(&path), value)?;
    }

    // `[driver]` arrives whole; split it by name into the common fields and the
    // driver-specific bag that `type` decides the struct for.
    if let Some(driver) = out
        .get_mut("model")
        .and_then(|m| m.get_mut("driver"))
        .and_then(|d| d.as_table_mut())
    {
        let mut options = toml::Table::new();
        let specific: Vec<String> = driver
            .keys()
            .filter(|k| !DRIVER_COMMON.contains(&k.as_str()))
            .cloned()
            .collect();
        for key in specific {
            if let Some(value) = driver.remove(&key) {
                options.insert(key, value);
            }
        }
        driver.insert("options".to_string(), toml::Value::Table(options));
    }
    Ok(out)
}

/// The internal path for one full file path, longest file prefix first.
fn to_internal_path(file_path: &str) -> String {
    let mut best: Option<(&str, &str)> = None;
    for (file, inner) in MOVES {
        if file_path == *file || file_path.starts_with(&format!("{file}.")) {
            if best.is_none_or(|(b, _)| file.len() > b.len()) {
                best = Some((file, inner));
            }
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
        assert_eq!(to_file_path("model.scheduler.request_timeout"), "runtime.request_timeout");
        assert_eq!(to_file_path("runtime.allow_network"), "sandbox.allow_network");
        assert_eq!(to_file_path("model.driver.type"), "driver.type");
        assert_eq!(
            to_file_path("model.driver.options.gpu_mem_utilization"),
            "driver.gpu_mem_utilization"
        );
        assert_eq!(to_file_path("telemetry.enabled"), "server.telemetry");
        assert_eq!(to_file_path("offload.enabled"), "cluster.offload");
        // Unmoved keys are themselves.
        assert_eq!(to_file_path("model.name"), "model.name");
        assert_eq!(to_file_path("server.host"), "server.host");
    }

    #[test]
    fn the_longest_prefix_wins() {
        // `runtime` -> `model.scheduler` and `runtime.worker_threads` ->
        // `server.worker_threads` both match an internal path starting
        // `runtime`; the specific one has to win or the tuning knob would be
        // reported as a scheduler key.
        assert_eq!(to_file_path("runtime.worker_threads"), "server.worker_threads");
        assert_eq!(to_file_path("runtime.max_upload"), "server.max_upload");
    }

    #[test]
    fn the_retired_sections_say_so_rather_than_being_ignored() {
        let mut file = toml::Table::new();
        file.insert("controller".into(), toml::Value::Table(Default::default()));
        let err = reshape(file).unwrap_err().to_string();
        // Names what to do instead, not only what is wrong.
        assert!(err.contains("[cluster]"), "got: {err}");
    }



    #[test]
    fn driver_keys_split_into_common_and_specific() {
        let file: toml::Table = toml::from_str(
            "[driver]\ntype = \"metal\"\ndevice = [\"metal:0\"]\ntotal_pages = 512\n",
        )
        .unwrap();
        let out = reshape(file).unwrap();
        let driver = out["model"]["driver"].as_table().unwrap();
        assert_eq!(driver["type"].as_str(), Some("metal"));
        // Everything the common list does not name lands in the bag `type`
        // picks the struct for -- which is what keeps deny_unknown_fields.
        assert_eq!(driver["options"]["total_pages"].as_integer(), Some(512));
        assert!(driver.get("total_pages").is_none());
    }
}
