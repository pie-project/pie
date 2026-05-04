//! `pie config { init | show | set }` — manage the user's config TOML.
//!
//! Mirrors `pie/src/pie_cli/config.py`. The dot-path setter
//! (`pie config set model.0.hf_repo Qwen/Qwen3-1.7B`) understands
//! TOML array-of-tables indexing, matching Python's behavior.

use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use clap::Subcommand;

use crate::paths;

mod template;
use template::default_config_content;

#[derive(Subcommand, Debug)]
pub enum ConfigCmd {
    /// Write a default config TOML to `~/.pie/config.toml` (or
    /// `--path`). Refuses to overwrite an existing file unless
    /// `--force` is passed.
    Init {
        #[arg(long)]
        path: Option<PathBuf>,
        #[arg(long)]
        force: bool,
    },

    /// Print the contents of the config TOML.
    Show {
        #[arg(long)]
        path: Option<PathBuf>,
    },

    /// Set a config value by dot-path. Numeric segments index into
    /// arrays-of-tables (e.g. `model.0.hf_repo`).
    Set {
        /// Dot-path key (e.g. `server.port`, `model.0.hf_repo`).
        key: String,
        /// Value to set. Parsed as bool / int / float / comma-list / str
        /// in that order.
        value: String,
        #[arg(long)]
        path: Option<PathBuf>,
    },
}

pub fn run(cmd: ConfigCmd) -> Result<()> {
    match cmd {
        ConfigCmd::Init { path, force } => init(path, force),
        ConfigCmd::Show { path } => show(path),
        ConfigCmd::Set { key, value, path } => set(key, value, path),
    }
}

fn init(path: Option<PathBuf>, force: bool) -> Result<()> {
    let cfg_path = path.unwrap_or_else(paths::default_config_path);
    if cfg_path.exists() && !force {
        bail!(
            "config file already exists at {cfg_path:?}; pass --force to overwrite"
        );
    }
    if let Some(parent) = cfg_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("create parent dir {parent:?}: {e}"))?;
    }
    std::fs::write(&cfg_path, default_config_content())
        .map_err(|e| anyhow!("write {cfg_path:?}: {e}"))?;
    println!("✓ Configuration file created at {cfg_path:?}");

    // Pre-fetch the Python WASM runtime so Python inferlets work
    // out of the box. Mirrors `pie/src/pie_cli/config.py::config_init`'s
    // explicit call to `bakery.py_runtime.ensure_installed()`.
    // Verbose here (not best-effort) so the user sees download
    // progress and any error message clearly.
    match crate::py_runtime::ensure_installed(/*quiet=*/ false) {
        Ok(_) => println!("✓ Python WASM runtime installed"),
        Err(e) => println!(
            "! Could not install Python WASM runtime: {e}\n  \
             Retry later with `pie config init --force`."
        ),
    }
    Ok(())
}

fn show(path: Option<PathBuf>) -> Result<()> {
    let cfg_path = path.unwrap_or_else(paths::default_config_path);
    if !cfg_path.exists() {
        bail!("config file not found at {cfg_path:?} (run `pie config init`)");
    }
    let content = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow!("read {cfg_path:?}: {e}"))?;
    print!("{content}");
    Ok(())
}

fn set(key: String, value: String, path: Option<PathBuf>) -> Result<()> {
    let cfg_path = path.unwrap_or_else(paths::default_config_path);
    if !cfg_path.exists() {
        bail!("config file not found at {cfg_path:?} (run `pie config init`)");
    }
    let content = std::fs::read_to_string(&cfg_path)
        .map_err(|e| anyhow!("read {cfg_path:?}: {e}"))?;
    let mut value_table: toml::Value = toml::from_str(&content)
        .map_err(|e| anyhow!("parse {cfg_path:?}: {e}"))?;

    let parsed = parse_value(&value);
    set_nested(&mut value_table, &key, parsed.clone())?;

    let serialized =
        toml::to_string(&value_table).map_err(|e| anyhow!("serialize TOML: {e}"))?;
    std::fs::write(&cfg_path, serialized)
        .map_err(|e| anyhow!("write {cfg_path:?}: {e}"))?;

    println!("✓ Set {key} = {}", display_value(&parsed));
    Ok(())
}

/// Parse a CLI string into the most specific TOML value it represents.
/// Order: bool → int → float → comma-list → string. Mirrors
/// `pie_cli/config.py::_parse_value`.
fn parse_value(s: &str) -> toml::Value {
    match s.to_ascii_lowercase().as_str() {
        "true" => return toml::Value::Boolean(true),
        "false" => return toml::Value::Boolean(false),
        _ => {}
    }
    if let Ok(n) = s.parse::<i64>() {
        return toml::Value::Integer(n);
    }
    if let Ok(f) = s.parse::<f64>() {
        return toml::Value::Float(f);
    }
    if s.contains(',') {
        // Comma-separated list — only flatten when every element is a
        // string. Mixed-type CSVs are rare and ambiguous; let the user
        // hand-edit the TOML for those.
        let elems: Vec<toml::Value> = s
            .split(',')
            .map(|e| toml::Value::String(e.trim().to_string()))
            .collect();
        return toml::Value::Array(elems);
    }
    toml::Value::String(s.to_string())
}

fn display_value(v: &toml::Value) -> String {
    match v {
        toml::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Walk a dot-path into the TOML tree, creating intermediate tables
/// as needed and respecting numeric segments as array indices.
/// Mirrors `pie_cli/config.py::_set_nested`.
fn set_nested(root: &mut toml::Value, key: &str, value: toml::Value) -> Result<()> {
    let parts: Vec<&str> = key.split('.').collect();
    if parts.is_empty() {
        bail!("empty key");
    }

    // Walk to the parent of the final segment.
    let mut cursor: &mut toml::Value = root;
    for (i, part) in parts.iter().take(parts.len() - 1).enumerate() {
        cursor = step(cursor, part, &parts[..=i])?;
    }

    // Set the final segment.
    let last = parts[parts.len() - 1];
    if let Ok(idx) = last.parse::<usize>() {
        let arr = cursor
            .as_array_mut()
            .ok_or_else(|| anyhow!("{} is not an array", parts.join(".")))?;
        if idx >= arr.len() {
            bail!(
                "index {idx} out of range (len={}) at {}",
                arr.len(),
                parts.join("."),
            );
        }
        arr[idx] = value;
    } else {
        let table = cursor
            .as_table_mut()
            .ok_or_else(|| anyhow!("{} is not a table", parts.join(".")))?;
        table.insert(last.to_string(), value);
    }
    Ok(())
}

fn step<'a>(
    cursor: &'a mut toml::Value,
    part: &str,
    breadcrumb: &[&str],
) -> Result<&'a mut toml::Value> {
    if let Ok(idx) = part.parse::<usize>() {
        let arr = cursor
            .as_array_mut()
            .ok_or_else(|| anyhow!("{} is not an array", breadcrumb.join(".")))?;
        if idx >= arr.len() {
            bail!(
                "index {idx} out of range (len={}) at {}",
                arr.len(),
                breadcrumb.join("."),
            );
        }
        return Ok(&mut arr[idx]);
    }
    let table = cursor
        .as_table_mut()
        .ok_or_else(|| anyhow!("{} is not a table", breadcrumb.join(".")))?;
    if !table.contains_key(part) {
        table.insert(part.to_string(), toml::Value::Table(Default::default()));
    }
    Ok(table.get_mut(part).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_value_preserves_type_order() {
        match parse_value("true") {
            toml::Value::Boolean(true) => {}
            v => panic!("expected bool, got {v:?}"),
        }
        match parse_value("42") {
            toml::Value::Integer(42) => {}
            v => panic!("expected int, got {v:?}"),
        }
        match parse_value("3.14") {
            toml::Value::Float(f) if (f - 3.14).abs() < 1e-9 => {}
            v => panic!("expected float, got {v:?}"),
        }
        match parse_value("a,b,c") {
            toml::Value::Array(a) if a.len() == 3 => {}
            v => panic!("expected array, got {v:?}"),
        }
        match parse_value("hello") {
            toml::Value::String(s) if s == "hello" => {}
            v => panic!("expected string, got {v:?}"),
        }
    }

    #[test]
    fn set_nested_top_level() {
        let mut t: toml::Value = toml::from_str("port = 8080\n").unwrap();
        set_nested(&mut t, "port", toml::Value::Integer(9090)).unwrap();
        assert_eq!(t["port"].as_integer().unwrap(), 9090);
    }

    #[test]
    fn set_nested_creates_intermediate_table() {
        let mut t: toml::Value = toml::from_str("").unwrap();
        set_nested(&mut t, "auth.enabled", toml::Value::Boolean(true)).unwrap();
        assert_eq!(t["auth"]["enabled"].as_bool().unwrap(), true);
    }

    #[test]
    fn set_nested_array_index() {
        let mut t: toml::Value = toml::from_str(
            r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"
"#,
        )
        .unwrap();
        set_nested(
            &mut t,
            "model.0.hf_repo",
            toml::Value::String("meta-llama/Llama-3.2-1B".to_string()),
        )
        .unwrap();
        assert_eq!(
            t["model"][0]["hf_repo"].as_str().unwrap(),
            "meta-llama/Llama-3.2-1B"
        );
    }

    #[test]
    fn set_nested_rejects_bad_index() {
        let mut t: toml::Value = toml::from_str(
            r#"
[[model]]
name = "default"
hf_repo = "x"
"#,
        )
        .unwrap();
        let err = set_nested(&mut t, "model.5.hf_repo", toml::Value::String("y".into()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("out of range"), "got: {err}");
    }
}
