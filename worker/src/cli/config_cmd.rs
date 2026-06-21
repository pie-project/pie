//! `pie config { init | show | set }` — manage the user's config TOML.
//!
//! Mirrors `pie/src/pie_cli/config.py`. The dot-path setter
//! (`pie config set model.0.hf_repo Qwen/Qwen3-1.7B`) understands
//! TOML array-of-tables indexing, matching Python's behavior.

use std::io::IsTerminal;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
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
        bail!("config file already exists at {cfg_path:?}; pass --force to overwrite");
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
    let content =
        std::fs::read_to_string(&cfg_path).map_err(|e| anyhow!("read {cfg_path:?}: {e}"))?;
    let cwd = std::env::current_dir().ok();
    let display = cwd
        .as_deref()
        .and_then(|c| cfg_path.strip_prefix(c).ok())
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| cfg_path.display().to_string());
    let colorize = std::io::stdout().is_terminal();
    if colorize {
        // Mimic the Python pie's `rich.Syntax(... title=path)` framing:
        // a thin separator line above and below labelled with the path.
        let dim = "\x1b[2m";
        let reset = "\x1b[0m";
        println!("{dim}── {display} ──{reset}");
        for line in content.lines() {
            println!("{}", colorize_toml_line(line));
        }
        println!("{dim}{}{reset}", "─".repeat(display.chars().count() + 6));
    } else {
        print!("{content}");
    }
    Ok(())
}

/// Colorize one line of TOML for an ANSI terminal. Mirrors the
/// "monokai"-ish palette `rich.Syntax(lexer="toml")` produces in the
/// Python pie's `pie config show`. Dependency-free — a tiny state
/// machine over the line characters is plenty for TOML's grammar.
fn colorize_toml_line(line: &str) -> String {
    const RESET: &str = "\x1b[0m";
    const COMMENT: &str = "\x1b[2;37m"; // dim grey
    const HEADER: &str = "\x1b[1;34m"; // bold blue
    const KEY: &str = "\x1b[36m"; // cyan
    const STRING: &str = "\x1b[32m"; // green
    const NUMBER: &str = "\x1b[33m"; // yellow
    const BOOL: &str = "\x1b[35m"; // magenta

    let trimmed_start = line.trim_start();
    let leading: String = line[..line.len() - trimmed_start.len()].to_string();

    // Whole-line comment.
    if trimmed_start.starts_with('#') {
        return format!("{leading}{COMMENT}{trimmed_start}{RESET}");
    }
    // Section header: [foo] / [[foo]].
    if trimmed_start.starts_with('[') {
        // Split off any trailing comment so it gets its own colour.
        let (head, tail) = split_trailing_comment(trimmed_start);
        let mut out = format!("{leading}{HEADER}{head}{RESET}");
        if let Some(c) = tail {
            out.push_str(&format!(" {COMMENT}{c}{RESET}"));
        }
        return out;
    }
    // key = value [# comment]
    let Some(eq) = trimmed_start.find('=') else {
        // No `=`: blank line or unrecognized — return as-is.
        return line.to_string();
    };
    let (key_part, rest) = trimmed_start.split_at(eq);
    let value_part = &rest[1..]; // drop '='
    let (value, comment) = split_trailing_comment(value_part);

    let mut out = String::new();
    out.push_str(&leading);
    out.push_str(KEY);
    out.push_str(key_part.trim_end());
    out.push_str(RESET);
    out.push_str(" = ");
    out.push_str(&colorize_value(value.trim_start(), STRING, NUMBER, BOOL));
    if let Some(c) = comment {
        out.push_str(&format!(" {COMMENT}{c}{RESET}"));
    }
    out
}

/// Split off a `#`-prefixed trailing comment, respecting `#` characters
/// inside double-quoted strings. Returns `(value, Option<comment>)`.
fn split_trailing_comment(s: &str) -> (&str, Option<&str>) {
    let mut in_string = false;
    for (i, ch) in s.char_indices() {
        match ch {
            '"' => in_string = !in_string,
            '#' if !in_string => return (s[..i].trim_end(), Some(s[i..].trim_end())),
            _ => {}
        }
    }
    (s.trim_end(), None)
}

fn colorize_value(v: &str, string: &str, number: &str, boolean: &str) -> String {
    const RESET: &str = "\x1b[0m";
    let trimmed = v.trim();
    if trimmed == "true" || trimmed == "false" {
        return format!("{boolean}{trimmed}{RESET}");
    }
    if trimmed.starts_with('"') {
        return format!("{string}{trimmed}{RESET}");
    }
    if trimmed.starts_with('[') {
        // Arrays: highlight individual elements, leaving brackets/commas
        // un-coloured. Cheap and good enough for typical config arrays.
        let inner = &trimmed[1..trimmed.len().saturating_sub(1)];
        let elems: Vec<String> = inner
            .split(',')
            .map(|e| colorize_value(e.trim(), string, number, boolean))
            .collect();
        return format!("[{}]", elems.join(", "));
    }
    if trimmed.parse::<f64>().is_ok() {
        return format!("{number}{trimmed}{RESET}");
    }
    trimmed.to_string()
}

fn set(key: String, value: String, path: Option<PathBuf>) -> Result<()> {
    let cfg_path = path.unwrap_or_else(paths::default_config_path);
    if !cfg_path.exists() {
        bail!("config file not found at {cfg_path:?} (run `pie config init`)");
    }
    let content =
        std::fs::read_to_string(&cfg_path).map_err(|e| anyhow!("read {cfg_path:?}: {e}"))?;
    let mut value_table: toml::Value =
        toml::from_str(&content).map_err(|e| anyhow!("parse {cfg_path:?}: {e}"))?;

    let parsed = parse_value(&value);
    set_nested(&mut value_table, &key, parsed.clone())?;

    let serialized = toml::to_string(&value_table).map_err(|e| anyhow!("serialize TOML: {e}"))?;
    let cfg: crate::config::Config =
        toml::from_str(&serialized).context("validating updated config")?;
    cfg.validate().context("validating updated config")?;
    std::fs::write(&cfg_path, serialized).map_err(|e| anyhow!("write {cfg_path:?}: {e}"))?;

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
        // `3.14` is a deliberate toml float fixture, not an approximation of PI.
        #[allow(clippy::approx_constant)]
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

    #[test]
    fn set_rejects_invalid_result_without_writing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");
        let original = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "dummy"
device = ["cpu"]

[model.driver.options]
vocab_size = 151936
arch_name = "qwen3"
"#;
        std::fs::write(&path, original).unwrap();

        let err = set(
            "runtime.worker_threads".to_string(),
            "0".to_string(),
            Some(path.clone()),
        )
        .unwrap_err();
        let err = format!("{err:#}");
        assert!(err.contains("worker_threads"), "got: {err}");
        assert_eq!(std::fs::read_to_string(path).unwrap(), original);
    }
}
