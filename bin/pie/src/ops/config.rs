//! `pie config { init | show | set }` — manage the user's config TOML.
//!
//! Mirrors `pie/src/pie_cli/config.py`. The dot-path setter
//! (`pie config set model.model Qwen/Qwen3-1.7B`) walks nested
//! TOML tables, matching Python's behavior.

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

    /// Set a config value by dot-path (e.g. `model.model`).
    Set {
        /// Dot-path key (e.g. `server.port`, `model.model`).
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
    match crate::ops::py_runtime::ensure_installed(/*quiet=*/ false) {
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
    // Validate the whole standalone config (all three sections) before writing.
    crate::derive::derive_standalone(&serialized).context("validating updated config")?;
    std::fs::write(&cfg_path, serialized).map_err(|e| anyhow!("write {cfg_path:?}: {e}"))?;

    // Report the key that was actually written, which a renamed one is not.
    let written = normalize_key(&key);
    if written == key {
        println!("✓ Set {key} = {}", display_value(&parsed));
    } else {
        println!(
            "✓ Set {written} = {} ({key} was renamed)",
            display_value(&parsed)
        );
    }
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

/// Keys that were renamed, and what they are now.
///
/// The setter writes whatever key string it is handed, so without this a
/// `pie config set model.hf_repo …` on a config that already uses the new
/// spelling would leave *both* in the file — and `ModelConfig` accepts
/// `hf_repo` only as an alias for `model`, so serde rejects a document
/// carrying the two as a duplicate field. Normalizing here makes a config
/// converge on one spelling however it is edited.
/// Matched against the *end* of a dot-path, because the same table is reached
/// as `model.hf_repo` in a worker config and `worker.model.hf_repo` in a
/// standalone one.
const RENAMED_KEYS: [(&str, &str); 1] = [("model.hf_repo", "model.model")];

/// Walk a dot-path into the TOML tree, creating intermediate tables
/// as needed. Mirrors `pie_cli/config.py::_set_nested`.
fn set_nested(root: &mut toml::Value, key: &str, value: toml::Value) -> Result<()> {
    let key = normalize_key(key);
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
    let table = cursor
        .as_table_mut()
        .ok_or_else(|| anyhow!("{} is not a table", parts.join(".")))?;
    table.insert(last.to_string(), value);

    // Having written the current spelling, drop the superseded one from the
    // same table. `ModelConfig` takes the old name only as an alias, so a
    // document carrying both is a duplicate field rather than a preference —
    // and this is the one place that can notice.
    if parts.len() >= 2 {
        for (old, new) in RENAMED_KEYS {
            let (old_table, stale) = old.rsplit_once('.').expect("renamed keys are dotted");
            let (_, current) = new.rsplit_once('.').expect("renamed keys are dotted");
            if last == current && parts[parts.len() - 2] == old_table {
                table.remove(stale);
            }
        }
    }
    Ok(())
}

/// Rewrites a renamed dot-path to its current spelling, matching on the tail
/// so both the worker and standalone shapes are covered.
///
/// Public spellings outlive the code that reads them: someone has
/// `pie config set model.hf_repo …` in a script, and the rename is pie's
/// problem rather than theirs.
pub(crate) fn normalize_key(key: &str) -> String {
    for (old, new) in RENAMED_KEYS {
        if key == old {
            return new.to_string();
        }
        if let Some(prefix) = key.strip_suffix(old) {
            if prefix.ends_with('.') {
                return format!("{prefix}{new}");
            }
        }
    }
    key.to_string()
}

fn step<'a>(
    cursor: &'a mut toml::Value,
    part: &str,
    breadcrumb: &[&str],
) -> Result<&'a mut toml::Value> {
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
    fn set_nested_model_field() {
        let mut t: toml::Value = toml::from_str(
            r#"
[model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"
"#,
        )
        .unwrap();
        set_nested(
            &mut t,
            "model.model",
            toml::Value::String("meta-llama/Llama-3.2-1B".to_string()),
        )
        .unwrap();
        assert_eq!(
            t["model"]["model"].as_str().unwrap(),
            "meta-llama/Llama-3.2-1B"
        );
        // The old spelling is removed rather than left beside the new one:
        // `ModelConfig` takes `hf_repo` as an alias for `model`, and a document
        // carrying both is a duplicate field, not a preference.
        assert!(t["model"].get("hf_repo").is_none());
    }

    /// The renamed key still works from the command line, and lands on the new
    /// spelling — in both config shapes. Someone with
    /// `pie config set model.hf_repo …` in a script should not have to learn
    /// about the rename to keep working.
    #[test]
    fn the_old_model_key_sets_the_new_one() {
        for (prefix, table) in [("", "model"), ("worker.", "worker")] {
            let doc = if prefix.is_empty() {
                "[model]\nname = \"default\"\nhf_repo = \"old\"\n".to_string()
            } else {
                "[worker.model]\nname = \"default\"\nhf_repo = \"old\"\n".to_string()
            };
            let mut t: toml::Value = toml::from_str(&doc).unwrap();
            set_nested(
                &mut t,
                &format!("{prefix}model.hf_repo"),
                toml::Value::String("Qwen/Qwen3-1.7B".to_string()),
            )
            .unwrap();
            let model = if table == "model" {
                &t["model"]
            } else {
                &t["worker"]["model"]
            };
            assert_eq!(model["model"].as_str().unwrap(), "Qwen/Qwen3-1.7B");
            assert!(
                model.get("hf_repo").is_none(),
                "the old spelling survived beside the new one ({prefix}model)"
            );
        }
    }

    #[test]
    fn set_rejects_invalid_result_without_writing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");
        let original = r#"
[worker.model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[worker.model.driver]
type = "dummy"
device = ["cpu"]

[worker.model.driver.options]
vocab_size = 151936
arch_name = "qwen3"
"#;
        std::fs::write(&path, original).unwrap();

        let err = set(
            "worker.runtime.worker_threads".to_string(),
            "0".to_string(),
            Some(path.clone()),
        )
        .unwrap_err();
        let err = format!("{err:#}");
        assert!(err.contains("worker_threads"), "got: {err}");
        assert_eq!(std::fs::read_to_string(path).unwrap(), original);
    }
}
