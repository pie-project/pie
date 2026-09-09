use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};

use crate::ui::Answer;
use clap::Subcommand;

mod template;
pub mod tune;
use template::default_config_content;
pub use tune::Objective;

#[cfg(test)]
pub(crate) fn default_config_for_test() -> String {
    template::config_content_with_any_engine()
}

#[derive(Subcommand, Debug)]
pub enum ConfigCmd {
    List {
        prefix: Option<String>,
    },

    Show {
        key: Option<String>,
    },

    Set {
        key: String,
        value: String,
    },

    Unset {
        key: String,
    },

    Edit,

    Init {
        #[arg(long)]
        force: bool,
    },

    Tune(tune::TuneArgs),
}

pub async fn run(cmd: ConfigCmd, global: &bootstrap::GlobalArgs) -> Result<Answer> {
    match cmd {
        ConfigCmd::List { prefix } => list(global, prefix),
        ConfigCmd::Show { key } => show(global, key),
        ConfigCmd::Set { key, value } => set(global, key, value),
        ConfigCmd::Unset { key } => unset(global, key),
        ConfigCmd::Edit => edit(global),
        ConfigCmd::Init { force } => init(global, force),
        ConfigCmd::Tune(args) => tune::run(global, args).await,
    }
}

fn config_path(global: &bootstrap::GlobalArgs) -> PathBuf {
    bootstrap::cli_config_path(global).0
}

fn init(global: &bootstrap::GlobalArgs, force: bool) -> Result<Answer> {
    let cfg_path = config_path(global);
    if cfg_path.exists() && !force {
        bail!("config file already exists at {cfg_path:?}; pass --force to overwrite");
    }
    if let Some(parent) = cfg_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("create parent dir {parent:?}: {e}"))?;
    }
    let content = default_config_content()?;
    std::fs::write(&cfg_path, content).map_err(|e| anyhow!("write {cfg_path:?}: {e}"))?;
    let did = format!("config written to {}", crate::ui::short_path(&cfg_path));
    Ok(Answer::did(did))
}

fn list(global: &bootstrap::GlobalArgs, prefix: Option<String>) -> Result<Answer> {
    let (cfg_path, origin) = bootstrap::cli_config_path(global);
    let file: toml::Value = match std::fs::read_to_string(&cfg_path) {
        Ok(content) => toml::from_str(&content).map_err(|e| anyhow!("parse {cfg_path:?}: {e}"))?,
        Err(_) if origin == bootstrap::Origin::Default => toml::Value::Table(Default::default()),
        Err(e) => bail!(
            "no config file at {} ({}): {e}",
            crate::ui::short_path(&cfg_path),
            origin.describe()
        ),
    };

    let engine = worker::config::schema::lookup(&file, "engine.type")
        .and_then(|v| v.as_str())
        .and_then(|s| match s {
            "cuda_native" | "cuda" => Some(worker::config::EngineKind::CudaNative),
            "metal" => Some(worker::config::EngineKind::Metal),
            "vulkan" => Some(worker::config::EngineKind::Vulkan),
            "wgpu" => Some(worker::config::EngineKind::Wgpu),
            _ => None,
        })
        .unwrap_or(default_engine_kind());

    let fields = worker::config::schema::fields(engine);
    let selected: Vec<_> = fields
        .iter()
        .filter(|f| match &prefix {
            None => true,
            Some(p) => f.key == *p || f.key.starts_with(&format!("{p}.")),
        })
        .collect();
    if selected.is_empty() {
        bail!(
            "no keys under {:?}; `pie config list` shows all of them",
            prefix.unwrap_or_default()
        );
    }

    Ok(Answer::report(ConfigList {
        config_path: cfg_path,
        fields: selected
            .iter()
            .map(|field| ConfigField {
                key: field.key.to_string(),
                doc: field.doc.to_string(),
                value: worker::config::schema::lookup(&file, &field.key)
                    .cloned()
                    .or_else(|| field.default.clone()),
                set: is_set(&file, &field.key),
                required: field.required,
                shown: effective(&file, field),
            })
            .collect(),
    }))
}

#[derive(serde::Serialize)]
pub struct ConfigList {
    config_path: PathBuf,
    fields: Vec<ConfigField>,
}

#[derive(serde::Serialize)]
struct ConfigField {
    key: String,
    doc: String,
    value: Option<toml::Value>,
    set: bool,
    required: bool,
    shown: String,
}

impl crate::ui::Report for ConfigList {
    fn render(&self, palette: &crate::ui::Palette) {
        let leaf = |key: &str| key.rsplit('.').next().unwrap_or(key).to_string();

        let mut order: Vec<&str> = Vec::new();
        let mut sections: std::collections::HashMap<&str, Vec<&ConfigField>> =
            std::collections::HashMap::new();
        for field in &self.fields {
            let parent = field.key.rsplit_once('.').map(|(p, _)| p).unwrap_or("");
            if !sections.contains_key(parent) {
                order.push(parent);
            }
            sections.entry(parent).or_default().push(field);
        }

        for (index, section) in order.iter().enumerate() {
            if index > 0 {
                println!();
            }
            println!("{}", palette.bold(format!("[{section}]")));
            let mut table = crate::ui::Table::new(
                [
                    crate::ui::Align::Left,
                    crate::ui::Align::Left,
                    crate::ui::Align::Left,
                ],
                2,
            );
            for field in &sections[section] {
                let mark = if field.set {
                    crate::ui::Mark::Chosen
                } else {
                    crate::ui::Mark::Plain
                };
                table.push(crate::ui::Row::new(
                    mark,
                    [leaf(&field.key), field.shown.clone(), plain(&field.doc)],
                ));
            }
            table.print(palette);
        }
        println!(
            "\n{}",
            palette.dim(format!(
                "• set in {}",
                crate::ui::short_path(&self.config_path)
            ))
        );
    }
}

fn effective(file: &toml::Value, field: &worker::config::schema::Field) -> String {
    if let Some(value) = worker::config::schema::lookup(file, &field.key) {
        return display_value(value);
    }
    match (&field.default, field.required) {
        (Some(default), _) => display_value(default),
        (None, true) => "(must be set)".to_string(),
        (None, false) => "(derived)".to_string(),
    }
}

fn schema_field(file: &toml::Value, key: &str) -> Result<worker::config::schema::Field> {
    let fields = worker::config::schema::fields(engine_kind(file));
    if let Some(found) = fields.iter().find(|f| f.key == key) {
        return Ok(found.clone());
    }
    let leaf = key.rsplit('.').next().unwrap_or(key);
    let near: Vec<&str> = fields
        .iter()
        .filter(|f| f.key.ends_with(&format!(".{leaf}")) || f.key == leaf)
        .map(|f| f.key.as_str())
        .collect();
    if near.is_empty() {
        bail!("unknown key {key:?}; `pie config list` shows every key");
    }
    bail!("unknown key {key:?}; did you mean {}?", near.join(" or "))
}

fn engine_kind(file: &toml::Value) -> worker::config::EngineKind {
    worker::config::schema::lookup(file, "engine.type")
        .and_then(|v| v.as_str())
        .and_then(|s| match s {
            "cuda_native" | "cuda" => Some(worker::config::EngineKind::CudaNative),
            "metal" => Some(worker::config::EngineKind::Metal),
            "vulkan" => Some(worker::config::EngineKind::Vulkan),
            "wgpu" => Some(worker::config::EngineKind::Wgpu),
            _ => None,
        })
        .unwrap_or(default_engine_kind())
}

fn default_engine_kind() -> worker::config::EngineKind {
    worker::config::EngineKind::CudaNative
}

fn is_set(file: &toml::Value, key: &str) -> bool {
    worker::config::schema::lookup(file, key).is_some()
}

fn plain(doc: &str) -> String {
    doc.replace("**", "")
}

#[derive(serde::Serialize)]
#[serde(untagged)]
pub enum ConfigShow {
    File {
        path: PathBuf,
        origin: String,
        content: Option<String>,
        #[serde(skip)]
        display: String,
        #[serde(skip)]
        redirected: bool,
    },
    Value {
        key: String,
        value: toml::Value,
    },
}

impl crate::ui::Report for ConfigShow {
    fn render(&self, palette: &crate::ui::Palette) {
        match self {
            ConfigShow::Value { value, .. } => println!("{}", display_value(value)),
            ConfigShow::File {
                path,
                origin,
                content: None,
                ..
            } => {
                println!("no config file at {}", path.display());
                println!("  looked there because of {origin}");
                println!(
                    "  pie is running on built-in defaults; `pie config init` writes them out"
                );
            }
            ConfigShow::File {
                origin,
                content: Some(content),
                display,
                redirected,
                ..
            } => {
                if !palette.enabled() {
                    print!("{content}");
                    return;
                }
                println!("{}", palette.dim(format!("── {display} ──")));
                if *redirected {
                    println!("{}", palette.dim(format!("   from {origin}")));
                }
                for line in content.lines() {
                    println!("{}", crate::ui::toml_line(line, palette));
                }
                println!("{}", palette.dim("─".repeat(display.chars().count() + 6)));
            }
        }
    }
}

fn show(global: &bootstrap::GlobalArgs, key: Option<String>) -> Result<Answer> {
    let (cfg_path, origin) = bootstrap::cli_config_path(global);
    if !cfg_path.exists() {
        if origin == bootstrap::Origin::Default {
            return Ok(Answer::report(ConfigShow::File {
                path: cfg_path,
                origin: origin.describe().to_string(),
                content: None,
                display: String::new(),
                redirected: false,
            }));
        }
        bail!(
            "no config file at {} ({}); pie will not start without it",
            crate::ui::short_path(&cfg_path),
            origin.describe()
        );
    }
    let content =
        std::fs::read_to_string(&cfg_path).map_err(|e| anyhow!("read {cfg_path:?}: {e}"))?;

    if let Some(key) = key {
        let root: toml::Value =
            toml::from_str(&content).map_err(|e| anyhow!("parse {cfg_path:?}: {e}"))?;
        if let Some(found) = get_nested(&root, &key) {
            let value = found.clone();
            return Ok(Answer::report(ConfigShow::Value { key, value }));
        }
        let field = schema_field(&root, &key)?;
        return match (&field.default, field.required) {
            (Some(default), _) => Ok(Answer::report(ConfigShow::Value {
                key,
                value: default.clone(),
            })),
            (None, true) => bail!("{key} has no value: the config must set it"),
            (None, false) => bail!("{key} is not set; pie derives it at bootstrap"),
        };
    }
    let cwd = std::env::current_dir().ok();
    let display = cwd
        .as_deref()
        .and_then(|c| cfg_path.strip_prefix(c).ok())
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| cfg_path.display().to_string());
    Ok(Answer::report(ConfigShow::File {
        path: cfg_path,
        origin: origin.describe().to_string(),
        content: Some(content),
        display,
        redirected: origin != bootstrap::Origin::Default,
    }))
}

fn set(global: &bootstrap::GlobalArgs, key: String, value: String) -> Result<Answer> {
    let cfg_path = config_path(global);
    if !cfg_path.exists() {
        bail!("config file not found at {cfg_path:?} (run `pie config init`)");
    }
    let content =
        std::fs::read_to_string(&cfg_path).map_err(|e| anyhow!("read {cfg_path:?}: {e}"))?;

    let (serialized, chosen) = typed_by_schema(&content, &key, &value)?;
    std::fs::write(&cfg_path, serialized).map_err(|e| anyhow!("write {cfg_path:?}: {e}"))?;
    let value = display_value(&chosen);
    Ok(Answer::did(format!("set {key} = {value}")))
}

fn typed_by_schema(content: &str, key: &str, value: &str) -> Result<(String, toml::Value)> {
    let parsed: toml::Value =
        toml::from_str(content).unwrap_or_else(|_| toml::Value::Table(Default::default()));
    schema_field(&parsed, key)?;
    let mut errors: Vec<anyhow::Error> = Vec::new();
    for candidate in candidates(value) {
        let mut root: toml::Value =
            toml::from_str(content).map_err(|e| anyhow!("parse config: {e}"))?;
        set_nested(&mut root, key, candidate.clone())?;
        let serialized = toml::to_string(&root).map_err(|e| anyhow!("serialize TOML: {e}"))?;
        match crate::derive::derive_standalone(&serialized) {
            Ok(_) => {
                let mut doc: toml_edit::DocumentMut =
                    content.parse().map_err(|e| anyhow!("parse config: {e}"))?;
                set_nested_doc(&mut doc, key, &candidate)?;
                return Ok((doc.to_string(), candidate));
            }
            Err(error) => errors.push(error),
        }
    }
    let reported = match errors.iter().position(|e| !is_type_mismatch(e)) {
        Some(i) => Some(errors.swap_remove(i)),
        None => errors.pop(),
    };
    Err(reported
        .unwrap_or_else(|| anyhow!("no valid value"))
        .context(format!("{key} does not accept {value:?}")))
}

fn is_type_mismatch(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|e| e.to_string().contains("invalid type:"))
}

fn candidates(value: &str) -> Vec<toml::Value> {
    let mut out = Vec::new();
    if let Some(parsed) = parse_toml_literal(value) {
        out.push(parsed);
    }
    match value.to_ascii_lowercase().as_str() {
        "true" => out.push(toml::Value::Boolean(true)),
        "false" => out.push(toml::Value::Boolean(false)),
        _ => {}
    }
    if let Ok(n) = value.parse::<i64>() {
        out.push(toml::Value::Integer(n));
    }
    if let Ok(f) = value.parse::<f64>() {
        out.push(toml::Value::Float(f));
    }
    if value.contains(',') {
        out.push(toml::Value::Array(
            value
                .split(',')
                .map(|e| toml::Value::String(e.trim().to_string()))
                .collect(),
        ));
    }
    out.push(toml::Value::String(value.to_string()));
    out.dedup();
    out
}

fn parse_toml_literal(value: &str) -> Option<toml::Value> {
    let trimmed = value.trim();
    let bracketed = (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2)
        || (trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2);
    if !bracketed {
        return None;
    }
    let table: toml::Value = toml::from_str(&format!("value = {trimmed}")).ok()?;
    table.get("value").cloned()
}

fn edit(global: &bootstrap::GlobalArgs) -> Result<Answer> {
    let (cfg_path, _) = bootstrap::cli_config_path(global);
    if !cfg_path.exists() {
        bail!(
            "no config file at {}; `pie config init` writes one",
            crate::ui::short_path(&cfg_path)
        );
    }
    let editor = std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .map_err(|_| anyhow!("set $EDITOR (or $VISUAL) to the editor to open"))?;

    let scratch = cfg_path.with_extension("toml.editing");
    if scratch.exists() {
        println!(
            "{} resuming the edit kept at {}",
            crate::ui::Mark::Warn
                .render(&crate::ui::Palette::for_stream(crate::ui::Stream::Stdout)),
            crate::ui::short_path(&scratch)
        );
    } else {
        std::fs::copy(&cfg_path, &scratch).map_err(|e| anyhow!("prepare {scratch:?}: {e}"))?;
    }

    let status = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!("{editor} \"$1\"", editor = editor))
        .arg("sh")
        .arg(&scratch)
        .status()
        .map_err(|e| anyhow!("run {editor}: {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&scratch);
        bail!("{editor} exited with {status}; config unchanged");
    }

    let edited = std::fs::read_to_string(&scratch).map_err(|e| anyhow!("read {scratch:?}: {e}"))?;
    if let Err(error) = crate::derive::derive_standalone(&edited) {
        return Err(error).with_context(|| {
            format!(
                "the edit is invalid and was NOT saved; it is kept at {}",
                crate::ui::short_path(&scratch)
            )
        });
    }
    std::fs::rename(&scratch, &cfg_path).map_err(|e| anyhow!("replace {cfg_path:?}: {e}"))?;
    Ok(Answer::did(format!(
        "saved {}",
        crate::ui::short_path(&cfg_path)
    )))
}

fn unset(global: &bootstrap::GlobalArgs, key: String) -> Result<Answer> {
    let cfg_path = config_path(global);
    if !cfg_path.exists() {
        bail!("config file not found at {cfg_path:?} (run `pie config init`)");
    }
    let content =
        std::fs::read_to_string(&cfg_path).map_err(|e| anyhow!("read {cfg_path:?}: {e}"))?;
    let mut root: toml::Value =
        toml::from_str(&content).map_err(|e| anyhow!("parse {cfg_path:?}: {e}"))?;

    schema_field(&root, &key)?;
    if !remove_nested(&mut root, &key)? {
        return Ok(Answer::noop(format!("{key} was already unset")));
    }
    let serialized = toml::to_string(&root).map_err(|e| anyhow!("serialize TOML: {e}"))?;
    crate::derive::derive_standalone(&serialized).with_context(|| format!("unsetting {key}"))?;
    let mut doc: toml_edit::DocumentMut = content
        .parse()
        .map_err(|e| anyhow!("parse {cfg_path:?}: {e}"))?;
    remove_nested_doc(&mut doc, &key)?;
    std::fs::write(&cfg_path, doc.to_string()).map_err(|e| anyhow!("write {cfg_path:?}: {e}"))?;
    Ok(Answer::did(format!("unset {key}")))
}

fn get_nested<'a>(root: &'a toml::Value, key: &str) -> Option<&'a toml::Value> {
    let mut cursor = root;
    for part in key.split('.') {
        cursor = cursor.as_table()?.get(part)?;
    }
    Some(cursor)
}

fn remove_nested_doc(doc: &mut toml_edit::DocumentMut, key: &str) -> Result<bool> {
    let parts: Vec<&str> = key.split('.').collect();
    let (last, parents) = parts.split_last().ok_or_else(|| anyhow!("empty key"))?;
    let mut cursor = doc.as_table_mut();
    for part in parents {
        let Some(next) = cursor.get_mut(part).and_then(|item| item.as_table_mut()) else {
            return Ok(false);
        };
        cursor = next;
    }
    Ok(cursor.remove(last).is_some())
}

fn remove_nested(root: &mut toml::Value, key: &str) -> Result<bool> {
    let parts: Vec<&str> = key.split('.').collect();
    let (last, parents) = parts.split_last().ok_or_else(|| anyhow!("empty key"))?;
    let mut cursor = root;
    for part in parents {
        let Some(next) = cursor.as_table_mut().and_then(|t| t.get_mut(*part)) else {
            return Ok(false);
        };
        cursor = next;
    }
    let Some(table) = cursor.as_table_mut() else {
        return Ok(false);
    };
    Ok(table.remove(*last).is_some())
}

fn display_value(v: &toml::Value) -> String {
    match v {
        toml::Value::String(s) if s.is_empty() => "\"\"".to_string(),
        toml::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn set_nested(root: &mut toml::Value, key: &str, value: toml::Value) -> Result<()> {
    let parts: Vec<&str> = key.split('.').collect();
    if parts.is_empty() {
        bail!("empty key");
    }

    let mut cursor: &mut toml::Value = root;
    for (i, part) in parts.iter().take(parts.len() - 1).enumerate() {
        cursor = step(cursor, part, &parts[..=i])?;
    }

    let last = parts[parts.len() - 1];
    let table = cursor
        .as_table_mut()
        .ok_or_else(|| anyhow!("{} is not a table", parts.join(".")))?;
    table.insert(last.to_string(), value);
    Ok(())
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

fn set_nested_doc(doc: &mut toml_edit::DocumentMut, key: &str, value: &toml::Value) -> Result<()> {
    let parts: Vec<&str> = key.split('.').collect();
    if parts.is_empty() {
        bail!("empty key");
    }

    let mut cursor = doc.as_table_mut();
    for (i, part) in parts.iter().take(parts.len() - 1).enumerate() {
        let entry = cursor
            .entry(part)
            .or_insert_with(|| toml_edit::Item::Table(toml_edit::Table::new()));
        cursor = entry
            .as_table_mut()
            .ok_or_else(|| anyhow!("{} is not a table", parts[..=i].join(".")))?;
    }

    let last = parts[parts.len() - 1];
    let item = edit_item(value)?;
    match cursor.get_mut(last) {
        Some(slot) => {
            let decor = slot.as_value().map(|v| v.decor().clone());
            *slot = item;
            if let (Some(decor), Some(v)) = (decor, slot.as_value_mut()) {
                *v.decor_mut() = decor;
            }
        }
        None => {
            cursor.insert(last, item);
        }
    }
    Ok(())
}

fn edit_item(value: &toml::Value) -> Result<toml_edit::Item> {
    let mut wrapper = toml::value::Table::new();
    wrapper.insert("value".to_string(), value.clone());
    let text = toml::to_string(&toml::Value::Table(wrapper))
        .map_err(|e| anyhow!("serialize value: {e}"))?;
    let doc: toml_edit::DocumentMut = text.parse().map_err(|e| anyhow!("re-parse value: {e}"))?;
    doc.get("value")
        .cloned()
        .ok_or_else(|| anyhow!("value did not round-trip"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_every_case() {
        show_and_list_agree_about_an_unset_key_with_a_default();
        effective_distinguishes_set_from_default_from_derived_from_required();
        the_most_specific_reading_is_offered_first();
        set_rejects_invalid_result_without_writing();
        a_numeric_literal_into_a_string_field_stays_a_string();
        an_array_literal_is_stored_as_an_array();
        setting_a_value_keeps_the_comments_around_it();
    }

    fn show_and_list_agree_about_an_unset_key_with_a_default() {
        let file: toml::Value = toml::from_str("[model]\nname = \"a\"\n").unwrap();
        let field = schema_field(&file, "server.port").unwrap();
        assert_eq!(field.default, Some(toml::Value::Integer(8080)));
        assert_eq!(effective(&file, &field), "8080");
    }

    fn effective_distinguishes_set_from_default_from_derived_from_required() {
        use worker::config::schema::Field;
        let file: toml::Value = toml::from_str("[server]\nport = 9090\n").unwrap();
        let field = |key: &str, default: Option<toml::Value>, required: bool| Field {
            key: key.to_string(),
            doc: String::new(),
            default,
            required,
        };
        assert_eq!(
            effective(
                &file,
                &field("server.port", Some(toml::Value::Integer(8080)), false)
            ),
            "9090"
        );
        assert_eq!(
            effective(
                &file,
                &field("server.host", Some("127.0.0.1".into()), false)
            ),
            "127.0.0.1"
        );
        assert_eq!(
            effective(
                &file,
                &field("runtime.max_concurrent_processes", None, false)
            ),
            "(derived)"
        );
        assert_eq!(
            effective(&file, &field("model.name", None, true)),
            "(must be set)"
        );
    }

    fn the_most_specific_reading_is_offered_first() {
        let head = |literal: &str| candidates(literal).remove(0);
        assert_eq!(head("true"), toml::Value::Boolean(true));
        assert_eq!(head("42"), toml::Value::Integer(42));
        #[allow(clippy::approx_constant)]
        {
            assert_eq!(head("3.14"), toml::Value::Float(3.14));
        }
        assert_eq!(head("a,b,c").as_array().unwrap().len(), 3);
        assert_eq!(head("hello"), toml::Value::String("hello".into()));
        assert!(candidates("42").contains(&toml::Value::Float(42.0)));
    }

    fn set_rejects_invalid_result_without_writing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");
        let original = r#"
[model]
name = "default"
model = "Qwen/Qwen3-0.6B"

[engine]
type = "vulkan"
device = ["vulkan:0"]
"#;
        std::fs::write(&path, original).unwrap();

        let err = format!(
            "{:#}",
            typed_by_schema(original, "server.worker_threads", "0").unwrap_err()
        );
        assert!(err.contains("worker_threads"), "got: {err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
    }

    fn fixture() -> &'static str {
        r#"
[server]
port = 8080

[model]
name = "default"
model = "Qwen/Qwen3-0.6B"

[engine]
type = "vulkan"
device = ["vulkan:0"]
"#
    }

    fn a_numeric_literal_into_a_string_field_stays_a_string() {
        let (_, chosen) = typed_by_schema(fixture(), "model.name", "3").unwrap();
        assert_eq!(chosen, toml::Value::String("3".into()));
    }

    fn an_array_literal_is_stored_as_an_array() {
        let (written, chosen) =
            typed_by_schema(fixture(), "engine.device", r#"["metal:0"]"#).unwrap();
        assert_eq!(
            chosen,
            toml::Value::Array(vec![toml::Value::String("metal:0".into())])
        );
        let back: toml::Value = toml::from_str(&written).unwrap();
        assert_eq!(
            get_nested(&back, "engine.device")
                .and_then(|v| v.as_array())
                .map(|a| a.len()),
            Some(1),
            "written file should hold an array, got: {written}"
        );
    }

    fn setting_a_value_keeps_the_comments_around_it() {
        let annotated = r#"# pie configuration
# Delete a line to get the default back.

[server]
# The port the engine listens on.
port = 8080

[model]
name = "default"
model = "Qwen/Qwen3-0.6B"

[engine]
type = "vulkan"         # trailing note
device = ["vulkan:0"]
"#;
        let (written, _) = typed_by_schema(annotated, "server.port", "9090").unwrap();
        assert!(written.contains("# pie configuration"), "got: {written}");
        assert!(
            written.contains("# The port the engine listens on."),
            "got: {written}"
        );
        assert!(written.contains("# trailing note"), "got: {written}");
        assert!(written.contains("port = 9090"), "got: {written}");
        assert!(!written.contains("port = 8080"), "got: {written}");
    }
}
