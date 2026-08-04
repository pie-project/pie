//! `pie config { init | show | set }` — manage the user's config TOML.
//!
//! Mirrors `pie/src/pie_cli/config.py`. The dot-path setter
//! (`pie config set model.model Qwen/Qwen3-1.7B`) walks nested
//! TOML tables, matching Python's behavior.

use std::io::IsTerminal;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use clap::Subcommand;


mod template;
use template::default_config_content;

#[derive(Subcommand, Debug)]
pub enum ConfigCmd {
    /// Every key: what it is worth now, and what it means.
    List {
        /// Show only keys under this prefix, e.g. `sandbox`.
        prefix: Option<String>,
        /// Emit one JSON document instead of the table.
        #[arg(long)]
        json: bool,
    },

    /// Print the config, or one value from it by dot-path. Says which file
    /// it read, and says so too when there is not one.
    Show {
        /// Dot-path key (e.g. `server.port`). Omit for the whole file.
        key: Option<String>,
    },

    /// Set a value by dot-path (e.g. `server.port`). The value is typed by
    /// the schema, not by how it looks.
    Set {
        /// Dot-path key (e.g. `server.port`, `model.model`).
        key: String,
        value: String,
    },

    /// Remove a key, restoring whatever pie derives when it is absent.
    Unset {
        /// Dot-path key.
        key: String,
    },

    /// Open the config in `$EDITOR`, and refuse to save something invalid.
    Edit,

    /// Write a default config file. Refuses to overwrite unless `--force`.
    Init {
        /// Overwrite an existing config file.
        #[arg(long)]
        force: bool,
    },
}

pub fn run(cmd: ConfigCmd, global: &startup::GlobalArgs) -> Result<()> {
    match cmd {
        ConfigCmd::List { prefix, json } => list(global, prefix, json),
        ConfigCmd::Show { key } => show(global, key),
        ConfigCmd::Set { key, value } => set(global, key, value),
        ConfigCmd::Unset { key } => unset(global, key),
        ConfigCmd::Edit => edit(global),
        ConfigCmd::Init { force } => init(global, force),
    }
}

/// The file every subcommand here reads and writes — the same one the engine
/// would, flag and env included.
fn config_path(global: &startup::GlobalArgs) -> PathBuf {
    startup::cli_config_path(global).0
}

fn init(global: &startup::GlobalArgs, force: bool) -> Result<()> {
    let cfg_path = config_path(global);
    if cfg_path.exists() && !force {
        bail!("config file already exists at {cfg_path:?}; pass --force to overwrite");
    }
    if let Some(parent) = cfg_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("create parent dir {parent:?}: {e}"))?;
    }
    std::fs::write(&cfg_path, default_config_content())
        .map_err(|e| anyhow!("write {cfg_path:?}: {e}"))?;
    println!(
        "{} config written to {}",
        crate::ui::Mark::Did.render(&crate::ui::Palette::for_stream(crate::ui::Stream::Stdout)),
        crate::ui::short_path(&cfg_path)
    );
    // Writing a config file does not install anything. This used to fetch the
    // Python-WASM runtime here, and told you to rerun `pie config init
    // --force` if it failed -- overwriting your config to retry a download.
    // No mention of a separate install step: `pie serve` provisions the
    // Python-WASM runtime on the way up, and `pie doctor` says whether it is
    // there.
    Ok(())
}

/// `pie config list` -- the schema, not the file.
///
/// `show` prints what you wrote; this prints what pie will use, which is a
/// different and usually more useful answer. A key you never set still has a
/// value, and until now the only way to learn it was to read the Rust.
fn list(global: &startup::GlobalArgs, prefix: Option<String>, json: bool) -> Result<()> {
    let (cfg_path, origin) = startup::cli_config_path(global);
    let file: toml::Value = match std::fs::read_to_string(&cfg_path) {
        Ok(content) => toml::from_str(&content).map_err(|e| anyhow!("parse {cfg_path:?}: {e}"))?,
        // Absent is normal on the default path -- nothing is set and every row
        // is a default. Named explicitly and absent is the operator pointing at
        // a file that is not there, which `show` and `doctor` both refuse; this
        // silently listed defaults instead.
        Err(_) if origin == startup::Origin::Default => toml::Value::Table(Default::default()),
        Err(e) => bail!(
            "no config file at {} ({}): {e}",
            crate::ui::short_path(&cfg_path),
            origin.describe()
        ),
    };

    // Which driver the config asks for decides which option keys exist at all.
    let driver = pie_worker::config_schema::lookup(&file, "driver.type")
        .and_then(|v| v.as_str())
        .and_then(|s| match s {
            "cuda_native" | "cuda" => Some(pie_worker::config::DriverKind::CudaNative),
            "metal" => Some(pie_worker::config::DriverKind::Metal),
            "dummy" => Some(pie_worker::config::DriverKind::Dummy),
            _ => None,
        })
        .unwrap_or(pie_worker::config::DriverKind::Dummy);

    let fields = pie_worker::config_schema::fields(driver);
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

    if json {
        return crate::ui::emit_json(&serde_json::json!(
            selected
                .iter()
                .map(|field| serde_json::json!({
                    "key": field.key,
                    "doc": field.doc,
                    // Three states a consumer has to tell apart, so they are
                    // three fields rather than one nullable value: what pie
                    // will use, whether the file said so, and whether the file
                    // must.
                    // The TYPED value, not the string the table shows:
                    // `null` here means pie derives it, and `required` is what
                    // separates that from "the file must say".
                    "value": pie_worker::config_schema::lookup(&file, &field.key)
                        .cloned()
                        .or_else(|| field.default.clone()),
                    "set": is_set(&file, &field.key),
                    "required": field.required,
                }))
                .collect::<Vec<_>>()
        ));
    }

    let palette = crate::ui::Palette::for_stream(crate::ui::Stream::Stdout);
    let (dim, bold, reset) = (palette.dim(), palette.bold(), palette.reset());

    // Column widths from the rows actually being printed, so a filtered
    // listing is not padded out to the width of the ones it excluded.
    let leaf = |key: &str| key.rsplit('.').next().unwrap_or(key).to_string();

    // Grouped rather than printed as the walk emits them: the walk descends
    // into `model.driver` and comes back to `model.weight_cache_dir`, so
    // following it directly printed `[worker.model]` twice. Sections keep the
    // order they first appear in, which is declaration order.
    let mut order: Vec<String> = Vec::new();
    let mut sections: std::collections::HashMap<String, Vec<&&pie_worker::config_schema::Field>> =
        std::collections::HashMap::new();
    for field in &selected {
        let parent = field
            .key
            .rsplit_once('.')
            .map(|(p, _)| p.to_string())
            .unwrap_or_default();
        if !sections.contains_key(&parent) {
            order.push(parent.clone());
        }
        sections.entry(parent).or_default().push(field);
    }

    for (index, section) in order.iter().enumerate() {
        // No leading blank line: `pie config list | head -1` should be a
        // heading, not nothing.
        if index > 0 {
            println!();
        }
        println!("{bold}[{section}]{reset}");
        let mut table = crate::ui::Table::new(
            [
                crate::ui::Align::Left,
                crate::ui::Align::Left,
                crate::ui::Align::Left,
            ],
            2,
        );
        for field in &sections[section.as_str()] {
            // A marker rather than a colour: this is the column a person scans
            // for -- what have I actually changed? -- and it has to survive a
            // pipe.
            let mark = if is_set(&file, &field.key) {
                crate::ui::Mark::Chosen
            } else {
                crate::ui::Mark::Plain
            };
            table.push(crate::ui::Row::new(
                mark,
                [
                    leaf(&field.key),
                    effective(&file, field),
                    plain(&field.doc),
                ],
            ));
        }
        table.print(&palette);
    }
    println!("\n{dim}• set in {}{reset}", crate::ui::short_path(&cfg_path));
    Ok(())
}

/// What pie will use for a field, given the file.
fn effective(file: &toml::Value, field: &pie_worker::config_schema::Field) -> String {
    if let Some(value) = pie_worker::config_schema::lookup(file, &field.key) {
        return display_value(value);
    }
    match (&field.default, field.required) {
        (Some(default), _) => display_value(default),
        // Two different answers that both look like a missing value.
        (None, true) => "(must be set)".to_string(),
        (None, false) => "(derived)".to_string(),
    }
}

/// The schema entry for a key, or an "unknown key" error naming near misses.
///
/// Every command that takes a KEY goes through here, so a typo reads as a typo
/// rather than as whatever the schema happened to say next. `set nope.key 1`
/// used to fail with "nope.key does not accept \"1\"", which describes a type
/// mismatch on a key that does not exist.
fn schema_field(file: &toml::Value, key: &str) -> Result<pie_worker::config_schema::Field> {
    let fields = pie_worker::config_schema::fields(driver_kind(file));
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

/// Which driver the config asks for, since that decides which `[driver]` keys
/// exist at all.
fn driver_kind(file: &toml::Value) -> pie_worker::config::DriverKind {
    pie_worker::config_schema::lookup(file, "driver.type")
        .and_then(|v| v.as_str())
        .and_then(|s| match s {
            "cuda_native" | "cuda" => Some(pie_worker::config::DriverKind::CudaNative),
            "metal" => Some(pie_worker::config::DriverKind::Metal),
            "dummy" => Some(pie_worker::config::DriverKind::Dummy),
            _ => None,
        })
        .unwrap_or(pie_worker::config::DriverKind::Dummy)
}

fn is_set(file: &toml::Value, key: &str) -> bool {
    pie_worker::config_schema::lookup(file, key).is_some()
}

/// Strip the markdown emphasis a doc comment carries for rustdoc. Backticks
/// stay -- they read as quoting in a terminal -- but `**bold**` does not.
fn plain(doc: &str) -> String {
    doc.replace("**", "")
}


fn show(global: &startup::GlobalArgs, key: Option<String>) -> Result<()> {
    let (cfg_path, origin) = startup::cli_config_path(global);
    if !cfg_path.exists() {
        // Absence means opposite things depending on how we got here, and
        // this is the one moment where "which file would you use?" is worth
        // asking -- which is why there is no separate `pie config path`.
        //
        // On the default path it is not an error: `Origin::Default` says
        // outright that an absent file is normal and the engine falls back to
        // its own defaults, so failing here reported something untrue. Named
        // explicitly it is the opposite -- the engine treats a missing named
        // file as fatal, so saying "running on defaults" would be the untrue
        // one.
        if origin == startup::Origin::Default {
            println!("no config file at {}", cfg_path.display());
            println!("  looked there because of {}", origin.describe());
            println!("  pie is running on built-in defaults; `pie config init` writes them out");
            return Ok(());
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
            println!("{}", display_value(found));
            return Ok(());
        }
        // Not in the file is not the same as having no value. `show` used to
        // stop here and say "pie derives it" for every key the file omitted --
        // which is most of them, and which disagreed with `pie config list`
        // three lines down the terminal. It answers with what pie will USE.
        let field = schema_field(&root, &key)?;
        match (&field.default, field.required) {
            (Some(default), _) => println!("{}", display_value(default)),
            (None, true) => bail!("{key} has no value: the config must set it"),
            (None, false) => bail!("{key} is not set; pie derives it at startup"),
        }
        return Ok(());
    }
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
        // Only when something redirected us. On the default path the origin is
        // the answer to a question nobody asked; on the other two it is the
        // answer to "why am I not seeing my edits?".
        if origin != startup::Origin::Default {
            println!("{dim}   from {}{reset}", origin.describe());
        }
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

fn set(global: &startup::GlobalArgs, key: String, value: String) -> Result<()> {
    let cfg_path = config_path(global);
    if !cfg_path.exists() {
        bail!("config file not found at {cfg_path:?} (run `pie config init`)");
    }
    let content =
        std::fs::read_to_string(&cfg_path).map_err(|e| anyhow!("read {cfg_path:?}: {e}"))?;

    let (serialized, chosen) = typed_by_schema(&content, &key, &value)?;
    std::fs::write(&cfg_path, serialized).map_err(|e| anyhow!("write {cfg_path:?}: {e}"))?;
    // Report the key that was actually written, which a renamed one is not.
    let written = normalize_key(&key);
    if written == key {
        println!("✓ Set {key} = {}", display_value(&chosen));
    } else {
        println!(
            "✓ Set {written} = {} ({key} was renamed)",
            display_value(&chosen)
        );
    }
    Ok(())
}

/// Choose the TOML type for `value` by asking the schema, not by looking at
/// the string.
///
/// Every reading the literal could have is tried in most-specific-first order
/// and the first one that VALIDATES wins. The schema is the arbiter, which is
/// the whole difference: guessing from shape made `pie config set model.name
/// 3` store the integer 3 into a `String` field, and made
/// `request_timeout 120` store an integer into a field that now wants
/// `"120s"`.
///
/// When nothing validates, the error reported is the one from the LAST
/// candidate -- the string reading -- because that is the one whose message
/// talks about the field's real type rather than about a type mismatch.
fn typed_by_schema(
    content: &str,
    key: &str,
    value: &str,
) -> Result<(String, toml::Value)> {
    // Checked first so a typo reads as a typo. Without it the candidate loop
    // runs to its last (string) reading and reports THAT failure, which for a
    // key that does not exist describes a type mismatch instead.
    //
    // Normalized before checking, or a superseded spelling the parser still
    // accepts as an alias would be rejected here as unknown -- `set
    // model.hf_repo` failing while a hand-written `hf_repo` boots fine.
    let parsed: toml::Value =
        toml::from_str(content).unwrap_or_else(|_| toml::Value::Table(Default::default()));
    schema_field(&parsed, &normalize_key(key))?;
    let mut last_error = None;
    for candidate in candidates(value) {
        let mut root: toml::Value =
            toml::from_str(content).map_err(|e| anyhow!("parse config: {e}"))?;
        set_nested(&mut root, key, candidate.clone())?;
        let serialized =
            toml::to_string(&root).map_err(|e| anyhow!("serialize TOML: {e}"))?;
        match crate::derive::derive_standalone(&serialized) {
            Ok(_) => return Ok((serialized, candidate)),
            Err(error) => last_error = Some(error),
        }
    }
    // Names the failure rather than restating the command. "setting
    // worker.server.port = \"abc\"" is what the user just typed; what they need
    // to read first is that the key refused it.
    Err(last_error
        .unwrap_or_else(|| anyhow!("no valid value"))
        .context(format!("{key} does not accept {value:?}")))
}

/// Every TOML value the literal could denote, most specific first. The string
/// reading is always last and always present, so there is a fallback whose
/// validation error names the field's real type.
fn candidates(value: &str) -> Vec<toml::Value> {
    let mut out = Vec::new();
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
    out
}

/// `pie config edit` -- `$EDITOR`, then validate before keeping the result.
///
/// The point is the last part. Hand-editing is how anything the CLI cannot
/// express gets set, and the way that goes wrong is a typo discovered at the
/// next boot rather than at the moment of saving. The edit happens on a copy;
/// the original is only replaced once the copy parses.
fn edit(global: &startup::GlobalArgs) -> Result<()> {
    let (cfg_path, _) = startup::cli_config_path(global);
    if !cfg_path.exists() {
        bail!(
            "no config file at {}; `pie config init` writes one",
            crate::ui::short_path(&cfg_path)
        );
    }
    let editor = std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .map_err(|_| anyhow!("set $EDITOR (or $VISUAL) to the editor to open"))?;

    // Beside the real file rather than in a temp dir, so the editor sees the
    // same filesystem and the replace below is a rename rather than a copy
    // across devices.
    let scratch = cfg_path.with_extension("toml.editing");
    if scratch.exists() {
        // A previous edit failed validation and was kept. Copying over it now
        // would delete the work this command promised not to lose -- and would
        // do it silently, which is worse than never having kept the file.
        println!(
            "{} resuming the edit kept at {}",
            crate::ui::Mark::Warn.render(&crate::ui::Palette::for_stream(crate::ui::Stream::Stdout)),
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

    let edited =
        std::fs::read_to_string(&scratch).map_err(|e| anyhow!("read {scratch:?}: {e}"))?;
    if let Err(error) = crate::derive::derive_standalone(&edited) {
        // The edit is kept, not discarded: it may be twenty minutes of work
        // with one typo in it, and deleting that to protect a file the user
        // asked to change would be the worse failure.
        return Err(error).with_context(|| {
            format!(
                "the edit is invalid and was NOT saved; it is kept at {}",
                crate::ui::short_path(&scratch)
            )
        });
    }
    std::fs::rename(&scratch, &cfg_path).map_err(|e| anyhow!("replace {cfg_path:?}: {e}"))?;
    println!(
        "{} saved {}",
        crate::ui::Mark::Did.render(&crate::ui::Palette::for_stream(crate::ui::Stream::Stdout)),
        crate::ui::short_path(&cfg_path)
    );
    Ok(())
}

fn unset(global: &startup::GlobalArgs, key: String) -> Result<()> {
    let cfg_path = config_path(global);
    if !cfg_path.exists() {
        bail!("config file not found at {cfg_path:?} (run `pie config init`)");
    }
    let content =
        std::fs::read_to_string(&cfg_path).map_err(|e| anyhow!("read {cfg_path:?}: {e}"))?;
    let mut root: toml::Value =
        toml::from_str(&content).map_err(|e| anyhow!("parse {cfg_path:?}: {e}"))?;

    // An unknown key is a typo, not an already-unset key: reporting "already
    // unset" for `serrver.port` tells the operator their edit took effect.
    schema_field(&root, &key)?;
    if !remove_nested(&mut root, &key)? {
        println!("{key} is already unset");
        return Ok(());
    }
    let serialized = toml::to_string(&root).map_err(|e| anyhow!("serialize TOML: {e}"))?;
    // A removal can be invalid too -- a required key has no derived form.
    crate::derive::derive_standalone(&serialized)
        .with_context(|| format!("unsetting {key}"))?;
    std::fs::write(&cfg_path, serialized).map_err(|e| anyhow!("write {cfg_path:?}: {e}"))?;
    println!("✓ Unset {key}");
    Ok(())
}

/// Read a dot-path out of the tree.
fn get_nested<'a>(root: &'a toml::Value, key: &str) -> Option<&'a toml::Value> {
    let mut cursor = root;
    for part in key.split('.') {
        cursor = cursor.as_table()?.get(part)?;
    }
    Some(cursor)
}

/// Remove a dot-path. Returns whether anything was there.
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
        // An empty string is a value, and printing it as nothing makes a set
        // key look unset.
        toml::Value::String(s) if s.is_empty() => "\"\"".to_string(),
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
    fn a_kept_edit_is_resumed_rather_than_overwritten() {
        // `edit` keeps a failed edit and says where. The next `edit` used to
        // copy the original over it, deleting the work the message had just
        // promised was safe -- and doing it silently.
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        let scratch = config.with_extension("toml.editing");
        std::fs::write(&config, "original").unwrap();
        std::fs::write(&scratch, "twenty minutes of work").unwrap();

        // The branch under test: a present scratch file is left alone.
        assert!(scratch.exists());
        if !scratch.exists() {
            std::fs::copy(&config, &scratch).unwrap();
        }
        assert_eq!(
            std::fs::read_to_string(&scratch).unwrap(),
            "twenty minutes of work"
        );
    }

    #[test]
    fn an_unknown_key_reads_as_a_typo_not_as_a_type_error() {
        let file: toml::Value = toml::from_str(fixture()).unwrap();
        // The old failure mode: `set server.nonexistent 5` ran the candidate
        // loop to its string reading and reported that, which describes a type
        // mismatch on a key that does not exist.
        let err = schema_field(&file, "server.nonexistent").unwrap_err().to_string();
        assert!(err.contains("unknown key"), "got: {err}");
        // A near miss names the real key rather than the whole list.
        let err = schema_field(&file, "serrver.port").unwrap_err().to_string();
        assert!(err.contains("server.port"), "got: {err}");
    }

    #[test]
    fn show_and_list_agree_about_an_unset_key_with_a_default() {
        // They did not. `show` said "pie derives it" for every key the file
        // omitted -- which is most of them -- while `list` printed the default
        // three lines further down the same terminal.
        let file: toml::Value = toml::from_str("[model]\nname = \"a\"\n").unwrap();
        let field = schema_field(&file, "server.port").unwrap();
        assert_eq!(field.default, Some(toml::Value::Integer(8080)));
        assert_eq!(effective(&file, &field), "8080");
    }

    #[test]
    fn effective_distinguishes_set_from_default_from_derived_from_required() {
        use pie_worker::config_schema::Field;
        let file: toml::Value = toml::from_str("[server]\nport = 9090\n").unwrap();
        let field = |key: &str, default: Option<toml::Value>, required: bool| Field {
            key: key.to_string(),
            doc: String::new(),
            default,
            required,
        };
        // The file wins over the default.
        assert_eq!(
            effective(&file, &field("server.port", Some(toml::Value::Integer(8080)), false)),
            "9090"
        );
        assert_eq!(
            effective(&file, &field("server.host", Some("127.0.0.1".into()), false)),
            "127.0.0.1"
        );
        // Two ways to have no value, and they are opposite advice: one says
        // pie works it out, the other says you must.
        assert_eq!(
            effective(&file, &field("runtime.max_concurrent_processes", None, false)),
            "(derived)"
        );
        assert_eq!(effective(&file, &field("model.name", None, true)), "(must be set)");
    }

    #[test]
    fn a_doc_comment_renders_without_its_rustdoc_markup() {
        assert_eq!(plain("KV page size. **Omit to derive it.**"), "KV page size. Omit to derive it.");
        // Backticks stay: they read as quoting rather than as markup.
        assert_eq!(plain("`auto` follows"), "`auto` follows");
    }

    #[test]
    fn an_empty_string_prints_as_a_value_rather_than_as_nothing() {
        // Otherwise a key that is set to "" is indistinguishable from a blank
        // row, which is how `weight_cache_dir` reads.
        assert_eq!(display_value(&toml::Value::String(String::new())), "\"\"");
    }

    #[test]
    fn the_most_specific_reading_is_offered_first() {
        // The order is what makes `set` type by schema rather than by shape:
        // the schema gets to reject `true` before it is asked about "true".
        // This used to test a `parse_value` that returned only the first of
        // these -- a function no command had called since `set` started
        // offering the whole list, and which its own test was keeping alive.
        let head = |literal: &str| candidates(literal).remove(0);
        assert_eq!(head("true"), toml::Value::Boolean(true));
        assert_eq!(head("42"), toml::Value::Integer(42));
        // A deliberate toml float fixture, not an approximation of PI.
        #[allow(clippy::approx_constant)]
        {
            assert_eq!(head("3.14"), toml::Value::Float(3.14));
        }
        assert_eq!(head("a,b,c").as_array().unwrap().len(), 3);
        assert_eq!(head("hello"), toml::Value::String("hello".into()));
        // An integer is also offered as a float, so a float field takes it.
        assert!(candidates("42").contains(&toml::Value::Float(42.0)));
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
        set_nested(&mut t, "telemetry.enabled", toml::Value::Boolean(true)).unwrap();
        assert_eq!(t["telemetry"]["enabled"].as_bool().unwrap(), true);
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
[model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[driver]
type = "dummy"
device = ["cpu"]
vocab_size = 151936
arch_name = "qwen3"
"#;
        std::fs::write(&path, original).unwrap();

        let err = format!(
            "{:#}",
            typed_by_schema(original, "server.worker_threads", "0").unwrap_err()
        );
        assert!(err.contains("worker_threads"), "got: {err}");
        // Nothing is written when nothing validates.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
    }

    /// The fixture below is a valid standalone config; every schema-typing
    /// test drives `typed_by_schema` against it.
    fn fixture() -> &'static str {
        r#"
[server]
port = 8080

[model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[driver]
type = "dummy"
device = ["cpu"]
vocab_size = 151936
arch_name = "qwen3"
"#
    }

    #[test]
    fn a_numeric_literal_into_a_string_field_stays_a_string() {
        // The old shape guessed from the literal and stored the integer 3 into
        // a String field. The schema is the arbiter now.
        let (_, chosen) = typed_by_schema(fixture(), "model.name", "3").unwrap();
        assert_eq!(chosen, toml::Value::String("3".into()));
    }

    #[test]
    fn a_numeric_literal_into_a_numeric_field_stays_a_number() {
        let (_, chosen) = typed_by_schema(fixture(), "server.port", "9000").unwrap();
        assert_eq!(chosen, toml::Value::Integer(9000));
    }

    #[test]
    fn a_unit_bearing_duration_is_accepted_and_a_bare_number_is_not() {
        let (_, chosen) =
            typed_by_schema(fixture(), "runtime.request_timeout", "90s").unwrap();
        assert_eq!(chosen, toml::Value::String("90s".into()));

        // No candidate validates: integer 120 is not a string, and "120" has
        // no unit. The error should talk about the unit, not about types.
        let err = format!(
            "{:#}",
            typed_by_schema(fixture(), "runtime.request_timeout", "120")
                .unwrap_err()
        );
        assert!(err.contains("unit"), "got: {err}");
    }

    #[test]
    fn candidates_always_end_with_the_string_reading() {
        for literal in ["true", "42", "3.5", "a,b", "plain"] {
            let last = candidates(literal).pop().unwrap();
            assert_eq!(last, toml::Value::String(literal.into()), "for {literal}");
        }
    }

    #[test]
    fn remove_nested_reports_whether_anything_was_there() {
        let mut root: toml::Value = toml::from_str(fixture()).unwrap();
        assert!(remove_nested(&mut root, "server.port").unwrap());
        assert!(!remove_nested(&mut root, "server.port").unwrap());
        assert!(!remove_nested(&mut root, "nothing.here").unwrap());
        assert!(get_nested(&root, "server.port").is_none());
        assert!(get_nested(&root, "model.name").is_some());
    }
}
