//! The `[worker]` config schema, read out of `config.rs` itself.
//!
//! `pie config list` prints every key a person may set, what it is currently
//! worth, and what it means. The first two come from serde: the key path is
//! how `Config` nests and the value from serializing `Config::default()`. The
//! third has no runtime representation at all -- Rust discards doc comments
//! long before anything can ask for them -- so this module parses them back
//! out of the source with [`include_str!`].
//!
//! Parsing Rust with string matching is crude, and it is still the honest
//! option here. The alternative sources for a description are a hand-written
//! table or a docs page, and both are copies that are free to drift from the
//! field. `configuration.mdx` documents four keys that do not exist, which is
//! what that drift looks like once it has had time. A doc comment cannot
//! describe the wrong field, because it is attached to the field.
//!
//! What keeps the crudeness from being silent: `Config::parse` is the schema's
//! only real authority, and `schema_covers_exactly_the_settable_keys` checks
//! this walk against what serde will actually accept. If the parse breaks, the
//! test fails rather than the listing quietly shrinking.

use std::collections::BTreeMap;

use crate::config::{Config, EngineKind};

const SOURCE: &str = include_str!("config.rs");

/// One settable key.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// Dotted path as `pie config set` spells it, e.g. `worker.server.port`.
    pub key: String,
    /// The field's summary paragraph, joined into one line. Empty only if the
    /// doc-comment test has been circumvented.
    pub doc: String,
    /// What the field is worth when the file does not say. `None` means the
    /// absence is the setting -- pie derives the value at bootstrap and there is
    /// no constant to print.
    ///
    /// Always `None` when `required`, which is the other reason a key can have
    /// no default and a different answer: there the file must say.
    pub default: Option<toml::Value>,
    /// The file has to carry this one. Distinguished from a derived field
    /// because "you must set this" and "pie works this out" are opposite
    /// advice, and both would otherwise print as a missing default.
    pub required: bool,
}

/// A field as it appears in the source, before nesting is resolved.
struct Parsed {
    name: String,
    ty: String,
    doc: String,
    skip: bool,
    /// No serde `default`, from the field or from the struct: omitting it is
    /// a parse error rather than a choice.
    required: bool,
}

/// Pull `(struct -> fields)` out of the source text.
///
/// Only `pub` fields of `pub struct`s, and only where an identifier is
/// followed immediately by `:` -- without that last part this also matches
/// `pub const fn from_secs(s: u64)` in the newtypes' impl blocks.
fn parse_structs() -> BTreeMap<String, Vec<Parsed>> {
    let mut out: BTreeMap<String, Vec<Parsed>> = BTreeMap::new();
    let mut current = String::new();
    let mut doc: Vec<String> = Vec::new();
    let mut attrs: Vec<String> = Vec::new();
    // `#[serde(default)]` on the struct makes every one of its fields
    // optional, so field attributes alone do not decide requiredness.
    let mut struct_defaults_all = false;

    for line in SOURCE.lines() {
        let trimmed = line.trim_start();

        if let Some(rest) = line.strip_prefix("pub struct ") {
            struct_defaults_all = attrs.iter().any(|a| {
                a.starts_with("#[serde(") && (a.contains("(default,") || a.contains("(default)"))
            });
            current = rest
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .next()
                .unwrap_or("")
                .to_string();
            doc.clear();
            attrs.clear();
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("/// ") {
            doc.push(rest.to_string());
            continue;
        }
        if trimmed == "///" {
            // A blank doc line ends the summary paragraph. Everything after it
            // is the rationale, which belongs in the source and not in a table
            // of 80-odd rows.
            doc.push(String::new());
            continue;
        }
        if trimmed.starts_with("#[") {
            attrs.push(trimmed.to_string());
            continue;
        }

        if !current.is_empty()
            && let Some(rest) = line.strip_prefix("    pub ")
        {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty()
                && let Some(ty) = rest[name.len()..].strip_prefix(':')
            {
                let attr_text = attrs.join(" ");
                let renamed = attr_text
                    .split_once("rename = \"")
                    .and_then(|(_, rest)| rest.split_once('"'))
                    .map(|(name, _)| name.to_string());
                let summary: Vec<&String> =
                    doc.iter().take_while(|line| !line.is_empty()).collect();
                out.entry(current.clone()).or_default().push(Parsed {
                    name: renamed.unwrap_or(name),
                    ty: ty.trim().trim_end_matches(',').to_string(),
                    doc: summary
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    // A skipped field is populated by pie from somewhere else,
                    // so listing it would offer a key that `set` cannot honour.
                    skip: attr_text.contains("skip)") || attr_text.contains("skip,"),
                    required: !struct_defaults_all && !attr_text.contains("default"),
                });
            }
        }
        doc.clear();
        attrs.clear();
    }
    out
}

/// The option struct an engine kind parses `[model.engine.options]` into, or
/// `None` for a kind no build hosts.
///
/// `None` is what makes the listing honest. The three retired kinds named
/// option structs here after R3 deleted the engines that read them, so
/// `pie config list` went on advertising sixteen `engine.*` knobs — twelve
/// under `metal`, three under `vulkan`, one under `wgpu` — that no build
/// would have obeyed. A key is listed when a seam reads it.
fn options_struct(engine: EngineKind) -> Option<&'static str> {
    match engine {
        EngineKind::CudaNative => Some("CudaNativeEngineOptions"),
        EngineKind::Metal | EngineKind::Vulkan | EngineKind::Wgpu => None,
    }
}

/// Every key settable under `[worker]`, in declaration order, for a config
/// using `engine`.
///
/// The engine matters because `[model.engine.options]` is an untyped table
/// until the engine kind picks the struct it parses into -- so a listing that
/// ignored it would either show the wrong knobs or none.
pub fn fields(engine: EngineKind) -> Vec<Field> {
    let structs = parse_structs();
    let defaults = default_values(engine);
    let mut out = Vec::new();
    walk(&structs, "Config", "", engine, &defaults, &mut out);
    // The struct paths are an implementation detail; what `pie config set`
    // accepts is what the file spells. One translation, at the boundary, so
    // the walk stays a walk.
    for field in &mut out {
        field.key = crate::config_layout::to_file_path(&field.key);
    }
    out.sort_by(|a, b| {
        let section = |k: &str| {
            k.rsplit_once('.')
                .map(|(s, _)| s.to_string())
                .unwrap_or_default()
        };
        section(&a.key).cmp(&section(&b.key))
    });
    out
}

fn walk(
    structs: &BTreeMap<String, Vec<Parsed>>,
    struct_name: &str,
    prefix: &str,
    engine: EngineKind,
    defaults: &toml::Value,
    out: &mut Vec<Field>,
) {
    let Some(fields) = structs.get(struct_name) else {
        return;
    };
    for field in fields {
        if field.skip {
            continue;
        }
        let key = if prefix.is_empty() {
            field.name.clone()
        } else {
            format!("{prefix}.{}", field.name)
        };
        // `options` is the one field whose shape depends on another field's
        // value, so it is the one place the walk consults the engine kind.
        let nested = if field.ty == "toml::Table" {
            // No struct, no keys, and the untyped table itself is not one
            // either: a kind no build hosts parses nothing out of
            // `[model.engine.options]`, so nothing under it is settable.
            let Some(inner) = options_struct(engine) else {
                continue;
            };
            Some(inner.to_owned())
        } else {
            let inner = field
                .ty
                .strip_prefix("Option<")
                .and_then(|t| t.strip_suffix('>'))
                .unwrap_or(&field.ty);
            structs.contains_key(inner).then(|| inner.to_string())
        };
        match nested {
            Some(inner) => walk(structs, &inner, &key, engine, defaults, out),
            None => out.push(Field {
                doc: field.doc.clone(),
                default: if field.required {
                    None
                } else {
                    lookup(defaults, &key).cloned()
                },
                required: field.required,
                key,
            }),
        }
    }
}

/// A config carrying nothing but its required keys, serialized -- so every
/// other value in it is the one serde reaches for when the file is silent.
///
/// Built by parsing rather than by `Config::default()`, which does not exist:
/// `model` has required fields, and a type that cannot be defaulted is exactly
/// how the schema says so. The placeholders below are never read, because a
/// required field reports no default.
fn default_values(engine: EngineKind) -> toml::Value {
    let minimal = format!(
        "[model]\nname = \"x\"\nmodel = \"x\"\n\
         [engine]\ntype = \"{}\"\ndevice = [\"x\"]\n",
        engine.as_str()
    );
    let Ok(parsed) = Config::parse(&minimal) else {
        return toml::Value::Table(Default::default());
    };
    let mut root =
        toml::Value::try_from(parsed).unwrap_or_else(|_| toml::Value::Table(Default::default()));

    // `options` round-trips as the empty table it was parsed from, so the
    // engine's own option defaults have to be asked for separately -- by
    // deserializing an empty table, which is serde applying the same defaults
    // it would apply to a config that omitted the section.
    let empty = toml::Value::Table(Default::default());
    // Deserialize then re-serialize: the round trip is what applies serde's
    // defaults, and the two halves have different error types, so neither is
    // chained onto the other.
    fn defaults_of<T>(empty: &toml::Value) -> Option<toml::Value>
    where
        T: serde::de::DeserializeOwned + serde::Serialize,
    {
        let parsed: T = empty.clone().try_into().ok()?;
        toml::Value::try_from(parsed).ok()
    }
    let options = match engine {
        EngineKind::CudaNative => defaults_of::<crate::config::CudaNativeEngineOptions>(&empty),
        EngineKind::Metal | EngineKind::Vulkan | EngineKind::Wgpu => None,
    };
    if let (Some(options), Some(engine_table)) = (
        options,
        root.get_mut("model")
            .and_then(|m| m.get_mut("engine"))
            .and_then(|d| d.as_table_mut()),
    ) {
        engine_table.insert("options".to_string(), options);
    }
    root
}

/// Follow a dotted path. Returns `None` for an absent key, which for a
/// serialized default means the field is an `Option` that is `None`.
pub fn lookup<'a>(root: &'a toml::Value, key: &str) -> Option<&'a toml::Value> {
    let mut cursor = root;
    for part in key.split('.') {
        cursor = cursor.get(part)?;
    }
    Some(cursor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(engine: EngineKind) -> Vec<String> {
        fields(engine).into_iter().map(|f| f.key).collect()
    }

    #[test]
    fn schema_covers_exactly_the_settable_keys() {
        // The walk is a parse of the source; `Config::parse` is the schema
        // itself. Serializing a default config produces every non-`Option` key
        // serde will accept, so anything there and not here means the listing
        // has a blind spot -- which is how a parse that quietly broke would
        // otherwise present itself.
        let listed: std::collections::BTreeSet<String> =
            keys(EngineKind::CudaNative).into_iter().collect();

        fn collect(value: &toml::Value, prefix: &str, out: &mut Vec<String>) {
            let Some(table) = value.as_table() else {
                out.push(prefix.to_string());
                return;
            };
            for (name, child) in table {
                let key = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{prefix}.{name}")
                };
                collect(child, &key, out);
            }
        }
        let mut serialized = Vec::new();
        collect(&default_values(EngineKind::CudaNative), "", &mut serialized);

        let missing: Vec<String> = serialized
            .iter()
            .map(|key| crate::config_layout::to_file_path(key))
            .filter(|key| !listed.contains(key))
            .collect();
        assert!(
            missing.is_empty(),
            "keys serde accepts but the listing omits: {missing:?}"
        );
    }

    #[test]
    fn a_kind_no_build_hosts_advertises_no_option_keys() {
        // `[model.engine.options]` is untyped until the engine kind names the
        // struct it parses into -- and three of the four kinds name none, so
        // the listing offers an operator nothing under them beyond the common
        // keys `EngineConfig` itself declares. It offered sixteen keys across
        // the three -- `engine.total_pages`, `engine.kv_cache_dtype` among
        // them -- for engines that leave `pie serve` refusing the config by
        // name before one of them is read.
        //
        // `engine.kv_pages` is NOT in that list any more and must not go back
        // into it: it is a field of `EngineConfig`, so it is common to every
        // kind, and the two backends that read it are two of the retired ones.
        let cuda = keys(EngineKind::CudaNative);
        assert!(cuda.contains(&"engine.gpu_mem_utilization".to_string()));
        for retired in [EngineKind::Metal, EngineKind::Vulkan, EngineKind::Wgpu] {
            let listed = keys(retired);
            assert!(
                listed.iter().all(|key| cuda.contains(key)),
                "{retired:?} lists a key the hosted engine does not: {listed:?}"
            );
            assert!(listed.contains(&"engine.kv_pages".to_string()));
            for key in ["total_pages", "kv_cache_dtype", "kernels"] {
                assert!(!listed.contains(&format!("engine.{key}")));
            }
        }
    }

    #[test]
    fn every_listed_key_carries_a_description() {
        // The reason this module exists. An empty column is worse than no
        // column: it reads as "this key means nothing".
        for field in fields(EngineKind::CudaNative) {
            assert!(!field.doc.is_empty(), "{} has no description", field.key);
        }
    }

    #[test]
    fn the_summary_stops_at_the_blank_doc_line() {
        // Several fields carry paragraphs of measurement rationale after the
        // summary. A table of 80 rows cannot hold those.
        let fields = fields(EngineKind::CudaNative);
        let threads = fields
            .iter()
            .find(|f| f.key == "server.worker_threads")
            .expect("worker_threads");
        assert!(threads.doc.starts_with("Tokio worker threads."));
        assert!(
            !threads.doc.contains("EPYC"),
            "rationale leaked into the summary: {}",
            threads.doc
        );
        // Multi-line summaries are joined rather than cut at the first line.
        let hosts = fields
            .iter()
            .find(|f| f.key == "sandbox.network_allowed_hosts")
            .expect("network_allowed_hosts");
        assert!(hosts.doc.ends_with("for any."), "got: {}", hosts.doc);
    }

    #[test]
    fn fields_pie_populates_itself_are_not_offered() {
        // `device` and `verbose` in the engine options are `#[serde(skip)]`:
        // pie fills them from `[engine] device` and `[server] verbose`. After
        // the options table is flattened into `[engine]`, the skipped `device`
        // would land on the same path as the real one -- so it has to be the
        // real one that survives, exactly once.
        let cuda = keys(EngineKind::CudaNative);
        assert_eq!(
            cuda.iter().filter(|k| *k == "engine.device").count(),
            1,
            "engine.device must appear once, from EngineConfig"
        );
        assert!(!cuda.contains(&"engine.verbose".to_string()));
        assert!(cuda.contains(&"server.verbose".to_string()));
    }

    #[test]
    fn a_derived_field_has_no_default_to_print() {
        let fields = fields(EngineKind::CudaNative);
        let by_key = |k: &str| fields.iter().find(|f| f.key == k).expect(k);
        // `Option` and `None`: absence is the setting.
        assert!(by_key("runtime.max_concurrent_processes").default.is_none());
        assert!(by_key("engine.kv_page_size").default.is_none());
        // A concrete default prints as itself.
        assert_eq!(
            by_key("server.port").default,
            Some(toml::Value::Integer(8080))
        );
    }
}
