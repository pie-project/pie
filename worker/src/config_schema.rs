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

use crate::config::{Config, DriverKind};

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
    /// absence is the setting -- pie derives the value at startup and there is
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

/// The option struct a driver kind parses `[model.driver.options]` into.
fn options_struct(driver: DriverKind) -> &'static str {
    match driver {
        DriverKind::CudaNative => "CudaNativeDriverOptions",
        DriverKind::Metal => "MetalDriverOptions",
        DriverKind::Dummy => "DummyDriverOptions",
    }
}

/// Every key settable under `[worker]`, in declaration order, for a config
/// using `driver`.
///
/// The driver matters because `[model.driver.options]` is an untyped table
/// until the driver kind picks the struct it parses into -- so a listing that
/// ignored it would either show the wrong knobs or none.
pub fn fields(driver: DriverKind) -> Vec<Field> {
    let structs = parse_structs();
    let defaults = default_values(driver);
    let mut out = Vec::new();
    walk(&structs, "Config", "", driver, &defaults, &mut out);
    // The struct paths are an implementation detail; what `pie config set`
    // accepts is what the file spells. One translation, at the boundary, so
    // the walk stays a walk.
    for field in &mut out {
        field.key = crate::config_layout::to_file_path(&field.key);
    }
    out.sort_by(|a, b| {
        let section = |k: &str| k.rsplit_once('.').map(|(s, _)| s.to_string()).unwrap_or_default();
        section(&a.key).cmp(&section(&b.key))
    });
    out
}

fn walk(
    structs: &BTreeMap<String, Vec<Parsed>>,
    struct_name: &str,
    prefix: &str,
    driver: DriverKind,
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
        // value, so it is the one place the walk consults the driver kind.
        let nested = if field.ty == "toml::Table" {
            Some(options_struct(driver).to_string())
        } else {
            let inner = field
                .ty
                .strip_prefix("Option<")
                .and_then(|t| t.strip_suffix('>'))
                .unwrap_or(&field.ty);
            structs.contains_key(inner).then(|| inner.to_string())
        };
        match nested {
            Some(inner) => walk(structs, &inner, &key, driver, defaults, out),
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
fn default_values(driver: DriverKind) -> toml::Value {
    let minimal = format!(
        "[model]\nname = \"x\"\nhf_repo = \"x\"\n\
         [driver]\ntype = \"{}\"\ndevice = [\"x\"]\n",
        driver.as_str()
    );
    let Ok(parsed) = Config::parse(&minimal) else {
        return toml::Value::Table(Default::default());
    };
    let mut root =
        toml::Value::try_from(parsed).unwrap_or_else(|_| toml::Value::Table(Default::default()));

    // `options` round-trips as the empty table it was parsed from, so the
    // driver's own option defaults have to be asked for separately -- by
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
    let options = match driver {
        DriverKind::CudaNative => defaults_of::<crate::config::CudaNativeDriverOptions>(&empty),
        DriverKind::Metal => defaults_of::<crate::config::MetalDriverOptions>(&empty),
        DriverKind::Dummy => defaults_of::<crate::config::DummyDriverOptions>(&empty),
    };
    if let (Some(options), Some(driver_table)) = (
        options,
        root.get_mut("model")
            .and_then(|m| m.get_mut("driver"))
            .and_then(|d| d.as_table_mut()),
    ) {
        driver_table.insert("options".to_string(), options);
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

    fn keys(driver: DriverKind) -> Vec<String> {
        fields(driver).into_iter().map(|f| f.key).collect()
    }

    #[test]
    fn schema_covers_exactly_the_settable_keys() {
        // The walk is a parse of the source; `Config::parse` is the schema
        // itself. Serializing a default config produces every non-`Option` key
        // serde will accept, so anything there and not here means the listing
        // has a blind spot -- which is how a parse that quietly broke would
        // otherwise present itself.
        let listed: std::collections::BTreeSet<String> =
            keys(DriverKind::Dummy).into_iter().collect();

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
        collect(&default_values(DriverKind::Dummy), "", &mut serialized);

        let missing: Vec<String> = serialized
            .iter()
            .map(|key| crate::config_layout::to_file_path(key))
            .filter(|key| !listed.contains(key))
            .collect();
        assert!(missing.is_empty(), "keys serde accepts but the listing omits: {missing:?}");
    }

    #[test]
    fn the_driver_decides_which_option_keys_exist() {
        // `[model.driver.options]` is untyped until the driver kind names the
        // struct it parses into.
        let cuda = keys(DriverKind::CudaNative);
        let metal = keys(DriverKind::Metal);
        assert!(cuda.contains(&"driver.gpu_mem_utilization".to_string()));
        assert!(!metal.contains(&"driver.gpu_mem_utilization".to_string()));
        assert!(metal.contains(&"driver.total_pages".to_string()));
    }

    #[test]
    fn every_listed_key_carries_a_description() {
        // The reason this module exists. An empty column is worse than no
        // column: it reads as "this key means nothing".
        for field in fields(DriverKind::CudaNative) {
            assert!(!field.doc.is_empty(), "{} has no description", field.key);
        }
    }

    #[test]
    fn the_summary_stops_at_the_blank_doc_line() {
        // Several fields carry paragraphs of measurement rationale after the
        // summary. A table of 80 rows cannot hold those.
        let fields = fields(DriverKind::CudaNative);
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
        // `device` and `verbose` in the driver options are `#[serde(skip)]`:
        // pie fills them from `[driver] device` and `[server] verbose`. After
        // the options table is flattened into `[driver]`, the skipped `device`
        // would land on the same path as the real one -- so it has to be the
        // real one that survives, exactly once.
        let cuda = keys(DriverKind::CudaNative);
        assert_eq!(
            cuda.iter().filter(|k| *k == "driver.device").count(),
            1,
            "driver.device must appear once, from DriverConfig"
        );
        assert!(!cuda.contains(&"driver.verbose".to_string()));
        assert!(cuda.contains(&"server.verbose".to_string()));
    }

    #[test]
    fn a_derived_field_has_no_default_to_print() {
        let fields = fields(DriverKind::CudaNative);
        let by_key = |k: &str| fields.iter().find(|f| f.key == k).expect(k);
        // `Option` and `None`: absence is the setting.
        assert!(by_key("runtime.max_concurrent_processes").default.is_none());
        assert!(by_key("driver.kv_page_size").default.is_none());
        // A concrete default prints as itself.
        assert_eq!(
            by_key("server.port").default,
            Some(toml::Value::Integer(8080))
        );
    }
}
