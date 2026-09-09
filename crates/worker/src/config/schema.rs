use std::collections::BTreeMap;

use crate::config::{Config, EngineKind};

const SOURCE: &str = concat!(
    include_str!("../config.rs"),
    include_str!("units.rs"),
    include_str!("backend.rs"),
);

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub key: String,
    pub doc: String,
    pub default: Option<toml::Value>,
    pub required: bool,
}

struct Parsed {
    name: String,
    ty: String,
    doc: String,
    skip: bool,
    required: bool,
}

fn parse_structs() -> BTreeMap<String, Vec<Parsed>> {
    let mut out: BTreeMap<String, Vec<Parsed>> = BTreeMap::new();
    let mut current = String::new();
    let mut doc: Vec<String> = Vec::new();
    let mut attrs: Vec<String> = Vec::new();
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

fn options_struct(engine: EngineKind) -> Option<&'static str> {
    match engine {
        EngineKind::CudaNative => Some("CudaNativeEngineOptions"),
        EngineKind::Metal => Some("MetalEngineOptions"),
        EngineKind::Vulkan => Some("VulkanEngineOptions"),
        EngineKind::Wgpu => Some("WgpuEngineOptions"),
    }
}

pub fn fields(engine: EngineKind) -> Vec<Field> {
    let structs = parse_structs();
    let defaults = default_values(engine);
    let mut out = Vec::new();
    walk(&structs, "Config", "", engine, &defaults, &mut out);
    for field in &mut out {
        field.key = crate::config::layout::to_file_path(&field.key);
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
        let nested = if field.ty == "toml::Table" {
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

    let empty = toml::Value::Table(Default::default());
    fn defaults_of<T>(empty: &toml::Value) -> Option<toml::Value>
    where
        T: serde::de::DeserializeOwned + serde::Serialize,
    {
        let parsed: T = empty.clone().try_into().ok()?;
        toml::Value::try_from(parsed).ok()
    }
    let options = match engine {
        EngineKind::CudaNative => defaults_of::<crate::config::CudaNativeEngineOptions>(&empty),
        EngineKind::Metal => defaults_of::<crate::config::MetalEngineOptions>(&empty),
        EngineKind::Vulkan => defaults_of::<crate::config::VulkanEngineOptions>(&empty),
        EngineKind::Wgpu => defaults_of::<crate::config::WgpuEngineOptions>(&empty),
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
    fn schema_every_case() {
        schema_covers_exactly_the_settable_keys();
        the_summary_stops_at_the_blank_doc_line();
        a_derived_field_has_no_default_to_print();
    }

    fn schema_covers_exactly_the_settable_keys() {
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
            .map(|key| crate::config::layout::to_file_path(key))
            .filter(|key| !listed.contains(key))
            .collect();
        assert!(
            missing.is_empty(),
            "keys serde accepts but the listing omits: {missing:?}"
        );
    }

    fn the_summary_stops_at_the_blank_doc_line() {
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
        let hosts = fields
            .iter()
            .find(|f| f.key == "sandbox.network_allowed_hosts")
            .expect("network_allowed_hosts");
        assert!(hosts.doc.ends_with("for any."), "got: {}", hosts.doc);
    }

    fn a_derived_field_has_no_default_to_print() {
        let fields = fields(EngineKind::CudaNative);
        let by_key = |k: &str| fields.iter().find(|f| f.key == k).expect(k);
        assert!(by_key("runtime.max_concurrent_processes").default.is_none());
        assert!(by_key("engine.kv_page_size").default.is_none());
        assert_eq!(
            by_key("server.port").default,
            Some(toml::Value::Integer(8080))
        );
    }
}
