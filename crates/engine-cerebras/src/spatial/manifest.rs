//! `pie_manifest.json`: what PIE built into a compiled output directory.
//!
//! [`super::build::compile`] writes it next to `out.json`, recording the
//! design and the exact compile parameters. Program readers take the design
//! and its parameters from it. A directory without one (compiled outside PIE)
//! falls back to reading `out.json`'s params and the older probes, with a
//! warning.

use std::path::Path;

type R<T> = std::result::Result<T, String>;

pub const FILE: &str = "pie_manifest.json";
/// The manifest format's version.
const VERSION: u64 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// The design (`serve2d`, `serve2d_pipe`).
    pub design: String,
    /// The compile parameters, in the layout's order.
    pub params: Vec<(String, String)>,
    /// The build's cache key (sources, params, fabric).
    pub key: String,
    pub fabric_dims: (u32, u32),
    pub fabric_offsets: (u32, u32),
    pub system: bool,
}

impl Manifest {
    pub fn to_json(&self) -> serde_json::Value {
        let params: serde_json::Map<String, serde_json::Value> =
            self.params.iter().map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone()))).collect();
        serde_json::json!({
            "version": VERSION,
            "design": self.design,
            "params": params,
            "param_order": self.params.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
            "key": self.key,
            "fabric_dims": [self.fabric_dims.0, self.fabric_dims.1],
            "fabric_offsets": [self.fabric_offsets.0, self.fabric_offsets.1],
            "system": self.system,
        })
    }

    pub fn from_json(v: &serde_json::Value) -> R<Manifest> {
        let version = v["version"].as_u64().ok_or("manifest: no version")?;
        if version != VERSION {
            return Err(format!("manifest: version {version}, this engine reads {VERSION}"));
        }
        let design = v["design"].as_str().ok_or("manifest: no design")?.to_string();
        let map = v["params"].as_object().ok_or("manifest: no params")?;
        let order: Vec<String> = match v["param_order"].as_array() {
            Some(a) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
            None => map.keys().cloned().collect(),
        };
        let params = order
            .into_iter()
            .map(|k| {
                let val = &map[&k];
                let s = val.as_str().map(String::from).or_else(|| val.as_u64().map(|n| n.to_string()));
                s.map(|s| (k.clone(), s)).ok_or_else(|| format!("manifest: param `{k}` is not a string or integer"))
            })
            .collect::<R<Vec<_>>>()?;
        let pair = |f: &str| -> R<(u32, u32)> {
            let a = v[f].as_array().filter(|a| a.len() == 2).ok_or_else(|| format!("manifest: no {f}"))?;
            Ok((a[0].as_u64().unwrap_or(0) as u32, a[1].as_u64().unwrap_or(0) as u32))
        };
        Ok(Manifest {
            design,
            params,
            key: v["key"].as_str().unwrap_or("").to_string(),
            fabric_dims: pair("fabric_dims")?,
            fabric_offsets: pair("fabric_offsets")?,
            system: v["system"].as_bool().unwrap_or(false),
        })
    }

    /// The manifest in `dir`, if PIE built it (and it parses).
    pub fn read(dir: &Path) -> Option<Manifest> {
        let text = std::fs::read_to_string(dir.join(FILE)).ok()?;
        let v: serde_json::Value = serde_json::from_str(&text).ok()?;
        match Manifest::from_json(&v) {
            Ok(m) => Some(m),
            Err(e) => {
                tracing::warn!(dir = %dir.display(), "{FILE} unreadable: {e}");
                None
            }
        }
    }

    pub fn write(&self, dir: &Path) -> R<()> {
        let path = dir.join(FILE);
        let text = serde_json::to_string_pretty(&self.to_json()).map_err(|e| e.to_string())?;
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// A parameter as a number.
    pub fn get(&self, k: &str) -> Option<usize> {
        self.params.iter().find(|(n, _)| n == k).and_then(|(_, v)| v.trim().parse().ok())
    }
}

/// A program directory's parameters: the manifest's when PIE built it,
/// else `out.json`'s, with a warning that the design is being inferred.
pub fn program_params(dir: &Path) -> R<(Option<String>, Vec<(String, String)>)> {
    if let Some(m) = Manifest::read(dir) {
        return Ok((Some(m.design), m.params));
    }
    let path = dir.join("out.json");
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?,
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    tracing::warn!(
        dir = %dir.display(),
        "no {FILE}: a program compiled outside PIE; its design is inferred from out.json"
    );
    let params = v["params"]
        .as_object()
        .map(|o| {
            o.iter()
                .map(|(k, x)| (k.clone(), x.as_str().map(String::from).unwrap_or_else(|| x.to_string())))
                .collect()
        })
        .unwrap_or_default();
    Ok((None, params))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trips() {
        let m = Manifest {
            design: "serve2d_pipe".into(),
            params: vec![("PX".into(), "4".into()), ("B".into(), "2".into()), ("vocab".into(), "250".into())],
            key: "abc".into(),
            fabric_dims: (24, 19),
            fabric_offsets: (4, 1),
            system: false,
        };
        let back = Manifest::from_json(&m.to_json()).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.get("vocab"), Some(250));
        assert_eq!(back.get("none"), None);
    }

    #[test]
    fn manifest_refuses_other_versions_and_bad_params() {
        let mut v = Manifest {
            design: "serve2d".into(),
            params: vec![],
            key: String::new(),
            fabric_dims: (1, 1),
            fabric_offsets: (4, 1),
            system: false,
        }
        .to_json();
        v["version"] = serde_json::json!(2);
        assert!(Manifest::from_json(&v).unwrap_err().contains("version 2"));
        v["version"] = serde_json::json!(1);
        v["params"] = serde_json::json!({"B": [1]});
        v["param_order"] = serde_json::json!(["B"]);
        assert!(Manifest::from_json(&v).is_err());
    }

    #[test]
    fn program_params_prefers_the_manifest() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("out.json"), r#"{"params": {"PX": "8"}}"#).unwrap();
        let (design, p) = program_params(dir.path()).unwrap();
        assert_eq!((design, p), (None, vec![("PX".to_string(), "8".to_string())]));
        Manifest {
            design: "serve2d".into(),
            params: vec![("PX".into(), "4".into())],
            key: "k".into(),
            fabric_dims: (1, 1),
            fabric_offsets: (4, 1),
            system: false,
        }
        .write(dir.path())
        .unwrap();
        let (design, p) = program_params(dir.path()).unwrap();
        assert_eq!(design.as_deref(), Some("serve2d"));
        assert_eq!(p, vec![("PX".to_string(), "4".to_string())]);
    }
}
