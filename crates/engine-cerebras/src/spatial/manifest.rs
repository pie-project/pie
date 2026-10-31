//! `pie_manifest.json`: what PIE built into a compiled output directory.
//!
//! [`super::build::compile`] writes it next to `out.json`: the program's
//! [`ProgramSpec`] (`spec`: its design and the parameters it renders, which
//! make it again exactly; the design and params also at the top level, as
//! manifests had them before the spec), the build's cache key and fabric.
//! The driver reads the spec from it ([`read_program`]); a manifest from
//! before the spec (still in the compile cache) is read from its params. A directory without one (compiled
//! outside PIE) is read from `out.json`'s params, its design inferred from
//! them, with a warning.

use std::path::Path;

use super::codegen::Design;
use super::program::ProgramSpec;

type R<T> = std::result::Result<T, String>;

pub const FILE: &str = "pie_manifest.json";
/// The manifest format's version.
const VERSION: u64 = 1;

pub use super::program::Params;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// The program built.
    pub program: ProgramSpec,
    /// The build's cache key (sources, params, fabric).
    pub key: String,
    pub fabric_dims: (u32, u32),
    pub fabric_offsets: (u32, u32),
    pub system: bool,
}

impl Manifest {
    /// The manifest's JSON: the spec as its design and parameters (in
    /// order), and, for readers of the file, the VM image the design's
    /// sources carry and how its KV staging slots pack.
    pub fn to_json(&self) -> serde_json::Value {
        let spec = &self.program;
        let params = spec.to_params();
        let map: serde_json::Map<String, serde_json::Value> = params
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();
        let kvst = Some(spec.kvst()).filter(|&s| s > 0);
        serde_json::json!({
            "version": VERSION,
            "design": spec.design.name(),
            "params": map,
            "param_order": params.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
            "key": self.key,
            "fabric_dims": [self.fabric_dims.0, self.fabric_dims.1],
            "fabric_offsets": [self.fabric_offsets.0, self.fabric_offsets.1],
            "system": self.system,
            "vm": spec.design.vm_image(),
            "kvst": kvst,
            "kvst_pack": kvst.map(|_| spec.design.kv_staging().0),
            "kvst_aliased": kvst.map(|_| spec.design.kv_staging().1),
            "spec": {
                "design": spec.design.name(),
                "params": params.iter().map(|(k, v)| [k, v]).collect::<Vec<_>>(),
            },
        })
    }

    pub fn from_json(v: &serde_json::Value) -> R<Manifest> {
        let version = v["version"].as_u64().ok_or("manifest: no version")?;
        if version != VERSION {
            return Err(format!(
                "manifest: version {version}, this engine reads {VERSION}"
            ));
        }
        let name = v["design"].as_str().ok_or("manifest: no design")?;
        let design = Design::parse(name).ok_or_else(|| {
            format!("manifest: PIE built a `{name}` program, which this engine does not serve")
        })?;
        let map = v["params"].as_object().ok_or("manifest: no params")?;
        let order: Vec<String> = match v["param_order"].as_array() {
            Some(a) => a
                .iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect(),
            None => map.keys().cloned().collect(),
        };
        let params = order
            .into_iter()
            .map(|k| {
                let val = &map[&k];
                let s = val
                    .as_str()
                    .map(String::from)
                    .or_else(|| val.as_u64().map(|n| n.to_string()));
                s.map(|s| (k.clone(), s))
                    .ok_or_else(|| format!("manifest: param `{k}` is not a string or integer"))
            })
            .collect::<R<Vec<_>>>()?;
        let pair = |f: &str| -> R<(u32, u32)> {
            let a = v[f]
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| format!("manifest: no {f}"))?;
            Ok((
                a[0].as_u64().unwrap_or(0) as u32,
                a[1].as_u64().unwrap_or(0) as u32,
            ))
        };
        // A manifest written since the program spec carries it (`spec`: its
        // design and rendered params, in order), and it must make one; an
        // older manifest (PIE's builds before it, still in the compile
        // cache) has only the design and params, which make the spec.
        let program = match v.get("spec") {
            Some(spec) => {
                let name = spec["design"]
                    .as_str()
                    .ok_or("manifest: a spec without a design")?;
                let design = Design::parse(name)
                    .ok_or_else(|| format!("manifest: a spec of an unknown design `{name}`"))?;
                let params = spec["params"]
                    .as_array()
                    .ok_or("manifest: a spec without params")?
                    .iter()
                    .map(|kv| match (kv[0].as_str(), kv[1].as_str()) {
                        (Some(k), Some(v)) => Ok((k.to_string(), v.to_string())),
                        _ => Err(format!("manifest: a spec param {kv} is not [name, value]")),
                    })
                    .collect::<R<Vec<_>>>()?;
                ProgramSpec::from_params(design, &params)
                    .map_err(|e| format!("manifest: its program spec: {e}"))?
            }
            None => {
                ProgramSpec::from_params(design, &params).map_err(|e| format!("manifest: {e}"))?
            }
        };
        Ok(Manifest {
            program,
            key: v["key"].as_str().unwrap_or("").to_string(),
            fabric_dims: pair("fabric_dims")?,
            fabric_offsets: pair("fabric_offsets")?,
            system: v["system"].as_bool().unwrap_or(false),
        })
    }

    /// The manifest in `dir`: `None` when there is none, an error when it
    /// does not read.
    pub fn load(dir: &Path) -> R<Option<Manifest>> {
        let path = dir.join(FILE);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Ok(None);
        };
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Manifest::from_json(&v)
            .map(Some)
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The manifest in `dir`, if PIE built it and it reads (an unreadable
    /// one is logged).
    pub fn read(dir: &Path) -> Option<Manifest> {
        Manifest::load(dir).unwrap_or_else(|e| {
            tracing::warn!("{e}");
            None
        })
    }

    pub fn write(&self, dir: &Path) -> R<()> {
        let path = dir.join(FILE);
        let text = serde_json::to_string_pretty(&self.to_json()).map_err(|e| e.to_string())?;
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// The program a compiled output directory holds: the manifest's spec
/// when PIE built it; otherwise `out.json`'s params, the design inferred
/// from them, with a warning.
pub fn read_program(dir: &Path) -> R<ProgramSpec> {
    if let Some(m) = Manifest::load(dir)? {
        return Ok(m.program);
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
    let params: Params = v["params"]
        .as_object()
        .map(|o| {
            o.iter()
                .map(|(k, x)| {
                    (
                        k.clone(),
                        x.as_str()
                            .map(String::from)
                            .unwrap_or_else(|| x.to_string()),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    ProgramSpec::infer_design(&params)
        .and_then(|design| ProgramSpec::from_params(design, &params))
        .map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str) -> ProgramSpec {
        let (_, design, params) = crate::spatial::golden::scenario_params()
            .into_iter()
            .find(|s| s.0 == name)
            .expect("the scenario");
        ProgramSpec::from_params(design, &params).unwrap()
    }

    fn manifest(program: ProgramSpec) -> Manifest {
        Manifest {
            program,
            key: "abc".into(),
            fabric_dims: (24, 19),
            fabric_offsets: (4, 1),
            system: false,
        }
    }

    #[test]
    fn manifest_round_trips() {
        for name in ["serve2d_pipe_ab", "serve8b_kvst", "serve8b_region_pipe"] {
            let m = manifest(spec(name));
            let v = m.to_json();
            assert_eq!(Manifest::from_json(&v).unwrap(), m, "{name}");
            assert_eq!(v["design"], m.program.design.name());
        }
        let v = manifest(spec("serve8b_kvst")).to_json();
        assert_eq!(
            (&v["vm"], &v["kvst"], &v["kvst_pack"], &v["kvst_aliased"]),
            (
                &serde_json::Value::Null,
                &serde_json::json!(1),
                &serde_json::json!("bf16x2"),
                &serde_json::json!(true)
            )
        );
    }

    #[test]
    fn manifest_refuses_other_versions_and_bad_params() {
        let mut v = manifest(spec("serve2d")).to_json();
        v["version"] = serde_json::json!(2);
        assert!(Manifest::from_json(&v).unwrap_err().contains("version 2"));
        v["version"] = serde_json::json!(1);
        let good = v.clone();
        v["params"]["B"] = serde_json::json!([1]);
        assert!(Manifest::from_json(&v).is_err());
        let mut v = good.clone();
        v["design"] = serde_json::json!("serve28s3");
        assert!(Manifest::from_json(&v).unwrap_err().contains("serve28s3"));
        // Params that make no program: an older manifest's, and a spec's.
        let mut v = good.clone();
        v.as_object_mut().unwrap().remove("spec");
        v["params"]["HEADMODE"] = serde_json::json!("0");
        assert!(Manifest::from_json(&v).unwrap_err().contains("HEADMODE"));
        let mut v = good;
        for kv in v["spec"]["params"].as_array_mut().unwrap() {
            if kv[0] == "HEADMODE" {
                kv[1] = serde_json::json!("0");
            }
        }
        assert!(Manifest::from_json(&v).unwrap_err().contains("HEADMODE"));
    }

    /// A manifest PIE wrote before the program spec (no `spec`: the design,
    /// the params and the derived fields only) loads through its params; a
    /// present spec that makes no program is an error.
    #[test]
    fn older_manifests_load_and_a_bad_spec_does_not() {
        let m = manifest(spec("serve8b_region_kvst"));
        let mut old = m.to_json();
        old.as_object_mut().unwrap().remove("spec");
        assert_eq!(Manifest::from_json(&old).unwrap(), m);
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(FILE), old.to_string()).unwrap();
        assert_eq!(read_program(dir.path()).unwrap(), m.program);
        let mut bad = m.to_json();
        bad["spec"]["params"]
            .as_array_mut()
            .unwrap()
            .retain(|kv| kv[0] != "HH");
        assert!(
            Manifest::from_json(&bad)
                .unwrap_err()
                .contains("its program spec")
        );
        let mut bad = m.to_json();
        bad["spec"]["design"] = serde_json::json!("serve28s3");
        assert!(Manifest::from_json(&bad).is_err());
    }

    #[test]
    fn read_program_prefers_the_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let plain = spec("serve2d");
        let out: serde_json::Map<String, serde_json::Value> = plain
            .to_params()
            .into_iter()
            .map(|(k, v)| (k, serde_json::Value::String(v)))
            .collect();
        std::fs::write(
            dir.path().join("out.json"),
            serde_json::json!({ "params": out }).to_string(),
        )
        .unwrap();
        // Without a manifest: the design inferred from out.json.
        assert_eq!(read_program(dir.path()).unwrap().design, Design::Serve2d);
        let kvst = spec("serve2d_kvst");
        manifest(kvst.clone()).write(dir.path()).unwrap();
        assert_eq!(read_program(dir.path()).unwrap(), kvst);
        std::fs::write(dir.path().join(FILE), "{").unwrap();
        assert!(read_program(dir.path()).is_err());
    }
}
