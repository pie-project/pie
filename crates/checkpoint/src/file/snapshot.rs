//! A checkpoint as it lies on disk, opened as one tensor name space: a
//! diffusers pipeline, a stamped artifact, or the safetensors / zt containers of
//! a snapshot directory. The configuration a snapshot ships beside its weights
//! rides along as the source's `config` attribute, so an import reads what the
//! checkpoint states of itself from the source it reads the weights from.

use std::path::{Path, PathBuf};

use ztensor::format::cbor::Value;

use crate::error::Error;
use crate::file::diffusers;

/// The attribute a snapshot's configuration rides under: the snapshot's
/// `config.json`, or for a diffusers pipeline a map from each component's
/// name prefix (`dit`, `te`, `vae`, ...) to that component's `config.json`.
pub const CONFIG: &str = "config";

/// The checkpoint at `path` as one source, its configuration attached.
pub fn open(path: &Path) -> Result<ztensor::Source, Error> {
    if diffusers::is_pipeline(path) {
        let components = diffusers::components(path)?;
        let source = diffusers::open_from(&components, path)?;
        let mut configs = Vec::new();
        for component in &components {
            let Some(config) = &component.config else {
                continue;
            };
            let key = component.prefix.trim_end_matches('.').to_string();
            configs.push((Value::Text(key), json_file(config)?));
        }
        return Ok(if configs.is_empty() {
            source
        } else {
            source.with_attribute(CONFIG, Value::Map(configs))
        });
    }
    let containers = containers(path)?;
    let source = if let [container] = containers.as_slice() {
        ztensor_compat::index(container)
            .or_else(|_| ztensor::Source::open(container))
            .map_err(|why| {
                Error::Checkpoint(format!(
                    "cannot open {} as a tensor container: {why}",
                    container.display()
                ))
            })?
    } else {
        ztensor_compat::index_all(&containers)
            .or_else(|_| ztensor::Source::open_all(&containers))
            .map_err(|why| {
                Error::Checkpoint(format!(
                    "cannot open the {} containers under {} as one tensor name space: {why}",
                    containers.len(),
                    path.display()
                ))
            })?
    };
    let config = path.join("config.json");
    Ok(if path.is_dir() && config.is_file() {
        source.with_attribute(CONFIG, json_file(&config)?)
    } else {
        source
    })
}

fn containers(path: &Path) -> Result<Vec<PathBuf>, Error> {
    if !path.is_dir() {
        return Ok(vec![path.to_path_buf()]);
    }
    let root = path.join("model.zt");
    if root.is_file() {
        return Ok(vec![root]);
    }
    let mut found: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|why| {
            Error::Checkpoint(format!(
                "cannot read the checkpoint directory {}: {why}",
                path.display()
            ))
        })?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    if found.is_empty() {
        return Err(Error::Checkpoint(format!(
            "{} holds no `.safetensors` and no `.zt` container",
            path.display()
        )));
    }
    Ok(found)
}

fn json_file(path: &Path) -> Result<Value, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|why| Error::Checkpoint(format!("cannot read {}: {why}", path.display())))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|why| Error::Checkpoint(format!("{} is not valid JSON: {why}", path.display())))?;
    Ok(cbor(&json))
}

/// A JSON value as the attribute value a source carries.
#[must_use]
fn cbor(json: &serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => match (n.as_u64(), n.as_i64(), n.as_f64()) {
            (Some(u), _, _) => Value::Uint(u),
            (None, Some(i), _) => Value::Nint(i.unsigned_abs() - 1),
            (_, _, Some(f)) => Value::Float(f),
            _ => Value::Null,
        },
        serde_json::Value::String(s) => Value::Text(s.clone()),
        serde_json::Value::Array(items) => Value::Array(items.iter().map(cbor).collect()),
        serde_json::Value::Object(map) => Value::Map(
            map.iter()
                .map(|(k, v)| (Value::Text(k.clone()), cbor(v)))
                .collect(),
        ),
    }
}
