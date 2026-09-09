use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::error::Error;

pub const PIPELINE_INDEX: &str = "model_index.json";

const STEMS: [&str; 2] = ["diffusion_pytorch_model", "model"];

fn is_weightless(folder: &str) -> bool {
    folder == "scheduler" || folder.starts_with("tokenizer") || folder.starts_with("feature_")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Component {
    pub folder: String,
    pub prefix: String,
    pub library: String,
    pub class: String,
    pub dir: PathBuf,
    pub weights: Vec<PathBuf>,
    pub config: Option<PathBuf>,
}

#[must_use]
pub fn prefix_of(folder: &str) -> String {
    let role = match folder {
        "transformer" => "dit",
        "transformer_2" => "dit2",
        "text_encoder" => "te",
        "text_encoder_2" => "te2",
        "vae" | "video_vae" => "vae",
        "image_encoder" => "ie",
        "audio_vae" => "avae",
        "vocoder" => "voc",
        other => other,
    };
    format!("{role}.")
}

#[must_use]
pub fn is_pipeline(dir: &Path) -> bool {
    dir.is_dir() && dir.join(PIPELINE_INDEX).is_file()
}

pub fn components(dir: &Path) -> Result<Vec<Component>, Error> {
    let index = dir.join(PIPELINE_INDEX);
    let text = std::fs::read_to_string(&index)
        .map_err(|err| Error::Checkpoint(format!("cannot read {}: {err}", index.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
        Error::Checkpoint(format!("{} is not valid JSON: {err}", index.display()))
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| Error::Checkpoint(format!("{} is not a JSON object", index.display())))?;

    let mut found = Vec::new();
    for (folder, entry) in object {
        let Some((library, class)) = pair(entry) else {
            continue;
        };
        if folder.starts_with('_') || is_weightless(folder) {
            continue;
        }
        let component_dir = dir.join(folder);
        if !component_dir.is_dir() {
            continue;
        }
        let weights = weight_files(&component_dir)?;
        if weights.is_empty() {
            continue;
        }
        let config = component_dir.join("config.json");
        found.push(Component {
            folder: folder.clone(),
            prefix: prefix_of(folder),
            library,
            class,
            dir: component_dir,
            weights,
            config: config.is_file().then_some(config),
        });
    }
    Ok(found)
}

fn pair(entry: &serde_json::Value) -> Option<(String, String)> {
    let array = entry.as_array()?;
    let [library, class] = array.as_slice() else {
        return None;
    };
    Some((library.as_str()?.to_string(), class.as_str()?.to_string()))
}

fn weight_files(dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let here = weight_files_in(dir)?;
    if !here.is_empty() {
        return Ok(here);
    }
    for sub in subdirectories(dir) {
        let below = weight_files_in(&sub)?;
        if !below.is_empty() {
            return Ok(below);
        }
    }
    Ok(Vec::new())
}

fn weight_files_in(dir: &Path) -> Result<Vec<PathBuf>, Error> {
    for stem in STEMS {
        let index = dir.join(format!("{stem}.safetensors.index.json"));
        if index.is_file() {
            return shards_from_index(dir, &index);
        }
        let single = dir.join(format!("{stem}.safetensors"));
        if single.is_file() {
            return Ok(vec![single]);
        }
        let loose = loose_shards(dir, stem);
        if !loose.is_empty() {
            return Ok(loose);
        }
    }
    Ok(Vec::new())
}

fn subdirectories(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| !name.starts_with('.'))
        })
        .collect();
    found.sort();
    found
}

fn shards_from_index(dir: &Path, index: &Path) -> Result<Vec<PathBuf>, Error> {
    let text = std::fs::read_to_string(index)
        .map_err(|err| Error::Checkpoint(format!("cannot read {}: {err}", index.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
        Error::Checkpoint(format!("{} is not valid JSON: {err}", index.display()))
    })?;
    let weight_map = value
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| Error::Checkpoint(format!("{} missing 'weight_map'", index.display())))?;
    let mut names = BTreeSet::new();
    for shard in weight_map.values() {
        let shard = shard.as_str().ok_or_else(|| {
            Error::Checkpoint(format!(
                "{} weight_map has a non-string shard",
                index.display()
            ))
        })?;
        names.insert(shard.to_string());
    }
    let mut paths = Vec::with_capacity(names.len());
    for name in names {
        let path = dir.join(&name);
        if !path.is_file() {
            return Err(Error::Checkpoint(format!(
                "{} names the shard {name}, which is not beside it; a sharded \
                 component is one checkpoint, and reading the shards that happen \
                 to be present would import a model with holes",
                index.display()
            )));
        }
        paths.push(path);
    }
    Ok(paths)
}

fn loose_shards(dir: &Path, stem: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with(&format!("{stem}-")) && name.ends_with(".safetensors")
                    })
        })
        .collect();
    found.sort();
    found
}

pub fn open(dir: &Path) -> Result<ztensor::Source, Error> {
    open_from(&components(dir)?, dir)
}

pub fn open_from(components: &[Component], dir: &Path) -> Result<ztensor::Source, Error> {
    if components.is_empty() {
        return Err(Error::Checkpoint(format!(
            "{} has a {PIPELINE_INDEX} but no component holds any safetensors; \
             a pipeline's weights live under its component folders, and a \
             bundle beside them is not read",
            dir.display()
        )));
    }
    let mut parts = Vec::with_capacity(components.len());
    for component in components {
        let source = ztensor_compat::index_all(&component.weights).map_err(|err| {
            Error::Checkpoint(format!(
                "cannot read the {} component of {}: {err}",
                component.folder,
                dir.display()
            ))
        })?;
        parts.push(source.under(&component.prefix).map_err(Error::from)?);
    }
    ztensor::Source::merge(parts).map_err(Error::from)
}

pub fn configs(dir: &Path) -> Result<Vec<(String, Vec<u8>)>, Error> {
    let mut carried = Vec::new();
    let index = dir.join(PIPELINE_INDEX);
    if let Some(bytes) = read_json(&index)? {
        carried.push((String::new(), bytes));
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(carried);
    };
    let mut folders: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    folders.sort();
    for folder in folders {
        let Some(name) = folder.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        for file in ["config.json", "scheduler_config.json"] {
            if let Some(bytes) = read_json(&folder.join(file))? {
                carried.push((name.to_string(), bytes));
                break;
            }
        }
    }
    Ok(carried)
}

fn read_json(path: &Path) -> Result<Option<Vec<u8>>, Error> {
    if !path.is_file() {
        return Ok(None);
    }
    let raw = std::fs::read(path)
        .map_err(|err| Error::Checkpoint(format!("cannot read {}: {err}", path.display())))?;
    serde_json::from_slice::<serde_json::Value>(&raw)
        .map_err(|err| Error::Checkpoint(format!("cannot parse {}: {err}", path.display())))?;
    Ok(Some(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffusers_every_case() {
        every_known_component_takes_its_role_as_a_prefix();
        a_null_pair_is_not_a_component();
    }

    fn every_known_component_takes_its_role_as_a_prefix() {
        assert_eq!(prefix_of("transformer"), "dit.");
        assert_eq!(prefix_of("transformer_2"), "dit2.");
        assert_eq!(prefix_of("text_encoder"), "te.");
        assert_eq!(prefix_of("text_encoder_2"), "te2.");
        assert_eq!(prefix_of("vae"), "vae.");
        assert_eq!(prefix_of("image_encoder"), "ie.");
        assert_eq!(prefix_of("audio_vae"), "avae.");
        assert_eq!(prefix_of("vocoder"), "voc.");
        assert_eq!(prefix_of("connector"), "connector.");
    }

    fn a_null_pair_is_not_a_component() {
        assert_eq!(
            pair(&serde_json::json!(["diffusers", "AutoencoderKL"])),
            Some(("diffusers".to_string(), "AutoencoderKL".to_string()))
        );
        assert_eq!(pair(&serde_json::json!([null, null])), None);
        assert_eq!(pair(&serde_json::json!("0.36.0")), None);
        assert_eq!(pair(&serde_json::json!(["one"])), None);
    }
}
