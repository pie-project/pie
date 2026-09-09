use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::error::Error;
use crate::file::zt;
use crate::file::{Attributes, Metadata, TokenizerTables, diffusers};

pub fn discover_safetensors_files(snapshot_dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let single = snapshot_dir.join("model.safetensors");
    let index = snapshot_dir.join("model.safetensors.index.json");

    if single.is_file() {
        return Ok(vec![single]);
    }

    if index.is_file() {
        let text = std::fs::read_to_string(&index)
            .map_err(|err| Error::Checkpoint(format!("cannot read {}: {err}", index.display())))?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|err| {
            Error::Checkpoint(format!("{} is not valid JSON: {err}", index.display()))
        })?;
        let weight_map = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                Error::Checkpoint(format!("{} missing 'weight_map'", index.display()))
            })?;
        let mut shard_names = BTreeSet::new();
        for shard in weight_map.values() {
            let shard = shard.as_str().ok_or_else(|| {
                Error::Checkpoint(format!(
                    "{} weight_map has a non-string shard",
                    index.display()
                ))
            })?;
            shard_names.insert(shard.to_string());
        }
        return Ok(shard_names
            .into_iter()
            .map(|s| snapshot_dir.join(s))
            .collect());
    }

    let named = named_safetensors_files(snapshot_dir);
    if !named.is_empty() {
        return Ok(named);
    }

    Err(Error::Checkpoint(format!(
        "no model.safetensors[.index.json] in {}",
        snapshot_dir.display()
    )))
}

fn named_safetensors_files(snapshot_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(snapshot_dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
        })
        .collect();
    found.sort();
    found
}

fn discover_gguf_files(snapshot_dir: &Path) -> Result<Option<Vec<PathBuf>>, Error> {
    let named = snapshot_dir.join("model.gguf");
    if named.is_file() {
        return Ok(Some(vec![named]));
    }
    let Ok(entries) = std::fs::read_dir(snapshot_dir) else {
        return Ok(None);
    };
    let mut ggufs: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        })
        .collect();
    ggufs.sort();
    match ggufs.first() {
        Some(first) => gguf_shard_set(first).map(Some),
        None => Ok(None),
    }
}

fn gguf_shard_set(path: &Path) -> Result<Vec<PathBuf>, Error> {
    let Some((prefix, own, count)) = split_shard_name(path) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut shards = Vec::with_capacity(count as usize);
    for index in 1..=count {
        let shard = dir.join(format!("{prefix}-{index:05}-of-{count:05}.gguf"));
        if !shard.is_file() {
            return Err(Error::Checkpoint(format!(
                "{} is shard {own} of {count}, and shard {index} is not beside \
                 it ({} is missing); a split GGUF is one checkpoint, and \
                 importing the shards that happen to be present would write a \
                 model with holes",
                path.display(),
                shard.display()
            )));
        }
        shards.push(shard);
    }
    Ok(shards)
}

fn split_shard_name(path: &Path) -> Option<(String, u32, u32)> {
    let stem = path.file_stem()?.to_str()?;
    let (head, count) = stem.rsplit_once("-of-")?;
    let (prefix, index) = head.rsplit_once('-')?;
    let count: u32 = count.parse().ok()?;
    let index: u32 = index.parse().ok()?;
    if count == 0 || index == 0 || index > count {
        return None;
    }
    Some((prefix.to_string(), index, count))
}

const ZT_NAMES: [&str; 2] = ["model.zt", "archive.zt"];

pub fn discover_zt_files(snapshot_dir: &Path) -> Vec<PathBuf> {
    if snapshot_dir.is_file()
        && snapshot_dir
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("zt"))
    {
        return vec![snapshot_dir.to_path_buf()];
    }
    if let Some(named) = ZT_NAMES
        .iter()
        .map(|name| snapshot_dir.join(name))
        .find(|named| named.is_file())
    {
        return vec![named];
    }
    specialized_zt_files(snapshot_dir)
}

pub fn discover_zt_file(snapshot_dir: &Path) -> Option<PathBuf> {
    let mut found = discover_zt_files(snapshot_dir);
    (found.len() == 1).then(|| found.remove(0))
}

fn specialized_zt_files(snapshot_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(snapshot_dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("zt"))
        })
        .collect();
    found.sort();
    found
}

pub fn read_meta(metadata: &Metadata, path: &str) -> Result<Option<Vec<u8>>, Error> {
    use std::io::{Read, Seek, SeekFrom};

    let Some(object) = metadata.meta_object(path) else {
        return Ok(None);
    };
    let file = metadata
        .files
        .iter()
        .find(|file| file.id == object.file_id)
        .ok_or_else(|| {
            Error::Checkpoint(format!(
                "{} points at a file the checkpoint lacks",
                object.name
            ))
        })?;
    let mut handle = std::fs::File::open(&file.path)
        .map_err(|err| Error::Checkpoint(format!("cannot open {}: {err}", file.path)))?;
    handle
        .seek(SeekFrom::Start(object.file_offset))
        .map_err(|err| Error::Checkpoint(format!("cannot seek in {}: {err}", file.path)))?;
    let mut bytes = vec![0u8; object.span_bytes as usize];
    handle.read_exact(&mut bytes).map_err(|err| {
        Error::Checkpoint(format!(
            "cannot read {} from {}: {err}",
            object.name, file.path
        ))
    })?;
    Ok(Some(bytes))
}

enum Discovered {
    One(PathBuf),
    Set(Vec<PathBuf>),
}

fn discover(snapshot_dir: &Path) -> Result<Discovered, Error> {
    if let Some(zt) = discover_zt_file(snapshot_dir) {
        return Ok(Discovered::One(zt));
    }
    let specialized = specialized_zt_files(snapshot_dir);
    if specialized.len() > 1 {
        return Err(Error::Checkpoint(format!(
            "{} holds {} serving artifacts and this load names none of them: {}. \
             Two artifacts of one model is what the naming is for — a different \
             backend, degree or precision — so `[model] model` has to name the \
             file rather than the directory.",
            snapshot_dir.display(),
            specialized.len(),
            specialized
                .iter()
                .filter_map(|path| path.file_name())
                .map(|name| format!("`{}`", name.to_string_lossy()))
                .collect::<Vec<_>>()
                .join(", "),
        )));
    }
    if snapshot_dir.is_file() {
        return Ok(one_or_set(gguf_shard_set(snapshot_dir)?));
    }
    match discover_safetensors_files(snapshot_dir) {
        Ok(files) => Ok(Discovered::Set(files)),
        Err(safetensors_err) => match discover_gguf_files(snapshot_dir)? {
            Some(files) => Ok(one_or_set(files)),
            None => Err(safetensors_err),
        },
    }
}

fn one_or_set(mut files: Vec<PathBuf>) -> Discovered {
    if files.len() == 1 {
        Discovered::One(files.remove(0))
    } else {
        Discovered::Set(files)
    }
}

pub fn parse_metadata(snapshot_dir: &Path) -> Result<Metadata, Error> {
    if diffusers::is_pipeline(snapshot_dir) {
        return zt::describe(&diffusers::open(snapshot_dir)?);
    }
    match discover(snapshot_dir)? {
        Discovered::One(path) => zt::parse(&path),
        Discovered::Set(paths) => zt::parse_files(&paths),
    }
}

pub fn parse_groups(snapshot_dir: &Path) -> Result<Vec<(String, Vec<String>)>, Error> {
    if diffusers::is_pipeline(snapshot_dir) {
        return zt::describe_groups(&diffusers::open(snapshot_dir)?);
    }
    let paths = match discover(snapshot_dir)? {
        Discovered::One(path) => vec![path],
        Discovered::Set(paths) => paths,
    };
    let mut groups = Vec::new();
    for path in &paths {
        groups.extend(zt::parse_groups(path)?);
    }
    Ok(groups)
}

pub fn parse_attributes(snapshot_dir: &Path) -> Result<Attributes, Error> {
    if diffusers::is_pipeline(snapshot_dir) {
        return Ok(Attributes::default());
    }
    match discover(snapshot_dir)? {
        Discovered::One(path) => zt::parse_attributes(&path),
        Discovered::Set(paths) => zt::parse_attributes_files(&paths),
    }
}

pub fn parse_tokenizer(snapshot_dir: &Path) -> Result<TokenizerTables, Error> {
    if diffusers::is_pipeline(snapshot_dir) {
        return Ok(TokenizerTables::default());
    }
    let path = match discover(snapshot_dir)? {
        Discovered::One(path) => path,
        Discovered::Set(mut paths) => {
            paths.sort();
            paths.remove(0)
        }
    };
    zt::parse_tokenizer_tables(&path)
}

pub fn verify_declared_files(
    plan: &crate::plan::LoadPlan,
    snapshot_dir: &Path,
) -> Result<(), Error> {
    for file in &plan.files {
        let path = snapshot_dir.join(&file.path);
        match std::fs::metadata(&path) {
            Ok(meta) if meta.len() == file.size_bytes => {}
            Ok(meta) => {
                return Err(Error::Checkpoint(format!(
                    "{} is {} bytes on disk, the plan declares {}",
                    path.display(),
                    meta.len(),
                    file.size_bytes
                )));
            }
            Err(err) => {
                return Err(Error::Checkpoint(format!("{}: {err}", path.display())));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        std::fs::write(path, b"stand-in for an artifact").unwrap();
    }

    #[test]
    fn read_every_case() {
        a_directory_of_specializations_discovers_all_of_them();
        a_lone_specialization_is_the_answer_to_both_questions();
        a_fixed_name_wins_over_everything_beside_it();
        a_file_names_itself();
        a_directory_without_artifacts_discovers_nothing();
    }

    fn a_directory_of_specializations_discovers_all_of_them() {
        let dir = tempfile::tempdir().unwrap();
        let cuda = dir.path().join("glm.glm53-flash-u8g64-kv-bf16.cuda.zt");
        let vulkan = dir.path().join("glm.glm53-flash-u8g64-kv-bf16.vulkan.zt");
        touch(&cuda);
        touch(&vulkan);

        assert_eq!(discover_zt_files(dir.path()), vec![cuda, vulkan]);
        assert_eq!(
            discover_zt_file(dir.path()),
            None,
            "a caller that cannot say which one it means still gets a refusal"
        );
    }

    fn a_lone_specialization_is_the_answer_to_both_questions() {
        let dir = tempfile::tempdir().unwrap();
        let only = dir.path().join("glm.glm53-flash-u8g64-kv-bf16.cuda.zt");
        touch(&only);

        assert_eq!(discover_zt_files(dir.path()), vec![only.clone()]);
        assert_eq!(discover_zt_file(dir.path()), Some(only));
    }

    fn a_fixed_name_wins_over_everything_beside_it() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("archive.zt");
        touch(&archive);
        touch(&dir.path().join("glm.glm53-flash-u8g64-kv-bf16.cuda.zt"));
        assert_eq!(discover_zt_files(dir.path()), vec![archive.clone()]);

        let model = dir.path().join("model.zt");
        touch(&model);
        assert_eq!(discover_zt_files(dir.path()), vec![model]);
    }

    fn a_file_names_itself() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("solo.zt");
        touch(&file);
        assert_eq!(discover_zt_files(&file), vec![file.clone()]);
        assert_eq!(discover_zt_file(&file), Some(file));
    }

    fn a_directory_without_artifacts_discovers_nothing() {
        let dir = tempfile::tempdir().unwrap();
        touch(&dir.path().join("model.safetensors"));
        assert!(discover_zt_files(dir.path()).is_empty());
        assert_eq!(discover_zt_file(dir.path()), None);
    }
}
