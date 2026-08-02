//! Finding the model a worker is configured to serve — **load-at-boot only**.
//!
//! The worker never downloads and never converts (R3). What it resolves is a
//! name or a path to something already on disk, and under this plan that
//! something is one `.zt` artifact: weights, compiled tokenizer and compiled
//! model descriptor together, written by `pie model download` or
//! `pie model convert`.
//!
//! Two forms, told apart by shape rather than by what happens to exist:
//!
//! - anything path-like — absolute, `./`-relative, or ending in a checkpoint
//!   extension — is used as given, so an artifact outside the store works;
//! - anything else is a store name, looked up in `PIE_HOME/models/`.
//!
//! Deciding by shape and not by probing is deliberate. A rule that fell back
//! from one to the other would make the error depend on which file happened to
//! be missing, and would let a typo in a store name quietly become a relative
//! path that does not exist either.
//!
//! An HF snapshot directory still resolves, because the driver can still load
//! one and the migration is not finished. It is not the intended input: it
//! carries no compiled metadata, so serving it goes back to parsing
//! `config.json` at boot, which is the thing the artifact exists to stop.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

/// What the worker was pointed at.
///
/// The distinction is drawn **once**, here, and then carried. Everything
/// downstream asks this type rather than re-reading the extension: the answer
/// was already paid for, and a second derivation is a second chance to
/// disagree with the first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Model {
    /// A `.zt` artifact: everything the driver needs, in one file.
    Artifact(PathBuf),
    /// A HuggingFace snapshot directory, or a single checkpoint file. Legacy:
    /// no compiled metadata, so the driver falls back to `config.json`.
    Snapshot(PathBuf),
}

impl Model {
    pub fn path(&self) -> &Path {
        match self {
            Model::Artifact(path) | Model::Snapshot(path) => path,
        }
    }

    /// Everything the runtime and the drivers need out of the artifact, lifted
    /// in **one** open.
    ///
    /// `Ok(None)` for a snapshot, and for an artifact carrying only part of it
    /// — the runtime wants all or nothing, since half the metadata compiled
    /// and half probed from files beside a directory is exactly the skew the
    /// artifact removes.
    pub fn metadata(&self) -> Result<Option<pie_model::ArtifactMetadata>> {
        let Model::Artifact(path) = self else {
            return Ok(None);
        };
        // One parse. For a sharded artifact the manifest read opens and
        // validates every shard, so doing it per consumer is not free.
        let checkpoint = pie_loader::checkpoint::read::parse_checkpoint_metadata(path)
            .map_err(|err| anyhow::anyhow!("cannot read {}: {err}", path.display()))?;

        let Some(descriptor) = pie_loader::checkpoint::read::read_meta(
            &checkpoint,
            pie_model_config::DESCRIPTOR_OBJECT,
        )?
        else {
            return Ok(None);
        };
        let mut tokenizer = Vec::with_capacity(pie_tokenizer::canonical::OBJECTS.len());
        for name in pie_tokenizer::canonical::OBJECTS {
            let Some(bytes) = pie_loader::checkpoint::read::read_meta(&checkpoint, name)? else {
                return Ok(None);
            };
            tokenizer.push((name.to_string(), bytes));
        }
        Ok(Some(pie_model::ArtifactMetadata {
            tokenizer,
            descriptor,
        }))
    }
}

/// `$PIE_HOME/models/` — the store `pie model download` writes into.
fn store_dir() -> PathBuf {
    crate::paths::pie_home().join("models")
}

/// Resolves `model` — a store name or a path — to something on disk.
pub fn resolve(model: &str) -> Result<Model> {
    if model.trim().is_empty() {
        bail!("[model].model is empty; set it to a store name or a path to a .zt artifact");
    }
    if looks_like_path(model) {
        let path = PathBuf::from(model);
        if path.is_file() {
            return Ok(if is_artifact_path(&path) {
                Model::Artifact(path)
            } else {
                Model::Snapshot(path)
            });
        }
        if path.is_dir() {
            return Ok(Model::Snapshot(path));
        }
        bail!("model {model:?} does not exist");
    }

    let artifact = store_dir().join(format!("{model}.zt"));
    if artifact.is_file() {
        return Ok(Model::Artifact(artifact));
    }
    // A repo ID is a store name spelled the other way, and typing it is the
    // obvious mistake — so answer it rather than reporting a missing file.
    let by_repo_id = store_dir().join(format!("{}.zt", model.replace('/', "--")));
    if by_repo_id.is_file() {
        return Ok(Model::Artifact(by_repo_id));
    }
    bail!(
        "no model {model:?} in {}; `pie model download {model}` fetches and converts one, \
         and `pie model list` shows what is there",
        store_dir().display()
    )
}

/// Whether `path` names a `.zt`. The one place the extension is judged.
pub(crate) fn is_artifact_path(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("zt"))
}

/// Extensions that make a bare word a file rather than a store name.
///
/// A closed list, not "has an extension": model names are full of dots, and
/// `Qwen--Qwen3-0.6B` would otherwise read as a file with extension `6B`.
const CHECKPOINT_EXTENSIONS: [&str; 3] = ["zt", "gguf", "safetensors"];

/// Whether a config value is meant as a path rather than a store name.
///
/// Syntactic on purpose: a store name and a relative path can be spelled the
/// same, and letting the filesystem decide would make the meaning of a config
/// depend on the working directory.
fn looks_like_path(value: &str) -> bool {
    value.starts_with('/')
        || value.starts_with("./")
        || value.starts_with("../")
        || value.starts_with('~')
        || value.contains('\\')
        || Path::new(value).extension().is_some_and(|ext| {
            CHECKPOINT_EXTENSIONS
                .iter()
                .any(|known| ext.eq_ignore_ascii_case(known))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_loader::checkpoint::write::CheckpointWriter;
    use pie_loader::types::{DType, Encoding, TensorDecl, TensorId};

    #[test]
    fn a_path_is_a_path_and_a_name_is_a_name() {
        assert!(looks_like_path("/data/model.zt"));
        assert!(looks_like_path("./model.zt"));
        assert!(looks_like_path("../models/foo.zt"));
        assert!(looks_like_path("relative/thing.zt"));
        assert!(looks_like_path("weights.gguf"));
        // Model names are full of dots, and a "has an extension" rule would
        // read `Qwen--Qwen3-0.6B` as a file with extension `6B`.
        assert!(!looks_like_path("Qwen--Qwen3-0.6B"));
        assert!(!looks_like_path("Qwen/Qwen3-0.6B"));
        assert!(!looks_like_path("meta-llama--Llama-3.1-8B"));
        assert!(!looks_like_path("mymodel"));
    }

    #[test]
    fn an_artifact_path_resolves_to_itself() {
        let dir = tempfile::tempdir().unwrap();
        let artifact = dir.path().join("model.zt");
        std::fs::write(&artifact, b"stand-in for an artifact").unwrap();
        assert_eq!(
            resolve(artifact.to_str().unwrap()).unwrap(),
            Model::Artifact(artifact)
        );
    }

    /// A snapshot directory and a lone checkpoint still resolve, and say they
    /// are legacy — the driver reads `config.json` for those, which is what
    /// the artifact removes.
    #[test]
    fn snapshots_resolve_as_legacy() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir.path().join("snap");
        std::fs::create_dir(&snapshot).unwrap();
        let resolved = resolve(snapshot.to_str().unwrap()).unwrap();
        assert_eq!(resolved, Model::Snapshot(snapshot));

        let gguf = dir.path().join("model.gguf");
        std::fs::write(&gguf, b"x").unwrap();
        assert_eq!(
            resolve(gguf.to_str().unwrap()).unwrap(),
            Model::Snapshot(gguf)
        );
    }

    #[test]
    fn a_missing_name_says_how_to_get_one() {
        let err = resolve("definitely-not-a-model").unwrap_err().to_string();
        assert!(err.contains("pie model download"), "{err}");
        let err = resolve("./nowhere.zt").unwrap_err().to_string();
        assert!(err.contains("does not exist"), "{err}");
        assert!(resolve("").is_err());
    }

    /// Writes an artifact carrying `descriptor` plus, optionally, the whole
    /// compiled tokenizer.
    fn artifact(dir: &Path, descriptor: &[u8], whole_tokenizer: bool) -> Model {
        let path = dir.join("model.zt");
        let canonical = pie_tokenizer::Tokenizer::from_vocab(&["a".to_string(), "b".to_string()])
            .to_canonical()
            .unwrap();
        let mut writer = CheckpointWriter::create(&path, &Default::default()).unwrap();
        // Ascending names: `model/…` sorts before `tokenizer/…`.
        writer
            .add_meta(pie_model_config::DESCRIPTOR_OBJECT, descriptor)
            .unwrap();
        for (name, bytes) in canonical.objects() {
            if !whole_tokenizer && name == pie_tokenizer::canonical::MERGE_TABLE {
                continue;
            }
            writer.add_meta(name, bytes).unwrap();
        }
        writer
            .add_tensor(
                &TensorDecl {
                    id: TensorId(0),
                    name: "w".to_string(),
                    shape: vec![4],
                    encoding: Encoding::Raw(DType::U8),
                    alignment: 1,
                    visibility: Default::default(),
                },
                &[1u8, 2, 3, 4],
            )
            .unwrap();
        writer.finish().unwrap();
        Model::Artifact(path)
    }

    /// An artifact hands over its compiled metadata whole, and the tokenizer
    /// that comes back tokenizes like the one that went in.
    #[test]
    fn an_artifact_hands_over_its_compiled_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let descriptor = br#"{"version":"pie.model/1","vocab_size":7,"num_hidden_layers":4}"#;
        let lifted = artifact(dir.path(), descriptor, true)
            .metadata()
            .unwrap()
            .expect("no metadata");

        assert_eq!(
            lifted.tokenizer.len(),
            pie_tokenizer::canonical::OBJECTS.len()
        );
        assert_eq!(lifted.descriptor, descriptor);

        // The runtime's own reconstruction, exercised here so a break shows up
        // as a worker test rather than only at serve time.
        let rebuilt = pie_tokenizer::canonical::CanonicalTokenizer::from_objects(|name| {
            lifted
                .tokenizer
                .iter()
                .find(|(have, _)| have == name)
                .map(|(_, bytes)| bytes.clone())
        })
        .unwrap();
        let rebuilt = pie_tokenizer::Tokenizer::from_canonical(&rebuilt).unwrap();
        assert_eq!(rebuilt.vocab_size(), 2);
    }

    /// All of the metadata or none of it: half compiled and half probed from
    /// files beside a snapshot is the skew the artifact removes.
    #[test]
    fn an_artifact_missing_part_of_its_metadata_hands_over_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let partial = artifact(dir.path(), br#"{"version":"pie.model/1"}"#, false);
        assert!(partial.metadata().unwrap().is_none());
    }

    /// A legacy snapshot has no metadata, and saying so is not an error — it
    /// is what selects the driver's fallback. Answered without opening
    /// anything.
    #[test]
    fn a_snapshot_carries_no_metadata() {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            Model::Snapshot(dir.path().to_path_buf())
                .metadata()
                .unwrap()
                .is_none()
        );
    }
}
