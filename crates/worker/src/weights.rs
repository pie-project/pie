//! Finding the model a worker is configured to serve — **load-at-boot only**.
//!
//! The worker never downloads and never converts (R3). What it resolves is a
//! name or a path to something already on disk, and under this plan that
//! something is one `.zt` artifact: weights, compiled tokenizer and compiled
//! checkpoint config together, written by `pie model import`.
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
//! An HF snapshot directory still resolves, because the engine can still load
//! one and the migration is not finished. It is not the intended input: it
//! carries no compiled metadata, so serving it goes back to parsing
//! `config.json` at boot, which is the thing the artifact exists to stop.

use std::path::{Path, PathBuf};

use ::runtime::model::ModelMetadata;
use anyhow::{Result, anyhow, bail};
// By item, not by module. The crate is `checkpoint` and the local below is
// also called `checkpoint`; spelling the calls `checkpoint::file::read::…`
// would put a third use of the word between them, and both names here say
// what they do without the path.
use checkpoint::file::read::{parse_metadata, read_meta};

/// The artifact object the checkpoint's own `config.json` is written under.
///
/// It was `model::serve::encoding::CONFIG_OBJECT`, beside an `Encoding` that
/// parsed the document. M18 deleted that module, and the parser did not come
/// with it: the loader reads a checkpoint's quantization off its STORED
/// tensor encodings now, not off what its config claims, so the one reader
/// this name had is gone. The NAME still has two — the writer below in this
/// file's tests, and the reader in [`Model::metadata`] — and both are here.
pub const CONFIG_OBJECT: &str = "model/config";

/// What the worker was pointed at.
///
/// The distinction is drawn **once**, here, and then carried. Everything
/// downstream asks this type rather than re-reading the extension: the answer
/// was already paid for, and a second derivation is a second chance to
/// disagree with the first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Model {
    /// A `.zt` artifact: everything the engine needs, in one file.
    Artifact(PathBuf),
    /// A HuggingFace snapshot directory, or a single checkpoint file. Legacy:
    /// no compiled metadata, so the engine falls back to `config.json`.
    Snapshot(PathBuf),
}

impl Model {
    pub fn path(&self) -> &Path {
        match self {
            Model::Artifact(path) | Model::Snapshot(path) => path,
        }
    }

    /// Everything the runtime and the engines need, lifted in **one** open.
    ///
    /// The config is always produced: an artifact carries it embedded and a
    /// snapshot has it on disk, and both hand over the SAME bytes. That is
    /// what lets everything downstream read one document instead of keeping a
    /// second path that parses the files beside a snapshot.
    ///
    /// It is carried and not parsed. The last in-tree reader of the config's
    /// own fields was `model::serve::encoding::Encoding`, which asked what
    /// quantization the checkpoint declared; M18 deleted it, because the
    /// loader reads that off the stored tensor encodings instead of believing
    /// the document. Everything else the old `pie.model/1` document carried
    /// is a catalog row's.
    ///
    /// The tokenizer half stays optional, and all-or-nothing: half the
    /// tokenizer compiled and half probed from files is the skew the artifact
    /// removes, so a partial one is treated as absent and the files win.
    pub fn metadata(&self) -> Result<ModelMetadata> {
        let Model::Artifact(path) = self else {
            return Ok(ModelMetadata {
                tokenizer: None,
                config: lift_snapshot_config(self.path())?,
            });
        };
        // One parse. For a sharded artifact the manifest read opens and
        // validates every shard, so doing it per consumer is not free.
        let checkpoint =
            parse_metadata(path).map_err(|err| anyhow!("cannot read {}: {err}", path.display()))?;

        let config = read_meta(&checkpoint, CONFIG_OBJECT)?
            .ok_or_else(|| {
                anyhow!(
                    "artifact {} carries no {}; it was written when an artifact \
                     carried a resolved `pie.model/1` document instead of the \
                     checkpoint's own config. Re-import it with `pie model import`",
                    path.display(),
                    CONFIG_OBJECT,
                )
            })?;

        let mut tokenizer = Vec::with_capacity(tokenizer::canonical::OBJECTS.len());
        for name in tokenizer::canonical::OBJECTS {
            let Some(bytes) = read_meta(&checkpoint, name)? else {
                tokenizer.clear();
                break;
            };
            tokenizer.push((name.to_string(), bytes));
        }
        Ok(ModelMetadata {
            tokenizer: (!tokenizer.is_empty()).then_some(tokenizer),
            config,
        })
    }
}

/// Lift a snapshot's `config.json`, verbatim.
///
/// # Why verbatim, when this used to normalize
///
/// It used to run the config through an 845-line normalizer into a
/// `pie.model/1` descriptor — ~40 fields, resolved from a 136-field
/// schema — so that an engine would not have to parse HuggingFace's
/// spelling variations itself. That was the right shape of answer to
/// the wrong question. Every one of those fields except three is a
/// fact about the MODEL, and a model is a catalog row now: the row
/// states its geometry, and a checkpoint is matched to it by its
/// TENSORS rather than believed on the strength of what its config
/// claims.
///
/// The three that remain are the declared quantization — method, bits,
/// group size — and they are the only ones a row cannot state, because
/// they are properties of the FILES and Qwen3-8B ships as four
/// different sets of them. Nothing in the tree reads them out of this
/// document any more — the loader takes a checkpoint's encodings off the
/// tensors themselves — but the config still crosses whole, because
/// carrying the checkpoint's own bytes is cheaper than deciding, here, which
/// of its fields a future reader will want.
///
/// Verbatim also removes a class of failure the normalizer had by
/// construction: it was a second reader of a document the checkpoint
/// already carries, and a normalizer that defaults a missing field
/// cannot be told apart from a config that states that value.
///
/// A missing or unreadable `config.json` is an error here, not a
/// fallback. It used to be one: the engine would find nothing and parse
/// the file itself. That branch is what this function exists to delete,
/// so restoring it as an error path would restore the thing being
/// removed.
fn lift_snapshot_config(path: &Path) -> Result<Vec<u8>> {
    // A snapshot may be the directory or a lone checkpoint file inside it.
    let dir = if path.is_dir() {
        path
    } else {
        path.parent().ok_or_else(|| {
            anyhow!(
                "checkpoint {} has no directory to read config.json from",
                path.display()
            )
        })?
    };
    let config = dir.join("config.json");
    let raw = std::fs::read_to_string(&config).map_err(|err| {
        anyhow!(
            "cannot read {}: {err}; a snapshot must carry the config.json its \
             encoding is read from (`pie model import` writes an artifact \
             that carries it already)",
            config.display(),
        )
    })?;
    // Parsed only to REFUSE a config that is not JSON. An engine that
    // received an unparseable document would refuse it too, but several
    // frames later and with a snapshot already half-opened.
    serde_json::from_str::<serde_json::Value>(&raw)
        .map_err(|err| anyhow!("cannot parse {}: {err}", config.display()))?;
    Ok(raw.into_bytes())
}

/// `$PIE_HOME/models/` — the store `pie model import` writes into.
///
/// The layout is one directory per model: `<name>/archive.zt` is the general
/// form, `<name>/runtime/<key>.zt` are per-target builds. `src/local/store.rs`
/// owns it and is where the shape is explained; this repeats only enough of it
/// to find a file, because the worker cannot depend on the CLI crate.
fn store_dir() -> PathBuf {
    crate::paths::pie_home().join("models")
}

/// The archive of the model stored under `name`, if there is one.
///
/// Falls back to a flat `<name>.zt`, which is what pie wrote before the store
/// gained a directory per model. Read, never written: a store that predates
/// the change should keep serving rather than report every model missing.
fn archive_in(store: &Path, name: &str) -> Option<PathBuf> {
    let two_layer = store.join(name).join("archive.zt");
    if two_layer.is_file() {
        return Some(two_layer);
    }
    let flat = store.join(format!("{name}.zt"));
    flat.is_file().then_some(flat)
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

    let store = store_dir();
    if let Some(artifact) = archive_in(&store, model) {
        return Ok(Model::Artifact(artifact));
    }
    // A repo ID is a store name spelled the other way, and typing it is the
    // obvious mistake — so answer it rather than reporting a missing file.
    if let Some(artifact) = archive_in(&store, &model.replace('/', "--")) {
        return Ok(Model::Artifact(artifact));
    }
    bail!(
        "no model {model:?} in {}; `pie model import {model}` fetches and converts one, \
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
    use checkpoint::file::write::Writer;
    use checkpoint::types::{DType, Encoding, TensorDecl, TensorId};

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
    /// are legacy — the engine reads `config.json` for those, which is what
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

    /// A store name finds `<name>/archive.zt`, and falls back to `<name>.zt`.
    ///
    /// The two-layer store gives each model a directory so that per-target
    /// builds have somewhere to live that is not a sibling of the archive.
    /// The flat spelling is what pie wrote before that, and it still resolves:
    /// dropping it would have made every model a user already had report as
    /// missing, which reads as data loss whether or not the bytes are there.
    ///
    /// The archive wins when both exist, because it is the one a current pie
    /// wrote.
    #[test]
    fn a_store_name_finds_the_archive_and_falls_back_to_the_flat_spelling() {
        let store = tempfile::tempdir().unwrap();
        let store = store.path();
        assert_eq!(archive_in(store, "qwen"), None);

        let flat = store.join("qwen.zt");
        std::fs::write(&flat, b"stand-in").unwrap();
        assert_eq!(archive_in(store, "qwen"), Some(flat));

        let archive = store.join("qwen").join("archive.zt");
        std::fs::create_dir_all(archive.parent().unwrap()).unwrap();
        std::fs::write(&archive, b"stand-in").unwrap();
        assert_eq!(
            archive_in(store, "qwen"),
            Some(archive),
            "the two-layer archive wins over a leftover flat file"
        );

        // A model directory with no archive in it is not a model. `runtime/`
        // alone is what an interrupted `pie model remove` can leave behind.
        std::fs::create_dir_all(store.join("empty").join("runtime")).unwrap();
        assert_eq!(archive_in(store, "empty"), None);
    }

    #[test]
    fn a_missing_name_says_how_to_get_one() {
        let err = resolve("definitely-not-a-model").unwrap_err().to_string();
        assert!(err.contains("pie model import"), "{err}");
        let err = resolve("./nowhere.zt").unwrap_err().to_string();
        assert!(err.contains("does not exist"), "{err}");
        assert!(resolve("").is_err());
    }

    /// Writes an artifact carrying `config` plus, optionally, the whole
    /// compiled tokenizer.
    fn artifact(dir: &Path, config: &[u8], whole_tokenizer: bool) -> Model {
        let path = dir.join("model.zt");
        let canonical = tokenizer::Tokenizer::from_vocab(&["a".to_string(), "b".to_string()])
            .to_canonical()
            .unwrap();
        let mut writer = Writer::create(&path, &Default::default()).unwrap();
        // Ascending names: `model/…` sorts before `tokenizer/…`.
        writer.add_meta(CONFIG_OBJECT, config).unwrap();
        for (name, bytes) in canonical.objects() {
            if !whole_tokenizer && name == tokenizer::canonical::MERGE_TABLE {
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
        let config = br#"{"version":"pie.model/1","vocab_size":7,"num_hidden_layers":4}"#;
        let lifted = artifact(dir.path(), config, true).metadata().unwrap();

        let objects = lifted.tokenizer.as_ref().expect("no compiled tokenizer");
        assert_eq!(objects.len(), tokenizer::canonical::OBJECTS.len());
        assert_eq!(lifted.config, config);

        // The runtime's own reconstruction, exercised here so a break shows up
        // as a worker test rather than only at serve time.
        let rebuilt = tokenizer::canonical::CanonicalTokenizer::from_objects(|name| {
            objects
                .iter()
                .find(|(have, _)| have == name)
                .map(|(_, bytes)| bytes.clone())
        })
        .unwrap();
        let rebuilt = tokenizer::Tokenizer::from_canonical(&rebuilt).unwrap();
        // AGAINST THE ORIGINAL, not against a number. `from_vocab(["a","b"])`
        // builds a byte-level tokenizer, so its vocabulary is 258 -- the 256
        // single-byte tokens plus the two named ones -- and the literal `2`
        // that used to be here was a guess about the constructor rather than
        // a claim about the round trip. Comparing the two ends says the thing
        // this test is named for, and says it whatever the constructor does.
        let original = tokenizer::Tokenizer::from_vocab(&["a".to_string(), "b".to_string()]);
        assert_eq!(rebuilt.vocab_size(), original.vocab_size());
    }

    /// All of the tokenizer or none of it: half compiled and half probed from
    /// files beside a snapshot is the skew the artifact removes.
    ///
    /// The config is unaffected — it is a different object with a
    /// different completeness question, and an artifact that carries one but
    /// not a whole tokenizer still has a model config worth reading.
    #[test]
    fn an_artifact_missing_part_of_its_tokenizer_hands_over_none_of_it() {
        let dir = tempfile::tempdir().unwrap();
        let config = br#"{"version":"pie.model/1","vocab_size":7}"#;
        let lifted = artifact(dir.path(), config, false).metadata().unwrap();
        assert!(lifted.tokenizer.is_none());
        assert_eq!(lifted.config, config);
    }

    /// **Both input forms hand over the checkpoint's own config, byte
    /// for byte.**
    ///
    /// This is the property that let the second and third normalizers go — the
    /// engines' `config.json` parsers and the runtime's own key probes — so it
    /// is pinned rather than left implied. If a snapshot ever again reaches
    /// them without one, there is nothing left to fall back to.
    ///
    /// Verbatim is stronger than the normalized document it replaced: two
    /// forms of the same model now hand over IDENTICAL bytes, where before
    /// they handed over two normalizations that had to agree.
    #[test]
    fn every_model_form_produces_the_checkpoints_config() {
        let dir = tempfile::tempdir().unwrap();

        // The artifact form: the embedded bytes, read back rather than
        // derived from a `config.json` it does not have.
        let compiled = br#"{"model_type":"llama","vocab_size":7,"num_hidden_layers":4}"#;
        let model = artifact(dir.path(), compiled, true);
        assert_eq!(model.metadata().unwrap().config, compiled);

        // The snapshot form: lifted here, from `config.json`, and with no
        // compiled tokenizer to hand over.
        let snap = dir.path().join("snapshot");
        std::fs::create_dir(&snap).unwrap();
        std::fs::write(
            snap.join("config.json"),
            br#"{"architectures":["LlamaForCausalLM"],"model_type":"llama",
                 "hidden_size":64,"num_hidden_layers":2,
                 "num_attention_heads":4,"num_key_value_heads":4,
                 "intermediate_size":128,"vocab_size":32,"max_position_embeddings":128,
                 "rms_norm_eps":1e-5,"rope_theta":10000.0}"#,
        )
        .unwrap();
        let lifted = Model::Snapshot(snap.clone()).metadata().unwrap();
        assert!(lifted.tokenizer.is_none());
        let doc: serde_json::Value = serde_json::from_slice(&lifted.config).unwrap();
        // VERBATIM: the keys are the checkpoint's own spelling, not a
        // normalizer's. `num_hidden_layers` and `vocab_size` are a row's
        // answers now, and nothing downstream reads them from here — this
        // asserts only that the bytes arrived unaltered.
        assert_eq!(doc["num_hidden_layers"], 2);
        assert_eq!(doc["vocab_size"], 32);
        assert_eq!(doc["model_type"], "llama");
        // THE QUANTIZATION BLOCK, absent here, which is not a defect: most
        // checkpoints declare none, and an absent block is an unquantized
        // checkpoint rather than a missing answer. This used to run it
        // through `model::serve::encoding::Encoding`; that parser is deleted
        // and had no non-test caller, so what is left to assert is that the
        // block is not there and nothing invented one.
        assert!(
            doc.get("quantization_config").is_none() && doc.get("quantization").is_none(),
            "an unquantized snapshot declares nothing"
        );

        // A checkpoint file inside the snapshot reads the config beside it.
        let gguf = snap.join("model.gguf");
        std::fs::write(&gguf, b"x").unwrap();
        assert_eq!(
            Model::Snapshot(gguf).metadata().unwrap().config,
            lifted.config
        );
    }

    /// A snapshot without a `config.json` is an error, not a silent fallback.
    #[test]
    fn a_snapshot_without_a_config_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let err = Model::Snapshot(dir.path().to_path_buf())
            .metadata()
            .unwrap_err()
            .to_string();
        assert!(err.contains("config.json"), "{err}");
    }
}
