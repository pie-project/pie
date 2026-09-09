use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use runtime::model::ModelMetadata;

use crate::backend::EngineCapabilities;
use crate::config;
use checkpoint::file::read::{parse_metadata, read_meta};

pub const CONFIG_OBJECT: &str = "model/config";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Model {
    Artifact(PathBuf),
    Snapshot(PathBuf),
}

impl Model {
    pub fn path(&self) -> &Path {
        match self {
            Model::Artifact(path) | Model::Snapshot(path) => path,
        }
    }

    pub fn metadata(&self) -> Result<ModelMetadata> {
        let Model::Artifact(path) = self else {
            return Ok(ModelMetadata {
                tokenizer: None,
                config: lift_snapshot_config(self.path())?,
            });
        };
        let checkpoint =
            parse_metadata(path).map_err(|err| anyhow!("cannot read {}: {err}", path.display()))?;

        let config = read_meta(&checkpoint, CONFIG_OBJECT)?.ok_or_else(|| {
            anyhow!(
                "artifact {} carries no {}; it was written when an artifact \
                     carried a resolved `pie.model/1` document instead of the \
                     checkpoint's own config. Re-import it with `pie model import`",
                path.display(),
                CONFIG_OBJECT,
            )
        })?;

        let mut tokenizer = Vec::with_capacity(tokenizer::canonical::OBJECTS.len() + 1);
        for name in tokenizer::canonical::OBJECTS {
            let Some(bytes) = read_meta(&checkpoint, name)? else {
                tokenizer.clear();
                break;
            };
            tokenizer.push((name.to_string(), bytes));
        }
        if !tokenizer.is_empty() {
            for name in tokenizer::canonical::OPTIONAL_OBJECTS {
                if let Some(bytes) = read_meta(&checkpoint, name)? {
                    tokenizer.push((name.to_string(), bytes));
                }
            }
        }
        Ok(ModelMetadata {
            tokenizer: (!tokenizer.is_empty()).then_some(tokenizer),
            config,
        })
    }
}

fn lift_snapshot_config(path: &Path) -> Result<Vec<u8>> {
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
    serde_json::from_str::<serde_json::Value>(&raw)
        .map_err(|err| anyhow!("cannot parse {}: {err}", config.display()))?;
    Ok(raw.into_bytes())
}

fn store_dir() -> PathBuf {
    bootstrap::paths::pie_home().join("models")
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Want<'a> {
    pub backend: Option<&'a str>,
    pub sku: Option<&'a str>,
}

fn archive_in(store: &Path, name: &str, want: Want<'_>) -> Result<Option<PathBuf>> {
    let model_dir = store.join(name);
    if model_dir.is_dir()
        && let Some(found) = pick(
            &model_dir,
            checkpoint::file::read::discover_zt_files(&model_dir),
            want,
        )?
    {
        return Ok(Some(found));
    }
    if let Ok(parsed) = checkpoint::serving::Name::parse(&format!("{name}.zt")) {
        let file = store.join(&parsed.slug).join(parsed.render());
        if file.is_file() {
            return Ok(Some(file));
        }
        if let Some(file) = std::fs::read_dir(store).ok().and_then(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_dir())
                .find(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| checkpoint::serving::slugify(name) == parsed.slug)
                })
                .map(|dir| dir.join(parsed.render()))
                .filter(|file| file.is_file())
        }) {
            return Ok(Some(file));
        }
    }
    let flat = store.join(format!("{name}.zt"));
    Ok(flat.is_file().then_some(flat))
}

fn pick(dir: &Path, found: Vec<PathBuf>, want: Want<'_>) -> Result<Option<PathBuf>> {
    if found.len() <= 1 {
        return Ok(found.into_iter().next());
    }
    let named: Vec<(PathBuf, Option<checkpoint::serving::Name>)> = found
        .into_iter()
        .map(|path| {
            let parsed = path
                .file_name()
                .and_then(|name| name.to_str())
                .and_then(|name| checkpoint::serving::Name::parse(name).ok());
            (path, parsed)
        })
        .collect();

    let mut wanted: Vec<&(PathBuf, Option<checkpoint::serving::Name>)> = named.iter().collect();
    if let Some(backend) = want.backend {
        wanted.retain(|(_, parsed)| parsed.as_ref().is_some_and(|it| it.backend == backend));
    }
    if wanted.len() > 1
        && let Some(sku) = want.sku
    {
        let sku = checkpoint::serving::slugify(sku);
        wanted.retain(|(_, parsed)| parsed.as_ref().is_some_and(|it| it.sku == sku));
    }
    if let [(path, _)] = wanted[..] {
        return Ok(Some(path.clone()));
    }

    let candidates = named
        .iter()
        .filter_map(|(path, _)| path.file_stem())
        .map(|stem| format!("`{}`", stem.to_string_lossy()))
        .collect::<Vec<_>>()
        .join(", ");
    let asked = match (want.backend, want.sku) {
        (Some(backend), Some(sku)) => {
            format!(" and none of them is `{sku}` for {backend}")
        }
        (Some(backend), None) => format!(" and {} of them are for {backend}", wanted.len()),
        _ => String::new(),
    };
    bail!(
        "{} holds {} artifacts of one model{asked}. Name the one to serve in \
         `[model] model` — {candidates} — or state `[model] sku` to pick \
         between rows of one backend.",
        dir.display(),
        named.len(),
    )
}

pub fn resolve(model: &str, want: Want<'_>) -> Result<Model> {
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
    if let Some(artifact) = archive_in(&store, model, want)? {
        return Ok(Model::Artifact(artifact));
    }
    if let Some(artifact) = archive_in(&store, &model.replace('/', "--"), want)? {
        return Ok(Model::Artifact(artifact));
    }
    bail!(
        "no model {model:?} in {}; `pie model import {model}` fetches and converts one, \
         and `pie model list` shows what is there",
        store_dir().display()
    )
}

pub(crate) fn is_artifact_path(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("zt"))
}

const CHECKPOINT_EXTENSIONS: [&str; 3] = ["zt", "gguf", "safetensors"];

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

pub(crate) fn model_identity(
    user_cfg: &config::Config,
    caps: &EngineCapabilities,
    artifact_digest: &[u8; 32],
    component: crate::executor::ModelComponent,
) -> Result<crate::executor::ModelIdentity> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(user_cfg.model.name.as_bytes());
    hasher.update(artifact_digest);
    hasher.update(caps.device.backend.as_bytes());
    hasher.update(&caps.profile.vocab.to_le_bytes());
    hasher.update(&caps.profile.num_layers.to_le_bytes());
    hasher.update(&caps.limits.max_context.to_le_bytes());
    hasher.update(caps.profile.activation_name().as_bytes());
    hasher.update(&caps.pools.kv_page_size.to_le_bytes());
    hasher.update(format!("{:?}", user_cfg.model.engine.kind).as_bytes());
    hasher.update(user_cfg.model.engine.activation_dtype.as_bytes());
    hasher.update(user_cfg.model.weight_dtype.as_bytes());
    Ok(crate::executor::ModelIdentity {
        hash: *hasher.finalize().as_bytes(),
        component,
    })
}

pub(crate) fn manifest_digest(path: &Path) -> Result<Option<[u8; 32]>> {
    let identity = checkpoint::file::zt::artifact_identity(path)
        .map_err(|err| anyhow!("reading the identity of {path:?}: {err}"))?;
    Ok(identity.map(|bytes| *blake3::hash(&bytes).as_bytes()))
}

pub(crate) fn model_artifact_digest(snapshot_dir: &Path) -> Result<[u8; 32]> {
    if let Some(digest) = manifest_digest(snapshot_dir)? {
        return Ok(digest);
    }

    let components = snapshot_dir.components().collect::<Vec<_>>();
    for pair in components.windows(2) {
        if pair[0].as_os_str() == "snapshots" {
            let revision = pair[1].as_os_str().to_string_lossy();
            if !revision.is_empty() {
                return Ok(*blake3::hash(revision.as_bytes()).as_bytes());
            }
        }
    }

    fn collect_files(current: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
        if current.is_file() {
            files.push(current.to_path_buf());
            return Ok(());
        }
        let mut entries = std::fs::read_dir(current)
            .with_context(|| format!("reading model artifact directory {current:?}"))?
            .collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() {
                let target = std::fs::canonicalize(&path)?;
                if target.is_file() {
                    files.push(path);
                }
            } else if metadata.is_dir() {
                collect_files(&path, files)?;
            } else if metadata.is_file() {
                files.push(path);
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    collect_files(snapshot_dir, &mut files)?;
    files.sort();
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    for path in files {
        use std::io::Read;

        let relative = path.strip_prefix(snapshot_dir).unwrap_or(&path);
        hasher.update(relative.to_string_lossy().as_bytes());
        let mut file = std::fs::File::open(&path)
            .with_context(|| format!("opening model artifact {path:?}"))?;
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }
    Ok(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint::file::write::Writer;
    use checkpoint::types::{DType, Encoding, TensorDecl, TensorId};

    #[test]
    fn weights_every_case() {
        an_artifact_path_resolves_to_itself();
        a_store_name_finds_the_name_a_stamped_import_wrote();
        a_store_name_naming_two_shells_resolves_to_the_one_this_build_hosts();
        two_rows_of_one_backend_are_told_apart_by_the_stated_sku();
        an_artifact_hands_over_its_compiled_metadata();
        every_model_form_produces_the_checkpoints_config();
        a_snapshot_without_a_config_says_so();
    }

    fn an_artifact_path_resolves_to_itself() {
        let dir = tempfile::tempdir().unwrap();
        let artifact = dir.path().join("model.zt");
        std::fs::write(&artifact, b"stand-in for an artifact").unwrap();
        assert_eq!(
            resolve(artifact.to_str().unwrap(), Want::default()).unwrap(),
            Model::Artifact(artifact)
        );
    }

    fn a_store_name_finds_the_name_a_stamped_import_wrote() {
        let store = tempfile::tempdir().unwrap();
        let store = store.path();

        let model = store.join("deepseek");
        std::fs::create_dir_all(&model).unwrap();
        let specialized = model.join("deepseek.dsv4-flash-u4g64-u2g64-kv-bf16.metal.zt");
        std::fs::write(&specialized, b"stand-in").unwrap();
        assert_eq!(
            archive_in(store, "deepseek", Want::default()).unwrap(),
            Some(specialized),
            "the name `pie model import` writes under a serving stamp has to \
             be the name the worker resolves"
        );
    }

    fn a_store_name_naming_two_shells_resolves_to_the_one_this_build_hosts() {
        let store = tempfile::tempdir().unwrap();
        let store = store.path();
        let model = store.join("glm");
        std::fs::create_dir_all(&model).unwrap();
        let cuda = model.join("glm.glm53-flash-u8g64-u2g64-kv-bf16.cuda.zt");
        let vulkan = model.join("glm.glm53-flash-u8g64-u2g64-kv-bf16.vulkan.zt");
        std::fs::write(&cuda, b"stand-in").unwrap();
        std::fs::write(&vulkan, b"stand-in").unwrap();

        for (backend, expected) in [("cuda", &cuda), ("vulkan", &vulkan)] {
            assert_eq!(
                archive_in(
                    store,
                    "glm",
                    Want {
                        backend: Some(backend),
                        sku: None
                    }
                )
                .unwrap()
                .as_ref(),
                Some(expected),
                "a {backend} build serves the {backend} artifact"
            );
        }

        let why = archive_in(
            store,
            "glm",
            Want {
                backend: Some("metal"),
                sku: None,
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            why.contains("glm.glm53-flash-u8g64-u2g64-kv-bf16.cuda"),
            "{why}"
        );
        assert!(
            why.contains("glm.glm53-flash-u8g64-u2g64-kv-bf16.vulkan"),
            "{why}"
        );

        assert_eq!(
            archive_in(
                store,
                "glm.glm53-flash-u8g64-u2g64-kv-bf16.vulkan",
                Want {
                    backend: Some("cuda"),
                    sku: None
                }
            )
            .unwrap(),
            Some(vulkan),
            "naming the file is how an operator overrides the pick"
        );
    }

    fn two_rows_of_one_backend_are_told_apart_by_the_stated_sku() {
        let store = tempfile::tempdir().unwrap();
        let store = store.path();
        let model = store.join("glm");
        std::fs::create_dir_all(&model).unwrap();
        let plain = model.join("glm.glm53-flash-u8g64-u2g64-kv-bf16.vulkan.zt");
        let mtp = model.join("glm.glm53-flash-mtp-u8g64-u2g64-kv-bf16.vulkan.zt");
        std::fs::write(&plain, b"stand-in").unwrap();
        std::fs::write(&mtp, b"stand-in").unwrap();

        let want = |sku| Want {
            backend: Some("vulkan"),
            sku: Some(sku),
        };
        assert_eq!(
            archive_in(store, "glm", want("glm53-flash-mtp-u8g64-u2g64-kv-bf16")).unwrap(),
            Some(mtp)
        );
        assert_eq!(
            archive_in(store, "glm", want("GLM53-Flash-u8g64-u2g64-kv-bf16")).unwrap(),
            Some(plain)
        );
        let why = archive_in(
            store,
            "glm",
            Want {
                backend: Some("vulkan"),
                sku: None,
            },
        )
        .unwrap_err()
        .to_string();
        assert!(why.contains("[model] sku"), "{why}");
    }

    fn artifact(dir: &Path, config: &[u8], whole_tokenizer: bool) -> Model {
        let path = dir.join("model.zt");
        let canonical = tokenizer::Tokenizer::from_vocab(&["a".to_string(), "b".to_string()])
            .to_canonical()
            .unwrap();
        let mut writer = Writer::create(&path, &Default::default()).unwrap();
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

    fn an_artifact_hands_over_its_compiled_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let config = br#"{"version":"pie.model/1","vocab_size":7,"num_hidden_layers":4}"#;
        let lifted = artifact(dir.path(), config, true).metadata().unwrap();

        let objects = lifted.tokenizer.as_ref().expect("no compiled tokenizer");
        assert_eq!(objects.len(), tokenizer::canonical::OBJECTS.len());
        assert_eq!(lifted.config, config);

        let rebuilt = tokenizer::canonical::CanonicalTokenizer::from_objects(|name| {
            objects
                .iter()
                .find(|(have, _)| have == name)
                .map(|(_, bytes)| bytes.clone())
        })
        .unwrap();
        let rebuilt = tokenizer::Tokenizer::from_canonical(&rebuilt).unwrap();
        let original = tokenizer::Tokenizer::from_vocab(&["a".to_string(), "b".to_string()]);
        assert_eq!(rebuilt.vocab_size(), original.vocab_size());
    }

    fn every_model_form_produces_the_checkpoints_config() {
        let dir = tempfile::tempdir().unwrap();

        let compiled = br#"{"model_type":"llama","vocab_size":7,"num_hidden_layers":4}"#;
        let model = artifact(dir.path(), compiled, true);
        assert_eq!(model.metadata().unwrap().config, compiled);

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
        assert_eq!(doc["num_hidden_layers"], 2);
        assert_eq!(doc["vocab_size"], 32);
        assert_eq!(doc["model_type"], "llama");
        assert!(
            doc.get("quantization_config").is_none() && doc.get("quantization").is_none(),
            "an unquantized snapshot declares nothing"
        );

        let gguf = snap.join("model.gguf");
        std::fs::write(&gguf, b"x").unwrap();
        assert_eq!(
            Model::Snapshot(gguf).metadata().unwrap().config,
            lifted.config
        );
    }

    fn a_snapshot_without_a_config_says_so() {
        let dir = tempfile::tempdir().unwrap();
        let err = Model::Snapshot(dir.path().to_path_buf())
            .metadata()
            .unwrap_err()
            .to_string();
        assert!(err.contains("config.json"), "{err}");
    }
}
