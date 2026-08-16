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
//! An HF snapshot directory still resolves, because the driver can still load
//! one and the migration is not finished. It is not the intended input: it
//! carries no compiled metadata, so serving it goes back to parsing
//! `config.json` at boot, which is the thing the artifact exists to stop.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};

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

    /// Everything the runtime and the drivers need, lifted in **one** open.
    ///
    /// The config is always produced: an artifact carries it embedded and a
    /// snapshot has it on disk, and both hand over the SAME bytes. That is
    /// what lets everything downstream read one document instead of keeping a
    /// second path that parses the files beside a snapshot.
    ///
    /// One reader is left, and it wants three fields:
    /// [`Encoding::from_config_json`](model::encoding::Encoding::from_config_json)
    /// asks what quantization the checkpoint declares. Everything else the
    /// old `pie.model/1` document carried is a catalog row's now.
    ///
    /// The tokenizer half stays optional, and all-or-nothing: half the
    /// tokenizer compiled and half probed from files is the skew the artifact
    /// removes, so a partial one is treated as absent and the files win.
    pub fn metadata(&self) -> Result<model::ModelMetadata> {
        let Model::Artifact(path) = self else {
            return Ok(model::ModelMetadata {
                tokenizer: None,
                config: lift_snapshot_config(self.path())?,
            });
        };
        // One parse. For a sharded artifact the manifest read opens and
        // validates every shard, so doing it per consumer is not free.
        let checkpoint = model_loader::checkpoint::read::parse_checkpoint_metadata(path)
            .map_err(|err| anyhow!("cannot read {}: {err}", path.display()))?;

        let config =
            model_loader::checkpoint::read::read_meta(&checkpoint, model::encoding::CONFIG_OBJECT)?
                .ok_or_else(|| {
                    anyhow!(
                        "artifact {} carries no {}; it was written when an artifact \
                 carried a resolved `pie.model/1` document instead of the \
                 checkpoint's own config. Re-import it with `pie model import`",
                        path.display(),
                        model::encoding::CONFIG_OBJECT,
                    )
                })?;

        let mut tokenizer = Vec::with_capacity(tokenizer::canonical::OBJECTS.len());
        for name in tokenizer::canonical::OBJECTS {
            let Some(bytes) = model_loader::checkpoint::read::read_meta(&checkpoint, name)? else {
                tokenizer.clear();
                break;
            };
            tokenizer.push((name.to_string(), bytes));
        }
        Ok(model::ModelMetadata {
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
/// schema — so that a driver would not have to parse HuggingFace's
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
/// different sets of them. [`model::encoding::Encoding::from_config_json`]
/// reads exactly those three, so what has to cross is the config
/// itself.
///
/// Verbatim also removes a class of failure the normalizer had by
/// construction: it was a second reader of a document the checkpoint
/// already carries, and a normalizer that defaults a missing field
/// cannot be told apart from a config that states that value.
///
/// A missing or unreadable `config.json` is an error here, not a
/// fallback. It used to be one: the driver would find nothing and parse
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
    // Parsed only to REFUSE a config that is not JSON. A driver that
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

/// What a boot is asking for, in the terms a runtime artifact states itself in.
///
/// Every field decides which tensors exist. `pie model build` writes the same
/// six facts into the artifact it produces (see
/// [`model_loader::checkpoint::meta`]), and a runtime is bound only when all
/// six agree — anything less is a file that answers a different question.
///
/// Note what is *not* here: a quantization or a MoE request. The serve path
/// authors with `RuntimeQuant::None` and `Mxfp4MoeRequest::Auto`, hardcoded in
/// `crates/model/src/boot.rs`, so those are constants of this side rather than
/// choices, and they are compared against the artifact as such. The day serve
/// learns to ask for a quantization is the day they become fields.
#[derive(Debug, Clone)]
pub struct Request {
    /// `cuda`, `metal`, `vulkan` or `wgpu` — `Flavor::as_str`, which is the
    /// same vocabulary `pie model build --backend` takes.
    pub backend: String,
    /// `driver_api::ModelComponent`'s lowercase name.
    pub component: String,
    /// The tensor-parallel degree this boot will bind at.
    pub tp_size: usize,
}

/// Bind the prebuilt runtime that answers `request`, if the store holds one.
///
/// # Why this is a separate step from [`resolve`]
///
/// [`resolve`] answers "which model", by shape, and must not probe. This
/// answers "which *lay-out* of that model", and is nothing but probing: it is
/// a cache lookup, and a cache lookup is allowed to come back empty. Keeping
/// them apart is what makes a miss free of consequence — the archive is
/// servable on its own, so the worst outcome here is that a boot pays the
/// family transforms `pie model build` exists to move offline.
///
/// # Why the artifact is matched on stated facts and not on a key
///
/// The obvious design is for a serve to recompute the cache key its build
/// wrote and look up the filename. It cannot, for two independent reasons:
///
/// 1. The key hashes the whole compiled plan, and `pie model build` compiles
///    for the *host converter's* `StorageTarget` while a serve compiles for
///    its device. Two different plans, so two different keys, always.
/// 2. The plan a serve would compile over the *archive* need not compile at
///    all — a transform outside the backend's tile-map mask is precisely the
///    work that has to be done offline. Requiring the key would make the
///    lookup fail hardest on the models the build exists for.
///
/// So the artifact says what it is for, and this asks whether that is what is
/// wanted. A fact the artifact does not state is a mismatch, never a shrug:
/// `pie model build` writes all six unconditionally, so a file missing one was
/// written by a pie that did not know it mattered.
///
/// A false miss costs one boot's transforms. A false hit serves silently wrong
/// numbers, so — like the cache key — this errs toward missing.
pub fn prefer_runtime(model: Model, request: &Request) -> Model {
    let Model::Artifact(archive) = &model else {
        return model;
    };
    let Some(runtime_dir) = runtime_dir_of(archive) else {
        return model;
    };
    // A runtime is a cache of ONE archive, and re-importing rewrites the
    // archive under the same path. Read once, here, rather than per candidate.
    let stat =
        model_loader::cache_key::snapshot_stat(archive.parent().unwrap_or_else(|| Path::new(".")));
    let Ok(entries) = std::fs::read_dir(&runtime_dir) else {
        return model;
    };
    let mut candidates: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| is_artifact_path(path))
        .collect();
    // Directory order is unspecified, and a boot that binds a different file
    // on each run for the same reason is not a cache.
    candidates.sort();
    for candidate in candidates {
        if runtime_answers(&candidate, request, &stat) {
            tracing::info!(
                runtime = %candidate.display(),
                backend = %request.backend,
                "binding a prebuilt runtime artifact"
            );
            return Model::Artifact(candidate);
        }
    }
    // SAID OUT LOUD, because the alternative to saying it is building at boot,
    // and a boot that quietly spends minutes materializing weights is a worse
    // answer than one that names the command an operator can run once. What is
    // served is correct either way -- the archive is general form -- so this is
    // an optimization that was available and not taken, which is exactly the
    // kind of thing a log line is for.
    tracing::info!(
        archive = %archive.display(),
        "no prebuilt runtime for this boot; serving the archive and doing its \
         transforms at load. `pie model build {} --backend {}` writes one",
        archive
            .parent()
            .and_then(Path::file_name)
            .map_or_else(|| archive.display().to_string(), |n| n.to_string_lossy().into_owned()),
        request.backend,
    );
    model
}

/// `<name>/runtime/` beside an archive, if this artifact is a store archive.
///
/// Named by the layout rather than by where the file came from, so an operator
/// who spells the archive as a path gets the same lookup a store name gets.
/// Everything else — a `--out` build, a legacy flat `<name>.zt`, a bare
/// checkpoint — has no `runtime/` beside it and is returned untouched.
fn runtime_dir_of(archive: &Path) -> Option<PathBuf> {
    if archive.file_name()? != "archive.zt" {
        return None;
    }
    let dir = archive.parent()?.join("runtime");
    dir.is_dir().then_some(dir)
}

/// Whether the runtime at `path` states exactly the request being made — and
/// is still a model this build knows how to serve.
///
/// Reads the file's attributes first — a header, not the weights — so a
/// candidate that answers a different request costs one small read. Only a
/// candidate that passes every stated fact is then identified, which is the
/// expensive half and the one that catches what the facts cannot.
///
/// # Why identification is checked here and not left to the driver
///
/// Measured, on `openai/gpt-oss-20b`: `pie model build --backend cuda` wrote an
/// artifact that `model::catalog::identify` refused — *"missing
/// `layer.{}.mlp.experts.gate_up_proj_bias`"* — because a built artifact's
/// tensors are post-transform and match no manifest by construction. That is
/// now fixed at the root (`catalog::identify_artifact`: a build states the row
/// this same identification settled against the archive), so the question
/// asked below is the one that can be answered.
///
/// The check stays regardless, because the *class* of failure is what a cache
/// must not propagate: an artifact naming a row this build does not serve, or
/// one the catalog has since dropped, would otherwise take a boot that worked
/// and stop it from booting at all. A cache whose hit can be worse than its
/// miss is not a cache. So the file is asked the question the driver will ask,
/// here, where the answer can still be "use the archive instead".
fn runtime_answers(path: &Path, request: &Request, source_stat: &str) -> bool {
    let Ok(attributes) = model_loader::checkpoint::zt::read_attributes(path) else {
        return false;
    };
    if !answers(&attributes, request, source_stat) {
        return false;
    }
    let Ok(metadata) = model_loader::checkpoint::read::parse_checkpoint_metadata(path) else {
        tracing::warn!(runtime = %path.display(), "a prebuilt runtime cannot be read; ignoring it");
        return false;
    };
    // The same map, in the shape the catalog reads. `read_attributes` already
    // kept only the text-valued entries, which is all provenance ever is, so
    // this re-wraps rather than re-reads.
    let stated = model_loader::checkpoint::Attributes::from_pairs(
        attributes
            .iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    model_loader::checkpoint::Attribute::Text(value.clone()),
                )
            })
            .collect::<Vec<_>>(),
    );
    // `Override::None` and not the boot's `--as`: the worker does not carry
    // one, and guessing wrong here can only cost a fallback to the archive.
    if let Err(why) =
        model::catalog::identify_artifact(&stated, &metadata, &model::catalog::Override::None)
    {
        tracing::warn!(
            runtime = %path.display(),
            "a prebuilt runtime was built for this boot but is not a model this build \
             can serve, so the archive is used instead: {why}"
        );
        return false;
    }
    true
}

/// The match itself, over an artifact's stated facts.
///
/// Split from the read so it can be tested against every way a runtime can
/// fail to be the one wanted, without a `.zt` per case. Every leg here is a
/// fact that decides which tensors exist, and a fact that is absent or
/// disagrees is a mismatch — never a shrug.
fn answers(
    attributes: &std::collections::BTreeMap<String, String>,
    request: &Request,
    source_stat: &str,
) -> bool {
    use model_loader::checkpoint::meta;

    let says = |key: &str| attributes.get(key).map(String::as_str);
    // A shard carries no provenance, so it fails here rather than needing a
    // shard rule of its own: only a checkpoint root is ever written with
    // attributes, and every one of these is required.
    says(meta::CONTRACT_KEY) == Some(meta::CONTRACT_REVISION.to_string().as_str())
        && says(meta::BACKEND_KEY) == Some(request.backend.as_str())
        && says(meta::COMPONENT_KEY) == Some(request.component.as_str())
        && says(meta::TP_SIZE_KEY) == Some(request.tp_size.to_string().as_str())
        && says(meta::SOURCE_STAT_KEY) == Some(source_stat)
        // `Auto` is what `crates/model/src/boot.rs` authors with, and it is not
        // a wildcard: a build asked for `routed` lowered the expert banks a way
        // an `Auto` boot did not ask for.
        && says(meta::MOE_KEY) == Some("auto")
        // Absent means "no runtime quantization" — that is what the key's own
        // documentation says it means, and what every unquantized build writes.
        // The serve path authors `RuntimeQuant::None`, so a quantized build
        // cannot answer it until serve learns to ask.
        && says(meta::RUNTIME_QUANT_KEY).is_none_or(|quant| quant == "none")
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_loader::checkpoint::write::CheckpointWriter;
    use model_loader::types::{DType, Encoding, TensorDecl, TensorId};

    /// The request `crates/model/src/boot.rs` makes today, and what
    /// `pie model build --backend cuda` states in answer to it.
    fn cuda_request() -> Request {
        Request {
            backend: "cuda".to_string(),
            component: "full".to_string(),
            tp_size: 1,
        }
    }

    fn built_for_cuda() -> std::collections::BTreeMap<String, String> {
        use model_loader::checkpoint::meta;
        [
            (meta::CONTRACT_KEY, meta::CONTRACT_REVISION.to_string()),
            (meta::BACKEND_KEY, "cuda".to_string()),
            (meta::COMPONENT_KEY, "full".to_string()),
            (meta::TP_SIZE_KEY, "1".to_string()),
            (meta::SOURCE_STAT_KEY, "0123456789abcdef".to_string()),
            (meta::MOE_KEY, "auto".to_string()),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
    }

    /// The control the six negative cases below are read against.
    #[test]
    fn a_build_for_this_request_answers_it() {
        assert!(answers(
            &built_for_cuda(),
            &cuda_request(),
            "0123456789abcdef"
        ));
    }

    /// The one that would be silent: same shapes, same names, different
    /// numbers. `--backend` decides projections and naming and lands entirely
    /// in the bytes, so nothing downstream would contradict a Metal build
    /// bound by a CUDA boot.
    #[test]
    fn a_build_for_another_backend_does_not_answer() {
        let mut attributes = built_for_cuda();
        attributes.insert(
            model_loader::checkpoint::meta::BACKEND_KEY.to_string(),
            "metal".to_string(),
        );
        assert!(!answers(&attributes, &cuda_request(), "0123456789abcdef"));
    }

    /// A runtime is a cache of ONE archive, and re-importing rewrites the
    /// archive in place. Measured end to end: touching `archive.zt` makes a
    /// boot that was binding a runtime fall back to the archive.
    #[test]
    fn a_build_from_an_archive_that_has_since_changed_does_not_answer() {
        assert!(!answers(
            &built_for_cuda(),
            &cuda_request(),
            "fedcba9876543210"
        ));
    }

    /// The serve path authors `RuntimeQuant::None`. An FP8 build is a finished
    /// product and cannot answer a request for unquantized weights.
    #[test]
    fn a_quantized_build_does_not_answer_an_unquantized_request() {
        let mut attributes = built_for_cuda();
        attributes.insert(
            model_loader::checkpoint::meta::RUNTIME_QUANT_KEY.to_string(),
            "fp8".to_string(),
        );
        assert!(!answers(&attributes, &cuda_request(), "0123456789abcdef"));
        // Absent and `none` are the same statement, and every unquantized
        // build makes one of them.
        attributes.insert(
            model_loader::checkpoint::meta::RUNTIME_QUANT_KEY.to_string(),
            "none".to_string(),
        );
        assert!(answers(&attributes, &cuda_request(), "0123456789abcdef"));
    }

    /// A whole model is not the slice a component boot asked for.
    #[test]
    fn a_whole_model_does_not_answer_a_request_for_a_slice() {
        let mut request = cuda_request();
        request.component = "text".to_string();
        assert!(!answers(&built_for_cuda(), &request, "0123456789abcdef"));
    }

    /// `pie model build` writes one unsharded artifact. Binding it on every
    /// rank of a tensor-parallel boot would give each rank the whole model
    /// under the names it expects its own shard at.
    #[test]
    fn an_unsharded_build_does_not_answer_a_sharded_boot() {
        let mut request = cuda_request();
        request.tp_size = 2;
        assert!(!answers(&built_for_cuda(), &request, "0123456789abcdef"));
    }

    /// Every fact is required, so a file written by a pie that did not know
    /// one of them mattered is a miss rather than a partial match. Checked one
    /// key at a time: a rule that only fires when several are missing is a
    /// rule that has already been passed by the interesting case.
    #[test]
    fn a_build_that_does_not_state_every_fact_does_not_answer() {
        for key in built_for_cuda().keys() {
            let mut attributes = built_for_cuda();
            attributes.remove(key);
            assert!(
                !answers(&attributes, &cuda_request(), "0123456789abcdef"),
                "an artifact silent about {key} was bound anyway"
            );
        }
    }

    /// A `--out` build, a legacy flat `<name>.zt` and a bare checkpoint have
    /// no `runtime/` beside them, and the lookup has to leave them alone
    /// rather than search their parent directory.
    #[test]
    fn only_a_store_archive_has_runtimes_to_choose_from() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let flat = dir.path().join("model.zt");
        std::fs::write(&flat, b"").expect("a file");
        assert_eq!(runtime_dir_of(&flat), None);

        let model_dir = dir.path().join("Qwen--Qwen3-0.6B");
        let archive = model_dir.join("archive.zt");
        std::fs::create_dir_all(&model_dir).expect("a model dir");
        std::fs::write(&archive, b"").expect("an archive");
        // No `runtime/` yet: an archive that has never been built for.
        assert_eq!(runtime_dir_of(&archive), None);
        std::fs::create_dir_all(model_dir.join("runtime")).expect("a runtime dir");
        assert_eq!(runtime_dir_of(&archive), Some(model_dir.join("runtime")));
    }

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
        let mut writer = CheckpointWriter::create(&path, &Default::default()).unwrap();
        // Ascending names: `model/…` sorts before `tokenizer/…`.
        writer
            .add_meta(model::encoding::CONFIG_OBJECT, config)
            .unwrap();
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
    /// drivers' `config.json` parsers and the runtime's own key probes — so it
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
        // THE ONE FIELD A READER STILL WANTS. Absent here, which is not a
        // defect: most checkpoints declare no quantization, and an absent
        // block is an unquantized checkpoint rather than a missing answer.
        assert!(
            model::encoding::Encoding::from_config_value(&doc).is_none(),
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
