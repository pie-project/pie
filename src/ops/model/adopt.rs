//! `pie model adopt`: what a checkpoint is, what this build knows of it, and
//! the package a family that recognizes it writes so that it serves.
//!
//! A Hugging Face id is read hollow: its configuration and tokenizer whole,
//! each weight file as its header and a hole, so a 70B is read in seconds and
//! nothing is downloaded that an import would not need anyway. The family's
//! formats recognize the checkpoint by its tensor names; its `adopt` reads
//! the numbers from `config.json`; the package it returns is checked by
//! reading the checkpoint through it, as an import would, before anything is
//! written to the models directory.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use runtime::engine::load::Platform;
use serde::Serialize;

use crate::ui::{Align, Mark, Palette, Row, Table};

#[derive(Args, Debug)]
pub struct AdoptArgs {
    /// A Hugging Face model (`owner/name`) or a checkpoint directory.
    pub source: String,
    /// The model id to adopt it as; by default its repository's name, lowercased.
    #[arg(long, value_name = "ID")]
    pub id: Option<String>,
    /// Say what would be adopted and print the package; write nothing.
    #[arg(long)]
    pub dry_run: bool,
    /// Print the package written.
    #[arg(long)]
    pub show: bool,
    /// Replace an earlier adoption of the same name.
    #[arg(long)]
    pub force: bool,
}

const PLATFORMS: [Platform; 6] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Vulkan,
    Platform::Wgpu,
    Platform::Xla,
    Platform::Cerebras,
];

pub fn run(args: AdoptArgs) -> Result<crate::ui::Answer> {
    let (dir, origin, hollow) = resolve(&args.source)?;
    let config = configuration(&dir)?;
    let src = runtime::engine::load::open_source(&dir)?;
    let metadata = runtime::engine::load::checkpoint_metadata(&dir)?;
    let card = Card::of(&origin, &card_view(&config), &metadata);
    let platform = runtime::engine::load::this_box().unwrap_or(Platform::Cuda);

    let catalog = runtime::catalog::catalog();
    let kin = catalog.kin(&src, platform);
    let mut report = Report {
        card,
        hollow,
        kin: kin.iter().map(Family::of).collect(),
        served: None,
        adoption: None,
        refused: None,
        passed_over: Vec::new(),
        unknown: None,
    };

    if kin.is_empty() {
        report.unknown = Some(Unknown::of(&metadata, &config));
        return Ok(crate::ui::Answer::report(report).with_code(std::process::ExitCode::from(2)));
    }
    if kin[0].nearest.departing.is_empty()
        && let Some((name, _)) =
            runtime::engine::load::conversion_contract(&src, &metadata, platform)
    {
        report.served = Some(name);
        return Ok(crate::ui::Answer::report(report));
    }
    if !kin.iter().any(|k| k.adopts) {
        report.refused = Some(format!(
            "`{}` recognizes it but lists its models by hand: it states no `adopt` to read \
             another's numbers from config.json",
            kin[0].package.name()
        ));
        return Ok(crate::ui::Answer::report(report).with_code(std::process::ExitCode::from(2)));
    }

    let id = match &args.id {
        Some(id) => id.clone(),
        None => model_id(&origin),
    };
    let name = package_name(&id);
    let target = crate::paths::models_dir().join(format!("{name}.poem"));
    let catalog_for_adoption = if target.exists() && args.force {
        // The earlier adoption is the one replaced: read the catalog without it.
        let mut tree = runtime::catalog::Tree::read(&crate::paths::models_dir())
            .map_err(|why| anyhow!("{why}"))?;
        tree.packages.retain(|(n, _)| *n != name);
        Some(runtime::catalog::Catalog::of_tree(&tree).0)
    } else {
        None
    };
    let catalog = catalog_for_adoption.as_ref().unwrap_or(catalog);
    let special = special_tokens(&dir);
    let checkpoint = runtime::catalog::Checkpoint {
        src: &src,
        config: &config,
        special: &special,
        origin: &origin,
    };
    // Each family that adopts, nearest first, until one does: a family
    // recognizing the names but refusing the numbers hands it to the next.
    let mut refusals = Vec::new();
    let mut chosen = None;
    for family in catalog.kin(&src, platform).into_iter().filter(|k| k.adopts) {
        match catalog.adopt(&family, &name, &id, &checkpoint, platform) {
            Ok(adopted) => {
                chosen = Some((family, adopted));
                break;
            }
            Err(why) => refusals.push(why.to_string()),
        }
    }
    let Some((family, adopted)) = chosen else {
        report.refused = Some(refusals.join("; and "));
        return Ok(crate::ui::Answer::report(report).with_code(std::process::ExitCode::from(2)));
    };
    report.passed_over = refusals;

    let mut deployments = Vec::new();
    let mut first: Option<runtime::catalog::Deployment> = None;
    for (deployment, read) in &adopted.deployments {
        let verdict = match read {
            Ok(contract) => runtime::engine::load::lands(&metadata, contract),
            Err(why) => Err(why.clone()),
        };
        if verdict.is_ok() && first.is_none() {
            first = Some(deployment.clone());
        }
        deployments.push(Landing {
            name: deployment.name.clone(),
            reads: verdict.is_ok(),
            why: verdict.err().map(|why| reason(&why)),
        });
    }
    let traces = match &first {
        Some(d) => PLATFORMS
            .iter()
            .map(|p| (p.backend().to_string(), d.try_trace(*p).is_ok()))
            .collect(),
        None => Vec::new(),
    };
    let package_text = adopted
        .adoption
        .files
        .iter()
        .map(|(_, text)| text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let tokenizer = tokenizer_of(&dir);
    let mut written = None;
    if first.is_some() && tokenizer.is_ok() && !args.dry_run {
        if target.exists() && !args.force {
            bail!(
                "{} holds an earlier adoption; pass --force to replace it, or --id to adopt it \
                 under another name",
                crate::ui::short_path(&target)
            );
        }
        write_package(&crate::paths::models_dir(), &adopted.adoption.files)?;
        written = Some(target.clone());
    }
    report.adoption = Some(Adoption {
        family: family.package.name().to_string(),
        id,
        package: name,
        rebinds: adopted.adoption.rebinds.clone(),
        deployments,
        traces,
        tokenizer: tokenizer.clone().err(),
        written,
        dry_run: args.dry_run,
        text: if args.show {
            Some(package_text)
        } else if args.dry_run {
            Some(adopted.adoption.block.clone())
        } else {
            None
        },
        whole: args.show,
        next: format!("pie model import {}", args.source),
    });
    let code = if first.is_some() && tokenizer.is_ok() {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::from(2)
    };
    Ok(crate::ui::Answer::report(report).with_code(code))
}

/// The checkpoint `source` names: a directory as it is, or a Hugging Face
/// model's snapshot, its weights' headers alone where none is downloaded.
fn resolve(source: &str) -> Result<(PathBuf, String, bool)> {
    let path = Path::new(source);
    if path.is_dir() {
        let origin = path
            .canonicalize()
            .map(|p| crate::ui::short_path(&p))
            .unwrap_or_else(|_| source.to_string());
        return Ok((path.to_path_buf(), origin, false));
    }
    if path.exists() {
        bail!(
            "{source}: a checkpoint is a directory holding config.json (or a pipeline's \
             model_index.json) beside its weights"
        );
    }
    let (owner, name) = source
        .split_once('/')
        .filter(|(o, n)| !o.is_empty() && !n.is_empty() && !n.contains('/'))
        .ok_or_else(|| {
            anyhow!("{source} is neither a directory nor a Hugging Face `owner/name`")
        })?;
    let repo_dir = crate::local::hf::resolve_cache_dir().join(format!("models--{owner}--{name}"));
    if let Some(snapshot) = std::fs::read_dir(repo_dir.join("snapshots"))
        .ok()
        .and_then(|entries| {
            entries.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| {
                p.join("config.json").is_file() || checkpoint::file::diffusers::is_pipeline(p)
            })
        })
        && (checkpoint::file::read::discover_safetensors_files(&snapshot).is_ok()
            || checkpoint::file::diffusers::components(&snapshot)
                .is_ok_and(|components| !components.is_empty()))
    {
        return Ok((snapshot, source.to_string(), false));
    }
    let root = crate::paths::pie_home()
        .join("adopt")
        .join(format!("{owner}--{name}"));
    println!("Reading {source}'s configuration and weight headers (no weights)");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let dir = runtime.block_on(crate::local::hf::hollow_snapshot(source, &root))?;
    Ok((dir, source.to_string(), true))
}

/// A model id of `origin`'s name: lowercase letters, digits, `.` and `-`.
fn model_id(origin: &str) -> String {
    let name = origin.rsplit('/').next().unwrap_or(origin);
    let mut id = String::new();
    for c in name.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() || c == '.' {
            id.push(c);
        } else if !id.ends_with('-') {
            id.push('-');
        }
    }
    id.trim_matches('-').to_string()
}

/// The package an adoption of `id` is written as.
fn package_name(id: &str) -> String {
    let mut name = "adopted_".to_string();
    for c in id.chars() {
        if c.is_ascii_alphanumeric() {
            name.push(c.to_ascii_lowercase());
        } else if !name.ends_with('_') {
            name.push('_');
        }
    }
    name.trim_end_matches('_').to_string()
}

fn write_package(models: &Path, files: &[(String, String)]) -> Result<()> {
    std::fs::create_dir_all(models)?;
    for (file, text) in files {
        let target = models.join(file);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let fresh = target.with_extension("poem.adopting");
        std::fs::write(&fresh, text)?;
        std::fs::rename(&fresh, &target)?;
    }
    Ok(())
}

fn json_file(path: &Path) -> Result<serde_json::Value> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&finite(&text)).with_context(|| format!("parsing {}", path.display()))
}

/// `text` with the `Infinity`, `-Infinity` and `NaN` Python's json writes
/// (and JSON does not) spelled as JSON can read them: the largest finite
/// double either way, and null.
fn finite(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if in_string {
            in_string = !(c == '"' && !escaped);
            escaped = c == '\\' && !escaped;
        } else if c == '"' {
            in_string = true;
        } else {
            let spelled = [
                ("-Infinity", "-1.7976931348623157e308"),
                ("Infinity", "1.7976931348623157e308"),
                ("NaN", "null"),
            ]
            .into_iter()
            .find(|(word, _)| rest.starts_with(word));
            if let Some((word, json)) = spelled {
                out.push_str(json);
                rest = &rest[word.len()..];
                continue;
            }
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

/// What a family's `adopt` is handed as the checkpoint's configuration: its
/// `config.json`, or for a diffusers pipeline each component's under the
/// role its tensors are prefixed by (`dit`, `te`, `vae`, …), beside the
/// pipeline's own index as `pipeline` and its scheduler's as `scheduler`.
fn configuration(dir: &Path) -> Result<serde_json::Value> {
    if checkpoint::file::diffusers::is_pipeline(dir) {
        let mut map = serde_json::Map::new();
        map.insert(
            "pipeline".to_string(),
            json_file(&dir.join(checkpoint::file::diffusers::PIPELINE_INDEX))?,
        );
        // The schedule is the scheduler's, which holds no weights: a
        // distilled release and its base can differ in nothing else.
        let scheduler = dir.join("scheduler").join("scheduler_config.json");
        if scheduler.is_file() {
            map.insert("scheduler".to_string(), json_file(&scheduler)?);
        }
        for component in checkpoint::file::diffusers::components(dir)? {
            if let Some(config) = &component.config {
                map.insert(
                    component.prefix.trim_end_matches('.').to_string(),
                    json_file(config)?,
                );
            }
        }
        return Ok(serde_json::Value::Object(map));
    }
    let path = dir.join("config.json");
    if !path.is_file() {
        bail!("{} states no config.json", crate::ui::short_path(dir));
    }
    json_file(&path)
}

/// The configuration a model card reads: a pipeline's denoiser's, named by
/// the pipeline's class.
fn card_view(config: &serde_json::Value) -> serde_json::Value {
    let Some(dit) = config.get("pipeline").and(config.get("dit")) else {
        return config.clone();
    };
    let mut view = dit.clone();
    if let (Some(view), Some(class)) = (
        view.as_object_mut(),
        config["pipeline"].get("_class_name").cloned(),
    ) {
        view.insert(
            "architectures".to_string(),
            serde_json::Value::Array(vec![class]),
        );
        view.insert("model_type".to_string(), "diffusers pipeline".into());
    }
    view
}

/// The special tokens `tokenizer_config.json` names, as text: what a family
/// adopting a checkpoint with a vocabulary of its own stops at. A pipeline
/// keeps its tokenizer in `tokenizer/`.
fn special_tokens(dir: &Path) -> serde_json::Value {
    let config: serde_json::Value = ["tokenizer_config.json", "tokenizer/tokenizer_config.json"]
        .iter()
        .find_map(|file| std::fs::read_to_string(dir.join(file)).ok())
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(serde_json::Value::Null);
    let mut special = serde_json::Map::new();
    for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
        let text = match config.get(key) {
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(serde_json::Value::Object(o)) => o
                .get("content")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            _ => None,
        };
        if let Some(text) = text {
            special.insert(key.to_string(), serde_json::Value::String(text));
        }
    }
    serde_json::Value::Object(special)
}

/// Whether the checkpoint's `tokenizer.json` compiles to one of the pipelines
/// pie runs, as an import would compile it; a checkpoint without one (a GGUF
/// carries its own) is not held against it.
fn tokenizer_of(dir: &Path) -> std::result::Result<(), String> {
    let path = dir.join("tokenizer.json");
    if !path.is_file() {
        return Ok(());
    }
    tokenizer::Tokenizer::from_file(&path)
        .map_err(|why| format!("{why:#}"))
        .and_then(|t| {
            t.to_canonical()
                .map(|_| ())
                .map_err(|why| format!("{why:#}"))
        })
}

/// What a refusal says, past the evaluator's traceback: the line a `fail`
/// or an error spoke, else its first.
fn reason(text: &str) -> String {
    let said = text
        .lines()
        .find_map(|line| line.trim().strip_prefix("error: fail: "))
        .or_else(|| {
            text.lines()
                .find_map(|line| line.trim().strip_prefix("error: "))
        })
        .unwrap_or_else(|| text.lines().next().unwrap_or(text));
    said.trim().to_string()
}

#[derive(Serialize)]
struct Report {
    card: Card,
    hollow: bool,
    kin: Vec<Family>,
    served: Option<String>,
    adoption: Option<Adoption>,
    refused: Option<String>,
    /// What each nearer family that refused it said.
    passed_over: Vec<String>,
    unknown: Option<Unknown>,
}

/// What a checkpoint states of itself, as a person would ask it.
#[derive(Serialize)]
struct Card {
    origin: String,
    architecture: Option<String>,
    model_type: Option<String>,
    parameters: u64,
    stored_bytes: u64,
    tensors: usize,
    files: usize,
    dtypes: BTreeMap<String, usize>,
    facts: Vec<(String, String)>,
    kv_bytes_per_token: Option<u64>,
    context: Option<u64>,
}

impl Card {
    fn of(origin: &str, config: &serde_json::Value, metadata: &checkpoint::file::Metadata) -> Card {
        // A multimodal checkpoint states its trunk under `text_config`.
        let text = config.get("text_config").unwrap_or(config);
        let int = |key: &str| text.get(key).and_then(serde_json::Value::as_u64);
        let float = |key: &str| text.get(key).and_then(serde_json::Value::as_f64);
        let mut facts: Vec<(String, String)> = Vec::new();
        let mut fact = |what: &str, value: String| facts.push((what.to_string(), value));

        let hidden = int("hidden_size");
        let layers = int("num_hidden_layers");
        let q = int("num_attention_heads");
        let kv = int("num_key_value_heads").or(q);
        let head = int("head_dim").or_else(|| Some(hidden? / q?));
        if let Some(h) = hidden {
            fact("width", format!("{h}"));
        }
        if let Some(l) = layers {
            fact("depth", format!("{l} layers"));
        }
        if let (Some(q), Some(kv)) = (q, kv) {
            let grouping = if kv == q {
                "multi-head".to_string()
            } else if kv == 1 {
                "multi-query".to_string()
            } else {
                format!("grouped {}:1", q / kv.max(1))
            };
            fact(
                "attention",
                format!(
                    "{q} query heads, {kv} key/value heads ({grouping}){}",
                    head.map(|d| format!(", {d} wide")).unwrap_or_default()
                ),
            );
        }
        if let Some(window) = int("sliding_window")
            && text
                .get("use_sliding_window")
                .and_then(serde_json::Value::as_bool)
                != Some(false)
        {
            fact("window", format!("sliding, {window} tokens"));
        }
        if let Some(types) = text
            .get("layer_types")
            .and_then(serde_json::Value::as_array)
        {
            let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
            for t in types.iter().filter_map(serde_json::Value::as_str) {
                *kinds.entry(t).or_default() += 1;
            }
            if kinds.len() > 1 {
                fact(
                    "layer kinds",
                    kinds
                        .iter()
                        .map(|(k, n)| format!("{n} {k}"))
                        .collect::<Vec<_>>()
                        .join(", "),
                );
            }
        }
        let experts = int("num_experts")
            .or_else(|| int("num_local_experts"))
            .or_else(|| int("n_routed_experts"));
        match experts {
            Some(e) if e > 1 => fact(
                "mlp",
                format!(
                    "{e} experts, {} routed a token{}{}",
                    int("num_experts_per_tok").map_or("?".to_string(), |k| k.to_string()),
                    int("moe_intermediate_size")
                        .map(|w| format!(", {w} wide"))
                        .unwrap_or_default(),
                    int("n_shared_experts")
                        .filter(|n| *n > 0)
                        .map(|n| format!(", {n} shared"))
                        .unwrap_or_default()
                ),
            ),
            _ => {
                if let Some(inter) = int("intermediate_size") {
                    let act = text
                        .get("hidden_act")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("?");
                    let kind = if act == "silu" {
                        "SwiGLU".to_string()
                    } else {
                        format!("gated {act}")
                    };
                    fact("mlp", format!("{kind}, {inter} wide"));
                }
            }
        }
        if let Some(vocab) = int("vocab_size")
            .or_else(|| config.get("vocab_size").and_then(serde_json::Value::as_u64))
        {
            let tied = config
                .get("tie_word_embeddings")
                .or_else(|| text.get("tie_word_embeddings"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            fact(
                "vocabulary",
                format!(
                    "{vocab} tokens{}",
                    if tied {
                        ", head tied to the embedding"
                    } else {
                        ", its own head"
                    }
                ),
            );
        }
        let context = int("max_position_embeddings");
        let theta = float("rope_theta").or_else(|| {
            text.get("rope_parameters")
                .and_then(|r| r.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
        });
        let scaling = text
            .get("rope_scaling")
            .filter(|s| !s.is_null())
            .map(|s| {
                let kind = s
                    .get("rope_type")
                    .or_else(|| s.get("type"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("scaled");
                match s.get("factor").and_then(serde_json::Value::as_f64) {
                    Some(f) => format!("{kind} ×{f}"),
                    None => kind.to_string(),
                }
            })
            .unwrap_or_else(|| "unscaled".to_string());
        fact(
            "position",
            format!(
                "rope{}, {scaling}{}",
                theta.map(|t| format!(" base {t}")).unwrap_or_default(),
                context
                    .map(|c| format!("; {c} positions"))
                    .unwrap_or_default()
            ),
        );
        if let Some(eps) = float("rms_norm_eps") {
            fact("norm", format!("RMS, ε {eps:e}"));
        }
        if let Some(q) = config.get("quantization_config") {
            let method = q
                .get("quant_method")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("quantized");
            let detail = q
                .get("fmt")
                .or_else(|| q.get("format"))
                .or_else(|| q.get("bits"))
                .map(|v| format!(" {}", v.to_string().trim_matches('"')))
                .unwrap_or_default();
            fact("quantization", format!("{method}{detail}"));
        }

        let mut dtypes: BTreeMap<String, usize> = BTreeMap::new();
        let mut parameters = 0u64;
        for tensor in &metadata.tensors {
            *dtypes
                .entry(format!("{:?}", tensor.encoding.dtype()).to_lowercase())
                .or_default() += 1;
            parameters += tensor
                .shape
                .iter()
                .map(|d| *d.max(&0) as u64)
                .product::<u64>();
        }
        // Keys and values each layer stores a token, as bf16: a layer of
        // full attention holds every token, one of another kind not.
        let full_layers = text
            .get("layer_types")
            .and_then(serde_json::Value::as_array)
            .map(|types| {
                types
                    .iter()
                    .filter(|t| t.as_str() == Some("full_attention"))
                    .count() as u64
            })
            .or(layers);
        let kv_bytes_per_token = match (full_layers, kv, head) {
            (Some(l), Some(kv), Some(d)) => Some(2 * l * kv * d * 2),
            _ => None,
        };
        Card {
            origin: origin.to_string(),
            architecture: config
                .get("architectures")
                .and_then(|a| a.get(0))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            model_type: config
                .get("model_type")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            parameters,
            stored_bytes: metadata.tensors.iter().map(|t| t.span_bytes).sum(),
            tensors: metadata.tensors.len(),
            files: metadata.files.len(),
            dtypes,
            facts,
            kv_bytes_per_token,
            context,
        }
    }
}

/// A family whose formats recognize the checkpoint.
#[derive(Serialize)]
struct Family {
    package: String,
    format: String,
    nearest: String,
    adopts: bool,
    agreeing: Vec<Value>,
    departing: Vec<Value>,
}

#[derive(Serialize)]
struct Value {
    key: String,
    built: f64,
    stated: Option<f64>,
    at_least: bool,
}

impl Family {
    fn of(kin: &runtime::catalog::Kin) -> Family {
        let value = |a: &runtime::catalog::Agreement| Value {
            key: a.key.clone(),
            built: a.built,
            stated: a.stated,
            at_least: a.at_least,
        };
        Family {
            package: kin.package.name().to_string(),
            format: kin.format.clone(),
            nearest: kin.nearest.id.clone(),
            adopts: kin.adopts,
            agreeing: kin.nearest.agreeing.iter().map(value).collect(),
            departing: kin.nearest.departing.iter().map(value).collect(),
        }
    }
}

#[derive(Serialize)]
struct Adoption {
    family: String,
    id: String,
    package: String,
    rebinds: Vec<String>,
    deployments: Vec<Landing>,
    traces: Vec<(String, bool)>,
    /// Why the checkpoint's tokenizer does not compile, where it does not.
    tokenizer: Option<String>,
    written: Option<PathBuf>,
    dry_run: bool,
    text: Option<String>,
    whole: bool,
    next: String,
}

#[derive(Serialize)]
struct Landing {
    name: String,
    reads: bool,
    why: Option<String>,
}

/// A checkpoint no family recognizes: what a layer of it holds, for whoever
/// writes its package.
#[derive(Serialize)]
struct Unknown {
    layer: Vec<(String, Vec<i64>)>,
    outside: Vec<(String, Vec<i64>)>,
}

impl Unknown {
    fn of(metadata: &checkpoint::file::Metadata, _config: &serde_json::Value) -> Unknown {
        let mut layer = Vec::new();
        let mut outside = Vec::new();
        for t in &metadata.tensors {
            match layer_suffix(&t.name) {
                Some((0, suffix)) => layer.push((suffix, t.shape.clone())),
                Some(_) => {}
                None => outside.push((t.name.clone(), t.shape.clone())),
            }
        }
        layer.sort();
        outside.sort();
        Unknown { layer, outside }
    }
}

/// `name`'s layer and what follows its index, where it names a layer's tensor.
fn layer_suffix(name: &str) -> Option<(u64, String)> {
    let parts: Vec<&str> = name.split('.').collect();
    let at = parts.windows(2).position(|w| {
        matches!(w[0], "layers" | "h" | "blocks" | "layer") && w[1].parse::<u64>().is_ok()
    })?;
    let index = parts[at + 1].parse().ok()?;
    Some((index, parts[at + 2..].join(".")))
}

fn number(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else if n.abs() < 1e-3 {
        format!("{n:e}")
    } else {
        format!("{n}")
    }
}

fn count(n: u64) -> String {
    match n {
        n if n >= 1_000_000_000 => format!("{:.2}B", n as f64 / 1e9),
        n if n >= 1_000_000 => format!("{:.1}M", n as f64 / 1e6),
        n => n.to_string(),
    }
}

impl crate::ui::Report for Report {
    fn render(&self, p: &Palette) {
        let c = &self.card;
        let what = match (&c.architecture, &c.model_type) {
            (Some(a), Some(t)) => format!("{a} ({t})"),
            (Some(a), None) => a.clone(),
            (None, Some(t)) => t.clone(),
            (None, None) => "no architecture stated".to_string(),
        };
        println!("{}  {}  {}", p.bold(&c.origin), p.dim("·"), what);
        println!(
            "  {} parameters as stored, {} in {} tensors over {} file{}{}",
            count(c.parameters),
            crate::ui::bytes(c.stored_bytes),
            c.tensors,
            c.files,
            if c.files == 1 { "" } else { "s" },
            if self.hollow {
                p.dim(" (headers read; no weights fetched)").to_string()
            } else {
                String::new()
            }
        );

        println!();
        println!("{}", p.bold("What it is"));
        let mut t = Table::new([Align::Left, Align::Left], 2);
        for (what, value) in &c.facts {
            t.push(Row::new(Mark::Plain, [what.clone(), value.clone()]));
        }
        t.push(Row::new(
            Mark::Plain,
            [
                "stored as".to_string(),
                c.dtypes
                    .iter()
                    .map(|(d, n)| format!("{d} ×{n}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            ],
        ));
        if let Some(per) = c.kv_bytes_per_token {
            let at = c.context.unwrap_or(32768);
            t.push(Row::new(
                Mark::Plain,
                [
                    "kv cache".to_string(),
                    format!(
                        "{} a token in bf16; {} for {at} tokens",
                        crate::ui::bytes(per),
                        crate::ui::bytes(per * at)
                    ),
                ],
            ));
        }
        t.print(p);

        println!();
        println!("{}", p.bold("What this build knows of it"));
        if self.kin.is_empty() {
            println!(
                "  {} no family's formats recognize these tensor names",
                Mark::Blocked.render(p)
            );
        }
        let others: Vec<&str> = self
            .kin
            .iter()
            .skip(1)
            .map(|f| f.package.as_str())
            .collect();
        for f in self.kin.iter().take(1) {
            println!(
                "  {} {} recognizes its tensor names (its `{}` format){}",
                Mark::Did.render(p),
                p.bold(&f.package),
                f.format,
                if f.adopts {
                    ""
                } else {
                    ", but lists its models by hand"
                }
            );
            println!(
                "    nearest listed model: {} ({} of {} numbers differ)",
                f.nearest,
                f.departing.len(),
                f.departing.len() + f.agreeing.len()
            );
            let mut t = Table::new([Align::Left, Align::Left, Align::Right, Align::Right], 9);
            t.push(Row::new(
                Mark::Plain,
                [
                    "".to_string(),
                    p.dim("config.json").to_string(),
                    p.dim(&f.nearest).to_string(),
                    p.dim("this checkpoint").to_string(),
                ],
            ));
            for v in &f.departing {
                t.push(Row::new(
                    Mark::Warn,
                    [
                        "".to_string(),
                        v.key.clone(),
                        number(v.built),
                        v.stated.map(number).unwrap_or_default(),
                    ],
                ));
            }
            for v in &f.agreeing {
                t.push(Row::new(
                    Mark::Absent,
                    [
                        "".to_string(),
                        p.dim(&v.key).to_string(),
                        p.dim(number(v.built)).to_string(),
                        p.dim(
                            v.stated
                                .map(number)
                                .unwrap_or_else(|| "unstated".to_string()),
                        )
                        .to_string(),
                    ],
                ));
            }
            t.print(p);
        }
        if !others.is_empty() {
            println!(
                "  {}",
                p.dim(format!(
                    "also recognized by name, further off: {}",
                    others.join(", ")
                ))
            );
        }

        if let Some(served) = &self.served {
            println!();
            println!(
                "{} Served as it is, as {}: {}",
                Mark::Did.render(p),
                p.bold(served),
                p.dim(format!("pie model import {}", c.origin))
            );
            return;
        }
        if let Some(unknown) = &self.unknown {
            println!();
            println!("{}", p.bold("What a package for it would read"));
            println!("  layer 0 holds:");
            let mut t = Table::new([Align::Left, Align::Left], 1);
            for (name, shape) in &unknown.layer {
                t.push(Row::new(Mark::Plain, [name.clone(), format!("{shape:?}")]));
            }
            t.print(p);
            if !unknown.outside.is_empty() {
                println!("  outside its layers:");
                let mut t = Table::new([Align::Left, Align::Left], 1);
                for (name, shape) in unknown.outside.iter().take(12) {
                    t.push(Row::new(Mark::Plain, [name.clone(), format!("{shape:?}")]));
                }
                t.print(p);
            }
            println!(
                "  {}",
                p.dim("A new architecture is a new package under models/ (see models/README.md).")
            );
            return;
        }
        if let Some(why) = &self.refused {
            println!();
            println!("{} Not adopted: {why}", Mark::Blocked.render(p));
            return;
        }
        let Some(a) = &self.adoption else { return };
        println!();
        println!("{}", p.bold("How Pie adjusted to it"));
        for why in &self.passed_over {
            println!("  {} passed over: {why}", Mark::Absent.render(p));
        }
        println!(
            "  {} {}'s adopt() read config.json as the model {}, rebinding {}",
            Mark::Did.render(p),
            a.family,
            p.bold(&a.id),
            a.rebinds.join(", ")
        );
        let mut t = Table::new([Align::Left, Align::Left], 1);
        for d in &a.deployments {
            let (mark, said) = if d.reads {
                (
                    Mark::Did,
                    "reads every tensor it needs and lands".to_string(),
                )
            } else {
                (Mark::Absent, d.why.clone().unwrap_or_default())
            };
            t.push(Row::new(mark, [d.name.clone(), said]));
        }
        t.print(p);
        if !a.traces.is_empty() {
            println!(
                "  its program traces for: {}",
                a.traces
                    .iter()
                    .map(|(b, ok)| format!(
                        "{b} {}",
                        if *ok {
                            Mark::Did.render(p)
                        } else {
                            Mark::Blocked.render(p)
                        }
                    ))
                    .collect::<Vec<_>>()
                    .join("  ")
            );
            println!(
                "  {}",
                p.dim(
                    "(a trace is the model's program; each engine's kernels are its own, and \
                     an op one lacks is refused at its first forward)"
                )
            );
        }
        match &a.tokenizer {
            None => println!("  {} its tokenizer compiles", Mark::Did.render(p)),
            Some(why) => println!(
                "  {} its tokenizer does not compile: {}",
                Mark::Blocked.render(p),
                reason(why)
            ),
        }
        if let Some(text) = &a.text {
            println!();
            println!(
                "{}",
                p.dim(format!(
                    "---- {}.poem{} ----",
                    a.package,
                    if a.whole {
                        ""
                    } else {
                        ", as it ends (--show prints it whole)"
                    }
                ))
            );
            println!("{text}");
        }
        println!();
        if a.tokenizer.is_some() {
            println!(
                "{} Its tokenizer is outside the pipelines pie compiles, so an import would \
                 refuse it; nothing written",
                Mark::Blocked.render(p)
            );
        } else if !a.deployments.iter().any(|d| d.reads) {
            println!(
                "{} No deployment of the adoption reads it; nothing written",
                Mark::Blocked.render(p)
            );
        } else if let Some(written) = &a.written {
            println!(
                "{} Wrote {}; this build now serves {}",
                Mark::Did.render(p),
                crate::ui::short_path(written),
                p.bold(&a.id)
            );
            println!("  next: {}", p.bold(&a.next));
        } else if a.dry_run {
            println!("{} Dry run: nothing written", Mark::Chosen.render(p));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_adoption_is_named_by_its_repository() {
        assert_eq!(model_id("Qwen/Qwen3-4B"), "qwen3-4b");
        assert_eq!(
            model_id("meta-llama/Llama-3.2-1B-Instruct"),
            "llama-3.2-1b-instruct"
        );
        assert_eq!(
            package_name("llama-3.2-1b-instruct"),
            "adopted_llama_3_2_1b_instruct"
        );
        assert_eq!(package_name("qwen3-4b"), "adopted_qwen3_4b");
    }

    #[test]
    fn a_configuration_python_wrote_reads() {
        let read: serde_json::Value = serde_json::from_str(&finite(
            r#"{"a": [0.0, Infinity, -Infinity, NaN], "b": "NaN \"Infinity"}"#,
        ))
        .unwrap();
        assert_eq!(read["a"][1], serde_json::json!(f64::MAX));
        assert_eq!(read["a"][2], serde_json::json!(-f64::MAX));
        assert!(read["a"][3].is_null());
        assert_eq!(read["b"], "NaN \"Infinity");
    }

    #[test]
    fn a_layer_tensor_is_named_past_its_index() {
        assert_eq!(
            layer_suffix("model.layers.0.self_attn.q_proj.weight"),
            Some((0, "self_attn.q_proj.weight".to_string()))
        );
        assert_eq!(layer_suffix("model.embed_tokens.weight"), None);
    }
}
