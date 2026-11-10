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

use crate::ui::{Mark, Palette};

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
    /// Show every setting compared, every deployment's reason, and the
    /// adoption block.
    #[arg(long, short = 'v')]
    pub verbose: bool,
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
        schema: SCHEMA,
        card,
        hollow,
        kin: kin.iter().map(Family::of).collect(),
        served: None,
        adoption: None,
        refused: None,
        passed_over: Vec::new(),
        unknown: None,
        verbose: args.verbose,
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
        deployments.push(Landing::of(deployment, verdict));
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
        } else if args.verbose {
            Some(adopted.adoption.block.clone())
        } else {
            None
        },
        whole: args.show,
        source: args.source.clone(),
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
    if crate::ui::is_interactive(crate::ui::Stream::Stderr) {
        eprintln!("Fetching {source} metadata (config, tokenizer, weight headers)…");
    }
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
    checkpoint::file::json::parse(&text).with_context(|| format!("parsing {}", path.display()))
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

/// The version of `--json`'s shape: a field renamed or removed bumps it.
const SCHEMA: u32 = 1;

#[derive(Serialize)]
struct Report {
    schema: u32,
    card: Card,
    hollow: bool,
    kin: Vec<Family>,
    served: Option<String>,
    adoption: Option<Adoption>,
    refused: Option<String>,
    /// What each nearer family that refused it said.
    passed_over: Vec<String>,
    unknown: Option<Unknown>,
    #[serde(skip)]
    verbose: bool,
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
            fact("Hidden size", grouped(h));
        }
        if let Some(l) = layers {
            let mut said = grouped(l);
            if let Some(types) = text
                .get("layer_types")
                .and_then(serde_json::Value::as_array)
            {
                let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
                for t in types.iter().filter_map(serde_json::Value::as_str) {
                    *kinds.entry(t).or_default() += 1;
                }
                if kinds.len() > 1 {
                    said.push_str(&format!(
                        " ({})",
                        kinds
                            .iter()
                            .map(|(k, n)| format!("{n} {}", k.replace('_', " ")))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
            fact("Layers", said);
        }
        if let (Some(q), Some(kv)) = (q, kv) {
            let kind = if kv == q {
                "MHA".to_string()
            } else if kv == 1 {
                "MQA".to_string()
            } else {
                "GQA".to_string()
            };
            fact(
                "Attention",
                format!(
                    "{kind} · {q} query / {kv} KV heads{}",
                    head.map(|d| format!(" · head dim {d}")).unwrap_or_default()
                ),
            );
        }
        if let Some(window) = int("sliding_window")
            && text
                .get("use_sliding_window")
                .and_then(serde_json::Value::as_bool)
                != Some(false)
        {
            fact("Sliding window", format!("{} tokens", grouped(window)));
        }
        let experts = int("num_experts")
            .or_else(|| int("num_local_experts"))
            .or_else(|| int("n_routed_experts"));
        match experts {
            Some(e) if e > 1 => fact(
                "MLP",
                format!(
                    "MoE · {e} experts, top-{}{}{}",
                    int("num_experts_per_tok").map_or("?".to_string(), |k| k.to_string()),
                    int("moe_intermediate_size")
                        .map(|w| format!(" · expert dim {}", grouped(w)))
                        .unwrap_or_default(),
                    int("n_shared_experts")
                        .filter(|n| *n > 0)
                        .map(|n| format!(" · {n} shared"))
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
                    fact("MLP", format!("{kind} · {}", grouped(inter)));
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
                "Vocabulary",
                format!(
                    "{}{}",
                    grouped(vocab),
                    if tied { " (tied embeddings)" } else { "" }
                ),
            );
        }
        let context = int("max_position_embeddings");
        let theta = float("rope_theta").or_else(|| {
            text.get("rope_parameters")
                .and_then(|r| r.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
        });
        let scaling = text.get("rope_scaling").filter(|s| !s.is_null()).map(|s| {
            let kind = s
                .get("rope_type")
                .or_else(|| s.get("type"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("scaled");
            match s.get("factor").and_then(serde_json::Value::as_f64) {
                Some(f) => format!(" · {kind} ×{}", number(f)),
                None => format!(" · {kind}"),
            }
        });
        if context.is_some() || theta.is_some() {
            fact(
                "Context",
                format!(
                    "{}{}{}",
                    context
                        .map(|c| format!("{} tokens", grouped(c)))
                        .unwrap_or_else(|| "—".to_string()),
                    theta
                        .map(|t| format!(" · RoPE θ {}", number(t)))
                        .unwrap_or_default(),
                    scaling.unwrap_or_default()
                ),
            );
        }
        if let Some(eps) = float("rms_norm_eps") {
            fact("Norm", format!("RMSNorm · ε {}", number(eps)));
        }
        if let Some(q) = config.get("quantization_config") {
            let method = q
                .get("quant_method")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("quantized");
            let detail = q
                .get("quant_algo")
                .or_else(|| q.get("fmt"))
                .or_else(|| q.get("format"))
                .or_else(|| q.get("bits"))
                .map(|v| format!(" · {}", v.to_string().trim_matches('"')))
                .unwrap_or_default();
            fact("Quantization", format!("{method}{detail}"));
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
        if let Some(per) = kv_bytes_per_token {
            fact(
                "KV cache (bf16)",
                format!(
                    "{} / token{}",
                    size(per),
                    context
                        .map(|c| format!(" · {} at {} tokens", size(per * c), grouped(c)))
                        .unwrap_or_default()
                ),
            );
        }
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

    /// The stored precisions, by how many tensors hold each, most first.
    fn stored(&self) -> Vec<String> {
        let mut by: Vec<(&String, &usize)> = self.dtypes.iter().collect();
        by.sort_by(|a, b| b.1.cmp(a.1));
        by.into_iter().map(|(d, _)| label(d)).collect()
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
    source: String,
}

#[derive(Serialize)]
struct Landing {
    name: String,
    /// Each weight role's precision, as people name them (`bf16`, `fp8`, `int4`).
    weights: Vec<String>,
    kv: String,
    extras: Vec<String>,
    reads: bool,
    why: Option<String>,
}

impl Landing {
    fn of(
        deployment: &runtime::catalog::Deployment,
        verdict: std::result::Result<(), String>,
    ) -> Landing {
        let deploy = &deployment.deploy;
        let mut weights: Vec<String> = Vec::new();
        for precision in &deploy.weights {
            let said = label(&poem::star::precision_word(*precision));
            if !weights.contains(&said) {
                weights.push(said);
            }
        }
        let mut extras: Vec<String> = deploy.parts.clone();
        extras.extend(deploy.drafter.iter().map(|d| format!("{d} drafter")));
        Landing {
            name: deployment.name.clone(),
            weights,
            kv: label(&poem::star::word(deploy.kv)),
            extras,
            reads: verdict.is_ok(),
            why: verdict.err().map(|why| reason(&why)),
        }
    }

    fn precision(&self) -> String {
        format!("{} weights · {} KV", self.weights.join(" + "), self.kv)
    }
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

/// A family's refusal split into the family and what it said:
/// "`llama_3`'s `adopt` refuses it: the config's ..." reads as
/// (`llama_3`, "the config's ...").
fn refusal(said: &str) -> (String, String) {
    let said = reason(said);
    let Some(rest) = said.strip_prefix('`') else {
        return (String::new(), said);
    };
    let Some((family, why)) = rest.split_once("`'s `adopt` refuses it: ") else {
        return (String::new(), said);
    };
    (family.to_string(), why.to_string())
}

/// A precision as people name it: `bf16`, `fp8`, `nvfp4`, `int4`.
fn label(word: &str) -> String {
    let word = word.to_ascii_lowercase();
    let named = [
        ("bf16", "bf16"),
        ("f16", "fp16"),
        ("f32", "fp32"),
        ("nvfp4", "nvfp4"),
        ("mxfp4", "mxfp4"),
        ("mxfp8", "mxfp8"),
        ("e4m3", "fp8"),
        ("e5m2", "fp8"),
        ("f8", "fp8"),
        ("fp4", "fp4"),
        ("u8", "int8"),
        ("i8", "int8"),
        ("u4", "int4"),
        ("i4", "int4"),
        ("u3", "int3"),
        ("u2", "int2"),
        ("u1", "int1"),
    ];
    named
        .iter()
        .find(|(prefix, _)| word.starts_with(prefix))
        .map_or(word.clone(), |(_, label)| (*label).to_string())
}

/// `n` with its thousands grouped: `151,936`.
fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn number(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        if n < 0.0 {
            format!("-{}", grouped((-n) as u64))
        } else {
            grouped(n as u64)
        }
    } else if n.abs() < 1e-3 {
        format!("{n:e}")
    } else {
        format!("{n}")
    }
}

/// `n` bytes with a space before the unit: `7.5 GiB`.
fn size(n: u64) -> String {
    let said = crate::ui::bytes(n);
    match said.find(|c: char| c.is_ascii_alphabetic()) {
        Some(at) => format!("{} {}", &said[..at], &said[at..]),
        None => said,
    }
}

fn count(n: u64) -> String {
    match n {
        n if n >= 1_000_000_000_000 => format!("{:.2}T", n as f64 / 1e12),
        n if n >= 1_000_000_000 => format!("{:.2}B", n as f64 / 1e9),
        n if n >= 1_000_000 => format!("{:.1}M", n as f64 / 1e6),
        n => grouped(n),
    }
}

/// A backend as its users name it.
fn backend(name: &str) -> &str {
    match name {
        "cuda" => "CUDA",
        "metal" => "Metal",
        "vulkan" => "Vulkan",
        "wgpu" => "WebGPU",
        "xla" => "XLA",
        "cerebras" => "Cerebras",
        other => other,
    }
}

/// Key–value rows under a section, the keys in one column.
const KEY: usize = 18;

fn section(p: &Palette, title: &str) {
    println!();
    println!("{}", p.bold(title));
}

fn row(p: &Palette, key: &str, value: impl std::fmt::Display) {
    println!("  {}{value}", p.dim(format!("{key:<KEY$}")));
}

fn more(p: &Palette, value: impl std::fmt::Display) {
    println!("  {}{value}", " ".repeat(KEY));
    let _ = p;
}

impl crate::ui::Report for Report {
    fn render(&self, p: &Palette) {
        let ok = Mark::Did.render(p);
        let no = Mark::Blocked.render(p);
        let c = &self.card;

        // Header: what was read.
        println!("{}", p.bold(&c.origin));
        let mut said = Vec::new();
        if let Some(a) = &c.architecture {
            said.push(a.clone());
        }
        said.push(format!("{} parameters", count(c.parameters)));
        said.push(size(c.stored_bytes));
        let stored = c.stored();
        if !stored.is_empty() {
            said.push(stored.join(" + "));
        }
        println!("  {}", said.join(" · "));
        if self.hollow {
            println!("  {}", p.dim("Metadata only: no weights were downloaded."));
        }

        // MODEL: the architecture, as config.json states it.
        if !c.facts.is_empty() {
            section(p, "MODEL");
            for (key, value) in &c.facts {
                row(p, key, value);
            }
        }

        // COMPATIBILITY: which family it is, and where it departs.
        section(p, "COMPATIBILITY");
        match self.kin.first() {
            None => row(
                p,
                "Family",
                format!("{no} no installed family matches these tensor names"),
            ),
            Some(_) if self.refused.is_some() => {
                row(
                    p,
                    "Family",
                    format!("{no} no installed family runs this architecture"),
                );
                row(
                    p,
                    "Considered",
                    p.dim(format!(
                        "{} (tensor names match)",
                        self.kin
                            .iter()
                            .map(|k| k.package.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )),
                );
            }
            Some(f) => {
                row(p, "Family", format!("{} {}", ok, f.package));
                let total = f.agreeing.len() + f.departing.len();
                if total > 0 {
                    row(
                        p,
                        "Closest model",
                        format!(
                            "{}  {}",
                            f.nearest,
                            p.dim(format!("{} of {total} settings match", f.agreeing.len()))
                        ),
                    );
                }
                let width = f.departing.iter().map(|v| v.key.len()).max().unwrap_or(0);
                for (i, v) in f.departing.iter().enumerate() {
                    let change = format!(
                        "{:<width$}  {} → {}",
                        v.key,
                        number(v.built),
                        v.stated.map_or("unstated".to_string(), number)
                    );
                    if i == 0 {
                        row(p, "Differences", change);
                    } else {
                        more(p, change);
                    }
                }
                if self.verbose {
                    for v in &f.agreeing {
                        more(
                            p,
                            p.dim(format!(
                                "{:<width$}  {} (same)",
                                v.key,
                                v.stated.map_or("unstated".to_string(), number)
                            )),
                        );
                    }
                    for other in self.kin.iter().skip(1) {
                        row(
                            p,
                            "Also matches",
                            p.dim(format!("{} (by tensor names)", other.package)),
                        );
                    }
                    for why in &self.passed_over {
                        row(p, "Passed over", p.dim(why));
                    }
                }
            }
        }

        if let Some(unknown) = &self.unknown {
            if self.verbose {
                section(p, "LAYER 0 TENSORS");
                for (name, shape) in &unknown.layer {
                    row(p, name, p.dim(format!("{shape:?}")));
                }
            }
            section(p, "RESULT");
            row(p, "Status", format!("{no} Unknown architecture"));
            row(
                p,
                "Next",
                "Support needs a new model package (see models/README.md).",
            );
            if !self.verbose {
                more(p, p.dim("Run with --verbose to list its layer-0 tensors."));
            }
            return;
        }
        if let Some(served) = &self.served {
            section(p, "RESULT");
            row(
                p,
                "Status",
                format!("{ok} Already supported as {}", p.bold(served)),
            );
            row(p, "Next", format!("pie model import {}", c.origin));
            return;
        }
        if let Some(why) = &self.refused {
            section(p, "RESULT");
            row(p, "Status", format!("{no} Not supported"));
            let refusals: Vec<(String, String)> = why.split("; and ").map(refusal).collect();
            let width = refusals.iter().map(|(f, _)| f.len()).max().unwrap_or(0);
            for (i, (family, why)) in refusals.iter().enumerate() {
                let said = if family.is_empty() {
                    why.clone()
                } else {
                    format!("{}  {why}", p.dim(format!("{family:<width$}")))
                };
                if i == 0 {
                    row(p, "Why", said);
                } else {
                    more(p, said);
                }
            }
            return;
        }
        let Some(a) = &self.adoption else { return };

        // DEPLOYMENTS: what it can be served as.
        section(p, "DEPLOYMENTS");
        let readable: Vec<&Landing> = a.deployments.iter().filter(|d| d.reads).collect();
        let width = readable
            .iter()
            .map(|d| d.precision().chars().count())
            .max()
            .unwrap_or(0);
        for d in &readable {
            let extras = if d.extras.is_empty() {
                String::new()
            } else {
                format!(" · {}", d.extras.join(" · "))
            };
            println!(
                "  {ok} {:<width$}{extras}  {}",
                d.precision(),
                p.dim(&d.name)
            );
        }
        let unreadable: Vec<&Landing> = a.deployments.iter().filter(|d| !d.reads).collect();
        if !unreadable.is_empty() {
            if self.verbose {
                for d in &unreadable {
                    println!(
                        "  {} {}  {}",
                        Mark::Absent.render(p),
                        d.precision(),
                        p.dim(d.why.as_deref().unwrap_or_default())
                    );
                }
            } else {
                let mut wanted: Vec<String> = Vec::new();
                for d in &unreadable {
                    for w in &d.weights {
                        if !stored.contains(w) && !wanted.contains(w) {
                            wanted.push(w.clone());
                        }
                    }
                }
                let said = if wanted.is_empty() {
                    format!(
                        "{} more not readable from this checkpoint",
                        unreadable.len()
                    )
                } else {
                    format!(
                        "{} weights need a checkpoint stored that way (this one is {})",
                        wanted.join(" / "),
                        stored.first().map_or("?", String::as_str)
                    )
                };
                println!("  {} {}", Mark::Absent.render(p), p.dim(said));
            }
        }

        // VALIDATION: what was checked before anything is written.
        section(p, "VALIDATION");
        if readable.is_empty() {
            row(
                p,
                "Weights",
                format!("{no} no deployment can read this checkpoint"),
            );
        } else {
            row(
                p,
                "Weights",
                format!("{ok} all required tensors present, shapes match"),
            );
        }
        match &a.tokenizer {
            None => row(p, "Tokenizer", format!("{ok} compiles")),
            Some(why) => row(p, "Tokenizer", format!("{no} {}", reason(why))),
        }
        if !a.traces.is_empty() {
            let traced = a
                .traces
                .iter()
                .map(|(b, traced)| format!("{} {}", if *traced { &ok } else { &no }, backend(b)))
                .collect::<Vec<_>>()
                .join("  ");
            row(p, "Graph", traced);
            more(
                p,
                p.dim("checks the model program per backend; kernels are verified at first run"),
            );
        }
        if self.verbose {
            row(
                p,
                "Family hook",
                p.dim(format!("{}.adopt() set {}", a.family, a.rebinds.join(", "))),
            );
        }

        if let Some(text) = &a.text {
            section(
                p,
                &format!(
                    "PACKAGE  {}",
                    p.dim(format!(
                        "{}.poem{}",
                        a.package,
                        if a.whole {
                            ""
                        } else {
                            ", adoption block (--show prints the whole file)"
                        }
                    ))
                ),
            );
            println!("{text}");
        }

        // RESULT: one verdict, and what to run next.
        section(p, "RESULT");
        let next_import = format!("pie model import {}", a.source);
        if a.tokenizer.is_some() {
            row(
                p,
                "Status",
                format!("{no} Not adoptable: its tokenizer is not supported"),
            );
        } else if readable.is_empty() {
            row(
                p,
                "Status",
                format!("{no} Not adoptable: no deployment reads this checkpoint"),
            );
        } else if let Some(written) = &a.written {
            row(p, "Status", format!("{ok} Adopted as {}", p.bold(&a.id)));
            row(p, "Package", crate::ui::short_path(written));
            row(p, "Next", next_import);
        } else if a.dry_run {
            row(
                p,
                "Status",
                format!(
                    "{ok} Ready to adopt as {}  {}",
                    p.bold(&a.id),
                    p.dim("(dry run: nothing written)")
                ),
            );
            row(p, "Next", format!("pie model adopt {}", a.source));
            more(p, next_import);
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
    fn numbers_and_precisions_read_as_people_write_them() {
        assert_eq!(grouped(151_936), "151,936");
        assert_eq!(grouped(36), "36");
        assert_eq!(number(1_000_000.0), "1,000,000");
        assert_eq!(number(1e-6), "1e-6");
        assert_eq!(size(7_500 << 20), "7.3 GiB");
        assert_eq!(label("e4m3tile128a8"), "fp8");
        assert_eq!(label("u4g64"), "int4");
        assert_eq!(label("nvfp4a4"), "nvfp4");
        assert_eq!(label("bf16"), "bf16");
    }

    #[test]
    fn a_refusal_names_its_family() {
        assert_eq!(
            refusal("`llama_3`'s `adopt` refuses it: the config's model_type is phi"),
            (
                "llama_3".to_string(),
                "the config's model_type is phi".to_string()
            )
        );
        assert_eq!(refusal("plain").0, "");
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
