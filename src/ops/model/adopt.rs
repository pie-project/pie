use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use runtime::engine::load::Platform;
use serde::Serialize;

use crate::ui::{Mark, Palette};

#[derive(Args, Debug)]
pub struct AdoptArgs {
    pub source: String,
    #[arg(long, short = 'v')]
    pub verbose: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Explain { verbose: bool },
    Write,
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
    let mode = Mode::Explain {
        verbose: args.verbose,
    };
    let (report, code) = evaluate(&dir, &origin, hollow, mode)?;
    Ok(crate::ui::Answer::report(report).with_code(code))
}

pub(crate) enum Behind {
    Adopted(String),
    Refused(String),
}

pub(crate) fn behind_the_scenes(dir: &Path, source: &str) -> Result<Behind> {
    let (report, _) = evaluate(dir, source, false, Mode::Write)?;
    if let Some(a) = report.adoption.as_ref().filter(|a| a.written.is_some()) {
        let family = report.kin.iter().find(|k| k.package == a.family);
        let changed: Vec<String> = family
            .map(|f| {
                f.departing
                    .iter()
                    .map(|v| {
                        format!(
                            "{} {} → {}",
                            v.key,
                            number(v.built),
                            v.stated.map_or("unstated".to_string(), number)
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        return Ok(Behind::Adopted(format!(
            "adopted {source} as {} on the {} family{}",
            a.id,
            a.family,
            if changed.is_empty() {
                String::new()
            } else {
                format!(" ({})", changed.join(", "))
            }
        )));
    }
    let why = if report.unknown.is_some() {
        "no installed model family matches its tensor names".to_string()
    } else if !report.refused.is_empty() {
        report
            .refused
            .iter()
            .map(|r| format!("{}: {}", r.family, r.reason))
            .collect::<Vec<_>>()
            .join("; ")
    } else if let Some(why) = report.adoption.as_ref().and_then(|a| a.tokenizer.as_ref()) {
        format!("its tokenizer is not supported: {}", reason(why))
    } else {
        "no deployment of its family reads it".to_string()
    };
    Ok(Behind::Refused(why))
}

fn evaluate(
    dir: &Path,
    origin: &str,
    hollow: bool,
    mode: Mode,
) -> Result<(Report, std::process::ExitCode)> {
    let verbose = matches!(mode, Mode::Explain { verbose: true });
    let (dir, origin) = (dir.to_path_buf(), origin.to_string());
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
        refused: Vec::new(),
        passed_over: Vec::new(),
        unknown: None,
        verbose,
    };

    if kin.is_empty() {
        report.unknown = Some(Unknown::of(&metadata, &config));
        return Ok((report, std::process::ExitCode::from(2)));
    }
    if kin[0].nearest.departing.is_empty()
        && let Some((name, _)) =
            runtime::engine::load::conversion_contract(&src, &metadata, platform)
    {
        report.served = Some(name);
        return Ok((report, std::process::ExitCode::SUCCESS));
    }
    if !kin.iter().any(|k| k.adopts) {
        report.refused = vec![Refused {
            family: kin[0].package.name().to_string(),
            reason: "it lists its models by hand and states no `adopt`".to_string(),
        }];
        return Ok((report, std::process::ExitCode::from(2)));
    }

    let id = model_id(&origin);
    let name = package_name(&id);
    let target = crate::paths::models_dir().join(format!("{name}.poem"));
    let catalog_for_adoption = if target.exists() {
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
    let mut refusals = Vec::new();
    let mut chosen = None;
    for family in catalog.kin(&src, platform).into_iter().filter(|k| k.adopts) {
        match catalog.adopt(&family, &name, &id, &checkpoint, platform) {
            Ok(adopted) => {
                chosen = Some((family, adopted));
                break;
            }
            Err(refusal) => refusals.push(Refused::from(refusal)),
        }
    }
    let Some((family, adopted)) = chosen else {
        report.refused = refusals;
        return Ok((report, std::process::ExitCode::from(2)));
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
    let platforms: &[Platform] = match mode {
        Mode::Explain { .. } => &PLATFORMS,
        Mode::Write => std::slice::from_ref(&platform),
    };
    let traces = match &first {
        Some(d) => platforms
            .iter()
            .map(|p| (p.backend().to_string(), d.try_trace(*p).is_ok()))
            .collect(),
        None => Vec::new(),
    };
    let tokenizer = tokenizer_of(&dir);
    let mut written = None;
    if mode == Mode::Write && first.is_some() && tokenizer.is_ok() {
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
        text: verbose.then(|| adopted.adoption.block.clone()),
        source: origin.clone(),
    });
    let code = if first.is_some() && tokenizer.is_ok() {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::from(2)
    };
    Ok((report, code))
}

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

fn configuration(dir: &Path) -> Result<serde_json::Value> {
    if checkpoint::file::diffusers::is_pipeline(dir) {
        let mut map = serde_json::Map::new();
        map.insert(
            "pipeline".to_string(),
            json_file(&dir.join(checkpoint::file::diffusers::PIPELINE_INDEX))?,
        );
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

const SCHEMA: u32 = 1;

#[derive(Serialize)]
struct Report {
    schema: u32,
    card: Card,
    hollow: bool,
    kin: Vec<Family>,
    served: Option<String>,
    adoption: Option<Adoption>,
    refused: Vec<Refused>,
    passed_over: Vec<Refused>,
    unknown: Option<Unknown>,
    #[serde(skip)]
    verbose: bool,
}

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

    fn stored(&self) -> Vec<String> {
        let mut by: Vec<(&String, &usize)> = self.dtypes.iter().collect();
        by.sort_by(|a, b| b.1.cmp(a.1));
        by.into_iter().map(|(d, _)| label(d)).collect()
    }
}

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
    tokenizer: Option<String>,
    written: Option<PathBuf>,
    text: Option<String>,
    source: String,
}

#[derive(Serialize)]
struct Landing {
    name: String,
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

fn layer_suffix(name: &str) -> Option<(u64, String)> {
    let parts: Vec<&str> = name.split('.').collect();
    let at = parts.windows(2).position(|w| {
        matches!(w[0], "layers" | "h" | "blocks" | "layer") && w[1].parse::<u64>().is_ok()
    })?;
    let index = parts[at + 1].parse().ok()?;
    Some((index, parts[at + 2..].join(".")))
}

#[derive(Serialize)]
struct Refused {
    family: String,
    reason: String,
}

impl From<runtime::catalog::Refusal> for Refused {
    fn from(refusal: runtime::catalog::Refusal) -> Refused {
        Refused {
            family: refusal.family,
            reason: refusal.reason,
        }
    }
}

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

const BACKENDS: [(&str, &str); 6] = [
    ("cuda", "CUDA"),
    ("metal", "Metal"),
    ("vulkan", "Vulkan"),
    ("wgpu", "WebGPU"),
    ("xla", "XLA"),
    ("cerebras", "Cerebras"),
];

fn backend(name: &str) -> &str {
    BACKENDS
        .iter()
        .find(|(word, _)| *word == name)
        .map_or(name, |(_, said)| said)
}

struct Sheet {
    title: String,
    about: Vec<Vec<Span>>,
    sections: Vec<Section>,
}

struct Section {
    title: &'static str,
    aside: Option<String>,
    lines: Vec<Line>,
}

enum Line {
    Field(String, Vec<Span>),
    Item(Mark, Vec<Vec<Span>>),
    Verbatim(String),
}

#[derive(Clone, Copy)]
enum Style {
    Plain,
    Dim,
    Bold,
    Mark(Mark),
}

struct Span(Style, String);

fn plain(text: impl Into<String>) -> Span {
    Span(Style::Plain, text.into())
}

fn dim(text: impl Into<String>) -> Span {
    Span(Style::Dim, text.into())
}

fn bold(text: impl Into<String>) -> Span {
    Span(Style::Bold, text.into())
}

fn mark(mark: Mark) -> Span {
    Span(Style::Mark(mark), mark.glyph().to_string())
}

fn status(m: Mark, text: impl Into<String>) -> Vec<Span> {
    vec![mark(m), plain(" "), plain(text)]
}

impl Section {
    fn new(title: &'static str) -> Section {
        Section {
            title,
            aside: None,
            lines: Vec::new(),
        }
    }

    fn field(&mut self, key: &str, value: Vec<Span>) {
        self.lines.push(Line::Field(key.to_string(), value));
    }

    fn fields(&mut self, key: &str, values: Vec<Vec<Span>>) {
        for (i, value) in values.into_iter().enumerate() {
            self.field(if i == 0 { key } else { "" }, value);
        }
    }

    fn item(&mut self, m: Mark, columns: Vec<Vec<Span>>) {
        self.lines.push(Line::Item(m, columns));
    }
}

const KEY: usize = 18;
const GAP: &str = "  ";

impl Span {
    fn width(&self) -> usize {
        self.1.chars().count()
    }

    fn paint(&self, p: &Palette) -> String {
        match self.0 {
            Style::Plain => self.1.clone(),
            Style::Dim => p.dim(&self.1).to_string(),
            Style::Bold => p.bold(&self.1).to_string(),
            Style::Mark(m) => m.render(p),
        }
    }
}

fn paint(spans: &[Span], p: &Palette) -> String {
    spans.iter().map(|s| s.paint(p)).collect()
}

fn width(spans: &[Span]) -> usize {
    spans.iter().map(Span::width).sum()
}

impl Sheet {
    fn print(&self, out: &mut dyn std::io::Write, p: &Palette) -> std::io::Result<()> {
        writeln!(out, "{}", p.bold(&self.title))?;
        for line in &self.about {
            writeln!(out, "  {}", paint(line, p))?;
        }
        for section in &self.sections {
            writeln!(out)?;
            match &section.aside {
                Some(aside) => writeln!(out, "{}  {}", p.bold(section.title), p.dim(aside))?,
                None => writeln!(out, "{}", p.bold(section.title))?,
            }
            let mut widths: Vec<usize> = Vec::new();
            for line in &section.lines {
                if let Line::Item(_, columns) = line {
                    let aligned = columns.len().saturating_sub(1);
                    for (i, cell) in columns.iter().take(aligned).enumerate() {
                        if widths.len() <= i {
                            widths.push(0);
                        }
                        widths[i] = widths[i].max(width(cell));
                    }
                }
            }
            for line in &section.lines {
                match line {
                    Line::Field(key, value) => {
                        let key = format!("{key:<KEY$}");
                        writeln!(out, "  {}{}", p.dim(key), paint(value, p))?;
                    }
                    Line::Item(m, columns) => {
                        let mut said = format!("  {} ", m.render(p));
                        for (i, cell) in columns.iter().enumerate() {
                            said.push_str(&paint(cell, p));
                            if i + 1 < columns.len() {
                                said.push_str(&" ".repeat(widths[i] - width(cell)));
                                said.push_str(GAP);
                            }
                        }
                        writeln!(out, "{}", said.trim_end())?;
                    }
                    Line::Verbatim(text) => writeln!(out, "{text}")?,
                }
            }
        }
        Ok(())
    }
}

impl Report {
    fn sheet(&self) -> Sheet {
        let c = &self.card;
        let stored = c.stored();
        let mut summary: Vec<String> = c.architecture.iter().cloned().collect();
        summary.push(format!("{} parameters", count(c.parameters)));
        summary.push(size(c.stored_bytes));
        if !stored.is_empty() {
            summary.push(stored.join(" + "));
        }
        let mut about = vec![vec![plain(summary.join(" · "))]];
        if self.hollow {
            about.push(vec![dim("Metadata only: no weights were downloaded.")]);
        }
        let mut sections = Vec::new();
        if !c.facts.is_empty() {
            let mut model = Section::new("MODEL");
            for (key, value) in &c.facts {
                model.field(key, vec![plain(value)]);
            }
            sections.push(model);
        }
        sections.push(self.compatibility());
        if let Some(unknown) = &self.unknown {
            if self.verbose {
                let mut layer = Section::new("LAYER 0 TENSORS");
                for (name, shape) in &unknown.layer {
                    layer.item(
                        Mark::Plain,
                        vec![vec![plain(name)], vec![dim(format!("{shape:?}"))]],
                    );
                }
                sections.push(layer);
            }
        } else if let Some(a) = self.adoption.as_ref().filter(|_| self.refused.is_empty()) {
            sections.push(self.deployments(a, &stored));
            sections.push(self.validation(a));
            if let Some(text) = &a.text {
                let mut package = Section::new("PACKAGE");
                package.aside = Some(format!("{}.poem, its adoption block", a.package));
                package.lines.push(Line::Verbatim(text.clone()));
                sections.push(package);
            }
        }
        sections.push(self.result());
        Sheet {
            title: c.origin.clone(),
            about,
            sections,
        }
    }

    fn compatibility(&self) -> Section {
        let mut s = Section::new("COMPATIBILITY");
        let Some(f) = self.kin.first() else {
            s.field(
                "Family",
                status(
                    Mark::Blocked,
                    "no installed family matches these tensor names",
                ),
            );
            return s;
        };
        if !self.refused.is_empty() {
            s.field(
                "Family",
                status(Mark::Blocked, "no installed family runs this architecture"),
            );
            let names: Vec<&str> = self.kin.iter().map(|k| k.package.as_str()).collect();
            s.field(
                "Considered",
                vec![dim(format!("{} (tensor names match)", names.join(", ")))],
            );
            return s;
        }
        s.field("Family", status(Mark::Did, &f.package));
        let total = f.agreeing.len() + f.departing.len();
        if total > 0 {
            s.field(
                "Closest model",
                vec![
                    plain(&f.nearest),
                    plain(GAP),
                    dim(format!("{} of {total} settings match", f.agreeing.len())),
                ],
            );
        }
        let width = f.departing.iter().map(|v| v.key.len()).max().unwrap_or(0);
        let said = |v: &Value| v.stated.map_or("unstated".to_string(), number);
        let mut differences: Vec<Vec<Span>> = f
            .departing
            .iter()
            .map(|v| {
                vec![plain(format!(
                    "{:<width$}{GAP}{} → {}",
                    v.key,
                    number(v.built),
                    said(v)
                ))]
            })
            .collect();
        if self.verbose {
            differences.extend(
                f.agreeing
                    .iter()
                    .map(|v| vec![dim(format!("{:<width$}{GAP}{} (same)", v.key, said(v)))]),
            );
        }
        s.fields("Differences", differences);
        if self.verbose {
            s.fields(
                "Also matches",
                self.kin
                    .iter()
                    .skip(1)
                    .map(|k| vec![dim(format!("{} (by tensor names)", k.package))])
                    .collect(),
            );
            s.fields(
                "Passed over",
                self.passed_over
                    .iter()
                    .map(|r| vec![dim(format!("{}: {}", r.family, r.reason))])
                    .collect(),
            );
        }
        s
    }

    fn deployments(&self, a: &Adoption, stored: &[String]) -> Section {
        let mut s = Section::new("DEPLOYMENTS");
        for d in a.deployments.iter().filter(|d| d.reads) {
            let mut said = d.precision();
            for extra in &d.extras {
                said.push_str(&format!(" · {extra}"));
            }
            s.item(Mark::Did, vec![vec![plain(said)], vec![dim(&d.name)]]);
        }
        let unreadable: Vec<&Landing> = a.deployments.iter().filter(|d| !d.reads).collect();
        if self.verbose {
            for d in &unreadable {
                let why = d.why.clone().unwrap_or_default();
                s.item(
                    Mark::Absent,
                    vec![vec![plain(d.precision())], vec![dim(why)]],
                );
            }
        } else if !unreadable.is_empty() {
            let mut wanted: Vec<&str> = Vec::new();
            for w in unreadable.iter().flat_map(|d| &d.weights) {
                if !stored.contains(w) && !wanted.contains(&w.as_str()) {
                    wanted.push(w);
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
            s.item(Mark::Absent, vec![vec![dim(said)]]);
        }
        s
    }

    fn validation(&self, a: &Adoption) -> Section {
        let mut s = Section::new("VALIDATION");
        s.field(
            "Weights",
            if a.deployments.iter().any(|d| d.reads) {
                status(Mark::Did, "all required tensors present, shapes match")
            } else {
                status(Mark::Blocked, "no deployment can read this checkpoint")
            },
        );
        s.field(
            "Tokenizer",
            match &a.tokenizer {
                None => status(Mark::Did, "compiles"),
                Some(why) => status(Mark::Blocked, reason(why)),
            },
        );
        if !a.traces.is_empty() {
            let mut graph = Vec::new();
            for (i, (name, traced)) in a.traces.iter().enumerate() {
                if i > 0 {
                    graph.push(plain(GAP));
                }
                graph.extend(status(
                    if *traced { Mark::Did } else { Mark::Blocked },
                    backend(name),
                ));
            }
            s.fields(
                "Graph",
                vec![
                    graph,
                    vec![dim(
                        "checks the model program per backend; kernels are verified at first run",
                    )],
                ],
            );
        }
        if self.verbose {
            s.field(
                "Family hook",
                vec![dim(format!(
                    "{}.adopt() set {}",
                    a.family,
                    a.rebinds.join(", ")
                ))],
            );
        }
        s
    }

    fn result(&self) -> Section {
        let mut s = Section::new("RESULT");
        let origin = &self.card.origin;
        if self.unknown.is_some() {
            s.field("Status", status(Mark::Blocked, "Unknown architecture"));
            let mut next = vec![vec![plain(
                "Support needs a new model package (see models/README.md).",
            )]];
            if !self.verbose {
                next.push(vec![dim("Run with --verbose to list its layer-0 tensors.")]);
            }
            s.fields("Next", next);
        } else if let Some(served) = &self.served {
            s.field(
                "Status",
                vec![
                    mark(Mark::Did),
                    plain(" Already supported as "),
                    bold(served),
                ],
            );
            s.field("Next", vec![plain(format!("pie model import {origin}"))]);
        } else if !self.refused.is_empty() {
            s.field("Status", status(Mark::Blocked, "Not supported"));
            let width = self
                .refused
                .iter()
                .map(|r| r.family.len())
                .max()
                .unwrap_or(0);
            s.fields(
                "Why",
                self.refused
                    .iter()
                    .map(|r| vec![dim(format!("{:<width$}{GAP}", r.family)), plain(&r.reason)])
                    .collect(),
            );
        } else if let Some(a) = &self.adoption {
            let import = vec![plain(format!("pie model import {}", a.source))];
            if a.tokenizer.is_some() {
                s.field(
                    "Status",
                    status(
                        Mark::Blocked,
                        "Not adoptable: its tokenizer is not supported",
                    ),
                );
            } else if !a.deployments.iter().any(|d| d.reads) {
                s.field(
                    "Status",
                    status(
                        Mark::Blocked,
                        "Not adoptable: no deployment reads this checkpoint",
                    ),
                );
            } else if let Some(written) = &a.written {
                s.field(
                    "Status",
                    vec![mark(Mark::Did), plain(" Adopted as "), bold(&a.id)],
                );
                s.field("Package", vec![plain(crate::ui::short_path(written))]);
                s.field("Next", import);
            } else {
                s.field(
                    "Status",
                    vec![
                        mark(Mark::Did),
                        plain(" Supported: import adopts it as "),
                        bold(&a.id),
                    ],
                );
                s.field("Next", import);
            }
        }
        s
    }
}

impl crate::ui::Report for Report {
    fn render(&self, p: &Palette) {
        let _ = self.sheet().print(&mut std::io::stdout().lock(), p);
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
    fn a_sheet_aligns_its_columns_and_keys() {
        let mut deployments = Section::new("DEPLOYMENTS");
        deployments.item(Mark::Did, vec![vec![plain("bf16 weights")], vec![dim("a")]]);
        deployments.item(Mark::Did, vec![vec![plain("int4")], vec![dim("b")]]);
        let mut result = Section::new("RESULT");
        result.fields("Next", vec![vec![plain("one")], vec![plain("two")]]);
        let sheet = Sheet {
            title: "org/model".into(),
            about: vec![vec![plain("about")]],
            sections: vec![deployments, result],
        };
        let mut out = Vec::new();
        sheet.print(&mut out, &Palette::forced(false)).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            [
                "org/model",
                "  about",
                "",
                "DEPLOYMENTS",
                "  ✓ bf16 weights  a",
                "  ✓ int4          b",
                "",
                "RESULT",
                "  Next              one",
                "                    two",
                "",
            ]
            .join("\n")
        );
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
