//! What this build says of a checkpoint or an adapter before a weight is
//! downloaded, and the adapter files an import does not fetch.
//!
//! | Command | Answers |
//! | --- | --- |
//! | `pie model families` | Every architecture a package's models are, with the package |
//! | `pie model check <source>` | Whether an import takes it, as which deployment, and whether this Mac holds it |
//! | `pie model fetch <repo> --out <dir>` | An adapter's files at a pinned commit, hash-checked |
//!
//! A check of a Hugging Face repository reads only what it states of itself
//! (`local::hf::stated_download`) and runs the matcher an import runs
//! (`conversion_contract`) and the planner a boot admits by
//! (`runtime::engine::load::need`). Answers are kept under
//! `$PIE_HOME/cache/check` by repository, commit and context.

use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

use crate::ui::{Answer, Palette};

/// The page size Pie's Metal engine lays kv in when `[engine] kv_page_size` is unstated.
const PAGE_SIZE: u32 = 32;

/// The checks `--many` runs at once.
const AT_ONCE: usize = 8;

/// Numbers the folders checks read into, so two checks of one repository do not share one.
static READS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[derive(clap::Args, Debug)]
pub struct CheckArgs {
    /// A Hugging Face repository (`owner/name`) or a folder on this machine.
    #[arg(required_unless_present = "many")]
    pub source: Option<String>,
    /// The commit to check, by its full sha.
    #[arg(long, value_name = "COMMIT")]
    pub revision: Option<String>,
    /// The tokens one chat holds, which the memory need is planned at.
    #[arg(long, default_value_t = 32768)]
    pub context: u32,
    /// Read sources from standard input, one a line as `<source> [<commit>]`,
    /// and print one JSON line for each as its check ends.
    #[arg(long, conflicts_with = "source")]
    pub many: bool,
}

#[derive(clap::Args, Debug)]
pub struct FetchArgs {
    /// The adapter's Hugging Face repository (`owner/name`).
    pub repo: String,
    /// The commit to fetch, by its full sha.
    #[arg(long, value_name = "COMMIT")]
    pub revision: String,
    /// The folder its files are linked into.
    #[arg(long)]
    pub out: PathBuf,
    /// Exit when standard input closes.
    #[arg(long)]
    pub parent_stdin: bool,
}

/// What a source is.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Kind {
    /// A whole model.
    Model,
    /// Weights that ride on a base model.
    Adapter,
}

/// What this build does with a source.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Verdict {
    /// An import takes it as stored.
    Ready,
    /// An import takes it only by quantizing it again, for the deployment named.
    Convert,
    /// Nothing here loads it; `reason` says why.
    Unsupported,
}

/// How a model's need compares with this Mac.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Fits {
    /// It needs at most half of this Mac's memory.
    Fits,
    /// It is within what Pie may take, but needs more than half of this Mac's memory.
    Most,
    /// It needs more than Pie may take here, so a boot refuses it.
    TooLarge,
}

/// An adapter's facts, as its own files state them.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Adapter {
    /// The kind of adapter: `lora` and the rest, or a Stanza program's name.
    #[serde(rename = "type")]
    pub kind: String,
    /// What it attaches to.
    pub targets: Vec<String>,
    /// Its rank, where it has one.
    pub rank: Option<u64>,
}

/// One check's answer.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Checked {
    /// The source as it was given.
    pub repo: String,
    /// The commit checked, for a repository.
    pub revision: Option<String>,
    /// A model or an adapter.
    pub kind: Kind,
    /// What this build does with it.
    pub verdict: Verdict,
    /// Why it is unsupported.
    pub reason: Option<String>,
    /// The package whose model it is.
    pub package: Option<String>,
    /// The deployment an import builds.
    pub deployment: Option<String>,
    /// The deployment's weight dtype, as its name spells it.
    pub quant: Option<String>,
    /// The bytes of its weight files.
    pub weights_bytes: Option<u64>,
    /// The bytes an import downloads.
    pub download_bytes: Option<u64>,
    /// What a boot of it at `--context` needs of this Mac.
    pub need_bytes: Option<u64>,
    /// How that need compares with this Mac.
    pub fits: Option<Fits>,
    /// The model an adapter rides on.
    pub base: Option<String>,
    /// An adapter's facts.
    pub adapter: Option<Adapter>,
}

/// A source `--many` could not check, and why.
#[derive(Serialize)]
struct Failed<'a> {
    /// The source as it was given.
    repo: &'a str,
    /// Why it could not be checked.
    error: String,
}

/// Every architecture this build's packages hold models of.
#[derive(Serialize)]
pub struct Families {
    /// One entry per architecture and package.
    families: Vec<Family>,
}

/// The models of one architecture in one package.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Family {
    /// The architecture, as a checkpoint's `model_type` states it.
    model_type: String,
    /// The package that holds them.
    package: String,
    /// The package's models of it, miniatures left out.
    models: Vec<String>,
    /// Its models make images rather than text.
    generative: bool,
}

impl crate::ui::Report for Families {
    fn render(&self, _: &Palette) {
        for family in &self.families {
            println!(
                "{:<20} {:<16} {}",
                family.model_type,
                family.package,
                family.models.join(", ")
            );
        }
    }
}

impl crate::ui::Report for Checked {
    fn render(&self, _: &Palette) {
        let verdict = match self.verdict {
            Verdict::Ready => "ready",
            Verdict::Convert => "convert",
            Verdict::Unsupported => "unsupported",
        };
        println!("{}: {verdict}", self.repo);
        for (label, value) in [
            ("deployment", self.deployment.clone()),
            ("base", self.base.clone()),
            ("needs", self.need_bytes.map(crate::ui::bytes)),
            ("reason", self.reason.clone()),
        ] {
            if let Some(value) = value {
                println!("  {label:<10} {value}");
            }
        }
    }
}

pub fn families() -> Result<Answer> {
    let catalog = runtime::catalog::catalog();
    let mut families: Vec<Family> = Vec::new();
    for (package, model) in catalog.models() {
        if model.mini || model.arch.is_empty() {
            continue;
        }
        let generative = catalog
            .deployments()
            .filter(|deployment| deployment.model.id == model.id)
            .any(|deployment| deployment.try_generative().ok().flatten().is_some());
        match families
            .iter_mut()
            .find(|f| f.model_type == model.arch && f.package == package.name())
        {
            Some(family) => family.models.push(model.id.clone()),
            None => families.push(Family {
                model_type: model.arch.clone(),
                package: package.name().to_string(),
                models: vec![model.id.clone()],
                generative,
            }),
        }
    }
    Ok(Answer::report(Families { families }))
}

pub fn run(args: CheckArgs) -> Result<Answer> {
    if !args.many {
        let source = args.source.as_deref().unwrap_or_default();
        return Ok(Answer::report(check(
            source,
            args.revision.as_deref(),
            args.context,
        )?));
    }
    let lines: Vec<String> = std::io::stdin()
        .lock()
        .lines()
        .collect::<std::io::Result<_>>()
        .context("reading the sources on standard input")?;
    let next = std::sync::Mutex::new(lines.iter().filter(|line| !line.trim().is_empty()));
    std::thread::scope(|scope| {
        for _ in 0..AT_ONCE {
            scope.spawn(|| {
                loop {
                    let Some(line) = next.lock().expect("sources poisoned").next() else {
                        return;
                    };
                    let mut words = line.split_whitespace();
                    let source = words.next().unwrap_or_default();
                    let line = match check(source, words.next(), args.context) {
                        Ok(checked) => serde_json::to_string(&checked),
                        Err(why) => serde_json::to_string(&Failed {
                            repo: source,
                            error: format!("{why:#}"),
                        }),
                    };
                    if let Ok(line) = line {
                        let mut out = std::io::stdout().lock();
                        let _ = std::io::Write::write_all(&mut out, format!("{line}\n").as_bytes());
                    }
                }
            });
        }
    });
    Ok(Answer::quiet())
}

/// `source` at `revision`, from the cache when it was checked at this context before.
fn check(source: &str, revision: Option<&str>, context: u32) -> Result<Checked> {
    let path = Path::new(source);
    if path.is_dir() {
        if revision.is_some() {
            bail!("--revision pins a commit on the hub, and {source} is a folder on this machine");
        }
        return Ok(of_folder(source, path, context));
    }
    super::parse_repo_id(source)?;
    if let Some(revision) = revision
        && (revision.len() != 40
            || !revision
                .bytes()
                .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
    {
        bail!("--revision {revision}: name the commit by its full sha");
    }
    let repo_dir = crate::paths::pie_home()
        .join("cache")
        .join("check")
        .join(source.replace('/', "--"));
    let kept = |sha: &str| repo_dir.join(format!("{sha}-{context}.json"));
    if let Some(revision) = revision
        && let Ok(text) = std::fs::read_to_string(kept(revision))
        && let Ok(checked) = serde_json::from_str::<Checked>(&text)
    {
        return Ok(checked);
    }
    let read = READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = repo_dir.join(format!("reading-{}-{read}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let stated = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(crate::local::hf::stated_download(source, revision, &dir));
    let checked = match stated {
        Ok(stated) => {
            let mut checked = of_folder(source, &dir, context);
            checked.revision = Some(stated.sha.clone());
            checked.download_bytes = Some(match checked.kind {
                Kind::Model => stated.download_bytes,
                Kind::Adapter => stated.adapter_bytes,
            });
            if checked.kind == Kind::Model {
                checked.weights_bytes = Some(stated.weights_bytes);
                if stated.weights_bytes == 0 {
                    checked = unsupported(
                        checked,
                        "it has no safetensors weights an import reads (`model*.safetensors`)",
                    );
                }
            }
            Ok(checked)
        }
        Err(why) => match why.downcast_ref::<crate::local::hf::Gated>() {
            Some(gated) => Ok(unsupported(blank(source, Kind::Model), &gated.to_string())),
            None => Err(why),
        },
    };
    let _ = std::fs::remove_dir_all(&dir);
    let checked = checked?;
    if let Some(sha) = &checked.revision
        && let Ok(text) = serde_json::to_string(&checked)
    {
        let _ = std::fs::write(kept(sha), text);
    }
    Ok(checked)
}

/// A check of the folder `dir`, which holds what `source` states of itself.
fn of_folder(source: &str, dir: &Path, context: u32) -> Checked {
    if dir.join("stanza.json").is_file() {
        return stanza(source, dir);
    }
    if dir.join("adapter_config.json").is_file() {
        return peft(source, dir);
    }
    model(source, dir, context)
}

fn blank(source: &str, kind: Kind) -> Checked {
    Checked {
        repo: source.to_string(),
        revision: None,
        kind,
        verdict: Verdict::Unsupported,
        reason: None,
        package: None,
        deployment: None,
        quant: None,
        weights_bytes: None,
        download_bytes: None,
        need_bytes: None,
        fits: None,
        base: None,
        adapter: None,
    }
}

fn unsupported(mut checked: Checked, reason: &str) -> Checked {
    checked.verdict = Verdict::Unsupported;
    checked.reason = Some(reason.to_string());
    checked.deployment = None;
    checked.need_bytes = None;
    checked.fits = None;
    checked
}

fn model(source: &str, dir: &Path, context: u32) -> Checked {
    let checked = blank(source, Kind::Model);
    let config: serde_json::Value = match std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    {
        Some(config) => config,
        None => return unsupported(checked, "it states no `config.json`"),
    };
    let Some(platform) = runtime::engine::load::this_box() else {
        return unsupported(checked, "this pie binary carries no engine");
    };
    let matched = runtime::engine::load::open_source(dir).and_then(|source| {
        let metadata = runtime::engine::load::checkpoint_metadata(dir)?;
        Ok(matched(&source, &metadata, platform))
    });
    let (deployment, verdict) = match matched {
        Ok(Some(matched)) => matched,
        Ok(None) => return unsupported(checked, &no_deployment(&config)),
        Err(why) => return unsupported(checked, &format!("{why:#}")),
    };
    let catalog = runtime::catalog::catalog();
    let package = catalog
        .parse(&deployment)
        .map(|d| d.package.name().to_string());
    let quant = deployment
        .split("-kv-")
        .next()
        .and_then(|weights| weights.rsplit('-').next())
        .map(str::to_string);
    let mut checked = Checked {
        verdict,
        package,
        quant,
        ..checked
    };
    match need(&deployment, context) {
        Ok(Some((bytes, fits))) => {
            checked.need_bytes = Some(bytes);
            checked.fits = Some(fits);
        }
        Ok(None) => {}
        Err(why) => checked.reason = Some(format!("{why:#}")),
    }
    checked.deployment = Some(deployment);
    checked
}

/// The deployment an import builds from `source`, in the order
/// `conversion_contract` tries them: the first that reads it as stored, else
/// the first that reads it only by quantizing it again, which an import
/// builds when it is named.
fn matched(
    source: &ztensor::Source,
    metadata: &checkpoint::file::Metadata,
    platform: runtime::engine::load::Platform,
) -> Option<(String, Verdict)> {
    let mut convert = None;
    for deployment in runtime::catalog::catalog().candidates_of(source, platform) {
        let Ok((name, contract)) = runtime::engine::load::conversion_contract_named(
            source,
            metadata,
            platform,
            &deployment.name,
        ) else {
            continue;
        };
        if runtime::catalog::requantizes(&contract).is_none() {
            return Some((name, Verdict::Ready));
        }
        convert.get_or_insert(name);
    }
    convert.map(|name| (name, Verdict::Convert))
}

/// Why no deployment reads a checkpoint whose configuration is `config`.
fn no_deployment(config: &serde_json::Value) -> String {
    let model_type = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let known = runtime::catalog::catalog()
        .models()
        .any(|(_, model)| !model.arch.is_empty() && model.arch == model_type);
    if !known {
        return format!("no package in this build holds a `{model_type}` architecture");
    }
    let quantized = config
        .get("quantization_config")
        .or_else(|| config.get("quantization"));
    match quantized {
        Some(quantization) => format!(
            "no deployment of the `{model_type}` models reads its quantization ({})",
            quantization_words(quantization)
        ),
        None => format!(
            "no deployment of the `{model_type}` models reads its weights as stored \
             (another size or layout than this build's models)"
        ),
    }
}

/// A checkpoint's quantization in a few words: its method, bits and group size.
fn quantization_words(quantization: &serde_json::Value) -> String {
    let words: Vec<String> = ["quant_method", "bits", "group_size"]
        .iter()
        .filter_map(|key| {
            let value = quantization.get(*key)?;
            Some(match value {
                serde_json::Value::String(text) => text.clone(),
                other => format!("{key} {other}"),
            })
        })
        .collect();
    if words.is_empty() {
        "stated in its configuration".to_string()
    } else {
        words.join(", ")
    }
}

/// What a boot of `deployment` at `context` tokens needs of this Mac, and how that fits it.
#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn need(deployment: &str, context: u32) -> Result<Option<(u64, Fits)>> {
    let (need, memory) = runtime::engine::load::need(deployment, context, PAGE_SIZE)?;
    let bytes = need.bytes();
    let fits = if bytes > need.ceiling {
        Fits::TooLarge
    } else if bytes > memory / 2 {
        Fits::Most
    } else {
        Fits::Fits
    };
    Ok(Some((bytes, fits)))
}

/// Planned on Apple Silicon only.
#[cfg(not(all(feature = "metal", target_vendor = "apple")))]
fn need(_deployment: &str, _context: u32) -> Result<Option<(u64, Fits)>> {
    let _ = PAGE_SIZE;
    Ok(None)
}

/// What a Stanza bundle (`stanza.json` beside `adapter.safetensors`) states.
#[derive(Deserialize)]
struct Stanza {
    /// The model it was trained on, by its id in a package.
    model: String,
    /// That model's package.
    package: String,
    /// The Stanza program that trained it.
    stanza: String,
    /// The banks it holds.
    banks: Vec<Bank>,
}

/// One bank of a Stanza bundle.
#[derive(Deserialize)]
struct Bank {
    /// Its name, as the base's program would register it.
    name: String,
    /// Its shape.
    shape: Vec<u64>,
}

/// A Stanza bundle: supported when every bank it holds is a bank its base's
/// program registers.
fn stanza(source: &str, dir: &Path) -> Checked {
    let checked = blank(source, Kind::Adapter);
    let stated: Stanza = match std::fs::read(dir.join("stanza.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    {
        Some(stated) => stated,
        None => return unsupported(checked, "its `stanza.json` cannot be read"),
    };
    let mut targets: Vec<String> = Vec::new();
    for bank in &stated.banks {
        let target = bank
            .name
            .split('.')
            .skip_while(|word| *word == "layers" || word.bytes().all(|b| b.is_ascii_digit()))
            .collect::<Vec<_>>();
        let target = target[..target.len().saturating_sub(1)].join(".");
        if !targets.contains(&target) {
            targets.push(target);
        }
    }
    let checked = Checked {
        base: Some(stated.model.clone()),
        package: Some(stated.package.clone()),
        adapter: Some(Adapter {
            kind: stated.stanza.clone(),
            targets,
            rank: stated
                .banks
                .iter()
                .find(|bank| bank.name.ends_with(".lora_a"))
                .and_then(|bank| bank.shape.first().copied()),
        }),
        ..checked
    };
    let catalog = runtime::catalog::catalog();
    let Some(base) = catalog
        .deployments()
        .find(|deployment| deployment.model.id == stated.model)
    else {
        return unsupported(
            checked,
            &format!("this build ships no `{}` for it to ride on", stated.model),
        );
    };
    let trace = match base.try_trace(runtime::engine::load::Platform::Metal) {
        Ok(trace) => trace,
        Err(why) => return unsupported(checked, &why.to_string()),
    };
    let registered: Vec<&str> = trace
        .params
        .iter()
        .filter(|param| param.source == poem::ParamSource::Registered)
        .map(|param| param.name.as_str())
        .collect();
    match stated
        .banks
        .iter()
        .find(|bank| !registered.contains(&bank.name.as_str()))
    {
        Some(bank) => unsupported(
            checked,
            &format!(
                "`{}`'s program has no attachment point for its bank `{}`",
                stated.model, bank.name
            ),
        ),
        None => Checked {
            verdict: Verdict::Ready,
            ..checked
        },
    }
}

/// What a PEFT adapter's `adapter_config.json` states.
#[derive(Deserialize)]
struct Peft {
    /// The model it was trained on.
    base_model_name_or_path: Option<String>,
    /// `LORA` and the rest.
    peft_type: Option<String>,
    /// The modules it attaches to.
    #[serde(default)]
    target_modules: serde_json::Value,
    /// Its rank.
    r: Option<u64>,
}

/// A PEFT adapter, which this build has no reader for.
fn peft(source: &str, dir: &Path) -> Checked {
    let checked = blank(source, Kind::Adapter);
    let stated: Peft = match std::fs::read(dir.join("adapter_config.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    {
        Some(stated) => stated,
        None => return unsupported(checked, "its `adapter_config.json` cannot be read"),
    };
    let targets = match stated.target_modules {
        serde_json::Value::Array(items) => items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .collect(),
        serde_json::Value::String(pattern) => vec![pattern],
        _ => Vec::new(),
    };
    let kind = stated
        .peft_type
        .unwrap_or_else(|| "lora".to_string())
        .to_ascii_lowercase();
    let checked = Checked {
        base: stated.base_model_name_or_path,
        adapter: Some(Adapter {
            kind: kind.clone(),
            targets,
            rank: stated.r,
        }),
        ..checked
    };
    unsupported(
        checked,
        &format!(
            "this build reads no PEFT {kind} adapter: no program here names its target modules \
             as attachment points"
        ),
    )
}

pub fn fetch(args: FetchArgs, json: bool) -> Result<Answer> {
    if args.parent_stdin {
        std::thread::spawn(|| {
            let _ = std::io::copy(&mut std::io::stdin().lock(), &mut std::io::sink());
            eprintln!("fetch: standard input closed; stopping");
            std::process::exit(1);
        });
    }
    let allow = crate::local::hf::ADAPTER_FILES.map(str::to_string);
    let snapshot = super::fetch_snapshot(&args.repo, Some(&args.revision), &allow, json)?;
    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;
    for entry in
        std::fs::read_dir(&snapshot).with_context(|| format!("reading {}", snapshot.display()))?
    {
        let entry = entry?;
        let from = std::fs::canonicalize(entry.path())?;
        if !from.is_file() {
            continue;
        }
        let to = args.out.join(entry.file_name());
        let _ = std::fs::remove_file(&to);
        std::fs::hard_link(&from, &to)
            .or_else(|_| std::fs::copy(&from, &to).map(drop))
            .map_err(|why| anyhow!("placing {}: {why}", to.display()))?;
    }
    Ok(Answer::did(format!(
        "fetched {}@{} into {}",
        args.repo,
        args.revision,
        crate::ui::short_path(&args.out)
    )))
}
