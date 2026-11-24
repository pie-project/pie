//! `pie model trace`: the run directory a trainer replays. It evaluates a
//! model package as the server does and writes what that records
//! (`trace.json`), the adapter banks the trace registers (`banks.json`) and
//! the files it read (`mount.json`). Given the checkpoint, it also writes
//! which checkpoint tensors fill each frozen weight (`import.json`); with
//! `--lowered`, how the trace lowers for each named platform and whether
//! every bank comes through it (`lowered.json`).

mod banks;
mod import;
mod lowered;
mod mount;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use poem::star::Deploy;
use poem::{Dtype, Param, ParamSource, Platform, Precision};
use runtime::catalog::{Catalog, Deployment, Tree, one_rank};
use serde_json::Value;

use crate::ui::Answer;

/// The flags of `pie model trace`.
#[derive(Args, Debug)]
pub struct TraceArgs {
    /// The model id, as its package lists it (`qwen3-0.6b`).
    pub model: String,

    /// The run directory to write; it is made if missing.
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,

    /// The models directory to read (default: `$PIE_HOME/models`).
    #[arg(long, value_name = "DIR")]
    pub models_dir: Option<PathBuf>,

    /// The checkpoint the trainer reads the frozen weights from; `import.json`
    /// is written only with it.
    #[arg(long, value_name = "DIR")]
    pub checkpoint: Option<PathBuf>,

    /// The precisions the weights are deployed in, one for each role the
    /// package's layout reads (`bf16`, or `bf16,mxfp4` for weights and experts).
    #[arg(long, default_value = "bf16", value_delimiter = ',', value_parser = precision)]
    pub weights: Vec<Precision>,

    /// The dtype the kv cache is deployed in.
    #[arg(long, default_value = "bf16", value_parser = dtype)]
    pub kv: Dtype,

    /// The platforms to lower the trace for, comma separated (`metal,cuda,xla,gpu,vulkan`).
    #[arg(long, value_delimiter = ',', value_name = "PLATFORMS")]
    pub lowered: Vec<String>,
}

/// Writes the run directory, and fails naming the bank when a lowering
/// loses one (after writing `lowered.json`, which shows how many).
pub fn run(args: TraceArgs) -> Result<Answer> {
    let platforms = args
        .lowered
        .iter()
        .map(|name| Ok((name, lowered::platform(name)?)))
        .collect::<Result<Vec<_>>>()?;
    let dir = args.models_dir.unwrap_or_else(crate::paths::models_dir);
    let (catalog, refused) = Catalog::of_tree(&Tree::read(&dir).map_err(|why| anyhow!("{why}"))?);
    let Some((package, model)) = catalog.model(&args.model) else {
        let why: String = refused.iter().map(|why| format!("\n  {why}")).collect();
        bail!(
            "no package under {} holds `{}`{why}",
            dir.display(),
            args.model
        );
    };
    let deploy = Deploy {
        weights: args.weights,
        ..one_rank(Dtype::Bf16, args.kv)
    };
    // Recorded for Metal, as a trainer replays it.
    let trace = package
        .trace(&model.id, &deploy, &model.id, Platform::Metal)
        .map_err(|why| anyhow!("{why:#}"))?;
    let banks: Vec<&Param> = trace
        .params
        .iter()
        .filter(|p| p.source == ParamSource::Registered)
        .collect();
    let mut summary = format!(
        "wrote {} for `{}` ({}): {} banks",
        crate::ui::short_path(&args.out),
        model.id,
        package.name(),
        banks.len(),
    );
    let mut files = vec![
        (
            "banks.json",
            banks::list(package.name(), &model.id, &banks)?,
        ),
        ("mount.json", mount::of(package, &dir, &banks)?),
    ];
    match &args.checkpoint {
        Some(checkpoint) => {
            let import = import::of(package, &model.id, &deploy, checkpoint, &trace)?;
            let reads = import["params"].as_array().map_or(0, Vec::len);
            summary.push_str(&format!(", {reads} checkpoint reads"));
            files.push(("import.json", import));
        }
        None => summary.push_str(", no import.json (no --checkpoint)"),
    }
    let mut lost = Vec::new();
    if !platforms.is_empty() {
        let deployment = Deployment::of(Arc::clone(package), model, deploy);
        let mut entries = serde_json::Map::new();
        for (name, platform) in platforms {
            let (entry, gone) = lowered::entry(&deployment, platform, &banks);
            match &entry["refused"] {
                Value::String(why) => summary.push_str(&format!("; {name} refused: {why}")),
                _ => summary.push_str(&format!(
                    "; {name} keeps {} of {} banks",
                    entry["banks_kept"], entry["banks_total"]
                )),
            }
            lost.extend(gone.map(|bank| format!("lowering for {name} loses the bank `{bank}`")));
            entries.insert(name.clone(), entry);
        }
        files.push(("lowered.json", Value::Object(entries)));
    }

    let out = &args.out;
    std::fs::create_dir_all(out).with_context(|| format!("make {}", out.display()))?;
    write(out, "trace.json", serde_json::to_string_pretty(&trace)?)?;
    for (file, value) in &files {
        write(out, file, serde_json::to_string_pretty(value)? + "\n")?;
    }
    if let Some(first) = lost.first() {
        let more = match lost.len() {
            1 => String::new(),
            n => format!(", and {} more platforms lose banks", n - 1),
        };
        bail!("{first} (absent, renamed or reshaped){more}");
    }
    Ok(Answer::did(summary))
}

/// The dtype `name` spells (`bf16`).
fn dtype(name: &str) -> Result<Dtype, String> {
    poem::star::dtype_of(name).ok_or_else(|| format!("no dtype `{name}`"))
}

/// The precision `name` spells (`bf16`, `mxfp4a8`).
fn precision(name: &str) -> Result<Precision, String> {
    poem::star::precision_of(name).ok_or_else(|| format!("no precision `{name}`"))
}

fn write(out: &Path, file: &str, text: String) -> Result<()> {
    let path = out.join(file);
    std::fs::write(&path, text).with_context(|| format!("write {}", path.display()))
}
