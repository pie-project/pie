//! `pie tune`: measure this machine's device, fit the profile decisions its
//! backend can measure, and write the result to the user's shelf.

use anyhow::{Result, anyhow, bail};
use clap::{Args, Subcommand};
use runtime::engine::profiles::backends;

use super::profile::text;
use crate::ui::{self, Answer, Palette};

#[derive(Args, Debug)]
pub struct TuneArgs {
    #[command(subcommand)]
    pub cmd: Option<TuneCmd>,

    /// The backend to tune; needed only where this build hosts several.
    pub backend: Option<String>,

    /// Measure the whole grid (about an hour) rather than the quick one
    /// (minutes). Expected for a profile you contribute.
    #[arg(long)]
    pub full: bool,

    /// Measure and print the profile, but write nothing.
    #[arg(long)]
    pub dry_run: bool,

    /// A deployment to serve end to end under today's profile and the
    /// tuned one, by name or substring; repeatable. Without it, the
    /// smallest imported artifact.
    #[arg(long)]
    pub check: Vec<String>,

    /// Skip the end-to-end check.
    #[arg(long)]
    pub no_check: bool,

    /// Write the profile even if the end-to-end check finds it slower.
    #[arg(long)]
    pub force: bool,

    /// Fit the measurements the last run kept, without measuring again.
    #[arg(long)]
    pub refit: bool,

    /// Check the estimate the load weighs its top rung by: time a fire at
    /// each rung width of these models (by name or substring; the smallest
    /// imported artifact without one), to four times the fire top, through
    /// the serving path; print the measured and estimated times a rung and
    /// both picks, and fail where the estimate's pick falls more than the
    /// stated bound short of the measured best. Writes nothing.
    #[arg(long, num_args = 0..)]
    pub check_estimate: Option<Vec<String>>,

    /// With `--check-estimate`: also dissect these rung widths (comma
    /// separated), each replayed and walked node by node beside the
    /// estimate's picks.
    #[arg(long, value_delimiter = ',', requires = "check_estimate")]
    pub dissect: Vec<u32>,

    /// Run the `[rates]` step alone: measure the device's roofline
    /// constants (launch, matmul and attention rates, bandwidths, scan).
    #[arg(long, conflicts_with = "check_estimate")]
    pub rates: bool,
}

#[derive(Subcommand, Debug)]
pub enum TuneCmd {
    /// Write a device's stored measurements (the user's profile a load
    /// keeps them in) as a committed measured profile,
    /// `<out>/<device>.measured.toml`: each entry with its key, in a fixed
    /// order, no date or path. Checked in under
    /// `crates/engine-<backend>/profiles/`, a load of that device takes each
    /// entry whose key matches and measures nothing more for it.
    Export {
        /// The engine whose stored measurements to write: `vulkan`, `wgpu`.
        #[arg(long)]
        engine: String,
        /// The device, by number (`vulkan:<n>`); the first without it.
        #[arg(long, value_name = "N")]
        device: Option<u32>,
        /// The directory to write into: the engine's `profiles/`.
        #[arg(long, value_name = "DIR")]
        out: std::path::PathBuf,
    },
}

pub fn run(args: TuneArgs) -> Result<Answer> {
    if let Some(TuneCmd::Export {
        engine,
        device,
        out,
    }) = &args.cmd
    {
        return export(engine, *device, out).map(text);
    }
    let mut all = backends();
    let names: Vec<&str> = all.iter().map(|b| b.name).collect();
    let tunable: Vec<&str> = all
        .iter()
        .filter(|b| b.tune.is_some())
        .map(|b| b.name)
        .collect();
    let at = match args.backend.as_deref() {
        Some(name) => all
            .iter()
            .position(|b| b.name == name)
            .ok_or_else(|| anyhow!("this build hosts no `{name}` backend; it hosts {names:?}"))?,
        None => match tunable.as_slice() {
            [one] => all.iter().position(|b| b.name == *one).expect("listed"),
            [] => bail!(
                "no backend this build hosts can tune yet (it hosts {names:?}); \
                 `pie tune` measures Metal on Apple silicon, CUDA and Vulkan"
            ),
            _ => bail!("name a backend to tune: {tunable:?}"),
        },
    };
    let backend = all.swap_remove(at);
    let dir = crate::paths::pie_home().join("profiles");
    let palette = Palette::for_stream(ui::Stream::Stderr);
    let mut say = |line: &str| eprintln!("{}", palette.dim(line));
    let artifacts = crate::paths::pie_home().join("artifacts");
    let cache = crate::paths::pie_home().join("tune");
    let mut check = args.check.clone();
    check.extend(args.check_estimate.iter().flatten().cloned());
    let req = runtime::engine::profiles::TuneRequest {
        full: args.full,
        check: &check,
        no_check: args.no_check,
        refit: args.refit,
        check_estimate: args.check_estimate.is_some(),
        dissect: &args.dissect,
        rates: args.rates,
        ..runtime::engine::profiles::TuneRequest::at(&dir, &artifacts, &cache)
    };
    // A check of the estimate is a test: it writes nothing.
    if req.check_estimate {
        let checked = runtime::engine::profiles::check_estimate(&backend, &req, &mut say)
            .map_err(|e| anyhow!(e))?;
        let mut out = String::new();
        for c in &checked {
            out.push_str(&c.report);
        }
        out.push_str(&format!(
            "the estimate's pick is within the bound of the measured best for {} model(s)\n",
            checked.len()
        ));
        return Ok(text(out));
    }
    if req.rates {
        let rated =
            runtime::engine::profiles::rates(&backend, &req, &mut say).map_err(|e| anyhow!(e))?;
        let head = format!(
            "{}: rates measured in {:.0} s\n{}",
            rated.device, rated.seconds, rated.block
        );
        return write_profile(&backend, &args, &dir, &rated.device, &rated.file, head).map(text);
    }
    let tune = backend.tune.ok_or_else(|| {
        anyhow!(
            "`pie tune` cannot measure {} yet; start from `pie profile dump {}` instead",
            backend.name,
            backend.name
        )
    })?;
    let tuned = tune(&req, &mut say).map_err(|e| anyhow!(e))?;

    let mut out = format!("{}: tuned in {:.0} s\n", tuned.device, tuned.seconds);
    for (id, text) in &tuned.lines {
        out.push_str(&format!("  {id}: {text}\n"));
    }
    if !tuned.wrong.is_empty() {
        out.push_str("candidates that answered wrongly, left out where they did:\n");
        for w in &tuned.wrong {
            out.push_str(&format!("  {w}\n"));
        }
    }
    let failed = tuned.checks.iter().any(|(_, passed)| !passed);
    if !tuned.changed {
        out.push_str("nothing changed: today's profile is already the fastest the search found\n");
    } else if tuned.checks.is_empty() {
        out.push_str(
            "end to end: not checked (no artifact for this backend imported, or --no-check)\n",
        );
    }
    for (line, passed) in &tuned.checks {
        out.push_str(&format!(
            "end to end: {line}{}\n",
            if *passed {
                ""
            } else {
                "  <- slower or different"
            }
        ));
    }
    if failed && !args.force && !args.dry_run {
        bail!(
            "{out}the tuned profile serves slower or reads differently than today's; \
             nothing written (--force writes it anyway)"
        );
    }
    let mut answer = write_profile(&backend, &args, &dir, &tuned.device, &tuned.file, out)?;
    if !args.full && !args.dry_run {
        answer.push_str(
            "This was the quick grid; run `pie tune --full` for a profile you contribute.\n",
        );
    }
    Ok(text(answer))
}

/// Write `file`, the profile of `device`, to the user's shelf (or, with
/// `--dry-run`, print it), the previous one kept beside it; `out` is what
/// the run says so far, the answer's head.
fn write_profile(
    backend: &runtime::engine::profiles::Backend,
    args: &TuneArgs,
    dir: &std::path::Path,
    device: &str,
    file: &str,
    mut out: String,
) -> Result<String> {
    if args.dry_run {
        out.push_str("\n--dry-run: nothing written. The profile:\n\n");
        out.push_str(file);
        return Ok(out);
    }
    let id = profile::id_of(device);
    let path = backend.shelf.user_path(dir, device);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if path.exists() {
        let kept = path.with_extension("toml.bak");
        std::fs::copy(&path, &kept)?;
        out.push_str(&format!(
            "the previous profile is kept at {}\n",
            ui::short_path(&kept)
        ));
    }
    std::fs::write(&path, file)?;
    out.push_str(&format!(
        "wrote {}; every load on this device now runs it.\n\
         Consider contributing it:\n    \
         pie profile export {id} > {id}.toml\n\
         and open a PR at https://github.com/pie-project/pie adding it under \
         crates/engine-{}/profiles/. The file holds device facts and measured \
         numbers, nothing else.\n",
        ui::short_path(&path),
        backend.name
    ));
    Ok(out)
}

/// `pie tune export`: the stored measurements of `engine`'s device `device`
/// written to `out` in committed form (`profile::export`).
fn export(engine: &str, device: Option<u32>, out: &std::path::Path) -> Result<String> {
    let all = backends();
    let names: Vec<&str> = all.iter().map(|b| b.name).collect();
    let backend = all
        .iter()
        .find(|b| b.name == engine)
        .ok_or_else(|| anyhow!("this build hosts no `{engine}` engine; it hosts {names:?}"))?;
    let dir = crate::paths::pie_home().join("profiles");
    let cache = crate::paths::pie_home().join("cache");
    let exported = runtime::engine::profiles::export(backend, &dir, &cache, device.unwrap_or(0))
        .map_err(|e| anyhow!(e))?;
    std::fs::create_dir_all(out)?;
    let path = out.join(&exported.name);
    std::fs::write(&path, &exported.text)?;
    let covers = match &exported.covers {
        Some((name, text)) => {
            let at = out.join(name);
            std::fs::write(&at, text)?;
            format!(
                "wrote {}: the host covers the home's cache holds for it; listed in \
                 crates/engine-{}/src/chip.rs `COVERS` (`include_str!(\"../profiles/{name}\")`), \
                 a boot whose cover key matches one takes it.\n",
                ui::short_path(&at),
                backend.name,
            )
        }
        None => "no host cover is cached for it in this home.\n".to_string(),
    };
    let main = format!(
        "wrote {} from {}: the stored measurements of {}, each under its key.\n\
         A build carries it once it is listed beside the engine's checked-in profiles \
         (crates/engine-{}/src/chip.rs, `\"{}\"`); a load of that device then takes each entry \
         whose key matches.\n",
        ui::short_path(&path),
        ui::short_path(&exported.from),
        exported.device,
        backend.name,
        exported.name,
    );
    Ok(main + &covers)
}
