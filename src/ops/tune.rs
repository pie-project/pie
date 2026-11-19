//! `pie tune`: measure this machine's device, fit the profile decisions its
//! backend can measure, and write the result to the user's shelf.

use anyhow::{Result, anyhow, bail};
use clap::Args;
use runtime::engine::profiles::backends;

use super::profile::text;
use crate::ui::{self, Answer, Palette};

#[derive(Args, Debug)]
pub struct TuneArgs {
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

/// `[engine.memory] budget_device` of the operator's config at `path`, in
/// bytes: none where the file states none or there is no file; a file that
/// does not read is refused by name.
fn budget_device(path: &std::path::Path) -> Result<Option<u64>> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Ok(None);
    };
    let config = worker::config::Config::parse_file(&text, Some(path))?;
    Ok(config.engine.memory.budget_device.map(|b| b.as_bytes()))
}

pub fn run(args: TuneArgs, global: &crate::args::GlobalArgs) -> Result<Answer> {
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
        budget_device: budget_device(&crate::args::config_path_or(global, "config.toml").0)?,
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

#[cfg(test)]
mod budget_tests {
    use super::budget_device;

    /// A tune reads the operator's `[engine.memory] budget_device` for its
    /// check's loads; none where the config states none or there is none.
    #[test]
    fn a_tune_reads_the_configs_device_budget() {
        let dir = std::env::temp_dir().join(format!("pie-tune-budget-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.toml");
        let engine = "[model]\nname = \"default\"\nsource = \"Qwen/Qwen3-0.6B\"\n\
                      [engine]\ntype = \"wgpu\"\ndevice = [\"wgpu:0\"]\n";
        std::fs::write(
            &path,
            format!("{engine}[engine.memory]\nbudget_device = \"12GiB\"\n"),
        )
        .unwrap();
        assert_eq!(budget_device(&path).unwrap(), Some(12 << 30));
        std::fs::write(&path, engine).unwrap();
        assert_eq!(budget_device(&path).unwrap(), None);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(budget_device(&dir.join("config.toml")).unwrap(), None);
    }
}
