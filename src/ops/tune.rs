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
}

pub fn run(args: TuneArgs) -> Result<Answer> {
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
    let tune = backend.tune.ok_or_else(|| {
        anyhow!(
            "`pie tune` cannot measure {} yet; start from `pie profile dump {}` instead",
            backend.name,
            backend.name
        )
    })?;
    let dir = crate::paths::pie_home().join("profiles");
    let palette = Palette::for_stream(ui::Stream::Stderr);
    let mut say = |line: &str| eprintln!("{}", palette.dim(line));
    let artifacts = crate::paths::pie_home().join("artifacts");
    let cache = crate::paths::pie_home().join("tune");
    let req = runtime::engine::profiles::TuneRequest {
        full: args.full,
        profiles: &dir,
        artifacts: &artifacts,
        check: &args.check,
        no_check: args.no_check,
        cache: &cache,
        refit: args.refit,
    };
    let tuned = tune(&req, &mut say).map_err(|e| anyhow!(e))?;

    let id = profile::id_of(&tuned.device);
    let path = backend.shelf.user_path(&dir, &tuned.device);
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
    if args.dry_run {
        out.push_str("\n--dry-run: nothing written. The profile:\n\n");
        out.push_str(&tuned.file);
        return Ok(text(out));
    }
    if failed && !args.force {
        bail!(
            "{out}the tuned profile serves slower or reads differently than today's; \
             nothing written (--force writes it anyway)"
        );
    }
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
    std::fs::write(&path, &tuned.file)?;
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
    if !args.full {
        out.push_str(
            "This was the quick grid; run `pie tune --full` for a profile you contribute.\n",
        );
    }
    Ok(text(out))
}
