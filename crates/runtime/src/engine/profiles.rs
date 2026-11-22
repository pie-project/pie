//! The device-profile registries this build carries, one per backend it
//! hosts: what `pie profile` and doctor read.

use std::path::Path;

use profile::{Facts, Profile, Registry, Report, Shelf};

/// One backend's profiles: its registry, its checked-in files, and how to
/// read the device it would serve.
pub struct Backend {
    pub name: &'static str,
    pub registry: fn() -> &'static Registry,
    pub shelf: &'static Shelf,
    /// The device this machine would serve, by name, with its facts.
    pub probe: fn() -> Option<(String, Facts)>,
    /// Resolve the profile a load on that device runs under, as the engine
    /// does: with the user's profiles directory.
    pub resolve: fn(&str, &Facts, Option<&Path>) -> (Profile, Report),
    /// Measure this machine's device and write its profile: `pie tune`.
    /// `None` where the backend cannot tune yet.
    pub tune: Option<Tune>,
    /// Measure the device's `[rates]` alone: `pie tune --rates`. `None`
    /// where the engine has no rates step.
    pub rates: Option<Rates>,
    /// The engine's loads of one model for the check of the estimate
    /// (`engine::load::check`): `pie tune --check-estimate`. `None` where
    /// the engine offers none.
    pub bench: Option<BenchOf>,
}

/// `pie tune --rates` for one backend.
pub type Rates =
    fn(&TuneRequest<'_>, say: &mut dyn FnMut(&str)) -> Result<engine::rates::Rated, String>;

/// A backend's loads of the model of `artifact` (its deployment named) for
/// the check of the estimate.
pub type BenchOf = for<'a> fn(
    &'a TuneRequest<'a>,
    artifact: &'a Path,
    deployment: &'a str,
) -> Result<Box<dyn engine::load::check::Bench + 'a>, String>;

/// `pie tune` for one backend.
pub type Tune = fn(&TuneRequest<'_>, say: &mut dyn FnMut(&str)) -> Result<Tuned, String>;

/// What `pie tune` asks of a backend.
pub struct TuneRequest<'a> {
    /// Measure the whole grid rather than the quick one.
    pub full: bool,
    /// The user's profiles directory.
    pub profiles: &'a Path,
    /// Where imported artifacts live, for the end-to-end check.
    pub artifacts: &'a Path,
    /// Deployments to check end to end, by name or substring; empty takes
    /// the smallest artifact the backend can serve.
    pub check: &'a [String],
    /// Skip the end-to-end check.
    pub no_check: bool,
    /// Where measurements are kept between runs.
    pub cache: &'a Path,
    /// Fit the kept measurements again instead of measuring.
    pub refit: bool,
    /// Check the estimate alone: time the check models' fires at each rung
    /// width and hold the estimate's pick to the measured best.
    pub check_estimate: bool,
    /// With `check_estimate`: rung widths dissected beside the picks.
    pub dissect: &'a [u32],
    /// Run the `[rates]` step alone.
    pub rates: bool,
}

impl<'a> TuneRequest<'a> {
    /// A quick tune over these directories, every choice off: what a caller
    /// fills the rest of a request from (`..TuneRequest::at(..)`), so a
    /// field this build carries under a feature the caller's crate does not
    /// see (the workspace's feature unification turns `engine-cuda` on here
    /// for one member and not another) still has its value.
    #[must_use]
    pub fn at(profiles: &'a Path, artifacts: &'a Path, cache: &'a Path) -> TuneRequest<'a> {
        TuneRequest {
            full: false,
            profiles,
            artifacts,
            check: &[],
            no_check: false,
            cache,
            refit: false,
            check_estimate: false,
            dissect: &[],
            rates: false,
        }
    }
}

/// What a tune produced.
pub struct Tuned {
    pub device: String,
    /// The complete profile, for the user's shelf.
    pub file: String,
    /// Each tunable decision and what became of it.
    pub lines: Vec<(&'static str, String)>,
    /// Candidates that answered wrongly, and where.
    pub wrong: Vec<String>,
    /// Each end-to-end check, described, and whether it passed.
    pub checks: Vec<(String, bool)>,
    /// Whether the tuned file differs from today's at all.
    pub changed: bool,
    pub seconds: f64,
}

/// The check models among the artifacts stamped for `backend`: those
/// `req.check` names, or the smallest.
fn pick_models(
    backend: &str,
    req: &TuneRequest<'_>,
) -> Result<Vec<(std::path::PathBuf, String)>, String> {
    if req.no_check {
        return Ok(Vec::new());
    }
    let found = artifacts(req.artifacts, backend);
    if req.check.is_empty() {
        if found.is_empty() && req.check_estimate {
            return Err(format!(
                "a check of the estimate needs a model stamped for {backend}, and none under {} \
                 is; make one with the import `pie serve` runs, `pie model import <its source, \
                 e.g. Qwen/Qwen3-0.6B>`",
                req.artifacts.display()
            ));
        }
        return Ok(found.into_iter().take(1).collect());
    }
    let mut picked = Vec::new();
    for name in req.check {
        let hit = found
            .iter()
            .find(|(path, d)| {
                d == name || d.contains(name.as_str()) || path.to_string_lossy().contains(name.as_str())
            })
            .ok_or_else(|| {
                let have: Vec<&str> = found.iter().map(|(_, d)| d.as_str()).collect();
                format!(
                    "no stamped artifact for this backend matches `{name}` (servable here: {have:?}); \
                     `pie model list` marks the rest unstamped, which an import from this \
                     pie restamps: `pie model import <source>`"
                )
            })?;
        picked.push(hit.clone());
    }
    Ok(picked)
}

/// The `.zt` artifacts under `store` stamped for `backend`, as (path,
/// deployment), smallest file first.
fn artifacts(store: &Path, backend: &str) -> Vec<(std::path::PathBuf, String)> {
    let mut out: Vec<(u64, std::path::PathBuf, String)> = Vec::new();
    let Ok(dirs) = std::fs::read_dir(store) else {
        return Vec::new();
    };
    for dir in dirs.flatten() {
        let Ok(files) = std::fs::read_dir(dir.path()) else {
            continue;
        };
        for file in files.flatten() {
            let path = file.path();
            if path.extension().is_none_or(|e| e != "zt") {
                continue;
            }
            let Some(stamp) = checkpoint::file::serve::stamp_of(&path).ok().flatten() else {
                continue;
            };
            if stamp.backend == backend {
                let bytes = file.metadata().map_or(u64::MAX, |m| m.len());
                out.push((bytes, path, stamp.deployment));
            }
        }
    }
    out.sort_by_key(|(bytes, ..)| *bytes);
    out.into_iter().map(|(_, p, d)| (p, d)).collect()
}

/// `pie tune --rates`: the backend's rates step, refused by name where the
/// engine has none.
pub fn rates(
    backend: &Backend,
    req: &TuneRequest<'_>,
    say: &mut dyn FnMut(&str),
) -> Result<engine::rates::Rated, String> {
    let step = backend.rates.ok_or_else(|| {
        format!(
            "`pie tune --rates` measures a device's [rates] by its engine's rates step, and \
             engine-{} has none",
            backend.name
        )
    })?;
    step(req, say)
}

/// `pie tune --check-estimate`: the check of the estimate
/// (`engine::load::check`) on each check model through the backend's
/// bench, refused by name where the engine offers none. Every model is
/// checked before any failure is returned.
pub fn check_estimate(
    backend: &Backend,
    req: &TuneRequest<'_>,
    say: &mut dyn FnMut(&str),
) -> Result<Vec<engine::load::check::Checked>, String> {
    let bench_of = backend.bench.ok_or_else(|| {
        format!(
            "`pie tune --check-estimate` times fires through its engine's check bench \
             (engine::load::check::Bench), and engine-{} offers none",
            backend.name
        )
    })?;
    let models = pick_models(backend.name, req)?;
    let mut checked = Vec::new();
    let mut failed = Vec::new();
    for (artifact, deployment) in &models {
        say(&format!("check of the estimate: {deployment}"));
        let outcome = bench_of(req, artifact, deployment)
            .and_then(|mut bench| engine::load::check::check(&mut *bench, req.dissect, say));
        match outcome {
            Ok(c) if c.passes() => checked.push(c),
            Ok(c) => failed.push(c.failure()),
            Err(why) => failed.push(format!("{deployment}: {why}")),
        }
    }
    if failed.is_empty() {
        Ok(checked)
    } else {
        Err(failed.join("\n"))
    }
}

/// A backend's tune output as [`Tuned`]: each backend has its own type, with
/// the same field names.
#[cfg(any(
    feature = "engine-cuda",
    feature = "vulkan",
    all(feature = "metal", target_vendor = "apple")
))]
macro_rules! tuned {
    ($out:ident, wrong: $wrong:expr) => {
        Tuned {
            device: $out.device,
            file: $out.file,
            lines: $out.lines.into_iter().map(|l| (l.id, l.text)).collect(),
            wrong: $wrong,
            checks: $out
                .checks
                .iter()
                .map(|c| (c.describe(), c.passes()))
                .collect(),
            changed: $out.changed,
            seconds: $out.seconds,
        }
    };
}

/// `pie tune` on a backend that measures a kernel grid at a depth.
#[cfg(any(feature = "vulkan", all(feature = "metal", target_vendor = "apple")))]
macro_rules! tune_grid {
    ($name:ident, $engine:ident, $backend:literal) => {
        fn $name(req: &TuneRequest<'_>, say: &mut dyn FnMut(&str)) -> Result<Tuned, String> {
            use $engine::tune::{Depth, run};
            let depth = if req.full { Depth::Full } else { Depth::Quick };
            let models = pick_models($backend, req)?;
            let out = run(depth, Some(req.profiles), req.cache, req.refit, &models, say)?;
            Ok(tuned!(out, wrong: out.wrong))
        }
    };
}

#[cfg(feature = "vulkan")]
tune_grid!(tune_vulkan, engine_vulkan, "vulkan");

#[cfg(all(feature = "metal", target_vendor = "apple"))]
tune_grid!(tune_metal, engine_metal, "metal");

#[cfg(feature = "vulkan")]
fn bench_vulkan<'a>(
    req: &'a TuneRequest<'a>,
    artifact: &'a Path,
    deployment: &'a str,
) -> Result<Box<dyn engine::load::check::Bench + 'a>, String> {
    let bench = engine_vulkan::check::bench(
        req.profiles,
        artifact,
        deployment,
        crate::engine::load::contract_for,
    )?;
    Ok(Box::new(bench))
}

#[cfg(feature = "wgpu")]
fn bench_wgpu<'a>(
    req: &'a TuneRequest<'a>,
    artifact: &'a Path,
    deployment: &'a str,
) -> Result<Box<dyn engine::load::check::Bench + 'a>, String> {
    let bench = engine_wgpu::check::bench(
        req.profiles,
        artifact,
        deployment,
        crate::engine::load::contract_for,
    )?;
    Ok(Box::new(bench))
}

/// `pie tune metal --rates`: the device's `[rates]` by Metal's rates step.
#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn rates_metal(
    req: &TuneRequest<'_>,
    say: &mut dyn FnMut(&str),
) -> Result<engine::rates::Rated, String> {
    engine_metal::tune::rates(Some(req.profiles), say)
}

/// `pie tune metal --check-estimate`: Metal's loads of a model for the
/// check of the estimate.
#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn bench_metal<'a>(
    req: &'a TuneRequest<'a>,
    artifact: &'a Path,
    deployment: &'a str,
) -> Result<Box<dyn engine::load::check::Bench + 'a>, String> {
    Ok(Box::new(engine_metal::tune::check_bench(
        Some(req.profiles),
        artifact,
        deployment,
    )?))
}

#[cfg(feature = "engine-cuda")]
fn tune_cuda(req: &TuneRequest<'_>, say: &mut dyn FnMut(&str)) -> Result<Tuned, String> {
    let models = pick_models("cuda", req)?;
    let out = engine_cuda::tune::run(Some(req.profiles), &models, say)?;
    Ok(tuned!(out, wrong: Vec::new()))
}

/// CUDA's loads of one model for the check of the estimate.
#[cfg(feature = "engine-cuda")]
fn bench_cuda<'a>(
    req: &'a TuneRequest<'a>,
    artifact: &'a Path,
    deployment: &'a str,
) -> Result<Box<dyn engine::load::check::Bench + 'a>, String> {
    Ok(Box::new(engine_cuda::check::bench(
        artifact,
        deployment,
        Some(req.profiles),
    )))
}

/// What `pie tune export` wrote of a device ([`export`]).
pub struct Exported {
    /// The device, by name.
    pub device: String,
    /// The user's file it was read from.
    pub from: std::path::PathBuf,
    /// Its committed file's name (`profile::measured_name`).
    pub name: String,
    /// The committed measured profile.
    pub text: String,
}

/// The stored measurements of `backend`'s device `index` (the user's file
/// under `profiles` a load keeps them in) in committed form
/// (`profile::export`): refused by name where the device is not found, its
/// user file holds none, or an entry is stored without a key.
pub fn export(backend: &Backend, profiles: &Path, index: u32) -> Result<Exported, String> {
    let device = device_name(backend, index)?;
    let from = backend.shelf.user_path(profiles, &device);
    let user = std::fs::read_to_string(&from).map_err(|e| {
        format!(
            "{}: {e}: no measurement of `{device}` is stored; a load of it stores them",
            from.display()
        )
    })?;
    let text = profile::export(
        (backend.registry)(),
        &device,
        &user,
        &from.display().to_string(),
    )?;
    Ok(Exported {
        name: profile::measured_name(&device),
        device,
        from,
        text,
    })
}

/// The records of what a deployment's bodies took (the cuda engine's
/// memory measurement) that this build made, from `cache_dir` into `out`'s
/// `bodies/`, beside its committed measured profile: the names written.
/// Another engine keeps none.
pub fn export_bodies(backend: &Backend, cache_dir: &Path, out: &Path) -> Result<Vec<String>, String> {
    match backend.name {
        #[cfg(feature = "engine-cuda")]
        "cuda" => engine_cuda::serve::export_bodies(cache_dir, out)
            .map_err(|e| format!("{}: {e}", out.display())),
        _ => {
            let _ = (cache_dir, out);
            Ok(Vec::new())
        }
    }
}

/// The name of `backend`'s device `index`, as its loads file its profiles.
fn device_name(backend: &Backend, index: u32) -> Result<String, String> {
    let probed = match backend.name {
        #[cfg(feature = "vulkan")]
        "vulkan" => engine_vulkan::chip::probe_at(index),
        #[cfg(feature = "wgpu")]
        "wgpu" => engine_wgpu::chip::probe_at(index),
        _ if index == 0 => (backend.probe)(),
        other => {
            return Err(format!(
                "`--device {index}`: the {other} engine names its first device alone here"
            ));
        }
    };
    probed
        .map(|(name, _)| name)
        .ok_or_else(|| format!("no {} device {index} on this machine", backend.name))
}

/// Every backend this build hosts.
#[must_use]
#[allow(clippy::vec_init_then_push)]
pub fn backends() -> Vec<Backend> {
    #[allow(unused_mut)]
    let mut out = Vec::new();
    #[cfg(feature = "engine-cuda")]
    out.push(Backend {
        name: "cuda",
        registry: engine_cuda::chip::registry,
        shelf: &engine_cuda::chip::SHELF,
        probe: || engine_cuda::chip::probe(0),
        resolve: engine_cuda::chip::resolve,
        tune: Some(tune_cuda),
        rates: Some(|req, say| engine_cuda::tune::rates(Some(req.profiles), say)),
        bench: Some(bench_cuda),
    });
    #[cfg(feature = "vulkan")]
    out.push(Backend {
        name: "vulkan",
        registry: engine_vulkan::chip::registry,
        shelf: &engine_vulkan::chip::SHELF,
        probe: engine_vulkan::chip::probe,
        resolve: engine_vulkan::chip::resolve_facts,
        tune: Some(tune_vulkan),
        rates: Some(|req, say| engine_vulkan::check::rated(req.profiles, say)),
        bench: Some(bench_vulkan),
    });
    #[cfg(feature = "wgpu")]
    out.push(Backend {
        name: "wgpu",
        registry: engine_wgpu::chip::registry,
        shelf: &engine_wgpu::chip::SHELF,
        probe: engine_wgpu::chip::probe,
        resolve: engine_wgpu::chip::resolve_facts,
        tune: None,
        rates: Some(|req, say| engine_wgpu::check::rated(req.profiles, say)),
        bench: Some(bench_wgpu),
    });
    #[cfg(feature = "xla")]
    out.push(Backend {
        name: "xla",
        registry: engine_xla::chip::registry,
        shelf: &engine_xla::chip::SHELF,
        // Opening a PJRT client to read the device kind takes the device;
        // name one with `--device`.
        probe: || None,
        resolve: |kind, _, profiles| engine_xla::chip::resolve(kind, profiles),
        tune: None,
        rates: None,
        bench: None,
    });
    #[cfg(feature = "metal")]
    out.push(Backend {
        name: "metal",
        registry: engine_metal::chip::registry,
        shelf: &engine_metal::chip::SHELF,
        probe: engine_metal::chip::probe,
        resolve: engine_metal::chip::resolve,
        #[cfg(target_vendor = "apple")]
        tune: Some(tune_metal),
        #[cfg(not(target_vendor = "apple"))]
        tune: None,
        #[cfg(target_vendor = "apple")]
        rates: Some(rates_metal),
        #[cfg(not(target_vendor = "apple"))]
        rates: None,
        #[cfg(target_vendor = "apple")]
        bench: Some(bench_metal),
        #[cfg(not(target_vendor = "apple"))]
        bench: None,
    });
    out
}
