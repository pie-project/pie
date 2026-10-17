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
}

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
#[cfg(any(
    feature = "cuda",
    feature = "vulkan",
    all(feature = "metal", target_vendor = "apple")
))]
fn pick_models(
    backend: &str,
    req: &TuneRequest<'_>,
) -> Result<Vec<(std::path::PathBuf, String)>, String> {
    if req.no_check {
        return Ok(Vec::new());
    }
    let found = artifacts(req.artifacts, backend);
    if req.check.is_empty() {
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
#[cfg(any(
    feature = "cuda",
    feature = "vulkan",
    all(feature = "metal", target_vendor = "apple")
))]
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

/// A backend's tune output as [`Tuned`]: each backend has its own type, with
/// the same field names.
#[cfg(any(
    feature = "cuda",
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

#[cfg(feature = "cuda")]
fn tune_cuda(req: &TuneRequest<'_>, say: &mut dyn FnMut(&str)) -> Result<Tuned, String> {
    let models = pick_models("cuda", req)?;
    let out = engine_cuda::tune::run(Some(req.profiles), &models, say)?;
    Ok(tuned!(out, wrong: Vec::new()))
}

/// Every backend this build hosts.
#[must_use]
#[allow(clippy::vec_init_then_push)]
pub fn backends() -> Vec<Backend> {
    #[allow(unused_mut)]
    let mut out = Vec::new();
    #[cfg(feature = "cuda")]
    out.push(Backend {
        name: "cuda",
        registry: engine_cuda::chip::registry,
        shelf: &engine_cuda::chip::SHELF,
        probe: || engine_cuda::chip::probe(0),
        resolve: engine_cuda::chip::resolve,
        tune: Some(tune_cuda),
    });
    #[cfg(feature = "vulkan")]
    out.push(Backend {
        name: "vulkan",
        registry: engine_vulkan::chip::registry,
        shelf: &engine_vulkan::chip::SHELF,
        probe: engine_vulkan::chip::probe,
        resolve: engine_vulkan::chip::resolve_facts,
        tune: Some(tune_vulkan),
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
    });
    out
}
