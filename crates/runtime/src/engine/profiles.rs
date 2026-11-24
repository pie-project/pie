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

/// What `pie tune export` wrote of a device ([`export`]): each part the
/// home holds, `None` for each it holds none of.
pub struct Exported {
    /// The device, by name.
    pub device: String,
    /// Why the engine was not probed, where the device is instead the one
    /// the home's stored measurements are filed under.
    pub unprobed: Option<String>,
    /// The user's file the measurements were read from.
    pub from: std::path::PathBuf,
    /// Its committed file's name (`profile::measured_name`) and text: `None`
    /// where the home stores no measurement of the device (every entry a
    /// load took matched a committed one), with why.
    pub measured: Result<(String, String), String>,
    /// What a boot of each model the export carries a settle of reads, and
    /// whether each is carried (`profile::settles_carried`), a line a model.
    pub report: Vec<String>,
    /// The host covers the home's cache holds for this device
    /// (`<device>.covers.json`, its name and text): each entry whose key
    /// names the device the measurements were taken on. `None` where it
    /// holds none.
    pub covers: Option<(String, String)>,
}

/// The name a device's committed host covers are filed under:
/// `<id>.covers.json`, beside its committed measured profile.
#[must_use]
pub fn covers_name(device: &str) -> String {
    format!("{}.covers.json", profile::id_of(device))
}

/// Every host cover the cache under `cache` holds (`<cache>/cover/*.json`)
/// whose key names `on` (the device and driver the measurements were taken
/// on, as an engine's cover fingerprint names them), sorted by key: a JSON
/// list of the cache's own entries, as the runtime reads a committed one.
/// `None` where it holds none.
#[must_use]
pub fn covers_of(cache: &Path, on: &str) -> Option<String> {
    let mut held: Vec<serde_json::Value> = std::fs::read_dir(cache.join("cover"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
        .filter_map(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .filter(|entry| {
            entry
                .get("key")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|key| key.split('|').any(|part| part == on))
        })
        .collect();
    if held.is_empty() {
        return None;
    }
    held.sort_by(|a, b| {
        let key = |v: &serde_json::Value| v.get("key").and_then(|k| k.as_str()).unwrap_or("").to_owned();
        key(a).cmp(&key(b))
    });
    serde_json::to_string_pretty(&held).ok().map(|text| text + "\n")
}

/// What the home holds of `backend`'s device `index`: its stored
/// measurements (the user's file under `profiles` a load keeps them in) in
/// committed form (`profile::export`; an entry stored without a key is
/// refused by name), and the host covers its cache (`cache`) holds for the
/// device ([`covers_of`]). A part the home holds none of is `None`, said
/// why; the device not found is refused.
pub fn export(
    backend: &Backend,
    profiles: &Path,
    cache: &Path,
    index: u32,
) -> Result<Exported, String> {
    let mut unprobed = None;
    let device = match device_name(backend, index) {
        Ok(device) => device,
        // An engine that cannot be probed beside a load (XLA: opening its
        // client takes the device): the device the home's stored entries
        // are filed under, where they are one device's.
        Err(why) => {
            let stored = stored_devices(backend, profiles);
            match stored.as_slice() {
                [only] => {
                    tracing::info!(
                        device = only.as_str(),
                        "pie tune export: the {} engine was not probed ({why}); the device is the \
                         one the home's stored measurements are filed under",
                        backend.name
                    );
                    unprobed = Some(why);
                    only.clone()
                }
                [] => return Err(why),
                many => {
                    return Err(format!(
                        "{why}; the home stores measurements of {} devices ({}): export names \
                         one only where it can probe it",
                        many.len(),
                        many.join(", ")
                    ));
                }
            }
        }
    };
    let from = backend.shelf.user_path(profiles, &device);
    let committed = backend
        .shelf
        .seeded
        .iter()
        .find(|(path, _)| *path == profile::measured_name(&device))
        .map(|(_, text)| *text);
    let now = device_now(backend, index, &device);
    let mut exported = held(
        (backend.registry)(),
        device.clone(),
        from,
        cache,
        committed,
        now.as_deref(),
        || device_on(backend, index, &device),
    );
    exported.unprobed = unprobed;
    Ok(exported)
}

/// The devices `backend`'s stored measurements in the home are filed
/// under (`Shelf::user_path`'s file names, each a device's id), sorted.
fn stored_devices(backend: &Backend, profiles: &Path) -> Vec<String> {
    let dir = backend
        .shelf
        .user_path(profiles, "x")
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let mut ids: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            name.strip_suffix(".toml").map(str::to_string)
        })
        .collect();
    ids.sort();
    ids
}

/// [`export`] of `device`, its user's file at `from` (read where it is) and
/// the home's cache `cache`; `on` names the device its covers key where the
/// file states none.
fn held(
    registry: &Registry,
    device: String,
    from: std::path::PathBuf,
    cache: &Path,
    committed: Option<&str>,
    now: Option<&dyn profile::fire::Now>,
    on: impl FnOnce() -> Option<String>,
) -> Exported {
    let origin = from.display().to_string();
    let user = std::fs::read_to_string(&from).map_err(|e| {
        format!(
            "{}: {e}: no measurement of `{device}` is stored (a load that took every entry from \
             a committed profile stores none)",
            from.display()
        )
    });
    let stored = user
        .as_ref()
        .ok()
        .map(|user| registry.parse(user, &origin, profile::Strict::User));
    let stored_on = match &stored {
        Some(Ok(file)) => file.tuned.rates.measured_on.clone(),
        _ => None,
    };
    let committed_rates = committed
        .map(|text| registry.parse(text, &profile::measured_name(&device), profile::Strict::Measured));
    let mut report = Vec::new();
    // The home's measured entries over what of the committed profile still
    // holds for this build, each said: a re-export never shrinks it.
    let measured = (|| -> Result<(String, String), String> {
        let mine = match stored {
            Some(Ok(file)) => file.tuned.rates,
            Some(Err(why)) => return Err(why.to_string()),
            None => profile::fire::Rates::default(),
        };
        let theirs = match committed_rates {
            Some(Ok(file)) => {
                for line in &file.skipped_rates {
                    report.push(format!("dropped from the committed profile: {line}"));
                }
                Some(file.tuned.rates)
            }
            // Not one entry of it reads (each list entry is read alone, so
            // this is the file as a whole): none carried, said by name.
            Some(Err(why)) => {
                report.push(format!("dropped: the committed profile does not read ({why})"));
                None
            }
            None => None,
        };
        if mine == profile::fire::Rates::default() && theirs.is_none() {
            return Err(user.clone().err().unwrap_or_else(|| format!("{origin}: stores nothing")));
        }
        let (merged, lines) = profile::merge_committed(mine, theirs, now)?;
        report.extend(lines);
        let (text, lines) = profile::export_rates(registry, &device, merged, &origin)?;
        report.extend(lines);
        Ok((profile::measured_name(&device), text))
    })();
    let covers = stored_on
        .or_else(on)
        .and_then(|on| covers_of(cache, &on))
        .map(|text| (covers_name(&device), text));
    Exported {
        device,
        unprobed: None,
        from,
        measured,
        report,
        covers,
    }
}

/// What this build keys `backend`'s device `index` (named `device`) by
/// (`profile::fire::Now`): what a committed entry is checked against before
/// an export carries it. `None` where the engine states none here (its
/// committed entries are carried unchecked, and a boot checks them).
fn device_now(backend: &Backend, index: u32, device: &str) -> Option<Box<dyn profile::fire::Now>> {
    let _ = (index, device);
    match backend.name {
        #[cfg(feature = "vulkan")]
        "vulkan" => engine_vulkan::chip::now_at(index),
        #[cfg(feature = "wgpu")]
        "wgpu" => engine_wgpu::chip::now_at(index),
        #[cfg(feature = "engine-cuda")]
        "cuda" => Some(Box::new(engine_cuda::stored::Keys::of(
            i32::try_from(index).unwrap_or(0),
            device,
        ))),
        _ => None,
    }
}

/// The device and driver `backend`'s device `index` (named `device`) keys
/// its measurements as taken on: what its host covers name, where no stored
/// file of the home states it.
fn device_on(backend: &Backend, index: u32, device: &str) -> Option<String> {
    let _ = (index, device);
    match backend.name {
        #[cfg(feature = "vulkan")]
        "vulkan" => engine_vulkan::chip::measured_on_at(index),
        #[cfg(feature = "wgpu")]
        "wgpu" => engine_wgpu::chip::measured_on_at(index),
        #[cfg(feature = "engine-cuda")]
        "cuda" => Some(profile::fire::Now::on(&engine_cuda::stored::Keys::of(
            i32::try_from(index).unwrap_or(0),
            device,
        ))),
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        "metal" => Some(engine_metal::stored::measured_on(device)),
        _ => None,
    }
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

#[cfg(test)]
mod cover_tests {
    use super::*;

    /// The covers a home's cache holds for a device are exported (another
    /// device's left out), and a boot reads each back by its exact key.
    #[test]
    fn a_homes_covers_export_and_read_back_by_key() {
        let cache = std::env::temp_dir().join(format!("pie-covers-{}", std::process::id()));
        let dir = cache.join("cover");
        std::fs::create_dir_all(&dir).unwrap();
        let on = "vulkan 10de:2c31 driver 1";
        let entry = |key: &str, cover: u32| {
            serde_json::json!({"key": key, "cover": cover, "frame_us": 5400.0, "turnaround_us": 97.0})
                .to_string()
        };
        let ours = format!("Spirv|{on}|kernels 24 0x1|model a|sampler 9|frame 1|depth 2|context 64");
        let theirs = "Spirv|vulkan 10de:9999 driver 1|kernels 24 0x1|model a".to_string();
        std::fs::write(dir.join("a.json"), entry(&ours, 1)).unwrap();
        std::fs::write(dir.join("b.json"), entry(&theirs, 2)).unwrap();
        let text = covers_of(&cache, on).expect("one cover of the device");
        std::fs::remove_dir_all(&cache).unwrap();
        assert!(!text.contains("10de:9999"), "{text}");
        assert_eq!(
            crate::bootstrap::committed_cover(&text, &ours),
            Some(crate::bootstrap::HeldCover {
                cover: 1,
                frame_us: 5400.0,
                turnaround_us: 97.0
            })
        );
        assert_eq!(crate::bootstrap::committed_cover(&text, "another key"), None);
        assert_eq!(covers_name("NVIDIA RTX PRO 4500 Blackwell"), "nvidia-rtx-pro-4500-blackwell.covers.json");
    }

    /// A home that stores no measurement of the device (every entry a load
    /// took matched a committed one) and caches its covers: the export
    /// writes the covers, by the device the engine names, and says the
    /// measured profile is none.
    #[test]
    fn a_home_with_only_covers_exports_its_covers() {
        let home = std::env::temp_dir().join(format!("pie-only-covers-{}", std::process::id()));
        let cache = home.join("cache");
        std::fs::create_dir_all(cache.join("cover")).unwrap();
        let on = "vulkan 10de:2c31 driver 1";
        let key = format!("Spirv|{on}|kernels 24 0x1|model a");
        std::fs::write(
            cache.join("cover").join("a.json"),
            serde_json::json!({"key": key, "cover": 1, "frame_us": 5.0, "turnaround_us": 1.0})
                .to_string(),
        )
        .unwrap();
        let registry = Registry::new("toy", &[], &[]);
        let exported = held(
            &registry,
            "Card X".into(),
            home.join("profiles/toy/user/card-x.toml"),
            &cache,
            None,
            None,
            || Some(on.to_string()),
        );
        std::fs::remove_dir_all(&home).unwrap();
        assert!(exported.measured.is_err(), "no measured profile");
        let (name, text) = exported.covers.expect("the covers are written");
        assert_eq!(name, "card-x.covers.json");
        assert!(text.contains(&key), "{text}");
    }
}

#[cfg(test)]
mod committed_tests {
    use super::*;

    /// Every committed measured profile this build carries reads whole, as
    /// committed (`Strict::Measured`): an entry another pie's form left (a
    /// format change landed without the device files re-exported) fails the
    /// build here, where a load would skip it and measure it again.
    #[test]
    fn every_committed_profile_reads_whole() {
        for backend in backends() {
            for (name, text) in backend.shelf.seeded {
                if !name.ends_with(profile::MEASURED_SUFFIX) {
                    continue;
                }
                let file = (backend.registry)()
                    .parse(text, name, profile::Strict::Measured)
                    .unwrap_or_else(|e| panic!("{}: {name}: {e}", backend.name));
                assert!(
                    file.skipped_rates.is_empty(),
                    "{}: {name} carries entries this build skips (re-export it): {:?}",
                    backend.name,
                    file.skipped_rates
                );
            }
        }
    }

    /// Every committed host covers file in the tree is a list of the cover
    /// cache's entries, each with its key and figures.
    #[test]
    fn every_committed_covers_file_reads() {
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut read = 0;
        for engine in std::fs::read_dir(&crates).unwrap().flatten() {
            let profiles = engine.path().join("profiles");
            for file in std::fs::read_dir(&profiles).into_iter().flatten().flatten() {
                let path = file.path();
                if !path.to_string_lossy().ends_with(".covers.json") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).unwrap();
                let all: Vec<serde_json::Value> = serde_json::from_str(&text)
                    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                for entry in &all {
                    let key = entry
                        .get("key")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_else(|| panic!("{}: an entry with no key", path.display()));
                    assert!(
                        crate::bootstrap::committed_cover(&text, key).is_some(),
                        "{}: `{key}` does not read as a cover",
                        path.display()
                    );
                }
                read += 1;
            }
        }
        assert!(read > 0, "no committed covers file found under {}", crates.display());
    }
}
