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
    pub seconds: f64,
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn tune_metal(req: &TuneRequest<'_>, say: &mut dyn FnMut(&str)) -> Result<Tuned, String> {
    use engine_metal::tune::{Depth, artifacts, run};
    let depth = if req.full { Depth::Full } else { Depth::Quick };
    let found = artifacts(req.artifacts);
    let models: Vec<_> = if req.no_check {
        Vec::new()
    } else if req.check.is_empty() {
        found.into_iter().take(1).collect()
    } else {
        let mut picked = Vec::new();
        for name in req.check {
            let hit = found
                .iter()
                .find(|(path, d)| d == name || d.contains(name.as_str()) || path.to_string_lossy().contains(name.as_str()))
                .ok_or_else(|| format!("no Metal artifact matches `{name}`; import one with `pie model import`"))?;
            picked.push(hit.clone());
        }
        picked
    };
    let out = run(depth, Some(req.profiles), req.cache, req.refit, &models, say)?;
    Ok(Tuned {
        device: out.device,
        file: out.file,
        lines: out.lines.into_iter().map(|l| (l.id, l.text)).collect(),
        wrong: out.wrong,
        checks: out.checks.iter().map(|c| (c.describe(), c.passes())).collect(),
        seconds: out.seconds,
    })
}

/// Every backend this build hosts.
#[must_use]
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
        tune: None,
    });
    #[cfg(feature = "vulkan")]
    out.push(Backend {
        name: "vulkan",
        registry: engine_vulkan::chip::registry,
        shelf: &engine_vulkan::chip::SHELF,
        probe: engine_vulkan::chip::probe,
        resolve: engine_vulkan::chip::resolve_facts,
        tune: None,
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
