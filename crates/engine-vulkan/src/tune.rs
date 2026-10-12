//! `pie tune` on Vulkan: `kernel.qmm` is fitted to a kernel bench (`qmm`),
//! then it and the other decisions are searched end to end on served models.

mod qmm;

pub use qmm::Depth;

use std::path::{Path, PathBuf};
use std::time::Instant;

use profile::search::{self, Bench, CONTEXT, Check, LANES, Line, PROMPT, STEPS, Serving};
use profile::{Facts, Profile};

/// The kernel bench fits `kernel.qmm`; one device with one queue has no
/// collectives or side streams.
const SEARCH_SKIPS: &[&str] = &[
    "kernel.qmm",
    "compiler.cost.collective_us",
    "compiler.side_streams",
];

/// The facts a tuned file assumes: what a kernel's requirement reads.
const ASSUME: &[&str] = &["vendor", "subgroup", "coopmat", "cores"];

/// What a tuned overlay on `base` may assume: its target, and the facts
/// [`ASSUME`] names as this device reports them.
#[cfg(any(test, feature = "vulkan"))]
fn target(base: &Profile, facts: &Facts) -> profile::Target {
    let mut target = base.target().clone();
    for name in ASSUME {
        let i = kernels_vulkan::tuning::FACTS
            .iter()
            .position(|v| v.name == *name)
            .expect("a Vulkan fact");
        if let Some(v) = facts.get(name)
            && !target.assumes.iter().any(|&(j, _)| j == i)
        {
            target.assumes.push((i, v));
        }
    }
    target
}

pub struct Outcome {
    pub device: String,
    /// The complete profile, ready for the user's shelf.
    pub file: String,
    pub lines: Vec<Line>,
    /// Kernel candidates that answered wrongly, and where.
    pub wrong: Vec<String>,
    pub checks: Vec<Check>,
    pub changed: bool,
    pub seconds: f64,
}

fn patches(trace: &poem_ir::Trace, slots: u32) -> Option<poem_compiler::PatchLadder> {
    crate::api::patch_ladder(
        trace,
        &engine::load::Budgets {
            max_lanes: slots,
            max_tokens: CONTEXT,
            ..engine::load::Budgets::default()
        },
    )
}

/// Serve `artifact` with `file` as the user's profile for `device`.
fn serve(artifact: &Path, deployment: &str, device: &str, file: &str) -> Result<Serving, String> {
    use poem::{Platform, Request};
    let slots = *LANES.iter().max().expect("lanes") as u32;
    let dir = std::env::temp_dir().join(format!("pie-tune-vulkan-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = crate::chip::SHELF.user_path(&dir, device);
    std::fs::create_dir_all(path.parent().expect("a parent")).map_err(|e| e.to_string())?;
    std::fs::write(&path, file).map_err(|e| e.to_string())?;
    let row = poem_compiler::catalog::deployment(deployment)
        .ok_or_else(|| format!("no catalog row `{deployment}`"))?;
    let trace = row.trace(Platform::Vulkan);
    let source = ztensor_compat::index(artifact).map_err(|e| e.to_string())?;
    let contract =
        poem::import::own_contract(&source, &trace.params, row.deploy.tp, Platform::Vulkan)
            .map_err(|e| e.to_string())?;
    drop(source);
    let boot = crate::api::DeviceBoot {
        profiles: Some(dir.clone()),
        ..crate::api::DeviceBoot::default()
    };
    let mut shell = crate::serve::Shell::load(crate::serve::Boot {
        trace: trace.clone(),
        contract: &contract,
        checkpoint: artifact,
        budget: poem_compiler::Budget::new(slots, CONTEXT),
        patches: patches(&trace, slots),
        voxels: None,
        profile: None,
        page_size: 16,
        context: CONTEXT,
        slots,
        runahead: engine::runahead::Runahead::F1,
        device: &boot,
        residency: engine::load::Residency::default(),
    })
    .map_err(|e| e.to_string())?;
    let token = |i: u32| 100 + (i * 7919) % 5000;
    let prompt: Vec<u32> = (0..PROMPT).map(token).collect();
    let fire = |shell: &mut crate::serve::Shell,
                lanes: &[(u32, &[u32])]|
     -> Result<(f64, Vec<f32>), String> {
        let lanes: Vec<crate::serve::Lane> = lanes
            .iter()
            .map(|&(slot, tokens)| crate::serve::Lane {
                slot,
                word: trace.facts.word(&Request::new(tokens.len() as u32, false)),
                tokens,
            })
            .collect();
        let began = Instant::now();
        let out = shell.fire(&lanes).map_err(|e| e.to_string())?;
        Ok((
            began.elapsed().as_secs_f64() * 1e3,
            out.into_iter().next().unwrap_or_default(),
        ))
    };
    let mut prefill = Vec::new();
    let mut readout = Vec::new();
    for round in 0..4 {
        shell.open(0).map_err(|e| e.to_string())?;
        let (ms, r) = fire(&mut shell, &[(0, &prompt)])?;
        if round > 0 {
            prefill.push(ms);
        }
        readout = r;
    }
    let history: Vec<u32> = (0..32).map(token).collect();
    for slot in 0..slots {
        shell.open(slot).map_err(|e| e.to_string())?;
        fire(&mut shell, &[(slot, &history)])?;
    }
    let mut decode = Vec::new();
    for &lanes in &LANES {
        let mut times = Vec::new();
        for step in 0..=STEPS {
            let next: Vec<[u32; 1]> = (0..lanes)
                .map(|l| [token((step * lanes + l) as u32)])
                .collect();
            let batch: Vec<(u32, &[u32])> = next
                .iter()
                .enumerate()
                .map(|(l, t)| (l as u32, &t[..]))
                .collect();
            let (ms, _) = fire(&mut shell, &batch)?;
            // The first step at a lane count compiles or records.
            if step > 0 {
                times.push(ms);
            }
        }
        decode.push(times);
    }
    let read = shell.tuning().profile().read();
    drop(shell);
    let _ = std::fs::remove_dir_all(&dir);
    Ok(Serving {
        prompt: prefill,
        steps: decode,
        readout,
        read,
        refused: None,
    })
}

/// A fingerprint of the fused trace and plan a load under `text` runs, as
/// `Shell::load` builds them.
fn plan_of(deployment: &str, text: &str, cores: u32) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    let chip = crate::chip::registry().load_user(text, "plan").ok()?;
    let row = poem_compiler::catalog::deployment(deployment)?;
    let trace = poem_compiler::fuse::fuse(
        row.trace(poem::Platform::Vulkan),
        &poem_compiler::fuse::flags::kernels(&chip),
    );
    let slots = *LANES.iter().max().expect("lanes") as u32;
    let mut budgets = poem_compiler::Budgets::of(poem_compiler::Budget::new(slots, CONTEXT));
    if let Some(ladder) = patches(&trace, slots) {
        budgets = budgets.with_patches(ladder);
    }
    let compiled = poem_compiler::compile_axes(
        &trace,
        &budgets,
        &poem_compiler::DeviceProfile::of(&chip, cores, &[], &[]),
    )
    .ok()?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{trace:?}{compiled:?}").hash(&mut h);
    Some(h.finish())
}

struct Vulkan<'a> {
    models: &'a [(PathBuf, String)],
    device: &'a str,
    base: &'a Profile,
    facts: &'a Facts,
    cores: u32,
}

impl Bench for Vulkan<'_> {
    fn models(&self) -> Vec<String> {
        self.models.iter().map(|(_, d)| d.clone()).collect()
    }

    fn file(&self, entries: &[(String, String)]) -> Result<String, String> {
        search::assemble(
            crate::chip::registry(),
            self.base,
            self.device,
            self.facts,
            &search::render_overrides(entries),
            &[],
            ASSUME,
        )
    }

    fn serve(&mut self, model: usize, file: &str) -> Result<Serving, String> {
        let (artifact, deployment) = &self.models[model];
        serve(artifact, deployment, self.device, file)
    }

    fn plan(&self, model: usize, file: &str) -> Option<u64> {
        plan_of(&self.models[model].1, file, self.cores)
    }
}

/// Measure `kernel.qmm` and fit its ladder; the measurements are kept under
/// `cache`, where `refit` reads them back.
fn bench_qmm(
    depth: Depth,
    name: &str,
    base: &Profile,
    cores: u32,
    cache: &Path,
    refit: bool,
    say: &mut dyn FnMut(&str),
) -> Result<(search::Entries, Vec<Line>, Vec<String>), String> {
    let grid = qmm::Grid::of(depth);
    let kept = cache
        .join(profile::id_of(name))
        .join(format!("vulkan-qmm-{}.txt", depth.word()));
    let measured = if refit {
        let text = std::fs::read_to_string(&kept)
            .map_err(|e| format!("no measurements at {}: {e}", kept.display()))?;
        say(&format!("refitting the measurements in {}", kept.display()));
        let all = qmm::candidates(base);
        let ran: Vec<_> = text
            .lines()
            .filter_map(|l| l.strip_prefix("candidate "))
            .filter_map(|b| all.iter().find(|q| qmm::body(q) == b).copied())
            .collect();
        qmm::Measured::from_text(&text, &ran)?
    } else {
        let m = measure(base, &grid, say)?;
        if let Some(dir) = kept.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(&kept, m.to_text(name)).map_err(|e| e.to_string())?;
        say(&format!("measurements kept in {}", kept.display()));
        m
    };
    let (fitted, lines) = qmm::fit(&measured, &grid, cores)?;
    let wrong = measured
        .wrong
        .iter()
        .map(|(body, n, first)| format!("{body}: wrong at {n} points; first at {first}"))
        .collect();
    Ok((fitted, lines, wrong))
}

#[cfg(feature = "vulkan")]
fn measure(
    base: &Profile,
    grid: &qmm::Grid,
    say: &mut dyn FnMut(&str),
) -> Result<qmm::Measured, String> {
    let device = crate::device::Context::bind(
        &crate::api::DeviceBoot::default(),
        engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
    )
    .map_err(|e| e.to_string())?;
    qmm::measure(&device, base, grid, say)
}

#[cfg(not(feature = "vulkan"))]
fn measure(_: &Profile, _: &qmm::Grid, _: &mut dyn FnMut(&str)) -> Result<qmm::Measured, String> {
    Err("this build carries no Vulkan device code".to_string())
}

/// Tune the device a load opens. `say` hears progress.
pub fn run(
    depth: Depth,
    profiles: Option<&Path>,
    cache: &Path,
    refit: bool,
    models: &[(PathBuf, String)],
    say: &mut dyn FnMut(&str),
) -> Result<Outcome, String> {
    let began = Instant::now();
    let (name, facts) = crate::chip::probe().ok_or("no Vulkan device answers")?;
    let cores = facts
        .get("cores")
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(crate::chip::UNREPORTED_CORES);
    let (base, report) = crate::chip::resolve_facts(&name, &facts, profiles);
    say(&format!(
        "{name}: {}; starting from {}",
        facts.describe().join(", "),
        report.used
    ));
    let (fitted, mut lines, wrong) = bench_qmm(depth, &name, &base, cores, cache, refit, say)?;
    // The fitted ladder must also win end to end to be kept.
    let mut knobs: Vec<search::Knob> = Vec::new();
    for (id, value) in &fitted {
        let (spec, rungs, _) = base
            .entries()
            .find(|(s, ..)| s.id == id.as_str())
            .expect("a Vulkan decision");
        let current = profile::render::value_toml(spec, rungs);
        if current.replace(char::is_whitespace, "") != value.replace(char::is_whitespace, "") {
            knobs.push(search::Knob {
                id: spec.id,
                current,
                candidates: vec![(value.clone(), String::new())],
                halfway: false,
                plan_only: false,
            });
        }
    }
    knobs.extend(search::knobs(&base, SEARCH_SKIPS, ""));
    let mut bench = Vulkan {
        models,
        device: &name,
        base: &base,
        facts: &facts,
        cores,
    };
    let mut chosen = Vec::new();
    if models.is_empty() {
        chosen = fitted;
        for k in knobs.iter().filter(|k| k.id != "kernel.qmm") {
            lines.push(Line {
                id: k.id,
                text: "not searched: no model to serve (import one, or pass --check)".to_string(),
            });
        }
    } else {
        let names: Vec<&str> = models.iter().map(|(_, d)| d.as_str()).collect();
        say(&format!(
            "searching {} decisions end to end on {}",
            knobs.len(),
            names.join(", ")
        ));
        let (found, searched) = search::search(&mut bench, chosen, &knobs, "", say)?;
        chosen = found;
        lines.extend(searched);
    }
    let tuned = search::render_overrides(&chosen);
    let today = bench.file(&[])?;
    let draft = bench.file(&chosen)?;
    let mut checks = Vec::new();
    let unchanged = draft == today;
    if unchanged && !models.is_empty() {
        say("end to end: nothing changed, so nothing to compare");
    }
    for (at, (_, deployment)) in models.iter().enumerate().filter(|_| !unchanged) {
        say(&format!(
            "end to end: serving {deployment} under today's profile and the tuned one"
        ));
        match search::check(&mut bench, at, &today, &draft) {
            Ok(c) => {
                say(&format!("  {}", c.describe()));
                checks.push(c);
            }
            Err(why) => say(&format!("  {deployment} did not serve: {why}")),
        }
    }
    // The kernel bench and the search both report `kernel.qmm`.
    let mut tuned_ids: Vec<String> = Vec::new();
    for l in lines.iter().filter(|l| !l.text.starts_with("not searched")) {
        let id = format!("\"{}\"", l.id);
        if !tuned_ids.contains(&id) {
            tuned_ids.push(id);
        }
    }
    let checked = if unchanged {
        "kernels against the f64 reference; nothing changed".to_string()
    } else if checks.is_empty() {
        "kernels against the f64 reference; no end-to-end run".to_string()
    } else {
        format!(
            "kernels against the f64 reference; end to end: {}",
            checks
                .iter()
                .map(Check::describe)
                .collect::<Vec<_>>()
                .join("; ")
        )
    };
    let seconds = began.elapsed().as_secs_f64();
    let provenance = [
        (
            "source",
            format!("\"pie tune --{} on this machine\"", depth.word()),
        ),
        ("measured", format!("[{}]", tuned_ids.join(", "))),
        ("from", format!("\"{}\"", report.used)),
        ("pie", format!("\"{}\"", env!("CARGO_PKG_VERSION"))),
        ("seconds", format!("{seconds:.0}")),
        ("checked", format!("\"{}\"", checked.replace('"', "'"))),
    ];
    let file = search::assemble(
        crate::chip::registry(),
        &base,
        &name,
        &facts,
        &tuned,
        &provenance,
        ASSUME,
    )?;
    Ok(Outcome {
        device: name,
        file,
        lines,
        wrong,
        checks,
        changed: !unchanged,
        seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_search_candidate_reads_against_the_registry() {
        for coopmat in [false, true] {
            let facts = profile::Facts::of(crate::chip::registry())
                .word("vendor", "nvidia")
                .int("subgroup", 32)
                .int("cores", 142)
                .flag("coopmat", coopmat);
            let (base, _) = crate::chip::resolve_facts("Test GPU", &facts, None);
            let knobs = search::knobs(&base, SEARCH_SKIPS, "");
            let ids: Vec<&str> = knobs.iter().map(|k| k.id).collect();
            assert!(ids.contains(&"kernel.sdpa.decode.threads"), "{ids:?}");
            assert!(ids.contains(&"engine.submit.split_rows"), "{ids:?}");
            assert!(!ids.contains(&"kernel.qmm"), "{ids:?}");
            for k in &knobs {
                for (label, _) in &k.candidates {
                    let entries = vec![(k.id.to_string(), label.clone())];
                    let got = search::assemble(
                        crate::chip::registry(),
                        &base,
                        "Test GPU",
                        &facts,
                        &search::render_overrides(&entries),
                        &[],
                        ASSUME,
                    );
                    if let Err(why) = got {
                        assert!(
                            !coopmat && why.contains("coopmat"),
                            "{} = {label}: {why}",
                            k.id
                        );
                    }
                }
            }
        }
    }
}
