//! `pie tune` on Vulkan: every decision this backend does not mark `never`
//! is searched end to end on served models (`profile::search`), starting
//! from the profile the device resolves to (a user file, a named one, or the
//! one its facts derive), and the result is written as a complete profile
//! for the user's shelf. `.wiki/pie-tune.md` holds the design.

use std::path::{Path, PathBuf};
use std::time::Instant;

use profile::search::{self, Bench, CONTEXT, Check, LANES, Line, PROMPT, STEPS, Serving};
use profile::{Facts, Profile};

/// Decisions the search leaves alone on Vulkan: one device, one queue.
pub const SEARCH_SKIPS: &[&str] = &["compiler.cost.collective_us", "compiler.side_streams"];

/// The facts a tuned file assumes where the profile it starts from assumes
/// nothing: what a kernel's requirement reads.
const ASSUME: &[&str] = &["vendor", "subgroup", "coopmat"];

/// What a run produced.
pub struct Outcome {
    pub device: String,
    /// The complete profile, ready for the user's shelf.
    pub file: String,
    pub lines: Vec<Line>,
    /// The end-to-end runs, today's profile against the tuned one.
    pub checks: Vec<Check>,
    /// Whether the tuned file differs from today's at all.
    pub changed: bool,
    pub seconds: f64,
}

/// The Vulkan-stamped artifacts under `store`, smallest first: what an
/// end-to-end check can serve.
#[must_use]
pub fn artifacts(store: &Path) -> Vec<(PathBuf, String)> {
    let mut out: Vec<(u64, PathBuf, String)> = Vec::new();
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
            if stamp.backend == "vulkan" {
                let bytes = file.metadata().map_or(u64::MAX, |m| m.len());
                out.push((bytes, path, stamp.deployment));
            }
        }
    }
    out.sort_by_key(|(bytes, ..)| *bytes);
    out.into_iter().map(|(_, p, d)| (p, d)).collect()
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

/// Serve `artifact` under the complete profile `file`, which the load
/// resolves as the user's file for `device`.
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
    let contract = poem::import::own_contract(&source, &trace.params, row.deploy.tp, Platform::Vulkan)
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
        profile: None,
        page_size: 16,
        context: CONTEXT,
        slots,
        pages: slots * CONTEXT / 16,
        runahead: engine::runahead::Runahead::F1,
        device: &boot,
        residency: engine::load::Residency::default(),
    })
    .map_err(|e| e.to_string())?;
    let token = |i: u32| 100 + (i * 7919) % 5000;
    let prompt: Vec<u32> = (0..PROMPT).map(token).collect();
    let fire = |shell: &mut crate::serve::Shell, lanes: &[(u32, &[u32])]| -> Result<(f64, Vec<f32>), String> {
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
        Ok((began.elapsed().as_secs_f64() * 1e3, out.into_iter().next().unwrap_or_default()))
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
            let next: Vec<[u32; 1]> = (0..lanes).map(|l| [token((step * lanes + l) as u32)]).collect();
            let batch: Vec<(u32, &[u32])> = next.iter().enumerate().map(|(l, t)| (l as u32, &t[..])).collect();
            let (ms, _) = fire(&mut shell, &batch)?;
            // The first step of a lane count compiles or records.
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

/// A fingerprint of the fused trace and compiled plan a load under `text`
/// runs, built as `Shell::load` builds them.
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

/// Vulkan's bench: the check models, served on the device a load opens.
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

/// Run the tune on the device a load opens: search, check, and assemble
/// the file. `say` hears progress.
pub fn run(
    profiles: Option<&Path>,
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
    let knobs = search::knobs(&base, SEARCH_SKIPS, "");
    let mut bench = Vulkan {
        models,
        device: &name,
        base: &base,
        facts: &facts,
        cores,
    };
    let mut lines = Vec::new();
    let mut chosen = Vec::new();
    if models.is_empty() {
        for k in &knobs {
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
        lines = searched;
    }
    let tuned = search::render_overrides(&chosen);
    let today = bench.file(&[])?;
    let draft = bench.file(&chosen)?;
    let mut checks = Vec::new();
    // A file that changes nothing serves as today's does.
    let unchanged = draft == today;
    if unchanged && !models.is_empty() {
        say("end to end: nothing changed, so nothing to compare");
    }
    for (at, (_, deployment)) in models.iter().enumerate().filter(|_| !unchanged) {
        say(&format!("end to end: serving {deployment} under today's profile and the tuned one"));
        match search::check(&mut bench, at, &today, &draft) {
            Ok(c) => {
                say(&format!("  {}", c.describe()));
                checks.push(c);
            }
            Err(why) => say(&format!("  {deployment} did not serve: {why}")),
        }
    }
    let tuned_ids: Vec<String> = lines
        .iter()
        .filter(|l| !l.text.starts_with("not searched"))
        .map(|l| format!("\"{}\"", l.id))
        .collect();
    let checked = if unchanged {
        "nothing changed".to_string()
    } else if checks.is_empty() {
        "no end-to-end run".to_string()
    } else {
        format!(
            "end to end: {}",
            checks.iter().map(Check::describe).collect::<Vec<_>>().join("; ")
        )
    };
    let seconds = began.elapsed().as_secs_f64();
    let provenance = [
        ("source", "\"pie tune on this machine\"".to_string()),
        ("measured", "true".to_string()),
        ("tuned", format!("[{}]", tuned_ids.join(", "))),
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
        checks,
        changed: !unchanged,
        seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every candidate the search offers on a derived profile reads as a
    /// file this device takes, on a device with and without coopmat.
    #[test]
    fn every_search_candidate_reads_against_the_registry() {
        for coopmat in [false, true] {
            let mut facts = profile::Facts::of(crate::chip::registry())
                .word("vendor", "nvidia")
                .int("subgroup", 32)
                .int("cores", 142);
            facts = facts.flag("coopmat", coopmat);
            let (base, _) = crate::chip::resolve_facts("Test GPU", &facts, None);
            let knobs = search::knobs(&base, SEARCH_SKIPS, "");
            let ids: Vec<&str> = knobs.iter().map(|k| k.id).collect();
            assert!(ids.contains(&"kernel.qmm.min_rows"), "{ids:?}");
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
                    // A coopmat kernel on a device without it is refused,
                    // which the search records and passes over.
                    if let Err(why) = got {
                        assert!(!coopmat && why.contains("coopmat"), "{} = {label}: {why}", k.id);
                    }
                }
            }
        }
    }
}
