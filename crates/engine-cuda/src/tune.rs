//! `pie tune` on CUDA: every searchable decision is searched end to end on
//! served models (`profile::search`); there is no kernel bench.

use std::path::{Path, PathBuf};
use std::time::Instant;

use profile::search::{self, Bench, CONTEXT, Check, LANES, Line, PROMPT, STEPS, Serving};
use profile::{Facts, Profile};

/// Decisions the search leaves alone on CUDA: a collective's price means
/// nothing to a single-GPU load.
pub const SEARCH_SKIPS: &[&str] = &["compiler.cost.collective_us"];

pub struct Outcome {
    pub device: String,
    pub file: String,
    pub lines: Vec<Line>,
    /// The end-to-end runs, today's profile against the tuned one.
    pub checks: Vec<Check>,
    pub changed: bool,
    pub seconds: f64,
}

/// Serve `artifact` under `chip`: a prompt, then steps at each lane count.
fn serve(artifact: &Path, deployment: &str, chip: Profile) -> Result<Serving, String> {
    use poem::{Platform, Request};
    let slots = *LANES.iter().max().expect("lanes") as u32;
    let row = poem_compiler::catalog::deployment(deployment)
        .ok_or_else(|| format!("no catalog row `{deployment}`"))?;
    let trace = row.trace(Platform::Cuda);
    let source = ztensor_compat::index(artifact).map_err(|e| e.to_string())?;
    let contract =
        poem::import::own_contract(&source, &trace.params, row.deploy.tp, Platform::Cuda)
            .map_err(|e| e.to_string())?;
    drop(source);
    let target =
        checkpoint::plan::StorageTarget::for_backend(checkpoint::types::BackendKind::Cuda, 0, 1);
    let prospect =
        crate::weights::prospect(&trace, &contract, artifact, target).map_err(|e| e.to_string())?;
    let residency = crate::experts::Plan::cut(
        &prospect.ranking,
        crate::experts::Budgets {
            device: None,
            host: None,
        },
    )
    .map_err(|e| e.to_string())?;
    let mut shell = crate::Shell::load(crate::Boot {
        voxels: None,
        deferred_tier: false,
        trace: trace.clone(),
        contract: &contract,
        checkpoint: artifact,
        budget: poem_compiler::Budget::new(slots, CONTEXT),
        patches: patches(&trace, slots),
        page_size: 16,
        context: CONTEXT,
        slots,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: crate::Knobs {
            chip: Some(chip.clone()),
            ..crate::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency,
        world: crate::World::default(),
        comm: core::ptr::null_mut(),
    })
    .map_err(|e| e.to_string())?;
    let token = |i: u32| 100 + (i * 7919) % 5000;
    let prompt: Vec<u32> = (0..PROMPT).map(token).collect();
    let fire =
        |shell: &mut crate::Shell, lanes: &[(u32, &[u32])]| -> Result<(f64, Vec<f32>), String> {
            let lanes: Vec<crate::Lane> = lanes
                .iter()
                .map(|&(slot, tokens)| crate::Lane {
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
            // The first step of a lane count compiles or records.
            if step > 0 {
                times.push(ms);
            }
        }
        decode.push(times);
    }
    drop(shell);
    Ok(Serving {
        prompt: prefill,
        steps: decode,
        readout,
        read: chip.read(),
        refused: None,
    })
}

/// A fingerprint of the fused trace and compiled plan a load under `chip`
/// runs, built as `serve/load.rs` builds them.
fn plan_of(deployment: &str, chip: &Profile, sms: u32) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    let row = poem_compiler::catalog::deployment(deployment)?;
    let engine = crate::chip::Engine::of(chip);
    let grouped: &[&str] = if engine.grouped { &crate::GROUPED } else { &[] };
    let profile = poem_compiler::DeviceProfile::of(chip, sms, &crate::EXCLUSIVE, grouped);
    let trace = poem_compiler::fuse::fuse(
        row.trace(poem::Platform::Cuda),
        &poem_compiler::fuse::flags::kernels(chip),
    );
    let slots = *LANES.iter().max().expect("lanes") as u32;
    let mut budget = poem_compiler::Budget::new(slots, CONTEXT);
    budget.buckets = crate::api::lattice(Vec::new(), budget.max_tokens, engine.lattice_steps);
    let mut budgets = poem_compiler::Budgets::of(budget);
    if let Some(ladder) = patches(&trace, slots) {
        budgets = budgets.with_patches(ladder);
    }
    let compiled = poem_compiler::compile_axes(&trace, &budgets, &profile).ok()?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{trace:?}{compiled:?}").hash(&mut h);
    Some(h.finish())
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

struct Cuda<'a> {
    models: &'a [(PathBuf, String)],
    device: &'a str,
    base: &'a Profile,
    facts: &'a Facts,
    sms: u32,
}

impl Cuda<'_> {
    fn profile_of(file: &str) -> Result<Profile, String> {
        crate::chip::registry()
            .load_user(file, "tuned")
            .map_err(|e| e.to_string())
    }
}

impl Bench for Cuda<'_> {
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
            &["cc"],
        )
    }

    fn serve(&mut self, model: usize, file: &str) -> Result<Serving, String> {
        let (artifact, deployment) = &self.models[model];
        serve(artifact, deployment, Self::profile_of(file)?)
    }

    fn plan(&self, model: usize, file: &str) -> Option<u64> {
        plan_of(
            &self.models[model].1,
            &Self::profile_of(file).ok()?,
            self.sms,
        )
    }
}

/// Run the tune on GPU 0: search, check, and assemble the file. `say` hears
/// progress.
pub fn run(
    profiles: Option<&Path>,
    models: &[(PathBuf, String)],
    say: &mut dyn FnMut(&str),
) -> Result<Outcome, String> {
    let began = Instant::now();
    let (name, facts) = crate::chip::probe(0).ok_or("no CUDA device answers")?;
    let sms = facts
        .get("sms")
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(1);
    let (base, report) = crate::chip::resolve(&name, &facts, profiles);
    say(&format!("{name}: {sms} SMs; starting from {}", report.used));
    let knobs = search::knobs(&base, SEARCH_SKIPS, "");
    let mut bench = Cuda {
        models,
        device: &name,
        base: &base,
        facts: &facts,
        sms,
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
    let mut tuned_ids: Vec<String> = Vec::new();
    for l in lines.iter().filter(|l| !l.text.starts_with("not searched")) {
        let id = format!("\"{}\"", l.id);
        if !tuned_ids.contains(&id) {
            tuned_ids.push(id);
        }
    }
    let checked = if unchanged {
        "nothing changed".to_string()
    } else if checks.is_empty() {
        "no end-to-end run".to_string()
    } else {
        format!(
            "end to end: {}",
            checks
                .iter()
                .map(Check::describe)
                .collect::<Vec<_>>()
                .join("; ")
        )
    };
    let seconds = began.elapsed().as_secs_f64();
    let provenance = [
        ("source", "\"pie tune on this machine\"".to_string()),
        ("measured", format!("[{}]", tuned_ids.join(", "))),
        ("from", format!("\"{}\"", report.used)),
        ("pie", format!("\"{}\"", env!("CARGO_PKG_VERSION"))),
        ("sms", sms.to_string()),
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
        &["cc"],
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

    #[test]
    fn every_search_candidate_reads_against_the_registry() {
        let facts = crate::chip::facts(8, 9, 142);
        let (base, _) = crate::chip::resolve("NVIDIA L40S", &facts, None);
        let registry = crate::chip::registry();
        let knobs = search::knobs(&base, SEARCH_SKIPS, "");
        let ids: Vec<&str> = knobs.iter().map(|k| k.id).collect();
        for want in [
            "kernel.moe.grouped",
            "engine.graphs",
            "engine.recording",
            "fuse.matmul_bias",
        ] {
            assert!(ids.contains(&want), "{want} not searched: {ids:?}");
        }
        for k in &knobs {
            for (label, _) in &k.candidates {
                let entries = vec![(k.id.to_string(), label.clone())];
                search::assemble(
                    registry,
                    &base,
                    "NVIDIA L40S",
                    &facts,
                    &search::render_overrides(&entries),
                    &[],
                    &["cc"],
                )
                .unwrap_or_else(|e| panic!("{} = {label}: {e}", k.id));
            }
        }
    }
}
