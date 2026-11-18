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

/// `pie tune cuda --rates`: GPU 0's roofline constants measured at
/// `engine::rates`' shapes ([`crate::rates::measure`]), laid over the
/// profile it runs.
pub fn rates(
    profiles: Option<&Path>,
    say: &mut dyn FnMut(&str),
) -> Result<engine::rates::Rated, String> {
    let began = Instant::now();
    let (name, facts) = crate::chip::probe(0).ok_or("no CUDA device answers")?;
    let (base, report) = crate::chip::resolve(&name, &facts, profiles);
    say(&format!("{name}: starting from {}", report.used));
    let cache = profiles
        .and_then(Path::parent)
        .map(|home| home.join("cache"));
    let measured = crate::rates::measure(0, &base, cache.as_deref())?;
    let r = measured.rates;
    let curve = |c: &Option<Vec<(u32, f64)>>| {
        c.iter()
            .flatten()
            .map(|(x, t)| format!("{x} {t:.1}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    say(&format!(
        "[rates]: matmul TFLOP/s by rows {}; attention by keys {} (at {} keys: {}); per-row ops \
         {:.0} GB/s, scan {:.0} G elements/s, {:.0} GB/s, launch {} ns",
        r.gemm_tflops_by_rows
            .iter()
            .flatten()
            .map(|(w, c)| format!("w {w}: {}", curve(&Some(c.clone()))))
            .collect::<Vec<_>>()
            .join("; "),
        curve(&r.attn_tflops_by_keys),
        engine::rates::ATTN_KEYS[engine::rates::ATTN_KEYS.len() - 1],
        measured
            .attn_by_dim
            .iter()
            .map(|(d, t)| format!("d {d} {t:.1}"))
            .collect::<Vec<_>>()
            .join(", "),
        r.elementwise_gbps.unwrap_or(0.0),
        r.scan_gelems.unwrap_or(0.0),
        r.hbm_gbps.unwrap_or(0.0),
        r.launch_ns.unwrap_or(0),
    ));
    let tuned = engine::rates::tuned(&r);
    let provenance = crate::rates::provenance(&name);
    let file = search::assemble(
        crate::chip::registry(),
        &base,
        &name,
        &facts,
        &tuned,
        std::slice::from_ref(&provenance),
        &[crate::chip::ARCH],
    )?;
    let block = format!(
        "{tuned}\n[provenance]\n{} = {}\n",
        provenance.0, provenance.1
    );
    Ok(engine::rates::Rated {
        device: name,
        file,
        block,
        seconds: began.elapsed().as_secs_f64(),
    })
}

/// How a load is shaped: its slots and compiled budget, its patch ladder,
/// its pages and context.
pub(crate) struct Shape {
    pub(crate) slots: u32,
    pub(crate) budget: poem_compiler::Budget,
    pub(crate) patches: Option<poem_compiler::PatchLadder>,
    pub(crate) page_size: u32,
    pub(crate) context: u32,
}

/// Load `artifact` under `knobs` shaped by `shape` (from the trace).
pub(crate) fn load(
    artifact: &Path,
    deployment: &str,
    knobs: crate::Knobs,
    shape: impl FnOnce(&poem_ir::Trace) -> Shape,
) -> Result<(crate::Shell, poem_ir::Trace), String> {
    use poem::Platform;
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
    let residency = crate::experts::Plan::cut(&prospect.ranking);
    let Shape {
        slots,
        budget,
        patches,
        page_size,
        context,
    } = shape(&trace);
    let shell = crate::Shell::load(crate::Boot {
        voxels: None,
        trace: trace.clone(),
        contract: &contract,
        checkpoint: artifact,
        budget,
        patches,
        page_size,
        context,
        slots,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs,
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency,
        world: crate::World::default(),
        expert: crate::World::default(),
        comm: core::ptr::null_mut(),
    })
    .map_err(|e| e.to_string())?;
    Ok((shell, trace))
}

/// Serve `artifact` under `chip`: a prompt, then steps at each lane count.
fn serve(
    artifact: &Path,
    deployment: &str,
    chip: Profile,
    profiles: Option<&Path>,
) -> Result<Serving, String> {
    use poem::Request;
    let slots = *LANES.iter().max().expect("lanes") as u32;
    let knobs = crate::Knobs {
        chip: Some(chip.clone()),
        profiles: profiles.map(Path::to_path_buf),
        ..crate::Knobs::default()
    };
    let (mut shell, trace) = load(artifact, deployment, knobs, |trace| Shape {
        slots,
        budget: poem_compiler::Budget::new(slots, CONTEXT),
        patches: patches(trace, slots),
        page_size: 16,
        context: CONTEXT,
    })?;
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
        poem_compiler::fuse::flags::on(chip, &crate::FUSIONS),
    );
    #[cfg(not(feature = "rocm"))]
    let trace = {
        let tuning = kernels_cuda::tuning::Tuning::of(chip);
        crate::narrow::gdn_out_bf16(
            crate::grouped_decode::rewrite_with(
                trace,
                tuning.decode_grouped_prefill_from(),
                tuning.decode_grouped_prefill_lse(),
                tuning.decode_grouped_prefill_window(),
                kernels_cuda::attn::decode_gqa::Gate::of(tuning),
            ),
            || tuning.ssm_gdn_out_bf16(),
        )
    };
    let slots = *LANES.iter().max().expect("lanes") as u32;
    let mut budget = poem_compiler::Budget::new(slots, CONTEXT);
    budget.buckets = crate::api::lattice(
        Vec::new(),
        budget.max_tokens,
        engine.lattice_steps,
        engine.prefill_lattice_from,
        engine.prefill_lattice_steps,
    );
    let mut budgets = poem_compiler::Budgets::of(budget);
    if let Some(ladder) = patches(&trace, slots) {
        budgets = budgets.with_patches(ladder);
    }
    let compiled = poem_compiler::compile_axes(&trace, &budgets, &profile).ok()?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{trace:?}{compiled:?}").hash(&mut h);
    Some(h.finish())
}

pub(crate) fn patches(trace: &poem_ir::Trace, slots: u32) -> Option<poem_compiler::PatchLadder> {
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
    profiles: Option<&'a Path>,
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
            &[crate::chip::ARCH],
        )
    }

    fn serve(&mut self, model: usize, file: &str) -> Result<Serving, String> {
        let (artifact, deployment) = &self.models[model];
        serve(artifact, deployment, Self::profile_of(file)?, self.profiles)
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
        profiles,
    };
    let mut lines = Vec::new();
    let mut chosen = Vec::new();
    if models.is_empty() {
        for k in &knobs {
            lines.push(Line {
                id: k.id,
                text: "not searched: no model to serve (import one, or pass --check)".to_string(),
                measured: false,
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
    // Only what the run measured: a decision no check model reads, or one
    // whose bar the noise pushed past the agreement tolerance, keeps its
    // line in the report but is not called measured.
    let mut tuned_ids: Vec<String> = Vec::new();
    for l in lines.iter().filter(|l| l.measured) {
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
        &[crate::chip::ARCH],
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

/// Contract A at eight decode lanes: each lane's greedy decode served in
/// one fire of eight (the few-rows products at eight rows) against the same
/// lane served alone (one row, the gemv), token for token; a step where the
/// two pick apart is admitted only as a near-tie (the drift over the full
/// row diffuse, and the decision's gap within twice its largest per-logit
/// move). Run by hand on a served model:
/// `PIE_CONTRACT_ARTIFACT=<.zt> PIE_CONTRACT_DEPLOYMENT=<row>`.
#[cfg(all(test, feature = "cuda"))]
mod eight_lanes {
    use super::*;

    const LANES: u32 = 8;
    const PROMPT: u32 = 128;
    const STEPS: usize = 32;

    fn argmax(row: &[f32]) -> usize {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(at, _)| at)
    }

    /// The decision's top-1 less its top-2.
    fn gap(row: &[f32]) -> f32 {
        let mut top = [f32::NEG_INFINITY; 2];
        for &v in row {
            if v > top[0] {
                top = [v, top[0]];
            } else if v > top[1] {
                top[1] = v;
            }
        }
        top[0] - top[1]
    }

    #[test]
    #[ignore = "a contract check run by hand on a served model"]
    fn eight_lanes_decode_as_each_lane_alone_up_to_near_ties() {
        let (Some(artifact), Some(deployment)) = (
            std::env::var_os("PIE_CONTRACT_ARTIFACT"),
            std::env::var("PIE_CONTRACT_DEPLOYMENT").ok(),
        ) else {
            eprintln!("skipping: PIE_CONTRACT_ARTIFACT / PIE_CONTRACT_DEPLOYMENT unset");
            return;
        };
        let slots = 2 * LANES;
        let (mut shell, trace) = load(
            Path::new(&artifact),
            &deployment,
            crate::Knobs::default(),
            |trace| Shape {
                slots,
                budget: poem_compiler::Budget::new(LANES, 0),
                patches: patches(trace, LANES),
                page_size: 16,
                context: 1024,
            },
        )
        .expect("the model loads");
        let word = |n: usize| trace.facts.word(&poem::Request::new(n as u32, false));
        let prompt = |lane: u32| -> Vec<u32> {
            (0..PROMPT)
                .map(|i| 1000 + (i * 7919 + lane * 104_729) % 20_000)
                .collect()
        };
        // Lane `l` runs in slot `l` with the others (batched) and in slot
        // `LANES + l` alone; both prefill alone, the same rows.
        let mut last = [[0u32; 2]; LANES as usize];
        for lane in 0..LANES {
            for (side, slot) in [(0, lane), (1, LANES + lane)] {
                shell.open(slot).expect("a slot opens");
                let tokens = prompt(lane);
                let rows = shell
                    .fire(&[crate::Lane {
                        slot,
                        word: word(tokens.len()),
                        tokens: &tokens,
                    }])
                    .expect("the prompt fires");
                last[lane as usize][side] = argmax(&rows[0]) as u32;
            }
        }
        let mut parted = vec![false; LANES as usize];
        let (mut same, mut ties) = (0usize, 0usize);
        for step in 0..STEPS {
            let next: Vec<[u32; 1]> = (0..LANES as usize).map(|l| [last[l][0]]).collect();
            let lanes: Vec<crate::Lane> = (0..LANES)
                .map(|l| crate::Lane {
                    slot: l,
                    word: word(1),
                    tokens: &next[l as usize],
                })
                .collect();
            let batched = shell.fire(&lanes).expect("the eight lanes fire");
            for lane in 0..LANES as usize {
                let token = [last[lane][1]];
                let alone = shell
                    .fire(&[crate::Lane {
                        slot: LANES + lane as u32,
                        word: word(1),
                        tokens: &token,
                    }])
                    .expect("the lane fires alone")
                    .swap_remove(0);
                let (a, b) = (&alone, &batched[lane]);
                let (ta, tb) = (argmax(a), argmax(b));
                if !parted[lane] && ta != tb {
                    let moved: Vec<f32> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
                    let drift = moved.iter().copied().fold(0.0f32, f32::max);
                    let elsewhere = moved
                        .iter()
                        .enumerate()
                        .filter(|(t, _)| *t != ta && *t != tb)
                        .map(|(_, d)| *d)
                        .fold(0.0f32, f32::max);
                    let diffuse = moved[ta] <= elsewhere && moved[tb] <= elsewhere;
                    let (ga, gb) = (gap(a), gap(b));
                    let tie = diffuse && ga.max(gb) <= 2.0 * drift;
                    println!(
                        "lane {lane} parts at step {step}: alone {ta} batched {tb}; gap {ga:.4} / \
                         {gb:.4}, max drift {drift:.4}, diffuse {diffuse}: {}",
                        if tie { "near-tie" } else { "NOT a near-tie" }
                    );
                    assert!(tie, "lane {lane} parts at step {step} on no near-tie");
                    parted[lane] = true;
                    ties += 1;
                } else if !parted[lane] {
                    same += 1;
                }
                // Side 0 is the batched slot's, side 1 the lone one's.
                last[lane] = [tb as u32, ta as u32];
            }
        }
        println!(
            "{deployment}: {LANES} lanes x {STEPS} steps: {same} steps the same token, {ties} \
             lane(s) parted on a near-tie"
        );
    }
}

/// A served model's logits, gathered at the readout rows, read back at each
/// lane's own row in a fire mixing a prompt between decodes: each lane's
/// last row is the one it answers alone. Run by hand on a served model:
/// `PIE_CONTRACT_ARTIFACT=<.zt> PIE_CONTRACT_DEPLOYMENT=<row>`.
#[cfg(all(test, feature = "cuda"))]
mod mixed_readout {
    use super::*;

    #[test]
    #[ignore = "a readout check run by hand on a served model"]
    fn a_prompt_beside_three_decodes_reads_each_lanes_own_row() {
        let (Some(artifact), Some(deployment)) = (
            std::env::var_os("PIE_CONTRACT_ARTIFACT"),
            std::env::var("PIE_CONTRACT_DEPLOYMENT").ok(),
        ) else {
            eprintln!("skipping: PIE_CONTRACT_ARTIFACT / PIE_CONTRACT_DEPLOYMENT unset");
            return;
        };
        let (mut shell, trace) = load(
            Path::new(&artifact),
            &deployment,
            crate::Knobs::default(),
            |trace| Shape {
                slots: 8,
                budget: poem_compiler::Budget::new(4, 0),
                patches: patches(trace, 4),
                page_size: 16,
                context: 1024,
            },
        )
        .expect("the model loads");
        let word = |n: usize| trace.facts.word(&poem::Request::new(n as u32, false));
        let prompt = |salt: u32, n: u32| -> Vec<u32> {
            (0..n).map(|i| 1000 + (i * 7919 + salt * 104_729) % 20_000).collect()
        };
        for slot in 0..8 {
            shell.open(slot).expect("a slot opens");
        }
        // Decoding lanes in slots 0..3 (twins 4..7, but for the prompt's 3 / 7).
        for (slot, salt) in [(0, 0), (4, 0), (1, 1), (5, 1), (2, 2), (6, 2)] {
            let tokens = prompt(salt, 96);
            shell
                .fire(&[crate::Lane { slot, word: word(tokens.len()), tokens: &tokens }])
                .expect("a prompt fires");
        }
        let fresh = prompt(9, 40);
        let next = [[11u32], [12], [13]];
        let mixed = shell
            .fire(&[
                crate::Lane { slot: 0, word: word(1), tokens: &next[0] },
                crate::Lane { slot: 3, word: word(fresh.len()), tokens: &fresh },
                crate::Lane { slot: 1, word: word(1), tokens: &next[1] },
                crate::Lane { slot: 2, word: word(1), tokens: &next[2] },
            ])
            .expect("the mixed fire serves");
        let alone: Vec<Vec<f32>> = [(4u32, &next[0][..]), (7, &fresh[..]), (5, &next[1][..]), (6, &next[2][..])]
            .iter()
            .map(|&(slot, tokens)| {
                shell
                    .fire(&[crate::Lane { slot, word: word(tokens.len()), tokens }])
                    .expect("a lane fires alone")
                    .swap_remove(0)
            })
            .collect();
        for (lane, (a, b)) in alone.iter().zip(&mixed).enumerate() {
            let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            let top = |row: &[f32]| {
                row.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |(at, _)| at)
            };
            println!("{deployment}: lane {lane} worst {worst} top {} / {}", top(a), top(b));
            assert!(worst < 0.5, "lane {lane}'s last row is not its own ({worst})");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "rocm"))]
    const GPU: (&str, u32, u32, u32, &str) = ("NVIDIA L40S", 8, 9, 142, "fuse.moe_experts");
    #[cfg(feature = "rocm")]
    const GPU: (&str, u32, u32, u32, &str) = (
        "AMD Instinct MI300X",
        9,
        4,
        304,
        "fuse.residual_add_rmsnorm",
    );

    #[test]
    fn every_search_candidate_reads_against_the_registry() {
        let (gpu, major, minor, sms, fused) = GPU;
        let facts = crate::chip::facts(major, minor, sms);
        let (base, _) = crate::chip::resolve(gpu, &facts, None);
        let registry = crate::chip::registry();
        let knobs = search::knobs(&base, SEARCH_SKIPS, "");
        let ids: Vec<&str> = knobs.iter().map(|k| k.id).collect();
        for want in ["kernel.moe.grouped", fused] {
            assert!(ids.contains(&want), "{want} not searched: {ids:?}");
        }
        for k in &knobs {
            for (label, _) in &k.candidates {
                let entries = vec![(k.id.to_string(), label.clone())];
                search::assemble(
                    registry,
                    &base,
                    gpu,
                    &facts,
                    &search::render_overrides(&entries),
                    &[],
                    &[crate::chip::ARCH],
                )
                .unwrap_or_else(|e| panic!("{} = {label}: {e}", k.id));
            }
        }
    }
}
