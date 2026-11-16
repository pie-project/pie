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
    let residency = crate::experts::Plan::cut(&prospect.ranking);
    let mut shell = crate::Shell::load(crate::Boot {
        voxels: None,
        trace: trace.clone(),
        contract: &contract,
        checkpoint: artifact,
        budget: poem_compiler::Budget::new(slots, CONTEXT),
        patches: patches(&trace, slots),
        page_size: 16,
        context: CONTEXT,
        slots,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: crate::Knobs {
            chip: Some(chip.clone()),
            ..crate::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency,
        world: crate::World::default(),
        expert: crate::World::default(),
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

/// How far the estimate's pick may fall short of the measured-best rung:
/// its score, under the measured times, within this share of the best's.
pub const CHECK_BOUND: f64 = 0.05;

/// The lanes a served CUDA load is asked (`pie serve`'s); its tokens are
/// left to the engine (`engine.forward_tokens`, fitted to the card).
const SERVED_LANES: u32 = 512;

/// The fires timed a rung, after two warm ones.
const TIMED_FIRES: usize = 5;

/// A served-shape load of `artifact` under `chip`, every rung the card
/// holds laid (`Knobs::every_rung`), one slot long enough for the widest.
fn load_every_rung(
    artifact: &Path,
    deployment: &str,
    chip: Profile,
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
    let mut budget = poem_compiler::Budget::new(SERVED_LANES, 0);
    budget.readouts_per_lane = poem_compiler::arena::readouts_per_lane(&trace);
    let shell = crate::Shell::load(crate::Boot {
        voxels: None,
        trace: trace.clone(),
        contract: &contract,
        checkpoint: artifact,
        budget,
        patches: patches(&trace, SERVED_LANES),
        page_size: 16,
        context: 32_768,
        slots: 1,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: crate::Knobs {
            chip: Some(chip),
            every_rung: true,
            ..crate::Knobs::default()
        },
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

/// One fire of `width` prompt tokens on a fresh slot 0: its wall time (ns)
/// and whether it replayed a recorded body.
fn fire_once(
    shell: &mut crate::Shell,
    trace: &poem_ir::Trace,
    width: u32,
) -> Result<(f64, bool), String> {
    let token = |i: u32| 100 + (i * 7919) % 5000;
    let prompt: Vec<u32> = (0..width).map(token).collect();
    shell.open(0).map_err(|e| e.to_string())?;
    let lane = crate::Lane {
        slot: 0,
        word: trace.facts.word(&poem::Request::new(width, false)),
        tokens: &prompt,
    };
    let hits = shell.body_stats().tally.hits;
    let began = Instant::now();
    shell
        .fire(std::slice::from_ref(&lane))
        .map_err(|e| e.to_string())?;
    let ns = began.elapsed().as_secs_f64() * 1e9;
    Ok((ns, shell.body_stats().tally.hits > hits))
}

/// The kind of work an op is, for the dissection's rollup.
fn kind_of(op: &str) -> &'static str {
    let moe = op.contains("moe") || op.contains("expert");
    match op.split('.').next().unwrap_or_default() {
        "linear" if moe => "gemm (experts)",
        "linear" => "gemm (dense)",
        "fused" if op.contains("anchored") => "gemm (dense)",
        "attn" => "attention",
        "ssm" | "conv" => "recurrent",
        "sampling" => "sampling/readout",
        "gather" => "gather/embed",
        "collective" => "collective",
        _ => "norms/elementwise",
    }
}

/// A rung dissected: its tokens, its replayed and estimated times (ns),
/// and the launches the estimate counts at it.
struct Dissected {
    tokens: u32,
    replayed: f64,
    estimated: f64,
    launches: u32,
}

/// A walked fire of each of `rungs` dissected by node, under `walked` (the
/// check's profile, eager): its ops' device time (each node drained before
/// and after: [`crate::dissect`]) rolled up by kind, beside the walked wall
/// and the replayed time served fires take.
fn dissect(
    artifact: &Path,
    deployment: &str,
    walked: Profile,
    rungs: &[Dissected],
    say: &mut dyn FnMut(&str),
) -> Result<(), String> {
    let launch_ns = walked.rates().launch_ns.unwrap_or(0) as f64;
    let (mut shell, trace) = load_every_rung(artifact, deployment, walked)?;
    // A drained pair around nothing: what each node's drains cost beside
    // its work, taken off every node.
    crate::dissect::start();
    for _ in 0..64 {
        crate::dissect::enter();
        crate::dissect::left("empty");
    }
    let floor = crate::dissect::take()
        .first()
        .map_or(0.0, |&(_, ns, n)| ns as f64 / f64::from(n));
    for rung in rungs {
        let width = rung.tokens;
        fire_once(&mut shell, &trace, width)?;
        let (walk, _) = fire_once(&mut shell, &trace, width)?;
        crate::dissect::start();
        let (wall, _) = fire_once(&mut shell, &trace, width)?;
        let mut ops = crate::dissect::take();
        ops.sort_by_key(|row| std::cmp::Reverse(row.1));
        let nodes: u32 = ops.iter().map(|row| row.2).sum();
        let net = |ns: u64, n: u32| (ns as f64 - floor * f64::from(n)).max(0.0);
        let device: f64 = ops.iter().map(|&(_, ns, n)| net(ns, n)).sum();
        let mut kinds: Vec<(&str, f64, u32)> = Vec::new();
        for &(op, ns, n) in &ops {
            let kind = kind_of(op);
            match kinds.iter_mut().find(|k| k.0 == kind) {
                Some(k) => {
                    k.1 += net(ns, n);
                    k.2 += n;
                }
                None => kinds.push((kind, net(ns, n), n)),
            }
        }
        kinds.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut out = format!(
            "dissected: a fire of {width} tokens: replayed {:.2} ms, estimated {:.2} ms; walked \
             {:.2} ms ({:.2} ms dissected), {nodes} nodes, {:.2} ms in them past the drains \
             ({:.1} us a drained pair); the estimate's {} launches x {launch_ns} ns = {:.2} ms\n",
            rung.replayed / 1e6,
            rung.estimated / 1e6,
            walk / 1e6,
            wall / 1e6,
            device / 1e6,
            floor / 1e3,
            rung.launches,
            f64::from(rung.launches) * launch_ns / 1e6,
        );
        for (kind, ns, n) in &kinds {
            out.push_str(&format!(
                "  {:>9.3} ms {:>5.1}%  x{n:<5} {kind}\n",
                ns / 1e6,
                100.0 * ns / device.max(1.0)
            ));
        }
        out.push_str("  by op:\n");
        for &(op, ns, n) in ops.iter().take(14) {
            out.push_str(&format!("    {:>9.3} ms  x{n:<5} {op}\n", net(ns, n) / 1e6));
        }
        say(&out);
    }
    Ok(())
}

/// `--check-estimate`: `artifact` loaded under `chip` as a served load is
/// asked, every rung the card holds laid; a fire of each rung's width timed
/// as served fires run (one lane on a fresh slot, its recorded body
/// replayed, the median of [`TIMED_FIRES`]), against the estimate the load
/// weighs its top by. Prints both a rung and the two picks, dissects the
/// narrowest, both picks and the widest under `walked`, and refuses where
/// the estimate's pick scores, under the measured times, more than
/// [`CHECK_BOUND`] below the measured best.
fn check_estimate_of(
    artifact: &Path,
    deployment: &str,
    chip: Profile,
    walked: Profile,
    say: &mut dyn FnMut(&str),
) -> Result<String, String> {
    let (mut shell, trace) = load_every_rung(artifact, deployment, chip)?;
    let model = engine::load::model_class(&trace).to_owned();
    let weighed = shell.weighed().cloned().ok_or_else(|| {
        "this device has no rates in its profile to estimate by: run `pie tune --rates`".to_string()
    })?;
    let laid = shell.budget().max_tokens;
    let rungs: Vec<engine::load::RungFacts> = weighed
        .rungs
        .iter()
        .filter(|r| r.infeasible.is_none() && r.tokens <= laid)
        .copied()
        .collect();
    say(&format!(
        "timing a fire at each of {} rung widths of {model} (laid to {laid})",
        rungs.len()
    ));
    let mut meas = Vec::new();
    for r in &rungs {
        for _ in 0..2 {
            fire_once(&mut shell, &trace, r.tokens)?;
        }
        let mut times = Vec::new();
        for _ in 0..TIMED_FIRES {
            let (ns, replayed) = fire_once(&mut shell, &trace, r.tokens)?;
            if !replayed {
                return Err(format!(
                    "a fire of {} tokens of {model} was walked, not replayed: its time is not \
                     what serving runs, so it is no measurement",
                    r.tokens
                ));
            }
            times.push(ns);
        }
        times.sort_by(f64::total_cmp);
        meas.push(times[times.len() / 2]);
    }
    drop(shell);
    let e = &weighed.estimate;
    let est: Vec<f64> = rungs.iter().map(|r| e.ns(r)).collect();
    let by_est = engine::load::pick(&rungs, &est, weighed.live).map_err(|e| e.to_string())?;
    let by_meas = engine::load::pick(&rungs, &meas, weighed.live).map_err(|e| e.to_string())?;
    let scores = engine::load::scored(&rungs, &meas, weighed.live);
    let score = |tokens: u32| {
        scores
            .iter()
            .find(|&&(t, _)| t == tokens)
            .map_or(0.0, |&(_, s)| s)
    };
    let (at_est, best) = (score(by_est.tokens), score(by_meas.tokens));
    let mut out = format!(
        "{model}: rung, measured ms, estimated ms, measured/estimated, measured tok/s, estimated \
         tok/s; the estimate's terms (ms): launches, weight read, matmul, readout, attention, kv, \
         recurrent, elementwise\n"
    );
    for (k, r) in rungs.iter().enumerate() {
        let tokens = f64::from(r.tokens);
        let t = e.terms(r);
        out.push_str(&format!(
            "  {:>6} {:>9.2} {:>9.2} {:>6.2}x {:>9.0} {:>9.0}   {:.2} {:.2} {:.2} {:.2} {:.2} {:.2} \
             {:.2} {:.2}\n",
            r.tokens,
            meas[k] / 1e6,
            est[k] / 1e6,
            meas[k] / est[k],
            tokens / meas[k] * 1e9,
            tokens / est[k] * 1e9,
            t.launch / 1e6,
            t.weights / 1e6,
            t.flops / 1e6,
            t.readout / 1e6,
            t.attention / 1e6,
            t.kv / 1e6,
            t.recurrent / 1e6,
            t.elementwise / 1e6,
        ));
    }
    let short = if best > 0.0 { 1.0 - at_est / best } else { 0.0 };
    out.push_str(&format!(
        "  pick by the estimate {}, by the measured times {}; the estimate's scores {:.1}% below \
         the best (bound {:.0}%)\n",
        by_est.tokens,
        by_meas.tokens,
        100.0 * short,
        100.0 * CHECK_BOUND
    ));
    say(&out);
    let mut widths: Vec<u32> = vec![by_est.tokens, by_meas.tokens];
    widths.extend(rungs.first().map(|r| r.tokens));
    widths.extend(rungs.last().map(|r| r.tokens));
    widths.sort_unstable();
    widths.dedup();
    let dissected: Vec<Dissected> = widths
        .iter()
        .filter_map(|&t| {
            let k = rungs.iter().position(|r| r.tokens == t)?;
            Some(Dissected {
                tokens: t,
                replayed: meas[k],
                estimated: est[k],
                launches: rungs[k].launches,
            })
        })
        .collect();
    dissect(artifact, deployment, walked, &dissected, say)?;
    if short > CHECK_BOUND {
        return Err(format!(
            "the estimate picks rung {} for {model}, {:.1}% below the measured best {}: past \
             the {:.0}% bound\n{out}",
            by_est.tokens,
            100.0 * short,
            by_meas.tokens,
            100.0 * CHECK_BOUND
        ));
    }
    Ok(out)
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
                tuning.decode_grouped_prefill_from,
                tuning.decode_grouped_prefill_lse,
                tuning.decode_grouped_prefill_window,
                kernels_cuda::attn::decode_gqa::Gate::of(tuning),
            ),
            tuning.ssm_gdn_out_bf16,
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
            &[crate::chip::ARCH],
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
    check_estimate: bool,
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
    // The check of the estimate alone: under today's profile, every model
    // checked before the run fails on any.
    if check_estimate {
        let today = bench.file(&[])?;
        let eager = [
            ("engine.graphs".to_string(), "\"off\"".to_string()),
            ("engine.recording".to_string(), "\"off\"".to_string()),
        ];
        let walked = bench.file(&eager)?;
        let mut failed = Vec::new();
        for (artifact, deployment) in models {
            say(&format!("check of the estimate: {deployment}"));
            if let Err(why) = check_estimate_of(
                artifact,
                deployment,
                Cuda::profile_of(&today)?,
                Cuda::profile_of(&walked)?,
                say,
            ) {
                failed.push(why);
            }
        }
        if !failed.is_empty() {
            return Err(failed.join("\n"));
        }
        return Ok(Outcome {
            device: name,
            file: today,
            lines: Vec::new(),
            checks: Vec::new(),
            changed: false,
            seconds: began.elapsed().as_secs_f64(),
        });
    }
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
        for want in [
            "kernel.moe.grouped",
            "engine.graphs",
            "engine.recording",
            fused,
        ] {
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
