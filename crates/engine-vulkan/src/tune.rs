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
        let i = crate::tuning::FACTS
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
    engine_gpu::api::patch_ladder(
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
    let mut shell = engine_gpu::serve::Shell::load(engine_gpu::serve::Boot {
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
        handed: None,
        // A device untuned has no rates to rule its top by; every rung is laid.
        top: engine_gpu::serve::Top::EveryRung,
    })
    .map_err(|e| e.to_string())?;
    let token = |i: u32| 100 + (i * 7919) % 5000;
    let prompt: Vec<u32> = (0..PROMPT).map(token).collect();
    let fire = |shell: &mut engine_gpu::serve::Shell<crate::device::Api>,
                lanes: &[(u32, &[u32])]|
     -> Result<(f64, Vec<f32>), String> {
        let lanes: Vec<engine_gpu::serve::Lane> = lanes
            .iter()
            .map(|&(slot, tokens)| engine_gpu::serve::Lane {
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

/// How far the estimate's pick may fall short of the measured-best rung:
/// its score, under the measured times, within this share of the best's.
pub const CHECK_BOUND: f64 = 0.05;

/// `--check-estimate`: `artifact` loaded under `file` as a served load is
/// asked ([`crate::FIRE`]), every rung one binding holds laid; a fire of
/// each rung's width timed through the serving path (one lane on a fresh
/// slot, replayed as served fires are), against the estimate the load
/// weighs its top by. Prints both a rung and the two picks; refuses where
/// the estimate's pick scores, under the measured times, more than
/// [`CHECK_BOUND`] below the measured best.
fn check_estimate(
    artifact: &Path,
    deployment: &str,
    device: &str,
    file: &str,
    say: &mut dyn FnMut(&str),
) -> Result<String, String> {
    use poem::{Platform, Request};
    let (lanes, tokens) = crate::FIRE;
    let dir = std::env::temp_dir().join(format!("pie-tune-vulkan-check-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = crate::chip::SHELF.user_path(&dir, device);
    std::fs::create_dir_all(path.parent().expect("a parent")).map_err(|e| e.to_string())?;
    std::fs::write(&path, file).map_err(|e| e.to_string())?;
    let row = poem_compiler::catalog::deployment(deployment)
        .ok_or_else(|| format!("no catalog row `{deployment}`"))?;
    let trace = row.trace(Platform::Vulkan);
    let model = engine::load::model_class(&trace).to_owned();
    let source = ztensor_compat::index(artifact).map_err(|e| e.to_string())?;
    let contract =
        poem::import::own_contract(&source, &trace.params, row.deploy.tp, Platform::Vulkan)
            .map_err(|e| e.to_string())?;
    drop(source);
    let boot = crate::api::DeviceBoot {
        profiles: Some(dir.clone()),
        ..crate::api::DeviceBoot::default()
    };
    // One slot holds the widest fire's tokens: its rows are all a fire
    // reads beside the weights.
    let mut shell = engine_gpu::serve::Shell::<crate::device::Api>::load(engine_gpu::serve::Boot {
        trace: trace.clone(),
        contract: &contract,
        checkpoint: artifact,
        budget: poem_compiler::Budget::new(lanes, tokens),
        patches: patches(&trace, 1),
        voxels: None,
        profile: None,
        page_size: 16,
        context: tokens,
        slots: 1,
        runahead: engine::runahead::Runahead::F1,
        device: &boot,
        residency: engine::load::Residency::default(),
        handed: None,
        top: engine_gpu::serve::Top::EveryRung,
    })
    .map_err(|e| e.to_string())?;
    let weighed = shell.weighed().cloned().ok_or_else(|| {
        format!(
            "{device} has no rates in its profile to estimate by: run `pie tune vulkan --rates`"
        )
    })?;
    let widths: Vec<u32> = weighed
        .rungs
        .iter()
        .filter(|r| r.infeasible.is_none())
        .map(|r| r.tokens)
        .collect();
    say(&format!(
        "timing a fire at each of {} rung widths of {model}",
        widths.len()
    ));
    let token = |i: u32| 100 + (i * 7919) % 5000;
    // The narrowest widths' fires dissected, before any is recorded: the
    // first walked and recorded with a timestamp around each dispatch, the
    // second replaying that recording (the form a served fire takes), its
    // timestamps written again and read back. Each dispatch's kernel and
    // groups, and each kernel's device time, of both.
    let mut dissect = |width: u32, say: &mut dyn FnMut(&str)| -> Result<(), String> {
        let prompt: Vec<u32> = (0..width).map(token).collect();
        let lane = engine_gpu::serve::Lane {
            slot: 0,
            word: trace.facts.word(&Request::new(width, false)),
            tokens: &prompt,
        };
        crate::encode::profile_kernels(true);
        let mut shapes = Vec::new();
        for form in ["walked", "replayed"] {
            shell.open(0).map_err(|e| e.to_string())?;
            crate::encode::capture_shapes(form == "walked");
            crate::encode::reset_kernel_profile();
            let before = shell.recordings();
            let began = Instant::now();
            let fired = shell.fire(std::slice::from_ref(&lane));
            let wall = began.elapsed().as_secs_f64() * 1e3;
            crate::encode::capture_shapes(false);
            if let Err(e) = fired {
                crate::encode::profile_kernels(false);
                return Err(e.to_string());
            }
            if form == "walked" {
                shapes = crate::encode::take_shapes();
            }
            let replayed = shell.recordings().map_or(0, |c| c.0) > before.map_or(0, |c| c.0);
            let mut rows = crate::encode::kernel_profile();
            rows.sort_by_key(|row| std::cmp::Reverse(row.1));
            let device: u64 = rows.iter().map(|row| row.1).sum();
            let mut out = format!(
                "dissected ({form}{}): a fire of {width} tokens of {model}: {wall:.2} ms wall, \
                 {:.2} ms in {} kernels' dispatches on the device (frame span {:.2} ms; {} \
                 dispatches encoded); on the host: submit {:.2} ms, fence wait {:.2} ms, \
                 encode {:.2} ms, the rest of the fire {:.2} ms\n",
                if replayed { ", replayed" } else { "" },
                device as f64 / 1e6,
                rows.len(),
                crate::encode::frame_span_ns() as f64 / 1e6,
                shapes.len(),
                crate::encode::submit_ns() as f64 / 1e6,
                crate::encode::fence_ns() as f64 / 1e6,
                crate::encode::host_encode_ns() as f64 / 1e6,
                (wall * 1e6
                    - (crate::encode::submit_ns()
                        + crate::encode::fence_ns()
                        + crate::encode::host_encode_ns()) as f64)
                    / 1e6,
            );
            for (name, ns, fires) in rows.iter().take(14) {
                let groups: Vec<String> = shapes
                    .iter()
                    .filter(|s| s.entrypoint == name.as_str())
                    .map(|s| format!("{:?}", s.groups))
                    .take(3)
                    .collect();
                out.push_str(&format!(
                    "  {:>9.3} ms  x{fires:<4} {name}  groups {}\n",
                    *ns as f64 / 1e6,
                    groups.join(" ")
                ));
            }
            say(&out);
        }
        crate::encode::profile_kernels(false);
        Ok(())
    };
    for width in widths.iter().copied().filter(|&w| w <= 16) {
        dissect(width, say)?;
    }
    let measured = search::time_fires(
        &widths,
        |width, timed| {
            let prompt: Vec<u32> = (0..width).map(token).collect();
            shell.open(0).map_err(|e| e.to_string())?;
            let lane = engine_gpu::serve::Lane {
                slot: 0,
                word: trace.facts.word(&Request::new(width, false)),
                tokens: &prompt,
            };
            let before = shell.recordings();
            let began = Instant::now();
            shell
                .fire(std::slice::from_ref(&lane))
                .map_err(|e| e.to_string())?;
            let ns = u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX);
            // Served fires replay what they recorded: a fire walked here
            // times the host's encoding beside the device's work, and is no
            // fire time of this model.
            let replayed = |c: Option<(u64, u64)>| c.map_or(0, |(replayed, _)| replayed);
            if timed && replayed(shell.recordings()) <= replayed(before) {
                return Err(format!(
                    "a fire of {width} tokens of {model} was walked, not replayed (recordings \
                     {:?}): its time is not what serving runs, so it is no measurement",
                    shell.recordings()
                ));
            }
            Ok(ns)
        },
        say,
    )?;
    drop(shell);
    let _ = std::fs::remove_dir_all(&dir);

    let rungs: Vec<engine::load::RungFacts> = weighed
        .rungs
        .iter()
        .filter(|r| r.infeasible.is_none())
        .copied()
        .collect();
    let est: Vec<f64> = rungs.iter().map(|r| weighed.estimate.ns(r)).collect();
    let meas: Vec<f64> = rungs
        .iter()
        .map(|r| {
            measured
                .iter()
                .find(|&&(t, _)| t == r.tokens)
                .map_or(f64::NAN, |&(_, ns)| ns as f64)
        })
        .collect();
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
    let mut out = format!("{model} on {device}: rung, measured ms, estimated ms\n");
    for (k, r) in rungs.iter().enumerate() {
        out.push_str(&format!(
            "  {:>6} {:>10.2} {:>10.2}\n",
            r.tokens,
            meas[k] / 1e6,
            est[k] / 1e6
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

/// A fingerprint of the fused trace and plan a load under `text` runs, as
/// `Shell::load` builds them.
fn plan_of(deployment: &str, text: &str, cores: u32) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    let chip = crate::chip::registry().load_user(text, "plan").ok()?;
    let row = poem_compiler::catalog::deployment(deployment)?;
    let trace = poem_compiler::fuse::fuse(
        row.trace(poem::Platform::Vulkan),
        poem_compiler::fuse::flags::on(&chip, &crate::FUSIONS),
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
    let device =
        crate::device::Context::bind(&crate::api::DeviceBoot::default(), engine::Allowance::WHOLE)
            .map_err(|e| e.to_string())?;
    qmm::measure(&device, base, grid, say)
}

#[cfg(not(feature = "vulkan"))]
fn measure(_: &Profile, _: &qmm::Grid, _: &mut dyn FnMut(&str)) -> Result<qmm::Measured, String> {
    Err("this build carries no Vulkan device code".to_string())
}

/// The steps a `pie tune` run takes: every decision's search by default;
/// the `[rates]` step, the check of the estimate, or both, else.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Steps {
    /// The `[rates]` step alone ([`crate::rates::measure`]).
    pub rates: bool,
    /// The check of the estimate alone (`--check-estimate`).
    pub check_estimate: bool,
}

impl Steps {
    fn search(self) -> bool {
        !self.rates && !self.check_estimate
    }

    fn checks(self) -> bool {
        self.check_estimate
    }
}

/// The `[rates]` step: the device's rates measured on it under `base`.
fn rates(base: &Profile, say: &mut dyn FnMut(&str)) -> Result<profile::fire::Rates, String> {
    let device =
        crate::device::Context::bind(&crate::api::DeviceBoot::default(), engine::Allowance::WHOLE)
            .map_err(|e| e.to_string())?;
    let tuning = crate::tuning::Tuning::new(base.clone(), device.limits());
    let rates = crate::rates::measure(&device, &tuning).map_err(|e| e.to_string())?;
    // The matmul path the profile selects on this card, which the rates
    // step measures through.
    let tier = <crate::device::Api as engine_gpu::serve::Serve>::tier(&device);
    let path = kernels_slang::linear::gemm::tile_for(
        kernels_slang::Target::Spirv,
        engine_gpu::backend::Tactics::gemm(&tuning, tier),
        engine_gpu::rates::GEMM,
        engine_gpu::rates::GEMM,
        engine_gpu::rates::GEMM,
    );
    say(&format!(
        "[rates]: the matmul runs {path:?} (tier {tier:?})"
    ));
    say(&format!(
        "[rates]: {:.1} TFLOP/s bf16, {:.0} GB/s, launch {} us",
        rates.bf16_flops_per_ns.unwrap_or(0.0) / 1e3,
        rates.bytes_per_ns.unwrap_or(0.0),
        rates.launch_ns.unwrap_or(0) / 1000
    ));
    Ok(rates)
}

/// Tune the device a load opens. `say` hears progress.
pub fn run(
    depth: Depth,
    profiles: Option<&Path>,
    cache: &Path,
    refit: bool,
    steps: Steps,
    models: &[(PathBuf, String)],
    say: &mut dyn FnMut(&str),
) -> Result<Outcome, String> {
    let fire_only = !steps.search();
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
    let (fitted, mut lines, wrong) = if fire_only {
        (Vec::new(), Vec::new(), Vec::new())
    } else {
        bench_qmm(depth, &name, &base, cores, cache, refit, say)?
    };
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
    if !fire_only {
        knobs.extend(search::knobs(&base, SEARCH_SKIPS, ""));
    }
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
    } else if !fire_only {
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
    let mut tuned = search::render_overrides(&chosen);
    if steps.rates {
        let r = rates(&base, say)?;
        tuned.push_str(&format!(
            "[rates]\nbf16_flops_per_ns = {:?}\nbytes_per_ns = {:?}\nlaunch_ns = {}\n",
            r.bf16_flops_per_ns.unwrap_or(1.0),
            r.bytes_per_ns.unwrap_or(1.0),
            r.launch_ns.unwrap_or(0)
        ));
    }
    let today = bench.file(&[])?;
    let draft = bench.file(&chosen)?;
    // The estimate each model's load weighs its top by, against its fires
    // measured, under the file this writes.
    // Every model is checked before the run fails on any.
    let mut failed = Vec::new();
    for (artifact, deployment) in models.iter().filter(|_| steps.checks()) {
        say(&format!("check of the estimate: {deployment}"));
        if let Err(why) = check_estimate(artifact, deployment, &name, &draft, say) {
            failed.push(why);
        }
    }
    if !failed.is_empty() {
        return Err(failed.join("\n"));
    }
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
