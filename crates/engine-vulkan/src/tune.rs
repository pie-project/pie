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
pub const CHECK_BOUND: f64 = engine::rates::AGREEMENT;

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
    // A shell of `slots` slots of `context` tokens each: one slot holding
    // the widest prompt, or a slot a lane for decode-shaped fires.
    let load = |slots: u32, context: u32| {
        engine_gpu::serve::Shell::<crate::device::Api>::load(engine_gpu::serve::Boot {
            trace: trace.clone(),
            contract: &contract,
            checkpoint: artifact,
            budget: poem_compiler::Budget::new(lanes, tokens),
            patches: patches(&trace, 1),
            voxels: None,
            profile: None,
            page_size: 16,
            context,
            slots,
            runahead: engine::runahead::Runahead::F1,
            device: &boot,
            residency: engine::load::Residency::default(),
            handed: None,
            top: engine_gpu::serve::Top::EveryRung,
        })
        .map_err(|e| e.to_string())
    };
    // The rungs come of a load; the shell each check fire runs on is then
    // sized from the fires (engine::load::CheckFire): its slots the most
    // lanes one fires, its context a lane's keys and new tokens.
    let mut shell = load(1, tokens)?;
    let weighed = shell.weighed().cloned().ok_or_else(|| {
        format!(
            "{device} has no rates in its profile to estimate by: run `pie tune vulkan --rates`"
        )
    })?;
    let sized = |decode: bool| {
        weighed
            .rungs
            .iter()
            .filter(|r| r.infeasible.is_none())
            .map(engine::load::RungFacts::check_fire)
            .filter(|f| (f.lanes > 1) == decode)
            .fold((1u32, 1u32), |(slots, context), f| {
                (
                    slots.max(f.lanes),
                    context.max(f.keys_per_lane + f.tokens_per_lane),
                )
            })
    };
    let (_, prompt_context) = sized(false);
    if prompt_context != tokens {
        drop(shell);
        shell = load(1, prompt_context)?;
    }
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
            // The recurrent scan and the conv of a hybrid, whatever their
            // place in the list.
            let recurrent: f64 = rows
                .iter()
                .filter(|(name, ..)| {
                    ["gdn", "ssm", "scan", "conv", "recurrent"]
                        .iter()
                        .any(|w| name.contains(w))
                })
                .map(|row| row.1 as f64)
                .sum();
            out.push_str(&format!(
                "  recurrent scan and conv dispatches: {:.3} ms\n",
                recurrent / 1e6
            ));
            // The attention's dispatches against the estimate's attention
            // term at this rung, and any fused matmul (a matmul fused with
            // its activation or a second matmul), which need not run at the
            // plain matmul's rates.
            let share = |words: &[&str]| -> f64 {
                rows.iter()
                    .filter(|(name, ..)| words.iter().any(|w| name.contains(w)))
                    .map(|row| row.1 as f64)
                    .sum()
            };
            let estimated = weighed
                .rungs
                .iter()
                .find(|r| r.tokens == width)
                .map_or(0.0, |r| weighed.estimate.terms(r).attention);
            out.push_str(&format!(
                "  attention (sdpa) dispatches: {:.3} ms; the estimate's attention term at this \
                 rung {:.3} ms\n",
                share(&["sdpa", "attn_"]) / 1e6,
                estimated / 1e6
            ));
            // The rates' attention micro-fire at this fire's shape (its rows
            // attending themselves at the model's head width), dissected the
            // same way, so a kernel the two do not share shows.
            if form == "replayed" && weighed.estimate.attn_head_dim > 0 {
                let served: Vec<String> = rows
                    .iter()
                    .filter(|(name, ..)| name.contains("sdpa"))
                    .map(|(name, ns, fires)| {
                        let groups: Vec<String> = shapes
                            .iter()
                            .filter(|s| s.entrypoint == name.as_str())
                            .map(|s| format!("{:?}", s.groups))
                            .take(1)
                            .collect();
                        format!(
                            "{name} groups {} {:.3} ms a dispatch",
                            groups.join(" "),
                            *ns as f64 / 1e6 / (*fires).max(1) as f64
                        )
                    })
                    .collect();
                let shape = engine::rates::ATTN_SHAPE
                    .at(width)
                    .with_dim(weighed.estimate.attn_head_dim);
                crate::encode::capture_shapes(true);
                crate::encode::reset_kernel_profile();
                let rate = crate::rates::attention_rate(
                    &shell.device,
                    &shell.tuning,
                    shape,
                    weighed.estimate.rates.launch_ns,
                );
                crate::encode::capture_shapes(false);
                let micro_shapes = crate::encode::take_shapes();
                let micro: Vec<String> = crate::encode::kernel_profile()
                    .iter()
                    .filter(|(name, ..)| name.contains("sdpa"))
                    .map(|(name, ns, fires)| {
                        let groups: Vec<String> = micro_shapes
                            .iter()
                            .filter(|s| s.entrypoint == name.as_str())
                            .map(|s| format!("{:?}", s.groups))
                            .take(1)
                            .collect();
                        format!(
                            "{name} groups {} {:.3} ms a dispatch",
                            groups.join(" "),
                            *ns as f64 / 1e6 / (*fires).max(1) as f64
                        )
                    })
                    .collect();
                out.push_str(&format!(
                    "  sdpa, served: {}\n  sdpa, the rates' micro-fire at {} rows x {} keys, d \
                     {} ({}): {}\n",
                    served.join("; "),
                    shape.rows,
                    shape.keys,
                    shape.head_dim,
                    match rate {
                        Ok(r) => format!("{:.1} TFLOP/s", r / 1e3),
                        Err(e) => format!("refused: {e}"),
                    },
                    micro.join("; ")
                ));
            }
            out.push_str(&format!(
                "  fused matmul dispatches (swiglu, glu, a matmul with its epilogue): {:.3} ms\n",
                share(&["swiglu", "matmul_glu", "fused"]) / 1e6
            ));
            if form == "replayed" || width <= 16 {
                say(&out);
            }
        }
        crate::encode::profile_kernels(false);
        Ok(())
    };
    for width in widths.iter().copied() {
        dissect(width, say)?;
    }
    // The matmul rate the device reaches at a few row counts on the
    // model's own dense shapes (its distinct planes, `columns x depth`).
    let mut shapes: Vec<(u32, u32)> = Vec::new();
    for param in &trace.params {
        if param.shape.len() == 2 {
            let (n, k) = (param.shape[0], param.shape[1]);
            if let (Ok(n), Ok(k)) = (u32::try_from(n), u32::try_from(k))
                && n >= 256
                && k >= 256
                && n <= 32_768
                && !shapes.contains(&(n, k))
            {
                shapes.push((n, k));
            }
        }
    }
    let mut rates_out = format!("matmul rate of {model}'s planes (TFLOP/s) at m =");
    const ROWS: [u32; 5] = [16, 64, 256, 1024, 4096];
    for m in ROWS {
        rates_out.push_str(&format!(" {m:>7}"));
    }
    rates_out.push('\n');
    for &(n, k) in shapes.iter().take(8) {
        rates_out.push_str(&format!("  {n:>6} x {k:<6}             "));
        for m in ROWS {
            let r = crate::rates::gemm_rate(&shell.device, &shell.tuning, (m, n, k))
                .map_err(|e| e.to_string())?;
            rates_out.push_str(&format!(" {:>7.2}", r / 1e3));
        }
        rates_out.push('\n');
    }
    say(&rates_out);
    // Each rung timed at the shape the estimate weighs it
    // (`RungFacts::sampled`): past the lanes, one prompt of its tokens on a
    // fresh slot; up to them, a lane a token, each on a fresh slot of its
    // own (a second shell, of a slot a lane).
    // Each rung's fire as the estimate weighs it (engine::load::CheckFire).
    let fire_of = |width: u32| {
        weighed
            .rungs
            .iter()
            .find(|r| r.tokens == width)
            .map(engine::load::RungFacts::check_fire)
    };
    let (decode, prompts): (Vec<u32>, Vec<u32>) = widths
        .iter()
        .copied()
        .partition(|&w| fire_of(w).is_some_and(|f| f.lanes > 1));
    let max_lanes = decode
        .iter()
        .filter_map(|&w| fire_of(w).map(|f| f.lanes))
        .max()
        .unwrap_or(1);
    let replayed = |c: Option<(u64, u64)>| c.map_or(0, |(replayed, _)| replayed);
    // Served fires replay what they recorded: a fire walked here times the
    // host's encoding beside the device's work, and is no fire time of this
    // model.
    let walked = |width: u32, recordings| {
        format!(
            "a fire of {width} tokens of {model} was walked, not replayed (recordings \
             {recordings:?}): its time is not what serving runs, so it is no measurement"
        )
    };
    let mut measured = search::time_fires(
        &prompts,
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
            // Timed to the device's finish (the fire's fence); its rows read
            // back after, outside the time.
            let (landed, _) = shell
                .fire_waited(std::slice::from_ref(&lane))
                .map_err(|e| e.to_string())?;
            let ns = u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX);
            shell.rows_of(&landed).map_err(|e| e.to_string())?;
            if timed && replayed(shell.recordings()) <= replayed(before) {
                return Err(walked(width, shell.recordings()));
            }
            Ok(ns)
        },
        say,
    )?;
    drop(shell);
    if !decode.is_empty() {
        say(&format!(
            "decode-shaped rungs (up to {max_lanes} lanes): a lane a token, each on a fresh slot"
        ));
        let (slots, context) = sized(true);
        let mut shell = load(slots.max(max_lanes), context)?;
        // Each lane's tokens: as many as the check fire lays on a lane.
        let most = decode
            .iter()
            .filter_map(|&w| fire_of(w).map(|f| f.tokens_per_lane))
            .max()
            .unwrap_or(1);
        let ones: Vec<Vec<u32>> = (0..max_lanes)
            .map(|i| (0..most).map(|t| token(i * most + t)).collect())
            .collect();
        measured.extend(search::time_fires(
            &decode,
            |width, timed| {
                let fire = fire_of(width);
                let lanes_n = fire.map_or(width, |f| f.lanes);
                let per = fire.map_or(1, |f| f.tokens_per_lane) as usize;
                let word = trace.facts.word(&Request::new(per as u32, false));
                for slot in 0..lanes_n {
                    shell.open(slot).map_err(|e| e.to_string())?;
                }
                let lanes: Vec<engine_gpu::serve::Lane<'_>> = (0..lanes_n)
                    .map(|slot| engine_gpu::serve::Lane {
                        slot,
                        word,
                        tokens: &ones[slot as usize][..per],
                    })
                    .collect();
                let before = shell.recordings();
                let began = Instant::now();
                // Timed to the device's finish; every lane's row read back
                // after, outside the time.
                let (landed, _) = shell.fire_waited(&lanes).map_err(|e| e.to_string())?;
                let ns = u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX);
                shell.rows_of(&landed).map_err(|e| e.to_string())?;
                if timed && replayed(shell.recordings()) <= replayed(before) {
                    return Err(walked(width, shell.recordings()));
                }
                Ok(ns)
            },
            say,
        )?);
    }
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
    let mut out = format!(
        "{model} on {device}: rung, measured ms, estimated ms = work + the launch floors' \
         share (each dispatch its work or a launch); work = max(weights, flops) + readout + \
         attention + kv + recurrent + elementwise\n"
    );
    for (k, r) in rungs.iter().enumerate() {
        let t = weighed.estimate.terms(r);
        out.push_str(&format!(
            "  {:>6} {:>9.2} {:>9.2} = {:>7.2} + floor {:>7.2} ({:>4.1}%); {:>7.2} = max({:>7.2}, \
             {:>7.2}) + {:>7.2} + {:>7.2} + {:>7.2} + {:>7.2} + {:>7.2}\n",
            r.tokens,
            meas[k] / 1e6,
            est[k] / 1e6,
            t.work() / 1e6,
            t.floor() / 1e6,
            100.0 * t.floor() / t.total().max(1.0),
            t.work() / 1e6,
            t.weights / 1e6,
            t.flops / 1e6,
            t.readout / 1e6,
            t.attention / 1e6,
            t.kv / 1e6,
            t.recurrent / 1e6,
            t.elementwise / 1e6
        ));
    }
    out.push_str(&engine::load::shape_table(&weighed.estimate, &rungs, &meas));
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
    // The step's device takes what its own fires lay, no load's budget.
    let device = crate::device::Context::bind(
        &crate::api::DeviceBoot::default(),
        engine::Allowance {
            budget: Some(engine::rates::step_bytes()),
        },
    )
    .map_err(|e| e.to_string())?;
    let tuning = crate::tuning::Tuning::new(base.clone(), device.limits());
    let rates = crate::rates::measure_saying(&device, &tuning, say).map_err(|e| e.to_string())?;
    // The matmul path the profile selects on this card, which the rates
    // step measures through.
    let tier = <crate::device::Api as engine_gpu::serve::Serve>::tier(&device);
    let gemm = engine_gpu::backend::Tactics::gemm(&tuning, tier);
    let w = engine::rates::GEMM_WIDTH;
    for (m, n, k) in engine::rates::GEMM_ROWS.map(|m| (m, w, w)) {
        let path =
            kernels_slang::linear::gemm::tile_for(kernels_slang::Target::Spirv, gemm, m, n, k);
        say(&format!(
            "[rates]: matmul {m}x{n}x{k} through {path:?} (tier {tier:?})"
        ));
    }
    say(&format!(
        "[rates]: matmul {:.1} TFLOP/s at its widest rows, attention {:.1} TFLOP/s at its \
         longest keys, per-row ops {:.0} GB/s, scan {:.2} G state elements/s, {:.0} GB/s, \
         launch {:.2} us",
        rates
            .gemm_tflops_by_rows
            .as_ref()
            .and_then(|c| c.last())
            .map_or(0.0, |&(_, t)| t),
        rates
            .attn_tflops_by_keys
            .as_ref()
            .and_then(|c| c.last())
            .map_or(0.0, |&(_, t)| t),
        rates.elementwise_gbps.unwrap_or(0.0),
        rates.scan_gelems.unwrap_or(0.0),
        rates.hbm_gbps.unwrap_or(0.0),
        rates.launch_ns.unwrap_or(0) as f64 / 1e3
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
        tuned.push_str(&engine::rates::tuned(&r));
    }
    let today = bench.file(&[])?;
    let draft = bench.file(&chosen)?;
    // The estimate each model's load weighs its top by, against its fires
    // measured, under the file this writes: its decisions and the rates this
    // run measured.
    let checked_under = search::assemble(
        crate::chip::registry(),
        &base,
        &name,
        &facts,
        &tuned,
        &[],
        ASSUME,
    )?;
    // Every model is checked before the run fails on any.
    let mut failed = Vec::new();
    for (artifact, deployment) in models.iter().filter(|_| steps.checks()) {
        say(&format!("check of the estimate: {deployment}"));
        if let Err(why) = check_estimate(artifact, deployment, &name, &checked_under, say) {
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
    let mut provenance = vec![
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
    if steps.rates {
        provenance.push(engine::rates::provenance(
            "vulkan",
            &format!(
                "the matmul through the profile's path, the host's clock around each frame; \
                 operands: engine::rates' pattern; at {}",
                engine::rates::build_commit()
            ),
        ));
    }
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
