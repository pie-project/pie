//! `wafer-bench`: a linked wafer program run bare (no Pie) on the engine's host code, against a
//! golden, on the simulator or on a system at `--cmaddr`. Prints `WAFER_BENCH ok` when every check
//! holds.
//!
//! ```text
//! wafer-bench --program <dir> --weights <dir>[:<prefix>] --golden <golden.json>
//!             [--steps N] [--tokens N] [--logits <out.json>] [--times <out.json>]
//! wafer-bench --reading <dir> --program <dir> --inputs <dir> [--grid T,H,W] [--steps N]
//!             [--golden <dir>] [--save <dir>] [--statics <dir> | --checkpoint <dir | .zt>]
//!             [--stamps <dir>] [--poll 1]
//! wafer-bench --decode <golden.json> --checkpoint <dir | .zt>
//!             (--reading <dir> --program <dir> | --model <deployment> --cache <dir>
//!              [--positions N] [--lanes B]) [--steps N] [--stamps 1 | <dir>] [--anneal N]
//! ```

use std::collections::HashMap;
use std::path::Path;

use engine_cerebras::artifact::{ArtifactModel, Weights, argmax, read};
use engine_cerebras::sdk::{Platform, Rect, Runtime, Sdk, Simulator, Target};
use kernels_cerebras::compile::emit::stamps::{self, Probe, Read};
use serde_json::{Value, json};

const USAGE: &str = "usage: wafer-bench --program <dir> --weights <dir>[:<prefix>] --golden \
                     <golden.json> [--steps N] [--tokens N] [--cmaddr <addr>] [--logits <out.json>] \
                     [--times <out.json>]\n       wafer-bench --reading <dir> --program <dir> --inputs \
                     <dir> [--grid T,H,W] [--steps N] [--golden <dir>] [--save <dir>] [--cmaddr <addr>] \
                     [--statics <dir> | --checkpoint <dir | .zt>] [--stamps <dir>] [--poll 1]\n       \
                     wafer-bench --decode <golden.json> --checkpoint <dir | .zt> (--reading <dir> \
                     --program <dir> | --model <deployment> --cache <dir> [--positions N] [--lanes B]) \
                     [--steps N] [--cmaddr <addr>] [--stamps 1 | <dir>] [--anneal <moves a block>]";

/// A headless slice's rows against the golden's: relative L2 and cosine.
const HIDDEN_REL: f64 = 3e-2;
const HIDDEN_COS: f64 = 0.999;

fn main() {
    match run() {
        Ok(true) => println!("WAFER_BENCH ok"),
        Ok(false) => {
            println!("WAFER_BENCH FAILED");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("wafer-bench: {e}");
            std::process::exit(2);
        }
    }
}

/// A greedy run's tokens, and its decode launches' (those after the prompt's) wall ms, steps,
/// count and host ns by part ([`ArtifactModel::host_ns`]).
struct Run {
    got: Vec<u32>,
    ms: f64,
    steps: usize,
    launches: usize,
    host: [u64; 4],
}

/// `n` tokens after `prompt`, `steps` a launch.
fn greedy(m: &mut ArtifactModel, prompt: &[u32], n: usize, steps: usize) -> Result<Run, String> {
    let mut got = Vec::new();
    let mut feed = prompt.to_vec();
    let mut start = 0;
    let mut run = Run {
        got: Vec::new(),
        ms: 0.0,
        steps: 0,
        launches: 0,
        host: [0; 4],
    };
    while got.len() < n {
        let extra = (steps - 1).min(n - got.len() - 1);
        let (t0, h0) = (std::time::Instant::now(), m.host_ns);
        let a = m.greedy(&feed, start, extra)?;
        if start > 0 {
            run.ms += t0.elapsed().as_secs_f64() * 1e3;
            run.steps += extra + 1;
            run.launches += 1;
            for (h, (now, was)) in run.host.iter_mut().zip(m.host_ns.iter().zip(h0)) {
                *h += now - was;
            }
        }
        let new = &a[feed.len() - 1..];
        start += feed.len() + new.len() - 1;
        got.extend_from_slice(new);
        feed = vec![*new.last().ok_or("a launch made no token")?];
    }
    got.truncate(n);
    run.got = got;
    Ok(run)
}

/// `b[i] - b[0]` for each word, modulo the 32-bit stamp (`None`: a word no pass stamped, still 0).
fn since(b: &[u32]) -> Vec<Option<f64>> {
    b.iter()
        .enumerate()
        .map(|(i, &v)| (i == 0 || v != 0).then(|| v.wrapping_sub(b[0]) as f64))
        .collect()
}

/// The element-wise mean of equal-length rows of stamps, over the rows that have each one.
fn mean_of(rows: &[Vec<Option<f64>>]) -> Vec<Option<f64>> {
    (0..rows.first().map_or(0, Vec::len))
        .map(|i| {
            let v: Vec<f64> = rows.iter().filter_map(|r| r[i]).collect();
            (!v.is_empty()).then(|| (v.iter().sum::<f64>() / v.len() as f64).round())
        })
        .collect()
}

/// Consecutive differences, modulo the 32-bit stamp.
fn diffs(b: &[u32]) -> Vec<f64> {
    b.windows(2)
        .map(|w| w[1].wrapping_sub(w[0]) as f64)
        .collect()
}

/// The element-wise mean of equal-length rows.
fn mean(rows: &[Vec<f64>]) -> Vec<f64> {
    let n = rows.len().max(1) as f64;
    let mut m = vec![0.0; rows.first().map_or(0, Vec::len)];
    for r in rows {
        for (a, v) in m.iter_mut().zip(r) {
            *a += v / n;
        }
    }
    m.iter().map(|v| v.round()).collect()
}

/// A greedy sequence as [`greedy`], every probe's `tdbg` read after each launch; the cycles of
/// the launches after the prompt's, averaged.
fn stamped(
    m: &mut ArtifactModel,
    probes: &[Probe],
    prompt: &[u32],
    n: usize,
    steps: usize,
) -> Result<Value, String> {
    let mut reads: Vec<Vec<Vec<u32>>> = Vec::new();
    let mut feed = prompt.to_vec();
    let (mut start, mut made, mut last_extra) = (0, 0, 0);
    while made < n {
        let extra = (steps - 1).min(n - made - 1);
        let a = m.greedy(&feed, start, extra)?;
        let new = &a[feed.len() - 1..];
        start += feed.len() + new.len() - 1;
        made += new.len();
        feed = vec![*new.last().ok_or("a launch made no token")?];
        let mut r = Vec::new();
        for p in probes {
            let at = Rect {
                x: p.at.x as i32,
                y: p.at.y as i32,
                w: 1,
                h: 1,
            };
            r.push(m.dev.get("tdbg", at, p.words as usize)?);
        }
        reads.push(r);
        last_extra = extra;
    }
    let decode = if reads.len() > 1 {
        &reads[1..]
    } else {
        &reads[..]
    };
    let (mut layers, mut units, mut send) = (Vec::new(), Vec::new(), Vec::new());
    let mut phases = serde_json::Map::new();
    let mut lp = Value::Null;
    for (i, p) in probes.iter().enumerate() {
        let rows: Vec<&Vec<u32>> = decode.iter().map(|r| &r[i]).collect();
        match p.read {
            Read::Layers { layers: k } => {
                let d: Vec<Vec<f64>> = rows.iter().map(|b| diffs(&b[..k as usize + 2])).collect();
                let d = mean(&d);
                units.push(d[..k as usize].iter().sum::<f64>());
                send.push(d[k as usize]);
                layers.push(d[..k as usize].to_vec());
            }
            Read::Phases => {
                let n = (p.words - p.counts) as usize;
                let d: Vec<_> = rows.iter().map(|b| since(&b[..n])).collect();
                phases.insert(p.name.clone(), json!(mean_of(&d)));
                if p.counts > 0 {
                    let c: Vec<_> = rows
                        .iter()
                        .map(|b| b[n..].iter().map(|&v| Some(v as f64)).collect())
                        .collect();
                    phases.insert(format!("{}.counts", p.name), json!(mean_of(&c)));
                }
            }
            Read::Loop => {
                let at = [0, 66, 67, 68, 69];
                let d: Vec<_> = rows
                    .iter()
                    .map(|b| since(&at.iter().map(|&j| b[j]).collect::<Vec<_>>()))
                    .collect();
                phases.insert(p.name.clone(), json!(mean_of(&d)));
                let last = rows.last().ok_or("no launch read")?;
                let s = (last_extra + 1).min(64);
                lp = json!(diffs(&last[2..2 + s]));
                let every: Vec<_> = rows[..rows.len() - 1]
                    .iter()
                    .map(|b| diffs(&b[2..2 + steps.min(64)]))
                    .chain(std::iter::once(diffs(&last[2..2 + s])))
                    .collect();
                phases.insert("loop_every".into(), json!(every));
            }
        }
    }
    Ok(json!({
        "launches": decode.len(),
        "layer_cycles_mean": layers,
        "unit_cycles_mean": units,
        "send_cycles_mean": send,
        "phases_mean": phases,
        "loop_cycles": lp,
    }))
}

/// `--decode`: a unit decode program run teacher-forced through the golden, token by token, each
/// step past the prompt held to the golden's next token. `--steps N`: the first N steps alone.
fn decode(get: &HashMap<String, String>) -> Result<bool, String> {
    use engine_cerebras::reading::Reading;
    let need = |k: &str| get.get(k).cloned().ok_or_else(|| USAGE.to_string());
    // the program given, or (`--cache`) the one the engine compiles on load for the model
    let (dir, program) = match get.get("cache") {
        None => (need("reading")?, need("program")?),
        Some(cache) => {
            let model = need("model")?;
            let trace = poem_compiler::catalog::repository()
                .parse(&model)
                .ok_or_else(|| format!("{model}: no such deployment"))?
                .trace(poem_ir::Platform::Cerebras);
            let cfg = engine_cerebras::config::Config {
                cache_dir: Some(cache.into()),
                cmaddr: get.get("cmaddr").cloned(),
                // (`--anneal <moves a block>`: the annealing placer on the unit program kept)
                wafer: engine_cerebras::config::Wafer {
                    anneal: get
                        .get("anneal")
                        .map(|a| a.parse::<u64>().map_err(|e| format!("--anneal: {e}")))
                        .transpose()?,
                    ..Default::default()
                },
                ..Default::default()
            };
            let positions = get
                .get("positions")
                .map(|p| p.parse::<u32>().map_err(|e| format!("--positions: {e}")))
                .transpose()?;
            let lanes = get
                .get("lanes")
                .map_or(Ok(1), |p| p.parse::<u32>())
                .map_err(|e| format!("--lanes: {e}"))?;
            let (d, l, at) = engine_cerebras::prepare::unit_program(
                &cfg,
                &trace,
                positions,
                engine_cerebras::prepare::Lanes::Exactly(lanes),
            )?;
            println!("DECODE_PROGRAM {} at {at} positions", d.display());
            (d.display().to_string(), l.display().to_string())
        }
    };
    let mut reading = Reading::open(Path::new(&dir))?;
    // (every PE's stamps read back only where asked)
    reading.stamped = get.contains_key("stamps");
    let trace = reading.trace()?;
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(need("decode")?).map_err(|e| format!("--decode: {e}"))?,
    )
    .map_err(|e| format!("--decode: {e}"))?;
    let ids = |k: &str| -> Vec<u32> {
        golden[k]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|t| t.as_u64())
                    .map(|t| t as u32)
                    .collect()
            })
            .unwrap_or_default()
    };
    let (prompt, seq) = (ids("prompt"), ids("tokens"));
    // (absolute: the import contract's files are the checkpoint's, joined to it again on load)
    let ckpt =
        std::fs::canonicalize(need("checkpoint")?).map_err(|e| format!("--checkpoint: {e}"))?;
    let t0 = std::time::Instant::now();
    // (a stamped artifact's own planes, or a raw checkpoint's through the import contract)
    let contract = poem_compiler::catalog::repository().contract_for(&trace, &ckpt)?;
    let landed = engine_cerebras::prepare::land(&trace, &contract, &ckpt)?;
    let of = |name: &str, count: usize| engine_cerebras::prepare::plane(&landed, name, count);
    let mut weights = engine_cpu::Forward::new(trace.clone(), None, &of)?;
    let statics = reading.statics(&mut weights, &reading.step(&trace, None, &[]))?;
    let mut host = engine_cpu::Forward::host_for(trace.clone(), &of, &reading.entry_values())?;
    println!(
        "DECODE {} landed and its statics made in {:.0} s",
        reading.model,
        t0.elapsed().as_secs_f64()
    );
    let lanes = reading
        .classes
        .iter()
        .map(|c| c.rows as usize)
        .sum::<usize>()
        .max(1);
    let platform = match get.get("cmaddr") {
        Some(c) => Platform::System { cmaddr: c.clone() },
        None => Platform::Simulator(Simulator::new(Target::Wse3, 8)),
    };
    let sdk = Sdk::open().map_err(|e| e.to_string())?;
    let mut rt =
        Runtime::new(sdk, Path::new(&program), platform, "WARNING").map_err(|e| e.to_string())?;
    rt.load();
    rt.run();
    reading.write_statics(&mut rt, &statics)?;
    // B lanes: lane t teacher-forced through the golden from step 3t, its pages its own
    let stagger = 3;
    if let Some(g) = reading.paging {
        // (the column's last page the scratch)
        let per = (g.pages - 1) / g.lanes;
        let mut table = vec![g.pages - 1; (g.lanes * g.pages) as usize];
        for t in 0..g.lanes {
            for j in 0..per {
                table[(t * g.pages + j) as usize] = t * per + j;
            }
        }
        reading.write_pages(&mut rt, &table)?;
    }
    let steps = get
        .get("steps")
        .map_or(Ok(seq.len().saturating_sub(1)), |v| v.parse::<usize>())
        .map_err(|e| format!("--steps: {e}"))?
        .min(seq.len().saturating_sub(1));
    let (mut held, mut checked) = (0, 0);
    for gs in 0..steps + (lanes - 1) * stagger {
        // each lane's step (an idle lane's row at its first position, re-stored when it starts)
        let local: Vec<Option<usize>> = (0..lanes)
            .map(|t| gs.checked_sub(t * stagger).filter(|&s| s < steps))
            .collect();
        let toks: Vec<u32> = local.iter().map(|s| seq[s.unwrap_or(0)]).collect();
        let pos: Vec<u32> = local.iter().map(|s| s.unwrap_or(0) as u32).collect();
        let (_, entries) = reading.decode_step(&mut host, &trace, &toks, &pos)?;
        let io0 = rt.io;
        reading.write_pos(&mut rt, pos[0])?;
        // lane t its own lane slot and state slot
        let rows: Vec<(u32, u32, u32)> = (0..lanes).map(|t| (t as u32, t as u32, pos[t])).collect();
        reading.write_rows(&mut rt, lanes, &rows)?;
        let t1 = std::time::Instant::now();
        let (got, cycles) = reading.launch(&mut rt, &entries)?;
        let wall_ms = t1.elapsed().as_secs_f64() * 1e3;
        let io1 = rt.io;
        reading.write_ahead(&mut rt, &mut host, &trace, &toks, &pos)?;
        // (`--stamps <dir>`: each step's stamps too, as `--reading` saves them; `--stamps 1` the
        // cycles alone)
        if let Some(dir) = get.get("stamps").filter(|d| d.as_str() != "1") {
            save_tsc(&mut rt, &reading, dir, gs)?;
        }
        // the program's own argmax (its head roots'), against the logits read back
        let on_wafer = engine_cerebras::reading::Fabric::argmax(&mut rt, &reading)?;
        // the step's copies (the stamps apart): before and through its launch, and after it
        let io2 = rt.io;
        let ms = |a: u64, b: u64| (a - b) as f64 / 1e6;
        let io = json!({"h2d": [io1.h2d.0 - io0.h2d.0, io1.h2d.2 - io0.h2d.2, ms(io1.h2d.1, io0.h2d.1)],
            "ahead": [io2.h2d.0 - io1.h2d.0, io2.h2d.2 - io1.h2d.2, ms(io2.h2d.1, io1.h2d.1)],
            "d2h": [io2.d2h.0 - io0.d2h.0, io2.d2h.2 - io0.d2h.2, ms(io2.d2h.1, io0.d2h.1)],
            "launch_ms": ms(io2.launch.1, io0.launch.1)});
        let back = reading.rows_back(&got)?;
        let logits = back
            .iter()
            .max_by_key(|c| c.width)
            .ok_or("no logits read back")?;
        let engine_cpu::forward::Values::F32(all) = &logits.values else {
            return Err("logits not f32".into());
        };
        let w = logits.width;
        // a headless slice (`wafer-compile --stages`): its rows held to the golden's value at each
        // position by relative L2 and cosine
        if let Some(hidden) = golden["hidden"].as_array() {
            for (t, s) in local.iter().enumerate() {
                let Some(s) = *s else { continue };
                let v = &all[t * w..(t + 1) * w];
                let want: Vec<f32> = hidden
                    .get(s)
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("golden.hidden has no position {s}"))?
                    .iter()
                    .filter_map(Value::as_f64)
                    .map(|x| x as f32)
                    .collect();
                if want.len() != w {
                    return Err(format!(
                        "golden.hidden is {} wide, the rows read back {w}",
                        want.len()
                    ));
                }
                let dot = |a: &[f32], b: &[f32]| -> f64 {
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| f64::from(*x) * f64::from(*y))
                        .sum()
                };
                let diff: Vec<f32> = v.iter().zip(&want).map(|(a, b)| a - b).collect();
                let rel = (dot(&diff, &diff) / dot(&want, &want).max(1e-30)).sqrt();
                let cos = dot(v, &want) / (dot(v, v) * dot(&want, &want)).sqrt().max(1e-30);
                let ok = rel <= HIDDEN_REL && cos >= HIDDEN_COS;
                checked += 1;
                held += usize::from(ok);
                println!(
                    "DECODE_STEP {}",
                    json!({"lane": t, "step": s, "token": seq[s], "value": golden["value"],
                        "rel_l2": rel, "cos": cos, "ok": ok, "cycles": cycles, "wall_ms": wall_ms,
                        "finite": v.iter().all(|x| x.is_finite())})
                );
            }
            continue;
        }
        for (t, s) in local.iter().enumerate() {
            let Some(s) = *s else { continue };
            let v = &all[t * w..(t + 1) * w];
            let (best, top) =
                v.iter().enumerate().fold(
                    (0usize, f32::MIN),
                    |b, (i, &x)| if x > b.1 { (i, x) } else { b },
                );
            let want = seq[s + 1];
            let past = s + 1 >= prompt.len();
            if past {
                checked += 1;
                held += usize::from(best as u32 == want);
            }
            // the golden's top-5 at this step (HF's own logits), for the margin a miss had
            let gtop = &golden["top5"][s + 1 - prompt.len().min(s + 1)];
            println!(
                "DECODE_STEP {}",
                json!({"lane": t, "step": s, "token": seq[s], "argmax": best, "logit": top, "want": want,
                    "checked": past, "ok": best as u32 == want, "cycles": cycles, "wall_ms": wall_ms,
                    "finite": v.iter().all(|x| x.is_finite()),
                    "wafer_argmax": on_wafer.as_ref().and_then(|a| a.get(t)),
                    "io": io,
                    "golden_top": if past { gtop.clone() } else { Value::Null }})
            );
        }
    }
    rt.stop();
    println!(
        "DECODE_RESULT {}",
        json!({"held": held, "checked": checked})
    );
    Ok(held == checked)
}

/// `--statics <dir>`: the static arrays made elsewhere, the `k`-th array's words in `<k>.u32`.
fn statics_of(
    reading: &engine_cerebras::reading::Reading,
    dir: &str,
) -> Result<Vec<Option<Vec<u32>>>, String> {
    (0..reading.h2d.len())
        .map(|k| {
            let p = Path::new(dir).join(format!("{k}.u32"));
            p.is_file().then(|| {
                std::fs::read(&p).map(|b| {
                    b.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes(*c))
                        .collect()
                })
            })
        })
        .map(Option::transpose)
        .collect::<Result<_, _>>()
        .map_err(|e| format!("{dir}: {e}"))
}

/// `--stamps <dir>`: step `i`'s stamps, every PE's three words, row major, in files of at most
/// 3 MB (`tsc<i>_<row>.tsc`: from fabric row `row` on).
fn save_tsc(
    rt: &mut Runtime,
    reading: &engine_cerebras::reading::Reading,
    dir: &str,
    i: usize,
) -> Result<(), String> {
    let Some(ts) = rt.tsc(reading)? else {
        return Ok(());
    };
    std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
    let row = 3 * reading.w as usize;
    let rows = ((3 << 20) / (4 * row)).max(1);
    for (k, part) in ts.chunks(rows * row).enumerate() {
        let p = Path::new(dir).join(format!("tsc{i}_{}.tsc", k * rows));
        let b: Vec<u8> = part.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(&p, b).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(())
}

/// `--reading`: a stream reading's steps on the program, the host's ops the engine's.
fn reading(get: &HashMap<String, String>) -> Result<bool, String> {
    use engine_cerebras::reading::{Denoise, Reading};
    let need = |k: &str| get.get(k).cloned().ok_or_else(|| USAGE.to_string());
    let mut reading = Reading::open(Path::new(&need("reading")?))?;
    // (`--poll 1`: each launch timed to its end on the device, the reads after it aside)
    reading.poll = get.get("poll").is_some_and(|v| v == "1");
    let trace = reading.trace()?;
    let grid: Option<[u32; 3]> = match get.get("grid") {
        Some(g) => {
            let v: Vec<u32> = g
                .split(',')
                .map(|x| x.parse().map_err(|e| format!("--grid {g}: {e}")))
                .collect::<Result<_, String>>()?;
            Some(<[u32; 3]>::try_from(v).map_err(|_| format!("--grid {g}: T,H,W"))?)
        }
        None => None,
    };
    // (`--inputs a,b`: a batch's requests' inputs, a directory each)
    let dirs: Vec<String> = need("inputs")?.split(',').map(str::to_string).collect();
    let dir = dirs[0].clone();
    let read_f32 = |p: &Path| -> Result<Vec<f32>, String> {
        let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    };
    let mut inputs: Vec<(String, Vec<f32>)> = Vec::new();
    for i in &trace.inputs {
        let p = Path::new(&dir).join(format!("{}.f32", i.name));
        if p.is_file() {
            inputs.push((i.name.clone(), read_f32(&p)?));
        }
    }
    let mut host = engine_cpu::Forward::host(trace.clone());
    // `--checkpoint <dir>`: the static arrays and the entries' weights from the checkpoint, landed
    // through the import contract as the engine lands it
    let made: Option<Vec<Option<Vec<u32>>>> = match get.get("checkpoint") {
        Some(c) => {
            let ckpt = std::fs::canonicalize(c).map_err(|e| format!("--checkpoint: {e}"))?;
            let t0 = std::time::Instant::now();
            // (a stamped artifact's own planes, or a raw checkpoint's through the import contract)
            let contract = poem_compiler::catalog::repository().contract_for(&trace, &ckpt)?;
            let landed = engine_cerebras::prepare::land(&trace, &contract, &ckpt)?;
            let of =
                |name: &str, count: usize| engine_cerebras::prepare::plane(&landed, name, count);
            let mut weights = engine_cpu::Forward::new(trace.clone(), None, &of)?;
            let statics = reading.statics(&mut weights, &reading.step(&trace, None, &[]))?;
            host = engine_cpu::Forward::host_for(trace.clone(), &of, &reading.entry_values())?;
            println!(
                "READING {} landed and its statics made in {:.0} s",
                reading.model,
                t0.elapsed().as_secs_f64()
            );
            Some(statics)
        }
        None => None,
    };
    let program = need("program")?;
    let steps = get
        .get("steps")
        .map_or(Ok(1), |v| v.parse::<u32>())
        .map_err(|e| format!("--steps: {e}"))?;
    let image_rows = reading
        .classes
        .iter()
        .zip(&reading.step(&trace, grid, &inputs).lanes)
        .filter(|(_, l)| l.clip.is_some())
        .map(|(c, _)| c.rows)
        .sum::<u32>();
    let d = Denoise::of(&reading.model, steps, Some(image_rows))?;
    let platform = match get.get("cmaddr") {
        Some(c) => Platform::System { cmaddr: c.clone() },
        None => Platform::Simulator(Simulator::new(Target::Wse3, 8)),
    };
    let sdk = Sdk::open().map_err(|e| e.to_string())?;
    let golden = get.get("golden").map(std::path::PathBuf::from);
    let rel_cos = |a: &[f32], b: &[f32]| {
        let dot = |x: &[f32], y: &[f32]| -> f64 {
            x.iter()
                .zip(y)
                .map(|(p, q)| f64::from(*p) * f64::from(*q))
                .sum()
        };
        let diff: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        (
            (dot(&diff, &diff) / dot(b, b)).sqrt(),
            dot(a, b) / (dot(a, a) * dot(b, b)).sqrt(),
        )
    };
    println!(
        "READING {} {}: sigmas {:?}",
        reading.model, reading.reading, d.sigmas
    );
    let mut ok = true;
    let lanes = reading.classes.len();
    let x0 = inputs
        .iter()
        .find(|(n, _)| *n == d.latent)
        .map(|(_, v)| v.clone())
        .ok_or("no latent input")?;
    // the program loaded once, its static arrays written once: a step is its entries and a launch
    let load = || -> Result<Runtime, String> {
        let t0 = std::time::Instant::now();
        let mut rt = Runtime::new(
            sdk.clone(),
            Path::new(&program),
            platform.clone(),
            "WARNING",
        )
        .map_err(|e| e.to_string())?;
        println!("PROGRESS runtime made {:.1} s", t0.elapsed().as_secs_f64());
        rt.load();
        let loaded = t0.elapsed().as_secs_f64();
        println!("PROGRESS loaded {loaded:.1} s");
        rt.run();
        println!("PROGRESS running {:.1} s", t0.elapsed().as_secs_f64());
        match (&made, get.get("statics")) {
            // the static arrays made from the checkpoint here, or elsewhere: the k-th array's words
            (Some(words), _) => reading.write_statics(&mut rt, words)?,
            (None, Some(dir)) => reading.write_statics(&mut rt, &statics_of(&reading, dir)?)?,
            (None, None) => reading.write_static(&mut rt)?,
        }
        println!(
            "READING_LOAD {}",
            json!({"load_s": loaded, "run_and_statics_s": t0.elapsed().as_secs_f64() - loaded,
                "statics": rt.io.to_string()})
        );
        Ok(rt)
    };
    // (the inputs bound before any program loads: a run that cannot reach the fabric fails here)
    {
        let step = reading.step(&trace, grid, &inputs);
        if reading.batch > 1 {
            reading.entries_batch(&mut host, std::slice::from_ref(&step))?;
        } else {
            reading.entries(&mut host, &step)?;
        }
        println!("PROGRESS the inputs bound");
    }
    let mut rt = load()?;
    let (_, vs) = d.run(&mut inputs, lanes, |i, ins| {
        let step = reading.step(&trace, grid, ins);
        // (a batched program: the request in its first slot standing for the rest)
        let entries = if reading.batch > 1 {
            reading.entries_batch(&mut host, std::slice::from_ref(&step))?
        } else {
            reading.entries(&mut host, &step)?
        };
        println!("PROGRESS step {i}: entries made, launching");
        let t0 = std::time::Instant::now();
        let a = rt.io;
        let (got, cycles) = reading.launch(&mut rt, &entries)?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        println!("PROGRESS step {i}: back, {cycles:?} cycles, {wall_ms:.3} ms");
        if let Some(dir) = get.get("stamps") {
            save_tsc(&mut rt, &reading, dir, i)?;
        }
        // the step's host side by kind: entries in, the launch, the rows and stamps back
        let b = rt.io;
        let ms = |x: u64, y: u64| (x - y) as f64 / 1e6;
        let io = json!({"h2d_ms": ms(b.h2d.1, a.h2d.1), "h2d_calls": b.h2d.0 - a.h2d.0,
            "h2d_words": b.h2d.2 - a.h2d.2, "launch_ms": ms(b.launch.1, a.launch.1),
            "d2h_ms": ms(b.d2h.1, a.d2h.1), "d2h_calls": b.d2h.0 - a.d2h.0,
            "d2h_words": b.d2h.2 - a.d2h.2, "done_ms": ms(b.done.1, a.done.1)});
        let back = reading.rows_back(&got)?;
        let back = if reading.batch > 1 {
            reading.request_of(&back, 0)
        } else {
            back
        };
        let v = reading
            .finish(&mut host, &trace, &step, &back, &[d.velocity.as_str()])?
            .remove(0);
        let mut line = json!({"step": i, "sigma": d.sigmas[i], "cycles": cycles,
            "wall_s": t0.elapsed().as_secs(), "wall_ms": wall_ms, "io": io,
            "finite": v.iter().all(|x| x.is_finite())});
        if let Some(g) = &golden {
            let p = g.join(format!("cpu_velocity{i}.f32"));
            if p.is_file() {
                let (rel, cos) = rel_cos(&v, &read_f32(&p)?);
                ok &= rel <= 1e-2 && cos >= 0.999;
                line["velocity"] = json!({"rel_l2": rel, "cosine": cos});
            }
        }
        println!("READING_STEP {line}");
        Ok(v)
    })?;
    rt.stop();
    // each step's latent, from the program's velocities (the golden's only on its own schedule)
    let golden_steps = golden.as_ref().map_or(0, |g| {
        (0..)
            .take_while(|k| g.join(format!("cpu_velocity{k}.f32")).is_file())
            .count()
    });
    if golden.is_some() && golden_steps != vs.len() {
        println!(
            "READING_LATENT not held: the golden's schedule is {golden_steps} steps, this run's {}",
            vs.len()
        );
    }
    let mut x = x0;
    for (i, v) in vs.iter().enumerate() {
        let dt = d.sigmas[i + 1] - d.sigmas[i];
        for (x, v) in x.iter_mut().zip(v) {
            *x += dt * v;
        }
        if let Some(dir) = get.get("save") {
            // the program's velocity and the latent after it, as the golden files lay them
            let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
            for (name, v) in [
                (format!("velocity{i}.f32"), v),
                (format!("latent{}.f32", i + 1), &x),
            ] {
                let p = Path::new(dir).join(name);
                std::fs::write(&p, bytes(v)).map_err(|e| format!("{}: {e}", p.display()))?;
            }
        }
        if let Some(g) = golden.as_ref().filter(|_| golden_steps == vs.len()) {
            let p = g.join(format!("cpu_latent{}.f32", i + 1));
            if p.is_file() {
                let (rel, cos) = rel_cos(&x, &read_f32(&p)?);
                ok &= rel <= 1e-2 && cos >= 0.999;
                println!(
                    "READING_LATENT {}",
                    json!({"after": i + 1, "rel_l2": rel, "cosine": cos})
                );
            }
        }
    }
    Ok(ok)
}

fn run() -> Result<bool, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut get = HashMap::new();
    for pair in args.chunks(2) {
        match pair {
            [k, v] if k.starts_with("--") => {
                get.insert(k.trim_start_matches("--").to_string(), v.clone());
            }
            _ => return Err(USAGE.into()),
        }
    }
    if get.contains_key("decode") && (get.contains_key("reading") || get.contains_key("cache")) {
        return decode(&get);
    }
    if get.contains_key("reading") {
        return reading(&get);
    }
    let need = |k: &str| get.get(k).cloned().ok_or_else(|| USAGE.to_string());
    let num = |k: &str, d: usize| -> Result<usize, String> {
        get.get(k)
            .map_or(Ok(d), |v| v.parse().map_err(|e| format!("--{k} {v}: {e}")))
    };
    let program = need("program")?;
    let weights = Weights::parse(&need("weights")?)?;
    let steps = num("steps", 8)?.max(1);
    let golden = need("golden")?;
    let gold: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&golden).map_err(|e| format!("{golden}: {e}"))?,
    )
    .map_err(|e| format!("{golden}: {e}"))?;
    let ids = |k: &str| -> Result<Vec<u32>, String> {
        gold[k]
            .as_array()
            .ok_or_else(|| format!("{golden}: no {k}"))?
            .iter()
            .map(|v| v.as_u64().map(|v| v as u32).ok_or(format!("{golden}: {k}")))
            .collect()
    };
    let (prompt, want) = (ids("prompt_ids")?, ids("tokens")?);
    let n = num("tokens", want.len())?.min(want.len());
    let art = read(Path::new(&program))?;
    let platform = match get.get("cmaddr") {
        Some(c) => Platform::System { cmaddr: c.clone() },
        None => Platform::Simulator(Simulator::new(Target::Wse3, 8)),
    };
    let sdk = Sdk::open().map_err(|e| e.to_string())?;
    let mut rt =
        Runtime::new(sdk, Path::new(&program), platform, "WARNING").map_err(|e| e.to_string())?;
    rt.load();
    rt.run();
    let vocab = art.header.model.vocab as usize;
    let mut m = ArtifactModel::open(Box::new(rt), &weights, &art, vocab)?;
    let mut ok = true;
    for rep in 0..2 {
        let t0 = std::time::Instant::now();
        let run = greedy(&mut m, &prompt, n, steps)?;
        let got = &run.got;
        let mis = got.iter().zip(&want).filter(|(g, w)| g != w).count();
        let from = got.iter().zip(&want).position(|(g, w)| g != w);
        ok &= mis == 0;
        println!(
            "SEQ {rep}: {mis} mismatches of {n}{}, {:.3} ms a token, launches {} steps {} got {:?}",
            from.map_or(String::new(), |i| format!(" from {i}")),
            t0.elapsed().as_secs_f64() * 1e3 / (prompt.len() + n) as f64,
            m.launches,
            m.steps,
            &got[..n.min(16)]
        );
        if run.launches > 0 {
            let a_launch = |ns: u64| ns as f64 / 1e6 / run.launches as f64;
            println!(
                "DECODE {rep}: {:.4} ms a step over {} steps, {steps} a launch; a launch's host ms: \
                 embedding {:.3} puts {:.3} compute {:.3} ring {:.3}",
                run.ms / run.steps as f64,
                run.steps,
                a_launch(run.host[0]),
                a_launch(run.host[1]),
                a_launch(run.host[2]),
                a_launch(run.host[3]),
            );
        }
    }
    // a rollback: the wafer holds the whole sequence; a run at the prompt + 2 replays from 0
    if n > 4 {
        let at = prompt.len() + 2;
        let a2 = m.greedy(&[want[1]], prompt.len() + 1, 1)?;
        let good = a2 == want[2..4];
        ok &= good;
        println!(
            "ROLLBACK to {at}: {a2:?} want {:?} {}",
            &want[2..4],
            if good { "ok" } else { "BAD" }
        );
    }
    if let Some(probes) = stamps::read(&Path::new(&program).join("artifact"))? {
        let times = stamped(&mut m, &probes, &prompt, n, steps)?;
        println!("TIMES {times}");
        if let Some(path) = get.get("times") {
            std::fs::write(path, times.to_string()).map_err(|e| format!("{path}: {e}"))?;
        }
    }
    let (_, l) = m.forward(&prompt, 0, true)?;
    let l = l.ok_or("no logits row")?;
    let best = argmax(&l).ok_or("an empty logits row")?;
    ok &= best == want[0];
    if let Some(path) = get.get("logits") {
        let text = serde_json::to_string(&l).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))?;
    }
    let mut s = l.clone();
    s.sort_by(|a, b| b.total_cmp(a));
    println!(
        "LOGITS argmax {best} want {} (top {:.4}, margin {:.4}, rows {})",
        want[0],
        s[0],
        s[0] - s.get(1).copied().unwrap_or(s[0]),
        l.len()
    );
    Ok(ok)
}
