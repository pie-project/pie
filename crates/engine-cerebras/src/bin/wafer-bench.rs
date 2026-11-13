//! `wafer-bench`: a linked wafer program run bare (no Pie), on the host code Pie's engine runs
//! (`ArtifactModel`), against a golden: two greedy sequences through the wafer token loop from
//! position 0 on one load (the state resets), a rollback (replayed from 0) and a logits row (its
//! argmax is the first token), with the time a token. On the simulator, or the chip at
//! `--cmaddr`.
//!
//! ```text
//! wafer-bench --program <program out dir> --weights <dir>[:<prefix>] --golden <golden.json>
//!             [--steps N] [--tokens N] [--cmaddr <addr>] [--logits <out.json>] [--times <out.json>]
//!             [--forced N]
//! ```
//!
//! `--steps`: tokens a launch (default 8); `--tokens`: the golden's first N (default all);
//! `--logits`: where to write the prompt's last logits row (a JSON array).
//! A greedy sequence's mismatches count every token past where it left the golden (`from`): one
//! flipped near-tie diverges the rest. `--forced N` also feeds the golden's first N tokens one a
//! launch (teacher-forced) and prints each step whose argmax is not the golden's next token, with
//! how far the golden's token is behind (`FORCED`; informational, not a check).
//! A program built with `TSTAMP:1` (its artifact's `stamps.json`) also gets a third sequence
//! with every probe's `tdbg` read after each launch: `TIMES` prints the cycles (each unit's
//! layers and send, each probe's phases from its pass start (`null`: a phase that path never
//! stamps; `<probe>.counts`: a kernel's counters, as read), the token loop's steps), also
//! written to `--times`.
//! Prints `WAFER_BENCH ok` and exits 0 when every check holds.

use std::collections::HashMap;
use std::path::Path;

use engine_cerebras::artifact::{ArtifactModel, Weights, argmax, read};
use engine_cerebras::sdk::{Platform, Rect, Runtime, Sdk, Simulator, Target};
use kernels_cerebras::emit::stamps::{self, Probe, Read};
use serde_json::{Value, json};

const USAGE: &str = "usage: wafer-bench --program <dir> --weights <dir>[:<prefix>] --golden \
                     <golden.json> [--steps N] [--tokens N] [--cmaddr <addr>] [--logits <out.json>] \
                     [--times <out.json>] [--forced N]";

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

/// A greedy run's tokens, and its decode launches' (the ones after the prompt's): their wall ms,
/// steps and count, and their host ns by part ([`ArtifactModel::host_ns`]). Decode is the rate
/// Pie's posted loops are timed at; the run's ms a token also holds the prompt's one-pass launches.
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

/// The golden's first `n` tokens fed one a launch after `prompt`: each step whose argmax is not
/// the next golden token, as (step, got, want, the logit gap got - want).
fn forced(
    m: &mut ArtifactModel,
    prompt: &[u32],
    want: &[u32],
    n: usize,
) -> Result<Vec<(usize, u32, u32, f32)>, String> {
    let mut miss = Vec::new();
    for i in 0..n {
        let (_, l) = if i == 0 {
            m.forward(prompt, 0, true)?
        } else {
            m.forward(&want[i - 1..i], prompt.len() + i - 1, true)?
        };
        let l = l.ok_or("no logits row")?;
        let got = argmax(&l).ok_or("an empty logits row")?;
        if got != want[i] {
            miss.push((i, got, want[i], l[got as usize] - l[want[i] as usize]));
        }
    }
    Ok(miss)
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
/// the launches after the first (the prompt's), averaged.
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
        None => Platform::Simulator(Simulator {
            num_threads: 8,
            ..Simulator::new(Target::Wse3)
        }),
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
    let nf = num("forced", 0)?.min(n);
    if nf > 0 {
        let miss = forced(&mut m, &prompt, &want, nf)?;
        let at: Vec<String> = miss
            .iter()
            .map(|(i, g, w, d)| format!("{i}: {g} want {w} ({d:+.4})"))
            .collect();
        println!(
            "FORCED {} mismatches of {nf}{}{}",
            miss.len(),
            if at.is_empty() { "" } else { ": " },
            at.join(", ")
        );
    }
    if let Some(probes) = stamps::read(&Path::new(&program).join("artifact"))? {
        let mut times = stamped(&mut m, &probes, &prompt, n, steps)?;
        if get.contains_key("again") {
            let again = stamped(&mut m, &probes, &prompt, n, steps)?;
            times["again"] = again["phases_mean"]["loop_every"].clone();
        }
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
