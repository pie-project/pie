//! The bench driver for a batched serve8b-family build, through the
//! engine's own wafer path (placement, tables, launches) without the Pie
//! runtime: `bench8b <program dir> <ckpt dir> <golden.json> <new> [cmaddr]`
//! (no cmaddr: the simulator). One lane: the golden's prompt and greedy
//! tokens, CAP - 1 passes a launch, every launch after the first timed; then
//! the tokens against the golden's (first divergence and the golden's
//! top-2 margin there). `BENCH8B_REPS` (default 2) runs it again on the
//! same pages. Prints one JSON line.

mod common;

use std::path::Path;
use std::time::Instant;

use engine_cerebras::sdk::Platform;
use engine_cerebras::spatial::serve2d::BatchLane;
use engine_cerebras::spatial::wafer::WaferModel;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let golden = common::read_json(&args[3])?;
    let new: usize = args[4].parse().map_err(|e| format!("{e}"))?;
    let t0 = Instant::now();
    let platform = match args.get(5) {
        Some(cmaddr) => Platform::System {
            cmaddr: cmaddr.clone(),
        },
        None => common::simulator(8),
    };
    let mut m = WaferModel::open(Path::new(&args[1]), Path::new(&args[2]), platform)?;
    let load_s = t0.elapsed().as_secs_f64();
    let (b_n, shape) = common::batched(&m)?;
    if b_n != 1 {
        return Err(format!("bench8b drives one lane; the build launches {b_n}"));
    }
    let cap = m.design().cap();
    let prompt: Vec<u32> = golden["prompt_ids"]
        .as_array()
        .ok_or("golden prompt_ids")?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();
    let want: Vec<u32> = golden["tokens"]
        .as_array()
        .ok_or("golden tokens")?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u32)
        .collect();
    let margins: Vec<f64> = golden["margins"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect())
        .unwrap_or_default();
    let total = prompt.len() + new;
    let ps = shape.page_size;
    let npg = (total + cap + 1).div_ceil(ps) as u32;
    if npg as usize > shape.pages() {
        return Err(format!(
            "{total} positions want {npg} pages; the pool has {}",
            shape.pages()
        ));
    }
    let pages: Vec<u32> = match std::env::var("BENCH8B_PAGEORDER").ok() {
        None => (0..npg).collect(),
        Some(spec) => ar_page_order(&spec, npg, &shape)?,
    };
    let lane = |pos: usize, forced: Vec<u32>| BatchLane {
        pos: pos as u32,
        slot: pos as u32,
        pages: pages.clone(),
        visible: None,
        forced,
        state: None,
    };
    let reps: usize = std::env::var("BENCH8B_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let mut runs = Vec::new();
    for rep in 0..reps {
        // The prompt and the first greedy tokens in one launch of CAP - 1
        // passes (a launch's most; the prompt fits), then continuations.
        let first = (cap - 1).max(prompt.len() + 1);
        let ta = Instant::now();
        let (out, _) = m.fire_batch(&[lane(0, prompt.clone())], first, false)?;
        let first_s = ta.elapsed().as_secs_f64();
        let mut got: Vec<u32> = out[0][prompt.len() - 1..].to_vec();
        let mut pos = first;
        let td = Instant::now();
        let mut launches = 0;
        while got.len() < new {
            let steps = (cap - 1).min(new - got.len());
            let (out, _) = m.fire_batch(&[lane(pos, vec![*got.last().unwrap()])], steps, false)?;
            got.extend_from_slice(&out[0][..steps]);
            pos += steps;
            launches += 1;
        }
        let decode_s = td.elapsed().as_secs_f64();
        got.truncate(new);
        let decoded = got.len() - (first - prompt.len() + 1).min(got.len());
        let diverge = got.iter().zip(&want).position(|(a, b)| a != b);
        runs.push(serde_json::json!({
            "rep": rep,
            "first_launch_s": first_s,
            "decode_launches": launches,
            "decode_tokens": decoded,
            "decode_s": decode_s,
            "tok_s": decoded as f64 / decode_s,
            "ms_token": 1e3 * decode_s / decoded.max(1) as f64,
            "first_divergence": diverge,
            "golden_margin_there": diverge.and_then(|i| margins.get(i)),
            "agree_prefix": diverge.unwrap_or(got.len().min(want.len())),
            "tokens": got,
        }));
    }
    // Teacher-forced: the prompt and the golden's tokens as every pass's
    // input, CAP - 1 a launch; each pass's argmax against the golden's next
    // token, every mismatch with the golden's top-2 margin there.
    let seq: Vec<u32> = prompt.iter().chain(&want).copied().collect();
    let mut mismatches = Vec::new();
    let mut at = 0;
    while at + 1 < seq.len() {
        let n = (cap - 1).min(seq.len() - 1 - at);
        let (out, _) = m.fire_batch(&[lane(at, seq[at..at + n].to_vec())], n, false)?;
        for (i, &got) in out[0].iter().enumerate().take(n) {
            let t = at + i + 1;
            if t >= prompt.len() && got != seq[t] {
                let k = t - prompt.len();
                mismatches.push(serde_json::json!({"token": k, "got": got, "want": seq[t], "margin": margins.get(k)}));
            }
        }
        at += n;
    }
    // BENCH8B_PROF=<launches>: the profile counters (a TIME 1 build) over
    // that many decode launches of CAP - 1 passes from position 256 (the
    // golden's tokens forced), zeroed first; each unit's and the head's
    // per-slot max and mean over its PEs, a timed pass (CAP - 2 a launch).
    let mut profile = serde_json::Value::Null;
    if let Some(k) = std::env::var("BENCH8B_PROF")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        let mut run = || -> Result<serde_json::Value, String> {
            m.profile(true)?;
            let at0 = 256.min(seq.len().saturating_sub(cap));
            // (every pass's position inside the lane's pages)
            let k = k
                .min((npg as usize * ps).saturating_sub(at0 + 1) / (cap - 1))
                .max(1);
            let tp = Instant::now();
            for j in 0..k {
                let at = at0 + j * (cap - 1);
                let forced: Vec<u32> = (0..cap - 1).map(|i| seq[(at + i) % seq.len()]).collect();
                m.fire_batch(&[lane(at, forced)], cap - 1, false)?;
            }
            let wall = tp.elapsed().as_secs_f64();
            let timed = (k * (cap - 2)) as f64;
            let units: Vec<serde_json::Value> = m
                .profile(false)?
                .into_iter()
                .map(|(r, mx, mean)| {
                    serde_json::json!({
                        "rect": [r.x, r.y, r.w, r.h],
                        "max": mx.iter().map(|&v| v as f64 / timed).collect::<Vec<_>>(),
                        "mean": mean.iter().map(|&v| v / timed).collect::<Vec<_>>(),
                    })
                })
                .collect();
            // BENCH8B_HEADMAP=1: the head's slots 0-5 averaged over each column
            // and each row (where the head's max comes from).
            let mut headmap = serde_json::Value::Null;
            if std::env::var("BENCH8B_HEADMAP").is_ok_and(|v| v == "1") {
                let (a, w) = m.profile_head_raw()?;
                let h = a.len() / 16 / w;
                let mut cols = vec![vec![0f64; w]; 6];
                let mut rows = vec![vec![0f64; h]; 6];
                let mut rmax = vec![vec![0f64; h]; 6];
                for y in 0..h {
                    for x in 0..w {
                        for s in 0..6 {
                            let v = a[(y * w + x) * 16 + s] as f64 / timed;
                            cols[s][x] += v / h as f64;
                            rows[s][y] += v / w as f64;
                            rmax[s][y] = rmax[s][y].max(v);
                        }
                    }
                }
                let r1 = |v: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
                    v.iter()
                        .map(|r| r.iter().map(|x| x.round()).collect())
                        .collect()
                };
                headmap = serde_json::json!({"cols": r1(&cols), "rows": r1(&rows), "rows_max": r1(&rmax)});
            }
            Ok(serde_json::json!({
                "headmap": headmap,
                "launches": k, "timed_passes": timed, "start": at0,
                "wall_s": wall, "ms_pass": 1e3 * wall / (k * (cap - 1)) as f64,
                "units": units,
            }))
        };
        profile = run().unwrap_or_else(|e| serde_json::json!({"error": e}));
    }
    println!(
        "{}",
        serde_json::json!({
            "program": args[1], "cap": cap, "load_s": load_s,
            "prompt_len": prompt.len(), "new": new, "runs": runs,
            "teacher": {"tokens": want.len(), "mismatches": mismatches},
            "profile": profile,
        })
    );
    Ok(())
}

/// `BENCH8B_PAGEORDER=ar:<py>:<gy>`: logical page i on
/// [`engine_cerebras::spatial::paged::RowOrder`]'s page for i (the rows by
/// o/l all-reduce slack; positions keep their order, only rows move).
fn ar_page_order(
    spec: &str,
    npg: u32,
    shape: &engine_cerebras::spatial::paged::Shape,
) -> Result<Vec<u32>, String> {
    let o = engine_cerebras::spatial::paged::RowOrder::parse(spec, shape)?;
    eprintln!(
        "BENCH8B_PAGEORDER rows first: {:?}",
        &o.rows()[..8.min(o.rows().len())]
    );
    Ok((0..npg).map(|i| o.map(i)).collect())
}
