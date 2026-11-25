//! `pie tune` on Metal: kernel benches fit `kernel.qmm`, `kernel.qmm_fp16`
//! and `kernel.dense` ([`dense`]); the other searchable decisions are
//! searched end to end on the served models.

pub(crate) mod dense;

use std::cell::RefCell;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use kernels_metal::linear::qmm;
use kernels_metal::tuning::{Qmm, QmmFp16};
use kernels_metal::{ArgValue, Encode, Error, Fire, Tensor, Tuning};
use poem_ir::Dtype;
use profile::search::{self, Bench, CONTEXT, Check, Entries, LANES, Line, PROMPT, STEPS, Serving};
use profile::tune::{Candidate, Options, Sample};
use profile::{Decision, Facts, Profile, Source};

use crate::device::{Buffer, Context, Handles, Pipelines};
use crate::encode::Sink;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Depth {
    Quick,
    Full,
}

impl Depth {
    fn word(self) -> &'static str {
        match self {
            Depth::Quick => "quick",
            Depth::Full => "full",
        }
    }
}

/// `fp16`: 4-bit, 64-group codes read through the fp16 GEMM format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bank {
    bits: u32,
    group: u32,
    fp16: bool,
}

/// The catalog's quantised bank shapes, `(n, k, banks)`, heaviest
/// (`n * k * banks`) first; `the_census_is_the_catalogs` keeps it current.
const CENSUS: &[(u32, u32, u32)] = &[
    (34816, 5120, 777),
    (5120, 17408, 777),
    (43008, 5376, 270),
    (16384, 5120, 540),
    (36864, 8192, 142),
    (57344, 8192, 80),
    (14336, 5120, 458),
    (4096, 8192, 958),
    (5376, 21504, 270),
    (24576, 4096, 300),
    (8192, 16384, 193),
    (6144, 16384, 241),
    (16384, 6144, 241),
    (5120, 6144, 722),
    (20480, 4096, 253),
    (39936, 6656, 76),
    (16384, 5376, 225),
    (8192, 28672, 80),
    (5120, 12288, 285),
    (51200, 5120, 64),
    (34816, 8192, 47),
    (4096, 4096, 767),
    (30720, 3840, 96),
    (6656, 19968, 76),
];

struct Grid {
    rows: Vec<u32>,
    /// `(n, k, weight)`, weighted by the shape's share of the census bytes.
    shapes: Vec<(u32, u32, f64)>,
    banks: Vec<Bank>,
}

impl Grid {
    fn of(depth: Depth) -> Grid {
        let fp16 = Bank {
            bits: 4,
            group: 64,
            fp16: true,
        };
        let bank = |bits, group| Bank {
            bits,
            group,
            fp16: false,
        };
        let shapes = |taken: Vec<(u32, u32, u32)>| {
            let total: f64 = taken
                .iter()
                .map(|&(n, k, c)| f64::from(n) * f64::from(k) * f64::from(c))
                .sum();
            taken
                .iter()
                .map(|&(n, k, c)| (n, k, f64::from(n) * f64::from(k) * f64::from(c) / total))
                .collect::<Vec<_>>()
        };
        match depth {
            Depth::Quick => Grid {
                rows: vec![1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 64, 128, 512],
                // Small shapes too: a crossover learned on large banks alone
                // misplaces small ones.
                shapes: shapes(
                    [0, 1, 2, 3, 9, 14, 18, 21]
                        .iter()
                        .map(|&i| CENSUS[i])
                        .collect(),
                ),
                banks: vec![fp16, bank(4, 64), bank(4, 32), bank(8, 64), bank(2, 64)],
            },
            Depth::Full => Grid {
                rows: (1..=16)
                    .chain([20, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048])
                    .collect(),
                shapes: shapes(CENSUS.to_vec()),
                banks: vec![
                    fp16,
                    bank(4, 64),
                    bank(4, 32),
                    bank(8, 64),
                    bank(2, 32),
                    bank(2, 64),
                    bank(2, 128),
                ],
            },
        }
    }
}

/// The standard workload mix: how much a point counts. A 4-bit 64-group
/// bank in the format `qmm_fp16` does not pick counts a tenth on top.
fn weight(grid: &Grid, rows: u32, n: u32, k: u32, bank: Bank, fp16_on: bool) -> f64 {
    let band = |lo: u32, hi: u32| grid.rows.iter().filter(|&&r| lo <= r && r <= hi).count();
    let per_rows = match rows {
        0..=8 => 0.5 / band(0, 8) as f64,
        9..=64 => 0.2 / band(9, 64) as f64,
        _ => 0.3 / band(65, u32::MAX) as f64,
    };
    let kind = |b: &Bank| match (b.bits, b.group) {
        (4, 64) => 0,
        (4, _) => 1,
        (8, _) => 2,
        _ => 3,
    };
    let share = [0.5, 0.1, 0.15, 0.25][kind(&bank)];
    let alike = grid
        .banks
        .iter()
        .filter(|b| kind(b) == kind(&bank) && b.fp16 == bank.fp16)
        .count()
        .max(1);
    let unchosen = if (bank.bits, bank.group) == (4, 64) && bank.fp16 != fp16_on {
        0.1
    } else {
        1.0
    };
    let per_shape = grid
        .shapes
        .iter()
        .find(|s| (s.0, s.1) == (n, k))
        .map_or(0.0, |s| s.2);
    per_rows * share * unchosen / alike as f64 * per_shape
}

/// A spread of constants, and every rung value `base` holds, that a GPU of
/// `family` answers right on.
fn candidates(base: &Profile, family: u32) -> Vec<Qmm> {
    let mut out = vec![Qmm::Mpp];
    for crossover_tg in [64, 96, 128, 160, 256] {
        out.push(Qmm::Tiled { crossover_tg });
    }
    for (rows_max, packs, crossover_tg) in [(8, 1, 160), (8, 1, 96), (4, 1, 160), (8, 2, 160)] {
        out.push(Qmm::Qmv {
            rows_max,
            packs,
            crossover_tg,
        });
    }
    out.push(Qmm::Staged);
    out.push(Qmm::Register);
    for q in base.ladder::<Qmm>().values() {
        if !out.contains(q) {
            out.push(*q);
        }
    }
    out.retain(|q| q.runs_on(family));
    out
}

fn body(q: &Qmm) -> String {
    match *q {
        Qmm::Mpp => "use = \"mpp\"".to_string(),
        Qmm::Staged => "use = \"staged\"".to_string(),
        Qmm::Register => "use = \"register\"".to_string(),
        Qmm::Tiled { crossover_tg } => format!("use = \"tiled\", crossover_tg = {crossover_tg}"),
        Qmm::Qmv {
            rows_max,
            packs,
            crossover_tg,
        } => format!(
            "use = \"qmv\", rows_max = {rows_max}, packs = {packs}, crossover_tg = {crossover_tg}"
        ),
    }
}

/// The `kernel.qmm` ladder that runs `q` wherever it can, and the vector
/// kernel elsewhere.
fn forcing(q: &Qmm) -> String {
    match q {
        Qmm::Qmv { .. } => format!("[{{ {} }}]", body(q)),
        _ => format!(
            "[{{ {} }}, {{ use = \"qmv\", rows_max = 8, packs = 1, crossover_tg = 160 }}]",
            body(q)
        ),
    }
}

/// Whether a launch that fired `files` ran `q`'s kernel.
fn ran(files: &[&str], q: &Qmm) -> bool {
    let has = |name: &str| files.iter().any(|f| f.contains(name));
    match q {
        Qmm::Mpp => has("linear/qmm_mpp.metal"),
        Qmm::Staged => has("linear/qmm_staged.metal"),
        Qmm::Register => has("linear/qmm_sg.metal"),
        Qmm::Tiled { .. } => {
            !has("linear/qmm_mpp.metal")
                && !has("linear/qmm_staged.metal")
                && !has("linear/qmm_sg.metal")
                && has("linear/qmm_t.metal")
        }
        Qmm::Qmv { .. } => {
            !has("linear/qmm_mpp.metal")
                && !has("linear/qmm_staged.metal")
                && !has("linear/qmm_sg.metal")
                && !has("linear/qmm_t.metal")
        }
    }
}

#[derive(Clone, Debug)]
struct Point {
    rows: u32,
    n: u32,
    k: u32,
    bank: Bank,
    /// Seconds per launch for each candidate; `None` where it did not run.
    times: Vec<Option<f64>>,
    /// The candidate today's profile runs here.
    baseline: Option<usize>,
    /// The timed candidates' median errors pooled, relative
    /// ([`profile::tune::Sample::spread`]).
    spread: f64,
}

#[derive(Clone, Debug, Default)]
struct Measured {
    candidates: Vec<Qmm>,
    points: Vec<Point>,
    /// `(candidate, points it answered wrongly at, the first)`.
    wrong: Vec<(String, usize, String)>,
}

impl Measured {
    /// What `pie tune --refit` reads back.
    fn to_text(&self, device: &str) -> String {
        let mut out = format!("# pie tune measurements, kernel.qmm, {device}\n");
        for q in &self.candidates {
            let _ = writeln!(out, "candidate {}", body(q));
        }
        for (b, n, first) in &self.wrong {
            let _ = writeln!(out, "wrong {n} {b} | {first}");
        }
        for p in &self.points {
            let times: Vec<String> = p
                .times
                .iter()
                .map(|t| t.map_or_else(|| "-".to_string(), |t| format!("{t:e}")))
                .collect();
            let _ = writeln!(
                out,
                "point {} {} {} {} {} {} {} {} {:e}",
                p.rows,
                p.n,
                p.k,
                p.bank.bits,
                p.bank.group,
                u8::from(p.bank.fp16),
                p.baseline
                    .map_or_else(|| "-".to_string(), |b| b.to_string()),
                times.join(","),
                p.spread
            );
        }
        out
    }

    /// A file measured over another candidate list is refused.
    fn from_text(text: &str, candidates: &[Qmm]) -> Result<Measured, String> {
        let bodies: Vec<String> = candidates.iter().map(body).collect();
        let mut seen = Vec::new();
        let mut out = Measured {
            candidates: candidates.to_vec(),
            ..Measured::default()
        };
        for line in text.lines() {
            if let Some(b) = line.strip_prefix("candidate ") {
                seen.push(b.to_string());
            } else if let Some(rest) = line.strip_prefix("wrong ") {
                let (count, rest) = rest.split_once(' ').ok_or("a wrong line")?;
                let (b, first) = rest.split_once(" | ").ok_or("a wrong line")?;
                out.wrong.push((
                    b.to_string(),
                    count.parse().map_err(|_| "a count")?,
                    first.to_string(),
                ));
            } else if let Some(rest) = line.strip_prefix("point ") {
                let f: Vec<&str> = rest.split(' ').collect();
                let num = |i: usize| -> Result<u32, String> {
                    f.get(i)
                        .and_then(|v| v.parse().ok())
                        .ok_or(format!("field {i} of `{line}`"))
                };
                let times: Vec<Option<f64>> = f
                    .get(7)
                    .ok_or("times")?
                    .split(',')
                    .map(|t| t.parse().ok())
                    .collect();
                out.points.push(Point {
                    rows: num(0)?,
                    n: num(1)?,
                    k: num(2)?,
                    bank: Bank {
                        bits: num(3)?,
                        group: num(4)?,
                        fp16: num(5)? == 1,
                    },
                    baseline: f.get(6).and_then(|b| b.parse().ok()),
                    times,
                    spread: f.get(8).and_then(|v| v.parse().ok()).ok_or(format!(
                        "`{line}` states no spread (measured by an older pie); measure again"
                    ))?,
                });
            }
        }
        if seen != bodies {
            return Err(
                "the measurements were taken over another candidate list; measure again".into(),
            );
        }
        Ok(out)
    }
}

/// Decisions the end-to-end search leaves alone: the kernel bench's, and
/// what means nothing on one queue on one device.
const SEARCH_SKIPS: &[&str] = &[
    "kernel.qmm",
    "kernel.qmm_fp16",
    "kernel.dense",
    "compiler.side_streams",
    "compiler.cost.collective_us",
];

fn fit(
    measured: &Measured,
    grid: &Grid,
    cores: Option<u32>,
) -> Result<(Entries, Vec<Line>), String> {
    let mut lines = Vec::new();
    // `qmm_fp16`: whether the fp16 format's best beats the emulated one's.
    let best = |p: &Point| {
        p.times
            .iter()
            .flatten()
            .copied()
            .fold(f64::INFINITY, f64::min)
    };
    let mut fp16_cost = 0.0;
    let mut emulated_cost = 0.0;
    for p in measured.points.iter().filter(|p| p.bank.fp16) {
        let Some(twin) = measured.points.iter().find(|q| {
            !q.bank.fp16
                && (q.bank.bits, q.bank.group) == (4, 64)
                && (q.rows, q.n, q.k) == (p.rows, p.n, p.k)
        }) else {
            continue;
        };
        let (a, b) = (best(p), best(twin));
        if a.is_finite() && b.is_finite() {
            let w = weight(grid, p.rows, p.n, p.k, p.bank, true);
            let floor = a.min(b);
            fp16_cost += w * a / floor;
            emulated_cost += w * b / floor;
        }
    }
    let fp16_on = fp16_cost <= emulated_cost || emulated_cost == 0.0;
    lines.push(Line {
        id: QmmFp16::spec().id,
        text: if emulated_cost == 0.0 {
            "on: no 4-bit, 64-group point measured in both formats".to_string()
        } else {
            format!(
                "{}: fp16 format at {:.3}x the per-point best, emulated at {:.3}x",
                if fp16_on { "on" } else { "off" },
                fp16_cost / fp16_cost.min(emulated_cost),
                emulated_cost / fp16_cost.min(emulated_cost),
            )
        },
        measured: emulated_cost != 0.0,
    });
    let fp16 = (QmmFp16::spec().id.to_string(), fp16_on.to_string());

    let spec = Qmm::spec();
    let Some(cores) = cores else {
        lines.push(Line {
            id: spec.id,
            text: "not searched: the device does not say its GPU cores, so `fill` is unknown"
                .to_string(),
            measured: false,
        });
        return Ok((vec![fp16], lines));
    };
    let candidates: Vec<Candidate> = measured
        .candidates
        .iter()
        .map(|q| Candidate {
            body: body(q),
            terminal: matches!(q, Qmm::Qmv { .. }),
        })
        .collect();
    // Coordinates in `kernel.qmm`'s variable order.
    let samples: Vec<Sample> = measured
        .points
        .iter()
        .map(|p| Sample {
            at: vec![
                i64::from(p.rows),
                kernels_metal::tuning::fill(p.rows, p.n, cores),
                i64::from(p.n),
                i64::from(p.k),
                i64::from(p.bank.bits),
                i64::from(p.bank.group),
                i64::from(!p.bank.fp16),
            ],
            weight: weight(grid, p.rows, p.n, p.k, p.bank, fp16_on),
            times: p.times.clone(),
            spread: p.spread,
        })
        .collect();
    let baseline: Vec<Option<usize>> = measured.points.iter().map(|p| p.baseline).collect();
    let fit = profile::tune::fit(spec, &candidates, &samples, &baseline, Options::default())
        .map_err(|e| format!("{}: {e}", spec.id))?;
    let ladder = fit.toml(spec, &candidates);
    let mut text = format!(
        "{} rungs, at {:.3}x the per-point best",
        fit.rungs.len(),
        fit.cost / fit.floor
    );
    if let Some(b) = fit.baseline {
        let _ = write!(text, "; today's ladder runs at {:.3}x", b / fit.floor);
    }
    lines.push(Line {
        id: spec.id,
        text,
        measured: true,
    });
    let entries = vec![(spec.id.to_string(), ladder), fp16];
    Ok((entries, lines))
}

fn assemble(
    base: &Profile,
    device: &str,
    facts: &Facts,
    tuned: &str,
    provenance: &[(&str, String)],
) -> Result<String, String> {
    profile::search::assemble(
        crate::chip::registry(),
        base,
        device,
        facts,
        tuned,
        provenance,
        &["family", "ane"],
    )
}

/// The user's file of `device` under `profiles`, less what this pie no
/// longer reads: where loads store what they measure, each under its key.
fn user_file(profiles: Option<&Path>, device: &str) -> Option<String> {
    let text = std::fs::read_to_string(crate::chip::user_path(profiles?, device)).ok()?;
    Some(crate::chip::registry().prune(&text).0)
}

/// `pie tune metal --rates`: this device's `[rates]` measured through
/// Metal's own kernels ([`crate::rates::measure`]) at the shapes every
/// backend times, keyed and saved over the user's file as a load saves what
/// it measures (`engine::rates::saved_text`).
pub fn rates(
    profiles: Option<&Path>,
    say: &mut dyn FnMut(&str),
) -> Result<engine::rates::Rated, String> {
    let began = Instant::now();
    let device = Context::bind().map_err(|e| e.to_string())?;
    let name = device.name().to_string();
    let facts = crate::chip::measured(&device, crate::chip::perf_cores());
    let (base, report) = crate::chip::resolve(&name, &facts, profiles);
    say(&format!(
        "{name}: measuring the rates, starting from {}",
        report.used
    ));
    let rates = crate::rates::measure(&device, base.clone(), say)?;
    let tuned = engine::rates::tuned(&rates);
    let seconds = began.elapsed().as_secs_f64();
    let provenance = [
        (
            "source",
            "\"pie tune metal --rates on this machine\"".to_string(),
        ),
        crate::rates::provenance(&name),
        ("from", format!("\"{}\"", report.used)),
        ("pie", format!("\"{}\"", env!("CARGO_PKG_VERSION"))),
        ("seconds", format!("{seconds:.0}")),
    ];
    let file = engine::rates::saved_text(
        user_file(profiles, &name).as_deref(),
        &rates,
        &crate::rates::provenance(&name),
    );
    let block = format!(
        "{}\n[provenance]\n{}\n",
        tuned.trim_end(),
        provenance
            .iter()
            .map(|(key, value)| format!("rates.{key} = {value}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    Ok(engine::rates::Rated {
        device: name,
        file,
        block,
        seconds,
    })
}

pub struct Outcome {
    pub device: String,
    pub file: String,
    pub lines: Vec<Line>,
    pub wrong: Vec<String>,
    /// Today's profile against the tuned one, end to end.
    pub checks: Vec<Check>,
    pub changed: bool,
    pub seconds: f64,
}

/// Serve `artifact` under the profile `file`: a prompt, then steps at each
/// lane count, timed once the Neural Engine's programs are ready.
fn serve(artifact: &Path, deployment: &str, device: &str, file: &str) -> Result<Serving, String> {
    use poem::{Platform, Request};
    let slots = *LANES.iter().max().expect("lanes") as u32;
    let dir = std::env::temp_dir().join(format!("pie-tune-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = crate::chip::user_path(&dir, device);
    std::fs::create_dir_all(path.parent().expect("a parent")).map_err(|e| e.to_string())?;
    std::fs::write(&path, file).map_err(|e| e.to_string())?;
    let row = poem_compiler::catalog::deployment(deployment)
        .ok_or_else(|| format!("no catalog row `{deployment}`"))?;
    let trace = row.trace(Platform::Metal);
    let source = ztensor_compat::index(artifact).map_err(|e| e.to_string())?;
    let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Metal)
        .map_err(|e| e.to_string())?;
    drop(source);
    let mut shell = crate::Shell::load(crate::Boot {
        voxels: None,
        trace: trace.clone(),
        contract: &contract,
        checkpoint: artifact,
        budget: poem_compiler::Budget::new(slots, CONTEXT),
        patches: patch_ladder(&trace),
        profile: None,
        page_size: 16,
        context: CONTEXT,
        slots,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        runahead: engine::runahead::Runahead::F1,
        profiles: Some(dir.clone()),
        residency: crate::ResidencyPlan::default(),
        top: crate::Top::EveryRung,
        measure: engine::rates::Measuring::Time,
    })
    .map_err(|e| e.to_string())?;
    // No runtime raises the rung here: the widest carve is backed.
    shell.raise_whole().map_err(|e| e.to_string())?;
    // Time the load as it serves, not the GPU-alone stretch before the
    // Neural Engine's programs compile.
    let mut refused = None;
    if let Some(ane) = shell.neural_engine() {
        let waited = Instant::now();
        loop {
            match ane.compiled() {
                Some(Ok(())) => break,
                Some(Err(why)) => {
                    refused = Some(why.lines().next().unwrap_or("").chars().take(100).collect());
                    break;
                }
                None if waited.elapsed().as_secs() > 1200 => {
                    return Err("the Neural Engine programs did not compile in 20 minutes".into());
                }
                None => std::thread::sleep(std::time::Duration::from_millis(250)),
            }
        }
    }
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
            let ms = began.elapsed().as_secs_f64() * 1e3;
            Ok((ms, out.into_iter().next().unwrap_or_default()))
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
    // Every slot holds a short history, then steps one token a lane.
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
    let read = shell.tuning().profile().read();
    drop(shell);
    let _ = std::fs::remove_dir_all(&dir);
    Ok(Serving {
        prompt: prefill,
        steps: decode,
        readout,
        read,
        refused,
        layers: trace.decoder_layers(),
    })
}

/// A hash of the fused trace and compiled plan a load under `text` runs:
/// two files with the same plan serve the same.
fn plan_of(deployment: &str, text: &str, cores: Option<u32>) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    let profile = crate::chip::registry().load_user(text, "plan").ok()?;
    let row = poem_compiler::catalog::deployment(deployment)?;
    let trace = poem_compiler::fuse::fuse(
        row.trace(poem::Platform::Metal),
        poem_compiler::fuse::flags::on(&profile, &crate::FUSIONS),
    );
    let slots = *LANES.iter().max().expect("lanes") as u32;
    let mut budgets = poem_compiler::Budgets::of(poem_compiler::Budget::new(slots, CONTEXT));
    if let Some(ladder) = patch_ladder(&trace) {
        budgets = budgets.with_patches(ladder);
    }
    let compiled = poem_compiler::compile_axes(
        &trace,
        &budgets,
        &poem_compiler::DeviceProfile::of(
            &profile,
            crate::chip::pricing_cores(&profile, cores),
            &[],
            &[],
        ),
    )
    .ok()?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("{trace:?}{compiled:?}").hash(&mut h);
    Some(h.finish())
}

/// A vision row states its patch ladder even for a text prompt.
fn patch_ladder(trace: &poem_ir::Trace) -> Option<poem_compiler::PatchLadder> {
    crate::api::patch_ladder(
        trace,
        &engine::load::Budgets {
            max_lanes: *LANES.iter().max().expect("lanes") as u32,
            max_tokens: CONTEXT,
            ..engine::load::Budgets::default()
        },
    )
}

struct Metal<'a> {
    models: &'a [(PathBuf, String)],
    device: &'a str,
    base: &'a Profile,
    facts: &'a profile::Facts,
    cores: Option<u32>,
    mix: search::Mix,
}

impl Bench for Metal<'_> {
    fn mix(&self) -> search::Mix {
        self.mix
    }

    fn agreement(&self) -> f64 {
        engine::rates::AGREEMENT
    }

    fn models(&self) -> Vec<String> {
        self.models.iter().map(|(_, d)| d.clone()).collect()
    }

    fn file(&self, entries: &[(String, String)]) -> Result<String, String> {
        assemble(
            self.base,
            self.device,
            self.facts,
            &search::render_overrides(entries),
            &[],
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

/// An `Encode` that remembers the files it fired.
struct Watch<'a> {
    inner: &'a dyn Encode,
    files: RefCell<Vec<&'static str>>,
}

impl Encode for Watch<'_> {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
        self.files.borrow_mut().push(fire.file);
        self.inner.fire(fire, args)
    }

    fn absent(&self) -> Result<ArgValue, Error> {
        self.inner.absent()
    }
}

fn noise(at: u64) -> u8 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 40) as u8
}

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    [(bits >> 16) as u8, (bits >> 24) as u8]
}

fn seen(v: f32) -> f64 {
    f64::from(f32::from_bits(v.to_bits() & 0xFFFF_0000))
}

fn to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits((u32::from(hi) << 24) | (u32::from(lo) << 16))
}

fn dtype(bits: u32, group: u32) -> Dtype {
    match (bits, group) {
        (4, 32) => Dtype::U4g32,
        (8, _) => Dtype::U8g64,
        (2, 32) => Dtype::U2g32,
        (2, 128) => Dtype::U2g128,
        (2, _) => Dtype::U2g64,
        _ => Dtype::U4g64,
    }
}

/// A bank in enough copies to stream past the last-level cache, with its
/// host values for the reference.
struct Weights {
    codes: Vec<u32>,
    scales: Vec<u32>,
    biases: Vec<u32>,
    q: Vec<u8>,
    scale: Vec<f64>,
    bias: Vec<f64>,
    _buffers: Vec<Buffer>,
}

/// More than any Apple GPU's system-level cache.
const STREAM_BYTES: u64 = 256 << 20;
const SHAPE_BYTES: u64 = 1 << 30;

fn weights(
    device: &Context,
    handles: &Handles,
    n: u32,
    k: u32,
    bits: u32,
    group: u32,
) -> Result<Weights, String> {
    let (nn, kk) = (n as usize, k as usize);
    let groups = kk / group as usize;
    let mut q = vec![0u8; nn * kk];
    let levels = 1u16 << bits;
    for (i, c) in q.iter_mut().enumerate() {
        *c = (u16::from(noise(i as u64 ^ 0x5151)) % levels) as u8;
    }
    let row_bytes = kk * bits as usize / 8;
    let mut packed = vec![0u8; nn * row_bytes];
    for col in 0..nn {
        let out = &mut packed[col * row_bytes..(col + 1) * row_bytes];
        let mut at = 0usize;
        for &c in &q[col * kk..(col + 1) * kk] {
            for b in 0..bits {
                if (c >> b) & 1 == 1 {
                    out[at / 8] |= 1 << (at % 8);
                }
                at += 1;
            }
        }
    }
    let mut scale = vec![0.0; nn * groups];
    let mut bias = vec![0.0; nn * groups];
    let mut sc = vec![0u8; nn * groups * 2];
    let mut bi = vec![0u8; nn * groups * 2];
    for i in 0..nn * groups {
        let s = 0.01 + 0.001 * f32::from(noise(i as u64 ^ 0xAA) % 8);
        let b = -0.05 + 0.01 * f32::from(noise(i as u64 ^ 0x55) % 8);
        sc[2 * i..2 * i + 2].copy_from_slice(&bf16(s));
        bi[2 * i..2 * i + 2].copy_from_slice(&bf16(b));
        scale[i] = seen(s);
        bias[i] = seen(b);
    }
    let bank_bytes = (packed.len() + sc.len() + bi.len()) as u64;
    if bank_bytes > SHAPE_BYTES / 2 {
        return Err(format!("{n} x {k} at {bits} bits is past the shape budget"));
    }
    let copies = (STREAM_BYTES.div_ceil(bank_bytes)).clamp(1, SHAPE_BYTES / bank_bytes);
    let mut buffers = Vec::new();
    let mut upload = |bytes: &[u8]| -> Result<u32, String> {
        let mut b = Buffer::zeroed(device, bytes.len() as u64).map_err(|e| e.to_string())?;
        b.write(0, bytes).map_err(|e| e.to_string())?;
        let h = handles.bind(&b, 0, b.bytes()).map_err(|e| e.to_string())?;
        buffers.push(b);
        Ok(h)
    };
    let (mut codes, mut scales, mut biases) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..copies {
        codes.push(upload(&packed)?);
        scales.push(upload(&sc)?);
        biases.push(upload(&bi)?);
    }
    Ok(Weights {
        codes,
        scales,
        biases,
        q,
        scale,
        bias,
        _buffers: buffers,
    })
}

/// Measure, fit, search end to end on `models` (path, deployment), and
/// assemble the file. Measurements are kept under `cache`; `refit` reads
/// them back instead of measuring.
pub fn run(
    depth: Depth,
    profiles: Option<&Path>,
    cache: &Path,
    refit: bool,
    models: &[(PathBuf, String)],
    mix: search::Mix,
    say: &mut dyn FnMut(&str),
) -> Result<Outcome, String> {
    let began = Instant::now();
    let device = Context::bind().map_err(|e| e.to_string())?;
    let name = device.name().to_string();
    let cores = device.cores();
    let facts = crate::chip::measured(&device, crate::chip::perf_cores());
    let (base, report) = crate::chip::resolve(&name, &facts, profiles);
    say(&format!(
        "{name}: {}, family {}; starting from {}",
        match (cores, device.cores_from()) {
            (Some(cores), Some(from)) => format!("{cores} GPU cores ({from})"),
            _ => "GPU cores unreported".to_string(),
        },
        device.family(),
        report.used
    ));
    let grid = Grid::of(depth);
    let kept = cache
        .join(profile::id_of(&name))
        .join(format!("qmm-{}.txt", depth.word()));
    let measured = if refit {
        let text = std::fs::read_to_string(&kept)
            .map_err(|e| format!("no measurements at {}: {e}", kept.display()))?;
        say(&format!("refitting the measurements in {}", kept.display()));
        Measured::from_text(&text, &candidates(&base, device.family()))?
    } else {
        let m = measure(&device, &base, cores, &grid, say)?;
        if let Some(dir) = kept.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(&kept, m.to_text(&name)).map_err(|e| e.to_string())?;
        say(&format!("measurements kept in {}", kept.display()));
        m
    };
    let (mut fitted, mut lines) = fit(&measured, &grid, cores)?;
    // `kernel.dense`: its own bench, kept beside the quantised one's.
    let kept_dense = cache
        .join(profile::id_of(&name))
        .join(format!("dense-{}.txt", depth.word()));
    let dense_measured = if refit {
        let text = std::fs::read_to_string(&kept_dense)
            .map_err(|e| format!("no measurements at {}: {e}", kept_dense.display()))?;
        dense::Measured::from_text(&text, device.family())?
    } else {
        let m = dense::measure(&device, &base, cores, depth, say)?;
        std::fs::write(&kept_dense, m.to_text(&name)).map_err(|e| e.to_string())?;
        say(&format!("measurements kept in {}", kept_dense.display()));
        m
    };
    let (dense_entry, dense_line) = dense::fit(&dense_measured, depth)?;
    fitted.push(dense_entry);
    lines.push(dense_line);
    // The fitted matmul must also win on the served models to be kept.
    let mut knobs: Vec<search::Knob> = Vec::new();
    for (id, value) in &fitted {
        let (spec, rungs, _) = base
            .entries()
            .find(|(s, ..)| s.id == id.as_str())
            .expect("a Metal decision");
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
    knobs.extend(search::knobs(&base, SEARCH_SKIPS, "kernel.ane"));
    let mut chosen: search::Entries = Vec::new();
    let mut bench = Metal {
        models,
        device: &name,
        base: &base,
        facts: &facts,
        cores,
        mix,
    };
    if models.is_empty() {
        chosen = fitted;
        for k in knobs
            .iter()
            .filter(|k| !["kernel.qmm", "kernel.qmm_fp16", "kernel.dense"].contains(&k.id))
        {
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
        let (found, searched) = search::search(&mut bench, chosen, &knobs, "kernel.ane", say)?;
        chosen = found;
        lines.extend(searched);
    }
    let tuned = search::render_overrides(&chosen);
    // Only what the run measured (`search::Line::measured`): a decision no
    // check model reads, or one whose bar the noise pushed past the
    // agreement tolerance, keeps its line in the report but is not called
    // measured.
    let mut tuned_ids: Vec<String> = Vec::new();
    for l in lines.iter().filter(|l| l.measured) {
        let id = format!("\"{}\"", l.id);
        if !tuned_ids.contains(&id) {
            tuned_ids.push(id);
        }
    }
    let today = assemble(&base, &name, &facts, "", &[])?;
    let draft = assemble(&base, &name, &facts, &tuned, &[])?;
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
        ("mix", "\"standard-1\"".to_string()),
    ];
    provenance.extend(cores.map(|cores| ("cores", cores.to_string())));
    provenance.extend([
        ("seconds", format!("{seconds:.0}")),
        ("checked", format!("\"{}\"", checked.replace('"', "'"))),
    ]);
    let file = assemble(&base, &name, &facts, &tuned, &provenance)?;
    let file = engine::rates::keeping_stored(crate::chip::registry(), file, user_file(profiles, &name).as_deref());
    Ok(Outcome {
        device: name,
        file,
        lines,
        wrong: measured
            .wrong
            .iter()
            .chain(&dense_measured.wrong)
            .map(|(body, n, first)| format!("{body}: wrong at {n} points; first at {first}"))
            .collect(),
        checks,
        changed: !unchanged,
        seconds,
    })
}

fn measure(
    device: &Context,
    base: &Profile,
    cores: Option<u32>,
    grid: &Grid,
    say: &mut dyn FnMut(&str),
) -> Result<Measured, String> {
    let candidates = candidates(base, device.family());
    // `base` with `overrides` under `[kernel]`, in each format.
    let tunings = |overrides: &str| -> Result<[Tuning; 2], String> {
        let at = |fp16: bool| -> Result<Tuning, String> {
            let p = crate::chip::registry()
                .overlay(
                    base,
                    &format!("[kernel]\n{overrides}qmm_fp16 = {fp16}\n"),
                    base.target().clone(),
                    Source::User("tune".into()),
                )
                .map_err(|e| e.to_string())?;
            Ok(Tuning::new(p, false, cores))
        };
        Ok([at(false)?, at(true)?])
    };
    let forced: Vec<[Tuning; 2]> = candidates
        .iter()
        .map(|q| tunings(&format!("qmm = {}\n", forcing(q))))
        .collect::<Result<_, _>>()?;
    let today = tunings("")?;

    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let widest = *grid.rows.iter().max().expect("rows");
    let cap = widest.div_ceil(32) * 32;
    let total = grid.shapes.len() * grid.banks.len() * grid.rows.len();
    let mut done = 0usize;
    let mut out = Measured {
        candidates: candidates.clone(),
        ..Measured::default()
    };
    for &(n, k, _) in &grid.shapes {
        let kk = k as usize;
        // Activations for the widest row count, bf16, and their values.
        let mut act_bytes = vec![0u8; cap as usize * kk * 2];
        let mut act = vec![0.0f64; cap as usize * kk];
        for i in 0..cap as usize * kk {
            let v = 0.02 * (f32::from(noise(i as u64 ^ 0x77) % 16) - 8.0);
            act_bytes[2 * i..2 * i + 2].copy_from_slice(&bf16(v));
            act[i] = seen(v);
        }
        let alloc = |bytes: u64| Buffer::zeroed(device, bytes).map_err(|e| e.to_string());
        let mut act_b = alloc(act_bytes.len() as u64)?;
        act_b.write(0, &act_bytes).map_err(|e| e.to_string())?;
        let out_b = alloc(u64::from(cap) * u64::from(n) * 2)?;
        let precast_b = alloc(u64::from(cap) * u64::from(k) * 2)?;
        let partial_b = alloc(8 * u64::from(cap) * u64::from(n) * 4)?;
        let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).map_err(|e| e.to_string());
        let (ha, ho, hp, hq) = (
            bind(&act_b)?,
            bind(&out_b)?,
            bind(&precast_b)?,
            bind(&partial_b)?,
        );
        let partial_rows = partial_b.bytes() / (u64::from(n) * 4);

        for bank in &grid.banks {
            if k % bank.group != 0 {
                done += grid.rows.len();
                continue;
            }
            let w = match weights(device, &handles, n, k, bank.bits, bank.group) {
                Ok(w) => w,
                Err(why) => {
                    say(&format!("  skipped {n} x {k}: {why}"));
                    done += grid.rows.len();
                    continue;
                }
            };
            let format = usize::from(bank.fp16);
            for &m in &grid.rows {
                done += 1;
                let launch = |sink: &dyn Encode, tuning: &Tuning, copy: usize| {
                    let copy = copy % w.codes.len();
                    let b = kernels_metal::Bank {
                        mpp_codes: None,
                        codes: Tensor::new(w.codes[copy], n, k, dtype(bank.bits, bank.group)),
                        scales: Tensor::new(w.scales[copy], n, k / bank.group, Dtype::Bf16),
                        biases: Some(Tensor::new(w.biases[copy], n, k / bank.group, Dtype::Bf16)),
                        group: bank.group,
                        bits: bank.bits,
                        ld: k,
                    };
                    let precast = |rows: u32, c: u32| {
                        (u64::from(rows) * u64::from(c) <= u64::from(cap) * u64::from(k))
                            .then(|| Tensor::new(hp, rows, c, Dtype::F16))
                    };
                    let partials = |rows: u32, width: u32| {
                        (width == n && u64::from(rows) <= partial_rows)
                            .then(|| Tensor::new(hq, rows, width, Dtype::F32))
                    };
                    qmm::matmul(
                        sink,
                        tuning,
                        Tensor::new(ha, m, k, Dtype::Bf16),
                        b,
                        Tensor::new(ho, m, n, Dtype::Bf16),
                        qmm::Scratch {
                            precast: &precast,
                            partials: &partials,
                            table: None,
                        },
                        cap,
                    )
                };
                // The candidates that run here, as themselves, and right.
                let mut live: Vec<usize> = Vec::new();
                for (c, q) in candidates.iter().enumerate() {
                    let t = &forced[c][format];
                    if !q.fits(&t.qmm_at(m, n, k, bank.bits, bank.group)) {
                        continue;
                    }
                    let frame = device.frame().map_err(|e| e.to_string())?;
                    let sink = Sink::new(device, &frame, &pipelines, &handles);
                    let watch = Watch {
                        inner: &sink,
                        files: RefCell::new(Vec::new()),
                    };
                    let launched = launch(&watch, t, 0).is_ok();
                    let as_itself = ran(&watch.files.borrow(), q);
                    drop(watch);
                    frame.commit().map_err(|e| e.to_string())?;
                    if !launched || !as_itself {
                        continue;
                    }
                    if let Some(why) = wrong(&handles, ho, m, n, k, bank, &w, &act) {
                        let first =
                            format!("{m} x {n} x {k}, {}-bit g{}: {why}", bank.bits, bank.group);
                        let name = body(q);
                        match out.wrong.iter_mut().find(|(b, ..)| *b == name) {
                            Some(entry) => entry.1 += 1,
                            None => out.wrong.push((name, 1, first)),
                        }
                        continue;
                    }
                    live.push(c);
                }
                let times = time(device, &pipelines, &handles, &live, |sink, c, copy| {
                    launch(sink, &forced[c][format], copy)
                })?;
                let mut row = vec![None; candidates.len()];
                for (c, (t, _)) in live.iter().zip(&times) {
                    row[*c] = Some(*t);
                }
                let spread = profile::tune::pooled(times.iter().map(|&(_, e)| e));
                // Today's ladder: the first of its rungs that ran.
                let at = today[format].qmm_at(m, n, k, bank.bits, bank.group);
                let baseline = today[format]
                    .qmm(&at)
                    .iter()
                    .filter_map(|q| candidates.iter().position(|c| c == q))
                    .find(|&c| row[c].is_some());
                out.points.push(Point {
                    rows: m,
                    n,
                    k,
                    bank: *bank,
                    times: row,
                    baseline,
                    spread,
                });
            }
            say(&format!(
                "  kernel.qmm: {done}/{total} points ({n} x {k}, {}-bit g{}{})",
                bank.bits,
                bank.group,
                if bank.fp16 { ", fp16" } else { "" }
            ));
        }
    }
    Ok(out)
}

/// Why the launch just run answered wrongly, if it did: a sample of outputs
/// against the f64 reference, allowing fp16 and bf16 rounding.
#[allow(clippy::too_many_arguments)]
fn wrong(
    handles: &Handles,
    ho: u32,
    m: u32,
    n: u32,
    k: u32,
    bank: &Bank,
    w: &Weights,
    act: &[f64],
) -> Option<String> {
    let raw = handles.read(ho, u64::from(m) * u64::from(n) * 2).ok()?;
    let (nn, kk) = (n as usize, k as usize);
    let groups = kk / bank.group as usize;
    for r in [0, m as usize / 2, m as usize - 1] {
        for s in 0..16usize {
            let col = (s * 7919 + r * 104_729) % nn;
            let (mut want, mut mass) = (0.0f64, 0.0f64);
            for c in 0..kk {
                let g = col * groups + c / bank.group as usize;
                let wv = w.scale[g] * f64::from(w.q[col * kk + c]) + w.bias[g];
                let t = act[r * kk + c] * wv;
                want += t;
                mass += t.abs();
            }
            let at = 2 * (r * nn + col);
            let have = f64::from(to_f32(raw[at], raw[at + 1]));
            let allowed = want.abs().max(have.abs()) / 128.0 + mass / 256.0;
            if (want - have).abs() > allowed {
                return Some(format!(
                    "row {r} column {col}: want {want:.5}, got {have:.5}"
                ));
            }
        }
    }
    None
}

/// Seconds per launch for each of `live`, timed round-robin so drift lands
/// on every candidate alike, each with its median's relative error.
fn time(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    live: &[usize],
    launch: impl Fn(&dyn Encode, usize, usize) -> Result<(), Error>,
) -> Result<Vec<(f64, f64)>, String> {
    Ok(time_spread(device, pipelines, handles, live, launch)?
        .into_iter()
        .map(|(median, .., error)| (median, error))
        .collect())
}

/// [`time`]'s medians with the fastest and slowest of each candidate's
/// rounds (what a dense curve states as its spread) and the median's
/// relative error ([`profile::tune::median_and_error`]).
fn time_spread(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    live: &[usize],
    launch: impl Fn(&dyn Encode, usize, usize) -> Result<(), Error>,
) -> Result<Vec<(f64, f64, f64, f64)>, String> {
    time_rounds(device, pipelines, handles, live, 7, launch)
}

/// [`time_spread`] over `rounds` timed rounds (at least 3).
fn time_rounds(
    device: &Context,
    pipelines: &Pipelines,
    handles: &Handles,
    live: &[usize],
    rounds: usize,
    launch: impl Fn(&dyn Encode, usize, usize) -> Result<(), Error>,
) -> Result<Vec<(f64, f64, f64, f64)>, String> {
    let rounds = rounds.max(3);
    const FRAME_S: f64 = 0.002;
    const SLOWER: f64 = 3.0;
    let frame = |c: usize, batch: usize| -> Result<f64, String> {
        let f = device.frame().map_err(|e| e.to_string())?;
        let sink = Sink::new(device, &f, pipelines, handles);
        for copy in 0..batch {
            launch(&sink, c, copy).map_err(|e| e.to_string())?;
        }
        f.commit_timed().map_err(|e| e.to_string())
    };
    // Warm each and time one launch. One `SLOWER` than the fastest keeps
    // that time: the fit only needs to know it loses.
    let mut once = Vec::new();
    for &c in live {
        frame(c, 1)?;
        once.push(frame(c, 1)?);
    }
    let fastest = once.iter().copied().fold(f64::INFINITY, f64::min);
    let timed: Vec<bool> = once.iter().map(|&t| t <= fastest * SLOWER).collect();
    let batch: Vec<usize> = once
        .iter()
        .map(|&one| ((FRAME_S / one.max(1e-7)) as usize).clamp(1, 256))
        .collect();
    let mut seen: Vec<Vec<f64>> = (0..live.len())
        .map(|i| if timed[i] { Vec::new() } else { vec![once[i]] })
        .collect();
    for _ in 0..rounds {
        for (i, &c) in live.iter().enumerate() {
            if timed[i] {
                seen[i].push(frame(c, batch[i])? / batch[i] as f64);
            }
        }
    }
    Ok(seen
        .into_iter()
        .map(|mut s| {
            let (median, error) = profile::tune::median_and_error(&s);
            s.sort_by(f64::total_cmp);
            (median, s[0], s[s.len() - 1], error)
        })
        .collect())
}

/// Metal's loads of one model for the check of the estimate
/// (`engine::load::check`): `pie tune metal --check-estimate`.
pub struct CheckBench<'a> {
    artifact: &'a Path,
    profiles: Option<&'a Path>,
    trace: poem::Trace,
    contract: checkpoint::contract::ModelContract,
    model: String,
    /// The served load's pool layout, once the check weighed it: what a
    /// shell's seats take is laid out as that pool lays it.
    layout: Option<crate::store::layout::Layout>,
}

/// The check bench of the model `deployment` names, its weights in
/// `artifact`, under the user's profiles in `profiles`.
///
/// # Errors
///
/// No catalog row of that name, or an artifact that does not read.
pub fn check_bench<'a>(
    profiles: Option<&'a Path>,
    artifact: &'a Path,
    deployment: &str,
) -> Result<CheckBench<'a>, String> {
    use poem::Platform;
    let row = poem_compiler::catalog::deployment(deployment)
        .ok_or_else(|| format!("no catalog row `{deployment}`"))?;
    let trace = row.trace(Platform::Metal);
    let model = engine::load::model_class(&trace).to_owned();
    let source = ztensor_compat::index(artifact).map_err(|e| e.to_string())?;
    let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Metal)
        .map_err(|e| e.to_string())?;
    Ok(CheckBench {
        artifact,
        profiles,
        trace,
        contract,
        model,
        layout: None,
    })
}

impl engine::load::check::Bench for CheckBench<'_> {
    /// `None`: the shape a served load is asked (the fire size `FIRE`, the
    /// pool its capacity), whose weighing alone gives the rungs, the
    /// estimate, the live range and the fire top; it fires nothing.
    /// `Some(shell)`: `slots` seats of `context` each, every rung to `rows`
    /// laid (`Top::EveryRung`), the pool holding those seats' runs and the
    /// widest fire's rows, raised to the widest rung for the check's fires.
    fn load(
        &mut self,
        run: engine::load::check::Run,
        shell: Option<engine::load::check::Shell>,
    ) -> Result<Box<dyn engine::load::check::Loaded + '_>, String> {
        let page = 16u32;
        let (lanes, tokens) = crate::FIRE;
        // A served shape takes the fire size's lanes and the room the device
        // leaves. A shell the check fires takes its own slots, the most lanes
        // one of its fires has, and a pool of what its seats hold: each
        // seat's run of its context's pages and its record's pieces, and the
        // widest fire's rows landing past them, laid out as the served
        // load's pool lays them (`Layout::shell_room`, the layout learned
        // from the weighing load, which the check takes first).
        let (slots, context, rows, kv_budget) = match shell {
            None => (lanes, tokens, tokens, None),
            Some(shell) => {
                let layout = self.layout.ok_or_else(|| {
                    "the check fired a shell before it weighed the served load".to_string()
                })?;
                let room = layout.shell_room(
                    shell.slots,
                    u64::from(shell.context.div_ceil(page)),
                    u64::from(shell.rows.div_ceil(page)),
                );
                (shell.slots, shell.context, shell.rows, Some(room))
            }
        };
        let mut loaded = crate::Shell::load(crate::Boot {
            voxels: None,
            trace: self.trace.clone(),
            contract: &self.contract,
            checkpoint: self.artifact,
            budget: poem_compiler::Budget::new(slots, rows),
            patches: patch_ladder(&self.trace),
            profile: None,
            page_size: page,
            context,
            slots,
            kv_budget,
            allowance: engine::Allowance::WHOLE,
            runahead: engine::runahead::Runahead::F1,
            profiles: self.profiles.map(Path::to_path_buf),
            residency: crate::ResidencyPlan::default(),
            top: crate::Top::EveryRung,
            measure: engine::rates::Measuring::Time,
        })
        .map_err(|e| e.to_string())?;
        if shell.is_some() {
            // No runtime raises the rung here: the check fires the shell
            // itself, at its widest.
            loaded.raise_whole().map_err(|e| e.to_string())?;
            loaded.back_seats().map_err(|e| e.to_string())?;
        } else {
            self.layout = Some(loaded.pool_layout());
        }
        Ok(Box::new(CheckLoad {
            shell: loaded,
            trace: &self.trace,
            model: &self.model,
            run,
        }))
    }
}

/// One load of a [`CheckBench`].
struct CheckLoad<'a> {
    shell: crate::Shell,
    trace: &'a poem::Trace,
    model: &'a str,
    run: engine::load::check::Run,
}

impl CheckLoad<'_> {
    /// `fire`'s lanes, each on a fresh slot, and the fire timed to its
    /// landing (ns).
    fn fire_once(&mut self, fire: engine::load::rung::CheckFire) -> Result<f64, String> {
        use poem::Request;
        let tokens: Vec<Vec<u32>> = (0..fire.lanes)
            .map(|lane| {
                (0..fire.tokens_per_lane)
                    .map(|i| 100 + (lane * 131 + i) * 7919 % 5000)
                    .collect()
            })
            .collect();
        for slot in 0..fire.lanes {
            self.shell.open(slot).map_err(|e| e.to_string())?;
        }
        let word = self
            .trace
            .facts
            .word(&Request::new(fire.tokens_per_lane, false));
        let held = fire.keys_per_lane.saturating_sub(fire.tokens_per_lane);
        let lanes: Vec<crate::Seated<'_>> = (0..fire.lanes)
            .map(|slot| crate::Seated {
                replay_stated: None,
                held: Some(held),
                ..crate::Seated::of(crate::Lane {
                    slot,
                    word,
                    tokens: &tokens[slot as usize],
                })
            })
            .collect();
        let began = Instant::now();
        self.shell.fire_seated(&lanes).map_err(|e| e.to_string())?;
        Ok(began.elapsed().as_nanos() as f64)
    }
}

impl engine::load::check::Loaded for CheckLoad<'_> {
    fn weighed(&self) -> Result<engine::load::check::Weighed, String> {
        let weighed = self.shell.weighed().ok_or_else(|| {
            format!(
                "{} has no [rates] in its profile to estimate by: run `pie tune --rates`",
                self.model
            )
        })?;
        Ok(engine::load::check::Weighed {
            model: self.model.to_owned(),
            rungs: weighed.rungs.clone(),
            estimate: weighed.estimate.clone(),
            live: weighed.live,
            fire_top: weighed.fire_top,
            laid: weighed.rungs.iter().map(|r| r.tokens).max().unwrap_or(0),
        })
    }

    /// Metal records no body: every serving fire encodes its own dispatch,
    /// so a fire of a load run as serving runs it is what serving runs.
    fn fire(
        &mut self,
        fire: engine::load::rung::CheckFire,
    ) -> Result<engine::load::check::Fired, String> {
        let ns = self.fire_once(fire)?;
        Ok(engine::load::check::Fired {
            ns,
            replayed: self.run == engine::load::check::Run::Replayed,
        })
    }

    /// Each dispatch committed and timed alone by the GPU's own clock
    /// (`crate::encode::profile_kernels`), told by its kernel: a dispatch's
    /// time holds no drain, so none is taken off.
    fn dissect(
        &mut self,
        fire: engine::load::rung::CheckFire,
    ) -> Result<engine::load::check::Dissection, String> {
        crate::encode::reset_kernel_profile();
        crate::encode::profile_kernels(true);
        let fired = self.fire_once(fire);
        crate::encode::profile_kernels(false);
        let wall_ns = fired?;
        let ops = crate::encode::kernel_profile()
            .into_iter()
            .map(|(name, ns, n)| (name, ns as f64, u32::try_from(n).unwrap_or(u32::MAX)))
            .collect();
        crate::encode::reset_kernel_profile();
        Ok(engine::load::check::Dissection {
            wall_ns,
            ops,
            drain_ns: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Keyed rates a load stored are still taken by the next load after a
    /// tune, and the tuned decision stands beside them.
    #[test]
    fn a_tune_keeps_the_stored_rates() {
        const DEVICE: &str = "Apple M1 Max";
        let home = std::env::temp_dir().join(format!("pie-tune-keeps-{}", std::process::id()));
        let path = crate::chip::user_path(&home, DEVICE);
        std::fs::create_dir_all(path.parent().expect("a parent")).expect("the user's directory");
        let keyed = profile::rates::Rates {
            launch_ns: Some(4321),
            measured_on: Some(crate::stored::measured_on(DEVICE)),
            measured_by: Some(vec![(
                "launch_ns".to_string(),
                crate::stored::by("launch_ns", &["ssm/conv1d.metal"]),
            )]),
            ..Default::default()
        };
        let how = crate::rates::provenance(DEVICE);
        std::fs::write(&path, engine::rates::saved_text(None, &keyed, &how)).unwrap();
        let facts = crate::chip::facts(7, Some(32), 8, true);
        let takes = |said: &str| {
            let (chip, report) = crate::chip::resolve_keyed(DEVICE, &facts, Some(&home));
            assert!(
                report.stored.iter().any(|line| line.starts_with("`launch_ns`: stored (user")),
                "{said}: {:#?}",
                report.stored
            );
            assert_eq!(chip.tuned().rates.launch_ns, Some(4321), "{said}");
            chip
        };
        let base = takes("a load's own save");

        let tuned = "[kernel]\nqmm = [{ use = \"tiled\", crossover_tg = 120 }, \"qmv\"]\n";
        let file = assemble(&base, DEVICE, &facts, tuned, &[]).expect("the tuned file");
        let file = engine::rates::keeping_stored(crate::chip::registry(), file, user_file(Some(&home), DEVICE).as_deref());
        std::fs::write(&path, file).unwrap();
        let chip = takes("a tune");
        assert_eq!(
            chip.ladder::<kernels_metal::tuning::Qmm>().values().next(),
            Some(&kernels_metal::tuning::Qmm::Tiled { crossover_tg: 120 }),
            "the tuned decision is there"
        );
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn the_standard_mix_weighs_one_over_the_grid() {
        for depth in [Depth::Quick, Depth::Full] {
            let grid = Grid::of(depth);
            let total: f64 = grid
                .shapes
                .iter()
                .flat_map(|s| grid.banks.iter().map(move |b| (s, b)))
                .flat_map(|(s, b)| grid.rows.iter().map(move |&r| (r, *s, *b)))
                .map(|(r, s, b)| weight(&grid, r, s.0, s.1, b, true))
                .sum();
            // The unchosen 4-bit, 64-group format counts a tenth on top.
            assert!((total - 1.05).abs() < 1e-9, "{depth:?}: {total}");
        }
    }

    #[test]
    fn every_candidate_forces_a_ladder_the_registry_reads() {
        let (base, _) = crate::chip::resolve(
            "Apple M4 Max",
            &crate::chip::facts(9, Some(40), 10, true),
            None,
        );
        let registry = crate::chip::registry();
        for q in candidates(&base, 10) {
            let p = registry
                .overlay(
                    &base,
                    &format!("[kernel]\nqmm = {}\n", forcing(&q)),
                    base.target().clone(),
                    Source::User("test".into()),
                )
                .unwrap_or_else(|e| panic!("{q:?}: {e}"));
            assert_eq!(p.ladder::<Qmm>().values().next(), Some(&q));
        }
    }

    #[test]
    fn a_fitted_ladder_assembles_into_a_file_this_device_takes() {
        let facts = crate::chip::facts(9, Some(40), 10, true);
        let (base, _) = crate::chip::resolve("Apple M4 Max", &facts, None);
        let grid = Grid::of(Depth::Quick);
        let candidates = candidates(&base, 9);
        // A synthetic device: the tiled kernel wins from 12 rows, MPP never
        // runs, and the vector kernel's two-pack variant answers wrongly.
        let mut points = Vec::new();
        for &(n, k, _) in &grid.shapes {
            for bank in &grid.banks {
                for &rows in &grid.rows {
                    let times = candidates
                        .iter()
                        .map(|q| match *q {
                            Qmm::Mpp | Qmm::Staged | Qmm::Register => None,
                            Qmm::Tiled { crossover_tg } => {
                                (rows >= 4).then(|| 12.0 + f64::from(crossover_tg) / 1000.0)
                            }
                            Qmm::Qmv { packs: 2, .. } => None,
                            Qmm::Qmv { rows_max, .. } => {
                                Some(f64::from(rows) + f64::from(8 - rows_max) / 10.0)
                            }
                        })
                        .collect();
                    points.push(Point {
                        rows,
                        n,
                        k,
                        bank: *bank,
                        times,
                        baseline: Some(1),
                        spread: 0.005,
                    });
                }
            }
        }
        let measured = Measured {
            candidates,
            points,
            wrong: Vec::new(),
        };
        let (unfitted, why) = fit(&measured, &grid, None).unwrap();
        assert_eq!(
            unfitted
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>(),
            ["kernel.qmm_fp16"],
            "without the cores, `fill` is unknown and the ladder is not fitted"
        );
        assert!(why[1].text.starts_with("not searched"), "{why:?}");
        let (entries, lines) = fit(&measured, &grid, Some(40)).unwrap();
        let tuned = profile::search::render_overrides(&entries);
        assert!(
            tuned.contains("{ when = \"rows >= 16\", use = \"tiled\", crossover_tg = 64 }"),
            "{tuned}"
        );
        assert!(
            tuned.contains("{ use = \"qmv\", rows_max = 8, packs = 1, crossover_tg = 160 }"),
            "{tuned}"
        );
        assert_eq!(lines.len(), 2);
        let file = assemble(
            &base,
            "Apple M4 Max",
            &facts,
            &tuned,
            &[("measured", "[\"kernel.qmm\"]".to_string())],
        )
        .unwrap();
        assert!(file.contains("device = \"Apple M4 Max\""), "{file}");
        assert!(
            file.trim_end().ends_with("measured = [\"kernel.qmm\"]"),
            "{file}"
        );
    }

    #[test]
    fn the_census_is_the_catalogs() {
        use std::collections::BTreeMap;
        let mut by: BTreeMap<(u64, u64), u64> = BTreeMap::new();
        for d in poem_compiler::catalog::deployments() {
            let Ok(trace) = d.try_trace(poem::Platform::Metal) else {
                continue;
            };
            for p in &trace.params {
                let quantised = format!("{:?}", p.dtype);
                if quantised.starts_with('U') && quantised.contains('g') && p.shape.len() == 2 {
                    let (n, k) = (p.shape[0], p.shape[1]);
                    if (256..100_000).contains(&n) && k >= 256 {
                        *by.entry((n, k)).or_default() += 1;
                    }
                }
            }
        }
        let mut heaviest: Vec<((u64, u64), u64)> = by.into_iter().collect();
        heaviest.sort_by_key(|((n, k), c)| std::cmp::Reverse(n * k * c));
        let want: Vec<(u32, u32, u32)> = heaviest
            .iter()
            .take(CENSUS.len())
            .map(|((n, k), c)| (*n as u32, *k as u32, *c as u32))
            .collect();
        assert_eq!(
            CENSUS,
            want.as_slice(),
            "the catalog's heaviest bank shapes moved; `CENSUS` should read:\n{}",
            want.iter()
                .map(|(n, k, c)| format!("    ({n}, {k}, {c}),"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn measurements_read_back_as_written() {
        let (base, _) = crate::chip::resolve(
            "Apple M4 Max",
            &crate::chip::facts(9, Some(40), 10, true),
            None,
        );
        let candidates = candidates(&base, 9);
        let mut times = vec![None; candidates.len()];
        times[0] = Some(1.5e-4);
        times[3] = Some(2.25e-5);
        let m = Measured {
            candidates: candidates.clone(),
            points: vec![Point {
                rows: 8,
                n: 5120,
                k: 17408,
                bank: Bank {
                    bits: 4,
                    group: 64,
                    fp16: true,
                },
                times,
                baseline: Some(3),
                spread: 0.005,
            }],
            wrong: vec![(
                "use = \"qmv\", rows_max = 8, packs = 2, crossover_tg = 160".into(),
                3,
                "8 x 1 x 2: off".into(),
            )],
        };
        let back = Measured::from_text(&m.to_text("Apple M4 Max"), &candidates).unwrap();
        assert_eq!(back.points[0].times, m.points[0].times);
        assert_eq!(back.points[0].baseline, Some(3));
        assert_eq!(back.points[0].bank, m.points[0].bank);
        assert_eq!(back.wrong, m.wrong);
        assert!(Measured::from_text(&m.to_text("x"), &candidates[1..]).is_err());
    }

    #[test]
    fn every_search_candidate_reads_against_the_registry() {
        let facts = crate::chip::facts(10, Some(40), 10, true);
        let (base, _) = crate::chip::resolve("Apple M5 Max", &facts, None);
        let knobs = profile::search::knobs(&base, SEARCH_SKIPS, "kernel.ane");
        let ids: Vec<&str> = knobs.iter().map(|k| k.id).collect();
        for want in [
            "kernel.sdpa.prefill",
            "kernel.gdn.scan",
            "kernel.ane.mlp",
            "fuse.matmul_swiglu_matmul",
            "compiler.launch_us",
        ] {
            assert!(ids.contains(&want), "{want} not searched: {ids:?}");
        }
        assert!(!ids.contains(&"kernel.sdpa.kv_q8") && !ids.contains(&"kernel.qmm"));
        for k in &knobs {
            for (label, text) in &k.candidates {
                assemble(&base, "Apple M5 Max", &facts, text, &[])
                    .unwrap_or_else(|e| panic!("{} = {label}: {e}", k.id));
            }
        }
    }

    #[test]
    fn a_load_reads_only_the_decisions_it_evaluates() {
        let (base, _) = crate::chip::resolve(
            "Apple M4 Max",
            &crate::chip::facts(9, Some(40), 10, true),
            None,
        );
        let tuning = kernels_metal::Tuning::new(base.clone(), false, Some(40));
        let _ = tuning.qmm_mpp();
        assert!(
            base.read().is_empty(),
            "sizing at load is not a read: {:?}",
            base.read()
        );
        let _ = tuning.qmm(&tuning.qmm_at(8, 4096, 4096, 4, 64));
        // The point reads the fp16 flag for its format.
        assert_eq!(base.read(), vec!["kernel.qmm", "kernel.qmm_fp16"]);
    }
}
