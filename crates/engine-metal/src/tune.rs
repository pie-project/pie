//! `pie tune` on Metal: measure the decisions this backend declares
//! `tune = search` on this machine's GPU, fit their ladders, and write a
//! complete profile for the user's shelf. `.wiki/pie-tune.md` holds the
//! design.
//!
//! v1 measures `kernel.qmm` and `kernel.qmm_fp16`: the dense quantised
//! matmul, which carries most of a model's time. Every other decision is
//! copied from the profile the device runs today, and the report says so.

use std::fmt::Write as _;

use kernels_metal::tuning::{Qmm, QmmFp16};
use profile::search::{Entries, Line};
use profile::tune::{Candidate, Fit, Options, Sample};
use profile::{Decision, Facts, Profile, Source};

/// How much to measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Depth {
    /// A coarse grid: minutes.
    Quick,
    /// The whole grid: expected for a contributed file.
    Full,
}

impl Depth {
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Depth::Quick => "quick",
            Depth::Full => "full",
        }
    }
}

/// One bank format: code width, group width, and whether 4-bit, 64-group
/// codes are read through the fp16 GEMM format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bank {
    pub bits: u32,
    pub group: u32,
    pub fp16: bool,
}

/// The quantised bank shapes of the models pie ships, `(n, k, banks)`:
/// output columns, contraction length, and how many banks of that shape
/// the catalog's deployments hold, heaviest (`n * k * banks`) first. The
/// first eight hold 62% of the catalog's quantised bytes, all 24 hold 84%.
/// Vocabulary heads and banks under 256 wide are left out. The test
/// `the_census_is_the_catalogs` keeps it in step with `models/`.
pub const CENSUS: &[(u32, u32, u32)] = &[
    (34816, 5120, 593),
    (5120, 17408, 593),
    (43008, 5376, 180),
    (16384, 5120, 432),
    (24576, 4096, 264),
    (5376, 21504, 180),
    (5120, 6144, 578),
    (39936, 6656, 52),
    (16384, 5376, 150),
    (4096, 4096, 699),
    (12288, 5120, 146),
    (4096, 8192, 272),
    (4096, 16384, 116),
    (16384, 2560, 183),
    (6656, 19968, 52),
    (8192, 4096, 202),
    (5376, 8192, 150),
    (12288, 4096, 104),
    (8192, 2816, 225),
    (4096, 12288, 88),
    (4096, 2048, 472),
    (2560, 6144, 246),
    (16384, 4096, 57),
    (20480, 5376, 30),
];

/// The points a dense matmul is measured at.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub rows: Vec<u32>,
    /// `(n, k, weight)`: output columns, contraction length, and the
    /// shape's share of the grid's census bytes.
    pub shapes: Vec<(u32, u32, f64)>,
    pub banks: Vec<Bank>,
}

impl Grid {
    /// The grid for `depth`. Shapes are the heaviest of the [`CENSUS`],
    /// weighted by their bytes, since a model's time goes where its bytes
    /// are; rows are every small count, then log steps, because Metal pads
    /// rows to no lattice by default.
    #[must_use]
    pub fn of(depth: Depth) -> Grid {
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
            let total: f64 = taken.iter().map(|&(n, k, c)| f64::from(n) * f64::from(k) * f64::from(c)).sum();
            taken
                .iter()
                .map(|&(n, k, c)| (n, k, f64::from(n) * f64::from(k) * f64::from(c) / total))
                .collect::<Vec<_>>()
        };
        match depth {
            Depth::Quick => Grid {
                rows: vec![1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 64, 128, 512],
                // The four heaviest, and four spread down the census to its
                // small attention projections: a crossover learned on large
                // banks alone misplaces small ones.
                shapes: shapes([0, 1, 2, 3, 9, 15, 20, 21].iter().map(|&i| CENSUS[i]).collect()),
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

/// The standard workload mix: how much a point counts. Half the weight is
/// decode (8 rows or fewer), a fifth small batches, the rest prefill. By
/// bank: 4-bit 64-group half, 4-bit 32-group a tenth, 8-bit 0.15, 2-bit a
/// quarter, each split over the grid's banks of that kind. A 4-bit 64-group
/// bank in the format the fitted `qmm_fp16` does not pick only reaches a
/// wide-range model, so it counts a tenth on top.
#[must_use]
pub fn weight(grid: &Grid, rows: u32, n: u32, k: u32, bank: Bank, fp16_on: bool) -> f64 {
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

/// What `kernel.qmm` is measured over: every rung value the current
/// profile holds, and a spread of constants around them.
#[must_use]
pub fn candidates(base: &Profile) -> Vec<Qmm> {
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
    for q in base.ladder::<Qmm>().values() {
        if !out.contains(q) {
            out.push(*q);
        }
    }
    out
}

/// A rung value as the TOML body of a rung.
#[must_use]
pub fn body(q: &Qmm) -> String {
    match *q {
        Qmm::Mpp => "use = \"mpp\"".to_string(),
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
/// kernel elsewhere: how one candidate is timed.
#[must_use]
pub fn forcing(q: &Qmm) -> String {
    match q {
        Qmm::Qmv { .. } => format!("[{{ {} }}]", body(q)),
        _ => format!(
            "[{{ {} }}, {{ use = \"qmv\", rows_max = 8, packs = 1, crossover_tg = 160 }}]",
            body(q)
        ),
    }
}

/// Which kernel a launch ran, from the files it fired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ran {
    Mpp,
    Tiled,
    Qmv,
}

impl Ran {
    #[must_use]
    pub fn of(files: &[&str]) -> Ran {
        if files.iter().any(|f| f.contains("quant_qmm_mpp")) {
            Ran::Mpp
        } else if files.iter().any(|f| f.contains("quant_qmm_t")) {
            Ran::Tiled
        } else {
            Ran::Qmv
        }
    }

    #[must_use]
    pub fn is(self, q: &Qmm) -> bool {
        matches!(
            (self, q),
            (Ran::Mpp, Qmm::Mpp) | (Ran::Tiled, Qmm::Tiled { .. }) | (Ran::Qmv, Qmm::Qmv { .. })
        )
    }
}

/// One measured point of the dense matmul.
#[derive(Clone, Debug)]
pub struct Point {
    pub rows: u32,
    pub n: u32,
    pub k: u32,
    pub bank: Bank,
    /// Seconds per launch for each candidate; `None` where it did not run.
    pub times: Vec<Option<f64>>,
    /// The candidate today's profile runs here.
    pub baseline: Option<usize>,
}

/// Everything measured for `kernel.qmm`.
#[derive(Clone, Debug, Default)]
pub struct Measured {
    pub candidates: Vec<Qmm>,
    pub points: Vec<Point>,
    /// Each candidate that answered wrongly: how many points, and the
    /// first.
    pub wrong: Vec<(String, usize, String)>,
}

impl Measured {
    /// The measurements as text: a header of candidates, then a line a
    /// point. What `pie tune --refit` reads back.
    #[must_use]
    pub fn to_text(&self, device: &str) -> String {
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
                "point {} {} {} {} {} {} {} {}",
                p.rows,
                p.n,
                p.k,
                p.bank.bits,
                p.bank.group,
                u8::from(p.bank.fp16),
                p.baseline.map_or_else(|| "-".to_string(), |b| b.to_string()),
                times.join(",")
            );
        }
        out
    }

    /// Read [`Measured::to_text`] back, against today's candidates: a file
    /// measured over another candidate list is refused.
    pub fn from_text(text: &str, candidates: &[Qmm]) -> Result<Measured, String> {
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
                    f.get(i).and_then(|v| v.parse().ok()).ok_or(format!("field {i} of `{line}`"))
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
                });
            }
        }
        if seen != bodies {
            return Err("the measurements were taken over another candidate list; measure again".into());
        }
        Ok(out)
    }
}

/// Decisions the end-to-end search leaves alone: the dense matmul, which
/// the kernel bench measures, and what means nothing on Metal.
pub const SEARCH_SKIPS: &[&str] = &[
    "kernel.qmm",
    "kernel.qmm_fp16",
    // Metal serves every region from one queue, on one device.
    "compiler.side_streams",
    "compiler.cost.collective_us",
];

/// The fitted values of the measured decisions, as profile TOML under
/// `[kernel]`, and the report lines.
pub fn fit(measured: &Measured, grid: &Grid, cores: u32) -> Result<(Entries, Vec<Line>), String> {
    let mut lines = Vec::new();
    // `qmm_fp16`: whether the fp16 format's best time beats the emulated
    // format's over the 4-bit, 64-group points, weighted as decode-heavy.
    let best = |p: &Point| p.times.iter().flatten().copied().fold(f64::INFINITY, f64::min);
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
    });

    let spec = Qmm::spec();
    let candidates: Vec<Candidate> = measured
        .candidates
        .iter()
        .map(|q| Candidate {
            body: body(q),
            terminal: matches!(q, Qmm::Qmv { .. }),
        })
        .collect();
    let mut samples = Vec::new();
    let mut baseline = Vec::new();
    for p in &measured.points {
        let at = coords(p, cores);
        samples.push(Sample {
            at,
            weight: weight(grid, p.rows, p.n, p.k, p.bank, fp16_on),
            times: p.times.clone(),
        });
        baseline.push(p.baseline);
    }
    let fit: Fit = profile::tune::fit(spec, &candidates, &samples, &baseline, Options::default())
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
    });
    let entries = vec![
        (spec.id.to_string(), ladder),
        (QmmFp16::spec().id.to_string(), fp16_on.to_string()),
    ];
    Ok((entries, lines))
}

/// A point's coordinates in `kernel.qmm`'s variable order: rows, fill, n,
/// k, bits, group, format.
#[must_use]
pub fn coords(p: &Point, cores: u32) -> Vec<i64> {
    let tiles = u64::from(p.rows.div_ceil(32)) * u64::from(p.n.div_ceil(64));
    vec![
        i64::from(p.rows),
        i64::try_from(tiles * 8 / u64::from(cores.max(1))).unwrap_or(i64::MAX),
        i64::from(p.n),
        i64::from(p.k),
        i64::from(p.bank.bits),
        i64::from(p.bank.group),
        i64::from(!p.bank.fp16),
    ]
}

/// The complete profile `pie tune` writes: `base` with the tuned decisions
/// replaced, its target the device, and a provenance that says how it was
/// made. Read back strictly and resolved against `facts` before it is
/// returned, so a file that would be refused is never written.
pub fn assemble(
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

pub use device::{Outcome, artifacts, run};

mod device {
    use std::cell::RefCell;
    use std::path::Path;
    use std::time::Instant;

    use kernels_metal::linear::quant;
    use kernels_metal::{ArgValue, Encode, Error, Fire, Tensor, Tuning};
    use profile::Decision as _;
    use poem_ir::Dtype;

    use profile::search::{self, Bench, CONTEXT, Check, LANES, Line, PROMPT, STEPS, Serving};

    use super::{Bank, Depth, Grid, Measured, Point, Profile, Ran, Source, candidates, forcing};
    use crate::device::{Buffer, Context, Handles, Pipelines};
    use crate::encode::Sink;

    /// What a run produced.
    pub struct Outcome {
        pub device: String,
        /// The complete profile, ready for the user's shelf.
        pub file: String,
        pub lines: Vec<Line>,
        pub wrong: Vec<String>,
        /// The end-to-end runs, today's profile against the tuned one.
        pub checks: Vec<Check>,
        pub seconds: f64,
    }

    /// The Metal-stamped artifacts under `store`, smallest first: what an
    /// end-to-end check can serve.
    #[must_use]
    pub fn artifacts(store: &Path) -> Vec<(std::path::PathBuf, String)> {
        let mut out: Vec<(u64, std::path::PathBuf, String)> = Vec::new();
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
                if stamp.backend == "metal" {
                    let bytes = file.metadata().map_or(u64::MAX, |m| m.len());
                    out.push((bytes, path, stamp.deployment));
                }
            }
        }
        out.sort_by_key(|(bytes, ..)| *bytes);
        out.into_iter().map(|(_, p, d)| (p, d)).collect()
    }

    /// Serve `artifact` under the complete profile `file`: a prompt, then
    /// steps at each lane count, timed, once the Neural Engine's programs
    /// are ready (or refused, which the serving records).
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
            // A vision row states its patch ladder even for a text prompt.
            patches: crate::api::patch_ladder(
                &trace,
                &engine::load::Budgets {
                    max_lanes: slots,
                    max_tokens: CONTEXT,
                    ..engine::load::Budgets::default()
                },
            ),
            profile: None,
            page_size: 16,
            context: CONTEXT,
            slots,
            pages: slots * CONTEXT / 16,
            runahead: engine::runahead::Runahead::F1,
            gpu_mem_utilization: crate::store::accounting::DEFAULT_GPU_MEM_UTILIZATION,
            profiles: Some(dir.clone()),
            residency: crate::ResidencyPlan::default(),
        })
        .map_err(|e| e.to_string())?;
        // A Neural Engine split runs only once its programs compile; time
        // the load as it will serve, not the GPU-alone stretch before.
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
        // One fire of `lanes`: each lane's tokens into its own slot.
        let fire = |shell: &mut crate::Shell, lanes: &[(u32, &[u32])]| -> Result<(f64, Vec<f32>), String> {
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
                let next: Vec<[u32; 1]> = (0..lanes).map(|l| [token((step * lanes + l) as u32)]).collect();
                let batch: Vec<(u32, &[u32])> =
                    next.iter().enumerate().map(|(l, t)| (l as u32, &t[..])).collect();
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
        })
    }

    /// The fused trace and compiled plan a load under `text` runs: two
    /// files with the same plan serve the same, so a price or a fusion that
    /// leaves the plan alone needs no serving.
    fn plan_of(deployment: &str, text: &str, cores: u32) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        let profile = crate::chip::registry().load_user(text, "plan").ok()?;
        let row = poem_compiler::catalog::deployment(deployment)?;
        let trace = poem_compiler::fuse::fuse(
            row.trace(poem::Platform::Metal),
            &poem_compiler::fuse::flags::kernels(&profile),
        );
        let slots = *LANES.iter().max().expect("lanes") as u32;
        let mut budgets = poem_compiler::Budgets::of(poem_compiler::Budget::new(slots, CONTEXT));
        let ladder = crate::api::patch_ladder(
            &trace,
            &engine::load::Budgets {
                max_lanes: slots,
                max_tokens: CONTEXT,
                ..engine::load::Budgets::default()
            },
        );
        if let Some(ladder) = ladder {
            budgets = budgets.with_patches(ladder);
        }
        let compiled = poem_compiler::compile_axes(
            &trace,
            &budgets,
            &poem_compiler::DeviceProfile::of(&profile, cores, &[], &[]),
        )
        .ok()?;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("{trace:?}{compiled:?}").hash(&mut h);
        Some(h.finish())
    }

    /// Metal's bench: the check models, served on this device.
    struct Metal<'a> {
        models: &'a [(std::path::PathBuf, String)],
        device: &'a str,
        base: &'a Profile,
        facts: &'a profile::Facts,
        cores: u32,
    }

    impl Bench for Metal<'_> {
        fn models(&self) -> Vec<String> {
            self.models.iter().map(|(_, d)| d.clone()).collect()
        }

        fn file(&self, entries: &[(String, String)]) -> Result<String, String> {
            super::assemble(
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

    /// An `Encode` that remembers the files it fired: which kernel a
    /// forced candidate actually ran.
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

    /// A bank of one `(n, k, bits, group)`, in enough copies to stream
    /// past the last-level cache, with its host values for the reference.
    struct Weights {
        codes: Vec<u32>,
        scales: Vec<u32>,
        biases: Vec<u32>,
        q: Vec<u8>,
        scale: Vec<f64>,
        bias: Vec<f64>,
        _buffers: Vec<Buffer>,
    }

    /// Bytes a copy rotation must exceed: more than any Apple GPU's
    /// system-level cache.
    const STREAM_BYTES: u64 = 256 << 20;
    /// The most memory one shape may take.
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
        let mut out = Weights {
            codes: Vec::new(),
            scales: Vec::new(),
            biases: Vec::new(),
            q,
            scale,
            bias,
            _buffers: Vec::new(),
        };
        for _ in 0..copies {
            for (bytes, into) in [(&packed, 0), (&sc, 1), (&bi, 2)] {
                let mut b = Buffer::zeroed(device, bytes.len() as u64).map_err(|e| e.to_string())?;
                b.write(0, bytes).map_err(|e| e.to_string())?;
                let h = handles.bind(&b, 0, b.bytes()).map_err(|e| e.to_string())?;
                match into {
                    0 => out.codes.push(h),
                    1 => out.scales.push(h),
                    _ => out.biases.push(h),
                }
                out._buffers.push(b);
            }
        }
        Ok(out)
    }

    /// Run the tune: measure, fit, and assemble the file. `say` hears
    /// progress.
    /// `models` are the artifacts (path, deployment) the end-to-end check
    /// serves; none skips it, and the provenance says so.
    ///
    /// The measurements are kept under `cache`; `refit` reads them from
    /// there instead of measuring.
    pub fn run(
        depth: Depth,
        profiles: Option<&Path>,
        cache: &Path,
        refit: bool,
        models: &[(std::path::PathBuf, String)],
        say: &mut dyn FnMut(&str),
    ) -> Result<Outcome, String> {
        let began = Instant::now();
        let device = Context::bind().map_err(|e| e.to_string())?;
        let name = device.name().to_string();
        let cores = device.cores();
        let facts = crate::chip::facts(device.family(), cores, crate::chip::ane_loads());
        let (base, report) = crate::chip::resolve(&name, &facts, profiles);
        say(&format!(
            "{name}: {cores} GPU cores, family {}; starting from {}",
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
            Measured::from_text(&text, &candidates(&base))?
        } else {
            let m = measure(&device, &base, cores, &grid, say)?;
            if let Some(dir) = kept.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            std::fs::write(&kept, m.to_text(&name)).map_err(|e| e.to_string())?;
            say(&format!("measurements kept in {}", kept.display()));
            m
        };
        let (fitted, mut lines) = super::fit(&measured, &grid, cores)?;
        // The fitted matmul is a candidate like any other: a ladder learned
        // on kernels alone must also win on the served models to be kept.
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
        knobs.extend(search::knobs(&base, super::SEARCH_SKIPS, "kernel.ane"));
        let mut chosen: search::Entries = Vec::new();
        let mut bench = Metal {
            models,
            device: &name,
            base: &base,
            facts: &facts,
            cores,
        };
        if models.is_empty() {
            // Nothing to serve: the kernel bench's fit stands alone, and
            // the provenance says no end-to-end run judged it.
            chosen = fitted;
            for k in knobs.iter().filter(|k| k.id != "kernel.qmm" && k.id != "kernel.qmm_fp16") {
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
            let (found, searched) = search::search(&mut bench, chosen, &knobs, "kernel.ane", say)?;
            chosen = found;
            lines.extend(searched);
        }
        let tuned = search::render_overrides(&chosen);
        let tuned_ids: Vec<String> = lines
            .iter()
            .filter(|l| !l.text.starts_with("not searched"))
            .map(|l| format!("\"{}\"", l.id))
            .collect();
        let today = super::assemble(&base, &name, &facts, "", &[])?;
        let draft = super::assemble(&base, &name, &facts, &tuned, &[])?;
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
        let checked = if unchanged {
            "kernels against the f64 reference; nothing changed".to_string()
        } else if checks.is_empty() {
            "kernels against the f64 reference; no end-to-end run".to_string()
        } else {
            format!(
                "kernels against the f64 reference; end to end: {}",
                checks.iter().map(Check::describe).collect::<Vec<_>>().join("; ")
            )
        };
        let seconds = began.elapsed().as_secs_f64();
        let provenance = [
            (
                "source",
                format!("\"pie tune --{} on this machine\"", depth.word()),
            ),
            ("measured", "true".to_string()),
            ("tuned", format!("[{}]", tuned_ids.join(", "))),
            ("from", format!("\"{}\"", report.used)),
            ("pie", format!("\"{}\"", env!("CARGO_PKG_VERSION"))),
            ("mix", "\"standard-1\"".to_string()),
            ("cores", cores.to_string()),
            ("seconds", format!("{seconds:.0}")),
            ("checked", format!("\"{}\"", checked.replace('"', "'"))),
        ];
        let file = super::assemble(&base, &name, &facts, &tuned, &provenance)?;
        Ok(Outcome {
            device: name,
            file,
            lines,
            wrong: measured
                .wrong
                .iter()
                .map(|(body, n, first)| format!("{body}: wrong at {n} points; first at {first}"))
                .collect(),
            checks,
            seconds,
        })
    }

    fn measure(
        device: &Context,
        base: &Profile,
        cores: u32,
        grid: &Grid,
        say: &mut dyn FnMut(&str),
    ) -> Result<Measured, String> {
        let registry = crate::chip::registry();
        let candidates = candidates(base);
        // One tuning per (candidate, format), and today's per format.
        let tuning = |qmm: &str, fp16: bool| -> Result<Tuning, String> {
            let p = registry
                .overlay(
                    base,
                    &format!("[kernel]\nqmm = {qmm}\nqmm_fp16 = {fp16}\n"),
                    base.target().clone(),
                    Source::User("tune".into()),
                )
                .map_err(|e| e.to_string())?;
            Ok(Tuning::new(p, false, cores))
        };
        let mut forced: Vec<[Tuning; 2]> = Vec::new();
        for q in &candidates {
            let f = forcing(q);
            forced.push([tuning(&f, false)?, tuning(&f, true)?]);
        }
        let today = |fp16: bool| -> Result<Tuning, String> {
            let p = registry
                .overlay(
                    base,
                    &format!("[kernel]\nqmm_fp16 = {fp16}\n"),
                    base.target().clone(),
                    Source::User("tune".into()),
                )
                .map_err(|e| e.to_string())?;
            Ok(Tuning::new(p, false, cores))
        };
        let today = [today(false)?, today(true)?];

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
            let (ha, ho, hp, hq) = (bind(&act_b)?, bind(&out_b)?, bind(&precast_b)?, bind(&partial_b)?);
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
                        quant::matmul(
                            sink,
                            tuning,
                            Tensor::new(ha, m, k, Dtype::Bf16),
                            b,
                            Tensor::new(ho, m, n, Dtype::Bf16),
                            quant::Scratch {
                                precast: &precast,
                                partials: &partials,
                            },
                            cap,
                        )
                    };
                    // Which candidates run here, as themselves, and right.
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
                        let ran = Ran::of(&watch.files.borrow());
                        drop(watch);
                        drop(sink);
                        frame.commit().map_err(|e| e.to_string())?;
                        if !launched || !ran.is(q) {
                            continue;
                        }
                        if let Some(why) = wrong(&handles, ho, m, n, k, bank, &w, &act) {
                            let first = format!(
                                "{m} x {n} x {k}, {}-bit g{}: {why}",
                                bank.bits, bank.group
                            );
                            let name = super::body(q);
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
                    for (c, t) in live.iter().zip(times) {
                        row[*c] = Some(t);
                    }
                    // Today's ladder: the first of its candidates that ran.
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

    /// Whether the launch just run answered wrongly: a sample of outputs
    /// against the f64 reference, with the fp16 and bf16 rounding the
    /// kernels may add.
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
        let rows: Vec<usize> = [0, m as usize / 2, m as usize - 1].into_iter().collect();
        for &r in &rows {
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

    /// Seconds per launch for each of `live`, timed round-robin so drift
    /// lands on every candidate alike, each frame rotating through the
    /// weight copies.
    fn time(
        device: &Context,
        pipelines: &Pipelines,
        handles: &Handles,
        live: &[usize],
        launch: impl Fn(&dyn Encode, usize, usize) -> Result<(), Error>,
    ) -> Result<Vec<f64>, String> {
        const ROUNDS: usize = 7;
        const FRAME_S: f64 = 0.002;
        /// How much slower than the fastest a candidate may be and still
        /// be timed to the end.
        const SLOWER: f64 = 3.0;
        let frame = |c: usize, batch: usize| -> Result<f64, String> {
            let f = device.frame().map_err(|e| e.to_string())?;
            let sink = Sink::new(device, &f, pipelines, handles);
            for copy in 0..batch {
                launch(&sink, c, copy).map_err(|e| e.to_string())?;
            }
            f.commit_timed().map_err(|e| e.to_string())
        };
        // Warm each and take one launch's time. A candidate several times
        // slower than the fastest keeps that one time: the fit only needs
        // to know it loses, and timing it to the end (a vector kernel at
        // hundreds of rows) would take most of the run.
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
        let mut seen: Vec<Vec<f64>> = once.iter().map(|&t| vec![t]).collect();
        for _ in 0..ROUNDS {
            for (i, &c) in live.iter().enumerate() {
                if timed[i] {
                    seen[i].push(frame(c, batch[i])? / batch[i] as f64);
                }
            }
        }
        for (i, s) in seen.iter_mut().enumerate() {
            if timed[i] {
                // The single warm launch is not one of the rounds.
                s.remove(0);
            }
        }
        Ok(seen
            .into_iter()
            .map(|mut s| {
                s.sort_by(f64::total_cmp);
                s[s.len() / 2]
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (base, _) = crate::chip::resolve("Apple M4 Max", &crate::chip::facts(9, 40, true), None);
        let registry = crate::chip::registry();
        for q in candidates(&base) {
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
        let facts = crate::chip::facts(9, 40, true);
        let (base, _) = crate::chip::resolve("Apple M4 Max", &facts, None);
        let grid = Grid::of(Depth::Quick);
        let candidates = candidates(&base);
        // A synthetic device: the tiled kernel wins from 12 rows, MPP never
        // runs, and the vector kernel's two-pack variant answers wrongly.
        let mut points = Vec::new();
        for &(n, k, _) in &grid.shapes {
            for bank in &grid.banks {
                for &rows in &grid.rows {
                    let times = candidates
                        .iter()
                        .map(|q| match *q {
                            Qmm::Mpp => None,
                            Qmm::Tiled { crossover_tg } => (rows >= 4)
                                .then(|| 12.0 + f64::from(crossover_tg) / 1000.0),
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
                    });
                }
            }
        }
        let measured = Measured {
            candidates,
            points,
            wrong: Vec::new(),
        };
        let (entries, lines) = fit(&measured, &grid, 40).unwrap();
        let tuned = profile::search::render_overrides(&entries);
        assert!(
            tuned.contains("{ when = \"rows >= 16\", use = \"tiled\", crossover_tg = 64 }"),
            "{tuned}"
        );
        assert!(tuned.contains("{ use = \"qmv\", rows_max = 8, packs = 1, crossover_tg = 160 }"), "{tuned}");
        assert_eq!(lines.len(), 2);
        let file = assemble(
            &base,
            "Apple M4 Max",
            &facts,
            &tuned,
            &[("measured", "true".to_string())],
        )
        .unwrap();
        assert!(file.contains("device = \"Apple M4 Max\""), "{file}");
        assert!(file.trim_end().ends_with("measured = true"), "{file}");
    }

    /// The census is the catalog's: recomputed from `models/`, the 24
    /// heaviest bank shapes are the table's, in order.
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
                    if n < 100_000 && n >= 256 && k >= 256 {
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
        let (base, _) = crate::chip::resolve("Apple M4 Max", &crate::chip::facts(9, 40, true), None);
        let candidates = candidates(&base);
        let mut times = vec![None; candidates.len()];
        times[0] = Some(1.5e-4);
        times[3] = Some(2.25e-5);
        let m = Measured {
            candidates: candidates.clone(),
            points: vec![Point {
                rows: 8,
                n: 5120,
                k: 17408,
                bank: Bank { bits: 4, group: 64, fp16: true },
                times,
                baseline: Some(3),
            }],
            wrong: vec![("use = \"qmv\", rows_max = 8, packs = 2, crossover_tg = 160".into(), 3, "8 x 1 x 2: off".into())],
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
        let facts = crate::chip::facts(10, 40, true);
        let (base, _) = crate::chip::resolve("Apple M5 Max", &facts, None);
        let registry = crate::chip::registry();
        let knobs = profile::search::knobs(&base, SEARCH_SKIPS, "kernel.ane");
        let ids: Vec<&str> = knobs.iter().map(|k| k.id).collect();
        for want in ["kernel.sdpa.prefill", "kernel.gdn.scan", "kernel.ane.mlp", "fuse.mlp_swiglu", "compiler.launch_us"] {
            assert!(ids.contains(&want), "{want} not searched: {ids:?}");
        }
        assert!(!ids.contains(&"kernel.sdpa.kv_q8") && !ids.contains(&"kernel.qmm"));
        for k in &knobs {
            for (label, text) in &k.candidates {
                registry
                    .overlay(&base, text, base.target().clone(), Source::User("t".into()))
                    .unwrap_or_else(|e| panic!("{} = {label}: {e}", k.id));
            }
        }
    }

    #[test]
    fn a_load_reads_only_the_decisions_it_evaluates() {
        let (base, _) = crate::chip::resolve("Apple M4 Max", &crate::chip::facts(9, 40, true), None);
        let tuning = kernels_metal::Tuning::new(base.clone(), false, 40);
        let _ = tuning.qmm_mpp();
        assert!(base.read().is_empty(), "sizing at load is not a read: {:?}", base.read());
        let _ = tuning.qmm(&tuning.qmm_at(8, 4096, 4096, 4, 64));
        // The point reads the fp16 flag for its format.
        assert_eq!(base.read(), vec!["kernel.qmm", "kernel.qmm_fp16"]);
    }

}
