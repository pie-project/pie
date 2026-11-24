//! `kernel.dense` on Metal: the bf16 dense matmul's kernel (the tensor-op
//! tile, the few-row vector forms, the simdgroup row tiles) timed at each
//! row count over the served models' weight shapes, and fitted to a ladder.

use std::cell::RefCell;
use std::fmt::Write as _;

use kernels_metal::linear::gemm;
use kernels_metal::tuning::Dense;
use kernels_metal::{Encode, Tensor, Tuning};
use poem_ir::Dtype;
use profile::search::Line;
use profile::tune::{Candidate, Options, Sample};
use profile::{Decision, Profile, Source};

use super::{Depth, Watch, bf16, noise, seen, time_rounds, to_f32};
use crate::device::{Buffer, Context, Handles, Pipelines};
use crate::encode::Sink;

/// The bf16 dense weight shapes `(n, k)` the catalog's small dense models
/// serve, each model's matmuls (Qwen3-0.6B, Qwen3.5-0.8B, Gemma 4 E4B; an
/// 8B's 4096-wide layer), heads included where they fit the shape budget.
const SHAPES: &[(u32, u32)] = &[
    // Qwen3-0.6B
    (6144, 1024),
    (4096, 1024),
    (1024, 3072),
    (1024, 2048),
    (151_936, 1024),
    // Qwen3.5-0.8B
    (8192, 1024),
    (7168, 1024),
    (5120, 1024),
    (1024, 3584),
    (248_320, 1024),
    // Gemma 4 E4B
    (20480, 2560),
    (10752, 2560),
    (6144, 2560),
    (4096, 2560),
    (3072, 2560),
    (2048, 2560),
    (2560, 10240),
    (2560, 4096),
    (2560, 2048),
    // An 8B's layer
    (12288, 4096),
    (4096, 4096),
    (4096, 12288),
];

fn rows(depth: Depth) -> Vec<u32> {
    match depth {
        Depth::Quick => vec![2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 128, 512],
        Depth::Full => (2..=16)
            .chain([20, 24, 28, 32, 40, 48, 56, 64, 96, 128, 256, 512])
            .collect(),
    }
}

/// How much a point counts: half the mix on decode-sized rows (2 to 8), a
/// fifth on 9 to 64, the rest on prompt-sized rows; every shape alike.
fn weight(rows_grid: &[u32], shapes: usize, rows: u32) -> f64 {
    let band = |lo: u32, hi: u32| rows_grid.iter().filter(|&&r| lo <= r && r <= hi).count();
    let per_rows = match rows {
        0..=8 => 0.5 / band(0, 8) as f64,
        9..=64 => 0.2 / band(9, 64) as f64,
        _ => 0.3 / band(65, u32::MAX) as f64,
    };
    per_rows / shapes.max(1) as f64
}

fn candidates(family: u32) -> Vec<Dense> {
    [Dense::Mpp, Dense::Rows, Dense::Tile]
        .into_iter()
        .filter(|d| d.runs_on(family))
        .collect()
}

fn body(d: Dense) -> &'static str {
    match d {
        Dense::Mpp => "use = \"mpp\"",
        Dense::Rows => "use = \"rows\"",
        Dense::Tile => "use = \"tile\"",
    }
}

/// Whether a launch that fired `files` ran `d`'s kernel.
fn ran(files: &[&str], d: Dense) -> bool {
    let mpp = files.iter().any(|f| f.contains("linear/gemm_mpp.metal"));
    let simd = files.iter().any(|f| f.ends_with("linear/gemm.metal"));
    match d {
        Dense::Mpp => mpp,
        Dense::Rows | Dense::Tile => simd && !mpp,
    }
}

#[derive(Clone, Debug)]
struct Point {
    rows: u32,
    n: u32,
    k: u32,
    /// Seconds per launch for each candidate; `None` where it did not run.
    times: Vec<Option<f64>>,
    /// Each candidate's spread over its timed rounds, `(slowest - fastest)
    /// / median` (not kept in the measurements' text).
    spread: Vec<Option<f64>>,
    /// The candidate today's profile runs here.
    baseline: Option<usize>,
}

/// What `kernel.dense`'s bench measured, and what `--refit` reads back.
#[derive(Clone, Debug, Default)]
pub(super) struct Measured {
    candidates: Vec<Dense>,
    points: Vec<Point>,
    /// `(candidate, points it answered wrongly at, the first)`.
    pub(super) wrong: Vec<(String, usize, String)>,
}

impl Measured {
    pub(super) fn to_text(&self, device: &str) -> String {
        let mut out = format!("# pie tune measurements, kernel.dense, {device}\n");
        for d in &self.candidates {
            let _ = writeln!(out, "candidate {}", body(*d));
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
                "point {} {} {} {} {}",
                p.rows,
                p.n,
                p.k,
                p.baseline
                    .map_or_else(|| "-".to_string(), |b| b.to_string()),
                times.join(",")
            );
        }
        out
    }

    /// A file measured over another candidate list is refused.
    pub(super) fn from_text(text: &str, family: u32) -> Result<Measured, String> {
        let candidates = candidates(family);
        let bodies: Vec<&str> = candidates.iter().map(|d| body(*d)).collect();
        let mut seen = Vec::new();
        let mut out = Measured {
            candidates,
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
                out.points.push(Point {
                    rows: num(0)?,
                    n: num(1)?,
                    k: num(2)?,
                    baseline: f.get(3).and_then(|b| b.parse().ok()),
                    spread: Vec::new(),
                    times: f
                        .get(4)
                        .ok_or("times")?
                        .split(',')
                        .map(|t| t.parse().ok())
                        .collect(),
                });
            }
        }
        if seen != bodies {
            return Err(
                "the kernel.dense measurements were taken over another candidate list; measure again"
                    .into(),
            );
        }
        Ok(out)
    }
}

/// The timed rounds a curve's point takes, and a point slower than both its
/// neighbours takes again.
const ROUNDS: usize = 7;
const RETIME_ROUNDS: usize = 21;

/// More than any Apple GPU's system-level cache.
const STREAM_BYTES: u64 = 256 << 20;
/// The most one shape's weight copies take.
const SHAPE_BYTES: u64 = 1 << 30;

/// Time every candidate that runs as itself, and right, at each row count
/// over each shape.
pub(super) fn measure(
    device: &Context,
    base: &Profile,
    cores: Option<u32>,
    depth: Depth,
    say: &mut dyn FnMut(&str),
) -> Result<Measured, String> {
    measure_at(device, base, cores, SHAPES, &rows(depth), say)
}

/// [`measure`] over `shapes` at the row counts `grid`.
fn measure_at(
    device: &Context,
    base: &Profile,
    cores: Option<u32>,
    shapes: &[(u32, u32)],
    grid: &[u32],
    say: &mut dyn FnMut(&str),
) -> Result<Measured, String> {
    let family = device.family();
    let candidates = candidates(family);
    let tuning = |overrides: &str| -> Result<Tuning, String> {
        let p = crate::chip::registry()
            .overlay(
                base,
                &format!("[kernel]\n{overrides}"),
                base.target().clone(),
                Source::User("tune".into()),
            )
            .map_err(|e| e.to_string())?;
        Ok(Tuning::new(p, false, cores).with_family(family))
    };
    let forced: Vec<Tuning> = candidates
        .iter()
        .map(|d| match d {
            Dense::Tile => tuning("dense = [{ use = \"tile\" }]\n"),
            _ => tuning(&format!(
                "dense = [{{ {} }}, {{ use = \"tile\" }}]\n",
                body(*d)
            )),
        })
        .collect::<Result<_, _>>()?;
    let today = tuning("")?;
    let widest = *grid.iter().max().ok_or("no rows to measure at")?;
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let mut out = Measured {
        candidates: candidates.clone(),
        ..Measured::default()
    };
    for (at, &(n, k)) in shapes.iter().enumerate() {
        let (nn, kk) = (n as usize, k as usize);
        let weight_bytes = u64::from(n) * u64::from(k) * 2;
        if weight_bytes > SHAPE_BYTES / 2 {
            say(&format!(
                "  kernel.dense: skipped {n} x {k}, past the shape budget"
            ));
            continue;
        }
        let alloc = |bytes: u64| Buffer::zeroed(device, bytes).map_err(|e| e.to_string());
        // Activations for the widest row count, and their values.
        let mut act_bytes = vec![0u8; widest as usize * kk * 2];
        let mut act = vec![0.0f64; widest as usize * kk];
        for i in 0..widest as usize * kk {
            let v = 0.02 * (f32::from(noise(i as u64 ^ 0x77) % 16) - 8.0);
            act_bytes[2 * i..2 * i + 2].copy_from_slice(&bf16(v));
            act[i] = seen(v);
        }
        let mut act_b = alloc(act_bytes.len() as u64)?;
        act_b.write(0, &act_bytes).map_err(|e| e.to_string())?;
        let out_b = alloc(u64::from(widest) * u64::from(n) * 2)?;
        // The weight, in enough copies to stream past the cache.
        let mut w_bytes = vec![0u8; nn * kk * 2];
        for i in 0..nn * kk {
            w_bytes[2 * i..2 * i + 2].copy_from_slice(&bf16(weight_at(i)));
        }
        let copies = STREAM_BYTES
            .div_ceil(weight_bytes)
            .clamp(1, (SHAPE_BYTES / weight_bytes).max(1));
        let mut buffers = Vec::new();
        let mut weights = Vec::new();
        for _ in 0..copies {
            let mut b = alloc(weight_bytes)?;
            b.write(0, &w_bytes).map_err(|e| e.to_string())?;
            weights.push(handles.bind(&b, 0, b.bytes()).map_err(|e| e.to_string())?);
            buffers.push(b);
        }
        drop(w_bytes);
        let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).map_err(|e| e.to_string());
        let (ha, ho) = (bind(&act_b)?, bind(&out_b)?);
        let launch_at = |m: u32| {
            let weights = &weights;
            move |sink: &dyn Encode, tuning: &Tuning, copy: usize| {
                gemm::matmul(
                    sink,
                    tuning,
                    Tensor::new(ha, m, k, Dtype::Bf16),
                    Tensor::new(weights[copy % weights.len()], n, k, Dtype::Bf16),
                    Tensor::new(ho, m, n, Dtype::Bf16),
                )
            }
        };
        let first = out.points.len();
        for &m in grid {
            let launch = launch_at(m);
            let mut live: Vec<usize> = Vec::new();
            for (c, d) in candidates.iter().enumerate() {
                if forced[c].dense(m, n, k) != *d {
                    continue;
                }
                let frame = device.frame().map_err(|e| e.to_string())?;
                let sink = Sink::new(device, &frame, &pipelines, &handles);
                let watch = Watch {
                    inner: &sink,
                    files: RefCell::new(Vec::new()),
                };
                let launched = launch(&watch, &forced[c], 0).is_ok();
                let as_itself = ran(&watch.files.borrow(), *d);
                drop(watch);
                drop(sink);
                frame.commit().map_err(|e| e.to_string())?;
                if !launched || !as_itself {
                    continue;
                }
                if let Some(why) = wrong(&handles, ho, m, n, k, &act) {
                    let first = format!("{m} x {n} x {k}: {why}");
                    let name = body(*d).to_string();
                    match out.wrong.iter_mut().find(|(b, ..)| *b == name) {
                        Some(entry) => entry.1 += 1,
                        None => out.wrong.push((name, 1, first)),
                    }
                    continue;
                }
                live.push(c);
            }
            // Each point the fastest of its rounds: noise only slows a
            // launch, so the fastest is the kernel's, and two measuring
            // boots agree on it where a median moves with the device's
            // state (engine-gpu's curve rule).
            let times = time_rounds(device, &pipelines, &handles, &live, ROUNDS, |sink, c, copy| {
                launch(sink, &forced[c], copy)
            })?;
            let mut row = vec![None; candidates.len()];
            let mut spread = vec![None; candidates.len()];
            for (c, (t, lo, hi)) in live.iter().zip(times) {
                row[*c] = Some(lo);
                spread[*c] = Some((hi - lo) / t.max(1e-12));
            }
            let baseline = candidates
                .iter()
                .position(|d| *d == today.dense(m, n, k))
                .filter(|&c| row[c].is_some());
            out.points.push(Point {
                rows: m,
                n,
                k,
                times: row,
                spread,
                baseline,
            });
        }
        // A point slower than both its neighbours along the rows (in
        // TFLOP/s) is timed again over more rounds, its fastest kept where
        // faster: re-measured, never overridden from its neighbours.
        for c in 0..candidates.len() {
            let along: Vec<usize> = (first..out.points.len())
                .filter(|&i| out.points[i].times[c].is_some())
                .collect();
            for w in along.windows(3) {
                let (before, at, after) = (w[0], w[1], w[2]);
                let rate = |i: usize| {
                    let p = &out.points[i];
                    f64::from(p.rows) / p.times[c].unwrap_or(f64::INFINITY)
                };
                if rate(at) >= rate(before) || rate(at) >= rate(after) {
                    continue;
                }
                let m = out.points[at].rows;
                let launch = launch_at(m);
                let again = time_rounds(device, &pipelines, &handles, &[c], RETIME_ROUNDS, |sink, c, copy| {
                    launch(sink, &forced[c], copy)
                })?;
                if let (Some(&(_, lo, _)), Some(old)) = (again.first(), out.points[at].times[c]) {
                    out.points[at].times[c] = Some(old.min(lo));
                }
            }
        }
        drop(buffers);
        say(&format!(
            "  kernel.dense: {}/{} shapes ({n} x {k})",
            at + 1,
            shapes.len()
        ));
    }
    Ok(out)
}

/// The weight's element `i` (row-major, `n` rows of `k`): a pattern, not
/// zeros, recomputed for the reference rather than held on the host.
fn weight_at(i: usize) -> f32 {
    0.01 * (f32::from(noise(i as u64 ^ 0x5151) % 16) - 8.0)
}

/// Why the launch just run answered wrongly, if it did: a sample of outputs
/// against the f64 reference, allowing bf16 rounding.
fn wrong(handles: &Handles, ho: u32, m: u32, n: u32, k: u32, act: &[f64]) -> Option<String> {
    let raw = handles.read(ho, u64::from(m) * u64::from(n) * 2).ok()?;
    let (nn, kk) = (n as usize, k as usize);
    for r in [0, m as usize / 2, m as usize - 1] {
        for s in 0..16usize {
            let col = (s * 7919 + r * 104_729) % nn;
            let (mut want, mut mass) = (0.0f64, 0.0f64);
            for c in 0..kk {
                let t = act[r * kk + c] * seen(weight_at(col * kk + c));
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

/// The fitted `kernel.dense` ladder and its report line.
pub(super) fn fit(measured: &Measured, depth: Depth) -> Result<((String, String), Line), String> {
    fit_at(measured, &rows(depth), SHAPES.len())
}

/// [`fit`] for measurements over `shapes` shapes at the row counts `grid`.
fn fit_at(
    measured: &Measured,
    grid: &[u32],
    shapes: usize,
) -> Result<((String, String), Line), String> {
    let spec = Dense::spec();
    let candidates: Vec<Candidate> = measured
        .candidates
        .iter()
        .map(|d| Candidate {
            body: body(*d).to_string(),
            terminal: matches!(d, Dense::Tile),
        })
        .collect();
    // Coordinates in `kernel.dense`'s variable order: rows; n, k.
    let samples: Vec<Sample> = measured
        .points
        .iter()
        .map(|p| Sample {
            at: vec![i64::from(p.rows), i64::from(p.n), i64::from(p.k)],
            weight: weight(grid, shapes, p.rows),
            times: p.times.clone(),
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
    Ok((
        (spec.id.to_string(), ladder),
        Line {
            id: spec.id,
            text,
            measured: true,
        },
    ))
}

/// The bf16 matrices a model's contract holds, `(n, k)` as a matmul reads
/// them (rows of the weight, the contraction), each once, ascending.
fn shapes_of(contract: &checkpoint::contract::ModelContract) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = contract
        .tensors
        .iter()
        .chain(contract.groups.iter().flat_map(|g| g.tensors.iter()))
        .filter(|t| t.encoding == checkpoint::types::Encoding::Raw(checkpoint::types::DType::Bf16))
        .filter_map(|t| match t.shape.as_deref() {
            Some(&[n, k]) => Some((u32::try_from(n).ok()?, u32::try_from(k).ok()?)),
            _ => None,
        })
        .filter(|&(n, k)| n > 1 && k > 1)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The row counts a load measures at: the decode and verify rows one by one
/// to 8, where the kernels cross, then the prompt rows a few at a time.
const LOAD_ROWS: [u32; 15] = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 128, 512];

/// The rungs of a `kernel.dense` value as TOML array elements, one a line.
fn rungs_of(value: &str) -> Vec<String> {
    let value = value.trim();
    let Some(inner) = value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) else {
        return vec![value.to_string()];
    };
    inner
        .lines()
        .map(|l| l.trim().trim_end_matches(',').trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

/// `measured` (fitted over `shapes` alone) held to those shapes, then
/// `base`'s rungs for every other shape: each measured rung gains the
/// shapes' guard, and its terminal rung becomes a guarded one.
fn composed(measured: &str, shapes: &[(u32, u32)], base: &str) -> String {
    let guard = shapes
        .iter()
        .map(|(n, k)| format!("(n == {n} and k == {k})"))
        .collect::<Vec<_>>()
        .join(" or ");
    let mut out = String::from("[\n");
    for rung in rungs_of(measured) {
        let body = rung.trim_start_matches('{').trim_end_matches('}').trim();
        let line = if let Some(rest) = body.strip_prefix("when = \"") {
            let (when, pick) = rest.split_once('"').unwrap_or((rest, ""));
            format!("  {{ when = \"({when}) and ({guard})\"{pick} }},\n")
        } else if body.starts_with('"') {
            format!("  {{ when = \"{guard}\", use = {body} }},\n")
        } else {
            format!("  {{ when = \"{guard}\", {body} }},\n")
        };
        out.push_str(&line);
    }
    for rung in rungs_of(base) {
        out.push_str(&format!("  {rung},\n"));
    }
    out.push(']');
    out
}

/// The curve `d` is stored as (`profile::curves::KernelCurve` kind and key):
/// the few-row vector forms as a vector kernel of 8 rows, the simdgroup
/// row tiles as a tile (key 0: its row tile follows the rows), the
/// tensor-op tile as an MPP tile of 32 rows.
fn curve_id(d: Dense) -> (u32, u32) {
    use profile::curves::KernelCurve;
    match d {
        Dense::Rows => (KernelCurve::VECTOR, Dense::ROWS_MAX),
        Dense::Tile => (KernelCurve::TILE, 0),
        Dense::Mpp => (KernelCurve::MPP, 32),
    }
}

/// Whether `d` can run `n x k` at all (the tensor-op tile only on whole
/// 64-wide column and contraction steps): a shape it cannot run needs no
/// curve of it.
fn runs_shape(d: Dense, n: u32, k: u32) -> bool {
    !matches!(d, Dense::Mpp) || (n % Dense::MPP_STEP == 0 && k % Dense::MPP_STEP == 0)
}

/// The source file `d`'s kernels are built from: its curve's key.
fn curve_file(d: Dense) -> &'static str {
    match d {
        Dense::Mpp => "linear/gemm_mpp.metal",
        Dense::Rows | Dense::Tile => "linear/gemm.metal",
    }
}

/// `measured`'s points as curves, one a shape and candidate: TFLOP/s by
/// rows with each point's spread, keyed by the candidate's kernel source.
/// The vector forms end at 8 rows, the most they take: their curve stops
/// there, as one beaten past it.
fn curves_of(measured: &Measured) -> Vec<profile::curves::KernelCurve> {
    use profile::curves::KernelCurve;
    let mut out: Vec<KernelCurve> = Vec::new();
    for p in &measured.points {
        for (c, d) in measured.candidates.iter().enumerate() {
            let Some(t) = p.times.get(c).copied().flatten() else {
                continue;
            };
            let (kind, key) = curve_id(*d);
            let flops = 2.0 * f64::from(p.rows) * f64::from(p.n) * f64::from(p.k);
            let tflops = flops / t.max(1e-12) / 1e12;
            let spread = p.spread.get(c).copied().flatten().unwrap_or(0.0);
            match out
                .iter_mut()
                .find(|curve| curve.id() == (kind, key, p.n, p.k))
            {
                Some(curve) => {
                    curve.curve.push((p.rows, tflops));
                    curve.spread.push((p.rows, spread));
                }
                None => out.push(KernelCurve {
                    kind,
                    key,
                    n: p.n,
                    k: p.k,
                    curve: vec![(p.rows, tflops)],
                    stop: None,
                    spread: vec![(p.rows, spread)],
                    by: crate::stored::by(profile::rates::KERNEL_CURVES, &[curve_file(*d)]),
                }),
            }
        }
    }
    let past = |rows: u32| LOAD_ROWS.iter().copied().find(|&r| r > rows);
    for curve in &mut out {
        curve.curve.sort_by_key(|(rows, _)| *rows);
        curve.spread.sort_by_key(|(rows, _)| *rows);
        if curve.kind == KernelCurve::VECTOR {
            curve.stop = past(Dense::ROWS_MAX).map(|rows| (rows, KernelCurve::BEATEN));
        }
    }
    out
}

/// The points `curves` state for `shapes`, one a shape and row count, in
/// [`Measured`]'s form for the fit: each candidate's time from its TFLOP/s;
/// `today` the route the load would take without them.
fn points_of(
    curves: &[profile::curves::KernelCurve],
    candidates: &[Dense],
    shapes: &[(u32, u32)],
    today: &kernels_metal::Tuning,
) -> Measured {
    let mut points = Vec::new();
    for &(n, k) in shapes {
        for &rows in &LOAD_ROWS {
            let times: Vec<Option<f64>> = candidates
                .iter()
                .map(|d| {
                    let (kind, key) = curve_id(*d);
                    curves
                        .iter()
                        .find(|c| c.id() == (kind, key, n, k))
                        .and_then(|c| c.curve.iter().find(|(r, _)| *r == rows))
                        .map(|(_, tflops)| {
                            2.0 * f64::from(rows) * f64::from(n) * f64::from(k)
                                / (tflops.max(1e-9) * 1e12)
                        })
                })
                .collect();
            if times.iter().all(Option::is_none) {
                continue;
            }
            let spread: Vec<Option<f64>> = candidates
                .iter()
                .map(|d| {
                    let (kind, key) = curve_id(*d);
                    curves
                        .iter()
                        .find(|c| c.id() == (kind, key, n, k))
                        .and_then(|c| c.spread.iter().find(|(r, _)| *r == rows))
                        .map(|(_, s)| *s)
                })
                .collect();
            let baseline = candidates
                .iter()
                .position(|d| *d == today.dense(rows, n, k))
                .filter(|&c| times[c].is_some());
            points.push(Point {
                rows,
                n,
                k,
                times,
                spread,
                baseline,
            });
        }
    }
    Measured {
        candidates: candidates.to_vec(),
        points,
        wrong: Vec::new(),
    }
}

/// The ladder a load's measurement states: at each measured shape and row
/// count the candidate that ran fastest, adjacent row counts with the same
/// winner merged into one rung (a band runs from its first row count to the
/// row before the next band's), no threshold and no cap, so a model's own
/// ladder is its measurement. Where candidates' timed rounds overlap the
/// winner's, the measurement does not order them, and the pick there is the
/// one that won a neighbouring row count (below first), else `today`'s, so a
/// repeat measurement cannot flip it on noise. Returns the ladder (closed by
/// the simdgroup tiles, which run anywhere) and what to say of it.
fn winners(measured: &Measured, today: &kernels_metal::Tuning) -> (String, String) {
    let mut shapes: Vec<(u32, u32)> = measured.points.iter().map(|p| (p.n, p.k)).collect();
    shapes.dedup();
    let mut rungs = Vec::new();
    let (mut points, mut unordered, mut today_cost, mut won_cost) = (0usize, 0usize, 0f64, 0f64);
    for &(n, k) in &shapes {
        let mut at: Vec<&Point> = measured
            .points
            .iter()
            .filter(|p| (p.n, p.k) == (n, k))
            .collect();
        at.sort_by_key(|p| p.rows);
        // Per row count: the winner, or every candidate the winner's rounds
        // do not separate it from.
        let picks: Vec<Vec<usize>> = at
            .iter()
            .map(|p| {
                let Some(best) = (0..p.times.len())
                    .filter(|&c| p.times[c].is_some())
                    .min_by(|&a, &b| {
                        let t = |c: usize| p.times[c].unwrap_or(f64::INFINITY);
                        t(a).total_cmp(&t(b))
                    })
                else {
                    return Vec::new();
                };
                // A candidate's rounds span from its fastest (its time) to
                // its time stretched by its stated spread.
                let range = |c: usize| {
                    let t = p.times.get(c).copied().flatten()?;
                    let s = p.spread.get(c).copied().flatten().unwrap_or(0.0);
                    Some((t, t * (1.0 + s)))
                };
                match range(best) {
                    Some((lo, hi)) => (0..p.times.len())
                        .filter(|&c| {
                            c == best
                                || range(c).is_some_and(|(clo, chi)| clo <= hi && lo <= chi)
                        })
                        .collect(),
                    None => vec![best],
                }
            })
            .collect();
        let mut chosen: Vec<Option<usize>> = picks
            .iter()
            .map(|tied| (tied.len() == 1).then(|| tied[0]))
            .collect();
        for i in 0..at.len() {
            if chosen[i].is_some() || picks[i].is_empty() {
                continue;
            }
            unordered += 1;
            let neighbour = (0..i)
                .rev()
                .find_map(|j| chosen[j])
                .filter(|c| picks[i].contains(c))
                .or_else(|| {
                    (i + 1..at.len())
                        .find_map(|j| chosen[j])
                        .filter(|c| picks[i].contains(c))
                });
            let profile = measured
                .candidates
                .iter()
                .position(|d| *d == today.dense(at[i].rows, n, k))
                .filter(|c| picks[i].contains(c));
            chosen[i] = neighbour.or(profile).or(Some(picks[i][0]));
        }
        for (i, p) in at.iter().enumerate() {
            if let Some(t) = chosen[i].and_then(|c| p.times[c]) {
                points += 1;
                won_cost += 1.0;
                today_cost += p.baseline.and_then(|b| p.times[b]).map_or(1.0, |b| b / t);
            }
        }
        // Merge adjacent row counts with one winner into a band.
        let mut i = 0;
        while i < at.len() {
            let Some(c) = chosen[i] else {
                i += 1;
                continue;
            };
            let mut j = i + 1;
            while j < at.len() && chosen[j] == Some(c) {
                j += 1;
            }
            let lo = at[i].rows;
            let rows = if j < at.len() {
                format!("rows >= {lo} and rows <= {}", at[j].rows - 1)
            } else {
                format!("rows >= {lo}")
            };
            rungs.push(format!(
                "  {{ when = \"n == {n} and k == {k} and {rows}\", {} }},",
                body(measured.candidates[c])
            ));
            i = j;
        }
    }
    let said = format!(
        "{} rungs, each measured point's winner ({unordered} of {points} points not ordered \
         by their timed rounds, settled by a neighbour or the profile); the profile's ladder \
         runs at {:.3}x them",
        rungs.len(),
        if won_cost > 0.0 { today_cost / won_cost } else { 1.0 }
    );
    (format!("[\n{}\n  {{ use = \"tile\" }},\n]", rungs.join("\n")), said)
}

/// `chip` with `kernel.dense` stated by this model's own bf16 shapes,
/// from each shape's curves (`[rates] kernel_curves`, one a
/// candidate kernel): taken where the user's file or the device's committed
/// measured profile states them under a matching key (`crate::stored`: the
/// device, its OS release, the kernel's source), else measured here before
/// ready and stored to the user's file. Under `measure = false` a shape no
/// layer states is not measured: it keeps the profile's route, named. The
/// ladder is each measured point's winner ([`winners`], no fit and no
/// threshold), held to the shapes it was measured over, the profile's
/// own ladder after it for every other shape (a head past the measurement's
/// weight budget). A kernel that answers wrongly refuses the load; a
/// measurement that fails otherwise (memory refused at boot) leaves the
/// profile's route serving, said.
pub(crate) fn at_load(
    device: &Context,
    chip: Profile,
    profiles: Option<&std::path::Path>,
    contract: &checkpoint::contract::ModelContract,
    measuring: engine::rates::Measuring,
) -> Result<Profile, String> {
    let shapes = shapes_of(contract);
    if shapes.is_empty() {
        return Ok(chip);
    }
    let name = device.name().to_string();
    let family = device.family();
    let candidates = candidates(family);
    let stated: Vec<profile::curves::KernelCurve> = chip
        .rates()
        .kernel_curves
        .clone()
        .unwrap_or_default();
    let top = *LOAD_ROWS.last().expect("rows");
    let lacks = |curves: &[profile::curves::KernelCurve], n: u32, k: u32| {
        candidates.iter().filter(|d| runs_shape(**d, n, k)).any(|d| {
            let (kind, key) = curve_id(*d);
            !curves
                .iter()
                .any(|c| c.id() == (kind, key, n, k) && c.serves(top))
        })
    };
    let lacking: Vec<(u32, u32)> = shapes
        .iter()
        .copied()
        .filter(|&(n, k)| u64::from(n) * u64::from(k) * 2 <= SHAPE_BYTES / 2)
        .filter(|&(n, k)| lacks(&stated, n, k))
        .collect();
    let mut chip = chip;
    let mut curves = stated;
    // A shape whose curves are stored under a stale key (a kernel's code or
    // the method changed) serves by those figures, a fair estimate, until
    // the next export round or `pie tune` refreshes them
    // (`engine::rates::StaleRates::Served`): nothing is timed before ready
    // that a figure exists for, and nothing behind serving (a timing lays
    // buffers of its own, and nothing is laid after ready). Only a shape no
    // layer states at all is timed here.
    let stale: Vec<profile::curves::KernelCurve> = chip
        .rates()
        .stale
        .as_ref()
        .and_then(|stale| stale.kernel_curves.clone())
        .unwrap_or_default();
    let lacking: Vec<(u32, u32)> = if stale.is_empty() || lacking.is_empty() {
        lacking
    } else {
        let with: Vec<profile::curves::KernelCurve> = curves
            .iter()
            .chain(stale.iter().filter(|s| !curves.iter().any(|c| c.id() == s.id())))
            .cloned()
            .collect();
        let (still, served): (Vec<(u32, u32)>, Vec<(u32, u32)>) =
            lacking.iter().partition(|&&(n, k)| lacks(&with, n, k));
        if !served.is_empty() {
            for &(n, k) in &served {
                tracing::info!(
                    "kernel.dense: {n}x{k}: served by the curves stored under their stale key, \
                     until the next export round or `pie tune` times them again"
                );
            }
            curves = with;
        }
        still
    };
    if !lacking.is_empty() && measuring == engine::rates::Measuring::Skip {
        let mut tuned = chip.tuned().clone();
        for &(n, k) in &lacking {
            tracing::info!(
                "kernel.dense: skipped: {n}x{k}: nothing timed (`[engine] measure = false`) and no \
                 layer states its curves under a matching key; the profile's route serves it"
            );
            tuned.skipped.push(format!("dense curves at {n}x{k}"));
        }
        chip = chip.with_tuned(tuned);
    } else if !lacking.is_empty() {
        let began = std::time::Instant::now();
        let measured =
            match measure_at(device, &chip, device.cores(), &lacking, &LOAD_ROWS, &mut |_| {}) {
                Ok(m) => m,
                Err(why) => {
                    eprintln!("{name}: kernel.dense not measured: {why}; the profile's route serves");
                    return Ok(chip);
                }
            };
        if let Some((body, n, first)) = measured.wrong.first() {
            return Err(format!(
                "kernel.dense at load: `{body}` answered wrongly at {n} points; first at {first}"
            ));
        }
        let fresh = curves_of(&measured);
        tracing::info!(
            "kernel.dense: measured {} curves over {} of the model's bf16 shapes in {:.1} s, \
             stored under their keys",
            fresh.len(),
            lacking.len(),
            began.elapsed().as_secs_f64()
        );
        if let Some(profiles) = profiles {
            let save = engine::rates::SaveTo {
                path: crate::chip::SHELF.user_path(profiles, &name),
                provenance: crate::rates::provenance(&name),
                registry: crate::chip::registry(),
            };
            let stored = profile::rates::Rates {
                kernel_curves: Some(fresh.clone()),
                measured_on: Some(crate::stored::measured_on(&name)),
                ..profile::rates::Rates::default()
            };
            if let Err(why) = engine::rates::save_rates(&save, &stored) {
                eprintln!("{name}: kernel.dense curves not stored: {why}");
            }
        }
        let mut tuned = chip.tuned().clone();
        tuned.timed.push(format!("dense curves ({})", fresh.len()));
        chip = chip.with_tuned(tuned);
        curves.retain(|c| !fresh.iter().any(|f| f.id() == c.id()));
        curves.extend(fresh);
    } else {
        tracing::info!(
            "kernel.dense: every curve of the model's {} bf16 shapes stored or committed (a stale \
             one served by its figure until the next export); none timed",
            shapes.len()
        );
    }
    let fitted: Vec<(u32, u32)> = shapes
        .iter()
        .copied()
        .filter(|&(n, k)| {
            candidates.iter().filter(|d| runs_shape(**d, n, k)).all(|d| {
                let (kind, key) = curve_id(*d);
                curves.iter().any(|c| c.id() == (kind, key, n, k))
            })
        })
        .collect();
    if fitted.is_empty() {
        return Ok(chip);
    }
    let today = kernels_metal::Tuning::new(chip.clone(), false, device.cores()).with_family(family);
    let points = points_of(&curves, &candidates, &fitted, &today);
    let (ladder, said) = winners(&points, &today);
    tracing::info!(
        "kernel.dense: the model's {} measured shapes' ladder (ladder {:016x}): {said}",
        fitted.len(),
        ladder
            .bytes()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
    );
    let spec = Dense::spec();
    let base = chip
        .entries()
        .find(|(s, ..)| s.id == spec.id)
        .map(|(s, rungs, _)| profile::render::value_toml(s, rungs))
        .unwrap_or_else(|| "\"tile\"".to_string());
    let over = |value: &str| {
        crate::chip::registry().overlay(
            &chip,
            &format!("[kernel]\ndense = {value}\n"),
            chip.target().clone(),
            chip.source().clone(),
        )
    };
    match over(&composed(&ladder, &fitted, &base)) {
        Ok(p) => Ok(p),
        // A composition the registry refuses (a profile rung the measured
        // ones leave unreachable): the measured ladder alone serves.
        Err(_) => over(&ladder).map_err(|e| format!("kernel.dense fitted at load: {e}")),
    }
}
