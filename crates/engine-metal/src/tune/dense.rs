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

use super::{Depth, Watch, bf16, noise, seen, time, to_f32};
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
        for &m in grid {
            let launch = |sink: &dyn Encode, tuning: &Tuning, copy: usize| {
                gemm::matmul(
                    sink,
                    tuning,
                    Tensor::new(ha, m, k, Dtype::Bf16),
                    Tensor::new(weights[copy % weights.len()], n, k, Dtype::Bf16),
                    Tensor::new(ho, m, n, Dtype::Bf16),
                )
            };
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
            let times = time(device, &pipelines, &handles, &live, |sink, c, copy| {
                launch(sink, &forced[c], copy)
            })?;
            let mut row = vec![None; candidates.len()];
            for (c, t) in live.iter().zip(times) {
                row[*c] = Some(t);
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
                baseline,
            });
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

/// FNV-1a over `bytes`, continuing from `hash`: a key that reads the same
/// on every build, unlike the standard library's hasher.
fn fnv(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// The OS build: the kernels compile with its Metal compiler, so an update
/// can move where they cross.
fn os_build() -> String {
    #[cfg(target_vendor = "apple")]
    {
        objc2_foundation::NSProcessInfo::processInfo()
            .operatingSystemVersionString()
            .to_string()
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        String::new()
    }
}

/// What a measured `kernel.dense` stands on: the device, the OS build, the
/// kernels it timed (their Metal sources) and the shapes. Not the pie
/// version: a build that changes neither kernel keeps the measurement.
fn load_key(device: &str, os: &str, shapes: &[(u32, u32)]) -> u64 {
    let mut hash = fnv(0xcbf2_9ce4_8422_2325, device.as_bytes());
    hash = fnv(hash, os.as_bytes());
    for file in ["linear/gemm.metal", "linear/gemm_mpp.metal"] {
        hash = fnv(hash, file.as_bytes());
        hash = fnv(hash, kernels_metal::sources::source(file).unwrap_or("").as_bytes());
    }
    for &(n, k) in shapes {
        hash = fnv(hash, &n.to_le_bytes());
        hash = fnv(hash, &k.to_le_bytes());
    }
    hash
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

/// `chip` with `kernel.dense` measured over this model's own bf16 shapes:
/// where the kernels cross (the tensor-op tile, the few-row vector forms
/// and the simdgroup row tiles on Apple10; the vector forms and the tiles
/// on every other GPU) the shapes decide with the rows, and a fixed rule
/// cost an M5 Pro's 8-row decode 16-19% (the tile from 8 rows) and an M1
/// Max's Gemma 4 E4B 29% (the vector forms to 8 rows). The ladder is fitted at the rows a load serves and kept under
/// `profiles` (`metal/dense/<device>-<key>.toml`, keyed on the device, the
/// OS build, the two kernels' sources and the shapes), so a later load of
/// the same shapes reads it back and measures nothing. The profile's own
/// `kernel.dense` (a checked-in device file's, the user's, or the default)
/// stays the route before a measurement and for every shape not measured.
/// A GPU without tensor ops never times the tensor-op tile. A kernel that
/// answers wrongly refuses the load; a measurement that fails otherwise
/// (memory refused at boot) leaves the profile's route serving, said.
pub(crate) fn at_load(
    device: &Context,
    chip: Profile,
    profiles: Option<&std::path::Path>,
    contract: &checkpoint::contract::ModelContract,
) -> Result<Profile, String> {
    let shapes = shapes_of(contract);
    if shapes.is_empty() {
        return Ok(chip);
    }
    let name = device.name().to_string();
    let key = load_key(&name, &os_build(), &shapes);
    let path = profiles.map(|p| {
        p.join("metal")
            .join("dense")
            .join(format!("{}-{key:016x}.toml", profile::id_of(&name)))
    });
    let kept = path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|text| {
            text.split_once("dense = ")
                .map(|(_, ladder)| ladder.trim().to_string())
        });
    let ladder = match kept {
        Some(ladder) => ladder,
        None => {
            let began = std::time::Instant::now();
            let measured =
                match measure_at(device, &chip, device.cores(), &shapes, &LOAD_ROWS, &mut |_| {}) {
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
            let ((_, ladder), line) = match fit_at(&measured, &LOAD_ROWS, shapes.len()) {
                Ok(fitted) => fitted,
                Err(why) => {
                    eprintln!("{name}: kernel.dense not measured: {why}; the profile's route serves");
                    return Ok(chip);
                }
            };
            eprintln!(
                "{name}: measured kernel.dense over the model's {} bf16 shapes in {:.1} s: {}",
                shapes.len(),
                began.elapsed().as_secs_f64(),
                line.text
            );
            if let Some(path) = &path {
                let text = format!(
                    "# kernel.dense measured at load on {name}, over {} bf16 shapes \
                     ({}) at rows {LOAD_ROWS:?}; key {key:016x}: the device, the OS build, \
                     linear/gemm.metal and linear/gemm_mpp.metal, the shapes. {}\n\
                     dense = {}\n",
                    shapes.len(),
                    shapes
                        .iter()
                        .map(|(n, k)| format!("{n}x{k}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                    line.text,
                    ladder.trim()
                );
                let written = path
                    .parent()
                    .map_or(Ok(()), std::fs::create_dir_all)
                    .and_then(|()| std::fs::write(path, text));
                if let Err(e) = written {
                    eprintln!("{name}: kernel.dense not kept at {}: {e}", path.display());
                }
            }
            ladder
        }
    };
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
    match over(&composed(&ladder, &shapes, &base)) {
        Ok(p) => Ok(p),
        // A composition the registry refuses (a profile rung the measured
        // ones leave unreachable): the measured ladder alone serves.
        Err(_) => over(&ladder).map_err(|e| {
            format!(
                "kernel.dense measured at load{}: {e}",
                path.map(|p| format!(" ({})", p.display()))
                    .unwrap_or_default()
            )
        }),
    }
}
