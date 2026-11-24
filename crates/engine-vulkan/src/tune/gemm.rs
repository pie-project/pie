//! The `kernel.gemm.path` bench: the tiled (register-tiled from 96 rows) and
//! cooperative-matrix dense matmuls timed over the bf16 projections of the
//! dense models pie ships, by rows, each checked against an f64 reference;
//! the crossover is fitted over `rows`.

use std::fmt::Write as _;

use crate::tuning::GemmPath;
use profile::Decision as _;
use profile::search::{Entries, Line};
use profile::tune::{Candidate, Fit, Options, Sample};

use super::Depth;

/// The bf16 projections of a dense layer, `(n, k, per layer)`: Qwen3-0.6B's
/// (q; k and v; o; gate and up; down), then Qwen3-8B's.
pub const CENSUS: &[(u32, u32, u32)] = &[
    (2048, 1024, 1),
    (1024, 1024, 2),
    (1024, 2048, 1),
    (3072, 1024, 2),
    (1024, 3072, 1),
    (4096, 4096, 2),
    (1024, 4096, 2),
    (12288, 4096, 2),
    (4096, 12288, 1),
];

#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub rows: Vec<u32>,
    /// `(n, k, weight)`: the shape's share of its layer's projections.
    pub shapes: Vec<(u32, u32, f64)>,
}

impl Grid {
    #[must_use]
    pub fn of(depth: Depth) -> Grid {
        let rows = match depth {
            Depth::Quick => vec![4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048],
            Depth::Full => (4..=16)
                .chain([20, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192])
                .chain([256, 384, 512, 768, 1024, 2048, 4096])
                .collect(),
        };
        let layer = |from: usize, to: usize| -> u32 { CENSUS[from..to].iter().map(|s| s.2).sum() };
        let (small, large) = (layer(0, 5), layer(5, CENSUS.len()));
        let shapes = CENSUS
            .iter()
            .enumerate()
            .map(|(i, &(n, k, c))| {
                let of = if i < 5 { small } else { large };
                (n, k, f64::from(c) / f64::from(of))
            })
            .collect();
        Grid { rows, shapes }
    }
}

/// Every path the registry holds.
#[must_use]
pub fn candidates() -> Vec<GemmPath> {
    vec![GemmPath::Tiled, GemmPath::Coopmat]
}

#[must_use]
pub fn body(p: &GemmPath) -> String {
    match p {
        GemmPath::Tiled => "use = \"tiled\"".to_string(),
        GemmPath::Coopmat => "use = \"coopmat\"".to_string(),
    }
}

#[derive(Clone, Debug)]
pub struct Point {
    pub rows: u32,
    pub n: u32,
    pub k: u32,
    pub weight: f64,
    /// Seconds per launch for each candidate; `None` where it did not run.
    pub times: Vec<Option<f64>>,
    /// The candidate today's profile runs here.
    pub baseline: Option<usize>,
    /// The timed candidates' median errors pooled, relative
    /// ([`profile::tune::Sample::spread`]).
    pub spread: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Measured {
    pub candidates: Vec<GemmPath>,
    pub points: Vec<Point>,
    /// Each candidate that answered wrongly: how many points, and the first.
    pub wrong: Vec<(String, usize, String)>,
}

impl Measured {
    /// What `pie tune --refit` reads back.
    #[must_use]
    pub fn to_text(&self, device: &str) -> String {
        let mut out = format!("# pie tune measurements, kernel.gemm.path, {device}\n");
        for p in &self.candidates {
            let _ = writeln!(out, "candidate {}", body(p));
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
                "point {} {} {} {} {} {} {:e}",
                p.rows,
                p.n,
                p.k,
                p.weight,
                p.baseline
                    .map_or_else(|| "-".to_string(), |b| b.to_string()),
                times.join(","),
                p.spread
            );
        }
        out
    }

    /// A file measured over another candidate list is refused.
    pub fn from_text(text: &str) -> Result<Measured, String> {
        let candidates = candidates();
        let bodies: Vec<String> = candidates.iter().map(body).collect();
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
                    weight: f
                        .get(3)
                        .and_then(|v| v.parse().ok())
                        .ok_or(format!("the weight of `{line}`"))?,
                    baseline: f.get(4).and_then(|b| b.parse().ok()),
                    times: f
                        .get(5)
                        .ok_or("times")?
                        .split(',')
                        .map(|t| t.parse().ok())
                        .collect(),
                    spread: f.get(6).and_then(|v| v.parse().ok()).ok_or(format!(
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

/// The fitted crossover as a profile entry, and its report line.
pub fn fit(measured: &Measured) -> Result<(Entries, Vec<Line>), String> {
    let spec = GemmPath::spec();
    let candidates: Vec<Candidate> = measured
        .candidates
        .iter()
        .map(|p| Candidate {
            body: body(p),
            terminal: true,
        })
        .collect();
    let samples: Vec<Sample> = measured
        .points
        .iter()
        .map(|p| Sample {
            at: vec![i64::from(p.rows)],
            weight: p.weight,
            times: p.times.clone(),
            spread: p.spread,
        })
        .collect();
    let baseline: Vec<Option<usize>> = measured.points.iter().map(|p| p.baseline).collect();
    let fit: Fit = profile::tune::fit(spec, &candidates, &samples, &baseline, Options::default())
        .map_err(|e| format!("{}: {e}", spec.id))?;
    let ladder = fit.toml(spec, &candidates);
    let mut text = format!(
        "{} rungs, at {:.3}x the per-point best",
        fit.rungs.len(),
        fit.cost / fit.floor
    );
    if let Some(b) = fit.baseline {
        let _ = write!(text, "; today's path runs at {:.3}x", b / fit.floor);
    }
    Ok((
        vec![(spec.id.to_string(), ladder)],
        vec![Line {
            id: spec.id,
            text,
            measured: true,
        }],
    ))
}

/// The registry's file forcing `p` at every row count.
#[cfg(any(test, feature = "vulkan"))]
fn forcing(p: &GemmPath) -> String {
    format!(
        "[kernel]\ngemm.path = \"{}\"\n",
        match p {
            GemmPath::Tiled => "tiled",
            GemmPath::Coopmat => "coopmat",
        }
    )
}

#[cfg(feature = "vulkan")]
pub use device::measure;

#[cfg(feature = "vulkan")]
mod device {
    use engine_gpu::backend::Tactics as _;
    use kernels_slang::Tensor;
    use poem_ir::Dtype;
    use profile::{Profile, Source};

    use super::{Grid, Measured, Point, body, candidates, forcing};
    use crate::device::{Api, Buffer, Context, Pipelines};
    use crate::tune::qmm::device::{SHAPE_BYTES, STREAM_BYTES, bf16, noise, seen, time, to_f32};
    use crate::tuning::Tuning;
    use engine_gpu::Handles;

    pub fn measure(
        device: &Context,
        base: &Profile,
        grid: &Grid,
        say: &mut dyn FnMut(&str),
    ) -> Result<Measured, String> {
        let registry = crate::chip::registry();
        let limits = device.limits();
        let target = crate::tune::target(base, &device.chip().facts());
        // The registry refuses a path the device cannot run.
        let mut kept = Vec::new();
        let mut forced = Vec::new();
        for p in candidates() {
            if let Ok(profile) = registry.overlay(
                base,
                &forcing(&p),
                target.clone(),
                Source::User("tune".into()),
            ) {
                kept.push(p);
                forced.push(Tuning::new(profile, limits));
            }
        }
        let today = Tuning::new(base.clone(), limits);
        let mut out = Measured {
            candidates: candidates(),
            ..Measured::default()
        };
        let handles = Handles::<Api>::new();
        let pipelines = Pipelines::new();
        let widest = *grid.rows.iter().max().expect("rows");
        let alloc = |bytes: u64| {
            Buffer::zeroed(device, bytes, engine_gpu::books::owners::TUNING)
                .map_err(|e| e.to_string())
        };
        for (at, &(n, k, share)) in grid.shapes.iter().enumerate() {
            let (nn, kk) = (n as usize, k as usize);
            let values = |len: usize, salt: u64| -> (Vec<u8>, Vec<f64>) {
                let mut bytes = vec![0u8; len * 2];
                let mut host = vec![0.0f64; len];
                for i in 0..len {
                    let v = 0.02 * (f32::from(noise(i as u64 ^ salt) % 16) - 8.0);
                    bytes[2 * i..2 * i + 2].copy_from_slice(&bf16(v));
                    host[i] = seen(v);
                }
                (bytes, host)
            };
            let (act_bytes, act) = values(widest as usize * kk, 0x77);
            let (w_bytes, w) = values(nn * kk, 0x99);
            let mut act_b = alloc(act_bytes.len() as u64)?;
            act_b.write(0, &act_bytes).map_err(|e| e.to_string())?;
            let out_b = alloc(u64::from(widest) * u64::from(n) * 2)?;
            // Enough weight copies to stream past the last-level cache, as
            // a model's layers do.
            let plane = w_bytes.len() as u64;
            let copies = STREAM_BYTES.div_ceil(plane).clamp(1, SHAPE_BYTES / plane);
            let mut planes = Vec::new();
            let mut hw = Vec::new();
            for _ in 0..copies {
                let mut b = alloc(plane)?;
                b.write(0, &w_bytes).map_err(|e| e.to_string())?;
                hw.push(handles.bind(&b, 0, b.bytes()).map_err(|e| e.to_string())?);
                planes.push(b);
            }
            let ha = handles
                .bind(&act_b, 0, act_b.bytes())
                .map_err(|e| e.to_string())?;
            let ho = handles
                .bind(&out_b, 0, out_b.bytes())
                .map_err(|e| e.to_string())?;
            for &m in &grid.rows {
                let launch = |sink: &dyn kernels_slang::Encode, t: &Tuning, copy: usize| {
                    kernels_slang::linear::gemm::matmul(
                        sink,
                        t.gemm(kernels_slang::Tier::Baseline, m),
                        Tensor::new(ha, m, k, Dtype::Bf16),
                        Tensor::new(hw[copy % hw.len()], n, k, Dtype::Bf16),
                        Tensor::new(ho, m, n, Dtype::Bf16),
                        kernels_slang::ssm::RowSpan::whole(Tensor::new(ha, m, k, Dtype::Bf16)),
                    )
                };
                let mut live = Vec::new();
                for (c, p) in kept.iter().enumerate() {
                    let frame = device.frame().map_err(|e| e.to_string())?;
                    let sink = crate::encode::Sink::new(device, &frame, &pipelines, &handles);
                    let launched = launch(&sink, &forced[c], 0).is_ok();
                    drop(sink);
                    frame.commit().map_err(|e| e.to_string())?;
                    if !launched {
                        continue;
                    }
                    if let Some(why) = wrong(&handles, ho, m, n, k, &w, &act) {
                        let first = format!("{m} x {n} x {k}: {why}");
                        let name = body(p);
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
                let all = candidates();
                let mut row = vec![None; all.len()];
                for (c, (t, _)) in live.iter().zip(&times) {
                    let i = all.iter().position(|p| *p == kept[*c]).expect("a path");
                    row[i] = Some(*t);
                }
                let spread = profile::tune::pooled(times.iter().map(|&(_, e)| e));
                let picked = if today.gemm_coopmat(m) {
                    crate::tuning::GemmPath::Coopmat
                } else {
                    crate::tuning::GemmPath::Tiled
                };
                out.points.push(Point {
                    rows: m,
                    n,
                    k,
                    weight: share / grid.rows.len() as f64,
                    baseline: all.iter().position(|p| *p == picked),
                    times: row,
                    spread,
                });
            }
            say(&format!(
                "  kernel.gemm.path: {}/{} shapes ({n} x {k})",
                at + 1,
                grid.shapes.len()
            ));
        }
        Ok(out)
    }

    /// A sample of the last launch's outputs against the f64 reference.
    fn wrong(
        handles: &Handles<Api>,
        ho: u32,
        m: u32,
        n: u32,
        k: u32,
        w: &[f64],
        act: &[f64],
    ) -> Option<String> {
        let raw = handles.read(ho, u64::from(m) * u64::from(n) * 2).ok()?;
        let (nn, kk) = (n as usize, k as usize);
        for r in [0, m as usize / 2, m as usize - 1] {
            for s in 0..16usize {
                let col = (s * 7919 + r * 104_729) % nn;
                let (mut want, mut mass) = (0.0f64, 0.0f64);
                for c in 0..kk {
                    let t = act[r * kk + c] * w[col * kk + c];
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_path_forces_a_file_the_registry_reads() {
        let facts = profile::Facts::of(crate::chip::registry())
            .word("vendor", "nvidia")
            .int("subgroup", 32)
            .int("cores", 82)
            .flag("coopmat", true);
        let (base, _) = crate::chip::resolve_facts("Test GPU", &facts, None);
        let target = crate::tune::target(&base, &facts);
        for p in candidates() {
            crate::chip::registry()
                .overlay(
                    &base,
                    &forcing(&p),
                    target.clone(),
                    profile::Source::User("test".into()),
                )
                .unwrap_or_else(|e| panic!("{}: {e:?}", body(&p)));
        }
    }

    /// A crossover measured at 128 rows fits to tiled below it and
    /// cooperative matrices from it, and reads back from its own text.
    #[test]
    fn a_measured_crossover_fits_a_rows_rung() {
        let grid = Grid::of(Depth::Quick);
        let mut measured = Measured {
            candidates: candidates(),
            ..Measured::default()
        };
        for &(n, k, share) in &grid.shapes {
            for &rows in &grid.rows {
                let tiled = f64::from(rows) * 1e-6;
                let coopmat = 128.0 * 1e-6 + f64::from(rows) * 0.5e-6;
                let coopmat = if rows >= 128 {
                    tiled.min(coopmat) * 0.9
                } else {
                    coopmat
                };
                measured.points.push(Point {
                    rows,
                    n,
                    k,
                    weight: share / grid.rows.len() as f64,
                    times: vec![Some(tiled), Some(coopmat)],
                    baseline: Some(1),
                    spread: 0.005,
                });
            }
        }
        let back = Measured::from_text(&measured.to_text("Test GPU")).unwrap();
        assert_eq!(back.points.len(), measured.points.len());
        let (entries, _) = fit(&back).unwrap();
        assert_eq!(entries[0].0, "kernel.gemm.path");
        // The fitted ladder, laid over a coopmat device's profile, takes the
        // tiled path below 128 rows and cooperative matrices from 128.
        let facts = profile::Facts::of(crate::chip::registry())
            .word("vendor", "nvidia")
            .int("subgroup", 32)
            .int("cores", 82)
            .flag("coopmat", true);
        let (base, _) = crate::chip::resolve_facts("Test GPU", &facts, None);
        let text = format!("[kernel]\ngemm.path = {}\n", entries[0].1);
        let tuned = crate::chip::registry()
            .overlay(
                &base,
                &text,
                crate::tune::target(&base, &facts),
                profile::Source::User("test".into()),
            )
            .unwrap_or_else(|e| panic!("{text}: {e:?}"));
        let limits = crate::tuning::Limits::ASSUMED;
        let t = crate::tuning::Tuning::new(tuned, limits);
        assert!(!t.gemm_coopmat(4) && !t.gemm_coopmat(127), "{text}");
        assert!(t.gemm_coopmat(128) && t.gemm_coopmat(2048), "{text}");
        // The profile it replaces takes cooperative matrices at every row
        // count: the crossover is the fit's, not the base's.
        let today = crate::tuning::Tuning::new(base, limits);
        assert!(today.gemm_coopmat(4) && today.gemm_coopmat(127));
    }
}
