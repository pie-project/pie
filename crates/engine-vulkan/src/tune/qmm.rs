//! The `kernel.qmm` bench: every candidate timed over a grid of dense
//! quantised matmuls, checked against an f64 reference, and a ladder fitted.

use std::fmt::Write as _;

use kernels_vulkan::tuning::Qmm;
use profile::Decision as _;
use profile::Profile;
use profile::search::{Entries, Line};
use profile::tune::{Candidate, Fit, Options, Sample};

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

/// One bank format: code width and group width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bank {
    pub bits: u32,
    pub group: u32,
}

/// The quantised bank shapes of the models pie ships, `(n, k, banks)`,
/// heaviest first; the same census the Metal bench measures.
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

#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub rows: Vec<u32>,
    /// `(n, k, weight)`: the shape's share of the grid's census bytes.
    pub shapes: Vec<(u32, u32, f64)>,
    pub banks: Vec<Bank>,
}

impl Grid {
    #[must_use]
    pub fn of(depth: Depth) -> Grid {
        let bank = |bits, group| Bank { bits, group };
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
                rows: vec![1, 2, 4, 8, 12, 16, 32, 64, 128, 512],
                shapes: shapes([0, 1, 3, 9, 15, 20].iter().map(|&i| CENSUS[i]).collect()),
                banks: vec![bank(4, 64), bank(4, 32), bank(8, 64), bank(2, 64)],
            },
            Depth::Full => Grid {
                rows: (1..=16)
                    .chain([20, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048])
                    .collect(),
                shapes: shapes(CENSUS.to_vec()),
                banks: vec![
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

/// Metal's workload mix: half decode (8 rows or fewer), a fifth small
/// batches, the rest prefill; by bank, 4-bit g64 half, 4-bit g32 a tenth,
/// 8-bit 0.15, 2-bit a quarter.
#[must_use]
pub fn weight(grid: &Grid, rows: u32, n: u32, k: u32, bank: Bank) -> f64 {
    let band = |lo: u32, hi: u32| {
        grid.rows
            .iter()
            .filter(|&&r| lo <= r && r <= hi)
            .count()
            .max(1)
    };
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
        .filter(|b| kind(b) == kind(&bank))
        .count()
        .max(1);
    let per_shape = grid
        .shapes
        .iter()
        .find(|s| (s.0, s.1) == (n, k))
        .map_or(0.0, |s| s.2);
    per_rows * share / alike as f64 * per_shape
}

/// Every kernel at a spread of its constants, and every rung the current
/// profile holds.
#[must_use]
pub fn candidates(base: &Profile) -> Vec<Qmm> {
    let mut out = vec![
        Qmm::Coopmat,
        Qmm::RegisterTiled { bm: 32, k16: true },
        Qmm::RegisterTiled { bm: 64, k16: true },
        Qmm::RegisterTiled { bm: 32, k16: false },
        Qmm::RegisterTiled { bm: 64, k16: false },
    ];
    for crossover_tg in [64, 96, 128, 160, 256] {
        out.push(Qmm::Tiled { crossover_tg });
    }
    for lanes in [8, 16, 32] {
        for rows in [1, 2] {
            out.push(Qmm::OneRow { lanes, rows });
        }
    }
    for vecs in [1, 2, 4, 8] {
        out.push(Qmm::Qmv { vecs });
    }
    for q in base.ladder::<Qmm>().values() {
        if !out.contains(q) {
            out.push(*q);
        }
    }
    out
}

#[must_use]
pub fn body(q: &Qmm) -> String {
    match *q {
        Qmm::Coopmat => "use = \"coopmat\"".to_string(),
        Qmm::RegisterTiled { bm, k16 } => {
            format!("use = \"register_tiled\", bm = {bm}, k16 = {k16}")
        }
        Qmm::Tiled { crossover_tg } => format!("use = \"tiled\", crossover_tg = {crossover_tg}"),
        Qmm::OneRow { lanes, rows } => {
            format!("use = \"one_row\", lanes = {lanes}, rows = {rows}")
        }
        Qmm::Qmv { vecs } => format!("use = \"qmv\", vecs = {vecs}"),
    }
}

/// The ladder that runs `q` wherever it can, and the vector kernel
/// elsewhere.
#[cfg(any(test, feature = "vulkan"))]
fn forcing(q: &Qmm) -> String {
    match q {
        Qmm::Qmv { .. } => format!("[{{ {} }}]", body(q)),
        _ => format!("[{{ {} }}, {{ use = \"qmv\", vecs = 4 }}]", body(q)),
    }
}

/// The kernel `kernels_vulkan::linear::quant::ran` names for `q`.
#[cfg(feature = "vulkan")]
fn kernel(q: &Qmm) -> &'static str {
    match q {
        Qmm::Coopmat => "coopmat",
        Qmm::RegisterTiled { .. } => "register_tiled",
        Qmm::Tiled { .. } => "tiled",
        Qmm::OneRow { .. } => "one_row",
        Qmm::Qmv { .. } => "qmv",
    }
}

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

#[derive(Clone, Debug, Default)]
pub struct Measured {
    pub candidates: Vec<Qmm>,
    pub points: Vec<Point>,
    /// Each candidate that answered wrongly: how many points, and the first.
    pub wrong: Vec<(String, usize, String)>,
}

impl Measured {
    /// What `pie tune --refit` reads back.
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
                "point {} {} {} {} {} {} {}",
                p.rows,
                p.n,
                p.k,
                p.bank.bits,
                p.bank.group,
                p.baseline
                    .map_or_else(|| "-".to_string(), |b| b.to_string()),
                times.join(",")
            );
        }
        out
    }

    /// A file measured over another candidate list is refused.
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
                    f.get(i)
                        .and_then(|v| v.parse().ok())
                        .ok_or(format!("field {i} of `{line}`"))
                };
                out.points.push(Point {
                    rows: num(0)?,
                    n: num(1)?,
                    k: num(2)?,
                    bank: Bank {
                        bits: num(3)?,
                        group: num(4)?,
                    },
                    baseline: f.get(5).and_then(|b| b.parse().ok()),
                    times: f
                        .get(6)
                        .ok_or("times")?
                        .split(',')
                        .map(|t| t.parse().ok())
                        .collect(),
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

/// A point's coordinates in `kernel.qmm`'s variable order.
fn coords(p: &Point, cores: u32) -> Vec<i64> {
    let at = kernels_vulkan::tuning::qmm_at(p.rows, p.n, p.k, p.bank.bits, p.bank.group, cores);
    vec![at.rows, at.fill, at.n, at.k, at.bits, at.group]
}

/// The fitted ladder as a profile entry, and its report line.
pub fn fit(measured: &Measured, grid: &Grid, cores: u32) -> Result<(Entries, Vec<Line>), String> {
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
        samples.push(Sample {
            at: coords(p, cores),
            weight: weight(grid, p.rows, p.n, p.k, p.bank),
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
    Ok((
        vec![(spec.id.to_string(), ladder)],
        vec![Line { id: spec.id, text }],
    ))
}

#[cfg(feature = "vulkan")]
pub use device::measure;

#[cfg(feature = "vulkan")]
mod device {
    use std::cell::RefCell;

    use kernels_vulkan::linear::quant;
    use kernels_vulkan::{ArgValue, Encode, Error, Fire, Tensor, Tuning};
    use poem_ir::Dtype;
    use profile::{Decision as _, Profile, Source};

    use super::{Bank, Grid, Measured, Point, body, candidates, forcing, kernel};
    use crate::device::{Buffer, Context, Handles, Pipelines};
    use crate::encode::Sink;

    /// Remembers the entries it fired: which kernel a forced candidate ran.
    struct Watch<'a> {
        inner: &'a dyn Encode,
        entries: RefCell<Vec<&'static str>>,
    }

    impl Encode for Watch<'_> {
        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
            self.entries.borrow_mut().push(fire.entrypoint);
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

    /// A bank in enough copies to stream past the last-level cache, with
    /// its host values for the reference.
    struct Weights {
        codes: Vec<u32>,
        scales: Vec<u32>,
        biases: Vec<u32>,
        q: Vec<u8>,
        scale: Vec<f64>,
        bias: Vec<f64>,
        _buffers: Vec<Buffer>,
    }

    /// More than any GPU's last-level cache.
    const STREAM_BYTES: u64 = 256 << 20;
    const SHAPE_BYTES: u64 = 1 << 30;

    fn weights(
        device: &Context,
        handles: &Handles,
        n: u32,
        k: u32,
        bank: Bank,
    ) -> Result<Weights, String> {
        let (bits, group) = (bank.bits, bank.group);
        let (nn, kk) = (n as usize, k as usize);
        let groups = kk / group as usize;
        let levels = 1u16 << bits;
        let q: Vec<u8> = (0..nn * kk)
            .map(|i| (u16::from(noise(i as u64 ^ 0x5151)) % levels) as u8)
            .collect();
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
        let copies = STREAM_BYTES
            .div_ceil(bank_bytes)
            .clamp(1, SHAPE_BYTES / bank_bytes);
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
                let mut b =
                    Buffer::zeroed(device, bytes.len() as u64).map_err(|e| e.to_string())?;
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

    pub fn measure(
        device: &Context,
        base: &Profile,
        grid: &Grid,
        say: &mut dyn FnMut(&str),
    ) -> Result<Measured, String> {
        let registry = crate::chip::registry();
        let limits = device.limits();
        let target = crate::tune::target(base, &device.chip().facts());
        let tuning = |text: &str| -> Option<Tuning> {
            registry
                .overlay(base, text, target.clone(), Source::User("tune".into()))
                .ok()
                .map(|p| Tuning::new(p, limits))
        };
        // The registry refuses a candidate the device cannot run.
        let mut kept = Vec::new();
        let mut forced = Vec::new();
        for q in candidates(base) {
            if let Some(t) = tuning(&format!("[kernel]\nqmm = {}\n", forcing(&q))) {
                kept.push(q);
                forced.push(t);
            }
        }
        let today = Tuning::new(base.clone(), limits);

        let handles = Handles::new();
        let pipelines = Pipelines::new();
        let widest = *grid.rows.iter().max().expect("rows");
        let cap = widest.div_ceil(64) * 64;
        let total = grid.shapes.len() * grid.banks.len() * grid.rows.len();
        let mut done = 0usize;
        let mut out = Measured {
            candidates: kept.clone(),
            ..Measured::default()
        };
        for &(n, k, _) in &grid.shapes {
            let kk = k as usize;
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
            let ha = handles
                .bind(&act_b, 0, act_b.bytes())
                .map_err(|e| e.to_string())?;
            let ho = handles
                .bind(&out_b, 0, out_b.bytes())
                .map_err(|e| e.to_string())?;

            for bank in &grid.banks {
                if k % bank.group != 0 {
                    done += grid.rows.len();
                    continue;
                }
                let w = match weights(device, &handles, n, k, *bank) {
                    Ok(w) => w,
                    Err(why) => {
                        say(&format!("  skipped {n} x {k}: {why}"));
                        done += grid.rows.len();
                        continue;
                    }
                };
                for &m in &grid.rows {
                    done += 1;
                    let launch = |sink: &dyn Encode, tuning: &Tuning, copy: usize| {
                        let copy = copy % w.codes.len();
                        let b = kernels_vulkan::Bank {
                            codes: Tensor::new(w.codes[copy], n, k, dtype(bank.bits, bank.group)),
                            scales: Tensor::new(w.scales[copy], n, k / bank.group, Dtype::Bf16),
                            biases: Some(Tensor::new(
                                w.biases[copy],
                                n,
                                k / bank.group,
                                Dtype::Bf16,
                            )),
                            group: bank.group,
                            bits: bank.bits,
                        };
                        quant::matmul(
                            sink,
                            tuning,
                            Tensor::new(ha, m, k, Dtype::Bf16),
                            b,
                            Tensor::new(ho, m, n, Dtype::Bf16),
                            cap,
                        )
                    };
                    let mut live: Vec<usize> = Vec::new();
                    for (c, q) in kept.iter().enumerate() {
                        let t = &forced[c];
                        if !q.fits(&t.qmm_at(m, n, k, bank.bits, bank.group)) {
                            continue;
                        }
                        let frame = device.frame().map_err(|e| e.to_string())?;
                        let sink = Sink::new(device, &frame, &pipelines, &handles);
                        let watch = Watch {
                            inner: &sink,
                            entries: RefCell::new(Vec::new()),
                        };
                        let launched = launch(&watch, t, 0).is_ok();
                        let ran = watch.entries.borrow().iter().find_map(|e| quant::ran(e));
                        drop(watch);
                        drop(sink);
                        frame.commit().map_err(|e| e.to_string())?;
                        if !launched || ran != Some(kernel(q)) {
                            continue;
                        }
                        if let Some(why) = wrong(&handles, ho, m, n, k, *bank, &w, &act) {
                            let first = format!(
                                "{m} x {n} x {k}, {}-bit g{}: {why}",
                                bank.bits, bank.group
                            );
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
                        launch(sink, &forced[c], copy)
                    })?;
                    let mut row = vec![None; kept.len()];
                    for (c, t) in live.iter().zip(times) {
                        row[*c] = Some(t);
                    }
                    let at = today.qmm_at(m, n, k, bank.bits, bank.group);
                    let baseline = today
                        .qmm(&at)
                        .iter()
                        .filter_map(|q| kept.iter().position(|c| c == q))
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
                    "  kernel.qmm: {done}/{total} points ({n} x {k}, {}-bit g{})",
                    bank.bits, bank.group
                ));
            }
        }
        Ok(out)
    }

    /// A sample of the last launch's outputs against the f64 reference.
    #[allow(clippy::too_many_arguments)]
    fn wrong(
        handles: &Handles,
        ho: u32,
        m: u32,
        n: u32,
        k: u32,
        bank: Bank,
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

    /// Seconds per launch for each of `live`, timed round-robin so drift
    /// lands on every candidate alike.
    fn time(
        device: &Context,
        pipelines: &Pipelines,
        handles: &Handles,
        live: &[usize],
        launch: impl Fn(&dyn Encode, usize, usize) -> Result<(), Error>,
    ) -> Result<Vec<f64>, String> {
        const ROUNDS: usize = 7;
        const FRAME_S: f64 = 0.002;
        /// A candidate this much slower than the fastest is timed once.
        const SLOWER: f64 = 3.0;
        let frame = |c: usize, batch: usize| -> Result<f64, String> {
            let f = device.frame().map_err(|e| e.to_string())?;
            let sink = Sink::new(device, &f, pipelines, handles);
            for copy in 0..batch {
                launch(&sink, c, copy).map_err(|e| e.to_string())?;
            }
            drop(sink);
            f.commit_timed().map_err(|e| e.to_string())
        };
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
    fn every_candidate_forces_a_ladder_the_registry_reads() {
        for (vendor, subgroup, coopmat) in [("nvidia", 32, true), ("qualcomm", 64, false)] {
            let facts = profile::Facts::of(crate::chip::registry())
                .word("vendor", vendor)
                .int("subgroup", subgroup)
                .int("cores", 142)
                .flag("coopmat", coopmat);
            let (base, _) = crate::chip::resolve_facts("Test GPU", &facts, None);
            for q in candidates(&base) {
                if q == Qmm::Coopmat && !coopmat {
                    continue;
                }
                let p = crate::chip::registry()
                    .overlay(
                        &base,
                        &format!("[kernel]\nqmm = {}\n", forcing(&q)),
                        crate::tune::target(&base, &facts),
                        profile::Source::User("t".into()),
                    )
                    .unwrap_or_else(|e| panic!("{vendor}: {}: {e}", body(&q)));
                assert_eq!(p.ladder::<Qmm>().values().next(), Some(&q));
            }
        }
    }

    #[test]
    fn a_fitted_ladder_assembles_into_a_file_this_device_takes() {
        let grid = Grid {
            rows: vec![1, 16, 512],
            shapes: vec![(4096, 4096, 1.0)],
            banks: vec![Bank { bits: 4, group: 64 }],
        };
        let candidates = vec![
            Qmm::Tiled { crossover_tg: 160 },
            Qmm::Qmv { vecs: 4 },
            Qmm::Qmv { vecs: 2 },
        ];
        let point = |rows, times: Vec<Option<f64>>| Point {
            rows,
            n: 4096,
            k: 4096,
            bank: grid.banks[0],
            times,
            baseline: Some(1),
        };
        let measured = Measured {
            candidates: candidates.clone(),
            points: vec![
                point(1, vec![Some(3.0), Some(2.0), Some(1.0)]),
                point(16, vec![Some(1.0), Some(2.0), Some(2.5)]),
                point(512, vec![Some(1.0), Some(9.0), Some(9.0)]),
            ],
            wrong: Vec::new(),
        };
        let back = Measured::from_text(&measured.to_text("Test GPU"), &candidates).unwrap();
        assert_eq!(back.points.len(), 3);
        let (entries, lines) = fit(&back, &grid, 142).unwrap();
        assert!(lines[0].text.contains("rungs"), "{lines:?}");
        let facts = profile::Facts::of(crate::chip::registry())
            .word("vendor", "nvidia")
            .int("subgroup", 32)
            .int("cores", 142)
            .flag("coopmat", false);
        let (base, _) = crate::chip::resolve_facts("Test GPU", &facts, None);
        let file = profile::search::assemble(
            crate::chip::registry(),
            &base,
            "Test GPU",
            &facts,
            &profile::search::render_overrides(&entries),
            &[],
            crate::tune::ASSUME,
        )
        .unwrap();
        assert!(file.contains("vecs = 2"), "{file}");
    }
}
